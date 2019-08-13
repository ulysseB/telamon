//! Describes a `Function` that is ready to execute on a device.
use std::{fmt, sync::Arc};

use fxhash::{FxHashMap, FxHashSet};
use itertools::Itertools;
use log::{debug, trace};
use matches::matches;
use utils::*;

use crate::codegen::{
    self, cfg, dimension, Cfg, Dimension, InductionLevel, InductionVar,
};
use crate::ir::{self, IrDisplay};
use crate::search_space::{self, DimKind, Domain, MemSpace, Order, SearchSpace};

use super::access::{IndexVarId, IndexVars, VarWalker};
use super::iteration::IterationVars;
use super::predicates::{PredicateId, PredicateKey, Predicates};

pub struct FunctionBuilder<'a> {
    space: &'a SearchSpace,
    predicated: bool,
}

impl<'a> FunctionBuilder<'a> {
    pub fn new(space: &'a SearchSpace) -> Self {
        FunctionBuilder {
            space,
            predicated: false,
        }
    }

    pub fn predicated(mut self, predicated: bool) -> Self {
        self.predicated = predicated;
        self
    }

    pub fn build(&self) -> Function<'a> {
        let mut dims = dimension::group_merged_dimensions(self.space);
        let (induction_vars, init_induction_levels) =
            dimension::register_induction_vars(&mut dims, self.space);
        trace!("dims = {:?}", dims);
        let insts = self
            .space
            .ir_instance()
            .insts()
            .map(|inst| Instruction::new(inst, self.space))
            .collect_vec();
        let mut device_code_args = dims
            .iter()
            .flat_map(|d| d.host_values(self.space))
            .chain(
                induction_vars
                    .iter()
                    .flat_map(|v| v.host_values(self.space)),
            )
            .chain(insts.iter().flat_map(|i| i.host_values(self.space)))
            .chain(
                init_induction_levels
                    .iter()
                    .flat_map(|l| l.host_values(self.space)),
            )
            .collect::<FxHashSet<_>>();
        let (block_dims, thread_dims, cfg) = cfg::build(self.space, insts, dims);
        let mem_blocks = register_mem_blocks(self.space, &block_dims);
        device_code_args.extend(
            mem_blocks
                .iter()
                .flat_map(|x| x.host_values(self.space, &block_dims)),
        );

        let merged_dims: dimension::MergedDimensions<'_> = cfg
            .dimensions()
            .chain(&block_dims)
            .chain(&thread_dims)
            .collect();

        // Iteration vars
        let mut index_vars = IndexVars::default();
        let mut walker = VarWalker {
            merged_dims: &merged_dims,
            space: &self.space,
            device_code_args: &mut device_code_args,
            index_vars: &mut index_vars,
        };

        let mut access_map = FxHashMap::default();
        for inst in cfg.instructions() {
            for operand in inst.operator().operands() {
                match operand {
                    ir::Operand::ComputedAddress(id) => {
                        let access = &self.space.ir_instance().accesses()[*id];
                        walker.process_parameter(access.base());

                        let strides = access
                            .strides()
                            .map(|(expr, stride)| {
                                let stride = codegen::Size::from_ir(
                                    &ir::PartialSize::from(stride.clone()),
                                    self.space,
                                );

                                walker.process_size(&stride);
                                (walker.process_index_expr(expr), stride)
                            })
                            .collect::<Vec<_>>();

                        access_map.insert(*id, strides);
                    }
                    _ => (),
                }
            }
        }

        let mut iteration_vars = IterationVars::default();
        let mut predicates = Predicates::default();
        let mut instruction_predicates = FxHashMap::default();
        let mut loop_predicate_def = FxHashMap::default();
        let mut global_predicate_def = Vec::new();

        for inst in cfg.instructions() {
            let mut pred_dim = None;
            let mut inst_preds = Vec::new();
            for pred in inst.operator().predicates() {
                let mut instantiation_dims = Vec::new();
                let mut loop_dims = Vec::new();
                let mut global_dims = Vec::new();

                for &(dim, ref stride) in self
                    .space
                    .ir_instance()
                    .induction_var(pred.induction_variable())
                    .dims()
                    .iter()
                {
                    let dim = merged_dims[dim].id();
                    let stride = codegen::Size::from_ir(stride, self.space);

                    match self.space.domain().get_dim_kind(dim) {
                        DimKind::LOOP => loop_dims.push((dim, stride)),
                        DimKind::INNER_VECTOR
                        | DimKind::OUTER_VECTOR
                        | DimKind::UNROLL => instantiation_dims.push((
                            dim,
                            stride.as_int().unwrap_or_else(|| {
                                panic!("predicate instantation with dynamic stride")
                            }),
                        )),
                        DimKind::BLOCK | DimKind::THREAD => {
                            global_dims.push((dim, stride))
                        }
                        _ => panic!("invalid dim kind"),
                    }
                }

                // Loop dimensions must be in nesting order for `IterationVars`
                loop_dims.sort_unstable_by(|&(lhs, _), &(rhs, _)| {
                    if lhs == rhs {
                        return std::cmp::Ordering::Equal;
                    }

                    match self.space.domain().get_order(lhs.into(), rhs.into()) {
                        Order::INNER => std::cmp::Ordering::Greater,
                        Order::OUTER => std::cmp::Ordering::Less,
                        Order::MERGED => {
                            panic!("found MERGED order between representants")
                        }
                        _ => panic!("invalid order for induction variable dimensions"),
                    }
                });

                // The dim where we compute the final predicate.  This is innermost.
                if let Some(inner_dim) = loop_dims.last().map(|&(dim, _)| dim) {
                    match pred_dim {
                        None => pred_dim = Some(inner_dim),
                        Some(old) if old == inner_dim => (),
                        Some(old) => {
                            match self
                                .space
                                .domain()
                                .get_order(old.into(), inner_dim.into())
                            {
                                Order::INNER => (),
                                Order::MERGED => panic!("MERGED representants"),
                                Order::OUTER => pred_dim = Some(inner_dim),
                                _ => panic!("invalid order"),
                            }
                        }
                    }
                }

                // The iteration variable which tells us where we are for the non unrolled,
                // non vector dimensions
                let iteration_var = iteration_vars.add(global_dims, loop_dims);

                // The predicate
                let predicate = predicates.add(PredicateKey {
                    iteration_var,
                    instantiation_dims,
                    bound: codegen::Size::from_ir(
                        &ir::PartialSize::from(pred.bound().clone()),
                        self.space,
                    ),
                });

                inst_preds.push(predicate);
            }

            if !inst_preds.is_empty() {
                let mut instantiation_dims = Vec::new();
                for &id in &inst_preds {
                    instantiation_dims.extend(predicates[id].instantiation_dims().map(
                        |&(dim, _)| {
                            (
                                dim,
                                // We need to go through `codegen::Size` here to simplify the tile
                                // dimension sizes
                                codegen::Size::from_ir(
                                    self.space.ir_instance().dim(dim).size(),
                                    self.space,
                                )
                                .as_int()
                                .unwrap() as usize,
                            )
                        },
                    ));
                }
                instantiation_dims.sort_unstable();
                instantiation_dims.dedup();
                instruction_predicates.insert(inst.id(), instantiation_dims);

                if let Some(dim) = pred_dim {
                    loop_predicate_def
                        .entry(dim)
                        .or_insert(Vec::new())
                        .push((inst.id(), inst_preds));
                } else {
                    global_predicate_def.push((inst.id(), inst_preds));
                }
            }
        }

        debug!("compiling cfg {:?}", cfg);
        Function {
            cfg,
            thread_dims,
            block_dims,
            induction_vars,
            device_code_args: device_code_args.into_iter().collect(),
            space: self.space,
            mem_blocks,
            variables: codegen::variable::wrap_variables(self.space),
            init_induction_levels,
            predicate_accesses: self.predicated,
            iteration_vars,
            index_vars,
            predicates,
            loop_predicate_def,
            global_predicate_def,
            instruction_predicates,
            access_map,
        }
    }
}

/// A function ready to execute on a device, derived from a constrained IR instance.
pub struct Function<'a> {
    cfg: Cfg<'a>,
    thread_dims: Vec<Dimension<'a>>,
    block_dims: Vec<Dimension<'a>>,
    device_code_args: Vec<ParamVal>,
    induction_vars: Vec<InductionVar<'a>>,
    mem_blocks: Vec<MemoryRegion>,
    init_induction_levels: Vec<InductionLevel<'a>>,
    variables: Vec<codegen::Variable<'a>>,
    iteration_vars: IterationVars,
    index_vars: IndexVars,
    predicates: Predicates,
    loop_predicate_def: FxHashMap<ir::DimId, Vec<(ir::InstId, Vec<PredicateId>)>>,
    global_predicate_def: Vec<(ir::InstId, Vec<PredicateId>)>,
    instruction_predicates: FxHashMap<ir::InstId, Vec<(ir::DimId, usize)>>,
    access_map: FxHashMap<ir::AccessId, Vec<(IndexVarId, codegen::Size)>>,
    // TODO(cleanup): remove dependency on the search space
    space: &'a SearchSpace,
    predicate_accesses: bool,
}

impl<'a> Function<'a> {
    /// Creates a device `Function` from an IR instance.
    pub fn build(space: &'a SearchSpace) -> Function<'a> {
        FunctionBuilder::new(space).build()
    }

    /// Returns the ordered list of thread dimensions.
    pub fn thread_dims(&self) -> &[Dimension<'a>] {
        &self.thread_dims
    }

    /// Returns the ordered list of block dimensions.
    pub fn block_dims(&self) -> &[Dimension<'a>] {
        &self.block_dims
    }

    /// Iterate on the function variables.
    pub fn variables(&self) -> impl Iterator<Item = &codegen::Variable> {
        self.variables.iter()
    }

    /// Iterates other all `codegen::Dimension`.
    pub fn dimensions(&self) -> impl Iterator<Item = &Dimension> {
        self.cfg
            .dimensions()
            .chain(&self.block_dims)
            .chain(&self.thread_dims)
    }

    /// Returns the list of induction variables.
    pub fn induction_vars(&self) -> &[InductionVar<'a>] {
        &self.induction_vars
    }

    /// Returns the total number of threads to allocate.
    pub fn num_threads(&self) -> u32 {
        self.thread_dims
            .iter()
            .map(|d| unwrap!(d.size().as_int()))
            .product()
    }

    /// Returns the values to pass from the host to the device.
    pub fn device_code_args(&self) -> impl Iterator<Item = &ParamVal> {
        self.device_code_args.iter()
    }

    pub fn predicate_accesses(&self) -> bool {
        self.predicate_accesses
    }

    pub fn iteration_variables(&self) -> &IterationVars {
        &self.iteration_vars
    }

    pub fn index_vars(&self) -> &IndexVars {
        &self.index_vars
    }

    pub fn access_map(
        &self,
    ) -> &FxHashMap<ir::AccessId, Vec<(IndexVarId, codegen::Size)>> {
        &self.access_map
    }

    pub fn global_predicates(&self) -> &[(ir::InstId, Vec<PredicateId>)] {
        &self.global_predicate_def
    }

    pub fn predicates(&self) -> &Predicates {
        &self.predicates
    }

    pub fn loop_predicates(&self, dim: ir::DimId) -> &[(ir::InstId, Vec<PredicateId>)] {
        if let Some(predicates) = self.loop_predicate_def.get(&dim) {
            &*predicates
        } else {
            &[]
        }
    }

    pub fn instruction_predicates(
        &self,
    ) -> &FxHashMap<ir::InstId, Vec<(ir::DimId, usize)>> {
        &self.instruction_predicates
    }

    /// Returns the control flow graph.
    pub fn cfg(&self) -> &Cfg<'a> {
        &self.cfg
    }

    /// Returns all the induction levels in the function.
    pub fn induction_levels(&self) -> impl Iterator<Item = &InductionLevel> {
        self.block_dims
            .iter()
            .chain(&self.thread_dims)
            .flat_map(|d| d.induction_levels())
            .chain(self.cfg.induction_levels())
            .chain(self.init_induction_levels())
    }

    /// Returns the memory blocks allocated by the function.
    pub fn mem_blocks(&self) -> impl Iterator<Item = &MemoryRegion> {
        self.mem_blocks.iter()
    }

    /// Returns the underlying implementation space.
    // TODO(cleanup): prefer access to the space from individual wrappers on ir objects.
    pub fn space(&self) -> &SearchSpace {
        self.space
    }

    /// Returns the name of the function.
    pub fn name(&self) -> &str {
        self.space.ir_instance().name()
    }

    /// Returns the induction levels computed at the beginning of the kernel. Levels must
    /// be computed in the provided order.
    pub fn init_induction_levels(&self) -> &[InductionLevel] {
        &self.init_induction_levels
    }
}

impl<'a> fmt::Display for Function<'a> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        writeln!(
            fmt,
            "BLOCKS[{}]({}) THREADS[{}]({})",
            self.block_dims.iter().map(|d| d.size()).format(", "),
            self.block_dims
                .iter()
                .map(|d| d.dim_ids().format(" = "))
                .format(", "),
            self.thread_dims.iter().map(|d| d.size()).format(", "),
            self.thread_dims
                .iter()
                .map(|d| d.dim_ids().format(" = "))
                .format(", "),
        )?;
        write!(fmt, "{}", self.cfg().display(self.space.ir_instance()))
    }
}

/// Represents the value of a parameter passed to the kernel by the host.
#[derive(Debug)]
pub enum ParamVal {
    /// A parameter given by the caller.
    External(Arc<ir::Parameter>, ir::Type),
    /// A tiled dimension size computed on the host.
    Size(codegen::Size),
    /// A pointer to a global memory block, allocated by the wrapper.
    GlobalMem(ir::MemId, codegen::Size, ir::Type),
    // Magic constant for division
    DivMagic(codegen::Size, ir::Type),
    // Shift amount for division
    DivShift(codegen::Size, ir::Type),
}

impl ParamVal {
    /// Builds the `ParamVal` needed to implement an operand, if any.
    pub fn from_operand(operand: &ir::Operand, space: &SearchSpace) -> Option<Self> {
        match operand {
            ir::Operand::Param(p) => {
                let t = unwrap!(space.ir_instance().device().lower_type(p.t, space));
                Some(ParamVal::External(p.clone(), t))
            }
            ir::Operand::ComputedAddress(access) => {
                let base = space.ir_instance().accesses()[*access].base();
                let t = space
                    .ir_instance()
                    .device()
                    .lower_type(base.t, space)
                    .unwrap();
                Some(ParamVal::External(base.clone(), t))
            }
            _ => None,
        }
    }

    /// Builds the `ParamVal` needed to get a size value, if any.
    pub fn from_size(size: &codegen::Size) -> Option<Self> {
        match *size.dividend() {
            [] => None,
            [ref p] if size.factor() == 1 && size.divisor() == 1 => {
                Some(ParamVal::External(p.clone(), ir::Type::I(32)))
            }
            _ => Some(ParamVal::Size(size.clone())),
        }
    }

    pub fn div_magic(size: &codegen::Size, t: ir::Type) -> Option<Self> {
        match *size.dividend() {
            [] => None,
            _ => Some(ParamVal::DivMagic(size.clone(), t)),
        }
    }

    pub fn div_shift(size: &codegen::Size, t: ir::Type) -> Option<Self> {
        match *size.dividend() {
            [] => None,
            _ => Some(ParamVal::DivShift(size.clone(), t)),
        }
    }

    /// Returns the type of the parameter.
    pub fn t(&self) -> ir::Type {
        match *self {
            ParamVal::External(_, t)
            | ParamVal::GlobalMem(.., t)
            | ParamVal::DivMagic(.., t)
            | ParamVal::DivShift(.., t) => t,
            ParamVal::Size(_) => ir::Type::I(32),
        }
    }

    /// Indicates if the parameter is a pointer.
    pub fn is_pointer(&self) -> bool {
        match *self {
            ParamVal::External(ref p, _) => matches!(p.t, ir::Type::PtrTo(_)),
            ParamVal::GlobalMem(..) => true,
            ParamVal::Size(_) | ParamVal::DivMagic(..) | ParamVal::DivShift(..) => false,
        }
    }

    /// Returns a unique identifier for the `ParamVal`.
    pub fn key(&self) -> ParamValKey<'_> {
        match *self {
            ParamVal::External(ref p, _) => ParamValKey::External(&*p),
            ParamVal::Size(ref s) => ParamValKey::Size(s),
            ParamVal::GlobalMem(mem, ..) => ParamValKey::GlobalMem(mem),
            ParamVal::DivMagic(ref size, t) => ParamValKey::DivMagic(size, t),
            ParamVal::DivShift(ref size, t) => ParamValKey::DivShift(size, t),
        }
    }
}

hash_from_key!(ParamVal, ParamVal::key);

/// Uniquely identifies a `ParamVal`.
#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone)]
pub enum ParamValKey<'a> {
    External(&'a ir::Parameter),
    Size(&'a codegen::Size),
    GlobalMem(ir::MemId),
    DivMagic(&'a codegen::Size, ir::Type),
    DivShift(&'a codegen::Size, ir::Type),
}

/// Generates the list of internal memory blocks, and creates the parameters needed to
/// back them.
fn register_mem_blocks<'a>(
    space: &'a SearchSpace,
    block_dims: &[Dimension<'a>],
) -> Vec<MemoryRegion> {
    let num_thread_blocks = block_dims.iter().fold(None, |pred, block| {
        if let Some(mut pred) = pred {
            pred *= block.size();
            Some(pred)
        } else {
            Some(block.size().clone())
        }
    });
    space
        .ir_instance()
        .mem_blocks()
        .map(|b| MemoryRegion::new(b, &num_thread_blocks, space))
        .collect()
}

/// A memory block allocated by the kernel.
pub struct MemoryRegion {
    id: ir::MemId,
    size: codegen::Size,
    num_private_copies: Option<codegen::Size>,
    mem_space: MemSpace,
    ptr_type: ir::Type,
}

/// Indicates how is a memory block allocated.
#[derive(PartialEq, Eq)]
pub enum AllocationScheme {
    Global,
    PrivatisedGlobal,
    Shared,
}

impl MemoryRegion {
    /// Creates a new MemoryRegion from an `ir::Mem`.
    pub fn new(
        block: &ir::mem::Block,
        num_threads_groups: &Option<codegen::Size>,
        space: &SearchSpace,
    ) -> Self {
        let mem_space = space.domain().get_mem_space(block.mem_id());
        assert!(mem_space.is_constrained());
        let mut size = codegen::Size::new(block.base_size(), vec![], 1);
        for &(dim, _) in block.mapped_dims() {
            let ir_size = space.ir_instance().dim(dim).size();
            size *= &codegen::Size::from_ir(ir_size, space);
        }
        let num_private_copies = if block.is_private() && mem_space == MemSpace::GLOBAL {
            num_threads_groups.clone()
        } else {
            None
        };
        let ptr_type = ir::Type::PtrTo(block.mem_id());
        let ptr_type = unwrap!(space.ir_instance().device().lower_type(ptr_type, space));
        MemoryRegion {
            id: block.mem_id(),
            size,
            mem_space,
            num_private_copies,
            ptr_type,
        }
    }

    /// Returns the value to pass from the host to the device to implement `self`.
    pub fn host_values(
        &self,
        space: &SearchSpace,
        block_dims: &[Dimension<'_>],
    ) -> Vec<ParamVal> {
        let mut out = if self.mem_space == MemSpace::GLOBAL {
            let t = ir::Type::PtrTo(self.id);
            let t = unwrap!(space.ir_instance().device().lower_type(t, space));
            vec![ParamVal::GlobalMem(self.id, self.alloc_size(), t)]
        } else {
            vec![]
        };
        let size = if self.num_private_copies.is_some() {
            Some(
                block_dims[1..]
                    .iter()
                    .map(|d| d.size())
                    .chain(std::iter::once(&self.size))
                    .flat_map(ParamVal::from_size),
            )
        } else {
            None
        };
        out.extend(size.into_iter().flat_map(|x| x));
        out
    }

    /// Returns the memory ID.
    pub fn id(&self) -> ir::MemId {
        self.id
    }

    /// Indicates how is the memory block allocated.
    pub fn alloc_scheme(&self) -> AllocationScheme {
        match self.mem_space {
            MemSpace::SHARED => AllocationScheme::Shared,
            MemSpace::GLOBAL if self.num_private_copies.is_some() => {
                AllocationScheme::PrivatisedGlobal
            }
            MemSpace::GLOBAL => AllocationScheme::Global,
            _ => unreachable!(),
        }
    }

    /// Generates the size of the memory to allocate.
    pub fn alloc_size(&self) -> codegen::Size {
        let mut out = self.size.clone();
        if let Some(ref s) = self.num_private_copies {
            out *= s
        }
        out
    }

    /// Returns the size of the part of the allocated memory accessible by each thread.
    pub fn local_size(&self) -> &codegen::Size {
        &self.size
    }

    /// Returns the memory space the block is allocated in.
    pub fn mem_space(&self) -> MemSpace {
        self.mem_space
    }

    /// Returns the type of the pointer to the memory block.
    pub fn ptr_type(&self) -> ir::Type {
        self.ptr_type
    }
}

/// An instruction to execute.
pub struct Instruction<'a> {
    instruction: &'a ir::Instruction,
    instantiation_dims: Vec<(ir::DimId, u32)>,
    mem_flag: Option<search_space::InstFlag>,
    t: Option<ir::Type>,
}

impl<'a> Instruction<'a> {
    /// Creates a new `Instruction`.
    pub fn new(instruction: &'a ir::Instruction, space: &SearchSpace) -> Self {
        let instantiation_dims = instruction
            .iteration_dims()
            .iter()
            .filter(|&&dim| {
                let kind = space.domain().get_dim_kind(dim);
                unwrap!(kind.is(DimKind::VECTOR | DimKind::UNROLL).as_bool())
            })
            .map(|&dim| {
                let size = space.ir_instance().dim(dim).size();
                (dim, unwrap!(codegen::Size::from_ir(size, space).as_int()))
            })
            .collect();
        let mem_flag = instruction
            .as_mem_inst()
            .map(|inst| space.domain().get_inst_flag(inst.id()));
        let t = instruction
            .t()
            .map(|t| unwrap!(space.ir_instance().device().lower_type(t, space)));
        Instruction {
            instruction,
            instantiation_dims,
            mem_flag,
            t,
        }
    }

    /// Returns the ID of the instruction.
    pub fn id(&self) -> ir::InstId {
        self.instruction.id()
    }

    /// Returns the values to pass from the host to implement this instruction.
    pub fn host_values<'b>(
        &'b self,
        space: &'b SearchSpace,
    ) -> impl Iterator<Item = ParamVal> + 'b {
        let operands = self.instruction.operator().operands();
        operands
            .into_iter()
            .flat_map(move |op| ParamVal::from_operand(op, space))
            .chain(
                self.instruction
                    .operator()
                    .predicates()
                    .flat_map(move |pred| {
                        ParamVal::from_size(&codegen::Size::from_ir(
                            &ir::PartialSize::from(pred.bound().clone()),
                            space,
                        ))
                    }),
            )
    }

    /// Returns the type of the instruction.
    pub fn t(&self) -> Option<ir::Type> {
        self.t
    }

    /// Returns the operator computed by the instruction.
    pub fn operator(&self) -> &ir::Operator {
        self.instruction.operator()
    }

    /// Returns the IR instruction from which this codegen instruction was created.
    pub fn ir_instruction(&self) -> &ir::Instruction {
        self.instruction
    }

    /// Returns the dimensions on which to instantiate the instruction.
    pub fn instantiation_dims(&self) -> &[(ir::DimId, u32)] {
        &self.instantiation_dims
    }

    /// Indicates if the instruction performs a reduction, in wich case it returns the
    /// instruction that initializes the reduction, the `DimMap` to readh it and the
    /// reduction dimensions.
    pub fn as_reduction(&self) -> Option<(ir::InstId, &ir::DimMap)> {
        self.instruction.as_reduction().map(|(x, y, _)| (x, y))
    }

    /// Returns the memory flag of the intruction, if any.
    pub fn mem_flag(&self) -> Option<search_space::InstFlag> {
        self.mem_flag
    }

    /// Indicates if the instruction has observable side effects.
    pub fn has_side_effects(&self) -> bool {
        self.instruction.has_side_effects()
    }

    /// Indicates where to store the result of the instruction.
    pub fn result_variable(&self) -> Option<ir::VarId> {
        self.instruction.result_variable()
    }
}

impl<'a> fmt::Display for Instruction<'a> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.instruction, fmt)
    }
}
