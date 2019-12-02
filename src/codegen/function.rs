//! Describes a `Function` that is ready to execute on a device.
use std::fmt;
use std::sync::Arc;

use fxhash::FxHashSet;
use itertools::Itertools;
use log::{debug, trace};
use utils::*;

use crate::codegen::{self, cfg, dimension, Cfg, Dimension};
use crate::ir::{self, IrDisplay};
use crate::search_space::{self, DimKind, Domain, MemSpace, SearchSpace};

/// A function ready to execute on a device, derived from a constrained IR instance.
pub struct Function<'a> {
    cfg: Cfg<'a>,
    thread_dims: Vec<Dimension>,
    block_dims: Vec<Dimension>,
    induction_vars: Vec<(ir::IndVarId, super::expr::ExprPtr)>,
    accesses: Vec<(ir::AccessId, super::expr::ExprPtr)>,
    device_code_args: Vec<ParamVal>,
    mem_blocks: Vec<MemoryRegion>,
    variables: Vec<codegen::Variable<'a>>,
    // TODO(cleanup): remove dependency on the search space
    space: &'a SearchSpace,
}

impl<'a> Function<'a> {
    /// Creates a device `Function` from an IR instance.
    pub fn build(space: &'a SearchSpace) -> Function<'a> {
        let dims = dimension::group_merged_dimensions(space);
        trace!("dims = {:?}", dims);

        let insts = space
            .ir_instance()
            .insts()
            .map(|inst| Instruction::new(inst, space))
            .collect_vec();
        let mut device_code_args = dims
            .iter()
            .flat_map(|d| d.host_values(space))
            .chain(insts.iter().flat_map(|i| i.host_values(space)))
            .collect::<FxHashSet<_>>();
        let (block_dims, thread_dims, cfg) = cfg::build(space, insts, dims);
        let merged_dims: dimension::MergedDimensions<'_> = cfg
            .dimensions()
            .chain(&block_dims)
            .chain(&thread_dims)
            .collect();

        let mut lowering =
            super::expr::ExprLowering::new(space, &merged_dims, &mut device_code_args);
        let induction_vars = space
            .ir_instance()
            .induction_vars()
            .map(|(id, ind_var)| (id, lowering.induction_var(ind_var)))
            .collect::<Vec<_>>();

        let accesses = space
            .ir_instance()
            .accesses()
            .iter()
            .map(|(aid, access)| (aid, lowering.access(access)))
            .collect::<Vec<_>>();

        device_code_args.extend(
            induction_vars
                .iter()
                .flat_map(|(_, expr)| expr.host_values(space, &merged_dims))
                .chain(
                    accesses
                        .iter()
                        .flat_map(|(_, expr)| expr.host_values(space, &merged_dims)),
                ),
        );

        let mem_blocks = register_mem_blocks(space, &block_dims);
        device_code_args.extend(
            mem_blocks
                .iter()
                .flat_map(|x| x.host_values(space, &block_dims)),
        );

        debug!("compiling cfg {:?}", cfg);
        Function {
            cfg,
            thread_dims,
            block_dims,
            induction_vars,
            accesses,
            device_code_args: device_code_args.into_iter().collect(),
            space,
            mem_blocks,
            variables: codegen::variable::wrap_variables(space),
        }
    }

    /// Returns the ordered list of thread dimensions.
    pub fn thread_dims(&self) -> &[Dimension] {
        &self.thread_dims
    }

    /// Returns the ordered list of block dimensions.
    pub fn block_dims(&self) -> &[Dimension] {
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

    /// Returns the control flow graph.
    pub fn cfg(&self) -> &Cfg<'a> {
        &self.cfg
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

    pub fn induction_vars(&self) -> &[(ir::IndVarId, super::expr::ExprPtr)] {
        &self.induction_vars
    }

    pub fn accesses(&self) -> &[(ir::AccessId, super::expr::ExprPtr)] {
        &self.accesses
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
}

impl ParamVal {
    /// Builds the `ParamVal` needed to implement an operand, if any.
    pub fn from_operand(operand: &ir::Operand, space: &SearchSpace) -> Option<Self> {
        match operand {
            ir::Operand::Param(p) => {
                let t = unwrap!(space.ir_instance().device().lower_type(p.t, space));
                Some(ParamVal::External(p.clone(), t))
            }
            _ => None,
        }
    }

    /// Builds the `ParamVal` needed to get a size value, if any.
    pub fn from_size(size: &codegen::Size) -> Option<Self> {
        match size.dividend() {
            [] => None,
            [p] if size.factor() == 1 && size.divisor() == 1 => {
                Some(ParamVal::External(p.clone(), ir::Type::I(32)))
            }
            _ => Some(ParamVal::Size(size.clone())),
        }
    }

    /// Returns the type of the parameter.
    pub fn t(&self) -> ir::Type {
        match *self {
            ParamVal::External(_, t) | ParamVal::GlobalMem(.., t) => t,
            ParamVal::Size(_) => ir::Type::I(32),
        }
    }

    pub fn elem_t(&self) -> Option<ir::Type> {
        match *self {
            ParamVal::External(ref p, _) => p.elem_t,
            ParamVal::GlobalMem(_, _, elem_t) => Some(elem_t),
            ParamVal::Size(_) => None,
        }
    }

    /// Returns a unique identifier for the `ParamVal`.
    pub fn key(&self) -> ParamValKey {
        match *self {
            ParamVal::External(ref p, _) => ParamValKey::External(&**p),
            ParamVal::Size(ref s) => ParamValKey::Size(s),
            ParamVal::GlobalMem(mem, ..) => ParamValKey::GlobalMem(mem),
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
}

impl fmt::Display for ParamValKey<'_> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParamValKey::External(p) => fmt.write_str(&p.name),
            ParamValKey::Size(size) => fmt::Display::fmt(size, fmt),
            ParamValKey::GlobalMem(memid) => write!(fmt, "pointer to {}", memid),
        }
    }
}

impl codegen::IdentDisplay for ParamValKey<'_> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            ParamValKey::External(p) => fmt.write_str(&p.name),
            ParamValKey::Size(size) => codegen::IdentDisplay::fmt(size, fmt),
            ParamValKey::GlobalMem(mem) => write!(fmt, "_gbl_mem_{}", mem.0),
        }
    }
}

/// Generates the list of internal memory blocks, and creates the parameters needed to
/// back them.
fn register_mem_blocks<'a>(
    space: &'a SearchSpace,
    block_dims: &[Dimension],
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
        block_dims: &[Dimension],
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
    pub fn host_values(
        &self,
        space: &'a SearchSpace,
    ) -> impl Iterator<Item = ParamVal> + 'a {
        let operands = self.instruction.operator().operands();
        operands
            .into_iter()
            .flat_map(move |op| ParamVal::from_operand(op, space))
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
