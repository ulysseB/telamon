//! Describes a `Function` that is ready to execute on a device.
use codegen::{self, cfg, dimension, Cfg, Dimension, InductionLevel, InductionVar};
use indexmap::IndexMap;
use ir;
use itertools::Itertools;
use search_space::*;
use std;
use utils::*;

/// A function ready to execute on a device, derived from a constrained IR instance.
pub struct Function<'a> {
    cfg: Cfg<'a>,
    thread_dims: Vec<Dimension<'a>>,
    block_dims: Vec<Dimension<'a>>,
    device_code_args: Vec<ParamVal<'a>>,
    induction_vars: Vec<InductionVar<'a>>,
    mem_blocks: Vec<MemoryRegion<'a>>,
    init_induction_levels: Vec<InductionLevel<'a>>,
    variables: IndexMap<ir::VarId, codegen::Variable<'a>>,
    // TODO(cleanup): remove dependency on the search space
    space: &'a SearchSpace<'a>,
}

impl<'a> Function<'a> {
    /// Creates a device `Function` from an IR instance.
    pub fn build(space: &'a SearchSpace<'a>) -> Function<'a> {
        let variables = codegen::variable::wrap_variables(space);
        let mut dims = dimension::group_merged_dimensions(space);
        let (induction_vars, init_induction_levels) =
            dimension::register_induction_vars(&mut dims, &variables, space);
        trace!("dims = {:?}", dims);
        let insts = space
            .ir_instance()
            .insts()
            .map(|inst| Instruction::new(inst, space))
            .collect_vec();
        let mut device_code_args = dims
            .iter()
            .flat_map(|d| d.host_values(space))
            .chain(induction_vars.iter().flat_map(|v| v.host_values(space)))
            .chain(insts.iter().flat_map(|i| i.host_values(space)))
            .chain(
                init_induction_levels
                    .iter()
                    .flat_map(|l| l.host_values(space)),
            ).collect::<HashSet<_>>();
        let (block_dims, thread_dims, cfg) = cfg::build(space, insts, dims);
        let mem_blocks = register_mem_blocks(space);
        device_code_args.extend(mem_blocks.iter().flat_map(|x| x.host_values()));
        debug!("compiling cfg {:?}", cfg);
        Function {
            cfg,
            thread_dims,
            block_dims,
            induction_vars,
            device_code_args: device_code_args.into_iter().collect(),
            space,
            mem_blocks,
            variables,
            init_induction_levels,
        }
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
        self.variables.values()
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
    pub fn device_code_args(&self) -> impl Iterator<Item = &ParamVal<'a>> {
        self.device_code_args.iter()
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

    /// Returns the induction levels computed at the beginning of the kernel. Levels must
    /// be computed in the provided order.
    pub fn init_induction_levels(&self) -> &[InductionLevel] {
        &self.init_induction_levels
    }

    /// Returns the stride between elements accessed by along a layout dimension.
    pub fn layout_dim_stride(&self, dim: ir::LayoutDimId) -> &codegen::Size<'a> {
        codegen::layout_dim_stride(dim, &self.variables, self.space)
    }
}

impl<'a> std::ops::Deref for Function<'a> {
    type Target = ir::Signature;

    fn deref(&self) -> &Self::Target {
        self.space.ir_instance()
    }
}

/// Represents the value of a parameter passed to the kernel by the host.
pub enum ParamVal<'a> {
    /// A parameter given by the caller.
    External(&'a ir::Parameter, ir::Type),
    /// A tiled dimension size computed on the host.
    Size(codegen::Size<'a>),
    /// A pointer to a global memory block, allocated by the wrapper.
    GlobalMem(ir::MemId, codegen::Size<'a>, ir::Type),
}

impl<'a> ParamVal<'a> {
    /// Builds the `ParamVal` needed to implement an operand, if any.
    pub fn from_operand(
        operand: &'a ir::Operand<'a>,
        space: &SearchSpace,
    ) -> Option<Self> {
        match *operand {
            ir::Operand::Param(p) => {
                let t = unwrap!(space.ir_instance().device().lower_type(p.t, space));
                Some(ParamVal::External(p, t))
            }
            _ => None,
        }
    }

    /// Builds the `ParamVal` needed to get a size value, if any.
    pub fn from_size(size: &codegen::Size<'a>) -> Option<Self> {
        match *size.dividend() {
            [] => None,
            [p] if size.factor() == 1 && size.divisor() == 1 => {
                Some(ParamVal::External(p, ir::Type::I(32)))
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

    /// Indicates if the parameter is a pointer.
    pub fn is_pointer(&self) -> bool {
        match *self {
            ParamVal::External(p, _) => matches!(p.t, ir::Type::PtrTo(_)),
            ParamVal::GlobalMem(..) => true,
            ParamVal::Size(_) => false,
        }
    }

    /// Returns a unique identifier for the `ParamVal`.
    pub fn key(&self) -> ParamValKey {
        match *self {
            ParamVal::External(p, _) => ParamValKey::External(p),
            ParamVal::Size(ref s) => ParamValKey::Size(s),
            ParamVal::GlobalMem(mem, ..) => ParamValKey::GlobalMem(mem),
        }
    }
}

hash_from_key!(ParamVal<'a>, ParamVal::key, 'a);

/// Uniquely identifies a `ParamVal`.
#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone)]
pub enum ParamValKey<'a> {
    External(&'a ir::Parameter),
    Size(&'a codegen::Size<'a>),
    GlobalMem(ir::MemId),
}

/// Generates the list of internal memory blocks, and creates the parameters needed to
/// back them.
fn register_mem_blocks<'a>(space: &'a SearchSpace<'a>) -> Vec<MemoryRegion<'a>> {
    let var_blocks = space
        .ir_instance()
        .memory_vars()
        .map(|var| MemoryRegion::new_var(var, space));
    space
        .ir_instance()
        .mem_blocks()
        .map(|b| MemoryRegion::new_fixed(b, space))
        .chain(var_blocks)
        .collect()
}

/// A memory block allocated by the kernel.
pub struct MemoryRegion<'a> {
    id: ir::ArrayId,
    len: codegen::Size<'a>,
    elements_type: ir::Type,
    memory_space: ir::MemorySpace,
    ptr_type: ir::Type,
}

impl<'a> MemoryRegion<'a> {
    /// Creates a new MemoryRegion from an `ir::Mem`.
    pub fn new_fixed(block: &ir::mem::Block, space: &SearchSpace) -> Self {
        let ptr_type = ir::Type::PtrTo(block.id.into());
        MemoryRegion {
            id: block.id.into(),
            len: codegen::Size::new(block.len, vec![], 1),
            elements_type: block.elements_type,
            memory_space: block.space,
            ptr_type: unwrap!(space.ir_instance().device().lower_type(ptr_type, space)),
        }
    }

    /// Creates a new memory region to store an `ir::Variable`.
    pub fn new_var(var: &ir::Variable, space: &SearchSpace<'a>) -> Self {
        let ptr_type = ir::Type::PtrTo(ir::ArrayId::Variable(var.id()));
        let len = var
            .layout()
            .iter()
            .filter(|&&layout_dim| {
                space.domain().get_is_instantiated(layout_dim) == IsInstantiated::TRUE
            }).map(|&layout_dim| {
                let dim = space.ir_instance().layout_dimension(layout_dim).dim();
                space.ir_instance().dim(dim).size()
            }).product();
        MemoryRegion {
            id: ir::ArrayId::Variable(var.id()),
            len: codegen::Size::from_ir(&len, space),
            elements_type: var.t(),
            memory_space: fixed_memory_space(space.domain().get_memory_space(var.id())),
            ptr_type: unwrap!(space.ir_instance().device().lower_type(ptr_type, space)),
        }
    }

    /// Returns the value to pass from the host to the device to implement `self`.
    pub fn host_values(&self) -> Option<ParamVal<'a>> {
        ParamVal::from_size(&self.size())
    }

    /// Returns the memory ID.
    pub fn id(&self) -> ir::ArrayId {
        self.id
    }

    /// Returns the size of the memory block in number of elements.
    pub fn len(&self) -> &codegen::Size<'a> {
        &self.len
    }

    /// Retrns the size of the memory in bytes.
    pub fn size(&self) -> codegen::Size<'a> {
        let element_size = unwrap!(self.elements_type.len_byte());
        let mut out = codegen::Size::new(element_size, vec![], 1);
        out *= &self.len;
        out
    }

    /// Indicates the types of the elements of the memory block.
    pub fn elements_type(&self) -> ir::Type {
        self.elements_type
    }

    /// Indicates where to allocate the memory block.
    pub fn memory_space(&self) -> ir::MemorySpace {
        self.memory_space
    }

    /// Returns the type of the pointer to the memory block.
    pub fn ptr_type(&self) -> ir::Type {
        self.ptr_type
    }
}

/// An instruction to execute.
pub struct Instruction<'a> {
    instruction: &'a ir::Instruction<'a>,
    instantiation_dims: Vec<(ir::DimId, u32)>,
    mem_access: Option<MemAccess<'a>>,
    dma_wait: Option<MemAccess<'a>>,
    t: Option<ir::Type>,
}

impl<'a> Instruction<'a> {
    /// Creates a new `Instruction`.
    pub fn new(instruction: &'a ir::Instruction<'a>, space: &SearchSpace) -> Self {
        let instantiation_dims = instruction
            .iteration_dims()
            .iter()
            .filter(|&&dim| {
                let kind = space.domain().get_dim_kind(dim);
                unwrap!(kind.is(DimKind::VECTOR | DimKind::UNROLL).as_bool())
            }).map(|&dim| {
                let size = space.ir_instance().dim(dim).size();
                (dim, unwrap!(codegen::Size::from_ir(size, space).as_int()))
            }).collect();
        let mem_access = instruction
            .operator()
            .mem_access_pattern()
            .map(|pattern| MemAccess::new(instruction, &pattern, space));
        let op = instruction.operator();
        let dma_wait = if let ir::op::DmaStart { dma_wait, .. } = op {
            let inst = space.ir_instance().inst(unwrap!(*dma_wait));
            if let ir::op::DmaWait { dst_pattern, .. } = inst.operator() {
                Some(MemAccess::new(inst, dst_pattern, space))
            } else {
                panic!("expected a DmaWait operator")
            }
        } else {
            None
        };
        let t = instruction
            .t()
            .map(|t| unwrap!(space.ir_instance().device().lower_type(t, space)));
        Instruction {
            instruction,
            instantiation_dims,
            mem_access,
            dma_wait,
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
        space: &'a SearchSpace<'a>,
    ) -> impl Iterator<Item = ParamVal<'a>> {
        let operands = self.instruction.operator().operands();
        let mem_access_stride = self
            .mem_access()
            .and_then(|x| x.stride.as_ref())
            .and_then(ParamVal::from_size);
        operands
            .into_iter()
            .flat_map(move |op| ParamVal::from_operand(op, space))
            .chain(mem_access_stride)
    }

    /// Returns the type of the instruction.
    pub fn t(&self) -> Option<ir::Type> {
        self.t
    }

    /// Returns the operator computed by the instruction.
    pub fn operator(&self) -> &ir::Operator {
        self.instruction.operator()
    }

    /// Returns the dimensions on which to instantiate the instruction.
    pub fn instantiation_dims(&self) -> &[(ir::DimId, u32)] {
        &self.instantiation_dims
    }

    /// Returns a description of how the instruction accesses the memory.
    pub fn mem_access(&self) -> Option<&MemAccess<'a>> {
        self.mem_access.as_ref()
    }

    /// Indicates how the DMA wait link to this instruction, if any, accesses the memory.
    pub fn dma_wait_access(&self) -> Option<&MemAccess<'a>> {
        self.dma_wait.as_ref()
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

/// Describes a memory access.
#[derive(Debug)]
pub struct MemAccess<'a> {
    pub stride: Option<codegen::Size<'a>>,
    pub space: ir::MemorySpace,
    pub flag: InstFlag,
}

impl<'a> MemAccess<'a> {
    pub fn new(
        inst: &ir::Instruction,
        access_pattern: &ir::AccessPattern,
        space: &SearchSpace,
    ) -> Self {
        let layout_dims = inst
            .mem_access_layout()
            .iter()
            .map(|&id| {
                let dim = space.ir_instance().layout_dimension(id);
                let rank_universe = unwrap!(dim.possible_ranks());
                let rank = space.domain().get_rank(id).as_constrained(rank_universe);
                (dim, unwrap!(rank))
            }).sorted_by(|(_, lhs), (_, rhs)| std::cmp::Ord::cmp(lhs, rhs));
        let first_outer_vec = layout_dims.iter().find(|(dim, _)| {
            space.domain().get_dim_kind(dim.dim()) == DimKind::OUTER_VECTOR
        });
        let last_inner_vec = layout_dims
            .iter()
            .rev()
            .find(|(dim, _)| {
                space.domain().get_dim_kind(dim.dim()) == DimKind::OUTER_VECTOR
            }).map_or(0, |&(_, rank)| rank);
        if let Some(&(dim, rank)) = first_outer_vec {
            let is_strided = rank > last_inner_vec + 1 || dim.is_strided();
            assert!(!is_strided, "strided access are not supported yet");
        }
        let array = access_pattern.accessed_array();
        MemAccess {
            stride: None,
            space: fixed_memory_space(array_memory_space(array, space)),
            flag: space.domain().get_inst_flag(inst.id()),
        }
    }
}

/// Converts a `SearchSpace::MemorySpace` into an `ir::MemorySpace`.
fn fixed_memory_space(space: MemorySpace) -> ir::MemorySpace {
    match space {
        MemorySpace::GLOBAL => ir::MemorySpace::Global,
        MemorySpace::SHARED => ir::MemorySpace::Shared,
        space => panic!("invalid memory space {:?}", space),
    }
}
