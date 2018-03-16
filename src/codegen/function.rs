//! Describes a `Function` that is ready to execute on a device.
use ir;
use ir::mem::Block;
use itertools::Itertools;
use search_space::{Domain, DomainStore, MemSpace, SearchSpace};
use codegen::{cfg, Cfg, dimension, Dimension, InductionVar, InductionLevel};
use utils::*;
use std;

/// A function ready to execute on a device, derived from a constrained IR instance.
pub struct Function<'a> {
    cfg: Cfg<'a>,
    thread_dims: Vec<Dimension<'a>>,
    block_dims: Vec<Dimension<'a>>,
    device_code_args: HashSet<ParamVal<'a>>,
    induction_vars: Vec<InductionVar<'a>>,
    mem_blocks: Vec<InternalMemBlock<'a>>,
    // TODO(cleanup): remove dependency on the search space
    space: &'a SearchSpace<'a>,
}

impl<'a> Function<'a> {
    /// Creates a device `Function` from an IR instance.
    pub fn build(space: &'a SearchSpace<'a>) -> Function<'a> {
        let mut dims = dimension::group_merged_dimensions(space);
        let (induction_vars, precomputed_indvar_levels) =
            dimension::register_induction_vars(&mut dims, space);
        let operands = space.ir_instance().insts().flat_map(|i| i.operands());
        let mut device_code_args = dims.iter().flat_map(|d| d.host_values())
            .chain(induction_vars.iter().flat_map(|v| v.host_values()))
            .chain(operands.flat_map(ParamVal::from_operand))
            .chain(precomputed_indvar_levels.iter().flat_map(|l| l.host_values()))
            .collect();
        let (block_dims, thread_dims, cfg) =
            cfg::build(space, dims, precomputed_indvar_levels);
        let mem_blocks = register_mem_blocks(space, &block_dims, &mut device_code_args);
        debug!("compiling cfg {:?}", cfg);
        Function {
            cfg, thread_dims, block_dims, induction_vars, device_code_args, space,
            mem_blocks,
        }
    }

    /// Returns the ordered list of thread dimensions.
    pub fn thread_dims(&self) -> &[Dimension<'a>] { &self.thread_dims }

    /// Returns the ordered list of block dimensions.
    pub fn block_dims(&self) -> &[Dimension<'a>] { &self.block_dims }

    /// Iterates other all `codegen::Dimension`.
    pub fn dimensions(&self) -> impl Iterator<Item=&Dimension> {
        self.cfg.dimensions().chain(&self.block_dims).chain(&self.thread_dims)
    }

    /// Returns the list of induction variables.
    pub fn induction_vars(&self) -> &[InductionVar<'a>] { &self.induction_vars }

    /// Returns the total number of threads to allocate.
    pub fn num_threads(&self) -> u32 {
        self.thread_dims.iter().map(|d| unwrap!(d.size().as_int())).product()
    }

    /// Returns the values to pass from the host to the device.
    pub fn device_code_args(&self) -> impl Iterator<Item=&ParamVal<'a>> {
        self.device_code_args.iter()
    }

    /// Returns the control flow graph.
    pub fn cfg(&self) -> &Cfg<'a> { &self.cfg }

    /// Returns all the induction levels in the function.
    pub fn induction_levels(&self) -> impl Iterator<Item=&InductionLevel> {
        self.dimensions().flat_map(|d| d.induction_levels())
            .chain(self.cfg.induction_levels())
    }

    /// Returns the memory blocks allocated by the function.
    pub fn mem_blocks(&self) -> impl Iterator<Item=&InternalMemBlock> {
        self.mem_blocks.iter()
    }

    // TODO(cleanup): remove unecessary methods
    /// Returns the parameters passed to the `Function` by the caller.
    pub fn external_params(&self) -> &[ir::Parameter] { &self.space.ir_instance().params }

    /// Returns the underlying domain.
    pub fn decisions(&self) -> &DomainStore { self.space.domain() }

    /// Returns the underlying implementation space.
    pub fn space(&self) -> &SearchSpace { self.space }
}

impl<'a> std::ops::Deref for Function<'a> {
    type Target = ir::Function<'a>;

    fn deref(&self) -> &ir::Function<'a> { self.space.ir_instance() }
}

/// Represents the value of a parameter passed to the kernel by the host.
#[derive(PartialEq, Eq, Hash)]
pub enum ParamVal<'a> {
    /// A parameter given by the caller.
    External(&'a ir::Parameter),
    /// A tiled dimension size computed on the host.
    Size(&'a ir::Size<'a>),
    /// A pointer to a global memory block, allocated by the wrapper.
    GlobalMem(ir::mem::InternalId, ir::Size<'a>),
}

impl<'a> ParamVal<'a> {
    /// Builds the `ParamVal` needed to implement an operand, if any.
    pub fn from_operand(operand: &'a ir::Operand<'a>) -> Option<Self> {
        match *operand {
            ir::Operand::Param(p) => Some(ParamVal::External(p)),
            ir::Operand::Size(ref s) => Self::from_size(s),
            _ => None,
        }
    }

    /// Builds the `ParamVal` needed to get a size value, if any.
    pub fn from_size(size: &'a ir::Size) -> Option<Self> {
        match *size.dividend() {
            [] => None,
            [p] if size.factor() == 1 && size.divisor() == 1 =>
                Some(ParamVal::External(p)),
            _ => Some(ParamVal::Size(size)),
        }
    }

    /// Returns the type of the parameter.
    pub fn t(&self) -> ir::Type {
        match *self {
            ParamVal::External(p) => p.t,
            ParamVal::Size(_) => ir::Type::I(32),
            ParamVal::GlobalMem(id, _) => ir::Type::PtrTo(id.into()),
        }
    }
}

/// Generates the list of internal memory blocks, and creates the parameters needed to
/// back them.
fn register_mem_blocks<'a>(space: &'a SearchSpace<'a>,
                           block_dims: &[Dimension<'a>],
                           device_code_args: &mut HashSet<ParamVal<'a>>)
    -> Vec<InternalMemBlock<'a>>
{
    let num_thread_blocks = block_dims.iter().fold(None, |pred, block| {
        if let Some(mut pred) = pred {
            pred *= block.size();
            Some(pred)
        } else { Some(block.size().clone()) }
    });
    let mem_blocks = space.ir_instance().internal_mem_blocks().map(|b| {
        InternalMemBlock::new(b, &num_thread_blocks, space)
    }).collect_vec();
    for block in &mem_blocks { device_code_args.extend(block.host_values()); }
    mem_blocks
}

/// A memory block allocated by the kernel.
pub struct InternalMemBlock<'a> {
    id: ir::mem::InternalId,
    size: &'a ir::Size<'a>,
    num_private_copies: Option<ir::Size<'a>>,
    mem_space: MemSpace,
}

/// Indicates how is a memory block allocated.
#[derive(PartialEq, Eq)]
pub enum AllocationScheme { Global, PrivatisedGlobal, Shared }

impl<'a> InternalMemBlock<'a> {
    /// Returns the value to pass from the host to the device to implement `self`.
    pub fn host_values(&self) -> impl Iterator<Item=ParamVal<'a>> {
        let ptr = if self.mem_space == MemSpace::GLOBAL {
            Some(ParamVal::GlobalMem(self.id, self.alloc_size()))
        } else { None };
        let size = if self.num_private_copies.is_some() {
            Some(ParamVal::Size(self.size))
        } else { None };
        ptr.into_iter().chain(size)
    }

    /// Returns the memory ID.
    pub fn id(&self) -> ir::mem::InternalId { self.id }

    /// Indicates how is the memory block allocated.
    pub fn alloc_scheme(&self) -> AllocationScheme {
        match self.mem_space {
            MemSpace::SHARED => AllocationScheme::Shared,
            MemSpace::GLOBAL if self.num_private_copies.is_some() =>
                AllocationScheme::PrivatisedGlobal,
            MemSpace::GLOBAL => AllocationScheme::Global,
            _ => unreachable!(),
        }
    }

    /// Generates the size of the memory to allocate.
    pub fn alloc_size(&self) -> ir::Size<'a> {
        let mut out = self.size.clone();
        if let Some(ref s) = self.num_private_copies { out *= s }
        out
    }

    /// Returns the size of the part of the allocated memory accessible by each thread.
    pub fn local_size(&self) -> &'a ir::Size<'a> { self.size }

    /// Returns the memory space the block is allocated in.
    pub fn mem_space(&self) -> MemSpace { self.mem_space }

    /// Creates a new InternalMemBlock from an `ir::mem::Internal`.
    fn new(block: &'a ir::mem::InternalBlock<'a>,
           num_threads_groups: &Option<ir::Size<'a>>,
           space: &'a SearchSpace<'a>) -> Self {
        let mem_space = space.domain().get_mem_space(block.mem_id());
        assert!(mem_space.is_constrained());
        let size = block.size();
        let num_private_copies = if block.is_private() && mem_space == MemSpace::GLOBAL {
            num_threads_groups.clone()
        } else { None };
        InternalMemBlock { id: block.id(), size, mem_space, num_private_copies }
    }
}
