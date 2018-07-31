//! Representation and manipulation of a set of possible implementation.
mod access_pattern;
mod basic_block;
mod dimension;
mod dim_map;
mod error;
mod function;
mod induction_var;
mod instruction;
mod operand;
mod operator;
mod size;
mod types;

pub use self::access_pattern::{Stride, AccessPattern};
pub use self::basic_block::{BasicBlock, BBId};
pub use self::dim_map::DimMap;
pub use self::dimension::{DimId, Dimension};
pub use self::error::{Error, TypeError};
pub use self::function::{Function, Signature, Parameter};
pub use self::instruction::{InstId, Instruction};
pub use self::induction_var::{IndVarId, InductionVar};
pub use self::mem::MemId;
pub use self::operand::{Operand, DimMapScope};
pub use self::operator::{BinOp, Operator};
pub use self::size::Size;
pub use self::types::Type;

pub mod mem;

/// Defines iteration dimensions properties.
pub mod dim {
    pub use super::dim_map::DimMap as Map;
}

/// Defines operators.
pub mod op {
    pub use super::operator::Operator::*;
    pub use super::operator::Rounding;
    pub use super::operator::BinOp;
}

/// Defines traits to import in the environment to use the IR.
pub mod prelude {
    pub use ir::basic_block::BasicBlock;
    pub use ir::mem::Block as MemBlock;
}

/// Stores the objects created by a lowering.
#[derive(Default, Debug)]
pub struct NewObjs {
    pub instructions: Vec<InstId>,
    pub dimensions: Vec<DimId>,
    pub basic_blocks: Vec<BBId>,
    pub mem_blocks: Vec<MemId>,
    pub internal_mem_blocks: Vec<mem::InternalId>,
    pub mem_insts: Vec<InstId>,
    pub iteration_dims: Vec<(InstId, DimId)>,
    pub thread_dims: Vec<DimId>,
}

impl NewObjs {
    /// Registers a new instruction.
    pub fn add_instruction(&mut self, inst: &Instruction) {
        self.add_bb(inst);
        for &dim in inst.iteration_dims() { self.iteration_dims.push((inst.id(), dim)); }
        self.instructions.push(inst.id());
    }

    /// Registers a new memory instruction.
    pub fn add_mem_instruction(&mut self, inst: &Instruction) {
        self.add_instruction(inst);
        self.mem_insts.push(inst.id());
    }

    /// Registers a new dimension.
    pub fn add_dimension(&mut self, dim: &Dimension) {
        self.add_bb(dim);
        self.dimensions.push(dim.id());
        if dim.is_thread_dim() { self.add_thread_dim(dim.id()); }
    }

    /// Registers a new basic block.
    pub fn add_bb(&mut self, bb: &BasicBlock) {
        self.basic_blocks.push(bb.bb_id());
    }

    /// Sets a dimension as a new iteration dimension.
    pub fn add_iteration_dim(&mut self, inst: InstId, dim: DimId) {
        self.iteration_dims.push((inst, dim));
    }

    /// Sets a dimension as a new thread dimension.
    pub fn add_thread_dim(&mut self, dim: DimId) { self.thread_dims.push(dim) }

    /// Registers a new memory block.
    pub fn add_mem_block(&mut self, id: mem::InternalId) {
        self.mem_blocks.push(id.into());
        self.internal_mem_blocks.push(id);
    }
}

/// A point-to-point communication lowered into a store and a load.
pub struct LoweredDimMap {
    pub mem: mem::InternalId,
    pub store: InstId,
    pub load: InstId,
    pub dimensions: Vec<(DimId, DimId)>
}

// TODO(perf): group static computations
