//! Representation and manipulation of a set of possible implementation.
mod access_pattern;
mod basic_block;
mod dimension;
mod dim_map;
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
pub use self::dimension::Dimension;
pub use self::function::{Function, Signature, Parameter};
pub use self::instruction::{InstId, Instruction};
pub use self::induction_var::{IndVarId, InductionVar};
pub use self::operand::{Operand, DimMapScope};
pub use self::operator::Operator;
pub use self::size::Size;
pub use self::types::Type;

pub mod mem;

/// Defines iteration dimensions properties.
pub mod dim {
    pub use super::dimension::Id;
    pub use super::dim_map::DimMap as Map;
}

/// Defines operators.
pub mod op {
    pub use super::operator::Operator::*;
    pub use super::operator::Rounding;
}

/// Defines traits to import in the environment to use the IR.
pub mod prelude {
    pub use ir::basic_block::BasicBlock;
    pub use ir::mem::Block as MemBlock;
}

// TODO(perf): group static computations
