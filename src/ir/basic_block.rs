//! Provides a generic decription of basic blocks.
use ir;
use std;
use utils::*;

/// Provides a unique identifer for a basic block.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BBId { Inst(ir::InstId), Dim(ir::dim::Id) }

impl From<ir::InstId> for BBId {
    fn from(id: ir::InstId) -> Self { BBId::Inst(id) }
}

impl From<ir::dim::Id> for BBId {
    fn from(id: ir::dim::Id) -> Self { BBId::Dim(id) }
}

impl std::fmt::Debug for BBId {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            BBId::Inst(id) => write!(f, "inst {}", id.id),
            BBId::Dim(id) => write!(f, "dim {}", id.id),
        }
    }
}

/// Represents a basic block in an Exhaust function.
pub trait BasicBlock<'a>: std::fmt::Debug {
    /// Returns the unique identifier of the `BasicBlock`.
    fn bb_id(&self) -> BBId;
    /// Returns 'self' if it is an instruction.
    fn as_inst(&self) -> Option<&ir::Instruction<'a>> { None }
    /// Returns 'self' if it is a dimension
    fn as_dim(&self) -> Option<&ir::Dimension<'a>> { None }
    /// The list of dimensions the instruction must be nested in.
    fn iteration_dims(&self) -> &HashSet<ir::dim::Id>;
    /// Adds a new iteration dimension. Indicates if the dimension was not already an
    /// iteration dimension.
    fn add_iteration_dimension(&mut self, dim: ir::dim::Id) -> bool;
}
