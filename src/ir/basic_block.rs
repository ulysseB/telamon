//! Provides a generic decription of basic blocks.
use ir;
use std;

/// Provides a unique identifer for a basic block.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum BBId { Inst(ir::InstId), Dim(ir::DimId) }

impl From<ir::InstId> for BBId {
    fn from(id: ir::InstId) -> Self { BBId::Inst(id) }
}

impl From<ir::DimId> for BBId {
    fn from(id: ir::DimId) -> Self { BBId::Dim(id) }
}

/// Represents a basic block in an Exhaust function.
pub trait BasicBlock<'a>: std::fmt::Debug {
    /// Returns the unique identifier of the `BasicBlock`.
    fn bb_id(&self) -> BBId;
    /// Returns 'self' if it is an instruction.
    fn as_inst(&self) -> Option<&ir::Instruction<'a>> { None }
    /// Returns 'self' if it is a dimension
    fn as_dim(&self) -> Option<&ir::Dimension<'a>> { None }
}
