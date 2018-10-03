//! Provides a generic decription of basic blocks.
use ir;
use std;
use utils::*;

/// Provides a unique identifer for a basic block.
#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[repr(C)]
pub enum StmtId {
    /// cbindgen:field-names=[id]
    Inst(ir::InstId),
    /// cbindgen:field-names=[id]
    Dim(ir::DimId),
}

impl From<ir::InstId> for StmtId {
    fn from(id: ir::InstId) -> Self {
        StmtId::Inst(id)
    }
}

impl From<ir::DimId> for StmtId {
    fn from(id: ir::DimId) -> Self {
        StmtId::Dim(id)
    }
}

/// Represents a basic block in an Exhaust function.
pub trait Statement<'a, L = ir::LoweringMap>: std::fmt::Debug {
    /// Returns the unique identifier of the `Statement`.
    fn stmt_id(&self) -> StmtId;

    /// Returns 'self' if it is an instruction.
    fn as_inst(&self) -> Option<&ir::Instruction<'a, L>> {
        None
    }

    /// Returns 'self' if it is a dimension
    fn as_dim(&self) -> Option<&ir::Dimension<'a>> {
        None
    }

    /// Lists the values defined at this statement.
    fn def_values(&self) -> &VecSet<ir::ValueId>;

    /// Lists the values defined used at this statement.
    fn used_values(&self) -> &VecSet<ir::ValueId>;
}
