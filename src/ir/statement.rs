//! Provides a generic decription of basic blocks.
use crate::ir;
use crate::utils::*;

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
pub trait Statement<'a, L = ir::LoweringMap> {
    /// Returns the unique identifier of the `Statement`.
    fn stmt_id(&self) -> StmtId;

    /// Returns 'self' if it is an instruction.
    fn as_inst(&self) -> Option<&ir::Instruction<'a, L>> {
        None
    }

    /// Returns 'self' if it is a dimension
    fn as_dim(&self) -> Option<&ir::Dimension<'a, L>> {
        None
    }

    /// Lists the variables defined at this statement.
    fn defined_vars(&self) -> &VecSet<ir::VarId>;

    /// Lists the variables defined used at this statement.
    fn used_vars(&self) -> &VecSet<ir::VarId>;

    /// Registers a variable use in this statement.
    fn register_defined_var(&mut self, var: ir::VarId);
}
