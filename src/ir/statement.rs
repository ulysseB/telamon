//! Provides a generic decription of basic blocks.
use ir;
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

/// Either a `StmtId` or a mutable reference to a statement. This is useful to abstract
/// over a statement that is either register in the functions or being created.
pub enum IdOrMut<'a: 'b, 'b> {
    Id(StmtId),
    Mut(&'b mut Statement<'a>),
}

impl<'a, 'b> From<StmtId> for IdOrMut<'a, 'b> {
    fn from(id: StmtId) -> Self {
        IdOrMut::Id(id)
    }
}

impl<'a, 'b> From<&'b mut Statement<'a>> for IdOrMut<'a, 'b> {
    fn from(stmt: &'b mut Statement<'a>) -> Self {
        IdOrMut::Mut(stmt)
    }
}

impl<'a, 'b> IdOrMut<'a, 'b> {
    /// Returns a mutable reference to the `Statement`.
    pub fn get_statement<'c, L>(
        &'c mut self,
        fun: &'c mut ir::Function<'a, L>,
    ) -> &'c mut Statement<'a> {
        match self {
            IdOrMut::Id(id) => fun.statement_mut(*id),
            IdOrMut::Mut(stmt) => *stmt,
        }
    }

    /// Returns the id of the `Statement`.
    pub fn id(&self) -> StmtId {
        match self {
            IdOrMut::Id(id) => *id,
            IdOrMut::Mut(stmt) => stmt.stmt_id(),
        }
    }
}

/// Represents a basic block in an Exhaust function.
pub trait Statement<'a> {
    /// Returns the unique identifier of the `Statement`.
    fn stmt_id(&self) -> StmtId;

    /// Returns 'self' if it is an instruction.
    fn as_inst(&self) -> Option<&ir::Instruction<'a>> {
        None
    }

    /// Returns 'self' if it is a dimension
    fn as_dim(&self) -> Option<&ir::Dimension<'a>> {
        None
    }

    /// Lists the variables defined at this statement.
    fn defined_vars(&self) -> &VecSet<ir::VarId>;

    /// Lists the variables defined used at this statement.
    fn used_vars(&self) -> &VecSet<ir::VarId>;

    /// Registers a variable use in this statement.
    fn register_defined_var(&mut self, var: ir::VarId);

    /// Registers a variables used as initialization by a `fby` along this dimensions.
    fn register_used_var(&mut self, var: ir::VarId);
}
