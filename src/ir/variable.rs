//! Encodes the data-flow information.
use ir;
use std;
use utils::*;

/// Uniquely identifies variables.
#[derive(
    Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize,
)]
pub struct VarId(pub u16);

impl From<VarId> for usize {
    fn from(val_id: VarId) -> Self {
        val_id.0 as usize
    }
}

/// A variable produced by the code.
#[derive(Clone, Debug)]
pub struct Variable {
    id: VarId,
    t: ir::Type,
    def: VarDef,
    use_points: VecSet<ir::StmtId>,
}

impl Variable {
    /// Creates a new variable with the given Id.
    pub fn new(id: VarId, t: ir::Type, def: VarDef) -> Self {
        Variable {
            id,
            t,
            def,
            use_points: Default::default(),
        }
    }

    /// Return the unique identifiers of the `Variable`.
    pub fn id(&self) -> VarId {
        self.id
    }

    /// Specifies how the variable is defined.
    pub fn def(&self) -> &VarDef {
        &self.def
    }

    /// Indicates the type of the variable.
    pub fn t(&self) -> ir::Type {
        self.t
    }

    /// Indicates the statements that define the variable.
    pub fn def_points(&self) -> impl Iterator<Item = ir::StmtId> + '_ {
        self.def.def_statements()
    }

    /// Indicates the statements that uses the variable.
    pub fn use_points(&self) -> impl Iterator<Item = ir::StmtId> + '_ {
        self.use_points.iter().cloned()
    }

    /// Registers that the variable is used by a statement.
    pub fn add_use(&mut self, stmt: ir::StmtId) {
        self.use_points.insert(stmt);
    }

    /// Returns the id of the instruction that produced the value if it was indeed produced by an
    /// instruction
    pub fn prod_inst_id(&self) -> Option<ir::InstId> {
        self.def().prod_inst_id()
    }
}

/// Specifies how is a `Variable` defined.
#[derive(Clone, Debug, Copy)]
pub enum VarDef {
    /// Takes the variable produced by an instruction.
    Inst(ir::InstId),
    // TODO(value):
    // - Last
    // - Dim Map
    // - Fby
    // - ExternalMem
}

impl VarDef {
    /// Registers the variable in the structures it references in the function.
    pub fn register<L>(self, self_id: VarId, function: &mut ir::Function<L>) {
        let VarDef::Inst(inst_id) = self;
        function.inst_mut(inst_id).set_result_variable(self_id);
    }

    /// Returns the type of the variable if used on the context of `function`.
    pub fn t<L>(self, fun: &ir::Function<L>) -> ir::Type {
        let VarDef::Inst(inst_id) = self;
        unwrap!(fun.inst(inst_id).t())
    }

    /// Ensures the definition is valid.
    pub fn check<L>(self, fun: &ir::Function<L>) -> Result<(), ir::TypeError> {
        let VarDef::Inst(inst) = self;
        if fun.inst(inst).t().is_none() {
            Err(ir::TypeError::ExpectedReturnType { inst })?;
        }
        Ok(())
    }

    /// Indicates in which statment the variable is defined.
    pub fn def_statements(self) -> impl Iterator<Item = ir::StmtId> {
        let VarDef::Inst(inst_id) = self;
        std::iter::once(inst_id.into())
    }

    /// Returns the id of the instruction that produced the value if it was indeed produced by an
    /// instruction
    pub fn prod_inst_id(&self) -> Option<ir::InstId> {
        let VarDef::Inst(id) = self;
        Some(*id)
    }
}
