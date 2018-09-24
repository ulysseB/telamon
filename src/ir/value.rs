//! Encodes the data-flow information.
use ir;
use std;
use utils::*;

/// Uniquely identifies values.
#[derive(
    Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize,
)]
pub struct ValueId(pub u16);

impl From<ValueId> for usize {
    fn from(val_id: ValueId) -> Self {
        val_id.0 as usize
    }
}

/// A value produced by the code.
#[derive(Clone, Debug)]
pub struct Value {
    id: ValueId,
    t: ir::Type,
    def: ValueDef,
    use_points: VecSet<ir::StmtId>,
}

impl Value {
    /// Creates a new value with the given Id.
    pub fn new(id: ValueId, t: ir::Type, def: ValueDef) -> Self {
        Value {
            id,
            t,
            def,
            use_points: Default::default(),
        }
    }

    /// Return the unique identifiers of the `Value`.
    pub fn id(&self) -> ValueId {
        self.id
    }

    /// Specifies how the value is defined.
    pub fn def(&self) -> &ValueDef {
        &self.def
    }

    /// Indicates the type of the value.
    pub fn t(&self) -> ir::Type {
        self.t
    }

    /// Indicates the statements that define the value.
    pub fn def_points(&self) -> impl Iterator<Item=ir::StmtId> + '_ {
        self.def.def_statements()
    }

    /// Indicates the statements that uses the value.
    pub fn use_points(&self) -> impl Iterator<Item = ir::StmtId> + '_ {
        self.use_points.iter().cloned()
    }

    /// Registers that the value is used by a statement.
    pub fn add_use(&mut self, stmt: ir::StmtId) {
        self.use_points.insert(stmt);
    }
}

/// Specifies how is a `Value` defined.
#[derive(Clone, Debug, Copy)]
pub enum ValueDef {
    /// Takes the value produced by an instruction.
    Inst(ir::InstId),
    // TODO(value):
    // - Last
    // - Dim Map
    // - Fby
    // - ExternalMem
}

impl ValueDef {
    /// Registers the value in the structures it references in the function.
    pub fn register<L>(self, self_id: ir::ValueId, function: &mut ir::Function<L>) {
        let ValueDef::Inst(inst_id) = self;
        function.inst_mut(inst_id).set_result_value(self_id);
    }

    /// Returns the type of the value if used on the context of `function`.
    pub fn t<L>(self, fun: &ir::Function<L>) -> ir::Type {
        let ValueDef::Inst(inst_id) = self;
        unwrap!(fun.inst(inst_id).t())
    }

    /// Ensures the definition is valid.
    pub fn check<L>(self, fun: &ir::Function<L>) -> Result<(), ir::TypeError> {
        let ValueDef::Inst(inst) = self;
        if fun.inst(inst).t().is_none() {
            Err(ir::TypeError::ExpectedReturnType { inst })?;
        }
        Ok(())
    }

    /// Indicates in which statment the value is defined.
    pub fn def_statements(self) -> impl Iterator<Item = ir::StmtId> {
        let ValueDef::Inst(inst_id) = self;
        std::iter::once(inst_id.into())
    }
}
