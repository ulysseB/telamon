//! Encodes the data-flow information.
use ir;

/// Uniquely identifies values.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize,
         Deserialize)]
pub struct ValueId(pub u16);

impl Into<usize> for ValueId {
    fn into(self) -> usize {
        self.0 as usize
    }
}

/// A value produced by the code.
#[derive(Clone, Debug)]
pub struct Value {
    id: ValueId,
    t: ir::Type,
    def: ValueDef,
}

impl Value {
    pub fn new(id: ValueId, t: ir::Type, def: ValueDef) -> Self {
        Value { id, t, def }
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
    pub fn t(&self) -> &ir::Type {
        &self.t
    }
}

/// Specifies how is a `Value` defined.
#[derive(Clone, Debug)]
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
    pub fn register<L>(&self, self_id: ir::ValueId, function: &mut ir::Function<'_, L>) {
        // TODO change this code when we add new variant for ValueDef
        let inst_id = match self {
            ValueDef::Inst(id) => id,
        };
        function.inst_mut(*inst_id).set_result_value(self_id);
    }
}
// - register in the instruction
// - retrieve the type
// FIXME: def point
// FIXME: use points
// FIXME: def dims
//  - use is in def dims ?
//  - use is before def
// FIXME: value in operand position
