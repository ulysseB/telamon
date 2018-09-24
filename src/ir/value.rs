//! Encodes the data-flow information.
use ir::{self, InstId};
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
    usepoints: HashSet<InstId>,
}

impl Value {
    pub fn new(id: ValueId, t: ir::Type, def: ValueDef) -> Self {
        Value {
            id,
            t,
            def,
            usepoints: HashSet::default(),
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

    pub fn usepoints(&self) -> impl Iterator<Item = &InstId> {
        self.usepoints.iter()
    }

    pub fn add_usepoint(&mut self, usepoint: InstId) {
        self.usepoints.insert(usepoint);
    }

    pub fn is_dependency_of(&self, usepoint: InstId) -> bool {
        self.usepoints.contains(&usepoint)
    }

    /// Returns the id of the instruction that produced the value if it was indeed produced by an
    /// instruction
    pub fn prod_inst_id(&self) -> Option<ir::InstId> {
        self.def().prod_inst_id()
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
    pub fn register<L>(self, self_id: ir::ValueId, function: &mut ir::Function<'_, L>) {
        // TODO change this code when we add new variant for ValueDef
        let ValueDef::Inst(inst_id) = self;
        function.inst_mut(inst_id).set_result_value(self_id);
    }

    /// Returns the id of the instruction that produced the value if it was indeed produced by an
    /// instruction
    pub fn prod_inst_id(&self) -> Option<ir::InstId> {
        let ValueDef::Inst(id) = self;
        Some(*id)
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
