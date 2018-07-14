pub mod enumeration;
pub mod integer;

pub use super::*;
pub use self::enumeration::EnumDef;
pub use self::integer::IntegerDef;

#[derive(Clone, Debug, PartialEq)]
pub enum ChoiceDef {
    CounterDef(CounterDef),
    EnumDef(EnumDef),
    IntegerDef(IntegerDef),
}

impl ChoiceDef {
    pub fn type_check(&self) -> Result<(), TypeError> {
        match self {
            ChoiceDef::CounterDef(_) => Ok(()),
            ChoiceDef::IntegerDef(integer_def) => integer_def.type_check(),
            ChoiceDef::EnumDef(enum_def) => enum_def.type_check(),
        }
    }
}

impl From<Statement> for ChoiceDef {
    fn from(stmt: Statement) -> Self {
        match stmt {
            Statement::ChoiceDef(def) => def,
            _ => unreachable!(),
        }
    }
}
