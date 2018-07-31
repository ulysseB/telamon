pub mod enumeration;
pub mod integer;

pub use super::*;
pub use self::enumeration::EnumDef;
pub use self::integer::IntegerDef;

#[derive(Debug, PartialEq, Clone)]
pub enum ChoiceDef {
    CounterDef(CounterDef),
    EnumDef(EnumDef),
    IntegerDef(IntegerDef),
}

impl ChoiceDef {
    pub fn declare(&self) -> Result<(), TypeError> {
        match self {
            ChoiceDef::CounterDef(_) => Ok(()),
            ChoiceDef::IntegerDef(integer_def) => integer_def.declare(),
            ChoiceDef::EnumDef(enum_def) => enum_def.declare(),
        }
    }

    pub fn define(&self) -> Result<(), TypeError> {
        match self {
            ChoiceDef::CounterDef(_) => Ok(()),
            ChoiceDef::IntegerDef(integer_def) => integer_def.define(),
            ChoiceDef::EnumDef(enum_def) => enum_def.define(),
        }
    }

    pub fn get_name(&self) -> Spanned<String> {
        match self {
            ChoiceDef::IntegerDef(choice) => choice.name.to_owned(),
            ChoiceDef::EnumDef(choice) => choice.name.to_owned(),
            ChoiceDef::CounterDef(choice) => choice.name.with_data(choice.name.data.to_string()),
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
