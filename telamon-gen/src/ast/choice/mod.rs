pub mod enumeration;
pub mod integer;

pub use self::enumeration::EnumDef;
pub use self::integer::IntegerDef;
pub use super::*;

#[derive(Debug, PartialEq, Clone)]
pub enum ChoiceDef {
    CounterDef(CounterDef),
    EnumDef(EnumDef),
    IntegerDef(IntegerDef),
}

impl ChoiceDef {
    pub fn declare(&self, context: &mut CheckerContext) -> Result<(), TypeError> {
        match self {
            ChoiceDef::IntegerDef(choice) => {
                context.declare_choice(choice.name.to_owned(), Hint::Integer)
            }
            ChoiceDef::EnumDef(choice) => {
                context.declare_choice(choice.name.to_owned(), Hint::Enum)
            }
            ChoiceDef::CounterDef(choice) => context.declare_choice(
                choice.name.with_data(choice.name.data.to_string()),
                Hint::Counter,
            ),
        }
    }

    pub fn define(
        self,
        context: &mut CheckerContext,
        tc: &mut TypingContext,
    ) -> Result<(), TypeError> {
        match self {
            ChoiceDef::CounterDef(_) => Ok(()),
            ChoiceDef::IntegerDef(integer_def) => integer_def.define(context, tc),
            ChoiceDef::EnumDef(enum_def) => enum_def.define(context, tc),
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
