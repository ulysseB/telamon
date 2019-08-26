mod counter;
mod enumeration;
mod integer;

pub use self::counter::CounterDef;
pub use self::enumeration::EnumDef;
pub use self::integer::IntegerDef;

use crate::ast::context::CheckerContext;
use crate::ast::error::{Hint, TypeError};
use crate::ast::{Constraint, Statement};

use crate::ir;

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
        ir_desc: &mut ir::IrDesc,
        constraints: &mut Vec<Constraint>,
        choice_defs: &mut Vec<ChoiceDef>,
    ) -> Result<(), TypeError> {
        match self {
            ChoiceDef::CounterDef(def) => def.define(context, choice_defs),
            ChoiceDef::IntegerDef(def) => def.define(context, ir_desc),
            ChoiceDef::EnumDef(def) => def.define(context, ir_desc, constraints),
        }
    }
}

impl From<Statement> for ChoiceDef {
    fn from(stmt: Statement) -> Self {
        match stmt {
            Statement::ChoiceDef(def) => *def,
            _ => unreachable!(),
        }
    }
}
