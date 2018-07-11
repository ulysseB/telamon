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

impl From<Statement> for ChoiceDef {
    fn from(stmt: Statement) -> Self {
        match stmt {
            Statement::CounterDef { name, doc, visibility, vars, body } => {
                ChoiceDef::CounterDef(CounterDef {
                    name, doc, visibility, vars, body
                })
            },
            Statement::EnumDef(EnumDef { name: Spanned {
                beg, end, data, }, doc, variables, statements }) => {
                ChoiceDef::EnumDef(EnumDef {
                    name: Spanned {
                        beg, end, data,
                    }, doc, variables, statements
                })
            },
            Statement::IntegerDef(IntegerDef { name, doc, variables, code }) => {
                ChoiceDef::IntegerDef(IntegerDef {
                    name, doc, variables, code
                })
            },
            _ => unreachable!(),
        }
    }
}
