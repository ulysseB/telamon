use ast::*;
use ir;

/// A toplevel definition or constraint.
#[derive(Debug)]
pub struct EnumDef {
    pub name: String,
    pub doc: Option<String>,
    pub variables: Vec<VarDef>,
    pub statements: Vec<EnumStatement>
}

impl PartialEq for EnumDef {
    fn eq(&self, rhs: &Self) -> bool {
        self.name == rhs.name
    }
}

#[derive(Debug)]
pub struct TriggerDef {
    pub foralls: Vec<VarDef>,
    pub conditions: Vec<Condition>,
    pub code: String,
}

#[derive(Debug)]
pub struct CounterDef {
    pub name: RcStr,
    pub doc: Option<String>,
    pub visibility: ir::CounterVisibility,
    pub vars: Vec<VarDef>,
    pub body: CounterBody,
}

impl PartialEq for CounterDef {
    fn eq(&self, rhs: &Self) -> bool {
        self.name == rhs.name
    }
}

#[derive(Debug, PartialEq)]
pub enum ChoiceDef {
    CounterDef(CounterDef),
    EnumDef(EnumDef),
}

impl From<Statement> for Result<ChoiceDef, TypeError> {
    fn from(stmt: Statement) -> Self {
        match stmt {
            Statement::CounterDef { name, doc, visibility, vars, body } => {
                Ok(ChoiceDef::CounterDef(CounterDef {
                    name, doc, visibility, vars, body
                }))
            },
            Statement::EnumDef { name, doc, variables, statements } => {
                    if let Some(statement) = statements.iter()
                             .find(|item|
                                 statements.iter()
                                     .skip_while(|subitem| subitem != item)
                                     .skip(1)
                                     .any(|ref subitem| subitem == item))
                             .and_then(|item: &EnumStatement| Some(item.clone())) {
                       Err(TypeError::EnumMultipleNameField(statement))
                    } else {
                       Ok(ChoiceDef::EnumDef(EnumDef {
                           name, doc, variables, statements
                       }))
                    }
            },
            _ => unreachable!(),
        }
    }
}

#[derive(Debug)]
pub struct SetDef {
    pub name: String,
    pub doc: Option<String>,
    pub arg: Option<VarDef>,
    pub superset: Option<SetRef>,
    pub disjoint: Vec<String>,
    pub keys: Vec<(ir::SetDefKey, Option<VarDef>, String)>,
    pub quotient: Option<Quotient>,
}
