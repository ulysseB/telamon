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

impl EnumDef {
    pub fn check_field_name_multi(&self) -> Result<(), TypeError> {
        if let Some(statement) = 
            self.statements.iter()
                .find(|item|
                    self.statements.iter()
                        .skip_while(|subitem| subitem != item).skip(1)
                        .any(|ref subitem| subitem == item))
                        .and_then(|item: &EnumStatement| Some(item.clone())) {
            Err(TypeError::EnumFieldNameMulti(statement))?
        }
        Ok(())
    }

    pub fn check_symmetric(&self) -> Result<(), TypeError> {
        if self.statements.contains(&EnumStatement::Symmetric) {
            match self.variables.len() {
                2 => {
                    if let Some((left, right)) =
                        self.variables.windows(2)
                            .find(|vars: &&[VarDef]| vars[0] != vars[1])
                            .and_then(|vars: &[VarDef]|
                                Some((vars[0].clone(), vars[1].clone()))) {
                        Err(TypeError::EnumSymmetricSameParametric(
                                left, right))?
                    }
                },
                n => Err(TypeError::EnumSymmetricTwoParametric(n))?,
            }
        }
        Ok(())
    }
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
                let enum_def: EnumDef = EnumDef {
                    name, doc, variables, statements
                };

                enum_def.check_field_name_multi()?;
                enum_def.check_symmetric()?;
                Ok(ChoiceDef::EnumDef(enum_def))
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
