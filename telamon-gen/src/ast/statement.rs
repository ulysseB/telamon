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
    /// A field from a enum should be unique.
    pub fn check_field_name_multi(&self) -> Result<(), TypeError> {
        for item in self.statements.iter() {
            for subitem in self.statements.iter()
                               .skip_while(|subitem| subitem != &item)
                               .skip(1) {
                if subitem == item {
                    Err(TypeError::EnumFieldNameMulti(item.clone()))?
                }
            }
        }
        Ok(())
    }

    /// An antisymmetric should refers to two same parametric.
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

    /// An Alias' value should exists.
    pub fn check_missing_alias_value(&self) -> Result<(), TypeError> {
        for item in self.statements.iter() {
            if let EnumStatement::Alias(_, _, values, ..) = item {
                for value in values {
                    if self.statements.iter().all(|ref subitem| {
                        if let EnumStatement::Value(subvalue, ..) = subitem {
                            value != subvalue
                        } else {
                            true
                        }
                    }) {
                        Err(TypeError::EnumAliasValueMissing(value.clone()))?
                    }
                }
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
                println!("> {} {:?} {:?}", name, doc, statements);
                let enum_def: EnumDef = EnumDef {
                    name, doc, variables, statements
                };

                enum_def.check_field_name_multi()?;
                enum_def.check_symmetric()?;
                enum_def.check_missing_alias_value()?;
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

impl SetDef {
    pub fn check_missing_key(&self) -> Result<(), TypeError> {
        Ok(())
    }
}
