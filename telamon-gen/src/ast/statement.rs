use ast::*;
use ir;

/// A toplevel definition or constraint.
#[derive(Clone, Debug)]
pub struct EnumDef {
    pub name: String,
    pub doc: Option<String>,
    pub variables: Vec<VarDef>,
    pub statements: Vec<EnumStatement>
}

impl Default for EnumDef {
    fn default() -> EnumDef {
        EnumDef {
            name: String::default(),
            doc: None,
            variables: vec![],
            statements: vec![],
        }
    }
}

impl EnumDef {
    /// A field from a enum should be unique.
    pub fn check_field_name_multi(&self) -> Result<(), TypeError> {
        for item in self.statements.iter() {
            for subitem in self.statements.iter()
                               .skip_while(|subitem| subitem != &item)
                               .skip(1) {
                if subitem == item {
                    Err(TypeError::EnumFieldRedefinition(self.clone(), item.clone()))?
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
                        Err(TypeError::EnumSymmetricUnsameParametric(
                                left, right))?
                    }
                },
                _ => Err(
                    TypeError::EnumSymmetricUntwoParametric(self.variables.clone())
                )?,
            }
        }
        Ok(())
    }

    /// An Alias' value should exists.
    pub fn check_undefined_value(&self) -> Result<(), TypeError> {
        let values: Vec<&String> =
            self.statements.iter()
                           .filter_map(|item| item.get_value().or(item.get_alias()))
                           .collect::<Vec<&String>>();

        for decisions in self.statements.iter()
                                        .filter_map(|item| item.get_alias_decisions()) {
            for value in decisions {
                if !values.contains(&value) {
                    Err(TypeError::EnumUndefinedValue(value.clone()))?
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

#[derive(Clone, Debug)]
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

#[derive(Clone, Debug, PartialEq)]
pub enum ChoiceDef {
    CounterDef(CounterDef),
    EnumDef(EnumDef),
}

impl ChoiceDef {
    pub fn get_name(&self) -> &String {
        match self {
            ChoiceDef::CounterDef(counter_def) => {
                &counter_def.name
            },
            ChoiceDef::EnumDef(enum_def) => {
                &enum_def.name
            },
        }
    }
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
                enum_def.check_undefined_value()?;
                Ok(ChoiceDef::EnumDef(enum_def))
            },
            _ => unreachable!(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SetDef {
    pub name: String,
    pub doc: Option<String>,
    pub arg: Option<VarDef>,
    pub superset: Option<SetRef>,
    pub disjoint: Vec<String>,
    pub keys: Vec<(ir::SetDefKey, Option<VarDef>, String)>,
    pub quotient: Option<Quotient>,
}

impl Default for SetDef {
    fn default() -> SetDef {
        SetDef {
            name: Default::default(),
            doc: None,
            arg: None,
            superset: None,
            disjoint: vec![],
            keys: vec![],
            quotient: None,
        }
    }
}

impl SetDef {
    /// A set should always have the keys: type, id_type, id_getter, iterator, getter.
    pub fn check_undefined_key(&self) -> Result<(), TypeError> {
        let keys = self.keys.iter().map(|(k, _, _)| k).collect::<Vec<&ir::SetDefKey>>();

        if !keys.contains(&&ir::SetDefKey::ItemType) {
            Err(TypeError::SetUndefinedKey(ir::SetDefKey::ItemType))?
        }
        if !keys.contains(&&ir::SetDefKey::IdType) {
            Err(TypeError::SetUndefinedKey(ir::SetDefKey::IdType))?
        }
        if !keys.contains(&&ir::SetDefKey::ItemGetter) {
            Err(TypeError::SetUndefinedKey(ir::SetDefKey::ItemGetter))?
        }
        if !keys.contains(&&ir::SetDefKey::IdGetter) {
            Err(TypeError::SetUndefinedKey(ir::SetDefKey::IdGetter))?
        }
        if !keys.contains(&&ir::SetDefKey::Iter) {
            Err(TypeError::SetUndefinedKey(ir::SetDefKey::Iter))?
        }
        Ok(())
    }

    pub fn check_undefined_superset_key(&self) -> Result<(), TypeError> {
        if self.superset.is_some() {
            if self.keys.iter().find(|(k, _, _)| k == &ir::SetDefKey::FromSuperset)
                               .is_none() {
                Err(TypeError::SetUndefinedKey(ir::SetDefKey::FromSuperset))?
            }
        }
        Ok(())
    }
}

impl PartialEq for SetDef {
    fn eq(&self, rhs: &Self) -> bool {
        self.name == rhs.name
    }
}

impl From<Statement> for Result<SetDef, TypeError> {
    fn from(stmt: Statement) -> Self {
        match stmt {
            Statement::SetDef {
                name, doc, arg, superset, disjoint, keys, quotient
            } => {
                let set_def: SetDef = SetDef {
                    name, doc, arg, superset, disjoint, keys, quotient
                };
                
                set_def.check_undefined_key()?;
                set_def.check_undefined_superset_key()?;
                Ok(set_def)
            },
            _ => unreachable!(),
        }
    }
}
