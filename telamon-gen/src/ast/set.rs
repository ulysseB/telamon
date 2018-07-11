use super::*;

#[derive(Debug, Clone)]
pub struct SetDef {
    pub name: Spanned<String>,
    pub doc: Option<String>,
    pub arg: Option<VarDef>,
    pub superset: Option<SetRef>,
    pub disjoint: Vec<String>,
    pub keys: Vec<(ir::SetDefKey, Option<VarDef>, String)>,
    pub quotient: Option<Quotient>,
}

impl SetDef {
    fn missing_entry(&self) -> Result<(), TypeError> {
        let keys = self.keys.iter()
                            .map(|(k, _, _)| k)
                            .collect::<Vec<&ir::SetDefKey>>();

        if !keys.contains(&&ir::SetDefKey::ItemType) {
            Err(TypeError::MissingEntry(self.name.data.to_owned(), Spanned {
                beg: self.name.beg, end: self.name.end,
                data: ir::SetDefKey::ItemType.to_string()
            }))?;
        }
        if !keys.contains(&&ir::SetDefKey::IdType) {
            Err(TypeError::MissingEntry(self.name.data.to_owned(), Spanned {
                beg: self.name.beg, end: self.name.end,
                data: ir::SetDefKey::IdType.to_string()
            }))?;
        }
        if !keys.contains(&&ir::SetDefKey::ItemGetter) {
            Err(TypeError::MissingEntry(self.name.data.to_owned(), Spanned {
                beg: self.name.beg, end: self.name.end,
                data: ir::SetDefKey::ItemGetter.to_string()
            }))?;
        }
        if !keys.contains(&&ir::SetDefKey::IdGetter) {
            Err(TypeError::MissingEntry(self.name.data.to_owned(), Spanned {
                beg: self.name.beg, end: self.name.end,
                data: ir::SetDefKey::IdGetter.to_string()
            }))?;
        }
        if !keys.contains(&&ir::SetDefKey::Iter) {
            Err(TypeError::MissingEntry(self.name.data.to_owned(), Spanned {
                beg: self.name.beg, end: self.name.end,
                data: ir::SetDefKey::Iter.to_string()
            }))?;
        }
        if self.superset.is_some() && !keys.contains(&&ir::SetDefKey::FromSuperset) {
            Err(TypeError::MissingEntry(self.name.data.to_owned(), Spanned {
                beg: self.name.beg, end: self.name.end,
                data: ir::SetDefKey::FromSuperset.to_string()
            }))?;
        }
        Ok(())
    }

    pub fn redefinition(&self) -> Result<(), TypeError> {
        let mut hash: HashMap<String, _> = HashMap::default();
        for (key, ..) in self.keys.iter() {
            if let Some(before) = hash.insert(key.to_string(), ()) {
                Err(TypeError::Redefinition(Spanned {
                    beg: Default::default(),
                    end: Default::default(),
                    data: Hint::Set,
                }, Spanned {
                    beg: Default::default(),
                    end: Default::default(),
                    data: key.to_string(),
                }))?;
            }
        }
        Ok(())
    }

    pub fn type_check(&self) -> Result<(), TypeError> {
        self.redefinition()?;
        self.missing_entry()?;
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
            Statement::SetDef(SetDef {
                name, doc, arg, superset, disjoint, keys, quotient
            }) => {
                Ok(SetDef {
                    name, doc, arg, superset, disjoint, keys, quotient
                })
            },
            _ => unreachable!(),
        }
    }
}
