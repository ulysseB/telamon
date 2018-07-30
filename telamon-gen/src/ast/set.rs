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

    /// This checks the presence of keys ItemType, IdType, ItemGetter, IdGetter and Iter.
    /// When there is a superset, this checks too the presence of FromSuperset keyword.
    fn check_missing_entry(&self) -> Result<(), TypeError> {
        let keys = self.keys.iter().map(|(k, _, _)| k)
                            .collect::<Vec<&ir::SetDefKey>>();

        if !keys.contains(&&ir::SetDefKey::ItemType) {
            Err(TypeError::MissingEntry(self.name.data.to_owned(),
                self.name.with_data(ir::SetDefKey::ItemType.to_string())))?;
        }
        if !keys.contains(&&ir::SetDefKey::IdType) {
            Err(TypeError::MissingEntry(self.name.data.to_owned(),
                self.name.with_data(ir::SetDefKey::IdType.to_string())))?;
        }
        if !keys.contains(&&ir::SetDefKey::ItemGetter) {
            Err(TypeError::MissingEntry(self.name.data.to_owned(),
                self.name.with_data(ir::SetDefKey::ItemGetter.to_string())))?;
        }
        if !keys.contains(&&ir::SetDefKey::IdGetter) {
            Err(TypeError::MissingEntry(self.name.data.to_owned(),
                self.name.with_data(ir::SetDefKey::IdGetter.to_string())))?;
        }
        if !keys.contains(&&ir::SetDefKey::Iter) {
            Err(TypeError::MissingEntry(self.name.data.to_owned(),
                self.name.with_data(ir::SetDefKey::Iter.to_string())))?;
        }
        if self.superset.is_some() && !keys.contains(&&ir::SetDefKey::FromSuperset) {
            Err(TypeError::MissingEntry(self.name.data.to_owned(),
                self.name.with_data(ir::SetDefKey::FromSuperset.to_string())))?;
        }
        Ok(())
    }

    /// This checks that thereisn't any keys doublon.
    /// TODO: fixe position, add Spanned to parser for key attributs.
    fn check_redefinition(&self) -> Result<(), TypeError> {
        let mut hash: HashSet<String> = HashSet::default();
        for (key, ..) in self.keys.iter() {
            if !hash.insert(key.to_string()) {
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

    /// Type checks the condition.
    pub fn type_check(&self) -> Result<(), TypeError> {
        self.check_redefinition()?;
        self.check_missing_entry()?;
        Ok(())
    }
}

impl PartialEq for SetDef {
    fn eq(&self, rhs: &Self) -> bool {
        self.name == rhs.name
    }
}
