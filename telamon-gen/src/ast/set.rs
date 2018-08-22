use super::*;

#[derive(Debug, Clone)]
pub struct SetDef {
    pub name: Spanned<String>,
    pub doc: Option<String>,
    pub arg: Option<VarDef>,
    pub superset: Option<SetRef>,
    pub disjoint: Vec<String>,
    pub keys: Vec<(Spanned<ir::SetDefKey>, Option<VarDef>, String)>,
    pub quotient: Option<Quotient>,
}

impl SetDef {
    /// This checks the presence of keys ItemType, IdType, ItemGetter, IdGetter and Iter.
    /// When there is a superset, this checks too the presence of FromSuperset keyword.
    fn check_missing_entry(&self) -> Result<(), TypeError> {
        let keys = self
            .keys
            .iter()
            .map(|(k, _, _)| k.data)
            .collect::<Vec<ir::SetDefKey>>();
    
        for ref key in ir::SetDefKey::REQUIRED.iter() {
            if !keys.contains(&key) {
                Err(TypeError::MissingEntry {
                    object_name: self.name.data.to_owned(),
                    object_field: self.name.with_data(key.to_string()),
                })?;
            }
        }
        if self.superset.is_some() && !keys.contains(&&ir::SetDefKey::FromSuperset) {
            Err(TypeError::MissingEntry {
                object_name: self.name.data.to_owned(),
                object_field: self
                    .name
                    .with_data(ir::SetDefKey::FromSuperset.to_string()),
            })?;
        }
        Ok(())
    }

    /// This checks that thereisn't any keys doublon.
    fn check_redefinition(&self) -> Result<(), TypeError> {
        let mut hash: HashMap<_, Spanned<()>> = HashMap::default();
        for (key, ..) in self.keys.iter() {
            if let Some(pre) = hash.insert(key.data.to_owned(), key.with_data(())) {
                Err(TypeError::Redefinition {
                    object_kind: pre.with_data(Hint::Set),
                    object_name: key.with_data(key.data.to_string()),
                })?;
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
