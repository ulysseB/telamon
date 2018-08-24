use super::*;
use std::ops::Deref;

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
    /// This checks that thereisn't any keys doublon.
    fn check_redefinition_key(&self) -> Result<(), TypeError> {
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

    /// This checks the presence of keys ItemType, IdType, ItemGetter,
    /// IdGetter and Iter. When there is a superset, this checks the
    /// presence of FromSuperset keyword.
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
        if self.superset.is_some() && !keys.contains(
            &&ir::SetDefKey::FromSuperset
        ) {
            Err(TypeError::MissingEntry {
                object_name: self.name.data.to_owned(),
                object_field: self
                    .name
                    .with_data(ir::SetDefKey::FromSuperset.to_string()),
            })?;
        }
        Ok(())
    }

    /// This checks if the argument is defined in the context.
    fn check_undefined_argument(
        &self, context: &CheckerContext
    ) -> Result<(), TypeError> {
        if let Some(VarDef { name: _, set: ref subset })= self.arg {
            if !context.check_set_define(subset) {
                let name: &String = subset.name.deref();

                Err(TypeError::Undefined {
                   object_name: self.name.with_data(name.to_owned()),
                })?;
            }
        }
        Ok(())
    }

    /// This checks if the superset is defined in the context.
    fn check_undefined_superset(
        &self, context: &CheckerContext
    ) -> Result<(), TypeError> {
        if let Some(ref subset) = self.superset {
            if !context.check_set_define(subset) {
                let name: &String = subset.name.deref();

                Err(TypeError::Undefined {
                   object_name: self.name.with_data(name.to_owned()),
                })?;
            }
        }
        Ok(())
    }

    /// Type checks the declare's condition.
    pub fn declare(&self, context: &mut CheckerContext) -> Result<(), TypeError> {
        context.declare_set(self.name.to_owned())
    }

    /// Type checks the define's condition.
    pub fn define(&self, context: &CheckerContext) -> Result<(), TypeError> {
        self.check_undefined_argument(context)?;
        self.check_undefined_superset(context)?;
        self.check_redefinition_key()?;
        self.check_missing_entry()?;
        Ok(())
    }
}

impl PartialEq for SetDef {
    fn eq(&self, rhs: &Self) -> bool {
        self.name == rhs.name
    }
}
