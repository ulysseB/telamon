use super::*;
use std::ops::Deref;

/// CheckContext is a type system.
#[derive(Debug, Default)]
pub struct CheckerContext {
    /// Map Name of unique identifiant.
    hash_set: HashMap<String, Spanned<Hint>>,
    hash_choice: HashMap<String, Spanned<Hint>>,
}

impl CheckerContext {
    /// This checks the redefinition of SetDef.
    pub fn declare_set(&mut self, object_name: Spanned<String>) -> Result<(), TypeError> {
        if let Some(pre) = self.hash_set.insert(
            object_name.data.to_owned(),
            object_name.with_data(Hint::Set),
        ) {
            Err(TypeError::Redefinition {
                object_kind: pre,
                object_name,
            })
        } else {
            Ok(())
        }
    }

    /// This checks the redefinition of ChoiceDef (EnumDef and IntegerDef).
    pub fn declare_choice(
        &mut self,
        object_name: Spanned<String>,
        object_type: Hint,
    ) -> Result<(), TypeError> {
        if let Some(pre) = self.hash_choice.insert(
            object_name.data.to_owned(),
            object_name.with_data(object_type),
        ) {
            Err(TypeError::Redefinition {
                object_kind: pre,
                object_name,
            })
        } else {
            Ok(())
        }
    }

    /// This checks the undefined of SetDef superset and arg.
    pub fn check_set_define(&self, subset: &SetRef) -> bool {
        self.hash_set.contains_key(subset.name.deref())
    }

    /// This checks the undefined of EnumDef or IntegerDef.
    pub fn check_choice_define(
        &self,
        object_name: &Spanned<String>,
        field_variables: &Vec<VarDef>,
    ) -> Result<(), TypeError> {
        for VarDef {
            name: _,
            set: SetRef { name, .. },
        } in field_variables
        {
            let name: &String = name.deref();
            if !self.hash_set.contains_key(name) {
                Err(TypeError::Undefined {
                    object_name: object_name.with_data(name.to_owned()),
                })?;
            }
        }
        Ok(())
    }
}
