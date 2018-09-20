use std::ops::Deref;

use super::error::{Hint, TypeError};
use super::SetRef;

use lexer::Spanned;
use utils::HashMap;

/// CheckContext is a type system.
#[derive(Debug, Default)]
pub struct CheckerContext {
    /// Map Name of unique identifiant.
    hash_set: HashMap<String, Spanned<Hint>>,
    hash_choice: HashMap<String, Spanned<Hint>>,
}

impl CheckerContext {
    /// Declares a set and ensures it is not defined twice.
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

    /// Declares a choice and ensures it is not defined twice.
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

    /// Check if the referenced set is defined.
    pub fn check_set_define(&self, subset: &SetRef) -> bool {
        self.hash_set.contains_key(subset.name.deref())
    }
}
