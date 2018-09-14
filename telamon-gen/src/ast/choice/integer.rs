use super::*;
use std::ops::Deref;

/// A toplevel integer
#[derive(Clone, Debug)]
pub struct IntegerDef {
    pub name: Spanned<String>,
    pub doc: Option<String>,
    pub variables: Vec<VarDef>,
    pub code: String, // varmap, type_check_code
}

impl IntegerDef {
    /// This checks if the variables are defined in the context.
    fn check_undefined_variables(
        &self,
        context: &CheckerContext,
    ) -> Result<(), TypeError> {
        for VarDef { name: _, ref set } in self.variables.iter() {
            if !context.check_set_define(set) {
                let name: &String = set.name.deref();

                Err(TypeError::Undefined {
                    object_name: self.name.with_data(name.to_owned()),
                })?;
            }
        }
        Ok(())
    }

    /// Type checks the declare's condition.
    pub fn declare(&self, context: &mut CheckerContext) -> Result<(), TypeError> {
        Ok(())
    }

    /// Type checks the define's condition.
    pub fn define(
        self,
        context: &mut CheckerContext,
        tc: &mut TypingContext,
    ) -> Result<(), TypeError> {
        self.check_undefined_variables(context)?;
        Ok(())
    }
}

impl PartialEq for IntegerDef {
    fn eq(&self, rhs: &Self) -> bool {
        self.name == rhs.name
    }
}
