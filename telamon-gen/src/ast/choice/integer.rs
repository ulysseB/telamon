use super::*;

/// A toplevel integer
#[derive(Clone, Debug)]
pub struct IntegerDef {
    pub name: Spanned<String>,
    pub doc: Option<String>,
    pub variables: Vec<VarDef>,
    pub code: String, // varmap, type_check_code
}

impl IntegerDef {
    /// Type checks the declare's condition.
    pub fn declare(&self, context: &mut CheckerContext) -> Result<(), TypeError> {
        Ok(())
    }

    /// Type checks the define's condition.
    pub fn define(&self, context: &CheckerContext) -> Result<(), TypeError> {
        context.check_choice_define(&self.name, &self.variables)?;
        Ok(())
    }
}

impl PartialEq for IntegerDef {
    fn eq(&self, rhs: &Self) -> bool {
        self.name == rhs.name
    }
}
