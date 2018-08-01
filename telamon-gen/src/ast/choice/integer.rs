use super::*;

/// A toplevel integer
#[derive(Clone, Debug)]
pub struct IntegerDef {
    /// Name of Interger.
    pub name: Spanned<String>,
    pub doc: Option<String>,
    pub variables: Vec<VarDef>,
    pub code: String, // varmap, type_check_code
}

impl IntegerDef {
    pub fn type_check(&self) -> Result<(), TypeError> {
        Ok(())
    }
}
impl PartialEq for IntegerDef {
    fn eq(&self, rhs: &Self) -> bool {
        self.name == rhs.name
    }
}
