use super::*;

/// A toplevel definition or constraint.
#[derive(Clone, Debug)]
pub struct EnumDef {
    pub name: Spanned<String>,
    pub doc: Option<String>,
    pub variables: Vec<VarDef>,
    pub statements: Vec<EnumStatement>
}

impl EnumDef {
    /// This checks the undefined of value or alias.
    fn check_undefined(&self) -> Result<(), TypeError> {
        let mut hash: HashMap<String, _> = HashMap::default();

        for stmt in self.statements.iter() {
            match stmt {
                EnumStatement::Value(spanned, ..) => {
                    hash.insert(spanned.data.to_owned(), spanned.with_data(()));
                },
                EnumStatement::Alias(spanned, _, sets, ..) => {
                    for set in sets {
                        if !hash.contains_key(set) {
                            Err(TypeError::Undefined(spanned.with_data(set.to_owned())))?;
                        }
                    }
                    hash.insert(spanned.data.to_owned(), spanned.with_data(()));
                },
                _ => {},
            }
        }
        Ok(())
    }

    /// This checks that there isn't any doublon.
    fn check_redefinition(&self) -> Result<(), TypeError> {
        let mut hash: HashMap<String, _> = HashMap::default();
        let mut symmetric: Option<Spanned<()>> = None;

        for stmt in self.statements.iter() {
            match stmt {
                EnumStatement::Symmetric(spanned) => {
                    if let Some(ref before) = symmetric {
                        Err(TypeError::Redefinition(before.with_data(Hint::EnumAttribute),
                                                    spanned.with_data(String::from("Symmetric"))))?;
                    } else {
                        symmetric = Some(spanned.with_data(()));
                    }
                },
                EnumStatement::Value(spanned, ..) |
                EnumStatement::Alias(spanned, ..) => {
                    if let Some(before) = hash.insert(spanned.data.to_owned(), spanned.with_data(())) {
                        Err(TypeError::Redefinition(
                            before.with_data(Hint::EnumAttribute),
                            spanned.with_data(spanned.data.to_owned())
                        ))?;
                    }
                },
                _ => {},
            }

        }
        Ok(())
    }

    /// Type checks the condition.
    pub fn type_check(&self) -> Result<(), TypeError> {
        self.check_undefined()?;
        self.check_redefinition()?;
        Ok(())
    }
}

impl PartialEq for EnumDef {
    fn eq(&self, rhs: &Self) -> bool {
        self.name == rhs.name
    }
}
