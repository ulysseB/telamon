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
        let mut antisymmetric: Option<Spanned<()>> = None;

        for stmt in self.statements.iter() {
            match stmt {
                EnumStatement::AntiSymmetric(spanned) => {
                    if let Some(ref before) = antisymmetric {
                        Err(TypeError::Redefinition(before.with_data(Hint::EnumAttribute),
                                                    spanned.with_data(String::from("Antisymmetric"))))?;
                    } else {
                        antisymmetric = Some(spanned.with_data(()));
                    }
                },
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
            }

        }
        Ok(())
    }

    /// This checks that there is two parameters if the field symmetric is defined.
    fn check_two_parameter(&self) -> Result<(), TypeError> {
        if self.statements.iter().find(|item| item.is_symmetric()
                                           || item.is_antisymmetric()).is_some() {
            if self.variables.len() != 2 {
                Err(TypeError::BadSymmetricArg(
                        self.name.to_owned(),
                        self.variables.to_owned())
                )?;
            }
        }
        Ok(())
    }

    fn check_same_parameter(&self) -> Result<(), TypeError> {
        if self.statements.iter().find(|item| item.is_symmetric()
                                           || item.is_antisymmetric()).is_some() {
            match self.variables.as_slice() {
                [VarDef { name, .. }, VarDef { name: rhs_name, .. }] => {
                    if name != rhs_name {
                        Err(TypeError::BadSymmetricArg(
                                self.name.to_owned(),
                                self.variables.to_owned())
                        )?;
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
        self.check_two_parameter()?;
        self.check_same_parameter()?;
        Ok(())
    }
}

impl PartialEq for EnumDef {
    fn eq(&self, rhs: &Self) -> bool {
        self.name == rhs.name
    }
}
