use super::*;

/// A toplevel definition or constraint.
#[derive(Clone, Debug)]
pub struct EnumDef {
    pub name: Spanned<String>,
    pub doc: Option<String>,
    pub variables: Vec<VarDef>,
    pub statements: Vec<EnumStatement>,
}

impl EnumDef {
    /// This checks the undefined of value or alias from alias or antisymmetric.
    fn check_undefined(&self) -> Result<(), TypeError> {
        let mut hash: HashMap<String, _> = HashMap::default();

        for stmt in self.statements.iter() {
            match stmt {
                EnumStatement::Value(spanned, ..) | EnumStatement::Alias(spanned, ..) => {
                    hash.insert(spanned.data.to_owned(), ());
                }
                _ => {}
            }
        }
        for stmt in self.statements.iter() {
            match stmt {
                EnumStatement::AntiSymmetric(spanned) => {
                    for (first, second) in spanned.data.iter() {
                        if !hash.contains_key(&first.to_owned()) {
                            Err(TypeError::Undefined {
                                object_name: spanned.with_data(first.to_owned()),
                            })?;
                        }
                        if !hash.contains_key(&second.to_owned()) {
                            Err(TypeError::Undefined {
                                object_name: spanned.with_data(second.to_owned()),
                            })?;
                        }
                    }
                }
                EnumStatement::Alias(spanned, _, sets, ..) => {
                    for set in sets {
                        if !hash.contains_key(&set.to_owned()) {
                            Err(TypeError::Undefined {
                                object_name: spanned.with_data(set.to_owned()),
                            })?;
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// This checks that there isn't any doublon in the field list.
    fn check_redefinition_field(&self) -> Result<(), TypeError> {
        let mut hash: HashMap<String, _> = HashMap::default();
        let mut symmetric: Option<Spanned<()>> = None;
        let mut antisymmetric: Option<Spanned<()>> = None;

        for stmt in self.statements.iter() {
            match stmt {
                EnumStatement::AntiSymmetric(spanned) => {
                    if let Some(ref before) = antisymmetric {
                        Err(TypeError::Redefinition {
                            object_kind: before.with_data(Hint::EnumAttribute),
                            object_name: spanned.with_data(String::from("Antisymmetric")),
                        })?;
                    } else {
                        antisymmetric = Some(spanned.with_data(()));
                    }
                }
                EnumStatement::Symmetric(spanned) => {
                    if let Some(ref before) = symmetric {
                        Err(TypeError::Redefinition {
                            object_kind: before.with_data(Hint::EnumAttribute),
                            object_name: spanned.with_data(String::from("Symmetric")),
                        })?;
                    } else {
                        symmetric = Some(spanned.with_data(()));
                    }
                }
                EnumStatement::Value(spanned, ..) | EnumStatement::Alias(spanned, ..) => {
                    if let Some(before) =
                        hash.insert(spanned.data.to_owned(), spanned.with_data(()))
                    {
                        Err(TypeError::Redefinition {
                            object_kind: before.with_data(Hint::EnumAttribute),
                            object_name: spanned.with_data(spanned.data.to_owned()),
                        })?;
                    }
                }
            }
        }
        Ok(())
    }

    /// This checks that there isn't any doublon in parameter list.
    fn check_redefinition_parameter(&self) -> Result<(), TypeError> {
        let mut hash: HashMap<String, _> = HashMap::default();
        for VarDef { name, .. } in self.variables.as_slice() {
            if let Some(before) = hash.insert(name.data.to_string(), name.with_data(())) {
                Err(TypeError::Redefinition {
                    object_kind: before.with_data(Hint::EnumAttribute),
                    object_name: name.with_data(name.data.to_string()),
                })?;
            }
        }
        Ok(())
    }

    /// This checks that both fields symmetric and antisymmetric aren't defined
    /// in the same enumeration.
    fn check_conflict(&self) -> Result<(), TypeError> {
        let mut symmetric: Option<Spanned<()>> = None;
        let mut antisymmetric: Option<Spanned<()>> = None;

        for stmt in self.statements.iter() {
            match stmt {
                EnumStatement::AntiSymmetric(spanned) => {
                    if let Some(ref symmetric) = symmetric {
                        Err(TypeError::Conflict {
                            object_fields: (
                                symmetric.with_data(String::from("Symmetric")),
                                spanned.with_data(String::from("Antisymmetric")),
                            ),
                        })?;
                    } else {
                        antisymmetric = Some(spanned.with_data(()));
                    }
                }
                EnumStatement::Symmetric(spanned) => {
                    if let Some(ref antisymmetric) = antisymmetric {
                        Err(TypeError::Conflict {
                            object_fields: (
                                antisymmetric.with_data(String::from("Antisymmetric")),
                                spanned.with_data(String::from("Symmetric")),
                            ),
                        })?;
                    } else {
                        symmetric = Some(spanned.with_data(()));
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// This checks that there is two parameters if the field symmetric is defined.
    fn check_two_parameter(&self) -> Result<(), TypeError> {
        if self
            .statements
            .iter()
            .find(|item| item.is_symmetric() || item.is_antisymmetric())
            .is_some()
        {
            if self.variables.len() != 2 {
                Err(TypeError::BadSymmetricArg {
                        object_name: self.name.to_owned(),
                        object_variables: self.variables.iter().map(|variable| {
                            let name: &String = variable.name.data.deref();
                            let set_name: &String = variable.set.name.deref();
                            (variable.name.with_data(name.to_owned()),
                             set_name.to_owned())
                        }).collect::<Vec<(Spanned<String>, String)>>()
                })?;
            }
        }
        Ok(())
    }

    /// This checkls that the parameters share the same type.
    fn check_same_parameter(&self) -> Result<(), TypeError> {
        if self
            .statements
            .iter()
            .find(|item| item.is_symmetric() || item.is_antisymmetric())
            .is_some()
        {
            match self.variables.as_slice() {
                [VarDef {
                    name: _,
                    set: SetRef { name, .. },
                }, VarDef {
                    name: _,
                    set: SetRef { name: rhs_name, .. },
                }] => {
                    if name != rhs_name {
                        Err(TypeError::BadSymmetricArg {
                            object_name: self.name.to_owned(),
                            object_variables: self.variables.iter().map(|variable| {
                                let name: &String = variable.name.data.deref();
                                let set_name: &String = variable.set.name.deref();
                                (variable.name.with_data(name.to_owned()),
                                 set_name.to_owned())
                            }).collect::<Vec<(Spanned<String>, String)>>()
                        })?;
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Type checks the condition.
    pub fn type_check(&self) -> Result<(), TypeError> {
        self.check_undefined()?;
        self.check_redefinition_parameter()?;
        self.check_redefinition_field()?;
        self.check_two_parameter()?;
        self.check_same_parameter()?;
        self.check_conflict()?;
        Ok(())
    }
}

impl PartialEq for EnumDef {
    fn eq(&self, rhs: &Self) -> bool {
        self.name == rhs.name
    }
}
