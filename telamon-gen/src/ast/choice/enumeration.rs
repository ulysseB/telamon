use super::*;

/// A toplevel definition or constraint.
#[derive(Clone, Debug)]
pub struct EnumDef {
    pub name: Spanned<String>,
    pub doc: Option<String>,
    pub variables: Vec<VarDef>,
    pub statements: Vec<Spanned<EnumStatement>>
}

impl EnumDef {
    pub fn undefined(&self) -> Result<(), TypeError> {
        let mut hash: HashMap<String, _> = HashMap::default();

        for stmt in self.statements.iter() {
            match stmt {
                Spanned { beg, end, data: EnumStatement::Value(name, ..) } => {
                    hash.insert(name.to_owned(), Spanned::default());
                },
                Spanned { beg, end, data: EnumStatement::Alias(name, _, sets, ..) } => {
                    for set in sets {
                        if !hash.contains_key(set) {
                            Err(TypeError::Undefined(Spanned {
                                beg: *beg, end: *end,
                                data: set.to_owned(),
                            }))?;
                        }
                    }
                    hash.insert(name.to_owned(), Spanned::default());
                },
                _ => {},
            }
        }
        Ok(())
    }

    pub fn redefinition(&self) -> Result<(), TypeError> {
        let mut hash: HashMap<String, _> = HashMap::default();
        let mut symmetric: Option<Spanned<()>> = None;

        for stmt in self.statements.iter() {
            match stmt {
                Spanned { beg, end, data: EnumStatement::Symmetric } => {
                    if let Some(before) = symmetric {
                        Err(TypeError::Redefinition(
                            Spanned {
                                beg: before.beg, end: before.end,
                                data: Hint::EnumAttribute
                            },
                            Spanned {
                                beg: *beg, end: *end,
                                data: EnumStatement::Symmetric.to_string()
                            },
                        ))?;
                    } else {
                        symmetric = Some(Spanned {
                            beg: *beg, end: *end,
                            data: (),
                        });
                    }
                },
                Spanned { beg, end, data: EnumStatement::Value(name, ..) } |
                Spanned { beg, end, data: EnumStatement::Alias(name, ..) } => {
                    if let Some(Spanned {
                        beg: beg_before,
                        end: end_before,
                        data: _
                    }) = hash.insert(name.to_owned(), Spanned::default()) {
                        Err(TypeError::Redefinition(
                            Spanned {
                                beg: *beg, end: *end,
                                data: Hint::EnumAttribute
                            },
                            Spanned {
                                beg: *beg, end: *end,
                                data: name.to_owned()
                            },
                        ))?;
                    }
                },
                _ => {},
            }

        }
        Ok(())
    }

    pub fn type_check(&self) -> Result<(), TypeError> {
        self.undefined()?;
        self.redefinition()?;
        Ok(())
    }
}

impl PartialEq for EnumDef {
    fn eq(&self, rhs: &Self) -> bool {
        self.name == rhs.name
    }
}
