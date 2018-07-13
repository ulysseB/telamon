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
                EnumStatement::Value(Spanned { beg, end, data: name }, ..) => {
                    hash.insert(name.to_owned(), Spanned {
                        beg: *beg, end: *end, data: ()
                    } );
                },
                EnumStatement::Alias(Spanned { beg, end, data: name }, _, sets, ..) => {
                    for set in sets {
                        if !hash.contains_key(set) {
                            Err(TypeError::Undefined(Spanned {
                                beg: *beg, end: *end,
                                data: set.to_owned(),
                            }))?;
                        }
                    }
                    hash.insert(name.to_owned(), Spanned {
                        beg: *beg, end: *end, data: ()
                    });
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
                EnumStatement::Symmetric(Spanned { beg, end, ..}) => {
                    if let Some(before) = symmetric {
                        Err(TypeError::Redefinition(
                            Spanned {
                                beg: before.beg, end: before.end,
                                data: Hint::EnumAttribute
                            },
                            Spanned {
                                beg: *beg, end: *end,
                                data: String::from("Symmetric"),
                            },
                        ))?;
                    } else {
                        symmetric = Some(Spanned {
                            beg: *beg, end: *end,
                            data: (),
                        });
                    }
                },
                EnumStatement::Value(Spanned { beg, end, data: name }, ..) |
                EnumStatement::Alias(Spanned { beg, end, data: name }, ..) => {
                    if let Some(Spanned {
                        beg: beg_before,
                        end: end_before,
                        data: _
                    }) = hash.insert(
                        name.to_owned(),
                        Spanned { beg: *beg, end: *end, data: () }
                    ) {
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
