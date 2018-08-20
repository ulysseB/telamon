pub use super::utils::RcStr;

pub use super::telamon_gen::ast::*;
pub use super::telamon_gen::lexer::{Lexer, LexerPosition, Position, Spanned};
pub use super::telamon_gen::parser;

/// Undefined
#[cfg(test)]
mod undefined {
    pub use super::*;

    /// Missing the set BasickBlock from a Emum.
    #[test]
    fn parameter() {
        assert_eq!(
            parser::parse_ast(Lexer::new(
                b"define enum foo($lhs in BasicBlock, $rhs in BasicBlock):
                    symmetric
                    value A:
                    value B:
              end"
                    .to_vec()
            )).unwrap()
                .type_check()
                .err(),
            Some(TypeError::Undefined {
                object_name: Spanned {
                    beg: Position {
                        position: LexerPosition {
                            line: 0,
                            column: 12
                        },
                        ..Default::default()
                    },
                    end: Position {
                        position: LexerPosition {
                            line: 0,
                            column: 15
                        },
                        ..Default::default()
                    },
                    data: String::from("BasicBlock"),
                }
            })
        );
    }

    /// Missing the set BasickBlock from a Emum.
    #[test]
    fn value() {
        assert_eq!(
            parser::parse_ast(Lexer::new(
                b"define enum foo():
                value A:
                alias AB = A | B:
              end"
                    .to_vec()
            )).unwrap()
                .type_check()
                .err(),
            Some(TypeError::Undefined {
                object_name: Spanned {
                    beg: Position {
                        position: LexerPosition {
                            line: 2,
                            column: 22
                        },
                        ..Default::default()
                    },
                    end: Position {
                        position: LexerPosition {
                            line: 2,
                            column: 24
                        },
                        ..Default::default()
                    },
                    data: String::from("B"),
                }
            })
        );
        assert_eq!(
            parser::parse_ast(Lexer::new(
                b"set BasicBlock:
                item_type = \"ir::inst::Obj\"
                id_type = \"ir::inst::Id\"
                item_getter = \"ir::inst::get($fun, $id)\"
                id_getter = \"ir::inst::Obj::id($item)\"
                iterator = \"ir::inst::iter($fun)\"
                var_prefix = \"inst\"
                new_objs = \"$objs.inst\"
              end
              define enum foo($lhs in BasicBlock, $rhs in BasicBlock):
                antisymmetric:
                  A -> B
                value A:
              end"
                    .to_vec()
            )).unwrap()
                .type_check()
                .err(),
            Some(TypeError::Undefined {
                object_name: Spanned {
                    beg: Position::new_optional(LexerPosition::new(10, 16), None),
                    end: Position::new_optional(LexerPosition::new(11, 24), None),
                    data: String::from("B"),
                }
            })
        );
    }
}

/// Redefinition
#[cfg(test)]
mod redefinition {
    pub use super::*;

    /// Redefinition of parameter name $lhs.
    #[test]
    fn parameter() {
        assert_eq!(
            parser::parse_ast(Lexer::new(
                b"set BasicBlock:
                item_type = \"ir::inst::Obj\"
                id_type = \"ir::inst::Id\"
                item_getter = \"ir::inst::get($fun, $id)\"
                id_getter = \"ir::inst::Obj::id($item)\"
                iterator = \"ir::inst::iter($fun)\"
                var_prefix = \"inst\"
                new_objs = \"$objs.inst\"
              end
              define enum foo($lhs in BasicBlock, $lhs in BasicBlock):
                symmetric
                value A:
                value B:
              end"
                    .to_vec()
            )).unwrap()
                .type_check()
                .err(),
            Some(TypeError::Redefinition {
                object_kind: Spanned {
                    beg: Position::new_optional(LexerPosition::new(9, 30), None),
                    end: Position::new_optional(LexerPosition::new(9, 34), None),
                    data: Hint::EnumAttribute,
                },
                object_name: Spanned {
                    beg: Position::new_optional(LexerPosition::new(9, 50), None),
                    end: Position::new_optional(LexerPosition::new(9, 54), None),
                    data: String::from("lhs"),
                }
            })
        );
    }

    /// Redefinition of the foo Enum.
    #[test]
    fn enum_() {
        assert_eq!(
            parser::parse_ast(Lexer::new(
                b"define enum foo():
              end

              define enum foo():
              end"
                    .to_vec()
            )).unwrap()
                .type_check()
                .err(),
            Some(TypeError::Redefinition {
                object_kind: Spanned {
                    beg: Position {
                        position: LexerPosition {
                            line: 0,
                            column: 12
                        },
                        ..Default::default()
                    },
                    end: Position {
                        position: LexerPosition {
                            line: 0,
                            column: 15
                        },
                        ..Default::default()
                    },
                    data: Hint::Enum,
                },
                object_name: Spanned {
                    beg: Position {
                        position: LexerPosition {
                            line: 3,
                            column: 26
                        },
                        ..Default::default()
                    },
                    end: Position {
                        position: LexerPosition {
                            line: 3,
                            column: 29
                        },
                        ..Default::default()
                    },
                    data: String::from("foo"),
                }
            })
        );
    }

    /// Redefinition of field.
    #[test]
    fn field() {
        /// Redefinition of the AB field Alias.
        assert_eq!(
            parser::parse_ast(Lexer::new(
                b"define enum foo():
                value A:
                value B:
                value C:
                alias AB = A | B:
                alias AB = A | B:
              end"
                    .to_vec()
            )).unwrap()
                .type_check()
                .err(),
            Some(TypeError::Redefinition {
                object_kind: Spanned {
                    beg: Position {
                        position: LexerPosition {
                            line: 4,
                            column: 22
                        },
                        ..Default::default()
                    },
                    end: Position {
                        position: LexerPosition {
                            line: 4,
                            column: 24
                        },
                        ..Default::default()
                    },
                    data: Hint::EnumAttribute,
                },
                object_name: Spanned {
                    beg: Position {
                        position: LexerPosition {
                            line: 5,
                            column: 22
                        },
                        ..Default::default()
                    },
                    end: Position {
                        position: LexerPosition {
                            line: 5,
                            column: 24
                        },
                        ..Default::default()
                    },
                    data: String::from("AB"),
                }
            })
        );

        /// Redefinition of the A field Value.
        assert_eq!(
            parser::parse_ast(Lexer::new(
                b"define enum foo():
                value A:
                value B:
                value A:
              end"
                    .to_vec()
            )).unwrap()
                .type_check()
                .err(),
            Some(TypeError::Redefinition {
                object_kind: Spanned {
                    beg: Position {
                        position: LexerPosition {
                            line: 1,
                            column: 22
                        },
                        ..Default::default()
                    },
                    end: Position {
                        position: LexerPosition {
                            line: 1,
                            column: 23
                        },
                        ..Default::default()
                    },
                    data: Hint::EnumAttribute,
                },
                object_name: Spanned {
                    beg: Position {
                        position: LexerPosition {
                            line: 3,
                            column: 22
                        },
                        ..Default::default()
                    },
                    end: Position {
                        position: LexerPosition {
                            line: 3,
                            column: 23
                        },
                        ..Default::default()
                    },
                    data: String::from("A"),
                }
            })
        );

        /// Redefinition of the field Antisymmetric.
        assert_eq!(
            parser::parse_ast(Lexer::new(
                b"set BasicBlock:
                item_type = \"ir::inst::Obj\"
                id_type = \"ir::inst::Id\"
                item_getter = \"ir::inst::get($fun, $id)\"
                id_getter = \"ir::inst::Obj::id($item)\"
                iterator = \"ir::inst::iter($fun)\"
                var_prefix = \"inst\"
                new_objs = \"$objs.inst\"
              end
              define enum foo($lhs in BasicBlock, $rhs in BasicBlock):
                antisymmetric:
                  A -> B
                antisymmetric:
                  A -> B
                value A:
                value B:
              end"
                    .to_vec()
            )).unwrap()
                .type_check()
                .err(),
            Some(TypeError::Redefinition {
                object_kind: Spanned {
                    beg: Position::new_optional(LexerPosition::new(10, 16), None),
                    end: Position::new_optional(LexerPosition::new(11, 24), None),
                    data: Hint::EnumAttribute,
                },
                object_name: Spanned {
                    beg: Position::new_optional(LexerPosition::new(12, 16), None),
                    end: Position::new_optional(LexerPosition::new(13, 24), None),
                    data: String::from("Antisymmetric"),
                }
            })
        );
    }

    mod symmetric {
        pub use super::*;

        /// Redefinition of the field Symmetric.
        #[test]
        fn field() {
            assert_eq!(
                parser::parse_ast(Lexer::new(
                    b"set BasicBlock:
                    item_type = \"ir::inst::Obj\"
                    id_type = \"ir::inst::Id\"
                    item_getter = \"ir::inst::get($fun, $id)\"
                    id_getter = \"ir::inst::Obj::id($item)\"
                    iterator = \"ir::inst::iter($fun)\"
                    var_prefix = \"inst\"
                    new_objs = \"$objs.inst\"
                  end
                  define enum foo($lhs in BasicBlock, $rhs in BasicBlock):
                    symmetric
                    symmetric
                    value A:
                    value B:
                  end"
                        .to_vec()
                )).unwrap()
                    .type_check()
                    .err(),
                Some(TypeError::Redefinition {
                    object_kind: Spanned {
                        beg: Position::new_optional(LexerPosition::new(10, 20), None),
                        end: Position::new_optional(LexerPosition::new(10, 29), None),
                        data: Hint::EnumAttribute,
                    },
                    object_name: Spanned {
                        beg: Position::new_optional(LexerPosition::new(11, 20), None),
                        end: Position::new_optional(LexerPosition::new(11, 29), None),
                        data: String::from("Symmetric"),
                    }
                })
            );
        }
    }
}

/// Parameter
#[cfg(test)]
mod parameter {
    pub use super::*;

    /// Unvalid parameter.
    mod antisymmetric {
        pub use super::*;

        /// Unvalid number of parameter.
        #[test]
        fn two() {
            assert_eq!(
                parser::parse_ast(Lexer::new(
                    b"define enum foo():
                    antisymmetric:
                      A -> B
                    value A:
                    value B:
                  end"
                        .to_vec()
                )).unwrap()
                    .type_check()
                    .err(),
                Some(TypeError::BadSymmetricArg {
                    object_name: Spanned {
                        beg: Position::new_optional(LexerPosition::new(0, 12), None),
                        end: Position::new_optional(LexerPosition::new(0, 15), None),
                        data: String::from("foo"),
                    },
                    object_variables: vec![],
                })
            );
            assert_eq!(
                parser::parse_ast(Lexer::new(
                    b"set BasicBlock:
                    item_type = \"ir::basic_block::Obj\"
                    id_type = \"ir::basic_block::Id\"
                    item_getter = \"ir::basic_block::get($fun, $id)\"
                    id_getter = \"ir::basic_block::Obj::id($item)\"
                    iterator = \"ir::basic_block::iter($fun)\"
                    var_prefix = \"bb\"
                    new_objs = \"$objs.basic_block\"
                  end
                  define enum foo($lhs in BasicBlock):
                    antisymmetric:
                      A -> B
                    value A:
                    value B:
                  end"
                        .to_vec()
                )).unwrap()
                    .type_check()
                    .err(),
                Some(TypeError::BadSymmetricArg {
                    object_name: Spanned {
                        beg: Position::new_optional(LexerPosition::new(9, 30), None),
                        end: Position::new_optional(LexerPosition::new(9, 33), None),
                        data: String::from("foo"),
                    },
                    object_variables: vec![VarDef {
                        name: Spanned {
                            beg: Position::new_optional(LexerPosition::new(9, 34), None),
                            end: Position::new_optional(LexerPosition::new(9, 38), None),
                            data: RcStr::new(String::from("lhs")),
                        },
                        set: SetRef {
                            name: RcStr::new(String::from("BasicBlock")),
                            var: None,
                        },
                    }],
                })
            );
        }

        /// Unvalid type parameter.
        #[test]
        fn same() {
            assert_eq!(
                parser::parse_ast(Lexer::new(
                    b"set BasicBlock:
                    item_type = \"ir::basic_block::Obj\"
                    id_type = \"ir::basic_block::Id\"
                    item_getter = \"ir::basic_block::get($fun, $id)\"
                    id_getter = \"ir::basic_block::Obj::id($item)\"
                    iterator = \"ir::basic_block::iter($fun)\"
                    var_prefix = \"bb\"
                    new_objs = \"$objs.basic_block\"
                  end
                  set BasicBlock2:
                    item_type = \"ir::basic_block::Obj\"
                    id_type = \"ir::basic_block::Id\"
                    item_getter = \"ir::basic_block::get($fun, $id)\"
                    id_getter = \"ir::basic_block::Obj::id($item)\"
                    iterator = \"ir::basic_block::iter($fun)\"
                    var_prefix = \"bb\"
                    new_objs = \"$objs.basic_block\"
                  end
                  define enum foo($lhs in BasicBlock, $rhs in BasicBlock2):
                    antisymmetric:
                      A -> B
                    value A:
                    value B:
                  end"
                        .to_vec()
                )).unwrap()
                    .type_check()
                    .err(),
                Some(TypeError::BadSymmetricArg {
                    object_name: Spanned {
                        beg: Position::new_optional(LexerPosition::new(18, 30), None),
                        end: Position::new_optional(LexerPosition::new(18, 33), None),
                        data: String::from("foo"),
                    },
                    object_variables: vec![
                        VarDef {
                            name: Spanned {
                                beg: Position::new_optional(
                                    LexerPosition::new(18, 34),
                                    None,
                                ),
                                end: Position::new_optional(
                                    LexerPosition::new(18, 38),
                                    None,
                                ),
                                data: RcStr::new(String::from("lhs")),
                            },
                            set: SetRef {
                                name: RcStr::new(String::from("BasicBlock")),
                                var: None,
                            },
                        },
                        VarDef {
                            name: Spanned {
                                beg: Position::new_optional(
                                    LexerPosition::new(18, 54),
                                    None,
                                ),
                                end: Position::new_optional(
                                    LexerPosition::new(18, 58),
                                    None,
                                ),
                                data: RcStr::new(String::from("rhs")),
                            },
                            set: SetRef {
                                name: RcStr::new(String::from("BasicBlock2")),
                                var: None,
                            },
                        },
                    ],
                })
            );
        }
    }

    /// Unvalid parameter.
    mod symmetric {
        pub use super::*;

        /// Unvalid number of parameter.
        #[test]
        fn two() {
            assert_eq!(
                parser::parse_ast(Lexer::new(
                    b"define enum foo():
                    symmetric
                    value A:
                    value B:
                  end"
                        .to_vec()
                )).unwrap()
                    .type_check()
                    .err(),
                Some(TypeError::BadSymmetricArg {
                    object_name: Spanned {
                        beg: Position::new_optional(LexerPosition::new(0, 12), None),
                        end: Position::new_optional(LexerPosition::new(0, 15), None),
                        data: String::from("foo"),
                    },
                    object_variables: vec![],
                })
            );
            assert_eq!(
                parser::parse_ast(Lexer::new(
                    b"set BasicBlock:
                    item_type = \"ir::basic_block::Obj\"
                    id_type = \"ir::basic_block::Id\"
                    item_getter = \"ir::basic_block::get($fun, $id)\"
                    id_getter = \"ir::basic_block::Obj::id($item)\"
                    iterator = \"ir::basic_block::iter($fun)\"
                    var_prefix = \"bb\"
                    new_objs = \"$objs.basic_block\"
                  end
                  define enum foo($lhs in BasicBlock):
                    symmetric
                    value A:
                    value B:
                  end"
                        .to_vec()
                )).unwrap()
                    .type_check()
                    .err(),
                Some(TypeError::BadSymmetricArg {
                    object_name: Spanned {
                        beg: Position::new_optional(LexerPosition::new(9, 30), None),
                        end: Position::new_optional(LexerPosition::new(9, 33), None),
                        data: String::from("foo"),
                    },
                    object_variables: vec![VarDef {
                        name: Spanned {
                            beg: Position::new_optional(LexerPosition::new(9, 34), None),
                            end: Position::new_optional(LexerPosition::new(9, 38), None),
                            data: RcStr::new(String::from("lhs")),
                        },
                        set: SetRef {
                            name: RcStr::new(String::from("BasicBlock")),
                            var: None,
                        },
                    }],
                })
            );
            assert_eq!(
                parser::parse_ast(Lexer::new(
                    b"set BasicBlock:
                    item_type = \"ir::basic_block::Obj\"
                    id_type = \"ir::basic_block::Id\"
                    item_getter = \"ir::basic_block::get($fun, $id)\"
                    id_getter = \"ir::basic_block::Obj::id($item)\"
                    iterator = \"ir::basic_block::iter($fun)\"
                    var_prefix = \"bb\"
                    new_objs = \"$objs.basic_block\"
                  end
                  define enum foo($lhs in BasicBlock,
                                  $chs in BasicBlock,
                                  $rhs in BasicBlock):
                    symmetric
                    value A:
                    value B:
                  end"
                        .to_vec()
                )).unwrap()
                    .type_check()
                    .err(),
                Some(TypeError::BadSymmetricArg {
                    object_name: Spanned {
                        beg: Position::new_optional(LexerPosition::new(9, 30), None),
                        end: Position::new_optional(LexerPosition::new(9, 33), None),
                        data: String::from("foo"),
                    },
                    object_variables: vec![
                        VarDef {
                            name: Spanned {
                                beg: Position::new_optional(
                                    LexerPosition::new(9, 34),
                                    None,
                                ),
                                end: Position::new_optional(
                                    LexerPosition::new(9, 38),
                                    None,
                                ),
                                data: RcStr::new(String::from("lhs")),
                            },
                            set: SetRef {
                                name: RcStr::new(String::from("BasicBlock")),
                                var: None,
                            },
                        },
                        VarDef {
                            name: Spanned {
                                beg: Position::new_optional(
                                    LexerPosition::new(10, 34),
                                    None,
                                ),
                                end: Position::new_optional(
                                    LexerPosition::new(10, 38),
                                    None,
                                ),
                                data: RcStr::new(String::from("chs")),
                            },
                            set: SetRef {
                                name: RcStr::new(String::from("BasicBlock")),
                                var: None,
                            },
                        },
                        VarDef {
                            name: Spanned {
                                beg: Position::new_optional(
                                    LexerPosition::new(11, 34),
                                    None,
                                ),
                                end: Position::new_optional(
                                    LexerPosition::new(11, 38),
                                    None,
                                ),
                                data: RcStr::new(String::from("rhs")),
                            },
                            set: SetRef {
                                name: RcStr::new(String::from("BasicBlock")),
                                var: None,
                            },
                        },
                    ],
                })
            );
        }

        /// Unvalid type parameter.
        #[test]
        fn same() {
            assert_eq!(
                parser::parse_ast(Lexer::new(
                    b"set BasicBlock:
                    item_type = \"ir::basic_block::Obj\"
                    id_type = \"ir::basic_block::Id\"
                    item_getter = \"ir::basic_block::get($fun, $id)\"
                    id_getter = \"ir::basic_block::Obj::id($item)\"
                    iterator = \"ir::basic_block::iter($fun)\"
                    var_prefix = \"bb\"
                    new_objs = \"$objs.basic_block\"
                  end
                  set BasicBlock2:
                    item_type = \"ir::basic_block::Obj\"
                    id_type = \"ir::basic_block::Id\"
                    item_getter = \"ir::basic_block::get($fun, $id)\"
                    id_getter = \"ir::basic_block::Obj::id($item)\"
                    iterator = \"ir::basic_block::iter($fun)\"
                    var_prefix = \"bb\"
                    new_objs = \"$objs.basic_block\"
                  end
                  define enum foo($lhs in BasicBlock, $rhs in BasicBlock2):
                    symmetric
                    value A:
                    value B:
                  end"
                        .to_vec()
                )).unwrap()
                    .type_check()
                    .err(),
                Some(TypeError::BadSymmetricArg {
                    object_name: Spanned {
                        beg: Position::new_optional(LexerPosition::new(18, 30), None),
                        end: Position::new_optional(LexerPosition::new(18, 33), None),
                        data: String::from("foo"),
                    },
                    object_variables: vec![
                        VarDef {
                            name: Spanned {
                                beg: Position::new_optional(
                                    LexerPosition::new(18, 34),
                                    None,
                                ),
                                end: Position::new_optional(
                                    LexerPosition::new(18, 38),
                                    None,
                                ),
                                data: RcStr::new(String::from("lhs")),
                            },
                            set: SetRef {
                                name: RcStr::new(String::from("BasicBlock")),
                                var: None,
                            },
                        },
                        VarDef {
                            name: Spanned {
                                beg: Position::new_optional(
                                    LexerPosition::new(18, 54),
                                    None,
                                ),
                                end: Position::new_optional(
                                    LexerPosition::new(18, 58),
                                    None,
                                ),
                                data: RcStr::new(String::from("rhs")),
                            },
                            set: SetRef {
                                name: RcStr::new(String::from("BasicBlock2")),
                                var: None,
                            },
                        },
                    ],
                })
            );
        }
    }
}

/// Illegal definition of Symmetric and Antisymmetric
/// in the same enumeration.
#[test]
fn conflict() {
    assert_eq!(
        parser::parse_ast(Lexer::new(
            b"set BasicBlock:
            item_type = \"ir::basic_block::Obj\"
            id_type = \"ir::basic_block::Id\"
            item_getter = \"ir::basic_block::get($fun, $id)\"
            id_getter = \"ir::basic_block::Obj::id($item)\"
            iterator = \"ir::basic_block::iter($fun)\"
            var_prefix = \"bb\"
            new_objs = \"$objs.basic_block\"
          end
          define enum foo($lhs in BasicBlock, $rhs in BasicBlock):
            symmetric
            antisymmetric:
              A -> B
            value A:
            value B:
          end"
                .to_vec()
        )).unwrap()
            .type_check()
            .err(),
        Some(TypeError::Conflict {
            object_fields: (
                Spanned {
                    beg: Position::new_optional(LexerPosition::new(10, 12), None),
                    end: Position::new_optional(LexerPosition::new(10, 21), None),
                    data: String::from("Symmetric"),
                },
                Spanned {
                    beg: Position::new_optional(LexerPosition::new(11, 12), None),
                    end: Position::new_optional(LexerPosition::new(12, 20), None),
                    data: String::from("Antisymmetric"),
                }
            )
        })
    )
}
