pub use super::utils::RcStr;

pub use super::telamon_gen::lexer::{Lexer, Spanned, Position};
pub use super::telamon_gen::parser;
pub use super::telamon_gen::ast::*;

/// Undefined
#[cfg(test)]
mod undefined {
    pub use super::*;

    /// Missing the set BasickBlock from a Emum.
    #[test]
    fn parameter() {
        assert_eq!(parser::parse_ast(Lexer::from(
                b"define enum foo($lhs in BasicBlock, $rhs in BasicBlock):
                    symmetric
                    value A:
                    value B:
              end".to_vec())).unwrap().type_check().err(),
            Some(TypeError::Undefined(Spanned {
                beg: Position { line: 0, column: 12},
                end: Position { line: 0, column: 15},
                data: String::from("BasicBlock")
            }))
        );
    }

    /// Missing the set BasickBlock from a Emum.
    #[test]
    fn value() {
        assert_eq!(parser::parse_ast(Lexer::from(
            b"define enum foo():
                value A:
                alias AB = A | B:
              end".to_vec())).unwrap().type_check().err(),
            Some(TypeError::Undefined(Spanned {
                beg: Position { line: 2, column: 22},
                end: Position { line: 2, column: 24},
                data: String::from("B")
            }))
        );
    }
}

/// Redefinition
#[cfg(test)]
mod redefinition {
    pub use super::*;

    /// Redefinition of the foo Enum.
    #[test]
    fn enum_() {
        assert_eq!(parser::parse_ast(Lexer::from(
            b"define enum foo():
              end
              
              define enum foo():
              end".to_vec())).unwrap().type_check().err(),
            Some(TypeError::Redefinition(Spanned {
                beg: Position { line: 0, column: 12},
                end: Position { line: 0, column: 15},
                data: Hint::Enum,
            }, Spanned {
                beg: Position { line: 3, column: 26},
                end: Position { line: 3, column: 29},
                data: String::from("foo"),
            }))
        );
    }

    #[test]
    fn field() {
        assert_eq!(parser::parse_ast(Lexer::from(
            b"define enum foo():
                value A:
                value B:
                value C:
                alias AB = A | B:
                alias AB = A | B:
              end".to_vec())).unwrap().type_check().err(),
            Some(TypeError::Redefinition(Spanned {
                beg: Position { line: 5, column: 22 },
                end: Position { line: 5, column: 24 },
                data: Hint::EnumAttribute,
            }, Spanned {
                beg: Position { line: 5, column: 22 },
                end: Position { line: 5, column: 24 },
                data: String::from("AB"),
            }))
        );
        assert_eq!(parser::parse_ast(Lexer::from(
            b"define enum foo():
                value A:
                value B:
                value A:
              end".to_vec())).unwrap().type_check().err(),
            Some(TypeError::Redefinition(Spanned {
                beg: Position { line: 3, column: 22 },
                end: Position { line: 3, column: 23 },
                data: Hint::EnumAttribute,
            }, Spanned {
                beg: Position { line: 3, column: 22 },
                end: Position { line: 3, column: 23 },
                data: String::from("A"),
            }))
        );
    }

    mod antisymmetric {
        pub use super::*;

        #[test]
        fn field() {
            assert_eq!(parser::parse_ast(Lexer::from(
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
                  end".to_vec())).unwrap().type_check().err(),
                Some(TypeError::Redefinition(Spanned {
                    beg: Position { line: 10, column: 20 },
                    end: Position { line: 11, column: 28 },
                    data: Hint::EnumAttribute,
                }, Spanned {
                    beg: Position { line: 12, column: 20 },
                    end: Position { line: 13, column: 28 },
                    data: String::from("Antisymmetric"),
                }))
            );
        }
    }

    mod symmetric {
        pub use super::*;

        #[test]
        fn field() {
            assert_eq!(parser::parse_ast(Lexer::from(
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
                  end".to_vec())).unwrap().type_check().err(),
                Some(TypeError::Redefinition(Spanned {
                    beg: Position { line: 10, column: 20 },
                    end: Position { line: 10, column: 29 },
                    data: Hint::EnumAttribute,
                }, Spanned {
                    beg: Position { line: 11, column: 20 },
                    end: Position { line: 11, column: 29 },
                    data: String::from("Symmetric"),
                }))
            );
        }
    }
}

/// Parameter
#[cfg(test)]
mod parameter {
    pub use super::*;

    mod antisymmetric {
        pub use super::*;

        #[test]
        fn two() {
            assert_eq!(parser::parse_ast(Lexer::from(
                b"define enum foo():
                    antisymmetric:
                      A -> B
                    value A:
                    value B:
                  end".to_vec())).unwrap().type_check().err(),
                Some(TypeError::BadSymmetricArg(Spanned {
                    beg: Position { line: 0, column: 12 },
                    end: Position { line: 0, column: 15 },
                    data: String::from("foo"),
                }, vec![]))
            );
            assert_eq!(parser::parse_ast(Lexer::from(
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
                  end".to_vec())).unwrap().type_check().err(),
                Some(TypeError::BadSymmetricArg(Spanned {
                    beg: Position { line: 9, column: 30 },
                    end: Position { line: 9, column: 33 },
                    data: String::from("foo"),
                }, vec![
                    VarDef {
                        name: Spanned {
                            beg: Position { line: 9, column: 34 },
                            end: Position { line: 9, column: 38 },
                            data: RcStr::new(String::from("lhs"))
                        },
                        set: SetRef {
                            name: RcStr::new(String::from("BasicBlock")),
                            var: None
                        }
                    }
                ]))
            );
            assert_eq!(parser::parse_ast(Lexer::from(
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
                    antisymmetric:
                      A -> B
                    value A:
                    value B:
                  end".to_vec())).unwrap().type_check().err(),
                Some(TypeError::BadSymmetricArg(Spanned {
                    beg: Position { line: 9, column: 30 },
                    end: Position { line: 9, column: 33 },
                    data: String::from("foo"),
                }, vec![
                    VarDef {
                        name: Spanned {
                            beg: Position { line: 9, column: 34 },
                            end: Position { line: 9, column: 38 },
                            data: RcStr::new(String::from("lhs"))
                        },
                        set: SetRef {
                            name: RcStr::new(String::from("BasicBlock")),
                            var: None
                        }
                    },
                    VarDef {
                        name: Spanned {
                            beg: Position { line: 10, column: 34 },
                            end: Position { line: 10, column: 38 },
                            data: RcStr::new(String::from("chs"))
                        },
                        set: SetRef {
                            name: RcStr::new(String::from("BasicBlock")),
                            var: None
                        }
                    },
                    VarDef {
                        name: Spanned {
                            beg: Position { line: 11, column: 34 },
                            end: Position { line: 11, column: 38 },
                            data: RcStr::new(String::from("rhs"))
                        },
                        set: SetRef {
                            name: RcStr::new(String::from("BasicBlock")),
                            var: None
                        }
                    },
                ]))
            );
        }

        #[test]
        fn same() {
            assert_eq!(parser::parse_ast(Lexer::from(
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
                  end".to_vec())).unwrap().type_check().err(),
                Some(TypeError::BadSymmetricArg(Spanned {
                    beg: Position { line: 18, column: 30 },
                    end: Position { line: 18, column: 33 },
                    data: String::from("foo"),
                }, vec![
                    VarDef {
                        name: Spanned {
                            beg: Position { line: 18, column: 34 },
                            end: Position { line: 18, column: 38 },
                            data: RcStr::new(String::from("lhs"))
                        },
                        set: SetRef {
                            name: RcStr::new(String::from("BasicBlock")),
                            var: None
                        }
                    },
                    VarDef {
                        name: Spanned {
                            beg: Position { line: 18, column: 54 },
                            end: Position { line: 18, column: 58 },
                            data: RcStr::new(String::from("rhs"))
                        },
                        set: SetRef {
                            name: RcStr::new(String::from("BasicBlock2")),
                            var: None
                        }
                    }
                ]))
            );
        }
    }

    mod symmetric {
        pub use super::*;

        #[test]
        fn two() {
            assert_eq!(parser::parse_ast(Lexer::from(
                b"define enum foo():
                    symmetric
                    value A:
                    value B:
                  end".to_vec())).unwrap().type_check().err(),
                Some(TypeError::BadSymmetricArg(Spanned {
                    beg: Position { line: 0, column: 12 },
                    end: Position { line: 0, column: 15 },
                    data: String::from("foo"),
                }, vec![]))
            );
            assert_eq!(parser::parse_ast(Lexer::from(
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
                  end".to_vec())).unwrap().type_check().err(),
                Some(TypeError::BadSymmetricArg(Spanned {
                    beg: Position { line: 9, column: 30 },
                    end: Position { line: 9, column: 33 },
                    data: String::from("foo"),
                }, vec![
                    VarDef {
                        name: Spanned {
                            beg: Position { line: 9, column: 34 },
                            end: Position { line: 9, column: 38 },
                            data: RcStr::new(String::from("lhs"))
                        },
                        set: SetRef {
                            name: RcStr::new(String::from("BasicBlock")),
                            var: None
                        }
                    }
                ]))
            );
            assert_eq!(parser::parse_ast(Lexer::from(
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
                  end".to_vec())).unwrap().type_check().err(),
                Some(TypeError::BadSymmetricArg(Spanned {
                    beg: Position { line: 9, column: 30 },
                    end: Position { line: 9, column: 33 },
                    data: String::from("foo"),
                }, vec![
                    VarDef {
                        name: Spanned {
                            beg: Position { line: 9, column: 34 },
                            end: Position { line: 9, column: 38 },
                            data: RcStr::new(String::from("lhs"))
                        },
                        set: SetRef {
                            name: RcStr::new(String::from("BasicBlock")),
                            var: None
                        }
                    },
                    VarDef {
                        name: Spanned {
                            beg: Position { line: 10, column: 34 },
                            end: Position { line: 10, column: 38 },
                            data: RcStr::new(String::from("chs"))
                        },
                        set: SetRef {
                            name: RcStr::new(String::from("BasicBlock")),
                            var: None
                        }
                    },
                    VarDef {
                        name: Spanned {
                            beg: Position { line: 11, column: 34 },
                            end: Position { line: 11, column: 38 },
                            data: RcStr::new(String::from("rhs"))
                        },
                        set: SetRef {
                            name: RcStr::new(String::from("BasicBlock")),
                            var: None
                        }
                    },
                ]))
            );
        }

        #[test]
        fn same() {
            assert_eq!(parser::parse_ast(Lexer::from(
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
                  end".to_vec())).unwrap().type_check().err(),
                Some(TypeError::BadSymmetricArg(Spanned {
                    beg: Position { line: 18, column: 30 },
                    end: Position { line: 18, column: 33 },
                    data: String::from("foo"),
                }, vec![
                    VarDef {
                        name: Spanned {
                            beg: Position { line: 18, column: 34 },
                            end: Position { line: 18, column: 38 },
                            data: RcStr::new(String::from("lhs"))
                        },
                        set: SetRef {
                            name: RcStr::new(String::from("BasicBlock")),
                            var: None
                        }
                    },
                    VarDef {
                        name: Spanned {
                            beg: Position { line: 18, column: 54 },
                            end: Position { line: 18, column: 58 },
                            data: RcStr::new(String::from("rhs"))
                        },
                        set: SetRef {
                            name: RcStr::new(String::from("BasicBlock2")),
                            var: None
                        }
                    }
                ]))
            );
        }
    }
}
