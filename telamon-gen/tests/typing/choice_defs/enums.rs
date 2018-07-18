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
        assert_eq!(parser::parse_ast(Lexer::new(
                b"define enum foo($lhs in BasicBlock, $rhs in BasicBlock):
                    symmetric
                    value A:
                    value B:
              end".to_vec())).unwrap().type_check().err(),
            Some(TypeError::Undefined(Spanned {
                beg: Position { line: 0, column: 12},
                end: Position { line: 0, column: 15},
                data: String::from("BasicBlock"),
                filename: Default::default()
            }))
        );
    }

    /// Missing the set BasickBlock from a Emum.
    #[test]
    fn value() {
        assert_eq!(parser::parse_ast(Lexer::new(
            b"define enum foo():
                value A:
                alias AB = A | B:
              end".to_vec())).unwrap().type_check().err(),
            Some(TypeError::Undefined(Spanned {
                beg: Position { line: 2, column: 22},
                end: Position { line: 2, column: 24},
                data: String::from("B"),
                filename: Default::default()
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
        assert_eq!(parser::parse_ast(Lexer::new(
            b"define enum foo():
              end
              
              define enum foo():
              end".to_vec())).unwrap().type_check().err(),
            Some(TypeError::Redefinition(Spanned {
                beg: Position { line: 0, column: 12},
                end: Position { line: 0, column: 15},
                data: Hint::Enum,
                filename: Default::default()
            }, Spanned {
                beg: Position { line: 3, column: 26},
                end: Position { line: 3, column: 29},
                data: String::from("foo"),
                filename: Default::default()
            }))
        );
    }

    #[test]
    fn field() {
        assert_eq!(parser::parse_ast(Lexer::new(
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
                filename: Default::default()
            }, Spanned {
                beg: Position { line: 5, column: 22 },
                end: Position { line: 5, column: 24 },
                data: String::from("AB"),
                filename: Default::default()
            }))
        );
        assert_eq!(parser::parse_ast(Lexer::new(
            b"define enum foo():
                value A:
                value B:
                value A:
              end".to_vec())).unwrap().type_check().err(),
            Some(TypeError::Redefinition(Spanned {
                beg: Position { line: 3, column: 22 },
                end: Position { line: 3, column: 23 },
                data: Hint::EnumAttribute,
                filename: Default::default()
            }, Spanned {
                beg: Position { line: 3, column: 22 },
                end: Position { line: 3, column: 23 },
                data: String::from("A"),
                filename: Default::default()
            }))
        );
        assert_eq!(parser::parse_ast(Lexer::new(
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
                beg: Position { line: 10, column: 16 },
                end: Position { line: 10, column: 25 },
                data: Hint::EnumAttribute,
                filename: Default::default()
            }, Spanned {
                beg: Position { line: 11, column: 16 },
                end: Position { line: 11, column: 25 },
                data: String::from("Symmetric"),
                filename: Default::default()
            }))
        );
    }
}

/*
#[test]
fn enum_symmetric_two_parameters() {
    assert_eq!(parser::parse_ast(Lexer::new(
        b"define enum foo():
            symmetric
            value A:
            value B:
          end".to_vec())).unwrap().type_check().err(),
        Some(Spanned {
            beg: Position { line: 0, column: 0},
            end: Position { line: 0, column: 18},
            data: TypeError::BadSymmetricArg(vec![]),
            filename: Default::default()
        })
    );
    assert_eq!(parser::parse_ast(Lexer::new(
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
        Some(Spanned {
            beg: Position { line: 9, column: 10},
            end: Position { line: 9, column: 46},
            data: TypeError::BadSymmetricArg(vec![
                VarDef {
                    name: RcStr::new(String::from("lhs")),
                    set: SetRef {
                        name: RcStr::new(String::from("BasicBlock")),
                        var: None
                    }
                },
            ]),
            filename: Default::default()
        })
    );
    assert_eq!(parser::parse_ast(Lexer::new(
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
        Some(Spanned {
            beg: Position { line: 9, column: 10},
            end: Position { line: 11, column: 46},
            data: TypeError::BadSymmetricArg(vec![
                VarDef {
                    name: RcStr::new(String::from("lhs")),
                    set: SetRef {
                        name: RcStr::new(String::from("BasicBlock")),
                        var: None
                    }
                },
                VarDef {
                    name: RcStr::new(String::from("chs")),
                    set: SetRef {
                        name: RcStr::new(String::from("BasicBlock")),
                        var: None
                    }
                },
                VarDef {
                    name: RcStr::new(String::from("rhs")),
                    set: SetRef {
                        name: RcStr::new(String::from("BasicBlock")),
                        var: None
                    }
                },
            ]),
            filename: Default::default()
        })
    );
}

#[test]
fn enum_symmetric_same_parameter() {
    assert_eq!(parser::parse_ast(Lexer::new(
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
        Some(Spanned {
            beg: Position { line: 18, column: 10},
            end: Position { line: 18, column: 67},
            data: TypeError::BadSymmetricArg(vec![
                VarDef {
                    name: RcStr::new(String::from("lhs")),
                    set: SetRef {
                        name: RcStr::new(String::from("BasicBlock")),
                        var: None
                    }
                },
                VarDef {
                    name: RcStr::new(String::from("rhs")),
                    set: SetRef {
                        name: RcStr::new(String::from("BasicBlock2")),
                        var: None
                    }
                }
            ]),
            filename: Default::default()
        })
    );
}
*/
