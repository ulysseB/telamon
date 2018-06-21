pub use super::utils::RcStr;

pub use super::telamon_gen::lexer::{Lexer, Spanned, Position};
pub use super::telamon_gen::parser;
pub use super::telamon_gen::ast::*;

/*
#[test]
fn enum_field_redefinition() {
    assert_eq!(parser::parse_ast(Lexer::from(
        b"define enum foo():
            value A:
            value B:
            value C:
            alias AB = A | B:
            alias AB = A | B:
          end".to_vec())).unwrap().type_check().err(),
        Some(Spanned {
            beg: Position { line: 0, column: 0},
            end: Position { line: 0, column: 18},
            data: TypeError::Redefinition(
                String::from("AB"),
                Hint::EnumAttribute,
            ),
        })
    );
    assert!(parser::parse_ast(Lexer::from(
        b"define enum foo():
            value A:
            value B:
            value C:
            alias AB = A | B:
            alias BC = B | C:
          end".to_vec())).unwrap().type_check().is_ok()
    );
    assert_eq!(parser::parse_ast(Lexer::from(
        b"define enum foo($lhs in BasicBlock, $rhs in BasicBlock):
            symmetric
            symmetric
            value A:
            value B:
          end".to_vec())).unwrap().type_check().err(),
        Some(Spanned {
            beg: Position { line: 0, column: 0},
            end: Position { line: 0, column: 56},
            data: TypeError::Redefinition(
                String::from("Symmetric"),
                Hint::EnumAttribute
            ),
        })
    );
    assert!(parser::parse_ast(Lexer::from(
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
            value A:
            value B:
          end".to_vec())).unwrap().type_check().is_ok()
    );
    assert_eq!(parser::parse_ast(Lexer::from(
        b"define enum foo():
            value A:
            value B:
            value A:
          end".to_vec())).unwrap().type_check().err(),
        Some(Spanned {
            beg: Position { line: 0, column: 0},
            end: Position { line: 0, column: 18},
            data: TypeError::Redefinition(
                String::from("A"),
                Hint::EnumAttribute
            ),
        })
    );
    assert!(parser::parse_ast(Lexer::from(
        b"define enum foo():
            value A:
            value B:
          end".to_vec())).unwrap().type_check().is_ok()
    );
    assert!(parser::parse_ast(Lexer::from(
        b"define enum foo():
            value A:
            value B:
            value C:
            alias AB = A | B:
            alias BC = B | C:
            alias ABBC = AB | BC:
          end".to_vec())).unwrap().type_check().is_ok()
    );
}

#[test]
fn enum_symmetric_two_parameter() {
    assert_eq!(parser::parse_ast(Lexer::from(
        b"define enum foo():
            symmetric
            value A:
            value B:
          end".to_vec())).unwrap().type_check().err(),
        Some(Spanned {
            beg: Position { line: 0, column: 0},
            end: Position { line: 0, column: 18},
            data: TypeError::BadSymmetricArg(vec![])
        })
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
            ])
        })
    );
    assert!(parser::parse_ast(Lexer::from(
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
            value A:
            value B:
          end".to_vec())).unwrap().type_check().is_ok()
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
            ])
        })
    );
}

#[test]
fn enum_symmetric_same_parameter() {
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
                ])
        })
    );
    assert!(parser::parse_ast(Lexer::from(
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
            value A:
            value B:
          end".to_vec())).unwrap().type_check().is_ok()
    );
}


#[test]
fn enum_undefined_value() {
    assert_eq!(parser::parse_ast(Lexer::from(
        b"define enum foo():
            value A:
            alias AB = A | B:
          end".to_vec())).unwrap().type_check().err(),
        Some(Spanned {
            beg: Position { line: 0, column: 0},
            end: Position { line: 0, column: 18},
            data: TypeError::Undefined(String::from("B"))
        })
    );
}
 
#[test]
fn enum_undefined_parameter() {
    assert_eq!(parser::parse_ast(Lexer::from(
            b"define enum foo($lhs in BasicBlock, $rhs in BasicBlock):
                symmetric
                value A:
                value B:
          end".to_vec())).unwrap().type_check().err(),
        Some(Spanned {
            beg: Position { line: 0, column: 0},
            end: Position { line: 0, column: 56},
            data: TypeError::Undefined(String::from("BasicBlock"))
        })
    );
}
*/
