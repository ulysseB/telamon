extern crate telamon_utils as utils;
extern crate telamon_gen;
extern crate lalrpop_util;

use utils::RcStr;

use telamon_gen::lexer::{Lexer, Spanned, Position};
use telamon_gen::parser;
use telamon_gen::ast::*;

#[test]
fn enum_name_multi() {
    assert_eq!(parser::parse_ast(Lexer::from(
        b"define enum foo():
            value A:
            value B:
            value C:
          end
          
          define enum foo():
            value A:
          end".to_vec())).unwrap().type_check().err(),
        Some(Spanned {
            leg: Position { line: 6, column: 10},
            end: Position { line: 6, column: 28},
            data: telamon_gen::ast::TypeError::EnumNameMulti(
                String::from("foo")
            )
        })
    );
    assert!(parser::parse_ast(Lexer::from(
        b"define enum foo():
            value A:
            value B:
            value C:
          end
          
          define enum bar():
            value A:
          end".to_vec())).unwrap().type_check().is_ok()
    );
}

#[test]
fn enum_field_name_multi() {
    assert_eq!(parser::parse_ast(Lexer::from(
        b"define enum foo():
            value A:
            value B:
            value C:
            alias AB = A | B:
            alias AB = A | B:
          end".to_vec())).unwrap().type_check().err(),
        Some(Spanned {
            leg: Position { line: 0, column: 0},
            end: Position { line: 0, column: 18},
            data: telamon_gen::ast::TypeError::EnumFieldNameMulti(
                EnumStatement::Alias(
                    String::from("AB"),
                    None,
                    vec![String::from("A"), String::from("B")],
                    vec![]
                )
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
            leg: Position { line: 0, column: 0},
            end: Position { line: 0, column: 56},
            data: telamon_gen::ast::TypeError::EnumFieldNameMulti(
                EnumStatement::Symmetric
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
            leg: Position { line: 0, column: 0},
            end: Position { line: 0, column: 18},
            data: telamon_gen::ast::TypeError::EnumFieldNameMulti(
                EnumStatement::Value(String::from("A"), None, vec![])
            ),
        })
    );
    assert!(parser::parse_ast(Lexer::from(
        b"define enum foo():
            value A:
            value B:
          end".to_vec())).unwrap().type_check().is_ok()
    );
}

#[test]
fn enum_symmetric_two_parametric() {
    assert_eq!(parser::parse_ast(Lexer::from(
        b"define enum foo():
            symmetric
            value A:
            value B:
          end".to_vec())).unwrap().type_check().err(),
        Some(Spanned {
            leg: Position { line: 0, column: 0},
            end: Position { line: 0, column: 18},
            data: telamon_gen::ast::TypeError::EnumSymmetricTwoParametric(0)
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
            leg: Position { line: 9, column: 10},
            end: Position { line: 9, column: 46},
            data: telamon_gen::ast::TypeError::EnumSymmetricTwoParametric(1)
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
            leg: Position { line: 9, column: 10},
            end: Position { line: 11, column: 46},
            data: telamon_gen::ast::TypeError::EnumSymmetricTwoParametric(3)
        })
    );
}

#[test]
fn enum_symmetric_same_parametric() {
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
            leg: Position { line: 18, column: 10},
            end: Position { line: 18, column: 67},
            data: telamon_gen::ast::TypeError::EnumSymmetricSameParametric(
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
            )
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
fn enum_alias_multi() {
    assert_eq!(parser::parse_ast(Lexer::from(
        b"define enum foo():
            value A:
            alias AB = A | B:
          end".to_vec())).unwrap().type_check().err(),
        Some(Spanned {
            leg: Position { line: 0, column: 0},
            end: Position { line: 0, column: 18},
            data: telamon_gen::ast::TypeError::EnumAliasValueMissing(
                String::from("B")
            )
        })
    );
    assert!(parser::parse_ast(Lexer::from(
        b"define enum foo():
            value A:
            value B:
            alias AB = A | B:
          end".to_vec())).unwrap().type_check().is_ok()
    );
}
