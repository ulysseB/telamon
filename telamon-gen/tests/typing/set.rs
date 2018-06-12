pub use super::utils::RcStr;

pub use super::telamon_gen::lexer::{Lexer, Spanned, Position};
pub use super::telamon_gen::parser;
pub use super::telamon_gen::ir;
pub use super::telamon_gen::ast::*;

#[test]
fn set_name_redefinition() {
    assert_eq!(parser::parse_ast(Lexer::from(
        b"set Foo:
            item_type = \"ir::inst::Obj\"
            id_type = \"ir::inst::Id\"
            item_getter = \"ir::inst::get($fun, $id)\"
            id_getter = \"ir::inst::Obj::id($item)\"
            iterator = \"ir::inst::iter($fun)\"
            var_prefix = \"inst\"
            new_objs = \"$objs.inst\"
          end
          
          set Foo:
            item_type = \"ir::inst::Obj\"
            id_type = \"ir::inst::Id\"
            item_getter = \"ir::inst::get($fun, $id)\"
            id_getter = \"ir::inst::Obj::id($item)\"
            iterator = \"ir::inst::iter($fun)\"
            var_prefix = \"inst\"
            new_objs = \"$objs.inst\"
          end".to_vec())).unwrap().type_check().err(),
        Some(Spanned {
            leg: Position { line: 10, column: 10},
            end: Position { line: 10, column: 18},
            data: TypeError::SetRedefinition(
                SetDef {
                    name: String::from("Foo"),
                    ..Default::default()
                }
            )
        })
    );
}

#[test]
fn set_undefined_key() {
    assert_eq!(parser::parse_ast(Lexer::from(
        b"set Instruction:
            id_type = \"ir::inst::Id\"
            item_getter = \"ir::inst::get($fun, $id)\"
            id_getter = \"ir::inst::Obj::id($item)\"
            iterator = \"ir::inst::iter($fun)\"
            var_prefix = \"inst\"
            new_objs = \"$objs.inst\"
          end".to_vec())).unwrap().type_check().err(),
        Some(Spanned {
            leg: Position { line: 0, column: 0},
            end: Position { line: 0, column: 16},
            data: TypeError::SetUndefinedKey(ir::SetDefKey::ItemType)
        })
    );
    assert_eq!(parser::parse_ast(Lexer::from(
        b"set Instruction:
            item_type = \"ir::inst::Obj\"
            item_getter = \"ir::inst::get($fun, $id)\"
            id_getter = \"ir::inst::Obj::id($item)\"
            iterator = \"ir::inst::iter($fun)\"
            var_prefix = \"inst\"
            new_objs = \"$objs.inst\"
          end".to_vec())).unwrap().type_check().err(),
        Some(Spanned {
            leg: Position { line: 0, column: 0},
            end: Position { line: 0, column: 16},
            data: TypeError::SetUndefinedKey(ir::SetDefKey::IdType)
        })
    );
    assert_eq!(parser::parse_ast(Lexer::from(
        b"set Instruction:
            item_type = \"ir::inst::Obj\"
            id_type = \"ir::inst::Id\"
            id_getter = \"ir::inst::Obj::id($item)\"
            iterator = \"ir::inst::iter($fun)\"
            var_prefix = \"inst\"
            new_objs = \"$objs.inst\"
          end".to_vec())).unwrap().type_check().err(),
        Some(Spanned {
            leg: Position { line: 0, column: 0},
            end: Position { line: 0, column: 16},
            data: TypeError::SetUndefinedKey(ir::SetDefKey::ItemGetter)
        })
    );
    assert_eq!(parser::parse_ast(Lexer::from(
        b"set Instruction:
            item_type = \"ir::inst::Obj\"
            id_type = \"ir::inst::Id\"
            item_getter = \"ir::inst::get($fun, $id)\"
            iterator = \"ir::inst::iter($fun)\"
            var_prefix = \"inst\"
            new_objs = \"$objs.inst\"
          end".to_vec())).unwrap().type_check().err(),
        Some(Spanned {
            leg: Position { line: 0, column: 0},
            end: Position { line: 0, column: 16},
            data: TypeError::SetUndefinedKey(ir::SetDefKey::IdGetter)
        })
    );
    assert_eq!(parser::parse_ast(Lexer::from(
        b"set Instruction:
            item_type = \"ir::inst::Obj\"
            id_type = \"ir::inst::Id\"
            item_getter = \"ir::inst::get($fun, $id)\"
            id_getter = \"ir::inst::Obj::id($item)\"
            var_prefix = \"inst\"
            new_objs = \"$objs.inst\"
          end".to_vec())).unwrap().type_check().err(),
        Some(Spanned {
            leg: Position { line: 0, column: 0},
            end: Position { line: 0, column: 16},
            data: TypeError::SetUndefinedKey(ir::SetDefKey::Iter)
        })
    );
    assert!(parser::parse_ast(Lexer::from(
        b"set Instruction:
            item_type = \"ir::inst::Obj\"
            id_type = \"ir::inst::Id\"
            item_getter = \"ir::inst::get($fun, $id)\"
            id_getter = \"ir::inst::Obj::id($item)\"
            iterator = \"ir::inst::iter($fun)\"
            var_prefix = \"inst\"
            new_objs = \"$objs.inst\"
          end".to_vec())).unwrap().type_check().is_ok()
    );
}

#[test]
fn set_undefined_parametric() {
    assert_eq!(parser::parse_ast(Lexer::from(
        b"set Operand($inst in Instruction):
            item_type = \"ir::operand::Obj\"
            id_type = \"ir::operand::Id\"
            item_getter = \"ir::operand::get($fun, $inst, $id)\"
            id_getter = \"$item.id()\"
            iterator = \"ir::operand::iter($fun, ir::inst::Obj::id($inst))\"
            var_prefix = \"op\"
            new_objs = \"$objs.operand\"
          end".to_vec())).unwrap().type_check().err(),
        Some(Spanned {
            leg: Position { line: 0, column: 0},
            end: Position { line: 0, column: 34},
            data: TypeError::SetUndefinedParametric(
                SetDef {
                    name: String::from("Instruction"),
                    ..Default::default()
                }
            )
        })
    );
    assert!(parser::parse_ast(Lexer::from(
        b"set Instruction:
            item_type = \"ir::inst::Obj\"
            id_type = \"ir::inst::Id\"
            item_getter = \"ir::inst::get($fun, $id)\"
            id_getter = \"ir::inst::Obj::id($item)\"
            iterator = \"ir::inst::iter($fun)\"
            var_prefix = \"inst\"
            new_objs = \"$objs.inst\"
          end

          set Operand($inst in Instruction):
            item_type = \"ir::operand::Obj\"
            id_type = \"ir::operand::Id\"
            item_getter = \"ir::operand::get($fun, $inst, $id)\"
            id_getter = \"$item.id()\"
            iterator = \"ir::operand::iter($fun, ir::inst::Obj::id($inst))\"
            var_prefix = \"op\"
            new_objs = \"$objs.operand\"
          end".to_vec())).unwrap().type_check().is_ok()
    );
}
