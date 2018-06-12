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
fn set_missing_key() {
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
            data: TypeError::SetMissingKey(ir::SetDefKey::ItemType)
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
