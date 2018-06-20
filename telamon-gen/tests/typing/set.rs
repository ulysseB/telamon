pub use super::utils::RcStr;

pub use super::telamon_gen::lexer::{Lexer, Spanned, Position};
pub use super::telamon_gen::parser;
pub use super::telamon_gen::ir;
pub use super::telamon_gen::ast::*;

/*

#[test]
fn set_field_redefinition() {
    assert_eq!(parser::parse_ast(Lexer::from(
        b"set Foo:
            item_type = \"ir::inst::Obj\"
            item_type = \"ir::inst::Obj\"
            id_type = \"ir::inst::Id\"
            item_getter = \"ir::inst::get($fun, $id)\"
            id_getter = \"ir::inst::Obj::id($item)\"
            iterator = \"ir::inst::iter($fun)\"
            var_prefix = \"inst\"
            new_objs = \"$objs.inst\"
          end".to_vec())).unwrap().type_check().err(),
        Some(Spanned {
            beg: Position { line: 0, column: 0},
            end: Position { line: 0, column: 8},
            data: TypeError::Redefinition(
                String::from("ItemType"),
                Hint::SetAttribute,
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
            beg: Position { line: 0, column: 0},
            end: Position { line: 0, column: 16},
            data: TypeError::Undefined(ir::SetDefKey::ItemType.to_string())
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
            beg: Position { line: 0, column: 0},
            end: Position { line: 0, column: 16},
            data: TypeError::Undefined(ir::SetDefKey::IdType.to_string())
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
            beg: Position { line: 0, column: 0},
            end: Position { line: 0, column: 16},
            data: TypeError::Undefined(ir::SetDefKey::ItemGetter.to_string())
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
            beg: Position { line: 0, column: 0},
            end: Position { line: 0, column: 16},
            data: TypeError::Undefined(ir::SetDefKey::IdGetter.to_string())
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
            beg: Position { line: 0, column: 0},
            end: Position { line: 0, column: 16},
            data: TypeError::Undefined(ir::SetDefKey::Iter.to_string())
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
fn set_undefined_parameter() {
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
            beg: Position { line: 0, column: 0},
            end: Position { line: 0, column: 34},
            data: TypeError::Undefined(String::from("Instruction"))
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

#[test]
fn set_undefined_subsetof() {
    assert_eq!(parser::parse_ast(Lexer::from(
        b"set Instruction subsetof BasicBlock:
            item_type = \"ir::inst::Obj\"
            id_type = \"ir::inst::Id\"
            item_getter = \"ir::inst::get($fun, $id)\"
            id_getter = \"ir::inst::Obj::id($item)\"
            iterator = \"ir::inst::iter($fun)\"
            var_prefix = \"inst\"
            new_objs = \"$objs.inst\"
            from_superset = \"ir::inst::from_superset($fun, $item)\"
         end".to_vec())).unwrap().type_check().err(),
        Some(Spanned {
            beg: Position { line: 0, column: 0},
            end: Position { line: 0, column: 36},
            data: TypeError::Undefined(String::from("BasicBlock")),
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
          
          set Instruction subsetof BasicBlock:
            item_type = \"ir::inst::Obj\"
            id_type = \"ir::inst::Id\"
            item_getter = \"ir::inst::get($fun, $id)\"
            id_getter = \"ir::inst::Obj::id($item)\"
            iterator = \"ir::inst::iter($fun)\"
            var_prefix = \"inst\"
            new_objs = \"$objs.inst\"
            from_superset = \"ir::inst::from_superset($fun, $item)\"
         end".to_vec())).unwrap().type_check().is_ok()
    );
}

#[test]
fn set_from_superset_key() {
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
          
          set Instruction subsetof BasicBlock:
            item_type = \"ir::inst::Obj\"
            id_type = \"ir::inst::Id\"
            item_getter = \"ir::inst::get($fun, $id)\"
            id_getter = \"ir::inst::Obj::id($item)\"
            iterator = \"ir::inst::iter($fun)\"
            var_prefix = \"inst\"
            new_objs = \"$objs.inst\"
         end".to_vec())).unwrap().type_check().err(),
        Some(Spanned {
            beg: Position { line: 10, column: 10},
            end: Position { line: 10, column: 46},
            data: TypeError::Undefined(ir::SetDefKey::FromSuperset.to_string())
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
          
          set Instruction subsetof BasicBlock:
            item_type = \"ir::inst::Obj\"
            id_type = \"ir::inst::Id\"
            item_getter = \"ir::inst::get($fun, $id)\"
            id_getter = \"ir::inst::Obj::id($item)\"
            iterator = \"ir::inst::iter($fun)\"
            var_prefix = \"inst\"
            new_objs = \"$objs.inst\"
            from_superset = \"ir::inst::from_superset($fun, $item)\"
         end".to_vec())).unwrap().type_check().is_ok()
    );
}
*/
