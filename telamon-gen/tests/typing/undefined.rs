pub use super::utils::RcStr;

pub use super::telamon_gen::lexer::{Lexer, Spanned, Position};
pub use super::telamon_gen::parser;
pub use super::telamon_gen::ast::*;
pub use super::telamon_gen::ir;

/// Missing the ItemType's key from Set.
#[test]
fn undefined_item_type_from_set() {
    assert_eq!(parser::parse_ast(Lexer::from(
        b"set Instruction:
            id_type = \"ir::inst::Id\"
            item_getter = \"ir::inst::get($fun, $id)\"
            id_getter = \"ir::inst::Obj::id($item)\"
            iterator = \"ir::inst::iter($fun)\"
            var_prefix = \"inst\"
            new_objs = \"$objs.inst\"
          end".to_vec())).unwrap().type_check().err(),
        Some(TypeError::MissingEntry(String::from("Instruction"), Spanned {
            beg: Position { line: 0, column: 0},
            end: Position { line: 0, column: 16},
            data: ir::SetDefKey::ItemType.to_string()
        }))
    );
}

/// Missing the IdType's key from Set.
#[test]
fn undefined_id_type_from_set() {
    assert_eq!(parser::parse_ast(Lexer::from(
        b"set Instruction:
            item_type = \"ir::inst::Obj\"
            item_getter = \"ir::inst::get($fun, $id)\"
            id_getter = \"ir::inst::Obj::id($item)\"
            iterator = \"ir::inst::iter($fun)\"
            var_prefix = \"inst\"
            new_objs = \"$objs.inst\"
          end".to_vec())).unwrap().type_check().err(),
        Some(TypeError::MissingEntry(String::from("Instruction"), Spanned {
            beg: Position { line: 0, column: 0},
            end: Position { line: 0, column: 16},
            data: ir::SetDefKey::IdType.to_string()
        }))
    );
}

/// Missing the ItemGetter's key from Set.
#[test]
fn undefined_item_getter_from_set() {
    assert_eq!(parser::parse_ast(Lexer::from(
        b"set Instruction:
            item_type = \"ir::inst::Obj\"
            id_type = \"ir::inst::Id\"
            id_getter = \"ir::inst::Obj::id($item)\"
            iterator = \"ir::inst::iter($fun)\"
            var_prefix = \"inst\"
            new_objs = \"$objs.inst\"
          end".to_vec())).unwrap().type_check().err(),
        Some(TypeError::MissingEntry(String::from("Instruction"), Spanned {
            beg: Position { line: 0, column: 0},
            end: Position { line: 0, column: 16},
            data: ir::SetDefKey::ItemGetter.to_string()
        }))
    );
}

/// Missing the IdGetter's key from Set.
#[test]
fn undefined_id_getter_from_set() {
    assert_eq!(parser::parse_ast(Lexer::from(
        b"set Instruction:
            item_type = \"ir::inst::Obj\"
            id_type = \"ir::inst::Id\"
            item_getter = \"ir::inst::get($fun, $id)\"
            iterator = \"ir::inst::iter($fun)\"
            var_prefix = \"inst\"
            new_objs = \"$objs.inst\"
          end".to_vec())).unwrap().type_check().err(),
        Some(TypeError::MissingEntry(String::from("Instruction"), Spanned {
            beg: Position { line: 0, column: 0},
            end: Position { line: 0, column: 16},
            data: ir::SetDefKey::IdGetter.to_string()
        }))
    );
}

/// Missing the Iter's key from Set.
#[test]
fn undefined_iter_from_set() {
    assert_eq!(parser::parse_ast(Lexer::from(
        b"set Instruction:
            item_type = \"ir::inst::Obj\"
            id_type = \"ir::inst::Id\"
            item_getter = \"ir::inst::get($fun, $id)\"
            id_getter = \"ir::inst::Obj::id($item)\"
            var_prefix = \"inst\"
            new_objs = \"$objs.inst\"
          end".to_vec())).unwrap().type_check().err(),
        Some(TypeError::MissingEntry(String::from("Instruction"), Spanned {
            beg: Position { line: 0, column: 0},
            end: Position { line: 0, column: 16},
            data: ir::SetDefKey::Iter.to_string()
        }))
    );
}

/// Missing the set MySet from a Integer.
#[test]
fn undefined_parameter_from_integer() {
    assert_eq!(parser::parse_ast(Lexer::from(
        b"define integer foo($arg in MySet): \"mycode\"
          end".to_vec())).unwrap().type_check().err(),
        Some(TypeError::Undefined(Spanned {
            beg: Position { line: 0, column: 0},
            end: Position { line: 1, column: 13},
            data: String::from("MySet"),
        }))
    );
}

/// Missing the set BasickBlock from a Emum.
#[test]
fn undefined_parameter_from_enum() {
    assert_eq!(parser::parse_ast(Lexer::from(
            b"define enum foo($lhs in BasicBlock, $rhs in BasicBlock):
                symmetric
                value A:
                value B:
          end".to_vec())).unwrap().type_check().err(),
        Some(TypeError::Undefined(Spanned {
            beg: Position { line: 0, column: 0},
            end: Position { line: 0, column: 56},
            data: String::from("BasicBlock")
        }))
    );
}

/// Missing the set Instruction from aSet
#[test]
fn undefined_parameter_from_set() {
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
        Some(TypeError::Undefined(Spanned {
            beg: Position { line: 0, column: 0},
            end: Position { line: 0, column: 34},
            data: String::from("Instruction"),
        }))
    );
}

/// Missing the subset BasicBlock from a Set.
#[test]
fn undefined_subsetof_from_set() {
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
        Some(TypeError::Undefined(Spanned {
            beg: Position { line: 0, column: 0},
            end: Position { line: 0, column: 36},
            data: String::from("BasicBlock"),
        }))
    );
}
