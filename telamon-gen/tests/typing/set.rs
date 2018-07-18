pub use super::utils::RcStr;

pub use super::telamon_gen::lexer::{Lexer, Spanned, Position};
pub use super::telamon_gen::parser;
pub use super::telamon_gen::ir;
pub use super::telamon_gen::ast::*;

#[cfg(test)]
mod redefinition {
    pub use super::*;

    /// Redefinition of the Foo Set.
    #[test]
    fn set() {
        assert_eq!(parser::parse_ast(Lexer::new(
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
            Some(TypeError::Redefinition(Spanned {
                beg: Position { line: 0, column: 4},
                end: Position { line: 0, column: 7},
                data: Hint::Set,
                filename: Default::default()
            }, Spanned {
                beg: Position { line: 9, column: 16},
                end: Position { line: 9, column: 19},
                data:  String::from("Foo"),
                filename: Default::default()
            }))
        );
    }

    /// Redefinition of the Field from Set.
    /// TODO: fixe position
    #[test]
    fn field() {
        assert_eq!(parser::parse_ast(Lexer::new(
          b"set Foo:
                item_type = \"ir::inst::Obj\"
                id_type = \"ir::inst::Id\"
                item_getter = \"ir::inst::get($fun, $id)\"
                id_getter = \"ir::inst::Obj::id($item)\"
                iterator = \"ir::inst::iter($fun)\"
                var_prefix = \"inst\"
                new_objs = \"$objs.inst\"
                new_objs = \"$objs.inst\"
            end".to_vec())).unwrap().type_check().err(),
            Some(TypeError::Redefinition(Spanned {
                beg: Default::default(),
                end: Default::default(),
                data: Hint::Set,
                filename: Default::default()
            }, Spanned {
                beg: Default::default(),
                end: Default::default(),
                data:  String::from("NewObjs"),
                filename: Default::default()
            }))
        );
    }
}

/// Undefined
#[cfg(test)]
mod undefined {
    pub use super::*;

    /// Missing the set Instruction from a Set
    #[test]
    fn parameter() {
        assert_eq!(parser::parse_ast(Lexer::new(
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
                beg: Position { line: 0, column: 4},
                end: Position { line: 0, column: 11},
                data: String::from("Instruction"),
                filename: Default::default()
            }))
        );
    }

    /// Missing the subset BasicBlock from a Set.
    #[test]
    fn subsetof() {
        assert_eq!(parser::parse_ast(Lexer::new(
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
                beg: Position { line: 0, column: 4},
                end: Position { line: 0, column: 15},
                data: String::from("BasicBlock"),
                filename: Default::default()
            }))
        );
    }
}

/// Missing Entry
#[cfg(test)]
mod missing_entry {
    pub use super::*;

    /// Missing the ItemType's key from Set.
    #[test]
    fn item_type() {
        assert_eq!(parser::parse_ast(Lexer::new(
            b"set Instruction:
                id_type = \"ir::inst::Id\"
                item_getter = \"ir::inst::get($fun, $id)\"
                id_getter = \"ir::inst::Obj::id($item)\"
                iterator = \"ir::inst::iter($fun)\"
                var_prefix = \"inst\"
                new_objs = \"$objs.inst\"
              end".to_vec())).unwrap().type_check().err(),
            Some(TypeError::MissingEntry(String::from("Instruction"), Spanned {
                beg: Position { line: 0, column: 4},
                end: Position { line: 0, column: 15},
                data: ir::SetDefKey::ItemType.to_string(),
                filename: Default::default()
            }))
        );
    }

    /// Missing the IdType's key from Set.
    #[test]
    fn id_type() {
        assert_eq!(parser::parse_ast(Lexer::new(
            b"set Instruction:
                item_type = \"ir::inst::Obj\"
                item_getter = \"ir::inst::get($fun, $id)\"
                id_getter = \"ir::inst::Obj::id($item)\"
                iterator = \"ir::inst::iter($fun)\"
                var_prefix = \"inst\"
                new_objs = \"$objs.inst\"
              end".to_vec())).unwrap().type_check().err(),
            Some(TypeError::MissingEntry(String::from("Instruction"), Spanned {
                beg: Position { line: 0, column: 4},
                end: Position { line: 0, column: 15},
                data: ir::SetDefKey::IdType.to_string(),
                filename: Default::default()
            }))
        );
    }

    /// Missing the ItemGetter's key from Set.
    #[test]
    fn item_getter() {
        assert_eq!(parser::parse_ast(Lexer::new(
            b"set Instruction:
                item_type = \"ir::inst::Obj\"
                id_type = \"ir::inst::Id\"
                id_getter = \"ir::inst::Obj::id($item)\"
                iterator = \"ir::inst::iter($fun)\"
                var_prefix = \"inst\"
                new_objs = \"$objs.inst\"
              end".to_vec())).unwrap().type_check().err(),
            Some(TypeError::MissingEntry(String::from("Instruction"), Spanned {
                beg: Position { line: 0, column: 4},
                end: Position { line: 0, column: 15},
                data: ir::SetDefKey::ItemGetter.to_string(),
                filename: Default::default()
            }))
        );
    }

    /// Missing the IdGetter's key from Set.
    #[test]
    fn id_getter() {
        assert_eq!(parser::parse_ast(Lexer::new(
            b"set Instruction:
                item_type = \"ir::inst::Obj\"
                id_type = \"ir::inst::Id\"
                item_getter = \"ir::inst::get($fun, $id)\"
                iterator = \"ir::inst::iter($fun)\"
                var_prefix = \"inst\"
                new_objs = \"$objs.inst\"
              end".to_vec())).unwrap().type_check().err(),
            Some(TypeError::MissingEntry(String::from("Instruction"), Spanned {
                beg: Position { line: 0, column: 4},
                end: Position { line: 0, column: 15},
                data: ir::SetDefKey::IdGetter.to_string(),
                filename: Default::default()
            }))
        );
    }

    /// Missing the Iter's key from Set.
    #[test]
    fn iter() {
        assert_eq!(parser::parse_ast(Lexer::new(
            b"set Instruction:
                item_type = \"ir::inst::Obj\"
                id_type = \"ir::inst::Id\"
                item_getter = \"ir::inst::get($fun, $id)\"
                id_getter = \"ir::inst::Obj::id($item)\"
                var_prefix = \"inst\"
                new_objs = \"$objs.inst\"
              end".to_vec())).unwrap().type_check().err(),
            Some(TypeError::MissingEntry(String::from("Instruction"), Spanned {
                beg: Position { line: 0, column: 4},
                end: Position { line: 0, column: 15},
                data: ir::SetDefKey::Iter.to_string(),
                filename: Default::default()
            }))
        );
    }

    /// Missing the SetDefKey's key from Set.
    #[test]
    fn from_superset() {
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
              
              set Instruction subsetof BasicBlock:
                item_type = \"ir::inst::Obj\"
                id_type = \"ir::inst::Id\"
                item_getter = \"ir::inst::get($fun, $id)\"
                id_getter = \"ir::inst::Obj::id($item)\"
                iterator = \"ir::inst::iter($fun)\"
                var_prefix = \"inst\"
                new_objs = \"$objs.inst\"
             end".to_vec())).unwrap().type_check().err(),
            Some(TypeError::MissingEntry(String::from("Instruction"), Spanned {
                beg: Position { line: 10, column: 18},
                end: Position { line: 10, column: 29},
                data: ir::SetDefKey::FromSuperset.to_string(),
                filename: Default::default()
            }))
        );
    }
}
