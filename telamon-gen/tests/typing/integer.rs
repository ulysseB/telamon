pub use super::utils::RcStr;

pub use super::telamon_gen::lexer::{Lexer, Spanned, Position};
pub use super::telamon_gen::parser;
pub use super::telamon_gen::ast::*;

#[test]
fn integer_redefinition() {
    assert_eq!(parser::parse_ast(Lexer::from(
        b"set MySet:
            item_type = \"ir::inst::Obj\"
            id_type = \"ir::inst::Id\"
            item_getter = \"ir::inst::get($fun, $id)\"
            id_getter = \"ir::inst::Obj::id($item)\"
            iterator = \"ir::inst::iter($fun)\"
            var_prefix = \"inst\"
            new_objs = \"$objs.inst\"
          end
          define integer foo($myarg in MySet) in \"mycode\"
          end
          define integer foo($myarg in MySet) in \"mycode\"
          end".to_vec())).unwrap().type_check().err(),
        Some(Spanned {
            beg: Position { line: 11, column: 10},
            end: Position { line: 12, column: 13},
            data: TypeError::Redefinition(
                String::from("foo"),
                Hint::Integer
            ),
        })
    );
    assert!(parser::parse_ast(Lexer::from(
        b"set MySet:
            item_type = \"ir::inst::Obj\"
            id_type = \"ir::inst::Id\"
            item_getter = \"ir::inst::get($fun, $id)\"
            id_getter = \"ir::inst::Obj::id($item)\"
            iterator = \"ir::inst::iter($fun)\"
            var_prefix = \"inst\"
            new_objs = \"$objs.inst\"
          end
          define integer foo($myarg in MySet) in \"mycode\"
          end".to_vec())).unwrap().type_check().is_ok()
    );

}

#[test]
fn integer_field_redefinition() {
    assert_eq!(parser::parse_ast(Lexer::from(
        b"set MySet:
            item_type = \"ir::inst::Obj\"
            id_type = \"ir::inst::Id\"
            item_getter = \"ir::inst::get($fun, $id)\"
            id_getter = \"ir::inst::Obj::id($item)\"
            iterator = \"ir::inst::iter($fun)\"
            var_prefix = \"inst\"
            new_objs = \"$objs.inst\"
          end
          define integer foo($arg in MySet, $arg in MySet) in \"mycode\"
          end".to_vec())).unwrap().type_check().err(),
        Some(Spanned {
            beg: Position { line: 9, column: 10},
            end: Position { line: 10, column: 13},
            data: TypeError::Redefinition(
                String::from("arg"),
                Hint::IntegerAttribute,
            )
        })
    );
}

#[test]
fn integer_undefined_parametric() {
    assert_eq!(parser::parse_ast(Lexer::from(
        b"define integer foo($arg in MySet) in \"mycode\"
          end".to_vec())).unwrap().type_check().err(),
        Some(Spanned {
            beg: Position { line: 0, column: 0},
            end: Position { line: 1, column: 13},
            data: TypeError::Undefined(String::from("MySet")),
        })
    );
}
