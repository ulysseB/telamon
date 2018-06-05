extern crate telamon_gen;
extern crate lalrpop_util;

use telamon_gen::lexer::{Lexer, Spanned, Position};
use telamon_gen::ast::EnumStatement;
use telamon_gen::parser;

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
            data: telamon_gen::ast::TypeError::EnumNameMulti
        })
    );
}

#[test]
fn enum_multi_name_field() {
    assert_eq!(parser::parse_ast(Lexer::from(
        b"define enum foo():
          value A:
          value B:
          value A:
          end".to_vec())).unwrap().type_check().err(),
        Some(Spanned {
            leg: Position { line: 0, column: 0},
            end: Position { line: 0, column: 18},
            data: telamon_gen::ast::TypeError::EnumMultipleNameField(
                EnumStatement::Value(String::from("A"), None, vec![])
            ),
        })
    );
}
