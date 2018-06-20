pub use super::utils::RcStr;

pub use super::telamon_gen::lexer::{Lexer, Spanned, Position};
pub use super::telamon_gen::parser;
pub use super::telamon_gen::ast::*;

#[test]
#[ignore] // TODO(test): raise an error as expected by the test
fn undefined() {
    assert_eq!(parser::parse_ast(Lexer::from(
        b"define integer foo($arg in MySet): \"mycode\"
          end".to_vec())).unwrap().type_check().err(),
        Some(TypeError::Undefined(Spanned {
            beg: Position { line: 0, column: 34},
            end: Position { line: 0, column: 36},
            data: String::from("MySet"),
        }))
    );
}
