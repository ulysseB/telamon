pub use super::utils::RcStr;

pub use super::telamon_gen::lexer::{Lexer, Spanned, Position};
pub use super::telamon_gen::parser;
pub use super::telamon_gen::ast::*;

#[test]
fn integer_redefinition() {
    assert_eq!(parser::parse_ast(Lexer::from(
        b"define integer foo($myarg in MySet) in \"mycode\"
          end
          define integer foo($myarg in MySet) in \"mycode\"
          end".to_vec())).unwrap().type_check().err(),
        Some(Spanned {
            beg: Position { line: 2, column: 10},
            end: Position { line: 3, column: 13},
            data: TypeError::Redefinition(
                String::from("foo"),
                Hint::Integer
            ),
        })
    );
    assert!(parser::parse_ast(Lexer::from(
        b"define integer foo($myarg in MySet) in \"mycode\"
          end".to_vec())).unwrap().type_check().is_ok()
    );

}
