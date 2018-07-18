pub use super::utils::RcStr;

pub use super::telamon_gen::lexer::{Lexer, Spanned, Position};
pub use super::telamon_gen::parser;
pub use super::telamon_gen::ast::*;

#[cfg(test)]
mod undefined {
    pub use super::*;

    /// Missing the set MySet from a Integer.
    #[test]
    fn parameter() {
        assert_eq!(parser::parse_ast(Lexer::new(
            b"define integer foo($arg in MySet): \"mycode\"
              end".to_vec())).unwrap().type_check().err(),
            Some(TypeError::Undefined(Spanned {
                beg: Position { line: 0, column: 15},
                end: Position { line: 0, column: 18},
                data: String::from("MySet"),
                filename: Default::default()
            }))
        );
    }
}

#[cfg(test)]
mod redefinition {
    pub use super::*;

    /// Redefinition of the foo Integer.
    #[test]
    fn integer() {
        assert_eq!(parser::parse_ast(Lexer::new(
            b"define integer foo(): \"mycode\"
              end
              define integer foo(): \"mycode\"
              end".to_vec())).unwrap().type_check().err(),
            Some(TypeError::Redefinition(Spanned {
                beg: Position { line: 0, column: 15},
                end: Position { line: 0, column: 18},
                data: Hint::Integer,
                filename: Default::default()
            }, Spanned {
                beg: Position { line: 2, column: 29},
                end: Position { line: 2, column: 32},
                data:  String::from("foo"),
                filename: Default::default()
            }))
        );
    }
}
