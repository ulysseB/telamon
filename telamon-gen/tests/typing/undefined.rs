pub use super::utils::RcStr;

pub use super::telamon_gen::lexer::{Lexer, Spanned, Position};
pub use super::telamon_gen::parser;
pub use super::telamon_gen::ast::*;
pub use super::telamon_gen::ir;

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
