extern crate telamon_gen;
extern crate lalrpop_util;

use telamon_gen::lexer::{Lexer, LexerPosition, Token, LexicalError, Position};
use telamon_gen::parser;
use telamon_gen::error;

use lalrpop_util::ParseError;

use std::path::Path;

#[test]
fn invalid_token() {
    assert_eq!(parser::parse_ast(Lexer::new(b"!".to_vec())).err(), Some(
                  ParseError::User {
                      error: LexicalError::InvalidToken(
                          Position::default(),
                          Token::InvalidToken(String::from("!")),
                          Position {
                              position: LexerPosition::new(0, 1),
                              ..Default::default()
                          }
                      ),
                  }
              ));

    assert_eq!(format!("{}",
                   parser::parse_ast(Lexer::new(b"!".to_vec()))
                          .map_err(|c|
                               error::ProcessError::from(
                                   (Path::new("exh").display(), c)))
                          .err().unwrap()),
               "InvalidToken(\"!\"), between line 0, column 0 and line 0, column 1 -> exh");
}

#[test]
fn integer_token() {
    assert!(parser::parse_ast(Lexer::new(
        b"define integer mychoice($myarg in MySet): \"mycode\" end".to_vec())).is_ok()
    );
}
