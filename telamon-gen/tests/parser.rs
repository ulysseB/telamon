extern crate telamon_gen;
extern crate lalrpop_util;

use telamon_gen::lexer::{Lexer, Token, LexicalError, Position};
use telamon_gen::parser;
use telamon_gen::error;

use lalrpop_util::ParseError;

use std::path::Path;

#[test]
fn invalid_token() {
    assert_eq!(parser::parse_ast(Lexer::from(b"!".to_vec())).err(), Some(
                  ParseError::User {
                      error: LexicalError::InvalidToken(
                          Position::default(),
                          Token::InvalidToken(String::from("!")),
                          Position { column: 1, ..Default::default() }
                      ),
                  }
              ));

    assert_eq!(format!("{}",
                   parser::parse_ast(Lexer::from(b"!".to_vec()))
                          .map_err(|c|
                               error::ProcessError::from(
                                   (Path::new("exh").display(), c)))
                          .err().unwrap()),
               "InvalidToken(\"!\") at [0; 0]:[0; 1] -> exh");
}
