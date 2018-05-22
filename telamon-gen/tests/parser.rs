extern crate telamon_gen;
extern crate lalrpop_util;

use telamon_gen::lexer::{Lexer, Token, LexicalError, Position};
use telamon_gen::parser;

use lalrpop_util::ParseError;

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
}
