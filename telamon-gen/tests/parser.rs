extern crate lalrpop_util;
extern crate telamon_gen;

use telamon_gen::error;
use telamon_gen::lexer::{Lexer, LexerPosition, LexicalError, Position, Token};
use telamon_gen::parser;

use lalrpop_util::ParseError;

use std::path::Path;

#[test]
fn parser_invalid_token() {
    assert_eq!(
        parser::parse_ast(Lexer::new(b"!".to_vec())).err(),
        Some(ParseError::User {
            error: LexicalError::InvalidToken(
                Position::default(),
                Token::InvalidToken(String::from("!")),
                Position {
                    position: LexerPosition::new(0, 1),
                    ..Default::default()
                }
            ),
        })
    );

    assert_eq!(
        format!(
            "{}",
            parser::parse_ast(Lexer::new(b"!".to_vec()))
                .map_err(|c| error::ProcessError::from((Path::new("exh").display(), c)))
                .err()
                .unwrap()
        ),
        "InvalidToken(\"!\"), between line 0, column 0 and line 0, column 1 -> exh"
    );
}

#[test]
fn parser_integer_token() {
    assert!(
        parser::parse_ast(Lexer::new(
            b"define integer mychoice($myarg in MySet): \"mycode\" end".to_vec()
        )).is_ok()
    );
}

#[test]
fn parser_include_set() {
    // Header include.
    // ```
    // include ab
    //      set a
    //          include c
    //               set c
    //      set b
    // ```
    let filename: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/extra/set/acb.exh");
    let include = format!("include \"{}\"", filename);

    // test the parse validity.
    assert!(parser::parse_ast(Lexer::new(include.as_bytes().to_vec()).collect::<Vec<_>>()).is_ok());
}
