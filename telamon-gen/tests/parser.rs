extern crate lalrpop_util;
extern crate telamon_gen;

use telamon_gen::lexer::{
    ErrorKind, Lexer, LexerPosition, LexicalError, Position, Spanned, Token,
};
use telamon_gen::{error, parser};

use lalrpop_util::ParseError;

use std::path::Path;

#[test]
fn parser_unexpected_token() {
    assert_eq!(
        parser::parse_ast(Lexer::new(b"define enum Uper".to_vec())).err(),
        Some(ParseError::UnrecognizedToken {
            token: Some((
                Position {
                    position: LexerPosition {
                        line: 0,
                        column: 12
                    },
                    filename: None
                },
                Token::SetIdent(String::from("Uper")),
                Position {
                    position: LexerPosition {
                        line: 0,
                        column: 16
                    },
                    filename: None
                }
            )),
            expected: vec![String::from("choice_ident")],
        })
    );

    assert_eq!(
        format!(
            "{}",
            parser::parse_ast(Lexer::new(b"define enum Uper".to_vec()))
                .map_err(|c| error::Error::from((Path::new("exh").to_path_buf(), c)))
                .err()
                .unwrap()
        ),
        "Unexpected token 'SetIdent(\"Uper\")', between line 0, \
         column 12 and line 0, column 16 -> exh"
    );
}

#[test]
fn parser_invalid_token() {
    assert_eq!(
        parser::parse_ast(Lexer::new(b"!".to_vec())).err(),
        Some(ParseError::User {
            error: LexicalError {
                cause: Spanned {
                    beg: Position::default(),
                    end: Position {
                        position: LexerPosition::new(0, 1),
                        ..Default::default()
                    },
                    data: ErrorKind::InvalidToken {
                        token: String::from("!")
                    },
                }
            },
        })
    );

    assert_eq!(
        format!(
            "{}",
            parser::parse_ast(Lexer::new(b"!".to_vec()))
                .map_err(|c| error::Error::from((Path::new("exh").to_path_buf(), c)))
                .err()
                .unwrap()
        ),
        "Invalid token \"!\", between line 0, column 0 and line 0, \
         column 1 -> exh"
    );
}

#[test]
fn parser_integer_token() {
    assert!(parser::parse_ast(Lexer::new(
        b"define integer mychoice($myarg in MySet): \"mycode\" end".to_vec()
    ))
    .is_ok());
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
    assert!(parser::parse_ast(
        Lexer::new(include.as_bytes().to_vec()).collect::<Vec<_>>()
    )
    .is_ok());
}
