extern crate lalrpop_util;
extern crate telamon_gen;

<<<<<<< HEAD
use telamon_gen::lexer::{Lexer, LexerPosition, ErrorKind, Spanned,
    LexicalError, Position};
use telamon_gen::parser;
=======
>>>>>>> 94b6ae433ce9060913e4f47af35f333acf93d84e
use telamon_gen::error;
use telamon_gen::lexer::{Lexer, LexerPosition, LexicalError, Position, Token};
use telamon_gen::parser;

use lalrpop_util::ParseError;

use std::path::Path;

#[test]
fn invalid_token() {
<<<<<<< HEAD
    assert_eq!(parser::parse_ast(Lexer::new(b"!".to_vec())).err(), Some(
                  ParseError::User {
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
                  }
              ));

    assert_eq!(format!("{}",
                   parser::parse_ast(Lexer::new(b"!".to_vec()))
                          .map_err(|c|
                               error::ProcessError::from(
                                   (Path::new("exh").display(), c)))
                          .err().unwrap()),
               "invalid token !, between line 0, column 0 and line 0, column 1 -> exh");
=======
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
>>>>>>> 94b6ae433ce9060913e4f47af35f333acf93d84e
}

#[test]
fn integer_token() {
    assert!(
        parser::parse_ast(Lexer::new(
            b"define integer mychoice($myarg in MySet): \"mycode\" end".to_vec()
        )).is_ok()
    );
}
