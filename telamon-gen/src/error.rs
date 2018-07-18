use super::lexer;
use super::lalrpop_util::*;

use std::{path, fmt};
use std::error::Error;

#[derive(Debug)]
pub enum Cause {
    /// Lalrpop
    Parse(ParseError<lexer::Position,
                     lexer::Token,
                     lexer::LexicalError>),
    /// Will be remplaced by field for Ast [...]
    Other,
}

#[derive(Debug)]
pub struct ProcessError<'a> {
    /// Display of filename.
    pub path: path::Display<'a>,
    /// Position of lexeme.
    pub span: Option<lexer::Span>,
    cause: Cause,
}

impl <'a>From<(path::Display<'a>,
               ParseError<lexer::LexerPosition,
                          lexer::Token,
                          lexer::LexicalError>
             )> for ProcessError<'a> {
    fn from((path, parse): (path::Display<'a>,
                                ParseError<lexer::LexerPosition,
                                           lexer::Token,
                                           lexer::LexicalError>
    )) -> Self {
        unimplemented!()
    }
}

impl <'a>From<(path::Display<'a>,
               ParseError<lexer::Position,
                                 lexer::Token,
                                 lexer::LexicalError>
             )> for ProcessError<'a> {
    fn from((path, parse): (path::Display<'a>,
                                ParseError<lexer::Position,
                                           lexer::Token,
                                           lexer::LexicalError>
    )) -> Self {
        match parse {
            ParseError::InvalidToken { location }
                => ProcessError {
                    path: path,
                    span: Some(lexer::Span { beg: location, ..Default::default() }),
                    cause: Cause::Parse(parse),
                },
            ParseError::UnrecognizedToken { token: None, .. }
                => ProcessError {
                    path: path,
                    span: None,
                    cause: Cause::Parse(parse),
                }, 
            ParseError::UnrecognizedToken { token: Some((l, .., e)), .. } |
            ParseError::ExtraToken { token: (l, .., e) } |
            ParseError::User {
                error: lexer::LexicalError::UnexpectedToken(
                    lexer::LexerPosition { position:  l, .. }, ..,
                    lexer::LexerPosition { position: e, .. }
                )
            } |
            ParseError::User {
                error: lexer::LexicalError::InvalidInclude(
                    lexer::LexerPosition { position:  l, .. }, ..,
                    lexer::LexerPosition { position: e, .. }
                )
            } |
            ParseError::User {
                error: lexer::LexicalError::InvalidToken(
                    lexer::LexerPosition { position:  l, .. }, ..,
                    lexer::LexerPosition { position: e, .. }
                )
            } => ProcessError {
                 path: path,
                 span: Some(lexer::Span { beg: l, end: Some(e) }),
                 cause: Cause::Parse(parse),
            },
        }
    }
}

impl <'a> fmt::Display for ProcessError<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
           ProcessError { path, span, cause: Cause::Parse(
                ParseError::UnrecognizedToken {
                    token: Some((_, ref token, _)), ..
                }), ..} |
           ProcessError { path, span, cause: Cause::Parse(ParseError::ExtraToken {
                    token: (_, ref token, _)
                }), ..} |
           ProcessError { path, span, cause: Cause::Parse(
                ParseError::User {
                    error: lexer::LexicalError::UnexpectedToken(_, ref token, _)
                }), ..} |
           ProcessError { path, span, cause: Cause::Parse(
                ParseError::User {
                    error: lexer::LexicalError::InvalidToken(_, ref token, _)
                }), ..} => {
                if let Some(span) = span {
                    write!(f, "{}, {} -> {}", token, span, path)
                } else {
                    write!(f, "{} -> {}", token, path)
                }
           },
           _ => Ok(()),
        }
    }
}

impl <'a>Error for ProcessError<'a> {
    fn description(&self) -> &str {
        "Process error"
    }

    fn cause(&self) -> Option<&Error> {
        if let Cause::Parse(ref parse) = self.cause {
            parse.cause()
        } else {
            None
        }
    }
}
