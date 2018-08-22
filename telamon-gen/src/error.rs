use super::lalrpop_util::*;
use super::lexer::{LexicalError, ErrorKind, Spanned, Position, Span, Token};

use std::error::Error;
use std::{fmt, path};

#[derive(Debug)]
pub enum Cause {
    /// Lalrpop
    Parse(ParseError<Position, Token, LexicalError>),
    /// Will be remplaced by field for Ast [...]
    Other,
}

#[derive(Debug)]
pub struct ProcessError<'a> {
    /// Display of filename.
    pub path: path::Display<'a>,
    /// Position of lexeme.
    pub span: Option<Span>,
    cause: Cause,
}

impl<'a> From<(path::Display<'a>, ParseError<Position, Token, LexicalError>)>
    for ProcessError<'a>
{
    fn from(
        (path, parse): (path::Display<'a>, ParseError<Position, Token, LexicalError>),
    ) -> Self {
        match parse {
            ParseError::InvalidToken {
                location: Position { position: beg, .. },
            } => ProcessError {
                path: path,
                span: Some(Span {
                    beg,
                    ..Default::default()
                }),
                cause: Cause::Parse(parse),
            },
            ParseError::UnrecognizedToken { token: None, .. } => ProcessError {
                path: path,
                span: None,
                cause: Cause::Parse(parse),
            },
            ParseError::UnrecognizedToken {
                token: Some((Position { position: l, .. }, ..,
                             Position { position: e, .. })), ..
            } |
            ParseError::ExtraToken {
                token: (Position { position: l, .. }, ..,
                        Position { position: e, .. })
            } |
            ParseError::User {
                error: LexicalError {
                    cause: Spanned {
                        beg: Position { position: l, .. },
                        end: Position { position: e, .. },
                        data: ErrorKind::InvalidToken { .. },
                    }
                }
            } |
            ParseError::User {
                error: LexicalError {
                    cause: Spanned {
                        beg: Position { position: l, .. },
                        end: Position { position: e, .. },
                        data: ErrorKind::InvalidInclude { .. },
                    }
                }
            } => ProcessError {
                path: path,
                span: Some(Span {
                    beg: l,
                    end: Some(e),
                }),
                cause: Cause::Parse(parse),
            },
        }
    }
}

impl<'a> fmt::Display for ProcessError<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
           ProcessError { path, span, cause: Cause::Parse(
                ParseError::UnrecognizedToken {
                    token: Some((_, ref token, _)), ..
                }), ..} |
           ProcessError { path, span, cause: Cause::Parse(ParseError::ExtraToken {
                    token: (_, ref token, _)
                }), ..} => {
                if let Some(span) = span {
                    write!(f, "{}, {} -> {}", token, span, path)
                } else {
                    write!(f, "{} -> {}", token, path)
                }
           },
           ProcessError { path, span, cause: Cause::Parse(ParseError::User {
                error: LexicalError{
                    cause: Spanned { data, .. }
                }
           }), ..} => {
                if let Some(span) = span {
                    write!(f, "{}, {} -> {}", data, span, path)
                } else {
                    write!(f, "{} -> {}", data, path)
                }
            },
           _ => Ok(()),
        }
    }
}

impl<'a> Error for ProcessError<'a> {
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
