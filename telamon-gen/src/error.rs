use super::lalrpop_util::*;
use super::lexer::{ErrorKind, LexicalError, Position, Span, Spanned, Token};

use std::{fmt, path};

#[derive(Debug)]
pub struct ProcessError<'a> {
    /// Display of filename.
    pub path: path::Display<'a>,
    /// Position of lexeme.
    pub span: Option<Span>,
    cause: ParseError<Position, Token, LexicalError>,
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
                path,
                span: Some(Span {
                    beg,
                    ..Default::default()
                }),
                cause: parse,
            },
            ParseError::UnrecognizedToken { token: None, .. } => ProcessError {
                path,
                span: None,
                cause: parse,
            },
            ParseError::UnrecognizedToken {
                token:
                    Some((Position { position: beg, .. }, .., Position { position: end, .. })),
                ..
            }
            | ParseError::ExtraToken {
                token:
                    (Position { position: beg, .. }, .., Position { position: end, .. }),
            }
            | ParseError::User {
                error:
                    LexicalError {
                        cause:
                            Spanned {
                                beg: Position { position: beg, .. },
                                end: Position { position: end, .. },
                                data: ErrorKind::InvalidToken { .. },
                            },
                    },
            }
            | ParseError::User {
                error:
                    LexicalError {
                        cause:
                            Spanned {
                                beg: Position { position: beg, .. },
                                end: Position { position: end, .. },
                                data: ErrorKind::InvalidInclude { .. },
                            },
                    },
            } => ProcessError {
                path,
                span: Some(Span {
                    beg,
                    end: Some(end),
                }),
                cause: parse,
            },
        }
    }
}

impl<'a> fmt::Display for ProcessError<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ProcessError {
                path,
                span,
                cause:
                    ParseError::UnrecognizedToken {
                        token: Some((_, ref token, _)),
                        ..
                    },
                ..
            }
            | ProcessError {
                path,
                span,
                cause:
                    ParseError::ExtraToken {
                        token: (_, ref token, _),
                    },
                ..
            } => {
                if let Some(span) = span {
                    write!(f, "Unexpected token '{:?}', {} -> {}", token, span, path)
                } else {
                    write!(f, "Unexpected token '{:?}' -> {}", token, path)
                }
            }
            ProcessError {
                path,
                span,
                cause: ParseError::User { error },
                ..
            } => {
                if let Some(span) = span {
                    write!(f, "{}, {} -> {}", error, span, path)
                } else {
                    write!(f, "{} -> {}", error, path)
                }
            }
            _ => Ok(()),
        }
    }
}
