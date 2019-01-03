use crate::lexer::{ErrorKind, LexicalError, Position, Span, Spanned, Token};
use lalrpop_util::ParseError;
use failure::Fail;
use std::fmt;
use std::path::PathBuf;

#[derive(Debug, Fail)]
pub struct Error {
    /// Display of filename.
    pub path: PathBuf,
    /// Position of lexeme.
    pub span: Option<Span>,
    cause: ParseError<Position, Token, LexicalError>,
}

impl From<(PathBuf, ParseError<Position, Token, LexicalError>)> for Error {
    fn from((path, parse): (PathBuf, ParseError<Position, Token, LexicalError>)) -> Self {
        match parse {
            ParseError::InvalidToken {
                location: Position { position: beg, .. },
            } => Error {
                path,
                span: Some(Span {
                    beg,
                    ..Default::default()
                }),
                cause: parse,
            },
            ParseError::UnrecognizedToken { token: None, .. } => Error {
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
            } => Error {
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

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error {
                path,
                span,
                cause:
                    ParseError::UnrecognizedToken {
                        token: Some((_, ref token, _)),
                        ..
                    },
                ..
            }
            | Error {
                path,
                span,
                cause:
                    ParseError::ExtraToken {
                        token: (_, ref token, _),
                    },
                ..
            } => {
                if let Some(span) = span {
                    write!(
                        f,
                        "Unexpected token '{:?}', {} -> {}",
                        token,
                        span,
                        path.display()
                    )
                } else {
                    write!(f, "Unexpected token '{:?}' -> {}", token, path.display())
                }
            }
            Error {
                path,
                span,
                cause: ParseError::User { error },
                ..
            } => {
                if let Some(span) = span {
                    write!(f, "{}, {} -> {}", error, span, path.display())
                } else {
                    write!(f, "{} -> {}", error, path.display())
                }
            }
            _ => Ok(()),
        }
    }
}
