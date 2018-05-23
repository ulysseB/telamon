use super::lexer;
use super::lalrpop_util::*;

use std::{path, fmt};
use std::error::Error;

#[derive(Debug)]
pub struct ProcessError<'a> {
    pub path: path::Display<'a>,
    pub span: Option<lexer::Span>,
    cause: ParseError<lexer::Position,
                      lexer::Token,
                      lexer::LexicalError>
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
                    span: Some(lexer::Span { leg: location, ..Default::default() }),
                    cause: parse,
                },
            ParseError::UnrecognizedToken { token: None, .. }
                => ProcessError {
                    path: path,
                    span: None,
                    cause: parse,
                }, 
           ParseError::UnrecognizedToken { token: Some((l, .., e)), .. } |
           ParseError::ExtraToken { token: (l, .., e) } |
           ParseError::User { error: lexer::LexicalError::UnexpectedToken(l, .., e) } |
           ParseError::User { error: lexer::LexicalError::InvalidToken(l, .., e) } 
                => ProcessError {
                    path: path,
                    span: Some(lexer::Span { leg: l, end: Some(e) }),
                    cause: parse,
                },
        }
    }
}

impl <'a> fmt::Display for ProcessError<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
           ProcessError { path, span, cause: ParseError::UnrecognizedToken {
               token: Some((_, ref token, _)), .. }, ..} |
           ProcessError { path, span, cause: ParseError::ExtraToken {
               token: (_, ref token, _) }, ..} |
           ProcessError { path, span, cause: ParseError::User {
               error: lexer::LexicalError::UnexpectedToken(_, ref token, _) }, ..} |
           ProcessError { path, span, cause: ParseError::User {
               error: lexer::LexicalError::InvalidToken(_, ref token, _) }, ..} => {
               if let Some(span) = span {
                   write!(f, "{} at {} -> {}", token, span, path)
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
        Some(&self.cause)
    }
}
