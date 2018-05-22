/// This lexer is a application of 
/// [Writing a custom lexer](https://github.com/lalrpop/lalrpop/blob/master/doc/src/lexer_tutorial/index.md)'s
/// documentation. This includes a Spanned definition and a Iterator.
mod ffi;
mod token;

use std::{io,ptr};
use std::error::Error;
use std::fmt;

pub use self::token::Token;    

use libc;

use std::ffi::CStr;

use self::ffi::{
    YyScan,
    YyBufferState,
    YyExtraType,
    yylex_init,
    yy_scan_bytes,
    yyset_lineno,
    yy_delete_buffer,
    yylex_destroy,
    yylex,
    YyToken,
    yyget_text,
    yyget_extra,
};

pub use self::ffi::Position;

#[derive(Debug, PartialEq)]
pub enum LexicalError {
    InvalidToken(Position, Token, Position),
    UnexpectedToken(Position, Token, Position),
}

impl Error for LexicalError {
    fn description(&self) -> &str {
        match self {
            LexicalError::InvalidToken(..) => "invalid token",
            LexicalError::UnexpectedToken(..) => "expected expression",
        }
    }
}

impl fmt::Display for LexicalError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            LexicalError::UnexpectedToken(leg, tok, end) => {
                write!(f, "{}, found '{:?}' between {}:{}",
                          self.description(),
                          tok,
                          leg,
                          end
                )
            },
            LexicalError::InvalidToken(leg, tok, end) => {
                write!(f, "{}, found '{:?}' between {}:{}",
                          self.description(),
                          tok,
                          leg,
                          end
                )
            },
        }
    }
}

/// The alias Spanned is a definition of the stream format.
/// The parser will accept an iterator where each item
/// in the stream has the following structure.
pub type Spanned<Tok, Pos, Err> = Result<(Pos, Tok, Pos), Err>;

pub struct Lexer {
    scanner: YyScan,
    buffer: YyBufferState,
}

impl Lexer {
    pub fn new(input: &mut io::Read) -> Self {
        let mut buffer = Vec::new();

        input.read_to_end(&mut buffer).unwrap();
        Lexer::from(buffer)
    }
}

impl From<Vec<u8>> for Lexer {
    fn from(buffer: Vec<u8>) -> Self {
        unsafe {
            let scanner: YyScan = ptr::null();

            // The function [yylex_init](https://westes.github.io/flex/manual/Init-and-Destroy-Functions.html#index-yylex_005finit)
            // innitializes the scanner.
            yylex_init(&scanner);

            // scans len bytes starting at location bytes. 
            let buffer: YyBufferState = yy_scan_bytes(buffer.as_ptr() as *const _, buffer.len() as _, scanner);

            // Issue [flex/60](https://github.com/westes/flex/issues/60)
            // yylineno should be set.
            // The function [yyset_lineno](https://westes.github.io/flex/manual/Reentrant-Functions.html#index-yyset_005flineno)
            // sets the current line number.
            yyset_lineno(0, scanner);
            Lexer {
                scanner: scanner,
                // The function  [yy_scan_bytes](https://westes.github.io/flex/manual/Multiple-Input-Buffers.html)
                // scans len bytes starting at location bytes. 
                buffer: buffer,
            }
        }
    }
}

impl Drop for Lexer {
    fn drop(&mut self) {
        unsafe {
            // The function [yy_delete_buffer](https://westes.github.io/flex/manual/Multiple-Input-Buffers.html)
            // clears the current contents of a buffer using.
            yy_delete_buffer(self.buffer, self.scanner);
            // The function [yylex_destroy](https://westes.github.io/flex/manual/Init-and-Destroy-Functions.html#index-yylex_005finit)
            // frees the resources used by the scanner.
            yylex_destroy(self.scanner);
        }
    }
}   

/// the Lalrpop Iterator is a exh implementation:for lexer.
impl Iterator for Lexer {
    type Item = Spanned<Token, Position, LexicalError>;

    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            // The function [yylex](https://westes.github.io/flex/manual/Generated-Scanner.html)
            // returns statement in one of the actions, the scanner may then
            // be called again and it will resume scanning where it left off. 
            let code: YyToken = yylex(self.scanner);
            // The accessor function [yyget_extra](https://westes.github.io/flex/manual/Extra-Data.html)
            // returns a extra copy.
            let extra: YyExtraType = yyget_extra(self.scanner);

            match code {
                YyToken::InvalidToken => {
                    let out = yyget_text(self.scanner);

                    CStr::from_ptr(out)
                         .to_str().ok()
                         .and_then(|s: &str| Some(Err(LexicalError::InvalidToken(extra.leg, Token::InvalidToken(s.to_owned()), extra.end))))
                },
                YyToken::ChoiceIdent => {
                    let out = yyget_text(self.scanner);

                    CStr::from_ptr(out)
                         .to_str().ok()
                         .and_then(|s: &str| Some(Ok((extra.leg, Token::ChoiceIdent(s.to_owned()), extra.end))))
                },
                YyToken::SetIdent => {
                    let out = yyget_text(self.scanner);

                    CStr::from_ptr(out)
                         .to_str().ok()
                         .and_then(|s: &str| Some(Ok((extra.leg, Token::SetIdent(s.to_owned()), extra.end))))
                },
                YyToken::ValueIdent => {
                    let out = yyget_text(self.scanner);

                    CStr::from_ptr(out)
                         .to_str().ok()
                         .and_then(|s: &str| Some(Ok((extra.leg, Token::ValueIdent(s.to_owned()), extra.end))))
                },
                YyToken::Var => {
                    let out = yyget_text(self.scanner);

                    CStr::from_ptr(out)
                         .to_str().ok()
                         .and_then(|s: &str| Some(Ok((extra.leg, Token::Var(s.to_owned()), extra.end))))
                },
                YyToken::Code => {
                    let out = yyget_text(self.scanner);
                    let len = libc::strlen(out)-1;

                    *out.offset(len as _) = b'\0' as _;
                    CStr::from_ptr(out)
                         .to_str().ok()
                         .and_then(|s: &str| Some(Ok((extra.leg, Token::Code(s.to_owned()), extra.end))))
                },
                YyToken::Doc => {
                    let out = yyget_text(self.scanner);

                    CStr::from_ptr(out)
                         .to_str().ok()
                         .and_then(|s: &str| Some(Ok((extra.leg, Token::Doc(s.to_owned()), extra.end))))
                },
                YyToken::Alias => Some(Ok((extra.leg, Token::Alias, extra.end))),
                YyToken::Counter => Some(Ok((extra.leg, Token::Counter, extra.end))),
                YyToken::Define => Some(Ok((extra.leg, Token::Define, extra.end))),
                YyToken::Enum => Some(Ok((extra.leg, Token::Enum, extra.end))),
                YyToken::Forall => Some(Ok((extra.leg, Token::Forall, extra.end))),
                YyToken::In => Some(Ok((extra.leg, Token::In, extra.end))),
                YyToken::Is => Some(Ok((extra.leg, Token::Is, extra.end))),
                YyToken::Not => Some(Ok((extra.leg, Token::Not, extra.end))),
                YyToken::Require => Some(Ok((extra.leg, Token::Require, extra.end))),
                YyToken::Requires => Some(Ok((extra.leg, Token::Requires, extra.end))),
                YyToken::CounterKind => Some(Ok((extra.leg, Token::CounterKind(extra.data.counter_kind), extra.end))),
                YyToken::Value => Some(Ok((extra.leg, Token::Value, extra.end))),
                YyToken::When => Some(Ok((extra.leg, Token::When, extra.end))),
                YyToken::Trigger => Some(Ok((extra.leg, Token::Trigger, extra.end))),
                YyToken::CounterVisibility => Some(Ok((extra.leg, Token::CounterVisibility(extra.data.counter_visibility), extra.end))),
                YyToken::Base => Some(Ok((extra.leg, Token::Base, extra.end))),
                YyToken::SetDefkey => Some(Ok((extra.leg, Token::SetDefKey(extra.data.set_def_key), extra.end))),
                YyToken::Set => Some(Ok((extra.leg, Token::Set, extra.end))),
                YyToken::SubsetOf => Some(Ok((extra.leg, Token::SubsetOf, extra.end))),
                YyToken::Disjoint => Some(Ok((extra.leg, Token::Disjoint, extra.end))),
                YyToken::Quotient => Some(Ok((extra.leg, Token::Quotient, extra.end))),
                YyToken::Of => Some(Ok((extra.leg, Token::Of, extra.end))),
                YyToken::Bool => Some(Ok((extra.leg, Token::Bool(extra.data.boolean), extra.end))),
                YyToken::Colon => Some(Ok((extra.leg, Token::Colon, extra.end))),
                YyToken::Comma => Some(Ok((extra.leg, Token::Comma, extra.end))),
                YyToken::LParen => Some(Ok((extra.leg, Token::LParen, extra.end))),
                YyToken::RParen => Some(Ok((extra.leg, Token::RParen, extra.end))),
                YyToken::BitOr => Some(Ok((extra.leg, Token::BitOr, extra.end))),
                YyToken::Or => Some(Ok((extra.leg, Token::Or, extra.end))),
                YyToken::And => Some(Ok((extra.leg, Token::And, extra.end))),
                YyToken::CmpOp => Some(Ok((extra.leg, Token::CmpOp(extra.data.cmp_op), extra.end))),
                YyToken::Equal => Some(Ok((extra.leg, Token::Equal, extra.end))),
                YyToken::End => Some(Ok((extra.leg, Token::End, extra.end))),
                YyToken::Symmetric => Some(Ok((extra.leg, Token::Symmetric, extra.end))),
                YyToken::AntiSymmetric => Some(Ok((extra.leg, Token::AntiSymmetric, extra.end))),
                YyToken::Arrow => Some(Ok((extra.leg, Token::Arrow, extra.end))),
                YyToken::Divide => Some(Ok((extra.leg, Token::Divide, extra.end))),
                // Return None to signal EOF.for a reached end of the string.
                YyToken::EOF => None,
            }
        }
    }
}
