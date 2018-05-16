/// Tokens from the textual representation of constraints.

mod ffi;
mod token;

use std::{io,ptr};

pub use self::token::Token;    

use libc;

use std::ffi::CStr;

use self::ffi::{
    YyScan,
    YyBufferState,
    YyExtraType,
    yylex_init,
    yy_scan_bytes,
    yy_delete_buffer,
    yylex_destroy,
    yylex,
    YyToken,
    yyget_text,
    yyget_extra,

};

pub use self::ffi::Position;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum LexicalError {
}

pub type Spanned<T, P, E> = Result<(P, T, P), E>;

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

            yylex_init(&scanner); // https://westes.github.io/flex/manual/Init-and-Destroy-Functions.html#index-yylex_005finit
            Lexer {
                scanner: scanner,
                buffer: yy_scan_bytes(buffer.as_ptr() as *const _, buffer.len() as _, scanner), // https://westes.github.io/flex/manual/Multiple-Input-Buffers.html
            }
        }
    }
}

impl Drop for Lexer {
    fn drop(&mut self) {
        unsafe {
            yy_delete_buffer(self.buffer, self.scanner); // https://westes.github.io/flex/manual/Multiple-Input-Buffers.html
            yylex_destroy(self.scanner); // https://westes.github.io/flex/manual/Init-and-Destroy-Functions.html#index-yylex_005finit
        }
    }
}   

impl Iterator for Lexer {
    type Item = Spanned<Token, Position, LexicalError>;

    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            let code: YyToken = yylex(self.scanner);
            let extra: YyExtraType = yyget_extra(self.scanner);

            match code {
                YyToken::InvalidToken => {
                    let out = yyget_text(self.scanner);

                    CStr::from_ptr(out)
                         .to_str().ok()
                         .and_then(|s: &str| Some(Ok((extra.leg, Token::InvalidToken(s.to_owned()), extra.end))))
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
                YyToken::EOF => None,
            }
        }
    }
}
