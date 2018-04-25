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
    yylex_init,
    yy_scan_buffer,
    yy_scan_bytes,
    yy_delete_buffer,
    yylex_destroy,
    yylex,
    YyToken,
    yylval,
    yyget_text,
};

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
    fn from(buffer:Vec<u8>) -> Self {
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
    type Item = Token;

    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            match yylex(self.scanner) {
                YyToken::Blank => self.next(),
                YyToken::InvalidToken => {
                    let out = ffi::yyget_text(self.scanner);

                    CStr::from_ptr(out)
                         .to_str().ok()
                         .and_then(|s: &str| Some(Token::InvalidToken(s.to_owned())))
                },
                YyToken::Blank => None,
                // C_CommentBeg
                YyToken::ChoiceIdent => {
                    let out = ffi::yyget_text(self.scanner);

                    CStr::from_ptr(out)
                         .to_str().ok()
                         .and_then(|s: &str| Some(Token::ChoiceIdent(s.to_owned())))
                },
                YyToken::SetIdent => {
                    let out = ffi::yyget_text(self.scanner);

                    CStr::from_ptr(out)
                         .to_str().ok()
                         .and_then(|s: &str| Some(Token::SetIdent(s.to_owned())))
                },
                YyToken::ValueIdent => {
                    let out = ffi::yyget_text(self.scanner);

                    CStr::from_ptr(out)
                         .to_str().ok()
                         .and_then(|s: &str| Some(Token::ValueIdent(s.to_owned())))
                },
                YyToken::Var => {
                    let out = ffi::yyget_text(self.scanner);

                    CStr::from_ptr(out.offset(1))
                         .to_str().ok()
                         .and_then(|s: &str| Some(Token::Var(s.to_owned())))
                },
                YyToken::Code => {
                    let out = ffi::yyget_text(self.scanner);
                    let len = libc::strlen(out)-1;

                    *out.offset(len as _) = b'\0' as _;
                    CStr::from_ptr(out.offset(1))
                         .to_str().ok()
                         .and_then(|s: &str| Some(Token::Code(s.to_owned())))
                },
                YyToken::Alias => Some(Token::Alias),
                YyToken::Counter => Some(Token::Counter),
                YyToken::Define => Some(Token::Define),
                YyToken::Enum => Some(Token::Enum),
                YyToken::Forall => Some(Token::Forall),
                YyToken::In => Some(Token::In),
                YyToken::Is => Some(Token::Is),
                YyToken::Not => Some(Token::Not),
                YyToken::Require => Some(Token::Require),
                YyToken::Requires => Some(Token::Requires),
                YyToken::CounterKind => Some(Token::CounterKind(yylval.counter_kind)),
                YyToken::Value => Some(Token::Value),
                YyToken::When => Some(Token::When),
                YyToken::Trigger => Some(Token::Trigger),
                YyToken::CounterVisibility => Some(Token::CounterVisibility(yylval.counter_visibility)),
                YyToken::Base => Some(Token::Base),
                YyToken::SetDefkey => Some(Token::SetDefKey(yylval.set_def_key)),
                YyToken::Set => Some(Token::Set),
                YyToken::SubsetOf => Some(Token::SubsetOf),
                YyToken::Disjoint => Some(Token::Disjoint),
                YyToken::Quotient => Some(Token::Quotient),
                YyToken::Of => Some(Token::Of),
                YyToken::Bool => Some(Token::Bool(yylval.boolean)),
                YyToken::Colon => Some(Token::Colon),
                YyToken::Comma => Some(Token::Comma),
                YyToken::LParen => Some(Token::LParen),
                YyToken::RParen => Some(Token::RParen),
                YyToken::BitOr => Some(Token::BitOr),
                YyToken::Or => Some(Token::Or),
                YyToken::And => Some(Token::And),
                YyToken::CmpOp => Some(Token::CmpOp(yylval.cmp_op)),
                YyToken::Equal => Some(Token::Equal),
//                YyToken::Doc => Some(Token::Doc),
                YyToken::End => Some(Token::End),
                YyToken::Symmetric => Some(Token::Symmetric),
                YyToken::AntiSymmetric => Some(Token::AntiSymmetric),
                YyToken::Arrow => Some(Token::Arrow),
                YyToken::Divide => Some(Token::Divide),
                YyToken::EOF => None,
                YyToken::Doc => {
                    None
                },
            }
        }
    }
}
