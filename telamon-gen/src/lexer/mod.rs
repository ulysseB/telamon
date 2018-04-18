/// Tokens from the textual representation of constraints.

mod ffi;
mod token;

use ir;
use std::{io,ptr};

pub use self::token::Token;

use self::ffi::{
    YyScan,
    YyBufferState,
    yylex_init,
    yy_scan_string,
    yy_delete_buffer,
    yylex_destroy,
    yylex,
    YyToken,
    yylval,

    CmpOp
};

pub struct Lexer {
    scanner: YyScan,
    buffer: YyBufferState,
}

impl Lexer {
    pub fn new(input: &mut io::Read) -> Self {
        let mut buffer = Vec::new();

        input.read_to_end(&mut buffer);
        unsafe {
            let scanner: YyScan = ptr::null();

            yylex_init(&scanner); // https://westes.github.io/flex/manual/Init-and-Destroy-Functions.html#index-yylex_005finit
            Lexer {
                scanner: scanner,
                buffer: yy_scan_string(buffer.as_ptr() as *const _, scanner), // https://westes.github.io/flex/manual/Multiple-Input-Buffers.html
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
                YyToken::EOF => None,
                YyToken::Alias => Some(Token::Alias),
                YyToken::Counter => Some(Token::Counter),
                YyToken::Define => Some(Token::Define),
                YyToken::Enum => Some(Token::Enum),
                YyToken::Forall => Some(Token::Forall),
                YyToken::In => Some(Token::In),
                YyToken::Is => Some(Token::Is),
                YyToken::Not => Some(Token::Not),
//                YyToken::Product => Some(Token::Product),
//                YyToken::Require => Some(Token::RequireKind),
                YyToken::Requires => Some(Token::Requires),
//              YyToken::Sum => Some(Token::RequireKind),
                YyToken::Value => Some(Token::Value),
                YyToken::When => Some(Token::When),
                YyToken::Trigger => Some(Token::Trigger),
//                YyToken::Half => Some(Token::CounterVisibility),
//                YyToken::HIDDEN => Some(Token::CounterVisibility),
                YyToken::Base => Some(Token::Base),
                YyToken::Set => Some(Token::Set),
                YyToken::SubsetOf => Some(Token::SubsetOf),
                YyToken::Disjoint => Some(Token::Disjoint),
                YyToken::Quotient => Some(Token::Quotient),
                YyToken::Of => Some(Token::Of),

                YyToken::Colon => Some(Token::Colon),
                YyToken::Comma => Some(Token::Comma),
                YyToken::LParen => Some(Token::LParen),
                YyToken::RParen => Some(Token::RParen),
                YyToken::BitOr => Some(Token::BitOr),
                YyToken::Or => Some(Token::Or),
                YyToken::And => Some(Token::And),
                YyToken::CmpOp => {
                    match yylval.cmp_op {
                        CmpOp::Gt => Some(Token::CmpOp(ir::CmpOp::Gt)),
                        CmpOp::Lt => Some(Token::CmpOp(ir::CmpOp::Lt)),
                        CmpOp::Geq => Some(Token::CmpOp(ir::CmpOp::Geq)),
                        CmpOp::Leq => Some(Token::CmpOp(ir::CmpOp::Leq)),
                        _ => None,
                    }
                },
//                YyToken::Equals => Some(Token::CmpOp(ir::CmpOp::Eq)),
//                YyToken::NotEquals => Some(Token::CmpOp(ir::CmpOp::Neq)),
                YyToken::Equal => Some(Token::Equal),
//                YyToken::Doc => Some(Token::Doc),
                YyToken::End => Some(Token::End),
                YyToken::Symmetric => Some(Token::Symmetric),
                YyToken::AntiSymmetric => Some(Token::AntiSymmetric),
                YyToken::Arrow => Some(Token::Arrow),
                YyToken::Divide => Some(Token::Divide),
                _ => unimplemented!(),
            }
        }
    }
}
