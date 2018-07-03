/// This lexer is a application of 
/// [Writing a custom lexer](https://github.com/lalrpop/lalrpop/blob/master/doc/src/lexer_tutorial/index.md)'s
/// documentation. This includes a Spanned definition and a Iterator.
mod ffi;
mod token;

use std::{io,ptr};
use std::error::Error;
use std::fmt;

pub use self::token::Token;    

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

pub use self::ffi::{Position, Spanned, Span};

#[derive(Debug, Clone, PartialEq)]
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
            LexicalError::UnexpectedToken(beg, tok, end) |
            LexicalError::InvalidToken(beg, tok, end) => {
                write!(f, "{}, found '{:?}' between {}:{}",
                          self.description(),
                          tok,
                          beg,
                          end
                )
            },
        }
    }
}

/// The alias Spanned is a definition of the stream format.
/// The parser will accept an iterator where each item
/// in the stream has the following structure.
pub type SpannedLexer<Tok, Pos, Err> = Result<(Pos, Tok, Pos), Err>;

pub struct Lexer {
    scanner: YyScan,
    buffer: YyBufferState,
    /// Stores the next token.
    lookahead_token: Option<SpannedLexer<Token, Position, LexicalError>>,
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
                lookahead_token: None,
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
    type Item = SpannedLexer<Token, Position, LexicalError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.lookahead_token.is_some() {
            self.lookahead_token.take()
        } else {
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
                             .and_then(|s: &str| Some(Err(LexicalError::InvalidToken(extra.beg, Token::InvalidToken(s.to_owned()), extra.end))))
                    },
                    YyToken::ChoiceIdent => {
                        let out = yyget_text(self.scanner);
    
                        CStr::from_ptr(out)
                             .to_str().ok()
                             .and_then(|s: &str| Some(Ok((extra.beg, Token::ChoiceIdent(s.to_owned()), extra.end))))
                    },
                    YyToken::SetIdent => {
                        let out = yyget_text(self.scanner);
    
                        CStr::from_ptr(out)
                             .to_str().ok()
                             .and_then(|s: &str| Some(Ok((extra.beg, Token::SetIdent(s.to_owned()), extra.end))))
                    },
                    YyToken::ValueIdent => {
                        let out = yyget_text(self.scanner);
    
                        CStr::from_ptr(out)
                             .to_str().ok()
                             .and_then(|s: &str| Some(Ok((extra.beg, Token::ValueIdent(s.to_owned()), extra.end))))
                    },
                    YyToken::Var => {
                        let out = yyget_text(self.scanner);
    
                        CStr::from_ptr(out)
                             .to_str().ok()
                             .and_then(|s: &str| Some(Ok((extra.beg, Token::Var(s.to_owned()), extra.end))))
                    },
                    YyToken::Code => {
                        let out = yyget_text(self.scanner);
                        CStr::from_ptr(out)
                             .to_str().ok()
                             .and_then(|s: &str| {
                                  let mut s: String = s.to_owned();
                                  let mut e = extra.end;
                                  loop {
                                      let lookahead_token = self.lookahead_token();
                                      if let Some(
                                          Ok((_, Token::Code(ref code), end))
                                      ) = lookahead_token {
                                          e = end;
                                          s.push_str(code);
                                      } else {
                                          self.lookahead_token = lookahead_token;
                                          return Some(
                                              Ok((extra.beg, Token::Code(s), e))
                                          )
                                      }
                                  }
                             })
                    },
                    YyToken::Doc => {
                        let out = yyget_text(self.scanner);
    
                        CStr::from_ptr(out)
                             .to_str().ok()
                             .and_then(|s: &str| Some(Ok((extra.beg, Token::Doc(s.to_owned()), extra.end))))
                    },
                    YyToken::Alias => Some(Ok((extra.beg, Token::Alias, extra.end))),
                    YyToken::Counter => Some(Ok((extra.beg, Token::Counter, extra.end))),
                    YyToken::Define => Some(Ok((extra.beg, Token::Define, extra.end))),
                    YyToken::Enum => Some(Ok((extra.beg, Token::Enum, extra.end))),
                    YyToken::Forall => Some(Ok((extra.beg, Token::Forall, extra.end))),
                    YyToken::In => Some(Ok((extra.beg, Token::In, extra.end))),
                    YyToken::Is => Some(Ok((extra.beg, Token::Is, extra.end))),
                    YyToken::Not => Some(Ok((extra.beg, Token::Not, extra.end))),
                    YyToken::Require => Some(Ok((extra.beg, Token::Require, extra.end))),
                    YyToken::Requires => Some(Ok((extra.beg, Token::Requires, extra.end))),
                    YyToken::CounterKind => Some(Ok((extra.beg, Token::CounterKind(extra.data.counter_kind), extra.end))),
                    YyToken::Value => Some(Ok((extra.beg, Token::Value, extra.end))),
                    YyToken::When => Some(Ok((extra.beg, Token::When, extra.end))),
                    YyToken::Trigger => Some(Ok((extra.beg, Token::Trigger, extra.end))),
                    YyToken::CounterVisibility => Some(Ok((extra.beg, Token::CounterVisibility(extra.data.counter_visibility), extra.end))),
                    YyToken::Base => Some(Ok((extra.beg, Token::Base, extra.end))),
                    YyToken::SetDefkey => Some(Ok((extra.beg, Token::SetDefKey(extra.data.set_def_key), extra.end))),
                    YyToken::Set => Some(Ok((extra.beg, Token::Set, extra.end))),
                    YyToken::SubsetOf => Some(Ok((extra.beg, Token::SubsetOf, extra.end))),
                    YyToken::Disjoint => Some(Ok((extra.beg, Token::Disjoint, extra.end))),
                    YyToken::Quotient => Some(Ok((extra.beg, Token::Quotient, extra.end))),
                    YyToken::Of => Some(Ok((extra.beg, Token::Of, extra.end))),
                    YyToken::Bool => Some(Ok((extra.beg, Token::Bool(extra.data.boolean), extra.end))),
                    YyToken::Colon => Some(Ok((extra.beg, Token::Colon, extra.end))),
                    YyToken::Comma => Some(Ok((extra.beg, Token::Comma, extra.end))),
                    YyToken::LParen => Some(Ok((extra.beg, Token::LParen, extra.end))),
                    YyToken::RParen => Some(Ok((extra.beg, Token::RParen, extra.end))),
                    YyToken::BitOr => Some(Ok((extra.beg, Token::BitOr, extra.end))),
                    YyToken::Or => Some(Ok((extra.beg, Token::Or, extra.end))),
                    YyToken::And => Some(Ok((extra.beg, Token::And, extra.end))),
                    YyToken::CmpOp => Some(Ok((extra.beg, Token::CmpOp(extra.data.cmp_op), extra.end))),
                    YyToken::Equal => Some(Ok((extra.beg, Token::Equal, extra.end))),
                    YyToken::End => Some(Ok((extra.beg, Token::End, extra.end))),
                    YyToken::Symmetric => Some(Ok((extra.beg, Token::Symmetric, extra.end))),
                    YyToken::AntiSymmetric => Some(Ok((extra.beg, Token::AntiSymmetric, extra.end))),
                    YyToken::Arrow => Some(Ok((extra.beg, Token::Arrow, extra.end))),
                    YyToken::Divide => Some(Ok((extra.beg, Token::Divide, extra.end))),
                    YyToken::Integer => Some(Ok((extra.beg, Token::Integer, extra.end))),
                    // Return None to signal EOF.for a reached end of the string.
                    YyToken::EOF => None,
                }
            }
        }
    }
}
