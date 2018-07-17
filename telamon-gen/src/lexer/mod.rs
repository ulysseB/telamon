/// This lexer is a application of 
/// [Writing a custom lexer](https://github.com/lalrpop/lalrpop/blob/master/doc/src/lexer_tutorial/index.md)'s
/// documentation. This includes a Spanned definition and a Iterator.
mod ffi;
mod token;

use std::{io, ptr, fs, path};
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
    YY_BUF_SIZE,
    yy_create_buffer,
    yypush_buffer_state,
    yypop_buffer_state,
    yy_delete_buffer,
    yylex_destroy,
    yylex,
    YyToken,
    yyget_text,
    yyget_extra,
};

pub use self::ffi::{Position, LexerPosition, Spanned, Span};

use ::libc;
use ::errno::{errno, set_errno};

pub type LexerItem = SpannedLexer<Token, LexerPosition, LexicalError>;

#[derive(Debug, Clone, PartialEq)]
pub enum LexicalError {
    InvalidToken(LexerPosition, Token, LexerPosition),
    InvalidInclude(LexerPosition, Token, LexerPosition),
    UnexpectedToken(LexerPosition, Token, LexerPosition),
}

impl Error for LexicalError {
    fn description(&self) -> &str {
        match self {
            LexicalError::InvalidToken(..) => "invalid token",
            LexicalError::InvalidInclude(..) => "invalid include header",
            LexicalError::UnexpectedToken(..) => "expected expression",
        }
    }
}

impl fmt::Display for LexicalError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            LexicalError::UnexpectedToken(beg, tok, end) |
            LexicalError::InvalidInclude(beg, tok, end) |
            LexicalError::InvalidToken(beg, tok, end) => {
                write!(f, "{}, found '{:?}' between {}:{}",
                          self.description(), tok, beg, end
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
    /// Name of entryfile.
    filename: Option<String>,
    scanner: YyScan,
    buffer: YyBufferState,
    /// Stores the next token.
    lookahead_token: Option<SpannedLexer<Token, LexerPosition, LexicalError>>,
    /// Multiple Input Buffers.
    include: Option<Box<Lexer>>,
    /// Top level module.
    root: bool,
}

impl Lexer {
    /// Returns a lexer interface for a iterable text.
    pub fn new(buffer: Vec<u8>) -> Self {
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
                filename: None,
                scanner: scanner,
                // The function  [yy_scan_bytes](https://westes.github.io/flex/manual/Multiple-Input-Buffers.html)
                // scans len bytes starting at location bytes. 
                buffer: buffer,
                lookahead_token: None,
                include: None,
                root: true,
            }
        }
    }

    /// Returns a lexer interface for a input stream.
    pub fn from_input(input: &mut io::Read) -> Self {
        let mut buffer = Vec::new();

        input.read_to_end(&mut buffer).unwrap();
        Lexer::new(buffer)
    }

    /// Returns a lexer interface for a file.
    pub fn from_file(input_path: &path::Path) -> Self {
        let mut input = fs::File::open(input_path).unwrap();
        let mut lexer = Lexer::from_input(&mut input);

        lexer.filename = Some(input_path.to_string_lossy().to_string());
        lexer
    }

    /// Returns a merged list of code terms into a code token.
    fn code(&mut self, extra: YyExtraType, buffer: &str) -> Option<LexerItem> {
        let mut buffer: String = buffer.to_owned();
        let mut acc_end: Position = extra.end;
        loop {
            match self.next() {
                Some(Ok((_, Token::Code(ref code), ref end))) => {
                    acc_end = end.position;
                    buffer.push_str(code);
                },
                next => {
                    self.lookahead_token = next;
                    return Some(Ok((LexerPosition {
                            position: extra.beg,
                            filename: self.filename.to_owned()
                        }, Token::Code(buffer), LexerPosition {
                        position: acc_end,
                        filename: self.filename.to_owned()
                    })))
                },
            }
        }
    }

    /// Links the new include buffer, Returns the first token from this include.
    fn include(&mut self, extra: YyExtraType, filename: &str) -> Option<LexerItem> {
        unsafe {
            // [Multiple Input Buffers](http://westes.github.io/flex/manual/Multiple-Input-Buffers.html#Multiple-Input-Buffers)
            // include file scanner.
            match libc::fopen(filename.as_ptr() as *const _, "r".as_ptr() as *const _) {
                yyin if yyin.is_null() => {
                    let why = errno();

                    set_errno(why);
                    Some(Err(LexicalError::InvalidInclude(
                        LexerPosition {
                            position: extra.beg,
                            filename: self.filename.to_owned()
                        }, Token::InvalidInclude(filename.to_string(), why), LexerPosition {
                            position: extra.end,
                            filename: self.filename.to_owned()
                        }
                    )))
                },
                yyin => {
                    let buffer: YyBufferState = yy_create_buffer(yyin, YY_BUF_SIZE, self.scanner);

                    yypush_buffer_state(buffer, self.scanner);
                    self.include = Some(Box::new(Lexer {
                        filename: Some(filename.to_string()),
                        scanner: self.scanner,
                        // The function  [yy_scan_bytes](https://westes.github.io/flex/manual/Multiple-Input-Buffers.html)
                        // scans len bytes starting at location bytes. 
                        buffer: self.buffer,
                        lookahead_token: None,
                        include: None,
                        root: false,
                    }));
                    self.next()
                },
            }
        }
    }
}

impl Drop for Lexer {
    fn drop(&mut self) {
        if self.root {
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
}   

/// the Lalrpop Iterator is a exh implementation:for lexer.
impl Iterator for Lexer {
    type Item = LexerItem;

    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            if let Some(ref mut module) = self.include {
                match module.next() {
                    token @ None | token @ Some(Err(..)) => {
                        yypop_buffer_state(self.scanner);
                        token
                    },
                    token => token,
                }
            } else {
                if self.lookahead_token.is_some() {
                    self.lookahead_token.take()
                } else {
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
                                 .and_then(|s: &str| Some(Err(LexicalError::InvalidToken(
                                        LexerPosition {
                                            position: extra.beg, filename: self.filename.to_owned()
                                        }, Token::InvalidToken(s.to_owned()), LexerPosition {
                                            position: extra.end, filename: self.filename.to_owned()
                                        },
                                ))))
                        },
                        YyToken::Include => {
                            let out = yyget_text(self.scanner);

                            CStr::from_ptr(out)
                                 .to_str().ok()
                                 .and_then(|s: &str| self.include(extra, s))
                        },
                        YyToken::ChoiceIdent => {
                            let out = yyget_text(self.scanner);

                            CStr::from_ptr(out)
                                 .to_str().ok()
                                 .and_then(|s: &str| Some(Ok((
                                        LexerPosition {
                                            position: extra.beg, filename: self.filename.to_owned()
                                        }, Token::ChoiceIdent(s.to_owned()), LexerPosition {
                                            position: extra.end, filename: self.filename.to_owned()
                                        }))))
                        },
                        YyToken::SetIdent => {
                            let out = yyget_text(self.scanner);

                            CStr::from_ptr(out)
                                 .to_str().ok()
                                 .and_then(|s: &str| Some(Ok((
                                        LexerPosition {
                                            position: extra.beg,
                                            filename: self.filename.to_owned()
                                        }, Token::SetIdent(s.to_owned()), LexerPosition {
                                            position: extra.end, filename: self.filename.to_owned()
                                        }))))
                        },
                        YyToken::ValueIdent => {
                            let out = yyget_text(self.scanner);

                            CStr::from_ptr(out)
                                 .to_str().ok()
                                 .and_then(|s: &str| Some(Ok((
                                        LexerPosition {
                                            position: extra.beg,
                                            filename: self.filename.to_owned()
                                        }, Token::ValueIdent(s.to_owned()), LexerPosition {
                                            position: extra.end, filename: self.filename.to_owned()
                                        }))))
                        },
                        YyToken::Var => {
                            let out = yyget_text(self.scanner);

                            CStr::from_ptr(out)
                                 .to_str().ok()
                                 .and_then(|s: &str| Some(Ok((LexerPosition {
                                            position: extra.beg,
                                            filename: self.filename.to_owned()
                                        }, Token::Var(s.to_owned()),
                                        LexerPosition {
                                            position: extra.end, filename: self.filename.to_owned() }))))
                        },
                        YyToken::Code => {
                            let out = yyget_text(self.scanner);
                            CStr::from_ptr(out)
                                 .to_str().ok()
                                 .and_then(|s: &str| self.code(extra, s))
                        },
                        YyToken::Doc => {
                            let out = yyget_text(self.scanner);

                            CStr::from_ptr(out)
                                 .to_str().ok()
                                 .and_then(|s: &str| Some(Ok((LexerPosition {
                                            position: extra.beg,
                                            filename: self.filename.to_owned()
                                        }, Token::Doc(s.to_owned()),
                                        LexerPosition {
                                            position: extra.end, filename: self.filename.to_owned() }))))
                        },
                        YyToken::Alias => Some(Ok((
                            LexerPosition {
                                position: extra.beg,
                                filename: self.filename.to_owned()
                            }, Token::Alias, LexerPosition {
                                position: extra.end,
                                filename: self.filename.to_owned()
                            }))),
                        YyToken::Counter => Some(Ok((
                            LexerPosition {
                                position: extra.beg,
                                filename: self.filename.to_owned()
                            }, Token::Counter, LexerPosition {
                                position: extra.end,
                                filename: self.filename.to_owned()
                            }))),
                        YyToken::Define => Some(Ok((
                            LexerPosition {
                                position: extra.beg,
                                filename: self.filename.to_owned()
                            }, Token::Define, LexerPosition {
                                position: extra.end,
                                filename: self.filename.to_owned()
                            }))),
                        YyToken::Enum => Some(Ok((
                            LexerPosition {
                                position: extra.beg,
                                filename: self.filename.to_owned()
                            }, Token::Enum, LexerPosition {
                                position: extra.end,
                                filename: self.filename.to_owned()
                            }))),
                        YyToken::Forall => Some(Ok((
                            LexerPosition {
                                position: extra.beg,
                                filename: self.filename.to_owned()
                            }, Token::Forall, LexerPosition {
                                position: extra.end,
                                filename: self.filename.to_owned()
                            }))),
                        YyToken::In => Some(Ok((
                            LexerPosition {
                                position: extra.beg,
                                filename: self.filename.to_owned()
                            }, Token::In, LexerPosition {
                                position: extra.end,
                                filename: self.filename.to_owned()
                            }))),
                        YyToken::Is => Some(Ok((
                            LexerPosition {
                                position: extra.beg,
                                filename: self.filename.to_owned()
                            }, Token::Is, LexerPosition {
                                position: extra.end,
                                filename: self.filename.to_owned()
                            }))),
                        YyToken::Not => Some(Ok((
                            LexerPosition {
                                position: extra.beg,
                                filename: self.filename.to_owned()
                            }, Token::Not, LexerPosition {
                                position: extra.end,
                                filename: self.filename.to_owned()
                            }))),
                        YyToken::Require => Some(Ok((
                            LexerPosition {
                                position: extra.beg,
                                filename: self.filename.to_owned()
                            }, Token::Require, LexerPosition {
                                position: extra.end,
                                filename: self.filename.to_owned()
                            }))),
                        YyToken::Requires => Some(Ok((
                            LexerPosition {
                                position: extra.beg,
                                filename: self.filename.to_owned()
                            }, Token::Requires, LexerPosition {
                                position: extra.end,
                                filename: self.filename.to_owned()
                            }))),
                        YyToken::CounterKind => Some(Ok((
                            LexerPosition {
                                position: extra.beg,
                                filename: self.filename.to_owned()
                            }, Token::CounterKind(extra.data.counter_kind), LexerPosition {
                                position: extra.end,
                                filename: self.filename.to_owned()
                            }))),
                        YyToken::Value => Some(Ok((
                            LexerPosition {
                                position: extra.beg,
                                filename: self.filename.to_owned()
                            }, Token::Value, LexerPosition {
                                position: extra.end,
                                filename: self.filename.to_owned()
                            }))),
                        YyToken::When => Some(Ok((
                            LexerPosition {
                                position: extra.beg,
                                filename: self.filename.to_owned()
                            }, Token::When, LexerPosition {
                                position: extra.end,
                                filename: self.filename.to_owned()
                            }))),
                        YyToken::Trigger => Some(Ok((
                            LexerPosition {
                                position: extra.beg,
                                filename: self.filename.to_owned()
                            }, Token::Trigger, LexerPosition {
                                position: extra.end,
                                filename: self.filename.to_owned()
                            }))),
                        YyToken::CounterVisibility => Some(Ok((
                            LexerPosition {
                                position: extra.beg,
                                filename: self.filename.to_owned()
                            }, Token::CounterVisibility(extra.data.counter_visibility),
                            LexerPosition {
                                position: extra.end,
                                filename: self.filename.to_owned()
                            }))),
                        YyToken::Base => Some(Ok((
                            LexerPosition {
                                position: extra.beg,
                                filename: self.filename.to_owned()
                            }, Token::Base, LexerPosition {
                                position: extra.end,
                                filename: self.filename.to_owned()
                            }))),
                        YyToken::SetDefkey => Some(Ok((
                            LexerPosition {
                                position: extra.beg,
                                filename: self.filename.to_owned()
                            }, Token::SetDefKey(extra.data.set_def_key), LexerPosition {
                                position: extra.end,
                                filename: self.filename.to_owned()
                            }))),
                        YyToken::Set => Some(Ok((
                            LexerPosition {
                                position: extra.beg,
                                filename: self.filename.to_owned()
                            }, Token::Set, LexerPosition {
                                position: extra.end,
                                filename: self.filename.to_owned()
                            }))),
                        YyToken::SubsetOf => Some(Ok((
                            LexerPosition {
                                position: extra.beg,
                                filename: self.filename.to_owned()
                            }, Token::SubsetOf, LexerPosition {
                                position: extra.end,
                                filename: self.filename.to_owned()
                            }))),
                        YyToken::Disjoint => Some(Ok((
                            LexerPosition {
                                position: extra.beg,
                                filename: self.filename.to_owned()
                            }, Token::Disjoint, LexerPosition {
                                position: extra.end,
                                filename: self.filename.to_owned()
                            }))),
                        YyToken::Quotient => Some(Ok((
                            LexerPosition {
                                position: extra.beg,
                                filename: self.filename.to_owned()
                            }, Token::Quotient, LexerPosition {
                                position: extra.end,
                                filename: self.filename.to_owned()
                            }))),
                        YyToken::Of => Some(Ok((
                            LexerPosition {
                                position: extra.beg,
                                filename: self.filename.to_owned()
                            }, Token::Of, LexerPosition {
                                position: extra.end,
                                filename: self.filename.to_owned()
                            }))),
                        YyToken::Bool => Some(Ok((
                            LexerPosition {
                                position: extra.beg,
                                filename: self.filename.to_owned()
                            }, Token::Bool(extra.data.boolean), LexerPosition {
                                position: extra.end,
                                filename: self.filename.to_owned()
                            }))),
                        YyToken::Colon => Some(Ok((
                            LexerPosition {
                                position: extra.beg,
                                filename: self.filename.to_owned()
                            }, Token::Colon, LexerPosition {
                                position: extra.end,
                                filename: self.filename.to_owned()
                            }))),
                        YyToken::Comma => Some(Ok((
                            LexerPosition {
                                position: extra.beg,
                                filename: self.filename.to_owned()
                            }, Token::Comma, LexerPosition {
                                position: extra.end,
                                filename: self.filename.to_owned()
                            }))),
                        YyToken::LParen => Some(Ok((
                            LexerPosition {
                                position: extra.beg,
                                filename: self.filename.to_owned()
                            }, Token::LParen, LexerPosition {
                                position: extra.end,
                                filename: self.filename.to_owned()
                            }))),
                        YyToken::RParen => Some(Ok((
                            LexerPosition {
                                position: extra.beg,
                                filename: self.filename.to_owned()
                            }, Token::RParen, LexerPosition {
                                position: extra.end,
                                filename: self.filename.to_owned()
                            }))),
                        YyToken::BitOr => Some(Ok((
                            LexerPosition {
                                position: extra.beg,
                                filename: self.filename.to_owned()
                            }, Token::BitOr, LexerPosition {
                                position: extra.end,
                                filename: self.filename.to_owned()
                            }))),
                        YyToken::Or => Some(Ok((
                            LexerPosition {
                                position: extra.beg,
                                filename: self.filename.to_owned()
                            }, Token::Or, LexerPosition {
                                position: extra.end,
                                filename: self.filename.to_owned()
                            }))),
                        YyToken::And => Some(Ok((
                            LexerPosition {
                                position: extra.beg,
                                filename: self.filename.to_owned()
                            }, Token::And, LexerPosition {
                                position: extra.end,
                                filename: self.filename.to_owned()
                            }))),
                        YyToken::CmpOp => Some(Ok((
                            LexerPosition {
                                position: extra.beg,
                                filename: self.filename.to_owned()
                            }, Token::CmpOp(extra.data.cmp_op), LexerPosition {
                                position: extra.end,
                                filename: self.filename.to_owned()
                            }))),
                        YyToken::Equal => Some(Ok((
                            LexerPosition {
                                position: extra.beg,
                                filename: self.filename.to_owned()
                            }, Token::Equal, LexerPosition {
                                position: extra.end,
                                filename: self.filename.to_owned()
                            }))),
                        YyToken::End => Some(Ok((
                            LexerPosition {
                                position: extra.beg,
                                filename: self.filename.to_owned()
                            }, Token::End, LexerPosition {
                                position: extra.end,
                                filename: self.filename.to_owned()
                            }))),
                        YyToken::Symmetric => Some(Ok((
                            LexerPosition {
                                position: extra.beg,
                                filename: self.filename.to_owned()
                            }, Token::Symmetric, LexerPosition {
                                position: extra.end,
                                filename: self.filename.to_owned()
                            }))),
                        YyToken::AntiSymmetric => Some(Ok((
                            LexerPosition {
                                position: extra.beg,
                                filename: self.filename.to_owned()
                            }, Token::AntiSymmetric, LexerPosition {
                                position: extra.end,
                                filename: self.filename.to_owned()
                            }))),
                        YyToken::Arrow => Some(Ok((
                            LexerPosition {
                                position: extra.beg,
                                filename: self.filename.to_owned()
                            }, Token::Arrow, LexerPosition {
                                position: extra.end,
                                filename: self.filename.to_owned()
                            }))),
                        YyToken::Divide => Some(Ok((
                            LexerPosition {
                                position: extra.beg,
                                filename: self.filename.to_owned()
                            }, Token::Divide, LexerPosition {
                                position: extra.end,
                                filename: self.filename.to_owned()
                            }))),
                        YyToken::Integer => Some(Ok((
                            LexerPosition {
                                position: extra.beg,
                                filename: self.filename.to_owned()
                            }, Token::Integer, LexerPosition {
                                position: extra.end,
                                filename: self.filename.to_owned()
                            }))),
                        // Return None to signal EOF.for a reached end of the string.
                        YyToken::EOF => None,
                    }
                }
            }
        }
    }
}
