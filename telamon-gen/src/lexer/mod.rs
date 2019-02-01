/// This lexer is a application of
/// [Writing a custom lexer](https://github.com/lalrpop/lalrpop/blob/master/doc/src/lexer_tutorial/index.md)'s
/// documentation. This includes a Spanned definition and a Iterator.
mod ffi;
mod token;

use std::ffi::{CStr, CString};
use std::fmt;
use std::path::{Path, PathBuf};
use std::{fs, io, path, ptr};

use errno::Errno;
use failure::Fail;

pub use self::ffi::Spanned;
pub use self::token::Token;

use self::ffi::{
    yy_create_buffer, yy_delete_buffer, yy_scan_bytes, yyget_extra, yyget_text, yylex,
    yylex_destroy, yylex_init, yypop_buffer_state, yypush_buffer_state, yyset_lineno,
    YyBufferState, YyExtraType, YyScan, YyToken, YY_BUF_SIZE,
};

pub use self::ffi::{LexerPosition, Position, Span};

use errno::{errno, set_errno};
use libc;

/// The alias Spanned is a definition of the stream format.
/// The parser will accept an iterator where each item
/// in the stream has the following structure.
pub type ParserSpanned<Tok, Pos, Err> = Result<(Pos, Tok, Pos), Err>;

pub type LexerItem = ParserSpanned<Token, Position, LexicalError>;

/// FD of include header.
type YyIn = *mut libc::FILE;

#[derive(Debug, Fail, Clone, PartialEq)]
pub enum ErrorKind {
    #[fail(display = "Invalid token \"{}\"", token)]
    InvalidToken { token: String },
    #[fail(display = "Invalid include header {}", name)]
    InvalidInclude { name: String, code: Errno },
}

#[derive(Debug, Fail, Clone, PartialEq)]
pub struct LexicalError {
    #[fail(display = "cause {}", cause)]
    pub cause: Spanned<ErrorKind>,
}

impl fmt::Display for LexicalError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.cause.data)
    }
}

pub struct Lexer {
    /// Name of entryfile.
    filename: Option<PathBuf>,
    scanner: YyScan,
    buffer: YyBufferState,
    /// Stores the next token.
    lookahead_token: Option<ParserSpanned<Token, Position, LexicalError>>,
    /// Multiple Input Buffers.
    include: Option<Box<Lexer>>,
    /// Sub level module yyin.
    sublevel: Option<YyIn>,
}

impl Lexer {
    /// Returns a lexer interface for a iterable text.
    pub fn new(buffer: Vec<u8>) -> Self {
        unsafe {
            let scanner: YyScan = ptr::null();
            yylex_init(&scanner);
            // scans len bytes starting at location bytes.
            let buffer: YyBufferState =
                yy_scan_bytes(buffer.as_ptr() as *const _, buffer.len() as _, scanner);
            yyset_lineno(0, scanner);
            Lexer {
                filename: None,
                scanner: scanner,
                buffer: buffer,
                lookahead_token: None,
                include: None,
                sublevel: None,
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

        lexer.filename = Some(input_path.to_path_buf());
        lexer
    }

    /// Returns a merged list of code terms into a code token.
    fn code(&mut self, extra: YyExtraType, buffer: &str) -> Option<LexerItem> {
        let mut buffer: String = buffer.to_owned();
        let mut acc_end: LexerPosition = extra.end;
        loop {
            match self.next() {
                Some(Ok((_, Token::Code(ref code), ref end))) => {
                    acc_end = end.position;
                    buffer.push_str(code);
                }
                next => {
                    self.lookahead_token = next;
                    return Some(Ok((
                        Position::new_optional(extra.beg, self.filename.to_owned()),
                        Token::Code(buffer),
                        Position::new_optional(acc_end, self.filename.to_owned()),
                    )));
                }
            }
        }
    }

    /// Links the new include buffer, Returns the first token from this include.
    fn include_open(
        &mut self,
        extra: YyExtraType,
        filename: CString,
    ) -> Option<LexerItem> {
        unsafe {
            match libc::fopen(filename.as_ptr(), "r".as_ptr() as _) {
                yyin if yyin.is_null() => {
                    let why = errno();

                    set_errno(why);
                    Some(Err(LexicalError {
                        cause: Spanned {
                            beg: Position::new_optional(
                                extra.beg,
                                self.filename.to_owned(),
                            ),
                            end: Position::new_optional(
                                extra.end,
                                self.filename.to_owned(),
                            ),
                            data: ErrorKind::InvalidInclude {
                                name: filename.into_string().unwrap(),
                                code: why,
                            },
                        },
                    }))
                }
                yyin => {
                    let buffer: YyBufferState =
                        yy_create_buffer(yyin, YY_BUF_SIZE, self.scanner);

                    yypush_buffer_state(buffer, self.scanner);
                    self.include = Some(Box::new(Lexer {
                        filename: Some(PathBuf::from(filename.to_str().unwrap())),
                        scanner: self.scanner,
                        buffer: self.buffer,
                        lookahead_token: None,
                        include: None,
                        sublevel: Some(yyin),
                    }));
                    self.next()
                }
            }
        }
    }

    /// Resources the parent folder of filename and opens the include.
    fn include(&mut self, extra: YyExtraType, filename: &str) -> Option<LexerItem> {
        let mut path: PathBuf = PathBuf::new();
        let filepath: &Path = Path::new(filename);

        if filepath.is_relative() {
            if let Some(ref parent_path) = self.filename {
                if let Some(ref parent) = parent_path.parent() {
                    path.push(parent.to_path_buf());
                }
            }
        }
        path.push(filepath);
        let filename: CString = CString::new(path.to_str().unwrap()).unwrap();
        self.include_open(extra, filename)
    }
}

/// the Lalrpop Iterator is a exh implementation:for lexer.
impl Iterator for Lexer {
    type Item = LexerItem;

    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            if self.include.is_some() {
                if let Some(ref mut module) = self.include {
                    match module.next() {
                        None => yypop_buffer_state(self.scanner),
                        token => return token,
                    };
                }
                self.include = None;
                self.next()
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

                            CStr::from_ptr(out).to_str().ok().and_then(|s: &str| {
                                Some(Err(LexicalError {
                                    cause: Spanned {
                                        beg: Position::new_optional(
                                            extra.beg,
                                            self.filename.to_owned(),
                                        ),
                                        end: Position::new_optional(
                                            extra.end,
                                            self.filename.to_owned(),
                                        ),
                                        data: ErrorKind::InvalidToken {
                                            token: s.to_owned(),
                                        },
                                    },
                                }))
                            })
                        }
                        YyToken::Include => {
                            let out = yyget_text(self.scanner);

                            CStr::from_ptr(out)
                                .to_str()
                                .ok()
                                .and_then(|s: &str| self.include(extra, s))
                        }
                        YyToken::ChoiceIdent => {
                            let out = yyget_text(self.scanner);

                            CStr::from_ptr(out).to_str().ok().and_then(|s: &str| {
                                Some(Ok((
                                    Position::new_optional(
                                        extra.beg,
                                        self.filename.to_owned(),
                                    ),
                                    Token::ChoiceIdent(s.to_owned()),
                                    Position::new_optional(
                                        extra.end,
                                        self.filename.to_owned(),
                                    ),
                                )))
                            })
                        }
                        YyToken::SetIdent => {
                            let out = yyget_text(self.scanner);

                            CStr::from_ptr(out).to_str().ok().and_then(|s: &str| {
                                Some(Ok((
                                    Position::new_optional(
                                        extra.beg,
                                        self.filename.to_owned(),
                                    ),
                                    Token::SetIdent(s.to_owned()),
                                    Position::new_optional(
                                        extra.end,
                                        self.filename.to_owned(),
                                    ),
                                )))
                            })
                        }
                        YyToken::ValueIdent => {
                            let out = yyget_text(self.scanner);

                            CStr::from_ptr(out).to_str().ok().and_then(|s: &str| {
                                Some(Ok((
                                    Position::new_optional(
                                        extra.beg,
                                        self.filename.to_owned(),
                                    ),
                                    Token::ValueIdent(s.to_owned()),
                                    Position::new_optional(
                                        extra.end,
                                        self.filename.to_owned(),
                                    ),
                                )))
                            })
                        }
                        YyToken::Var => {
                            let out = yyget_text(self.scanner);

                            CStr::from_ptr(out).to_str().ok().and_then(|s: &str| {
                                Some(Ok((
                                    Position::new_optional(
                                        extra.beg,
                                        self.filename.to_owned(),
                                    ),
                                    Token::Var(s.to_owned()),
                                    Position::new_optional(
                                        extra.end,
                                        self.filename.to_owned(),
                                    ),
                                )))
                            })
                        }
                        YyToken::Code => {
                            let out = yyget_text(self.scanner);
                            CStr::from_ptr(out)
                                .to_str()
                                .ok()
                                .and_then(|s: &str| self.code(extra, s))
                        }
                        YyToken::Doc => {
                            let out = yyget_text(self.scanner);

                            CStr::from_ptr(out).to_str().ok().and_then(|s: &str| {
                                Some(Ok((
                                    Position::new_optional(
                                        extra.beg,
                                        self.filename.to_owned(),
                                    ),
                                    Token::Doc(s.to_owned()),
                                    Position::new_optional(
                                        extra.end,
                                        self.filename.to_owned(),
                                    ),
                                )))
                            })
                        }
                        YyToken::Alias => Some(Ok((
                            Position::new_optional(extra.beg, self.filename.to_owned()),
                            Token::Alias,
                            Position::new_optional(extra.end, self.filename.to_owned()),
                        ))),
                        YyToken::Counter => Some(Ok((
                            Position::new_optional(extra.beg, self.filename.to_owned()),
                            Token::Counter,
                            Position::new_optional(extra.end, self.filename.to_owned()),
                        ))),
                        YyToken::Define => Some(Ok((
                            Position::new_optional(extra.beg, self.filename.to_owned()),
                            Token::Define,
                            Position::new_optional(extra.end, self.filename.to_owned()),
                        ))),
                        YyToken::Enum => Some(Ok((
                            Position::new_optional(extra.beg, self.filename.to_owned()),
                            Token::Enum,
                            Position::new_optional(extra.end, self.filename.to_owned()),
                        ))),
                        YyToken::Forall => Some(Ok((
                            Position::new_optional(extra.beg, self.filename.to_owned()),
                            Token::Forall,
                            Position::new_optional(extra.end, self.filename.to_owned()),
                        ))),
                        YyToken::In => Some(Ok((
                            Position::new_optional(extra.beg, self.filename.to_owned()),
                            Token::In,
                            Position::new_optional(extra.end, self.filename.to_owned()),
                        ))),
                        YyToken::Is => Some(Ok((
                            Position::new_optional(extra.beg, self.filename.to_owned()),
                            Token::Is,
                            Position::new_optional(extra.end, self.filename.to_owned()),
                        ))),
                        YyToken::Not => Some(Ok((
                            Position::new_optional(extra.beg, self.filename.to_owned()),
                            Token::Not,
                            Position::new_optional(extra.end, self.filename.to_owned()),
                        ))),
                        YyToken::Require => Some(Ok((
                            Position::new_optional(extra.beg, self.filename.to_owned()),
                            Token::Require,
                            Position::new_optional(extra.end, self.filename.to_owned()),
                        ))),
                        YyToken::Requires => Some(Ok((
                            Position::new_optional(extra.beg, self.filename.to_owned()),
                            Token::Requires,
                            Position::new_optional(extra.end, self.filename.to_owned()),
                        ))),
                        YyToken::CounterKind => Some(Ok((
                            Position::new_optional(extra.beg, self.filename.to_owned()),
                            Token::CounterKind(extra.data.counter_kind),
                            Position::new_optional(extra.end, self.filename.to_owned()),
                        ))),
                        YyToken::Value => Some(Ok((
                            Position::new_optional(extra.beg, self.filename.to_owned()),
                            Token::Value,
                            Position::new_optional(extra.end, self.filename.to_owned()),
                        ))),
                        YyToken::When => Some(Ok((
                            Position::new_optional(extra.beg, self.filename.to_owned()),
                            Token::When,
                            Position::new_optional(extra.end, self.filename.to_owned()),
                        ))),
                        YyToken::Trigger => Some(Ok((
                            Position::new_optional(extra.beg, self.filename.to_owned()),
                            Token::Trigger,
                            Position::new_optional(extra.end, self.filename.to_owned()),
                        ))),
                        YyToken::CounterVisibility => Some(Ok((
                            Position::new_optional(extra.beg, self.filename.to_owned()),
                            Token::CounterVisibility(extra.data.counter_visibility),
                            Position::new_optional(extra.end, self.filename.to_owned()),
                        ))),
                        YyToken::Base => Some(Ok((
                            Position::new_optional(extra.beg, self.filename.to_owned()),
                            Token::Base,
                            Position::new_optional(extra.end, self.filename.to_owned()),
                        ))),
                        YyToken::SetDefkey => Some(Ok((
                            Position::new_optional(extra.beg, self.filename.to_owned()),
                            Token::SetDefKey(extra.data.set_def_key),
                            Position::new_optional(extra.end, self.filename.to_owned()),
                        ))),
                        YyToken::Set => Some(Ok((
                            Position::new_optional(extra.beg, self.filename.to_owned()),
                            Token::Set,
                            Position::new_optional(extra.end, self.filename.to_owned()),
                        ))),
                        YyToken::SubsetOf => Some(Ok((
                            Position::new_optional(extra.beg, self.filename.to_owned()),
                            Token::SubsetOf,
                            Position::new_optional(extra.end, self.filename.to_owned()),
                        ))),
                        YyToken::Disjoint => Some(Ok((
                            Position::new_optional(extra.beg, self.filename.to_owned()),
                            Token::Disjoint,
                            Position::new_optional(extra.end, self.filename.to_owned()),
                        ))),
                        YyToken::Quotient => Some(Ok((
                            Position::new_optional(extra.beg, self.filename.to_owned()),
                            Token::Quotient,
                            Position::new_optional(extra.end, self.filename.to_owned()),
                        ))),
                        YyToken::Of => Some(Ok((
                            Position::new_optional(extra.beg, self.filename.to_owned()),
                            Token::Of,
                            Position::new_optional(extra.end, self.filename.to_owned()),
                        ))),
                        YyToken::Bool => Some(Ok((
                            Position::new_optional(extra.beg, self.filename.to_owned()),
                            Token::Bool(extra.data.boolean),
                            Position::new_optional(extra.end, self.filename.to_owned()),
                        ))),
                        YyToken::Colon => Some(Ok((
                            Position::new_optional(extra.beg, self.filename.to_owned()),
                            Token::Colon,
                            Position::new_optional(extra.end, self.filename.to_owned()),
                        ))),
                        YyToken::Comma => Some(Ok((
                            Position::new_optional(extra.beg, self.filename.to_owned()),
                            Token::Comma,
                            Position::new_optional(extra.end, self.filename.to_owned()),
                        ))),
                        YyToken::LParen => Some(Ok((
                            Position::new_optional(extra.beg, self.filename.to_owned()),
                            Token::LParen,
                            Position::new_optional(extra.end, self.filename.to_owned()),
                        ))),
                        YyToken::RParen => Some(Ok((
                            Position::new_optional(extra.beg, self.filename.to_owned()),
                            Token::RParen,
                            Position::new_optional(extra.end, self.filename.to_owned()),
                        ))),
                        YyToken::BitOr => Some(Ok((
                            Position::new_optional(extra.beg, self.filename.to_owned()),
                            Token::BitOr,
                            Position::new_optional(extra.end, self.filename.to_owned()),
                        ))),
                        YyToken::Or => Some(Ok((
                            Position::new_optional(extra.beg, self.filename.to_owned()),
                            Token::Or,
                            Position::new_optional(extra.end, self.filename.to_owned()),
                        ))),
                        YyToken::And => Some(Ok((
                            Position::new_optional(extra.beg, self.filename.to_owned()),
                            Token::And,
                            Position::new_optional(extra.end, self.filename.to_owned()),
                        ))),
                        YyToken::CmpOp => Some(Ok((
                            Position::new_optional(extra.beg, self.filename.to_owned()),
                            Token::CmpOp(extra.data.cmp_op),
                            Position::new_optional(extra.end, self.filename.to_owned()),
                        ))),
                        YyToken::Equal => Some(Ok((
                            Position::new_optional(extra.beg, self.filename.to_owned()),
                            Token::Equal,
                            Position::new_optional(extra.end, self.filename.to_owned()),
                        ))),
                        YyToken::End => Some(Ok((
                            Position::new_optional(extra.beg, self.filename.to_owned()),
                            Token::End,
                            Position::new_optional(extra.end, self.filename.to_owned()),
                        ))),
                        YyToken::Symmetric => Some(Ok((
                            Position::new_optional(extra.beg, self.filename.to_owned()),
                            Token::Symmetric,
                            Position::new_optional(extra.end, self.filename.to_owned()),
                        ))),
                        YyToken::AntiSymmetric => Some(Ok((
                            Position::new_optional(extra.beg, self.filename.to_owned()),
                            Token::AntiSymmetric,
                            Position::new_optional(extra.end, self.filename.to_owned()),
                        ))),
                        YyToken::Arrow => Some(Ok((
                            Position::new_optional(extra.beg, self.filename.to_owned()),
                            Token::Arrow,
                            Position::new_optional(extra.end, self.filename.to_owned()),
                        ))),
                        YyToken::Divide => Some(Ok((
                            Position::new_optional(extra.beg, self.filename.to_owned()),
                            Token::Divide,
                            Position::new_optional(extra.end, self.filename.to_owned()),
                        ))),
                        YyToken::Integer => Some(Ok((
                            Position::new_optional(extra.beg, self.filename.to_owned()),
                            Token::Integer,
                            Position::new_optional(extra.end, self.filename.to_owned()),
                        ))),
                        // Return None to signal EOF.for a reached end of the string.
                        YyToken::EOF => None,
                    }
                }
            }
        }
    }
}

impl Drop for Lexer {
    fn drop(&mut self) {
        unsafe {
            if let Some(yyin) = self.sublevel {
                libc::fclose(yyin);
            } else {
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
