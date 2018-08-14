/// This lexer is a application of
/// [Writing a custom lexer](https://github.com/lalrpop/lalrpop/blob/master/doc/src/lexer_tutorial/index.md)'s
/// documentation. This includes a Spanned definition and a Iterator.
mod ffi;
mod token;

use std::{io, ptr, fs, path};
use std::error::Error;
use std::fmt;

pub use self::token::Token;
pub use self::ffi::Spanned;

use std::path::{PathBuf, Path};
use std::ffi::{CStr, CString};

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

pub use self::ffi::{LexerPosition, Position, Span};

use ::libc;
use ::errno::{errno, set_errno};

/// The alias Spanned is a definition of the stream format.
/// The parser will accept an iterator where each item
/// in the stream has the following structure.
pub type ParserSpanned<Tok, Pos, Err> = Result<(Pos, Tok, Pos), Err>;

pub type LexerItem = ParserSpanned<Token, Position, LexicalError>;

/// FD of include header.
type YyIn = *mut libc::FILE;

#[derive(Debug, Clone, PartialEq)]
/// TODO; struct Error {
///     
/// }
///
/// LexicalError {
///     beg
///     end
///     cause: ErrorKind
/// }
///
/// enum ErrorKind {
///     InvalidToken(String)
///     InvalidInclude(String, Errno)
/// }
pub enum LexicalError {
    InvalidToken(Position, Token, Position),
    InvalidInclude(Position, Token, Position),
    UnexpectedToken(Position, Token, Position),
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
                },
                next => {
                    self.lookahead_token = next;
                    return Some(Ok((Position::new_optional(extra.beg, self.filename.to_owned()),
                                    Token::Code(buffer),
                                    Position::new_optional(acc_end, self.filename.to_owned())
                    )))
                },
            }
        }
    }

    /// Links the new include buffer, Returns the first token from this include.
    fn include_open(&mut self, extra: YyExtraType, filename: CString) -> Option<LexerItem> {
        unsafe {
            // [Multiple Input Buffers](http://westes.github.io/flex/manual/Multiple-Input-Buffers.html#Multiple-Input-Buffers)
            // include file scanner.
            match libc::fopen(filename.to_bytes_with_nul()
                                      .as_ptr() as *const _, "r".as_ptr() as *const _) {
                yyin if yyin.is_null() => {
                    let why = errno();

                    set_errno(why);
                    Some(Err(LexicalError::InvalidInclude(
                        Position::new_optional(extra.beg, self.filename.to_owned()),
                        Token::InvalidInclude(filename.to_str().unwrap().to_string(), why),
                        Position::new_optional(extra.end, self.filename.to_owned())
                    )))
                },
                yyin => {
                    let buffer: YyBufferState = yy_create_buffer(yyin, YY_BUF_SIZE, self.scanner);

                    yypush_buffer_state(buffer, self.scanner);
                    self.include = Some(Box::new(Lexer {
                        filename: Some(PathBuf::from(filename.to_str().unwrap())),
                        scanner: self.scanner,
                        // The function  [yy_scan_bytes](https://westes.github.io/flex/manual/Multiple-Input-Buffers.html)
                        // scans len bytes starting at location bytes.
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

                            CStr::from_ptr(out)
                                 .to_str().ok()
                                 .and_then(|s: &str| Some(Err(
                                    LexicalError::InvalidToken(
                                        Position::new_optional(
                                            extra.beg, self.filename.to_owned()
                                        ),
                                        Token::InvalidToken(s.to_owned()),
                                        Position::new_optional(
                                            extra.end, self.filename.to_owned()
                                        ),
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
                                        Position::new_optional(
                                            extra.beg, self.filename.to_owned()
                                        ),
                                        Token::ChoiceIdent(s.to_owned()),
                                        Position::new_optional(
                                            extra.end, self.filename.to_owned()
                                        ),
                                ))))
                        },
                        YyToken::SetIdent => {
                            let out = yyget_text(self.scanner);

                            CStr::from_ptr(out)
                                 .to_str().ok()
                                 .and_then(|s: &str| Some(Ok((
                                        Position::new_optional(
                                            extra.beg, self.filename.to_owned()
                                        ),
                                        Token::SetIdent(s.to_owned()),
                                        Position::new_optional(
                                            extra.end, self.filename.to_owned()
                                        ),
                                ))))
                        },
                        YyToken::ValueIdent => {
                            let out = yyget_text(self.scanner);

                            CStr::from_ptr(out)
                                 .to_str().ok()
                                 .and_then(|s: &str| Some(Ok((
                                        Position::new_optional(
                                            extra.beg, self.filename.to_owned()
                                        ),
                                        Token::ValueIdent(s.to_owned()),
                                        Position::new_optional(
                                            extra.end, self.filename.to_owned()
                                        ),
                                ))))
                        },
                        YyToken::Var => {
                            let out = yyget_text(self.scanner);

                            CStr::from_ptr(out)
                                 .to_str().ok()
                                 .and_then(|s: &str| Some(Ok((
                                        Position::new_optional(
                                            extra.beg, self.filename.to_owned()
                                        ),
                                        Token::Var(s.to_owned()),
                                        Position::new_optional(
                                            extra.end, self.filename.to_owned()
                                        ),
                                ))))
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
                                 .and_then(|s: &str| Some(Ok((
                                        Position::new_optional(
                                            extra.beg, self.filename.to_owned()
                                        ),
                                        Token::Doc(s.to_owned()),
                                        Position::new_optional(
                                            extra.end, self.filename.to_owned()
                                        ),
                                ))))
                        },
                        YyToken::Alias => Some(Ok((
                            Position::new_optional(
                                extra.beg, self.filename.to_owned()
                            ),
                            Token::Alias,
                            Position::new_optional(
                                extra.end, self.filename.to_owned()
                            ),
                        ))),
                        YyToken::Counter => Some(Ok((
                            Position::new_optional(
                                extra.beg, self.filename.to_owned()
                            ),
                            Token::Counter,
                            Position::new_optional(
                                extra.end, self.filename.to_owned()
                            ),
                        ))),
                        YyToken::Define => Some(Ok((
                            Position::new_optional(
                                extra.beg, self.filename.to_owned()
                            ),
                            Token::Define,
                            Position::new_optional(
                                extra.end, self.filename.to_owned()
                            ),
                        ))),
                        YyToken::Enum => Some(Ok((
                            Position::new_optional(
                                extra.beg, self.filename.to_owned()
                            ),
                            Token::Enum,
                            Position::new_optional(
                                extra.end, self.filename.to_owned()
                            ),
                        ))),
                        YyToken::Forall => Some(Ok((
                            Position::new_optional(
                                extra.beg, self.filename.to_owned()
                            ),
                            Token::Forall,
                            Position::new_optional(
                                extra.end, self.filename.to_owned()
                            ),
                        ))),
                        YyToken::In => Some(Ok((
                            Position::new_optional(
                                extra.beg, self.filename.to_owned()
                            ),
                            Token::In,
                            Position::new_optional(
                                extra.end, self.filename.to_owned()
                            ),
                        ))),
                        YyToken::Is => Some(Ok((
                            Position::new_optional(
                                extra.beg, self.filename.to_owned()
                            ),
                            Token::Is,
                            Position::new_optional(
                                extra.end, self.filename.to_owned()
                            ),
                        ))),
                        YyToken::Not => Some(Ok((
                            Position::new_optional(
                                extra.beg, self.filename.to_owned()
                            ),
                            Token::Not,
                            Position::new_optional(
                                extra.end, self.filename.to_owned()
                            ),
                        ))),
                        YyToken::Require => Some(Ok((
                            Position::new_optional(
                                extra.beg, self.filename.to_owned()
                            ),
                            Token::Require,
                            Position::new_optional(
                                extra.end, self.filename.to_owned()
                            ),
                        ))),
                        YyToken::Requires => Some(Ok((
                            Position::new_optional(
                                extra.beg, self.filename.to_owned()
                            ),
                            Token::Requires,
                            Position::new_optional(
                                extra.end, self.filename.to_owned()
                            ),
                        ))),
                        YyToken::CounterKind => Some(Ok((
                            Position::new_optional(
                                extra.beg, self.filename.to_owned()
                            ),
                            Token::CounterKind(extra.data.counter_kind),
                            Position::new_optional(
                                extra.end, self.filename.to_owned()
                            ),
                        ))),
                        YyToken::Value => Some(Ok((
                            Position::new_optional(
                                extra.beg, self.filename.to_owned()
                            ),
                            Token::Value,
                            Position::new_optional(
                                extra.end, self.filename.to_owned()
                            ),
                        ))),
                        YyToken::When => Some(Ok((
                            Position::new_optional(
                                extra.beg, self.filename.to_owned()
                            ),
                            Token::When,
                            Position::new_optional(
                                extra.end, self.filename.to_owned()
                            ),
                        ))),
                        YyToken::Trigger => Some(Ok((
                            Position::new_optional(
                                extra.beg, self.filename.to_owned()
                            ),
                            Token::Trigger,
                            Position::new_optional(
                                extra.end, self.filename.to_owned()
                            ),
                        ))),
                        YyToken::CounterVisibility => Some(Ok((
                            Position::new_optional(
                                extra.beg, self.filename.to_owned()
                            ),
                            Token::CounterVisibility(extra.data.counter_visibility),
                            Position::new_optional(
                                extra.end, self.filename.to_owned()
                            ),
                        ))),
                        YyToken::Base => Some(Ok((
                            Position::new_optional(
                                extra.beg, self.filename.to_owned()
                            ),
                            Token::Base,
                            Position::new_optional(
                                extra.end, self.filename.to_owned()
                            ),
                        ))),
                        YyToken::SetDefkey => Some(Ok((
                            Position::new_optional(
                                extra.beg, self.filename.to_owned()
                            ),
                            Token::SetDefKey(extra.data.set_def_key),
                            Position::new_optional(
                                extra.end, self.filename.to_owned()
                            ),
                        ))),
                        YyToken::Set => Some(Ok((
                            Position::new_optional(
                                extra.beg, self.filename.to_owned()
                            ),
                            Token::Set,
                            Position::new_optional(
                                extra.end, self.filename.to_owned()
                            ),
                        ))),
                        YyToken::SubsetOf => Some(Ok((
                            Position::new_optional(
                                extra.beg, self.filename.to_owned()
                            ),
                            Token::SubsetOf,
                            Position::new_optional(
                                extra.end, self.filename.to_owned()
                            ),
                        ))),
                        YyToken::Disjoint => Some(Ok((
                            Position::new_optional(
                                extra.beg, self.filename.to_owned()
                            ),
                            Token::Disjoint,
                            Position::new_optional(
                                extra.end, self.filename.to_owned()
                            ),
                        ))),
                        YyToken::Quotient => Some(Ok((
                            Position::new_optional(
                                extra.beg, self.filename.to_owned()
                            ),
                            Token::Quotient,
                            Position::new_optional(
                                extra.end, self.filename.to_owned()
                            ),
                        ))),
                        YyToken::Of => Some(Ok((
                            Position::new_optional(
                                extra.beg, self.filename.to_owned()
                            ),
                            Token::Of,
                            Position::new_optional(
                                extra.end, self.filename.to_owned()
                            ),
                        ))),
                        YyToken::Bool => Some(Ok((
                            Position::new_optional(
                                extra.beg, self.filename.to_owned()
                            ),
                            Token::Bool(extra.data.boolean),
                            Position::new_optional(
                                extra.end, self.filename.to_owned()
                            ),
                        ))),
                        YyToken::Colon => Some(Ok((
                            Position::new_optional(
                                extra.beg, self.filename.to_owned()
                            ),
                            Token::Colon,
                            Position::new_optional(
                                extra.end, self.filename.to_owned()
                            ),
                        ))),
                        YyToken::Comma => Some(Ok((
                            Position::new_optional(
                                extra.beg, self.filename.to_owned()
                            ),
                            Token::Comma,
                            Position::new_optional(
                                extra.end, self.filename.to_owned()
                            ),
                        ))),
                        YyToken::LParen => Some(Ok((
                            Position::new_optional(
                                extra.beg, self.filename.to_owned()
                            ),
                            Token::LParen,
                            Position::new_optional(
                                extra.end, self.filename.to_owned()
                            ),
                        ))),
                        YyToken::RParen => Some(Ok((
                            Position::new_optional(
                                extra.beg, self.filename.to_owned()
                            ),
                            Token::RParen,
                            Position::new_optional(
                                extra.end, self.filename.to_owned()
                            ),
                        ))),
                        YyToken::BitOr => Some(Ok((
                            Position::new_optional(
                                extra.beg, self.filename.to_owned()
                            ),
                            Token::BitOr,
                            Position::new_optional(
                                extra.end, self.filename.to_owned()
                            ),
                        ))),
                        YyToken::Or => Some(Ok((
                            Position::new_optional(
                                extra.beg, self.filename.to_owned()
                            ),
                            Token::Or,
                            Position::new_optional(
                                extra.end, self.filename.to_owned()
                            ),
                        ))),
                        YyToken::And => Some(Ok((
                            Position::new_optional(
                                extra.beg, self.filename.to_owned()
                            ),
                            Token::And,
                            Position::new_optional(
                                extra.end, self.filename.to_owned()
                            ),
                        ))),
                        YyToken::CmpOp => Some(Ok((
                            Position::new_optional(
                                extra.beg, self.filename.to_owned()
                            ),
                            Token::CmpOp(extra.data.cmp_op),
                            Position::new_optional(
                                extra.end, self.filename.to_owned()
                            ),
                        ))),
                        YyToken::Equal => Some(Ok((
                            Position::new_optional(
                                extra.beg, self.filename.to_owned()
                            ),
                            Token::Equal,
                            Position::new_optional(
                                extra.end, self.filename.to_owned()
                            ),
                        ))),
                        YyToken::End => Some(Ok((
                            Position::new_optional(
                                extra.beg, self.filename.to_owned()
                            ),
                            Token::End,
                            Position::new_optional(
                                extra.end, self.filename.to_owned()
                            ),
                        ))),
                        YyToken::Symmetric => Some(Ok((
                            Position::new_optional(
                                extra.beg, self.filename.to_owned()
                            ),
                            Token::Symmetric,
                            Position::new_optional(
                                extra.end, self.filename.to_owned()
                            ),
                        ))),
                        YyToken::AntiSymmetric => Some(Ok((
                            Position::new_optional(
                                extra.beg, self.filename.to_owned()
                            ),
                            Token::AntiSymmetric,
                            Position::new_optional(
                                extra.end, self.filename.to_owned()
                            ),
                        ))),
                        YyToken::Arrow => Some(Ok((
                            Position::new_optional(
                                extra.beg, self.filename.to_owned()
                            ),
                            Token::Arrow,
                            Position::new_optional(
                                extra.end, self.filename.to_owned()
                            ),
                        ))),
                        YyToken::Divide => Some(Ok((
                            Position::new_optional(
                                extra.beg, self.filename.to_owned()
                            ),
                            Token::Divide,
                            Position::new_optional(
                                extra.end, self.filename.to_owned()
                            ),
                        ))),
                        YyToken::Integer => Some(Ok((
                            Position::new_optional(
                                extra.beg, self.filename.to_owned()
                            ),
                            Token::Integer,
                            Position::new_optional(
                                extra.end, self.filename.to_owned()
                            ),
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
