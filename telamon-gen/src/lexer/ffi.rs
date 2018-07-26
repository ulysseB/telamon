#![allow(dead_code)]
use ::libc;
use ::ir;

use std::fmt;
use std::path::PathBuf;

/// A [yyscan](https://westes.github.io/flex/manual/About-yyscan_005ft.html) type is the internal
/// representation of a [yylex_init](https://westes.github.io/flex/manual/Init-and-Destroy-Functions.html) structure.
pub type YyScan = *const libc::c_void;
/// State per character.
pub type YyBufferState = *const libc::c_void;
/// Unsigned integer type used to represent the sizes f/lex.
pub type YySize = libc::size_t;

/// According to the [Default Memory Management](http://westes.github.io/flex/manual/The-Default-Memory-Management.html),
/// the input buffer is 16kB.
pub const YY_BUF_SIZE: libc::c_int = 16384;

/// A sequence's row/column position
#[derive(Copy, Clone)]
#[repr(C)]
pub union YyLval {
    /// Indicate a comparison operators.
    pub cmp_op: ir::CmpOp,
    pub boolean: bool,
    /// Indicates whether a counter sums or adds.
    pub counter_kind: ir::CounterKind,
    /// Indicates how a counter exposes how its maximum value.
    pub counter_visibility: ir::CounterVisibility,
    pub set_def_key: ir::SetDefKey,
}

/// A sequence's row/column position
#[derive(Default, Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct LexerPosition {
    pub line: libc::c_uint,
    pub column: libc::c_uint,
}

impl LexerPosition {

    // Returns a LexerPosition interface from a couple of line/column.
    pub fn new(line: libc::c_uint, column: libc::c_uint) -> Self {
        LexerPosition {
            line,
            column
        }
    }
}

impl fmt::Display for LexerPosition {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "line {}, column {}", self.line, self.column)
    }
}

#[derive(Default, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Position {
    pub position: LexerPosition,
    pub filename: Option<String>,
}

impl Position {
    // Returns a Position interface from LexerPosition with filename.
    pub fn new(position: LexerPosition, filename: String) -> Self {
        Position {
            position,
            filename: Some(filename),
        }
    }

    // Returns a Position interface from LexerPosition with optional filename.
    pub fn new_optional(position: LexerPosition, filename: Option<PathBuf>) -> Self {
        Position {
            position,
            filename: filename.and_then(|path| Some(path.to_string_lossy().to_string())),
        }
    }
}

impl From<LexerPosition> for Position {
    fn from(position: LexerPosition) -> Self {
        Position {
            position,
            ..Default::default()
        }
    }
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "position: {:?}, filename: {:?}", self.position, self.filename)
    }
}

/// A double sequence's row/column position
#[derive(Default, Clone, Debug, PartialEq)]
pub struct Span {
    pub beg: LexerPosition,
    pub end: Option<LexerPosition>,
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(end) = self.end {
            write!(f, "between {} and {}", self.beg, end)
        } else {
            write!(f, "at {}", self.beg)
        }
    }
}

/// A F/lex's token with a span.
#[derive(Default, Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub struct LexerSpanned<Y> {
    pub beg: LexerPosition,
    pub end: LexerPosition,
    /// Spanned data
    pub data: Y,
}

pub type YyExtraType = LexerSpanned<YyLval>;

#[derive(Default, Clone, PartialEq, Debug)]
pub struct Spanned<Y> {
    pub beg: Position,
    pub end: Position,
    /// Spanned data
    pub data: Y,
}

impl <Y> Spanned<Y> {
    pub fn with_data<T>(&self, data: T)  -> Spanned<T> {
        Spanned {
            beg: self.beg.to_owned(),
            end: self.end.to_owned(),
            data,
        }
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub enum YyToken {
    ValueIdent,
    ChoiceIdent,
    Var,
    Doc,
    CmpOp,
    InvalidToken,
    Code,
    CounterKind,
    Bool,
    CounterVisibility,
    And,
    Trigger,
    When,
    Alias,
    Counter,
    Define,
    Enum,
    Equal,
    Forall,
    In,
    Is,
    Not,
    Require,
    Requires,
    Value,
    End,
    Symmetric,
    AntiSymmetric,
    Arrow,
    Colon,
    Comma,
    LParen,
    RParen,
    BitOr,
    Or,
    SetDefkey,
    Set,
    SubsetOf,
    SetIdent,
    Base,
    Disjoint,
    Quotient,
    Of,
    Divide,
    Integer,
    /// Include exh header file.
    Include,
    /// End-of-File
    EOF = libc::EOF as _,
}

extern {
    pub fn yylex_init(scanner: *const YyScan) -> libc::c_int;
    pub fn yy_scan_string(yy_str: *const libc::c_char, yyscanner: YyScan) -> YyBufferState;
    pub fn yy_scan_buffer(base: *const libc::c_char, size: YySize, yyscanner: YyScan) -> YyBufferState;
    pub fn yy_scan_bytes(base: *const libc::c_char, len: libc::c_int, yyscanner: YyScan) -> YyBufferState;
    pub fn yy_create_buffer(file: *const libc::FILE, size: libc::c_int, yyscanner: YyScan) -> YyBufferState;
    pub fn yypush_buffer_state(buffer: YyBufferState, yyscanner: YyScan) -> libc::c_void;
    pub fn yypop_buffer_state(yyscanner: YyScan) -> libc::c_void;
    pub fn yyget_extra(yyscanner: YyScan) -> YyExtraType;
    pub fn yylex(yyscanner: YyScan) -> YyToken;
    pub fn yyget_text(yyscanner: YyScan) -> *mut libc::c_char;
    pub fn yyset_lineno(line_number: libc::c_int, yyscanner: YyScan) -> libc::c_int;
    pub fn yy_delete_buffer(b: YyBufferState, yyscanner: YyScan);
    pub fn yylex_destroy(yyscanner: YyScan) -> libc::c_int;
}
