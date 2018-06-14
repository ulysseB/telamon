#![allow(dead_code)]
use ::libc;
use ::ir;

use std::fmt;

/// A [yyscan](https://westes.github.io/flex/manual/About-yyscan_005ft.html) type is the internal
/// representation of a [yylex_init](https://westes.github.io/flex/manual/Init-and-Destroy-Functions.html) structure.
pub type YyScan = *const libc::c_void;
/// State per character.
pub type YyBufferState = *const libc::c_void;
/// Unsigned integer type used to represent the sizes f/lex.
pub type YySize = libc::size_t;

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
pub struct Position {
    pub line: libc::c_uint,
    pub column: libc::c_uint,
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "line {}, column {}", self.line, self.column)
    }
}

/// A double sequence's row/column position
#[derive(Default, Copy, Clone, Debug, PartialEq)]
pub struct Span {
    pub leg: Position,
    pub end: Option<Position>,
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(end) = self.end {
            write!(f, "between {} and {}", self.leg, end)
        } else {
            write!(f, "at {}", self.leg)
        }
    }
}

/// A F/lex's token with a span.
#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub struct Spanned<Y> {
    pub leg: Position,
    pub end: Position,
    /// Spanned data
    pub data: Y,
}

pub type YyExtraType = Spanned<YyLval>;

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
    /// End-of-File
    EOF = libc::EOF as _,
}

extern {
    pub fn yylex_init(scanner: *const YyScan) -> libc::c_int;
    pub fn yy_scan_string(yy_str: *const libc::c_char, yyscanner: YyScan) -> YyBufferState;
    pub fn yy_scan_buffer(base: *const libc::c_char, size: YySize, yyscanner: YyScan) -> YyBufferState;
    pub fn yy_scan_bytes(base: *const libc::c_char, len: libc::c_int, yyscanner: YyScan) -> YyBufferState;
    pub fn yyget_extra(yyscanner: YyScan) -> YyExtraType;
    pub fn yylex(yyscanner: YyScan) -> YyToken;
    pub fn yyget_text(yyscanner: YyScan) -> *mut libc::c_char;
    pub fn yyset_lineno(line_number: libc::c_int, yyscanner: YyScan) -> libc::c_int;
    pub fn yy_delete_buffer(b: YyBufferState, yyscanner: YyScan);
    pub fn yylex_destroy(yyscanner: YyScan) -> libc::c_int;
}
