use ::libc;

/// https://westes.github.io/flex/manual/About-yyscan_005ft.html
pub type YyScan = *const libc::c_void;
pub type YyBufferState = *const libc::c_void;

#[derive(Copy, Clone)]
#[repr(C)]
pub union YyLval {
    val: libc::c_int,
    pub cmp_op: CmpOp,
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
    EOF = libc::EOF as _,
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub enum CmpOp {
    Lt,
    Gt,
    Leq,
    Geq,
    Eq,
    Neq,
}

extern {
    pub static yylval: YyLval;

    pub fn yylex_init(scanner: *const YyScan) -> libc::c_int;
    pub fn yy_scan_string(yy_str: *const libc::c_char, yyscanner: YyScan) -> YyBufferState;
    pub fn yylex(yyscanner: YyScan) -> YyToken;
    pub fn yyget_text(yyscanner: YyScan) -> *const libc::c_char;
    pub fn yy_delete_buffer(b: YyBufferState, yyscanner: YyScan);
    pub fn yylex_destroy(yyscanner: YyScan) -> libc::c_int;
}
