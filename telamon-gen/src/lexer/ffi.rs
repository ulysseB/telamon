use ::libc;

/// https://westes.github.io/flex/manual/About-yyscan_005ft.html
pub type YyScan = *const libc::c_void;
pub type YyBufferState = *const libc::c_void;

extern {
    pub static lines: libc::c_int;

    pub fn yylex_init(scanner: *const YyScan) -> libc::c_int;
    pub fn yy_scan_string(yy_str: *const libc::c_char, yyscanner: YyScan) -> YyBufferState;
    pub fn yylex(yyscanner: YyScan) -> libc::c_int;
    pub fn yyget_text(yyscanner: YyScan) -> *const libc::c_char;
    pub fn yy_delete_buffer(b: YyBufferState, yyscanner: YyScan);
    pub fn yylex_destroy(yyscanner: YyScan) -> libc::c_int;
}
