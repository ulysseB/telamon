/// Tokens from the textual representation of constraints.

mod ffi;
mod token;

use std::{io,ptr};

use ::libc;
pub use self::token::Token;

pub struct Lexer {
    scanner: ffi::YyScan,
    buffer: ffi::YyBufferState,
}

impl Lexer {
    pub fn new(input: &mut io::Read) -> Self {
        let mut buffer = Vec::new();

        input.read_to_end(&mut buffer);
        unsafe {
            let scanner: ffi::YyScan = ptr::null();

            ffi::yylex_init(&scanner); // https://westes.github.io/flex/manual/Init-and-Destroy-Functions.html#index-yylex_005finit
            Lexer {
                scanner: scanner,
                buffer: ffi::yy_scan_string(buffer.as_ptr() as *const _, scanner), // https://westes.github.io/flex/manual/Multiple-Input-Buffers.html
            }
        }
    }
}

impl Drop for Lexer {
    fn drop(&mut self) {
        unsafe {
                libc::write(1, b"\n".as_ptr() as *const _, 1);
            ffi::yy_delete_buffer(self.buffer, self.scanner); // https://westes.github.io/flex/manual/Multiple-Input-Buffers.html
            ffi::yylex_destroy(self.scanner); // https://westes.github.io/flex/manual/Init-and-Destroy-Functions.html#index-yylex_005finit
        }
    }
}   

impl Iterator for Lexer {
    type Item = Token;

    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}
