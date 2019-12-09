//! Helpers to generate code from an IR instance and fully specified decisions.
mod cfg;
mod dimension;
mod expr;
mod function;
pub mod llir;
mod name_map;
mod printer;
mod size;
mod variable;

pub use self::cfg::Cfg;
pub use self::dimension::Dimension;
pub use self::function::*;
pub use self::name_map::{Interner, NameGenerator, NameMap};
pub use self::printer::{IdentDisplay, InstPrinter, Printer};
pub use self::size::Size;
pub use self::variable::Variable;

fn i32_div_magic_and_shift(val: i32) -> (i32, i32) {
    let l = (32 - (val - 1).leading_zeros()).max(1) as i32;
    let m = 1 + (1u64 << (31 + l)) / (val as u64);
    ((m.wrapping_sub(1u64 << 32)) as i32, l - 1)
}

pub fn i32_div_magic(val: i32) -> i32 {
    let (magic, _shift) = i32_div_magic_and_shift(val);
    magic
}

pub fn i32_div_shift(val: i32) -> i32 {
    let (_magic, shift) = i32_div_magic_and_shift(val);
    shift
}

// TODO(cleanup): refactor function
// - extend instructions with additional information: vector factor, flag, instantiated dims
// TODO(cleanup): refactor namer
