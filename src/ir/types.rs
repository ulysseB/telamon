/// Describes the types instruction and operands can take.
use crate::ir;
use serde::{Deserialize, Serialize};
use std::fmt;
use utils::*;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
/// Values and intructions types.
pub enum Type {
    /// Type for integer values, with a fixed number of bits.
    I(u16),
    /// Type for unsigned integer values, with a fixed number of bits.
    U(u16),
    /// Type for floating point values, with a fixed number of bits.
    F(u16),
    /// Pointer type of the given memory space.
    PtrTo(ir::MemId),
}

pub struct IdentName<'a> {
    t: &'a Type,
}

impl fmt::Display for IdentName<'_> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.t {
            Type::I(s) => write!(fmt, "i{}", s),
            Type::U(s) => write!(fmt, "u{}", s),
            Type::F(s) => write!(fmt, "f{}", s),
            Type::PtrTo(mem) => write!(fmt, "memptr{}", mem.0),
        }
    }
}

impl Type {
    pub fn ident_name(&self) -> IdentName<'_> {
        IdentName { t: self }
    }

    /// Returns true if the type is an integer.
    pub fn is_integer(self) -> bool {
        match self {
            Type::I(_) | Type::PtrTo(_) | Type::U(_) => true,
            Type::F(_) => false,
        }
    }

    /// Returns true if the type is a float.
    pub fn is_float(self) -> bool {
        match self {
            Type::F(_) => true,
            Type::I(_) | Type::PtrTo(..) | Type::U(_) => false,
        }
    }

    /// Returns the number of bytes of the type.
    pub fn len_byte(self) -> Option<u32> {
        match self {
            Type::I(i) | Type::F(i) | Type::U(i) => Some(u32::from(div_ceil(i, 8))),
            Type::PtrTo(_) => None,
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Type::I(s) => write!(f, "i{}", s),
            Type::U(s) => write!(f, "u{}", s),
            Type::F(s) => write!(f, "f{}", s),
            Type::PtrTo(mem) => write!(f, "ptr to {:?}", mem),
        }
    }
}
