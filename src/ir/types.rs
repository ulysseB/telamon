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
    /// Type for floating point values, with a fixed number of bits.
    F(u16),
    /// Pointer type of the given memory space.
    PtrTo(ir::MemId),
}

impl Type {
    /// Returns true if the type is an integer.
    pub fn is_integer(self) -> bool {
        match self {
            Type::I(_) | Type::PtrTo(_) => true,
            Type::F(_) => false,
        }
    }

    /// Returns true if the type is a float.
    pub fn is_float(self) -> bool {
        match self {
            Type::F(_) => true,
            Type::I(_) | Type::PtrTo(..) => false,
        }
    }

    /// Return the number of bits of the type
    pub fn bitwidth(self) -> Option<u32> {
        match self {
            Type::I(bits) | Type::F(bits) => Some(u32::from(bits)),
            _ => None,
        }
    }

    /// Returns the number of bytes of the type.
    pub fn len_byte(self) -> Option<u32> {
        self.bitwidth().map(|bits| div_ceil(bits, 8))
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Type::I(s) => write!(f, "i{}", s),
            Type::F(s) => write!(f, "f{}", s),
            Type::PtrTo(mem) => write!(f, "ptr to {:?}", mem),
        }
    }
}
