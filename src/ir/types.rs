/// Describes the types instruction and operands can take.
use ir;
use std::fmt;
use utils::*;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
/// Values and intructions types.
pub enum Type {
    /// Type for instructions that do not produce a value.
    Void,
    /// Type for integer values, with a fixed number of bits.
    I(u16),
    /// Type for floating point values, with a fixed number of bits.
    F(u16),
    /// Pointer type of the given memory space.
    PtrTo(ir::mem::Id),
}

impl Type {
    /// Returns true if the type is an integer.
    pub fn is_integer(&self) -> bool {
        match *self {
            Type::I(_) | Type::PtrTo(_) => true,
            Type::Void | Type::F(_) => false,
        }
    }

    /// Returns true if the type is a float.
    pub fn is_float(&self) -> bool {
        match *self {
            Type::F(_) => true,
            Type::Void | Type::I(_) | Type::PtrTo(..) => false,
        }
    }

    /// Returns the number of bytes of the type.
    pub fn len_byte(&self) -> Option<u32> {
        match *self {
            Type::I(i) | Type::F(i) => Some(u32::from(div_ceil(i, 8))),
            Type::Void => Some(0),
            Type::PtrTo(_) => None,
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Type::Void => write!(f, "void"),
            Type::I(s) => write!(f, "i{}", s),
            Type::F(s) => write!(f, "f{}", s),
            Type::PtrTo(mem) => write!(f, "ptr to {:?}", mem),
        }
    }
}
