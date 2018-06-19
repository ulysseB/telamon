/// Provides a way to represent the stride of a given variable.
use ir;
use utils::*;

/// A stride on a given dimensions.
#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub enum Stride {
    /// A fixed stride.
    Int(i32),
    /// A stride that is not statically known.
    Unknown,
}

impl Stride {
    /// Unwrap the stride or return the given value.
    pub fn unwrap_or(self, default: i32) -> i32 {
        match self {
            Stride::Int(stride) => stride,
            Stride::Unknown => default,
        }
    }
}

#[derive(Clone, Debug)]
pub enum AccessPattern<'a> {
    /// Unknown access pattern.
    Unknown { mem_id: ir::mem::Id },
    /// Access with a fixed stride on each dimensions. Accesses on two different
    /// dimensions should not overlap.
    Tensor { mem_id: ir::mem::Id, dims: HashMap<ir::dim::Id, ir::Size<'a>> },
}

impl<'a> AccessPattern<'a> {
    /// Returns the stride on a given dimension.
    pub fn stride(&self, dim: ir::dim::Id) -> Stride {
        match *self {
            AccessPattern::Unknown { .. } => Stride::Unknown,
            AccessPattern::Tensor { ref dims, .. } => {
                dims.get(&dim).map(|s| {
                    s.as_int().map(|x| Stride::Int(x as i32)).unwrap_or(Stride::Unknown)
                }).unwrap_or(Stride::Int(0))
            },
        }
    }

    /// Returns the id of the memory block accessed.
    pub fn mem_block(&self) -> ir::mem::Id {
        match *self {
            AccessPattern::Unknown { mem_id } |
            AccessPattern::Tensor { mem_id, .. } => mem_id,
        }
    }
}
