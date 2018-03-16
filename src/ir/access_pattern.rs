/// Provides a way to represent the stride of a given variable.
use ir;

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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AccessPattern {
    /// Unknown access pattern.
    Unknown { mem_id: ir::mem::Id },
    /// Corresponds to accessing a tensor stored in a contiguous array in memory.
    Tensor { mem_id: ir::mem::Id, stride: i32, dims: Vec<ir::dim::Id> },
}

impl AccessPattern {
    /// Returns the stride on a given dimension.
    pub fn stride(&self, dim: ir::dim::Id) -> Stride {
        match *self {
            AccessPattern::Unknown { .. } => Stride::Unknown,
            AccessPattern::Tensor { stride, ref dims, .. } => {
                if dims.last() == Some(&dim) {
                    Stride::Int(stride)
                } else if dims.iter().any(|x| *x == dim) {
                    Stride::Unknown
                } else {
                    Stride::Int(0)
                }
            },
        }
    }

    /// Renames a basic block in the stride map.
    pub fn rename(&mut self, old: ir::dim::Id, new: ir::dim::Id) {
        match *self {
            AccessPattern::Unknown { .. } => (),
            AccessPattern::Tensor { ref mut dims, .. } =>
                for dim in dims { if *dim == old { *dim = new; } },
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
