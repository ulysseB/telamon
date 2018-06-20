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

#[derive(Clone, Debug)]
pub enum AccessPattern<'a> {
    /// Unknown access pattern.
    Unknown { mem_id: ir::mem::Id },
    /// Access with a fixed stride on each dimensions. Accesses on two different
    /// dimensions should not overlap.
    Tensor { mem_id: ir::mem::Id, dims: HashMap<ir::dim::Id, ir::Size<'a>> },
}

impl<'a> AccessPattern<'a> {
    /// Indicates if emory accesses access to consecutive elements on the given dimension.
    pub fn is_consecutive(&self, dim: ir::dim::Id, t: &ir::Type) -> bool {
        match self {
            AccessPattern::Unknown { .. } => false,
            AccessPattern::Tensor { dims, .. } => {
                dims.get(&dim).map(|s| s.as_fixed() == t.len_byte()).unwrap_or(false)
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
