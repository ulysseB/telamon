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
    /// Indicates if memory accesses access to consecutive elements on the given dimension.
    pub fn is_consecutive(&self, dim: ir::dim::Id, t: &ir::Type) -> bool {
        match self {
            AccessPattern::Unknown { .. } => false,
            AccessPattern::Tensor { dims, .. } => {
                dims.get(&dim).and_then(|stride| stride.as_int())
                    .map(|stride| Some(stride) == t.len_byte())
                    .unwrap_or(false)
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

    /// Ensure the access pattern is valid for an instruction nested in the dimensions
    /// given in `iter_dims`.
    pub fn check(&self, iter_dims: &HashSet<ir::dim::Id>) -> Result<(), ir::Error> {
        match self {
            AccessPattern::Unknown { .. } => Ok(()),
            AccessPattern::Tensor { dims, .. } => {
                // Ensures all dimensions referenced in the pattern are nested outside
                // the access pattern.
                for (&dim, _) in dims {
                    if !iter_dims.contains(&dim) {
                        return Err(ir::Error::InvalidDimInPattern { dim });
                    }
                }
                Ok(())
            },
        }
    }
}
