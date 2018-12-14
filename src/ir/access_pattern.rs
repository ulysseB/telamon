/// Provides a way to represent the stride of a given variable.
use crate::device::Device;
use crate::ir;
use crate::search_space::MemSpace;
use telamon_utils::*;

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
    Unknown(Option<ir::MemId>),
    /// Access with a fixed stride on each dimensions. Accesses on two different
    /// dimensions should not overlap.
    Tensor {
        mem_id: Option<ir::MemId>,
        dims: HashMap<ir::DimId, ir::PartialSize<'a>>,
    },
}

impl<'a> AccessPattern<'a> {
    /// Indicates if memory accesses access to consecutive elements on the given dimension.
    pub fn is_consecutive(&self, dim: ir::DimId, t: &ir::Type) -> bool {
        match self {
            AccessPattern::Unknown(..) => false,
            AccessPattern::Tensor { dims, .. } => dims
                .get(&dim)
                .and_then(|stride| stride.as_int())
                .map(|stride| Some(stride) == t.len_byte())
                .unwrap_or(false),
        }
    }

    /// Returns the id of the memory block accessed.
    pub fn mem_block(&self) -> Option<ir::MemId> {
        match *self {
            AccessPattern::Unknown(mem_id) | AccessPattern::Tensor { mem_id, .. } => {
                mem_id
            }
        }
    }

    /// Ensure the access pattern is valid for an instruction nested in the dimensions
    /// given in `iter_dims`.
    pub fn check(&self, iter_dims: &HashSet<ir::DimId>) -> Result<(), ir::Error> {
        match self {
            AccessPattern::Unknown(..) => Ok(()),
            AccessPattern::Tensor { dims, .. } => {
                // Ensures all dimensions referenced in the pattern are nested outside
                // the access pattern.
                for (&dim, _) in dims.iter() {
                    if !iter_dims.contains(&dim) {
                        return Err(ir::Error::MissingIterationDim { dim });
                    }
                }
                Ok(())
            }
        }
    }

    /// Returns the type of pointer to use for the access.
    pub fn pointer_type(&self, device: &Device) -> ir::Type {
        // We either have a memory ID or the array is located in global memory.
        self.mem_block()
            .map(ir::Type::PtrTo)
            .unwrap_or_else(|| device.pointer_type(MemSpace::GLOBAL))
    }
}
