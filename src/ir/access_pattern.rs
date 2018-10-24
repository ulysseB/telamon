/// Provides a way to represent the stride of a given variable.
use device::Device;
use indexmap::IndexMap;
use ir;
use itertools::Itertools;
use search_space::MemSpace;
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
    Unknown(Option<ir::MemId>),
    /// Access with a fixed stride on each dimensions. Accesses on two different
    /// dimensions should not overlap.
    Tensor {
        /// The memory block accessed.
        mem_id: Option<ir::MemId>,
        /// The elements type of the tensor.
        t: ir::Type,
        /// The dimensions of the tensor, with the innermost dimension first.
        dims: IndexMap<ir::DimId, ir::PartialSize<'a>>,
    },
}

impl<'a> AccessPattern<'a> {
    /// Indicates if memory accesses access to consecutive elements on the given dimension.
    pub fn is_layout_dimension(&self, dim: ir::DimId) -> bool {
        match self {
            AccessPattern::Unknown(..) => false,
            AccessPattern::Tensor { dims, .. } => dims.contains_key(&dim),
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

    /// Indicates the number of dimensions in the underlying layout.
    pub fn num_layout_dimensions(&self) -> usize {
        match self {
            AccessPattern::Unknown(..) => 0,
            AccessPattern::Tensor { dims, .. } => dims.len(),
        }
    }
}

/// Uniquely identifies a `LayoutDimension`.
#[derive(
    Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord, Debug, Serialize, Deserialize,
)]
pub struct LayoutDimId(pub usize);

impl From<LayoutDimId> for usize {
    fn from(id: LayoutDimId) -> Self {
        id.0
    }
}

/// Represents a dimension in a tensor.
#[derive(Clone, Debug)]
pub struct LayoutDimension {
    id: LayoutDimId,
    inst: Option<ir::InstId>,
    dim: ir::DimId,
    is_strided: bool,
    possible_ranks: Option<VecSet<u32>>,
}

impl LayoutDimension {
    /// Returns the unique identifier of the `LayoutDimension`.
    pub fn id(&self) -> LayoutDimId {
        self.id
    }

    /// Indicates the statement dimension that maps to this layout dimension.
    pub fn dim(&self) -> ir::DimId {
        self.dim
    }

    /// Indicates if the layout dimension is a dimension of a memory block.
    pub fn is_memory_layout(&self) -> bool {
        self.possible_ranks.is_some()
    }

    /// Indicates if the layout is accessed by an instruction.
    pub fn access_inst(&self) -> Option<ir::InstId> {
        self.inst
    }

    /// Indicates the possible orders the dimension can have if the layout is for a
    /// memory block.
    pub fn possible_ranks(&self) -> Option<&VecSet<u32>> {
        self.possible_ranks.as_ref()
    }

    /// Indicates if the layout dimension is strided with regard to the immediately inner
    /// dimension.
    pub fn is_strided(&self) -> bool {
        self.is_strided
    }

    /// Creates the layout dimensions that correspond to an access pattern. Returns the patterns in
    /// the order the ids where given.
    ///
    /// To obtain the number of ids to pass to this method, call
    /// `pattern.num_layout_dimensions()`.
    pub fn from_access_pattern<L>(
        ids: &[ir::LayoutDimId],
        inst: ir::InstId,
        pattern: &AccessPattern,
        fun: &ir::Function<L>,
    ) -> Vec<Self> {
        match pattern {
            AccessPattern::Unknown { .. } => {
                assert!(ids.is_empty());
                vec![]
            }
            AccessPattern::Tensor { dims, t, .. } => {
                let type_len = unwrap!(t.len_byte());
                let mut current_stride = ir::PartialSize::new(type_len, vec![]);
                dims.iter()
                    .zip_eq(ids)
                    .enumerate()
                    .map(|(rank, ((&dim, stride), &id))| {
                        let is_strided = *stride != current_stride;
                        current_stride = stride.clone() * fun.dim(dim).size();
                        LayoutDimension::new_static(
                            id,
                            dim,
                            rank as u32 + 1, // Ranks start at 1.
                            is_strided,
                            inst,
                        )
                    }).collect()
            }
        }
    }

    /// Creates a new dimension for a static layout.
    fn new_static(
        id: ir::LayoutDimId,
        dim: ir::DimId,
        rank: u32,
        is_strided: bool,
        inst: ir::InstId,
    ) -> Self {
        LayoutDimension {
            id,
            dim,
            possible_ranks: Some(VecSet::new(vec![rank])),
            is_strided,
            inst: Some(inst),
        }
    }

    /// Registers `self` in the other `fun` structures. Does not registers `self` in
    /// `self.inst` as inst is created as the same time as `self`.
    pub fn register<L>(&self, fun: &mut ir::Function<L>) {
        fun.dim_mut(self.dim).register_layout_dim(self.id);
    }
}
