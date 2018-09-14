//! Represents iteration dimensions.
use ir::{self, Statement};
use std::fmt;
use utils::*;

/// Provides a unique identifier for iteration dimensions.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
#[repr(C)]
/// cbindgen:field-names=[id]
pub struct DimId(pub u32);

impl Into<usize> for DimId {
    fn into(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for DimId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Represents an iteration dimension.
#[derive(Clone, Debug)]
pub struct Dimension<'a> {
    id: DimId,
    size: ir::PartialSize<'a>,
    possible_sizes: Vec<u32>,
    iterated: Vec<ir::InstId>,
    is_thread_dim: bool,
    logical_dim: Option<LogicalDimId>,
    mapped_dims: VecSet<DimMappingId>,
}

impl<'a> Dimension<'a> {
    /// Creates a new dimension.
    pub fn new(size: ir::PartialSize, id: DimId) -> Result<Dimension, ir::Error> {
        let possible_sizes = if let Some(size) = size.as_int() {
            if size == 1 {
                return Err(ir::Error::InvalidDimSize);
            }
            vec![size]
        } else {
            vec![]
        };
        Ok(Dimension {
            size,
            id,
            possible_sizes,
            iterated: Vec::new(),
            is_thread_dim: false,
            logical_dim: None,
            mapped_dims: VecSet::default(),
        })
    }

    /// Creates a new dimension with the same size as an existing one.
    pub fn with_same_size(id: DimId, other: &Self) -> Self {
        Dimension {
            size: other.size().clone(),
            possible_sizes: other.possible_sizes.clone(),
            id: id,
            iterated: Vec::new(),
            is_thread_dim: false,
            logical_dim: None,
            mapped_dims: VecSet::default(),
        }
    }

    /// Retruns the size of the dimension.
    pub fn size(&self) -> &ir::PartialSize<'a> {
        &self.size
    }

    /// Returns the values the size can take, if it is statically known.
    pub fn possible_sizes(&self) -> Option<&[u32]> {
        if self.possible_sizes.is_empty() {
            None
        } else {
            Some(&self.possible_sizes)
        }
    }

    /// Returns the id of the dimension.
    pub fn id(&self) -> DimId {
        self.id
    }

    /// Returns the constructs iterated along this dimension.
    pub fn iterated(&self) -> impl Iterator<Item = ir::InstId> + '_ {
        self.iterated.iter().cloned()
    }

    /// Adds a stmt that is iterated along self.
    pub fn add_iterated(&mut self, inst: ir::InstId) {
        self.iterated.push(inst);
    }

    /// Indicates if the dimension is a thread dimension.
    pub fn is_thread_dim(&self) -> bool {
        self.is_thread_dim
    }

    /// Sets the dimension as a thread dimension.
    pub fn set_thread_dim(&mut self) {
        self.is_thread_dim = true
    }

    /// Returns the logical dimension this dimension is part of, if any.
    pub fn logical_dim(&self) -> Option<LogicalDimId> {
        self.logical_dim
    }

    /// Returns the list of dimensions mapping containing this one.
    pub fn dim_mappings(&self) -> &VecSet<DimMappingId> {
        &self.mapped_dims
    }

    /// Register a dimension mapping.
    pub fn register_dim_mapping(&mut self, mapping: &DimMapping) {
        self.mapped_dims.insert(mapping.id);
        assert!(mapping.dims.contains(&self.id()));
    }
}

impl<'a> Statement<'a> for Dimension<'a> {
    fn stmt_id(&self) -> ir::StmtId {
        self.id.into()
    }

    fn as_dim(&self) -> Option<&Dimension<'a>> {
        Some(self)
    }
}

/// Provides a unique identifier for logic dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(C)]
pub struct LogicalDimId(pub u32);

/// A logic dimension composed of multiple `Dimension`s.
#[derive(Clone, Debug)]
pub struct LogicalDim<'a> {
    id: LogicalDimId,
    static_dims: Vec<DimId>,
    nonstatic_dim: Option<DimId>,
    possible_tilings: Vec<u32>,
    total_size: ir::Size<'a>,
}

impl<'a> LogicalDim<'a> {
    /// Creates a new logical dimension, composed only of static dimensions.
    pub fn new_static(
        id: LogicalDimId,
        static_dims: Vec<DimId>,
        total_size: u32,
    ) -> Self {
        LogicalDim {
            id,
            static_dims,
            nonstatic_dim: None,
            possible_tilings: vec![total_size],
            total_size: ir::Size::new_const(total_size),
        }
    }

    /// Creates a new logical dimension, composed of static dimensions and one
    /// dynamically-sized dimension.
    pub fn new_dynamic(
        id: LogicalDimId,
        dynamic_dim: DimId,
        static_dims: Vec<DimId>,
        possible_tilings: Vec<u32>,
        total_size: ir::Size<'a>,
    ) -> Self {
        LogicalDim {
            id,
            static_dims,
            nonstatic_dim: Some(dynamic_dim),
            possible_tilings,
            total_size,
        }
    }

    /// Returns a unique identifier for the logic dimension.
    pub fn id(&self) -> LogicalDimId {
        self.id
    }

    /// Returns the tiling dimensions, i.e. the dimensions with a static size.
    pub fn tile_dimensions(&self) -> impl Iterator<Item = DimId> + '_ {
        self.static_dims.iter().cloned()
    }

    /// Return the tiled dimensions, i.e. the dimension with a non-static size, if any.
    pub fn tiled_dimension(&self) -> Option<DimId> {
        self.nonstatic_dim
    }

    /// Returns the possible tiling factors.
    pub fn possible_tilings(&self) -> &[u32] {
        &self.possible_tilings
    }

    /// Returns all the dimensions constituing the logical dimension, from the inner-most
    /// to the outer-most.
    pub fn dimensions(&self) -> impl Iterator<Item = DimId> + '_ {
        self.static_dims
            .iter()
            .rev()
            .cloned()
            .chain(self.nonstatic_dim)
    }

    /// Returns the size of the logical dimension, i.e. the product of the sizes of its
    /// dimensions.
    pub fn total_size(&self) -> &ir::Size<'a> {
        &self.total_size
    }
}

/// Uniquely identifies a pair of mapped dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DimMappingId(pub u16);

impl From<DimMappingId> for usize {
    fn from(id: DimMappingId) -> usize {
        id.0 as usize
    }
}

/// Specifies that two dimensions should be mapped together.
#[derive(Clone, Copy, Debug)]
pub struct DimMapping {
    id: DimMappingId,
    dims: [DimId; 2],
}

impl DimMapping {
    /// Creates a `DimMapping`. Panics if the provided dimensions are the same.
    pub fn new(id: DimMappingId, dims: [DimId; 2]) -> Self {
        DimMapping { id, dims }
    }

    /// Returns the unique identifier of the `DimMapping`.
    pub fn id(&self) -> DimMappingId {
        self.id
    }

    /// Returns the mapped dims.
    pub fn dims(&self) -> [DimId; 2] {
        self.dims
    }
}
