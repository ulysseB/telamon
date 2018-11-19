//! Represents iteration dimensions.
use ir::{self, Statement};
use std;
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

impl std::fmt::Display for DimId {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// Represents an iteration dimension.
#[derive(Clone, Debug)]
pub struct Dimension<'a> {
    id: DimId,
    size: ir::PartialSize<'a>,
    possible_sizes: VecSet<u32>,
    iterated: Vec<ir::InstId>,
    is_thread_dim: bool,
    logical_dim: Option<LogicalDimId>,
    defined_vars: VecSet<ir::VarId>,
    used_vars: VecSet<ir::VarId>,
    inner_vars: VecSet<ir::VarId>,
    layout_dims: VecSet<ir::LayoutDimId>,
    is_parallelizable: bool,
}

impl<'a> Dimension<'a> {
    /// Creates a new dimension.
    pub fn new(
        id: DimId,
        size: ir::PartialSize<'a>,
        possible_sizes: VecSet<u32>,
        logical_dim: Option<LogicalDimId>,
    ) -> Result<Self, ir::Error> {
        if size.as_int() == Some(1) || possible_sizes.contains(&1) {
            return Err(ir::Error::InvalidDimSize);
        }
        Ok(Dimension {
            size,
            id,
            logical_dim,
            possible_sizes,
            iterated: Vec::new(),
            is_thread_dim: false,
            defined_vars: VecSet::default(),
            inner_vars: VecSet::default(),
            used_vars: VecSet::default(),
            layout_dims: VecSet::default(),
            is_parallelizable: true,
        })
    }

    /// Creates a dimension with a statically known size, picked in a list of
    /// possibilities.
    pub fn new_tile(
        id: DimId,
        possible_sizes: VecSet<u32>,
        logical_dim: LogicalDimId,
    ) -> Result<Self, ir::Error> {
        assert!(!possible_sizes.is_empty());
        let size = ir::PartialSize::new_dim_size(id);
        Self::new(id, size, possible_sizes, Some(logical_dim))
    }

    /// Creates a new dimension with the same size as an existing one.
    pub fn with_same_size(id: DimId, other: &Self) -> Self {
        let size = other.size.clone();
        // Cannot fail because the checks already passed when `other` was created.
        unwrap!(Self::new(id, size, other.possible_sizes.clone(), None))
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

    /// Returns the list of variables available inside the dimension.
    pub fn inner_vars(&self) -> &VecSet<ir::VarId> {
        &self.inner_vars
    }

    /// Register a variable available inside the dimension.
    pub fn register_inner_var(&mut self, var: ir::VarId) {
        self.inner_vars.insert(var);
    }

    /// Indicates the dimension cannot be parallelized.
    pub fn set_sequential(&mut self) {
        self.is_parallelizable = false;
    }

    /// Indicates if the dimension can be parallelized.
    pub fn is_parallelizable(&self) -> bool {
        self.is_parallelizable
    }

    /// Indicates the layout dimensions that maps to this dimension.
    pub fn layout_dims(&self) -> &VecSet<ir::LayoutDimId> {
        &self.layout_dims
    }

    /// Reigseter that a layout dimension is mapped to this dimension.
    pub fn register_layout_dim(&mut self, dim: ir::LayoutDimId) {
        self.layout_dims.insert(dim);
    }
}

impl<'a> Statement<'a> for Dimension<'a> {
    fn stmt_id(&self) -> ir::StmtId {
        self.id.into()
    }

    fn as_dim(&self) -> Option<&Dimension<'a>> {
        Some(self)
    }

    fn defined_vars(&self) -> &VecSet<ir::VarId> {
        &self.defined_vars
    }

    fn used_vars(&self) -> &VecSet<ir::VarId> {
        &self.used_vars
    }

    fn register_defined_var(&mut self, var: ir::VarId) {
        self.defined_vars.insert(var);
    }

    fn register_used_var(&mut self, var: ir::VarId) {
        self.used_vars.insert(var);
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
    possible_tilings: VecSet<u32>,
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
            possible_tilings: VecSet::new(vec![total_size]),
            total_size: ir::Size::new_const(total_size),
        }
    }

    /// Creates a new logical dimension, composed of static dimensions and one
    /// dynamically-sized dimension.
    pub fn new_dynamic(
        id: LogicalDimId,
        dynamic_dim: DimId,
        static_dims: Vec<DimId>,
        possible_tilings: VecSet<u32>,
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
