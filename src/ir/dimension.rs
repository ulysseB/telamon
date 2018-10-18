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
pub struct Dimension<'a, L = ir::LoweringMap> {
    id: DimId,
    size: ir::PartialSize<'a>,
    possible_sizes: VecSet<u32>,
    iterated: Vec<ir::InstId>,
    is_thread_dim: bool,
    logical_dim: Option<LogicalDimId>,
    mapped_dims: VecSet<DimMappingId>,
    defined_vars: VecSet<ir::VarId>,
    inner_vars: VecSet<ir::VarId>,
    is_parallelizable: bool,
    freeze_marker: std::marker::PhantomData<L>,
}

impl<'a> Dimension<'a, ()> {
    /// Sets the dimension as frozen.
    pub fn freeze(self) -> Dimension<'a> {
        Dimension {
            id: self.id,
            is_thread_dim: self.is_thread_dim,
            size: self.size,
            possible_sizes: self.possible_sizes,
            iterated: self.iterated,
            logical_dim: self.logical_dim,
            mapped_dims: self.mapped_dims,
            defined_vars: self.defined_vars,
            inner_vars: self.inner_vars,
            is_parallelizable: self.is_parallelizable,
            freeze_marker: std::marker::PhantomData,
        }
    }
}

impl<'a, L> Dimension<'a, L> {
    /// Creates a new dimension.
    pub fn new(
        id: DimId,
        size: ir::PartialSize<'a>,
        logical_dim: Option<LogicalDimId>,
    ) -> Result<Self, ir::Error> {
        let possible_sizes = if let Some(size) = size.as_int() {
            if size == 1 {
                return Err(ir::Error::InvalidDimSize);
            }
            VecSet::new(vec![size])
        } else {
            VecSet::default()
        };
        trace!("new dim {:?}, size = {:?}", id, size);
        Ok(Dimension {
            size,
            id,
            logical_dim,
            possible_sizes,
            iterated: Vec::new(),
            is_thread_dim: false,
            mapped_dims: VecSet::default(),
            defined_vars: VecSet::default(),
            inner_vars: VecSet::default(),
            is_parallelizable: true,
            freeze_marker: std::marker::PhantomData,
        })
    }

    /// Creates a dimension with a statically known size, picked in a list of
    /// possibilities.
    pub fn new_static(
        id: DimId,
        possible_sizes: VecSet<u32>,
        logical_dim: Option<LogicalDimId>,
    ) -> Result<Self, ir::Error> {
        if possible_sizes.contains(&1) {
            return Err(ir::Error::InvalidDimSize);
        }
        trace!("new static {:?}, size = {:?}", id, possible_sizes);
        Ok(Dimension {
            size: ir::PartialSize::new_dim_size(id),
            id,
            possible_sizes,
            logical_dim,
            iterated: Vec::new(),
            is_thread_dim: false,
            mapped_dims: VecSet::default(),
            defined_vars: VecSet::default(),
            inner_vars: VecSet::default(),
            is_parallelizable: true,
            freeze_marker: std::marker::PhantomData,
        })
    }

    /// Creates a new dimension with the same size as an existing one.
    pub fn with_same_size(id: DimId, other: &Self) -> Self {
        // Cannot fail because the checks already passed when `other` was created.
        unwrap!(if other.possible_sizes.is_empty() {
            Self::new(id, other.size().clone(), None)
        } else {
            Self::new_static(id, other.possible_sizes.clone(), None)
        })
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
}

lazy_static! {
    // This empty set is necessary because `Statement` must return references the the sets of
    // variables it uses and defines but does not contains any. Thus, instead of creating fields with
    // empty set we return a reference to this global variable.
    static ref NO_VALUES: VecSet<ir::VarId> = VecSet::default();
}

impl<'a, L> Statement<'a, L> for Dimension<'a, L> {
    fn stmt_id(&self) -> ir::StmtId {
        self.id.into()
    }

    fn as_dim(&self) -> Option<&Dimension<'a, L>> {
        Some(self)
    }

    fn defined_vars(&self) -> &VecSet<ir::VarId> {
        &self.defined_vars
    }

    fn used_vars(&self) -> &VecSet<ir::VarId> {
        &NO_VALUES
    }

    fn register_defined_var(&mut self, var: ir::VarId) {
        self.defined_vars.insert(var);
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

/// Uniquely identifies a pair of mapped dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DimMappingId(pub u16);

impl From<DimMappingId> for usize {
    fn from(id: DimMappingId) -> usize {
        id.0 as usize
    }
}

/// Specifies that two dimensions should be mapped together.
#[derive(Clone, Debug)]
pub struct DimMapping {
    id: DimMappingId,
    dims: [DimId; 2],
    variables: VecSet<ir::VarId>,
}

impl DimMapping {
    /// Creates a `DimMapping`. Panics if the provided dimensions are the same.
    pub fn new(id: DimMappingId, dims: [DimId; 2]) -> Self {
        DimMapping {
            id,
            dims,
            variables: VecSet::default(),
        }
    }

    /// Returns the unique identifier of the `DimMapping`.
    pub fn id(&self) -> DimMappingId {
        self.id
    }

    /// Returns the mapped dims.
    pub fn dims(&self) -> [DimId; 2] {
        self.dims
    }

    /// Returns the variables that rely on this mapping.
    pub fn users(&self) -> &VecSet<ir::VarId> {
        &self.variables
    }

    /// Registers that a variable uses this mapping.
    pub fn register_user(&mut self, user: ir::VarId) {
        self.variables.insert(user);
    }
}
