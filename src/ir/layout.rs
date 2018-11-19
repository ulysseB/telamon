/// Provides a way to represent the stride of a given variable.
use device::Device;
use indexmap::IndexMap;
use ir;
use itertools::Itertools;
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
    /// Access a variable, with the layout specified by `LayoutDimension` objects.
    Variable {
        /// Id of the variable accessed.
        id: ir::VarId,
        /// Mapping from the memory access dimensions to the variable dimensions.
        dims: HashMap<ir::DimId, ir::DimId>,
    },
}

impl<'a> AccessPattern<'a> {
    /// Indicates if memory accesses access to consecutive elements on the given dimension.
    pub fn is_layout_dimension(&self, dim: ir::DimId) -> bool {
        match self {
            AccessPattern::Unknown(..) => false,
            AccessPattern::Tensor { dims, .. } => dims.contains_key(&dim),
            AccessPattern::Variable { dims, .. } => dims.contains_key(&dim),
        }
    }

    /// Returns the id of the memory block accessed.
    pub fn accessed_array(&self) -> ir::ArrayId {
        match *self {
            AccessPattern::Unknown(Some(id))
            | AccessPattern::Tensor {
                mem_id: Some(id), ..
            } => id.into(),
            AccessPattern::Variable { id, .. } => ir::ArrayId::Variable(id),
            _ => ir::ArrayId::External,
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
            AccessPattern::Variable { dims, .. } => {
                // Ensures all dimensions referenced in the pattern are nested outside
                // the access pattern. We can't factor with the `AccessPattern::Tensor`
                // case as `dims` variables have different types.
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
        match self.accessed_array() {
            ir::ArrayId::External => device.pointer_type(ir::MemorySpace::Global),
            ir::ArrayId::Static(id) => ir::Type::PtrTo(id.into()),
            ir::ArrayId::Variable(var) => ir::Type::PtrTo(ir::ArrayId::Variable(var)),
        }
    }

    /// Indicates the number of dimensions in the underlying layout.
    pub fn num_layout_dimensions(&self) -> usize {
        match self {
            AccessPattern::Unknown(..) => 0,
            AccessPattern::Tensor { dims, .. } => dims.len(),
            AccessPattern::Variable { dims, .. } => dims.len(),
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
    variable: Option<ir::VarId>,
    accessed_variable: Option<ir::VarId>,
    dim: ir::DimId,
    is_strided: bool,
    possible_ranks: Option<VecSet<u32>>,
    predecessors: VecSet<LayoutDimId>,
    successors: VecSet<LayoutDimId>,
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

    /// Indicates to which variable the layout belongs, if any.
    pub fn variable(&self) -> Option<ir::VarId> {
        self.variable
    }

    /// Indicates if the layout is to access a variable. A layout may access a variable
    /// without belonging to it. The variable may have its own layout dimensions
    /// and `self`be the layout of a memory access instruction instead.
    pub fn accessed_variable(&self) -> Option<ir::VarId> {
        self.accessed_variable
    }

    /// Indicates the possible orders the dimension can have if the layout is for a
    /// memory block.
    pub fn possible_ranks(&self) -> Option<&VecSet<u32>> {
        self.possible_ranks.as_ref()
    }

    /// Registers the layout dimension is for a memory block and sets the list of
    /// possible ranks. Panics if ranks where already set.
    pub fn set_possible_ranks(&mut self, ranks: VecSet<u32>) {
        assert!(self.possible_ranks.is_none());
        self.possible_ranks = Some(ranks);
    }

    /// Indicates if the layout dimension is strided.
    ///
    /// A dimension is strided if the layout is fixed and the amount to increment the
    /// pointer at each iteration is not the product of the inner dimensions. For example:
    /// ```
    /// let array = [42; 40];
    /// for i in 0..8 { // Not strided
    ///     for j in 0..4 { // Strided
    ///         let x = array[i+10*j];
    ///     }
    /// }
    /// ```
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
            AccessPattern::Variable { dims, id: var_id } => {
                let reverse_dims: HashMap<_, _> =
                    dims.iter().map(|(&k, &v)| (v, k)).collect();
                fun.variable(*var_id)
                    .layout()
                    .iter()
                    .flat_map(|&layout_dim| {
                        let var_dim = fun.layout_dimension(layout_dim).dim();
                        reverse_dims.get(&var_dim).map(|&dim| (layout_dim, dim))
                    }).zip_eq(ids)
                    .map(|((var_layout_dim_id, access_dim), &id)| LayoutDimension {
                        id,
                        dim: access_dim,
                        inst: Some(inst),
                        variable: None,
                        accessed_variable: Some(*var_id),
                        is_strided: false,
                        possible_ranks: Some(VecSet::new((0..16).collect())),
                        predecessors: VecSet::new(vec![var_layout_dim_id]),
                        successors: VecSet::default(),
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
            variable: None,
            accessed_variable: None,
            predecessors: VecSet::default(),
            successors: VecSet::default(),
        }
    }

    /// Create a new dimension for a dynamic layout.
    pub fn new_dynamic(id: ir::LayoutDimId, dim: ir::DimId, variable: ir::VarId) -> Self {
        LayoutDimension {
            id,
            dim,
            possible_ranks: None,
            is_strided: false,
            inst: None,
            variable: Some(variable),
            accessed_variable: Some(variable),
            predecessors: VecSet::default(),
            successors: VecSet::default(),
        }
    }

    /// Registers `self` in the other `fun` structures. Does not registers `self` in
    /// `self.inst` as inst is created as the same time as `self`.
    pub fn register<L>(&self, fun: &mut ir::Function<L>) {
        fun.dim_mut(self.dim).register_layout_dim(self.id);
        for &pred in &self.predecessors {
            fun.layout_dimension_mut(pred).successors.insert(self.id());
        }
    }

    /// List the layout dimensions of preceeding aliasing variables or memory accesses
    /// that are mapped to this layout dimension.
    pub fn predecessors(&self) -> &VecSet<ir::LayoutDimId> {
        &self.predecessors
    }

    /// Adds a layout to the list of predecessors.
    pub fn add_predecessor(&mut self, pred: ir::LayoutDimId) {
        self.predecessors.insert(pred);
    }

    /// List the layout dimensions of dependent aliasing variables or memory accesses that
    /// are mapped to this layout dimension.
    pub fn successors(&self) -> &VecSet<ir::LayoutDimId> {
        &self.successors
    }

    /// Adds a layout dimensions to the list of successors.
    pub fn add_successor(&mut self, succ: ir::LayoutDimId) {
        self.successors.insert(succ);
    }
}
