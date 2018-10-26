//! Representation and manipulation of a set of possible implementation.
mod dim_map;
mod dimension;
mod error;
mod function;
mod induction_var;
mod instruction;
mod layout;
mod operand;
mod operator;
mod size;
mod statement;
mod types;
mod variable;

use itertools::Itertools;
use std;
use std::marker::PhantomData;

pub use self::dim_map::DimMap;
pub use self::dimension::{
    DimId, DimMapping, DimMappingId, Dimension, LogicalDim, LogicalDimId,
};
pub use self::error::{Error, TypeError};
pub use self::function::{Function, Parameter, Signature};
pub use self::induction_var::{IndVarId, InductionVar};
pub use self::instruction::{InstId, Instruction};
pub use self::layout::*;
pub use self::mem::MemId;
pub use self::operand::{DimMapScope, LoweringMap, Operand};
pub use self::operator::{BinOp, Operator, UnaryOp};
pub use self::size::{PartialSize, Size};
pub use self::statement::{Statement, StmtId};
pub use self::types::Type;
pub use self::variable::{MemoryLevel, VarDef, VarId, Variable};

pub mod mem;

/// Defines iteration dimensions properties.
pub mod dim {
    pub use super::dim_map::DimMap as Map;
}

/// Defines operators.
pub mod op {
    pub use super::operator::BinOp;
    pub use super::operator::Operator::*;
    pub use super::operator::Rounding;
}

/// Defines traits to import in the environment to use the IR.
pub mod prelude {
    pub use ir::mem::Block as MemoryRegion;
    pub use ir::statement::Statement;
}

/// Stores the objects created by a lowering.
#[derive(Default, Debug)]
pub struct NewObjs {
    pub instructions: Vec<InstId>,
    pub dimensions: Vec<DimId>,
    pub static_dims: Vec<DimId>,
    pub statements: Vec<StmtId>,
    pub mem_blocks: Vec<MemId>,
    pub mem_insts: Vec<InstId>,
    pub iteration_dims: Vec<(InstId, DimId)>,
    pub thread_dims: Vec<DimId>,
    pub logical_dims: Vec<LogicalDimId>,
    pub tile_dimensions: Vec<(LogicalDimId, DimId)>,
    pub tiled_dimensions: Vec<(LogicalDimId, DimId)>,
    pub dim_mappings: Vec<DimMappingId>,
    pub mapped_dims: Vec<(DimMappingId, DimId)>,
    pub static_mapped_dims: Vec<(DimMappingId, DimId)>,
    pub variables: Vec<VarId>,
    pub use_statements: Vec<(VarId, StmtId)>,
    pub def_statements: Vec<(VarId, StmtId)>,
    pub var_dims: Vec<(VarId, DimId)>,
    pub var_mappings: Vec<(VarId, DimMappingId)>,
    pub layout_dims: Vec<LayoutDimId>,
    pub mem_layout_dims: Vec<LayoutDimId>,
    pub actual_layout_dims: Vec<(LayoutDimId, DimId)>,
    pub mem_access_layout: Vec<(InstId, LayoutDimId)>,
}

impl NewObjs {
    /// Registers a new instruction.
    pub fn add_instruction(&mut self, inst: &Instruction) {
        self.add_stmt(inst);
        self.iteration_dims
            .extend(inst.iteration_dims().iter().map(|&dim| (inst.id(), dim)));
        self.mem_access_layout
            .extend(inst.mem_access_layout().iter().map(|&dim| (inst.id(), dim)));
        if inst.as_mem_inst().is_some() {
            self.mem_insts.push(inst.id());
        }
        self.instructions.push(inst.id());
    }

    /// Registers a new dimension.
    pub fn add_dimension(&mut self, dim: &Dimension) {
        self.add_stmt(dim);
        self.dimensions.push(dim.id());
        if dim.possible_sizes().is_some() {
            self.static_dims.push(dim.id());
        }
        if dim.is_thread_dim() {
            self.add_thread_dim(dim.id());
        }
    }

    /// Registers a new statement
    pub fn add_stmt(&mut self, stmt: &Statement) {
        self.statements.push(stmt.stmt_id());
    }

    /// Sets a dimension as a new iteration dimension.
    pub fn add_iteration_dim(&mut self, inst: InstId, dim: DimId) {
        self.iteration_dims.push((inst, dim));
    }

    /// Sets a dimension as a new thread dimension.
    pub fn add_thread_dim(&mut self, dim: DimId) {
        self.thread_dims.push(dim)
    }

    /// Registers a new memory block.
    pub fn add_mem_block(&mut self, id: MemId) {
        self.mem_blocks.push(id.into());
    }

    /// Adds a mapping between dimensions.
    pub fn add_dim_mapping(&mut self, mapping: &DimMapping, fun: &Function) {
        self.dim_mappings.push(mapping.id());
        for &dim in &mapping.dims() {
            self.mapped_dims.push((mapping.id(), dim));
            if fun.dim(dim).possible_sizes().is_some() {
                self.static_mapped_dims.push((mapping.id(), dim));
            }
        }
    }

    pub fn add_variable(&mut self, var: &Variable) {
        self.variables.push(var.id());
        self.def_statements
            .extend(var.def_points().map(|stmt| (var.id(), stmt)));
        self.use_statements
            .extend(var.use_points().map(|stmt| (var.id(), stmt)));
        self.var_dims
            .extend(var.dimensions().iter().map(|&dim| (var.id(), dim)));
        self.var_mappings
            .extend(var.def().mapped_dims().map(|id| (var.id(), id)));
    }

    /// Adds a layout dimension.
    pub fn add_layout_dim(&mut self, dim: &LayoutDimension) {
        self.layout_dims.push(dim.id());
        self.actual_layout_dims.push((dim.id(), dim.dim()));
        if dim.is_memory_layout() {
            self.mem_layout_dims.push(dim.id());
        }
    }
}

/// A point-to-point communication lowered into a store and a load.
pub struct LoweredDimMap {
    pub mem: MemId,
    pub store: InstId,
    pub load: InstId,
    /// Mapping from production dimensions to store dimensions.
    pub st_dims_mapping: Vec<(DimMappingId, [DimId; 2])>,
    /// Mapping from consumption dimensions to load dimensions.
    pub ld_dims_mapping: Vec<(DimMappingId, [DimId; 2])>,
}

impl LoweredDimMap {
    /// Adds the objects created by the lowering to the list of new objects.
    pub fn register_new_objs(&self, fun: &Function, new_objs: &mut NewObjs) {
        new_objs.add_mem_block(self.mem);
        new_objs.add_instruction(fun.inst(self.store));
        new_objs.add_instruction(fun.inst(self.load));
        let mappings = self.st_dims_mapping.iter().chain(&self.ld_dims_mapping);
        for &(mapping, [_, new_dim]) in mappings {
            new_objs.add_dimension(fun.dim(new_dim));
            new_objs.add_dim_mapping(fun.dim_mapping(mapping), fun);
        }
    }

    /// Returns the dimensions of the memory layout to create. For each dimension, gives
    /// a pair `(store dim, load dim)`.
    pub fn mem_dimensions(&self) -> impl Iterator<Item = (DimId, DimId)> + '_ {
        let st_dims = self.st_dims_mapping.iter().map(|&(_, [_, dim])| dim);
        let ld_dims = self.ld_dims_mapping.iter().map(|&(_, [_, dim])| dim);
        st_dims.zip_eq(ld_dims)
    }

    /// Returns the dimensions that store the variable.
    pub fn store_dims(&self) -> impl Iterator<Item = DimId> + '_ {
        self.st_dims_mapping.iter().map(|&(_, [_, dim])| dim)
    }

    /// Returns the dimensions that load the variable.
    pub fn load_dims(&self) -> impl Iterator<Item = DimId> + '_ {
        self.ld_dims_mapping.iter().map(|&(_, [_, dim])| dim)
    }
}

/// A vector with holes. This provides a similar interface as a
/// standard `Vec` does, but also provides a `set_lazy` method which
/// can be used to assign new values past the end of the vector and
/// will allocate holes in between if needed. Hole spaces must first
/// be reserved through `expand_to`.
///
/// An index type `I` is also used in order to provide strong typing
/// guarantees since `SparseVec` is meant to hold `DimId -> Dimension`
/// and `InstId -> Instruction` mappings.
#[derive(Clone, Debug)]
pub(crate) struct SparseVec<I, T> {
    vec: Vec<Option<T>>,
    capacity: usize,
    _marker: PhantomData<I>,
}

impl<T, I> SparseVec<I, T>
where
    I: Into<usize>,
{
    /// Creates a new sparse vector.
    pub fn new() -> Self {
        SparseVec {
            vec: Vec::new(),
            capacity: 0,
            _marker: PhantomData,
        }
    }

    pub fn from_vec(vec: Vec<Option<T>>) -> Self {
        let capacity = vec.len();
        SparseVec {
            vec,
            capacity,
            _marker: PhantomData,
        }
    }

    /// Returns the length of the sparse vector. This includes holes.
    pub fn len(&self) -> usize {
        self.capacity
    }

    /// Append an element to the back of the collection.
    pub fn push(&mut self, value: T) {
        self.make_holes();
        assert!(self.vec.len() == self.capacity);
        self.capacity += 1;
        self.vec.push(Some(value))
    }

    /// Add holes and the end of the underlying vector so that
    /// `self.vec.len() == self.capacity`.
    fn make_holes(&mut self) {
        if self.vec.len() < self.capacity {
            let extra = (self.vec.len()..=self.capacity).map(|_| None);
            self.vec.extend(extra);
        }
    }

    /// Increase the possible size of the vector by adding holes at
    /// the end.
    pub fn expand_to(&mut self, capacity: usize) {
        assert!(capacity >= self.capacity);
        self.capacity = capacity;
    }

    /// Initializes a hole. The index can be past the length of the
    /// vector.
    ///
    /// # Panics
    ///
    /// Panics if the element at index `index` is not a hole.
    pub fn set_lazy(&mut self, index: I, value: T) {
        let index = index.into();
        if index >= self.capacity {
            panic!("Cannot fill a hole that was not declared.");
        }

        if index >= self.vec.len() {
            let extra = (self.vec.len()..=index).map(|_| None);
            self.vec.extend(extra);
        }

        let old = ::std::mem::replace(&mut self.vec[index], Some(value));
        if old.is_some() {
            panic!("can only set a lazy entry once")
        }
    }

    /// Returns an iterator over the filled elements of the
    /// slice. Holes are skipped.
    pub fn iter<'a>(&'a self) -> impl Iterator<Item = &'a T> + Clone {
        self.vec.iter().filter_map(Option::as_ref)
    }

    /// Returns a mutable iterator over the filled elements of the
    /// slice. Holes are skipped.
    pub fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut T> {
        self.vec.iter_mut().filter_map(Option::as_mut)
    }
}

impl<I, T> Default for SparseVec<I, T>
where
    I: Into<usize>,
{
    fn default() -> Self {
        SparseVec::new()
    }
}

impl<I, T> IntoIterator for SparseVec<I, T> {
    type Item = Option<T>;
    type IntoIter = <Vec<Option<T>> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.vec.into_iter()
    }
}

/// Implements the indexing operation (`vec[index]`). The `index` must
/// be of type `I` as declared in the `SparseVec` type. Panics when
/// accessing an index which is out of bounds *or* contains a hole.
impl<T, I> ::std::ops::Index<I> for SparseVec<I, T>
where
    I: Into<usize>,
{
    type Output = T;

    fn index(&self, index: I) -> &T {
        unwrap!(self.vec[index.into()].as_ref())
    }
}

/// Implements the mutable indexing operation (`vec[index]`). The
/// `index` must be of type `I` as declared in the `SparseVec`
/// type. Panics when accessing an index which is out of bounds *or*
/// contains a hole.
impl<T, I> ::std::ops::IndexMut<I> for SparseVec<I, T>
where
    I: Into<usize>,
{
    fn index_mut(&mut self, index: I) -> &mut T {
        unwrap!(self.vec[index.into()].as_mut())
    }
}

impl<I, T> std::iter::Extend<T> for SparseVec<I, T>
where
    I: Into<usize>,
{
    fn extend<ITER: IntoIterator<Item = T>>(&mut self, iter: ITER) {
        self.make_holes();
        assert!(self.vec.len() == self.capacity);
        self.vec.extend(iter.into_iter().map(Some));
        self.capacity = self.vec.len();
    }
}

/// A wrapper used to count extra dimensions that will be needed in
/// the future and allocates IDs for them. This is used when freezing
/// in order to pre-allocate IDs for the various objects (internal
/// memory block, instructions, dimensions, etc.) required for future
/// lowering.
pub struct Counter {
    pub next_mem: usize,
    pub next_inst: usize,
    pub next_dim: usize,
    pub next_dim_mapping: u16,
    pub next_layout_dim: usize,
}

impl Counter {
    pub fn next_mem(&mut self) -> MemId {
        let next = MemId(self.next_mem as u32);
        self.next_mem += 1;
        next
    }

    pub fn next_inst(&mut self) -> InstId {
        let next = InstId(self.next_inst as u32);
        self.next_inst += 1;
        next
    }

    pub fn next_dim(&mut self) -> DimId {
        let next = DimId(self.next_dim as u32);
        self.next_dim += 1;
        next
    }

    pub fn next_dim_mapping(&mut self) -> DimMappingId {
        let next = DimMappingId(self.next_dim_mapping);
        self.next_dim_mapping += 1;
        next
    }

    pub fn next_layout_dim(&mut self) -> LayoutDimId {
        let next = LayoutDimId(self.next_layout_dim);
        self.next_layout_dim += 1;
        next
    }
}

// TODO(perf): group static computations
