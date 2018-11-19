//! Representation and manipulation of a set of possible implementation.
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

use std;
use std::marker::PhantomData;

pub use self::dimension::{DimId, Dimension, LogicalDim, LogicalDimId};
pub use self::error::{Error, TypeError};
pub use self::function::{Function, Parameter, Signature};
pub use self::induction_var::{IndVarId, InductionVar};
pub use self::instruction::{InstId, Instruction};
pub use self::layout::*;
pub use self::mem::{ArrayId, MemId, MemorySpace};
pub use self::operand::Operand;
pub use self::operator::{BinOp, Operator, UnaryOp};
pub use self::size::{MemAccessStride, PartialSize, Size};
pub use self::statement::{Statement, StmtId};
pub use self::types::Type;
pub use self::variable::{
    FutureInstruction, FutureMemAccess, FutureVariable, ProductionPoint, VarDef,
    VarDefMode, VarId, VarUseMode, Variable,
};

pub mod mem;

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
    pub mem_insts: Vec<InstId>,
    pub iteration_dims: Vec<(InstId, DimId)>,
    pub thread_dims: Vec<DimId>,
    pub logical_dims: Vec<LogicalDimId>,
    pub tile_dimensions: Vec<(LogicalDimId, DimId)>,
    pub tiled_dimensions: Vec<(LogicalDimId, DimId)>,
    pub variables: Vec<VarId>,
    pub var_layout: Vec<(VarId, LayoutDimId)>,
    pub mem_var_layout: Vec<(VarId, LayoutDimId)>,
    pub use_statements: Vec<(VarId, StmtId)>,
    pub def_statements: Vec<(VarId, StmtId)>,
    pub predecessors: Vec<(VarId, VarId)>,
    pub layout_dims: Vec<LayoutDimId>,
    pub mem_layout_dims: Vec<LayoutDimId>,
    pub actual_layout_dims: Vec<(LayoutDimId, DimId)>,
    pub actual_layout_static_dims: Vec<(LayoutDimId, DimId)>,
    pub mem_access_layout: Vec<(InstId, LayoutDimId)>,
    pub predecessor_layout_dims: Vec<(LayoutDimId, LayoutDimId)>,
    pub predecessor_mem_layout_dims: Vec<(LayoutDimId, LayoutDimId)>,
    pub memory_vars: Vec<VarId>,
    pub accessed_var: Vec<(InstId, VarId)>,
}

impl NewObjs {
    /// Registers a new instruction.
    pub fn add_instruction(&mut self, inst: &Instruction) {
        self.add_stmt(inst);
        self.iteration_dims
            .extend(inst.iteration_dims().iter().map(|&dim| (inst.id(), dim)));
        if inst.as_mem_inst().is_some() {
            self.mem_insts.push(inst.id());
        }
        self.instructions.push(inst.id());
        for &layout_dim in inst.mem_access_layout() {
            self.mem_access_layout.push((inst.id(), layout_dim));
        }
        for var in inst.operator().accessed_mem_var() {
            self.accessed_var.push((inst.id(), var));
        }
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
        self.use_statements
            .extend(stmt.used_vars().iter().map(|&var| (var, stmt.stmt_id())));
    }

    /// Sets a dimension as a new iteration dimension.
    pub fn add_iteration_dim(&mut self, inst: InstId, dim: DimId) {
        self.iteration_dims.push((inst, dim));
    }

    /// Sets a dimension as a new thread dimension.
    pub fn add_thread_dim(&mut self, dim: DimId) {
        self.thread_dims.push(dim)
    }

    /// Registers a new variable.
    pub fn add_variable(&mut self, var: &Variable, fun: &Function) {
        self.variables.push(var.id());
        self.def_statements
            .extend(var.def_points().map(|stmt| (var.id(), stmt)));
        self.predecessors
            .extend(var.predecessors().iter().map(|&id| (var.id(), id)));
        for &layout_dim in var.layout() {
            self.add_layout_dim(fun.layout_dimension(layout_dim), fun);
        }
        if var.is_memory() {
            self.memory_vars.push(var.id());
        }
    }

    /// Adds a layout dimension.
    pub fn add_layout_dim(&mut self, dim: &LayoutDimension, fun: &Function) {
        self.layout_dims.push(dim.id());
        self.actual_layout_dims.push((dim.id(), dim.dim()));
        if fun.dim(dim.dim()).possible_sizes().is_some() {
            self.actual_layout_static_dims.push((dim.id(), dim.dim()));
        }
        if dim.is_memory_layout() {
            self.mem_layout_dims.push(dim.id());
        }
        for &pred in dim.predecessors() {
            self.predecessor_layout_dims.push((dim.id(), pred));
            if fun.layout_dimension(pred).is_memory_layout() {
                self.predecessor_mem_layout_dims.push((dim.id(), pred));
            }
        }
        if let Some(var) = dim.variable() {
            self.var_layout.push((var, dim.id()));
            if dim.is_memory_layout() {
                self.mem_var_layout.push((var, dim.id()));
            }
        }
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

    /// Returns the value stored at a given index and replaces it with a hole.
    pub fn remove(&mut self, index: I) -> Option<T> {
        std::mem::replace(&mut self.vec[index.into()], None)
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
/// in order to pre-allocate IDs for the various objects (variables,
/// instructions, dimensions, etc.) required for future lowering.
pub struct Counter {
    pub next_inst: usize,
    pub next_dim: usize,
    pub next_layout_dim: usize,
    pub next_variable: usize,
}

impl Counter {
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

    pub fn next_layout_dim(&mut self) -> LayoutDimId {
        let next = LayoutDimId(self.next_layout_dim);
        self.next_layout_dim += 1;
        next
    }

    pub fn next_variable(&mut self) -> VarId {
        let next = VarId(self.next_variable as u16);
        self.next_variable += 1;
        next
    }
}

// TODO(perf): group static computations
