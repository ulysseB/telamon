//! Representation and manipulation of a set of possible implementation.
mod access_pattern;
mod basic_block;
mod dimension;
mod dim_map;
mod error;
mod function;
mod induction_var;
mod instruction;
mod operand;
mod operator;
mod size;
mod types;

use std::marker::PhantomData;

pub use self::access_pattern::{Stride, AccessPattern};
pub use self::basic_block::{BasicBlock, BBId};
pub use self::dim_map::DimMap;
pub use self::dimension::{DimId, Dimension};
pub use self::error::{Error, TypeError};
pub use self::function::{Function, Signature, Parameter};
pub use self::instruction::{InstId, Instruction};
pub use self::induction_var::{IndVarId, InductionVar};
pub use self::mem::MemId;
pub use self::operand::{Operand, DimMapScope, LoweringMap};
pub use self::operator::{BinOp, Operator};
pub use self::size::Size;
pub use self::types::Type;

pub mod mem;

/// Defines iteration dimensions properties.
pub mod dim {
    pub use super::dim_map::DimMap as Map;
}

/// Defines operators.
pub mod op {
    pub use super::operator::Operator::*;
    pub use super::operator::Rounding;
    pub use super::operator::BinOp;
}

/// Defines traits to import in the environment to use the IR.
pub mod prelude {
    pub use ir::basic_block::BasicBlock;
    pub use ir::mem::Block as MemBlock;
}

/// Stores the objects created by a lowering.
#[derive(Default, Debug)]
pub struct NewObjs {
    pub instructions: Vec<InstId>,
    pub dimensions: Vec<DimId>,
    pub static_dims: Vec<DimId>,
    pub basic_blocks: Vec<BBId>,
    pub mem_blocks: Vec<MemId>,
    pub internal_mem_blocks: Vec<mem::InternalId>,
    pub mem_insts: Vec<InstId>,
    pub iteration_dims: Vec<(InstId, DimId)>,
    pub thread_dims: Vec<DimId>,
}

impl NewObjs {
    /// Registers a new instruction.
    pub fn add_instruction(&mut self, inst: &Instruction) {
        self.add_bb(inst);
        for &dim in inst.iteration_dims() { self.iteration_dims.push((inst.id(), dim)); }
        self.instructions.push(inst.id());
    }

    /// Registers a new memory instruction.
    pub fn add_mem_instruction(&mut self, inst: &Instruction) {
        self.add_instruction(inst);
        self.mem_insts.push(inst.id());
    }

    /// Registers a new dimension.
    pub fn add_dimension(&mut self, dim: &Dimension) {
        self.add_bb(dim);
        self.dimensions.push(dim.id());
        if dim.possible_sizes().is_some() { self.static_dims.push(dim.id()); }
        if dim.is_thread_dim() { self.add_thread_dim(dim.id()); }
    }

    /// Registers a new basic block.
    pub fn add_bb(&mut self, bb: &BasicBlock) {
        self.basic_blocks.push(bb.bb_id());
    }

    /// Sets a dimension as a new iteration dimension.
    pub fn add_iteration_dim(&mut self, inst: InstId, dim: DimId) {
        self.iteration_dims.push((inst, dim));
    }

    /// Sets a dimension as a new thread dimension.
    pub fn add_thread_dim(&mut self, dim: DimId) { self.thread_dims.push(dim) }

    /// Registers a new memory block.
    pub fn add_mem_block(&mut self, id: mem::InternalId) {
        self.mem_blocks.push(id.into());
        self.internal_mem_blocks.push(id);
    }
}

/// A point-to-point communication lowered into a store and a load.
pub struct LoweredDimMap {
    pub mem: mem::InternalId,
    pub store: InstId,
    pub load: InstId,
    pub dimensions: Vec<(DimId, DimId)>
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
where I: Into<usize>
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
        if self.vec.len() < self.capacity {
            let extra = (self.vec.len()..=self.capacity).map(|_| None);
            self.vec.extend(extra);
        }
        assert!(self.vec.len() == self.capacity);

        self.capacity += 1;
        self.vec.push(Some(value))
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
where I: Into<usize>
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
where I: Into<usize>
{
    fn index_mut(&mut self, index: I) -> &mut T {
        unwrap!(self.vec[index.into()].as_mut())
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
}

impl Counter {
    pub fn next_mem(&mut self) -> mem::InternalId {
        let next = mem::InternalId(self.next_mem as u32);
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
}

// TODO(perf): group static computations
