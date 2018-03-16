//! Represents iteration dimensions.
use ir::{self, BasicBlock};
use std::fmt;
use std::hash::{Hash, Hasher};
use utils::*;

/// Provides a unique identifier for iteration dimensions.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Id { pub id: u32 }

impl fmt::Debug  for Id {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { self.id.fmt(f) }
}

/// Represents an iteration dimension.
#[derive(Clone, Debug)]
pub struct Dimension<'a> {
    size: ir::Size<'a>,
    id: Id,
    iterated: Vec<ir::BBId>,
    iter_dims: HashSet<ir::dim::Id>,
}

impl<'a> Dimension<'a> {
    /// Creates a new dimension.
    pub fn new(size: ir::Size, id: Id) -> Dimension {
        assert_ne!(size.as_int(), Some(1));
        Dimension { size, id, iterated: Vec::new(), iter_dims: HashSet::default() }
    }

    /// Retruns the size of the dimension.
    pub fn size(&self) -> &ir::Size<'a> { &self.size }

    /// Returns the id of the dimension.
    pub fn id(&self) -> Id { self.id }

    /// Returns the constructs iterated along this dimension.
    pub fn iterated<'b>(&'b self) -> impl Iterator<Item=ir::BBId> + 'b {
        self.iterated.iter().cloned()
    }

    /// Adds a bb that is iterated along self.
    pub fn add_iterated(&mut self, bb: ir::BBId) { self.iterated.push(bb); }
}

impl<'a> BasicBlock<'a> for Dimension<'a> {
    fn bb_id(&self) -> ir::BBId { self.id.into() }

    fn as_dim(&self) -> Option<&Dimension<'a>> { Some(self) }

    fn iteration_dims(&self) -> &HashSet<ir::dim::Id> { &self.iter_dims }

    fn add_iteration_dimension(&mut self, dim: ir::dim::Id) -> bool {
        self.iter_dims.insert(dim)
    }
}

// Dimension equality is based on `BBId`.
impl<'a> PartialEq for Dimension<'a> {
    fn eq(&self, other: &Dimension<'a>) -> bool { self.bb_id() == other.bb_id() }
}

impl<'a> Eq for Dimension<'a> {}

// Dimension hash based on `BBId`.
impl<'a> Hash for Dimension<'a> {
    fn hash<H: Hasher>(&self, state: &mut H) { self.bb_id().hash(state) }
}
