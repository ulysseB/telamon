//! Represents iteration dimensions.
use ir::{self, BasicBlock};
use std::fmt;
use std::hash::{Hash, Hasher};

/// Provides a unique identifier for iteration dimensions.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Id(pub u32);

impl fmt::Debug  for Id {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "dim::Id({})", self.0)
    }
}

impl fmt::Display for Id {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { self.0.fmt(f) }
}

/// Represents an iteration dimension.
#[derive(Clone, Debug)]
pub struct Dimension<'a> {
    size: ir::Size<'a>,
    id: Id,
    iterated: Vec<ir::InstId>,
    is_thread_dim: bool,
}

impl<'a> Dimension<'a> {
    /// Creates a new dimension.
    pub fn new(size: ir::Size, id: Id) -> Result<Dimension, ir::Error> {
        if size.as_int().map(|i| i <= 1).unwrap_or(false) {
            return Err(ir::Error::InvalidDimSize);
        }
        Ok(Dimension {
            size, id,
            iterated: Vec::new(),
            is_thread_dim: false,
        })
    }

    /// Retruns the size of the dimension.
    pub fn size(&self) -> &ir::Size<'a> { &self.size }

    /// Returns the id of the dimension.
    pub fn id(&self) -> Id { self.id }

    /// Returns the constructs iterated along this dimension.
    pub fn iterated<'b>(&'b self) -> impl Iterator<Item=ir::InstId> + 'b {
        self.iterated.iter().cloned()
    }

    /// Adds a bb that is iterated along self.
    pub fn add_iterated(&mut self, inst: ir::InstId) { self.iterated.push(inst); }

    /// Indicates if the dimension is a thread dimension.
    pub fn is_thread_dim(&self) -> bool { self.is_thread_dim }

    /// Sets the dimension as a thread dimension.
    pub fn set_thread_dim(&mut self) { self.is_thread_dim = true }
}

impl<'a> BasicBlock<'a> for Dimension<'a> {
    fn bb_id(&self) -> ir::BBId { self.id.into() }

    fn as_dim(&self) -> Option<&Dimension<'a>> { Some(self) }
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
