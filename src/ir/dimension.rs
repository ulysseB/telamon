//! Represents iteration dimensions.
use ir::{self, BasicBlock};
use std::fmt;

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
    pub fn new(size: ir::Size, id: Id) -> Dimension {
        if size.universe().map(|u| u.contains(&1)).unwrap_or(false) {
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

    /// Returns the logical dimension this real dimension is part of, if any.
    pub fn logical_dim(&self) -> Option<LogicalId> { None } // FIXME
}

impl<'a> BasicBlock<'a> for Dimension<'a> {
    fn bb_id(&self) -> ir::BBId { self.id.into() }

    fn as_dim(&self) -> Option<&Dimension<'a>> { Some(self) }
}

hash_from_key!(Dimension<'a>, &Dimension::id, 'a);

/// Provides a unique identifier for logic dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LogicalId(pub u32);

/// A logic dimension composed of multiple `Dimension`s.
#[derive(Clone, Debug)]
pub struct LogicalDim {
    id: LogicalId,
    static_dims: Vec<Id>,
    nonstatic_dim: Option<Id>,
    max_size: u32,
    has_dyn_dims: bool,
}

impl LogicalDim {
    /// Returns a unique identifier for the logic dimension.
    pub fn id(&self) -> LogicalId { self.id }

    /// Returns the dimensions with a static size in the logic dimension.
    pub fn static_dims(&self) -> impl Iterator<Item=Id> + '_ {
        self.static_dims.iter().cloned()
    }

    pub fn nonstatic_dim(&self) -> Option<ir::dim::Id> { self.nonstatic_dim }

    /// Returns the maximum size of combined static dimensions.
    pub fn max_static_size(&self) -> u32 { self.max_size }

    /// Returns the minimum size of combined static dimensions.
    pub fn min_static_size(&self) -> u32 {
        if self.has_dyn_dims { 1 } else { self.max_size }
    }
}

hash_from_key!(LogicalDim, &LogicalDim::id);
