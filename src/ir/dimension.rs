//! Represents iteration dimensions.
use ir::{self, BasicBlock};
use std::fmt;

/// Provides a unique identifier for iteration dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[derive(Serialize, Deserialize)]
#[repr(C)]
pub struct DimId(pub u32);

impl fmt::Display for DimId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { self.0.fmt(f) }
}

/// Represents an iteration dimension.
#[derive(Clone, Debug)]
pub struct Dimension<'a> {
    size: ir::Size<'a>,
    id: DimId,
    iterated: Vec<ir::InstId>,
    is_thread_dim: bool,
}

impl<'a> Dimension<'a> {
    /// Creates a new dimension.
    pub fn new(size: ir::Size, id: DimId) -> Result<Dimension, ir::Error> {
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
    pub fn id(&self) -> DimId { self.id }

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
