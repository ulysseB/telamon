//! Represents iteration dimensions.
use ir::{self, BasicBlock};
use std::fmt;

/// Provides a unique identifier for iteration dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[derive(Serialize, Deserialize)]
#[repr(C)]
/// cbindgen:field-names=[id]
pub struct DimId(pub u32);

impl Into<usize> for DimId {
    fn into(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for DimId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { self.0.fmt(f) }
}

/// Represents an iteration dimension.
#[derive(Clone, Debug)]
pub struct Dimension<'a> {
    id: DimId,
    size: ir::Size<'a>,
    possible_sizes: Vec<u32>,
    iterated: Vec<ir::InstId>,
    is_thread_dim: bool,
}

impl<'a> Dimension<'a> {
    /// Creates a new dimension.
    pub fn new(size: ir::Size, id: DimId) -> Result<Dimension, ir::Error> {
        let possible_sizes = if let Some(size) = size.as_int() {
            if size == 1 { return Err(ir::Error::InvalidDimSize); }
            vec![size]
        } else {
            vec![]
        };
        Ok(Dimension {
            size, id, possible_sizes,
            iterated: Vec::new(),
            is_thread_dim: false,
        })
    }

    /// Creates a new dimension with the same size as an existing one.
    pub fn with_same_size(id: DimId, other: &Self) -> Self {
        Dimension {
            size: other.size().clone(),
            possible_sizes: other.possible_sizes.clone(),
            id: id,
            iterated: Vec::new(),
            is_thread_dim: false,
        }
    }

    /// Retruns the size of the dimension.
    pub fn size(&self) -> &ir::Size<'a> { &self.size }

    /// Returns the values the size can take, if it is statically known.
    pub fn possible_sizes(&self) -> Option<&[u32]> {
        if self.possible_sizes.is_empty() { None } else { Some(&self.possible_sizes) }
    }

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
