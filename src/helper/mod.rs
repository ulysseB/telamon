//! Helper functions to build an IR instance.
mod builder;
mod operand;
mod signature;

pub mod tensor;

pub use self::builder::Builder;
pub use self::signature::Builder as SignatureBuilder;
pub use self::operand::{AutoOperand, Reduce, TmpArray};

use ir;
use std;

/// A logical dimension, possible composed of multiple nested dimensions.
pub trait MetaDimension {
    /// Returns the ids of the underlying dimensions.
    fn ids<'a>(&'a self) -> Box<DoubleEndedIterator<Item=ir::dim::Id> + 'a>;
}

impl MetaDimension for ir::dim::Id {
    fn ids<'a>(&'a self) -> Box<DoubleEndedIterator<Item=ir::dim::Id> + 'a> {
        Box::new(std::iter::once(*self))
    }
}

/// A groups of dimensions that act as a single logical dimension.
pub struct DimGroup { dims: Vec<ir::dim::Id> }

impl DimGroup {
    /// Creates a dimension group containing the given dimensions.
    pub fn new(dims: Vec<ir::dim::Id>) -> Self { DimGroup { dims } }
}

impl MetaDimension for DimGroup {
    fn ids<'a>(&'a self) -> Box<DoubleEndedIterator<Item=ir::dim::Id> + 'a> {
        Box::new(self.dims.iter().cloned())
    }
}

impl std::ops::Index<usize> for DimGroup {
    type Output = ir::dim::Id;

    fn index(&self, index: usize) -> &ir::dim::Id { &self.dims[index] }
}

/// A logical basic block, that can actually be implemented by multiple ones.
pub trait MetaBasicBlock {
    /// Returns the ids on the underlying basic blocks.
    fn ids<'a>(&'a self) -> Box<Iterator<Item=ir::BBId> + 'a>;
}

impl MetaBasicBlock for ir::BBId {
    fn ids<'a>(&'a self) -> Box<Iterator<Item=ir::BBId> + 'a> {
        Box::new(std::iter::once(*self))
    }
}

impl MetaBasicBlock for ir::InstId {
    fn ids<'a>(&'a self) -> Box<Iterator<Item=ir::BBId> + 'a> {
        Box::new(std::iter::once((*self).into()))
    }
}

impl<T> MetaBasicBlock for T where T: MetaDimension {
    fn ids<'a>(&'a self) -> Box<Iterator<Item=ir::BBId> + 'a> {
        Box::new(MetaDimension::ids(self).map(|x| x.into()))
    }
}
