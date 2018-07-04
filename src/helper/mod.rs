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

impl<T> MetaDimension for T where T: std::borrow::Borrow<[ir::dim::Id]> {
    fn ids<'b>(&'b self) -> Box<DoubleEndedIterator<Item=ir::dim::Id> + 'b> {
        Box::new(self.borrow().iter().cloned())
    }
}

/// A groups of dimensions that act as a single logical dimension.
#[derive(Clone, Default)]
pub struct DimGroup {
    dims: Vec<ir::dim::Id>
}

impl DimGroup {
    /// Creates a dimension group containing the given dimensions.
    #[deprecated]
    pub fn new(dims: Vec<ir::dim::Id>) -> Self { DimGroup { dims } }

    /// Iterates over the sub-dimensions of the group.
    pub fn iter(&self) -> std::iter::Cloned<std::slice::Iter<ir::dim::Id>> {
        self.into_iter()
    }
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

impl<'a> IntoIterator for &'a DimGroup {
    type Item = ir::dim::Id;
    type IntoIter = std::iter::Cloned<std::slice::Iter<'a, ir::dim::Id>>;

    fn into_iter(self) -> Self::IntoIter { self.dims.iter().cloned() }
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
