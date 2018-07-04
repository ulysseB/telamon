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
#[derive(Clone)]
pub enum LogicalDim {
    /// A single concrete dimension.
    Simple(ir::dim::Id),
    /// Multiple dimensions forming a single logical once.
    Composite(ir::dim::LogicalId, Vec<ir::dim::Id>, Vec<u32>),
}

impl LogicalDim {
    pub fn iter<'a>(&'a self) -> Box<DoubleEndedIterator<Item=ir::dim::Id> + 'a> {
        self.into_iter()
    }
}

impl From<ir::dim::Id> for LogicalDim {
    fn from(id: ir::dim::Id) -> Self { LogicalDim::Simple(id) }
}

impl MetaDimension for LogicalDim {
    fn ids<'a>(&'a self) -> Box<DoubleEndedIterator<Item=ir::dim::Id> + 'a> {
        self.into_iter()
    }
}

impl std::ops::Index<usize> for LogicalDim {
    type Output = ir::dim::Id;

    fn index(&self, index: usize) -> &ir::dim::Id {
        match self {
            LogicalDim::Simple(id) if index == 0 => id,
            LogicalDim::Simple(_) => panic!("out of bounds index {}", index),
            LogicalDim::Composite(_, dims, _) => &dims[index],

        }
    }
}

impl<'a> IntoIterator for &'a LogicalDim {
    type Item = ir::dim::Id;
    type IntoIter = Box<DoubleEndedIterator<Item=ir::dim::Id> + 'a>;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            LogicalDim::Simple(dim) => Box::new(std::iter::once(*dim)),
            LogicalDim::Composite(_, dims, _) => Box::new(dims.iter().cloned()),
        }
    }
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
