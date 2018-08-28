//! Helper functions to build an IR instance.
mod builder;
mod operand;
mod signature;

pub mod tensor;

pub use self::builder::Builder;
pub use self::operand::{AutoOperand, Reduce, TmpArray};
pub use self::signature::Builder as SignatureBuilder;

use ir;
use std;

/// A logical dimension, possible composed of multiple nested dimensions.
pub trait MetaDimension {
    /// Returns the ids of the underlying dimensions.
    fn ids(&self) -> Box<DoubleEndedIterator<Item = ir::DimId> + '_>;
}

impl MetaDimension for ir::DimId {
    fn ids(&self) -> Box<DoubleEndedIterator<Item = ir::DimId> + '_> {
        Box::new(std::iter::once(*self))
    }
}

impl<T> MetaDimension for T
where
    T: std::borrow::Borrow<[ir::DimId]>,
{
    fn ids(&self) -> Box<DoubleEndedIterator<Item = ir::DimId> + '_> {
        Box::new(self.borrow().iter().cloned())
    }
}

/// A groups of dimensions that act as a single logical dimension.
#[derive(Clone)]
pub enum LogicalDim<'a> {
    /// A single concrete dimension.
    Simple(ir::DimId),
    /// Multiple dimensions forming a single logical once.
    Composite {
        id: ir::LogicalDimId,
        dims: Vec<ir::DimId>,
        size: ir::Size<'a>,
        tiling_factors: Vec<u32>,
        // TODO(strip-minig): remove this parameter once we enable unconstrained tile sizes.
        tile_sizes: Vec<u32>,
    },
}

impl<'b> LogicalDim<'b> {
    pub fn iter(&self) -> Box<DoubleEndedIterator<Item = ir::DimId> + '_> {
        self.into_iter()
    }
}

impl From<ir::DimId> for LogicalDim<'static> {
    fn from(id: ir::DimId) -> Self {
        LogicalDim::Simple(id)
    }
}

impl<'b> MetaDimension for LogicalDim<'b> {
    fn ids(&self) -> Box<DoubleEndedIterator<Item = ir::DimId> + '_> {
        self.into_iter()
    }
}

impl<'a> std::ops::Index<usize> for LogicalDim<'a> {
    type Output = ir::DimId;

    fn index(&self, index: usize) -> &ir::DimId {
        match self {
            LogicalDim::Simple(id) if index == 0 => id,
            LogicalDim::Simple(_) => panic!("out of bounds index {}", index),
            LogicalDim::Composite { dims, .. } => &dims[index],
        }
    }
}

impl<'a> IntoIterator for &'a LogicalDim<'a> {
    type Item = ir::DimId;
    type IntoIter = Box<DoubleEndedIterator<Item = ir::DimId> + 'a>;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            LogicalDim::Simple(dim) => Box::new(std::iter::once(*dim)),
            LogicalDim::Composite { dims, .. } => Box::new(dims.iter().cloned()),
        }
    }
}

/// A logical basic block, that can actually be implemented by multiple ones.
pub trait MetaStatement {
    /// Returns the ids on the underlying basic blocks.
    fn ids(&self) -> Box<Iterator<Item = ir::BBId> + '_>;
}

impl MetaStatement for ir::BBId {
    fn ids(&self) -> Box<Iterator<Item = ir::BBId> + '_> {
        Box::new(std::iter::once(*self))
    }
}

impl MetaStatement for ir::InstId {
    fn ids(&self) -> Box<Iterator<Item = ir::BBId> + '_> {
        Box::new(std::iter::once((*self).into()))
    }
}

impl<T> MetaStatement for T
where
    T: MetaDimension,
{
    fn ids(&self) -> Box<Iterator<Item = ir::BBId> + '_> {
        Box::new(MetaDimension::ids(self).map(|x| x.into()))
    }
}
