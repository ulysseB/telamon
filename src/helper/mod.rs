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

/// A groups of dimensions that act as a single logical dimension.
#[derive(Clone)]
pub struct LogicalDim {
    logical_id: ir::LogicalDimId,
    real_ids: Vec<ir::DimId>,
    // TODO(strip-minig): remove this parameter once we enable unconstrained tile sizes.
    tile_sizes: Vec<u32>,
}

impl LogicalDim {
    /// Iterates on the reals IDs, from the outermost to the innermost.
    pub fn iter(&self) -> impl Iterator<Item = ir::DimId> + '_ {
        self.into_iter()
    }

    /// Returns the id of the logical dimension.
    pub fn id(&self) -> ir::LogicalDimId {
        self.logical_id
    }
}

impl std::ops::Index<usize> for LogicalDim {
    type Output = ir::DimId;

    fn index(&self, index: usize) -> &ir::DimId {
        &self.real_ids[index]
    }
}

impl<'a> IntoIterator for &'a LogicalDim {
    type Item = ir::DimId;
    type IntoIter = std::iter::Cloned<std::slice::Iter<'a, ir::DimId>>;

    fn into_iter(self) -> Self::IntoIter {
        self.real_ids.iter().cloned()
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

impl MetaStatement for LogicalDim {
    fn ids(&self) -> Box<Iterator<Item = ir::BBId> + '_> {
        Box::new(self.iter().map(|id| id.into()))
    }
}
