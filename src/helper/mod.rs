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
use utils::*;

/// A groups of dimensions that act as a single logical dimension.
#[derive(Clone)]
pub struct LogicalDim {
    logical_id: ir::LogicalDimId,
    real_ids: Vec<ir::DimId>,
    tile_sizes: Vec<VecSet<u32>>,
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
    fn ids(&self) -> Box<Iterator<Item = ir::StmtId> + '_>;
}

impl<T> MetaStatement for T
where
    T: Into<ir::StmtId> + Copy,
{
    fn ids(&self) -> Box<Iterator<Item = ir::StmtId> + '_> {
        Box::new(std::iter::once((*self).into()))
    }
}

impl MetaStatement for Option<LogicalDim> {
    fn ids(&self) -> Box<Iterator<Item = ir::StmtId> + '_> {
        Box::new(self.iter().flat_map(|dim| dim.ids()))
    }
}

impl MetaStatement for LogicalDim {
    fn ids(&self) -> Box<Iterator<Item = ir::StmtId> + '_> {
        Box::new(self.iter().map(|id| id.into()))
    }
}

/// Indicates how a dimension should be tiled.
#[derive(Clone, Debug)]
pub struct TilingPattern {
    tiling_factors: Vec<u32>,
    tile_sizes: Vec<VecSet<u32>>,
}

impl TilingPattern {
    /// Creates a new fixed tiling pattern, with dimensions of the given sizes.
    pub fn new_fixed(dim_sizes: &[u32]) -> Self {
        let tiling_factor = dim_sizes.iter().product::<u32>();
        TilingPattern {
            tiling_factors: vec![tiling_factor],
            tile_sizes: dim_sizes.iter().map(|&s| VecSet::new(vec![s])).collect(),
        }
    }

    /// Infer a tiling pattern for a dimension whose size is a multiple of `gcd_size`.
    /// `max_tile_sizes` limits the maximal tile sizes for each tiling dimension.
    pub fn infer_pattern(gcd_size: u32, max_tile_sizes: &[u32]) -> Self {
        let multiples: Vec<_> = (2..gcd_size)
            .filter(|x| (gcd_size % x) == 0)
            .take(16)
            .collect();
        let tile_sizes = max_tile_sizes
            .iter()
            .map(|max| {
                VecSet::new(multiples.iter().cloned().take_while(|x| x < max).collect())
            }).collect();
        TilingPattern {
            tiling_factors: multiples,
            tile_sizes,
        }
    }
}

impl Default for TilingPattern {
    fn default() -> Self {
        TilingPattern {
            tiling_factors: vec![1],
            tile_sizes: vec![],
        }
    }
}
