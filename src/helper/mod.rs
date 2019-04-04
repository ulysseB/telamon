//! Helper functions to build an IR instance.
mod builder;
mod operand;
mod signature;

pub mod tensor;

pub use self::builder::Builder;
pub use self::operand::{AutoOperand, Reduce, TmpArray};
pub use self::signature::Builder as SignatureBuilder;

use crate::ir;
use serde::{Deserialize, Serialize};
use std;
use utils::*;

/// A groups of dimensions that act as a single logical dimension.
#[derive(Clone)]
pub struct LogicalDim {
    logical_id: ir::LogicalDimId,
    real_ids: Vec<ir::DimId>,
    tiling_pattern: TilingPattern,
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
    fn ids(&self) -> Box<dyn Iterator<Item = ir::StmtId> + '_>;
}

impl<T> MetaStatement for T
where
    T: Into<ir::StmtId> + Copy,
{
    fn ids(&self) -> Box<dyn Iterator<Item = ir::StmtId> + '_> {
        Box::new(std::iter::once((*self).into()))
    }
}

impl MetaStatement for Option<LogicalDim> {
    fn ids(&self) -> Box<dyn Iterator<Item = ir::StmtId> + '_> {
        Box::new(self.iter().flat_map(|dim| dim.ids()))
    }
}

impl MetaStatement for LogicalDim {
    fn ids(&self) -> Box<dyn Iterator<Item = ir::StmtId> + '_> {
        Box::new(self.iter().map(|id| id.into()))
    }
}

/// Indicates how a logical dimension should be tiled.
///
/// In details, if we have dimension `d0` of size `size(d0)`, we create `tile_sizes.len()`
/// additional tiling dimensions. `tile_sizes` corresponds to the sizes each tiling dimension can
/// have. `tiling_factors` contrain the total tiling factors of `d0`: the product of the sizes of
/// the tiling dimensions must be in `tiling_factors`.
///
/// For example, we want to tile 2 times a dimension of size 128, we could have:
/// ```
/// let tiling_factors = [4, 8, 16, 32, 64];
/// let tile_sizes = [[2, 4, 8, 16, 32], [2, 4, 8, 16, 32]];
/// ```
/// Each tiling dimension can have size 32, but not both simultaneously because 1024 = 32x32 is not
/// in the tiling factors.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TilingPattern {
    tiling_factors: VecSet<u32>,
    tile_sizes: Vec<VecSet<u32>>,
}

impl TilingPattern {
    /// Creates a new fixed tiling pattern, with dimensions of the given sizes.
    pub fn new_fixed(dim_sizes: &[u32]) -> Self {
        let tiling_factor = dim_sizes.iter().product::<u32>();
        TilingPattern {
            tiling_factors: VecSet::new(vec![tiling_factor]),
            tile_sizes: dim_sizes.iter().map(|&s| VecSet::new(vec![s])).collect(),
        }
    }

    /// Infer a tiling pattern for a dimension whose size is a multiple of `gcd_size`.
    /// `max_tile_sizes` limits the maximal tile sizes for each tiling dimension.
    pub fn infer_pattern(gcd_size: u32, max_tile_sizes: &[u32]) -> Self {
        let multiples: VecSet<_> = (2..gcd_size)
            .filter(|x| (gcd_size % x) == 0)
            .take(16)
            .collect();
        let tile_sizes = max_tile_sizes
            .iter()
            .map(|max| {
                VecSet::new(multiples.iter().cloned().take_while(|x| x <= max).collect())
            })
            .collect();
        TilingPattern {
            tiling_factors: multiples,
            tile_sizes,
        }
    }
}

impl<'a> From<&'a [u32]> for TilingPattern {
    fn from(dim_sizes: &'a [u32]) -> Self {
        TilingPattern::new_fixed(dim_sizes)
    }
}

impl Default for TilingPattern {
    fn default() -> Self {
        TilingPattern {
            tiling_factors: VecSet::new(vec![1]),
            tile_sizes: vec![],
        }
    }
}
