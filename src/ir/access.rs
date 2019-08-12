use std::ops;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::{LogicalDimId, Parameter, Size, Type};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct PackedDimId(usize);

/// A packed dimension.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackedDim {
    logical_dim: LogicalDimId,
    sizes: Vec<Size>,
}

impl PackedDim {
    pub fn logical_dim(&self) -> LogicalDimId {
        self.logical_dim
    }
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
pub struct UnpackedDimId(usize, usize);

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PackedDims {
    packed_dims: Vec<PackedDim>,
}

impl PackedDims {
    pub fn unpack(
        &mut self,
        logical_dim: LogicalDimId,
        sizes: Vec<Size>,
    ) -> Vec<UnpackedDimId> {
        let packed_id = self.packed_dims.len();
        let unpacked_ids = (0..sizes.len())
            .map(|i| UnpackedDimId(packed_id, i))
            .collect();
        self.packed_dims.push(PackedDim { logical_dim, sizes });
        unpacked_ids
    }
}

impl ops::Index<PackedDimId> for PackedDims {
    type Output = PackedDim;

    fn index(&self, idx: PackedDimId) -> &PackedDim {
        &self.packed_dims[idx.0]
    }
}

impl ops::Index<UnpackedDimId> for PackedDims {
    type Output = PackedDim;

    fn index(&self, idx: UnpackedDimId) -> &PackedDim {
        &self.packed_dims[idx.0]
    }
}

/// An index expression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexExpr {
    /// The current position when iterating over a logical dimension
    LogicalDim(LogicalDimId),
    /// The current position on an unpacked dimension
    Unpack(UnpackedDimId),
    /// A kernel parameter.  Mainly used in `Sum` expressions.
    Parameter(Arc<Parameter>),
    /// A constant value.  Mainly used in `Sum` expressions.
    Constant(i32),
    /// A sum of expressions.
    ///
    /// This is used for the `p + r` expression in convolutions.
    Sum(Vec<IndexExpr>),
}

impl IndexExpr {
    fn collect_indices<'a>(&'a self, indices: &mut Vec<&'a Self>) {
        indices.push(self);

        match self {
            IndexExpr::Sum(others) => {
                for other in others {
                    other.collect_indices(indices);
                }
            }
            _ => (),
        }
    }
}

/// A memory access
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Access {
    base: Arc<Parameter>,
    strides: Vec<(IndexExpr, Size)>,
}

impl Access {
    pub fn new(base: Arc<Parameter>, strides: Vec<(IndexExpr, Size)>) -> Self {
        Access { base, strides }
    }

    pub fn t(&self) -> Type {
        self.base.t
    }

    pub fn base(&self) -> &Arc<Parameter> {
        &self.base
    }

    pub fn strides(&self) -> impl Iterator<Item = &'_ (IndexExpr, Size)> + '_ {
        self.strides.iter()
    }

    /// Iterate (recursively) over all the indices used in this access
    pub fn indices(&self) -> impl Iterator<Item = &'_ IndexExpr> + '_ {
        let mut indices = Vec::new();
        for (expr, _) in &self.strides {
            expr.collect_indices(&mut indices);
        }

        indices.into_iter()
    }
}

pub enum Predicate {
    /// A range predicate.  True iff the corresponding index expression is in min..=max
    Range {
        index: IndexExpr,
        min: Option<Size>,
        max: Option<Size>,
    },
}
