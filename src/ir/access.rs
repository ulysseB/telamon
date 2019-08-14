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

    pub fn sizes(&self) -> &[Size] {
        &self.sizes
    }
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
pub struct UnpackedDimId(usize, usize);

impl UnpackedDimId {
    pub fn unpack_index(self) -> usize {
        self.1
    }
}

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
    // Marker for an unchecked expr (must not generate predicate)
    Unchecked(Box<IndexExpr>),
}

impl IndexExpr {
    pub fn unchecked(self) -> IndexExpr {
        IndexExpr::Unchecked(Box::new(self))
    }
}

pub trait IntoIndexExpr {
    fn into_index_expr(self) -> IndexExpr;
}

impl IntoIndexExpr for IndexExpr {
    fn into_index_expr(self) -> IndexExpr {
        self
    }
}

impl IntoIndexExpr for &'_ IndexExpr {
    fn into_index_expr(self) -> IndexExpr {
        self.clone()
    }
}

impl IntoIndexExpr for LogicalDimId {
    fn into_index_expr(self) -> IndexExpr {
        IndexExpr::LogicalDim(self)
    }
}

impl IntoIndexExpr for UnpackedDimId {
    fn into_index_expr(self) -> IndexExpr {
        IndexExpr::Unpack(self)
    }
}

impl IntoIndexExpr for i32 {
    fn into_index_expr(self) -> IndexExpr {
        IndexExpr::Constant(self)
    }
}

impl IntoIndexExpr for Arc<Parameter> {
    fn into_index_expr(self) -> IndexExpr {
        IndexExpr::Parameter(self)
    }
}

impl IntoIndexExpr for Parameter {
    fn into_index_expr(self) -> IndexExpr {
        Arc::new(self).into_index_expr()
    }
}

impl<Rhs: IntoIndexExpr> ops::Add<Rhs> for IndexExpr {
    type Output = IndexExpr;

    fn add(self, other: Rhs) -> Self::Output {
        let other = other.into_index_expr();

        match (self, other) {
            (IndexExpr::Constant(lhs), IndexExpr::Constant(rhs)) => {
                IndexExpr::Constant(lhs + rhs)
            }
            (IndexExpr::Sum(mut lhs), IndexExpr::Sum(rhs)) => {
                lhs.extend(rhs);
                IndexExpr::Sum(lhs)
            }
            (IndexExpr::Sum(mut sum), other) | (other, IndexExpr::Sum(mut sum)) => {
                sum.push(other);
                IndexExpr::Sum(sum)
            }
            (lhs, rhs) => IndexExpr::Sum(vec![lhs, rhs]),
        }
    }
}

impl<Rhs: IntoIndexExpr> ops::Add<Rhs> for &'_ IndexExpr {
    type Output = IndexExpr;

    fn add(self, other: Rhs) -> Self::Output {
        self.clone() + other
    }
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexDimension {
    pub expr: IndexExpr,
    pub stride: Size,
    pub min: Option<Size>,
    pub max: Option<Size>,
}

/// A memory access
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Access {
    base: Arc<Parameter>,
    strides: Vec<IndexDimension>,
}

impl Access {
    pub fn new(base: Arc<Parameter>, strides: Vec<IndexDimension>) -> Self {
        Access { base, strides }
    }

    pub fn t(&self) -> Type {
        self.base.t
    }

    pub fn base(&self) -> &Arc<Parameter> {
        &self.base
    }

    pub fn index_dims(&self) -> &[IndexDimension] {
        &self.strides
    }

    /// Iterate (recursively) over all the indices used in this access
    pub fn indices(&self) -> impl Iterator<Item = &'_ IndexExpr> + '_ {
        let mut indices = Vec::new();
        for access_dim in &self.strides {
            access_dim.expr.collect_indices(&mut indices);
        }

        indices.into_iter()
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Accesses {
    accesses: Vec<Access>,
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
#[repr(transparent)]
pub struct AccessId(usize);

impl Accesses {
    pub fn add(&mut self, access: Access) -> AccessId {
        let id = AccessId(self.accesses.len());
        self.accesses.push(access);
        id
    }

    pub fn iter(&self) -> impl Iterator<Item = (AccessId, &Access)> {
        self.accesses
            .iter()
            .enumerate()
            .map(|(id, access)| (AccessId(id), access))
    }
}

impl ops::Index<AccessId> for Accesses {
    type Output = Access;

    fn index(&self, idx: AccessId) -> &Access {
        &self.accesses[idx.0]
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
