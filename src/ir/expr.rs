use std::collections::HashSet;
use std::fmt;
use std::hash::BuildHasher;
use std::ops;
use std::sync::Arc;

use fxhash::{FxHashMap, FxHashSet};
use itertools::Itertools;
use log::{trace, warn};

use super::{AccessPattern, Function, IrDisplay, LogicalDimId, Parameter, Size, Type};

/// An index expression
#[derive(Debug, Clone)]
pub enum IndexExpr {
    /// The current position when iterating over a logical dimension
    LogicalDim(LogicalDimId),
    /// A kernel parameter.  Mainly used in `Sum` expressions.
    Parameter(Arc<Parameter>),
    /// A sum of expressions and a constant.
    ///
    /// This is used for the `p + r` expression in convolutions.
    Sum(i32, Vec<IndexExpr>),
}

impl<L> IrDisplay<L> for IndexExpr {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>, function: &Function<L>) -> fmt::Result {
        match self {
            IndexExpr::LogicalDim(lid) => write!(fmt, "{}", lid),
            IndexExpr::Parameter(p) => write!(fmt, "{}", p),
            IndexExpr::Sum(cst, exprs) => write!(
                fmt,
                "{} + {}",
                cst,
                exprs.iter().map(|e| e.display(function)).format(" + ")
            ),
        }
    }
}

impl IndexExpr {
    fn collect_logical_dims<S: BuildHasher>(
        &self,
        logical_dims: &mut HashSet<LogicalDimId, S>,
    ) {
        match self {
            &IndexExpr::LogicalDim(id) => {
                logical_dims.insert(id);
            }
            IndexExpr::Parameter(..) => (),
            IndexExpr::Sum(_, exprs) => {
                for expr in exprs {
                    expr.collect_logical_dims(logical_dims)
                }
            }
        }
    }

    pub fn as_logical_dim(&self) -> Option<LogicalDimId> {
        match self {
            IndexExpr::LogicalDim(id) => Some(*id),
            _ => None,
        }
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

impl IntoIndexExpr for LogicalDimId {
    fn into_index_expr(self) -> IndexExpr {
        IndexExpr::LogicalDim(self)
    }
}

impl IntoIndexExpr for i32 {
    fn into_index_expr(self) -> IndexExpr {
        IndexExpr::Sum(self, Vec::new())
    }
}

impl IntoIndexExpr for Arc<Parameter> {
    fn into_index_expr(self) -> IndexExpr {
        IndexExpr::Parameter(self)
    }
}

impl<Rhs: IntoIndexExpr> ops::Add<Rhs> for IndexExpr {
    type Output = IndexExpr;

    fn add(self, other: Rhs) -> Self::Output {
        let other = other.into_index_expr();

        match (self, other) {
            (IndexExpr::Sum(lhs_cst, mut lhs), IndexExpr::Sum(rhs_cst, rhs)) => {
                lhs.extend(rhs);
                IndexExpr::Sum(lhs_cst + rhs_cst, lhs)
            }
            (IndexExpr::Sum(cst, mut sum), other)
            | (other, IndexExpr::Sum(cst, mut sum)) => {
                sum.push(other);
                IndexExpr::Sum(cst, sum)
            }
            (lhs, rhs) => IndexExpr::Sum(0, vec![lhs, rhs]),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Access {
    id: AccessId,
    base: Arc<Parameter>,
    strides: Vec<(IndexExpr, Size)>,
}

impl<L> IrDisplay<L> for Access {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>, function: &Function<L>) -> fmt::Result {
        write!(
            fmt,
            "{}[{}]",
            self.base,
            self.strides.iter().format_with(" + ", |(expr, stride), f| {
                if stride.as_constant() == Some(1) {
                    f(&expr.display(function))
                } else {
                    f(&format_args!("{}*{}", expr.display(function), stride))
                }
            })
        )
    }
}

impl Access {
    pub fn id(&self) -> AccessId {
        self.id
    }

    fn t(&self) -> Type {
        self.base.t
    }

    pub fn base(&self) -> &Arc<Parameter> {
        &self.base
    }

    pub fn strides(&self) -> &[(IndexExpr, Size)] {
        &self.strides
    }

    pub fn logical_dims(&self) -> impl Iterator<Item = LogicalDimId> + '_ {
        let mut logical_dims = FxHashSet::default();
        for (expr, _) in &self.strides {
            expr.collect_logical_dims(&mut logical_dims);
        }

        logical_dims.into_iter()
    }

    pub fn access_pattern(&self, function: &Function<()>) -> AccessPattern {
        let elem_size = self.base.elem_t.unwrap().len_byte().unwrap();
        let mem_id = None;
        let mut dims = FxHashMap::default();
        for (expr, stride) in &self.strides {
            if let Some(id) = expr.as_logical_dim() {
                let logical_dim = function.logical_dim(id);
                // TODO: dimensions could be in incorrect order. Maybe.
                let mut stride = super::PartialSize::from(stride * elem_size);
                for did in logical_dim.dimensions() {
                    let dim = function.dim(did);

                    if dims.insert(did, stride.clone()).is_some() {
                        warn!(
                            "Unknown access pattern: multiple strides for dimension {}",
                            did
                        );
                        return AccessPattern::Unknown(mem_id);
                    }

                    stride *= dim.size();
                }
            } else {
                warn!(
                    "Unknown access pattern: complex index expression `{}` with stride `{}`",
                    expr.display(function), stride
                );
                return AccessPattern::Unknown(mem_id);
            }
        }
        let ap = AccessPattern::Tensor { mem_id, dims };
        trace!("Tensor access `{}` for `{}`", ap, self.display(function));
        ap
    }
}

#[derive(Debug, Clone, Default)]
pub struct Accesses {
    accesses: Vec<Access>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AccessId(usize);

impl fmt::Display for AccessId {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(fmt, "*{}", self.0)
    }
}

impl Accesses {
    pub fn add(
        &mut self,
        base: Arc<Parameter>,
        strides: Vec<(IndexExpr, Size)>,
    ) -> AccessId {
        assert!(base.elem_t.is_some());

        let id = AccessId(self.accesses.len());
        self.accesses.push(Access { id, base, strides });
        id
    }

    pub fn iter(&self) -> AccessesIter<'_> {
        AccessesIter {
            iter: self.accesses.iter().enumerate(),
        }
    }
}

impl ops::Index<AccessId> for Accesses {
    type Output = Access;

    fn index(&self, idx: AccessId) -> &Access {
        &self.accesses[idx.0]
    }
}

impl<'a> IntoIterator for &'a Accesses {
    type Item = (AccessId, &'a Access);

    type IntoIter = AccessesIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

pub struct AccessesIter<'a> {
    iter: std::iter::Enumerate<std::slice::Iter<'a, Access>>,
}

impl<'a> Iterator for AccessesIter<'a> {
    type Item = (AccessId, &'a Access);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(id, access)| (AccessId(id), access))
    }
}
