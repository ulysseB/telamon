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
    /// A computed size.
    //
    // TODO(bclement): Technically this supersedes Parameter...
    Size(Size),

    /// Tuple projection.
    Proj(Arc<TupleExpr>, usize),
}

#[derive(Debug, Clone)]
pub enum TupleExpr {
    /// Delinearization
    Delinearize(IndexExpr, Vec<Size>),
}

#[derive(Debug, Clone)]
pub enum IndexPredicate {
    InRange(IndexExpr, ops::Range<IndexExpr>),
    And(Vec<IndexPredicate>),
}

impl<L> IrDisplay<L> for IndexPredicate {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>, function: &Function<L>) -> fmt::Result {
        match self {
            IndexPredicate::InRange(expr, range) => write!(
                fmt,
                "{} in {}..{}",
                expr.display(function),
                range.start.display(function),
                range.end.display(function)
            ),
            IndexPredicate::And(preds) => write!(
                fmt,
                "{}",
                preds
                    .iter()
                    .format_with(" && ", |p, f| f(&p.display(function)))
            ),
        }
    }
}

impl IndexPredicate {
    fn collect_logical_dims<S: BuildHasher>(
        &self,
        logical_dims: &mut HashSet<LogicalDimId, S>,
    ) {
        match self {
            IndexPredicate::InRange(expr, range) => {
                expr.collect_logical_dims(logical_dims);
                range.start.collect_logical_dims(logical_dims);
                range.end.collect_logical_dims(logical_dims);
            }
            IndexPredicate::And(preds) => {
                for pred in preds {
                    pred.collect_logical_dims(logical_dims);
                }
            }
        }
    }
}

impl ops::BitAndAssign for IndexPredicate {
    fn bitand_assign(&mut self, other: Self) {
        match self {
            IndexPredicate::And(preds) => preds.push(other),
            _ => {
                let this = std::mem::replace(
                    self,
                    IndexPredicate::And(match other {
                        IndexPredicate::And(preds) => preds,
                        _ => vec![other],
                    }),
                );

                match self {
                    IndexPredicate::And(preds) => {
                        preds.push(this);
                    }
                    _ => unreachable!(),
                }
            }
        }
    }
}

impl ops::BitAnd for IndexPredicate {
    type Output = Self;

    fn bitand(mut self, other: Self) -> Self {
        self &= other;
        self
    }
}

impl<L> IrDisplay<L> for IndexExpr {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>, function: &Function<L>) -> fmt::Result {
        match self {
            IndexExpr::LogicalDim(lid) => write!(fmt, "{}", lid),
            IndexExpr::Parameter(p) => write!(fmt, "{}", p),
            IndexExpr::Size(size) => write!(fmt, "{}", size),
            IndexExpr::Sum(cst, exprs) => write!(
                fmt,
                "{} + {}",
                cst,
                exprs.iter().map(|e| e.display(function)).format(" + ")
            ),
            IndexExpr::Proj(tuple, idx) => {
                write!(fmt, "{}.{}", tuple.display(function), idx)
            }
        }
    }
}

impl<L> IrDisplay<L> for TupleExpr {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>, function: &Function<L>) -> fmt::Result {
        match self {
            TupleExpr::Delinearize(expr, sizes) => write!(
                fmt,
                "({} :: ({}))",
                expr.display(function),
                sizes.iter().format(", ")
            ),
        }
    }
}

impl TupleExpr {
    fn collect_logical_dims<S: BuildHasher>(
        &self,
        logical_dims: &mut HashSet<LogicalDimId, S>,
    ) {
        match self {
            TupleExpr::Delinearize(expr, _) => expr.collect_logical_dims(logical_dims),
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
            IndexExpr::Size(..) => (),
            IndexExpr::Sum(_, exprs) => {
                for expr in exprs {
                    expr.collect_logical_dims(logical_dims)
                }
            }
            IndexExpr::Proj(tuple, _) => tuple.collect_logical_dims(logical_dims),
        }
    }

    pub fn logical_dims(&self) -> impl Iterator<Item = LogicalDimId> + '_ {
        let mut logical_dims = FxHashSet::default();
        self.collect_logical_dims(&mut logical_dims);
        logical_dims.into_iter()
    }

    pub fn as_logical_dim(&self) -> Option<LogicalDimId> {
        match self {
            IndexExpr::LogicalDim(id) => Some(*id),
            _ => None,
        }
    }

    pub fn delinearize(self, sizes: Vec<Size>) -> Vec<Self> {
        assert!(!sizes.is_empty());

        if sizes.len() == 1 {
            return vec![self];
        }

        let len = sizes.len();
        let delinearize = Arc::new(TupleExpr::Delinearize(self, sizes));
        (0..len)
            .map(|idx| IndexExpr::Proj(delinearize.clone(), idx))
            .collect()
    }

    pub fn in_range<T: IntoIndexExpr>(self, range: ops::Range<T>) -> IndexPredicate {
        IndexPredicate::InRange(
            self,
            range.start.into_index_expr()..range.end.into_index_expr(),
        )
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

impl IntoIndexExpr for Size {
    fn into_index_expr(self) -> IndexExpr {
        IndexExpr::Size(self)
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

impl<Rhs> ops::Add<Rhs> for &'_ IndexExpr
where
    IndexExpr: ops::Add<Rhs>,
{
    type Output = <IndexExpr as ops::Add<Rhs>>::Output;

    fn add(self, other: Rhs) -> Self::Output {
        self.clone() + other
    }
}

#[derive(Debug, Clone)]
pub struct Access {
    id: AccessId,
    base: Arc<Parameter>,
    strides: Vec<(IndexExpr, Size)>,
    predicate: Option<IndexPredicate>,
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

    pub fn predicate(&self) -> Option<&IndexPredicate> {
        self.predicate.as_ref()
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
        let mut unknown_dims = FxHashSet::default();
        for (expr, stride) in &self.strides {
            if let Some(id) = expr.as_logical_dim() {
                let logical_dim = function.logical_dim(id);
                let mut stride = super::PartialSize::from(stride * elem_size);
                for did in logical_dim.dimensions() {
                    let dim = function.dim(did);

                    if unknown_dims.contains(&did)
                        || dims.insert(did, stride.clone()).is_some()
                    {
                        warn!(
                            "Unknown access pattern for {}: multiple strides for dimension {}",
                            self.display(function), did
                        );

                        unknown_dims.insert(did);
                        dims.remove(&did);
                        // return AccessPattern::Unknown(mem_id);
                    }

                    stride *= dim.size();
                }
            } else {
                warn!(
                    "Unknown access pattern for {}: complex index expression `{}` with stride `{}`",
                    self.display(function), expr.display(function), stride
                );

                for lid in expr.logical_dims() {
                    for did in function.logical_dim(lid).dimensions() {
                        dims.remove(&did);
                        unknown_dims.insert(did);
                    }
                }

                // return AccessPattern::Unknown(mem_id);
            }
        }

        if dims.is_empty() {
            trace!("Unknown access for `{}`", self.display(function));
            AccessPattern::Unknown(mem_id)
        } else {
            let ap = AccessPattern::Tensor { mem_id, dims };
            trace!("Tensor access `{}` for `{}`", ap, self.display(function));
            ap
        }
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
        predicate: Option<IndexPredicate>,
    ) -> AccessId {
        assert!(base.elem_t.is_some());

        let id = AccessId(self.accesses.len());
        self.accesses.push(Access {
            id,
            base,
            strides,
            predicate,
        });
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
