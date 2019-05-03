use std::any::Any;
use std::cell::Cell;
use std::cmp;
use std::fmt;
use std::rc::{Rc, Weak};

use std::hash::{BuildHasher, Hash};

use crate::memo::{Memoized, MemoizedHash};

trait Op: Copy + Eq + 'static {}

trait Apply<O, Rhs = Self> {
    type Output;

    fn apply(&self, op: O, other: &Rhs) -> Self::Output;
}

trait ApplyNum<O, N> {
    type Output;

    fn apply_num(&self, op: O, other: &N) -> Self::Output;
}

trait ApplySym<O, S> {
    type Output;

    fn apply_sym(&self, op: O, other: &S) -> Self::Output;
}

trait RevApplyNum<O, N> {
    type Output;

    fn rev_apply_num(&self, op: O, other: &N) -> Self::Output;
}

trait RevApplySym<O, S> {
    type Output;

    fn rev_apply_sym(&self, op: O, other: &S) -> Self::Output;
}

trait ApplyAssign<O, Rhs = Self> {
    fn apply_assign(&mut self, op: O, other: &Rhs);
}

trait ApplyAssignNum<O, N> {
    fn apply_assign_num(&mut self, op: O, other: &N);
}

trait ApplyAssignSym<O, S> {
    fn apply_assign_sym(&mut self, op: O, other: &S);
}

trait NumExpr {
    type Numeric;

    fn as_numeric(&self) -> Option<&Self::Numeric>;

    fn as_numeric_mut(&mut self) -> Option<&mut Self::Numeric>;
}

trait AsExpr<O> {
    type Expr;

    fn as_expr(&self) -> Option<&Self::Expr> {
        None
    }

    fn as_expr_mut(&mut self) -> Option<&mut Self::Expr> {
        None
    }
}

macro_rules! forward_op {
    (impl<$gen:ident $(, $p:ident)*> $imp:ident<$op:ident, $rhs:ident, Output = $u:ty>, $method:ident for $t:ty $(where $($where:tt)*)?) => {
        impl<$op, $rhs, $gen $(, $p)*> $imp<$op, $rhs> for $t
        where
            T: $imp<$op, $rhs>,
            $($($where)*)?
        {
            type Output = $u;

            fn $method(&self, op: $op, other: &$rhs) -> $u {
                self.as_ref().$method(op, other).into()
            }
        }
    };

    (impl<$gen:ident $(, $p:ident)*> $imp:ident<$op:ident, $rhs:ident>, $method:ident for $t:ty $(where $($where:tt)*)?) => {
        impl<$op, $rhs, $gen $(, $p)*> $imp<$op, $rhs> for $t
        where
            T: $imp<$op, $rhs>,
            $($($where)*)?
        {
            fn $method(&mut self, op: $op, other: &$rhs) {
                self.as_mut().$method(op, other);
            }
        }
    };
}

macro_rules! forward_all {
    (impl<$gen:ident $(, $p:ident)*> ...Num<$op:ident, $rhs:ident, Output = $u:ty> for $t:ty $(where $($where:tt)*)?) => {
        forward_op!(impl<$gen $(, $p)*> ApplyNum<$op, $rhs, Output = $u>, apply_num for $t $(where $($where)*)?);
        forward_op!(impl<$gen $(, $p)*> RevApplyNum<$op, $rhs, Output = $u>, rev_apply_num for $t $(where $($where)*)?);
        forward_op!(impl<$gen $(, $p)*> ApplyAssignNum<$op, $rhs>, apply_assign_num for $t $(where $($where)*)?);
    };

    (impl<$gen:ident $(, $p:ident)*> ...Sym<$op:ident, $rhs:ident, Output = $u:ty> for $t:ty $(where $($where:tt)*)?) => {
        forward_op!(impl<$gen $(, $p)*> ApplySym<$op, $rhs, Output = $u>, apply_sym for $t $(where $($where)*)?);
        forward_op!(impl<$gen $(, $p)*> RevApplySym<$op, $rhs, Output = $u>, rev_apply_sym for $t $(where $($where)*)?);
        forward_op!(impl<$gen $(, $p)*> ApplyAssignSym<$op, $rhs>, apply_assign_sym for $t $(where $($where)*)?);
    };

    (impl<$gen:ident $(, $p:ident)*> ...<$op:ident, $rhs:ident, Output = $u:ty> for $t:ty $(where $($where:tt)*)?) => {
        forward_op!(impl<$gen $(, $p)*> Apply<$op, $rhs, Output = $u>, apply for $t $(where $($where)*)?);
        forward_op!(impl<$gen $(, $p)*> ApplyAssign<$op, $rhs>, apply_assign for $t $(where $($where)*)?);

        forward_all!(impl<$gen $(, $p)*> ...Num<$op, $rhs, Output = $u> for $t $(where $($where)*)?);
        forward_all!(impl<$gen $(, $p)*> ...Sym<$op, $rhs, Output = $u> for $t $(where $($where)*)?);
    };
}

forward_all!(impl<T, M> ...<O, Rhs, Output = Memoized<T::Output, M>> for Memoized<T, M> where M: Default);
forward_all!(impl<T, S> ...<O, Rhs, Output = MemoizedHash<T::Output, S>> for MemoizedHash<T, S> where S: BuildHasher + Default);

#[derive(Debug, Clone, Hash, Eq, PartialEq, Ord, PartialOrd)]
struct Ops<S> {
    inner: S,
}

impl<S> fmt::Display for Ops<S>
where
    S: fmt::Display,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.inner, fmt)
    }
}

impl<S> AsRef<S> for Ops<S> {
    fn as_ref(&self) -> &S {
        &self.inner
    }
}

impl<S> AsMut<S> for Ops<S> {
    fn as_mut(&mut self) -> &mut S {
        &mut self.inner
    }
}

impl<S> From<S> for Ops<S> {
    fn from(inner: S) -> Self {
        Ops { inner }
    }
}

impl<O, N, E, T> ApplyAssignNum<O, N> for Ops<T>
where
    T: AsExpr<O, Expr = E> + NumExpr<Numeric = N> + ApplyAssignNum<O, N>,
    E: ApplyAssignNum<O, N>,
    N: ApplyAssign<O>,
{
    fn apply_assign_num(&mut self, op: O, other: &N) {
        if let Some(self_num) = self.inner.as_numeric_mut() {
            self_num.apply_assign(op, other);
        } else if let Some(self_expr) = self.inner.as_expr_mut() {
            self_expr.apply_assign_num(op, other);
        } else {
            self.inner.apply_assign_num(op, other);
        }
    }
}

impl<O, N, E, T> ApplyNum<O, N> for Ops<T>
where
    T: AsExpr<O, Expr = E> + NumExpr<Numeric = N> + ApplyNum<O, N>,
    E: ApplyNum<O, N, Output = E> + Into<T::Output>,
    N: Apply<O, N, Output = N> + Into<T::Output>,
{
    type Output = Ops<T::Output>;

    fn apply_num(&self, op: O, other: &N) -> Self::Output {
        if let Some(self_num) = self.inner.as_numeric() {
            Ops::from(self_num.apply(op, other).into())
        } else if let Some(self_expr) = self.inner.as_expr() {
            Ops::from(self_expr.apply_num(op, other).into())
        } else {
            Ops::from(self.inner.apply_num(op, other))
        }
    }
}

impl<O, N, T> RevApplyNum<O, N> for Ops<T>
where
    T: AsExpr<O> + NumExpr<Numeric = N> + RevApplyNum<O, N>,
    T::Expr: RevApplyNum<O, N, Output = T::Expr> + Into<T::Output>,
    N: Apply<O, N, Output = N> + Into<T::Output>,
{
    type Output = Ops<T::Output>;

    fn rev_apply_num(&self, op: O, other: &N) -> Self::Output {
        if let Some(self_num) = self.inner.as_numeric() {
            Ops::from(other.apply(op, self_num).into())
        } else if let Some(self_expr) = self.inner.as_expr() {
            Ops::from(self_expr.rev_apply_num(op, other).into())
        } else {
            Ops::from(self.inner.rev_apply_num(op, other))
        }
    }
}

forward_op!(impl<T> ApplySym<Op, Rhs, Output = Ops<T::Output>>, apply_sym for Ops<T>);
forward_op!(impl<T> RevApplySym<Op, Rhs, Output = Ops<T::Output>>, rev_apply_sym for Ops<T>);
forward_op!(impl<T> ApplyAssignSym<Op, Rhs>, apply_assign_sym for Ops<T>);

impl<O, N, T, Rhs> ApplyAssign<O, Ops<Rhs>> for Ops<T>
where
    T: AsExpr<O> + NumExpr<Numeric = N> + ApplyAssignNum<O, N> + ApplyAssign<O, Rhs>,
    Rhs: AsExpr<O> + NumExpr<Numeric = N> + RevApplyNum<O, N, Output = T>,
    T::Expr: ApplyAssignNum<O, N> + ApplyAssign<O, Rhs::Expr> + ApplyAssignSym<O, Rhs>,
    Rhs::Expr: RevApplyNum<O, N, Output = T> + RevApplySym<O, T, Output = T>,
    N: ApplyAssign<O>,
{
    fn apply_assign(&mut self, op: O, other: &Ops<Rhs>) {
        if let Some(other_num) = other.inner.as_numeric() {
            self.apply_assign_num(op, other_num);
        } else if let Some(other_expr) = other.inner.as_expr() {
            if let Some(self_num) = self.inner.as_numeric() {
                self.inner = other_expr.rev_apply_num(op, self_num);
            } else if let Some(self_expr) = self.inner.as_expr_mut() {
                self_expr.apply_assign(op, other_expr);
            } else {
                self.inner = other_expr.rev_apply_sym(op, &self.inner);
            }
        } else if let Some(self_num) = self.inner.as_numeric() {
            *self = other.inner.rev_apply_num(op, self_num);
        } else if let Some(self_expr) = self.inner.as_expr_mut() {
            self_expr.apply_assign_sym(op, &other.inner);
        } else {
            self.inner.apply_assign(op, &other.inner);
        }
    }
}

impl<O, N, T, Rhs, Out> Apply<O, Ops<Rhs>> for Ops<T>
where
    T: AsExpr<O>
        + NumExpr<Numeric = N>
        + ApplyNum<O, N, Output = Out>
        + Apply<O, Rhs, Output = Out>,
    Rhs: AsExpr<O> + NumExpr<Numeric = N> + RevApplyNum<O, N, Output = Out>,
    T::Expr: ApplyNum<O, N, Output = Out>
        + Apply<O, Rhs::Expr, Output = Out>
        + ApplySym<O, Rhs, Output = Out>,
    Rhs::Expr: RevApplyNum<O, N, Output = Out> + RevApplySym<O, T, Output = Out>,
    N: Apply<O, N, Output = N> + Into<Out>,
{
    type Output = Ops<Out>;

    fn apply(&self, op: O, other: &Ops<Rhs>) -> Self::Output {
        if let Some(self_num) = self.inner.as_numeric() {
            if let Some(other_num) = other.inner.as_numeric() {
                Ops::from(self_num.apply(op, other_num).into())
            } else if let Some(other_expr) = other.inner.as_expr() {
                other_expr.rev_apply_num(op, self_num).into()
            } else {
                other.inner.rev_apply_num(op, self_num).into()
            }
        } else if let Some(self_expr) = self.inner.as_expr() {
            if let Some(other_num) = other.inner.as_numeric() {
                self_expr.apply_num(op, other_num).into()
            } else if let Some(other_expr) = other.inner.as_expr() {
                self_expr.apply(op, other_expr).into()
            } else {
                self_expr.apply_sym(op, &other.inner).into()
            }
        } else if let Some(other_num) = other.inner.as_numeric() {
            self.inner.apply_num(op, other_num).into()
        } else if let Some(other_expr) = other.inner.as_expr() {
            other_expr.rev_apply_sym(op, &self.inner).into()
        } else {
            self.inner.apply(op, &other.inner).into()
        }
    }
}

trait CmpOp<T>: Sized {
    fn merge(&self, other: Self) -> Self {
        other
    }

    fn op_cmp(&self, lhs: &T, rhs: &T) -> cmp::Ordering;

    fn is_greater(&self, lhs: &T, rhs: &T) -> bool {
        self.op_cmp(lhs, rhs) == cmp::Ordering::Greater
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
struct MaxOp;

impl<T> CmpOp<T> for MaxOp
where
    T: Ord,
{
    fn op_cmp(&self, lhs: &T, rhs: &T) -> cmp::Ordering {
        lhs.cmp(rhs)
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
struct MinOp;

impl<T> CmpOp<T> for MinOp
where
    T: Ord,
{
    fn op_cmp(&self, lhs: &T, rhs: &T) -> cmp::Ordering {
        rhs.cmp(lhs)
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
struct F64MaxOp;

impl CmpOp<f64> for F64MaxOp {
    fn op_cmp(&self, lhs: &f64, rhs: &f64) -> cmp::Ordering {
        if let Some(ordering) = lhs.partial_cmp(rhs) {
            ordering
        } else if rhs.is_nan() {
            cmp::Ordering::Greater
        } else {
            cmp::Ordering::Less
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
struct F64MinOp;

impl CmpOp<f64> for F64MinOp {
    fn op_cmp(&self, lhs: &f64, rhs: &f64) -> cmp::Ordering {
        if let Some(ordering) = rhs.partial_cmp(lhs) {
            ordering
        } else if rhs.is_nan() {
            cmp::Ordering::Greater
        } else {
            cmp::Ordering::Less
        }
    }
}

#[derive(Debug, Clone, Hash, Eq, PartialEq, Ord, PartialOrd)]
struct Maximum<O, N, S> {
    op: O,
    constant: N,
    values: Vec<S>,
}

impl<O, N, S> Maximum<O, N, S>
where
    O: CmpOp<N> + Copy,
    S: ToRange2<O, N>,
{
    fn retain_values(&mut self, constant: &N) {
        self.values.retain({
            let op = self.op;
            move |sym| op.is_greater(&sym.max_value(op), constant)
        });
    }

    fn filtered_values(&'_ self, constant: &'_ N) -> impl Iterator<Item = &'_ S> + '_ {
        self.values.iter().filter({
            let op = self.op;
            move |sym| op.is_greater(&sym.max_value(op), constant)
        })
    }
}

struct Range<N> {
    min: N,
    max: N,
}

trait ToRange2<O, N> {
    fn min_value(&self, op: O) -> N;

    fn max_value(&self, op: O) -> N;
}

macro_rules! commutative_apply {
    (impl<$($p:ident),*> $rev_impl:ident<$op:ty, $rhs:ty>, $rev_method:ident for $t:ty where $impl:ident, $method:ident) => {
        impl<$($p),*> $rev_impl<$op, $rhs> for $t
        where
            Self: $impl<$op, $rhs>,
        {
            type Output = <Self as $impl<$op, $rhs>>::Output;

            fn $rev_method(&self, op: $op, other: &$rhs) -> Self::Output {
                self.$method(op, other)
            }
        }
    };
}

commutative_apply!(impl<O, N, S> RevApplyNum<O, N>, rev_apply_num for Maximum<O, N, S> where ApplyNum, apply_num);
commutative_apply!(impl<O, N, S> RevApplySym<O, N>, rev_apply_sym for Maximum<O, N, S> where ApplySym, apply_sym);

impl<O, N, S> ApplyNum<O, N> for Maximum<O, N, S>
where
    O: CmpOp<N> + Copy,
    S: ToRange2<O, N> + Clone,
    N: Clone,
{
    type Output = Maximum<O, N, S>;

    fn apply_num(&self, op: O, other: &N) -> Self {
        if op.is_greater(other, &self.constant) {
            Maximum {
                op,
                constant: other.clone(),
                values: self
                    .values
                    .iter()
                    .filter(|sym| op.is_greater(&sym.max_value(op), other))
                    .cloned()
                    .collect(),
            }
        } else {
            self.clone()
        }
    }
}

impl<O, N, S> ApplyAssignNum<O, N> for Maximum<O, N, S>
where
    O: CmpOp<N> + Copy,
    S: ToRange2<O, N>,
    N: Clone,
{
    fn apply_assign_num(&mut self, op: O, other: &N) {
        if op.is_greater(other, &self.constant) {
            self.constant = other.clone();
            self.retain_values(other);
        }
    }
}

impl<O, N, S> ApplyAssignSym<O, S> for Maximum<O, N, S>
where
    O: CmpOp<N> + Copy,
    S: ToRange2<O, N> + Clone,
{
    fn apply_assign_sym(&mut self, op: O, other: &S) {
        if op.is_greater(&other.max_value(op), &self.constant) {
            let min_value = other.min_value(op);
            if op.is_greater(&min_value, &self.constant) {
                self.retain_values(&min_value);
                self.constant = min_value;
            }

            // TODO: sorted insert
            self.values.push(other.clone());
        }
    }
}

impl<O, N, S> ApplySym<O, S> for Maximum<O, N, S>
where
    O: CmpOp<N> + Copy,
    S: ToRange2<O, N> + Clone,
    N: Clone,
{
    type Output = Maximum<O, N, S>;

    fn apply_sym(&self, op: O, other: &S) -> Self {
        if op.is_greater(&other.max_value(op), &self.constant) {
            let min_value = other.min_value(op);
            let (mut values, constant) = if op.is_greater(&min_value, &self.constant) {
                (
                    self.filtered_values(&min_value).cloned().collect(),
                    min_value,
                )
            } else {
                (self.values.clone(), self.constant.clone())
            };

            // TODO sorted insert
            values.push(other.clone());
            Maximum {
                op,
                constant,
                values,
            }
        } else {
            self.clone()
        }
    }
}

impl<O, N, S> ApplyAssign<O> for Maximum<O, N, S>
where
    O: CmpOp<N> + Copy,
    S: ToRange2<O, N> + Clone,
    N: Clone,
{
    fn apply_assign(&mut self, op: O, other: &Maximum<O, N, S>) {
        match op.op_cmp(&self.constant, &other.constant) {
            cmp::Ordering::Greater => self
                .values
                .extend(other.filtered_values(&self.constant).cloned()),
            cmp::Ordering::Less => {
                self.retain_values(&other.constant);
                self.values.extend(other.values.iter().cloned());
                self.constant = other.constant.clone();
            }
            cmp::Ordering::Equal => self.values.extend(other.values.iter().cloned()),
        }
    }
}

impl<O, N, S> Apply<O> for Maximum<O, N, S>
where
    O: CmpOp<N> + Copy,
    S: ToRange2<O, N> + Clone,
    N: Clone,
{
    type Output = Maximum<O, N, S>;

    fn apply(&self, op: O, other: &Maximum<O, N, S>) -> Maximum<O, N, S> {
        // TODO: could be more efficient.
        let mut result = self.clone();
        result.apply_assign(op, other);
        result
    }
}

#[derive(Debug, Clone)]
struct MaxExpr(Vec<FloatInner>);

#[derive(Debug, Clone)]
struct MinExpr(Vec<FloatInner>);

#[derive(Debug, Clone)]
enum FloatInner {
    Max(MaxExpr),
    Min(MinExpr),
}

impl AsExpr<MaxOp> for FloatInner {
    type Expr = MaxExpr;

    fn as_expr(&self) -> Option<&MaxExpr> {
        match self {
            FloatInner::Max(expr) => Some(expr),
            _ => None,
        }
    }

    fn as_expr_mut(&mut self) -> Option<&mut MaxExpr> {
        match self {
            FloatInner::Max(expr) => Some(expr),
            _ => None,
        }
    }
}

/*
impl OpAssign<MaxOp> for FloatInner {
    fn new_op_assign(&mut self, MaxOp: MaxOp, other: &FloatInner) {
        *self = FloatInner::Max(MaxExpr(vec![self.clone(), other.clone()]));
    }
}
*/

/*
macro_rules! forward_apply_symbolic {
    (impl<$t:ident $(, $p:ident)*> Apply<Output = $Co:ty> for $Ct:ty) => {
        impl<O, $t, Rhs $(, $p)*> Apply<O, Symbolic<Rhs>> for $Ct
        where
            O: Op,
            $t: Apply<O, Symbolic<Rhs>>,
        {
            type Output = $Co;

            fn apply(&self, op: O, other: Symbolic<Rhs>) -> $Co {
                self.as_ref().apply(op, other).into()
            }
        }
    };
}

macro_rules! forward_apply_inner {
    (impl<$t:ident, $rhs:ident $(, $p:ident)*> Apply<&$Crhs:ty, Output = $Co:ty> for $Ct:ty) => {
        impl<'a, O, $t, $rhs $(, $p)*> Apply<O, &'a $Crhs> for $Ct
        where
            O: Op,
            $t: Apply<O, &'a $rhs>,
        {
            type Output = $Co;

            fn apply(&self, op: O, other: &'a $Crhs) -> $Co {
                self.as_ref().apply(op, other.as_ref()).into()
            }
        }
    };
}

macro_rules! forward_apply_assign_inner {
    (impl<$t:ident, $rhs:ident $(, $p:ident)*>
     ApplyAssign<$op:ty, &$Frhs:ty> for $Ft:ty) => {
        impl<'a, $t, $rhs $(, $p)*> ApplyAssign<$op, &'a $Frhs> for $Ft
        where
            $op: Op,
            $t: ApplyAssign<$op, &'a $rhs>,
        {
            fn apply_assign(&mut self, op: $op, other: &'a $Frhs) {
                <$Ft as ToMut<$t>>::to_mut(self).apply_assign(
                    op, <$Frhs as AsRef<$rhs>>::as_ref(other));
            }
        }
    };
}

macro_rules! forward_apply_assign {
    (impl<$t:ident, $rhs:ident $(, $p:ident)*>
     ApplyAssign<$op:ty, $Grhs:ty> for $Ft:ty) => {
        impl<$t, $rhs $(, $p)*> ApplyAssign<$op, $Grhs> for $Ft
        where
            $op: Op,
            $t: ApplyAssign<$op, $Grhs>,
        {
            fn apply_assign(&mut self, op: $op, other: $Grhs) {
                <$Ft as ToMut<$t>>::to_mut(self).apply_assign(op, other);
            }
        }
    };
}

forward_apply_symbolic!(impl<T> Apply<Output = Rc<T::Output>> for Rc<T>);
forward_apply_inner!(impl<T, Rhs> Apply<&Rc<Rhs>, Output = Rc<T::Output>> for Rc<T>);

impl<O, T, Rhs> ApplyAssign<O, Symbolic<Rhs>> for Rc<T>
where
    O: Op,
    T: ApplyAssign<O, Symbolic<Rhs>> + Apply<O, Symbolic<Rhs>, Output = T>,
{
    fn apply_assign(&mut self, op: O, other: Symbolic<Rhs>) {
        if Rc::get_mut(self).is_some() {
            Rc::get_mut(self).unwrap().apply_assign(op, other);
        } else {
            *self = self.apply(op, other);
        }
    }
}

impl<'a, O, T, Rhs> ApplyAssign<O, &'a Rc<Rhs>> for Rc<T>
where
    O: Op,
    T: ApplyAssign<O, &'a Rhs> + Apply<O, &'a Rhs, Output = T>,
{
    fn apply_assign(&mut self, op: O, other: &'a Rc<Rhs>) {
        if Rc::get_mut(self).is_some() {
            Rc::get_mut(self).unwrap().apply_assign(op, other.as_ref());
        } else {
            *self = self.apply(op, other);
        }
    }
}

// Memoized hash

impl<O, T, Rhs, S> Apply<O, Symbolic<Rhs>> for MemoizedHash<T, S>
where
    O: Op,
    T: Apply<O, Symbolic<Rhs>> + Hash,
    T::Output: Hash,
    S: BuildHasher + Default,
{
    type Output = MemoizedHash<T::Output, S>;

    fn apply(&self, op: O, other: Symbolic<Rhs>) -> MemoizedHash<T::Output, S> {
        self.as_ref().apply(op, other).into()
    }
}

forward_apply_assign!(
    impl<T, Rhs, O, S> ApplyAssign<O, Symbolic<Rhs>> for MemoizedHash<T, S>);

impl<'a, O, T, Rhs, S> Apply<O, &'a MemoizedHash<Rhs, S>> for MemoizedHash<T, S>
where
    O: Op,
    T: Apply<O, &'a Rhs> + Hash,
    T::Output: Hash,
    Rhs: Hash,
    S: BuildHasher + Default,
{
    type Output = MemoizedHash<T::Output, S>;

    fn apply(
        &self,
        op: O,
        other: &'a MemoizedHash<Rhs, S>,
    ) -> MemoizedHash<T::Output, S> {
        self.as_ref().apply(op, other.as_ref()).into()
    }
}

forward_apply_assign_inner!(
    impl<T, Rhs, O, S> ApplyAssign<O, &MemoizedHash<Rhs>> for MemoizedHash<T, S>);

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
struct MaxOp;

impl Op for MaxOp {}

#[derive(Debug, Clone)]
struct Max<N, S> {
    numeric: N,
    symbolic: Vec<S>,
}

impl<'a, N, S> ApplyAssign<MaxOp, &'a Max<N, S>> for Max<N, S>
where
    S: Clone,
{
    fn apply_assign(&mut self, MaxOp: MaxOp, other: &'a Max<N, S>) {
        self.symbolic.extend(other.symbolic.iter().cloned());
    }
}

impl<'a, N, S> ApplyAssign<MaxOp, Symbolic<&'a S>> for Max<N, S>
where
    S: Clone,
{
    fn apply_assign(&mut self, MaxOp: MaxOp, other: Symbolic<&'a S>) {
        self.symbolic.push(other.0.clone());
    }
}

impl<'a, N, S> Apply<MaxOp, &'a Max<N, S>> for Max<N, S>
where
    N: Clone,
    S: Clone,
{
    type Output = Max<N, S>;

    fn apply(&self, MaxOp: MaxOp, other: &'a Max<N, S>) -> Max<N, S> {
        let mut this = self.clone();
        this.apply_assign(MaxOp, other);
        this
    }
}

impl<'a, N, S> Apply<MaxOp, Symbolic<&'a S>> for Max<N, S>
where
    N: Clone,
    S: Clone,
{
    type Output = Max<N, S>;

    fn apply(&self, MaxOp: MaxOp, other: Symbolic<&'a S>) -> Max<N, S> {
        let mut this = self.clone();
        this.apply_assign(MaxOp, other);
        this
    }
}

/*
enum FloatInner<N> {
    /// A numeric value
    Numeric(N),
    /// Max
    Max(MemoizedBinop<Supremum<N, Float<N>, MaxOp>>),
    /// Min
    Min(MemoizedBinop<Supremum<N, Float<N>, MinOp>>),
    // TODO: Pair<N, Ratio<A>>? (question is how to memo)
    // probably MemoizedBinop<Ratio<A>>
    Ratio(Pair<N, MemoizedBinop<Reduction<Float<N>, MulOp, DivOp>>>),
    Diff(Pair<N, MemoizedBinop<Reduction<Float<N>, AddOp, SubOp>>>),

    /// maybe Ratio should have the DivCeil
    /// no because we can have many DivCeil
    /// -> need to take into account when computing stuff on the Ratio
    /// (oh shit :p)
    /// note when we create:
    ///  - if the lcm always a multiple, divide everywhere, then divide by `1`
    ///  - if furthermore there is a single value in the lcm, return a ratio instead
    /// ceil(lcm(x..)/n)
    DivCeil(N, Vec<(N, Ratio<A>)>, u32),
}

struct Float<N> {
    inner: Expr<N, MemoizedBinop<MemoizedRange<N, MemoizedHash<FloatInner<N>>>>>,
}

enum IntInner<N, A> {
    // TODO: things in the lcm are only Mul
    Lcm(MemoizedBinop<Supremum<N, Int<N, A>, LcmOp>>),
    Mul(Pair<N, Ratio<A>>),
}

struct Int<N> {
    inner: Expr<N, MemoizedBinop<MemoizedRange<N, MemoizedHash<IntInner<N>>>>>,
}
*/

// (3*x) + 1
// |-> Expr::Symbolic(3*x) + Expr::Numeric(1)
// |-> Expr::Symbolic(3*x) + Expr::Symbolic(1.into())

// GOGOGO

/*
struct OpMemo<Rhs, Output> {
    op: Box<dyn Any>,
    other: Weak<Rhs>,
    output: Weak<Output>,
}

struct MemoizedOp<T, Rhs = Self, Output = Self> {
    value: T,
    memo: Cell<Option<OpMemo<Rhs, Output>>>,
}

impl<T, Rhs, Output> PartialEq for MemoizedOp<T, Rhs, Output>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.value.eq(&other.value)
    }
}

impl<T, Rhs, Output> Eq for MemoizedOp<T, Rhs, Output> where T: Eq {}

struct WeakAny(Box<dyn Any + 'static>);

impl WeakAny {
    fn upgrade<T>(&self) -> Option<Rc<T>>
    where
        T: Any,
    {
        self.0.downcast_ref::<Weak<T>>().and_then(Weak::upgrade)
    }
}

struct AnyMemo {
    op: Box<dyn Any>,
    other: WeakAny,
    result: WeakAny,
}

struct MemoizedAny<T> {
    value: T,
    memo: Cell<Option<AnyMemo>>,
}

struct MemoizedRc<T> {
    inner: Rc<MemoizedAny<T>>,
}

impl<T> PartialEq for MemoizedAny<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.value.eq(&other.value)
    }
}

impl<'a, O, T, Rhs> ApplyAssign<O, &'a Rc<Rhs>> for MemoizedRc<T>
where
    O: Op,
    T: Eq + ApplyAssign<O, &'a Rhs> + 'static,
    Rhs: Eq + 'static,
    for<'b> &'b T: Apply<O, &'a Rhs, Output = T>,
{
    fn apply_assign(&mut self, op: O, other: &'a Rc<Rhs>) {
        if let Some(memo) = self.inner.memo.take() {
            if memo.op.downcast_ref::<O>() == Some(&op) {
                if let Some(prev_other) = memo.other.upgrade::<Rhs>() {
                    if prev_other == *other {
                        if let Some(prev_result) = memo.result.upgrade::<MemoizedAny<T>>()
                        {
                            *self = MemoizedRc { inner: prev_result };
                            return;
                        }
                    }
                }
            }
        }

        if Rc::get_mut(&mut self.inner).is_some() {
            Rc::get_mut(&mut self.inner)
                .unwrap()
                .value
                .apply_assign(op, &*other);
        } else {
            let result = Rc::new(MemoizedAny {
                value: self.inner.value.apply(op, other),
                memo: Cell::new(None),
            });

            self.inner.memo.set(Some(AnyMemo {
                op: Box::new(op),
                other: WeakAny(Box::new(Rc::downgrade(other))),
                result: WeakAny(Box::new(Rc::downgrade(&result))),
            }));

            *self = MemoizedRc { inner: result };
        }
    }
}

#[derive(Debug, Clone)]
struct Max<N, S> {
    numeric: N,
    symbolic: Vec<S>,
}

// Diff(Expr<N, MemoizedRc<Diff<Float<N>>, FloatInner<N>>)
// Diff(Expr<N, MemoizedRc<Diff<Float<N>>>>>),

// TODO: do I want to memo in addition to `SymbolicFloat + SymbolicFloat`
// `Max + SymbolicFloat`?
// or juste `Max + Max` ?
// ok for max it doesn't make sense.
//
// for Diff.
// Diff(Expr<N, Rc<..>>)
// -> N + N
// & Rc<..> + Rc<..> [memoized]
// or: Rc<..> + SymbolicFloat [memoized]
// argh

enum FloatInner<N> {
    Max(MemoizedRc<Max<N, SymbolicFloat<N>>>),
}

enum Float<N> {
    Numeric(N),
    Symbolic(FloatInner<N>),
}

type SymbolicFloat<N> = MemoizedRc<Float<N>>;

trait AsNumeric {
    type Numeric;

    fn as_numeric(&self) -> Option<&Self::Numeric>;

    fn as_numeric_mut(&mut self) -> Option<&mut Self::Numeric>;

    fn into_numeric(self) -> Result<Self::Numeric, Self>
    where
        Self: Sized;
}

impl<N, S> AsNumeric for Max<N, S> {
    type Numeric = N;

    fn as_numeric(&self) -> Option<&N> {
        if self.symbolic.is_empty() {
            Some(&self.numeric)
        } else {
            None
        }
    }

    fn as_numeric_mut(&mut self) -> Option<&mut N> {
        if self.symbolic.is_empty() {
            Some(&mut self.numeric)
        } else {
            None
        }
    }

    fn into_numeric(self) -> Result<N, Self> {
        if self.symbolic.is_empty() {
            Ok(self.numeric)
        } else {
            Err(self)
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
struct MaxOp;

impl Op for MaxOp {}

pub enum Expr<N, S> {
    Numeric(N),
    Symbolic(S),
}

trait Expression<N, S = Self> {
    fn into_expr(self) -> Expr<N, S>
    where
        Self: Sized;
}

impl<'a, O, N, S> ApplyAssign<O, &'a Expr<N, S>> for Expr<N, S>
where
    O: Op,
    N: Default + ApplyAssign<O, &'a N>,
    for<'b> Numeric<&'b N>: Apply<O, &'a S, Output = S>,
    S: Expression<N> + ApplyAssign<O, &'a N> + ApplyAssign<O, &'a S>,
{
    fn apply_assign(&mut self, op: O, other: &'a Expr<N, S>) {
        match self {
            Expr::Numeric(lhs) => match other {
                Expr::Numeric(rhs) => lhs.apply_assign(op, rhs),
                Expr::Symbolic(rhs) => *self = Numeric(&*lhs).apply(op, rhs).into_expr(),
            },
            Expr::Symbolic(lhs) => {
                match other {
                    Expr::Numeric(rhs) => lhs.apply_assign(op, rhs),
                    Expr::Symbolic(rhs) => lhs.apply_assign(op, rhs),
                }

                // Lower numeric values if possible.  This can happen when simplifications occur
                // during the computation (e.g. `[x] *= 0` but also `[x + 1] -= x`)
                if let Expr::Symbolic(lhs) =
                    std::mem::replace(self, Expr::Numeric(Default::default()))
                {
                    *self = lhs.into_expr();
                } else {
                    unreachable!()
                }
            }
        }
    }
}

/*
impl<'a, 'b, N, S> Apply<MaxOp, &'a Max<N, S>> for Numeric<&'a N>
where
    N: Clone,
    S: Clone,
{
    type Output = Max<N, S>;

    fn apply(self, MaxOp: MaxOp, other: &'a Max<N, S>) -> Max<N, S> {
        // TODO: do not clone if not needed to do anything etc.
        let mut max = other.clone();
        max.apply_assign(MaxOp, self);
        max
    }
}
*/

impl<'a, N, S> ApplyAssign<MaxOp, Numeric<&'a N>> for Max<N, S> {
    fn apply_assign(&mut self, MaxOp: MaxOp, other: Numeric<&'a N>) {
        // self.numeric.max_assign(other)
    }
}

impl<'a, N, S> ApplyAssign<MaxOp, &'a Max<N, S>> for Max<N, S>
where
    S: Clone,
{
    fn apply_assign(&mut self, MaxOp: MaxOp, other: &'a Max<N, S>) {
        // self.numeric.max_assign(&other.numeric);
        self.symbolic.extend(other.symbolic.iter().cloned())
    }
}

impl<'a, N, S> ApplyAssign<MaxOp, Symbolic<&'a S>> for Max<N, S>
where
    S: Clone,
{
    fn apply_assign(&mut self, MaxOp: MaxOp, other: Symbolic<&'a S>) {
        // TODO: check + maybe update
        self.symbolic.push(other.0.clone())
    }
}

/*
#[cfg(test)]
mod lol {
    #[derive(Debug, Copy, Clone)]
    struct AddOp;

    #[derive(Debug, Clone, PartialEq)]
    struct Diff<S> {
        positive: Vec<S>,
        negative: Vec<S>,
    }

    trait Z: Sized {
        fn zero() -> Self;

        fn is_zero(&self) -> bool;
    }

    impl<S> Z for Diff<S> {
        fn zero() -> Self {
            Diff {
                positive: Vec::new(),
                negative: Vec::new(),
            }
        }

        fn is_zero(&self) -> bool {
            self.positive.is_empty() && self.negative.is_empty()
        }
    }

    struct WeakBinop<Lhs, Rhs = Lhs> {
        op: Box<dyn Any>,
        rhs: Weak<Rhs>,
        result: Weak<Lhs>,
    }

    struct DiffExpr<N, S>(Expr<N, Rc<MemoizedBinop<Diff<S>>>>);

    enum FloatInner<N> {
        Max(Rc<MemoizedBinop<Max<N, Float<N>>>>),
        Min(Rc<MemoizedBinop<Min<N, Float<N>>>>),
        Ratio(Expr<N, Rc<MemoizedBinop<Ratio<Float<N>>>>>),
        Diff(Expr<N, Rc<MemoizedBinop<Diff<Float<N>>>>>),
    }

    // ApplyAssign<Op, S> for Truc where S: AsExpr<Truc>,
    // Truc: ApplyAssign<Op, S>

    enum Float<N> {
        Numeric(N),
        // v-- SymbolicFloat<N> = Rc<...>
        Symbolic(Rc<MemoizedBinop<MemoizedRange<MemoizedHash<FloatInner<N>>>>>),
    }

    // Expr<N, Rc<Memoized<Diff<S>, WeakBinop<
    // Diff(N, Rc<Memoized<Diff<S>, WeakBinop<Diff<S>, Rhs>>,>)

    //impl<'a, O, T, Rhs> ApplyAssign<O, &'a Rc<Rhs>> for Rc<U>
    //where U = Memoized<T, WeakBinop<U, Rhs>>

    impl<'a, O, T, Rhs> ApplyAssign<O, &'a Rc<Rhs>> for Rc<MemoizedBinop<T, Rhs>>
    where
        T: ApplyAssign<O, &'a Rc<Rhs>>,
        for<'b> &'b T: Apply<O, &'a Rc<Rhs>>,
    {
        fn apply_assign(&mut self, op: O, other: &'a Rc<Rhs>) {
            if let Some(weak) = Memoized::get_memo(&*self) {
                if let Some(weak_op) = weak.downcast_ref::<O>() {
                    if op == *weak_op {
                        if let Some(old_rhs) = weak.rhs.upgrade() {
                            if old_rhs == *other {
                                if let Some(result) = weak.result.upgrade() {
                                    *self = result;
                                    return;
                                }
                            }
                        }
                    }
                }
            }

            if Rc::get_mut(self).is_some() {
                Rc::get_mut(self).unwrap().to_mut().apply_assign(op, other);
            } else {
                let result = self.apply(op, other);

                Memoized::set_memo(
                    &*self,
                    WeakBinop {
                        op: Box::new(op),
                        rhs: Rc::downgrade(other),
                        result: Rc::downgrade(&result),
                    },
                );

                *self = result;
            }
        }
    }

    impl<'a, S> ApplyAssign<AddOp, &'a Diff<S>> for Diff<S>
    where
        S: Clone,
    {
        fn apply_assign(&mut self, AddOp: AddOp, other: &'a Diff<S>) {
            // Can do it more smart
            self.positive.extend(other.positive.iter().cloned());
            self.negative.extend(other.negative.iter().cloned());
        }
    }

    struct Expr<N, S> {
        numeric: N,
        symbolic: S,
    }

    type ExprRef<'a, N, S> = Expr<&'a N, &'a S>;
    type ExprMut<'a, N, S> = Expr<&'a mut N, &'a mut S>;

    impl<'a, 'b, N, S> ApplyAssign<AddOp, ExprRef<'a, N, S>> for ExprMut<'b, N, S>
    where
        N: ApplyAssign<AddOp, &'a N>,
        S: ApplyAssign<AddOp, &'a S>,
    {
        fn apply_assign(&mut self, op: AddOp, other: ExprRef<'a, N, S>) {
            self.numeric.apply_assign(op, other.numeric);
            self.symbolic.apply_assign(op, other.symbolic);
        }
    }
}
*/
*/
*/
