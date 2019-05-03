use std::convert::{AsMut, AsRef};
use std::hash::{BuildHasher, Hash};
use std::iter;
use std::ops::{Add, AddAssign};

use std::rc::Rc;

pub mod memo;

use memo::{Memoized, MemoizedHash};

/// An operator `O`.
///
/// Note that `Rhs` is `Self` by default, but this is not mandatory.  For example, a type
/// representing a sum of elements of type `T` could implement `Apply<Add, T>` in addition to
/// `Apply<Add>`.
pub trait Apply<O, Rhs = Self> {
    /// The resulting type after applying the operator.  This is usually `Self`.
    type Output;

    /// Computes `self op other`.
    fn apply(&self, op: O, other: &Rhs) -> Self::Output;
}

/// The reverse of an operator `O`.
///
/// This is used to work around limitations of Rust's trait system.
pub trait RevApply<O, Lhs = Self> {
    /// The resulting type after applying the operator.  This is usually `Lhs`.
    type Output;

    /// Computes `other op self`.
    fn rev_apply(&self, op: O, other: &Lhs) -> Self::Output;
}

/// The assignment operator `op=`
pub trait ApplyAssign<O, Rhs = Self> {
    /// Computes `self op= other`
    fn apply_assign(&mut self, op: O, other: &Rhs);
}

/// The reverse assignment operator `op=`
pub trait RevApplyAssign<O, Lhs = Self> {
    /// Computes `other op= self`
    fn rev_apply_assign(&self, op: O, other: &mut Lhs);
}

pub trait AsNum: Sized {
    type Num: Sized;

    fn from_numeric(numeric: Self::Num) -> Self;

    fn as_numeric(&self) -> Option<&Self::Num>;

    fn as_numeric_mut(&mut self) -> Option<&mut Self::Num>;
}

pub trait ApplyNum<O>: AsNum {
    fn apply_num(&self, op: O, other: &Self::Num) -> Self;
}

pub trait RevApplyNum<O, Lhs = Self>: AsNum {
    fn rev_apply_num(&self, op: O, other: &Self::Num) -> Lhs;
}

pub trait AsExpr<O> {
    type Expr;

    fn from_expr(expr: Self::Expr) -> Self;

    fn as_expr(&self) -> Option<&Self::Expr>;

    fn as_expr_mut(&mut self) -> Option<&mut Self::Expr>;
}

macro_rules! forward_as {
    (impl<$($gen:ident),*> $impl:ident$(<$($args:ident),+>)?, $con:ident, $from:ident, $as_ref:ident, $as_mut:ident for $t:ty [$rec:ident] $(where $($where:tt)*)?) => {
        impl<$($gen),*> $impl$(<$($args),+>)? for $t
        where
            $rec: $impl$(<$($args),+>)?,
            $($($where)*)?
        {
            type $con = $rec::$con;

            #[inline]
            fn $from(value: Self::$con) -> Self {
                Self::from($rec::$from(value))
            }

            #[inline]
            fn $as_ref(&self) -> Option<&Self::$con> {
                self.as_ref().$as_ref()
            }

            #[inline]
            fn $as_mut(&mut self) -> Option<&mut Self::$con> {
                self.as_mut().$as_mut()
            }
        }
    };
}

macro_rules! forward_apply {
    (impl<$($gen:ident),+> $imp:ident<$op:ident, $other:ident, Output = $out:ty>, $method:ident, $rev_method:ident for $t:ty $(where $($where:tt)*)?) => {
        impl<$($gen),+> $imp<$op, $other> for $t
        $(where $($where)*)?
        {
            type Output = $out;

            fn $method(&self, op: $op, other: &$other) -> $out {
                other.$rev_method(op, self.as_ref()).into()
            }
        }
    };
}

macro_rules! forward_apply_assign {
    (impl<$($gen:ident),+> $imp:ident<$op:ident, $other:ident>, $method:ident, $rev_method:ident for $t:ty $(where $($where:tt)*)?) => {
        impl<$($gen),+> $imp<$op, $other> for $t
        $(where $($where)*)?
        {
            fn $method(&mut self, op: $op, other: &$other) {
                other.$rev_method(op, self.as_mut())
            }
        }
    };
}

macro_rules! forward_rev_apply_assign {
    (impl<$($gen:ident),+> $imp:ident<$op:ident, $other:ident>, $method:ident, $rev_method:ident for $t:ty $(where $($where:tt)*)?) => {
        impl<$($gen),+> $imp<$op, $other> for $t
        $(where $($where)*)?
        {
            fn $method(&self, op: $op, other: &mut $other) {
                other.$rev_method(op, self.as_ref())
            }
        }
    };
}

macro_rules! forward_all {
    (impl<$($gen:ident),+> for $t:ty [$rec:ident] $(where $($where:tt)*)?) => {
        forward_as!(
            impl<$($gen),+> AsNum, Num, from_numeric, as_numeric, as_numeric_mut for $t [$rec] $(where $($where)*)?);
        forward_as!(
            impl<$($gen, )+ __OP> AsExpr<__OP>, Expr, from_expr, as_expr, as_expr_mut for $t [$rec] $(where $($where)*)?);

        forward_apply!(
            impl<$($gen),+, __OP, __RHS> Apply<__OP, __RHS, Output = Self>, apply, rev_apply for $t where __RHS: RevApply<__OP, $rec, Output = $rec>, $($($where)*)?);
        forward_apply!(
            impl<$($gen),+, __OP, __LHS> RevApply<__OP, __LHS, Output = __LHS>, rev_apply, apply for $t where __LHS: Apply<__OP, $rec, Output = __LHS>, $($($where)*)?);

        forward_apply_assign!(
            impl<$($gen),+, __OP, __RHS> ApplyAssign<__OP, __RHS>, apply_assign, rev_apply_assign for $t where __RHS: RevApplyAssign<__OP, $rec>, $($($where)*)?);
        forward_rev_apply_assign!(
            impl<$($gen),+, __OP, __LHS> RevApplyAssign<__OP, __LHS>, rev_apply_assign, apply_assign for $t where __LHS: ApplyAssign<__OP, $rec>, $($($where)*)?);
    }
}

forward_all!(impl<T, M> for Memoized<T, M> [T] where M: Default);
forward_all!(impl<T, S> for MemoizedHash<T, S> [T] where S: BuildHasher + Default);

impl<T> AsNum for Rc<T>
where
    T: AsNum + Clone,
{
    type Num = T::Num;

    fn from_numeric(numeric: Self::Num) -> Self {
        Rc::new(T::from_numeric(numeric))
    }

    fn as_numeric(&self) -> Option<&Self::Num> {
        self.as_ref().as_numeric()
    }

    fn as_numeric_mut(&mut self) -> Option<&mut Self::Num> {
        if Rc::get_mut(self).is_some() {
            Rc::get_mut(self).unwrap().as_numeric_mut()
        } else if self.as_numeric().is_some() {
            Rc::make_mut(self).as_numeric_mut()
        } else {
            None
        }
    }
}

impl<T, O> AsExpr<O> for Rc<T>
where
    T: AsExpr<O> + Clone,
{
    type Expr = T::Expr;

    fn from_expr(expr: Self::Expr) -> Self {
        Rc::new(T::from_expr(expr))
    }

    fn as_expr(&self) -> Option<&Self::Expr> {
        self.as_ref().as_expr()
    }

    fn as_expr_mut(&mut self) -> Option<&mut Self::Expr> {
        if Rc::get_mut(self).is_some() {
            Rc::get_mut(self).unwrap().as_expr_mut()
        } else if self.as_expr().is_some() {
            Rc::make_mut(self).as_expr_mut()
        } else {
            None
        }
    }
}

impl<T, O> ApplyNum<O> for Rc<T>
where
    T: ApplyNum<O> + Clone,
{
    fn apply_num(&self, op: O, other: &Self::Num) -> Self {
        self.as_ref().apply_num(op, other).into()
    }
}

impl<T, O, U> RevApplyNum<O, U> for Rc<T>
where
    T: RevApplyNum<O, U> + Clone,
{
    fn rev_apply_num(&self, op: O, other: &Self::Num) -> U {
        self.as_ref().rev_apply_num(op, other)
    }
}

impl<T, O, Rhs> Apply<O, Rhs> for Rc<T>
where
    Rhs: RevApply<O, T, Output = T>,
{
    type Output = Self;

    fn apply(&self, op: O, other: &Rhs) -> Self {
        other.rev_apply(op, self.as_ref()).into()
    }
}

impl<T, O, Lhs> RevApply<O, Lhs> for Rc<T>
where
    Lhs: Apply<O, T, Output = Lhs>,
{
    type Output = Lhs;

    fn rev_apply(&self, op: O, other: &Lhs) -> Lhs {
        other.apply(op, self.as_ref()).into()
    }
}

#[derive(Debug, Clone, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct Ops<T> {
    pub inner: T,
}

impl<T> From<T> for Ops<T> {
    fn from(inner: T) -> Self {
        Ops { inner }
    }
}

impl<T> AsRef<T> for Ops<T> {
    fn as_ref(&self) -> &T {
        &self.inner
    }
}

impl<T> AsMut<T> for Ops<T> {
    fn as_mut(&mut self) -> &mut T {
        &mut self.inner
    }
}

forward_as!(impl<T> AsNum, Num, from_numeric, as_numeric, as_numeric_mut for Ops<T> [T]);
forward_as!(impl<T, O> AsExpr<O>, Expr, from_expr, as_expr, as_expr_mut for Ops<T> [T]);

impl<O, N, T, E> ApplyNum<O> for Ops<T>
where
    T: ApplyNum<O, Num = N> + AsExpr<O, Expr = E>,
    N: Apply<O, Output = N> + ApplyAssign<O>,
    E: ApplyNum<O, Num = N>,
{
    fn apply_num(&self, op: O, other: &Self::Num) -> Self {
        if let Some(self_num) = self.as_numeric() {
            Self::from_numeric(self_num.apply(op, other))
        } else if let Some(self_expr) = self.as_expr() {
            Self::from_expr(self_expr.apply_num(op, other))
        } else {
            Self::from(self.inner.apply_num(op, other))
        }
    }
}

impl<O, N, E, T /*, U*/> RevApplyNum<O /*, U*/> for Ops<T>
where
    N: Apply<O, Output = N>,
    T: AsNum<Num = N> + AsExpr<O, Expr = E> + RevApplyNum<O /*, U */>,
    // U: AsNum<Num = N> + AsExpr<O>,
    E: RevApplyNum<O, /* U::Expr, */ Num = N>,
{
    fn rev_apply_num(&self, op: O, other: &Self::Num) -> Self {
        if let Some(self_num) = self.as_numeric() {
            Self::from_numeric(other.apply(op, self_num))
        } else if let Some(self_expr) = self.as_expr() {
            // T::apply_num(other, op, self)
            Self::from_expr(self_expr.rev_apply_num(op, other))
        } else {
            // T::apply_num(other, op, self)
            Self::from(self.inner.rev_apply_num(op, other))
        }
    }
}

impl<O, N, E, T, Rhs> Apply<O, Rhs> for Ops<T>
where
    T: AsNum<Num = N> + AsExpr<O, Expr = E>,
    E: Apply<O, Rhs, Output = E>,
    Rhs: RevApplyNum<O, /* T, */ Num = N> + RevApply<O, T, Output = T> + Into<Self>,
{
    type Output = Self;

    fn apply(&self, op: O, other: &Rhs) -> Self {
        if let Some(self_num) = self.as_numeric() {
            other.rev_apply_num(op, self_num).into()
        } else if let Some(self_expr) = self.as_expr() {
            Self::from_expr(self_expr.apply(op, other))
        } else {
            Self::from(other.rev_apply(op, &self.inner))
        }
    }
}

impl<O, N, E, T, Lhs> RevApply<O, Lhs> for Ops<T>
where
    T: AsNum<Num = N> + AsExpr<O, Expr = E>,
    Lhs: ApplyNum<O, Num = N> + Apply<O, E, Output = Lhs> + Apply<O, T, Output = Lhs>,
{
    type Output = Lhs;

    fn rev_apply(&self, op: O, other: &Lhs) -> Lhs {
        if let Some(self_num) = self.as_numeric() {
            other.apply_num(op, self_num)
        } else if let Some(self_expr) = self.as_expr() {
            other.apply(op, self_expr)
        } else {
            other.apply(op, &self.inner)
        }
    }
}

pub struct AddOp;

#[derive(Debug, Clone)]
pub struct Sum<N, S> {
    constant: N,
    values: Vec<S>,
}

impl<N, S> AsNum for Sum<N, S> {
    type Num = N;

    fn from_numeric(constant: N) -> Self {
        Sum {
            constant,
            values: Vec::new(),
        }
    }

    fn as_numeric(&self) -> Option<&N> {
        if self.values.is_empty() {
            Some(&self.constant)
        } else {
            None
        }
    }

    fn as_numeric_mut(&mut self) -> Option<&mut N> {
        if self.values.is_empty() {
            Some(&mut self.constant)
        } else {
            None
        }
    }
}

impl<N, S> ApplyNum<AddOp> for Sum<N, S>
where
    N: Apply<AddOp, Output = N>,
    S: Clone,
{
    fn apply_num(&self, AddOp: AddOp, other: &N) -> Self {
        Sum {
            constant: self.constant.apply(AddOp, other),
            values: self.values.clone(),
        }
    }
}

impl<N, S, T> RevApplyNum<AddOp, T> for Sum<N, S>
where
    N: Apply<AddOp, Output = N>,
    S: Clone,
    Self: Into<T>,
{
    fn rev_apply_num(&self, AddOp: AddOp, other: &N) -> T {
        self.apply_num(AddOp, other).into()
    }
}

impl<N, S> Apply<AddOp> for Sum<N, S>
where
    N: Apply<AddOp, Output = N>,
    S: Clone,
{
    type Output = Sum<N, S>;

    fn apply(&self, AddOp: AddOp, other: &Self) -> Self {
        Sum {
            constant: self.constant.apply(AddOp, &other.constant),
            values: self
                .values
                .iter()
                .chain(other.values.iter())
                .cloned()
                .collect(),
        }
    }
}

impl<N, S> Apply<AddOp, S> for Sum<N, S>
where
    N: Clone,
    S: Clone,
{
    type Output = Sum<N, S>;

    fn apply(&self, AddOp: AddOp, other: &S) -> Self {
        Sum {
            constant: self.constant.clone(),
            values: self
                .values
                .iter()
                .cloned()
                .chain(iter::once(other.clone()))
                .collect(),
        }
    }
}

impl<N, S, T> RevApply<AddOp, T> for Sum<N, S>
where
    T: Apply<AddOp, Self, Output = T>,
{
    type Output = T;

    fn rev_apply(&self, AddOp: AddOp, other: &T) -> T {
        other.apply(AddOp, self)
    }
}

#[derive(Debug)]
pub struct Clovis;

impl Clone for Clovis {
    fn clone(&self) -> Self {
        println!("Cloning clovis");
        Clovis
    }
}

#[derive(Debug, Clone)]
pub enum FloatInner<N> {
    Numeric(N),
    Atom(Clovis),
    Diff(Sum<N, Float<N>>),
}

impl<N> AsNum for FloatInner<N> {
    type Num = N;

    fn from_numeric(numeric: Self::Num) -> Self {
        FloatInner::Numeric(numeric)
    }

    fn as_numeric(&self) -> Option<&N> {
        match self {
            FloatInner::Numeric(n) => Some(n),
            FloatInner::Diff(sum) => sum.as_numeric(),
            FloatInner::Atom(_) => None,
        }
    }

    fn as_numeric_mut(&mut self) -> Option<&mut N> {
        match self {
            FloatInner::Numeric(n) => Some(n),
            FloatInner::Diff(sum) => sum.as_numeric_mut(),
            FloatInner::Atom(_) => None,
        }
    }
}

impl<N> AsExpr<AddOp> for FloatInner<N> {
    type Expr = Sum<N, Float<N>>;

    fn from_expr(expr: Self::Expr) -> Self {
        FloatInner::Diff(expr)
    }

    fn as_expr(&self) -> Option<&Self::Expr> {
        match self {
            FloatInner::Diff(sum) => Some(sum),
            _ => None,
        }
    }

    fn as_expr_mut(&mut self) -> Option<&mut Self::Expr> {
        match self {
            FloatInner::Diff(sum) => Some(sum),
            _ => None,
        }
    }
}

impl<N> ApplyNum<AddOp> for Rc<FloatInner<N>>
where
    N: Clone,
{
    fn apply_num(&self, AddOp: AddOp, other: &N) -> Self {
        Rc::new(FloatInner::Diff(Sum {
            constant: other.clone(),
            values: vec![Ops::from(self.clone())],
        }))
    }
}

impl<N> Apply<AddOp, Rc<FloatInner<N>>> for Rc<FloatInner<N>>
where
    N: Clone + Default,
{
    type Output = Self;

    fn apply(&self, AddOp: AddOp, other: &Rc<FloatInner<N>>) -> Self {
        Rc::new(FloatInner::Diff(Sum {
            constant: N::default(),
            values: vec![Ops::from(self.clone()), Ops::from(other.clone())],
        }))
    }
}

impl<N> RevApply<AddOp, Rc<FloatInner<N>>> for Rc<FloatInner<N>>
where
    N: Clone + Default,
{
    type Output = Rc<FloatInner<N>>;

    fn rev_apply(&self, AddOp: AddOp, other: &Rc<FloatInner<N>>) -> Rc<FloatInner<N>> {
        other.apply(AddOp, self)
    }
}

impl<N> Apply<AddOp, Sum<N, Float<N>>> for Rc<FloatInner<N>>
where
    N: Clone,
{
    type Output = Self;

    fn apply(&self, AddOp: AddOp, other: &Sum<N, Float<N>>) -> Self {
        Self::from_expr(other.apply(AddOp, &Ops::from(self.clone())))
    }
}

impl<N> RevApplyNum<AddOp> for Rc<FloatInner<N>>
where
    N: Clone,
{
    fn rev_apply_num(&self, AddOp: AddOp, other: &N) -> Self {
        self.apply_num(AddOp, other)
    }
}

pub type Float<N> = Ops<Rc<FloatInner<N>>>;
pub type Floflo<N> = Rc<Rc<FloatInner<N>>>;

impl Apply<AddOp> for f64 {
    type Output = f64;

    fn apply(&self, AddOp: AddOp, other: &f64) -> f64 {
        *self + *other
    }
}

pub trait AssertAdd<Rhs = Self>: Sized + Apply<AddOp, Rhs, Output = Self> {}
impl<N> AssertAdd for Float<N> where N: Apply<AddOp, Output = N> + Clone + Default {}

impl<N> AssertAdd<Sum<N, Float<N>>> for Float<N> where
    N: Apply<AddOp, Output = N> + Clone + Default
{
}

impl<N> AssertAdd for Floflo<N> where N: Apply<AddOp, Output = N> + Clone + Default {}
