use log::{debug, info};
use std::borrow::Borrow;
use std::cmp;
use std::fmt;
use std::iter;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
use std::rc::Rc;

use num::{BigUint, Integer, Num, One, ToPrimitive, Zero};

use crate::model::size::{Range, ToRange};

pub trait Atom: ToRange + Clone + fmt::Debug + fmt::Display + PartialEq {}

// REDO
//
/*
/// A trait representing objects that can perform assignment to a reference-counted pointer.
trait Assigner<T>: Sized {
    /// Perfom the assignment.
    ///
    /// Calling `assign` should conceptually be identical to `*dst = Rc::new(src)`, but allows
    /// implementers to perform additional bookkeeping in addition to the assignment.
    ///
    /// The default implementation simply creates a new `Rc` and assigns `dst` to it.
    fn assign(self, dst: &mut Rc<T>, src: T) {
        *dst = Rc::new(src)
    }

    fn self_assign(self, dst: &mut T) {}
}

/// A default implementation of the `Assigner` that does nothing but perform the assignment.
#[derive(Copy, Clone, Debug)]
struct DefaultAssigner;

impl<T> Assigner<T> for DefaultAssigner {}

enum DeferredAssignmentInner<'a, T> {
    Initial(&'a mut Rc<T>),
    Unique(&'a mut T),
    Multiple { initial: &'a mut Rc<T>, new: T },
}

struct DeferredAssignment<'a, T, A = DefaultAssigner>
where
    A: Assigner<T>,
{
    inner: Option<DeferredAssignmentInner<'a, T>>,
    assigner: Option<A>,
}

impl<'a, T, A> DeferredAssignment<'a, T, A>
where
    T: Clone,
    A: Assigner<T>,
{
    fn new(rc: &'a mut Rc<T>, assigner: A) -> Self {
        DeferredAssignment {
            inner: Some(DeferredAssignmentInner::Initial(rc)),
            assigner: Some(assigner),
        }
    }

    fn as_ref(&self) -> &T {
        use DeferredAssignmentInner::*;
        match self.inner.as_ref().unwrap() {
            Initial(initial) => &**initial,
            Multiple { new, .. } => new,
            Unique(new) => new,
        }
    }

    fn to_mut(&mut self) -> &mut T {
        use DeferredAssignmentInner::*;
        if {
            if let Initial(_) = self.inner.as_ref().unwrap() {
                false
            } else {
                true
            }
        } {
            let initial = if let Initial(initial) = self.inner.take().unwrap() {
                initial
            } else {
                unreachable!()
            };

            if Rc::get_mut(initial).is_some() {
                self.inner = Some(Unique(Rc::get_mut(initial).unwrap()));
            } else {
                self.inner = Some(Multiple {
                    new: (**initial).clone(),
                    initial,
                })
            }
        }

        match self.inner.as_mut().unwrap() {
            Initial(_) => unreachable!(),
            Unique(new) => new,
            Multiple { new, .. } => new,
        }
    }
}

impl<'a, T, A> Drop for DeferredAssignment<'a, T, A>
where
    A: Assigner<T>,
{
    fn drop(&mut self) {
        use DeferredAssignmentInner::*;

        match self.inner.take().unwrap() {
            Initial(_) => (),
            Unique(new) => self.assigner.take().unwrap().self_assign(new),
            Multiple { initial, new } => self.assigner.take().unwrap().assign(initial, new),
        }
    }
}

/// This trait should be implemented by symbolic types.
trait AsNumeric: Clone {
    /// The corresponding numeric type.
    type Numeric: Num + Clone;

    /// Returns the numeric value of the symbolic expression if it is known to be constant.
    ///
    /// `as_numeric` may return `None` even if the expression is not constant if doing so would
    /// require additional simplifications.
    fn as_numeric(&self) -> Option<&Self::Numeric>;

    /// Mutable version of `as_numeric`.
    ///
    /// # Notes
    ///
    /// `as_numeric_mut` should never return `None` unless `as_numeric` also does.
    fn as_numeric_mut(&mut self) -> Option<&mut Self::Numeric>;
}

/// A symbolic expression.
///
/// Symbolic expressions are split into two parts, a numeric part and a symbolic part.
///
/// This type is used to represent partially exploded expressions.
///
/// This is basically a tuple with two elements with some added methods for convenience.
#[derive(Debug, Copy, Clone)]
struct Expr<N, E> {
    numeric: N,
    symbolic: E,
}

type ExprRef<'a, N, E> = Expr<&'a N, &'a E>;

type ExprMut<'a, N, E> = Expr<&'a mut N, &'a mut E>;

impl<'a, N, E> ExprRef<'a, N, E>
where
    N: Clone,
    E: Clone,
{
    /// Converts this reference to an actual expression by cloning each components.
    fn to_expr(self) -> Expr<N, E> {
        Expr {
            numeric: self.numeric.clone(),
            symbolic: self.symbolic.clone(),
        }
    }
}

impl<N, E> Expr<N, E> {
    /// Get a mutable reference to the underlying expression.
    fn as_mut(&mut self) -> ExprMut<N, E> {
        Expr {
            numeric: &mut self.numeric,
            symbolic: &mut self.symbolic,
        }
    }
}

/// A trait to design operators on a symbolic expression.
trait Op<N, S>: Copy {
    /// The type of expressions built using this operator.
    type Expr;
}

/// A trait for operators which have a neutral element, such as `0` for addition or `1` for
/// multiplication.
trait Neutral<N, S>: Op<N, S> {
    /// Returns the neutral element for the operator.
    fn neutral(self) -> Expr<N, Self::Expr>;
}

trait Apply<Lhs, Rhs = Lhs>: Copy {
    type Output;

    fn apply(self, lhs: Lhs, rhs: Rhs) -> Self::Output;
}

trait ApplyAssign<Lhs, Rhs = Lhs>: Copy {
    fn apply_assign(self, lhs: &mut Lhs, rhs: Rhs);
}

impl<'a, O, Lhs, Rhs> ApplyAssign<Option<Lhs>, &'a Option<Rhs>> for O
where
    O: ApplyAssign<Lhs, &'a Rhs>,
    Rhs: Clone + Into<Lhs>,
{
    fn apply_assign(self, lhs: &mut Option<Lhs>, rhs: &'a Option<Rhs>) {
        match (lhs, rhs) {
            (Some(_), None) => {}
            (Some(lhs), Some(rhs)) => self.apply_assign(lhs, rhs),
            (None, Some(rhs)) => *lhs = Some(rhs.clone().into()),
            (None, None) => {}
        }
    }
}

impl<'a, O, Lhs, Rhs> ApplyAssign<Rc<Lhs>, &'a Rc<Rhs>> for O
where
    O: ApplyAssign<Lhs, &'a Rhs>,
    Lhs: Clone,
{
    fn apply_assign(self, lhs: &mut Rc<Lhs>, rhs: &'a Rc<Rhs>) {
        self.apply_assign(Rc::make_mut(lhs), &**rhs)
    }
}

// S = RangeMemo<HashMemo<RawS>>
// Expr<N, Option<Rc<S>>>

impl<N, M, O, Lhs, Rhs> ApplyAssign<Expr<N, Lhs>, Expr<M, Rhs>> for O
where
    O: ApplyAssign<N, M>,
    O: ApplyAssign<Lhs, Rhs>,
{
    fn apply_assign(self, lhs: &mut Expr<N, Lhs>, rhs: Expr<M, Rhs>) {
        self.apply_assign(&mut lhs.numeric, rhs.numeric);
        self.apply_assign(&mut lhs.symbolic, rhs.symbolic);
    }
}

#[derive(Debug, Copy, Clone)]
struct AddOp;

#[derive(Debug, Copy, Clone)]
struct SubOp;

#[derive(Debug, Copy, Clone)]
struct MulOp;

#[derive(Debug, Copy, Clone)]
struct DivOp;

#[derive(Debug, Copy, Clone)]
struct MinOp;

#[derive(Debug, Copy, Clone)]
struct MaxOp;

#[derive(Debug, Copy, Clone)]
struct LcmOp;

pub trait ToMut<T: ?Sized> {
    fn to_mut(this: &mut Self) -> &mut T;
}

impl<T: ?Sized> ToMut<T> for T {
    fn to_mut(this: &mut Self) -> &mut T {
        this
    }
}

// Op: BinOp<N, S> -> ApplyAssign<S>
//
// AddOp: ApplyAssign<RangeMemo<Diff>>
// -> AddOp: ApplyAssign<RcMemo<RangeMemo<Diff>>>
// (also ApplyAssign<Rc<RangeMemo<Diff>>> etc.)
// (in particular ApplyAssign<N, RcMemo<RangeMemo<Diff>>>)
//
// AddOp: Binop<N, Rc<AnyOp>>
//  N, &RangeMemo<Diff> -> (N, RangeMemo<Diff>)
//  N, Rc<AnyOp> -> (N, RangeMemo<Diff>)
//  mut RangeMemo<Diff>, N -> (N, RangeMemo<Diff>) [[ auto? not for min/max :( ]]
//  mut RangeMemo<Diff>, RangeMemo<Diff> -> RangeMemo<Diff> [[ auto? not for min/max :( ]]
//  mut RangeMemo<Diff>, Rc<AnyOp> -> RangeMemo<Diff>
//  Rc<AnyOp>, N -> (N, RangeMemo<Diff>)
//  Rc<AnyOp>, RangeMemo<Diff> -> (0, RangeMemo<Diff>)
//  Rc<AnyOp>, Rc<AnyOp> -> (0, RangeMemo<Diff>)
//
//[no save]
//  -> AddOp: ApplyAssign<Expr<N, Rc<AnyOp>>>
//
//  -> AddOp: ApplyAssign<Expr<N, RcMemo<Expr<N, Rc<AnyOp>>>>>
//  [ need FastPathApplyAssign ]
//
//
// Op: ApplyAssign<S, S> -> Op: ApplyAssign<Rc<S>, Rc<S>>

trait BinOp<N, S>: Op<N, S> + for<'a, 'b> Apply<&'a N, &'b N, Output = N> {
    fn fast_apply_assign<D>(self, _lhs: &mut D, _rhs: &S) -> bool
    where
        D: AsRef<S> + ToMut<S>,
    {
        false
    }

    fn apply_num_expr(self, lhs: &N, rhs: ExprRef<N, Self::Expr>) -> Expr<N, Self::Expr>;

    fn apply_num_other(self, lhs: &N, rhs: &S) -> Expr<N, Self::Expr>;

    fn apply_assign_num(self, lhs: ExprMut<N, Self::Expr>, rhs: &N);

    fn apply_assign_expr(self, lhs: ExprMut<N, Self::Expr>, rhs: ExprRef<N, Self::Expr>);

    fn apply_assign_other(self, lhs: ExprMut<N, Self::Expr>, rhs: &S);

    fn apply_other_num(self, lhs: &S, rhs: &N) -> Expr<N, Self::Expr>;

    fn apply_other_expr(self, lhs: &S, rhs: ExprRef<N, Self::Expr>) -> Expr<N, Self::Expr>;

    fn apply_other_other(self, lhs: &S, rhs: &S) -> Expr<N, Self::Expr>;
}

/// Trait used for deconstructing symbolic expressions.
trait AsExpr<E> {
    /// Deconstructs the symbolic expression into its numeric and symbolic part, if it has the
    /// appropriate constructor.
    fn as_expr(&self) -> Option<&E>;

    /// Mutable version of `as_expr`.
    ///
    /// # Notes
    ///
    /// `as_expr_mut` should never return `None` unless `as_expr` also does.
    fn as_expr_mut(&mut self) -> Option<&mut E>;
}

impl<S, E> AsExpr<E> for Rc<S>
where
    S: AsExpr<E> + Clone,
{
    fn as_expr(&self) -> Option<&E> {
        self.as_expr()
    }

    fn as_expr_mut(&mut self) -> Option<&mut E> {
        if self.as_expr().is_some() {
            Rc::make_mut(self).as_expr_mut()
        } else {
            None
        }
    }
}

impl<S, E> AsExpr<E> for Option<S>
where
    S: AsExpr<E>,
{
    fn as_expr(&self) -> Option<&E> {
        self.as_ref().and_then(S::as_expr)
    }

    fn as_expr_mut(&mut self) -> Option<&mut E> {
        self.as_mut().and_then(S::as_expr_mut)
    }
}

impl<N, S> Expr<N, S> {
    fn new_numeric(numeric: N) -> Self
    where
        S: Default,
    {
        Expr {
            numeric,
            symbolic: S::default(),
        }
    }

    fn as_expr<E>(&self) -> Option<ExprRef<N, E>>
    where
        S: AsExpr<E>,
    {
        self.symbolic.as_expr().map(|symbolic| Expr {
            numeric: &self.numeric,
            symbolic,
        })
    }

    fn as_expr_mut<E>(&mut self) -> Option<ExprMut<N, E>>
    where
        S: AsExpr<E>,
    {
        self.symbolic.as_expr_mut().map({
            let numeric = &mut self.numeric;
            move |symbolic| Expr { numeric, symbolic }
        })
    }
}

// BinOp<N, S> => ApplyAssign<DeferredAssignment<S>
impl<'a, N, S, O> ApplyAssign<Expr<N, Option<Rc<S>>>, &'a Expr<N, Option<Rc<S>>>> for O
where
    O: BinOp<N, Expr<N, Option<Rc<S>>>>,
    S: AsExpr<O::Expr> + Clone,
{
    fn apply_assign(self, lhs: &mut Expr<N, Option<Rc<S>>>, rhs: &Expr<N, Option<Rc<S>>>) {
        let mut lhs = DeferredAssignment::new(lhs, BinOpAssigner::new(self, rhs));
        if self.fast_apply_assign(lhs, rhs) {
            // Assignment done in the fast path
        } else if let Some(rhs_num) = rhs.as_numeric() {
            if let Some(lhs_num) = lhs.as_numeric_mut() {
                *lhs.to_mut() = Expr::new_numeric(self.apply(lhs_num, rhs_num))
            } else if let Some(lhs_expr) = lhs.as_expr_mut() {
                self.apply_assign_num(lhs_expr, rhs_num);
            } else {
                *lhs.to_mut() = self.apply_other_num(lhs.as_ref(), rhs_num);
            }
        } else if let Some(rhs_expr) = rhs.as_expr() {
            if let Some(lhs_num) = lhs.as_numeric() {
                *lhs.to_mut() = self.apply_num_expr(lhs_num, rhs_expr);
            } else if let Some(lhs_expr) = lhs.as_expr_mut() {
                self.apply_assign_expr(lhs_expr, rhs_expr);
            } else {
                *lhs.to_mut() = self.apply_other_expr(lhs.as_ref(), rhs_expr);
            }
        } else if let Some(lhs_num) = lhs.as_numeric() {
            *lhs.to_mut() = self.apply_num_other(lhs_num, rhs);
        } else if let Some(lhs_expr) = lhs.as_expr_mut() {
            self.apply_assign_other(lhs_expr, rhs);
        } else {
            *lhs.to_mut() = self.apply_other_other(lhs.as_ref(), rhs);
        }
    }
}
*/
// END REDO

// implements "T op U" based on top of "T op &U"
macro_rules! forward_val_val_binop {
    (impl($($gen:tt)*) $imp:ident<Output = $o:ty>, $method:ident for $t:ty, $u:ty) => {
        impl<$($gen)*> $imp<$u> for $t {
            type Output = $o;

            #[inline]
            fn $method(self, other: $u) -> $t {
                $imp::$method(self, &other)
            }
        }
    };
}

// implements "T op U" based on top of "U op T" where op is commutative
macro_rules! forward_val_val_binop_commutative {
    (impl($($gen:tt)*) $imp:ident<Output = $o:ty>, $method:ident for $t:ty, $u:ty) => {
        impl<$($gen)*> $imp<$u> for $t {
            type Output = $o;

            #[inline]
            fn $method(self, other: $u) -> $o {
                $imp::$method(other, self)
            }
        }
    };
}

// implements "&T op U" based on top of "&T op &U"
macro_rules! forward_ref_val_binop {
    (impl($($gen:tt)*) $imp:ident<Output = $o:ty>, $method:ident for $t:ty, $u:ty) => {
        impl<'a, $($gen)*> $imp<$u> for &'a $t {
            type Output = $o;

            #[inline]
            fn $method(self, other: $u) -> $o {
                $imp::$method(self, &other)
            }
        }
    };
}

// implements "&T op U" based on top of "U op &T" where op is commutative
macro_rules! forward_ref_val_binop_commutative {
    (impl($($gen:tt)*) $imp:ident<Output = $o:ty>, $method:ident for $t:ty, $u:ty) => {
        impl<'a, $($gen)*> $imp<$u> for &'a $t {
            type Output = $o;

            #[inline]
            fn $method(self, other: $u) -> $o {
                $imp::$method(other, self)
            }
        }
    };
}

// implements "T op &U" based on top of "&T op &U"
macro_rules! forward_val_ref_binop {
    (impl($($gen:tt)*) $imp:ident<Output = $o:ty>, $method:ident for $t:ty, $u:ty) => {
        impl<'a, $($gen)*> $imp<&'a $u> for $t {
            type Output = $o;

            #[inline]
            fn $method(self, other: &'a $u) -> $o {
                $imp::$method(&self, other)
            }
        }
    };
}

// implements "&T op &U" based on top of "T op &U" where T is expected to be `Clone`able
macro_rules! forward_ref_ref_binop {
    (impl($($gen:tt)*) $imp:ident<Output = $o:ty>, $method:ident for $t:ty, $u:ty) => {
        impl<'a, 'b, $($gen)*> $imp<&'a $u> for &'b $t {
            type Output = $o;

            #[inline]
            fn $method(self, other: &'a $u) -> $o {
                $imp::$method(self.clone(), other)
            }
        }
    };
}

// implements "&T op &U" based on top of "&U op &T" where op is commutative
macro_rules! forward_ref_ref_binop_commutative {
    (impl($($gen:tt)*) $imp:ident<Output = $o:ty>, $method:ident for $t:ty, $u:ty) => {
        impl<'a, 'b, $($gen)*> $imp<&'a $u> for &'b $t {
            type Output = $o;

            #[inline]
            fn $method(self, other: &'a $u) -> $o {
                $imp::$method(other, self)
            }
        }
    };
}

// forward all to "&T op &U"
macro_rules! forward_binop_to_ref_ref {
    (impl($($gen:tt)*) $imp:ident<Output = $o:ty>, $method:ident for $t:ty, $u:ty) => {
        forward_val_val_binop!(impl($($gen)*) $imp<Output = $o>, $method for $t, $u);
        forward_val_ref_binop!(impl($($gen)*) $imp<Output = $o>, $method for $t, $u);
        forward_ref_val_binop!(impl($($gen)*) $imp<Output = $o>, $method for $t, $u);
    };
}

// forward all to "T op &U" where T is expected to be `Clone`able
macro_rules! forward_binop_to_val_ref {
    (impl($($gen:tt)*) $imp:ident<Output = $o:ty>, $method:ident for $t:ty, $u:ty) => {
        forward_val_val_binop!(impl($($gen)*) $imp<Output = $o>, $method for $t, $u);
        forward_ref_val_binop!(impl($($gen)*) $imp<Output = $o>, $method for $t, $u);
        forward_ref_ref_binop!(impl($($gen)*) $imp<Output = $o>, $method for $t, $u);
    };
}

// forward "T op U", "&T op U" and "&T op &U" to "T op &U" or "U op &T" where T is expected to be
// `Clone`able and op is commutative
macro_rules! forward_binop_to_val_ref_commutative {
    (impl($($gen:tt)*) $imp:ident<Output = $o:ty>, $method:ident for $t:ty, $u:ty) => {
        forward_val_val_binop!(impl($($gen)*) $imp<Output = $o>, $method for $t, $u);
        forward_ref_val_binop_commutative!(impl($($gen)*) $imp<Output = $o>, $method for $t, $u);
        forward_ref_ref_binop!(impl($($gen)*) $imp<Output = $o>, $method for $t, $u);
    };
}

// implements "T op= U" based on top of "T op= &U"
macro_rules! forward_val_op_assign {
    (impl($($gen:tt)*) $imp:ident, $method:ident for $t:ty, $u:ty) => {
        impl<$($gen)*> $imp<$u> for $t {
            #[inline]
            fn $method(&mut self, other: $u) {
                $imp::$method(self, &other)
            }
        }
    };
}

// implements "T op &U" based on top of "T op= &U"
macro_rules! forward_val_ref_to_op_assign {
    (impl($($gen:tt)*) $imp:ident, $method:ident for $t:ty, $u:ty, $imp_assign:ident, $method_assign:ident) => {
        impl<'a, $($gen)*> $imp<&'a $u> for $t {
            type Output = $t;

            #[inline]
            fn $method(mut self, other: &'a $u) -> $t {
                $imp_assign::$method_assign(&mut self, other);
                self
            }
        }
    };
}

// forward "T op U", "T op &U", "&T op U", "&T op &U" and "T op= U" to "T op= &U" where T is
// expected to be `Clone`able
macro_rules! forward_binop_to_op_assign {
    (impl($($gen:tt)*) $imp:ident, $method:ident for $t:ty, $u:ty, $imp_assign:ident, $method_assign:ident) => {
        forward_binop_to_val_ref!(impl($($gen)*) $imp<Output = $t>, $method for $t, $u);
        forward_val_ref_to_op_assign!(impl($($gen)*) $imp, $method for $t, $u, $imp_assign, $method_assign);
        forward_val_op_assign!(impl($($gen)*) $imp_assign, $method_assign for $t, $u);
    };
}

// forward "T op U", "T op &U", "&T op U", "&T op &U" and "T op= U" to "T op= &U" or "U op &T"
// where T is expected to be `Clone`able and op is commutative
macro_rules! forward_binop_to_op_assign_commutative {
    (impl($($gen:tt)*) $imp:ident, $method:ident for $t:ty, $u:ty, $imp_assign:ident, $method_assign:ident) => {
        forward_binop_to_val_ref_commutative!(impl($($gen)*) $imp<Output = $t>, $method for $t, $u);
        forward_val_ref_to_op_assign!(impl($($gen)*) $imp, $method for $t, $u, $imp_assign, $method_assign);
        forward_val_op_assign!(impl($($gen)*) $imp_assign, $method_assign for $t, $u);
    };
}

// forward "T op U", "T op &U", "&T op &U" and "&T op U" to "U op &T" where op is commutative
macro_rules! forward_binop_to_ref_val_commutative {
    (impl($($gen:tt)*) $imp:ident<Output = $o:ty>, $method:ident for $t:ty, $u:ty) => {
        forward_val_val_binop_commutative!(impl($($gen)*) $imp<Output = $o>, $method for $t, $u);
        forward_val_ref_binop!(impl($($gen)*) $imp<Output = $o>, $method for $t, $u);
        forward_ref_ref_binop_commutative!(impl($($gen)*) $imp<Output = $o>, $method for $t, $u);
        forward_ref_val_binop_commutative!(impl($($gen)*) $imp<Output = $o>, $method for $t, $u);
    };
}

// Integer division of parameters.  There is the assumption that whatever values of the
// parameters are chosen, the ratio is an integer.
#[derive(Debug, Clone, PartialEq)]
struct RatioInner<P> {
    factor: BigUint,
    numer: Vec<P>,
    denom: Vec<P>,
}

#[derive(Clone, PartialEq)]
struct Ratio<P> {
    inner: Rc<RatioInner<P>>,
}

impl<P> fmt::Debug for Ratio<P>
where
    P: fmt::Debug,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_struct("Ratio")
            .field("factor", &self.inner.factor)
            .field("numer", &self.inner.numer)
            .field("denom", &self.inner.denom)
            .finish()
    }
}

impl<P> fmt::Display for RatioInner<P>
where
    P: Atom + fmt::Display,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        use itertools::Itertools;

        if self.numer.is_empty() {
            write!(fmt, "{}", self.factor)?;
        } else {
            if !self.factor.is_one() {
                write!(fmt, "{}*", self.factor)?;
            }

            write!(fmt, "{}", self.numer.iter().format("*"))?;
        }

        if !self.denom.is_empty() {
            write!(fmt, "/{}", self.denom.iter().format("/"))?;
        }

        Ok(())
    }
}

impl<P> fmt::Display for Ratio<P>
where
    P: Atom + fmt::Display,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.inner, fmt)
    }
}

impl<P> From<RatioInner<P>> for Ratio<P>
where
    P: Atom,
{
    fn from(inner: RatioInner<P>) -> Self {
        Ratio {
            inner: Rc::new(inner),
        }
    }
}

impl<P> Ratio<P>
where
    P: Atom,
{
    fn new(factor: BigUint, numer: Vec<P>, denom: Vec<P>) -> Self {
        RatioInner {
            factor,
            numer,
            denom,
        }
        .into()
    }

    fn to_u32(&self) -> Option<u32> {
        if self.inner.numer.is_empty() && self.inner.denom.is_empty() {
            Some(
                self.inner
                    .factor
                    .to_u32()
                    .unwrap_or_else(|| panic!("Unable to represent factor as u32")),
            )
        } else {
            None
        }
    }

    fn one() -> Self {
        Self::new(1u32.into(), Vec::new(), Vec::new())
    }

    fn as_biguint(&self) -> Option<&BigUint> {
        if self.inner.numer.is_empty() && self.inner.denom.is_empty() {
            Some(&self.inner.factor)
        } else {
            None
        }
    }

    fn as_biguint_mut(&mut self) -> Option<&mut BigUint> {
        if self.inner.numer.is_empty() && self.inner.denom.is_empty() {
            let inner = Rc::make_mut(&mut self.inner);
            if inner.numer.is_empty() && inner.denom.is_empty() {
                Some(&mut inner.factor)
            } else {
                unreachable!()
            }
        } else {
            None
        }
    }
}

impl<P: ToRange> ToRange for RatioInner<P> {
    fn to_range(&self) -> Range {
        let factor = self
            .factor
            .to_u64()
            .unwrap_or_else(|| panic!("Unable to represent factor as u64"));
        let numer_ranges = self.numer.iter().map(ToRange::to_range).collect::<Vec<_>>();
        let denom_ranges = self.denom.iter().map(ToRange::to_range).collect::<Vec<_>>();
        let numer_min = numer_ranges.iter().map(|r| r.min).product::<u64>();
        let numer_max = numer_ranges.iter().map(|r| r.max).product::<u64>();
        let denom_min = denom_ranges.iter().map(|r| r.min).product::<u64>();
        let denom_max = denom_ranges.iter().map(|r| r.max).product::<u64>();
        Range {
            min: (factor * numer_min) / denom_max,
            max: (factor * numer_max) / denom_min,
        }
    }
}

impl<P: ToRange> ToRange for Ratio<P> {
    fn to_range(&self) -> Range {
        self.inner.to_range()
    }
}

impl<'a, P> MulAssign<&'a Ratio<P>> for Ratio<P>
where
    P: Atom,
{
    fn mul_assign(&mut self, rhs: &'a Ratio<P>) {
        let lhs = Rc::make_mut(&mut self.inner);
        lhs.factor *= &rhs.inner.factor;
        for numer in rhs.inner.numer.iter() {
            if let Some(pos) = lhs.denom.iter().position(|d| d == numer) {
                lhs.denom.swap_remove(pos);
            } else {
                lhs.numer.push(numer.clone());
            }
        }
        for denom in rhs.inner.denom.iter() {
            if let Some(pos) = lhs.numer.iter().position(|n| n == denom) {
                lhs.numer.swap_remove(pos);
            } else {
                lhs.denom.push(denom.clone());
            }
        }
    }
}

forward_binop_to_op_assign_commutative!(impl(P: Atom) Mul, mul for Ratio<P>, Ratio<P>, MulAssign, mul_assign);

trait ReductionKind<C> {
    fn reduce(&self, lhs: &C, rhs: &C) -> C;

    fn reduce_assign(&self, lhs: &mut C, rhs: &C);
}

trait ReductionSkip<C, S>: ReductionKind<C> {
    fn should_skip(&self, _constant: &C, _other: &S) -> bool {
        false
    }

    fn get_bound(&self, _other: &S) -> Option<C> {
        None
    }
}

trait AsReduction<K, S = Self> {
    type Value: Into<S> + Clone;

    fn as_reduction(&self, kind: K) -> Option<&Reduction<K, Self::Value, S>>;

    fn as_reduction_mut(&mut self, kind: K) -> Option<&mut Reduction<K, Self::Value, S>>;
}

trait ReduceAssign<K, Rhs = Self> {
    fn reduce_assign(&mut self, kind: K, rhs: Rhs);
}

impl<'a, K, S, C> ReduceAssign<K, &'a S> for S
where
    K: ReductionSkip<C, S> + Copy,
    S: AsReduction<K, S, Value = C>
        + From<Reduction<K, C, S>>
        + From<C>
        + AsValue<C>
        + Clone
        + fmt::Display,
    C: Clone + fmt::Display,
{
    fn reduce_assign(&mut self, kind: K, rhs: &'a S) {
        if let Some(lhs_cst) = self.as_value() {
            if let Some(rhs_cst) = rhs.as_value() {
                *self = kind.reduce(lhs_cst, rhs_cst).into();
            } else if let Some(rhs_red) = rhs.as_reduction(kind) {
                *self = rhs_red.clone().reduce(Some(lhs_cst), iter::empty()).into();
            } else {
                *self = Reduction::new_constant(kind, lhs_cst.clone())
                    .reduce(None, iter::once(rhs))
                    .into();
            }
        } else if let Some(lhs_red) = self.as_reduction_mut(kind) {
            if let Some(rhs_cst) = rhs.as_value() {
                lhs_red.reduce_assign(Some(rhs_cst), iter::empty());
            } else if let Some(rhs_red) = rhs.as_reduction(kind) {
                lhs_red.reduce_assign(rhs_red.constant.as_ref(), rhs_red.others.iter());
            } else {
                lhs_red.reduce_assign(None, iter::once(rhs));
            }
        } else if let Some(rhs_cst) = rhs.as_value() {
            *self = Reduction::new_constant(kind, rhs_cst.clone())
                .reduce(None, iter::once(&*self))
                .into();
        } else if let Some(rhs_red) = rhs.as_reduction(kind) {
            *self = rhs_red.clone().reduce(None, iter::once(&*self)).into()
        } else {
            *self = Reduction::new_unknown(kind, self.clone())
                .reduce(None, iter::once(rhs))
                .into();
        }
    }
}

impl<P> AsReduction<IntReductionKind, Int<P>> for Int<P>
where
    P: Atom,
{
    type Value = BigUint;

    fn as_reduction(&self, kind: IntReductionKind) -> Option<&IntReduction<P>> {
        match &*self.inner {
            IntInner::Reduction(reduction) if reduction.kind == kind => Some(reduction),
            _ => None,
        }
    }

    fn as_reduction_mut(
        &mut self,
        kind: IntReductionKind,
    ) -> Option<&mut IntReduction<P>> {
        if self.as_reduction(kind).is_some() {
            Some(match Rc::make_mut(&mut self.inner) {
                IntInner::Reduction(reduction) if reduction.kind == kind => reduction,
                _ => unreachable!(),
            })
        } else {
            None
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum IntReductionKind {
    /// Computes the lowest common multiple of its arguments.
    Lcm,
    /// Computes the minimum of its arguments.
    Min,
}

impl fmt::Display for IntReductionKind {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            IntReductionKind::Lcm => write!(fmt, "lcm"),
            IntReductionKind::Min => write!(fmt, "min"),
        }
    }
}

impl ReductionKind<BigUint> for IntReductionKind {
    fn reduce(&self, lhs: &BigUint, rhs: &BigUint) -> BigUint {
        match self {
            IntReductionKind::Lcm => lhs.lcm(rhs),
            IntReductionKind::Min => {
                if lhs < rhs {
                    lhs.clone()
                } else {
                    rhs.clone()
                }
            }
        }
    }

    fn reduce_assign(&self, lhs: &mut BigUint, rhs: &BigUint) {
        match self {
            IntReductionKind::Lcm => *lhs = lhs.lcm(rhs),
            IntReductionKind::Min => {
                if rhs < lhs {
                    *lhs = rhs.clone()
                }
            }
        }
    }
}

impl<P> ReductionSkip<BigUint, Int<P>> for IntReductionKind
where
    P: Atom,
{
    /// Indicates whether to skip adding `other` to a reduction which contains `constant`.  Returns
    /// `true` when we are able to prove that `reduction(constant, other) = constant`.
    fn should_skip(&self, constant: &BigUint, other: &Int<P>) -> bool {
        info!("Try skipping: {} for {}", constant, other);

        match self {
            // lcm(ka, a/b) = ka
            IntReductionKind::Lcm => {
                if let Some(ratio) = other.as_ratio() {
                    ratio.inner.numer.is_empty()
                        && constant.is_multiple_of(&ratio.inner.factor)
                } else {
                    false
                }
            }
            // min(a, b) = a if a <= b
            IntReductionKind::Min => constant < &BigUint::from(other.range().min),
        }
    }

    fn get_bound(&self, other: &Int<P>) -> Option<BigUint> {
        match self {
            IntReductionKind::Min => Some(other.range().max.into()),
            IntReductionKind::Lcm => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Reduction<K, C, S> {
    kind: K,
    constant: Option<C>,
    others: Vec<S>,
}

impl<K, C, S> fmt::Display for Reduction<K, C, S>
where
    K: fmt::Display,
    C: fmt::Display,
    S: fmt::Display,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        use itertools::Itertools;

        if let Some(constant) = &self.constant {
            if self.others.is_empty() {
                write!(fmt, "{}({})", self.kind, constant)
            } else {
                write!(
                    fmt,
                    "{}({}, {})",
                    self.kind,
                    constant,
                    self.others.iter().format(", ")
                )
            }
        } else {
            write!(fmt, "{}({})", self.kind, self.others.iter().format(", "))
        }
    }
}

trait AsValue<C> {
    fn as_value(&self) -> Option<&C>;
    fn as_value_mut(&mut self) -> Option<&mut C>;
}

impl<K, C, S> AsValue<C> for Reduction<K, C, S> {
    fn as_value(&self) -> Option<&C> {
        if self.others.is_empty() {
            Some(self.constant.as_ref().unwrap())
        } else {
            None
        }
    }

    fn as_value_mut(&mut self) -> Option<&mut C> {
        if self.others.is_empty() {
            Some(self.constant.as_mut().unwrap())
        } else {
            None
        }
    }
}

impl<K, C, S> Reduction<K, C, S>
where
    K: ReductionSkip<C, S>,
{
    fn new_constant(kind: K, constant: C) -> Self {
        Reduction {
            kind,
            constant: Some(constant),
            others: Vec::new(),
        }
    }

    fn new_unknown(kind: K, other: S) -> Self {
        Reduction {
            constant: kind.get_bound(&other),
            kind,
            others: vec![other],
        }
    }

    fn reduce<'a, I>(mut self, constant: Option<&C>, others: I) -> Self
    where
        I: Iterator<Item = &'a S>,
        C: fmt::Display + Clone,
        S: fmt::Display + Clone + 'a,
    {
        self.reduce_assign(constant, others);
        self
    }

    fn reduce_assign<'a, I>(&mut self, constant: Option<&C>, others: I)
    where
        I: Iterator<Item = &'a S>,
        C: fmt::Display + Clone,
        S: fmt::Display + Clone + 'a,
    {
        let kind = &self.kind;
        if let Some(rhs_constant) = constant {
            if let Some(lhs_constant) = &mut self.constant {
                self.kind.reduce_assign(lhs_constant, rhs_constant);
            } else {
                self.constant = Some(rhs_constant.clone());
            }

            self.others
                .retain(move |elem| !kind.should_skip(rhs_constant, elem));
        }

        if let Some(constant) = &self.constant {
            self.others.extend(
                others
                    .filter(move |elem| !kind.should_skip(constant, elem))
                    .cloned(),
            )
        } else {
            for other in others {
                if self.constant.is_none() {
                    if let Some(bound) = kind.get_bound(other) {
                        info!("Setting bound to {} from {}", bound, other);
                        self.constant = Some(bound);
                    }
                }
                self.others.push(other.clone());
            }
        }

        // Update the constant and retain
        if let Some(constant) = &mut self.constant {
            for other in &self.others {
                if let Some(other_bound) = kind.get_bound(other) {
                    self.kind.reduce_assign(constant, &other_bound);
                }
            }
            self.others
                .retain(move |elem| !kind.should_skip(constant, elem));
        }
    }
}

type IntReduction<P> = Reduction<IntReductionKind, BigUint, Int<P>>;

impl<P> IntReduction<P>
where
    P: Atom,
{
    fn as_biguint(&self) -> Option<&BigUint> {
        self.as_value()
    }

    fn as_biguint_mut(&mut self) -> Option<&mut BigUint> {
        self.as_value_mut()
    }
}

#[derive(Debug, Clone, PartialEq)]
enum IntInner<P> {
    // Ceil division: ceil(numer / denom)
    DivCeil(Int<P>, u32),
    /// Minimum/Lowest common multiple of all arguments.
    /// TODO: split the constant element when there is one.
    /// TODO: The constant element should include the `gcd` of all others?
    /// TODO: Incorporate a `Mul`?
    Reduction(IntReduction<P>),
    /// a - b
    Sub(Int<P>, u32),
    // Reduction(IntReduction<P>)
    /// Multiplication of all arguments.  We keep the Ratio separate to make simplifications
    /// easier.
    Mul(Ratio<P>, Vec<Int<P>>),
}

impl<P> fmt::Display for IntInner<P>
where
    P: Atom + fmt::Display,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        use itertools::Itertools;
        use IntInner::*;

        match self {
            DivCeil(numer, denom) => write!(fmt, "div_ceil({}, {})", numer, denom),
            Reduction(reduction) => write!(fmt, "{}", reduction),
            Sub(lhs, rhs) => write!(fmt, "{} - {}", lhs, rhs),
            Mul(ratio, args) if args.is_empty() => write!(fmt, "{}", ratio),
            Mul(ratio, args) => {
                if !ratio.as_biguint().map(One::is_one).unwrap_or(false) {
                    write!(fmt, "{}*", ratio)?;
                }
                write!(fmt, "{}", args.iter().format("*"))
            }
        }
    }
}

impl<P> IntInner<P>
where
    P: Atom,
{
    fn is_mul(&self) -> bool {
        if let IntInner::Mul(_, _) = self {
            true
        } else {
            false
        }
    }
}

#[derive(PartialEq)]
pub struct Int<P> {
    inner: Rc<IntInner<P>>,
}

impl<P> fmt::Debug for Int<P>
where
    P: fmt::Debug,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&self.inner, fmt)
    }
}

impl<P> fmt::Display for Int<P>
where
    P: Atom + fmt::Display,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.inner, fmt)
    }
}

impl<P> Clone for Int<P> {
    fn clone(&self) -> Self {
        Int {
            inner: Rc::clone(&self.inner),
        }
    }
}

impl<P> From<IntReduction<P>> for Int<P>
where
    P: Atom,
{
    fn from(reduction: IntReduction<P>) -> Self {
        IntInner::Reduction(reduction).into()
    }
}

impl<P> From<IntInner<P>> for Int<P>
where
    P: Atom,
{
    fn from(inner: IntInner<P>) -> Self {
        Int {
            inner: Rc::new(inner),
        }
    }
}

impl<P, T> From<T> for Int<P>
where
    P: Atom,
    T: Into<BigUint>,
{
    fn from(inner: T) -> Self {
        Int::ratio(inner, Vec::new(), Vec::new())
    }
}

impl<P> AsValue<BigUint> for Int<P>
where
    P: Atom,
{
    fn as_value(&self) -> Option<&BigUint> {
        self.as_biguint()
    }

    fn as_value_mut(&mut self) -> Option<&mut BigUint> {
        self.as_biguint_mut()
    }
}

impl<P> Int<P>
where
    P: Atom,
{
    pub fn div_ceil(lhs: &Self, rhs: u32) -> Self {
        lhs.as_biguint()
            .map(|lhs| Int::from((lhs + rhs - 1u32) / rhs))
            .unwrap_or_else(|| IntInner::DivCeil(lhs.clone(), rhs).into())
    }

    pub fn to_symbolic_float(&self) -> Float<P> {
        match &*self.inner {
            IntInner::Reduction(IntReduction {
                kind: IntReductionKind::Min,
                ..
            }) => {
                // TODO: convert to float minimum (actually -- do we ever need this?)
                unimplemented!()
            }
            IntInner::Sub(lhs, rhs) => lhs.to_symbolic_float() - f64::from(*rhs),
            IntInner::Mul(ratio, args) => unimplemented!(),
            IntInner::DivCeil(_, _) | IntInner::Reduction(_) => {
                FloatInner::Mul(self.clone().into(), Vec::new()).into()
            }
        }
    }

    pub fn ratio<T: Into<BigUint>>(factor: T, numer: Vec<P>, denom: Vec<P>) -> Self {
        IntInner::Mul(Ratio::new(factor.into(), numer, denom), Vec::new()).into()
    }

    pub fn one() -> Self {
        Self::from(1u32)
    }

    pub fn fast_eq(lhs: &Self, rhs: &Self) -> bool {
        Rc::ptr_eq(&lhs.inner, &rhs.inner)
            || lhs
                .as_biguint()
                .and_then(|lhs| rhs.as_biguint().map(|rhs| lhs == rhs))
                .unwrap_or(false)
    }

    fn is_one(&self) -> bool {
        self.as_biguint().map(One::is_one).unwrap_or(false)
    }

    fn as_biguint(&self) -> Option<&BigUint> {
        match &*self.inner {
            IntInner::Mul(ratio, args) if args.is_empty() => ratio.as_biguint(),
            IntInner::Reduction(reduction) => reduction.as_biguint(),
            _ => None,
        }
    }

    fn as_biguint_mut(&mut self) -> Option<&mut BigUint> {
        if self.as_biguint().is_some() {
            Some(match Rc::make_mut(&mut self.inner) {
                IntInner::Mul(ratio, args) if args.is_empty() => {
                    ratio.as_biguint_mut().unwrap()
                }
                IntInner::Reduction(reduction) => reduction.as_biguint_mut().unwrap(),
                _ => unreachable!(),
            })
        } else {
            None
        }
    }

    fn as_sub(&self) -> Option<(&Int<P>, u32)> {
        match &*self.inner {
            IntInner::Sub(lhs, rhs) => Some((lhs, *rhs)),
            _ => None,
        }
    }

    fn as_ratio(&self) -> Option<&Ratio<P>> {
        match &*self.inner {
            IntInner::Mul(ratio, args) if args.is_empty() => Some(ratio),
            _ => None,
        }
    }

    fn as_ratio_mut(&mut self) -> Option<&mut Ratio<P>> {
        if self.as_ratio().is_some() {
            Some(match Rc::make_mut(&mut self.inner) {
                IntInner::Mul(ratio, args) if args.is_empty() => ratio,
                _ => unreachable!(),
            })
        } else {
            None
        }
    }

    pub fn min_assign(&mut self, rhs: &Int<P>) {
        // TODO: if `rhs.max <= self.min` there is nothing to do.
        self.reduce_assign(IntReductionKind::Min, rhs);
    }

    pub fn lcm_assign(&mut self, rhs: &Int<P>) {
        // TODO: If `self.gcd` is a multiple of `rhs.lcm` there is nothing to do.
        self.reduce_assign(IntReductionKind::Lcm, rhs);
    }

    pub fn to_u32(&self) -> Option<u32> {
        match &*self.inner {
            IntInner::DivCeil(lhs, rhs) => lhs.to_u32().map(|lhs| (lhs + rhs - 1) / rhs),
            IntInner::Reduction(_) => unimplemented!("to_u32 for {}", self),
            IntInner::Sub(lhs, rhs) => lhs.to_u32().map(|lhs| lhs - rhs),
            IntInner::Mul(ratio, args) => ratio.to_u32().and_then(|ratio| {
                args.iter()
                    .map(|arg| arg.to_u32().ok_or(()))
                    .product::<Result<u32, ()>>()
                    .ok()
                    .map(|res| res * ratio)
            }),
        }
    }

    pub fn range(&self) -> crate::model::size::Range {
        use crate::model::size::Range;
        info!("range for {}", self);

        match &*self.inner {
            IntInner::DivCeil(numer, denom) => {
                let numer_range = numer.range();
                let denom = u64::from(*denom);
                // TODO: should take gcd for le min
                Range {
                    min: (numer_range.min + denom - 1) / denom,
                    max: (numer_range.max + denom - 1) / denom,
                }
            }
            IntInner::Reduction(_) => unimplemented!("range for {}", self),
            IntInner::Sub(lhs, rhs) => {
                let lhs_range = lhs.range();
                Range {
                    min: lhs_range.min - u64::from(*rhs),
                    max: lhs_range.max - u64::from(*rhs),
                }
            }
            IntInner::Mul(ratio, args) => {
                let mut range = ratio.to_range();
                for arg in args.iter() {
                    let arg_range = arg.range();
                    range.min *= arg_range.min;
                    range.max *= arg_range.max;
                }
                range
            }
        }
    }
}

impl<'a, P> SubAssign<&'a u32> for Int<P>
where
    P: Atom,
{
    #[inline]
    fn sub_assign(&mut self, rhs: &'a u32) {
        *self = IntInner::Sub(self.clone(), *rhs).into();
    }
}

forward_binop_to_op_assign!(impl(P: Atom) Sub, sub for Int<P>, u32, SubAssign, sub_assign);

impl<'a, P> MulAssign<&'a Int<P>> for Int<P>
where
    P: Atom,
{
    fn mul_assign(&mut self, rhs: &'a Int<P>) {
        if rhs.is_one() {
            return;
        }

        if self.inner.is_mul() {
            match Rc::make_mut(&mut self.inner) {
                IntInner::Mul(lhs_ratio, lhs_args) => match &*rhs.inner {
                    IntInner::Mul(rhs_ratio, rhs_args) => {
                        *lhs_ratio *= rhs_ratio;
                        lhs_args.extend(rhs_args.iter().cloned());
                    }
                    _ => lhs_args.push(rhs.clone()),
                },
                _ => unreachable!(),
            }
        } else if let IntInner::Mul(rhs_ratio, rhs_args) = &*rhs.inner {
            *self = IntInner::Mul(
                rhs_ratio.clone(),
                iter::once(self.clone())
                    .chain(rhs_args.iter().cloned())
                    .collect(),
            )
            .into();
        } else {
            *self = IntInner::Mul(Ratio::one(), vec![self.clone(), rhs.clone()]).into();
        }
    }
}

forward_binop_to_op_assign_commutative!(impl(P: Atom) Mul, mul for Int<P>, Int<P>, MulAssign, mul_assign);

impl<'a, P> MulAssign<&'a u64> for Int<P>
where
    P: Atom,
{
    #[inline]
    fn mul_assign(&mut self, rhs: &'a u64) {
        *self *= Self::from(*rhs);
    }
}

forward_binop_to_op_assign!(impl(P: Atom) Mul, mul for Int<P>, u64, MulAssign, mul_assign);

impl<'a, P: 'a> iter::Product<&'a Int<P>> for Int<P>
where
    P: Atom,
{
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Int<P>>,
    {
        let mut res = Self::one();
        for elem in iter {
            res *= elem;
        }
        res
    }
}

// factor * float(numer) / float(denom)
#[derive(Debug, PartialEq)]
struct FloatRatioInner<P> {
    factor: f64,
    numer: Int<P>,
    denom: Int<P>,
    // should be: factor * float(numer/denom) * float(numer)/float(denom)
}

impl<P> Clone for FloatRatioInner<P> {
    fn clone(&self) -> Self {
        FloatRatioInner {
            factor: self.factor,
            numer: self.numer.clone(),
            denom: self.denom.clone(),
        }
    }
}

impl<'a, P> MulAssign<&'a FloatRatioInner<P>> for FloatRatioInner<P>
where
    P: Atom,
{
    fn mul_assign(&mut self, rhs: &'a FloatRatioInner<P>) {
        self.factor *= rhs.factor;
        self.numer *= &rhs.numer;
        self.denom *= &rhs.denom;
    }
}

forward_binop_to_op_assign_commutative!(impl(P: Atom) Mul, mul for FloatRatioInner<P>, FloatRatioInner<P>, MulAssign, mul_assign);

impl<P> fmt::Display for FloatRatioInner<P>
where
    P: Atom + fmt::Display,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        if self.factor == 1f64 {
            write!(fmt, "{}", self.numer)?;
        } else if self.numer.is_one() {
            write!(fmt, "{}", self.factor)?;
        } else {
            write!(fmt, "{}*{}", self.factor, self.numer)?;
        }

        if !self.denom.is_one() {
            write!(fmt, "/{}", self.denom)?;
        }

        Ok(())
    }
}

#[derive(PartialEq)]
struct FloatRatio<P> {
    inner: Rc<FloatRatioInner<P>>,
}

impl<P> Clone for FloatRatio<P> {
    fn clone(&self) -> Self {
        FloatRatio {
            inner: Rc::clone(&self.inner),
        }
    }
}

impl<'a, P> MulAssign<&'a FloatRatio<P>> for FloatRatio<P>
where
    P: Atom,
{
    fn mul_assign(&mut self, rhs: &'a FloatRatio<P>) {
        *Rc::make_mut(&mut self.inner) *= &*rhs.inner;
    }
}

forward_binop_to_op_assign_commutative!(impl(P: Atom) Mul, mul for FloatRatio<P>, FloatRatio<P>, MulAssign, mul_assign);

impl<P> fmt::Debug for FloatRatio<P>
where
    P: fmt::Debug,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&self.inner, fmt)
    }
}

impl<P> fmt::Display for FloatRatio<P>
where
    P: Atom + fmt::Display,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.inner, fmt)
    }
}

impl<P> FloatRatioInner<P>
where
    P: Atom,
{
    fn recip(mut self) -> Self {
        self.factor = self.factor.recip();
        std::mem::swap(&mut self.numer, &mut self.denom);
        self
    }
}

impl<P> From<FloatRatioInner<P>> for FloatRatio<P>
where
    P: Atom,
{
    fn from(inner: FloatRatioInner<P>) -> Self {
        FloatRatio {
            inner: Rc::new(inner),
        }
    }
}

impl<P> From<f64> for FloatRatioInner<P>
where
    P: Atom,
{
    fn from(constant: f64) -> Self {
        FloatRatioInner {
            factor: constant,
            numer: Int::one(),
            denom: Int::one(),
        }
    }
}

impl<P> From<f64> for FloatRatio<P>
where
    P: Atom,
{
    fn from(constant: f64) -> Self {
        FloatRatioInner::from(constant).into()
    }
}

impl<P> From<Int<P>> for FloatRatio<P>
where
    P: Atom,
{
    fn from(int: Int<P>) -> Self {
        FloatRatioInner {
            factor: 1f64,
            numer: int,
            denom: Int::one(),
        }
        .into()
    }
}

impl<P> FloatRatio<P>
where
    P: Atom,
{
    fn recip(denom: Int<P>) -> Self {
        if let Some(val) = denom.as_biguint() {
            val.to_f64().expect("bigint too big").recip().into()
        } else {
            FloatRatioInner {
                factor: 1f64,
                numer: Int::one(),
                denom,
            }
            .into()
        }
    }

    fn is_one(&self) -> bool {
        self.inner.factor == 1f64
            && self.inner.numer.is_one()
            && self.inner.denom.is_one()
    }

    fn is_zero(&self) -> bool {
        self.inner.factor == 0f64
            && self.inner.numer.is_one()
            && self.inner.denom.is_one()
    }

    fn as_f64_ref(&self) -> Option<&f64> {
        if self.inner.numer.is_one() && self.inner.denom.is_one() {
            Some(&self.inner.factor)
        } else {
            None
        }
    }

    fn as_f64_mut(&mut self) -> Option<&mut f64> {
        if self.inner.numer.is_one() && self.inner.denom.is_one() {
            Some(&mut Rc::make_mut(&mut self.inner).factor)
        } else {
            None
        }
    }

    fn as_f64(&self) -> Option<f64> {
        let numer = self.inner.numer.as_biguint()?.to_f64()?;
        let denom = self.inner.denom.as_biguint()?.to_f64()?;
        Some(self.inner.factor * numer / denom)
    }

    fn to_f64(&self) -> Option<f64> {
        let numer = f64::from(self.inner.numer.to_u32()?);
        let denom = f64::from(self.inner.denom.to_u32()?);
        Some(self.inner.factor * numer / denom)
    }

    fn min_value(&self) -> f64
    where
        P: Atom,
    {
        self.inner.factor * self.inner.numer.range().min as f64
            / self.inner.denom.range().max as f64
    }

    fn max_value(&self) -> f64
    where
        P: Atom,
    {
        self.inner.factor * self.inner.numer.range().max as f64
            / self.inner.denom.range().min as f64
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Diff<P> {
    constant: f64,
    positive: Vec<Float<P>>,
    negative: Vec<Float<P>>,
}

impl<P> fmt::Display for Diff<P>
where
    P: Atom + fmt::Display,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        use itertools::Itertools;

        if self.positive.is_empty() {
            if self.negative.is_empty() {
                write!(fmt, "{}", self.constant)
            } else if self.constant.is_zero() {
                write!(fmt, "-{}", self.negative.iter().format(" - "))
            } else {
                write!(
                    fmt,
                    "{} - {}",
                    self.constant,
                    self.negative.iter().format(" - ")
                )
            }
        } else {
            if !self.constant.is_zero() {
                write!(fmt, "{} + ", self.constant)?;
            }

            write!(fmt, "{}", self.positive.iter().format(" + "),)?;

            if !self.negative.is_empty() {
                write!(fmt, " - {}", self.negative.iter().format(" - "))?;
            }
            Ok(())
        }
    }
}

impl<P> AsValue<f64> for Diff<P> {
    fn as_value(&self) -> Option<&f64> {
        if self.positive.is_empty() && self.negative.is_empty() {
            Some(&self.constant)
        } else {
            None
        }
    }

    fn as_value_mut(&mut self) -> Option<&mut f64> {
        if self.positive.is_empty() && self.negative.is_empty() {
            Some(&mut self.constant)
        } else {
            None
        }
    }
}

impl<P> Diff<P> {
    fn new_constant(constant: f64) -> Self {
        Diff {
            constant,
            positive: Vec::new(),
            negative: Vec::new(),
        }
    }

    fn new_unknown(positive: Vec<Float<P>>, negative: Vec<Float<P>>) -> Self {
        Diff {
            constant: 0f64,
            positive,
            negative,
        }
    }

    fn min_value(&self) -> f64
    where
        P: Atom,
    {
        self.constant + self.positive.iter().map(Float::min_value).sum::<f64>()
            - self.negative.iter().map(Float::max_value).sum::<f64>()
    }

    fn max_value(&self) -> f64
    where
        P: Atom,
    {
        self.constant + self.positive.iter().map(Float::max_value).sum::<f64>()
            - self.negative.iter().map(Float::min_value).sum::<f64>()
    }

    fn neg(mut self) -> Self {
        self.constant = -self.constant;
        std::mem::swap(&mut self.positive, &mut self.negative);
        self
    }

    fn add<'a, Pos, Neg>(mut self, constant: f64, positive: Pos, negative: Neg) -> Self
    where
        P: 'a,
        Pos: IntoIterator<Item = &'a Float<P>>,
        Neg: IntoIterator<Item = &'a Float<P>>,
    {
        self.add_assign(constant, positive, negative);
        self
    }

    fn add_assign<'a, Pos, Neg>(&mut self, constant: f64, positive: Pos, negative: Neg)
    where
        P: 'a,
        Pos: IntoIterator<Item = &'a Float<P>>,
        Neg: IntoIterator<Item = &'a Float<P>>,
    {
        self.constant += constant;
        self.positive.extend(positive.into_iter().cloned());
        self.negative.extend(negative.into_iter().cloned());
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum FloatReductionKind {
    Min,
    Max,
}

impl fmt::Display for FloatReductionKind {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FloatReductionKind::Min => write!(fmt, "min"),
            FloatReductionKind::Max => write!(fmt, "max"),
        }
    }
}

impl ReductionKind<f64> for FloatReductionKind {
    fn reduce(&self, lhs: &f64, rhs: &f64) -> f64 {
        match self {
            FloatReductionKind::Min => lhs.min(*rhs),
            FloatReductionKind::Max => lhs.max(*rhs),
        }
    }

    fn reduce_assign(&self, lhs: &mut f64, rhs: &f64) {
        *lhs = self.reduce(lhs, rhs);
    }
}

impl<P> ReductionSkip<f64, Float<P>> for FloatReductionKind
where
    P: Atom,
{
    fn should_skip(&self, constant: &f64, other: &Float<P>) -> bool {
        match self {
            FloatReductionKind::Min => other.min_value() >= *constant,
            FloatReductionKind::Max => other.max_value() <= *constant,
        }
    }

    fn get_bound(&self, other: &Float<P>) -> Option<f64> {
        info!("Getting bound from {}", other);
        Some(match self {
            FloatReductionKind::Min => other.max_value(),
            FloatReductionKind::Max => other.min_value(),
        })
    }
}

type FloatReduction<P> = Reduction<FloatReductionKind, f64, Float<P>>;

impl<P> FloatReduction<P>
where
    P: Atom,
{
    fn min_value(&self) -> f64 {
        match self.kind {
            FloatReductionKind::Min => self
                .others
                .iter()
                .map(Float::min_value)
                .fold(self.constant.unwrap(), f64::min),
            FloatReductionKind::Max => self.constant.unwrap(),
        }
    }

    fn max_value(&self) -> f64 {
        match self.kind {
            FloatReductionKind::Max => self
                .others
                .iter()
                .map(Float::max_value)
                .fold(self.constant.unwrap(), f64::max),
            FloatReductionKind::Min => self.constant.unwrap(),
        }
    }
}

/*
#[derive(Debug, Clone)]
struct IntMemo {
    hash: Cell<Option<u64>>,
    range: Cell<Option<Range<BigUint>>>,
    factors: Cell<Option<Factors<BigUint>>>,
}

#[derive(Debug, Clone)]
struct FloatMemo {
    hash: Cell<Option<u64>>,
    range: Cell<Option<Range<f64>>>,
}
*/

#[derive(Debug, Clone, PartialEq)]
enum FloatInner<P> {
    /// Product of all arguments.  We keep the `FloatRatio` separate to make simplifications
    /// easier.
    Mul(FloatRatio<P>, Vec<Float<P>>),
    Reduction(FloatReduction<P>),
    Diff(Diff<P>),
}

impl<P> fmt::Display for FloatInner<P>
where
    P: Atom + fmt::Display,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        use itertools::Itertools;
        use FloatInner::*;

        match self {
            Mul(ratio, args) if args.is_empty() => write!(fmt, "{}", ratio),
            Mul(ratio, args) => {
                if !ratio.is_one() {
                    write!(fmt, "{}*", ratio)?;
                }

                write!(fmt, "{}", args.iter().format("*"))
            }
            Reduction(reduction) => write!(fmt, "{}", reduction),
            Diff(diff) => write!(fmt, "{}", diff),
        }
    }
}

impl<P> FloatInner<P>
where
    P: Atom,
{
    fn is_mul(&self) -> bool {
        if let FloatInner::Mul(_, _) = self {
            true
        } else {
            false
        }
    }

    fn is_diff(&self) -> bool {
        if let FloatInner::Diff(_) = self {
            true
        } else {
            false
        }
    }
}

#[derive(PartialEq)]
pub struct Float<P> {
    inner: Rc<FloatInner<P>>,
}

impl<P> fmt::Debug for Float<P>
where
    P: fmt::Debug,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&self.inner, fmt)
    }
}

impl<P> fmt::Display for Float<P>
where
    P: Atom + fmt::Display,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.inner, fmt)
    }
}

impl<P> From<Diff<P>> for Float<P>
where
    P: Atom,
{
    fn from(diff: Diff<P>) -> Self {
        FloatInner::Diff(diff).into()
    }
}

impl<P> From<FloatInner<P>> for Float<P>
where
    P: Atom,
{
    fn from(inner: FloatInner<P>) -> Self {
        Float {
            inner: Rc::new(inner),
        }
    }
}

impl<P> From<f64> for Float<P>
where
    P: Atom,
{
    fn from(constant: f64) -> Self {
        FloatInner::Mul(constant.into(), Vec::new()).into()
    }
}

impl<P> Clone for Float<P> {
    fn clone(&self) -> Self {
        Float {
            inner: Rc::clone(&self.inner),
        }
    }
}

impl<P> AsValue<f64> for Float<P>
where
    P: Atom,
{
    fn as_value(&self) -> Option<&f64> {
        match &*self.inner {
            FloatInner::Mul(ratio, args) if args.is_empty() => ratio.as_f64_ref(),
            _ => None,
        }
    }

    fn as_value_mut(&mut self) -> Option<&mut f64> {
        if self.as_value().is_some() {
            Some(match Rc::make_mut(&mut self.inner) {
                FloatInner::Mul(ratio, args) if args.is_empty() => {
                    ratio.as_f64_mut().unwrap()
                }
                _ => unreachable!(),
            })
        } else {
            None
        }
    }
}

impl<P> AsReduction<FloatReductionKind, Float<P>> for Float<P>
where
    P: Atom,
{
    type Value = f64;

    fn as_reduction(&self, kind: FloatReductionKind) -> Option<&FloatReduction<P>> {
        match &*self.inner {
            FloatInner::Reduction(reduction) if reduction.kind == kind => Some(reduction),
            _ => None,
        }
    }

    fn as_reduction_mut(
        &mut self,
        kind: FloatReductionKind,
    ) -> Option<&mut FloatReduction<P>> {
        if self.as_reduction(kind).is_some() {
            Some(match Rc::make_mut(&mut self.inner) {
                FloatInner::Reduction(reduction) if reduction.kind == kind => reduction,
                _ => unreachable!(),
            })
        } else {
            None
        }
    }
}

impl<P> From<FloatReduction<P>> for Float<P>
where
    P: Atom,
{
    fn from(reduction: FloatReduction<P>) -> Self {
        FloatInner::Reduction(reduction).into()
    }
}

impl<P> Float<P>
where
    P: Atom,
{
    fn fast_eq(lhs: &Self, rhs: &Self) -> bool {
        Rc::ptr_eq(&lhs.inner, &rhs.inner)
            || lhs
                .as_f64()
                .and_then(|lhs| rhs.as_f64().map(|rhs| lhs.to_bits() == rhs.to_bits()))
                .unwrap_or(false)
    }

    fn is_one(&self) -> bool {
        match &*self.inner {
            FloatInner::Mul(ratio, args) if args.is_empty() => ratio.is_one(),
            _ => false,
        }
    }

    fn as_diff(&self) -> Option<&Diff<P>> {
        match &*self.inner {
            FloatInner::Diff(diff) => Some(diff),
            _ => None,
        }
    }

    fn as_diff_mut(&mut self) -> Option<&mut Diff<P>> {
        if self.as_diff().is_some() {
            Some(match Rc::make_mut(&mut self.inner) {
                FloatInner::Diff(diff) => diff,
                _ => unreachable!(),
            })
        } else {
            None
        }
    }

    fn is_zero(&self) -> bool {
        self.as_value().map(Zero::is_zero).unwrap_or(false)
    }

    pub fn min_assign(&mut self, rhs: &Float<P>) {
        self.reduce_assign(FloatReductionKind::Min, rhs);
    }

    pub fn max_assign(&mut self, rhs: &Float<P>) {
        self.reduce_assign(FloatReductionKind::Max, rhs);
        if let Some(red) = self.as_reduction(FloatReductionKind::Max) {
            if red.others.len() == 1 && red.others[0].min_value() >= red.constant.unwrap()
            {
                *self = self
                    .as_reduction_mut(FloatReductionKind::Max)
                    .unwrap()
                    .others
                    .swap_remove(0);
            }
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match &*self.inner {
            FloatInner::Mul(ratio, args) if args.is_empty() => ratio.as_f64(),
            _ => None,
        }
    }

    pub fn to_f64(&self) -> Option<f64> {
        match &*self.inner {
            FloatInner::Mul(ratio, args) => ratio.to_f64().and_then(|ratio| {
                args.iter()
                    .map(|arg| arg.to_f64().ok_or(()))
                    .product::<Result<f64, ()>>()
                    .ok()
                    .map(|result| ratio * result)
            }),
            FloatInner::Reduction(_) => unimplemented!("to_f64 for {}", self),
            FloatInner::Diff(_) => unimplemented!("to_f64 for {}", self),
        }
    }

    pub fn min_value(&self) -> f64 {
        info!("min_value for {}", self);

        match &*self.inner {
            FloatInner::Mul(ratio, args) => {
                let mut min = ratio.min_value();
                for arg in args.iter() {
                    min *= arg.min_value();
                }
                min
            }
            FloatInner::Reduction(red) => red.min_value(),
            FloatInner::Diff(diff) => diff.min_value(),
        }
    }

    pub fn max_value(&self) -> f64 {
        info!("max_value for {}", self);

        match &*self.inner {
            FloatInner::Mul(ratio, args) => {
                let mut min = ratio.max_value();
                for arg in args.iter() {
                    min *= arg.max_value();
                }
                min
            }
            FloatInner::Reduction(red) => red.max_value(),
            FloatInner::Diff(diff) => diff.max_value(),
        }
    }
}

impl<'a, P> MulAssign<&'a Float<P>> for Float<P>
where
    P: Atom,
{
    fn mul_assign(&mut self, rhs: &'a Float<P>) {
        if rhs.is_one() || self.is_zero() {
            // Nothing to do
        } else if rhs.is_zero() {
            *self = rhs.clone();
        } else if self.inner.is_mul() {
            let inner_mut = Rc::make_mut(&mut self.inner);
            match inner_mut {
                FloatInner::Mul(lhs_ratio, lhs_args) => match &*rhs.inner {
                    FloatInner::Mul(rhs_ratio, rhs_args) => {
                        *lhs_ratio *= rhs_ratio;
                        lhs_args.extend(rhs_args.iter().cloned());
                    }
                    _ => lhs_args.push(rhs.clone()),
                },
                _ => unreachable!(),
            }
        } else if let FloatInner::Mul(rhs_ratio, rhs_args) = &*rhs.inner {
            *self = FloatInner::Mul(
                rhs_ratio.clone(),
                iter::once(self.clone())
                    .chain(rhs_args.iter().cloned())
                    .collect(),
            )
            .into();
        } else {
            *self = FloatInner::Mul(1f64.into(), vec![self.clone(), rhs.clone()]).into();
        }
    }
}

forward_binop_to_op_assign_commutative!(impl(P: Atom) Mul, mul for Float<P>, Float<P>, MulAssign, mul_assign);

impl<'a, P> MulAssign<&'a Int<P>> for Float<P>
where
    P: Atom,
{
    fn mul_assign(&mut self, rhs: &'a Int<P>) {
        *self *= rhs.to_symbolic_float();
    }
}

forward_binop_to_op_assign!(impl(P: Atom) Mul, mul for Float<P>, Int<P>, MulAssign, mul_assign);
forward_binop_to_ref_val_commutative!(impl(P: Atom) Mul<Output = Float<P>>, mul for Int<P>, Float<P>);

impl<'a, P> AddAssign<&'a Float<P>> for Float<P>
where
    P: Atom,
{
    fn add_assign(&mut self, rhs: &'a Float<P>) {
        if rhs.is_zero() {
            // Nothing to do
        } else if self.is_zero() {
            *self = rhs.clone();
        } else if let Some(lhs_val) = self.as_value() {
            if let Some(rhs_val) = rhs.as_value() {
                *self = (*lhs_val + *rhs_val).into();
            } else if let Some(rhs_diff) = rhs.as_diff() {
                *self = rhs_diff
                    .clone()
                    .add(*lhs_val, iter::empty(), iter::empty())
                    .into();
            } else {
                *self = Diff::new_constant(*lhs_val)
                    .add(0f64, iter::once(rhs), iter::empty())
                    .into();
            }
        } else if let Some(lhs_diff) = self.as_diff_mut() {
            if let Some(rhs_val) = rhs.as_value() {
                lhs_diff.add_assign(*rhs_val, iter::empty(), iter::empty());
            } else if let Some(rhs_diff) = rhs.as_diff() {
                lhs_diff.add_assign(
                    rhs_diff.constant,
                    rhs_diff.positive.iter(),
                    rhs_diff.negative.iter(),
                );
            } else {
                lhs_diff.add_assign(0f64, iter::once(rhs), iter::empty());
            }
        } else if let Some(rhs_val) = rhs.as_value() {
            *self = Diff::new_constant(*rhs_val)
                .add(0f64, iter::once(&*self), iter::empty())
                .into();
        } else if let Some(rhs_diff) = rhs.as_diff() {
            *self = rhs_diff
                .clone()
                .add(0f64, iter::once(&*self), iter::empty())
                .into();
        } else {
            *self = Diff::new_unknown(vec![self.clone(), rhs.clone()], Vec::new()).into();
        }
    }
}

forward_binop_to_op_assign_commutative!(impl(P: Atom) Add, add for Float<P>, Float<P>, AddAssign, add_assign);

impl<'a, P> DivAssign<&'a Int<P>> for Float<P>
where
    P: Atom,
{
    fn div_assign(&mut self, rhs: &'a Int<P>) {
        self.mul_assign(&Float::from(FloatInner::Mul(
            FloatRatio::recip(rhs.clone()),
            Vec::new(),
        )));
    }
}

forward_binop_to_op_assign!(impl(P: Atom) Div, div for Float<P>, Int<P>, DivAssign, div_assign);

impl<'a, P> DivAssign<&'a f64> for Float<P>
where
    P: Atom,
{
    fn div_assign(&mut self, rhs: &'a f64) {
        self.mul_assign(&Float::from(rhs.recip()));
    }
}

forward_binop_to_op_assign!(impl(P: Atom) Div, div for Float<P>, f64, DivAssign, div_assign);

impl<'a, P> SubAssign<&'a Float<P>> for Float<P>
where
    P: Atom,
{
    fn sub_assign(&mut self, rhs: &'a Float<P>) {
        if self.is_zero() {
            *self = rhs.clone();
        } else if rhs.is_zero() {
            // Nothing to do
        } else if self == rhs {
            *self = 0f64.into()
        } else if let Some(lhs_val) = self.as_value() {
            if let Some(rhs_val) = rhs.as_value() {
                *self = (*lhs_val - *rhs_val).into();
            } else if let Some(rhs_diff) = rhs.as_diff() {
                *self = rhs_diff
                    .clone()
                    .neg()
                    .add(*lhs_val, iter::empty(), iter::empty())
                    .into();
            } else {
                *self = Diff::new_constant(*lhs_val)
                    .add(0f64, iter::empty(), iter::once(rhs))
                    .into();
            }
        } else if let Some(lhs_diff) = self.as_diff_mut() {
            if let Some(rhs_val) = rhs.as_value() {
                lhs_diff.add_assign(-*rhs_val, iter::empty(), iter::empty());
            } else if let Some(rhs_diff) = rhs.as_diff() {
                lhs_diff.add_assign(
                    -rhs_diff.constant,
                    rhs_diff.negative.iter(),
                    rhs_diff.positive.iter(),
                );
            } else {
                lhs_diff.add_assign(0f64, iter::empty(), iter::once(rhs));
            }
        } else if let Some(rhs_val) = rhs.as_value() {
            *self = Diff::new_constant(*rhs_val)
                .add(0f64, iter::empty(), iter::once(&*self))
                .neg()
                .into();
        } else if let Some(rhs_diff) = rhs.as_diff() {
            *self = rhs_diff
                .clone()
                .add(0f64, iter::empty(), iter::once(&*self))
                .neg()
                .into();
        } else {
            *self = Diff::new_unknown(vec![self.clone()], vec![rhs.clone()]).into();
        }
    }
}

forward_binop_to_op_assign!(impl(P: Atom) Sub, sub for Float<P>, Float<P>, SubAssign, sub_assign);

impl<'a, P> SubAssign<&'a f64> for Float<P>
where
    P: Atom,
{
    fn sub_assign(&mut self, rhs: &'a f64) {
        *self -= Float::from(*rhs);
    }
}

forward_binop_to_op_assign!(impl(P: Atom) Sub, sub for Float<P>, f64, SubAssign, sub_assign);
