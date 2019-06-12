use log::{info, trace};
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::iter;
use std::ops::{self, Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
use std::rc::Rc;

use num::{BigUint, Integer, One, ToPrimitive, Zero};

mod hash;

fn is_close(lhs: f64, rhs: f64) -> bool {
    (lhs - rhs).abs() < 1e-8 + 1e-5 * rhs
}

pub trait Range<N = u64> {
    fn min_value(&self) -> N;

    fn max_value(&self) -> N;
}

pub trait Factor<N = u64> {
    fn gcd_value(&self) -> N;

    fn lcm_value(&self) -> N;
}

pub trait Atom:
    Range + Factor + Clone + fmt::Debug + fmt::Display + PartialEq + Ord + Hash
{
}

impl<T> Atom for T where
    T: ?Sized
        + Range
        + Factor
        + Clone
        + fmt::Debug
        + fmt::Display
        + PartialEq
        + Ord
        + Hash
{
}

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
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct RatioInner<P> {
    factor: BigUint,
    numer: Vec<P>,
    denom: Vec<P>,
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

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Ratio<P> {
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

impl<P> fmt::Display for Ratio<P>
where
    P: Atom + fmt::Display,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.inner, fmt)
    }
}

impl<P> Clone for Ratio<P> {
    fn clone(&self) -> Self {
        Ratio {
            inner: Rc::clone(&self.inner),
        }
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

impl<'a, P: 'a> iter::Product<&'a Ratio<P>> for Ratio<P>
where
    P: Atom,
{
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Ratio<P>>,
    {
        let mut res = Self::one();
        for elem in iter {
            res *= elem;
        }
        res
    }
}

impl<P> Ratio<P>
where
    P: Atom,
{
    pub fn new(factor: BigUint, mut numer: Vec<P>, mut denom: Vec<P>) -> Self {
        numer.sort();
        denom.sort();

        RatioInner {
            factor,
            numer,
            denom,
        }
        .into()
    }

    pub fn one() -> Self {
        Self::new(1u32.into(), Vec::new(), Vec::new())
    }

    pub fn to_symbolic_float(&self) -> Float<P> {
        FloatInner::Mul(self.clone().into(), Vec::new()).into()
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

    fn as_biguint(&self) -> Option<&BigUint> {
        if self.inner.numer.is_empty() && self.inner.denom.is_empty() {
            Some(&self.inner.factor)
        } else {
            None
        }
    }

    fn is_multiple_of(&self, other: &Ratio<P>) -> bool {
        use itertools::{EitherOrBoth::*, Itertools};

        let (left_gcd, right_lcm) =
            (self.inner.factor.clone(), other.inner.factor.clone());
        let (left_gcd, right_lcm) = self
            .inner
            .numer
            .iter()
            .merge_join_by(&other.inner.numer, |lhs, rhs| lhs.cmp(rhs))
            .fold(
                (left_gcd, right_lcm),
                |(left_gcd, right_lcm), either| match either {
                    Left(lhs) => (left_gcd * lhs.gcd_value(), right_lcm),
                    Right(rhs) => (left_gcd, right_lcm * rhs.lcm_value()),
                    Both(_, _) => (left_gcd, right_lcm),
                },
            );
        let (left_gcd, right_lcm) = self
            .inner
            .denom
            .iter()
            .merge_join_by(&other.inner.denom, |lhs, rhs| lhs.cmp(rhs))
            .fold(
                (left_gcd, right_lcm),
                |(left_gcd, right_lcm), either| match either {
                    Left(lhs) => (left_gcd / lhs.lcm_value(), right_lcm),
                    Right(rhs) => (left_gcd, right_lcm / rhs.gcd_value()),
                    Both(_, _) => (left_gcd, right_lcm),
                },
            );
        left_gcd.is_multiple_of(&right_lcm)
    }

    fn is_less_than(&self, other: &Ratio<P>) -> bool {
        // TODO
        self.max_value() <= other.min_value()
    }
}

impl<P> Range for RatioInner<P>
where
    P: Range,
{
    fn min_value(&self) -> u64 {
        let factor = self
            .factor
            .to_u64()
            .unwrap_or_else(|| panic!("Unable to represent factor as u64"));
        let numer_min = self.numer.iter().map(Range::min_value).product::<u64>();
        let denom_max = self.denom.iter().map(Range::max_value).product::<u64>();
        (factor * numer_min) / denom_max
    }

    fn max_value(&self) -> u64 {
        let factor = self
            .factor
            .to_u64()
            .unwrap_or_else(|| panic!("Unable to represent factor as u64"));
        let numer_max = self.numer.iter().map(Range::max_value).product::<u64>();
        let denom_min = self.denom.iter().map(Range::min_value).product::<u64>();
        (factor * numer_max) / denom_min
    }
}

impl<P> Factor for RatioInner<P>
where
    P: Factor,
{
    fn gcd_value(&self) -> u64 {
        let factor = self
            .factor
            .to_u64()
            .unwrap_or_else(|| panic!("Unable to represent factor as u64"));
        let numer_gcd = self.numer.iter().map(Factor::gcd_value).product::<u64>();
        let denom_lcm = self.denom.iter().map(Factor::lcm_value).product::<u64>();
        assert!((factor * numer_gcd).is_multiple_of(&denom_lcm));

        (factor * numer_gcd) / denom_lcm
    }

    fn lcm_value(&self) -> u64 {
        let factor = self
            .factor
            .to_u64()
            .unwrap_or_else(|| panic!("Unable to represent factor as u64"));
        let numer_lcm = self.numer.iter().map(Factor::lcm_value).product::<u64>();
        let denom_gcd = self.denom.iter().map(Factor::gcd_value).product::<u64>();
        assert!((factor * numer_lcm).is_multiple_of(&denom_gcd));

        (factor * numer_lcm) / denom_gcd
    }
}

impl<P> Range for Ratio<P>
where
    P: Range,
{
    fn min_value(&self) -> u64 {
        self.inner.min_value()
    }

    fn max_value(&self) -> u64 {
        self.inner.max_value()
    }
}

impl<P> Factor for Ratio<P>
where
    P: Factor,
{
    fn gcd_value(&self) -> u64 {
        self.inner.gcd_value()
    }

    fn lcm_value(&self) -> u64 {
        self.inner.lcm_value()
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

        lhs.numer.sort();
        lhs.denom.sort();
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
        + Ord
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

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct Reduction<K, C, S> {
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
        S: fmt::Display + Ord + Clone + 'a,
    {
        self.reduce_assign(constant, others);
        self
    }

    fn reduce_assign<'a, I>(&mut self, constant: Option<&C>, others: I)
    where
        I: Iterator<Item = &'a S>,
        C: fmt::Display + Clone,
        S: fmt::Display + Ord + Clone + 'a,
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
            );
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

        self.simplify();
    }

    fn simplify(&mut self)
    where
        S: Ord,
    {
        self.others.sort();
        self.others.dedup();
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Ord, PartialOrd, Hash)]
pub struct LcmExpr<P> {
    gcd: BigUint,
    lcm: BigUint,
    args: VecSet<Ratio<P>>,
}

impl<P> fmt::Display for LcmExpr<P>
where
    P: Atom,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        use itertools::Itertools;

        if self.args.is_empty() {
            write!(fmt, "lcm({})", self.lcm)
        } else {
            write!(fmt, "lcm({}, {})", self.lcm, self.args.iter().format(", "))
        }
    }
}

impl<P> From<LcmExpr<P>> for Int<P>
where
    P: Atom,
{
    fn from(lmin: LcmExpr<P>) -> Self {
        lmin.simplified()
            .map(|ratio| IntInner::Mul(ratio).into())
            .unwrap_or_else(|lcm| IntInner::Lcm(lcm).into())
    }
}

impl<P> From<Ratio<P>> for LcmExpr<P>
where
    P: Atom,
{
    fn from(value: Ratio<P>) -> Self {
        LcmExpr {
            gcd: value.gcd_value().into(),
            lcm: value.lcm_value().into(),
            args: VecSet::from_sorted_iter(iter::once(value)),
        }
    }
}

impl<P> LcmExpr<P>
where
    P: Atom,
{
    pub fn new<II>(iter: II) -> Option<Self>
    where
        II: IntoIterator<Item = Ratio<P>>,
    {
        let mut iter = iter.into_iter();
        let mut args: Vec<Ratio<P>> = Vec::with_capacity(iter.size_hint().1.unwrap_or(0));

        if let Some(first) = iter.next() {
            let mut gcd = BigUint::from(first.gcd_value());
            let mut lcm = BigUint::from(first.lcm_value());

            'elem: for elem in iter {
                let mut to_remove = vec![];
                for (ix, arg) in args.iter().enumerate() {
                    if arg.is_multiple_of(&elem) {
                        continue 'elem;
                    }

                    if elem.is_multiple_of(&arg) {
                        to_remove.push(ix);
                    }
                }

                // Need to iterate backwards so that indices stay valid
                for ix in to_remove.into_iter().rev() {
                    args.remove(ix);
                }

                gcd = gcd.lcm(&BigUint::from(elem.gcd_value()));
                lcm = lcm.lcm(&BigUint::from(elem.lcm_value()));
                args.push(elem);
            }

            Some(LcmExpr {
                gcd,
                lcm,
                args: args.into_iter().collect(),
            })
        } else {
            None
        }
    }

    pub fn one() -> Self {
        LcmExpr {
            gcd: 1u32.into(),
            lcm: 1u32.into(),
            args: VecSet::default(),
        }
    }

    fn as_biguint(&self) -> Option<&BigUint> {
        if self.is_constant() {
            Some(&self.lcm)
        } else {
            None
        }
    }

    fn to_u32(&self) -> Option<u32> {
        self.as_biguint().map(|value| {
            value
                .to_u32()
                .expect("Unable to convert constant lcm to u32")
        })
    }

    fn is_constant(&self) -> bool {
        self.gcd == self.lcm
    }

    fn is_single_value(&self) -> bool {
        self.args.len() == 1 && self.gcd == BigUint::from(self.args[0].gcd_value())
    }

    fn can_simplify(&self) -> bool {
        return self.is_constant() || self.is_single_value();
    }

    fn simplified(self) -> Result<Ratio<P>, Self> {
        if self.is_constant() {
            Ok(Ratio::new(self.gcd, Vec::new(), Vec::new()))
        } else if self.is_single_value() {
            Ok(self.args.values[0].clone())
        } else {
            Err(self)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Ord, PartialOrd, Hash)]
pub struct MinExpr<P> {
    min: BigUint,
    max: BigUint,
    values: VecSet<Ratio<P>>,
}

impl<P> fmt::Display for MinExpr<P>
where
    P: Atom,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        use itertools::Itertools;

        if let Some(minmax) = self.values.iter().map(Range::max_value).min() {
            if BigUint::from(minmax) == self.max {
                write!(fmt, "min({})", self.values.iter().format(", "))
            } else {
                write!(
                    fmt,
                    "min({}, {})",
                    self.max,
                    self.values.iter().format(", ")
                )
            }
        } else {
            assert!(self.values.is_empty());
            assert_eq!(self.min, self.max);

            write!(fmt, "min({})", self.min)
        }
    }
}

impl<P> From<MinExpr<P>> for Int<P>
where
    P: Atom,
{
    fn from(emin: MinExpr<P>) -> Self {
        emin.simplified()
            .map(|ratio| IntInner::Mul(ratio).into())
            .unwrap_or_else(|min| IntInner::Min(min).into())
    }
}

impl<P> From<Ratio<P>> for Int<P>
where
    P: Atom,
{
    fn from(ratio: Ratio<P>) -> Self {
        IntInner::Mul(ratio).into()
    }
}

impl<P> From<Ratio<P>> for MinExpr<P>
where
    P: Atom,
{
    fn from(value: Ratio<P>) -> Self {
        MinExpr {
            min: value.min_value().into(),
            max: value.max_value().into(),
            values: VecSet::from_sorted_iter(iter::once(value)),
        }
    }
}

impl<P> From<u32> for MinExpr<P>
where
    P: Atom,
{
    fn from(constant: u32) -> Self {
        MinExpr {
            min: constant.into(),
            max: constant.into(),
            values: VecSet::default(),
        }
    }
}

impl<P> MinExpr<P>
where
    P: Atom,
{
    pub fn new<II>(iter: II) -> Option<Self>
    where
        II: IntoIterator<Item = Ratio<P>>,
    {
        let mut iter = iter.into_iter();
        let mut args: Vec<Ratio<P>> = Vec::with_capacity(iter.size_hint().1.unwrap_or(0));

        if let Some(first) = iter.next() {
            let mut min = BigUint::from(first.min_value());
            let mut max = BigUint::from(first.max_value());

            'elem: for elem in iter {
                if max <= BigUint::from(elem.min_value()) {
                    continue;
                }

                let mut to_remove = vec![];
                for (ix, arg) in args.iter().enumerate() {
                    if arg.is_less_than(&elem) {
                        continue 'elem;
                    }

                    if elem.is_less_than(&arg) {
                        to_remove.push(ix);
                    }
                }

                // Need to iterate backwards so that indices stay valid
                for ix in to_remove.into_iter().rev() {
                    args.remove(ix);
                }

                min = min.min(BigUint::from(elem.min_value()));
                max = max.min(BigUint::from(elem.max_value()));
                args.push(elem);
            }

            Some(MinExpr {
                min,
                max,
                values: args.into_iter().collect(),
            })
        } else {
            None
        }
    }

    pub fn one() -> Self {
        1u32.into()
    }

    pub fn to_symbolic_float(&self) -> Float<P> {
        self.clone()
            .simplified()
            .map(|ratio| ratio.to_symbolic_float())
            .unwrap_or_else(|emin| {
                FloatInner::Reduction(FloatReduction {
                    kind: FloatReductionKind::Min,
                    constant: Some(emin.min.to_u32().unwrap() as f64),
                    others: emin
                        .values
                        .iter()
                        .cloned()
                        .map(FloatRatioInner::from)
                        .map(FloatRatio::from)
                        .map(|ratio| FloatInner::Mul(ratio, Vec::new()).into())
                        .collect(),
                })
                .into()
            })
    }

    fn as_biguint(&self) -> Option<&BigUint> {
        if self.is_constant() {
            debug_assert_eq!(self.min, self.max);

            Some(&self.min)
        } else {
            None
        }
    }

    fn to_u32(&self) -> Option<u32> {
        self.as_biguint().map(|value| value.to_u32().unwrap())
    }

    fn is_constant(&self) -> bool {
        self.min == self.max
    }

    fn is_single_value(&self) -> bool {
        self.values.len() == 1 && self.max == BigUint::from(self.values[0].max_value())
    }

    fn can_simplify(&self) -> bool {
        return self.is_constant() || self.is_single_value();
    }

    fn simplified(self) -> Result<Ratio<P>, Self> {
        if self.is_constant() {
            debug_assert_eq!(self.min, self.max);

            Ok(Ratio::new(self.min, Vec::new(), Vec::new()))
        } else if self.is_single_value() {
            debug_assert!(self.values.len() == 1);

            Ok(self.values.values[0].clone())
        } else {
            Err(self)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum IntInner<P> {
    Lcm(LcmExpr<P>),
    Min(MinExpr<P>),
    Mul(Ratio<P>),
}

impl<P> fmt::Display for IntInner<P>
where
    P: Atom + fmt::Display,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            IntInner::Lcm(lcm) => write!(fmt, "{}", lcm),
            IntInner::Min(min) => write!(fmt, "{}", min),
            IntInner::Mul(ratio) => write!(fmt, "{}", ratio),
        }
    }
}

impl<P> IntInner<P>
where
    P: Atom,
{
    fn is_mul(&self) -> bool {
        if let IntInner::Mul(_) = self {
            true
        } else {
            false
        }
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash)]
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

impl<P> Int<P>
where
    P: Atom,
{
    pub fn to_symbolic_float(&self) -> Float<P> {
        match &*self.inner {
            IntInner::Min(emin) => emin.to_symbolic_float(),
            IntInner::Mul(ratio) => ratio.to_symbolic_float(),
            IntInner::Lcm(_) => unimplemented!("Can't convert {} to float", self),
        }
    }

    pub fn ratio<T: Into<BigUint>>(factor: T, numer: Vec<P>, denom: Vec<P>) -> Self {
        IntInner::Mul(Ratio::new(factor.into(), numer, denom)).into()
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
            IntInner::Mul(ratio) => ratio.as_biguint(),
            IntInner::Min(emin) => emin.as_biguint(),
            IntInner::Lcm(elcm) => elcm.as_biguint(),
        }
    }

    fn simplify(&mut self) {
        match &*self.inner {
            IntInner::Lcm(lcm) if lcm.can_simplify() => {
                *self = IntInner::Mul(lcm.clone().simplified().unwrap()).into();
            }
            IntInner::Min(emin) if emin.can_simplify() => {
                *self = IntInner::Mul(emin.clone().simplified().unwrap()).into();
            }
            _ => (),
        }
    }

    pub fn to_u32(&self) -> Option<u32> {
        match &*self.inner {
            IntInner::Min(emin) => emin.to_u32(),
            IntInner::Lcm(elcm) => elcm.to_u32(),
            IntInner::Mul(ratio) => ratio.to_u32(),
        }
    }
}

impl<P> Range for Int<P>
where
    P: Atom,
{
    fn min_value(&self) -> u64 {
        match &*self.inner {
            IntInner::Lcm(elcm) => unimplemented!("min_value for {}", elcm), // elcm.gcd.to_u64().unwrap(),
            IntInner::Min(emin) => emin.min.to_u64().unwrap(),
            IntInner::Mul(ratio) => ratio.min_value(),
        }
    }

    fn max_value(&self) -> u64 {
        match &*self.inner {
            IntInner::Lcm(elcm) => unimplemented!("max_value for {}", elcm), // elcm.lcm.to_u64().unwrap(),
            IntInner::Min(emin) => emin.max.to_u64().unwrap(),
            IntInner::Mul(ratio) => ratio.max_value(),
        }
    }
}

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
                IntInner::Mul(lhs_ratio) => match &*rhs.inner {
                    IntInner::Mul(rhs_ratio) => {
                        *lhs_ratio *= rhs_ratio;
                    }
                    _ => unimplemented!("mul of {} and {}", self, rhs),
                },
                _ => unreachable!(),
            }
        } else if let IntInner::Mul(rhs_ratio) = &*rhs.inner {
            unimplemented!("mul of {} and {}", self, rhs);
        } else {
            unimplemented!("mul of {} and {}", self, rhs);
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
#[derive(Debug, Clone, PartialEq, PartialOrd)]
struct FloatRatioInner<P> {
    factor: f64,
    numer: Vec<P>,
    denom: Vec<P>,
    // should be: factor * float(numer/denom) * float(numer)/float(denom)
}

impl<P> From<Ratio<P>> for FloatRatioInner<P>
where
    P: Atom,
{
    fn from(ratio: Ratio<P>) -> Self {
        FloatRatioInner {
            factor: ratio.inner.factor.to_u64().unwrap() as f64,
            numer: ratio.inner.numer.clone(),
            denom: ratio.inner.denom.clone(),
        }
    }
}

impl<P> Eq for FloatRatioInner<P> where P: Eq {}

impl<P> Ord for FloatRatioInner<P>
where
    P: Ord,
{
    fn cmp(&self, other: &FloatRatioInner<P>) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl<P> Hash for FloatRatioInner<P>
where
    P: Hash,
{
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.factor.to_bits().hash(state);
        self.numer.hash(state);
        self.denom.hash(state);
    }
}

impl<'a, P> MulAssign<&'a FloatRatioInner<P>> for FloatRatioInner<P>
where
    P: Atom,
{
    fn mul_assign(&mut self, rhs: &'a FloatRatioInner<P>) {
        self.factor *= &rhs.factor;
        for numer in rhs.numer.iter() {
            if let Some(pos) = self.denom.iter().position(|d| d == numer) {
                self.denom.swap_remove(pos);
            } else {
                self.numer.push(numer.clone());
            }
        }
        for denom in rhs.denom.iter() {
            if let Some(pos) = self.numer.iter().position(|n| n == denom) {
                self.numer.swap_remove(pos);
            } else {
                self.denom.push(denom.clone());
            }
        }

        self.numer.sort();
        self.denom.sort();
    }
}

forward_binop_to_op_assign_commutative!(impl(P: Atom) Mul, mul for FloatRatioInner<P>, FloatRatioInner<P>, MulAssign, mul_assign);

impl<P> fmt::Display for FloatRatioInner<P>
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

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FloatRatio<P> {
    inner: Rc<FloatRatioInner<P>>,
}

impl<P> From<Ratio<P>> for FloatRatio<P>
where
    P: Atom,
{
    fn from(ratio: Ratio<P>) -> Self {
        FloatRatio {
            inner: Rc::new(ratio.into()),
        }
    }
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
            numer: Vec::new(),
            denom: Vec::new(),
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

impl<P> FloatRatio<P>
where
    P: Atom,
{
    fn new_constant(factor: f64) -> Self {
        FloatRatio::new(factor, Vec::new(), Vec::new())
    }

    fn new(factor: f64, numer: Vec<P>, denom: Vec<P>) -> Self {
        FloatRatioInner {
            factor,
            numer,
            denom,
        }
        .into()
    }

    fn is_one(&self) -> bool {
        self.inner.factor == 1f64
            && self.inner.numer.is_empty()
            && self.inner.denom.is_empty()
    }

    fn is_zero(&self) -> bool {
        self.inner.factor == 0f64
            && self.inner.numer.is_empty()
            && self.inner.denom.is_empty()
    }

    fn as_f64_ref(&self) -> Option<&f64> {
        if self.inner.numer.is_empty() && self.inner.denom.is_empty() {
            Some(&self.inner.factor)
        } else {
            None
        }
    }

    fn as_f64_mut(&mut self) -> Option<&mut f64> {
        if self.inner.numer.is_empty() && self.inner.denom.is_empty() {
            Some(&mut Rc::make_mut(&mut self.inner).factor)
        } else {
            None
        }
    }

    fn as_f64(&self) -> Option<f64> {
        if self.inner.numer.is_empty() && self.inner.denom.is_empty() {
            Some(self.inner.factor)
        } else {
            None
        }
    }

    fn to_f64(&self) -> Option<f64> {
        self.as_f64()
    }

    fn min_value(&self) -> f64
    where
        P: Atom,
    {
        let min = self.inner.factor;
        let min = self
            .inner
            .numer
            .iter()
            .fold(min, |min, n| min * n.min_value() as f64);
        self.inner
            .denom
            .iter()
            .fold(min, |min, d| min / d.max_value() as f64)
    }

    fn max_value(&self) -> f64
    where
        P: Atom,
    {
        let max = self.inner.factor;
        let max = self
            .inner
            .numer
            .iter()
            .fold(max, |max, n| max * n.max_value() as f64);
        self.inner
            .denom
            .iter()
            .fold(max, |max, d| max / d.min_value() as f64)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DiffExpr<P>
where
    P: Atom,
{
    constant: f64,
    values: WeightedVecSet<Float<P>>,
}

impl<P> Eq for DiffExpr<P> where P: Atom + Eq {}

/*
impl<P> PartialOrd for DiffExpr<P>
where
    P: Atom,
{
    fn partial_cmp(&self, other: &DiffExpr<P>) -> Option<Ordering> {
        let mut seen_lt = false;
        let mut seen_gt = false;

        if !is_close(self.constant, other.constant) {
            if self.constant < other.constant {
                seen_lt = true;
            } else {
                seen_gt = true;
            }
        }

        for (self_weight, other_weight, _) in
            WeightedZip::new(self.values.iter(), other.values.iter())
        {
            if !is_close(self_weight, other_weight) {
                if self_weight < other_weight {
                    if seen_gt {
                        return None;
                    }
                    seen_lt = true;
                } else {
                    if seen_lt {
                        return None;
                    }
                    seen_gt = true;
                }
            }
        }

        assert!(!seen_lt || !seen_gt);

        Some(if seen_lt {
            Ordering::Less
        } else if seen_gt {
            Ordering::Greater
        } else {
            Ordering::Equal
        })
    }
}
*/

impl<P> PartialOrd for DiffExpr<P>
where
    P: Atom + PartialOrd,
{
    fn partial_cmp(&self, other: &DiffExpr<P>) -> Option<Ordering> {
        self.values.partial_cmp(&other.values).map(|ord| {
            ord.then_with(|| self.constant.partial_cmp(&other.constant).unwrap())
        })
    }
}

impl<P> Ord for DiffExpr<P>
where
    P: Atom + Ord,
{
    fn cmp(&self, other: &DiffExpr<P>) -> Ordering {
        self.values
            .cmp(&other.values)
            .then_with(|| self.constant.partial_cmp(&other.constant).unwrap())
    }
}

impl<P> Hash for DiffExpr<P>
where
    P: Atom,
{
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.constant.to_bits().hash(state);
        self.values.hash(state);
    }
}

impl<P> fmt::Display for DiffExpr<P>
where
    P: Atom + fmt::Display,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        use itertools::Itertools;

        if self.values.is_empty() {
            write!(fmt, "{}", self.constant)
        } else {
            let (positive, negative) = self
                .values
                .iter()
                .partition::<Vec<_>, _>(|(weight, _)| weight > &0.);

            let mut has_written = false;
            if !self.constant.is_zero() {
                write!(fmt, "{}", self.constant)?;

                has_written = true;
            }

            if !positive.is_empty() {
                write!(
                    fmt,
                    "{}{}",
                    if has_written { " + " } else { "" },
                    positive
                        .iter()
                        .map(|(weight, item)| format!("{}*{}", weight, item))
                        .format(" + "),
                )?;

                has_written = true;
            }

            if !negative.is_empty() {
                write!(
                    fmt,
                    "{}{}",
                    if has_written { " - " } else { "-" },
                    negative
                        .iter()
                        .map(|(weight, item)| format!("{}*{}", -weight, item))
                        .format(" - ")
                )?;
            }

            Ok(())
        }
    }
}

impl<'a, 'b, P> ops::Add<&'a f64> for &'b DiffExpr<P>
where
    P: Atom,
{
    type Output = DiffExpr<P>;

    fn add(self, other: &'a f64) -> DiffExpr<P> {
        let result = DiffExpr {
            constant: self.constant + other,
            values: self.values.clone(),
        };

        #[cfg(verify)]
        {
            assert!(is_close(self.min_value() + other, result.min_value()));
            assert!(is_close(self.max_value() + other, result.max_value()));
        }

        result
    }
}

forward_binop_to_ref_ref!(impl(P: Atom) Add<Output = DiffExpr<P>>, add for DiffExpr<P>, f64);

impl<'a, 'b, P> ops::Add<&'a DiffExpr<P>> for &'b DiffExpr<P>
where
    P: Atom,
{
    type Output = DiffExpr<P>;

    fn add(self, other: &'a DiffExpr<P>) -> DiffExpr<P> {
        let result = DiffExpr {
            constant: self.constant + other.constant,
            values: self.values.union(&other.values),
        };

        #[cfg(verify)]
        {
            assert!(is_close(
                self.min_value() + other.min_value(),
                result.min_value()
            ));
            assert!(is_close(
                self.max_value() + other.max_value(),
                result.max_value()
            ));
        }

        result
    }
}

forward_binop_to_ref_ref!(impl(P: Atom) Add<Output = DiffExpr<P>>, add for DiffExpr<P>, DiffExpr<P>);

impl<'a, 'b, P> ops::Sub<&'a f64> for &'b DiffExpr<P>
where
    P: Atom,
{
    type Output = DiffExpr<P>;

    fn sub(self, other: &'a f64) -> DiffExpr<P> {
        let result = DiffExpr {
            constant: self.constant - other,
            values: self.values.clone(),
        };

        #[cfg(verify)]
        {
            assert!(is_close(self.min_value() - other, result.min_value()));
            assert!(is_close(self.max_value() - other, result.max_value()));
        }

        result
    }
}

forward_binop_to_ref_ref!(impl(P: Atom) Sub<Output = DiffExpr<P>>, sub for DiffExpr<P>, f64);

impl<'a, 'b, P> ops::Sub<&'a DiffExpr<P>> for &'b DiffExpr<P>
where
    P: Atom,
{
    type Output = DiffExpr<P>;

    fn sub(self, other: &'a DiffExpr<P>) -> DiffExpr<P> {
        let result = DiffExpr {
            constant: self.constant - other.constant,
            values: self
                .values
                .union_sorted(other.values.iter().map(|(weight, item)| (-weight, item))),
        };

        #[cfg(verify)]
        {
            assert!(is_close(
                self.min_value() - other.max_value(),
                result.min_value()
            ));
            assert!(is_close(
                self.max_value() - other.min_value(),
                result.max_value()
            ));
        }

        result
    }
}

forward_binop_to_ref_ref!(impl(P: Atom) Sub<Output = DiffExpr<P>>, sub for DiffExpr<P>, DiffExpr<P>);

impl<'a, 'b, P> ops::Mul<&'a f64> for &'b DiffExpr<P>
where
    P: Atom,
{
    type Output = DiffExpr<P>;

    fn mul(self, other: &'a f64) -> DiffExpr<P> {
        let result = DiffExpr {
            constant: self.constant * other,
            values: WeightedVecSet::from_sorted_iter(
                self.values
                    .iter()
                    .map(|(weight, item)| (weight * other, item.clone())),
            ),
        };

        #[cfg(verify)]
        {
            if other > 0. {
                assert!(is_close(self.min_value() * other, result.min_value()));
                assert!(is_close(self.max_value() * other, result.max_value()));
            } else {
                assert!(is_close(self.min_value() * other, result.max_value()));
                assert!(is_close(self.max_value() * other, result.min_value()));
            }
        }

        result
    }
}

forward_binop_to_ref_ref!(impl(P: Atom) Mul<Output = DiffExpr<P>>, mul for DiffExpr<P>, f64);

impl<P> AsValue<f64> for DiffExpr<P>
where
    P: Atom,
{
    fn as_value(&self) -> Option<&f64> {
        if self.values.is_empty() {
            Some(&self.constant)
        } else {
            None
        }
    }

    fn as_value_mut(&mut self) -> Option<&mut f64> {
        if self.values.is_empty() {
            Some(&mut self.constant)
        } else {
            None
        }
    }
}

impl<P> From<f64> for DiffExpr<P>
where
    P: Atom,
{
    fn from(constant: f64) -> Self {
        DiffExpr {
            constant,
            values: Default::default(),
        }
    }
}

impl<P> From<Float<P>> for DiffExpr<P>
where
    P: Atom,
{
    fn from(value: Float<P>) -> Self {
        match value.as_f64() {
            Some(val) => DiffExpr {
                constant: val,
                values: Default::default(),
            },
            None => match &*value {
                FloatInner::Diff(ediff) => ediff.clone(),
                FloatInner::Mul(ratio, args) => DiffExpr {
                    constant: 0.,
                    values: WeightedVecSet::from_sorted_iter(iter::once((
                        ratio.inner.factor,
                        FloatInner::Mul(
                            FloatRatio::new(
                                1.,
                                ratio.inner.numer.clone(),
                                ratio.inner.denom.clone(),
                            ),
                            args.clone(),
                        )
                        .into(),
                    ))),
                },
                _ => DiffExpr {
                    constant: 0.,
                    values: WeightedVecSet::from_sorted_iter(iter::once((1., value))),
                },
            },
        }
    }
}

impl<P> DiffExpr<P>
where
    P: Atom,
{
    fn min_value(&self) -> f64
    where
        P: Atom,
    {
        self.constant
            + self
                .values
                .iter()
                .map(|(weight, value)| {
                    assert!(value.min_value() >= 0.);

                    if weight < 0. {
                        weight * value.max_value()
                    } else {
                        weight * value.min_value()
                    }
                })
                .sum::<f64>()
    }

    fn max_value(&self) -> f64
    where
        P: Atom,
    {
        self.constant
            + self
                .values
                .iter()
                .map(|(weight, value)| {
                    assert!(value.min_value() >= 0.);

                    if weight < 0. {
                        weight * value.min_value()
                    } else {
                        weight * value.max_value()
                    }
                })
                .sum::<f64>()
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FloatReductionKind {
    Min,
}

impl fmt::Display for FloatReductionKind {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FloatReductionKind::Min => write!(fmt, "min"),
        }
    }
}

impl<P> Hash for FloatReduction<P>
where
    P: Atom + Hash,
{
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.kind.hash(state);
        self.constant.map(f64::to_bits).hash(state);
        self.others.hash(state);
    }
}

impl ReductionKind<f64> for FloatReductionKind {
    fn reduce(&self, lhs: &f64, rhs: &f64) -> f64 {
        match self {
            FloatReductionKind::Min => lhs.min(*rhs),
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
        }
    }

    fn get_bound(&self, other: &Float<P>) -> Option<f64> {
        info!("Getting bound from {}", other);
        Some(match self {
            FloatReductionKind::Min => other.max_value(),
        })
    }
}

type FloatReduction<P> = Reduction<FloatReductionKind, f64, Float<P>>;

impl<P> Eq for FloatReduction<P> where P: Atom + Eq {}

impl<P> Ord for FloatReduction<P>
where
    P: Atom + Ord,
{
    fn cmp(&self, other: &FloatReduction<P>) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

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
        }
    }

    fn max_value(&self) -> f64 {
        match self.kind {
            FloatReductionKind::Min => self.constant.unwrap(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct FMaxExpr<P>
where
    P: Atom,
{
    min: f64,
    max: f64,
    values: VecSet<DiffExpr<P>>,
}

impl<P> Eq for FMaxExpr<P> where P: Atom + Eq {}

impl<P> Ord for FMaxExpr<P>
where
    P: Atom + Ord,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl<P> Hash for FMaxExpr<P>
where
    P: Atom + Hash,
{
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.values.hash(state);
    }
}

impl<P> fmt::Display for FMaxExpr<P>
where
    P: Atom,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        use itertools::Itertools;

        if self.values.is_empty() {
            debug_assert_eq!(self.min, self.max);

            write!(fmt, "max({})", self.min)
        } else {
            let maxmin = self
                .values
                .iter()
                .skip(1)
                .map(DiffExpr::min_value)
                .fold(self.values[0].min_value(), f64::max);
            assert!(maxmin.is_finite());

            if maxmin == self.min {
                write!(fmt, "max({})", self.values.iter().format(", "))
            } else {
                write!(
                    fmt,
                    "max({}, {})",
                    self.min,
                    self.values.iter().format(", ")
                )
            }
        }
    }
}

impl<P> From<FMaxExpr<P>> for Float<P>
where
    P: Atom,
{
    fn from(emax: FMaxExpr<P>) -> Self {
        emax.simplified()
            .unwrap_or_else(|emax| FloatInner::Max(emax).into())
    }
}

impl<P> From<Float<P>> for FMaxExpr<P>
where
    P: Atom,
{
    fn from(value: Float<P>) -> Self {
        match value.as_f64() {
            Some(val) => FMaxExpr {
                min: val,
                max: val,
                values: Default::default(),
            },
            None => match &*value {
                FloatInner::Max(emax) => emax.clone(),
                _ => FMaxExpr {
                    min: value.min_value(),
                    max: value.max_value(),
                    values: VecSet::new(DiffExpr::from(value)),
                },
            },
        }
    }
}

impl<'a, 'b, P> Add<&'a f64> for &'b FMaxExpr<P>
where
    P: Atom,
{
    type Output = FMaxExpr<P>;

    fn add(self, other: &'a f64) -> FMaxExpr<P> {
        FMaxExpr {
            min: self.min + other,
            max: self.max + other,
            values: self.values.iter().map(|item| item + other).collect(),
        }
    }
}

impl<P> FMaxExpr<P>
where
    P: Atom,
{
    fn new(min: f64, max: f64, values: Vec<DiffExpr<P>>) -> Self {
        assert!(min.is_finite() && max.is_finite());

        FMaxExpr {
            min,
            max,
            values: values.into_iter().collect(),
        }
    }

    /*
    fn deplane(mut self) -> Self {
        use std::collections::{HashMap, HashSet};
        let mut planes = Vec::with_capacity(self.values.len());
        let mut ndiffs = Vec::with_capacity(self.values.len());
        'values: for value in self.values {
            match &*value.inner {
                FloatInner::Diff(diff) => {
                    let mut slack = HashMap::new();
                    for positive in &diff.positive {
                        assert!(positive.min_value() >= 0.);

                        match &*positive.inner {
                            FloatInner::Mul(ratio, args) => {
                                *slack
                                    .entry((
                                        ratio.inner.numer.clone(),
                                        ratio.inner.denom.clone(),
                                        args.clone(),
                                    ))
                                    .or_insert(0f64) += ratio.inner.factor
                            }
                            _ => {
                                ndiffs.push(value);
                                continue 'values;
                            }
                        }
                    }
                    for negative in &diff.negative {
                        assert!(negative.min_value() >= 0.);

                        match &*negative.inner {
                            FloatInner::Mul(ratio, args) => {
                                *slack
                                    .entry((
                                        ratio.inner.numer.clone(),
                                        ratio.inner.denom.clone(),
                                        args.clone(),
                                    ))
                                    .or_insert(0f64) -= ratio.inner.factor
                            }
                            _ => {
                                ndiffs.push(value);
                                continue 'values;
                            }
                        }
                    }
                    planes.push((diff.constant, slack));
                }
                FloatInner::Mul(ratio, args) => {
                    planes.push((
                        0f64,
                        iter::once((
                            (
                                ratio.inner.numer.clone(),
                                ratio.inner.denom.clone(),
                                args.clone(),
                            ),
                            ratio.inner.factor,
                        ))
                        .collect(),
                    ));
                }
                _ => ndiffs.push(value),
            }
        }

        if !planes.is_empty() {
            let mut lol = HashSet::new();
            for (_, slack) in &planes {
                lol.extend(slack.keys());
            }

            let mut points: Vec<Vec<_>> = Vec::with_capacity(planes.len());
            for (factor, slack) in &planes {
                let mut values = Vec::with_capacity(lol.len() + 1);
                values.push(factor);
                for key in &lol {
                    values.push(slack.get(key).unwrap_or(&0f64));
                }

                let mut skip = false;
                let mut to_remove = Vec::new();
                for (ix, other) in points.iter().enumerate() {
                    assert_eq!(other.len(), values.len());

                    if other.iter().zip(values.iter()).all(|(o, v)| o >= v) {
                        skip = true;
                    } else if other.iter().zip(values.iter()).all(|(o, v)| o <= v) {
                        to_remove.push(ix);
                    }
                }

                let mut nremoved = 0;
                for ix in to_remove {
                    points.remove(ix - nremoved);
                    nremoved += 1;
                }

                if !skip {
                    points.push(values);
                }
            }

            assert!(!points.is_empty());

            for point in points {
                let factor = point[0];
                let (positive, negative) = point
                    .into_iter()
                    .skip(1)
                    .zip(lol.iter())
                    .filter(|(v, _)| !v.is_zero())
                    .map(|(v, (n, d, args))| {
                        let val = v.abs();
                        (
                            v > &0f64,
                            Float::ratio(val, n.clone(), d.clone(), args.clone()),
                        )
                    })
                    .partition::<Vec<_>, _>(|(sign, _)| *sign);

                ndiffs.push(
                    Diff {
                        constant: *factor,
                        positive: positive.into_iter().map(|(_, flt)| flt).collect(),
                        negative: negative.into_iter().map(|(_, flt)| flt).collect(),
                    }
                    .into(),
                );
            }
        }

        FMaxExpr {
            min: self.min,
            max: self.max,
            values: ndiffs,
        }
    }
    */

    fn fmax(&self, other: &FMaxExpr<P>) -> FMaxExpr<P> {
        if self.min >= other.max {
            self.clone()
        } else if other.min >= self.max {
            other.clone()
        } else {
            let min = self.min.max(other.min);
            let max = self.max.max(other.max);

            let values = VecSet::from_sorted_iter(
                Union::new(
                    self.values.iter().filter(|value| value.max_value() > min),
                    other.values.iter().filter(|value| value.max_value() > min),
                )
                .cloned(),
            );

            FMaxExpr { min, max, values }
        }
    }

    fn is_constant(&self) -> bool {
        self.min == self.max
    }

    fn is_single_value(&self) -> bool {
        // Need to check that the min value is close to the min value of the single arg, because it
        // is possible that we computed a min with a constant value
        self.values.len() == 1 && is_close(self.values[0].min_value(), self.min)
    }

    fn simplified(self) -> Result<Float<P>, Self> {
        if self.is_constant() {
            debug_assert_eq!(self.min, self.max);

            Ok(self.min.into())
        } else if self.is_single_value() {
            debug_assert!(self.values.len() == 1);

            Ok(self.values.values[0].clone().into())
        } else {
            Err(self)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FloatInner<P>
where
    P: Atom,
{
    /// Product of all arguments.  We keep the `FloatRatio` separate to make simplifications
    /// easier.
    Mul(FloatRatio<P>, Vec<Float<P>>),
    Reduction(FloatReduction<P>),
    Max(FMaxExpr<P>),
    Diff(DiffExpr<P>),
    // Ceil division: ceil(numer / denom)
    DivCeil(Int<P>, u32),
}

impl<P> fmt::Display for FloatInner<P>
where
    P: Atom + fmt::Display,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        use itertools::Itertools;
        use FloatInner::*;

        match self {
            DivCeil(numer, denom) => write!(fmt, "div_ceil({}, {})", numer, denom),
            Mul(ratio, args) if args.is_empty() => write!(fmt, "{}", ratio),
            Mul(ratio, args) => {
                if !ratio.is_one() {
                    write!(fmt, "{}*", ratio)?;
                }

                write!(fmt, "{}", args.iter().format("*"))
            }
            Reduction(reduction) => write!(fmt, "{}", reduction),
            Max(emax) => write!(fmt, "{}", emax),
            Diff(diff) => write!(fmt, "{}", diff),
        }
    }
}

#[derive(PartialEq, Eq, Hash)]
pub struct Float<P>
where
    P: Atom,
{
    inner: Rc<hash::MemoizedHash<FloatInner<P>>>,
}

impl<P> PartialOrd for Float<P>
where
    P: Atom + PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if Rc::ptr_eq(&self.inner, &other.inner) {
            Some(Ordering::Equal)
        } else {
            self.inner.partial_cmp(&other.inner)
        }
    }
}

impl<P> Ord for Float<P>
where
    P: Atom + Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        if Rc::ptr_eq(&self.inner, &other.inner) {
            Ordering::Equal
        } else {
            self.inner.cmp(&other.inner)
        }
    }
}

impl<P> AsRef<FloatInner<P>> for Float<P>
where
    P: Atom,
{
    fn as_ref(&self) -> &FloatInner<P> {
        self
    }
}

impl<P> Float<P>
where
    P: Atom,
{
    fn make_mut(&mut self) -> &mut FloatInner<P> {
        hash::MemoizedHash::make_mut(Rc::make_mut(&mut self.inner))
    }
}

impl<P> ops::Deref for Float<P>
where
    P: Atom,
{
    type Target = FloatInner<P>;

    fn deref(&self) -> &FloatInner<P> {
        &**self.inner
    }
}

impl<P> fmt::Debug for Float<P>
where
    P: Atom,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&self.inner, fmt)
    }
}

impl<P> fmt::Display for Float<P>
where
    P: Atom,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.inner, fmt)
    }
}

impl<P> Clone for Float<P>
where
    P: Atom,
{
    fn clone(&self) -> Self {
        Float {
            inner: Rc::clone(&self.inner),
        }
    }
}

impl<P> From<DiffExpr<P>> for Float<P>
where
    P: Atom,
{
    fn from(diff: DiffExpr<P>) -> Self {
        if diff.values.is_empty() {
            diff.constant.into()
        } else if diff.values.len() == 1 && diff.constant == 0. {
            let (weight, item) = diff.values.values[0].clone();
            item * weight
        } else {
            FloatInner::Diff(diff).into()
        }
    }
}

impl<P> From<FloatInner<P>> for Float<P>
where
    P: Atom,
{
    fn from(inner: FloatInner<P>) -> Self {
        Float {
            inner: Rc::new(hash::MemoizedHash::new(inner)),
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

impl<P> AsValue<f64> for Float<P>
where
    P: Atom,
{
    fn as_value(&self) -> Option<&f64> {
        match &**self {
            FloatInner::Mul(ratio, args) if args.is_empty() => ratio.as_f64_ref(),
            _ => None,
        }
    }

    fn as_value_mut(&mut self) -> Option<&mut f64> {
        if self.as_value().is_some() {
            Some(match self.make_mut() {
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
        match &**self {
            FloatInner::Reduction(reduction) if reduction.kind == kind => Some(reduction),
            _ => None,
        }
    }

    fn as_reduction_mut(
        &mut self,
        kind: FloatReductionKind,
    ) -> Option<&mut FloatReduction<P>> {
        if self.as_reduction(kind).is_some() {
            Some(match self.make_mut() {
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
    fn ratio(factor: f64, numer: Vec<P>, denom: Vec<P>, args: Vec<Float<P>>) -> Self {
        FloatInner::Mul(
            FloatRatio::from(FloatRatioInner {
                factor,
                numer,
                denom,
            }),
            args,
        )
        .into()
    }

    pub fn div_ceil(lhs: &Int<P>, rhs: u32) -> Self {
        match (&*lhs.inner, lhs.as_biguint()) {
            (_, Some(value)) => Int::from((value + rhs - 1u32) / rhs).to_symbolic_float(),
            // TODO: This should be a check on factors!
            (IntInner::Mul(ratio), None)
                if /* ratio.inner.denom.is_empty()
                    && */ ratio.inner.factor.is_multiple_of(&rhs.into()) =>
            {
                // TODO: This is no longer an integer division!!!!  The denominator does not
                // necessarily divide the numerator anymore.
                Int::from(IntInner::Mul(Ratio::new(
                    &ratio.inner.factor / rhs,
                    ratio.inner.numer.clone(),
                    ratio.inner.denom.clone(),
                )))
                .to_symbolic_float()
            }
            (IntInner::Mul(ratio), None) => {
                let big_rhs = BigUint::from(rhs);
                let gcd = ratio.inner.factor.gcd(&big_rhs);

                FloatInner::DivCeil(
                    IntInner::Mul(Ratio::new(
                        &ratio.inner.factor / &gcd,
                        ratio.inner.numer.clone(),
                        ratio.inner.denom.clone(),
                    ))
                    .into(),
                    (big_rhs / gcd).to_u32().unwrap(),
                )
                .into()
            }
            (_, None) => FloatInner::DivCeil(lhs.clone(), rhs).into(),
        }
    }

    fn fast_eq(lhs: &Self, rhs: &Self) -> bool {
        Rc::ptr_eq(&lhs.inner, &rhs.inner)
            || lhs
                .as_f64()
                .and_then(|lhs| rhs.as_f64().map(|rhs| lhs.to_bits() == rhs.to_bits()))
                .unwrap_or(false)
    }

    fn is_one(&self) -> bool {
        match &**self {
            FloatInner::Mul(ratio, args) if args.is_empty() => ratio.is_one(),
            _ => false,
        }
    }

    fn is_zero(&self) -> bool {
        self.as_value().map(Zero::is_zero).unwrap_or(false)
    }

    pub fn min_assign(&mut self, rhs: &Float<P>) {
        self.reduce_assign(FloatReductionKind::Min, rhs);
    }

    pub fn max(&self, other: &Float<P>) -> Float<P> {
        trace!("max lhs: {}", self);
        trace!("max rhs: {}", other);

        let result = FMaxExpr::from(self.clone())
            .fmax(&FMaxExpr::from(other.clone()))
            .into();

        trace!("max out: {}", result);
        result
    }

    pub fn max_assign(&mut self, rhs: &Float<P>) {
        *self = Float::max(&self.clone(), rhs);
    }

    pub fn as_f64(&self) -> Option<f64> {
        match &**self {
            FloatInner::Mul(ratio, args) if args.is_empty() => ratio.as_f64(),
            _ => None,
        }
    }

    pub fn to_f64(&self) -> Option<f64> {
        match &**self {
            FloatInner::DivCeil(lhs, rhs) => {
                lhs.to_u32().map(|lhs| ((lhs + rhs - 1) / rhs) as f64)
            }
            FloatInner::Mul(ratio, args) => ratio.to_f64().and_then(|ratio| {
                args.iter()
                    .map(|arg| arg.to_f64().ok_or(()))
                    .product::<Result<f64, ()>>()
                    .ok()
                    .map(|result| ratio * result)
            }),
            FloatInner::Max(emax) => unimplemented!("to_f64 for {}", emax),
            FloatInner::Reduction(_) => unimplemented!("to_f64 for {}", self),
            FloatInner::Diff(_) => unimplemented!("to_f64 for {}", self),
        }
    }

    pub fn min_value(&self) -> f64 {
        info!("min_value for {}", self);

        match &**self {
            FloatInner::DivCeil(numer, denom) => {
                let numer_min = numer.min_value();
                let denom = u64::from(*denom);
                // TODO: should take gcd for le min
                ((numer_min + denom - 1) / denom) as f64
            }
            FloatInner::Mul(ratio, args) => {
                // Find the DivCeils and collect theirs factors
                let (divceils, others) = args.iter().fold(
                    (
                        Vec::with_capacity(args.len()),
                        Vec::with_capacity(args.len()),
                    ),
                    |(mut divceils, mut others), arg| {
                        match &**arg {
                            FloatInner::DivCeil(numer, denom) => match &*numer.inner {
                                IntInner::Mul(ratio) if ratio.inner.denom.len() == 0 => {
                                    divceils.push((
                                        &ratio.inner.factor,
                                        &ratio.inner.numer,
                                        denom,
                                    ))
                                }
                                _ => unimplemented!("min_value containing {}", arg),
                            },
                            _ => others.push(arg),
                        };
                        (divceils, others)
                    },
                );

                let min = ratio.inner.factor.clone();
                let min = ratio
                    .inner
                    .numer
                    .iter()
                    .fold(min, |min, num| min * num.min_value() as f64);

                let min = if divceils.len() > 1 {
                    unimplemented!("min_value for {} (more than 1 divceil)", self);
                } else if divceils.len() == 1 {
                    let (dcfactor, dcnumer, dcdenom) = divceils[0];

                    // denom is only on the denominator
                    let mut denom = ratio.inner.denom.iter().collect::<Vec<_>>();
                    // dconly is only on the lcm numer
                    let mut dc_numer_value = dcfactor.clone();

                    let mut min = min;
                    for dcn in dcnumer {
                        if let Some(pos) = denom.iter().position(|elem| *elem == dcn) {
                            // Don't use it twice!
                            denom.swap_remove(pos);

                            let lcm_value = dcn.lcm_value();
                            dc_numer_value *= lcm_value;
                            min /= lcm_value as f64;
                        } else {
                            dc_numer_value *= dcn.gcd_value();
                        }
                    }

                    let min = min
                        * ((dc_numer_value + dcdenom - 1u32) / dcdenom)
                            .to_u64()
                            .unwrap() as f64;

                    denom
                        .into_iter()
                        .fold(min, |min, denom| min / denom.max_value() as f64)

                // dcnumer_only -> min (gcd?)
                // common -> lcm
                // denom_only -> max
                } else {
                    assert!(divceils.is_empty());

                    ratio
                        .inner
                        .denom
                        .iter()
                        .fold(min, |min, denom| min / denom.max_value() as f64)
                };

                others
                    .into_iter()
                    .fold(min, |min, other| min * other.min_value())
            }
            FloatInner::Max(emax) => emax.min,
            FloatInner::Reduction(red) => red.min_value(),
            FloatInner::Diff(diff) => diff.min_value(),
        }
    }

    pub fn max_value(&self) -> f64 {
        info!("max_value for {}", self);

        match &**self {
            FloatInner::DivCeil(numer, denom) => {
                let denom = u64::from(*denom);
                ((numer.max_value() + denom - 1) / denom) as f64
            }
            FloatInner::Mul(ratio, args) => {
                let mut min = ratio.max_value();
                for arg in args.iter() {
                    min *= arg.max_value();
                }
                min
            }
            FloatInner::Max(emax) => emax.max,
            FloatInner::Reduction(red) => red.max_value(),
            FloatInner::Diff(diff) => diff.max_value(),
        }
    }
}

impl<P> MulAssign<&'_ Float<P>> for Float<P>
where
    P: Atom,
{
    fn mul_assign(&mut self, rhs: &'_ Float<P>) {
        *self = ops::Mul::mul(&self.clone(), rhs);
    }
}

impl<P> MulAssign<Float<P>> for Float<P>
where
    P: Atom,
{
    fn mul_assign(&mut self, rhs: Float<P>) {
        *self = ops::Mul::mul(&self.clone(), rhs);
    }
}

impl<'a, 'b, P> Mul<&'a f64> for &'b Float<P>
where
    P: Atom,
{
    type Output = Float<P>;

    fn mul(self, other: &'a f64) -> Float<P> {
        use FloatInner::*;

        let result = match self.as_value() {
            Some(value) => (value * other).into(),
            None => match &**self {
                Mul(ratio, args) => {
                    let mut ratio = ratio.clone();
                    ratio *= FloatRatio::new_constant(*other);
                    Mul(ratio, args.clone()).into()
                }
                Diff(ediff) => DiffExpr {
                    constant: ediff.constant * other,
                    values: WeightedVecSet::from_sorted_iter(
                        ediff
                            .values
                            .iter()
                            .map(|(weight, item)| (weight * other, item.clone())),
                    ),
                }
                .into(),
                Max(emax) => FMaxExpr::new(
                    emax.min * other,
                    emax.max * other,
                    emax.values
                        .iter()
                        .map(|value| ops::Mul::mul(value, other))
                        .collect(),
                )
                .into(),
                _ => Mul((*other).into(), vec![self.clone()]).into(),
            },
        };

        #[cfg(verify)]
        {
            if other > 0. {
                assert!(is_close(self.min_value() * other, result.min_value()));
                assert!(is_close(self.max_value() * other, result.max_value()));
            } else {
                assert!(is_close(self.min_value() * other, result.max_value()));
                assert!(is_close(self.max_value() * other, result.min_value()));
            }
        }

        result
    }
}

forward_binop_to_ref_ref!(impl(P: Atom) Mul<Output = Float<P>>, mul for Float<P>, f64);

impl<'a, 'b, P> Mul<&'a Float<P>> for &'b Float<P>
where
    P: Atom,
{
    type Output = Float<P>;

    fn mul(self, other: &'a Float<P>) -> Float<P> {
        use FloatInner::*;
        let result;

        trace!("mul lhs: {}", self);
        trace!("mul rhs: {}", other);

        if self.is_one() || other.is_zero() {
            result = other.clone();
        } else if other.is_one() || self.is_zero() {
            result = self.clone();
        } else if let Some(value) = other.as_value() {
            result = ops::Mul::mul(self, value)
        } else if let Some(value) = self.as_value() {
            result = ops::Mul::mul(other, value)
        } else {
            result = match (&**self, &**other) {
                (Mul(lhs_ratio, lhs_args), Mul(rhs_ratio, rhs_args)) => {
                    let mut args = lhs_args.clone();
                    args.extend(rhs_args.iter().cloned());
                    Mul(lhs_ratio * rhs_ratio, args).into()
                }
                (Mul(lhs_ratio, lhs_args), Diff(ediff)) => {
                    let factor = lhs_ratio.inner.factor;
                    let naked = Float::from(Mul(
                        FloatRatio::new(
                            1f64,
                            lhs_ratio.inner.numer.clone(),
                            lhs_ratio.inner.denom.clone(),
                        ),
                        lhs_args.clone(),
                    ));
                    DiffExpr {
                        constant: 0f64,
                        values: if ediff.constant.is_zero() {
                            None
                        } else {
                            Some((factor * ediff.constant, naked.clone()))
                        }
                        .into_iter()
                        .chain(
                            ediff
                                .values
                                .iter()
                                .map(|(weight, item)| (weight * factor, item * &naked)),
                        )
                        .collect(),
                    }
                    .into()
                }
                (Mul(lhs_ratio, lhs_args), _) => {
                    let mut args = lhs_args.clone();
                    args.push(other.clone());
                    Mul(lhs_ratio.clone(), args).into()
                }
                (_, Mul(_, _)) => ops::Mul::mul(other, self),
                (_, _) => {
                    trace!("true mul: {:?} and {:?}", self, other);
                    Mul(1f64.into(), vec![self.clone(), other.clone()]).into()
                }
            }
        }

        trace!("mul out: {}", result);

        #[cfg(verify)]
        {
            assert!(self.min_value() > 0.);
            assert!(other.min_value() > 0.);

            assert!(is_close(
                self.min_value() * other.min_value(),
                result.min_value()
            ));
            assert!(is_close(
                self.max_value() * other.max_value(),
                result.max_value()
            ));
        }

        result
    }
}

forward_binop_to_ref_ref!(impl(P: Atom) Mul<Output = Float<P>>, mul for Float<P>, Float<P>);

impl<P> MulAssign<&'_ Int<P>> for Float<P>
where
    P: Atom,
{
    fn mul_assign(&mut self, rhs: &'_ Int<P>) {
        MulAssign::mul_assign(self, rhs.to_symbolic_float());
    }
}

forward_binop_to_op_assign!(impl(P: Atom) Mul, mul for Float<P>, Int<P>, MulAssign, mul_assign);
forward_binop_to_ref_val_commutative!(impl(P: Atom) Mul<Output = Float<P>>, mul for Int<P>, Float<P>);

impl<'a, 'b, P> Add<&'a f64> for &'b Float<P>
where
    P: Atom,
{
    type Output = Float<P>;

    fn add(self, other: &'a f64) -> Float<P> {
        if let Some(value) = self.as_value() {
            (value + other).into()
        } else if *other == 0. {
            self.clone()
        } else {
            match &**self {
                FloatInner::Max(max) => ops::Add::add(max, other).into(),
                _ => ops::Add::add(DiffExpr::from(self.clone()), other).into(),
            }
        }
    }
}

impl<P> AddAssign<&'_ Float<P>> for Float<P>
where
    P: Atom,
{
    fn add_assign(&mut self, rhs: &'_ Float<P>) {
        *self = Add::add(&self.clone(), rhs);
    }
}

impl<P> AddAssign<Float<P>> for Float<P>
where
    P: Atom,
{
    fn add_assign(&mut self, rhs: Float<P>) {
        *self = Add::add(&self.clone(), rhs);
    }
}

impl<'a, 'b, P> Add<&'a Float<P>> for &'b Float<P>
where
    P: Atom,
{
    type Output = Float<P>;

    fn add(self, other: &'a Float<P>) -> Float<P> {
        trace!("add lhs: {}", self);
        trace!("add rhs: {}", other);

        let result = if let Some(value) = other.as_value() {
            ops::Add::add(self, value)
        } else if let Some(value) = self.as_value() {
            ops::Add::add(other, value)
        } else {
            ops::Add::add(DiffExpr::from(self.clone()), DiffExpr::from(other.clone()))
                .into()
        };

        trace!("add out: {}", result);

        #[cfg(verify)]
        {
            assert!(is_close(
                self.min_value() + other.min_value(),
                result.min_value()
            ));
            assert!(is_close(
                self.max_value() + other.max_value(),
                result.max_value()
            ));
        }

        result
    }
}

forward_binop_to_ref_ref!(impl(P: Atom) Add<Output = Float<P>>, add for Float<P>, Float<P>);

impl<'a, P> DivAssign<&'a Int<P>> for Float<P>
where
    P: Atom,
{
    fn div_assign(&mut self, rhs: &'a Int<P>) {
        trace!("div lhs: {}", self);
        trace!("div rhs: {}", rhs);

        match &*rhs.inner {
            IntInner::Mul(ratio) => {
                let mut flt_ratio = FloatRatioInner::from(ratio.clone());
                std::mem::swap(&mut flt_ratio.numer, &mut flt_ratio.denom);
                flt_ratio.factor = flt_ratio.factor.recip();
                *self *= Float::from(FloatInner::Mul(
                    FloatRatio {
                        inner: Rc::new(flt_ratio),
                    },
                    vec![],
                ));
            }
            _ => unimplemented!("{} / {}", self, rhs),
        }

        trace!("div out: {}", self);
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

impl<P> SubAssign<&'_ f64> for Float<P>
where
    P: Atom,
{
    fn sub_assign(&mut self, other: &'_ f64) {
        *self = ops::Sub::sub(self.clone(), other);
    }
}

impl<P> SubAssign<f64> for Float<P>
where
    P: Atom,
{
    fn sub_assign(&mut self, other: f64) {
        *self = ops::Sub::sub(self.clone(), other);
    }
}

impl<'a, 'b, P> Sub<&'a f64> for &'b Float<P>
where
    P: Atom,
{
    type Output = Float<P>;

    fn sub(self, other: &'a f64) -> Float<P> {
        if let Some(value) = self.as_value() {
            (value - other).into()
        } else if *other == 0. {
            self.clone()
        } else {
            ops::Sub::sub(DiffExpr::from(self.clone()), other).into()
        }
    }
}

forward_binop_to_ref_ref!(impl(P: Atom) Sub<Output = Float<P>>, sub for Float<P>, f64);

impl<'a, 'b, P> Sub<&'a Float<P>> for &'b Float<P>
where
    P: Atom,
{
    type Output = Float<P>;

    fn sub(self, other: &'a Float<P>) -> Float<P> {
        trace!("sub lhs: {}", self);
        trace!("sub rhs: {}", other);

        let result = if self == other {
            0f64.into()
        } else if let Some(value) = other.as_value() {
            ops::Sub::sub(self, value)
        } else {
            ops::Sub::sub(DiffExpr::from(self.clone()), DiffExpr::from(other.clone()))
                .into()
        };

        trace!("sub out: {}", result);
        #[cfg(verify)]
        {
            assert!(is_close(
                self.min_value() - other.max_value(),
                result.min_value()
            ));
            assert!(is_close(
                self.max_value() - other.min_value(),
                result.max_value()
            ));
        }

        result
    }
}

forward_binop_to_ref_ref!(impl(P: Atom) Sub<Output = Float<P>>, sub for Float<P>, Float<P>);

impl<P> SubAssign<Float<P>> for Float<P>
where
    P: Atom,
{
    fn sub_assign(&mut self, rhs: Float<P>) {
        *self = Sub::sub(self.clone(), rhs);
    }
}

impl<P> SubAssign<&'_ Float<P>> for Float<P>
where
    P: Atom,
{
    fn sub_assign(&mut self, rhs: &'_ Float<P>) {
        *self = Sub::sub(self.clone(), rhs);
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::fmt;

    #[derive(Clone, Debug, Hash)]
    struct Size<'a> {
        name: Cow<'a, str>,
        min: u64,
        max: u64,
    }

    impl fmt::Display for Size<'_> {
        fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
            fmt::Display::fmt(&self.name, fmt)
        }
    }

    impl PartialEq for Size<'_> {
        fn eq(&self, other: &Size<'_>) -> bool {
            self.name == other.name
        }
    }

    impl Eq for Size<'_> {}

    impl Ord for Size<'_> {
        fn cmp(&self, other: &Size<'_>) -> std::cmp::Ordering {
            self.name.cmp(&other.name)
        }
    }

    impl PartialOrd for Size<'_> {
        fn partial_cmp(&self, other: &Size<'_>) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }

    impl super::Range for Size<'_> {
        fn min_value(&self) -> u64 {
            self.min
        }

        fn max_value(&self) -> u64 {
            self.max
        }
    }

    impl super::Factor for Size<'_> {
        fn gcd_value(&self) -> u64 {
            self.min
        }

        fn lcm_value(&self) -> u64 {
            self.max
        }
    }

    impl<'a> Size<'a> {
        fn new<N>(name: N, min: u64, max: u64) -> Self
        where
            N: Into<Cow<'a, str>>,
        {
            Size {
                name: name.into(),
                min,
                max,
            }
        }
    }

    fn make<'a, N>(
        name: N,
        min: u64,
        max: u64,
    ) -> (super::Int<Size<'a>>, super::Int<Size<'a>>)
    where
        N: Into<Cow<'a, str>>,
    {
        let name = name.into();
        let size = Size::new(name.clone(), min, max);
        (
            super::Int::ratio(1u32, vec![size.clone()], vec![]),
            super::Int::ratio(1u32, vec![], vec![size]),
        )
    }

    fn symbolic(value: u64) -> super::Int<Size<'static>> {
        super::Int::ratio(
            1u32,
            vec![Size::new(format!("_{}_", value), value, value)],
            vec![],
        )
    }

    #[test]
    fn it_works() {
        let (x, _invx) = make("x", 1, 10);

        // 1 + max(3, x) -> max(4, 1 + x)
        let mut expr = x.to_symbolic_float();
        expr.max_assign(&3f64.into());
        let expr = expr + super::Float::from(1f64);

        assert_eq!(format!("{}", expr), "max(4, 1 + x)");
    }

    #[test]
    fn still_works() {
        let (x, _invx) = make("x", 1, 10);
        let (y, _invy) = make("y", 1, 10);

        // y + max(3, x) -> max(4, y + 3, y + x)
        let mut expr = x.to_symbolic_float();
        expr.max_assign(&3f64.into());
        let expr = expr + y.to_symbolic_float();

        assert_eq!(format!("{}", expr), "max(4, x + y, 3 + y)");
    }

    #[test]
    fn not_broken() {
        let (x, _invx) = make("x", 1, 10);
        let (y, _invy) = make("y", 1, 10);

        // (9 + y) + max(3, x) -> max(13, y + 12, 9 + y + x)
        let mut expr = x.to_symbolic_float();
        expr.max_assign(&3f64.into());
        let expr = expr + (y.to_symbolic_float() + super::Float::from(9f64));

        assert_eq!(format!("{}", expr), "max(13, 9 + y + x, 12 + y)");
    }

    #[test]
    fn maximax() {
        let (x, _invx) = make("x", 1, 10);
        let (y, _invy) = make("y", 1, 10);

        // max(9, y) + max(3, x) -> max(13, y + 12, 9 + y + x)
        let mut expr = x.to_symbolic_float();
        expr.max_assign(&3f64.into());
        let mut ymax = y.to_symbolic_float();
        ymax.max_assign(&9f64.into());
        let expr = expr + ymax;

        assert_eq!(format!("{}", expr), "max(3, x) + max(9, y)");
    }

    #[test]
    fn max_dedup() {
        let (x, _) = make("x", 1, 10);

        let mut expr = x.to_symbolic_float();
        expr.max_assign(&x.to_symbolic_float());

        assert_eq!(format!("{}", expr), "x");
    }

    #[test]
    fn max_dedup_2() {
        let (x, _) = make("x", 3, 3);

        let mut expr = x.to_symbolic_float();
        expr.max_assign(&symbolic(2).to_symbolic_float());

        println!("{:?}", expr);

        assert_eq!(format!("{}", expr), "3");
    }

    #[test]
    fn max_lin() {
        let (x, _) = make("x", 1, 10);

        let mut expr = x.to_symbolic_float() * 2f64;
        expr.max_assign(&(x.to_symbolic_float() * 3f64));

        assert_eq!(format!("{}", expr), "3*x");
    }

    // TODO: test that `(a + b) * c => a * c + b * c
    // need to check with "args" in c, eg. 3 * div_ceil(...)
    //
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn sub_zero(x: super::Float<Size<'static>>) {
            let delta = x - x;

            assert!(delta.min_value().is_zero() && delta.max_value().is_zero());
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct VecSet<T> {
    values: Rc<Vec<T>>,
}

impl<T> Default for VecSet<T>
where
    T: Ord + Clone,
{
    fn default() -> Self {
        iter::empty().collect()
    }
}

impl<T> iter::FromIterator<T> for VecSet<T>
where
    T: Ord + Clone,
{
    fn from_iter<II>(iter: II) -> Self
    where
        II: IntoIterator<Item = T>,
    {
        let mut values = Vec::from_iter(iter);
        values.sort();
        values.dedup();

        VecSet {
            values: Rc::new(values),
        }
    }
}

impl<T> VecSet<T> {
    fn len(&self) -> usize {
        self.values.len()
    }

    fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    fn iter(&self) -> std::slice::Iter<'_, T> {
        self.values.iter()
    }
}

impl<T, I> ops::Index<I> for VecSet<T>
where
    Vec<T>: ops::Index<I>,
{
    type Output = <Vec<T> as ops::Index<I>>::Output;

    fn index(&self, index: I) -> &Self::Output {
        ops::Index::index(self.values.as_ref(), index)
    }
}

struct Union<L, R>
where
    L: Iterator,
    R: Iterator,
{
    left: std::iter::Peekable<L>,
    right: std::iter::Peekable<R>,
}

impl<T, L, R> Union<L, R>
where
    T: Ord,
    L: Iterator<Item = T>,
    R: Iterator<Item = T>,
{
    fn new(left: L, right: R) -> Self {
        Union {
            left: left.peekable(),
            right: right.peekable(),
        }
    }
}

impl<T, L, R> Iterator for Union<L, R>
where
    T: Ord,
    L: Iterator<Item = T>,
    R: Iterator<Item = T>,
{
    type Item = T;

    fn next(&mut self) -> Option<T> {
        if let Some(litem) = self.left.peek() {
            if let Some(ritem) = self.right.peek() {
                match litem.cmp(ritem) {
                    Ordering::Less => {
                        Some(self.left.next().unwrap_or_else(|| unreachable!()))
                    }
                    Ordering::Greater => {
                        Some(self.right.next().unwrap_or_else(|| unreachable!()))
                    }
                    Ordering::Equal => {
                        self.left.next().unwrap_or_else(|| unreachable!());
                        Some(self.right.next().unwrap_or_else(|| unreachable!()))
                    }
                }
            } else {
                Some(self.left.next().unwrap_or_else(|| unreachable!()))
            }
        } else {
            self.right.next()
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (llow, lhigh) = self.left.size_hint();
        let (rlow, rhigh) = self.right.size_hint();

        (
            llow.min(rlow),
            lhigh.and_then(|lhigh| rhigh.map(|rhigh| lhigh + rhigh)),
        )
    }
}

impl<T> VecSet<T>
where
    T: Ord + Clone,
{
    fn new(value: T) -> Self {
        VecSet {
            values: Rc::new(vec![value]),
        }
    }

    fn from_sorted_iter<II>(values: II) -> Self
    where
        II: IntoIterator<Item = T>,
    {
        VecSet {
            values: Rc::new(values.into_iter().collect()),
        }
    }
}

struct WeightedUnion<L, R>
where
    L: Iterator,
    R: Iterator,
{
    left: std::iter::Peekable<L>,
    right: std::iter::Peekable<R>,
}

impl<T, L, R> WeightedUnion<L, R>
where
    T: Ord,
    L: Iterator<Item = (f64, T)>,
    R: Iterator<Item = (f64, T)>,
{
    fn new(left: L, right: R) -> Self {
        WeightedUnion {
            left: left.peekable(),
            right: right.peekable(),
        }
    }
}

impl<T, L, R> Iterator for WeightedUnion<L, R>
where
    T: Ord,
    L: Iterator<Item = (f64, T)>,
    R: Iterator<Item = (f64, T)>,
{
    type Item = (f64, T);

    fn next(&mut self) -> Option<(f64, T)> {
        loop {
            if let Some((_, litem)) = self.left.peek() {
                if let Some((_, ritem)) = self.right.peek() {
                    match litem.cmp(ritem) {
                        Ordering::Less => {
                            break Some(
                                self.left.next().unwrap_or_else(|| unreachable!()),
                            )
                        }
                        Ordering::Greater => {
                            break Some(
                                self.right.next().unwrap_or_else(|| unreachable!()),
                            )
                        }
                        Ordering::Equal => {
                            let (lweight, litem) =
                                self.left.next().unwrap_or_else(|| unreachable!());
                            let (rweight, _ritem) =
                                self.right.next().unwrap_or_else(|| unreachable!());

                            let weight = lweight + rweight;

                            if is_close(weight, 0.) {
                                continue;
                            } else {
                                break Some((lweight + rweight, litem));
                            }
                        }
                    }
                } else {
                    break Some(self.left.next().unwrap_or_else(|| unreachable!()));
                }
            } else {
                break self.right.next();
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (llow, lhigh) = self.left.size_hint();
        let (rlow, rhigh) = self.right.size_hint();

        (
            llow.min(rlow),
            lhigh.and_then(|lhigh| rhigh.map(|rhigh| lhigh.max(rhigh))),
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
struct WeightedVecSet<T> {
    values: Rc<Vec<(f64, T)>>,
}

impl<T> Default for WeightedVecSet<T> {
    fn default() -> Self {
        WeightedVecSet {
            values: Rc::default(),
        }
    }
}

impl<T> Eq for WeightedVecSet<T> where T: Eq {}

impl<T> PartialOrd for WeightedVecSet<T>
where
    T: Ord,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for WeightedVecSet<T>
where
    T: Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        if self.len() < other.len() {
            Ordering::Less
        } else if self.len() > other.len() {
            Ordering::Greater
        } else {
            for ((_, lhs), (_, rhs)) in self.iter().zip(other.iter()) {
                match lhs.cmp(rhs) {
                    Ordering::Less => return Ordering::Less,
                    Ordering::Greater => return Ordering::Greater,
                    Ordering::Equal => (),
                }
            }

            for ((lhs, _), (rhs, _)) in self.iter().zip(other.iter()) {
                if lhs < rhs {
                    return Ordering::Less;
                } else if lhs > rhs {
                    return Ordering::Greater;
                }
            }

            Ordering::Equal
        }
    }
}

impl<T> Hash for WeightedVecSet<T>
where
    T: Hash,
{
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        for (factor, item) in self.values.as_ref() {
            factor.to_bits().hash(state);
            item.hash(state)
        }
    }
}

impl<T> WeightedVecSet<T> {
    fn len(&self) -> usize {
        self.values.len()
    }
}

impl<T> iter::FromIterator<(f64, T)> for WeightedVecSet<T>
where
    T: Ord,
{
    fn from_iter<II>(iter: II) -> Self
    where
        II: IntoIterator<Item = (f64, T)>,
    {
        let mut values = Vec::from_iter(iter);
        values.sort_by(|(_, lhs), (_, rhs)| lhs.cmp(rhs));

        // TODO: sumdups()

        WeightedVecSet::from_sorted_iter(values)
    }
}

impl<T> WeightedVecSet<T>
where
    T: Ord,
{
    fn from_sorted_iter<II>(values: II) -> Self
    where
        II: IntoIterator<Item = (f64, T)>,
    {
        WeightedVecSet {
            values: Rc::new(values.into_iter().collect()),
        }
    }

    fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    fn iter<'a>(&'a self) -> impl Iterator<Item = (f64, &'a T)> {
        self.values.iter().map(|(weight, item)| (*weight, item))
    }
}

impl<T> WeightedVecSet<T>
where
    T: Ord + Clone,
{
    fn union(&self, other: &WeightedVecSet<T>) -> WeightedVecSet<T> {
        if self.is_empty() {
            other.clone()
        } else {
            self.union_sorted(other.iter())
        }
    }

    fn union_sorted<'a, II>(&'a self, other: II) -> WeightedVecSet<T>
    where
        II: IntoIterator<Item = (f64, &'a T)>,
        T: 'a,
    {
        let other = other.into_iter();
        if other.size_hint().1 == Some(0) {
            self.clone()
        } else {
            WeightedVecSet::from_sorted_iter(
                WeightedUnion::new(self.iter(), other)
                    .map(|(weight, item)| (weight, item.clone())),
            )
        }
    }
}
