use log::{debug, info};
use std::borrow::Borrow;
use std::cmp;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::iter;
use std::ops::{self, Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
use std::rc::Rc;

use num::{BigUint, Integer, Num, One, ToPrimitive, Zero};

mod hash;

pub trait Range<N = u64> {
    fn min_value(&self) -> N;

    fn max_value(&self) -> N;
}

pub trait Atom: Range + Clone + fmt::Debug + fmt::Display + PartialEq + Ord {}

impl<T> Atom for T where
    T: ?Sized + Range + Clone + fmt::Debug + fmt::Display + PartialEq + Ord
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

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
            IntReductionKind::Min => constant < &BigUint::from(other.min_value()),
        }
    }

    fn get_bound(&self, other: &Int<P>) -> Option<BigUint> {
        match self {
            IntReductionKind::Min => Some(other.max_value().into()),
            IntReductionKind::Lcm => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
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

type IntReduction<P> = Reduction<IntReductionKind, BigUint, Int<P>>;

impl<P> Eq for IntReduction<P> where P: Eq {}

impl<P> Ord for IntReduction<P>
where
    P: Ord,
{
    fn cmp(&self, other: &IntReduction<P>) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl<P> Hash for IntReduction<P>
where
    P: Hash,
{
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.kind.hash(state);
        self.constant.hash(state);
        self.others.hash(state);
    }
}

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

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
    fn div_ceil(lhs: &Self, rhs: u32) -> Self {
        match (&*lhs.inner, lhs.as_biguint()) {
            (_, Some(value)) => Int::from((value + rhs - 1u32) / rhs),
            // TODO: This should be a check on factors!
            (IntInner::Mul(ratio, args), None)
                if ratio.inner.factor.is_multiple_of(&rhs.into()) =>
            {
                // TODO: This is no longer an integer division!!!!  The denominator does not
                // necessarily divide the numerator anymore.
                IntInner::Mul(
                    Ratio::new(
                        &ratio.inner.factor / rhs,
                        ratio.inner.numer.clone(),
                        ratio.inner.denom.clone(),
                    ),
                    args.clone(),
                )
                .into()
            }
            (_, None) => IntInner::DivCeil(lhs.clone(), rhs).into(),
        }
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
            IntInner::Mul(ratio, args) => {
                if let Some(value) = self.to_u32() {
                    Float::from(value as f64)
                } else {
                    /* TODO */
                    FloatInner::Mul(self.clone().into(), Vec::new()).into()
                }
            }
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
}

impl<P> Range for Int<P>
where
    P: Atom,
{
    fn min_value(&self) -> u64 {
        match &*self.inner {
            IntInner::DivCeil(numer, denom) => {
                let numer_min = numer.min_value();
                let denom = u64::from(*denom);
                // TODO: should take gcd for le min
                (numer_min + denom - 1) / denom
            }
            IntInner::Reduction(_) => unimplemented!("min_value for {}", self),
            IntInner::Sub(lhs, rhs) => lhs.min_value() - u64::from(*rhs),
            IntInner::Mul(ratio, args) => args
                .iter()
                .fold(ratio.min_value(), |min, arg| min * arg.min_value()),
        }
    }

    fn max_value(&self) -> u64 {
        match &*self.inner {
            IntInner::DivCeil(numer, denom) => {
                let denom = u64::from(*denom);
                (numer.max_value() + denom - 1) / denom
            }
            IntInner::Reduction(_) => unimplemented!("max_value for {}", self),
            IntInner::Sub(lhs, rhs) => lhs.max_value() - u64::from(*rhs),
            IntInner::Mul(ratio, args) => args
                .iter()
                .fold(ratio.max_value(), |max, arg| max * arg.max_value()),
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
#[derive(Debug, PartialEq, PartialOrd)]
struct FloatRatioInner<P> {
    factor: f64,
    numer: Int<P>,
    denom: Int<P>,
    // should be: factor * float(numer/denom) * float(numer)/float(denom)
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
            write!(fmt, "/({})", self.denom)?;
        }

        Ok(())
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash)]
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
    fn new_constant(factor: f64) -> Self {
        FloatRatioInner {
            factor,
            numer: Int::one(),
            denom: Int::one(),
        }
        .into()
    }

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
        self.inner.factor * self.inner.numer.min_value() as f64
            / self.inner.denom.max_value() as f64
    }

    fn max_value(&self) -> f64
    where
        P: Atom,
    {
        self.inner.factor * self.inner.numer.max_value() as f64
            / self.inner.denom.min_value() as f64
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
struct Diff<P> {
    constant: f64,
    positive: Vec<Float<P>>,
    negative: Vec<Float<P>>,
}

impl<P> Eq for Diff<P> where P: Eq {}

impl<P> Ord for Diff<P>
where
    P: Ord,
{
    fn cmp(&self, other: &Diff<P>) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl<P> Hash for Diff<P>
where
    P: Hash,
{
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.constant.to_bits().hash(state);
        self.positive.hash(state);
        self.negative.hash(state);
    }
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

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
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

impl<P> Hash for FloatReduction<P>
where
    P: Hash,
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

impl<P> Eq for FloatReduction<P> where P: Eq {}

impl<P> Ord for FloatReduction<P>
where
    P: Ord,
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

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
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

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash)]
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

impl<P> Clone for Float<P> {
    fn clone(&self) -> Self {
        Float {
            inner: Rc::clone(&self.inner),
        }
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
    pub fn div_ceil(lhs: &Int<P>, rhs: u32) -> Self {
        Int::div_ceil(lhs, rhs).to_symbolic_float()
    }

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

impl<'a, 'b, P> Mul<&'a Float<P>> for &'b Float<P>
where
    P: Atom,
{
    type Output = Float<P>;

    fn mul(self, other: &'a Float<P>) -> Float<P> {
        use FloatInner::*;

        if self.is_one() || other.is_zero() {
            return other.clone();
        } else if other.is_one() || self.is_zero() {
            return self.clone();
        }

        match (
            (&*self.inner, self.as_value()),
            (&*other.inner, other.as_value()),
        ) {
            ((_, Some(lhs)), (_, Some(rhs))) => (*lhs * *rhs).into(),
            ((Mul(ratio, args), None), (_, Some(value)))
            | ((_, Some(value)), (Mul(ratio, args), None)) => {
                let mut ratio = ratio.clone();
                ratio *= FloatRatio::new_constant(*value);
                Mul(ratio, args.clone()).into()
            }
            ((Mul(lhs_ratio, lhs_args), None), (Mul(rhs_ratio, rhs_args), None)) => {
                let mut args = lhs_args.clone();
                args.extend(rhs_args.iter().cloned());
                Mul(lhs_ratio * rhs_ratio, args).into()
            }
            (
                (_, Some(value)),
                (
                    Diff(self::Diff {
                        constant,
                        positive,
                        negative,
                    }),
                    None,
                ),
            )
            | (
                (
                    Diff(self::Diff {
                        constant,
                        positive,
                        negative,
                    }),
                    None,
                ),
                (_, Some(value)),
            ) => {
                let orig_value = *value;
                let (value, positive, negative) = if orig_value < 0f64 {
                    (-orig_value, negative, positive)
                } else {
                    (orig_value, positive, negative)
                };

                Diff(self::Diff {
                    constant: constant * orig_value,
                    positive: positive.iter().map(|p| Float::from(value) * p).collect(),
                    negative: negative.iter().map(|n| Float::from(value) * n).collect(),
                })
                .into()
            }
            (
                (Mul(ratio, args), _),
                (
                    Diff(self::Diff {
                        constant,
                        positive,
                        negative,
                    }),
                    None,
                ),
            )
            | (
                (
                    Diff(self::Diff {
                        constant,
                        positive,
                        negative,
                    }),
                    None,
                ),
                (Mul(ratio, args), _),
            ) => {
                let factor = Float::from(Mul(ratio.clone(), args.clone()));
                let mut positive =
                    positive.iter().map(|p| &factor * p).collect::<Vec<_>>();
                let constant_ratio = FloatRatio::new_constant(*constant);
                let new_ratio = ratio * &constant_ratio;
                let constant_flt = Float::from(Mul(constant_ratio, vec![]));
                positive.push(
                    Mul(new_ratio, args.iter().map(|a| a * &constant_flt).collect())
                        .into(),
                );
                let negative = negative.iter().map(|n| &factor * n).collect::<Vec<_>>();

                Diff(self::Diff {
                    constant: 0f64,
                    positive,
                    negative,
                })
                .into()
            }
            ((Mul(lhs_ratio, lhs_args), _), _) => {
                let mut args = lhs_args.clone();
                args.push(other.clone());
                Mul(lhs_ratio.clone(), args).into()
            }
            (_, (Mul(_, _), _)) => ops::Mul::mul(other, self),
            (_, _) => Mul(1f64.into(), vec![self.clone(), other.clone()]).into(),
        }
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
        use FloatInner::*;

        if self.is_zero() {
            return other.clone();
        } else if other.is_zero() {
            return self.clone();
        }

        match (
            (&*self.inner, self.as_value()),
            (&*other.inner, other.as_value()),
        ) {
            ((_, Some(lhs)), (_, Some(rhs))) => (*lhs + *rhs).into(),
            ((Diff(diff), None), (_, Some(value)))
            | ((_, Some(value)), (Diff(diff), None)) => {
                let mut new = diff.clone();
                new.add_assign(*value, iter::empty(), iter::empty());
                Diff(new).into()
            }
            ((_, Some(value)), (Reduction(red), None))
            | ((Reduction(red), None), (_, Some(value))) => match red.kind {
                FloatReductionKind::Min | FloatReductionKind::Max => {
                    let mut red = red.clone();
                    red.constant = Some(red.constant.unwrap() + value);
                    for other in &mut red.others {
                        *other += Float::from(*value);
                    }
                    red.simplify();
                    Reduction(red).into()
                }
            },
            ((Diff(lhs), None), (Diff(rhs), None)) => {
                let mut lhs = lhs.clone();
                lhs.add_assign(rhs.constant, rhs.positive.iter(), rhs.negative.iter());
                Diff(lhs).into()
            }
            ((Reduction(_), None), (Reduction(_), None)) => {
                // Prevent term size explosion when adding reductions together
                self::Diff::new_unknown(vec![self.clone(), other.clone()], vec![]).into()
            }
            ((_, None), (Reduction(red), None)) => {
                let mut red = red.clone();
                let constant = red.constant.unwrap();
                let bound = match red.kind {
                    FloatReductionKind::Min => self.max_value(),
                    FloatReductionKind::Max => self.min_value(),
                };
                red.constant = Some(constant + bound);
                for other in &mut red.others {
                    *other += self;
                }
                red.reduce_assign(None, iter::once(&(self + Float::from(constant))));
                Reduction(red).into()
            }
            ((Reduction(_), None), (_, None)) => Add::add(other, self),
            ((Diff(lhs), _), (_, None)) => {
                let mut lhs = lhs.clone();
                lhs.add_assign(0f64, iter::once(other), iter::empty());
                Diff(lhs).into()
            }
            ((_, None), (Diff(_), _)) => Add::add(other, self),
            ((_, None), (_, Some(value))) => self::Diff::new_constant(*value)
                .add(0f64, iter::once(self), iter::empty())
                .into(),
            ((_, Some(value)), (_, None)) => self::Diff::new_constant(*value)
                .add(0f64, iter::once(other), iter::empty())
                .into(),
            ((_, None), (_, None)) => {
                self::Diff::new_unknown(vec![self.clone(), other.clone()], vec![]).into()
            }
        }
    }
}

forward_binop_to_ref_ref!(impl(P: Atom) Add<Output = Float<P>>, add for Float<P>, Float<P>);

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

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::fmt;

    #[derive(Clone, Debug)]
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
}
