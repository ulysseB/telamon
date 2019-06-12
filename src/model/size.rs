//! Size evaluation and manipulation primitives.
use crate::device::Context;
use crate::ir;
use crate::search_space::{NumSet, SearchSpace};
use num::{bigint::ToBigUint, Integer, ToPrimitive, Zero};
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

use utils::*;

#[derive(Debug, Eq)]
struct DimSizeInner {
    id: ir::DimId,
    possible_values: Vec<u32>,
    gcd: u64,
    lcm: u64,
}

impl fmt::Display for DimSizeInner {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{:?}", self.id)
    }
}

impl PartialEq for DimSizeInner {
    fn eq(&self, other: &DimSizeInner) -> bool {
        self.id == other.id
    }
}

impl Ord for DimSizeInner {
    fn cmp(&self, other: &DimSizeInner) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for DimSizeInner {
    fn partial_cmp(&self, other: &DimSizeInner) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Hash for DimSizeInner {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.id.hash(state);
    }
}

#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct DimSize {
    inner: Rc<DimSizeInner>,
}

impl fmt::Debug for DimSize {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&self.inner, fmt)
    }
}

impl fmt::Display for DimSize {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.inner, fmt)
    }
}

impl DimSize {
    pub(super) fn new<II>(id: ir::DimId, possible_values: II) -> Self
    where
        II: IntoIterator<Item = u32>,
    {
        let mut possible_values = possible_values.into_iter().collect::<Vec<_>>();
        assert!(!possible_values.is_empty(), "Impossible size.");
        assert!(
            possible_values.len() > 1,
            "DimSize should not be singleton, duh."
        );

        possible_values.sort();

        let (gcd, lcm) = possible_values.iter().skip(1).cloned().map(u64::from).fold(
            (u64::from(possible_values[0]), u64::from(possible_values[0])),
            |(gcd, lcm), possible_value| {
                (gcd.gcd(&possible_value), lcm.lcm(&possible_value))
            },
        );

        DimSize {
            inner: Rc::new(DimSizeInner {
                id,
                possible_values,
                gcd,
                lcm,
            }),
        }
    }
}

pub type SymbolicInt = sym::Int<DimSize>;
pub type SymbolicFloat = sym::Float<DimSize>;
pub type Ratio = sym::Ratio<DimSize>;
pub type Lcm = sym::LcmExpr<DimSize>;
pub type Min = sym::MinExpr<DimSize>;

impl sym::Range for DimSize {
    fn min_value(&self) -> u64 {
        self.inner.possible_values[0].into()
    }

    fn max_value(&self) -> u64 {
        self.inner.possible_values[self.inner.possible_values.len() - 1].into()
    }
}

impl sym::Factor for DimSize {
    fn gcd_value(&self) -> u64 {
        self.inner.gcd
    }

    fn lcm_value(&self) -> u64 {
        self.inner.lcm
    }
}

/// A span of values.
#[derive(Debug, Copy, Clone)]
pub struct Range {
    pub min: u64,
    pub max: u64,
}

impl Range {
    pub const ZERO: Self = Range { min: 0, max: 0 };

    pub const ONE: Self = Range { min: 1, max: 1 };

    /// Creates a `Range` containing a single value.
    pub fn new_fixed(val: u64) -> Self {
        Range { min: val, max: val }
    }

    /// Indicates if the `Range` contains a single value.
    pub fn is_constrained(&self) -> bool {
        self.min == self.max
    }
}

/// Bounds the values a size can take, in the given context.
pub fn bounds(size: &ir::PartialSize, space: &SearchSpace, ctx: &dyn Context) -> Range {
    let (factor, param_factors, dim_size_factors) = size.factors();
    let divisors = size.divisors();
    let factor = param_factors
        .iter()
        .map(|p| u64::from(ctx.param_as_size(&p.name).unwrap()))
        .product::<u64>()
        * u64::from(factor);
    let mut total_min = factor.to_biguint().unwrap();
    let mut total_max = total_min.clone();
    for &dim in dim_size_factors {
        let size = dim_bounds(dim, space);
        total_min *= size.min;
        total_max *= size.max;
    }
    for &dim in divisors {
        let size = dim_bounds(dim, space);
        total_min /= size.max.to_biguint().unwrap().gcd(&total_min);
        total_max /= size.min;
    }
    assert!(!total_min.is_zero());
    assert!(!total_max.is_zero());
    Range {
        min: total_min.to_u64().unwrap(),
        max: total_max.to_u64().unwrap(),
    }
}

/// Returns the `Range` a static dimension size can take.
pub fn dim_bounds(dim: ir::DimId, space: &SearchSpace) -> Range {
    let size = space.domain().get_size(dim);
    let universe = unwrap!(space.ir_instance().dim(dim).possible_sizes());
    Range {
        min: size.min_value(universe).into(),
        max: size.max_value(universe).into(),
    }
}

/// A span of values, in term of factors. The actual value is a mulitpe of `gcd` and
/// a divisor of `lcm`.
#[derive(Debug, Copy, Clone)]
pub struct FactorRange {
    pub gcd: u64,
    pub lcm: u64,
}

impl FactorRange {
    pub const ZERO: Self = FactorRange { gcd: 0, lcm: 0 };

    /// Create a `FactorRange` containing a single point.
    pub fn new_fixed(val: u64) -> Self {
        FactorRange { gcd: val, lcm: val }
    }
}

/// Returns a factor and a multiple of `size`.
pub fn factors(
    size: &ir::PartialSize,
    space: &SearchSpace,
    ctx: &dyn Context,
) -> FactorRange {
    let (factor, param_factors, dim_size_factors) = size.factors();
    let divisors = size.divisors();
    let factor = param_factors
        .iter()
        .map(|p| u64::from(ctx.param_as_size(&p.name).unwrap()))
        .product::<u64>()
        * u64::from(factor);
    let mut total_gcd = factor.to_biguint().unwrap();
    let mut total_lcm = total_gcd.clone();
    for &dim in dim_size_factors {
        let size = dim_factors(dim, space);
        total_gcd *= size.gcd;
        total_lcm *= size.lcm;
    }
    for &dim in divisors {
        let size = dim_factors(dim, space);
        total_gcd /= size.lcm.to_biguint().unwrap().gcd(&total_gcd);
        total_lcm /= size.gcd;
    }
    FactorRange {
        gcd: total_gcd.to_u64().unwrap(),
        lcm: total_lcm.to_u64().unwrap(),
    }
}

/// Returns the `FactorRane` a static dimension size can take.
pub fn dim_factors(dim: ir::DimId, space: &SearchSpace) -> FactorRange {
    let size = space.domain().get_size(dim);
    let universe = unwrap!(space.ir_instance().dim(dim).possible_sizes());
    FactorRange {
        gcd: size.gcd(universe).into(),
        lcm: size.lcm(universe).into(),
    }
}
