//! Size evaluation and manipulation primitives.
use crate::device::Context;
use crate::ir;
use crate::search_space::{NumSet, SearchSpace};
use num::{bigint::ToBigUint, Integer, ToPrimitive, Zero};
use utils::*;

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
pub fn bounds(size: &ir::PartialSize, space: &SearchSpace, ctx: &Context) -> Range {
    let (factor, param_factors, dim_size_factors) = size.factors();
    let divisors = size.divisors();
    let factor = param_factors
        .iter()
        .map(|p| unwrap!(ctx.param_as_size(&p.name)) as u64)
        .product::<u64>()
        * factor as u64;
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
        min: size.min(universe) as u64,
        max: size.max(universe) as u64,
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
    ctx: &Context,
) -> FactorRange {
    let (factor, param_factors, dim_size_factors) = size.factors();
    let divisors = size.divisors();
    let factor = param_factors
        .iter()
        .map(|p| unwrap!(ctx.param_as_size(&p.name)) as u64)
        .product::<u64>()
        * factor as u64;
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
        gcd: size.gcd(universe) as u64,
        lcm: size.lcm(universe) as u64,
    }
}
