//! Size evaluation and manipulation primitives.
use device::Context;
use ir;
use num::integer;
use search_space::{SearchSpace, NumDomain};

/// A span of values.
#[derive(Debug, Copy, Clone)]
pub struct Range {
    pub min: u64,
    pub max: u64,
}

impl Range {
    pub const ZERO: Self = Range { min: 0, max: 0 };

    pub const ONE: Self = Range { min: 1, max: 1 };

    /// Indicates if the size can only take a single value.
    pub fn is_constrained(&self) -> bool { self.min == self.max }
}

/// Bounds the values a size can take, in the given context.
pub fn bounds(size: &ir::Size, space: &SearchSpace, ctx: &Context) -> Range {
    let (mut factor, params, dividend_dims, divisor_dims) = size.members();
    for p in params { factor *= unwrap!(ctx.param_as_size(&p.name)); }
    let mut total = Range { min: factor as u64, max: factor as u64 };
    for &d in dividend_dims {
        let mut size = space.domain().get_size(d);
        total.min *= size.min() as u64;
        total.max *= size.max() as u64;
    }
    for &d in divisor_dims {
        let mut size = space.domain().get_size(d);
        total.min /= size.min() as u64;
        total.max /= size.max() as u64;
    }
    total
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
    pub fn new_fixed(val: u64) -> Self { FactorRange { gcd: val, lcm: val } }
}

/// Returns a factor and a multiple of `size`.
pub fn factors(size: &ir::Size, space: &SearchSpace, ctx: &Context) -> FactorRange {
    let (mut factor, params, dividend_dims, divisor_dims) = size.members();
    for p in params { factor *= unwrap!(ctx.param_as_size(&p.name)); }
    let mut total = FactorRange::new_fixed(factor as u64);
    for &d in dividend_dims {
        let mut size = space.domain().get_size(d);
        total.gcd *= size.gcd() as u64;
        total.lcm *= size.lcm() as u64;
    }
    let mut divisor = FactorRange::new_fixed(1);
    for &d in divisor_dims {
        let mut size = space.domain().get_size(d);
        divisor.gcd *= size.gcd() as u64;
        divisor.lcm *= size.lcm() as u64;
    }
    FactorRange {
        gcd: total.gcd / integer::gcd(total.gcd, divisor.lcm),
        lcm: total.lcm / divisor.gcd
    }
}
