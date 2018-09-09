//! Size evaluation and manipulation primitives.
use device::Context;
use ir;
use search_space::SearchSpace;

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
}

/// Bounds the values a size can take, in the given context.
pub fn bounds(size: &ir::PartialSize, space: &SearchSpace, ctx: &Context) -> Range {
    // FIXME: the size may not have a fixed value.
    let val = ctx.eval_size(&::codegen::Size::from_ir(size, space)) as u64;
    Range::new_fixed(val)
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
    // FIXME: the size may not have a fixed value.
    let val = ctx.eval_size(&::codegen::Size::from_ir(size, space)) as u64;
    FactorRange::new_fixed(val)
}
