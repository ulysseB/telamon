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
    /// Temporary function that assumes the range only contains a single value.
    pub fn fixed_val(&self) -> u64 {
        assert_eq!(self.min, self.max);
        self.min
    }
}

/// Bounds the values a size can take, in the given context.
pub fn bounds(size: &ir::Size, _: &SearchSpace, ctx: &Context) -> Range {
    let mut total = size.factor();
    for p in size.dividend() { total *= unwrap!(ctx.param_as_size(&p.name)); }
    total /= size.divisor();
    Range { min: total as u64, max: total as u64 }
}
