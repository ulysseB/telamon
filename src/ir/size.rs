use crate::ir;
use std;

use utils::*;

/// A fully specified size.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Size<'a> {
    factor: u32,
    params: Vec<&'a ir::Parameter>,
    max_val: u32,
}

impl<'a> Size<'a> {
    /// Create a new fully specified size.
    pub fn new(factor: u32, params: Vec<&'a ir::Parameter>, max_val: u32) -> Self {
        Size {
            factor,
            params,
            max_val,
        }
    }

    /// Creates a new constant size.
    pub fn new_const(factor: u32) -> Self {
        Size {
            factor,
            max_val: factor,
            ..Size::default()
        }
    }

    /// Creates a new size equal to a parameter.
    pub fn new_param(param: &'a ir::Parameter, max_val: u32) -> Size {
        Size {
            params: vec![param],
            max_val,
            ..Size::default()
        }
    }

    /// Returns the size if it is a constant.
    pub fn as_constant(&self) -> Option<u32> {
        if self.params.is_empty() {
            Some(self.factor)
        } else {
            None
        }
    }

    /// Returns the maximum value the size can take.
    pub fn max(&self) -> u32 {
        self.max_val
    }
}

impl<'a> Default for Size<'a> {
    fn default() -> Self {
        Size {
            factor: 1,
            params: Vec::new(),
            max_val: 1,
        }
    }
}

impl<'a, T> std::ops::MulAssign<T> for Size<'a>
where
    T: std::borrow::Borrow<Size<'a>>,
{
    fn mul_assign(&mut self, rhs: T) {
        let rhs = rhs.borrow();
        self.factor *= rhs.factor;
        self.params.extend(rhs.params.iter().cloned());
        self.max_val = self.max_val.saturating_mul(rhs.max_val);
    }
}

/// A size whose exact value is not yet decided. The value of `size` is
/// `product(size.factors())/product(size.divisors())`.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct PartialSize<'a> {
    static_factor: u32,
    param_factors: Vec<&'a ir::Parameter>,
    dim_factors: VecSet<ir::DimId>,
    divisors: VecSet<ir::DimId>,
}

impl<'a> PartialSize<'a> {
    /// Creates a new 'PartialSize'.
    pub fn new(factor: u32, params: Vec<&'a ir::Parameter>) -> Self {
        assert!(factor != 0);
        PartialSize {
            static_factor: factor,
            param_factors: params,
            ..Self::default()
        }
    }

    /// Creates a new `PartialSize` equals to the size of a dimension.
    pub fn new_dim_size(dim: ir::DimId) -> Self {
        PartialSize {
            dim_factors: VecSet::new(vec![dim]),
            ..Self::default()
        }
    }

    /// Add divisors to the size.
    pub fn add_divisors(&mut self, divisors: &VecSet<ir::DimId>) {
        self.divisors = self.divisors.union(divisors);
        self.simplify();
    }

    /// Returns the size of a dimension if it is staticaly known.
    pub fn as_int(&self) -> Option<u32> {
        let no_params = self.param_factors.is_empty();
        if no_params && self.dim_factors.is_empty() && self.divisors.is_empty() {
            Some(self.static_factor)
        } else {
            None
        }
    }

    /// Simplifies the fraction factor/divisor.
    fn simplify(&mut self) {
        let dim_factors = std::mem::replace(&mut self.dim_factors, VecSet::default());
        let divisors = std::mem::replace(&mut self.divisors, VecSet::default());
        let (new_dim_factors, new_divisors) = dim_factors.relative_difference(divisors);
        self.dim_factors = new_dim_factors;
        self.divisors = new_divisors;
    }

    /// Returns the factors composing the size.
    pub fn factors(&self) -> (u32, &[&'a ir::Parameter], &[ir::DimId]) {
        (self.static_factor, &self.param_factors, &self.dim_factors)
    }

    /// Returns the divisors composing the size.
    pub fn divisors(&self) -> &[ir::DimId] {
        &self.divisors
    }
}

impl<'a> Default for PartialSize<'a> {
    fn default() -> Self {
        PartialSize {
            static_factor: 1,
            param_factors: Vec::new(),
            dim_factors: VecSet::default(),
            divisors: VecSet::default(),
        }
    }
}

impl<'a, 'b> std::ops::MulAssign<&'b PartialSize<'a>> for PartialSize<'a> {
    fn mul_assign(&mut self, rhs: &'b PartialSize<'a>) {
        self.static_factor *= rhs.static_factor;
        self.param_factors.extend(rhs.param_factors.iter().cloned());
        self.dim_factors = self.dim_factors.union(&rhs.dim_factors);
        self.divisors = self.divisors.union(&rhs.divisors);
        self.simplify();
    }
}

impl<'a, 'b> std::ops::Mul<&'b PartialSize<'a>> for PartialSize<'a> {
    type Output = Self;

    fn mul(mut self, rhs: &PartialSize<'a>) -> Self {
        self *= rhs;
        self
    }
}

impl<'a, 'b> std::iter::Product<&'b PartialSize<'a>> for PartialSize<'a>
where
    'a: 'b,
{
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'b PartialSize<'a>>,
    {
        let mut static_factor = 1;
        let mut param_factors = vec![];
        let mut dim_factors = vec![];
        let mut divisors = vec![];
        for s in iter {
            static_factor *= s.static_factor;
            param_factors.extend(s.param_factors.iter().cloned());
            dim_factors.extend(s.dim_factors.iter().cloned());
            divisors.extend(s.divisors.iter().cloned());
        }
        let dim_factors = VecSet::new(dim_factors);
        let divisors = VecSet::new(divisors);
        let mut total = PartialSize {
            static_factor,
            param_factors,
            dim_factors,
            divisors,
        };
        total.simplify();
        total
    }
}

impl<'a> From<Size<'a>> for PartialSize<'a> {
    fn from(size: Size<'a>) -> PartialSize<'a> {
        PartialSize::new(size.factor, size.params)
    }
}
