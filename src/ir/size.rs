use serde::{Deserialize, Serialize};
use std;
use std::sync::Arc;

use utils::*;

use crate::ir;

/// A fully specified size.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Size {
    factor: u32,
    params: Vec<Arc<ir::Parameter>>,
    max_val: u32,
}

impl Size {
    /// Create a new fully specified size.
    pub fn new(factor: u32, params: Vec<Arc<ir::Parameter>>, max_val: u32) -> Self {
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
    pub fn new_param(param: Arc<ir::Parameter>, max_val: u32) -> Size {
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

impl Default for Size {
    fn default() -> Self {
        Size {
            factor: 1,
            params: Vec::new(),
            max_val: 1,
        }
    }
}

impl<T> std::ops::MulAssign<T> for Size
where
    T: std::borrow::Borrow<Size>,
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
#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct PartialSize {
    static_factor: u32,
    param_factors: Vec<Arc<ir::Parameter>>,
    dim_factors: VecSet<ir::DimId>,
    divisors: VecSet<ir::DimId>,
}

impl PartialSize {
    /// Creates a new 'PartialSize'.
    pub fn new(factor: u32, params: Vec<Arc<ir::Parameter>>) -> Self {
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
    pub fn factors(&self) -> (u32, &[Arc<ir::Parameter>], &[ir::DimId]) {
        (self.static_factor, &self.param_factors, &self.dim_factors)
    }

    /// Returns the divisors composing the size.
    pub fn divisors(&self) -> &[ir::DimId] {
        &self.divisors
    }
}

impl Default for PartialSize {
    fn default() -> Self {
        PartialSize {
            static_factor: 1,
            param_factors: Vec::new(),
            dim_factors: VecSet::default(),
            divisors: VecSet::default(),
        }
    }
}

impl<'a> std::ops::MulAssign<&'a PartialSize> for PartialSize {
    fn mul_assign(&mut self, rhs: &'a PartialSize) {
        self.static_factor *= rhs.static_factor;
        self.param_factors.extend(rhs.param_factors.iter().cloned());
        self.dim_factors = self.dim_factors.union(&rhs.dim_factors);
        self.divisors = self.divisors.union(&rhs.divisors);
        self.simplify();
    }
}

impl<'a> std::ops::Mul<&'a PartialSize> for PartialSize {
    type Output = Self;

    fn mul(mut self, rhs: &PartialSize) -> Self {
        self *= rhs;
        self
    }
}

impl<'a> std::iter::Product<&'a PartialSize> for PartialSize {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a PartialSize>,
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

impl From<Size> for PartialSize {
    fn from(size: Size) -> PartialSize {
        PartialSize::new(size.factor, size.params)
    }
}
