use std::sync::Arc;
use std::{fmt, ops};

use num;
use utils::unwrap;

use crate::ir;
use crate::search_space::{NumSet, SearchSpace};

/// The size of an iteration dimension. The size is of the form:
/// `(factor * dividend_0 * dividend_1 * ...)) / divisor`
/// where the remainder of the division is null.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Size {
    factor: u32,
    dividend: Vec<Arc<ir::Parameter>>,
    divisor: u32,
}

impl Size {
    /// Creates a new 'Size'.
    pub fn new(factor: u32, dividend: Vec<Arc<ir::Parameter>>, divisor: u32) -> Self {
        assert!(divisor != 0);
        let mut new = Size {
            factor,
            dividend,
            divisor,
        };
        new.simplify();
        new
    }

    /// Converts an `ir::PartialSize` to `Self`.
    pub fn from_ir(size: &ir::PartialSize, space: &SearchSpace) -> Self {
        let (cst_factor, param_factors, dim_size_factors) = size.factors();
        let dim_size_divisors = size.divisors();
        let factor = cst_factor
            * dim_size_factors
                .iter()
                .map(|&d| dim_size(d, space))
                .product::<u32>();
        let divisor = dim_size_divisors
            .iter()
            .map(|&d| dim_size(d, space))
            .product();
        Self::new(factor, param_factors.to_vec(), divisor)
    }

    /// Returns the size of a dimension if it is staticaly known.
    pub fn as_int(&self) -> Option<u32> {
        if self.dividend.is_empty() {
            assert_eq!(self.divisor, 1);

            Some(self.factor)
        } else {
            None
        }
    }

    /// Returns the dividends.
    pub fn dividend(&self) -> &[Arc<ir::Parameter>] {
        &self.dividend
    }

    /// Returns the divisor.
    pub fn divisor(&self) -> u32 {
        self.divisor
    }

    /// Returns the factor.
    pub fn factor(&self) -> u32 {
        self.factor
    }

    /// Simplifies the fraction factor/divisor.
    fn simplify(&mut self) {
        let gcd = num::integer::gcd(self.factor, self.divisor);
        self.factor /= gcd;
        self.divisor /= gcd;
        self.dividend.sort_by(|lhs, rhs| lhs.name.cmp(&rhs.name));
    }
}

impl ops::MulAssign<u32> for Size {
    fn mul_assign(&mut self, other: u32) {
        self.factor *= other;
    }
}

impl ops::MulAssign<&'_ Size> for Size {
    fn mul_assign(&mut self, other: &'_ Size) {
        self.factor *= other.factor;
        self.dividend.extend(other.dividend.iter().cloned());
        self.divisor *= other.divisor;
        self.simplify();
    }
}

impl ops::MulAssign<&'_ ir::Size> for Size {
    fn mul_assign(&mut self, other: &'_ ir::Size) {
        self.factor *= other.factor();
        self.dividend.extend(other.params().iter().cloned());
        self.simplify();
    }
}

impl<T> ops::Mul<T> for Size
where
    Size: ops::MulAssign<T>,
{
    type Output = Size;

    fn mul(mut self, other: T) -> Self::Output {
        self *= other;
        self
    }
}

impl<T> ops::Mul<T> for &'_ Size
where
    Size: ops::Mul<T>,
{
    type Output = <Size as ops::Mul<T>>::Output;

    fn mul(self, other: T) -> Self::Output {
        self.clone() * other
    }
}

impl fmt::Display for Size {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let mut pre = if self.factor != 1 {
            write!(fmt, "{}", self.factor)?;
            true
        } else {
            false
        };

        for p in &self.dividend {
            if pre {
                write!(fmt, "*")?;
            }
            write!(fmt, "{}", p)?;
            pre = true;
        }

        if self.divisor != 1 {
            write!(fmt, "/{}", self.divisor)?;
        }

        Ok(())
    }
}

impl From<u32> for Size {
    fn from(size: u32) -> Self {
        Size::new(size, Vec::new(), 1u32)
    }
}

impl From<&'_ ir::Size> for Size {
    fn from(size: &'_ ir::Size) -> Self {
        Size::new(size.factor(), size.params().to_vec(), 1u32)
    }
}

/// Returns the size of a static dimension from the domain.
fn dim_size(dim: ir::DimId, space: &SearchSpace) -> u32 {
    let universe = unwrap!(space.ir_instance().dim(dim).possible_sizes());
    let size = space.domain().get_size(dim).as_constrained(universe);
    unwrap!(size, "dim {} is not constrained", dim)
}
