use ir;
use num;
use std;

/// A fully specified size.
#[derive(Clone, Debug)]
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

/// A size whose exact value is not yet decided.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct PartialSize<'a> {
    factor: u32,
    dividend: Vec<&'a ir::Parameter>,
    divisor: u32,
}

impl<'a> PartialSize<'a> {
    /// Creates a new 'PartialSize'.
    pub fn new(factor: u32, dividend: Vec<&'a ir::Parameter>, divisor: u32) -> Self {
        assert!(factor != 0);
        assert!(divisor != 0);
        let mut new = PartialSize {
            factor,
            dividend,
            divisor,
        };
        new.simplify();
        new
    }

    /// Returns the size of a dimension if it is staticaly known.
    pub fn as_int(&self) -> Option<u32> {
        if self.dividend.is_empty() {
            Some(self.factor)
        } else {
            None
        }
    }

    /// Indicates if the size is constant.
    pub fn is_constant(&self) -> bool {
        self.dividend.is_empty()
    }

    /// Returns the dividends.
    pub fn dividend(&self) -> &[&'a ir::Parameter] {
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

    /// Multiplies the divisor by the given factor.
    pub fn mul_divisor(&mut self, d: u32) {
        assert_ne!(d, 0);
        self.divisor *= d;
        self.simplify();
    }

    /// Multiplies the factor by the given factor.
    pub fn mul_factor(&mut self, d: u32) {
        assert_ne!(d, 0);
        self.factor *= d;
        self.simplify();
    }

    /// Simplifies the fraction factor/divisor.
    fn simplify(&mut self) {
        let gcd = num::integer::gcd(self.factor, self.divisor);
        self.factor /= gcd;
        self.divisor /= gcd;
    }

    /// Indicates if two sizes may be equal, meaning they are equal appart from the
    /// decisions they depend on.
    pub fn is_compatible_with(&self, other: &ir::PartialSize) -> bool {
        self.factor == other.factor
            && self.divisor == other.divisor
            && self.dividend == other.dividend
    }
}

impl<'a, 'b> std::ops::MulAssign<&'b PartialSize<'a>> for PartialSize<'a> {
    fn mul_assign(&mut self, rhs: &'b PartialSize<'a>) {
        self.factor *= rhs.factor;
        self.dividend.extend(rhs.dividend.iter().cloned());
        self.divisor *= rhs.divisor;
        self.simplify();
    }
}

impl<'a> From<Size<'a>> for PartialSize<'a> {
    fn from(size: Size<'a>) -> PartialSize<'a> {
        PartialSize {
            factor: size.factor,
            dividend: size.params,
            divisor: 1,
        }
    }
}
