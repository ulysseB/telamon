use ir;
use num;
use std;
use search_space::SearchSpace;

/// The size of an iteration dimension. The size is of the form:
/// `(factor * dividend_0 * dividend_1 * ...)) / divisor`
/// where the reminder of the division is null.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Size<'a> {
    factor: u32,
    dividend: Vec<&'a ir::Parameter>,
    divisor: u32,
}

impl<'a> Size<'a> {
    /// Creates a new 'Size'.
    pub fn new(factor: u32, dividend: Vec<&'a ir::Parameter>, divisor: u32) -> Self {
        assert!(factor != 0);
        assert!(divisor != 0);
        let mut new = Size { factor, dividend, divisor };
        new.simplify();
        new
    }

    /// Converts an `ir::Size` to `Self`.
    pub fn from_ir(size: &ir::Size<'a>, _: &SearchSpace) -> Self {
        let dividend = size.dividend().iter().cloned().collect();
        Size::new(size.factor(), dividend, size.divisor())
    }

    /// Returns the size of a dimension if it is staticaly known.
    pub fn as_int(&self) -> Option<u32> {
        if self.dividend.is_empty() { Some(self.factor) } else { None }
    }

    /// Indicates if the size is constant.
    pub fn is_constant(&self) -> bool { self.dividend.is_empty() }

    /// Returns the dividends.
    pub fn dividend(&self) -> &[&'a ir::Parameter] { &self.dividend }

    /// Returns the divisor.
    pub fn divisor(&self) -> u32 { self.divisor }

    /// Returns the factor.
    pub fn factor(&self) -> u32 { self.factor }

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
}

impl<'a, 'b> std::ops::MulAssign<&'b Size<'a>> for Size<'a> {
    fn mul_assign(&mut self, rhs: &'b Size<'a>) {
        self.factor *= rhs.factor;
        self.dividend.extend(rhs.dividend.iter().cloned());
        self.divisor *= rhs.divisor;
        self.simplify();
    }
}
