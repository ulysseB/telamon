use ir;
use std;
use std::borrow::Borrow;
use utils::*;

/// The size of an iteration dimension. The size is of the form:
/// `(factor * dividend_0 * dividend_1 * ...)) / divisor`
/// where the reminder of the division is null.
#[derive(Clone, Debug)]
pub struct Size<'a> {
    static_factor: u32,
    params_factor: Vec<&'a ir::Parameter>,
    dividend: VecSet<ir::dim::Id>,
    divisor: VecSet<ir::dim::Id>,
}

impl<'a> Size<'a> {
    /// Creates a new `Size` representing the constant `1`.
    pub fn one() -> Self {
        Size {
            static_factor: 1,
            params_factor: vec![],
            dividend: VecSet::default(),
            divisor: VecSet::default(),
        }
    }

    /// Creates a new size.
    pub fn new(static_factor: u32, params_factor: Vec<&'a ir::Parameter>) -> Self {
        Size { static_factor, params_factor, .. Self::one() }
    }

    /// Creates a size that takes the value of a parameter.
    pub fn new_param(p: &'a ir::Parameter) -> Self {
        Size { params_factor: vec![p], .. Self::one() }
    }

    /// Creates a size that has a constant value.
    pub fn new_const(static_factor: u32) -> Self {
        Size { static_factor, .. Self::one() }
    }

    /// Creates a size equal to the number of iterations of a dimension.
    pub fn new_dim(id: ir::dim::Id) -> Self {
        Size { dividend: VecSet::new(vec![id]), .. Self::one() }
    }

    /// Returns the members composing the size in the form `(a, b, c, d)` where:
    /// `size = a * product(b) * product(c) / product(d)`.
    pub fn members(&self) -> (u32, &[&'a ir::Parameter], &[ir::dim::Id], &[ir::dim::Id]) {
        (self.static_factor, &self.params_factor, &self.dividend, &self.divisor)
    }

    /// Indicates if the size depends on the size of a dimension.
    pub fn depends_on_dim(&self) -> bool {
        !self.dividend.is_empty() || !self.divisor.is_empty()
    }

    /// Returns the size of a dimension if it is staticaly known and doesn't depend on
    /// any decisions.
    pub fn as_fixed(&self) -> Option<u32> {
        if self.params_factor.is_empty() &&
            self.dividend.is_empty() &&
            self.divisor.is_empty()
        {
            Some(self.static_factor)
        } else { None }
    }

    /// Divides the size by the size of the given dimensions.
    pub fn tile(&mut self, dims: &VecSet<ir::dim::Id>) {
        self.divisor = self.divisor.union(&dims);
        self.simplify();
    }

    /// Simplifies the fraction factor/divisor.
    fn simplify(&mut self) {
        let dividend = std::mem::replace(&mut self.dividend, VecSet::default());
        let divisor = std::mem::replace(&mut self.divisor, VecSet::default());
        let (new_dividend, new_divisor) = dividend.symmetric_difference(divisor);
        self.dividend = new_dividend;
        self.divisor = new_divisor;
    }

    /// Indicates if the dimension can be merged with another, assuming they have the
    /// same tiling factor and size choice value.
    pub fn is_compatible_with(&self, other: &ir::Size) -> bool {
        self.static_factor == other.static_factor
            && self.params_factor == other.params_factor
    }
}

impl<'a, T> std::ops::MulAssign<T> for Size<'a> where T: Borrow<ir::Size<'a>> {
    fn mul_assign(&mut self, rhs: T) {
        let rhs = rhs.borrow();
        self.static_factor *= rhs.static_factor;
        self.params_factor.extend(rhs.params_factor.iter().cloned());
        self.dividend = self.dividend.union(&rhs.dividend);
        self.divisor = self.divisor.union(&rhs.divisor);
        self.simplify();
    }
}

impl<'a, T> std::ops::Mul<T> for Size<'a> where T: Borrow<ir::Size<'a>> {
    type Output = Size<'a>;

    fn mul(self, rhs: T) -> Self::Output { (&self) * rhs }
}

impl<'a, 'b, T> std::ops::Mul<T> for &'b Size<'a> where T: Borrow<ir::Size<'a>> {
    type Output = Size<'a>;

    fn mul(self, rhs: T) -> Self::Output {
        let rhs = rhs.borrow();
        let static_factor = self.static_factor * rhs.static_factor;
        let params_factor = self.params_factor.iter().chain(&rhs.params_factor)
            .cloned().collect();
        let dividend = self.dividend.union(&rhs.dividend);
        let divisor = self.divisor.union(&rhs.divisor);
        let mut size = Size { static_factor, params_factor, dividend, divisor };
        size.simplify();
        size
    }
}

impl<'a, 'b> std::iter::Product<&'b ir::Size<'a>> for ir::Size<'a> where 'a: 'b  {
    fn product<I>(iter: I) -> Self where I: Iterator<Item=&'b ir::Size<'a>> {
        let mut static_factor = 1;
        let mut params_factor = vec![];
        let mut dividend = vec![];
        let mut divisor = vec![];
        for s in iter {
            static_factor *= s.static_factor;
            params_factor.extend(s.params_factor.iter().cloned());
            dividend.extend(s.dividend.iter().cloned());
            divisor.extend(s.divisor.iter().cloned());
        }
        let dividend = VecSet::new(dividend);
        let divisor = VecSet::new(divisor);
        let mut size = Size { static_factor, params_factor, dividend, divisor };
        size.simplify();
        size
    }
}
