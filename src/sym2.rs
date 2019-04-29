use num::Num;

/// A trait representing symbolic types.
trait Symbolic: AsNumeric<Self::Numeric> {
    /// The base numeric type for this type.
    type Numeric: Num + Into<Self>;
}

struct Expr<N, E> {
    numeric: N,
    symbolic: E,
}

type ExprRef<'a, E> = Expr<&'a N, &'a E>;
type ExprMut<'a, E> = Expr<&'a mut N, &'a mut E>;

impl<'a, N, E> ExprRef<'a, N, E>
where
    N: Clone,
    E: Clone,
{
    fn to_expr(self) -> Expr<N, E> {
        Expr {
            numeric: self.numeric.clone(),
            symbolic: self.symbolic.clone(),
        }
    }
}

impl<N, E> Expr<N, E> {
    fn as_mut(&mut self) -> ExprMut<N, E> {
        Expr {
            numeric: &mut self.numeric,
            symbolic: &mut self.symbolic,
        }
    }
}

trait AsExpr<E>: Symbolic + From<Expr<Self::Numeric, E>> {
    fn as_expr(&self) -> Option<ExprRef<Self::Numeric, E>>;
    fn as_expr_mut(&mut self) -> Option<ExprMut<Self::Numeric, E>>;
}

trait BinOp<S>
where
    S: Symbolic,
{
    type Expr;

    fn fast_apply_assign(&self, _lhs: &mut DeferredBinOp<Self, S>, _rhs: &S) -> bool {
        false
    }

    fn apply_num_num(&self, lhs: &S::Numeric, rhs: &S::Numeric) -> S;

    fn apply_num_expr(&self, lhs: &S::Numeric, rhs: ExprRef<S::Numeric, Self::Expr>)
        -> S;

    fn apply_num_other(
        &self,
        lhs: &S::Numeric,
        rhs: ExprRef<S::Numeric, Self::Expr>,
    ) -> S;

    fn apply_assign_num(&self, lhs: ExprMut<S::Numeric, Self::Expr>, rhs: &S::Numeric);

    fn apply_assign_expr(
        &self,
        lhs: ExprMut<S::Numeric, Self::Expr>,
        rhs: ExprRef<S::Numeric, Self::Expr>,
    );

    fn apply_assign_other(&self, lhs: ExprMut<S::Numeric, Self::Expr>, rhs: &S);

    fn apply_other_num(&self, lhs: &S, rhs: &S::Numeric) -> S;

    fn apply_other_expr(&self, lhs: &S, rhs: ExprRef<S::Numeric, Self::Expr>) -> S;

    fn apply_other_other(&self, lhs: &S, rhs: &S) -> S;
}

trait NeutralElement<S>: BinOp<S>
where
    S: Symbolic,
{
    fn neutral(&self) -> Self::Expr;
}

trait Reduce<S, A = S> {
    fn reduce<I>(&self, mut iter: I) -> S
    where
        I: Iterator<Item = A>;
}

struct Mul;

impl<S, A> iter::Product<A> for S
where
    S: Symbolic,
    Add: Reduce<S, A>,
{
    fn product<I>(iter: I) -> S
    where
        I: Iterator<Item = A>,
    {
        Mul.reduce(iter)
    }
}

struct Add;

impl<S, A> iter::Sum<A> for S
where
    S: Symbolic,
    Add: Reduce<S, A>,
{
    fn sum<I>(iter: I) -> S
    where
        I: Iterator<Item = A>,
    {
        Add.reduce(iter)
    }
}

impl<S> Reduce<S>
where
    Self: NeutralElement<S> + BinOp<S>,
    S: Symbolic + AsExpr<Self::Expr>,
{
    fn reduce<I>(&self, mut iter: I) -> S
    where
        I: Iterator<Item = S>,
    {
        self.maybe_reduce(iter).unwrap_or_else(|| self.neutral());
    }
}

impl<'a, S, O> Reduce<S, &'a S> for O
where
    O: NeutralElement<S> + BinOp<S>,
    S: AsExpr<Self::Expr>,
{
    fn reduce<I>(mut iter: I) -> S
    where
        I: Iterator<Item = &'a S>,
    {
        self.maybe_reduce(iter).unwrap_or_else(Self::neutral)
    }
}

trait Fold<S, A = S>: BinOp<S>
where
    S: Symbolic + AsExpr<Self::Expr>,
{
    fn fold<I>(init: S, iter: I) -> S
    where
        I: Iterator<Item = A>;
}

impl<S, O> Reduce1<S> for O
where
    O: BinOp<S>,
    S: Symbolic + AsExpr<Self::Expr>,
{
    fn fold<I>(mut init: Expr<S::Numeric, Self::Expr>, iter: I) -> S
    where
        I: Iterator<Item = S>,
    {
        for item in iter {
            if let Some(num) = item.as_numeric() {
                self.apply_assign_num(&mut init, num);
            } else if let Some(expr) = item.as_expr() {
                self.apply_assign_expr(&mut init, expr);
            } else {
                self.apply_assign_other(&mut init, item);
            }
        }

        init.into()
    }
}

impl<'a, S, O> MaybeReduce<S, &'a S> for O
where
    O: BinOp<S>,
    S: AsExpr<Self::Expr> + Clone,
{
    fn maybe_reduce<I>(mut iter: I) -> Option<S>
    where
        I: Iterator<Item = &'a S>,
    {
        if let Some(result) = iter.next() {
            let mut result = result.clone();

            for item in iter {
                if let Some(num) = item.as_numeric() {
                    self.apply_assign_num(&mut result, num);
                } else if let Some(expr) = item.as_expr() {
                    self.apply_assign_expr(&mut result, expr);
                } else {
                    self.apply_assign_other(&mut result, item);
                }
            }

            Some(result.into())
        } else {
            None
        }
    }
}

trait Apply<Lhs, Rhs = Lhs> {
    type Output;

    fn apply(&self, lhs: Lhs, rhs: Rhs) -> Self::Output;
}

trait ApplyAssign<Lhs, Rhs = Lhs> {
    fn apply_assign(&self, lhs: &mut Lhs, rhs: Rhs);
}

impl<Lhs, Rhs, O> Apply<Lhs, Rhs> for O
where
    O: ApplyAssign<Lhs, Rhs>,
    Lhs: Symbolic,
{
    type Output = Lhs;

    fn apply(&self, mut lhs: Lhs, rhs: Rhs) -> Self::Output {
        self.apply_assign(&mut lhs, rhs);
        lhs
    }
}

impl<'a, S, O> ApplyAssign<S, &'a S> for O
where
    O: BinOp<S>,
    S: Symbolic + AsExpr<O::Expr>,
{
    fn apply_assign(&self, lhs: &mut S, rhs: &S) {
        let mut lhs = DeferredAssignment::new(lhs, BinOpAssigner::new(self, rhs));
        if self.fast_apply_assign(&mut lhs, rhs) {
            // Assignment done in the fast path
        } else if let Some(rhs_num) = rhs.as_numeric() {
            if let Some(lhs_num) = lhs.as_numeric_mut() {
                *lhs.to_mut() = self.apply_num_num(lhs, rhs);
            } else if let Some(lhs_red) = lhs.as_expr_mut() {
                self.reduce_assign_num(lhs_red, rhs_num);
            } else {
                *lhs.to_mut() = self.apply_other_num(&*lhs, rhs_num);
            }
        } else if let Some(rhs_red) = rhs.as_expr() {
            if let Some(lhs_num) = lhs.as_numeric() {
                *lhs.to_mut() = self.apply_num_expr(lhs_num, rhs_red);
            } else if let Some(lhs_red) = lhs.as_expr_mut() {
                self.apply_assign_expr(lhs_red, rhs_red);
            } else {
                *lhs.to_mut() = self.apply_other_expr(&*lhs, rhs_red);
            }
        } else if let Some(lhs_num) = lhs.as_numeric() {
            *lhs.to_mut() = self.apply_num_other(lhs_num, rhs);
        } else if let Some(lhs_red) = lhs.as_expr_mut() {
            self.apply_assign_other(lhs_red, rhs);
        } else {
            *lhs.to_mut() = self.apply_other_other(&*lhs, rhs);
        }
    }
}

impl<S, Rhs> AddAssign<Rhs> for S
where
    S: Symbolic,
    Add: ApplyAssign<S, Rhs>,
{
    fn add_assign(&mut self, other: &Rhs) {
        Add.apply_assign(self, other)
    }
}

struct Sub;

impl<S, Rhs> SubAssign<Rhs> for S
where
    S: Symbolic,
    Sub: ApplyAssign<S, Rhs>,
{
    fn sub_assign(&mut self, other: &Rhs) {
        Sub.apply_assign(self, other)
    }
}

impl<S, Rhs> MulAssign<Rhs> for S
where
    S: Symbolic,
    Mul: ApplyAssign<S, Rhs>,
{
    fn mul_assign(&mut self, other: &Rhs) {
        Mul.apply_assign(self, other)
    }
}

struct Div;

impl<S, Rhs> DivAssign<Rhs> for S
where
    S: Symbolic,
    Div: ApplyAssign<S, Rhs>,
{
    fn div_assign(&mut self, other: &Rhs) {
        Div.apply_assign(self, other)
    }
}

impl<S, Rhs> ops::Add<Rhs> for S
where
    S: Symbolic,
    Add: Apply<S, Rhs>,
{
    type Output = <Add as Apply<S, Rhs>>::Output;

    fn add(self, other: Rhs) -> Self::Output {
        Add.apply(self, other)
    }
}
