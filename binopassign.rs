use std::borrow::Borrow;
use std::cell::Cell;
use std::cmp;
use std::collections::hash_map::RandomState;
use std::convert::AsRef;
use std::fmt;
use std::hash::{BuildHasher, BuildHasherDefault, Hash, Hasher};
use std::iter;
use std::ops::{
    Add, AddAssign, Deref, DerefMut, Div, DivAssign, Mul, MulAssign, Sub, SubAssign,
};
use std::rc::Rc;

/// A trait representing objects that can perform assignment to a reference-counted pointer.
trait Assigner<T>: Sized {
    /// Perfom the assignment.
    ///
    /// Calling `assign` should conceptually be identical to `*dst = Rc::new(src)`, but allows
    /// implementers to perform additional bookkeeping in addition to the assignment.
    ///
    /// The default implementation simply creates a new `Rc` and assigns `dst` to it.
    fn assign(self, dst: &mut Rc<T>, src: T) {
        *dst = Rc::new(src)
    }

    fn self_assign(self, dst: &mut T) {}
}

/// A default implementation of the `Assigner` that does nothing but perform the assignment.
#[derive(Copy, Clone, Debug)]
struct DefaultAssigner;

impl<T> Assigner<T> for DefaultAssigner {}

enum DeferredAssignmentInner<'a, T> {
    Initial(&'a mut Rc<T>),
    Unique(&'a mut T),
    Multiple { initial: &'a mut Rc<T>, new: T },
}

struct DeferredAssignment<'a, T, A = DefaultAssigner>
where
    A: Assigner<T>,
{
    inner: Option<DeferredAssignmentInner<'a, T>>,
    assigner: Option<A>,
}

impl<'a, T, A> DeferredAssignment<'a, T, A>
where
    T: Clone,
    A: Assigner<T>,
{
    fn new(rc: &'a mut Rc<T>, assigner: A) -> Self {
        DeferredAssignment {
            inner: Some(DeferredAssignmentInner::Initial(rc)),
            assigner: Some(assigner),
        }
    }

    fn as_ref(&self) -> &T {
        use DeferredAssignmentInner::*;
        match self.inner.as_ref().unwrap() {
            Initial(initial) => &**initial,
            Multiple { new, .. } => new,
            Unique(new) => new,
        }
    }

    fn to_mut(&mut self) -> &mut T {
        use DeferredAssignmentInner::*;
        if {
            if let Initial(_) = self.inner.as_ref().unwrap() {
                false
            } else {
                true
            }
        } {
            let initial = if let Initial(initial) = self.inner.take().unwrap() {
                initial
            } else {
                unreachable!()
            };

            if Rc::get_mut(initial).is_some() {
                self.inner = Some(Unique(Rc::get_mut(initial).unwrap()));
            } else {
                self.inner = Some(Multiple {
                    new: (**initial).clone(),
                    initial,
                })
            }
        }

        match self.inner.as_mut().unwrap() {
            Initial(_) => unreachable!(),
            Unique(new) => new,
            Multiple { new, .. } => new,
        }
    }
}

impl<'a, T, A> Drop for DeferredAssignment<'a, T, A>
where
    A: Assigner<T>,
{
    fn drop(&mut self) {
        use DeferredAssignmentInner::*;

        match self.inner.take().unwrap() {
            Initial(_) => (),
            Unique(new) => self.assigner.take().unwrap().self_assign(new),
            Multiple { initial, new } => {
                self.assigner.take().unwrap().assign(initial, new)
            }
        }
    }
}

trait AsRed<R> {
    fn as_reduction(&self) -> Option<&R>;
    fn as_reduction_mut(&mut self) -> Option<&mut R>;
}

impl<'a, T, F, R> AsRed<R> for DeferredAssignment<'a, T, F>
where
    T: Clone + AsRed<R>,
    F: Assigner<T>,
{
    fn as_reduction(&self) -> Option<&R> {
        self.as_ref().as_reduction()
    }

    fn as_reduction_mut(&mut self) -> Option<&mut R> {
        // Do not force a clone if we are not actually a reduction.
        if self.as_ref().as_reduction().is_none() {
            None
        } else {
            Some(self.to_mut().as_reduction_mut().unwrap())
        }
    }
}

// let lhs = DeferredAssignment::new(lhs, BinOpAssigner::new(op, rhs));

struct BinOpAssign<'a, O, Lhs, Rhs = Lhs> {
    op: O,
    orig_lhs: Option<&'a mut Rc<Lhs>>,
    lhs: Option<Result<&'a mut Lhs, Lhs>>,
    rhs: &'a Rhs,
}

impl<'a, O, Lhs, Rhs> BinOpAssign<'a, O, Lhs, Rhs> {
    fn new(op: O, lhs: &'a mut Rc<Lhs>, rhs: &'a Rhs) -> Self {
        BinOpAssign {
            op,
            orig_lhs: Some(lhs),
            lhs: None,
            rhs,
        }
    }

    fn lhs(&self) -> &Lhs {
        match &self.lhs {
            None => &**self.orig_lhs.as_ref().unwrap(),
            Some(Ok(lhs)) => lhs,
            Some(Err(lhs)) => lhs,
        }
    }

    fn lhs_mut(&mut self) -> &mut Lhs
    where
        Lhs: Clone,
    {
        if self.lhs.is_none() {
            let orig_lhs = self.orig_lhs.take().unwrap();
            if Rc::get_mut(orig_lhs).is_some() {
                self.lhs = Some(Ok(Rc::get_mut(orig_lhs).unwrap()));
            } else {
                self.lhs = Some(Err((**orig_lhs).clone()));
                self.orig_lhs = Some(orig_lhs);
            }
        }

        // Deref magic happens below
        match self.lhs.as_mut().unwrap() {
            Ok(lhs) => lhs,
            Err(lhs) => lhs,
        }
    }

    fn rhs(&self) -> &Rhs {
        self.rhs
    }
}

impl<'a, O, Lhs, Rhs> Drop for BinOpAssign<'a, O, Lhs, Rhs> {
    fn drop(&mut self) {}
}

impl<'a, O, Lhs, Rhs, R> AsRed<R> for BinOpAssign<'a, O, Lhs, Rhs>
where
    Lhs: Clone + AsRed<R>,
{
    fn as_reduction(&self) -> Option<&R> {
        self.lhs().as_reduction()
    }

    fn as_reduction_mut(&mut self) -> Option<&mut R> {
        // Do not force a clone if it is not needed
        if self.lhs.is_none() && self.orig_lhs.as_ref().unwrap().as_reduction().is_none()
        {
            None
        } else {
            self.lhs_mut().as_reduction_mut()
        }
    }
}

fn main() {}
