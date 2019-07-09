use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use std::{iter, ops};

use super::hash::MemoizedHash;

pub trait Comparator<T> {
    #[must_use]
    fn cmp(&self, lhs: &T, rhs: &T) -> Ordering;

    #[must_use]
    fn lt(&self, lhs: &T, rhs: &T) -> bool {
        match self.cmp(lhs, rhs) {
            Ordering::Less => true,
            _ => false,
        }
    }

    #[must_use]
    fn le(&self, lhs: &T, rhs: &T) -> bool {
        match self.cmp(lhs, rhs) {
            Ordering::Less | Ordering::Equal => true,
            _ => false,
        }
    }

    #[must_use]
    fn gt(&self, lhs: &T, rhs: &T) -> bool {
        match self.cmp(lhs, rhs) {
            Ordering::Greater => true,
            _ => false,
        }
    }

    #[must_use]
    fn ge(&self, lhs: &T, rhs: &T) -> bool {
        match self.cmp(lhs, rhs) {
            Ordering::Greater | Ordering::Equal => true,
            _ => false,
        }
    }
}

#[derive(Debug, Copy, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OrdComparator;

impl<T: Ord> Comparator<T> for OrdComparator {
    fn cmp(&self, lhs: &T, rhs: &T) -> Ordering {
        lhs.cmp(rhs)
    }

    fn lt(&self, lhs: &T, rhs: &T) -> bool {
        *lhs < *rhs
    }

    fn le(&self, lhs: &T, rhs: &T) -> bool {
        *lhs <= *rhs
    }

    fn gt(&self, lhs: &T, rhs: &T) -> bool {
        *lhs > *rhs
    }

    fn ge(&self, lhs: &T, rhs: &T) -> bool {
        *lhs >= *rhs
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VecSet<T, C = OrdComparator> {
    comparator: C,
    values: Vec<T>,
}

fn cmp<T, C>(comparator: &C, lhs: &[T], rhs: &[T]) -> Ordering
where
    C: Comparator<T>,
{
    match lhs.len().cmp(&rhs.len()) {
        Ordering::Equal => {
            // Eliminates bounds checks
            let l = std::cmp::min(lhs.len(), rhs.len());
            let lhs = &lhs[..l];
            let rhs = &rhs[..l];

            for i in 0..l {
                match comparator.cmp(&lhs[i], &rhs[i]) {
                    Ordering::Equal => (),
                    ord => return ord,
                }
            }

            Ordering::Equal
        }
        ord => ord,
    }
}

impl<T, C> PartialOrd for VecSet<T, C>
where
    T: PartialEq,
    C: Comparator<T> + PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.comparator.partial_cmp(&other.comparator) {
            Some(Ordering::Equal) => {
                Some(cmp(&self.comparator, &self.values, &other.values))
            }
            ord => ord,
        }
    }
}

impl<T, C> Ord for VecSet<T, C>
where
    T: Eq,
    C: Comparator<T> + Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        match Ord::cmp(&self.comparator, &other.comparator) {
            Ordering::Equal => cmp(&self.comparator, &self.values, &other.values),
            ord => ord,
        }
    }
}

impl<T, C> Default for VecSet<T, C>
where
    C: Comparator<T> + Default,
{
    fn default() -> Self {
        VecSet {
            comparator: C::default(),
            values: Vec::default(),
        }
    }
}

impl<T, C> iter::FromIterator<T> for VecSet<T, C>
where
    T: PartialEq,
    C: Comparator<T> + Default,
{
    fn from_iter<II>(iter: II) -> Self
    where
        II: IntoIterator<Item = T>,
    {
        let comparator = C::default();
        let mut values = Vec::from_iter(iter);
        values.sort_by(|lhs, rhs| comparator.cmp(lhs, rhs));
        values.dedup();

        VecSet { comparator, values }
    }
}

impl<T, C> VecSet<T, C> {
    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.values.iter()
    }

    pub fn sort(&mut self)
    where
        T: PartialEq,
        C: Comparator<T>,
    {
        self.values.sort_by({
            let comparator = &self.comparator;
            move |lhs, rhs| comparator.cmp(lhs, rhs)
        });
        self.values.dedup();
    }

    pub(super) fn unchecked_iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        // TODO: This breaks the order!  Need resort.
        self.values.iter_mut()
    }

    pub(super) fn unchecked_remove(&mut self, pos: usize) {
        self.values.remove(pos);
    }

    pub(super) fn unchecked_insert(&mut self, pos: usize, item: T) {
        self.values.insert(pos, item);
    }

    pub fn singleton(value: T) -> Self
    where
        C: Default + Comparator<T>,
    {
        VecSet {
            comparator: C::default(),
            values: vec![value],
        }
    }

    pub fn from_sorted_iter<II>(values: II) -> Self
    where
        C: Default + Comparator<T>,
        II: IntoIterator<Item = T>,
    {
        VecSet {
            comparator: C::default(),
            values: values.into_iter().collect(),
        }
    }
}

impl<T, C, I> ops::Index<I> for VecSet<T, C>
where
    Vec<T>: ops::Index<I>,
{
    type Output = <Vec<T> as ops::Index<I>>::Output;

    fn index(&self, index: I) -> &Self::Output {
        ops::Index::index(&self.values, index)
    }
}

impl<T, C> AsRef<[T]> for VecSet<T, C> {
    fn as_ref(&self) -> &[T] {
        &self.values
    }
}
