//! Sets backed by ordered vectors.
use std;
use std::cmp::{Ordering, PartialOrd};

/// A set backed by an ordered vector.
#[derive(Clone, Eq, PartialEq, Hash)]
pub struct VecSet<T> {
    data: Vec<T>,
}

impl<T> VecSet<T>
where T: Ord
{
    /// Creates a new `VecSet` with the given data.
    pub fn new(mut data: Vec<T>) -> Self {
        data.sort();
        data.dedup();
        VecSet { data }
    }

    /// Indicates if the `VecSet` is empty.
    pub fn is_empty(&self) -> bool { self.data.is_empty() }

    /// Returns the number of elements in the set.
    pub fn len(&self) -> usize { self.data.len() }

    /// Iterates over the set, in order.
    pub fn iter(&self) -> std::slice::Iter<T> { self.data.iter() }

    /// Returns the elements in self but not in other.
    pub fn difference<'a>(&'a self, other: &'a VecSet<T>) -> Difference<T> {
        Difference {
            lhs: self,
            lhs_cursor: 0,
            rhs: other,
            rhs_cursor: 0,
        }
    }

    /// Returns `rhs-other` and `rhs-self`.
    pub fn symmetric_difference(mut self, mut other: Self) -> (Self, Self) {
        let lhs_old_len = self.data.len();
        let rhs_old_len = other.data.len();
        unsafe {
            // Make sure the moved elements are not drop twice if an exception occurs.
            self.data.set_len(0);
            other.data.set_len(0);
            // Iterate simultaneously on both vectors to remove duplicate elements.
            let (mut lhs_idx, mut lhs_del, mut rhs_idx, mut rhs_del) = (0, 0, 0, 0);
            while lhs_idx < lhs_old_len && rhs_idx < rhs_old_len {
                let lhs = self.data.as_mut_ptr().offset(lhs_idx as isize);
                let rhs = other.data.as_mut_ptr().offset(rhs_idx as isize);
                match (*lhs).cmp(&*rhs) {
                    std::cmp::Ordering::Less => {
                        if lhs_del > 0 {
                            let lhs_dst = self.data.as_mut_ptr().offset((lhs_idx - lhs_del) as isize);
                            std::ptr::copy_nonoverlapping(lhs, lhs_dst, 1);
                        }
                        lhs_idx += 1;
                    },
                    std::cmp::Ordering::Greater => {
                        if rhs_del > 0 {
                            let rhs_dst = other.data.as_mut_ptr().offset((rhs_idx - rhs_del) as isize);
                            std::ptr::copy_nonoverlapping(rhs, rhs_dst, 1);
                        }
                        rhs_idx += 1;
                    },
                    std::cmp::Ordering::Equal => {
                        std::mem::drop(&*lhs);
                        std::mem::drop(&*rhs);
                        lhs_idx += 1;
                        lhs_del += 1;
                        rhs_idx += 1;
                        rhs_del += 1;
                    }
                }
            }
            // Complete vectors that are not yet explored.
            if lhs_idx < lhs_old_len && lhs_del > 0 {
                let lhs_src = self.data.as_mut_ptr().offset(lhs_idx as isize);
                let lhs_dst = self.data.as_mut_ptr().offset((lhs_idx - lhs_del) as isize);
                std::ptr::copy(lhs_src, lhs_dst, lhs_old_len - lhs_idx);
            }
            if rhs_idx < rhs_old_len && rhs_del > 0 {
                let rhs_src = other.data.as_mut_ptr().offset(rhs_idx as isize);
                let rhs_dst = other.data.as_mut_ptr().offset((rhs_idx - rhs_del) as isize);
                std::ptr::copy(rhs_src, rhs_dst, rhs_old_len - rhs_idx);
            }
            // Set the size of vectors to the correct size since we can now safely panic.
            self.data.set_len(lhs_old_len - lhs_del);
            other.data.set_len(rhs_old_len - rhs_del);
        }
        (self, other)
    }

    /// Returns a set containing the elements present in both `self` and
    /// `other`.
    pub fn intersection<'a>(&'a self, other: &'a VecSet<T>) -> Intersection<T> {
        Intersection {
            lhs: self,
            lhs_cursor: 0,
            rhs: other,
            rhs_cursor: 0,
        }
    }

    /// In-place intersection with another `VecSet`.
    pub fn intersect(&mut self, other: &VecSet<T>) {
        let mut other_cursor = 0;
        self.data.retain(|item| loop {
            if other_cursor >= other.data.len() {
                return false;
            }
            if *item < other.data[other_cursor] {
                return false;
            }
            if *item == other.data[other_cursor] {
                return true;
            }
            other_cursor += 1;
        });
    }

    /// Returns a set containing the elements present in either self` or
    /// `other`.
    pub fn union(&self, other: &VecSet<T>) -> VecSet<T>
    where T: Clone {
        let mut data = Vec::new();
        let mut self_cursor = 0;
        let mut other_cursor = 0;
        while self_cursor < self.data.len() && other_cursor < other.data.len() {
            match self.data[self_cursor].cmp(&other.data[other_cursor]) {
                Ordering::Equal => {
                    data.push(self.data[self_cursor].clone());
                    self_cursor += 1;
                    other_cursor += 1;
                }
                Ordering::Less => {
                    data.push(self.data[self_cursor].clone());
                    self_cursor += 1
                }
                Ordering::Greater => {
                    data.push(other.data[other_cursor].clone());
                    other_cursor += 1
                }
            }
        }
        data.extend(self.data[self_cursor..].iter().cloned());
        data.extend(other.data[other_cursor..].iter().cloned());
        VecSet { data }
    }

    /// Returns a new `VecSet` with only the elements for which the predicate returned
    /// `true`.
    pub fn filter<P>(&self, mut predicate: P) -> Self
    where
        T: Clone,
        P: FnMut(&T) -> bool,
    {
        VecSet {
            data: self.data
                .iter()
                .filter(|&x| predicate(x))
                .cloned()
                .collect(),
        }
    }

    /// Filters out elements for wich the predicate returns false.
    pub fn retain<P>(&mut self, predicate: P)
    where P: FnMut(&T) -> bool {
        self.data.retain(predicate);
    }

    /// Inserts an element in the `VecSet`. This operation has a complexity in
    /// O(n). Returns `false` if the item was already present.
    pub fn insert(&mut self, item: T) -> bool {
        match self.data.binary_search(&item) {
            Ok(_) => false,
            Err(pos) => {
                self.data.insert(pos, item);
                true
            },
        }
    }
}

impl<T> Default for VecSet<T> {
    fn default() -> Self { VecSet { data: Vec::new() } }
}

impl<T> std::fmt::Debug for VecSet<T>
where T: std::fmt::Debug
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { self.data.fmt(f) }
}

impl<T> PartialOrd for VecSet<T>
where T: Ord
{
    fn partial_cmp(&self, other: &VecSet<T>) -> Option<Ordering> {
        let mut self_cursor = 0;
        let mut other_cursor = 0;
        let mut prev_ord = Ordering::Equal;
        // Iterate simultaneously on both vectors.
        loop {
            // If self if finished.
            if self_cursor >= self.data.len() {
                if other.data.len() == other_cursor {
                    return Some(prev_ord);
                } else if prev_ord == Ordering::Greater {
                    return None;
                } else {
                    return Some(Ordering::Less);
                }
            // If other is finished but not self
            } else if other_cursor >= other.data.len() {
                if prev_ord == Ordering::Less {
                    return None;
                } else {
                    return Some(Ordering::Greater);
                }
            }
            // If both elements are present.
            let ord = self.data[self_cursor].cmp(&other.data[other_cursor]);
            match (ord, prev_ord) {
                (Ordering::Less, Ordering::Less)
                | (Ordering::Greater, Ordering::Greater) => return None,
                (Ordering::Equal, _) => {
                    self_cursor += 1;
                    other_cursor += 1;
                }
                (Ordering::Less, _) => {
                    prev_ord = Ordering::Greater;
                    self_cursor += 1;
                }
                (Ordering::Greater, _) => {
                    prev_ord = Ordering::Less;
                    other_cursor += 1;
                }
            };
        }
    }
}

impl<T> std::iter::IntoIterator for VecSet<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter { self.data.into_iter() }
}

impl<'a, T> std::iter::IntoIterator for &'a VecSet<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter { self.data.iter() }
}

impl<T> std::iter::FromIterator<T> for VecSet<T>
where T: Ord
{
    fn from_iter<IT>(iter: IT) -> Self
    where IT: IntoIterator<Item = T> {
        VecSet::new(Vec::from_iter(iter))
    }
}

impl<T> std::ops::Deref for VecSet<T> {
    type Target = [T];

    fn deref(&self) -> &[T] { &self.data }
}

pub struct Difference<'a, T: 'a> {
    lhs: &'a VecSet<T>,
    lhs_cursor: usize,
    rhs: &'a VecSet<T>,
    rhs_cursor: usize,
}

impl<'a, T> Iterator for Difference<'a, T>
where T: Ord + 'a
{
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        'lhs: loop {
            if self.lhs_cursor == self.lhs.data.len() {
                return None;
            }
            let item = &self.lhs.data[self.lhs_cursor];
            self.lhs_cursor += 1;
            while self.rhs_cursor < self.rhs.data.len() {
                match self.rhs.data[self.rhs_cursor].cmp(item) {
                    Ordering::Less => self.rhs_cursor += 1,
                    Ordering::Greater => break,
                    Ordering::Equal => continue 'lhs,
                }
            }
            return Some(item);
        }
    }
}

pub struct Intersection<'a, T: 'a> {
    lhs: &'a VecSet<T>,
    lhs_cursor: usize,
    rhs: &'a VecSet<T>,
    rhs_cursor: usize,
}

impl<'a, T> Iterator for Intersection<'a, T>
where T: Ord + 'a
{
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        loop {
            if self.lhs_cursor >= self.lhs.data.len()
                || self.rhs_cursor >= self.rhs.data.len()
            {
                return None;
            }
            let lhs_item = &self.lhs.data[self.lhs_cursor];
            match lhs_item.cmp(&self.rhs.data[self.rhs_cursor]) {
                Ordering::Less => self.lhs_cursor += 1,
                Ordering::Equal => {
                    self.lhs_cursor += 1;
                    self.rhs_cursor += 1;
                    return Some(lhs_item);
                }
                Ordering::Greater => self.rhs_cursor += 1,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::Ordering;

    /// Test the comparison operator.
    #[test]
    fn test_cmp() {
        let v0 = VecSet::new(vec![0]);
        let v1 = VecSet::new(vec![1]);
        let v01 = VecSet::new(vec![0, 1]);
        assert_eq!(v0.partial_cmp(&v1), None);
        assert_eq!(v1.partial_cmp(&v0), None);
        assert_eq!(v0.partial_cmp(&v01), Some(Ordering::Less));
        assert_eq!(v1.partial_cmp(&v01), Some(Ordering::Less));
        assert_eq!(v01.partial_cmp(&v0), Some(Ordering::Greater));
        assert_eq!(v01.partial_cmp(&v1), Some(Ordering::Greater));
    }

    #[test]
    fn symmetric_difference() {
        let v0 = VecSet::new(vec![0, 2, 3, 5]);
        let v1 = VecSet::new(vec![0, 1, 2, 4, 6]);
        let (d0, d1) = v0.symmetric_difference(v1);
        assert_eq!(d0, VecSet::new(vec![3, 5]));
        assert_eq!(d1, VecSet::new(vec![1, 4, 6]))
    }
}
