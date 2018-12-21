//! Extension of the iterator library.
use itertools::Itertools;
use linked_list;
use std;
use crate::HashMap;

/// Iterates over a linked list while removing some items.
pub fn filter_list<'a, T, F>(
    list: &'a mut linked_list::LinkedList<T>,
    filter: F,
) -> FilterList<'a, T, F>
where
    T: 'a,
    F: FnMut(&mut T) -> bool,
{
    FilterList {
        cursor: list.cursor(),
        filter,
    }
}

/// Iterates over a linked list while removing some items.
pub struct FilterList<'a, T, F>
where
    T: 'a,
    F: FnMut(&mut T) -> bool,
{
    cursor: linked_list::Cursor<'a, T>,
    filter: F,
}

impl<'a, T, F> Iterator for FilterList<'a, T, F>
where
    F: FnMut(&mut T) -> bool,
{
    type Item = T;

    fn next(&mut self) -> Option<T> {
        while !(self.filter)(self.cursor.peek_next()?) {
            self.cursor.next();
        }
        self.cursor.remove()
    }
}

/// Zip copies of an object with an iterator.
pub struct ZipCopy<I: Iterator, T: Clone> {
    it: std::iter::Peekable<I>,
    object: Option<T>,
}

impl<I: Iterator, T: Clone> Iterator for ZipCopy<I, T> {
    type Item = (I::Item, T);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(lhs) = self.it.next() {
            let rhs = if self.it.peek().is_none() {
                let mut out = None;
                std::mem::swap(&mut out, &mut self.object);
                out.unwrap()
            } else {
                self.object.as_ref().unwrap().clone()
            };
            Some((lhs, rhs))
        } else {
            None
        }
    }
}

/// Zip copies of an object with an iterator.
pub fn zip_copy<I: IntoIterator, T: Clone>(it: I, object: T) -> ZipCopy<I::IntoIter, T> {
    let mut peek_it = it.into_iter().peekable();
    let object_option = peek_it.peek().map(|_| object);
    ZipCopy {
        it: peek_it,
        object: object_option,
    }
}

/// Ensures an iterator has at most one element.
pub fn at_most_one<IT: Iterator>(mut it: IT) -> Option<IT::Item> {
    let out = it.next();
    assert!(it.next().is_none());
    out
}

/// Transforms an iterator into an `HashMap`. Redundant nodes are merged using
/// `merge`.
pub fn to_map<K: Eq + std::hash::Hash, V, IT, M>(it: IT, merge: M) -> HashMap<K, V>
where
    IT: Iterator<Item = (K, V)>,
    M: Fn(V, V) -> V,
{
    let mut map = HashMap::default();
    for (k, v) in it {
        let v = if let Some(old_v) = map.remove(&k) {
            merge(old_v, v)
        } else {
            v
        };
        map.insert(k, v);
    }
    map
}

pub struct PartialPermutations<T: Clone + Ord> {
    permutation: Vec<T>,
    finished: bool,
    k: usize,
}

impl<T: Clone + Ord> PartialPermutations<T> {
    /// Returns an iterator of the partial permuations of size `k` of values. Permutations
    /// are returned in lexicographical order.
    pub fn new<IT>(values: IT, k: usize) -> Self
    where
        IT: IntoIterator<Item = T>,
    {
        let mut values = values.into_iter().sorted();
        if k < values.len() {
            values[k..].reverse();
        }
        PartialPermutations {
            finished: k > values.len(),
            permutation: values,
            k,
        }
    }
}

impl<T: Clone + Ord> Iterator for PartialPermutations<T> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Vec<T>> {
        if self.finished {
            return None;
        }
        let combination = self.permutation.iter().take(self.k).cloned().collect();
        let mut i = std::cmp::min(self.k, self.permutation.len().saturating_sub(1));
        while i > 0 && self.permutation[i - 1] > self.permutation[i] {
            i -= 1;
        }
        if i == 0 {
            self.finished = true;
        } else {
            let swap_id = self
                .permutation
                .iter()
                .rposition(|x| *x > self.permutation[i - 1])
                .unwrap();
            self.permutation.swap(i - 1, swap_id);
            unsafe {
                let k = self.k;
                let n = self.permutation.len();
                let perm = self.permutation.as_mut_ptr();
                // Move the end of the new permutation into a temporary buffer.
                let mut tmp_vec = Vec::with_capacity(k - i);
                let buffer = tmp_vec.as_mut_ptr();
                std::ptr::copy_nonoverlapping(
                    perm.offset((n - k + i) as isize),
                    buffer,
                    k - i,
                );
                // Copy the new unused values.
                std::ptr::copy(perm.offset(i as isize), perm.offset(k as isize), n - k);
                // Copy andreverse the new end of the permutation
                for j in 0..(k - i) {
                    let src = buffer.offset(j as isize);
                    let dst = perm.offset((k - j - 1) as isize);
                    std::ptr::copy_nonoverlapping(src, dst, 1);
                }
            }
        }
        Some(combination)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use itertools::Itertools;

    /// Ensures `PartialPermuations` works when the number of values is 0.
    #[test]
    fn partial_permutations_n0() {
        let res: Vec<Vec<()>> = PartialPermutations::new(vec![], 0).collect_vec();
        assert_eq!(&res, &[Vec::<()>::new()]);
    }

    /// Ensures `PartialPermutations` works when  k==0.
    #[test]
    fn partial_permutations_k0() {
        let res = PartialPermutations::new(0..3, 0).collect_vec();
        assert_eq!(&res, &[Vec::<usize>::new()]);
    }

    /// Ensures `PartialPermutations` works when the number of values equals k.
    #[test]
    fn partial_permutations_kn() {
        let res = PartialPermutations::new(0..3, 3).collect_vec();
        let expected = vec![
            vec![0, 1, 2],
            vec![0, 2, 1],
            vec![1, 0, 2],
            vec![1, 2, 0],
            vec![2, 0, 1],
            vec![2, 1, 0],
        ];
        assert_eq!(res, expected);
    }

    /// Ensures `PartialPermutations` works when the number of values is
    /// smaller than k.
    #[test]
    fn partial_permutations_k_gt_n() {
        let res = PartialPermutations::new(0..3, 4).collect_vec();
        assert_eq!(res, Vec::<Vec<usize>>::new());
    }

    /// Ensures `PartialPermuations` works in the general case.
    #[test]
    fn partial_permutations() {
        let res = PartialPermutations::new(0..5, 3).collect_vec();
        let expected = vec![
            vec![0, 1, 2],
            vec![0, 1, 3],
            vec![0, 1, 4],
            vec![0, 2, 1],
            vec![0, 2, 3],
            vec![0, 2, 4],
            vec![0, 3, 1],
            vec![0, 3, 2],
            vec![0, 3, 4],
            vec![0, 4, 1],
            vec![0, 4, 2],
            vec![0, 4, 3],
            vec![1, 0, 2],
            vec![1, 0, 3],
            vec![1, 0, 4],
            vec![1, 2, 0],
            vec![1, 2, 3],
            vec![1, 2, 4],
            vec![1, 3, 0],
            vec![1, 3, 2],
            vec![1, 3, 4],
            vec![1, 4, 0],
            vec![1, 4, 2],
            vec![1, 4, 3],
            vec![2, 0, 1],
            vec![2, 0, 3],
            vec![2, 0, 4],
            vec![2, 1, 0],
            vec![2, 1, 3],
            vec![2, 1, 4],
            vec![2, 3, 0],
            vec![2, 3, 1],
            vec![2, 3, 4],
            vec![2, 4, 0],
            vec![2, 4, 1],
            vec![2, 4, 3],
            vec![3, 0, 1],
            vec![3, 0, 2],
            vec![3, 0, 4],
            vec![3, 1, 0],
            vec![3, 1, 2],
            vec![3, 1, 4],
            vec![3, 2, 0],
            vec![3, 2, 1],
            vec![3, 2, 4],
            vec![3, 4, 0],
            vec![3, 4, 1],
            vec![3, 4, 2],
            vec![4, 0, 1],
            vec![4, 0, 2],
            vec![4, 0, 3],
            vec![4, 1, 0],
            vec![4, 1, 2],
            vec![4, 1, 3],
            vec![4, 2, 0],
            vec![4, 2, 1],
            vec![4, 2, 3],
            vec![4, 3, 0],
            vec![4, 3, 1],
            vec![4, 3, 2],
        ];
        assert_eq!(res, expected);
    }
}
