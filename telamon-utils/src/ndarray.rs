//! An array with a variable number of dimensions.
use itertools::Itertools;
use num::Integer;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

/// An array with a variable number of dimensions.
#[derive(Debug)]
pub struct NDArray<T> {
    pub dims: Vec<usize>,
    data: Vec<T>,
}

impl<T> NDArray<T> {
    /// Create a NDArray and initializes it with the generator.
    pub fn new(dims: Vec<usize>, data: Vec<T>) -> NDArray<T> {
        assert_eq!(data.len(), dims.iter().product());
        NDArray { dims, data }
    }

    /// Initializes an NDArray with default values.
    pub fn init_default(dims: Vec<usize>) -> Self
    where
        T: Default,
    {
        let len = dims.iter().product();
        NDArray {
            dims,
            data: (0..len).map(|_| T::default()).collect(),
        }
    }

    /// Returns the number of dimensions in the array.
    pub fn num_dims(&self) -> usize {
        self.dims.len()
    }

    /// Converts a ND index into a 1D index.
    fn nd_to_1d(&self, indexes: &[usize]) -> usize {
        assert_eq!(self.dims.len(), indexes.len());
        self.dims
            .iter()
            .zip(indexes.iter())
            .fold(0, |x, (y, z)| x * y + z)
    }

    /// Returns a mutable view on the NDArray.
    pub fn view_mut(&mut self) -> ViewMut<T> {
        ViewMut {
            array: self,
            bounds: self.dims.clone(),
            fixed_indexes: Vec::new(),
            marker: PhantomData,
        }
    }
}

impl<'a, T> Index<&'a [usize]> for NDArray<T> {
    type Output = T;

    fn index(&self, indexes: &'a [usize]) -> &T {
        &self.data[self.nd_to_1d(indexes)]
    }
}

impl<'a, T> IndexMut<&'a [usize]> for NDArray<T> {
    fn index_mut(&mut self, indexes: &'a [usize]) -> &mut T {
        let idx = self.nd_to_1d(indexes);
        &mut self.data[idx]
    }
}

/// A N-dimentional range.
pub struct NDRange<'a, T>
where
    T: 'a + Integer + Clone,
{
    max: &'a [T],
    current: Option<Vec<T>>,
}

impl<'a, T> NDRange<'a, T>
where
    T: 'a + Integer + Clone,
{
    /// Creates a ND range with the given bounds.
    pub fn new(bounds: &'a [T]) -> NDRange<'a, T> {
        let min = if bounds.iter().any(|x| *x == T::zero()) {
            None
        } else {
            Some((0..bounds.len()).map(|_| T::zero()).collect_vec())
        };
        NDRange {
            max: bounds,
            current: min,
        }
    }
}

impl<'a, T> Iterator for NDRange<'a, T>
where
    T: 'a + Integer + Clone,
{
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        let out = self.current.clone();
        if let Some(ref mut current) = self.current {
            if !current.is_empty() {
                let mut i = current.len() - 1;
                loop {
                    current[i] = current[i].clone() + T::one();
                    if current[i] != self.max[i] || i == 0 {
                        break;
                    }
                    current[i] = T::zero();
                    i -= 1;
                }
            }
        }
        if self
            .current
            .as_ref()
            .map(|x| x.is_empty() || x[0] == self.max[0])
            .unwrap_or(false)
        {
            self.current = None;
        }
        out
    }
}

/// A mutable view on a `NDArray`, with some dimensions fixed.
pub struct ViewMut<'a, T> {
    /// Points to the underlying array.
    array: *mut NDArray<T>,
    /// The size of each dimension.
    bounds: Vec<usize>,
    /// Lists the dimensions that have a fixed value, in increasing order. The second
    /// element of the tuple is the value of the fixed index on this dimension.
    fixed_indexes: Vec<(usize, usize)>,
    /// Ensures the pointer is valid.
    marker: PhantomData<&'a ()>,
}

impl<'a, T> ViewMut<'a, T> {
    /// Returns the number of non-fixed dimensions in the view.
    pub fn num_dims(&self) -> usize {
        self.bounds.len()
    }

    /// Splits the view on the given dimension.
    pub fn split(&mut self, logical_dim: usize) -> Vec<ViewMut<T>> {
        // Build the fixed index vector.
        let mut bounds = self.bounds.clone();
        let dim_len = bounds.remove(logical_dim);
        // The fixed dimensions `j` at `fixed_indexes[i]` has `j-i` free dimensions
        // with a lower id. The new fixed dimension must thus be placed at the
        // first place where `logical_dim < j-i`.
        let pos_in_fixed_indexes = self
            .fixed_indexes
            .iter()
            .enumerate()
            .position(|(fixed_pos, &(raw_pos, _))| fixed_pos + logical_dim < raw_pos)
            .unwrap_or_else(|| self.fixed_indexes.len());
        let raw_dim = pos_in_fixed_indexes + logical_dim;
        // at cell i: (dim, _) in fixed_indexes, dim - i = number of outer mouving dims
        // Spawn the views
        (0..dim_len)
            .map(|idx| {
                let mut fixed_indexes = self.fixed_indexes.clone();
                fixed_indexes.insert(pos_in_fixed_indexes, (raw_dim, idx));
                ViewMut {
                    fixed_indexes,
                    bounds: bounds.clone(),
                    ..*self
                }
            })
            .collect()
    }

    /// Computes the index of an element in the underlying vector.
    fn flat_index(&self, indexes: &[usize]) -> usize {
        let mut next_fixed = 0;
        let mut next_index = 0;
        let mut flat_index = 0;
        for (dim_id, dim_size) in unsafe { (*self.array).dims.iter().enumerate() } {
            let idx;
            let num_fixed = self.fixed_indexes.len();
            if next_fixed < num_fixed && self.fixed_indexes[next_fixed].0 == dim_id {
                idx = self.fixed_indexes[next_fixed].1;
                next_fixed += 1;
            } else {
                idx = indexes[next_index];
                next_index += 1;
            }
            flat_index = flat_index * dim_size + idx;
        }
        flat_index
    }

    /// Enumerates the elements with their indexes.
    pub fn enumerate<'b>(&'b self) -> impl Iterator<Item = (Vec<usize>, &'b T)> + 'b {
        NDRange::new(&self.bounds).map(move |idx| {
            let item = &self[&idx[..]];
            (idx, item)
        })
    }

    /// Produces mutable references to the elements, with their indexes.
    pub fn enumerate_mut(
        &'a mut self,
    ) -> impl Iterator<Item = (Vec<usize>, &'a mut T)> + 'a {
        let self_ptr: *mut ViewMut<'a, _> = self;
        NDRange::new(&self.bounds).map(move |idx| {
            let item = unsafe { (*self_ptr).index_mut(&idx[..]) };
            (idx, item)
        })
    }
}

impl<'a, 'b, T> Index<&'b [usize]> for ViewMut<'a, T> {
    type Output = T;

    fn index(&self, indexes: &'b [usize]) -> &T {
        let idx = self.flat_index(indexes);
        unsafe { &(*self.array).data[idx] }
    }
}

impl<'a, 'b, T> IndexMut<&'b [usize]> for ViewMut<'a, T> {
    fn index_mut(&mut self, indexes: &'b [usize]) -> &mut T {
        let idx = self.flat_index(indexes);
        unsafe { &mut (*self.array).data[idx] }
    }
}

impl<'a, 'b, T> IntoIterator for &'b ViewMut<'a, T>
where
    'a: 'b,
{
    type Item = &'b T;
    type IntoIter = ViewMutIter<'b, T>;

    fn into_iter(self) -> Self::IntoIter {
        ViewMutIter {
            view: self,
            index: NDRange::new(&self.bounds),
        }
    }
}

/// Iterator over the elements in a mutable view.
pub struct ViewMutIter<'a, T: 'a> {
    view: &'a ViewMut<'a, T>,
    index: NDRange<'a, usize>,
}

impl<'a, T> Iterator for ViewMutIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.index.next().map(|idx| &self.view[&idx[..]])
    }
}

impl<'a, 'b, T> IntoIterator for &'b mut ViewMut<'a, T>
where
    T: 'a,
    'a: 'b,
{
    type Item = &'b mut T;
    type IntoIter = ViewIterMut<'a, 'b, T>;

    fn into_iter(self) -> Self::IntoIter {
        ViewIterMut {
            view: self,
            index: NDRange::new(&self.bounds),
        }
    }
}

/// Mutable iterator over the elements in a mutable view.
pub struct ViewIterMut<'a, 'b, T>
where
    T: 'a,
    'a: 'b,
{
    view: *mut ViewMut<'a, T>,
    index: NDRange<'b, usize>,
}

impl<'a, 'b, T> Iterator for ViewIterMut<'a, 'b, T> {
    type Item = &'b mut T;

    fn next(&mut self) -> Option<Self::Item> {
        self.index
            .next()
            .map(|idx| unsafe { &mut (*self.view)[&idx[..]] })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use itertools::Itertools;
    use std::fmt::Debug;

    /// Ensures `NDRange` works correctly in the common case.
    #[test]
    fn ndrange() {
        let bound = vec![2, 3];
        let mut it = NDRange::new(&bound);
        assert_eq!(it.next(), Some(vec![0, 0]));
        assert_eq!(it.next(), Some(vec![0, 1]));
        assert_eq!(it.next(), Some(vec![0, 2]));
        assert_eq!(it.next(), Some(vec![1, 0]));
        assert_eq!(it.next(), Some(vec![1, 1]));
        assert_eq!(it.next(), Some(vec![1, 2]));
        assert_eq!(it.next(), None);
        assert_eq!(it.next(), None);
    }

    /// Ensures a `NDRange` with zero dimension has exactly one element.
    #[test]
    fn ndrange_zero_dim() {
        let bound: Vec<u32> = vec![];
        let mut it = NDRange::new(&bound);
        assert_eq!(it.next(), Some(vec![]));
        assert_eq!(it.next(), None);
        assert_eq!(it.next(), None);
    }

    /// Ensure an empty `NDRange` has zero element.
    #[test]
    fn ndrange_empty() {
        let bound = vec![3, 0, 3];
        let mut it = NDRange::new(&bound);
        assert_eq!(it.next(), None);
        assert_eq!(it.next(), None);
    }

    /// Ensures `NDArray` works in the common case.
    #[test]
    fn ndarray() {
        let bound = vec![2, 3];
        let array = NDArray::new(bound.clone(), (0..6).collect());
        let values = NDRange::new(&bound)
            .map(|idx| array[&idx[..]])
            .collect_vec();
        assert_eq!(values, (0..6).collect_vec());
    }

    /// Ensures `NDArray` works with zero dimension.
    #[test]
    fn ndarray_zero_dim() {
        let array = NDArray::new(vec![], vec![42]);
        assert_eq!(array[&[][..]], 42);
    }

    /// Ensures `NDArray` works with an empty dimension.
    #[test]
    fn ndarry_empty() {
        NDArray::<u32>::new(vec![2, 0, 3], vec![]);
    }

    /// Ensures we can correctly iterate on a view.
    #[test]
    fn view_iter() {
        let mut array = NDArray::new(vec![2, 3], (0..6).collect());
        let mut view = array.view_mut();
        let split0 = view.split(0);
        test_iter((&split0[0]).into_iter().cloned(), &[0, 1, 2]);
    }

    /// Ensures an iterator has the expected values.
    fn test_iter<T, IT>(iter: IT, expected: &[T])
    where
        IT: IntoIterator<Item = T>,
        T: PartialEq + Debug,
    {
        assert_eq!(&iter.into_iter().collect_vec()[..], expected);
    }
}
