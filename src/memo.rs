use std::borrow::Borrow;
use std::cell::Cell;
use std::cmp;
use std::collections::hash_map::RandomState;
use std::convert::AsRef;
use std::fmt;
use std::hash::{BuildHasher, BuildHasherDefault, Hash, Hasher};
use std::ops::Deref;

pub trait ToMut<T: ?Sized> {
    fn to_mut(this: &mut Self) -> &mut T;
}

impl<T: ?Sized> ToMut<T> for T {
    fn to_mut(this: &mut Self) -> &mut T {
        this
    }
}

#[derive(Debug, Default)]
pub struct Memoized<T: ?Sized, M> {
    memo: M,
    value: T,
}

impl<T, M> Memoized<T, M>
where
    M: Default,
{
    pub fn new(value: T) -> Self {
        Memoized {
            value,
            memo: M::default(),
        }
    }
}

impl<T, M> From<T> for Memoized<T, M>
where
    M: Default,
{
    fn from(value: T) -> Self {
        Self::new(value)
    }
}

impl<T: ?Sized, M> fmt::Display for Memoized<T, M>
where
    T: fmt::Display,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.value, fmt)
    }
}

impl<T, M> Clone for Memoized<T, M>
where
    T: Clone,
    M: Default,
{
    fn clone(&self) -> Self {
        Memoized {
            memo: M::default(),
            value: self.value.clone(),
        }
    }
}

impl<T, M> Hash for Memoized<T, M>
where
    T: Hash + ?Sized,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}

impl<T, M> PartialEq for Memoized<T, M>
where
    T: PartialEq + ?Sized,
{
    fn eq(&self, other: &Self) -> bool {
        self.value.eq(&other.value)
    }
}

impl<T, M> Eq for Memoized<T, M> where T: Eq + ?Sized {}

impl<T, M> PartialOrd for Memoized<T, M>
where
    T: PartialOrd + ?Sized,
{
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl<T, M> Ord for Memoized<T, M>
where
    T: Ord + ?Sized,
{
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.value.cmp(&other.value)
    }
}

impl<T, M> Deref for Memoized<T, M>
where
    T: ?Sized,
{
    type Target = T;

    fn deref(&self) -> &T {
        &self.value
    }
}

impl<T, M> AsRef<T> for Memoized<T, M>
where
    T: ?Sized,
{
    fn as_ref(&self) -> &T {
        &*self
    }
}

impl<T, M> Borrow<T> for Memoized<T, M>
where
    T: ?Sized,
{
    fn borrow(&self) -> &T {
        &*self
    }
}

impl<T, M> ToMut<T> for Memoized<T, M>
where
    T: ?Sized,
    M: Default,
{
    fn to_mut(this: &mut Self) -> &mut T {
        this.memo = M::default();
        &mut this.value
    }
}

#[derive(Debug, Clone, Default)]
struct MemoizedHash<T: ?Sized, S = BuildHasherDefault<RandomState>> {
    hash_builder: S,
    hash: Cell<Option<u64>>,
    value: T,
}

impl<T, S> From<T> for MemoizedHash<T, BuildHasherDefault<S>>
where
    T: Hash,
    S: Hasher + Default,
{
    fn from(value: T) -> Self {
        MemoizedHash {
            hash_builder: BuildHasherDefault::default(),
            hash: Cell::new(None),
            value,
        }
    }
}

impl<T, S> fmt::Display for MemoizedHash<T, S>
where
    T: fmt::Display + ?Sized,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.value, fmt)
    }
}

impl<T, S> PartialEq for MemoizedHash<T, S>
where
    T: PartialEq + Hash + ?Sized,
    S: BuildHasher,
{
    fn eq(&self, other: &Self) -> bool {
        if MemoizedHash::get_hash(self) != MemoizedHash::get_hash(other) {
            false
        } else {
            self.value.eq(&other.value)
        }
    }
}

impl<T, S> Eq for MemoizedHash<T, S>
where
    T: Eq + Hash + ?Sized,
    S: BuildHasher,
{
}

impl<T, S> PartialOrd for MemoizedHash<T, S>
where
    T: PartialOrd + Hash + ?Sized,
    S: BuildHasher,
{
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        match MemoizedHash::get_hash(self).cmp(&MemoizedHash::get_hash(other)) {
            cmp::Ordering::Less => Some(cmp::Ordering::Less),
            cmp::Ordering::Equal => self.value.partial_cmp(&other.value),
            cmp::Ordering::Greater => Some(cmp::Ordering::Greater),
        }
    }
}

impl<T, S> Ord for MemoizedHash<T, S>
where
    T: Ord + Hash + ?Sized,
    S: BuildHasher,
{
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        MemoizedHash::get_hash(self)
            .cmp(&MemoizedHash::get_hash(other))
            .then_with(|| self.value.cmp(&other.value))
    }
}

impl<T, S> MemoizedHash<T, S>
where
    T: Hash,
    S: BuildHasher,
{
    pub fn with_hasher(value: T, hash_builder: S) -> Self {
        MemoizedHash {
            hash_builder,
            hash: Cell::new(None),
            value,
        }
    }
}

impl<T, S> MemoizedHash<T, S>
where
    T: Hash + ?Sized,
    S: BuildHasher,
{
    fn get_hash(&self) -> u64 {
        match self.hash.get() {
            None => {
                let mut hasher = self.hash_builder.build_hasher();
                self.value.hash(&mut hasher);
                let hash = hasher.finish();
                self.hash.set(Some(hash));
                hash
            }
            Some(hash) => hash,
        }
    }
}

impl<T, S> Hash for MemoizedHash<T, S>
where
    T: Hash + ?Sized,
    S: BuildHasher,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        MemoizedHash::get_hash(self).hash(state);
    }
}

impl<T, S> Borrow<T> for MemoizedHash<T, S>
where
    T: ?Sized,
{
    fn borrow(&self) -> &T {
        &*self
    }
}

impl<T, S> AsRef<T> for MemoizedHash<T, S>
where
    T: ?Sized,
{
    fn as_ref(&self) -> &T {
        &*self
    }
}

impl<T, S> Deref for MemoizedHash<T, S>
where
    T: ?Sized,
{
    type Target = T;

    fn deref(&self) -> &T {
        &self.value
    }
}

impl<T, S> ToMut<T> for MemoizedHash<T, S>
where
    T: ?Sized,
{
    fn to_mut(this: &mut Self) -> &mut T {
        this.hash.set(None);
        &mut this.value
    }
}
