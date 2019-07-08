use std::cell::Cell;
use std::cmp;
use std::collections::hash_map::DefaultHasher;
use std::convert::{AsMut, AsRef};
use std::fmt;
use std::hash::{BuildHasher, BuildHasherDefault, Hash, Hasher};
use std::ops::Deref;

#[derive(Default)]
pub struct Memoized<T: ?Sized, M> {
    memo: M,
    value: T,
}

impl<T, M> Memoized<T, M> {
    pub fn new(value: T) -> Self
    where
        M: Default,
    {
        Memoized {
            value,
            memo: M::default(),
        }
    }

    pub fn memo(this: &Self) -> &M {
        &this.memo
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

impl<T: ?Sized, M> fmt::Debug for Memoized<T, M>
where
    T: fmt::Debug,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&self.value, fmt)
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

impl<T, M> AsMut<T> for Memoized<T, M>
where
    T: ?Sized,
    M: Default,
{
    fn as_mut(&mut self) -> &mut T {
        self.memo = M::default();
        &mut self.value
    }
}
