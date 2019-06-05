use std::cell::Cell;
use std::cmp::Ordering;
use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ops;

#[derive(Debug, Clone)]
struct HashMemo<T, B = DefaultHasher> {
    _hasher: PhantomData<fn() -> B>,
    hash: Cell<Option<u64>>,
    value: T,
}

impl<T, B> Hash for HashMemo<T, B>
where
    T: Hash,
    B: Hasher + Default,
{
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        state.write_u64(self.hash())
    }
}

impl<T, B> HashMemo<T, B> {
    fn hash(&self) -> u64
    where
        T: Hash,
        B: Hasher + Default,
    {
        if let Some(hash) = self.hash.get() {
            hash
        } else {
            let mut hasher = B::default();
            self.value.hash(&mut hasher);
            let hash = hasher.finish();
            self.hash.set(Some(hash));
            hash
        }
    }
}

impl<T, B> fmt::Display for HashMemo<T, B>
where
    T: fmt::Display,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.value, fmt)
    }
}

impl<T, B> PartialEq for HashMemo<T, B>
where
    T: PartialEq + Hash,
    B: Hasher + Default,
{
    fn eq(&self, other: &HashMemo<T, B>) -> bool {
        if self.hash() == other.hash() {
            self.value == other.value
        } else {
            false
        }
    }
}

impl<T, B> Eq for HashMemo<T, B>
where
    T: Eq + Hash,
    B: Hasher + Default,
{
}

impl<T, B> PartialOrd for HashMemo<T, B>
where
    T: PartialOrd + Hash,
    B: Hasher + Default,
{
    fn partial_cmp(&self, other: &HashMemo<T, B>) -> Option<Ordering> {
        match self.hash().cmp(&other.hash()) {
            Ordering::Greater => Some(Ordering::Greater),
            Ordering::Less => Some(Ordering::Less),
            Ordering::Equal => self.value.partial_cmp(&other.value),
        }
    }
}

impl<T, B> Ord for HashMemo<T, B>
where
    T: Ord + Hash,
    B: Hasher + Default,
{
    fn cmp(&self, other: &HashMemo<T, B>) -> Ordering {
        self.hash()
            .cmp(&other.hash())
            .then_with(|| self.value.cmp(&other.value))
    }
}

impl<T, B> ops::Deref for HashMemo<T, B> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.value
    }
}
