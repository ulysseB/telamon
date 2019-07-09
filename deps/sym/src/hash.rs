use fxhash::FxHasher;
use std::cell::{Cell, RefCell};
use std::cmp::Ordering;
use std::fmt;
use std::hash::{BuildHasher, BuildHasherDefault, Hash, Hasher};
use std::ops;

#[derive(Clone, Default)]
pub struct MemoizedHash<T: ?Sized, S = BuildHasherDefault<FxHasher>> {
    hash_builder: S,
    hash: Cell<Option<u64>>,
    value: T,
}

impl<T, S> From<T> for MemoizedHash<T, S>
where
    S: BuildHasher + Default,
{
    fn from(value: T) -> Self {
        MemoizedHash {
            hash_builder: S::default(),
            hash: Cell::new(None),
            value,
        }
    }
}

impl<T, S> fmt::Debug for MemoizedHash<T, S>
where
    T: fmt::Debug + ?Sized,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&self.value, fmt)
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
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match MemoizedHash::get_hash(self).cmp(&MemoizedHash::get_hash(other)) {
            Ordering::Less => Some(Ordering::Less),
            Ordering::Equal => {
                if cfg!(feature = "ignore_collisions") {
                    Some(Ordering::Equal)
                } else {
                    self.value.partial_cmp(&other.value)
                }
            }
            Ordering::Greater => Some(Ordering::Greater),
        }
    }
}

impl<T, S> Ord for MemoizedHash<T, S>
where
    T: Ord + Hash + ?Sized,
    S: BuildHasher,
{
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        MemoizedHash::get_hash(self)
            .cmp(&MemoizedHash::get_hash(other))
            .then_with(|| {
                if cfg!(feature = "ignore_collisions") {
                    Ordering::Equal
                } else {
                    self.value.cmp(&other.value)
                }
            })
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

    pub fn new(value: T) -> Self
    where
        S: Default,
    {
        MemoizedHash::with_hasher(value, S::default())
    }

    pub fn fast_ne(this: &Self, other: &Self) -> bool {
        MemoizedHash::get_hash(this) != MemoizedHash::get_hash(other)
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

impl<T, S> AsRef<T> for MemoizedHash<T, S>
where
    T: ?Sized,
{
    fn as_ref(&self) -> &T {
        &*self
    }
}

impl<T, S> ops::Deref for MemoizedHash<T, S>
where
    T: ?Sized,
{
    type Target = T;

    fn deref(&self) -> &T {
        &self.value
    }
}

impl<T, S> MemoizedHash<T, S>
where
    T: ?Sized,
{
    pub fn make_mut(this: &mut Self) -> &mut T {
        this.hash.set(None);
        &mut this.value
    }
}
