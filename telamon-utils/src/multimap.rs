//! A `HashMap` with mutiple values for each key.
use serde::ser::{Serialize, Serializer};
use std;
use std::borrow::Borrow;
use std::collections::hash_map;
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hash};

/// A `HashMap` with mutiple values for each key.
#[derive(Clone)]
pub struct MultiHashMap<K, V, S = RandomState>
where
    K: Hash + Eq,
    S: BuildHasher,
{
    map: hash_map::HashMap<K, Vec<V>, S>,
}

impl<K: Hash + Eq, V, B: BuildHasher> Serialize for MultiHashMap<K, V, B>
where
    K: Serialize,
    V: Serialize,
{
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.map.serialize(serializer)
    }
}

impl<K: Hash + Eq, V, S: BuildHasher> MultiHashMap<K, V, S> {
    /// Creates an empty `MultiHashMap`.
    pub fn new() -> MultiHashMap<K, V, RandomState> {
        MultiHashMap {
            map: hash_map::HashMap::new(),
        }
    }

    /// Creates an empty hash map with the given initial capacity.
    pub fn with_capacity(capacity: usize) -> MultiHashMap<K, V, RandomState> {
        MultiHashMap {
            map: hash_map::HashMap::with_capacity(capacity),
        }
    }

    /// Creates an empty `MultiHashMap` which will use the given hash builder
    /// to hash keys.
    pub fn with_hasher(hash_builder: S) -> Self {
        MultiHashMap {
            map: hash_map::HashMap::with_hasher(hash_builder),
        }
    }

    /// Creates an empty `MultiHashMap` with space for at least `capacity`
    /// elements, using `hasher` to hash the keys.
    pub fn with_capacity_and_hasher(capacity: usize, hasher: S) -> Self {
        MultiHashMap {
            map: hash_map::HashMap::with_capacity_and_hasher(capacity, hasher),
        }
    }

    /// Returns the number of elements the map can hold without reallocating.
    pub fn capacity(&self) -> usize { self.map.capacity() }

    /// Reserves capacity for at least `additional` more elements to be
    /// inserted in the `MultiHashMap`. The collection may reserve more
    /// space to avoid frequent reallocations.
    pub fn reserve(&mut self, additional: usize) { self.map.reserve(additional); }

    /// Shrinks the capacity of the map as much as possible. It will drop down
    /// as much as possible while maintaining the internal rules and
    /// possibly leaving some space in accordance with the resize policy.
    pub fn shrink_to_fit(&mut self) { self.map.shrink_to_fit(); }

    /// An iterator visiting all keys in arbitrary order.
    pub fn keys(&self) -> hash_map::Keys<K, Vec<V>> { self.map.keys() }

    /// An iterator visiting all values in arbitrary order.
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.map.values().flat_map(|x| x.iter())
    }

    /// An iterator visitinf all values mutably in arbitrary order.
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut V> {
        self.map.values_mut().flat_map(|x| x.iter_mut())
    }

    /// Iterates over all the keys and returns the associated values.
    pub fn iter(&self) -> hash_map::Iter<K, Vec<V>> { self.map.iter() }

    /// Returns the number of keys.
    pub fn num_keys(&self) -> usize { self.map.len() }

    /// Indicates if the map contains no elements.
    pub fn is_empty(&self) -> bool { self.map.is_empty() }

    /// Clears the map, returning all key-value pairs as an iterator. Keeps the
    /// allocated memory for reuse.
    pub fn drain(&mut self) -> hash_map::Drain<K, Vec<V>> { self.map.drain() }

    /// Clears the map, removing all keys and values.
    pub fn clear(&mut self) { self.map.clear(); }

    /// Returns the values mapped to the key.
    pub fn get<'a, Q: ?Sized>(&'a self, k: &Q) -> impl Iterator<Item = &'a V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.map.get(k).into_iter().flat_map(move |x| x.iter())
    }

    /// Indicates if the map contains a value for the specified key.
    pub fn contains_key<Q: ?Sized>(&self, k: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.map.contains_key(k)
    }

    /// Returns an iterator over mutable reference to the values mapped to the
    /// key.
    pub fn get_mut<'a, Q: ?Sized>(&'a mut self, k: &Q) -> impl Iterator<Item = &'a mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.map
            .get_mut(k)
            .into_iter()
            .flat_map(move |x| x.iter_mut())
    }

    /// Inserts new value to the map.
    pub fn insert(&mut self, k: K, v: V) {
        self.map.entry(k).or_insert_with(Vec::new).push(v);
    }

    /// Inserts new value to the map.
    pub fn insert_many(&mut self, k: K, values: Vec<V>) {
        if !values.is_empty() {
            match self.map.entry(k) {
                hash_map::Entry::Occupied(mut entry) => entry.get_mut().extend(values),
                hash_map::Entry::Vacant(entry) => {
                    entry.insert(values);
                }
            }
        }
    }

    /// Removes all the elements bound to a key.
    pub fn remove<Q: ?Sized>(&mut self, k: &Q) -> Vec<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.map.remove(k).unwrap_or_else(Vec::new)
    }

    // Not implemented:
    // * iter_mut
    // * into_iter(&mut self)
    // * entry
    // * extend
    // * extend from copy
}

impl<K, V, S> PartialEq for MultiHashMap<K, V, S>
where
    K: Eq + Hash,
    V: PartialEq,
    S: BuildHasher,
{
    fn eq(&self, other: &Self) -> bool { self.map.eq(&other.map) }
}

impl<K, V, S> Eq for MultiHashMap<K, V, S>
where
    K: Eq + Hash,
    V: Eq,
    S: BuildHasher,
{}

impl<K, V, S> std::fmt::Debug for MultiHashMap<K, V, S>
where
    K: Eq + Hash + std::fmt::Debug,
    V: std::fmt::Debug,
    S: BuildHasher,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { self.map.fmt(f) }
}

impl<K, V, S> Default for MultiHashMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher + Default,
{
    fn default() -> Self { Self::with_hasher(Default::default()) }
}

impl<'a, K, Q: ?Sized, V, S> std::ops::Index<&'a Q> for MultiHashMap<K, V, S>
where
    K: Eq + Hash + Borrow<Q>,
    Q: Eq + Hash,
    S: BuildHasher,
{
    type Output = [V];

    fn index(&self, index: &Q) -> &[V] { self.map.index(index) }
}

impl<'a, K, V, S> IntoIterator for &'a MultiHashMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    type Item = (&'a K, &'a Vec<V>);
    type IntoIter = hash_map::Iter<'a, K, Vec<V>>;

    fn into_iter(self) -> Self::IntoIter { self.iter() }
}

impl<K, V, S> IntoIterator for MultiHashMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    type Item = (K, Vec<V>);
    type IntoIter = hash_map::IntoIter<K, Vec<V>>;

    fn into_iter(self) -> Self::IntoIter { self.map.into_iter() }
}

impl<K, V, S> std::iter::FromIterator<(K, V)> for MultiHashMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher + Default,
{
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let mut map = MultiHashMap::default();
        for (k, v) in iter {
            map.insert(k, v);
        }
        map
    }
}

impl<K, V, S> std::iter::FromIterator<(K, Vec<V>)> for MultiHashMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher + Default,
{
    fn from_iter<T: IntoIterator<Item = (K, Vec<V>)>>(iter: T) -> Self {
        MultiHashMap {
            map: iter
                .into_iter()
                .filter(|&(_, ref vec)| !vec.is_empty())
                .collect(),
        }
    }
}
