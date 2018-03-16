//! Thread-safe LRU Cache.
use super::*;
use linked_hash_map::LinkedHashMap;
use std;
use std::hash::Hash;
use std::sync::{Arc, RwLock};

/// A thread-safe LRU Cache.
#[derive(Debug)]
pub struct Cache<K, V>
where K: Hash + Eq
{
    map: RwLock<linked_hash_map::LinkedHashMap<K, Arc<V>>>,
    capacity: usize,
}

impl<K, V> Cache<K, V>
where K: Hash + Eq + Clone
{
    /// Returns a new `Cache` that can store `capacity` elements.
    pub fn new(capacity: usize) -> Self {
        let map = RwLock::new(LinkedHashMap::with_capacity(capacity + 1));
        Cache { map, capacity }
    }

    /// Returns the element associated to `key` in the cache. Generates the element with
    /// `gen` and store it in the cache if it is not already present.
    pub fn get<F>(&self, key: &K, gen: F) -> Arc<V>
    where F: FnOnce() -> V {
        // Check if the entry alread exists.
        {
            let guard = self.map.read().unwrap();
            if let Some(v) = guard.get(key) {
                return v.clone();
            }
            std::mem::drop(guard);
        }
        // Otherwise tack an exclusive lock
        let mut map = self.map.write().unwrap();
        let v = map.entry(key.clone())
            .or_insert_with(move || Arc::new(gen()))
            .clone();
        if map.len() > self.capacity {
            map.pop_front();
        }
        v
    }

    /// Removes all elements from the `Cache`.
    pub fn clear(&mut self) { self.map.get_mut().unwrap().clear(); }
}
