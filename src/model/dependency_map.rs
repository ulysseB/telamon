//! Holds the latency between each node and its dependencies.
use crate::model::FastBound;
use itertools::Itertools;
use std::collections::hash_map;

use utils::*;

/// Holds the latency between each node and its dependencies. Nodes must be sorted.
#[derive(Clone, Debug)]
pub struct DependencyMap {
    deps: Vec<FnvHashMap<usize, FastBound>>,
}

impl DependencyMap {
    /// Creates an empty dependency map.
    pub fn new(size: usize) -> DependencyMap {
        DependencyMap {
            deps: (0..size).map(|_| FnvHashMap::default()).collect(),
        }
    }

    /// Returns the dependencies of a node.
    pub fn deps(&self, to: usize) -> &FnvHashMap<usize, FastBound> {
        &self.deps[to]
    }

    /// Add a dependency to a node.
    pub fn add_dep(&mut self, from: usize, to: usize, latency: FastBound) {
        assert!(from < to, "invalid dependency: {} -- {}", from, to);
        match self.deps[to].entry(from) {
            hash_map::Entry::Vacant(entry) => {
                entry.insert(latency);
            }
            hash_map::Entry::Occupied(mut entry) => {
                let old = entry.get_mut();
                if latency.is_better_than(old) {
                    *old = latency;
                }
            }
        }
    }

    /// Computes the latency between two nodes.
    pub fn latency(&self, from: usize, to: usize) -> Option<FastBound> {
        // TODO(cc_perf): only the latencies between from and to are needed.
        self.latency_to(to)[from].take()
    }

    /// Computes the latency to a given node.
    pub fn latency_to(&self, to: usize) -> Vec<Option<FastBound>> {
        let mut latencies = (0..to).map(|_| None).collect_vec();
        for (&pred, latency) in &self.deps[to] {
            latencies[pred] = Some(latency.clone());
        }
        for i in (0..to).rev() {
            if let Some(lat_to_dest) = latencies[i].clone() {
                for (&pred, lat_to_i) in &self.deps[i] {
                    let new_lat = lat_to_i.clone().chain(i, lat_to_dest.clone());
                    let old_lat = &mut latencies[pred];
                    if old_lat
                        .as_ref()
                        .map(|x| new_lat.is_better_than(x))
                        .unwrap_or(true)
                    {
                        *old_lat = Some(new_lat);
                    }
                }
            }
        }
        latencies
    }
}
