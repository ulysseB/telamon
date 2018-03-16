use super::*;
/// Directed acyclic graph.
use itertools::Itertools;
use std;
use std::cmp::{self, Ordering};

/// A directed acyclic graph given by adjacency list.
#[derive(Debug)]
pub struct Dag<T> {
    nodes: Vec<T>,
    before: Vec<Vec<usize>>,
    after: Vec<Vec<usize>>,
}

impl<T> Dag<T> {
    /// Computes a minimal DAG from a partial order.
    pub fn from_order<C>(mut nodes: Vec<T>, cmp: C) -> Dag<T>
    where C: Fn(&T, &T) -> Option<Ordering> {
        // Compute the set of nodes lesser than other nodes.
        let mut lesser_than = nodes.iter().map(|_| HashSet::default()).collect_vec();
        let mut is_duplicate = nodes.iter().map(|_| false).collect_vec();
        for (i, lhs) in nodes.iter().enumerate() {
            for (j, rhs) in nodes[..i].iter().enumerate() {
                match cmp(lhs, rhs) {
                    Some(Ordering::Less) => {
                        lesser_than[j].insert(i);
                    }
                    Some(Ordering::Equal) => {
                        is_duplicate[cmp::max(i, j)] = true;
                    }
                    Some(Ordering::Greater) => {
                        lesser_than[i].insert(j);
                    }
                    None => (),
                }
            }
        }
        // Compute the permutation and remove duplicates.
        let mut permutation = (0..nodes.len()).collect_vec();
        permutation.sort_by_key(|&i| lesser_than[i].len());
        permutation.retain(|&i| !is_duplicate[i]);
        // Compute the inverse of the permutation.
        let mut inv_permutation = nodes.iter().map(|_| nodes.len()).collect_vec();
        for (new_pos, &old_pos) in permutation.iter().enumerate() {
            inv_permutation[old_pos] = new_pos;
        }
        // Apply the permutation and drop duplicates.
        let mut new_nodes = Vec::with_capacity(nodes.len());
        unsafe {
            new_nodes.set_len(permutation.len());
            for (old_pos, &new_pos) in inv_permutation.iter().enumerate() {
                if new_pos == inv_permutation.len() {
                    std::ptr::drop_in_place(&mut nodes[old_pos]);
                } else {
                    let new_ptr = &mut new_nodes[new_pos];
                    std::ptr::copy_nonoverlapping(&nodes[old_pos], new_ptr, 1);
                }
            }
            nodes.set_len(0);
        }
        // Minimize and rename lesser sets. Also remove duplicates from the lesser sets.
        let before = permutation
            .iter()
            .map(|&old_pos| {
                lesser_than[old_pos]
                    .iter()
                    .cloned()
                    .filter(|lesser| {
                        !lesser_than[old_pos].iter().any(|&other_lesser| {
                            lesser_than[other_lesser].contains(lesser)
                        })
                    })
                    .map(|old_lesser| inv_permutation[old_lesser])
                    .filter(|&lesser| lesser != inv_permutation.len())
                    .collect_vec()
            })
            .collect_vec();
        // Inverse before to compute after.
        let mut after = new_nodes.iter().map(|_| Vec::new()).collect_vec();
        for after_id in 0..new_nodes.len() {
            for &before_id in &before[after_id] {
                after[before_id].push(after_id);
            }
        }
        // Build and return the Dag structure.
        Dag {
            nodes: new_nodes,
            before,
            after,
        }
    }

    /// Returns the list of nodes, in increasing order.
    pub fn nodes(&self) -> &[T] { &self.nodes }

    /// Returns the predecessors of the given node.
    pub fn before(&self, id: usize) -> &[usize] { &self.before[id] }

    /// Returns the successors of the given node.
    pub fn after(&self, id: usize) -> &[usize] { &self.after[id] }

    /// Returns the id of nodes without predecessors.
    pub fn minima(&self) -> Vec<usize> {
        self.before
            .iter()
            .enumerate()
            .filter(|&(_, x)| x.is_empty())
            .map(|x| x.0)
            .collect()
    }

    /// Returns all the predecessors of a node.
    pub fn predecessors(&self, id: usize) -> Vec<usize> {
        let mut visited = (0..id).map(|_| false).collect_vec();
        let mut stack = vec![id];
        let mut preds = vec![];
        while let Some(id) = stack.pop() {
            for &before in self.before(id) {
                if visited[before] {
                    continue;
                }
                visited[before] = true;
                preds.push(before);
                stack.push(before);
            }
        }
        preds
    }
}
