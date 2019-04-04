//! Provides different methods to select a candidate in a list.
use crate::device::Context;
use crate::explorer::candidate::Candidate;
use crate::explorer::choice;
use crate::explorer::config::{ChoiceOrdering, NewNodeOrder};
use rand::distributions::{Weighted, WeightedChoice};
use rand::prelude::*;
use std;
use utils::*;

/// A random rollout configuration
pub struct Rollout<'a> {
    /// The order in which choices should be considered during the rollout
    pub choice_order: &'a ChoiceOrdering,
    /// The policy to use when selecting among the available actions
    pub node_order: &'a NewNodeOrder,
    /// The context to use for propagation
    pub context: &'a dyn Context,
    /// Current best score.  Used in a branch-and-bound fashion with the lower-bound from the
    /// performance model, and possibly in other policy-specific computations.
    pub cut: f64,
}

pub enum RolloutError {
    /// The candidate is a dead-end
    DeadEnd,
    /// The candidate is fully specified
    Implementation,
}

impl<'a> Rollout<'a> {
    /// Repeatedly perform rollout steps on the `candidate` until it is fully specified,
    /// backtracking when deadends are reached.  Returns `None` if the whole subtree is dead.
    pub fn descend_backtrack<'c>(
        &self,
        candidate: Candidate<'c>,
    ) -> Option<Candidate<'c>> {
        let choice = choice::list(self.choice_order, &candidate.space).next();
        if let Some(choice) = choice {
            let mut children = choice
                .into_iter()
                .enumerate()
                .map(|(ix, action)| {
                    candidate
                        .apply_decision(self.context, action)
                        .ok()
                        .map(|child| (ix, child.bound.value(), Box::new(child)))
                })
                .collect::<Vec<_>>();
            loop {
                // Select a child among the valid ones
                if let Some(ix) = self.node_order.pick_index(
                    children
                        .iter()
                        .filter_map(|t| {
                            t.as_ref().map(|&(ix, bound, ref _cand)| (ix, bound))
                        })
                        .collect::<Vec<_>>()
                        .into_iter(),
                    self.cut,
                ) {
                    // If we fail the recursive `descend` call, the corresponding entry in the
                    // vector will be set to `None` and never retried.
                    if let Some(implementation) =
                        self.descend_backtrack(*children[ix].take().unwrap().2)
                    {
                        // Found something
                        break Some(implementation);
                    }
                } else {
                    // All the children are now invalid: this candidate is dead.
                    break None;
                }
            }
        } else {
            // This is already a fully-specified implementation
            Some(candidate)
        }
    }

    /// Perform one rollout step: select a set of actions according to the choice ordering, apply
    /// them, and select among the resulting candidates according to the rollout policy.
    fn step<'c>(&self, candidate: &Candidate<'c>) -> Result<Candidate<'c>, RolloutError> {
        if let Some(choice) = choice::list(self.choice_order, &candidate.space).next() {
            let mut children = candidate.apply_choice(self.context, choice);
            if let Some(idx) = self.node_order.pick_candidate(&children, self.cut) {
                Ok(children.swap_remove(idx))
            } else {
                Err(RolloutError::DeadEnd)
            }
        } else {
            Err(RolloutError::Implementation)
        }
    }

    /// Repeatedly perform rollout steps on the `candidate` until it is fully specified or a
    /// deadend is reached, in which case `None` is returned.
    pub fn descend<'c>(&self, mut candidate: Candidate<'c>) -> Option<Candidate<'c>> {
        loop {
            match self.step(&candidate) {
                Ok(next) => candidate = next,
                Err(RolloutError::DeadEnd) => break None,
                Err(RolloutError::Implementation) => break Some(candidate),
            }
        }
    }

    /// Identical to `descend`, except that all intermediate candidates (including the initial
    /// candidate) are appended to the back of the `path`.  This makes it easier to backtrack or
    /// investigate the tree locally after a dead-end.
    pub fn descend_with_path<'c>(
        &self,
        mut candidate: Candidate<'c>,
        path: &mut Vec<Candidate<'c>>,
    ) -> Option<Candidate<'c>> {
        loop {
            match self.step(&candidate) {
                Ok(next) => {
                    path.push(candidate);
                    candidate = next;
                }
                Err(RolloutError::Implementation) => break Some(candidate),
                Err(RolloutError::DeadEnd) => {
                    path.push(candidate);
                    break None;
                }
            }
        }
    }
}

/// A recursive function that takes a candidate and expands it until we have a completely specified
/// candidate that we can pass to the evaluator, or we find a dead-end
pub fn descend<'a>(
    choice_order: &ChoiceOrdering,
    node_order: NewNodeOrder,
    context: &dyn Context,
    candidate: Candidate<'a>,
    cut: f64,
) -> Option<Candidate<'a>> {
    Rollout {
        choice_order,
        node_order: &node_order,
        context,
        cut,
    }
    .descend(candidate)
}

impl NewNodeOrder {
    /// Called in montecarlo_descend, dispatch the choice of the next candidate according to our
    /// configuration
    pub fn pick_candidate<'a>(
        self,
        new_nodes: &[Candidate<'a>],
        cut: f64,
    ) -> Option<usize> {
        let items = new_nodes.iter().map(|c| c.bound.value()).enumerate();
        self.pick_index(items, cut)
    }

    /// Returns the index of the next candidate to consider.
    pub fn pick_index<IT>(self, nodes: IT, cut: f64) -> Option<usize>
    where
        IT: Iterator<Item = (usize, f64)> + Clone,
    {
        let mut nodes = nodes.filter(|&(_, b)| b < cut);
        match self {
            NewNodeOrder::Api => nodes.next().map(|(idx, _)| idx),
            NewNodeOrder::WeightedRandom => choose_cand_weighted(nodes, cut),
            NewNodeOrder::Bound => choose_cand_best(nodes),
            NewNodeOrder::Random => choose_cand_rand(nodes),
        }
    }
}

/// Given a vector of candidate reference, returns the index of the candidate with the minimum
/// bound.
fn choose_cand_best<IT>(nodes: IT) -> Option<usize>
where
    IT: Iterator<Item = (usize, f64)>,
{
    nodes.min_by(|x1, x2| cmp_f64(x1.1, x2.1)).map(|x| x.0)
}

/// Given a vector of candidate reference, just choose randomly the next candidate
fn choose_cand_rand<IT>(mut nodes: IT) -> Option<usize>
where
    IT: Iterator<Item = (usize, f64)> + Clone,
{
    let len = nodes.clone().count();
    if len == 0 {
        None
    } else {
        nodes.nth(thread_rng().gen_range(0, len)).map(|x| x.0)
    }
}

/// Given a vector of candidate references, returns the index of a weighted sort on the candidate
/// bounds
fn choose_cand_weighted<IT>(nodes: IT, cut: f64) -> Option<usize>
where
    IT: Iterator<Item = (usize, f64)> + Clone,
{
    let mut weighted_items = vec![];
    let mut rng = thread_rng();
    let max_bound = nodes
        .clone()
        .max_by(|&x1, &x2| cmp_f64(x1.1, x2.1))
        .map(|x| x.1)?;
    for (ind, x) in nodes {
        if cut.is_infinite() {
            let x_weight = 1 + (10f64 * max_bound / x).floor() as u32;
            weighted_items.push(Weighted {
                weight: x_weight,
                item: ind,
            });
        } else {
            assert!(
                x <= cut,
                "Compare bound fail, cut {:.3e}, cand: {:.3e}",
                cut,
                x
            );
            let weight = (1000f64 * (1f64 - x / cut)).floor() as u32;
            let weight = std::cmp::max(1, weight);
            weighted_items.push(Weighted { weight, item: ind });
        }
    }
    if weighted_items.is_empty() {
        None
    } else {
        Some(WeightedChoice::new(&mut weighted_items).sample(&mut rng))
    }
}
