//! Exploration of the search space.

use device::Context;
use explorer::candidate::Candidate;
use explorer::{choice, local_selection};
use explorer::config::{BanditConfig, NewNodeOrder, OldNodeOrder};
use explorer::store::Store;
use itertools::Itertools;
use std;
use std::f64;
use std::sync::{Weak, Arc, RwLock};
use std::sync::atomic::{AtomicUsize, Ordering};
use utils::*;

struct TreeStats {
    num_deadends: AtomicUsize,
}

impl TreeStats {
    fn new() -> Self {
        TreeStats {
            num_deadends: AtomicUsize::new(0),
        }
    }
}

/// A search tree to perform a multi-armed bandit search.
pub struct Tree<'a, 'b> {
    shared_tree: Arc<RwLock<SubTree<'a>>>,
    cut: RwLock<f64>,
    config: &'b BanditConfig,
    stats: TreeStats,
}

impl<'a, 'b> Tree<'a, 'b> {
    /// Creates a new search tree containing the given candidates.
    pub fn new(candidates: Vec<Candidate<'a>>, config: &'b BanditConfig) -> Self {
        let root = SubTree::from_candidates(candidates, std::f64::INFINITY);
        Tree {
            shared_tree: Arc::new(RwLock::new(root)),
            cut: RwLock::new(std::f64::INFINITY),
            config,
            stats: TreeStats::new(),
        }
    }

    /// Removes the dead ends along the given path. Assumes the path points to a dead-end.
    /// Updates bounds along the way.
    fn clean_deadends(&self, mut path: Path<'a>, cut: f64) {
        // A `None` bound indicates the path points to a dead-end.
        let mut bound = None;
        while let Some((node, pos)) = path.0.pop() {
            if let Some(node) = node.upgrade() {
                let mut lock = unwrap!(node.write());
                if let Some(bound) = bound {
                    lock.children[pos].update_bound(bound);
                    // If the bound was set, then we finished removing deadends.
                    return;
                } else {
                    lock.children[pos] = SubTree::Empty;
                    bound = lock.bound();
                    // Children with a bound above the cut are considered as dead-ends.
                    if bound.map(|b| b >= cut).unwrap_or(false) { bound = None; }
                }
            } else { return; }
        }
        // If we did not returned before, we have reached the root of the tree.
        let mut lock = unwrap!(self.shared_tree.write());
        if let Some(bound) = bound { lock.update_bound(bound); } else {
            *lock = SubTree::Empty
        }
    }

    /// Descend in the tree and tries to reach a leaf from the given `DescendState`.
    fn descend(&self, context: &Context, mut state: DescendState<'a>, cut: f64)
        -> Option<(Candidate<'a>, Path<'a>)>
    {
        let mut path = Path::default();
        loop {
            match std::mem::replace(&mut state, DescendState::DeadEnd) {
                DescendState::DeadEnd => {
                    self.clean_deadends(path, cut);
                    return None
                }
                DescendState::Leaf(leaf) => {
                    self.clean_deadends(path.clone(), cut);
                    return Some((leaf, path))
                }
                DescendState::InternalNode(node, is_new) => {
                    if is_new && self.config.monte_carlo {
                        let res = unwrap!(node.write())
                            .descend_noexpand(&self.config, context, cut);
                        // We manually process the first two levels of the search as they
                        // are still in the tree and thus must be updated if needed.
                        if let Some((idx, maybe_candidate)) = res {
                            path.0.push((Arc::downgrade(&node), idx));
                            match maybe_candidate {
                                Ok(candidate) => {
                                    let res = local_selection::descend(
                                        self.config.new_nodes_order, context, candidate, cut);
                                    return res.map(|c| (c, path));
                                }
                                Err(new_state) => state = new_state,
                            }
                        } else {
                            continue;
                        }
                    } else {
                        let next = unwrap!(node.write())
                            .descend(&self.config, context, cut);
                        if let Some((idx, new_state)) = next {
                            path.0.push((Arc::downgrade(&node), idx));
                            state = new_state;
                        }
                    }
                }
            }
        }
    }
}

impl<'a, 'b> Store<'a> for Tree<'a, 'b> {
    type PayLoad = Path<'a>;

    fn update_cut(&self, new_cut: f64) {
        *unwrap!(self.cut.write()) = new_cut;
        let root = unwrap!(self.shared_tree.write()).trim(new_cut);
        let mut stack = root.into_iter().collect_vec();
        while let Some(node) = stack.pop() {
            let mut lock = unwrap!(node.write());
            for subtree in &mut lock.children {
                stack.extend(subtree.trim(new_cut));
            }
        }
        info!("trimming finished");
    }

    fn commit_evaluation(&self, mut path: Self::PayLoad, eval: f64) {
        while let Some((node, idx)) = path.0.pop() {
            if let Some(node) = node.upgrade() {
                if !unwrap!(node.write()).update_rewards(&self.config, idx, eval) {
                    return;
                }
            }
        }
    }

    fn explore(&self, context: &Context) -> Option<(Candidate<'a>, Self::PayLoad)> {
        loop {
            let cut: f64 = { *unwrap!(self.cut.read()) };
            let state = unwrap!(self.shared_tree.write()).descend(context, cut);
            if let DescendState::DeadEnd = state { return None; }
            let res = self.descend(context, state, cut);
            if res.is_some() {
                return res;
            } else {
                self.stats.num_deadends.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    fn print_stats(&self) {
        warn!("=== Bandit statistics ===");
        warn!("Deadends encountered: {}", self.stats.num_deadends.load(Ordering::Relaxed));
    }
}



/// Path to follow to reach a leaf in the tree.
#[derive(Clone, Default)]
pub struct Path<'a>(Vec<(Weak<RwLock<Children<'a>>>, usize)>);

/// The search tree that will be traversed
enum SubTree<'a> {
    /// The subtree has been expanded and has children.
    InternalNode(Arc<RwLock<Children<'a>>>, f64),
    /// The subtree has not been expanded yet.
    UnexpandedNode(Candidate<'a>),
    /// The subtree is empty.
    Empty,
}

impl<'a> SubTree<'a> {
    /// Creates a `SubTree` containing the given list of candidates.
    fn from_candidates(candidates: Vec<Candidate<'a>>, cut: f64) -> SubTree<'a> {
        let children = Children::from_candidates(candidates, cut);
        if let Some(bound) = children.bound() {
            SubTree::InternalNode(Arc::new(RwLock::new(children)), bound)
        } else { SubTree::Empty }
    }

    /// Trims the branch if it has with an evaluation time guaranteed to be worse than
    /// `cut`. Returns the childrens to trim if any,
    fn trim(&mut self, cut: f64) -> Option<Arc<RwLock<Children<'a>>>> {
        if self.bound() >= cut {
            *self = SubTree::Empty;
            None 
        } else if let SubTree::InternalNode(ref inner, _) = *self {
            Some(inner.clone())
        } else { None }
    }
    
    /// Returns the lower bound on the execution time on the `SubTree`.
    fn bound(&self) -> f64 {
        match *self {
            SubTree::InternalNode(_, bound) => bound,
            SubTree::UnexpandedNode(ref cand) => cand.bound.value(),
            SubTree::Empty => std::f64::INFINITY,
        }
    }

    /// Indicates if the `SubTree` is empty.
    fn is_empty(&self) -> bool {
        if let SubTree::Empty = *self { true } else { false }
    }

    /// Descend one level in the tree, expanding it if necessary.
    fn descend(&mut self, context: &Context, cut: f64) -> DescendState<'a> {
        match std::mem::replace(self, SubTree::Empty) {
            SubTree::InternalNode(node, bound) => {
                *self = SubTree::InternalNode(node.clone(), bound);
                DescendState::InternalNode(node, false)
            },
            SubTree::UnexpandedNode(candidate) => {
                let choice = choice::list(&candidate.space).next();
                if let Some(choice) = choice {
                    let candidates = candidate.apply_choice(context, choice);
                    let children = Children::from_candidates(candidates, cut);
                    if let Some(bound) = children.bound() {
                        let children = Arc::new(RwLock::new(children));
                        *self = SubTree::InternalNode(children.clone(), bound);
                        DescendState::InternalNode(children, true)
                    } else { DescendState::DeadEnd }
                } else { DescendState::Leaf(candidate) } 
            },
            SubTree::Empty => DescendState::DeadEnd,
        }
    }

    /// Update the bound registered in the node, if possible.
    fn update_bound(&mut self, bound: f64) {
        if let SubTree::InternalNode(_, ref mut old_bound) = *self { *old_bound = bound }
    }
}

/// Indicates the current state of a descent in the search tree.
enum DescendState<'a> {
    /// The descent reached an internal node with multiple children. Indicates if the
    /// children were created during the descent.
    InternalNode(Arc<RwLock<Children<'a>>>, bool),
    /// The descent reached a leaf of the tree.
    Leaf(Candidate<'a>),
    /// The descent reached a dead-end.
    DeadEnd,
}

/// Holds the children of a `SubTree::InternalNode`.
pub struct Children<'a> {
    children: Vec<SubTree<'a>>,
    rewards: Vec<(Vec<f64>, usize)>,
}

impl<'a> Children<'a> {
    /// Creates a new children containing the given candidates, if any. Only keeps
    /// candidates above the cut.
    fn from_candidates(candidates: Vec<Candidate<'a>>, cut: f64) -> Self {
        let children = candidates.into_iter().filter(|c| c.bound.value() < cut)
            .map(SubTree::UnexpandedNode).collect_vec();
        let rewards = children.iter().map(|_| (vec![], 0)).collect();
        Children { children: children, rewards }
    }

    /// Returns the lowest bound of the children, if any.
    fn bound(&self) -> Option<f64> {
        self.children.iter().map(|c| c.bound()).min_by(|&lhs, &rhs| cmp_f64(lhs, rhs))
    }

    /// Trim children (that is, replace with an empty `SubTree`) children whose bounds are
    /// higher than the cut. Also clean-up rewards.
    fn trim(&mut self, cut: f64) {
        for (child, reward) in self.children.iter_mut().zip_eq(&mut self.rewards) {
            if child.bound() >= cut {
                *child = SubTree::Empty;
                *reward = (vec![], 0);
            }
        }
    }

    /// Descend one level in the tree, expanding it if necessary.
    fn descend(&mut self, config: &BanditConfig, context: &Context, cut: f64)
        -> Option<(usize, DescendState<'a>)>
    {
        self.trim(cut);
        self.pick_child(config, cut).map(|idx| {
            self.rewards[idx].1 += 1;
            (idx, self.children[idx].descend(context, cut))
        })
    }

    /// Descends one level in the tree and apply an action on the candidate reached to
    /// generate a candidate that is not owned by the tree. Assumes all the `Children` are
    /// either dead-ends or unexpanded nodes. Returns the index of the selected child,
    /// the generated `Candidate` or the current state of the exploration if the candidate
    /// could not be generated. Returns `None` if a dead-end is reached before we could
    /// descend.
    fn descend_noexpand(&mut self, config: &BanditConfig, context: &Context, cut: f64)
        -> Option<(usize, Result<Candidate<'a>, DescendState<'a>>)>
    {
        self.trim(cut);
        self.pick_child(config, cut).map(|idx| {
            self.rewards[idx].1 += 1;
            let node = std::mem::replace(&mut self.children[idx], SubTree::Empty);
            let cand = match node {
                SubTree::UnexpandedNode(c) => c,
                SubTree::Empty => return (idx, Err(DescendState::DeadEnd)),
                SubTree::InternalNode(ref node, _) =>
                    return (idx, Err(DescendState::InternalNode(node.clone(), true))),
            };
            let choice = choice::list(&cand.space).next();
            let out = if let Some(choice) = choice {
                let cands = cand.apply_choice(context, choice);
                self.children[idx] = SubTree::UnexpandedNode(cand);
                let order = config.new_nodes_order;
                if let Some(cand) = local_selection::pick_candidate(order, cands, cut) {
                    Ok(cand)
                } else { Err(DescendState::DeadEnd) }
            } else { Err(DescendState::Leaf(cand)) };
            (idx, out)
        })
    }

    /// Picks a child to descend in. Returns `None` if all children are cut.
    fn pick_child(&self, config: &BanditConfig, cut: f64) -> Option<usize> {
        let new_nodes = self.children.iter().map(|c| c.bound()).enumerate()
            .filter(|&(idx, _)| self.rewards[idx].1 == 0);
        local_selection::pick_index(config.new_nodes_order, new_nodes, cut).or_else(|| {
            match config.old_nodes_order {
                OldNodeOrder::Bound => {
                    let children = self.children.iter().map(|c| c.bound()).enumerate();
                    local_selection::pick_index(NewNodeOrder::Bound, children, cut)
                }
                OldNodeOrder::WeightedRandom => {
                    let children = self.children.iter().map(|c| c.bound()).enumerate();
                    let order = NewNodeOrder::WeightedRandom;
                    local_selection::pick_index(order, children, cut)
                }
                OldNodeOrder::Bandit => {
                    pick_bandit_arm(config, &self.children, &self.rewards, cut)
                }
            }
        })
    }


    /// Update a rewards list given a new value and the position where it was found
    /// returns true if the value was inserted in the node.
    fn update_rewards(&mut self, config: &BanditConfig, pos: usize, val: f64) -> bool {
        if self.children[pos].is_empty() { return false; }
        let total_trials = self.rewards.iter().map(|x| x.0.len()).sum::<usize>();
        // If total trials is less than THRESHOLD, then we simply push our new value
        // in the node where it was found.
        if total_trials < config.threshold {
            self.rewards[pos].0.push(val);
            true
        } else {
            let (child, idx, max) = unwrap!(self.find_max_rewards());
            if val < max {
                self.rewards[child].0[idx] = val;
                true
            } else { false }
        }
    }

    /// Returns the tuple `(child_index, eval_index, value)` containing the position and
    /// the value of the biggest reward registered.
    fn find_max_rewards(&self) -> Option<(usize, usize, f64)> {
        self.rewards.iter().enumerate().flat_map(|(child_idx, rewards)| {
            let max = rewards.0.iter().cloned().enumerate()
                .max_by(|lhs, rhs| cmp_f64(lhs.1, rhs.1));
            max.map(|(idx, value)| (child_idx, idx, value))
        }).max_by(|x1, x2| cmp_f64(x1.2, x2.2))
    }
}

/// Picks a candidate below the bound following a multi-armed bandit approach.
fn pick_bandit_arm(config: &BanditConfig,
                   children: &[SubTree],
                   rewards: &[(Vec<f64>, usize)],
                   cut: f64) -> Option<usize> {
    let nb_tested = rewards.iter().fold(0, |acc, ref x| acc + x.1);
    rewards.iter().enumerate().filter(|&(i, _)| children[i].bound() < cut)
        .map(|(ind, x)| (ind, heval(config, x.0.len(), x.1, nb_tested, rewards.len())))
        .max_by(|x1, x2| cmp_f64(x1.1, x2.1))
        .map(|(ind, _)| ind)
}

/// gives a "score" to a branch of the tree at a given node n_successes is the number of
/// successes of that branch (that is, the number of leaves that belong to the THRESHOLD
/// best of that node and which come from that particular branch).
/// * `n_branch_trials` is the number of trials of that branch (both failed and succeeded),
/// * `n_trials` is  the number of trials of the node and k the number of branches in the
///   node.
fn heval(config: &BanditConfig,
         n_successes: usize,
         n_branch_trials: usize,
         n_trials: usize,
         n_branches: usize) -> f64 {
    assert!(n_branches > 0);
    assert!(n_branch_trials <= n_trials);
    if n_branch_trials == 0 { std::f64::INFINITY } else {
        let alpha = (2. * (n_trials * n_branches) as f64 / config.delta).ln().max(0.);
        let sqrt_body = alpha * (2. * n_successes as f64 + alpha);
        (n_successes as f64 + alpha + sqrt_body.sqrt()) / n_branch_trials as f64
    }
}
