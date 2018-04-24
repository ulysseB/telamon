//! Exploration of the search space.

use device::Context;
use explorer::candidate::Candidate;
use explorer::{choice, montecarlo};
use explorer::config::{BanditConfig, NewNodeOrder, OldNodeOrder};
use explorer::store::Store;
use itertools::Itertools;
use std;
use std::f64;
use std::sync::{ Weak, Arc, RwLock};
use utils::*;

/// A search tree to perform a multi-armed bandit search.
pub struct Tree<'a, 'b> {
    // FIXME: replace SubTree by Children
    shared_tree: Arc<RwLock<SubTree<'a>>>,
    cut: RwLock<f64>,
    config: &'b BanditConfig,
}

impl<'a, 'b> Tree<'a, 'b> {
    /// Creates a new search tree containing the given candidates.
    pub fn new(candidates: Vec<Candidate<'a>>, config: &'b BanditConfig) -> Self {
        let root = SubTree::from_candidates(candidates, std::f64::INFINITY);
        Tree {
            shared_tree: Arc::new(RwLock::new(root)),
            cut: RwLock::new(std::f64::INFINITY),
            config,
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
            let mut state = unwrap!(self.shared_tree.write()).descend(context, cut);
            if let DescendState::DeadEnd = state { return None; }
            // FIXME: >>>>>>>>>>>>
            // FIXME: match state in a loop
            match state {
                DescendState::DeadEnd => {
                    panic!()
                    // FIXME: ascend no val and break
                },
                DescendState::Leaf(_) => panic!("unexpected leaf"),
                DescendState::InternalNode(node, is_complete) => {
                    match iter_descend(self.config, context, node, cut) {
                        // FIXME: get rid of OldDescendResult
                        OldDescendResult::Finished => { return None; }
                        OldDescendResult::DeadEnd(path) => {
                            self.clean_deadends(path, cut);
                        }
                        OldDescendResult::Leaf(cand, path) => {
                            self.clean_deadends(path.clone(), cut);
                            return Some((cand, path));
                        }
                        OldDescendResult::MonteCarloLeaf(cand, path) => {
                            // FIXME: ensure we cleanup deadends if the montecarlo has 0 levels
                            return Some((cand, path));
                        }
                        // We have no information on where and how the search fail, so we can not
                        // update the tree in any way.
                        OldDescendResult::FailedMonteCarlo => {}
                    }
                },
            }
        }
    }
}



/// Path to follow to reach a leaf in the tree.
#[derive(Clone)]
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

    /// Picks a child to descend in. Returns `None` if all children are cut.
    fn pick_child(&self, config: &BanditConfig, cut: f64) -> Option<usize> {
        let new_nodes = self.children.iter().map(|c| c.bound()).enumerate()
            .filter(|&(idx, _)| self.rewards[idx].1 == 0);
        montecarlo::next_cand_index(config.new_nodes_order, new_nodes, cut).or_else(|| {
            match config.old_nodes_order {
                OldNodeOrder::Bound => {
                    let children = self.children.iter().map(|c| c.bound()).enumerate();
                    montecarlo::next_cand_index(NewNodeOrder::Bound, children, cut)
                }
                OldNodeOrder::WeightedRandom => {
                    let children = self.children.iter().map(|c| c.bound()).enumerate();
                    let order = NewNodeOrder::WeightedRandom;
                    montecarlo::next_cand_index(order, children, cut)
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

// FIXME: >>>>>>>>>>>>>>>..
// FIXME: add stats on the number of nodes inside the tree
// FIXME: propagate the bound upward when expanding and deleting branches
//  - this should remove the need for is_complete in the payload
// FIXME: must cut nodes above the bound when exploring the tree
/// Called in thread_descend_tree, iter on the value of descend_node
/// Builds the parents stack and returns an appropriate value at the end
fn iter_descend<'a>(config: &BanditConfig,
                    context: &Context,
                    node_root: Arc<RwLock<Children<'a>>>,
                    cut: f64) -> OldDescendResult<'a> {
    let mut parent_stack = vec![];
    let mut search_node_lock = node_root;
    loop {
        let next_node;
        {
            let mut search_node = search_node_lock.write().unwrap();
            if let Some((idx, state)) = search_node.descend(config, context, cut) {
                let weak_ref = Arc::downgrade(&search_node_lock);
                parent_stack.push((weak_ref, idx));
                match state {
                    DescendState::DeadEnd => {
                        return OldDescendResult::DeadEnd(Path(parent_stack))
                    },
                    DescendState::Leaf(candidate) => {
                        return OldDescendResult::Leaf(candidate, Path(parent_stack))
                    },
                    DescendState::InternalNode(children, is_new) => {
                        if is_new && config.monte_carlo {
                            std::mem::drop(search_node);
                            // FIXME: simplify monte-carlo
                            let monte_cand_opt = unwrap!(children.write())
                                .start_montecarlo(config, context, cut);
                            if let Some(cand) = monte_cand_opt {
                                return handle_montecarlo_descend(
                                    config, context, cand, cut, Path(parent_stack));
                            } else { return OldDescendResult::FailedMonteCarlo; }
                        } else { next_node = children.clone(); }
                    }
                }
            } else {
                return if parent_stack.is_empty() { OldDescendResult::Finished } else {
                    OldDescendResult::DeadEnd(Path(parent_stack))
                }
            }
        }
        search_node_lock = next_node;
    }
}

// These types are used as return type for the functions traversing the tree
pub enum OldDescendResult<'a> {
    Finished,
    DeadEnd(Path<'a>),
    Leaf(Candidate<'a>, Path<'a>),
    MonteCarloLeaf(Candidate<'a>, Path<'a>),
    FailedMonteCarlo,
}

/// Handles the descend from a candidate and returns an appropriate OldDescendResult
fn handle_montecarlo_descend<'a>(config: &BanditConfig,
                                 context: &Context,
                                 cand: Candidate<'a>,
                                 cut: f64,
                                 parent_stack: Path<'a>) -> OldDescendResult<'a> {
    let order = config.new_nodes_order;
    if let Some(cand) = montecarlo::descend(order, context, cand, cut) {
        OldDescendResult::MonteCarloLeaf(cand, parent_stack)
    } else { OldDescendResult::FailedMonteCarlo }
}

impl<'a> Children<'a> {
    /// We have a newly expanded node, we want to do a montecarlo descend on it
    /// As we cannot directly own the original candidate (which must stay in the tree, and
    /// which we do not want to clone) we have to do this on the freshly expanded node.
    fn start_montecarlo(&mut self, config: &BanditConfig, context: &Context, cut: f64)
        -> Option<Candidate<'a>>
    {
        if self.children.is_empty() {
            panic!("called montecarlo on a node with no children")
        }
        let ind;
        {
            let new_nodes = self.children.iter().map(|node| {
                if let SubTree::UnexpandedNode(ref cand) = *node { cand } else {
                    // We must only call this function on a newly expanded node, which
                    // means that there must nothing but unexpandedNode in it.
                    panic!()
                }
            });
            let node_bounds = new_nodes.map(|c| c.bound.value()).enumerate();
            ind = unwrap!(montecarlo::next_cand_index(config.new_nodes_order, node_bounds, cut));
            let cand_ref = if let SubTree::UnexpandedNode(ref cand) = self.children[ind]
            {
                cand
            } else { panic!() };
            let choice_opt = choice::list(&cand_ref.space).next();
            if let Some(choice) = choice_opt {
                let new_nodes = cand_ref.apply_choice(context, choice);
                return montecarlo::choose_next_cand(
                    config.new_nodes_order, new_nodes, cut);
            }
        }
        // This is, logically speaking, the else branch of of the last if let Some(choice)
        // as we can only be here if this 'if' branch was not taken - there is a return in
        // each other branches. We need to do that so we can hold a mutable reference on
        // self.children.
        let node = std::mem::replace(&mut self.children[ind], SubTree::Empty);
        if let SubTree::UnexpandedNode(cand) = node { Some(cand) } else { panic!() }
    }
}
// FIXME: <<<<<<<<<<<<<<<<

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
    if n_trials == 0 { std::f64::INFINITY } else {
        let f = (n_trials * n_branches) as f64;
        let alpha = f.ln() / config.delta;
        let sqrt_body = alpha * (2. * n_successes as f64 + alpha);
        (n_successes as f64 + alpha + sqrt_body.sqrt()) / n_branch_trials as f64
    }
}
