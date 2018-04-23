//! Exploration of the search space.

use device::Context;
use explorer::candidate::Candidate;
use explorer::{choice, montecarlo};
use explorer::config::{BanditConfig, NewNodeOrder, OldNodeOrder};
use explorer::store::Store;
use itertools::Itertools;
use rand::{Rng, thread_rng};
use rand::distributions::{ Weighted, WeightedChoice, IndependentSample};
use std;
use std::f64;
use std::sync::{ Weak, Arc, RwLock};
use utils::*;

/// A search tree to perform a multi-armed bandit search.
pub struct Tree<'a, 'b> {
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
}

impl<'a, 'b> Store<'a> for Tree<'a, 'b> {
    type PayLoad = Payload<'a>;

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

    fn commit_evaluation(&self, mut payload: Self::PayLoad, eval: f64) {
// FIXME: >>>>>>>>>>>>
// FIXME: call SubTree Node or Subtree and call `Node` `Children`
// FIXME: add stats on the number of nodes inside the tree
// FIXME: propagate the bound upward when expanding and deleting branches
        let parent_opt = payload.path.pop();
        if let Some((weak_node, pos)) = parent_opt {
            if let Some(node_lock) = weak_node.upgrade() {
                iter_ascend(node_lock, eval, pos, payload.leaf_pos, &mut payload.path,
                            payload.is_complete)
            }
        }
    }

    /// Here, explore is constantly trying to find a new completely specified candidate by
    /// calling thread_descend_tree - a thread safe seach in the tree - continuously.
    fn explore(&self, context: &Context) -> Option<(Candidate<'a>, Self::PayLoad)> {
        loop {
            let shared_tree = Arc::clone(&self.shared_tree);
            // FIXME: get rid of DescendResult
            match thread_descend_tree(self.config, context, shared_tree, &self.cut) {
                DescendResult::Finished => { return None; }
                DescendResult::DeadEnd(pos, parent_stack) => {
                    // FIXME: This remove the dead branch, ensure it is not called from montecarlo
                    thread_ascend_tree_no_val(pos, parent_stack);
                }
                DescendResult::Leaf(cand, leaf_pos, path) => {
                    return Some((cand, Payload { path, leaf_pos, is_complete: true }));
                }
                DescendResult::MonteCarloLeaf(cand, leaf_pos, path) => {
                    // FIXME: ensure we return complete if the montecarlo has 0 levels
                    return Some((cand, Payload { path, leaf_pos, is_complete: false }));
                }
                // We have no information on where and how the search fail, so we can not
                // update the tree in any way.
                DescendResult::FailedMonteCarlo => {}
            }
        }
    }
}


/// Transmits the information needed to update the tree after an update.
pub struct Payload<'a> {
    /// Path in the tree that lead to the evaluated branch.
    path: NodeStack<'a>,
    /// Indicates the position of the leaf in the last node.
    leaf_pos: usize,
    /// Indicates if the branch was fully evaluated and should be cut or not.
    is_complete: bool,
}


/// The search tree that will be traversed
enum SubTree<'a> {
    /// The subtree has been expanded and has children.
    InternalNode(Arc<RwLock<Node<'a>>>, f64),
    /// The subtree has not been expanded yet.
    UnexpandedNode(Candidate<'a>),
    /// The subtree is empty.
    Empty,
}

impl<'a> SubTree<'a> {
    /// Creates a `SubTree` containing the given list of candidates.
    fn from_candidates(candidates: Vec<Candidate<'a>>, cut: f64) -> SubTree<'a> {
        let bound = candidates.iter().map(|c| c.bound.value())
            .max_by(|&lhs, &rhs| cmp_f64(lhs, rhs)).unwrap_or(std::f64::INFINITY);
        let children = candidates.into_iter().filter(|c| c.bound.value() < cut)
            .map(SubTree::UnexpandedNode).collect_vec();
        if children.is_empty() { SubTree::Empty } else {
            SubTree::InternalNode(Arc::new(RwLock::new(Node::new(children))), bound)
        }
    }

    /// Trims the branch if it has with an evaluation time guaranteed to be worse than
    /// `cut`. Returns the childrens to trim if any,
    fn trim(&mut self, cut: f64) -> Option<Arc<RwLock<Node<'a>>>> {
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
}

// FIXME: >>>>>>>>>>>>>>>..
pub struct Node<'a> {
    children: Vec<SubTree<'a>>,
    rewards: Vec<(Vec<f64>, usize)>,
}

impl<'a> Node<'a> {
    /// Creates a new children of a node containing the given `SubTree`s.
    fn new(children: Vec<SubTree<'a>>) -> Self {
        let rewards = children.iter().map(|_| (vec![], 0)).collect();
        Node { children: children, rewards }
    }
}

type NodeStack<'a> = Vec<(Weak<RwLock<Node<'a>>>, Option<usize>)>;


// These types are used as return type for the functions traversing the tree
pub enum DescendResult<'a> {
    Finished,
    DeadEnd(usize, NodeStack<'a>),
    Leaf(Candidate<'a>, usize, NodeStack<'a>),
    MonteCarloLeaf(Candidate<'a>, usize, NodeStack<'a>),
    FailedMonteCarlo,
}

enum NodeDescendResult<'a> {
    Node(Arc<RwLock<Node<'a>>>, usize),
    MonteCarlo(Arc<RwLock<Node<'a>>>, usize),
    Leaf(Candidate<'a>, usize),
    DeadEndFromExpand(usize),
    DeadEnd,
}

pub enum NodeAscendResult<'a> {
    Node(Arc<RwLock<Node<'a>>>, Option<usize>),
    NodeNoVal(Arc<RwLock<Node<'a>>>, Option<usize>),
    NodeNotCompleted(Arc<RwLock<Node<'a>>>, Option<usize>),
    // FIXME: can merge some variants
    InvalidParent,
    TreeUpdated,
    Root,
}

enum ExpandRes<'a> {
    DeadEnd,
    Node(SubTree<'a>),
    Leaf(Candidate<'a>),
}

enum UpdateRes {
    KeepGoing,
    ValNotInserted,
    TreeUpdated,
}


/// Call descend_node in a loop until it finds either a DeadEnd or a Leaf
/// if the deadend is found at the root, then the return value states this fact
fn thread_descend_tree<'a>(
    bandit_config: &BanditConfig,
    context: &Context,
    root_lock: Arc<RwLock<SubTree<'a>>>,
    best_val: &RwLock<f64>) -> DescendResult<'a>
{
    let node_root;
    {
        let root = root_lock.read().unwrap();
        if let SubTree::Empty = *root {
          return DescendResult::Finished;
        }
        node_root = match *root {
            SubTree::InternalNode(ref n, _) => Arc::clone(n),
            _ => {panic!("At this point, root should be a node");}
        };
    }
    let best_val = *best_val.read().unwrap();
    iter_descend(bandit_config, context, node_root, best_val)
}


/// Called in thread_descend_tree, iter on the value of descend_node
/// Builds the parents stack and returns an appropriate value at the end
fn iter_descend<'a>(config: &BanditConfig,
                    context: &Context,
                    node_root: Arc<RwLock<Node<'a>>>,
                    best_val: f64) -> DescendResult<'a> {
    let mut parent_stack = vec![];
    let mut search_node_lock = node_root;
    let mut current_pos = None;
    loop {
        let next_node;
        //let mut montecarlo_candidate = None;
        {
            let mut search_node = search_node_lock.write().unwrap();
            match search_node.descend_node(context, config, best_val) {
                NodeDescendResult::Node(subtree_arc, pos) => {
                    let weak_ref = Arc::downgrade(&search_node_lock);
                    parent_stack.push((weak_ref, current_pos));
                    current_pos = Some(pos);
                    next_node = Arc::clone(&subtree_arc);
                }
                NodeDescendResult::MonteCarlo(subtree_arc, pos) => {
                    let weak_ref = Arc::downgrade(&search_node_lock);
                    //next_node = Arc::clone(&subtree_arc);
                    parent_stack.push((weak_ref, current_pos));
                    let monte_cand_opt = subtree_arc.write().unwrap()
                        .start_montecarlo(config, context, best_val);
                    // We don't need to hold this lock during the montecarlo descend, so we just
                    // drop it
                    std::mem::drop(search_node);
                    if let Some(cand) = monte_cand_opt {
                        return handle_montecarlo_descend(
                            config, context, cand, pos, best_val, parent_stack);
                    } else { return DescendResult::FailedMonteCarlo; }
                }
                NodeDescendResult::Leaf(candidate, pos) => {
                    let weak_ref = Arc::downgrade(&search_node_lock);
                    parent_stack.push((weak_ref, current_pos));
                    return DescendResult::Leaf(candidate, pos, parent_stack);
                }
                NodeDescendResult::DeadEndFromExpand(pos) => {
                    let weak_ref = Arc::downgrade(&search_node_lock);
                    parent_stack.push((weak_ref, current_pos));
                    return DescendResult::DeadEnd(pos, parent_stack);
                }
                // We do not want to push the node in DeadEnd in parent_stack as
                // it is not an interesting node anymore !  We want to start
                // ascend from the parent, which will update the deadend node
                // properly
                NodeDescendResult::DeadEnd => {
                    // If current_pos is None, then the root is a deadend
                    // (meaning that all subbranches have been either pruned or
                    // explored), therefore we have finished our search. Else we
                    // found the deadend elsewhere and node is necessarily a
                    // child
                    if let Some(pos) = current_pos {
                        return DescendResult::DeadEnd(pos, parent_stack);
                    } else {return DescendResult::Finished;}
                }
            }
        }
        search_node_lock = next_node;
    }
}

/// Handles the descend from a candidate and returns an appropriate DescendResult
fn handle_montecarlo_descend<'a>(config: &BanditConfig,
                                 context: &Context,
                                 cand: Candidate<'a>,
                                 pos: usize,
                                 cut: f64,
                                 parent_stack: NodeStack<'a>) -> DescendResult<'a> {
    let order = config.new_nodes_order;
    if let Some(cand) = montecarlo::descend(order, context, cand, cut) {
        DescendResult::MonteCarloLeaf(cand, pos, parent_stack)
    } else { DescendResult::FailedMonteCarlo }
}


/// Called when commiting an evaluation, loop on results retrieved from ascend_node and ascend
/// node_no_val call the corresponding function to update the tree.
fn iter_ascend<'a>(node_arc: Arc<RwLock<Node<'a>>>,
                   val: f64,
                   current_pos: Option<usize>,
                   pos_last_child: usize,
                   parent_stack: &mut NodeStack<'a>,
                   last_child_completed: bool) {
    let mut is_val_inserted = true;
    let mut node_arc = node_arc;
    let mut completed = last_child_completed;
    let mut pos_child = pos_last_child;
    let mut pos_parent = current_pos;
    loop {
        let node_res = if is_val_inserted {
            node_arc.write().unwrap().ascend_node(parent_stack, pos_child, completed, val)
        } else {
            node_arc.write().unwrap().ascend_node_no_val(parent_stack, pos_child)
        };
        match node_res {
            NodeAscendResult::Node(parent_arc, pos_opt) => {
                // If we are here, then the node we just called ascend_node on
                // is itself a child, therefore pos_parent should be Some
                pos_child = pos_parent.expect("parent is not a child 0");
                pos_parent = pos_opt;
                node_arc = parent_arc;
            }
            NodeAscendResult::NodeNoVal(parent_arc, pos_opt) => {
                is_val_inserted = false;
                pos_child = pos_parent.expect("parent is not a child 1");
                pos_parent = pos_opt;
                node_arc = parent_arc;
            }
            NodeAscendResult::NodeNotCompleted(parent_arc, pos_opt) => {
                completed = false;
                pos_child = pos_parent.expect("parent is not a child 2");
                pos_parent = pos_opt;
                node_arc = parent_arc;
            }
            NodeAscendResult::InvalidParent |
            NodeAscendResult::TreeUpdated |
            NodeAscendResult::Root => return,
        }
    }
}

/// We are coming from a deadend, so we did not retrieve any value here
/// We still want to update the tree for completed children
/// Takes a parent stack (of weak pointers) and iterates on it
pub fn thread_ascend_tree_no_val(pos_last_child: usize, 
                                 mut parent_stack: NodeStack) {
    let parent_opt = parent_stack.pop();
    if let Some((weak_node, pos_opt)) = parent_opt {
        if let Some(node_lock) = weak_node.upgrade() {
            iter_ascend_no_val(node_lock, pos_opt, pos_last_child, &mut parent_stack)
        }
    }
}

/// Called by thread_ascend_tree_no_val
/// Iterates on the values retrieved from ascend_node_no_val
fn iter_ascend_no_val<'a>(node_lock: Arc<RwLock<Node<'a>>>,
                          current_pos: Option<usize>,
                          pos_last_child: usize,
                          parent_stack: &mut NodeStack<'a>) {
    let mut node_arc = node_lock;
    let mut pos_child = pos_last_child;
    let mut pos_parent = current_pos;
    loop {
        let node_res;
        {
            node_res = node_arc.write().unwrap().ascend_node_no_val(parent_stack, pos_child);
        }
        match node_res {
            NodeAscendResult::NodeNoVal(parent_arc, pos_opt) => {
                pos_child = pos_parent.expect("IN NO VAL, parent is not a child !");
                pos_parent = pos_opt;
                node_arc = parent_arc;
            }
            NodeAscendResult::Node(..)  => { panic!("ascend_no_val returned Node"); }
            _ => return,
        }
    }
}



impl<'a> Node<'a> {
    /// Called on a node, returns None if we find a childless node, else returns the next
    /// node to visit. `best_time` is the time of the best candidate at the time this
    /// thread started traversing the tree.
    fn descend_node(&mut self, context: &Context, config: &BanditConfig, best_time: f64)
        -> NodeDescendResult<'a>
    {
        if self.children.is_empty() {
            panic!("We should never have an empty node");
        }
        // removing the children whose bounds are over the current best score
        self.remove_children(best_time);
        if let Some(un_pos) = self.find_unexpanded_node(config, best_time) {
            self.expand_child(context, config, un_pos, best_time)
        } else { self.descend_expanded_node(config, best_time) }
    }

    fn find_unexpanded_node(&self, config: &BanditConfig, best_time: f64) -> Option<usize> {
      match config.new_nodes_order {
          NewNodeOrder::Api => self.find_standard_unexpanded_node(),
          NewNodeOrder::WeightedRandom => self.find_mixed_unexpanded_node(best_time),
          NewNodeOrder::Random => self.find_rand_unexpanded_node(),
          NewNodeOrder::Bound => self.find_best_unexpanded_node(),
        }
    }


    /// Returns the position of an unexpanded node if there is some, else returns None
    fn find_standard_unexpanded_node(&self) -> Option<usize> {
        self.children.iter().enumerate()
            .find(|&(_, x)| match *x {
                SubTree::UnexpandedNode(..) =>  {true}
                _ => {false}})
            .map( |x| x.0)
    }

    /// Returns the position of the unexpanded node with the best bound if there is some, else
    /// returns None
    fn find_best_unexpanded_node(&self) -> Option<usize> {
        self.children.iter().enumerate()
            .filter(|&(_, x)| match *x {
                SubTree::UnexpandedNode(..) =>  {true}
                _ => {false}})
            .min_by(|x1, x2| cmp_f64(x1.1.bound(), x2.1.bound()))
            .map( |x| x.0)
    }


    /// Returns the position of an randomly chosen unexpanded node if there is some, else returns
    /// None
    fn find_rand_unexpanded_node(&self) -> Option<usize> {
        let mut rng = thread_rng();
        let unexpanded_index_list = self.children.iter().enumerate()
            .filter(|&(_, x)| match *x {
                SubTree::UnexpandedNode(..) =>  {true}
                _ => {false}})
            .map( |x| x.0).collect::<Vec<_>>();
        if unexpanded_index_list.len() == 0 {
            None
        }
        else {
            let rand_ind = rng.gen_range(0, unexpanded_index_list.len());
            Some(unexpanded_index_list[rand_ind])
        }
    }


    /// Returns the position of an randomly chosen unexpanded node if there is some, else returns
    /// None
    fn find_mixed_unexpanded_node(&self, best_score: f64) -> Option<usize> {
        let unexpanded_node_list = self.children.iter().enumerate()
            .filter(|&(_, x)| match *x {
                SubTree::UnexpandedNode(..) =>  {true}
                _ => {false}})
            .collect::<Vec<_>>();
        if unexpanded_node_list.len() == 0 {
            None
        }
        else {
            let ind = Node::select_node(unexpanded_node_list, best_score);
            Some(ind)
        }
    }

    /// Given a vect of (index, node) tuples and a - possibly infinite - best
    /// time, we choose a node according to a mixed strategy : we make a weighted
    /// random selection (weights are correlated to the distance between the
    /// bound and the best time)
    fn select_node(node_list: Vec<(usize, &SubTree)>,
        best_score: f64) -> usize {
        if node_list.is_empty() {
            panic!("not supposed to have an empty vec");
        }
        let mut weighted_items = vec![];
        let mut rng = thread_rng();
        let max_bound = node_list.iter().max_by(|x1, x2| cmp_f64(x1.1.bound(), x2.1.bound()))
            .map(|x| x.1.bound()).unwrap();
        for (ind, x) in node_list {
            if best_score.is_infinite() {
                let x_weight = std::cmp::max(1, (10f64 * max_bound / x.bound()).floor() as u32);
                weighted_items.push(Weighted{weight: x_weight, item: ind});
            } else {
                assert!(x.bound() <= best_score);
                let weight = (1000f64 * (1f64 - x.bound()/best_score)).floor() as u32;
                let weight = std::cmp::max(1, weight);
                weighted_items.push(Weighted { weight, item: ind });
            }
        }
        WeightedChoice::new(&mut weighted_items).ind_sample(&mut rng)
    }


    /// Called in descend_node
    /// We know that self contains no unexpanded node
    /// Find a suitable child, treat it and returns a NodeDescendResult
    fn descend_expanded_node(&mut self, config: &BanditConfig, best_time: f64) ->
      NodeDescendResult<'a> {
        let ind_opt = self.decide_next_child(config, best_time);
        if let Some(index) = ind_opt {
            self.rewards[index].1 += 1;
            match self.children[index] {
                SubTree::InternalNode(ref arc_node, _) => { NodeDescendResult::Node(
                    Arc::clone(&arc_node), index)}
                SubTree::UnexpandedNode(..) => {panic!("Found an unexpanded node");}
                SubTree::Empty => {panic!("Found a NoGo");}
            }
        } else { NodeDescendResult::DeadEnd }
    }


    /// Returns which child will be visited next by dispatching the real work according to config
    /// Returns None if children is empty
    fn decide_next_child(&self, config: &BanditConfig, best_time: f64) -> Option<usize> {
      match config.old_nodes_order {
        OldNodeOrder::Bandit => self.decide_next_child_bandit_arm(config),
        OldNodeOrder::Bound => self.decide_next_child_best(),
        OldNodeOrder::WeightedRandom => self.decide_next_child_mixed(best_time),
      }
    }

    /// Returns the index of the child with the minimum bound
    /// Returns None if children is empty
    fn decide_next_child_best(&self) -> Option<usize> {
      self.children.iter().enumerate().filter(|&(_, x)| {
          if let SubTree::Empty = *x {false} else {true}
      }).min_by(|x: &(usize, &SubTree), y:&(usize, &SubTree)| {
          cmp_f64(x.1.bound(), y.1.bound())
      }).map(|x| x.0)
    }

    /// Decides which child will be next according to a random algorithm weighted with the bounds
    /// of the children. Returns its index
    /// Returns None if children is empty
    fn decide_next_child_mixed(&self, best_score: f64)
        -> Option<usize>
    {
        let node_list = self.children.iter().enumerate()
            .filter(|&(_, x)| match *x {
                SubTree::Empty =>  {false}
                _ => {true}})
            .collect::<Vec<_>>();
        if node_list.len() == 0 {
            None
        }
        else {
            let ind = Node::select_node(node_list, best_score);
            Some(ind)
        }
    }


    /// Given the list of rewards, returns the index of the next
    /// child to go - or None if the list is empty
    fn decide_next_child_bandit_arm(&self, config: &BanditConfig) -> Option<usize> {
        assert_eq!(self.children.len(), self.rewards.len());
        let nb_tested = self.rewards.iter().fold(0, |acc, ref x| acc + x.1);
        let nb_children = self.rewards.len();
        self.rewards.iter().enumerate()
            .filter(|&(i, _)| {if let SubTree::Empty = self.children[i] {false}
                else {true}})
            .map(|(ind, ref x)| (ind, heval(config, x.0.len(), x.1, nb_tested, nb_children)))
            .max_by( |x1:&(usize, f64), x2:&(usize, f64)| cmp_f64(x1.1, x2.1))
            .map( |(ind, _)| ind)
    }

    /// "Remove" (that is, replace with a NoGo variant) children whose bounds are higher than best
    /// candidates score
    // FIXME: this is redundant with trim
    fn remove_children(&mut self, best_score: f64) -> usize {
        let mut to_remove = vec![];
        for (ind, child) in self.children.iter().enumerate() {
            match *child {
                SubTree::InternalNode(_, ref bound) => {
                    if *bound >= best_score { to_remove.push(ind); }
                }
                SubTree::UnexpandedNode(ref cand) => {
                    if cand.bound.value() >= best_score { to_remove.push(ind); }
                }
                SubTree::Empty => {}
            }
        }
        let nb_removed = to_remove.len();
        for ind in to_remove {
            self.children[ind] = SubTree::Empty;
        }
        nb_removed
    }

    /// Given a vector of children and pos, which is the position of an unexpanded node, this
    /// function expands the node, remove the child if it is a deadend or replace the
    /// unexpanded child with the expanded one
    fn expand_child(&mut self, context: &Context,
                    config: &BanditConfig,
                    pos: usize, cut: f64) -> NodeDescendResult<'a> {
        match self.children[pos].expand_node(context, cut) {
            ExpandRes::DeadEnd => {
                self.children[pos] = SubTree::Empty;
                NodeDescendResult::DeadEndFromExpand(pos)
            }
            ExpandRes::Leaf(candidate) => {
                self.children[pos] = SubTree::Empty;
                NodeDescendResult::Leaf(candidate, pos)
            }
            ExpandRes::Node(node) => {
                self.children[pos] = node;
                if let SubTree::InternalNode(ref node_arc, _) = self.children[pos] {
                    if config.monte_carlo {
                        NodeDescendResult::MonteCarlo(Arc::clone(node_arc), pos)
                    } else {NodeDescendResult::Node(Arc::clone(node_arc), pos)}
                } else { panic!("We should have a Node here"); }
            }
        }
    }

    /// pop the stack - which gives the parent of the node self, update the parent
    /// and returns it
    fn ascend_node(&mut self, parents_stack: &mut NodeStack<'a>, pos_child: usize,
                   child_completed: bool,  val: f64) -> NodeAscendResult<'a> {
        self.remove_children(val);
        match self.update_as_parent(val, pos_child, child_completed) {
            UpdateRes::KeepGoing => self.pop_parent(parents_stack, true),
            UpdateRes::ValNotInserted =>  self.pop_parent(parents_stack, false),
            UpdateRes::TreeUpdated => NodeAscendResult::TreeUpdated
        }
    }



    /// Update rewards in a node with val, also replace child with NoGo if needed
    fn update_as_parent(&mut self, val: f64, pos: usize, completed: bool) -> UpdateRes {
        if completed {
            self.children[pos] = SubTree::Empty;
        }
        let is_val_inserted = self.update_rewards(pos, val);
        match (is_val_inserted, completed) {
            (false, false) => UpdateRes::TreeUpdated,
            (false, true) => UpdateRes::ValNotInserted,
            (_, _) => UpdateRes::KeepGoing
        }
    }

    /// Basically the same function that ascend_node, besides the fact that we know that there
    /// is no need to update the rewards as we come from a deadend, we just need to update the
    /// completed branches
    fn ascend_node_no_val(&mut self, parents_stack: &mut NodeStack<'a>, pos_child: usize)
        -> NodeAscendResult<'a> {
        // If we call this function, we know that our child is complete (this is this function
        // only purpose)
        self.children[pos_child] = SubTree::Empty;
        self.pop_parent(parents_stack, false)
    }

    /// Take a parents_stack, pop it and returns the parent node -if there is one - in a
    /// NodeAscendResult
    fn pop_parent(&mut self, parents_stack: &mut NodeStack<'a>, val_inserted: bool)
        -> NodeAscendResult<'a>
    {
        let stack_opt = parents_stack.pop();
        if let Some((parent_ref, pos)) = stack_opt {
            let parent_opt = parent_ref.upgrade();
            if let Some(ref parent_lock) = parent_opt {
                let completed = self.children.iter().fold(true, |acc, ref x| if let
                                                          SubTree::Empty =
                                                          *x {acc && true} else {false});
                match (val_inserted, completed) {
                    (true, true) => NodeAscendResult::Node(Arc::clone(parent_lock), pos),
                    (false, true) => NodeAscendResult::NodeNoVal(Arc::clone(parent_lock), pos),
                    (true, false) => NodeAscendResult::NodeNotCompleted(Arc::clone(parent_lock), pos),
                    (false, false) => NodeAscendResult::TreeUpdated
                }
            } else { NodeAscendResult::InvalidParent }
        } else { NodeAscendResult::Root }
    }


    /// Update a rewards list given a new value and the position where it was found
    /// returns true if the value was inserted in the node
    fn update_rewards(&mut self, pos: usize, val: f64) -> bool {
        let total_trials = self.rewards.iter().map(|x| x.0.len()).sum::<usize>();
        // If total trials is less than THRESHOLD, then we simply push our new value
        // in the node where it was found.
        if total_trials < 10 { // FIXME: use the real threshold
            self.rewards[pos].0.push(val);
            true
        } else {
            // Now we have to find the minimum value of all vectors and the place where
            // we found it.
            let (ind, int_ind, min) = unwrap!(self.find_max_rewards());
            if val < min {
                self.rewards[ind].0.swap_remove(int_ind);
                self.rewards[pos].0.push(val);
                true
            } else { false }
        }
    }

    /// Returns the tuple (outer_index, int_index, max) which is the maximum value found
    /// in rewards with the indexes where we found it - index on outer dimension, then on
    /// inner.
    fn find_max_rewards(&self) -> Option<(usize, usize, f64)> {
        self.rewards.iter().enumerate().flat_map(|(out_idx, rewards)| {
            let max = rewards.0.iter().cloned().enumerate()
                .max_by(|lhs, rhs| cmp_f64(lhs.1, rhs.1));
            max.map(|(idx, value)| (out_idx, idx, value))
        }).max_by(|x1, x2| cmp_f64(x1.2, x2.2))
    }

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
            ind = unwrap!(montecarlo::next_cand_index(
                    config.new_nodes_order, new_nodes, cut));
            let cand_ref = if let SubTree::UnexpandedNode(ref cand) = self.children[ind]
            {
                cand
            } else { panic!() };
            let choice_opt = choice::list(&cand_ref.space).next();
            if let Some(choice) = choice_opt {
                let new_nodes = cand_ref.apply_choice(context, choice).into_iter()
                    .filter(|x| x.bound.value() < cut)
                    .collect_vec();
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

// FIXME: merge with other impl
impl<'a> SubTree<'a> {
    /// Given an expanded Node, returns a list of candidates corresponding to the choices
    /// applied to this candidate. If the list is empty, it means that no choice can be
    /// applied and this node is therefore a leaf in the tree, it has been completely
    /// constrained and must now be evaluated.
    fn expand_node(&mut self, context: &Context, cut: f64) -> ExpandRes<'a>  {
        // FIXME: should reassign inside
        let node = std::mem::replace(self, SubTree::Empty);
        if let SubTree::UnexpandedNode(candidate) = node {
            let choice_opt = choice::list(&candidate.space).next();
            if let Some(choice) = choice_opt {
                let candidates = candidate.apply_choice(context, choice);
                let new_tree = SubTree::from_candidates(candidates, cut);
                if new_tree.is_empty() {
                    ExpandRes::DeadEnd
                } else { ExpandRes::Node(new_tree) }
            } else { ExpandRes::Leaf(candidate) }
        } else { panic!("Trying to expand an already expanded node !!!");}
    }
}

// FIXME: <<<<<<<<<<<<<<<<

/// gives a "score" to a branch of the tree at a given node n_successes is the number of
/// successes of that branch (that is, the number of leaves that belong to the THRESHOLD
/// best of that node and which come from that particular branch).
/// * `n_branch_trials` is the number of trials of that branch (both failed and succeeded),
/// * `n_trials` is  the number of trials of the node and k the number of branches in the
///   node.
fn heval(config: &BanditConfig,
         n_successes: usize
         n_branch_trials: usize,
         n_trials: usize,
         n_branches: usize) -> f64 {
    if n_trials == 0 { std::f64::INFINITY } else {
        let f = (n_trials * n_branches) as f64;
        let alpha= f.ln() / config.delta;
        let sqrt_body = alpha * (2. * n_successes as f64 + alpha);
        (n_successes as f64 + alpha + sqrt_body.sqrt()) / n_branch_trials as f64
    }
}
