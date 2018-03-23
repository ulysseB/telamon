//! Exploration of the search space.

use device::Context;
use std;
use std::f64;
use std::sync::{ Weak, Arc, RwLock};

use utils::*;
use itertools::Itertools;

use rand::{Rng, thread_rng};
use rand::distributions::{ Weighted, WeightedChoice, IndependentSample};

use explorer::candidate::Candidate;
use explorer::choice;

use explorer::config::{Config, BanditConfig, NewNodeOrder, OldNodeOrder};



use  explorer::store::Store;
pub struct SafeTree<'a, 'b, 'c> {
    shared_tree: Arc<RwLock<SearchTree<'a, 'b>>>,
    cut: RwLock<f64>,
    config: &'c BanditConfig,
}

impl<'a, 'b, 'c> SafeTree<'a, 'b, 'c> {
    pub fn new(tree: SearchTree<'a, 'b>, config: &'c BanditConfig) -> Self {
        SafeTree {
            shared_tree: Arc::new(RwLock::new(tree)), 
            cut: RwLock::new(std::f64::INFINITY), 
            config,
        }
    }
}

impl<'a, 'b, 'c> Store<'a> for SafeTree<'a, 'b, 'c> {
    type PayLoad = (NodeStack<'a, 'b>, usize);

    fn update_cut(&self, new_cut: f64) {
        let mut cut = self.cut.write().unwrap();
        *cut = new_cut;
        let node_root; 
        {
            let root = self.shared_tree.read().unwrap();
            if let EnumTree::NoGoBranch = root.tree {
                return;
            }
            node_root = match &root.tree {
                &EnumTree::Node(ref n, _) => Arc::clone(n), 
                _ => panic!("At this point, root should be a node"),
            };
            let root_completed =  node_root.read().unwrap().children.iter()
                .fold(true, |acc, x| {
                    if let EnumTree::NoGoBranch = x.tree {acc && true} else {false}
                });
            if root_completed { return;};
        }
        warn!("Starting a collection");
        collect_descend( node_root, new_cut);
    }

    fn commit_evaluation(&self, _config: &Config, payload: Self::PayLoad, eval: f64) {
        let (ascend_stack, pos_last_child) = payload;
        thread_ascend_tree(eval, pos_last_child, ascend_stack);
    }

    fn explore(&self, config: &Config,  context: &Context) -> Option<(Candidate<'a>, Self::PayLoad)> {
        loop {
            match thread_descend_tree(config, self.config, context, Arc::clone(&self.shared_tree), &self.cut) {
                DescendResult::Finished => { return None; }
                DescendResult::DeadEnd(pos, parent_stack) => {
                    thread_ascend_tree_no_val(pos, parent_stack);
                }
                DescendResult::Leaf(cand, pos, parent_stack) => {
                    return Some((cand, (parent_stack, pos)));
                }
                DescendResult::MonteCarloLeaf(cand, pos, parent_stack) => {
                    return Some((cand, (parent_stack, pos)));
                }
                DescendResult::FailedMonteCarlo => {}
            }
        }
    }
}

type NodeRewards =  Vec<(Vec<f64>, usize)>;

pub struct Node<'a, 'b> { 
    children: Vec<SearchTree<'a, 'b>>, 
    rewards: NodeRewards, 
    id: String
}

/// The search tree that will be traversed
pub enum EnumTree<'a, 'b> {
    Node(Arc<RwLock<Node<'a, 'b>>>, f64 ),
    UnexpandedNode(Candidate<'a>, f64),
    NoGoBranch,
}

const THRESHOLD : usize = 10;
pub struct SearchTree<'a, 'b> { pub tree: EnumTree<'a, 'b>,  context: &'b Context<'b> }

type NodeStack<'a, 'b> = Vec<(Weak<RwLock<Node<'a, 'b>>>, Option<usize>)>;


// These types are used as return type for the functions traversing the tree
pub enum DescendResult<'a, 'b> {
    Finished,
    DeadEnd(usize, NodeStack<'a, 'b>),
    Leaf(Candidate<'a>, usize, NodeStack<'a, 'b>),
    MonteCarloLeaf(Candidate<'a>, usize, NodeStack<'a, 'b>),
    FailedMonteCarlo,
}

pub enum AscendResult {
    InvalidParent,
    TreeUpdated,
    Root,
}

enum NodeDescendResult<'a, 'b> {
    Node(Arc<RwLock<Node<'a, 'b>>>, usize),
    MonteCarlo(Arc<RwLock<Node<'a, 'b>>>, usize),
    Leaf(Candidate<'a>, usize),
    DeadEndFromExpand(usize),
    DeadEnd,
}

pub enum NodeAscendResult<'a, 'b> {
    Node(Arc<RwLock<Node<'a, 'b>>>, Option<usize>),
    NodeNoVal(Arc<RwLock<Node<'a, 'b>>>, Option<usize>),
    NodeNotCompleted(Arc<RwLock<Node<'a, 'b>>>, Option<usize>),
    InvalidParent,
    TreeUpdated,
    Root,
}

enum ExpandRes<'a, 'b> {
    DeadEnd,
    Node(SearchTree<'a, 'b>),
    Leaf(Candidate<'a>),
}

enum UpdateRes {
    KeepGoing,
    ValNotInserted,
    TreeUpdated,
}


/// Call descend_node in a loop until it finds either a DeadEnd or a Leaf
/// if the deadend is found at the root, then the return value states this fact
pub fn thread_descend_tree<'a, 'b>(
    // TODO(cleanup): remove the dependency on Config
    _config: &Config,
    bandit_config: &BanditConfig,
    context: &Context,
    root_lock: Arc<RwLock<SearchTree<'a, 'b>>>, 
    best_val: &RwLock<f64>) -> DescendResult<'a, 'b>
{
    debug!("IN DESCEND TREE");
    let node_root;
    {
        let root = root_lock.read().unwrap();
        if let EnumTree::NoGoBranch = root.tree {
          return DescendResult::Finished;
        }
        node_root = match &root.tree {
            &EnumTree::Node(ref n, _) => Arc::clone(n), 
                _ => {panic!("At this point, root should be a node");}
        };
    }
    let best_val = *best_val.read().unwrap();
    iter_descend(bandit_config, context, node_root, best_val)
}


/// Called in thread_descend_tree, iter on the value of descend_node
/// Builds the parents stack and returns an appropriate value at the end
fn iter_descend<'a, 'b>(config: &BanditConfig,
                        context: &Context,
                        node_root: Arc<RwLock<Node<'a, 'b>>>,
                        best_val: f64) -> DescendResult<'a, 'b> {
    debug!("IN ITER DESCEND");
    let mut parent_stack = vec![];
    let mut search_node_lock = node_root;
    let mut current_pos = None;
    loop {
        let next_node;
        let mut montecarlo_candidate = None;
        {
            let mut search_node = search_node_lock.write().unwrap();
            match search_node.descend_node(config, best_val) {
                NodeDescendResult::Node(subtree_arc, pos) => {
                    let weak_ref = Arc::downgrade(&search_node_lock);
                    parent_stack.push((weak_ref, current_pos));
                    current_pos = Some(pos);
                    next_node = Arc::clone(&subtree_arc);
                }
                NodeDescendResult::MonteCarlo(subtree_arc, pos) => {
                    let weak_ref = Arc::downgrade(&search_node_lock);
                    next_node = Arc::clone(&subtree_arc);
                    parent_stack.push((weak_ref, current_pos));
                    montecarlo_candidate = Some((subtree_arc.write().unwrap()
                        .start_montecarlo(config, context, best_val).unwrap(), pos));
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
        // A bit weird, but we do that so we don't have to hold the lock while descending on
        // candidate
        if let Some((cand, pos)) = montecarlo_candidate {
            return handle_montecarlo_descend(config, context, cand, pos, best_val, parent_stack);
        }
    }
}

fn handle_montecarlo_descend<'a, 'b>(config: &BanditConfig, 
                  context: &Context, 
                  cand: Candidate<'a>, 
                  pos: usize, 
                  cut: f64,
                  parent_stack: NodeStack<'a, 'b>) -> DescendResult<'a, 'b> {
    if let Some(cand) = montecarlo_descend(config, context, cand, cut) {
        return DescendResult::MonteCarloLeaf(
            cand,
            pos,
            parent_stack);
    }
    else { return DescendResult::FailedMonteCarlo;}

}

///  iter on all nodes in the tree and call node_collect on them
fn collect_descend<'a, 'b>(node_root: Arc<RwLock<Node<'a, 'b>>>, best_val: f64) {
    debug!("IN COLLECT TREE");
    let mut node_stack = vec![node_root];
    let mut collected_nodes = 0;
    let mut visited_nodes = 1;
    while let Some(search_node_lock) = node_stack.pop() {
        let mut search_node = search_node_lock.write().unwrap();
        let (node_children, nb_removed) = search_node.node_collect(best_val);
        collected_nodes += nb_removed;
        visited_nodes += node_children.len();
        node_stack.extend(node_children);
    }
    warn!("Removed {} nodes, visited {}", collected_nodes, visited_nodes);
}


/// Take a node stack and a score as argument and update the tree from leaf to root until there is
/// no information to be passed
pub fn thread_ascend_tree<'a, 'b>(val: f64, 
                                  pos_last_child: usize, 
                                  mut parent_stack: NodeStack<'a, 'b>)
    -> AscendResult 
{
    debug!("IN ASCEND TREE");
    let parent_opt = parent_stack.pop();
    if let Some((weak_node, pos)) = parent_opt {
        if let Some(node_lock) = weak_node.upgrade() {
            iter_ascend(node_lock, val, pos, pos_last_child, &mut parent_stack)
        } else { AscendResult::InvalidParent }
    } else { AscendResult::Root }
}

/// Called by thread_ascend_tree, loop on results retrieved from ascend_node and ascend_node_no_val
/// call the corresponding function to update the tree
fn iter_ascend<'a, 'b>(node_arc: Arc<RwLock<Node<'a, 'b>>>,
                       val: f64,
                       current_pos: Option<usize>,
                       pos_last_child: usize,
                       parent_stack: &mut NodeStack<'a, 'b>) -> AscendResult {
    debug!("IN ITER ASCEND");
    let mut is_val_inserted = true;
    let mut node_arc = node_arc;
    let mut completed = true;
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
            NodeAscendResult::InvalidParent => {
                return AscendResult::InvalidParent;
            }
            NodeAscendResult::TreeUpdated => {
                return AscendResult::TreeUpdated;
            }
            NodeAscendResult::Root => {
                return AscendResult::Root;
            }
        }
    }
}

/// We are coming from a deadend, so we did not retrieve any value here
/// We still want to update the tree for completed children
/// Takes a parent stack (of weak pointers) and iterates on it
pub fn thread_ascend_tree_no_val<'a, 'b>(
    pos_last_child: usize,  mut parent_stack: NodeStack<'a, 'b>) -> AscendResult
{
    debug!("IN ASCEND TREE NO VAL");
    let parent_opt = parent_stack.pop();
    if let Some((weak_node, pos_opt)) = parent_opt {
        if let Some(node_lock) = weak_node.upgrade() {
            iter_ascend_no_val(node_lock, pos_opt, pos_last_child, &mut parent_stack)
        }
        else { AscendResult::InvalidParent }
    } else { AscendResult::Root }
}

/// Called by thread_ascend_tree_no_val
/// Iterates on the values retrieved from ascend_node_no_val
fn iter_ascend_no_val<'a, 'b>(node_lock: Arc<RwLock<Node<'a, 'b>>>,
                              current_pos: Option<usize>, 
                              pos_last_child: usize,
                              parent_stack: &mut NodeStack<'a, 'b>) -> AscendResult {
    debug!("IN ITER ASCEND NO VAL");
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
            NodeAscendResult::Node(..)  => {panic!("ascend_no_val returned Node");}
            NodeAscendResult::InvalidParent => { return AscendResult::InvalidParent; }
            NodeAscendResult::NodeNotCompleted(..) => { return AscendResult::TreeUpdated; }
            NodeAscendResult::TreeUpdated => { return AscendResult::TreeUpdated; }
            NodeAscendResult::Root => { return AscendResult::Root; }
        }
    }
}



impl<'a, 'b> Node<'a, 'b> {
    /// Called on a node, returns None if we find a childless node, else returns the next node to
    /// visit.
    /// best_time is the time of the best candidate at the time this thread started traversing the
    /// tree
    fn descend_node(&mut self, config: &BanditConfig, best_time: f64)
        -> NodeDescendResult<'a, 'b>
    {
        debug!("IN DESCEND NODE on node {}: {} children", self.id, self.children.len());
        if self.children.is_empty() {
            panic!("We should never have an empty node");
        } 
        // removing the children whose bounds are over the current best score
        self.remove_children(best_time);
        if let Some(un_pos) = self.find_unexpanded_node(config, best_time) {
            self.expand_child(config, un_pos)
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
            .find(|&(_, x)| match x.tree { 
                EnumTree::UnexpandedNode(..) =>  {true} 
                _ => {false}})
            .map( |x| x.0)
    }
    
    /// Returns the position of the unexpanded node with the best bound if there is some, else
    /// returns None
    fn find_best_unexpanded_node(&self) -> Option<usize> {
        self.children.iter().enumerate()
            .filter(|&(_, x)| match x.tree { 
                EnumTree::UnexpandedNode(..) =>  {true} 
                _ => {false}})
            .min_by(|x1, x2| SearchTree::compare_bound(x1.1, x2.1))
            .map( |x| x.0)
    }
    
    
    /// Returns the position of an randomly chosen unexpanded node if there is some, else returns
    /// None
    fn find_rand_unexpanded_node(&self) -> Option<usize> {
        let mut rng = thread_rng();
        let unexpanded_index_list = self.children.iter().enumerate()
            .filter(|&(_, x)| match x.tree { 
                EnumTree::UnexpandedNode(..) =>  {true} 
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
            .filter(|&(_, x)| match x.tree { 
                EnumTree::UnexpandedNode(..) =>  {true} 
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
    fn select_node(node_list: Vec<(usize, &SearchTree)>,
        best_score: f64) -> usize {
        if node_list.is_empty() {
            panic!("not supposed to have an empty vec");
        }
        let mut weighted_items = vec![];
        let mut rng = thread_rng();
        let max_bound = node_list.iter().max_by(|x1, x2|
            SearchTree::compare_bound(x1.1, x2.1))
            .map(|x| x.1.bound()).unwrap();
        for (ind, x) in node_list {
            if best_score.is_infinite() {
                let x_weight = (10f64 * max_bound / x.bound()).floor() as u32 ;
                weighted_items.push(Weighted{weight: x_weight, item: ind});
            } else {
                assert!(x.bound() <= best_score);
                let weight = (1000f64 * (1f64 - x.bound()/best_score)).floor() as u32;
                let weight = std::cmp::max(1, weight);
                weighted_items.push(Weighted { weight, item: ind });
            }
        }
        let ind = WeightedChoice::new(&mut weighted_items).ind_sample(&mut rng);
        debug!("Chosen ind: {}", ind);
        ind
    }


    /// Called in descend_node
    /// We know that self contains no unexpanded node
    /// Find a suitable child, treat it and returns a NodeDescendResult
    fn descend_expanded_node(&mut self, config: &BanditConfig, best_time: f64) ->
      NodeDescendResult<'a, 'b> {
        debug!("IN DESCEND EXPANDED NODE on node {}: {} children", 
            self.id, self.children.len());
        let ind_opt = self.decide_next_child(config, best_time);
        if let Some(index) = ind_opt {
            self.rewards[index].1 += 1;
            match self.children[index].tree {
                EnumTree::Node(ref arc_node, _) => { NodeDescendResult::Node( 
                    Arc::clone(&arc_node), index)}
                EnumTree::UnexpandedNode(..) => {panic!("Found an unexpanded node");}
                EnumTree::NoGoBranch => {panic!("Found a NoGo");}
            }
        } else { NodeDescendResult::DeadEnd }
    }


    fn decide_next_child(&self, config: &BanditConfig, best_time: f64) -> Option<usize> {
      match config.old_nodes_order {
        OldNodeOrder::Bandit => self.decide_next_child_bandit_arm(config),
        OldNodeOrder::Bound => self.decide_next_child_best(),
        OldNodeOrder::WeightedRandom => self.decide_next_child_mixed(best_time),
      }
    }

    fn decide_next_child_best(&self) -> Option<usize> {
      self.children.iter().enumerate().filter(|&(_, x)| {
          if let EnumTree::NoGoBranch = x.tree {false} else {true}
      }).min_by(|x: &(usize, &SearchTree), y:&(usize, &SearchTree)| {
          cmp_f64(x.1.bound(), y.1.bound())
      }).map(|x| x.0)
    }

    fn decide_next_child_mixed(&self, best_score: f64)
        -> Option<usize>
    {
        let node_list = self.children.iter().enumerate()
            .filter(|&(_, x)| match x.tree { 
                EnumTree::NoGoBranch =>  {false} 
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
        debug!("IN DECIDE CHILD on node {}", self.id);
        assert_eq!(self.children.len(), self.rewards.len());
        let nb_tested = self.rewards.iter().fold(0, |acc, ref x| acc + x.1);
        let nb_children = self.rewards.len();
        self.rewards.iter().enumerate()
            .filter(|&(i, _)| {if let EnumTree::NoGoBranch = self.children[i].tree {false} 
                else {true}})
            .map(|(ind, ref x)| (ind, heval(config, x.0.len(), x.1, nb_tested, nb_children)))
            .max_by( |x1:&(usize, f64), x2:&(usize, f64)| cmp_f64(x1.1, x2.1))
            .map( |(ind, _)| ind)
    }

    /// "Remove" (that is, replace with a NoGo variant) children whose bounds are higher than best
    /// candidates score
    fn remove_children(&mut self, best_score: f64) -> usize {
        debug!("IN REMOVE CHILDREN on node {}", self.id);
        let mut to_remove = vec![];
        for (ind, child) in self.children.iter().enumerate() {
            match child.tree {
                EnumTree::Node(_, ref bound) => {
                    if *bound >= best_score { to_remove.push(ind); }
                }
                EnumTree::UnexpandedNode(_, ref bound) => {
                    if *bound >= best_score { to_remove.push(ind); }
                }
                EnumTree::NoGoBranch => {}
            }
        }
        let nb_removed = to_remove.len();
        for ind in to_remove {
            self.children[ind].tree = EnumTree::NoGoBranch;
        }
        nb_removed
    }

    /// Given a vector of children and pos, which is the position of an unexpanded node, this
    /// function expands the node, remove the child if it is a deadend or replace the
    /// unexpanded child with the expanded one
    fn expand_child(&mut self, config: &BanditConfig, pos: usize) -> NodeDescendResult<'a, 'b> {
        debug!("IN EXPAND CHILD on node {}", self.id);
        let mut id_child = self.id.clone();
        id_child.push_str(&String::from(" "));
        id_child.push_str(&pos.to_string());
        match self.children[pos].expand_node(id_child) {
            ExpandRes::DeadEnd => {
                self.children[pos].tree = EnumTree::NoGoBranch;
                NodeDescendResult::DeadEndFromExpand(pos)
            }
            ExpandRes::Leaf(candidate) => {
                self.children[pos].tree = EnumTree::NoGoBranch;
                NodeDescendResult::Leaf(candidate, pos)
            }
            ExpandRes::Node(node) => {
                self.children[pos] = node;
                if let &EnumTree::Node(ref node_arc, _) = &self.children[pos].tree {
                    if config.monte_carlo {
                     NodeDescendResult::Node(Arc::clone(node_arc), pos)
                    } else { NodeDescendResult::MonteCarlo(Arc::clone(node_arc), pos)}
                } else { panic!("We should have a Node here"); }
            }
        } 
    }


    /// remove all childrens and returns a vector containing all node children
    fn node_collect(&mut self, best_val: f64) -> (Vec<Arc<RwLock<Node<'a, 'b>>>>, usize) {
        let nb_removed = self.remove_children(best_val);
        (self.children.iter().filter(|x| if let EnumTree::Node(..) = x.tree {true} else {false})
            .map(|x| if let EnumTree::Node(ref node, _) = x.tree {Arc::clone(node)} 
                 else {panic!("We should only have nodes here");})
            .collect::<Vec<_>>(),
            nb_removed)
    }

    /// pop the stack - which gives the parent of the node self, update the parent 
    /// and returns it 
    fn ascend_node(&mut self, parents_stack: &mut NodeStack<'a, 'b>, pos_child: usize, 
                   child_completed: bool,  val: f64) -> NodeAscendResult<'a, 'b> {
        debug!("IN ASCEND NODE on node {}, {} children", self.id, self.children.len());
        self.remove_children(val);
        match self.update_as_parent(val, pos_child, child_completed) {
            UpdateRes::KeepGoing => self.pop_parent(parents_stack, true), 
            UpdateRes::ValNotInserted =>  self.pop_parent(parents_stack, false),
            UpdateRes::TreeUpdated => NodeAscendResult::TreeUpdated 
        }
    }



    /// Update rewards in a node with val, also replace child with NoGo if needed
    fn update_as_parent(&mut self, val: f64, pos: usize, completed: bool) -> UpdateRes {
        debug!("IN UPDATE AS PARENT on node {}, {} children", self.id, self.children.len());
        if completed {
            self.children[pos].tree = EnumTree::NoGoBranch;
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
    fn ascend_node_no_val(&mut self, parents_stack: &mut NodeStack<'a, 'b>, pos_child: usize) 
        -> NodeAscendResult<'a, 'b> {
        // If we call this function, we know that our child is complete (this is this function
        // only purpose)
        self.children[pos_child].tree = EnumTree::NoGoBranch;
        self.pop_parent(parents_stack, false)
    }

    /// Take a parents_stack, pop it and returns the parent node -if there is one - in a
    /// NodeAscendResult
    fn pop_parent(&mut self, parents_stack: &mut NodeStack<'a, 'b>, val_inserted: bool) 
        -> NodeAscendResult<'a, 'b>
    {
        let stack_opt = parents_stack.pop(); 
        if let Some((parent_ref, pos)) = stack_opt {
            let parent_opt = parent_ref.upgrade();
            if let Some(ref parent_lock) = parent_opt {
                let completed = self.children.iter().fold(true, |acc, ref x| if let
                                                          EnumTree::NoGoBranch =
                                                          x.tree {acc && true} else {false});
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
        debug!("IN UPDATE REWARDS on node {}", self.id);
        let total_trials = self.rewards.iter().fold( 0, |acc, x| acc + x.1);
        // If total trials is less than THRESHOLD, then we simply push our new value
        // in the node where it was found. 
        if total_trials < THRESHOLD {
            self.rewards[pos].0.push(val);
            true
        } else {
            // Now we have to find the minimum value of all vectors and the place where
            // we found it
            // We just iterate on the two level of vect and retain the minimum we find
            let min_elem = self.find_min_rewards();  
            if let Some((ind, int_ind, min)) = min_elem {
                if val > min {
                    self.rewards[ind].0.swap_remove(int_ind);
                    self.rewards[pos].0.push(val);
                    true
                } else { false }
            } else {
                // Very unlikely, but it is possible that all rewards lists are empty
                // in the case that we only encountered deadend in this node
                self.rewards[pos].0.push(val);
                true
            }
        }
    }

    /// Returns the tuple (outer_index, int_index, min) which is the minimum value found
    /// in rewards with the indexes where we found it - index on outer dimension, then on
    /// inner.
    fn find_min_rewards(&self) -> Option<(usize, usize, f64)> {
        self.rewards.iter().enumerate().filter(|&(_, elem)| !elem.0.is_empty())
            .map(|(out_ind, elem)|
                 (out_ind, 
                  elem.0.iter().enumerate()
                  .min_by(|x1, x2| cmp_f64(*x1.1, *x2.1))
                  .unwrap())
                )
            .map( |(out_ind, (in_ind, min_val))| (out_ind, in_ind, *min_val))
            .min_by(|x1, x2| cmp_f64(x1.2, x2.2))
    }

    /// We have a newly expanded node, we want to do a montecarlo descend on it
    fn start_montecarlo(&mut self, config: &BanditConfig, context: &Context, cut: f64) 
        -> Option<Candidate<'a>> 
    {
        let ind;
        {
            let new_nodes = self.children.iter()
                .map( |node| 
                      if let EnumTree::UnexpandedNode(ref cand, _) = node.tree 
                      {cand} else {panic!()}).collect_vec();
            ind = match config.new_nodes_order {
                NewNodeOrder::Api => 0,
                NewNodeOrder::WeightedRandom => choose_cand_weighted(&new_nodes, cut),
                NewNodeOrder::Bound => choose_cand_best(&new_nodes),
                NewNodeOrder::Random => choose_cand_rand(&new_nodes),
            };
            let cand_ref = if let EnumTree::UnexpandedNode(ref cand, _) = self.children[ind].tree
            {cand} else {panic!()};
            let choice_opt = choice::list(&cand_ref.space).next();
            if let Some(choice) = choice_opt {
                let new_nodes = cand_ref.apply_choice(context, choice);
                if new_nodes.is_empty() {
                    return None;
                } else { 
                    let chosen_candidate = choose_next_cand(config, new_nodes, cut);
                    return Some(chosen_candidate);
                } 
            }
    }
        let node = std::mem::replace(&mut self.children[ind].tree, EnumTree::NoGoBranch);
        if let EnumTree::UnexpandedNode(cand, _) = node {Some(cand)} else {panic!()}
    }
}


impl<'a, 'b> SearchTree<'a, 'b> {
    pub fn new(candidates: Vec<Candidate<'a>>, context: &'b Context<'b>) -> Self {
        SearchTree::create_node(candidates, context, 0.0, String::from("0"))
    }

    /// cut all work done on the tree in a safer way than just dropping it
    /// (could make some trouble to do so...)
    pub fn cut_work(&mut self) {
      self.tree = EnumTree::NoGoBranch;
    }

    /// Given an expanded Node, returns a list of candidates corresponding to the choices applied
    /// to this candidate.
    /// If the list is empty, it means that no choice can be applied and this node is therefore a
    /// leaf in the tree, it has been completely constrained and must now be evaluated
    fn expand_node(&mut self, id_node: String) -> ExpandRes<'a, 'b>  {
        debug!("IN EXPAND NODE");
        let node;
        {
            node = std::mem::replace(&mut self.tree, EnumTree::NoGoBranch);
        }
        if let EnumTree::UnexpandedNode(candidate, bound) = node {
            let choice_opt = choice::list(&candidate.space).next();
            if let Some(choice) = choice_opt {
                let new_nodes = candidate.apply_choice(self.context, choice);
                if new_nodes.is_empty() {
                    ExpandRes::DeadEnd
                } else { ExpandRes::Node(
                        SearchTree::create_node(new_nodes, self.context, bound, id_node))
                }
            } else { ExpandRes::Leaf(candidate) }
        } else { panic!("Trying to expand an already expanded node !!!");}
    }

    fn compare_bound(tree1: &SearchTree, tree2: &SearchTree) -> std::cmp::Ordering {
        match (&tree1.tree, &tree2.tree) {
            (&EnumTree::UnexpandedNode(_, b1), &EnumTree::UnexpandedNode(_, b2)) |
            (&EnumTree::UnexpandedNode(_, b1), &EnumTree::Node(_, b2)) |
            (&EnumTree::Node(_, b1), &EnumTree::UnexpandedNode(_, b2)) |
            (&EnumTree::Node(_, b1), &EnumTree::Node(_, b2)) => cmp_f64(b1, b2),
            (_, _) => {panic!("Can not compare bounds of NoGoBranch !");}
        }
    }

    fn bound(&self) -> f64 {
        match &self.tree {
            &EnumTree::Node(_, bound) | &EnumTree::UnexpandedNode(_, bound) =>
                bound,
                _ => {
                    panic!("Trying to get bound of an No Go Branch");
                }
        }
    }

    /// Given a list of candidates, create a Node which has these candidates as children 
    /// children being created as unexpanded nodes
    fn create_node(new_nodes: Vec<Candidate<'a>>, context: &'b Context<'b>, 
                   bound: f64, id_node: String) -> SearchTree<'a, 'b> {
        let mut rew_vec = vec![];
        let mut arc_nodes: Vec<SearchTree<'a, 'b>> = vec![];
        for node in new_nodes {
            let cand_bound = node.bound.value();
            rew_vec.push((vec![], 0));
            arc_nodes.push(SearchTree{tree: EnumTree::UnexpandedNode(node, cand_bound),
            context });
        }
        SearchTree{
            tree: EnumTree::Node(Arc::new(RwLock::new(
                              Node {children: arc_nodes, rewards: rew_vec, id: id_node})), 
                      bound),
                      context}
    }
}



/// gives a "score" to a branch of the tree at a given node n_successes is the number of successes
/// of that branch (that is, the number of leaves that belong to the THRESHOLD best of that node
/// and which come from that particular branch) 
/// n_branch_trials is the number of trials of that branch (both failed and succeeded), n_trials
/// the number of trials of the node and k the number of branches in the node
fn heval(config: &BanditConfig, n_successes: usize, n_branch_trials: usize, n_trials: usize,
         n_branches: usize) -> f64 {
    if n_trials == 0 {
        std::f64::INFINITY
    }
    else {
        let f = (n_trials * n_branches) as f64;
        let alpha= f.ln() / config.delta;
        let ret = ((n_successes as f64) + alpha + (2. * (n_successes as f64) * alpha + alpha *
                                                   alpha).sqrt()) / (n_branch_trials as f64);
        ret
    }
}

fn montecarlo_descend<'a>(config: &BanditConfig, 
                          context: &Context, 
                          candidate: Candidate<'a>, 
                          cut: f64) 
    -> Option<Candidate<'a>>
{
    let choice_opt = choice::list(&candidate.space).next();
    if let Some(choice) = choice_opt {
        let new_nodes = candidate.apply_choice(context, choice);
                if new_nodes.is_empty() {
                    None
                } else { 
                    let chosen_candidate = choose_next_cand(config, new_nodes, cut);
                    montecarlo_descend(config, context, chosen_candidate, cut)
                } 
    }
    else { Some(candidate) }
}

fn choose_next_cand<'a>(config: &BanditConfig, 
                        mut new_nodes: Vec<Candidate<'a>>, 
                        cut: f64) -> Candidate<'a> {
    let ind = match config.new_nodes_order {
        NewNodeOrder::Api => 0,
        NewNodeOrder::WeightedRandom => choose_cand_weighted(&new_nodes.iter().collect_vec(), cut),
          NewNodeOrder::Bound => choose_cand_best(&new_nodes.iter().collect_vec()),
          NewNodeOrder::Random => choose_cand_rand(&new_nodes.iter().collect_vec()),
    };
    new_nodes.remove(ind)
}


//fn choose_cand_best<'a, T>(new_nodes: T) -> usize where T: Iterator<Item=&'a Candidate<'a>>{
fn choose_cand_best<'a>(new_nodes: &Vec<&Candidate<'a>>) -> usize {
    new_nodes.iter().enumerate().min_by(|x1, x2| cmp_f64(x1.1.bound.value(), x2.1.bound.value()))
        .map(|x| x.0)
    // We checked in montecarlo_descend that new_nodes is not empty
        .unwrap()
}

//fn choose_cand_rand<'a, T>(new_nodes: T) -> usize where T: Iterator<Item=&'a Candidate<'a>>{
fn choose_cand_rand<'a>(new_nodes: &Vec<&Candidate<'a>>) -> usize {
    let mut rng = thread_rng();
    rng.gen_range(0, new_nodes.len())
}

fn choose_cand_weighted<'a>(new_nodes: &Vec<&'a Candidate<'a>>, cut: f64) -> usize {
//fn choose_cand_weighted<'a, T>(new_nodes: T, cut: f64) 
//  -> usize where T: Iterator<Item=&'a Candidate<'a>>
    let mut weighted_items = vec![];
    let mut rng = thread_rng();
    let max_bound = new_nodes.iter().max_by(|x1, x2|
                                            cmp_f64(x1.bound.value(), x2.bound.value()))
        .map(|x| x.bound.value()).unwrap();
    for (ind, x) in new_nodes.iter().enumerate() {
        if cut.is_infinite() {
            let x_weight = (10f64 * max_bound / x.bound.value()).floor() as u32 ;
            weighted_items.push(Weighted{weight: x_weight, item: ind});
        } else {
            assert!(x.bound.value() <= cut);
            let weight = (1000f64 * (1f64 - x.bound.value()/cut)).floor() as u32;
            let weight = std::cmp::max(1, weight);
            weighted_items.push(Weighted { weight, item: ind});
        }
    }
    WeightedChoice::new(&mut weighted_items).ind_sample(&mut rng)
}
