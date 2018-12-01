///! Exploration of the search space.
use device::Context;
use explorer::candidate::Candidate;
use explorer::config::{BanditConfig, NewNodeOrder, OldNodeOrder};
use explorer::logger::LogMessage;
use explorer::store::Store;
use explorer::{choice, local_selection};
use itertools::Itertools;
use rpds::List;
use std;
use std::f64;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock, Weak};
use utils::*;

/// An environment in which candidates can be refined.
pub struct Env<'a> {
    config: &'a BanditConfig,
    context: &'a dyn Context,
    cut: f64,
}

impl<'a> Env<'a> {
    /// List the available actions for a given candidate.
    ///
    /// If `list_actions` return `None`, the candidate is a fully-specified implementation.
    pub fn list_actions(&self, candidate: &Candidate<'_>) -> Option<Vec<choice::ActionEx>> {
        choice::list(&self.config.choice_ordering, &candidate.space).next()
    }

    /// Apply a choice to a candidate.
    ///
    /// This returns a list of candidates, one for each potential decision.  Note that the
    /// resulting vector of candidates may be shorter than the number of decisions in the choice in
    /// two cases: some actions can be discarded through propagation or by applying a
    /// branch-and-bound algorithm.
    pub fn apply_choice<'c>(
        &self,
        candidate: &Candidate<'c>,
        actions: Vec<choice::ActionEx>,
    ) -> Vec<Candidate<'c>> {
        candidate
            .apply_choice(self.context, actions)
            .into_iter()
            .filter(|candidate| candidate.bound.value() < self.cut)
            .collect::<Vec<_>>()
    }
}

/// Policy to use when descending in the tree.
pub trait TreePolicy: Sync + Sized {
    /// Statistics stored on the nodes.
    type NodeStats: Default + Send + Sync;

    /// Statistics stored on the edges.
    type EdgeStats: Default + Send + Sync;

    /// Pick a child in the given environment.  Returns an index for the child, or `None` if the
    /// node has no children.
    fn pick_child(&self, env: &Env<'_>, node: &Node<'_, Self>) -> Option<usize>;

    /// Record an evaluation across an edge.  This indicates that an evaluation of `eval` was found
    /// in a path which contains edge `node.children[idx]`.
    fn backpropagate(&self, node: &Node<Self>, idx: usize, eval: f64);
}

/// Global tree statistics
struct TreeStats {
    /// The number of deadends encountered during the descent.
    num_deadends: AtomicUsize,
}

impl Default for TreeStats {
    fn default() -> Self {
        TreeStats {
            num_deadends: AtomicUsize::new(0),
        }
    }
}

#[derive(Serialize, Deserialize)]
pub enum TreeEvent {
    Evaluation {
        actions: Sequence<choice::ActionEx>,
        score: f64,
    },
}

/// A search tree to perform a multi-armed bandit search.
pub struct Tree<'a, 'b, P: TreePolicy> {
    root: Edge<'a, P>,
    stop: AtomicBool,
    cut: RwLock<f64>,
    config: &'b BanditConfig,
    policy: P,
    stats: TreeStats,
    log: std::sync::mpsc::SyncSender<LogMessage<TreeEvent>>,
}

impl<'a, 'b, P: TreePolicy> Tree<'a, 'b, P> {
    /// Creates a new search tree containing the given candidates.
    pub fn new(
        candidates: Vec<Candidate<'a>>,
        config: &'b BanditConfig,
        policy: P,
        log_sender: std::sync::mpsc::SyncSender<LogMessage<TreeEvent>>,
    ) -> Self {
        Tree {
            root: SubTree::from(Node::from_candidates(candidates)).into(),
            stop: AtomicBool::new(false),
            cut: RwLock::new(std::f64::INFINITY),
            config,
            policy,
            stats: TreeStats::default(),
            log: log_sender,
        }
    }

    /// Removes the dead ends along the given path. Assumes the path points to a dead-end.
    /// Updates bounds along the way.
    fn clean_deadends(&self, mut path: Path<'a, P>, cut: f64) {
        // A `None` bound indicates the path points to a dead-end.
        let mut bound = None;
        while let Some((node, pos)) = path.0.pop() {
            if let Some(node) = node.upgrade() {
                if let Some(bound) = bound {
                    node.children[pos].update_bound(bound);
                    // If the bound was set, then we finished removing deadends.
                    return;
                } else {
                    node.children[pos].kill();

                    bound = node.bound();
                    // Node with a bound above the cut are considered as dead-ends.
                    if bound.map(|b| b >= cut).unwrap_or(false) {
                        bound = None;
                    }
                }
            } else {
                return;
            }
        }

        // If we did not returned before, we have reached the root of the tree.
        if let Some(bound) = bound {
            self.root.update_bound(bound);
        } else {
            self.root.kill();
        }
    }
}

impl<'a, 'b, P: TreePolicy> Store<'a> for Tree<'a, 'b, P> {
    type PayLoad = Path<'a, P>;

    type Event = TreeEvent;

    fn update_cut(&self, new_cut: f64) {
        *unwrap!(self.cut.write()) = new_cut;
        let mut stack = self.root.trim(new_cut).into_iter().collect::<Vec<_>>();
        while let Some(subtree) = stack.pop() {
            if let SubTree::InternalNode(node) = &*unwrap!(subtree.read()) {
                for edge in &node.children {
                    stack.extend(edge.trim(new_cut))
                }
            }
        }
        info!("trimming finished");
    }

    fn commit_evaluation(
        &self,
        actions: &List<choice::ActionEx>,
        mut path: Self::PayLoad,
        eval: f64,
    ) {
        unwrap!(self.log.send(LogMessage::Event(TreeEvent::Evaluation {
            actions: Sequence::List(actions.clone()),
            score: eval,
        })));

        while let Some((node, idx)) = path.0.pop() {
            if let Some(node) = node.upgrade() {
                self.policy.backpropagate(&node, idx, eval);
            }
        }
    }

    fn explore(&self, context: &Context) -> Option<(Candidate<'a>, Self::PayLoad)> {
        // Retry loop (in case of deadends)
        loop {
            if self.stop.load(Ordering::Relaxed) {
                return None;
            }

            let cut: f64 = { *unwrap!(self.cut.read()) };
            let env = Env {
                config: &self.config,
                context,
                cut,
            };

            // Bail out early if the root is a deadend
            let mut state = self.root.descend(&env);
            if let SubTree::Empty = state {
                return None;
            }

            // Descent loop
            let mut path = Path::default();
            loop {
                match state {
                    SubTree::Empty => {
                        self.clean_deadends(path, env.cut);
                        self.stats.num_deadends.fetch_add(1, Ordering::Relaxed);
                        break;
                    }
                    SubTree::Leaf(leaf) => {
                        if let Some(candidate) = local_selection::descend(
                            &env.config.choice_ordering,
                            env.config.new_nodes_order,
                            env.context,
                            *leaf,
                            env.cut,
                        ) {
                            return Some((candidate, path));
                        } else {
                            // Deadend reached while exploring; restart from the root
                            // TODO(bclement): We should backpropagate explicitely here.
                            self.stats.num_deadends.fetch_add(1, Ordering::Relaxed);

                            break;
                        }
                    }
                    SubTree::InternalNode(node) => {
                        if let Some((idx, next)) = node.descend(&env, &self.policy) {
                            path.0.push((Arc::downgrade(&node), idx));
                            state = next;
                        } else {
                            state = SubTree::Empty;
                        }
                    }
                }
            }
        }
    }

    fn stop_exploration(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }

    fn print_stats(&self) {
        warn!("=== Bandit statistics ===");
        warn!(
            "Deadends encountered: {}",
            self.stats.num_deadends.load(Ordering::Relaxed)
        );
    }
}

/// Path to follow to reach a leaf in the tree.
#[derive(Clone)]
pub struct Path<'a, P: TreePolicy>(Vec<(Weak<Node<'a, P>>, usize)>);

impl<'a, P: TreePolicy> Default for Path<'a, P> {
    fn default() -> Self {
        Path(vec![])
    }
}

/// The search tree that will be traversed
enum SubTree<'a, P: TreePolicy> {
    /// The subtree has been expanded and has children.
    InternalNode(Arc<Node<'a, P>>),
    /// The subtree has not been expanded yet.  This is a leaf in the MCTS tree.
    Leaf(Box<Candidate<'a>>),
    /// The subtree is empty.
    Empty,
}

impl<'a, P: TreePolicy> From<Node<'a, P>> for SubTree<'a, P> {
    fn from(node: Node<'a, P>) -> Self {
        if node.is_deadend() {
            SubTree::Empty
        } else {
            SubTree::InternalNode(Arc::new(node))
        }
    }
}

impl<'a, P: TreePolicy> SubTree<'a, P> {
    fn bound(&self) -> Option<f64> {
        match self {
            SubTree::InternalNode(node) => node.bound(),
            SubTree::Leaf(candidate) => Some(candidate.bound.value()),
            SubTree::Empty => None,
        }
    }
}

/// An edge in the tree
struct Edge<'a, P: TreePolicy> {
    /// The destination of the edge.  This may not be expanded yet.
    dst: Arc<RwLock<SubTree<'a, P>>>,

    /// Edge statistics
    stats: P::EdgeStats,

    /// The current bound for the pointed-to node.
    bound: RwLock<f64>,
}

impl<'a, P: TreePolicy> From<SubTree<'a, P>> for Edge<'a, P> {
    fn from(subtree: SubTree<'a, P>) -> Self {
        Edge {
            stats: P::EdgeStats::default(),
            bound: RwLock::new(subtree.bound().unwrap_or(std::f64::INFINITY)),
            dst: Arc::new(RwLock::new(subtree)),
        }
    }
}

impl<'a, P: TreePolicy> From<Candidate<'a>> for Edge<'a, P> {
    fn from(candidate: Candidate<'a>) -> Self {
        SubTree::Leaf(Box::new(candidate)).into()
    }
}

impl<'a, P: TreePolicy> Edge<'a, P> {
    /// Kill the edge, replacing it with a dead end.  The bound is erased.
    fn kill(&self) {
        *unwrap!(self.dst.write()) = SubTree::Empty;
        self.update_bound(std::f64::INFINITY);
    }

    // TODO(bclement):  Does this actually help at all?
    /// Update the bound.  This is typically called after a cut made more information available in
    /// the subtree.
    fn update_bound(&self, bound: f64) {
        *unwrap!(self.bound.write()) = bound;
    }

    /// Return the bound on execution time for the node this edge points to.
    fn bound(&self) -> f64 {
        *unwrap!(self.bound.read())
    }

    /// Trims the branch if it has an evaluation time guaranteed to be worse than
    /// `cut`. Returns the childrens to trim if any,
    fn trim(&self, cut: f64) -> Option<Arc<RwLock<SubTree<'a, P>>>> {
        if self.bound() >= cut {
            self.kill();
            None
        } else if let SubTree::InternalNode(_) = *unwrap!(self.dst.read()) {
            Some(Arc::clone(&self.dst))
        } else {
            None
        }
    }

    /// Descend one level in the tree, expanding it if necessary.
    fn descend(&self, env: &Env<'_>) -> SubTree<'a, P> {
        loop {
            // Most of the time we only need read access
            {
                match &*unwrap!(self.dst.read()) {
                    SubTree::Empty => return SubTree::Empty,
                    SubTree::InternalNode(node) => return SubTree::InternalNode(Arc::clone(node)),
                    SubTree::Leaf(_) => {
                        // Need write access, see below
                    }
                }
            }

            // Some times we do need write acces to expand a leaf candidate... but another thread
            // could beat us to the punch and expand it before us, in which case we won't see a
            // leaf below and loop again (where the first read access should not fail).
            {
                let dst = &mut *unwrap!(self.dst.write());
                if let SubTree::Leaf(_) = dst {
                    // We got write access to the leaf; expand it.
                    if let SubTree::Leaf(candidate) = std::mem::replace(dst, SubTree::Empty) {
                        let choice = env.list_actions(&candidate);
                        if let Some(choice) = choice {
                            let node = Node::from_candidates(env.apply_choice(&candidate, choice));
                            if node.is_deadend() {
                                // Actual dead-end; leave the SubTree::Empty there.
                                return SubTree::Empty;
                            } else {
                                // Newly expanded node, with no stats yet.
                                *dst = SubTree::InternalNode(Arc::new(node));
                                return SubTree::Leaf(candidate);
                            }
                        } else {
                            // Fully specified implementation; we leave the SubTree::Empty here because if
                            // we come back, this becomes a dead-end.  It may not be the smartest thing to
                            // do because it could throw off the search, but that is probably pretty rate
                            // anyways.
                            return SubTree::Leaf(candidate);
                        }
                    } else {
                        // We checked we were in the Leaf case before the mem::replace
                        unreachable!()
                    }
                } else {
                    // The leaf was expanded by another thread; try again on the expanded node.
                    continue;
                }
            }
        }
    }
}

/// Holds the children of a `SubTree::InternalNode`.
pub struct Node<'a, P: TreePolicy> {
    children: Vec<Edge<'a, P>>,

    /// Node statistics
    _stats: P::NodeStats,
}

impl<'a, P: TreePolicy> Node<'a, P> {
    /// Creates a new children containing the given candidates, if any.
    fn from_candidates(candidates: Vec<Candidate<'a>>) -> Self {
        let children = candidates.into_iter().map(Edge::from).collect_vec();
        Node {
            children,
            _stats: Default::default(),
        }
    }

    /// Returns the lowest bound of the children, if any.
    fn bound(&self) -> Option<f64> {
        self.children
            .iter()
            .map(|edge| edge.bound())
            .min_by(|&lhs, &rhs| cmp_f64(lhs, rhs))
    }

    fn is_deadend(&self) -> bool {
        self.bound().is_none()
    }

    /// Trim children (that is, replace with an empty `SubTree`) children whose bounds are
    /// higher than the cut. Also clean-up evaluations.
    fn trim(&self, cut: f64) {
        for edge in &self.children {
            edge.trim(cut);
        }
    }

    /// Descend one level in the tree, expanding it if necessary.
    fn descend(&self, env: &Env<'_>, policy: &P) -> Option<(usize, SubTree<'a, P>)> {
        self.trim(env.cut);

        policy
            .pick_child(env, self)
            .map(|idx| (idx, self.children[idx].descend(env)))
    }
}

impl<'a> TreePolicy for &'a BanditConfig {
    type NodeStats = ();
    type EdgeStats = TAGStats;

    fn pick_child(&self, env: &Env<'_>, node: &Node<'_, Self>) -> Option<usize> {
        // Pick a new node if there are any remaining.
        let new_nodes = node
            .children
            .iter()
            .enumerate()
            .filter(|(_, edge)| edge.stats.num_visits() == 0)
            .map(|(idx, edge)| (idx, edge.bound()));

        self.new_nodes_order
            .pick_index(new_nodes, env.cut)
            .or_else(|| match self.old_nodes_order {
                OldNodeOrder::Bound => NewNodeOrder::Bound.pick_index(
                    node.children.iter().map(|edge| edge.bound()).enumerate(),
                    env.cut,
                ),
                OldNodeOrder::WeightedRandom => NewNodeOrder::WeightedRandom.pick_index(
                    node.children.iter().map(|edge| edge.bound()).enumerate(),
                    env.cut,
                ),
                OldNodeOrder::Bandit => pick_tag_arm(self.delta, self.threshold, node, env.cut),
            }).map(|idx| {
                node.children[idx].stats.down();
                idx
            })
    }

    fn backpropagate(&self, node: &Node<Self>, idx: usize, eval: f64) {
        node.children[idx].stats.up(self, eval);
    }
}

/// Picks a candidate below the bound using TAG formula.
fn pick_tag_arm<'a>(
    delta: f64,
    threshold: usize,
    node: &Node<'a, &'_ BanditConfig>,
    cut: f64,
) -> Option<usize> {
    // Ignore cut children
    let children = node
        .children
        .iter()
        .enumerate()
        .filter(|(_, edge)| edge.bound() < cut);

    // Compute the threshold to use so that we only have `config.threshold` children
    let threshold = {
        let mut evalns = Evaluations::with_capacity(threshold);
        for (_, edge) in children.clone() {
            // Evaluations are sorted; we can bail out early.
            for &eval in &*unwrap!(edge.stats.evaluations.read()) {
                if !evalns.record(eval, threshold) {
                    break;
                }
            }
        }

        // It could happen that all edges have num_visits > 0 but still we don't have
        // any recorded evaluations if none of the descents have finished yet.
        evalns.max().unwrap_or(std::f64::INFINITY)
    };

    // Precompute statistics for each child to ensure the number of visits of a child is not
    // incremented concurrently after we have computed the sum.
    let stats = children
        .map(|(ix, edge)| {
            (
                ix,
                unwrap!(edge.stats.evaluations.read()).count_lte(threshold),
                edge.stats.num_visits(),
            )
        }).collect::<Vec<_>>();

    // Total number of visits on the node
    let num_visits = stats
        .iter()
        .map(|(_, _, num_visits)| num_visits)
        .sum::<usize>();

    // Total number of children, excluding ones which were cut
    let num_children = stats.len();

    stats
        .into_iter()
        .map(|(ix, child_successes, child_visits)| {
            let score = heval(
                delta,
                child_successes,
                child_visits,
                num_visits,
                num_children,
            );
            (ix, score)
        }).max_by(|x1, x2| cmp_f64(x1.1, x2.1))
        .map(|(ix, _)| ix)
}

/// gives a "score" to a branch of the tree at a given node n_successes is the number of
/// successes of that branch (that is, the number of leaves that belong to the THRESHOLD
/// best of that node and which come from that particular branch).
/// * `n_branch_trials` is the number of trials of that branch (both failed and succeeded),
/// * `n_trials` is  the number of trials of the node and k the number of branches in the
///   node.
fn heval(
    delta: f64,
    n_successes: usize,
    n_branch_trials: usize,
    n_trials: usize,
    n_branches: usize,
) -> f64 {
    assert!(n_branches > 0);
    assert!(n_branch_trials <= n_trials);
    if n_branch_trials == 0 {
        std::f64::INFINITY
    } else {
        let alpha = (2. * (n_trials * n_branches) as f64 / delta).ln().max(0.);
        let sqrt_body = alpha * (2. * n_successes as f64 + alpha);
        (n_successes as f64 + alpha + sqrt_body.sqrt()) / n_branch_trials as f64
    }
}

/// Holds the TAG statistics for a given edge.
pub struct TAGStats {
    /// All evaluations seen for the pointed-to node.
    evaluations: RwLock<Evaluations>,

    /// Number of visits across this edge.  Note that this is the number of descents; there may
    /// have been less backpropagations due to dead-ends.
    num_visits: AtomicUsize,
}

impl Default for TAGStats {
    fn default() -> Self {
        TAGStats {
            evaluations: RwLock::new(Evaluations::new()),
            num_visits: AtomicUsize::new(0),
        }
    }
}

impl TAGStats {
    /// Called when the edge is selected during a descent
    fn down(&self) {
        self.num_visits.fetch_add(1, Ordering::Relaxed);
    }

    /// Called when backpropagating across this edge after an evaluation
    fn up(&self, config: &BanditConfig, eval: f64) {
        unwrap!(self.evaluations.write()).record(eval, config.threshold);
    }

    /// The number of visits through this edge.
    fn num_visits(&self) -> usize {
        self.num_visits.load(Ordering::Relaxed)
    }
}

/// Holds the evaluations seen for a given node or edge.
struct Evaluations(Vec<f64>);

impl Evaluations {
    /// Create an new evaluation vector
    fn new() -> Evaluations {
        Evaluations(vec![])
    }

    fn with_capacity(capacity: usize) -> Self {
        Evaluations(Vec::with_capacity(capacity))
    }

    /// Returns the number of evaluations below the given threshold.
    fn count_lte(&self, threshold: f64) -> usize {
        match self.0.binary_search_by(|&probe| cmp_f64(probe, threshold)) {
            Ok(mut pos) => {
                // Get the highest value matching the threshold
                while pos < self.0.len()
                    && cmp_f64(self.0[pos], threshold) == std::cmp::Ordering::Equal
                {
                    pos += 1;
                }
                pos
            }
            Err(pos) => pos,
        }
    }

    /// Record a new evaluation.  Returns `true` if the evaluation is among the top `threshold`.
    fn record(&mut self, eval: f64, threshold: usize) -> bool {
        let pos = self
            .0
            .binary_search_by(|&probe| cmp_f64(probe, eval))
            .unwrap_or_else(|e| e);
        if pos < threshold {
            if self.0.len() >= threshold {
                self.0.pop();
            }
            self.0.insert(pos, eval);
            true
        } else {
            false
        }
    }

    /// Returns the maximum recorded reward.  Note that some higher evaluations may have been seen, but
    /// were discarded due to being too large according to the configured threshold.
    fn max(&self) -> Option<f64> {
        self.0.last().cloned()
    }
}

impl<'a> IntoIterator for &'a Evaluations {
    type Item = &'a f64;
    type IntoIter = std::slice::Iter<'a, f64>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}
