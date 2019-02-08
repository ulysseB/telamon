///! Exploration of the search space.
use crate::device::Context;
use crate::explorer::candidate::Candidate;
use crate::explorer::config::{self, BanditConfig, NewNodeOrder};
use crate::explorer::logger::LogMessage;
use crate::explorer::store::Store;
use crate::explorer::{choice, local_selection};
use log::{debug, info, trace, warn};
use rpds::List;
use serde::{Deserialize, Serialize};
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
    pub fn list_actions(
        &self,
        candidate: &Candidate<'_>,
    ) -> Option<Vec<choice::ActionEx>> {
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
pub trait TreePolicy: Sized {
    /// Statistics stored on the edges.
    type EdgeStats: Default;

    /// Pick a child in the given environment.  Returns an index for the child, or `None` if the
    /// node has no children.
    fn pick_child(
        &self,
        env: &Env<'_>,
        node: &Node<'_, Self::EdgeStats>,
    ) -> Option<usize>;

    /// Record an evaluation across an edge.  This indicates that an evaluation of `eval` was found
    /// in a path which contains edge `node.children[idx]`.
    fn backpropagate(&self, node: &Node<'_, Self::EdgeStats>, idx: usize, eval: f64);
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
pub enum DeadEndSource {
    /// Dead-end encountered in the tree
    Tree,
    /// Dead-end encountered in the rollout phase
    Rollout {
        /// List of actions defining the dead-end candidate
        actions: List<choice::ActionEx>,
        /// Depth in the tree.  The remaining actions were selected during rollout.
        depth: usize,
        /// Performance model bound
        bound: f64,
        /// Current cut value
        cut: f64,
    },
}

/// The possible tree events.
/// WARNING:  Changing the enums *will break* any pre-existing eventlog files.  Adding new cases
/// *at the end only* is safe.
#[derive(Serialize, Deserialize)]
pub enum TreeEvent {
    Evaluation {
        actions: List<choice::ActionEx>,
        score: f64,
    },

    /// A fully-specified implementation was found and evaluated
    EvaluationV2 {
        /// List of actions defining the implementation
        actions: List<choice::ActionEx>,
        /// Depth in the tree.  The remaining actions were selected during rollout.
        depth: usize,
        /// Execution time
        score: f64,
        /// Performance model lower bound
        bound: f64,
        /// Cut value when the implementation was found.  This is the best implementation at the
        /// time the descent started from the root, as threads only synchronize the cut at the
        /// root.
        cut: f64,
        /// Time at which the implementation was found
        search_end_time: f64,
        /// Time at which the evaluation finished
        evaluation_end_time: f64,
        /// ID of the thread that found this implementation
        thread: String,
    },

    /// A dead-end was reached
    DeadEnd {
        /// Source of this deadend
        source: DeadEndSource,
        /// Time at which the deadend was found after the start of the program
        time: f64,
        /// ID of the thread that found the deadend
        thread: String,
    },
}

/// A search tree to perform a multi-armed bandit search.
pub struct Tree<'a, 'b, P: TreePolicy> {
    root: RwLock<Option<Arc<Node<'a, P::EdgeStats>>>>,
    bound: RwLock<f64>,
    stop: AtomicBool,
    cut: RwLock<f64>,
    config: &'b BanditConfig,
    policy: P,
    stats: TreeStats,
    log: std::sync::mpsc::SyncSender<LogMessage<TreeEvent>>,
    start_time: std::time::Instant,
}

impl<'a, 'b, P: TreePolicy> Tree<'a, 'b, P> {
    /// Creates a new search tree containing the given candidates.
    pub fn new(
        candidates: Vec<Candidate<'a>>,
        config: &'b BanditConfig,
        policy: P,
        log_sender: std::sync::mpsc::SyncSender<LogMessage<TreeEvent>>,
    ) -> Self {
        let root = Node::try_from_candidates(candidates);
        let bound = root.as_ref().and_then(|root| root.bound());

        Tree {
            root: RwLock::new(root),
            stop: AtomicBool::new(false),
            cut: RwLock::new(config.initial_cut.unwrap_or(std::f64::INFINITY)),
            bound: RwLock::new(bound.unwrap_or(std::f64::INFINITY)),
            config,
            policy,
            stats: TreeStats::default(),
            log: log_sender,
            start_time: std::time::Instant::now(),
        }
    }

    fn thread(&self) -> String {
        format!("{:?}", std::thread::current().id())
    }

    fn timestamp(&self) -> f64 {
        let time = self.start_time.elapsed();
        time.as_secs() as f64 + time.subsec_nanos() as f64 * 1e-9
    }

    /// Removes the dead ends along the given path. Assumes the path points to a dead-end.
    /// Updates bounds along the way.
    fn clean_deadends(&self, path: &Path<'a, P::EdgeStats>, cut: f64) {
        // A `None` bound indicates the path points to a dead-end.
        let mut bound = None;
        for &(ref node, pos) in path.0.iter().rev() {
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
            trace!("upgrading root bound to {}", bound);
            *unwrap!(self.bound.write()) = bound;
        } else {
            trace!("killing root");
            *unwrap!(self.root.write()) = None;
        }
    }
}

impl<'a, 'b, P: TreePolicy> Store<'a> for Tree<'a, 'b, P>
where
    P: Send + Sync,
    P::EdgeStats: Send + Sync,
{
    type PayLoad = ImplInfo<'a, P::EdgeStats>;

    type Event = TreeEvent;

    fn update_cut(&self, new_cut: f64) {
        *unwrap!(self.cut.write()) = new_cut;

        let mut stack = match &*unwrap!(self.root.read()) {
            Some(node) => vec![Arc::clone(node)],
            None => Vec::new(),
        };

        while let Some(node) = stack.pop() {
            for edge in &node.children {
                stack.extend(edge.trim(new_cut))
            }
        }

        info!("cut: trimming finished");
    }

    fn commit_evaluation(
        &self,
        actions: &List<choice::ActionEx>,
        mut info: Self::PayLoad,
        eval: f64,
    ) {
        unwrap!(self.log.send(LogMessage::Event(TreeEvent::EvaluationV2 {
            actions: actions.clone(),
            depth: info.path.0.len(),
            score: eval,
            bound: info.bound,
            cut: info.cut,
            search_end_time: info.time,
            evaluation_end_time: self.timestamp(),
            thread: info.thread,
        })));

        while let Some((node, idx)) = info.path.0.pop() {
            if let Some(node) = node.upgrade() {
                self.policy.backpropagate(&node, idx, eval);
            }
        }
    }

    fn explore(&self, context: &Context) -> Option<(Candidate<'a>, Self::PayLoad)> {
        // Retry loop (in case of deadends)
        loop {
            if self.stop.load(Ordering::Relaxed) {
                debug!("stopping: requested");
                return None;
            }

            let cut: f64 = { *unwrap!(self.cut.read()) };
            let env = Env {
                config: &self.config,
                context,
                cut,
            };

            // Bail out early if the root is a deadend
            let mut state = SubTree::InternalNode(match &*unwrap!(self.root.read()) {
                Some(node) => Arc::clone(node),
                None => {
                    debug!("stopping: deadend at root");
                    return None;
                }
            });

            // Rollout configuration
            let rollout = local_selection::Rollout {
                choice_order: &env.config.choice_ordering,
                node_order: &env.config.new_nodes_order,
                context: env.context,
                cut: env.cut,
            };

            // Descent loop
            let mut path = Path::default();
            loop {
                match state {
                    SubTree::Empty => {
                        info!("Deadend found in the tree.");
                        unwrap!(self.log.send(LogMessage::Event(TreeEvent::DeadEnd {
                            source: DeadEndSource::Tree,
                            time: self.timestamp(),
                            thread: self.thread(),
                        })));

                        self.clean_deadends(&path, env.cut);
                        self.stats.num_deadends.fetch_add(1, Ordering::Relaxed);
                        break;
                    }
                    SubTree::Leaf(leaf) => {
                        let mut rollout_path = Vec::new();
                        if let Some(implementation) =
                            rollout.descend_with_path(*leaf, &mut rollout_path)
                        {
                            info!("Implementation found.");

                            let info = ImplInfo {
                                path,
                                bound: implementation.bound.value(),
                                cut: env.cut,
                                time: self.timestamp(),
                                thread: self.thread(),
                            };

                            return Some((implementation, info));
                        } else {
                            info!("Deadend found during rollout.");

                            if let Some(dead) = rollout_path.last() {
                                unwrap!(self.log.send(LogMessage::Event(
                                    TreeEvent::DeadEnd {
                                        source: DeadEndSource::Rollout {
                                            actions: dead.actions.clone(),
                                            depth: path.0.len(),
                                            bound: dead.bound.value(),
                                            cut: env.cut,
                                        },
                                        time: self.timestamp(),
                                        thread: self.thread(),
                                    }
                                )));
                            } else {
                                warn!("Empty rollout.");
                            }

                            // Deadend reached while exploring; restart from the root
                            // TODO(bclement): We should backpropagate explicitely here.
                            self.stats.num_deadends.fetch_add(1, Ordering::Relaxed);

                            break;
                        }
                    }
                    SubTree::InternalNode(node) => {
                        node.trim(env.cut);

                        if let Some(idx) = self.policy.pick_child(&env, &node) {
                            path.0.push((Arc::downgrade(&node), idx));
                            state = node.children[idx].descend(&env);
                        } else {
                            trace!("no child available: deadend");
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

/// Informations on a fully-specified implementation
#[derive(Clone)]
pub struct ImplInfo<'a, E> {
    /// Path to the implementation (in the tree)
    path: Path<'a, E>,
    /// Bound from the performance model
    bound: f64,
    /// Cut at the time the implementation was found
    cut: f64,
    /// Time at which the implementation was found
    time: f64,
    /// ID of the thread which found the implementation
    thread: String,
}

/// Path to follow to reach a leaf in the tree.
#[derive(Clone, Default)]
pub struct Path<'a, E>(Vec<(Weak<Node<'a, E>>, usize)>);

/// The search tree that will be traversed
enum SubTree<'a, E> {
    /// The subtree has been expanded and has children.
    InternalNode(Arc<Node<'a, E>>),
    /// The subtree has not been expanded yet.  This is a leaf in the MCTS tree.
    Leaf(Box<Candidate<'a>>),
    /// The subtree is empty.
    Empty,
}

/// An edge in the tree
struct Edge<'a, E> {
    /// The destination of the edge.  This may not be expanded yet.
    dst: RwLock<SubTree<'a, E>>,

    /// Edge statistics
    stats: E,

    /// The current bound for the pointed-to node.
    bound: RwLock<f64>,
}

impl<'a, E: Default> Edge<'a, E> {
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
    fn trim(&self, cut: f64) -> Option<Arc<Node<'a, E>>> {
        if self.bound() >= cut {
            self.kill();
            None
        } else {
            let subtree = unwrap!(self.dst.read());
            if let SubTree::InternalNode(node) = &*subtree {
                Some(Arc::clone(node))
            } else {
                None
            }
        }
    }

    /// Descend one level in the tree, expanding it if necessary.
    fn descend(&self, env: &Env<'_>) -> SubTree<'a, E> {
        loop {
            // Most of the time we only need read access
            {
                match &*unwrap!(self.dst.read()) {
                    SubTree::Empty => return SubTree::Empty,
                    SubTree::InternalNode(node) => {
                        return SubTree::InternalNode(Arc::clone(node));
                    }
                    SubTree::Leaf(_) => {
                        // Need write access to expand, see below
                    }
                }
            }

            // Some times we do need write acces to expand a leaf candidate... but another thread
            // could beat us to the punch and expand it before us, in which case we won't see a
            // leaf below and loop again (where the first read access should not fail).
            {
                let dst = &mut *unwrap!(self.dst.write());
                if let SubTree::Leaf(_) = &*dst {
                    // We got write access to the leaf; expand it.
                    if let SubTree::Leaf(candidate) =
                        std::mem::replace(dst, SubTree::Empty)
                    {
                        let choice = env.list_actions(&candidate);
                        if let Some(choice) = choice {
                            if let Some(node) = Node::try_from_candidates(
                                env.apply_choice(&candidate, choice),
                            ) {
                                // Newly expanded node, with no stats yet.
                                *dst = SubTree::InternalNode(node);
                                return SubTree::Leaf(candidate);
                            } else {
                                // Actual dead-end; leave the SubTree::Empty there.
                                return SubTree::Empty;
                            }
                        } else {
                            // Fully specified implementation; we leave the SubTree::Empty here because if
                            // we come back, this becomes a dead-end.  It may not be the smartest thing to
                            // do because it could throw off the search, but that is probably pretty rate
                            // anyways.
                            debug!("implementation reached at depth {}", candidate.depth);
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
pub struct Node<'a, E> {
    children: Vec<Edge<'a, E>>,
}

impl<'a, E: Default> Node<'a, E> {
    /// Creates a new children containing the given candidates, if any.
    fn try_from_candidates(candidates: Vec<Candidate<'a>>) -> Option<Arc<Self>> {
        if candidates.is_empty() {
            None
        } else {
            Some(Arc::new(Node {
                children: candidates
                    .into_iter()
                    .map(|candidate| Edge {
                        bound: RwLock::new(candidate.bound.value()),
                        dst: RwLock::new(SubTree::Leaf(Box::new(candidate))),
                        stats: Default::default(),
                    })
                    .collect::<Vec<_>>(),
            }))
        }
    }

    /// Returns the lowest bound of the children, if any.
    fn bound(&self) -> Option<f64> {
        self.children
            .iter()
            .map(|edge| edge.bound())
            .min_by(|&lhs, &rhs| cmp_f64(lhs, rhs))
    }

    /// Trim children (that is, replace with an empty `SubTree`) children whose bounds are
    /// higher than the cut. Also clean-up evaluations.
    fn trim(&self, cut: f64) {
        for edge in &self.children {
            edge.trim(cut);
        }
    }
}

impl TreePolicy for NewNodeOrder {
    type EdgeStats = ();

    fn pick_child(&self, env: &Env<'_>, node: &Node<'_, ()>) -> Option<usize> {
        self.pick_index(
            node.children.iter().map(|edge| edge.bound()).enumerate(),
            env.cut,
        )
    }

    fn backpropagate(&self, _node: &Node<()>, _idx: usize, _eval: f64) {}
}

/// TODO(bclement):  The UCT formula is wrong, because 1) we are optimising as a reward while we
/// actually have a cost and 2) the scale is wrong (evaluations are in the e6+ range but we do as
/// if they were in 0-1).
pub struct UCTPolicy {
    exploration_constant: f64,
    normalization: Option<config::Normalization>,
    value_reduction: config::ValueReduction,
    reward: config::Reward,
    formula: config::Formula,
}

impl From<config::UCTConfig> for UCTPolicy {
    fn from(config: config::UCTConfig) -> Self {
        let config::UCTConfig {
            exploration_constant,
            normalization,
            value_reduction,
            reward,
            formula,
        } = config;
        UCTPolicy {
            exploration_constant,
            normalization,
            value_reduction,
            reward,
            formula,
        }
    }
}

impl UCTPolicy {
    fn exploration_factor(&self, env: &Env<'_>) -> f64 {
        use self::config::Normalization;

        match self.normalization {
            Some(Normalization::GlobalBest) => {
                self.exploration_constant * self.reward(env.cut).abs()
            }
            None => self.exploration_constant,
        }
    }

    fn exploration_term(
        &self,
        env: &Env<'_>,
        visits: f64,
        total_visits: f64,
        num_children: usize,
    ) -> f64 {
        use self::config::Formula;

        self.exploration_factor(env)
            * match self.formula {
                Formula::Uct => (total_visits.ln() / visits).sqrt(),
                Formula::AlphaPuct => {
                    // TODO(bclement): Support non-uniform priors here.
                    (num_children as f64).recip() * total_visits.sqrt() / (1. + visits)
                }
            }
    }

    fn value(&self, stats: &UCTStats) -> (f64, usize) {
        use self::config::ValueReduction;

        let num_visits = stats.num_visits();

        let value = match self.value_reduction {
            ValueReduction::Best => stats.best_evaluation(),
            ValueReduction::Mean => stats.sum_evaluations() / num_visits as f64,
        };

        (value, num_visits)
    }

    fn reward(&self, evaln: f64) -> f64 {
        use self::config::Reward;

        match self.reward {
            Reward::NegTime => -evaln,
            Reward::Speed => evaln.recip(),
            Reward::LogSpeed => -evaln.ln(),
        }
    }
}

impl TreePolicy for UCTPolicy {
    type EdgeStats = UCTStats;

    fn pick_child(&self, env: &Env<'_>, node: &Node<'_, UCTStats>) -> Option<usize> {
        let stats = node
            .children
            .iter()
            .enumerate()
            .map(|(idx, edge)| (idx, edge.bound(), self.value(&edge.stats)))
            .filter(|(_, bound, (_value, _visits))| *bound < env.cut)
            .collect::<Vec<_>>();

        // Pick an edge which was not explored yet, if there is some...
        NewNodeOrder::WeightedRandom
            .pick_index(
                stats
                    .iter()
                    .filter(|(_, _bound, (_value, visits))| {
                        // Use the default policy if one of the following is true:
                        //  1. The node was never visited
                        //  2. The cut is infinite (we never got back any evaluation results yet)
                        env.cut.is_infinite() || *visits == 0
                    })
                    .map(|(idx, bound, (_value, _visits))| (*idx, *bound)),
                env.cut,
            )
            .or_else(|| {
                // Otherwise apply the UCT formula
                let total_visits = stats
                    .iter()
                    .map(|(_idx, _bound, (_value, visits))| visits)
                    .sum::<usize>() as f64;

                let num_children = stats.len();

                stats
                    .into_iter()
                    .map(|(idx, _bound, (value, visits))| {
                        (
                            idx,
                            value
                                + self.exploration_term(
                                    env,
                                    visits as f64,
                                    total_visits,
                                    num_children,
                                ),
                        )
                    })
                    .max_by(|lhs, rhs| cmp_f64(lhs.1, rhs.1))
                    .map(|(idx, _)| idx)
            })
            .map(|idx| {
                node.children[idx].stats.down();
                idx
            })
    }

    fn backpropagate(&self, node: &Node<UCTStats>, idx: usize, eval: f64) {
        node.children[idx].stats.up(self.reward(eval))
    }
}

pub struct UCTStats {
    best_evaluation: RwLock<f64>,

    sum_evaluations: RwLock<f64>,

    num_visits: AtomicUsize,
}

impl Default for UCTStats {
    fn default() -> Self {
        UCTStats {
            best_evaluation: RwLock::new(std::f64::NEG_INFINITY),
            sum_evaluations: RwLock::new(0f64),
            num_visits: AtomicUsize::new(0),
        }
    }
}

impl UCTStats {
    fn down(&self) {
        self.num_visits.fetch_add(1, Ordering::Relaxed);
    }

    fn up(&self, eval: f64) {
        {
            let mut best = unwrap!(self.best_evaluation.write());
            if eval > *best {
                *best = eval;
            }
        }

        *unwrap!(self.sum_evaluations.write()) += eval;
    }

    fn best_evaluation(&self) -> f64 {
        *unwrap!(self.best_evaluation.read())
    }

    fn sum_evaluations(&self) -> f64 {
        *unwrap!(self.sum_evaluations.read())
    }

    fn num_visits(&self) -> usize {
        self.num_visits.load(Ordering::Relaxed)
    }
}

pub struct TAGPolicy {
    delta: f64,
    topk: usize,
}

impl From<config::TAGConfig> for TAGPolicy {
    fn from(config: config::TAGConfig) -> Self {
        let config::TAGConfig { delta, topk } = config;
        TAGPolicy { delta, topk }
    }
}

impl TAGPolicy {
    /// gives a "score" to a branch of the tree at a given node n_successes is the number of
    /// successes of that branch (that is, the number of leaves that belong to the THRESHOLD
    /// best of that node and which come from that particular branch).
    /// * `n_branch_trials` is the number of trials of that branch (both failed and succeeded),
    /// * `n_trials` is  the number of trials of the node and k the number of branches in the
    ///   node.
    fn heval(
        &self,
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
            let alpha = (2. * (n_trials * n_branches) as f64 / self.delta)
                .ln()
                .max(0.);
            let sqrt_body = alpha * (2. * n_successes as f64 + alpha);
            (n_successes as f64 + alpha + sqrt_body.sqrt()) / n_branch_trials as f64
        }
    }
}

impl TreePolicy for TAGPolicy {
    type EdgeStats = TAGStats;

    fn pick_child(&self, env: &Env<'_>, node: &Node<'_, TAGStats>) -> Option<usize> {
        // Ignore cut children.  Also, we compute the number of visits beforehand to ensure that it
        // doesn't get changed by concurrent accesses.
        let children = node
            .children
            .iter()
            .map(|edge| (edge, edge.stats.num_visits()))
            .enumerate()
            .filter(|(_idx, (edge, _num_visits))| edge.bound() < env.cut)
            .collect::<Vec<_>>();

        // Pick an edge which was not explored yet, if there is some
        NewNodeOrder::WeightedRandom
            .pick_index(
                children
                    .iter()
                    .filter(|(_idx, (_edge, num_visits))| *num_visits == 0)
                    .map(|(idx, (edge, _num_visits))| (*idx, edge.bound())),
                env.cut,
            )
            .or_else(move || {
                // Compute the threshold to use so that we only have `config.topk` children
                let threshold = {
                    let mut evalns = Evaluations::with_capacity(self.topk);
                    for (_idx, (edge, _num_visits)) in &children {
                        // Evaluations are sorted; we can bail out early.
                        for &eval in &*unwrap!(edge.stats.evaluations.read()) {
                            if !evalns.record(eval, self.topk) {
                                break;
                            }
                        }
                    }

                    // It could happen that all edges have num_visits > 0 but still we don't have
                    // any recorded evaluations if none of the descents have finished yet.
                    evalns.max().unwrap_or(std::f64::INFINITY)
                };

                let stats = children
                    .into_iter()
                    .map(|(ix, (edge, num_visits))| {
                        (
                            ix,
                            unwrap!(edge.stats.evaluations.read()).count_lte(threshold),
                            num_visits,
                        )
                    })
                    .collect::<Vec<_>>();

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
                        let score = self.heval(
                            child_successes,
                            child_visits,
                            num_visits,
                            num_children,
                        );
                        (ix, score)
                    })
                    .max_by(|x1, x2| cmp_f64(x1.1, x2.1))
                    .map(|(ix, _)| ix)
            })
            .map(|idx| {
                node.children[idx].stats.down();
                idx
            })
    }

    fn backpropagate(&self, node: &Node<TAGStats>, idx: usize, eval: f64) {
        node.children[idx].stats.up(eval, self.topk)
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
    fn up(&self, eval: f64, topk: usize) {
        unwrap!(self.evaluations.write()).record(eval, topk);
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

    /// Record a new evaluation.  Returns `true` if the evaluation is among the top `topk`.
    fn record(&mut self, eval: f64, topk: usize) -> bool {
        let pos = self
            .0
            .binary_search_by(|&probe| cmp_f64(probe, eval))
            .unwrap_or_else(|e| e);
        if pos < topk {
            if self.0.len() >= topk {
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
