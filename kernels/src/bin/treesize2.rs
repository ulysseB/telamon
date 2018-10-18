extern crate crossbeam;
extern crate env_logger;
#[macro_use]
extern crate log;
extern crate dot;
extern crate rand;
extern crate serde_json;
extern crate stats;

extern crate telamon;
extern crate telamon_kernels as kernels;
extern crate telamon_utils as utils;

use std::borrow::Borrow;
use std::collections::HashMap;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{Deref, Index};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use rand::distributions::{IndependentSample, Weighted, WeightedChoice};
use rand::{thread_rng, ThreadRng};

use stats::Commute;

use telamon::{
    device::{cuda::Gpu, fake::FakeContext, Context},
    explorer::{choice, Candidate},
    helper::TilingPattern,
};

use utils::lazy::Lazy;

/// Newtype wrapper for probabilities.  This represents a floating
/// point number in [0, 1].
#[derive(Copy, Clone, Debug)]
struct Probability(f64);

impl Probability {
    /// Create a new Probability from a number in [0, 1].
    ///
    /// # Panics
    ///
    /// This function will panic if `p` is not in the [0, 1] range.
    fn new(p: f64) -> Self {
        assert!(0f64 <= p && p <= 1f64, "probability must be in [0, 1]");
        Probability(p)
    }

    /// Converts the probability back to a floating point number.
    fn into_f64(self) -> f64 {
        self.0
    }
}

impl From<f64> for Probability {
    fn from(p: f64) -> Probability {
        Probability::new(p)
    }
}

impl Into<f64> for Probability {
    fn into(self) -> f64 {
        self.into_f64()
    }
}

/// Wrapper for a value which was sampled with some probability.
#[derive(Copy, Clone, Debug)]
struct Sampled<T> {
    /// The sampled value.
    pub value: T,

    /// The probability that this value would have been sampled.
    pub probability: Probability,
}

trait Environment {
    type Action: Debug + Clone;
    type State;

    fn list_actions(&self, state: &Self::State) -> Option<Vec<Self::Action>>;

    fn apply_action(
        &self,
        state: &Self::State,
        action: &Self::Action,
    ) -> Option<Self::State>;
}

struct ContextEnvironment<'a, C: Context + 'a> {
    context: &'a C,
    invalid_actions_cnt: AtomicUsize,
}

impl<'a, C: Context + 'a> Environment for ContextEnvironment<'a, C> {
    type Action = choice::ActionEx;
    type State = Candidate<'a>;

    fn list_actions(&self, candidate: &Candidate<'a>) -> Option<Vec<choice::ActionEx>> {
        choice::default_list(&candidate.space).next()
    }

    fn apply_action(
        &self,
        candidate: &Candidate<'a>,
        action: &choice::ActionEx,
    ) -> Option<Candidate<'a>> {
        candidate
            .apply_decision(self.context, action.clone())
            .map_err(|()| self.invalid_actions_cnt.fetch_add(1, Ordering::Relaxed))
            .ok()
    }
}

trait Evaluator<E: Environment> {
    type Evaluation;

    /// Evaluate a new candidate.
    fn evaluate(
        &self,
        state: Arc<E::State>,
        actions: Vec<(E::Action, Option<Arc<E::State>>)>,
        rng: &mut ThreadRng,
    ) -> Self::Evaluation;
}

/// A tree policy represents the policy to use on the already expanded
/// tree.
trait TreePolicy<E: Environment> {
    type EdgePolicyData;

    fn compute_probabilities<B>(
        &self,
        edges: impl ExactSizeIterator<Item = B>,
    ) -> Vec<Self::EdgePolicyData>
    where
        B: TryDeref<Target = E::State>;

    fn sample<B>(
        &self,
        probabilities: impl ExactSizeIterator<Item = Option<B>>,
        rng: &mut ThreadRng,
    ) -> Option<Sampled<EdgeIndex>>
    where
        B: Deref<Target = Self::EdgePolicyData>;
}

trait SearchSpec: Sized {
    type Environment: Environment;
    type Evaluator: Evaluator<Self::Environment>;
    type TreePolicy: TreePolicy<Self::Environment>;
}

type Action<Spec> = <<Spec as SearchSpec>::Environment as Environment>::Action;
type State<Spec> = <<Spec as SearchSpec>::Environment as Environment>::State;
type EdgePolicyData<Spec> = <<Spec as SearchSpec>::TreePolicy as TreePolicy<
    <Spec as SearchSpec>::Environment,
>>::EdgePolicyData;
type Evaluation<Spec> = <<Spec as SearchSpec>::Evaluator as Evaluator<
    <Spec as SearchSpec>::Environment,
>>::Evaluation;

#[derive(Clone, Debug)]
struct CompleteTreeSizeRatioPolicy {
    epsilon: f64,
}

#[derive(Debug)]
struct CompleteTreeSizeRatio {
    weighted: Weighted<Sampled<EdgeIndex>>,
}

struct PolicyEvaluator<P, E> {
    policy: P,
    environment: E,
}

trait TryDeref {
    type Target: ?Sized;

    fn try_deref(&self) -> Option<&Self::Target>;
}

struct Thunk<T, F: FnOnce() -> T> {
    lazy: Lazy<T, F>,
}

impl<T, F: FnOnce() -> T> Thunk<T, F> {
    fn new(fun: F) -> Self {
        Thunk {
            lazy: Lazy::new(Arc::new(fun)),
        }
    }

    fn unwrap(thunk: Self) -> T {
        thunk
            .lazy
            .into_inner(|f| Arc::try_unwrap(f).ok().unwrap()())
    }
}

struct ThunkRef<'a, T: 'a, F: FnOnce() -> Option<T> + 'a>(&'a Thunk<Option<T>, F>);

impl<'a, T: 'a, F: FnOnce() -> Option<T> + 'a> TryDeref for ThunkRef<'a, T, F> {
    type Target = T;

    fn try_deref(&self) -> Option<&T> {
        self.0.try_deref()
    }
}

impl<'a, T, F: FnOnce() -> Option<T>> TryDeref for Thunk<Option<T>, F> {
    type Target = T;

    fn try_deref(&self) -> Option<&T> {
        self.lazy
            .force(|f| Arc::try_unwrap(f).ok().unwrap()())
            .as_ref()
    }
}

impl<T> TryDeref for Option<T>
where
    T: Deref,
{
    type Target = <T as Deref>::Target;

    fn try_deref(&self) -> Option<&Self::Target> {
        self.as_ref().map(Deref::deref)
    }
}

impl<E: Environment, P: TreePolicy<E>> Evaluator<E> for PolicyEvaluator<P, E> {
    type Evaluation = Option<f64>;

    fn evaluate(
        &self,
        _state: Arc<E::State>,
        actions: Vec<(E::Action, Option<Arc<E::State>>)>,
        rng: &mut ThreadRng,
    ) -> Option<f64> {
        if actions.len() == 0 {
            trace!("Terminal node evaluated.");

            // Terminal node, always has size 1.
            return 1f64.into();
        }

        let (mut estimate, mut candidate);
        {
            let probabilities = self.policy.compute_probabilities(actions.iter().map(
                |(_, candidate)| candidate.as_ref().map(|candidate| candidate.deref()),
            ));
            if let Some(sampled) = self.policy.sample(probabilities.iter().map(Some), rng)
            {
                if let Some(cand) = actions[sampled.value.0].1.as_ref().map(Arc::clone) {
                    candidate = cand;
                    estimate = sampled.probability.into_f64().recip();
                } else {
                    // The selected child was a deadend.
                    return None;
                }
            } else {
                // All children were dead
                return None;
            }
        }

        loop {
            let choice = self.environment.list_actions(&candidate);
            if let Some(choice) = choice {
                let arc = Arc::new(candidate);
                let mut candidates = choice
                    .into_iter()
                    .map(|action| {
                        let env_ref = &self.environment;
                        let clone = Arc::clone(&arc);
                        Thunk::new(move || env_ref.apply_action(&clone, &action))
                    }).collect::<Vec<_>>();

                let probabilities = self
                    .policy
                    .compute_probabilities(candidates.iter().map(ThunkRef));
                if let Some(sampled) =
                    self.policy.sample(probabilities.iter().map(Some), rng)
                {
                    match Thunk::unwrap(
                        candidates.swap_remove(sampled.value.into_usize()),
                    ) {
                        Some(c) => {
                            candidate = Arc::new(c);
                            estimate *= sampled.probability.into_f64().recip();
                        }
                        None => {
                            // The sampled subtree was empty; this is a
                            // dead end.
                            return None;
                        }
                    }
                } else {
                    // All subtrees were empty; this is a dead end.
                    return None;
                }
            } else {
                // Terminal node reached.
                return estimate.into();
            }
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct UniformPolicy {}

impl<E: Environment> TreePolicy<E> for UniformPolicy {
    type EdgePolicyData = bool;

    fn sample<B>(
        &self,
        ratios: impl ExactSizeIterator<Item = Option<B>>,
        rng: &mut ThreadRng,
    ) -> Option<Sampled<EdgeIndex>>
    where
        B: Deref<Target = bool>,
    {
        let mut weighted = ratios
            .enumerate()
            .flat_map(|(index, b)| {
                if b.map(|x| *x).unwrap_or(false) {
                    Some(Weighted {
                        item: EdgeIndex(index),
                        weight: 1u32,
                    })
                } else {
                    None
                }
            }).collect::<Vec<_>>();

        if weighted.is_empty() {
            return None;
        }

        let p = 1f64 / weighted.len() as f64;

        Some(Sampled {
            value: WeightedChoice::new(&mut weighted).ind_sample(rng),
            probability: Probability(p),
        })
    }

    fn compute_probabilities<B>(
        &self,
        candidates: impl ExactSizeIterator<Item = B>,
    ) -> Vec<bool>
    where
        B: TryDeref<Target = E::State>,
    {
        candidates.map(|b| b.try_deref().is_some()).collect()
    }
}

impl<'a, E: Environment<State = Candidate<'a>>> TreePolicy<E>
    for CompleteTreeSizeRatioPolicy
{
    type EdgePolicyData = CompleteTreeSizeRatio;

    fn sample<B>(
        &self,
        ratios: impl ExactSizeIterator<Item = Option<B>>,
        rng: &mut ThreadRng,
    ) -> Option<Sampled<EdgeIndex>>
    where
        B: Deref<Target = CompleteTreeSizeRatio>,
    {
        let mut probas = ratios
            .flat_map(|ratio| ratio.map(|ratio| ratio.deref().weighted))
            .collect::<Vec<_>>();

        let total_proba = probas
            .iter()
            .map(|weighted| weighted.item.probability.0)
            .sum::<f64>();

        if probas.len() == 0 {
            return None;
        }

        let sample = WeightedChoice::new(&mut probas).ind_sample(rng);

        Some(Sampled {
            probability: Probability::new(sample.probability.0 / total_proba),
            ..sample
        })
    }

    fn compute_probabilities<B>(
        &self,
        candidates: impl ExactSizeIterator<Item = B>,
    ) -> Vec<CompleteTreeSizeRatio>
    where
        B: TryDeref<Target = E::State>,
    {
        let log_weights = candidates
            .map(|candidate| {
                candidate
                    .try_deref()
                    .map(|candidate| {
                        choice::default_list(&candidate.space)
                            .map(|choice| (choice.len() as f64).ln())
                            .sum::<f64>()
                    }).unwrap_or(std::f64::NEG_INFINITY)
            }).collect::<Vec<_>>();

        // Do not count empty subtrees, but keep them in the vector so
        // that we can map the CompleteTreeSizeRatio to the input
        // Candidates.
        let len = log_weights.iter().filter(|x| x.is_finite()).count();

        if len == 0 {
            // If all subtrees are empty, we can select any of them --
            // we will end up in a deadend anyways.
            let p = 1f64 / log_weights.len() as f64;

            return log_weights
                .into_iter()
                .enumerate()
                .map(|(index, _)| CompleteTreeSizeRatio {
                    weighted: Weighted {
                        item: Sampled {
                            value: EdgeIndex(index),
                            probability: Probability::new(p),
                        },
                        weight: 1u32,
                    },
                }).collect();
        }

        // Use log sum exp trick for better accuracy when computing
        // the total weight.
        let max_log_weight = log_weights
            .iter()
            .cloned()
            .fold(std::f64::NEG_INFINITY, f64::max);
        let log_total_weight = max_log_weight + log_weights
            .iter()
            .map(|&log_weight| (log_weight - max_log_weight).exp())
            .sum::<f64>()
            .ln();

        // Scale the epsilon according to the number of samples so
        // that we are actually mixing with an uniform distribution.
        let epsilon = self.epsilon / len as f64;

        // The sampling procedure uses u32 so we need to ensure the
        // total sum of weights can fit.
        let resolution = (u32::max_value() / len as u32) as f64;

        log_weights
            .into_iter()
            .enumerate()
            .map(|(index, log_weight)| {
                let proba = if log_weight.is_finite() {
                    (log_weight - log_total_weight).exp() * (1f64 - self.epsilon)
                        + epsilon
                } else {
                    0f64
                };

                CompleteTreeSizeRatio {
                    weighted: Weighted {
                        item: Sampled {
                            value: EdgeIndex(index),
                            probability: Probability::new(proba),
                        },
                        weight: (proba * resolution) as u32,
                    },
                }
            }).collect()
    }
}

struct AtomicF64(AtomicUsize);

impl AtomicF64 {
    pub fn new(val: f64) -> Self {
        AtomicF64(AtomicUsize::new(val.to_bits() as usize))
    }

    pub fn load(&self) -> f64 {
        f64::from_bits(self.0.load(Ordering::Relaxed) as u64)
    }

    pub fn try_add(&self, val: f64) -> Result<f64, ()> {
        let cur = self.0.load(Ordering::Relaxed);
        let new = (f64::from_bits(cur as u64) + val).to_bits() as usize;
        if self.0.compare_and_swap(cur, new, Ordering::Relaxed) == cur {
            Ok(f64::from_bits(cur as u64))
        } else {
            Err(())
        }
    }
}

/// A node in the tree portion of the search.
struct Node<Spec: SearchSpec> {
    /// The outgoing edges for that node.
    edges: Edges<Edge<Spec>>,

    /// Whether this is a terminal node.
    terminal: bool,

    dead: AtomicBool,

    /// The cumulated score value.
    total_estimate: AtomicF64,

    /// The number of times this node has been visited.  Deadends are
    /// counted as visits.
    num_visits: AtomicUsize,

    /// The number of deadends encountered when visiting this node.
    num_deadends: AtomicUsize,
}

// gviz
struct NodeInfo {
    estimate: f64,
    num_visits: usize,
    num_deadends: usize,
    truncated: bool,
    terminal: bool,
    deadend: bool,
    explored: bool,
}

struct EdgeInfo<Spec: SearchSpec> {
    action: Action<Spec>,
}

struct TreeInfo<Spec: SearchSpec> {
    nodes: Vec<NodeInfo>,
    edges: Vec<(usize, usize, EdgeInfo<Spec>)>,
}

impl<Spec: SearchSpec> TreeInfo<Spec> {
    fn new(node: &Node<Spec>, min_visits: usize) -> Self {
        let mut worklist = vec![node];
        let mut node_infos = Vec::new();
        let mut edge_infos = Vec::new();

        while let Some(node) = worklist.pop() {
            let num_visits = node.num_visits.load(Ordering::Relaxed);
            let num_deadends = node.num_deadends.load(Ordering::Relaxed);

            node_infos.push((
                node as *const _ as usize,
                NodeInfo {
                    terminal: node.terminal,
                    deadend: node.is_dead(),
                    estimate: node.total_estimate.load() / num_visits as f64,
                    truncated: num_visits < min_visits,
                    explored: num_visits > 0,
                    num_visits: num_visits,
                    num_deadends: num_deadends,
                },
            ));
            /*
            if node.is_dead() {
                assert!(node.edges.len() != 0);
            }
*/

            if num_visits >= min_visits {
                for edge in node.edges.iter() {
                    if let Some(dst) = edge.dst.get() {
                        worklist.push(dst);

                        edge_infos.push((
                            (node as *const _ as usize, dst as *const _ as usize),
                            EdgeInfo {
                                action: edge.action.clone(),
                            },
                        ));
                    } else {
                        node_infos.push((
                            edge as *const _ as usize,
                            NodeInfo {
                                terminal: false,
                                deadend: false,
                                estimate: std::f64::NAN,
                                truncated: true,
                                explored: false,
                                num_visits: 0,
                                num_deadends: 0,
                            },
                        ));

                        edge_infos.push((
                            (node as *const _ as usize, edge as *const _ as usize),
                            EdgeInfo {
                                action: edge.action.clone(),
                            },
                        ));
                    }
                }
            }
        }

        let mut node_index = HashMap::new();
        for (index, (nid, _)) in node_infos.iter().enumerate() {
            node_index.insert(*nid, index);
        }

        TreeInfo {
            nodes: node_infos.into_iter().map(|(_, info)| info).collect(),
            edges: edge_infos
                .into_iter()
                .map(|(eid, einfo)| (node_index[&eid.0], node_index[&eid.1], einfo))
                .collect(),
        }
    }
}

type Nd<'a> = (usize, &'a NodeInfo);
type Ed<'a, Spec> = &'a (usize, usize, EdgeInfo<Spec>);

impl<'a, Spec: SearchSpec> dot::Labeller<'a, Nd<'a>, Ed<'a, Spec>> for TreeInfo<Spec> {
    fn graph_id(&'a self) -> dot::Id<'a> {
        dot::Id::new("example2").unwrap()
    }

    fn node_id(&'a self, n: &Nd<'a>) -> dot::Id<'a> {
        dot::Id::new(format!("N{}", n.0)).unwrap()
    }

    fn node_label<'b>(&'b self, n: &Nd<'a>) -> dot::LabelText<'b> {
        if !n.1.explored {
            dot::LabelText::label("?")
        } else {
            dot::LabelText::label(format!("{:.2e}", n.1.estimate))
                .suffix_line(dot::LabelText::label(format!("visits: {}", n.1.num_visits)))
                .suffix_line(dot::LabelText::label(format!(
                    "deadends: {}",
                    n.1.num_deadends
                )))
        }
    }

    fn node_style(&'a self, n: &Nd<'a>) -> dot::Style {
        if n.1.deadend {
            dot::Style::Filled
        } else if n.1.terminal {
            dot::Style::Bold
        } else if n.1.truncated {
            dot::Style::Dotted
        } else {
            dot::Style::None
        }
    }

    fn edge_label<'b>(&'b self, e: &Ed<'a, Spec>) -> dot::LabelText<'b> {
        dot::LabelText::LabelStr(format!("{:?}", e.2.action).into())
    }
}

impl<'a, Spec: SearchSpec> dot::GraphWalk<'a, Nd<'a>, Ed<'a, Spec>> for TreeInfo<Spec> {
    fn nodes(&'a self) -> dot::Nodes<'a, Nd<'a>> {
        self.nodes.iter().enumerate().collect()
    }

    fn edges(&'a self) -> dot::Edges<'a, Ed<'a, Spec>> {
        self.edges.iter().collect()
    }

    fn source(&'a self, e: &Ed<'a, Spec>) -> Nd<'a> {
        (e.0, &self.nodes[e.0])
    }

    fn target(&'a self, e: &Ed<'a, Spec>) -> Nd<'a> {
        (e.1, &self.nodes[e.1])
    }
}

impl<Spec: SearchSpec> Node<Spec> {
    /// Create a new node given its edges.
    fn new(edges: Vec<Edge<Spec>>) -> Self {
        Node {
            edges: edges.into(),
            terminal: false,
            dead: AtomicBool::new(false),
            total_estimate: AtomicF64::new(0f64),
            num_visits: AtomicUsize::new(0usize),
            num_deadends: AtomicUsize::new(0usize),
        }
    }

    fn terminal() -> Self {
        Node {
            terminal: true,
            ..Self::new(Vec::new())
        }
    }

    fn deadend() -> Self {
        Node {
            dead: AtomicBool::new(true),
            total_estimate: AtomicF64::new(837483748f64),
            ..Self::new(Vec::new())
        }
    }

    fn is_dead(&self) -> bool {
        self.dead.load(Ordering::Relaxed)
    }
}

/// An edge between nodes in the tree portion of the search.
struct Edge<Spec: SearchSpec> {
    action: Action<Spec>,

    /// The target node.  This is either an actual node (the common
    /// case), or a state which can be used to initialize the node.
    dst: Lazy<Node<Spec>, State<Spec>>,

    /// Some local data for the tree policy to use.
    policy_data: EdgePolicyData<Spec>,
}

/// A newtype wrapper representing a set of edges.  The API exposed by
/// Edges over the underlying vector is minimal on purpose and only
/// allows iteration and immutable indexing.
struct Edges<E>(Vec<E>);

impl<E> Edges<E> {
    /// Return the number of edges.
    fn len(&self) -> usize {
        self.0.len()
    }

    /// An iterator over all the edges.
    fn iter(&self) -> impl ExactSizeIterator<Item = &E> {
        self.0.iter()
    }
}

/// A newtype wrapper to indicate that an integer is an index into the
/// outgoing edges of a node.
#[derive(Copy, Clone, Debug)]
struct EdgeIndex(usize);

impl EdgeIndex {
    fn into_usize(self) -> usize {
        self.0
    }
}

impl<E> Index<EdgeIndex> for Edges<E> {
    type Output = E;

    fn index(&self, index: EdgeIndex) -> &E {
        &self.0[index.0]
    }
}

impl<E> From<Vec<E>> for Edges<E> {
    fn from(edges: Vec<E>) -> Self {
        Edges(edges)
    }
}

/// A path in the search tree down to a selected leaf.  The leaf has
/// just been expanded and was typically never evaluated before.
struct TreePath<'a, Spec: SearchSpec + 'a> {
    /// Path to the selected leaf.
    pub path: Vec<(&'a Node<Spec>, Sampled<EdgeIndex>)>,

    /// The selected leaf, which has already been expanded.
    pub leaf: &'a Node<Spec>,
}

struct ExpansionResult<Spec: SearchSpec> {
    state: Arc<State<Spec>>,
    actions: Vec<(Action<Spec>, Option<Arc<State<Spec>>>)>,
}

impl<Spec: SearchSpec> ExpansionResult<Spec> {
    fn new(
        state: Arc<State<Spec>>,
        actions: Vec<(Action<Spec>, Option<Arc<State<Spec>>>)>,
    ) -> Self {
        ExpansionResult { state, actions }
    }

    fn terminal(state: Arc<State<Spec>>) -> Self {
        ExpansionResult {
            state,
            actions: Vec::new().into(),
        }
    }
}

struct BackpropResult {
    depth: u64,
    estimate: Option<f64>,
}

struct Tree<Spec: SearchSpec> {
    _spec: Spec,
    policy: Spec::TreePolicy,
    environment: Spec::Environment,
    evaluator: Spec::Evaluator,
}

impl<Spec: SearchSpec> Tree<Spec>
where
    Spec::Evaluator: Evaluator<Spec::Environment, Evaluation = Option<f64>>,
{
    fn selection_expansion_steps<'a>(
        &'a self,
        mut node: &'a Node<Spec>,
        rng: &mut ThreadRng,
    ) -> (TreePath<'a, Spec>, Option<ExpansionResult<Spec>>) {
        trace!("Starting selection");

        assert!(node.edges.len() > 0);

        let mut path = Vec::with_capacity(64);

        loop {
            // We reached a terminal leaf or deadend.
            if node.is_dead() || node.terminal {
                return (TreePath { path, leaf: node }, None);
            }

            // We only sample non-dead children, and we let the policy
            // tell us when all our children are dead.
            let sampled = match self.policy.sample(
                node.edges.iter().map(|edge| {
                    if edge.dst.get().map(|node| node.is_dead()).unwrap_or(false) {
                        None
                    } else {
                        Some(&edge.policy_data)
                    }
                }),
                rng,
            ) {
                None => {
                    assert!(node.edges.len() != 0);
                    node.dead.store(true, Ordering::Relaxed);

                    return (TreePath { path, leaf: node }, None);
                }
                Some(sampled) => sampled,
            };

            let edge = &node.edges[sampled.value];
            path.push((node, sampled));

            let mut expansion_result = None;
            let dst = edge.dst.force(|candidate| {
                let (node, expansion) = self.expansion_step(candidate);
                expansion_result = Some(expansion);
                node
            });

            if let Some(expansion) = expansion_result {
                return (TreePath { path, leaf: dst }, Some(expansion));
            }

            assert!(!node.is_dead());

            node = dst;
        }
    }

    fn expansion_step(
        &self,
        state: Arc<State<Spec>>,
    ) -> (Node<Spec>, ExpansionResult<Spec>) {
        trace!("Starting expansion");

        match self.environment.list_actions(state.borrow()) {
            None => (Node::terminal(), ExpansionResult::terminal(state)),
            Some(actions) => {
                assert!(actions.len() > 0);

                let children = actions
                    .into_iter()
                    .map(|action| {
                        let child =
                            self.environment.apply_action(state.borrow(), &action);
                        (action, child)
                    }).collect::<Vec<_>>();

                let probas = self.policy.compute_probabilities(
                    children.iter().map(|(_, state)| state.as_ref()),
                );

                let mut edges = Vec::with_capacity(children.len());
                let mut actions = Vec::with_capacity(children.len());

                for ((action, child), proba) in children.into_iter().zip(probas) {
                    match child {
                        Some(child) => {
                            let child = Arc::new(child);
                            edges.push(Edge {
                                action: action.clone(),
                                dst: Lazy::new(Arc::clone(&child)),
                                policy_data: proba,
                            });

                            actions.push((action, Some(child)))
                        }
                        None => {
                            edges.push(Edge {
                                action: action.clone(),
                                dst: Lazy::from_val(Node::deadend()),
                                policy_data: proba,
                            });

                            actions.push((action, None));
                        }
                    }
                }

                (Node::new(edges), ExpansionResult::new(state, actions))
            }
        }
    }

    fn simulation_step(
        &self,
        expanded: ExpansionResult<Spec>,
        rng: &mut ThreadRng,
    ) -> Evaluation<Spec> {
        trace!("Starting simulation");

        self.evaluator
            .evaluate(expanded.state, expanded.actions, rng)
    }

    fn backpropagation_step(
        &self,
        path: TreePath<'_, Spec>,
        mut estimate: Evaluation<Spec>,
    ) -> BackpropResult {
        trace!("Starting backpropagation");

        let depth = (path.path.len() + 1) as u64;

        path.leaf.num_visits.fetch_add(1, Ordering::Relaxed);
        match estimate {
            None => {
                path.leaf.num_deadends.fetch_add(1, Ordering::Relaxed);
            }
            Some(estimate) => {
                while path.leaf.total_estimate.try_add(estimate).is_err() {}
            }
        }

        for (node, edge_sample) in path.path.into_iter().rev() {
            node.num_visits.fetch_add(1, Ordering::Relaxed);

            match estimate {
                None => {
                    node.num_deadends.fetch_add(1, Ordering::Relaxed);
                }
                Some(ref mut estimate) => {
                    *estimate *= edge_sample.probability.into_f64().recip();

                    while node.total_estimate.try_add(*estimate).is_err() {}
                }
            }
        }

        BackpropResult { depth, estimate }
    }

    fn playout(&self, root: &Node<Spec>, rng: &mut ThreadRng) -> Option<BackpropResult> {
        loop {
            let (path, expanded) = self.selection_expansion_steps(root, rng);

            let result = if let Some(expanded) = expanded {
                self.simulation_step(expanded, rng)
            } else {
                // We reached an already expanded terminal node (which
                // may be a dead end).
                if path.leaf.terminal {
                    Some(1f64)
                } else {
                    None
                }
            };

            return Some(self.backpropagation_step(path, result));
        }
    }
}

struct TreeSizeEstimation<'a, C: Context + 'a, P: TreePolicy<ContextEnvironment<'a, C>>>(
    PhantomData<(&'a (), fn(C, P))>,
);

impl<'a, C: Context + 'a, P: TreePolicy<ContextEnvironment<'a, C>>>
    TreeSizeEstimation<'a, C, P>
{
    fn new() -> Self {
        TreeSizeEstimation(PhantomData)
    }
}

impl<'a, C: Context + 'a, P: TreePolicy<ContextEnvironment<'a, C>>> SearchSpec
    for TreeSizeEstimation<'a, C, P>
{
    type Environment = ContextEnvironment<'a, C>;
    type Evaluator = PolicyEvaluator<P, ContextEnvironment<'a, C>>;
    type TreePolicy = P;
}

use kernels::{linalg, Kernel};

fn main() {
    env_logger::init();

    let proba = CompleteTreeSizeRatioPolicy { epsilon: 0.1f64 };
    let proba = UniformPolicy {};
    let num_playouts = 10_000;

    let gpu: Gpu = serde_json::from_reader(
        &std::fs::File::open("/home/elarnon/.config/telamon/cuda_gpus.json").unwrap(),
    ).unwrap();

    let estimates = linalg::MatMul::<f32>::with_candidates(
        linalg::MatMulP::new(1024, 1024, 1024)
            .tile_m(TilingPattern::new_fixed(&[32, 4]))
            .tile_n(TilingPattern::new_fixed(&[32, 4]))
            .tile_k(TilingPattern::new_fixed(&[32])),
        true,
        &mut FakeContext::new(gpu),
        move |mut candidates, context| {
            let tree = Tree {
                _spec: TreeSizeEstimation::new(),
                environment: ContextEnvironment {
                    context: context,
                    invalid_actions_cnt: AtomicUsize::new(0),
                },
                policy: proba.clone(),
                evaluator: PolicyEvaluator {
                    environment: ContextEnvironment {
                        context: context,
                        invalid_actions_cnt: AtomicUsize::new(0),
                    },
                    policy: proba,
                },
            };

            let (mut root, _) = tree.expansion_step(Arc::new(candidates.swap_remove(0)));

            // TODO: parallel
            let depths = Mutex::new(stats::OnlineStats::new());
            let estimates = Mutex::new(stats::OnlineStats::new());

            let playouts_done = AtomicUsize::new(0);

            crossbeam::scope(|scope| {
                for ix in 0..8 {
                    scope
                        .builder()
                        .name(format!("Telamon - Search Thread #{}", ix))
                        .spawn(|| {
                            let rng = &mut thread_rng();

                            let mut thread_depths = stats::OnlineStats::new();
                            let mut thread_estimates = stats::OnlineStats::new();

                            while playouts_done.fetch_add(1, Ordering::Relaxed)
                                < num_playouts
                            {
                                if let Some(result) = tree.playout(&root, rng) {
                                    if let Some(estimate) = result.estimate {
                                        thread_depths.add(result.depth);
                                        thread_estimates.add(estimate);
                                    } else {
                                        thread_depths.add_null();
                                        thread_estimates.add_null();
                                    }
                                }
                            }

                            depths.lock().unwrap().merge(thread_depths);
                            estimates.lock().unwrap().merge(thread_estimates);
                        }).unwrap();
                }
            });

            let estimates = estimates.into_inner().unwrap();
            let depths = depths.into_inner().unwrap();

            println!(
                "Average depth {:.2e} with stddev {:.2e}",
                depths.mean(),
                depths.stddev(),
            );

            println!(
                "Estimated {:.2e} with stddev {:.2e}",
                estimates.mean(),
                estimates.stddev()
            );

            let info = TreeInfo::new(&root, num_playouts / 10);
            let mut f = std::fs::File::create("out.dot").unwrap();
            dot::render(&info, &mut f).unwrap();

            // TODO
            vec![root.total_estimate.load() / ((*root.num_visits.get_mut()) as f64)]
        },
    );

    println!(
        "Got avg of {}",
        estimates.iter().sum::<f64>() / estimates.len() as f64
    );
}
