extern crate crossbeam;
extern crate csv;
extern crate env_logger;
extern crate structopt;
#[macro_use]
extern crate log;
extern crate dot;
extern crate indicatif;
extern crate rand;
extern crate rayon;
#[macro_use]
extern crate serde_derive;
extern crate rpds;
extern crate serde_json;
extern crate serde_yaml;
extern crate stats;

extern crate telamon;
extern crate telamon_kernels as kernels;
extern crate telamon_utils as utils;

use std::borrow::Borrow;
use std::collections::{hash_map::Entry, BinaryHeap, HashMap};
use std::fmt::Debug;
use std::iter::FromIterator;
use std::marker::PhantomData;
use std::ops::{Deref, Index};
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::{thread, time};

use indicatif::ProgressBar;
use rand::distributions::{Bernoulli, Weighted, WeightedChoice};
use rand::prelude::*;
use rayon::join;
use rpds::List;
use structopt::StructOpt;

use telamon::{
    device::{cuda::Gpu, fake::FakeContext, ArgMap, Context},
    explorer::{choice, config, Candidate},
    helper::TilingPattern,
};

use utils::atomic::AtomicF64;
use utils::ops::TryDeref;
use utils::sync::{Lazy, Thunk};

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
}

impl From<Probability> for f64 {
    fn from(p: Probability) -> f64 {
        p.0
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

trait AsF64 {
    fn as_f64(&self) -> f64;
}

impl AsF64 for f64 {
    fn as_f64(&self) -> f64 {
        *self
    }
}

trait HasData {
    type Data: AsF64 + Default + Clone;

    fn get_data(&self) -> Self::Data;
}

impl<'a> HasData for Candidate<'a> {
    type Data = f64;

    fn get_data(&self) -> Self::Data {
        self.bound.value()
    }
}

trait Environment {
    type Action: Debug + Clone;
    type State: HasData;

    fn list_actions(&self, state: &Self::State) -> Option<Vec<Self::Action>>;

    fn apply_action(
        &self,
        state: &Self::State,
        action: &Self::Action,
    ) -> Option<Self::State>;
}

struct ContextEnvironment<'a, C: Context + 'a> {
    context: &'a C,
    ordering: choice::ChoiceOrdering,
    invalid_actions_cnt: AtomicUsize,
}

trait Depth {
    fn depth(&self) -> usize;
}

impl<'a> Depth for Candidate<'a> {
    fn depth(&self) -> usize {
        self.actions.len()
    }
}

impl<'a, C: Context + 'a> Environment for ContextEnvironment<'a, C> {
    type Action = choice::ActionEx;
    type State = Candidate<'a>;

    fn list_actions(&self, candidate: &Candidate<'a>) -> Option<Vec<choice::ActionEx>> {
        choice::list(self.ordering.iter(), &candidate.space).next()
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

    fn environment(&self) -> &Self::Environment;
    fn evaluator(&self) -> &Self::Evaluator;
    fn policy(&self) -> &Self::TreePolicy;
}

trait SearchSpecExt: SearchSpec {
    fn expansion_step(
        &self,
        state: Arc<State<Self>>,
    ) -> (Node<Self>, ExpansionResult<Self>) {
        trace!("Starting expansion");

        match self.environment().list_actions(state.borrow()) {
            None => (
                Node::terminal(state.get_data()),
                ExpansionResult::terminal(state),
            ),
            Some(actions) => {
                assert!(!actions.is_empty());

                let children = actions
                    .into_iter()
                    .map(|action| {
                        let child =
                            self.environment().apply_action(state.borrow(), &action);
                        (action, child)
                    }).collect::<Vec<_>>();

                let probas = self.policy().compute_probabilities(
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
                                dst: Lazy::from_val(Node::deadend(Default::default())),
                                policy_data: proba,
                            });

                            actions.push((action, None));
                        }
                    }
                }

                (
                    Node::new(state.get_data(), edges),
                    ExpansionResult::new(state, actions),
                )
            }
        }
    }
}

impl<Spec: SearchSpec> SearchSpecExt for Spec {}

type Action<Spec> = <<Spec as SearchSpec>::Environment as Environment>::Action;
type State<Spec> = <<Spec as SearchSpec>::Environment as Environment>::State;
type EdgePolicyData<Spec> = <<Spec as SearchSpec>::TreePolicy as TreePolicy<
    <Spec as SearchSpec>::Environment,
>>::EdgePolicyData;
type Evaluation<Spec> = <<Spec as SearchSpec>::Evaluator as Evaluator<
    <Spec as SearchSpec>::Environment,
>>::Evaluation;
type NodeData<Spec> = <State<Spec> as HasData>::Data;

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

impl<E: Environment, P: TreePolicy<E>> Evaluator<E> for PolicyEvaluator<P, E> {
    type Evaluation = (bool, f64);

    fn evaluate(
        &self,
        _state: Arc<E::State>,
        actions: Vec<(E::Action, Option<Arc<E::State>>)>,
        rng: &mut ThreadRng,
    ) -> (bool, f64) {
        if actions.len() == 0 {
            trace!("Terminal node evaluated.");

            // Terminal node, always has size 1.
            return (true, 1f64);
        }

        let (mut proba, mut candidate);
        {
            let probabilities = self.policy.compute_probabilities(actions.iter().map(
                |(_, candidate)| candidate.as_ref().map(|candidate| candidate.deref()),
            ));
            if let Some(sampled) = self.policy.sample(probabilities.iter().map(Some), rng)
            {
                if let Some(cand) = actions[sampled.value.0].1.as_ref().map(Arc::clone) {
                    candidate = cand;
                    proba = f64::from(sampled.probability);
                } else {
                    // The selected child was a deadend.
                    return (false, f64::from(sampled.probability));
                }
            } else {
                // All children were dead
                return (false, 1f64);
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
                    .compute_probabilities(candidates.iter().map(Thunk::as_ref));
                if let Some(sampled) =
                    self.policy.sample(probabilities.iter().map(Some), rng)
                {
                    match Thunk::unwrap(
                        candidates.swap_remove(sampled.value.into_usize()),
                    ) {
                        Some(c) => {
                            candidate = Arc::new(c);
                            proba *= f64::from(sampled.probability);
                        }
                        None => {
                            // The sampled subtree was empty; this is a
                            // dead end.
                            return (false, proba * f64::from(sampled.probability));
                        }
                    }
                } else {
                    // All subtrees were empty; this is a dead end.
                    return (false, proba);
                }
            } else {
                // Terminal node reached.
                return (true, proba);
            }
        }
    }
}

trait Stratifier<E: Environment> {
    type Strate: Hash + Eq + Ord + Clone;

    fn strate(&self, state: &E::State) -> Self::Strate;
}

struct MyStratifier;

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
struct OrdF64(f64);

impl OrdF64 {
    fn new(x: f64) -> Option<Self> {
        if x.is_nan() {
            None
        } else {
            Some(OrdF64(x))
        }
    }
}

impl Eq for OrdF64 {}

impl Hash for OrdF64 {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        f64::to_bits(self.0).hash(state)
    }
}

impl Ord for OrdF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl<'a, E: Environment<State = Candidate<'a>>> Stratifier<E> for MyStratifier {
    type Strate = (usize, u64);

    fn strate(&self, state: &E::State) -> Self::Strate {
        /*
        let log_weight = choice::default_list(&state.space)
            .map(|choice| (choice.len() as f64).ln())
            .sum::<f64>();
            */
        (
            usize::max_value() - state.actions.len(),
            choice::default_list(&state.space).count() as u64
            // log_weight.to_bits() >> (std::f64::MANTISSA_DIGITS - 6),
        )
    }
}

struct StratifiedEvaluator<E, P, S> {
    environment: E,
    policy: P,
    stratifier: S,
}

struct Stratification<'a, E: Environment + 'a, S: Stratifier<E> + 'a> {
    environment: &'a E,
    stratifier: &'a S,
    queue: HashMap<S::Strate, (E::State, f64)>,
    heap: BinaryHeap<S::Strate>,
    pub total: f64,
}

impl<'a, E: Environment, S: Stratifier<E> + 'a> Stratification<'a, E, S> {
    fn new(environment: &'a E, stratifier: &'a S) -> Self {
        Stratification {
            environment,
            stratifier,
            queue: HashMap::new(),
            heap: BinaryHeap::new(),
            total: 0.,
        }
    }

    fn process<A: Borrow<E::State>>(
        &mut self,
        state: &A,
        weight: f64,
        rng: &mut impl Rng,
    ) {
        if let Some(actions) = self.environment.list_actions(state.borrow()) {
            for action in actions {
                if let Some(child) =
                    self.environment.apply_action(state.borrow(), &action)
                {
                    let strate = self.stratifier.strate(&child);
                    self.push(strate, child, weight, rng);
                }
            }
        } else {
            self.total += weight;
        }
    }

    fn push(
        &mut self,
        strate: S::Strate,
        state: E::State,
        weight: f64,
        rng: &mut impl Rng,
    ) {
        match self.queue.entry(strate) {
            Entry::Occupied(mut entry) => {
                let (ref mut s_state, ref mut s_weight) = entry.get_mut();
                *s_weight += weight;

                if Bernoulli::new(weight / *s_weight).sample(rng) {
                    *s_state = state;
                }
            }
            Entry::Vacant(entry) => {
                self.heap.push(entry.key().clone());
                entry.insert((state, weight));
            }
        }
    }

    fn pop(&mut self) -> Option<(E::State, f64)> {
        self.heap.pop().map(|strate| {
            self.queue
                .remove(&strate)
                .expect("Missing strate in the queue.")
        })
    }
}

impl<E: Environment, P: TreePolicy<E>, S: Stratifier<E>> Evaluator<E>
    for StratifiedEvaluator<E, P, S>
{
    type Evaluation = (bool, f64);

    fn evaluate(
        &self,
        state: Arc<E::State>,
        _actions: Vec<(E::Action, Option<Arc<E::State>>)>,
        rng: &mut ThreadRng,
    ) -> Self::Evaluation {
        let mut stratification = Stratification::new(&self.environment, &self.stratifier);

        stratification.process(&state, 1., rng);

        while let Some((state, weight)) = stratification.pop() {
            stratification.process(&state, weight, rng);
        }

        (true, stratification.total.recip())
    }
}

struct PartialBacktrackEvaluator<P, E> {
    policy: P,
    environment: E,
}

impl<E: Environment, P: TreePolicy<E>> Evaluator<E> for PartialBacktrackEvaluator<P, E>
where
    E::State: Depth,
{
    type Evaluation = (bool, f64);

    fn evaluate(
        &self,
        state: Arc<E::State>,
        actions: Vec<(E::Action, Option<Arc<E::State>>)>,
        rng: &mut ThreadRng,
    ) -> (bool, f64) {
        if actions.len() == 0 {
            trace!("Terminal node evaluated.");

            return (true, 1f64);
        }

        let probabilities =
            self.policy
                .compute_probabilities(actions.iter().map(|(_, candidate)| {
                    candidate.as_ref().map(|candidate| candidate.deref())
                }));

        // TODO: capacity
        let mut worklist = Vec::new();
        let mut num_terminal = 0;
        let mut num_dead = 0;
        let mut estimate = 0f64;
        let depth = state.depth();
        let num_samples = if depth > 100 { 2 } else { 1 };

        for sampled in (0..num_samples)
            .map(|_| self.policy.sample(probabilities.iter().map(Some), rng))
            .filter_map(|x| x)
        {
            let proba = f64::from(sampled.probability);

            if let Some(cand) = actions[sampled.value.0].1.as_ref().map(Arc::clone) {
                worklist.push((cand, proba, depth + 1));
            } else {
                // The selected child was a deadend
                num_dead += 1;
            }
        }

        while let Some((candidate, proba, depth)) = worklist.pop() {
            let num_samples = if depth > 100 { 2 } else { 1 };

            let choice = self.environment.list_actions(&candidate);
            if let Some(choice) = choice {
                let arc = Arc::new(candidate);
                let mut candidates = choice
                    .into_iter()
                    .map(|action| {
                        let env_ref = &self.environment;
                        let clone = Arc::clone(&arc);
                        Some(Thunk::new(move || env_ref.apply_action(&clone, &action)))
                    }).collect::<Vec<_>>();

                // We can safely unwrap here because we have put only `Some` values in the
                // candidates vector.
                let probabilities = self.policy.compute_probabilities(
                    candidates.iter().map(|x| x.as_ref().unwrap().as_ref()),
                );

                for sampled in (0..num_samples)
                    .map(|_| self.policy.sample(probabilities.iter().map(Some), rng))
                    .filter_map(|x| x)
                {
                    // This could fail if we sampled the same value twice.
                    if let Some(candidate) = candidates[sampled.value.into_usize()].take()
                    {
                        let proba = proba * f64::from(sampled.probability);

                        match Thunk::unwrap(candidate) {
                            Some(candidate) => {
                                worklist.push((Arc::new(candidate), proba, depth + 1));
                            }
                            None => {
                                // Dead node
                                num_dead += 1;
                            }
                        }
                    }
                }
            } else {
                // Terminal node reached
                num_terminal += 1;
                estimate += proba.recip();
            }
        }

        if num_terminal == 0 {
            (false, 1f64)
        } else {
            // Ergh we return the *proba* not the estimate...
            (true, (num_terminal + num_dead) as f64 / estimate)
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
            value: WeightedChoice::new(&mut weighted).sample(rng),
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

        let sample = WeightedChoice::new(&mut probas).sample(rng);

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

/// A node in the tree portion of the search.
struct Node<Spec: SearchSpec> {
    /// The outgoing edges for that node.
    edges: Edges<Edge<Spec>>,

    data: NodeData<Spec>,

    /// Whether this is a terminal node.
    terminal: bool,

    dead: AtomicBool,

    /// The cumulated score value.
    total_estimate: AtomicF64,

    /// The cumulated proba value.  See Weighted Backtrack Estimator.
    total_proba: AtomicF64,

    /// The number of times this node has been visited.  Deadends are
    /// counted as visits.
    num_visits: AtomicUsize,

    /// The number of deadends encountered when visiting this node.
    num_deadends: AtomicUsize,
}

// gviz
struct NodeInfo {
    estimate: f64,
    bound: f64,
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
                    bound: node.data.as_f64(),
                    terminal: node.terminal,
                    deadend: node.is_dead(),
                    estimate: node.total_estimate.load() / num_visits as f64,
                    truncated: num_visits < min_visits,
                    explored: num_visits > 0,
                    num_visits,
                    num_deadends,
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
                                bound: std::f64::NAN,
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
                ))).suffix_line(dot::LabelText::label(format!("bound: {:.2e}", n.1.bound)))
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
    fn new(data: NodeData<Spec>, edges: Vec<Edge<Spec>>) -> Self {
        Node {
            data: data,
            edges: edges.into(),
            terminal: false,
            dead: AtomicBool::new(false),
            total_estimate: AtomicF64::new(0f64),
            total_proba: AtomicF64::new(0f64),
            num_visits: AtomicUsize::new(0usize),
            num_deadends: AtomicUsize::new(0usize),
        }
    }

    fn terminal(data: NodeData<Spec>) -> Self {
        Node {
            terminal: true,
            ..Self::new(data, Vec::new())
        }
    }

    fn deadend(data: NodeData<Spec>) -> Self {
        Node {
            dead: AtomicBool::new(true),
            total_estimate: AtomicF64::new(837_483_748f64),
            ..Self::new(data, Vec::new())
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

struct TreePathList<'a, Spec: SearchSpec + 'a> {
    pub path: List<&'a Node<Spec>>,

    pub leaf: &'a Node<Spec>,

    pub weight: f64,
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
    path: Vec<usize>,
    estimate: Option<f64>,
}

struct Tree<Spec: SearchSpec> {
    spec: Spec,
}

impl<Spec: SearchSpec> Tree<Spec>
where
    Spec::Evaluator: Evaluator<Spec::Environment, Evaluation = (bool, f64)>,
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
            let sampled = match self.spec.policy().sample(
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
                let (node, expansion) = self.spec.expansion_step(candidate);
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

    fn simulation_step(
        &self,
        expanded: ExpansionResult<Spec>,
        rng: &mut ThreadRng,
    ) -> Evaluation<Spec> {
        trace!("Starting simulation");

        self.spec
            .evaluator()
            .evaluate(expanded.state, expanded.actions, rng)
    }

    fn backpropagation_step(
        &self,
        path: TreePath<'_, Spec>,
        estimate: Evaluation<Spec>,
    ) -> BackpropResult {
        trace!("Starting backpropagation");

        let edgepath = path
            .path
            .iter()
            .map(|(_, edge_sample)| edge_sample.value.0)
            .collect::<Vec<_>>();

        let (terminal, mut proba) = estimate;

        path.leaf.num_visits.fetch_add(1, Ordering::Relaxed);
        while path.leaf.total_proba.try_add(proba).is_err() {}
        if terminal {
            while path.leaf.total_estimate.try_add(proba.recip()).is_err() {}
        } else {
            path.leaf.num_deadends.fetch_add(1, Ordering::Relaxed);
        }

        for (node, edge_sample) in path.path.into_iter().rev() {
            node.num_visits.fetch_add(1, Ordering::Relaxed);
            proba *= f64::from(edge_sample.probability);
            while node.total_proba.try_add(proba).is_err() {}

            if terminal {
                while node.total_estimate.try_add(proba.recip()).is_err() {}
            } else {
                node.num_deadends.fetch_add(1, Ordering::Relaxed);
            }
        }

        BackpropResult {
            path: edgepath,
            estimate: if terminal { Some(proba.recip()) } else { None },
        }
    }

    fn playout(&self, root: &Node<Spec>, rng: &mut ThreadRng) -> Option<BackpropResult> {
        let (path, expanded) = self.selection_expansion_steps(root, rng);

        let result = if let Some(expanded) = expanded {
            self.simulation_step(expanded, rng)
        } else if path.leaf.terminal {
            (true, 1f64)
        } else {
            (false, 1f64)
        };

        return Some(self.backpropagation_step(path, result));
    }
}

use std::hash::Hash;

trait SpecStratifier<Spec: SearchSpec> {
    type Strate: Hash + Eq + Ord + Clone;

    fn strate(&self, node: &Node<Spec>) -> Self::Strate;
}

struct Stratified<Spec, S> {
    spec: Spec,
    stratifier: S,
}

impl<Spec: SearchSpec, S: SpecStratifier<Spec>> Stratified<Spec, S> {
    fn selection_expansion_steps<'a>(
        &'a self,
        root: &'a Node<Spec>,
        rng: &mut ThreadRng,
    ) -> Vec<(TreePathList<'a, Spec>, Option<ExpansionResult<Spec>>)> {
        assert!(root.edges.len() > 0);

        let mut queue = HashMap::new();
        let mut heap = BinaryHeap::new();

        let root_strate = self.stratifier.strate(root);
        queue.insert(
            root_strate.clone(),
            TreePathList {
                path: List::new(),
                leaf: root,
                weight: 1.,
            },
        );
        heap.push(root_strate);

        let mut paths = Vec::new();

        while let Some(strate) = heap.pop() {
            let path = queue.remove(&strate).expect("Missing strate in the queue.");

            // We reached a terminal leaf or deadend
            if path.leaf.is_dead() || path.leaf.terminal {
                paths.push((path, None));
                continue;
            }

            for edge in path.leaf.edges.iter() {
                let node;
                let expanded = {
                    let mut expanded = None;
                    node = edge.dst.force(|candidate| {
                        let (node, expansion) = self.spec.expansion_step(candidate);
                        expanded = Some(expansion);
                        node
                    });
                    expanded
                };

                let path = TreePathList {
                    path: path.path.push_front(path.leaf),
                    leaf: node,
                    weight: path.weight,
                };

                if let Some(expansion) = expanded {
                    paths.push((path, Some(expansion)));
                } else {
                    let strate = self.stratifier.strate(&node);

                    match queue.entry(strate) {
                        Entry::Occupied(mut entry) => {
                            let path_s = entry.get_mut();
                            path_s.weight += path.weight;

                            if Bernoulli::new(path.weight / path_s.weight).sample(rng) {
                                *path_s = path;
                            }
                        }
                        Entry::Vacant(entry) => {
                            heap.push(entry.key().clone());
                            entry.insert(path);
                        }
                    }
                }
            }
        }

        paths
    }
}

struct TreeSizeEstimation<Env, Eval, P> {
    environment: Env,
    policy: P,
    evaluator: Eval,
}

impl<Env: Environment, Eval: Evaluator<Env>, P: TreePolicy<Env>> SearchSpec
    for TreeSizeEstimation<Env, Eval, P>
{
    type Environment = Env;
    type Evaluator = Eval;
    type TreePolicy = P;

    fn environment(&self) -> &Self::Environment {
        &self.environment
    }

    fn evaluator(&self) -> &Self::Evaluator {
        &self.evaluator
    }

    fn policy(&self) -> &Self::TreePolicy {
        &self.policy
    }
}

use kernels::{linalg, Kernel, Scalar};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Prefix(Vec<usize>);

impl FromStr for Prefix {
    type Err = ::std::num::ParseIntError;

    fn from_str(s: &str) -> Result<Prefix, ::std::num::ParseIntError> {
        Ok(Prefix(
            s.split_terminator('.')
                .map(str::parse)
                .collect::<Result<Vec<_>, _>>()?,
        ))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExactChoice(ExactChoiceImpl);

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ExactChoiceImpl {
    Auto,
    Always,
    Never,
    Cached(usize),
}

impl Default for ExactChoice {
    fn default() -> ExactChoice {
        ExactChoice(ExactChoiceImpl::Auto)
    }
}

impl ExactChoice {
    fn auto() -> ExactChoice {
        ExactChoice(ExactChoiceImpl::Auto)
    }

    fn always() -> ExactChoice {
        ExactChoice(ExactChoiceImpl::Always)
    }

    fn never() -> ExactChoice {
        ExactChoice(ExactChoiceImpl::Never)
    }

    fn cached(value: usize) -> ExactChoice {
        ExactChoice(ExactChoiceImpl::Cached(value))
    }

    fn compute(
        &self,
        estimate: f64,
        context: &impl Context,
        ordering: &choice::ChoiceOrdering,
        candidates: Vec<Candidate<'_>>,
    ) -> Option<usize> {
        match self.0 {
            ExactChoiceImpl::Never => None,
            ExactChoiceImpl::Cached(size) => Some(size),
            ExactChoiceImpl::Auto if estimate > 5e5 => {
                trace!(
                    "Estimated size is larger than 5e5; skipping exact size computation."
                );
                return None;
            }
            ExactChoiceImpl::Auto | ExactChoiceImpl::Always => {
                let num_leafs = AtomicUsize::new(0);
                let done = AtomicBool::new(false);
                crossbeam::scope(|scope| {
                    scope.spawn(|| {
                        let bar = ProgressBar::new_spinner();
                        bar.set_style(
                            indicatif::ProgressStyle::default_spinner()
                                .template(concat!("[{elapsed_precise}] {spinner} {pos}")),
                        );

                        while !done.load(Ordering::Acquire) {
                            bar.set_position(num_leafs.load(Ordering::Relaxed) as u64);
                            thread::sleep(time::Duration::from_millis(1_000));
                        }

                        // Ensure we properly set the final position even if we were done by the
                        // first time we ran the above loop.
                        bar.set_position(num_leafs.load(Ordering::Relaxed) as u64);
                        bar.finish_and_clear();
                    });

                    exact_count(context, &ordering, candidates, &num_leafs);
                    done.store(true, Ordering::Release);
                });

                Some(num_leafs.load(Ordering::Relaxed))
            }
        }
    }
}

impl FromStr for ExactChoice {
    type Err = ::std::num::ParseIntError;

    fn from_str(s: &str) -> Result<ExactChoice, ::std::num::ParseIntError> {
        match s {
            "auto" => Ok(ExactChoice::auto()),
            "always" | "yes" | "on" => Ok(ExactChoice::always()),
            "never" | "no" | "off" => Ok(ExactChoice::never()),
            _ => Ok(ExactChoice::cached(str::parse(s)?)),
        }
    }
}

#[derive(StructOpt, Debug, Serialize, Deserialize)]
#[structopt(name = "treesize2")]
struct Opt {
    #[structopt(long = "num-playouts", default_value = "10000")]
    num_playouts: usize,

    #[structopt(long = "output", short = "o")]
    output: String,

    #[structopt(long = "prefix", default_value = "",)]
    prefix: Prefix,

    #[structopt(
        long = "ordering",
        default_value = "lower_layout,size,dim_kind,dim_map,mem_space,order,inst_flag"
    )]
    ordering: config::ChoiceOrdering,

    #[structopt(long = "exact", default_value = "auto",)]
    exact: ExactChoice,

    #[structopt(long = "dummy")]
    #[serde(rename = "dummy")]
    dummy: bool,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
struct Record {
    id: usize,
    estimate: f64,
}

fn exact_count<'a>(
    context: &impl Context,
    ordering: &choice::ChoiceOrdering,
    mut candidates: Vec<Candidate<'a>>,
    num_leafs: &AtomicUsize,
) {
    if let Some(candidate) = candidates.pop() {
        if let Some(choice) = choice::list(ordering.iter(), &candidate.space).next() {
            // If children is empty, we reached a deadend -- the call to exact_count will return 0.
            let children = candidate.apply_choice(context, choice);
            let ((), ()) = join(
                || exact_count(context, ordering, children, num_leafs),
                || exact_count(context, ordering, candidates, num_leafs),
            );
        } else {
            // If no choice is aavailable, we reached a leaf.
            num_leafs.fetch_add(1, Ordering::Relaxed);

            // We still need to count the remaining candidates!
            exact_count(context, ordering, candidates, num_leafs);
        }
    }
}

trait CandidateBuilder<'a, T, F, C>
where
    F: FnOnce(Vec<Candidate<'_>>, &C) -> T,
    C: ArgMap + Context + 'a,
{
    fn with_candidates(&self, context: &mut C, body: F) -> T;
}

impl<'a, Params, T, F, C> CandidateBuilder<'a, T, F, C> for Params
where
    Params: KernelParameters<'a>,
    F: FnOnce(Vec<Candidate<'_>>, &C) -> T,
    C: ArgMap + Context + 'a,
{
    fn with_candidates(&self, context: &mut C, body: F) -> T {
        <Params as KernelParameters<'a>>::with_candidates(self, context, body)
    }
}

trait KernelParameters<'a> {
    type Kernel: Kernel<'a>;

    fn as_parameters(&self) -> <Self::Kernel as Kernel<'a>>::Parameters;

    fn with_candidates<T, F, C>(&self, context: &mut C, body: F) -> T
    where
        F: FnOnce(Vec<Candidate<'_>>, &C) -> T,
        C: ArgMap + Context + 'a,
    {
        Self::Kernel::superman(
            self.as_parameters(),
            true,
            context,
            move |_kernel, candidates, context| body(candidates, context),
        )
    }
}

#[derive(Copy, Clone, Debug)]
pub struct AxpyParameters<S> {
    dim: i32,
    generic: bool,
    scalar: PhantomData<fn() -> S>,
}

impl<S: Scalar> From<(i32, bool)> for AxpyParameters<S> {
    fn from((dim, generic): (i32, bool)) -> Self {
        AxpyParameters {
            dim,
            generic,
            scalar: PhantomData,
        }
    }
}

impl<'a, S: Scalar> KernelParameters<'a> for AxpyParameters<S> {
    type Kernel = linalg::Axpy<'a, S>;

    fn as_parameters(&self) -> (i32, bool) {
        (self.dim, self.generic)
    }
}

#[derive(Clone)]
pub struct MatMulParameters<S> {
    params: linalg::MatMulP<S>,
    scalar: PhantomData<fn() -> S>,
}

impl<S: Scalar> From<linalg::MatMulP<S>> for MatMulParameters<S> {
    fn from(params: linalg::MatMulP<S>) -> Self {
        MatMulParameters {
            params,
            scalar: PhantomData,
        }
    }
}

impl<'a, S: Scalar> KernelParameters<'a> for MatMulParameters<S> {
    type Kernel = linalg::MatMul<'a, S>;

    fn as_parameters(&self) -> linalg::MatMulP<S> {
        self.params.clone()
    }
}

fn main() {
    env_logger::init();

    let opt = Opt::from_args();

    let out_dir = std::path::PathBuf::from(&opt.output);

    // Dummy safeguard
    let dummy_path = out_dir.join("DUMMY");
    std::fs::create_dir_all(&out_dir.parent().unwrap())
        .expect("Error creating parent directory");

    std::fs::create_dir(&out_dir)
        .or_else(|err| {
            if err.kind() == std::io::ErrorKind::AlreadyExists {
                if opt.dummy && dummy_path.exists() {
                    debug!("Overwriting existing dummy output.");

                    Ok(())
                } else {
                    panic!("I will not overwrite non-dummy files.")
                }
            } else {
                Err(err)
            }
        }).expect("Error creating directory");

    if opt.dummy {
        std::fs::write(&dummy_path, b"").expect("Error creating DUMMY file marker");
    }

    let config_path = out_dir.join("config.yaml");
    serde_yaml::to_writer(&mut std::fs::File::create(&config_path).unwrap(), &opt)
        .unwrap();

    let estimates_path = out_dir.join("estimates.csv");
    let descents_path = out_dir.join("descents.csv");
    let dot_path = out_dir.join("tree.dot");

    let num_playouts = opt.num_playouts;

    let ordering = choice::ChoiceOrdering::from_config_ref(&opt.ordering);

    // let proba = CompleteTreeSizeRatioPolicy { epsilon: 0.1f64 };
    let proba = UniformPolicy {};

    let gpu: Gpu = serde_json::from_reader(
        &std::fs::File::open("/home/elarnon/.config/telamon/cuda_gpus.json").unwrap(),
    ).unwrap();

    let params: Box<CandidateBuilder<'_, _, _, _>> = if true {
        /*
        Box::new(MatMulParameters::<f32>::from(linalg::MatMulP::new(
            16, 16, 16,
        )))*/
        Box::new(AxpyParameters::<f32>::from((1 << 26, true)))
    } else {
        Box::new(MatMulParameters::<f32>::from(
            linalg::MatMulP::new(1024, 1024, 1024)
                .tile_m(TilingPattern::new_fixed(&[32, 4]))
                .tile_n(TilingPattern::new_fixed(&[32, 4]))
                .tile_k(TilingPattern::new_fixed(&[32])),
        ))
    };

    let estimates = params.with_candidates(
        &mut FakeContext::new(gpu),
        move |root_candidates, context| {
            let candidate = {
                let mut candidates = root_candidates.clone();

                assert!(candidates.len() == 1);
                let mut candidate = candidates.pop().unwrap();

                for &index in &opt.prefix.0 {
                    // We need a local variable here otherwise rust gets confused about lifetimes.
                    let choice = choice::list(ordering.iter(), &candidate.space).next();
                    if let Some(mut choice) = choice {
                        println!(
                            "[{}] Selecting {:?} from {:?}",
                            index, choice[index], choice
                        );

                        if let Ok(child) =
                            candidate.apply_decision(context, choice.swap_remove(index))
                        {
                            candidate = child
                        } else {
                            panic!("Invalid decision.");
                        }
                    } else {
                        panic!("No path.");
                    }
                }
                candidate
            };

            let tree = Tree {
                spec: TreeSizeEstimation {
                    environment: ContextEnvironment {
                        context,
                        ordering: choice::ChoiceOrdering::from_config_ref(&opt.ordering),
                        invalid_actions_cnt: AtomicUsize::new(0),
                    },
                    policy: proba.clone(),
                    evaluator: StratifiedEvaluator {
                        stratifier: MyStratifier,
                        //PartialBacktrackEvaluator {
                        //PolicyEvaluator {
                        environment: ContextEnvironment {
                            context,
                            ordering: choice::ChoiceOrdering::from_config_ref(
                                &opt.ordering,
                            ),
                            invalid_actions_cnt: AtomicUsize::new(0),
                        },
                        policy: proba,
                    },
                },
            };

            let (mut root, _) = tree.spec.expansion_step(Arc::new(candidate.clone()));
            let num_threads = 8;

            let (estimates, descents) = {
                let mut all_estimates = (0..num_threads)
                    .map(|_| Vec::with_capacity(num_playouts))
                    .collect::<Vec<_>>();
                let mut all_descents = (0..num_threads)
                    .map(|_| Vec::with_capacity(num_playouts))
                    .collect::<Vec<_>>();
                let playouts_done = AtomicUsize::new(0);

                crossbeam::scope(|scope| {
                    for (ix, (estimate_mut, descent_mut)) in all_estimates
                        .iter_mut()
                        .zip(all_descents.iter_mut())
                        .enumerate()
                    {
                        scope
                            .builder()
                            .name(format!("TlmnSearch #{}", ix))
                            .spawn(|| {
                                let rng = &mut thread_rng();

                                let thread_estimates = estimate_mut;
                                let thread_descents = descent_mut;

                                loop {
                                    let playout_id =
                                        playouts_done.fetch_add(1, Ordering::Relaxed);
                                    if playout_id >= num_playouts {
                                        break;
                                    }

                                    if let Some(result) = tree.playout(&root, rng) {
                                        if let Some(estimate) = result.estimate {
                                            thread_estimates.push(Record {
                                                id: playout_id,
                                                estimate,
                                            });
                                        } else {
                                            thread_estimates.push(Record {
                                                id: playout_id,
                                                estimate: 0f64,
                                            });
                                        }

                                        thread_descents.push(result.path);
                                    }
                                }
                            }).unwrap();
                    }

                    scope
                        .builder()
                        .name("TlmnMonitor".to_string())
                        .spawn(|| {
                            let bar = ProgressBar::new(num_playouts as u64);
                            bar.set_style(
                                indicatif::ProgressStyle::default_bar()
                                    .template(concat!(
                                        "[{elapsed_precise}] ",
                                        "{bar:40.cyan/blue} ",
                                        "{pos:>7}/{len:<7} ",
                                        "{wide_msg}",
                                        "({eta})"
                                    )).progress_chars(r"##-"),
                            );

                            loop {
                                let total_estimate = root.total_estimate.load();
                                let total_proba = root.total_proba.load();
                                let num_visits = root.num_visits.load(Ordering::Relaxed);
                                let num_deadends =
                                    root.num_deadends.load(Ordering::Relaxed);

                                let playout_id = playouts_done.load(Ordering::Relaxed);
                                bar.set_position(playout_id as u64);
                                bar.set_message(&format!(
                                    "size ~{:.2e} ~{:.2e}(deadends: {})",
                                    total_estimate / num_visits as f64,
                                    (num_visits - num_deadends) as f64 / total_proba,
                                    num_deadends,
                                ));

                                if playout_id >= num_playouts {
                                    bar.finish();
                                    break;
                                }

                                thread::sleep(time::Duration::from_millis(1_000));
                            }
                        }).unwrap();
                });

                let mut estimates = Vec::with_capacity(num_playouts);
                estimates.extend(
                    all_estimates
                        .into_iter()
                        .flat_map(|thread_estimates| thread_estimates.into_iter()),
                );

                let mut descents = Vec::with_capacity(num_playouts);
                descents.extend(
                    all_descents
                        .into_iter()
                        .flat_map(|thread_descents| thread_descents.into_iter()),
                );

                (estimates, descents)
            };

            {
                let mut writer = csv::Writer::from_path(&estimates_path).unwrap();
                for result in estimates.iter() {
                    writer.serialize(result).unwrap();
                }
                writer.flush().unwrap();
            }

            {
                let mut writer = csv::Writer::from_path(&descents_path).unwrap();
                writer.write_record(&["Id", "Position", "Action"]).unwrap();
                for (ix, row) in descents.iter().enumerate() {
                    for (pos, elt) in row.iter().enumerate() {
                        writer
                            .write_record(&[
                                ix.to_string(),
                                pos.to_string(),
                                elt.to_string(),
                            ]).unwrap();
                    }
                }
                writer.flush().unwrap();
            }

            let estimate_stats = stats::OnlineStats::from_iter(
                estimates
                    .into_iter()
                    .map(|Record { estimate, .. }| estimate),
            );

            println!(
                "Estimated {:.2e} with stddev {:.2e}",
                estimate_stats.mean(),
                estimate_stats.stddev(),
            );

            let info = TreeInfo::new(&root, num_playouts / 10);
            let mut f = std::fs::File::create(&dot_path).unwrap();
            dot::render(&info, &mut f).unwrap();

            if let Some(true_size) = opt.exact.compute(
                estimate_stats.mean(),
                context,
                &choice::ChoiceOrdering::from_config_ref(&opt.ordering),
                vec![candidate.clone()],
            ) {
                println!("True size: {} ({:e})", true_size, true_size as f64);
                println!(
                    "Error (log scale): {:>3.0}%",
                    ((true_size as f64).ln() - estimate_stats.mean().ln()).abs()
                        / 10.0f64.ln()
                        * 100f64
                );
            }

            // TODO
            let total_proba = root.total_proba.load();
            let num_visits = root.num_visits.load(Ordering::Relaxed);
            let num_deadends = root.num_deadends.load(Ordering::Relaxed);
            println!(
                "Other: {:.2e}",
                (num_visits - num_deadends) as f64 / total_proba,
            );
            vec![root.total_estimate.load() / ((*root.num_visits.get_mut()) as f64)]
        },
    );

    println!(
        "Got avg of {}",
        estimates.iter().sum::<f64>() / estimates.len() as f64
    );
}
