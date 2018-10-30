//! Count the tree size

extern crate env_logger;
#[macro_use]
extern crate log;
extern crate rand;

extern crate telamon;
extern crate telamon_kernels as kernels;

use std::cell::RefCell;

use std::borrow::Borrow;

use rand::distributions::{IndependentSample, Weighted, WeightedChoice};
use rand::{thread_rng, Rng, ThreadRng};

use telamon::{
    device::Context,
    explorer::{choice, Candidate},
};

use std::marker::PhantomData;
use std::rc::Rc;

/// Defines the parameters of the search
pub trait SearchSpec: Sized {
    /// The type of actions.
    type Action;

    /// Cached state stored on the leafs.
    type LeafState;

    /// Algorithm-specific data stored on the nodes.
    type NodeData;

    /// Node statistics
    type NodeStats: NodeStats<Self>;

    /// Algorithm-specific data stored on the edges.
    type EdgeData;

    /// Edge statistics
    type EdgeStats: EdgeStats<Self>;

    /// Evaluator to use for leaf nodes.
    type Evaluator: Evaluator<Self>;

    /// Policy to use when exploring the tree.
    type TreePolicy: TreePolicy<Self>;
}

type TreePolicyThreadData<Spec> =
    <<Spec as SearchSpec>::TreePolicy as TreePolicy<Spec>>::ThreadLocalData;
type StateEvaluation<Spec> =
    <<Spec as SearchSpec>::Evaluator as Evaluator<Spec>>::StateEvaluation;
type ActionEvaluation<Spec> =
    <<Spec as SearchSpec>::Evaluator as Evaluator<Spec>>::ActionEvaluation;

/// The policy to use when exploring a tree.
pub trait TreePolicy<Spec: SearchSpec>: Sync + Sized {
    /// Type for per-thread local data used by the policy, such as a
    /// random number generator.
    type ThreadLocalData;

    /// Select a child node among possible candidates.
    ///
    /// # Panics
    ///
    /// `pick_child` should panic if `children` is empty.
    fn pick_child<'a>(
        &self,
        children: impl ExactSizeIterator<Item = &'a Edge<Spec>> + Clone,
        tld: &mut Self::ThreadLocalData,
    ) -> &'a Edge<Spec>;
}

/// The policy to evaluate a leaf node, which may or may not be
/// terminal.  This can use random rollouts, a neural network model,
/// etc.
pub trait Evaluator<Spec: SearchSpec>: Sync {
    type StateEvaluation;
    type ActionEvaluation;

    /// Evaluate a leaf node with state `state` and available actions
    /// `actions`.
    fn evaluate(
        &self,
        state: &State<Spec>,
        actions: &[(Action<Spec>, LeafState<Spec>)],
    ) -> (Self::StateEvaluation, Vec<Self::ActionEvaluation>);
}

struct SizeEstimate {
    num_deadends: usize,
    total_size: f64,
    num_descents: usize,
}

impl SizeEstimate {
    fn new() -> Self {
        SizeEstimate {
            num_deadends: 0usize,
            total_size: 0f64,
            num_descents: 0usize,
        }
    }

    fn add(&mut self, estimate: Option<f64>) {
        match estimate {
            Some(estimate) => {
                self.total_size += estimate;
                self.num_descents += 1;
            }
            None => self.num_deadends += 1,
        }
    }
}

impl<'a, Spec> Evaluator<Spec> for RandomRollout
where
    Spec: SearchSpec<LeafState = Candidate<'a>>,
{
    /// The state evaluation.  This is the list of all
    type StateEvaluation = ();
    type ActionEvaluation = ();

    fn evaluate(
        &self,
        _: &State<Spec>,
        actions: &[(Action<Spec>, Spec::LeafState)],
    ) -> (SizeEstimate, ()) {
        // Terminal node
        if actions.len() == 0 {
            return (SizeEstimate::new(1f64), vec![]);
        }

        let leaf_estimates = self
            .proba
            .estimate(actions.iter().map(|(_, candidate)| candidate));

        let (mut estimate, mut candidates);
        {
            let index = self.proba.sample(leaf_estimates, rng);
            let candidate = actions[index].1;
            estimate = leaf_estimates[index].size_estimate;

            let choice = choice::list(&candidate.space).next();
            if let Some(choice) = choice {
                candidates = candidate.apply_choice(&self.context, choice);
            } else {
                // Terminal node reached after 1 expansion
                return (SizeEstimate::new(estimate), leaf_estimates);
            }
        };

        loop {
            if candidates.len() == 0 {
                // No remaining choices after propagation: dead end.
                return (SizeEstimate::deadend(), leaf_estimates);
            }

            let estimates = self.proba.estimate(candidates.iter());
            let index = self.proba.sample(estimates, rng);
            let candidate = candidates.swap_remove(index);
            estimate *= estimates[index].size_estimate;

            let choice = choice::list(&candidate.space).next();
            if let Some(choice) = choice {
                candidates = candidate.apply_choice(&self.context, choice);
            } else {
                // Terminal node reached
                return (SizeEstimate::new(estimate), leaf_estimates);
            }
        }
    }
}

/// A node in the search tree.
pub struct SearchNode<Spec: SearchSpec> {
    /// The list of outgoing edges for this node.
    edges: Vec<SearchEdge<Spec>>,
    data: Spec::NodeData,
    stats: Spec::NodeStats,
}

/// A possibly non-expanded search node.
enum LazyNode<Spec: SearchSpec> {
    /// A non-expanded search node.
    LeafState(Spec::LeafState),
    /// A fully expanded node.
    Node(Rc<Node<Spec>>),
}

/// An edge between nodes in the search tree.  We use an explicit
/// representation of edges so that we are able to annotate the edges
/// with extra data when there are transpositions in the search tree.
pub struct SearchEdge<Spec: SearchSpec> {
    /// The action this edge represents
    action: Spec::Action,
    /// The destination node for this edge.  This is a `LeafState`
    /// value before the edge is first expanded.
    dst: RefCell<LazyNode<Spec>>,
}

struct Lazy<T, F: FnOnce() -> T> {
    once: Once,
    val: UnsafeCell<T>,
    fun: UnsafeCell<F>,
}

impl<T, F: FnOnce() -> T> Lazy<T, F> {
    pub fn from_fun(fun: F) -> Self {
        Lazy {
            once: Once::new(),
            val: UnsafeCell::new(None),
            fun: UnsafeCell::new(Some(fun)),
        }
    }

    pub fn from_val(val: T) -> Self {
        Lazy {
            once: Once::new(),
            val: UnsafeCell::new(Some(val)),
            fun: UnsafeCell::new(None),
        }
    }

    pub fn force(&self) -> (&T, bool) {
        unsafe {
            let mut created = false;
            self.once.call_once(|| {
                if let Some(fun) = (&mut *self.fun.get()).take() {
                    *self.val.get() = fun();
                    created = true;
                }
            });
            (&*self.val.get(), created)
        }
    }
}

struct Lazy<T, F: FnOnce() -> T> {
    fun: Atomic<F>,
    val: Atomic<T>,
}

enum LazyForceResult<'g, T> {
    /// Returned a pre-existing value.
    Existing(&'g T),
    /// Created a new value
    Created(&'g T, bool),
    /// Creating the value resulted in a cycle.
    Undefined,
}

impl<'g, T> LazyForceResult<'g, T> {
    pub fn created(&self) -> bool {
        match self {
            &LazyForceResult::Existing(_) => false,
            &LazyForceResult::Created(_, _) => true,
        }
    }

    pub fn had_contention(&self) -> bool {
        match self {
            &LazyForceResult::Existing(_) => false,
            &LazyForceResult::Created(_, contention) => contention,
        }
    }

    pub fn unwrap(self) -> &'g T {
        match self {
            LazyForceResult::Existing(t) => t,
            LazyForceResult::Created(t, _) => t,
        }
    }
}

impl<'g, T> Into<&'g T> for LazyForceResult<'g, T> {
    fn into(self) -> &'g T {
        self.unwrap()
    }
}

impl<T, F: FnOnce() -> T> Lazy<T, F> {
    pub fn from_fun(fun: F) -> Self {
        Lazy {
            fun: Atomic::new(fun),
            val: Atomic::null(),
        }
    }

    pub fn from_val(val: T) -> Self {
        Lazy {
            fun: Atomic::null(),
            val: Atomic::new(val),
        }
    }

    pub fn is_val<'g>(&'g self, guard: &'g Guard) -> bool {
        self.value.load_consume(guard).is_null()
    }

    pub fn get<'g>(&'g self, guard: &'g Guard) -> Option<&'g T> {
        unsafe { self.value.load_consume(guard).as_ref() }
    }

    pub fn try_force<'g>(&'g self, guard: &'g Guard) -> Option<&'g T> {
        let fun = self.fun.swap(Shared::null(), Ordering::AcqRel, guard);
        if fun.is_null() {
            return None;
        }

        let val = (*fun.into_owned().into_box())();
    }

    pub fn force<'g>(&'g self, guard: &'g Guard) -> LazyForceResult<'g, T> {
        unsafe {
            match self.get() {
                // Fast path: there is already a value, we don't have
                // to force anything.
                Some(value) => LazyForceResult::Existing(value),

                None => {
                    let create =
                        self.create.swap(Shared::null(), Ordering::AcqRel, guard);
                    if create.is_null() {
                        // There is no value and no function for
                        // creating one: another thread is already
                        // creating the value, so we just wait for
                        // them to be done.
                        loop {
                            if let Some(value) = self.get() {
                                return LazyForceResult::Created(value, true);
                            }
                        }
                    }

                    // Create the value and store it.  We can safely
                    // `deref` the shared value.
                    let value = (*create.into_owned().into_box())();
                    let shared_value = Owned::new(value).into_shared();
                    self.value
                        .store(shared_value.clone(), Ordering::Release, guard);
                    LazyForceResult::Created(shared_value.deref(), false)
                }
            }
        }
    }
}

impl<T, F: FnOnce() -> T> Drop for Lazy<T, T> {
    fn drop(&mut self) {
        unsafe {
            let guard = epoch::pin();
            let shared_create = self.value.swap(Shared::null(), Ordering::AcqRel, guard);
            let shared_value = self.value.swap(Shared::null(), Ordering::AcqRel, guard);

            if !shared_create.is_null() {
                guard.defer_destroy(shared_create);
            }

            if !shared_value.is_null() {
                guard.defer_destroy(shared_value);
            }
        }
    }
}

impl<Spec: SearchSpec> SearchEdge<Spec> {
    fn expand<'g>(&'g self, handle: &'g SearchHandle) -> &'g Node<Spec> {
        self.dst.force(self.handle.epoch)
    }
}

impl<Spec: SearchSpec> Drop for SearchEdge<Spec> {
    fn drop(&mut self) {
        unsafe {
            let guard = epoch::pin();
            let shared_dst = self.dst.swap(Shared::null(), Ordering::Acquire, guard);
            if !shared_dst.is_null() {
                guard.defer_destroy(shared_ptr);
            }
        }
    }
}

trait NodeStats<Spec: SearchSpec>: Default {
    fn down(&self, search: &Spec);

    fn up(&self, search: &Spec);
}

impl<Spec: SearchSpec<NodeStats = ()>> NodeStats<Spec> for () {
    fn down(&self, _: &Spec) {}

    fn up(&self, _: &Spec) {}
}

trait EdgeStats<Spec: SearchSpec>: Default {
    fn down(&self, search: &Spec);

    fn up(&self, search: &Spec);
}

impl<Spec: SearchSpec<EdgeStats = ()>> EdgeStats<Spec> for () {
    fn down(&self, _: &Spec) {}

    fn up(&self, _: &Spec) {}
}

type Action<Spec> = <Spec as SearchSpec>::Action;
type LeafState<Spec> = <Spec as SearchSpec>::LeafState;
type NodeData<Spec> = <Spec as SearchSpec>::NodeData;
type PolicyLocalData<Spec> =
    <<Spec as SearchSpec>::TreePolicy as TreePolicy<Spec>>::ThreadLocalData;

pub struct Tree<Spec: SearchSpec> {
    root: Node<Spec>,
    policy: Spec::TreePolicy,
    evaluator: Spec::Evaluator,
    search: Spec,
}

impl<Spec: SearchSpec> Tree<Spec> {
    /// Select the node to expand.
    fn selection(&self, tld: &mut PolicyLocalData<Spec>) -> Option<&Edge<Spec>> {
        let mut node = &self.root;

        // Uninteresting case of an empty tree.
        if node.edges.len() == 0 {
            None
        }

        // Selection phase
        loop {
            let edge = self.policy.pick_child(node.edges.iter(), tld);
            let num_visits = edge.stats.down(&self.search);

            match edge.dst.acquire() {
                Existing(dst) => node = dst,
                Created(dst_ref, candidate, contention) => {
                    // Expansion phase
                    let choice = choice::list(&candidate.space).next();
                    if let Some(choice) = choice {
                        let children = choice.into_iter().flat_map(|action| {
                            candidate
                                .apply_decision(&self.context, action)
                                .map_err(|()| {
                                    self.search
                                        .num_invalid_actions
                                        .fetch_add(1, Ordering::Relaxed)
                                }).map(|candidate| Edge {
                                    action,
                                    dst: candidate,
                                }).ok()
                        });
                        if children.is_empty() {
                            return None;
                        }

                        let expanded = Node { edges: children };
                        return Some(dst_ref.set(expanded));
                    }
                }
            }
        }
    }

    fn expansion(&self, initializer: _, tld: &mut PolicyLocalData<Spec>) {
        initializer.initialize(|candidate| {});
    }

    fn simulation(&self, node: &Node<Spec>, tld: &mut PolicyLocalData<Spec>) {}

    fn backpropagation(&self) {}
}

pub struct Edge<Spec: SearchSpec> {
    action: Action<Spec>,
    evaluation: Spec::ActionEvaluation,
    dst: RefCell<Option<Rc<Node<Spec>>>>,
    data: Spec::EdgeData,
    stats: Spec::EdgeStats,
}

impl<Spec: SearchSpec> AsRef<Action<Spec>> for Edge<Spec> {
    fn as_ref(&self) -> &Action<Spec> {
        &self.action
    }
}

pub struct Node<Spec: SearchSpec> {
    edges: Vec<Edge<Spec>>,
    data: NodeData<Spec>,
}

impl<Spec: SearchSpec> Node<Spec> {
    fn new(state: State<Spec>, edges: Vec<Edge<Spec>>) -> Self
    where
        NodeData<Spec>: Default,
    {
        Self::with_data(state, edges, Default::default())
    }

    fn with_data(
        state: State<Spec>,
        edges: Vec<Edge<Spec>>,
        data: NodeData<Spec>,
    ) -> Self {
        Node { state, edges, data }
    }
}

struct TreeSizeSearch<'a>(PhantomData<&'a ()>);

pub struct TreeSizeEstimation {
    size_estimate: f64,
    proba: f64,
    weight: u32,
}

impl<'a> SearchSpec for TreeSizeSearch<'a> {
    type State = Candidate<'a>;
    type LeafState = Candidate<'a>;
    type Action = choice::ActionEx;
    type NodeData = ();
    type NodeStats = ();
    type EdgeData = ();
    type EdgeStats = ();

    type ActionEvaluation = TreeSizeEstimation;
    type StateEvaluation = ();

    type TreePolicy = UniformPolicy;
}

impl<
        'a,
        C: Context,
        Spec: SearchSpec<
            LeafState = Candidate<'a>,
            Action = choice::ActionEx,
            StateEvaluation = f64,
            ActionEvaluation = TreeSizeEstimation,
        >,
    > Evaluator<Spec> for CompleteTreeSizeRatioEstimator<C>
{
    fn evaluate(
        &self,
        _state: &State<Spec>,
        actions: &[(Action<Spec>, LeafState<Spec>)],
    ) -> (f64, Vec<TreeSizeEstimation>) {
        let action_evaluations =
            self.estimate(actions.iter().map(|(_action, candidate)| candidate));
        let state_evaluation = self.probe(
            actions.iter().map(|(_action, candidate)| candidate),
            &mut thread_rng(),
        );
        (state_evaluation, action_evaluations)
    }
}

pub struct UniformPolicy;

impl<Spec: SearchSpec> TreePolicy<Spec> for UniformPolicy {
    type ThreadLocalData = ThreadRng;

    fn pick_child<'a>(
        &self,
        mut children: impl ExactSizeIterator<Item = &'a Edge<Spec>> + Clone,
        rng: &mut ThreadRng,
    ) -> &'a Edge<Spec> {
        assert!(children.len() > 0);

        let index = rng.gen_range(0, children.len());
        children.nth(index).unwrap()
    }
}

pub struct CompleteTreeSizeRatioEstimator<C: Context> {
    epsilon: f64,
    context: C,
}

impl<C: Context> CompleteTreeSizeRatioEstimator<C> {
    fn new(epsilon: f64, context: C) -> Self {
        Self { epsilon, context }
    }

    fn estimate<'a, B: 'a>(
        &self,
        candidates: impl Iterator<Item = B>,
    ) -> Vec<TreeSizeEstimation>
    where
        B: Borrow<Candidate<'a>>,
    {
        let log_weights = candidates
            .map(|candidate| {
                choice::list(&candidate.borrow().space)
                    .map(|choice| (choice.len() as f64).ln())
                    .sum::<f64>()
            }).collect::<Vec<_>>();

        // Only take into account valid actions
        let len = log_weights.iter().map(|x| x.is_finite()).count();

        // Use log sum exp trick for better accuracy when computing
        // the total weight.  Note that since `(-inf).exp() = 0` we
        // don't have to filter out the invalid children.
        let max_log_weight = log_weights
            .iter()
            .cloned()
            .fold(std::f64::NEG_INFINITY, f64::max);
        let log_total_weight = max_log_weight + log_weights
            .iter()
            .map(|&log_weight| (log_weight - max_log_weight).exp())
            .sum::<f64>();

        let epsilon = self.epsilon / len as f64;
        let resolution = (u32::max_value() / len as u32) as f64;

        log_weights
            .into_iter()
            .map(|log_weight| {
                let proba = if log_weight.is_finite() {
                    (log_weight - log_total_weight).exp() * (1f64 - epsilon) + epsilon
                } else {
                    0f64
                };
                TreeSizeEstimation {
                    size_estimate: proba.recip(),
                    proba,
                    weight: (proba * resolution) as u32,
                }
            }).collect::<Vec<_>>()
    }

    fn sample<B>(estimates: impl Iterator<Item = B>, rng: &mut impl Rng) -> usize
    where
        B: Borrow<TreeSizeEstimation>,
    {
        WeightedChoice::new(
            &mut estimates
                .enumerate()
                .map(|(index, estimate)| Weighted {
                    item: index,
                    weight: estimate.borrow().weight,
                }).collect::<Vec<_>>(),
        ).ind_sample(rng)
    }

    fn probe<'a, B>(
        &self,
        candidates: impl ExactSizeIterator<Item = B>,
        rng: &mut impl Rng,
    ) -> f64
    where
        B: Borrow<Candidate<'a>>,
    {
        // At the first iteration we may not own the Candidates and so
        // we need a separate first step.
        let mut estimate;
        let mut candidates = {
            let candidate = {
                let mut candidates = candidates.collect::<Vec<_>>();
                let mut estimates = self.estimate(candidates.iter().map(Borrow::borrow));
                let index = Self::sample(estimates.iter(), rng);
                estimate = estimates[index].size_estimate;
                candidates.swap_remove(index)
            };

            let maybe_choice = choice::list(&candidate.borrow().space).next();
            if let Some(choice) = maybe_choice {
                candidate.borrow().apply_choice(&self.context, choice)
            } else {
                return estimate;
            }
        };

        loop {
            assert!(candidates.len() > 0);

            let candidate = {
                let mut estimates = self.estimate(candidates.iter());
                let index = Self::sample(estimates.iter(), rng);
                estimate *= estimates[index].size_estimate;
                candidates.swap_remove(index)
            };

            let maybe_choice = choice::list(&candidate.space).next();
            if let Some(choice) = maybe_choice {
                candidates = candidate.apply_choice(&self.context, choice);
            } else {
                return estimate;
            }
        }
    }
}

impl<C: Context, Spec: SearchSpec<ActionEvaluation = TreeSizeEstimation>> TreePolicy<Spec>
    for CompleteTreeSizeRatioEstimator<C>
{
    type ThreadLocalData = ThreadRng;

    fn pick_child<'a>(
        &self,
        mut children: impl ExactSizeIterator<Item = &'a Edge<Spec>> + Clone,
        rng: &mut ThreadRng,
    ) -> &'a Edge<Spec> {
        assert!(children.len() > 0);

        let index = Self::sample(children.clone().map(|edge| &edge.evaluation), rng);
        children.nth(index).unwrap()
    }
}

enum EdgeProbability {
    Uniform,
    CompleteTreeSizeRatio { epsilon: f64 },
}

struct ExpandedCandidate<C> {
    candidate: C,
    choices: Option<Vec<choice::Choice>>,
}

impl<'a, C: Borrow<Candidate<'a>>> ExpandedCandidate<C> {
    fn new(candidate: C) -> Self {
        ExpandedCandidate {
            candidate,
            choices: None,
        }
    }

    fn with_choices(candidate: C, choices: Vec<choice::Choice>) -> Self {
        ExpandedCandidate {
            candidate,
            choices: Some(choices),
        }
    }

    fn expand(self, context: &impl Context) -> Option<Vec<Candidate<'a>>> {
        let candidate = self.candidate.borrow();
        self.choices
            .map_or_else(
                || choice::list(&candidate.space).next(),
                |choices| choices.into_iter().next(),
            ).and_then(move |choice| {
                let candidates = candidate.apply_choice(context, choice);
                if candidates.len() == 0 {
                    None
                } else {
                    Some(candidates)
                }
            })
    }
}

impl EdgeProbability {
    /// Randomly select one of the available candidates according to
    /// the edge probability.
    ///
    /// This returns a pair `(candidate, inv_prob)` where `candidate`
    /// is the selected candidate and `inv_prob` is the inverse
    /// probability of selecting that specific candidate.  The inverse
    /// probability is meant to be used in tree size estimation.
    ///
    /// # Panics
    ///
    /// Panics if `candidates` is empty.
    fn select<'a, C>(
        &self,
        rng: &mut impl Rng,
        mut candidates: impl ExactSizeIterator<Item = C>,
    ) -> Option<((usize, usize), ExpandedCandidate<C>, f64)>
    where
        C: Borrow<Candidate<'a>>,
    {
        let len = candidates.len();
        if len == 1 {
            return Some((
                (0usize, 1usize),
                ExpandedCandidate::new(candidates.next().unwrap()),
                1f64,
            ));
        }

        match self {
            EdgeProbability::Uniform => {
                let index = rng.gen_range(0, len);
                candidates.nth(index).map(|candidate| {
                    ((index, len), ExpandedCandidate::new(candidate), len as f64)
                })
            }
            EdgeProbability::CompleteTreeSizeRatio { epsilon } => {
                if len == 0 {
                    None
                } else {
                    let mut expanded_candidates: Vec<_> = candidates
                        .map(|candidate| {
                            let choices: Vec<_> =
                                choice::list(&candidate.borrow().space).collect();
                            (candidate, choices)
                        }).collect();
                    let log_weights: Vec<_> = expanded_candidates
                        .iter()
                        .map(|(_, choices)| {
                            choices.iter().map(|c| (c.len() as f64).ln()).sum::<f64>()
                        }).collect();
                    let max_log_weight = log_weights
                        .iter()
                        .cloned()
                        .fold(std::f64::NEG_INFINITY, f64::max);
                    // Use log sum exp trick for better accuracy
                    let log_total_weight = max_log_weight + log_weights
                        .iter()
                        .map(|&log_weight| (log_weight - max_log_weight).exp())
                        .sum::<f64>();
                    let probas: Vec<_> = log_weights
                        .into_iter()
                        .map(|log_weight| {
                            (log_weight - log_total_weight).exp() * (1f64 - epsilon)
                                + epsilon
                        }).collect();
                    let resolution = (u32::max_value() / len as u32) as f64;
                    let index = {
                        let mut weighted_indices = probas
                            .iter()
                            .map(|&proba| (proba * resolution) as u32)
                            .enumerate()
                            .map(|(item, weight)| Weighted { item, weight })
                            .collect::<Vec<_>>();
                        WeightedChoice::new(&mut weighted_indices).ind_sample(rng)
                    };
                    let (candidate, choices) = expanded_candidates.swap_remove(index);
                    Some((
                        (index, len),
                        ExpandedCandidate::with_choices(candidate, choices),
                        probas[index].recip(),
                    ))
                }
            }
        }
    }

    fn probe<'a>(
        &self,
        rng: &mut impl Rng,
        candidates: impl ExactSizeIterator<Item = Candidate<'a>>,
        context: &impl Context,
    ) -> f64 {
        if let Some((index, mut candidate, mut estimate)) = self.select(rng, candidates) {
            let mut path = Vec::with_capacity(50);
            path.push(index);
            loop {
                if let Some(mut candidates) = candidate.expand(context) {
                    assert!(candidates.len() > 0);

                    if candidates.len() == 1 {
                        candidate = ExpandedCandidate::new(candidates.pop().unwrap());
                    } else if let Some((index, next, inv_prob)) =
                        self.select(rng, candidates.into_iter())
                    {
                        path.push(index);
                        candidate = next;
                        estimate *= inv_prob;
                    } else {
                        println!("Path: {:?}", path);
                        break;
                    }
                } else {
                    println!("Path: {:?}", path);
                    break;
                }
            }
            estimate
        } else {
            0f64
        }
    }
}

use kernels::{linalg, Kernel};
use telamon::device::x86;

fn main() {
    env_logger::init();

    let rng = &mut thread_rng();
    let proba = EdgeProbability::CompleteTreeSizeRatio { epsilon: 0.1f64 };
    let num_descents = 100;

    let estimates = linalg::MatMul::<f32>::with_candidates(
        linalg::MatMulP::new(1024, 1024, 1024),
        true,
        &mut x86::Context::default(),
        move |candidates, context| {
            let mut estimates = Vec::with_capacity(num_descents);
            for ix in 0..num_descents {
                if let Some((_, candidate, mut estimate)) =
                    proba.select(rng, candidates.iter())
                {
                    if let Some(children) = candidate.expand(context) {
                        estimate *= proba.probe(rng, children.into_iter(), context);
                    }
                    estimates.push(estimate);
                } else {
                    estimates.push(1f64);
                }
                println!("Doing {} [{:?}]", ix, estimates);
            }
            estimates
        },
    );

    println!(
        "Got avg of {}",
        estimates.iter().sum::<f64>() / estimates.len() as f64
    );
}
<
