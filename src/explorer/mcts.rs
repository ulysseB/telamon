//! Search space exploration using MCTS-style algorithm
//!
//! This module provides functionality for exposing branch-and-bound candidate trees, as well as
//! MCTS-style algorithms for exploring them.
//!
//! The trees are defined using `Node` and `Edge`, and a trace-enabled `NodeCursor` is provided to
//! move in the tree while recording a sequence of events.
//!
//! The MctsWalker, parameterized with the appropriate bandit and rollout policies, can be used to
//! explore the resulting search tree in an MCTS-like fashion.

#![allow(clippy::type_complexity)]

use std::cell::RefCell;
use std::cmp::PartialEq;
use std::fmt::{self, Debug, Display};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    mpsc, Arc, RwLock, Weak,
};
use std::{cmp, iter, ops, slice};

use rand::distributions::{Weighted, WeightedChoice};
use rand::prelude::*;
use rpds::List;
use serde::{Deserialize, Serialize};
use utils::cmp_f64;

use crate::device::Context;
use crate::explorer::{
    candidate::Candidate,
    choice::{self, ActionEx as Action},
    config::{self, BanditConfig, ChoiceOrdering, NewNodeOrder},
    logger::LogMessage,
    store::Store,
};
use crate::model::{bound, Bound};
use crate::search_space::SearchSpace;

/// Newtype wrapper to represent a node identifier.  Node identifiers should be unique inside a
/// tree.  We use a fixed-size representation for consistency of the serialization format.
#[derive(Copy, Clone, Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
#[repr(transparent)]
pub struct NodeId(u64);

impl Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "#{}", self.0)
    }
}

impl From<NodeId> for u64 {
    fn from(v: NodeId) -> Self {
        v.0
    }
}

/// Newtype wrapper to represent an edge index.  Like `NodeId`, we use a fixed-size representation
/// for consistency of the serialization format.
#[derive(Copy, Clone, Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
#[repr(transparent)]
pub struct EdgeIndex(u16);

impl Display for EdgeIndex {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, ".[{}]", self.0)
    }
}

impl From<EdgeIndex> for u16 {
    fn from(v: EdgeIndex) -> Self {
        v.0
    }
}

/// The possible causes for which a node can be killed.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum CauseOfDeath {
    /// Unsatisfied constraints.
    Constraints,
    /// Cut by the performance model.
    PerfModel { cut: f64 },
    /// All child nodes are dead.
    Backtrack,
}

/// The internal structure of a node.  This should only be accessed through `Node` getters.
struct NodeInner<'c, N, E> {
    /// Node identifier.  Unique in a single tree.
    id: NodeId,

    /// Depth in the tree.   This is zero for the root.
    depth: usize,

    /// Parent edge.  `None` for the root.
    ///
    /// This is directly a pointer to the parent node and edge index in order to avoid an
    /// additional indirection through the `Edge` when following a path upwards.
    parent: Option<(WeakNode<'c, N, E>, EdgeIndex)>,

    /// Child edges.  Edges have a backward pointer to the node and additional metadata such as the
    /// action it correspond to.
    children: Vec<Edge<'c, N, E>>,

    /// Bound from the performance model.  If None, the node was dead due to constraint propagation
    /// and was never live.
    bound: Option<Box<Bound>>,

    /// Whether the node is dead.
    dead: AtomicBool,

    /// Whether the node was expanded.
    ///
    /// We use a lock instead of an atomic because during the expansion we take the candidate and
    /// store candidates on the child nodes and we want that to happen only once.  Using a lock on
    /// `expanded` is the easy way out.
    /// TODO(bclement): Move to data.
    expanded: RwLock<bool>,

    /// An optional candidate can be stored in the node for later retrieval.  When a node is killed
    /// the candidate is immediately cleared to free memory.
    /// TODO(bclement): Move to data.
    candidate: RwLock<Option<Box<SearchSpace<'c>>>>,

    /// Additional algorithm-specific data associated with the node.
    #[allow(dead_code)]
    data: N,
}

/// Represents a node in the search tree.  This is represented by a reference-counted pointer to an
/// internal structure; multiple clones point to the same in-memory structure.
pub struct Node<'c, N, E> {
    inner: Arc<NodeInner<'c, N, E>>,
}

impl<'c, N, E> Clone for Node<'c, N, E> {
    fn clone(&self) -> Self {
        Node {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<'c, N, E> PartialEq for Node<'c, N, E> {
    fn eq(&self, other: &Node<'c, N, E>) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl<'c, N, E> Node<'c, N, E> {
    /// The node identifier.  Node identifiers are unique for nodes in the same tree.
    pub fn id(&self) -> NodeId {
        self.inner.id
    }

    /// Depth of the node.  The root is at depth 0.
    pub fn depth(&self) -> usize {
        self.inner.depth
    }

    /// Returns the actions used for creating the node.  The actions are returned in the reverse
    /// order in which they were taken, i.e. the first action taken is last in the vector.
    pub fn rev_actions(&self) -> Vec<Action> {
        let mut actions = Vec::with_capacity(self.depth());

        let mut node = self.clone();
        while let Some((parent, index)) = node.inner.parent.as_ref() {
            let parent = parent.upgrade().expect("node removed from tree");
            actions.push(parent[*index].action().clone());
            node = parent;
        }

        actions
    }

    /// List of actions used for creating the node, in order of creation.
    pub fn actions(&self) -> Vec<Action> {
        let mut actions = self.rev_actions();
        actions.reverse();
        actions
    }

    /// Bound from the performance model.  If `None`, the node is dead and was killed by constraint
    /// propagation.
    pub fn bound(&self) -> Option<&Bound> {
        self.inner.bound.as_ref().map(Box::as_ref)
    }

    /// Returns whether the node is still live.
    pub fn is_live(&self) -> bool {
        self.inner.bound.is_some() && !self.inner.dead.load(Ordering::SeqCst)
    }

    fn kill(&self) {
        self.inner.dead.store(true, Ordering::SeqCst);

        // Clear the candidate if there is one.  Always do it after killing the node so that if
        // somebody tries to take the candidate and it fails, they will always see the node as dead
        // afterwards.
        self.take_candidate();

        // TODO: node.data.clear()
        // TODO: for child in node.children { child.data.clear() }
    }

    /// Returns whether the node is an implementation, i.e. represents a fully-specified candidate.
    ///
    /// An node is an implementation if it has a bound (i.e. was not killed by constraint
    /// propagation) and has no children.
    fn is_implementation(&self) -> bool {
        self.inner.bound.is_some() && self.inner.children.is_empty()
    }

    /// List the child edges.
    fn edges(&self) -> &[Edge<'c, N, E>] {
        &self.inner.children[..]
    }

    /// Create a new weak pointer to the node.
    fn downgrade(&self) -> WeakNode<'c, N, E> {
        WeakNode {
            inner: Arc::downgrade(&self.inner),
        }
    }

    /// Pointer to the algorithm-specific data payload.
    #[allow(dead_code)]
    fn data(&self) -> &N {
        &self.inner.data
    }

    /// Return whether the node is already expanded.
    fn is_expanded(&self) -> bool {
        *self.inner.expanded.read().expect("expanded: poisoned")
    }

    /// Take the candidate if there is one.
    fn take_candidate(&self) -> Option<SearchSpace<'c>> {
        self.inner
            .candidate
            .write()
            .expect("candidate: poisoned")
            .take()
            .map(|candidate| *candidate)
    }

    /// Store a candidate in the node.
    fn store_candidate(&self, candidate: SearchSpace<'c>) {
        *self.inner.candidate.write().expect("candidate: poisoned") =
            Some(Box::new(candidate));
    }
}

impl<'c, N, E> ops::Index<EdgeIndex> for Node<'c, N, E> {
    type Output = Edge<'c, N, E>;

    fn index(&self, index: EdgeIndex) -> &Self::Output {
        &self.inner.children[index.0 as usize]
    }
}

/// Non-owning reference to a node.  The node can be accessed through `upgrade`.
pub struct WeakNode<'c, N, E> {
    inner: Weak<NodeInner<'c, N, E>>,
}

impl<'c, N, E> WeakNode<'c, N, E> {
    /// Attempts to upgrade to a `Node`.  Returns `None` if the node has since been dropped.
    fn upgrade(&self) -> Option<Node<'c, N, E>> {
        Weak::upgrade(&self.inner).map(|inner| Node { inner })
    }
}

/// The internal structure of an edge.  This should only be accessed through `Edge` getters.
struct EdgeInner<'c, N, E> {
    /// The node pointed to by the edge.
    node: RwLock<Option<Node<'c, N, E>>>,

    /// Edge index across the parent's children
    index: EdgeIndex,

    /// Action associated with the edge.
    action: Action,

    /// Additional algorithm-specific data associated with the edge.
    data: E,
}

/// An edge in the search tree, which can contain additional data.
pub struct Edge<'c, N, E> {
    inner: Arc<EdgeInner<'c, N, E>>,
}

impl<'c, N, E> Clone for Edge<'c, N, E> {
    fn clone(&self) -> Self {
        Edge {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<'c, N, E> PartialEq for Edge<'c, N, E> {
    fn eq(&self, other: &Edge<'c, N, E>) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl<'c, N, E> Edge<'c, N, E> {
    /// Call a function on the destination node if there is one, and returns the option-wrapped
    /// result.
    pub fn try_with_node<F, T>(&self, func: F) -> Option<T>
    where
        F: FnOnce(&Node<'c, N, E>) -> T,
    {
        self.inner
            .node
            .read()
            .expect("node: poisoned")
            .as_ref()
            .map(func)
    }

    /// Edge index across the parent's children.
    pub fn index(&self) -> EdgeIndex {
        self.inner.index
    }

    /// The action associated with this edge.
    pub fn action(&self) -> &Action {
        &self.inner.action
    }

    /// Algorithm-specific data associated with the edge.
    pub fn data(&self) -> &E {
        &self.inner.data
    }
}

/// An environment in which candidates can be refined.
#[derive(Clone)]
pub struct Env<'a> {
    /// The order in which choices should be considered.
    choice_ordering: &'a ChoiceOrdering,
    /// The context to use for constraint propagation.
    context: &'a dyn Context,
}

impl<'a> Env<'a> {
    /// Create a new environment.
    pub fn new(choice_ordering: &'a ChoiceOrdering, context: &'a dyn Context) -> Self {
        Env {
            choice_ordering,
            context,
        }
    }

    /// List the available actions for a candidate.
    ///
    /// This includes all actions, even those that may be removed by further propagation.  Hence,
    /// the resulting vector is empty only when the candidate is a fully-specified implementation.
    pub fn list_actions(&self, candidate: &SearchSpace<'_>) -> Vec<Action> {
        choice::list(self.choice_ordering, candidate)
            .next()
            .unwrap_or_default()
    }

    /// Apply an action to an existing candidate, consuming the existing candidate.
    pub fn apply_action<'c>(
        &self,
        mut candidate: SearchSpace<'c>,
        action: Action,
    ) -> Option<SearchSpace<'c>> {
        if let Ok(()) = match action {
            Action::Action(action) => candidate.apply_decisions(vec![action]),
            Action::LowerLayout {
                mem,
                st_dims,
                ld_dims,
            } => candidate.lower_layout(mem, &st_dims, &ld_dims),
        } {
            Some(candidate)
        } else {
            None
        }
    }

    /// Compute the performance model bound for a candidate.
    pub fn bound<'c>(&self, candidate: &SearchSpace<'c>) -> Bound {
        bound(candidate, self.context)
    }
}

/// The types of policy used.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Policy {
    Bandit,
    Default,
}

/// The possible events in a trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Event {
    /// Move to a node given by its ID.  This is typically used at the start of the trace and for
    /// backtracking purposes.
    SelectNode(NodeId),
    /// Select the `n`th child of the current node.
    SelectChild(EdgeIndex, Policy, Selector<EdgeIndex>),
    /// Expand the current node.
    Expand,
    /// Kill the current node for the given reason.
    Kill(CauseOfDeath),
    /// Kill the `n`th child of the current node for the given reason.
    KillChild(EdgeIndex, CauseOfDeath),
    /// An implementation was found.  This should occur at most once per trace and will be the last
    /// event on the trace when it does occur.
    Implementation,
}

/// Wrapper struct to annotate events with timing information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Timed<T> {
    pub start_time: std::time::Duration,
    pub end_time: std::time::Duration,
    pub value: T,
}

/// A log message.
///
/// The way that the log is generated ensures that all nodes referenced in a `Trace` or
/// `Evaluation` event will have had a corresponding `Node` event previously in the log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Message {
    /// A new node was created
    Node {
        /// Id of the newly created node
        id: NodeId,
        /// Parent of the node, if it exists.  This is a pair containing the node identifier of the
        /// parent and index in the `children` list.
        parent: Option<(NodeId, EdgeIndex)>,
        /// All available children for this node.  This includes *all* actions ever considered,
        /// even those that end up being forbidden due to constraint propagation.
        children: Vec<Action>,
        /// Explained bound from the performance model.  `None` if the node was killed due to
        /// constraint propagation, in which case we can't run the performance model.
        bound: Option<Bound>,
        /// Time at which the node was discovered.
        discovery_time: std::time::Duration,
    },

    /// Sequence of actions (moves in the tree) performed by a specific thread.  Starts at the root
    /// of the tree.
    Trace {
        /// The thread performing the descent.  There can be multiple traces per thread, and they
        /// will share the `thread` field.
        thread: String,
        /// Sequence of events performed by the thread.
        events: Vec<Timed<Event>>,
    },

    /// A node was evaluated
    Evaluation {
        /// Identifier of the evaluated node
        id: NodeId,
        /// Evaluation result.  If `None`, the node was cut at evaluation time or otherwise timed
        /// out.
        value: Option<f64>,
        /// Time at which the evaluation results were made available and backpropagation started.
        result_time: std::time::Duration,
    },
}

/// A path in the tree.
pub struct Trace<'c, N, E> {
    /// List of edges taken.  For each edge, we also record the policy that was used to select it,
    /// so that it can be used appropriately for backpropagation.
    path: Vec<(Policy, WeakNode<'c, N, E>, EdgeIndex)>,
    /// The final node reached at the end of the trace.  This is provided for convenience and
    /// should always be the node pointed to by the last edge in the `path`.
    node: Node<'c, N, E>,
}

/// Helper structure to manipulate a tree.
///
/// Despite the name, the `Tree` does not hold pointers to actual nodes.
pub struct Tree<'a> {
    /// The environment to use for computing children.
    env: Env<'a>,
    /// Sequential counter for ID attribution.
    id_counter: &'a AtomicUsize,
    /// Channel to send log events to.
    logger: &'a mpsc::SyncSender<LogMessage<Message>>,
    /// Time at which exploration started.  Used as an epoch for timestamps.
    epoch: std::time::Instant,
}

impl<'a> Tree<'a> {
    /// Create a new tree.
    fn new(
        env: Env<'a>,
        id_counter: &'a AtomicUsize,
        logger: &'a mpsc::SyncSender<LogMessage<Message>>,
        epoch: std::time::Instant,
    ) -> Self {
        Tree {
            env,
            id_counter,
            logger,
            epoch,
        }
    }

    /// Create a new node.
    ///
    /// If parent is not provided, this will create a root node which must have an associated
    /// candidate.
    pub fn node<'c, N, E>(
        &self,
        parent: Option<(&Node<'c, N, E>, EdgeIndex)>,
        candidate: Option<&SearchSpace<'c>>,
    ) -> Node<'c, N, E>
    where
        N: Default,
        E: Default,
    {
        assert!(parent.is_some() || candidate.is_some());

        let (children, bound);
        if let Some(candidate) = candidate {
            children = self
                .env
                .list_actions(candidate)
                .into_iter()
                .enumerate()
                .map(|(ix, action)| Edge {
                    inner: Arc::new(EdgeInner {
                        node: RwLock::new(None),
                        index: EdgeIndex(ix as u16),
                        action,
                        data: E::default(),
                    }),
                })
                .collect();
            bound = Some(self.env.bound(candidate));
        } else {
            children = Vec::new();
            bound = None;
        }

        let id = NodeId(self.id_counter.fetch_add(1, Ordering::Relaxed) as u64);

        // Log node creation so that other events can refer to it using its ID only.
        self.log(Message::Node {
            id,
            parent: parent.map(|(parent, index)| (parent.id(), index)),
            children: children.iter().map(|edge| edge.action().clone()).collect(),
            bound: bound.clone(),
            discovery_time: self.epoch.elapsed(),
        });

        Node {
            inner: Arc::new(NodeInner {
                id,
                depth: parent.map(|(parent, _)| parent.depth() + 1).unwrap_or(0),
                parent: parent.map(|(parent, index)| (parent.downgrade(), index)),
                children,
                dead: AtomicBool::new(bound.is_none()),
                bound: bound.map(Box::new),
                data: N::default(),
                candidate: RwLock::new(None),
                expanded: RwLock::new(false),
            }),
        }
    }

    fn log(&self, message: Message) {
        self.logger
            .send(LogMessage::Event(message))
            .expect("sending message");
    }
}

/// A cursor which can be moved in the tree and remembers its trajectory.
pub struct NodeCursor<'a, 'c, N, E> {
    events: RefCell<Vec<Timed<Event>>>,
    cut: f64,
    cut_epoch: usize,
    path: Vec<(Policy, WeakNode<'c, N, E>, EdgeIndex)>,
    node: Node<'c, N, E>,
    tree: Tree<'a>,
    helper: WalkHelper<'a>,
}

impl<'a, 'c, N, E> NodeCursor<'a, 'c, N, E>
where
    N: Debug + Default,
    E: Debug + Default,
{
    fn check_stop(mut self) -> Result<Self, Error<'a, 'c, N, E>> {
        if self.helper.stop.load(Ordering::Relaxed) {
            Err(Error::Aborted)
        } else {
            let cut_epoch = self.helper.cut_epoch.load(Ordering::Relaxed);
            if cut_epoch != self.cut_epoch {
                // The cut epoch changed; update the known cut and abort the descent if the current
                // node is dead.
                self.cut_epoch = cut_epoch;
                self.cut = *self.helper.cut.read().expect("cut: poisoned");
                if self.cut() {
                    Err(Error::DeadEnd(self))
                } else {
                    Ok(self)
                }
            } else if self.node.is_live() {
                Ok(self)
            } else {
                Err(Error::DeadEnd(self))
            }
        }
    }

    /// Apply the current cut to the given node and returns whether it is now dead.  This function
    /// only returns false when the node is still live.
    fn cut_node<F>(&self, node: &Node<'c, N, E>, event_fn: F) -> bool
    where
        F: FnOnce(CauseOfDeath) -> Event,
    {
        if node.is_live() {
            if node.bound().unwrap().value() < self.cut {
                return false;
            }

            self.kill_node(node, CauseOfDeath::PerfModel { cut: self.cut }, event_fn);
        }

        true
    }

    /// Apply the current cut to the pointed-to node.  Returns `true` when the node has been cut or
    /// was otherwise already dead, and `false` if the node is still live.
    pub fn cut(&self) -> bool {
        self.cut_node(&self.node, Event::Kill)
    }

    /// An iterator on the live children of the pointed-to node.
    ///
    /// If the nodes pointed to by some of the edges don't exist, return the edge instead (it may
    /// be live, but the candidate is needed to compute the bound).
    pub fn live_children_iter(
        &'_ self,
    ) -> impl Iterator<
        Item = Result<(&'_ Edge<'c, N, E>, Node<'c, N, E>), &'_ Edge<'c, N, E>>,
    > + '_ {
        self.node
            .edges()
            .iter()
            .map(move |edge| {
                if let Some(opt) = edge.try_with_node(|node| {
                    if self.cut_node(node, |cause| Event::KillChild(edge.index(), cause))
                    {
                        None
                    } else {
                        Some((edge, node.clone()))
                    }
                }) {
                    Ok(opt)
                } else {
                    Err(edge)
                }
            })
            .filter_map(|result| match result {
                Ok(None) => None,
                Ok(Some((edge, node))) => Some(Ok((edge, node))),
                Err(err) => Some(Err(err)),
            })
    }

    /// An iterator on the live children of the pointed-to nodes with their associated search
    /// space.
    ///
    /// The iterated items are tuples `(edge, node, candidate)` where `candidate` is only provided
    /// is the node was freshly created.  If the node is live and the candidate is not provided,
    /// the caller must manually call `apply_action` to get the candidate.
    pub fn live_children_iter_with_candidates<'b>(
        &'b self,
        candidate: &'b SearchSpace<'c>,
    ) -> impl Iterator<Item = (&'b Edge<'c, N, E>, Node<'c, N, E>, Option<SearchSpace<'c>>)> + 'b
    {
        self.live_children_iter().filter_map(move |result| {
            result
                .map(|(edge, node)| (edge, node, None))
                .or_else(|edge| {
                    // Note that we have a lock on the node here, so we can't call methods which
                    // would try to acquire one, even for assertion purposes!
                    let mut node = edge.inner.node.write().expect("node: poisoned");
                    if let Some(node) = &*node {
                        return if self.cut_node(&node, |cause| {
                            Event::KillChild(edge.index(), cause)
                        }) {
                            Err(())
                        } else {
                            Ok((edge, node.clone(), None))
                        };
                    }

                    let child = self
                        .tree
                        .env
                        .apply_action(candidate.clone(), edge.action().clone());
                    let child_node = self
                        .tree
                        .node(Some((&self.node, edge.index())), child.as_ref());

                    if child.is_none() {
                        assert!(!child_node.is_live());
                        self.kill_node(&child_node, CauseOfDeath::Constraints, |cause| {
                            Event::KillChild(edge.index(), cause)
                        });
                    }

                    if self.cut_node(&child_node, |cause| {
                        Event::KillChild(edge.index(), cause)
                    }) {
                        *node = Some(child_node);
                        Err(())
                    } else {
                        assert!(child.is_some());
                        *node = Some(child_node.clone());
                        Ok((edge, child_node, child))
                    }
                })
                .ok()
        })
    }

    /// Kill the given node.
    fn kill_node<F>(&self, node: &Node<'c, N, E>, cause: CauseOfDeath, event_fn: F)
    where
        F: FnOnce(CauseOfDeath) -> Event,
    {
        // TODO: Do not overwrite cause if there already is one?
        self.event(self.tree.epoch.elapsed(), event_fn(cause));

        node.kill();
    }

    /// Kill the currently pointed-to node.
    pub fn kill(&self, cause: CauseOfDeath) {
        self.kill_node(&self.node, cause, Event::Kill)
    }

    /// Log an event which started at `start_time` and ended now.
    fn event(&self, start_time: std::time::Duration, event: Event) {
        self.events.borrow_mut().push(Timed {
            start_time,
            end_time: self.tree.epoch.elapsed(),
            value: event,
        });
    }

    pub fn evaluate(
        self,
        candidate: SearchSpace<'c>,
    ) -> Result<(SearchSpace<'c>, Trace<'c, N, E>), Self> {
        if self.cut() {
            Err(self)
        } else {
            self.event(self.tree.epoch.elapsed(), Event::Implementation);
            self.tree.log(Message::Trace {
                thread: format!("{:?}", std::thread::current().id()),
                events: self.events.into_inner(),
            });

            Ok((
                candidate,
                Trace {
                    path: self.path,
                    node: self.node,
                },
            ))
        }
    }

    pub fn deadend(self) {
        self.tree.log(Message::Trace {
            thread: format!("{:?}", std::thread::current().id()),
            events: self.events.into_inner(),
        });
    }

    /// Make a checkpoint of the current pointed-to node then call `func`.  If `func` fails,
    /// restore the resulting cursor to the checkpointed state.
    fn checkpoint<F, T>(self, func: F) -> Result<Result<T, Error<'a, 'c, N, E>>, Self>
    where
        F: FnOnce(Self) -> Result<Result<T, Error<'a, 'c, N, E>>, Self>,
    {
        let path_len = self.path.len();
        let checkpoint = self.node.clone();

        let result = if self.helper.config.backtrack_deadends {
            match func(self) {
                Ok(Err(Error::DeadEnd(cursor))) => Err(cursor),
                result => result,
            }
        } else {
            func(self)
        };

        match result {
            Ok(v) => Ok(v),
            Err(mut cursor) => {
                cursor.event(
                    cursor.tree.epoch.elapsed(),
                    Event::SelectNode(checkpoint.id()),
                );
                cursor.path.truncate(path_len);
                cursor.node = checkpoint;
                Err(cursor)
            }
        }
    }

    /// Select a child node.
    ///
    /// If `func` returns `Some`, the cursor moves to the corresponding child and is returned with
    /// the associated payload in an `Ok` result; otherwise, the cursor is left unmodified and
    /// returned in an `Err` result.
    fn select_child<F, T>(mut self, func: F) -> Result<(Self, T), Self>
    where
        F: FnOnce(
            &Self,
        )
            -> Option<(Policy, Selector<EdgeIndex>, EdgeIndex, Node<'c, N, E>, T)>,
    {
        let start_time = self.tree.epoch.elapsed();
        if let Some((policy, selector, eindex, node, value)) = func(&self) {
            self.event(start_time, Event::SelectChild(eindex, policy, selector));
            self.path.push((policy, self.node.downgrade(), eindex));
            self.node = node;
            Ok((self, value))
        } else {
            Err(self)
        }
    }

    /// Expand the currently pointed-to node and returns the resulting candidate.
    ///
    /// Returns `None` if the pointed-to node is either dead or already expanded.
    pub fn expand(&self) -> Option<SearchSpace<'c>> {
        if self.cut() || self.node.is_expanded() {
            return None;
        }

        let start_time = self.tree.epoch.elapsed();

        let mut expanded_mut = self
            .node
            .inner
            .expanded
            .write()
            .expect("expanded: poisoned");
        // Someone may have beaten us to the punch
        if *expanded_mut {
            return None;
        }

        if let Some(candidate) = self.node.take_candidate() {
            for (edge, node, child_candidate) in
                self.live_children_iter_with_candidates(&candidate)
            {
                node.store_candidate(child_candidate.unwrap_or_else(|| {
                    self.tree
                        .env
                        .apply_action(candidate.clone(), edge.action().clone())
                        .unwrap()
                }));
            }

            *expanded_mut = true;

            self.event(start_time, Event::Expand);
            Some(candidate)
        } else {
            // If there is no candidate it must be because the node was killed
            assert!(!self.node.is_live());

            None
        }

        // Lock on `expanded` is released here
    }
}

/// Errors which we can encounter during a descent
enum Error<'a, 'c, N, E> {
    /// A dead-end was encountered
    DeadEnd(NodeCursor<'a, 'c, N, E>),
    /// The search was aborted e.g. due to a timeout
    Aborted,
}

impl<'a, 'c, N, E> Debug for Error<'a, 'c, N, E> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::DeadEnd(_cursor) => write!(f, "DeadEnd(NodeCursor {{ .. }})"),
            Error::Aborted => write!(f, "Aborted"),
        }
    }
}

impl<'a, 'c, N, E> Display for Error<'a, 'c, N, E> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::DeadEnd(cursor) => write!(f, "dead end at node {}", cursor.node.id()),
            Error::Aborted => write!(f, "aborted"),
        }
    }
}

impl<'a, 'c, N, E> std::error::Error for Error<'a, 'c, N, E> {}

pub trait TreePolicy<'c, N: 'c, E: 'c>: Send + Sync {
    fn pick_child(
        &'_ self,
        cut: f64,
        children: &NodeView<'_, 'c, N, E>,
    ) -> Option<(EdgeViewIndex, Selector<EdgeIndex>)>;

    fn backpropagate(
        &'_ self,
        _parent: &'_ Node<'c, N, E>,
        _index: EdgeIndex,
        _eval: Option<f64>,
    ) {
    }
}

#[derive(Copy, Clone)]
struct WalkHelper<'a> {
    stop: &'a AtomicBool,
    cut: &'a RwLock<f64>,
    cut_epoch: &'a AtomicUsize,
    config: &'a BanditConfig,
}

/// Helper structure to walk the tree following a specific policy.
struct PolicyWalker<'a, 'c, N, E> {
    policy: &'a dyn TreePolicy<'c, N, E>,
}

impl<'a, 'c, N, E> PolicyWalker<'a, 'c, N, E>
where
    N: 'c + Send + Sync + Debug + Default,
    E: 'c + Send + Sync + Debug + Default,
{
    fn walk(
        &self,
        mut cursor: NodeCursor<'a, 'c, N, E>,
        candidate: SearchSpace<'c>,
    ) -> Result<(SearchSpace<'c>, Trace<'c, N, E>), Error<'a, 'c, N, E>> {
        // If we point to an implementation, we are done.  We mark it as dead to avoid evaluating
        // it again later.
        if cursor.node.is_implementation() {
            return cursor
                .evaluate(candidate)
                .map(|(candidate, trace)| {
                    trace.node.kill();
                    (candidate, trace)
                })
                .map_err(Error::DeadEnd);
        }

        loop {
            cursor = match cursor.check_stop()?.checkpoint(|cursor| {
                cursor
                    .select_child(|cursor| {
                        let (mut edges, mut candidates): (Vec<_>, Vec<_>) = cursor
                            .live_children_iter_with_candidates(&candidate)
                            .map(|(edge, node, child_candidate)| {
                                ((edge, node), child_candidate)
                            })
                            .unzip();

                        if let Some((index, selector)) = self
                            .policy
                            .pick_child(cursor.cut, &NodeView::new(&cursor.node, &edges))
                        {
                            let (edge, node) = edges.swap_remove(usize::from(index));
                            let child_candidate =
                                candidates.swap_remove(usize::from(index));

                            Some((
                                Policy::Default,
                                selector,
                                edge.index(),
                                node,
                                child_candidate.unwrap_or_else(|| {
                                    cursor
                                        .tree
                                        .env
                                        .apply_action(
                                            candidate.clone(),
                                            edge.action().clone(),
                                        )
                                        .unwrap()
                                }),
                            ))
                        } else {
                            assert!(edges.iter().all(|(_edge, node)| !node.is_live()));

                            cursor.kill(CauseOfDeath::Backtrack);
                            None
                        }
                    })
                    .map(|(cursor, candidate)| self.walk(cursor, candidate))
            }) {
                Ok(result) => return result,
                Err(cursor) => cursor,
            }
        }
    }
}

/// Helper structure to walk the tree using a MCTS algorithm.
struct MctsWalker<'a, 'c, N, E> {
    /// The walker to use to perform rollouts.
    default_walker: PolicyWalker<'a, 'c, N, E>,
    /// The policy to use in the explicit tree where statistics are available.
    tree_policy: &'a dyn TreePolicy<'c, N, E>,
}

impl<'a, 'c, N, E> MctsWalker<'a, 'c, N, E>
where
    N: 'c + Send + Sync + Debug + Default,
    E: 'c + Send + Sync + Debug + Default,
{
    /// Evaluate the underlying node
    fn evaluate(
        &self,
        cursor: NodeCursor<'a, 'c, N, E>,
        candidate: SearchSpace<'c>,
    ) -> Result<(SearchSpace<'c>, Trace<'c, N, E>), Error<'a, 'c, N, E>> {
        self.default_walker.walk(cursor, candidate)
    }

    /// Select a child in the explicit tree.  The node pointed to by the cursor must already be
    /// expanded (= in the explicit tree), and must not be an implementation (in that case it
    /// should be evaluated instead).
    fn select_intree(
        &self,
        mut cursor: NodeCursor<'a, 'c, N, E>,
    ) -> Result<(SearchSpace<'c>, Trace<'c, N, E>), Error<'a, 'c, N, E>> {
        assert!(cursor.node.is_expanded(), "not in the explicit tree");
        assert!(
            !cursor.node.is_implementation(),
            "implementation in the explicit tree"
        );

        // Information about the selected chidl
        struct SelectedChild {
            // Whether the child was selected for expansion or not
            expand: bool,
        };

        loop {
            cursor = match cursor.check_stop()?.checkpoint(|cursor| {
                cursor
                    .select_child(|cursor| {
                        let mut expanded = Vec::with_capacity(cursor.node.edges().len());
                        let mut unexpanded = Vec::new();
                        for (edge, node) in cursor
                            .live_children_iter()
                            .map(|r| r.ok().expect("missing in-tree child"))
                        {
                            if node.is_expanded() {
                                expanded.push((edge, node));
                            } else {
                                unexpanded.push((edge, node));
                            }
                        }

                        if let Some((index, selector)) =
                            self.default_walker.policy.pick_child(
                                cursor.cut,
                                &NodeView::new(&cursor.node, &unexpanded),
                            )
                        {
                            let (edge, node) = unexpanded.swap_remove(usize::from(index));

                            Some((
                                Policy::Default,
                                selector,
                                edge.index(),
                                node,
                                SelectedChild { expand: true },
                            ))
                        } else {
                            assert!(
                                unexpanded
                                    .into_iter()
                                    .all(|(_edge, node)| !node.is_live()),
                                "live unexpanded child was not selected"
                            );

                            if let Some((index, selector)) = self.tree_policy.pick_child(
                                cursor.cut,
                                &NodeView::new(&cursor.node, &expanded),
                            ) {
                                let (edge, node) =
                                    expanded.swap_remove(usize::from(index));

                                Some((
                                    Policy::Bandit,
                                    selector,
                                    edge.index(),
                                    node,
                                    SelectedChild { expand: false },
                                ))
                            } else {
                                assert!(
                                    expanded.iter().all(|(_edge, node)| !node.is_live()),
                                    "live child was not selected"
                                );

                                cursor.kill(CauseOfDeath::Backtrack);
                                None
                            }
                        }
                    })
                    .and_then(|(cursor, selected)| {
                        if selected.expand {
                            if let Some(candidate) = cursor.expand() {
                                Ok(self.evaluate(cursor, candidate))
                            } else {
                                // If expansion fails, the node was killed or expanded by
                                // another thread between the selection and expansion step;
                                // retry with another child.
                                Err(cursor)
                            }
                        } else {
                            assert!(cursor.node.is_expanded());

                            if cursor.node.is_implementation() {
                                // If the target node is an implementation but was expanded, we
                                // must have selected it while another thread was expanding it but
                                // before it marked it as already evaluated; try again.
                                Err(cursor)
                            } else {
                                Ok(self.select_intree(cursor))
                            }
                        }
                    })
            }) {
                Ok(result) => return result,
                Err(cursor) => cursor,
            }
        }
    }
}

/// Wrapper to interact with the `Store` trait.
pub struct MctsStore<'a, 'c, N, E> {
    root: Node<'c, N, E>,

    default_policy: Box<dyn TreePolicy<'c, N, E>>,

    tree_policy: Box<dyn TreePolicy<'c, N, E>>,

    /// Best evaluation found so far
    cut: RwLock<f64>,

    cut_epoch: AtomicUsize,

    /// Whether evaluation should be stopped
    stop: AtomicBool,

    /// Counter for the node IDs
    id_counter: AtomicUsize,

    /// Sender to the log queue
    logger: mpsc::SyncSender<LogMessage<Message>>,

    /// Bandit configuration
    config: &'a BanditConfig,

    /// Time at which the search started.  Used as an epoch for timestamps.
    epoch: std::time::Instant,
}

impl<'a, 'c, N, E> MctsStore<'a, 'c, N, E>
where
    N: Send + Sync + Debug + Default,
    E: Send + Sync + Debug + Default,
{
    pub fn new(
        space: SearchSpace<'c>,
        context: &dyn Context,
        config: &'a BanditConfig,
        tree_policy: Box<dyn TreePolicy<'c, N, E>>,
        default_policy: Box<dyn TreePolicy<'c, N, E>>,
        logger: mpsc::SyncSender<LogMessage<Message>>,
    ) -> Self {
        let epoch = std::time::Instant::now();

        let id_counter = AtomicUsize::new(0);
        let root = Tree::new(
            Env::new(&config.choice_ordering, context),
            &id_counter,
            &logger,
            epoch,
        )
        .node(None, Some(&space));
        root.store_candidate(space);

        MctsStore {
            root,
            default_policy,
            tree_policy,
            cut: RwLock::new(config.initial_cut.unwrap_or(std::f64::INFINITY)),
            cut_epoch: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
            id_counter,
            logger,
            config,
            epoch,
        }
    }

    fn cursor<'b>(&'b self, context: &'b dyn Context) -> NodeCursor<'b, 'c, N, E> {
        NodeCursor {
            events: Vec::new().into(),
            cut: *self.cut.read().expect("cut: poisoned"),
            cut_epoch: self.cut_epoch.load(Ordering::Relaxed),
            path: Vec::new(),
            node: self.root.clone(),
            tree: Tree::new(
                Env::new(&self.config.choice_ordering, context),
                &self.id_counter,
                &self.logger,
                self.epoch,
            ),
            helper: WalkHelper {
                stop: &self.stop,
                cut: &self.cut,
                cut_epoch: &self.cut_epoch,
                config: self.config,
            },
        }
    }

    fn walker(&self) -> MctsWalker<'_, 'c, N, E> {
        MctsWalker {
            default_walker: PolicyWalker {
                policy: self.default_policy.as_ref(),
            },
            tree_policy: self.tree_policy.as_ref(),
        }
    }
}

impl<'a, 'c, N: 'c, E: 'c> Store<'c> for MctsStore<'a, 'c, N, E>
where
    N: Send + Sync + Debug + Default,
    E: Send + Sync + Debug + Default,
{
    type PayLoad = Trace<'c, N, E>;

    type Event = Message;

    fn update_cut(&self, new_cut: f64) {
        *self.cut.write().expect("cut: poisoned") = new_cut;
        self.cut_epoch.fetch_add(1, Ordering::Relaxed);

        // TODO: trim the tree?
    }

    fn commit_evaluation(
        &self,
        _actions: &List<choice::ActionEx>,
        trace: Self::PayLoad,
        eval: f64,
    ) {
        let result_time = self.epoch.elapsed();
        let id = trace.node.id();
        let eval = if eval.is_finite() { Some(eval) } else { None };

        // Backpropagate only when the parent is expanded
        for (policy, parent, index) in trace.path {
            match policy {
                Policy::Bandit => {
                    if let Some(parent) = parent.upgrade() {
                        if parent.is_expanded() {
                            self.tree_policy.backpropagate(&parent, index, eval);
                        }
                    }
                }
                Policy::Default => {}
            }
        }

        self.logger
            .send(LogMessage::Event(Message::Evaluation {
                id,
                result_time,
                value: eval,
            }))
            .expect("sending message");
    }

    fn explore(&self, context: &dyn Context) -> Option<(Candidate<'c>, Self::PayLoad)> {
        loop {
            let cursor = self.cursor(context);
            let walker = self.walker();

            // Stop if the root node is dead.
            if cursor.cut() {
                break None;
            }

            // Expand the root node if it has not yet been expanded
            if !cursor.node.is_expanded() {
                if let Some(candidate) = cursor.expand() {
                    match walker.evaluate(cursor, candidate) {
                        Ok((candidate, trace)) => break Some((candidate, trace)),
                        Err(Error::DeadEnd(cursor)) => {
                            cursor.deadend();
                            continue;
                        }
                        Err(_err) => break None,
                    }
                }
            }

            // Otherwise perform monte-carlo selection
            match walker.select_intree(cursor) {
                Ok((candidate, trace)) => break Some((candidate, trace)),
                Err(Error::DeadEnd(cursor)) => {
                    cursor.deadend();
                    continue;
                }
                Err(_err) => break None,
            }
        }
        .map(|(candidate, trace)| {
            (
                Candidate::with_actions(
                    candidate,
                    trace.node.bound().unwrap().clone(),
                    trace.node.actions(),
                ),
                trace,
            )
        })
    }

    fn stop_exploration(&self) {
        self.stop.store(true, Ordering::Relaxed)
    }

    fn print_stats(&self) {}
}

impl NewNodeOrder {
    pub fn into_selector<IT, T>(self, cut: f64, bounds: IT) -> Option<Selector<T>>
    where
        IT: Iterator<Item = (T, f64)>,
    {
        let bounds = bounds.filter(|(_, b)| *b < cut);
        match self {
            NewNodeOrder::Api => bounds
                .take(1)
                .map(|(idx, _)| idx)
                .next()
                .map(Selector::exact),
            NewNodeOrder::WeightedRandom => {
                if cut.is_infinite() {
                    let epsilon = 1e-6;
                    Selector::try_random(
                        bounds
                            .map(|(idx, b)| (idx, (b + epsilon).recip()))
                            .collect(),
                    )
                } else {
                    Selector::try_random(
                        bounds.map(|(idx, b)| (idx, 1. - b / cut)).collect(),
                    )
                }
            }
            NewNodeOrder::Bound => {
                Selector::try_maximum(bounds.map(|(idx, b)| (idx, -b)).collect())
            }
            NewNodeOrder::Random => {
                Selector::try_random(bounds.map(|(idx, _)| (idx, 1.)).collect())
            }
        }
    }
}

impl<'c, N: 'c, E: 'c> TreePolicy<'c, N, E> for NewNodeOrder {
    fn pick_child(
        &'_ self,
        cut: f64,
        children: &NodeView<'_, 'c, N, E>,
    ) -> Option<(EdgeViewIndex, Selector<EdgeIndex>)> {
        self.into_selector(
            cut,
            children.iter().filter_map(|(idx, _edge, node)| {
                let b = node.bound().unwrap().value();
                if b < cut {
                    Some((idx, b))
                } else {
                    None
                }
            }),
        )
        .map(|selector| children.select_with(selector))
    }
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
    fn exploration_factor(&self, cut: f64) -> f64 {
        use self::config::Normalization;

        match self.normalization {
            Some(Normalization::GlobalBest) => {
                self.exploration_constant * self.reward(cut).abs()
            }
            None => self.exploration_constant,
        }
    }

    fn exploration_term(
        &self,
        cut: f64,
        visits: f64,
        total_visits: f64,
        num_children: usize,
    ) -> f64 {
        use self::config::Formula;

        self.exploration_factor(cut)
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

        let num_visits = stats.common.num_visits();

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

/// A newtype wrapper holding the indices in the `frozen` children list, which typically doesn't
/// contain all the children and hence does not use `EdgeIndex`.
#[derive(Debug, Copy, Clone)]
pub struct EdgeViewIndex(usize);

impl From<EdgeViewIndex> for usize {
    fn from(v: EdgeViewIndex) -> Self {
        v.0
    }
}

type ChildView<'a, 'c, N, E> = (&'a Edge<'c, N, E>, Node<'c, N, E>);

/// A locally frozen view on a node, where only some children may be present.  This typically only
/// contains the children satisfying a certain condition, such as live children or (un)expanded
/// children.
pub struct NodeView<'a, 'c, N, E> {
    #[allow(dead_code)]
    parent: &'a Node<'c, N, E>,
    edges: &'a [ChildView<'a, 'c, N, E>],
}

impl<'a, 'c, N, E> NodeView<'a, 'c, N, E> {
    fn new(parent: &'a Node<'c, N, E>, edges: &'a [ChildView<'a, 'c, N, E>]) -> Self {
        NodeView { parent, edges }
    }

    fn iter(&'_ self) -> ChildViewIter<'_, 'a, 'c, N, E> {
        ChildViewIter {
            iter: self.edges.iter().enumerate(),
        }
    }

    fn select_with(
        &self,
        selector: Selector<EdgeViewIndex>,
    ) -> (EdgeViewIndex, Selector<EdgeIndex>) {
        (
            selector.select(),
            selector.map(|index| self[index].0.index()),
        )
    }
}

impl<'a, 'c, N, E> ops::Index<EdgeViewIndex> for NodeView<'a, 'c, N, E> {
    type Output = (&'a Edge<'c, N, E>, Node<'c, N, E>);

    fn index(&self, index: EdgeViewIndex) -> &Self::Output {
        &self.edges[usize::from(index)]
    }
}

pub struct ChildViewIter<'a, 'b, 'c, N, E> {
    iter: iter::Enumerate<slice::Iter<'a, (&'b Edge<'c, N, E>, Node<'c, N, E>)>>,
}

impl<'a, 'b, 'c, N, E> Iterator for ChildViewIter<'a, 'b, 'c, N, E> {
    type Item = (EdgeViewIndex, &'a &'b Edge<'c, N, E>, &'a Node<'c, N, E>);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter
            .next()
            .map(|(idx, (edge, node))| (EdgeViewIndex(idx), edge, node))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Selector<T> {
    Random { weights: Vec<(T, f64)> },
    Maximum { scores: Vec<(T, f64)> },
    Exact { value: T },
}

impl<T> Selector<T> {
    pub fn try_random(weights: Vec<(T, f64)>) -> Option<Self> {
        if weights.is_empty() {
            None
        } else {
            Some(Selector::Random { weights })
        }
    }

    pub fn try_maximum(scores: Vec<(T, f64)>) -> Option<Self> {
        if scores.is_empty() {
            None
        } else {
            Some(Selector::Maximum { scores })
        }
    }

    pub fn exact(value: T) -> Self {
        Selector::Exact { value }
    }

    pub fn map<F, U>(self, f: F) -> Selector<U>
    where
        F: Fn(T) -> U,
    {
        match self {
            Selector::Random { weights } => Selector::Random {
                weights: weights
                    .into_iter()
                    .map(|(value, w)| (f(value), w))
                    .collect(),
            },
            Selector::Maximum { scores } => Selector::Maximum {
                scores: scores.into_iter().map(|(value, w)| (f(value), w)).collect(),
            },
            Selector::Exact { value } => Selector::Exact { value: f(value) },
        }
    }
}

impl<T: Clone> Selector<T> {
    pub fn select(&self) -> T {
        match self {
            Selector::Random { weights } => {
                let resolution = f64::from(u32::max_value() / weights.len() as u32);
                let total_weight = weights.iter().map(|&(_, w)| w).sum::<f64>();
                let index = WeightedChoice::new(
                    &mut weights
                        .iter()
                        .enumerate()
                        .map(|(idx, &(_, w))| Weighted {
                            item: idx,
                            weight: ((w / total_weight) * resolution) as u32,
                        })
                        .collect::<Vec<_>>(),
                )
                .sample(&mut thread_rng());
                weights[index].0.clone()
            }
            Selector::Maximum { scores } => scores
                .iter()
                .max_by(|(_, lhs), (_, rhs)| cmp_f64(*lhs, *rhs))
                .unwrap()
                .0
                .clone(),
            Selector::Exact { value } => value.clone(),
        }
    }
}

impl<'c, N: 'c> TreePolicy<'c, N, UCTStats> for UCTPolicy {
    fn pick_child(
        &'_ self,
        cut: f64,
        children: &NodeView<'_, 'c, N, UCTStats>,
    ) -> Option<(EdgeViewIndex, Selector<EdgeIndex>)> {
        let stats = children
            .iter()
            .map(|(index, edge, node)| {
                (
                    index,
                    (node.bound().unwrap().value(), self.value(edge.data())),
                )
            })
            .collect::<Vec<_>>();

        // If there are unvisited nodes, pick from them
        NewNodeOrder::WeightedRandom
            .into_selector(
                cut,
                stats
                    .iter()
                    .cloned()
                    .filter(|(_idx, (_bound, (_value, visits)))| {
                        cut.is_infinite() || *visits == 0
                    })
                    .map(|(idx, (b, _))| (idx, b)),
            )
            .or_else(move || {
                // Otherwise apply the UCT formula
                let total_visits = stats
                    .iter()
                    .map(|(_idx, (_bound, (_value, visits)))| visits)
                    .sum::<usize>() as f64;

                let num_children = stats.len();

                Selector::try_maximum(
                    stats
                        .into_iter()
                        .map(|(idx, (_bound, (value, visits)))| {
                            (
                                idx,
                                value
                                    + self.exploration_term(
                                        cut,
                                        visits as f64,
                                        total_visits,
                                        num_children,
                                    ),
                            )
                        })
                        .collect(),
                )
            })
            .map(|selector| {
                let (index, selector) = children.select_with(selector);
                children[index].0.data().down();
                (index, selector)
            })
    }

    fn backpropagate(
        &'_ self,
        parent: &'_ Node<'c, N, UCTStats>,
        index: EdgeIndex,
        eval: Option<f64>,
    ) {
        if let Some(eval) = eval {
            parent[index].data().up(self.reward(eval))
        }
    }
}

#[derive(Debug)]
pub struct CommonStats {
    /// Number of visits across this edge.  Note that this is the number of descents; there may
    /// have been less backpropagations due to dead-ends.
    num_visits: AtomicUsize,
}

impl Default for CommonStats {
    fn default() -> Self {
        CommonStats {
            num_visits: AtomicUsize::new(0),
        }
    }
}

impl CommonStats {
    /// Call when the edge is selected during a descent
    fn down(&self) {
        self.num_visits.fetch_add(1, Ordering::Relaxed);
    }

    /// The number of visits through this edge.
    fn num_visits(&self) -> usize {
        self.num_visits.load(Ordering::Relaxed)
    }
}

#[derive(Debug)]
pub struct UCTStats {
    best_evaluation: RwLock<f64>,

    sum_evaluations: RwLock<f64>,

    common: CommonStats,
}

impl Default for UCTStats {
    fn default() -> Self {
        UCTStats {
            best_evaluation: RwLock::new(std::f64::NEG_INFINITY),
            sum_evaluations: RwLock::new(0f64),
            common: CommonStats::default(),
        }
    }
}

impl UCTStats {
    fn down(&self) {
        self.common.down()
    }

    fn up(&self, eval: f64) {
        {
            let mut best = self
                .best_evaluation
                .write()
                .expect("best_evaluation: poisoned");
            if eval > *best {
                *best = eval;
            }
        }

        *self
            .sum_evaluations
            .write()
            .expect("sum_evaluations: poisoned") += eval;
    }

    fn best_evaluation(&self) -> f64 {
        *self
            .best_evaluation
            .read()
            .expect("best_evaluations: poisoned")
    }

    fn sum_evaluations(&self) -> f64 {
        *self
            .sum_evaluations
            .read()
            .expect("sum_evaluations: poisoned")
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

impl<'c, N: 'c> TreePolicy<'c, N, TAGStats> for TAGPolicy {
    fn pick_child(
        &'_ self,
        cut: f64,
        children: &NodeView<'_, 'c, N, TAGStats>,
    ) -> Option<(EdgeViewIndex, Selector<EdgeIndex>)> {
        // Ignore cut children.  Also, we compute the number of visits beforehand to ensure that it
        // doesn't get changed by concurrent accesses.
        let edges = children
            .iter()
            .map(|(index, edge, node)| {
                (index, (edge, node, edge.data().common.num_visits()))
            })
            .filter(|(_idx, (_edge, node, _num_visits))| {
                node.bound().unwrap().value() < cut
            })
            .collect::<Vec<_>>();

        NewNodeOrder::WeightedRandom
            .into_selector(
                cut,
                edges
                    .iter()
                    .filter(|(_idx, (_edge, _node, num_visits))| *num_visits == 0)
                    .map(|(idx, (_edge, node, _num_visits))| {
                        (*idx, node.bound().unwrap().value())
                    }),
            )
            .or_else(move || {
                // Compute the threshold to use so that we only have `config.topk` children
                let threshold = {
                    let mut evalns = Evaluations::with_capacity(self.topk);
                    for (_idx, (edge, _node, _num_visits)) in &edges {
                        // Evaluations are sorted; we can bail out early.
                        for &eval in &*edge
                            .data()
                            .evaluations
                            .read()
                            .expect("evaluations: poisoned")
                        {
                            if !evalns.record(eval, self.topk) {
                                break;
                            }
                        }
                    }

                    // It could happen that all edges have num_visits > 0 but still we don't have
                    // any recorded evaluations if none of the descents have finished yet.
                    evalns.max().unwrap_or(std::f64::INFINITY)
                };

                let stats = edges
                    .into_iter()
                    .map(|(ix, (edge, _node, num_visits))| {
                        (
                            ix,
                            edge.data()
                                .evaluations
                                .read()
                                .expect("evaluations: poisoned")
                                .count_lte(threshold),
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

                Selector::try_maximum(
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
                        .collect(),
                )
            })
            .map(|selector| {
                let (index, selector) = children.select_with(selector);
                children[index].0.data().down();
                (index, selector)
            })
    }

    fn backpropagate(
        &'_ self,
        parent: &'_ Node<'c, N, TAGStats>,
        index: EdgeIndex,
        eval: Option<f64>,
    ) {
        if let Some(eval) = eval {
            parent[index].data().up(eval, self.topk)
        }
    }
}

/// Holds the TAG statistics for a given edge.
#[derive(Debug)]
pub struct TAGStats {
    /// All evaluations seen for the pointed-to node.
    evaluations: RwLock<Evaluations>,
    common: CommonStats,
}

impl Default for TAGStats {
    fn default() -> Self {
        TAGStats {
            evaluations: RwLock::new(Evaluations::new()),
            common: CommonStats::default(),
        }
    }
}

impl TAGStats {
    /// Called when the edge is selected during a descent
    fn down(&self) {
        self.common.down()
    }

    /// Called when backpropagating across this edge after an evaluation
    fn up(&self, eval: f64, topk: usize) {
        self.evaluations
            .write()
            .expect("evaluations: poisoned")
            .record(eval, topk);
    }
}

/// Holds the evaluations seen for a given node or edge.
#[derive(Debug)]
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
                    && cmp_f64(self.0[pos], threshold) == cmp::Ordering::Equal
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
    type IntoIter = slice::Iter<'a, f64>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

pub struct RoundRobinPolicy;

impl<'c, N: 'c> TreePolicy<'c, N, CommonStats> for RoundRobinPolicy {
    fn pick_child(
        &'_ self,
        _cut: f64,
        view: &NodeView<'_, 'c, N, CommonStats>,
    ) -> Option<(EdgeViewIndex, Selector<EdgeIndex>)> {
        Selector::try_maximum(
            view.iter()
                .map(|(index, edge, _node)| (index, -(edge.data().num_visits() as f64)))
                .collect(),
        )
        .map(|selector| {
            let (index, selector) = view.select_with(selector);
            view[index].0.data().down();
            (index, selector)
        })
    }

    fn backpropagate(
        &self,
        _parent: &Node<'c, N, CommonStats>,
        _index: EdgeIndex,
        _eval: Option<f64>,
    ) {
    }
}
