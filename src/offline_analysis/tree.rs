///! Data structures and function that allow for the recreation of a
///! candidate tree from a log file
use crate::explorer::choice::ActionEx as Action;
use crate::explorer::mcts::{EdgeIndex, NodeId};
use crate::model::Bound;
use std::cell::{Ref, RefCell};
use std::rc::{Rc, Weak};
use std::time::Duration;
use utils::FxHashMap;

/// Outgoing Edge to a child annotated with the action for the
/// child. If `child` is None, the child node corresponding to the
/// action has not been computed or hasn't been added to the tree,
/// yet.
struct ChildEdge {
    action: Action,
    child: Option<Weak<RefCell<CandidateNodeInner>>>,
}

/// Edge to parent with `child_idx` indicating the index within the
/// list of children of `parent`.
pub struct ParentEdge {
    parent: Weak<RefCell<CandidateNodeInner>>,
    child_idx: EdgeIndex,
}

impl From<EdgeIndex> for usize {
    fn from(v: EdgeIndex) -> Self {
        usize::from(u16::from(v))
    }
}

/// Inner object for nodes of in the resconstructed candidate tree, wrapped by the proxy CandidateNode
struct CandidateNodeInner {
    /// Incoming edge from the parent to this node. May be None if
    /// this is the root node.
    incoming_edge: Option<ParentEdge>,

    /// Time at which the node was discovered
    discovery_time: Duration,

    /// Time at which the node was marked as an internal node
    internal_time: Option<Duration>,

    /// Time at which the node was marked as a rollout node
    rollout_time: Option<Duration>,

    /// Time at which the node was marked as an implementation
    implementation_time: Option<Duration>,

    /// Time at which the node was marked as a deadend
    deadend_time: Option<Duration>,

    /// Bound determined by the performance model
    bound: Option<Bound>,

    /// List of edges to the node's children with the corresponding
    /// action.
    outgoing_edges: Vec<ChildEdge>,

    /// ID of this node
    id: NodeId,

    /// Score from the evaluation
    score: Option<f64>,
}

trait ReplaceDurationIfLower {
    /// Set duration to d if currently undefined or if d is lower
    fn replace_if_lower(&mut self, d: Duration);
}

impl ReplaceDurationIfLower for Option<Duration> {
    fn replace_if_lower(&mut self, d: Duration) {
        if self.is_none() || self.unwrap() < d {
            self.replace(d);
        }
    }
}

/// A node of in the resconstructed candidate tree
pub struct CandidateNode {
    inner: Rc<RefCell<CandidateNodeInner>>,
}

/// A node in the re-created tree representing a candidate
impl CandidateNode {
    /// Returns the time at which the node was discovered
    pub fn discovery_time(&self) -> Duration {
        self.inner.borrow().discovery_time
    }

    /// Returns the time at which the node was marked as an internal
    /// node
    pub fn internal_time(&self) -> Option<Duration> {
        self.inner.borrow().internal_time
    }

    /// Returns the time at which the node was marked as a rollout
    /// node
    pub fn rollout_time(&self) -> Option<Duration> {
        self.inner.borrow().rollout_time
    }

    /// Returns the time at which the node was marked as an
    /// implementation
    pub fn implementation_time(&self) -> Option<Duration> {
        self.inner.borrow().implementation_time
    }

    /// Returns the time at which the node was marked as a deadend
    pub fn deadend_time(&self) -> Option<Duration> {
        self.inner.borrow().deadend_time
    }

    /// Returns the bound from the performance model for this candidate
    pub fn bound(&self) -> Ref<Option<Bound>> {
        Ref::map(self.inner.borrow(), |inner| &inner.bound)
    }

    /// Returns the score from the evaluation of this candidate
    pub fn score(&self) -> Option<f64> {
        self.inner.borrow().score
    }

    /// Returns the parent node or None if this is the root node
    pub fn parent(&self) -> Option<CandidateNode> {
        self.inner
            .borrow()
            .incoming_edge
            .as_ref()
            .map(|edge| CandidateNode {
                inner: edge.parent.upgrade().unwrap(),
            })
    }

    /// Returns true if this node has been explored (i.e., it is a
    /// rollout node or an internal node)
    pub fn is_explored(&self) -> bool {
        self.is_internal_node() || self.is_rollout_node()
    }

    /// Returns the child at index `child_idx`. This may be None if
    /// the child hasn't been set before.
    ///
    /// # Panics
    /// Panics if the index is invalid.
    pub fn child(&self, child_idx: usize) -> Option<CandidateNode> {
        let num_outgoing_edges = self.inner.borrow().outgoing_edges.len();

        self.inner
            .borrow()
            .outgoing_edges
            .get(child_idx)
            .unwrap_or_else(|| {
                panic!(
                    "Attempting to retrieve child with index {}, but node has only {} children.",
                    child_idx,
                    num_outgoing_edges)
            })
            .child
            .as_ref()
            .map(|child_weak| CandidateNode {
                inner: child_weak.upgrade().unwrap(),
            })
    }

    /// Returns an interator that allows for iteration over all
    /// children of this node as Option<CandidateNode>, including
    /// unexplored children represented by None values.
    pub fn children(&self) -> impl Iterator<Item = Option<CandidateNode>> {
        let node = CandidateNode {
            inner: Rc::clone(&self.inner),
        };

        (0..self.num_children()).map(move |i| node.child(i))
    }

    /// Returns the number of children, including unexplored children
    pub fn num_children(&self) -> usize {
        self.inner.borrow().outgoing_edges.len()
    }

    /// Returns the ID of this node
    pub fn id(&self) -> NodeId {
        self.inner.borrow().id
    }

    /// Indicates whether this is the virtual root node
    pub fn is_root(&self) -> bool {
        u64::from(self.id()) == 0
    }

    /// Returns the action associated to the edge from the parent of
    /// this node to the node. This might be None if the node does not
    /// have a parent (i.e., if this is the root).
    pub fn action(&self) -> Option<Action> {
        self.inner
            .borrow()
            .incoming_edge
            .as_ref()
            .map(|incoming_edge| {
                incoming_edge
                    .parent
                    .upgrade()
                    .unwrap()
                    .borrow()
                    .outgoing_edges[usize::from(incoming_edge.child_idx)]
                .action
                .clone()
            })
    }

    /// Returns the action associated to the edge from the parent of
    /// this node to the node as a string. If no action is associated
    /// with the edge, an empty string is returned.
    pub fn action_str(&self) -> String {
        self.action()
            .map_or(Default::default(), |action| format!("{:?}", action))
    }

    /// Returns true if this node was declared as a deadend by at
    /// least one thread
    pub fn is_deadend(&self) -> bool {
        self.deadend_time().is_some()
    }

    /// Returns true if this node was declared as a rollout node by at
    /// least one thread
    pub fn is_rollout_node(&self) -> bool {
        self.rollout_time().is_some()
    }

    /// Returns true if this node was declared as an internal node by
    /// at least one thread
    pub fn is_internal_node(&self) -> bool {
        self.internal_time().is_some()
    }

    /// Returns true if this node was declared as an implementation by
    /// at least one thread
    pub fn is_implementation(&self) -> bool {
        self.implementation_time().is_some()
    }

    /// Marks this node as a deadend. The `time` passed as a parameter
    /// becomes the deadend time if no other thread has declared the
    /// node as a deadend with a lower timestamp.
    pub fn declare_deadend(&mut self, timestamp: Duration) {
        self.inner
            .borrow_mut()
            .deadend_time
            .replace_if_lower(timestamp);
    }

    /// Marks this node as an internal node. The `time` passed as a
    /// parameter becomes the internal time if no other thread has
    /// declared the node as internal with a lower timestamp.
    pub fn declare_internal(&mut self, timestamp: Duration) {
        self.inner
            .borrow_mut()
            .internal_time
            .replace_if_lower(timestamp);
    }

    /// Marks this node as a rollout node. The `time` passed as a
    /// parameter becomes the rollout time if no other thread has
    /// declared the node as a rollout node with a lower timestamp.
    pub fn declare_rollout(&mut self, timestamp: Duration) {
        self.inner
            .borrow_mut()
            .rollout_time
            .replace_if_lower(timestamp);
    }

    /// Marks this node as an implementation. The `time` passed as a
    /// parameter becomes the implementation time if no other thread has
    /// declared the node as an implementation with a lower timestamp.
    pub fn declare_implementation(&mut self, timestamp: Duration) {
        self.inner
            .borrow_mut()
            .implementation_time
            .replace_if_lower(timestamp);
    }

    /// Sets the score from an evaluation
    ///
    /// # Panics
    /// Panics if the score has been set beforehand
    pub fn set_score(&mut self, score: f64) {
        assert!(
            self.score().is_none(),
            "Score already set for candidate with ID {}",
            self.id()
        );
        self.inner.borrow_mut().score.replace(score);
    }
}

/// A reconstructed tree
#[derive(Default)]
pub struct CandidateTree {
    /// Virtual root node of the reconstructed tree
    root: Option<Weak<RefCell<CandidateNodeInner>>>,

    /// Mapping node ID -> Candidate nodes
    nodes: FxHashMap<NodeId, Rc<RefCell<CandidateNodeInner>>>,
}

impl CandidateTree {
    /// Creates a new, empty tree
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new inner node
    fn new_node(
        &mut self,
        node_id: NodeId,
        discovery_time: Duration,
        parent: Option<(NodeId, EdgeIndex)>,
        bound: Option<Bound>,
        child_actions: &mut Vec<Action>,
    ) -> CandidateNodeInner {
        CandidateNodeInner {
            incoming_edge: parent.map(|(parent_id, child_idx)| ParentEdge {
                parent: Rc::downgrade(&self.nodes[&parent_id]),
                child_idx,
            }),
            discovery_time,
            internal_time: None,
            rollout_time: None,
            implementation_time: None,
            deadend_time: None,
            bound,
            outgoing_edges: child_actions
                .drain(..)
                .map(|action| ChildEdge {
                    action,
                    child: None,
                })
                .collect(),
            id: node_id,
            score: None,
        }
    }

    /// Returns the root node
    ///
    /// # Panics
    /// Panics if no root has been created beforehand
    pub fn get_root(&self) -> CandidateNode {
        CandidateNode {
            inner: self.root.as_ref().unwrap().upgrade().unwrap(),
        }
    }

    /// Returns the node with the given `id`
    ///
    /// # Panics
    /// Panics if no such node exists in the tree.
    pub fn get_node(&self, id: NodeId) -> CandidateNode {
        CandidateNode {
            inner: Rc::clone(self.nodes.get(&id).unwrap_or_else(|| {
                panic!("Attempting to retrieve unknown node with id {}", id)
            })),
        }
    }

    /// Checks whether a node with the given `id` exists in the tree.
    pub fn has_node(&self, id: NodeId) -> bool {
        self.nodes.contains_key(&id)
    }

    /// Sets the root node to `root`
    ///
    /// # Panics
    /// Panics if the root has been set beforehand or if the root's ID is not 0.
    fn set_root(&mut self, new_root: Weak<RefCell<CandidateNodeInner>>) {
        let new_id = new_root.upgrade().unwrap().borrow().id;

        assert!(self.root.is_none(),
                "Attempting to add second root node with id {}, but already set to node with id {}",
                new_id,
                self.root.as_ref().unwrap().upgrade().unwrap().borrow().id);

        assert!(
            u64::from(new_id) == 0,
            "Attempting to add root node with an ID != 0"
        );

        self.root = Some(new_root);
    }

    /// Adds a new mapping from the given `id` to the node `n`
    ///
    /// # Panics
    /// Panics if a mapping for the given ID already exists.
    fn add_node_mapping(&mut self, id: NodeId, n: Rc<RefCell<CandidateNodeInner>>) {
        assert!(
            !self.has_node(id),
            "Attempting to add duplicate node with id {}",
            id
        );

        self.nodes.insert(id, n);
    }

    /// Creates a new candidate node with the ID `node_id` and inserts
    /// it into the tree. If the `parent` is not None, a parent-child
    /// relationship between the parent and the new node is set
    /// up. The parameter `bound` is the optional bound for this
    /// candidate from the performance model. `Actions` is an array of
    /// actions associated to the outgoing edges of this candidate to
    /// its children and may be empty if this node does not have any
    /// children in the final recreated tree.
    ///
    /// Automatically sets the root of the tree to the newly created
    /// node if `parent` is None.
    ///
    /// # Panics
    /// Panics If `parent` is not None and the ID provided for the
    /// parent node is unknown.
    pub fn extend(
        &mut self,
        node_id: NodeId,
        discovery_time: Duration,
        parent: Option<(NodeId, EdgeIndex)>,
        bound: Option<Bound>,
        actions: &mut Vec<Action>,
    ) {
        let new_node = Rc::new(RefCell::new(self.new_node(
            node_id,
            discovery_time,
            parent,
            bound,
            actions,
        )));

        if let Some((parent_id, child_idx)) = parent {
            let parent_node = self.get_node(parent_id);
            let uidx = usize::from(child_idx);

            {
                let parent_out_edge =
                    &mut parent_node.inner.borrow_mut().outgoing_edges[uidx];

                assert!(
                    parent_out_edge.child.is_none(),
                    "Attempting to re-assign child at index {} of node {}",
                    child_idx,
                    parent_node.id()
                );
                parent_out_edge.child = Some(Rc::downgrade(&new_node));
            }
        } else {
            self.set_root(Rc::downgrade(&new_node));
        }

        self.add_node_mapping(node_id, new_node);
    }
}
