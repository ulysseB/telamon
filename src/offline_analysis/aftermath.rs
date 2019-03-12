///! Functions converting recreated tree and events into Aftermath
///! trace
use crate::explorer::mcts::{CauseOfDeath, EdgeIndex, NodeId, Selector};
use crate::offline_analysis::tree::{CandidateNode, CandidateTree};
use byteorder::{LittleEndian, WriteBytesExt};
use std::fs::File;
use std::io;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Duration;

/// Constants from Aftermath
const AM_TRACE_VERSION: u32 = 18;
const AM_TELAMON_CANDIDATE_FLAG_INTERNAL_NODE: u32 = (1 << 0);
const AM_TELAMON_CANDIDATE_FLAG_ROLLOUT_NODE: u32 = (1 << 1);
const AM_TELAMON_CANDIDATE_FLAG_IMPLEMENTATION: u32 = (1 << 2);
const AM_TELAMON_CANDIDATE_FLAG_DEADEND: u32 = (1 << 3);
const AM_TELAMON_CANDIDATE_FLAG_PERFMODEL_BOUND_VALID: u32 = (1 << 4);
const AM_TELAMON_CANDIDATE_FLAG_SCORE_VALID: u32 = (1 << 5);

const AM_TELAMON_CANDIDATE_EVALUATION_FLAG_SCORE_VALID: u8 = (1 << 0);

const HIERARCHY_DEFAULT_ID: u32 = 0;
const HIERARCHY_ROOT_NODE_ID: u32 = 1;

enum CandidateKillActionCause {
    Constraints = 0,
    PerfModel = 1,
    Backtrack = 2,
}

impl From<CandidateKillActionCause> for u8 {
    fn from(c: CandidateKillActionCause) -> Self {
        c as u8
    }
}

enum ChildSelectorType {
    Random = 0,
    Maximum = 1,
    Exact = 2,
}

impl From<ChildSelectorType> for u8 {
    fn from(t: ChildSelectorType) -> Self {
        t as u8
    }
}

/// IDs for frame type ID <-> frame type associations in the trace
enum FrameType {
    EventCollectionFrame = 1,
    EventMappingFrame = 2,
    HierarchyDescriptionFrame = 3,
    HierarchyNodeFrame = 4,
    TelamonCandidateFrame = 5,
    TelamonThreadTrace = 6,
    TelamonCandidateKillAction = 7,
    TelamonCandidateSelectAction = 8,
    TelamonCandidateSelectChildAction = 9,
    TelamonCandidateExpandAction = 10,
    TelamonCandidateMarkImplementationAction = 11,
    TelamonCandidateEvaluateAction = 12,
}

impl From<FrameType> for u32 {
    fn from(v: FrameType) -> Self {
        v as u32
    }
}

/// Converts a Duration to a u64 in nanoseconds
fn duration_ns(d: Duration) -> u64 {
    d.as_secs() * 1000000000u64 + d.subsec_nanos() as u64
}

/// Converts a Duration to a u64 in nanoseconds, or to 0 if the Option
/// is None
fn duration_ns_default(d: Option<Duration>) -> u64 {
    d.map_or(0, |d| duration_ns(d))
}

/// Writer producing trace files in Aftermath format
pub struct TraceWriter {
    bw: BufWriter<File>,
}

impl TraceWriter {
    /// Creates a new trace writer for the file whose path is passed
    /// as the argument. If the file already exists, it will be
    /// truncated.
    pub fn new(path: &Path) -> Result<Self, io::Error> {
        let fp = File::create(path)?;
        Ok(TraceWriter {
            bw: BufWriter::new(fp),
        })
    }

    /// Writes a u8 to the output file
    fn write_u8(&mut self, val: u8) -> Result<(), io::Error> {
        self.bw.write(&[val])?;

        Ok(())
    }

    /// Writes a u16 in its Aftermath on-disk representation to the
    /// output file
    fn write_u16(&mut self, val: u16) -> Result<(), io::Error> {
        self.bw.write_u16::<LittleEndian>(val)
    }

    /// Writes a u32 in its Aftermath on-disk representation to the
    /// output file
    fn write_u32(&mut self, val: u32) -> Result<(), io::Error> {
        self.bw.write_u32::<LittleEndian>(val)
    }

    /// Writes a u64 in its Aftermath on-disk representation to the
    /// output file
    fn write_u64(&mut self, val: u64) -> Result<(), io::Error> {
        self.bw.write_u64::<LittleEndian>(val)
    }

    /// Writes an f64 in its Aftermath on-disk representation to the
    /// output file
    fn write_f64(&mut self, val: f64) -> Result<(), io::Error> {
        self.bw.write_f64::<LittleEndian>(val)
    }

    /// Writes a NodeId in its Aftermath on-disk representation to the
    /// output file
    fn write_node_id(&mut self, val: NodeId) -> Result<(), io::Error> {
        self.write_u64(u64::from(val))
    }

    /// Writes a string in its Aftermath on-disk representation to the
    /// output file
    fn write_string(&mut self, str: &str) -> Result<(), io::Error> {
        self.write_u32(str.len() as u32)?;
        self.bw.write_all(str.as_bytes())
    }

    /// Writes the header for an Aftermath trace to the output file
    pub fn write_default_header(&mut self) -> Result<(), io::Error> {
        self.bw.write(b"OSTV")?;
        self.write_u32(AM_TRACE_VERSION)
    }

    /// Writes a single structure associating a frame type with a
    /// numerical ID to the output file
    fn write_default_frame_id(
        &mut self,
        frame_str: &str,
        frame_type: FrameType,
    ) -> Result<(), io::Error> {
        // Frame ID 0 = frame type ID association
        self.write_u32(0)?;

        // ID that will be used for the type
        self.write_u32(frame_type as u32)?;

        // Name of the type
        self.write_string(frame_str)
    }

    /// Writes structures to the output file associating the Telamon
    /// frame types with their default numerical IDs
    pub fn write_default_frame_ids(&mut self) -> Result<(), io::Error> {
        self.write_default_frame_id(
            "am_dsk_event_collection",
            FrameType::EventCollectionFrame,
        )?;
        self.write_default_frame_id(
            "am_dsk_event_mapping",
            FrameType::EventMappingFrame,
        )?;
        self.write_default_frame_id(
            "am_dsk_hierarchy_description",
            FrameType::HierarchyDescriptionFrame,
        )?;
        self.write_default_frame_id(
            "am_dsk_hierarchy_node",
            FrameType::HierarchyNodeFrame,
        )?;
        self.write_default_frame_id(
            "am_dsk_telamon_candidate",
            FrameType::TelamonCandidateFrame,
        )?;
        self.write_default_frame_id(
            "am_dsk_telamon_thread_trace",
            FrameType::TelamonThreadTrace,
        )?;
        self.write_default_frame_id(
            "am_dsk_telamon_candidate_kill_action",
            FrameType::TelamonCandidateKillAction,
        )?;
        self.write_default_frame_id(
            "am_dsk_telamon_candidate_select_action",
            FrameType::TelamonCandidateSelectAction,
        )?;
        self.write_default_frame_id(
            "am_dsk_telamon_candidate_select_child_action",
            FrameType::TelamonCandidateSelectChildAction,
        )?;
        self.write_default_frame_id(
            "am_dsk_telamon_candidate_expand_action",
            FrameType::TelamonCandidateExpandAction,
        )?;
        self.write_default_frame_id(
            "am_dsk_telamon_candidate_mark_implementation_action",
            FrameType::TelamonCandidateMarkImplementationAction,
        )?;
        self.write_default_frame_id(
            "am_dsk_telamon_candidate_evaluate_action",
            FrameType::TelamonCandidateEvaluateAction,
        )
    }

    /// Writes a single candidate to the output file
    pub fn write_candidate(
        &mut self,
        candidate: &CandidateNode,
        parent_id: u64,
    ) -> Result<(), io::Error> {
        let mut flags = 0;

        if candidate.is_internal_node() {
            flags |= AM_TELAMON_CANDIDATE_FLAG_INTERNAL_NODE;
        }

        if candidate.is_rollout_node() {
            flags |= AM_TELAMON_CANDIDATE_FLAG_ROLLOUT_NODE;
        }

        if candidate.is_implementation() {
            flags |= AM_TELAMON_CANDIDATE_FLAG_IMPLEMENTATION;
        }

        if candidate.is_deadend() {
            flags |= AM_TELAMON_CANDIDATE_FLAG_DEADEND;
        }

        if candidate.bound().is_some() {
            flags |= AM_TELAMON_CANDIDATE_FLAG_PERFMODEL_BOUND_VALID;
        }

        if candidate.score().is_some() {
            flags |= AM_TELAMON_CANDIDATE_FLAG_SCORE_VALID;
        }

        // Quick sanity check: unexplored nodes cannot have children
        if let Some(parent) = candidate.parent() {
            if !parent.is_explored() && parent.is_root() {
                panic!(
                    "Attempting to write node (ID {}) with unexplored parent",
                    candidate.id()
                );
            }
        }

        let bound = candidate
            .bound()
            .as_ref()
            .map_or(0.0, |bound| bound.value());

        self.write_u32(FrameType::TelamonCandidateFrame.into())?;
        self.write_u64(u64::from(candidate.id()))?;
        self.write_u64(parent_id)?;

        self.write_u64(duration_ns(candidate.discovery_time()))?;
        self.write_u64(duration_ns_default(candidate.internal_time()))?;
        self.write_u64(duration_ns_default(candidate.rollout_time()))?;
        self.write_u64(duration_ns_default(candidate.implementation_time()))?;
        self.write_u64(duration_ns_default(candidate.deadend_time()))?;

        self.write_u32(flags)?;
        self.write_f64(bound)?;
        self.write_f64(candidate.score().unwrap_or(0.0))?;
        self.write_string(&candidate.action_str())
    }

    /// Writes a candidate and all of its descendants in the candidate
    /// tree to the output file
    fn write_candidates_rec(
        &mut self,
        root: &CandidateNode,
        parent_id: u64,
    ) -> Result<(), io::Error> {
        self.write_candidate(root, parent_id)?;

        for child in root.children() {
            if let Some(child_node) = child {
                self.write_candidates_rec(&child_node, u64::from(root.id()))?;
            }
        }

        Ok(())
    }

    /// Writes all candidates of a candidate tree to the output file
    pub fn write_candidates(&mut self, t: &CandidateTree) -> Result<(), io::Error> {
        self.write_candidates_rec(&t.get_root(), 0)
    }

    /// Writes a data structure to the output file that declares a
    /// single Aftermath event collection for the thread whose ID is
    /// passed as the argument.
    pub fn write_event_collection(&mut self, id: u32) -> Result<(), io::Error> {
        self.write_u32(FrameType::EventCollectionFrame.into())?;
        self.write_u32(id)?;
        self.write_string(&format!("Thread {}", id))
    }

    /// Writes a single trace action to the output file
    pub fn write_trace_action(
        &mut self,
        thread_id: u32,
        start: Duration,
        end: Duration,
    ) -> Result<(), io::Error> {
        // Frame Type
        self.write_u32(FrameType::TelamonThreadTrace.into())?;
        self.write_u32(thread_id)?;
        self.write_u64(duration_ns(start))?;
        self.write_u64(duration_ns(end))
    }

    /// Writes a single candidate kill action to the output file
    pub fn write_candidate_kill_action(
        &mut self,
        thread_id: u32,
        candidate_id: NodeId,
        start: Duration,
        end: Duration,
        cause: &CauseOfDeath,
    ) -> Result<(), io::Error> {
        let cause_numeric: u8;

        match cause {
            CauseOfDeath::Constraints => {
                cause_numeric = CandidateKillActionCause::Constraints.into()
            }
            CauseOfDeath::PerfModel { .. } => {
                cause_numeric = CandidateKillActionCause::PerfModel.into()
            }
            CauseOfDeath::Backtrack => {
                cause_numeric = CandidateKillActionCause::Backtrack.into()
            }
        }

        // Frame Type
        self.write_u32(FrameType::TelamonCandidateKillAction.into())?;
        self.write_u32(thread_id)?;
        self.write_node_id(candidate_id)?;
        self.write_u64(duration_ns(start))?;
        self.write_u64(duration_ns(end))?;
        self.write_u8(cause_numeric)
    }

    /// Writes a single candidate evaluation action to the output file
    pub fn write_candidate_evaluate_action(
        &mut self,
        candidate_id: NodeId,
        timestamp: Duration,
        score: Option<f64>,
    ) -> Result<(), io::Error> {
        let mut flags = 0u8;

        // Frame Type
        self.write_u32(FrameType::TelamonCandidateEvaluateAction.into())?;

        // Candidate ID
        self.write_node_id(candidate_id)?;

        // Timestamp of the beginning of the backpropagation
        self.write_u64(duration_ns(timestamp))?;

        // Score
        self.write_f64(score.unwrap_or(0.0))?;

        if score.is_some() {
            flags |= AM_TELAMON_CANDIDATE_EVALUATION_FLAG_SCORE_VALID;
        }

        // Flags
        self.write_u8(flags)
    }

    /// Writes a single candidate select action to the output file
    pub fn write_candidate_select_action(
        &mut self,
        thread_id: u32,
        candidate_id: NodeId,
        start: Duration,
        end: Duration,
    ) -> Result<(), io::Error> {
        // Frame Type
        self.write_u32(FrameType::TelamonCandidateSelectAction.into())?;
        self.write_u32(thread_id)?;
        self.write_node_id(candidate_id)?;
        self.write_u64(duration_ns(start))?;
        self.write_u64(duration_ns(end))
    }

    /// Writes a single child selector entry to the output file
    fn write_child_selector_entry(
        &mut self,
        edge_index: u16,
        val: f64,
    ) -> Result<(), io::Error> {
        self.write_u16(edge_index)?;
        self.write_f64(val)
    }

    /// Writes a single child selector to the output file
    fn write_child_selector(
        &mut self,
        selector: &Selector<EdgeIndex>,
    ) -> Result<(), io::Error> {
        // Child selector
        match selector {
            Selector::Random { weights } => {
                self.write_u8(ChildSelectorType::Random.into())?;
                self.write_u16(weights.len() as u16)?;

                for (edge_index, weight) in weights {
                    self.write_child_selector_entry(u16::from(*edge_index), *weight)?;
                }
            }

            Selector::Maximum { scores } => {
                self.write_u8(ChildSelectorType::Maximum.into())?;
                self.write_u16(scores.len() as u16)?;

                for (edge_index, score) in scores {
                    self.write_child_selector_entry(u16::from(*edge_index), *score)?;
                }
            }

            Selector::Exact { .. } => {
                self.write_u8(ChildSelectorType::Exact.into())?;
                self.write_u16(0)?;
            }
        }

        Ok(())
    }

    /// Writes a single candidate select child action to the output file
    pub fn write_candidate_select_child_action(
        &mut self,
        thread_id: u32,
        parent_id: NodeId,
        selector: &Selector<EdgeIndex>,
        child_idx: u16,
        child_id: NodeId,
        start: Duration,
        end: Duration,
    ) -> Result<(), io::Error> {
        // Frame Type
        self.write_u32(FrameType::TelamonCandidateSelectChildAction.into())?;
        self.write_u32(thread_id)?;
        self.write_node_id(parent_id)?;
        self.write_node_id(child_id)?;
        self.write_u16(child_idx)?;
        self.write_u64(duration_ns(start))?;
        self.write_u64(duration_ns(end))?;
        self.write_child_selector(&selector)?;

        Ok(())
    }

    /// Writes a single candidate expand action to the output file
    pub fn write_candidate_expand_action(
        &mut self,
        thread_id: u32,
        candidate_id: NodeId,
        start: Duration,
        end: Duration,
    ) -> Result<(), io::Error> {
        // Frame Type
        self.write_u32(FrameType::TelamonCandidateExpandAction.into())?;
        self.write_u32(thread_id)?;
        self.write_node_id(candidate_id)?;
        self.write_u64(duration_ns(start))?;
        self.write_u64(duration_ns(end))
    }

    /// Writes a single action marking a candidate as an
    /// implementation to the output file
    pub fn write_candidate_mark_implementation_action(
        &mut self,
        thread_id: u32,
        candidate_id: NodeId,
        start: Duration,
        end: Duration,
    ) -> Result<(), io::Error> {
        // Frame Type
        self.write_u32(FrameType::TelamonCandidateMarkImplementationAction.into())?;
        self.write_u32(thread_id)?;
        self.write_node_id(candidate_id)?;
        self.write_u64(duration_ns(start))?;
        self.write_u64(duration_ns(end))
    }

    /// Writes a single Aftermath hierarchy description to the output
    /// file
    fn write_hierarchy_description(
        &mut self,
        id: u32,
        name: &str,
    ) -> Result<(), io::Error> {
        // Frame Type
        self.write_u32(FrameType::HierarchyDescriptionFrame.into())?;

        // Hierarchy ID
        self.write_u32(id)?;

        // Hierarchy name
        self.write_string(name)
    }

    /// Writes the data structure of a single Aftermath hierarchy node
    /// to the output file
    pub fn write_hierarchy_node(
        &mut self,
        hierarchy_id: u32,
        name: &str,
        id: u32,
        parent_id: u32,
    ) -> Result<(), io::Error> {
        // Frame Type
        self.write_u32(FrameType::HierarchyNodeFrame.into())?;

        // Hierarchy ID
        self.write_u32(hierarchy_id)?;

        // Hierarchy node ID
        self.write_u32(id)?;

        // ID of the parent hierarchy node
        self.write_u32(parent_id)?;

        // Node name
        self.write_string(name)
    }

    /// Returns the ID of the Aftermath hierarchy node associated to
    /// the thread whose ID is provided in `thread_id`.
    fn thread_id_to_hierarchy_node_id(thread_id: u32) -> u32 {
        thread_id + 2
    }

    /// Writes an entire Aftermath hierarchy to the output file. The
    /// hierarchy will be composed of a virtual root node called
    /// "Root" with one descendant per thread specified in
    /// `thread_ids`. The IDs of the hierarchy nodes correspond to the
    /// thread IDs incremented by 2.
    pub fn write_hierarchy(&mut self, thread_ids: &Vec<u32>) -> Result<(), io::Error> {
        // There is only one hierarchy, let this be 0
        self.write_hierarchy_description(HIERARCHY_DEFAULT_ID, "Threads")?;

        // Root node with special parent id 0 (= no parent)
        self.write_hierarchy_node(
            HIERARCHY_DEFAULT_ID,
            "Root",
            HIERARCHY_ROOT_NODE_ID,
            0,
        )?;

        for thread_id in thread_ids {
            let hnode_id = Self::thread_id_to_hierarchy_node_id(*thread_id);
            self.write_hierarchy_node(
                HIERARCHY_DEFAULT_ID,
                &format!("Thread {}", thread_id),
                hnode_id,
                HIERARCHY_ROOT_NODE_ID,
            )?;
        }

        Ok(())
    }

    /// Writes a single Aftermath event mapping (associating an event
    /// collection with a hierarchy node) to the output file for the
    /// thread with the specified ID. The thread's event collection is
    /// associated with the thread's hierarchy node for the entire
    /// duration of the trace.
    pub fn write_event_mapping(&mut self, thread_id: u32) -> Result<(), io::Error> {
        // Frame Type
        self.write_u32(FrameType::EventMappingFrame.into())?;

        // Collection ID
        self.write_u32(thread_id)?;

        // Hierarchy ID
        self.write_u32(0)?;

        // Node ID
        self.write_u32(Self::thread_id_to_hierarchy_node_id(thread_id))?;

        // Associate forever, i.e., the interval [0; UINT64_MAX]
        self.write_u64(0)?;
        self.write_u64(std::u64::MAX)
    }

    /// Writes the Aftermath event mappings for all threads in
    /// thread_ids to the output file.
    pub fn write_event_mappings(
        &mut self,
        thread_ids: &Vec<u32>,
    ) -> Result<(), io::Error> {
        for tid in thread_ids {
            self.write_event_mapping(*tid)?;
        }

        Ok(())
    }
}
