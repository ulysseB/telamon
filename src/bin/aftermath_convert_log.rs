///! Aftermath_convert_log converts a Telamon logfile into a trace
///! file that can be loaded into the Aftermath trace analysis tool
///! from https://www.aftermath-tracing.com
use log::debug;
use std::collections::HashSet;
use std::io;
use std::path::PathBuf;
use std::time::Duration;
use structopt::StructOpt;
use telamon::explorer::eventlog::EventLog;
use telamon::explorer::mcts::{Event, Message, Policy};
use telamon::offline_analysis::aftermath::TraceWriter;
use telamon::offline_analysis::tree::CandidateTree;

#[derive(Debug, StructOpt)]
#[structopt(name = "aftermath_convert_log")]
struct Opt {
    #[structopt(
        parse(from_os_str),
        short = "i",
        long = "input",
        default_value = "eventlog.tfrecord.gz"
    )]
    eventlog: PathBuf,

    #[structopt(
        parse(from_os_str),
        short = "o",
        long = "output",
        default_value = "eventlog.ost"
    )]
    output: PathBuf,
}

fn main() -> io::Result<()> {
    env_logger::try_init().unwrap();

    let opt = Opt::from_args();
    let mut tw = TraceWriter::new(&opt.output.as_path()).unwrap();
    let mut thread_ids = HashSet::new();
    let mut t = CandidateTree::new();

    // Write file header and declare all Aftermath frame types used in
    // the trace
    tw.write_default_header()?;
    tw.write_default_frame_ids()?;

    // Process log messages
    for record_bytes in EventLog::open(&opt.eventlog)?.records() {
        match bincode::deserialize(&record_bytes?).unwrap() {
            // Discovery of a new node
            Message::Node {
                id,
                parent,
                bound,
                children,
                discovery_time,
            } => {
                debug!("Node (ID {}) [discovery time: {:?}]", id, discovery_time);

                t.extend(
                    id.into(),
                    discovery_time.clone(),
                    parent,
                    bound.clone(),
                    &mut children.clone(),
                );
            }

            // Series of timed events performed by one thread
            Message::Trace { thread, events } => {
                // ID of the node that was last selected by an action;
                // The root node is implicitly selected at the start
                // of a thread trace
                let mut curr_node_id = t.get_root().id();
                let thread_id = thread
                    .trim_start_matches("ThreadId")
                    .trim_matches(|p| p == '(' || p == ')')
                    .parse::<u32>()
                    .unwrap();
                let mut trace_start: Option<Duration> = None;
                let mut trace_end: Option<Duration> = None;

                // If the thread is yet unknown, start a new Aftermath
                // event collection for its events
                if thread_ids.insert(thread_id) {
                    tw.write_event_collection(thread_id)?;
                }

                debug!("Trace (thread {}):", thread_id);
                debug!("  Implicit SelectNode {}", curr_node_id);

                for timed_event in &events {
                    let start = timed_event.start_time;
                    let end = timed_event.end_time;

                    // Set start timestamp of the thread trace to the
                    // beginning of the first event of the trace and
                    // the end timestamp to the end of the last event
                    if trace_start.is_none() {
                        trace_start = Some(start.clone());
                    }

                    trace_end = Some(end.clone());

                    match timed_event.value {
                        // Explicit selection of a candidate
                        Event::SelectNode(id) => {
                            curr_node_id = id.into();

                            tw.write_candidate_select_action(
                                thread_id,
                                curr_node_id,
                                start,
                                end,
                            )?;

                            debug!(
                                "  SelectNode {} [interval: {:?} to {:?}]",
                                curr_node_id, start, end
                            );
                        }

                        // Selection of a child of the current candidate
                        Event::SelectChild(idx, policy, ref selector) => {
                            let mut child_node =
                                t.get_node(curr_node_id).child(usize::from(idx)).unwrap();

                            match policy {
                                Policy::Bandit => {
                                    child_node.declare_internal(end);
                                }
                                Policy::Default => {
                                    child_node.declare_rollout(end);
                                }
                            }

                            tw.write_candidate_select_child_action(
                                thread_id as u32,
                                curr_node_id,
                                selector,
                                u16::from(idx),
                                child_node.id(),
                                start,
                                end,
                            )?;

                            debug!(
                                "  SelectChild [idx: {} -> ID: {}, policy: {:?}, interval: {:?} to {:?}]",
                                u16::from(idx), child_node.id(), policy, start, end
                            );

                            curr_node_id = child_node.id();
                        }

                        // Expansion of the current candidate
                        Event::Expand => {
                            t.get_node(curr_node_id).declare_internal(end);
                            tw.write_candidate_expand_action(
                                thread_id as u32,
                                curr_node_id,
                                start,
                                end,
                            )?;

                            debug!("  Expand [interval: {:?} to {:?}]", start, end);
                        }

                        // Declaring the current candidate as a
                        // deadend
                        Event::Kill(cause) => {
                            t.get_node(curr_node_id).declare_deadend(end);

                            tw.write_candidate_kill_action(
                                thread_id as u32,
                                curr_node_id,
                                start,
                                end,
                                &cause,
                            )?;

                            debug!("  Kill [interval: {:?} to {:?}]", start, end);
                        }

                        // Declaration of a child of the current
                        // candidate as a deadend
                        Event::KillChild(child_idx, cause) => {
                            let mut child = t
                                .get_node(curr_node_id)
                                .child(usize::from(child_idx))
                                .unwrap();

                            child.declare_deadend(end);

                            tw.write_candidate_kill_action(
                                thread_id as u32,
                                child.id(),
                                start,
                                end,
                                &cause,
                            )?;

                            debug!(
                                "  KillChild [idx: {} -> ID: {}, interval: {:?} to {:?}]",
                                u16::from(child_idx),
                                child.id(),
                                start,
                                end
                            );
                        }

                        // Declaration of the current candidate as an
                        // implementation
                        Event::Implementation {} => {
                            t.get_node(curr_node_id).declare_implementation(end);

                            tw.write_candidate_mark_implementation_action(
                                thread_id as u32,
                                curr_node_id,
                                start,
                                end,
                            )?;

                            debug!(
                                "  Implementation [interval: {:?} to {:?}]",
                                start, end
                            );
                        }
                    }
                }

                // Write trace event only if trace has at least one event
                if let Some(trace_start_ts) = trace_start {
                    tw.write_trace_action(
                        thread_id as u32,
                        trace_start_ts,
                        trace_end.unwrap(),
                    )?;
                }
            }

            // Execution of an implementation
            Message::Evaluation {
                id,
                value,
                result_time,
            } => {
                if value.is_some() {
                    t.get_node(id.into()).set_score(value.unwrap());
                }

                tw.write_candidate_evaluate_action(id.into(), result_time, value)?;
            }
        }
    }

    let mut thread_ids_vec = Vec::<u32>::new();

    for thr in thread_ids.iter() {
        thread_ids_vec.push(*thr);
    }

    thread_ids_vec.sort();

    // Write hierarchy of threads represented by a flat Aftermath
    // hierarchy composed of one virtual root thread with one child
    // per thread
    tw.write_hierarchy(&thread_ids_vec)?;

    // Write event mappings associating event collections with
    // hierarchy nodes
    tw.write_event_mappings(&thread_ids_vec)?;

    // Write the reconstructed candidate tree
    tw.write_candidates(&t)?;

    Ok(())
}
