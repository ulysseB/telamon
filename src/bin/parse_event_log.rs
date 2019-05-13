/// This allows dumping a CSV of all the evaluations that were performed during a run, as well as a
/// graphviz (.dot) graph recapitulating the run.
///
/// NOTE: This applies to the old log format, which is was less precise and does not contain
/// precise run details, such as actions not taken, or timings.  This file is kept for historical
/// reasons so that we still have a way to parse those old log files.
use std::collections::HashMap;

use std::fs::File;
use std::io;
use std::path::PathBuf;
use telamon::explorer::{choice::ActionEx, eventlog::EventLog};

use rpds::List;
use serde::{Deserialize, Serialize};
use structopt::StructOpt;

#[derive(Serialize, Deserialize)]
pub enum DeadEndSource {
    /// Dead-end encountered in the tree
    Tree {
        /// List of actions defining the dead-end candidate
        actions: List<ActionEx>,
    },
    /// Dead-end encountered in the rollout phase
    Rollout {
        /// List of actions defining the dead-end candidate
        actions: List<ActionEx>,
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
        actions: List<ActionEx>,
        score: f64,
    },

    /// A fully-specified implementation was found and evaluated
    EvaluationV2 {
        /// List of actions defining the implementation
        actions: List<ActionEx>,
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
        discovery_time: f64,
        /// Time at which the evaluation finished.  Note that evaluations are performed by a
        /// specific thread, not the one that found the implementation.
        evaluation_end_time: f64,
        /// ID of the thread that found this implementation
        thread: String,
    },

    /// A dead-end was reached
    DeadEnd {
        /// Source of this deadend
        source: DeadEndSource,
        /// Time at which the deadend was found after the start of the program
        discovery_time: f64,
        /// ID of the thread that found the deadend
        thread: String,
    },
}

struct Edge {
    action: ActionEx,
    node: Box<Node>,
}

struct Node {
    children: HashMap<ActionEx, Edge>,
    evaluations: Vec<f64>,
    id: usize,
    tag: Option<f64>,
}

impl Node {
    fn compute_top(&mut self, k: usize) {
        let mut buf = Vec::with_capacity(k);

        for (_action, edge) in self.children.iter() {
            for eval in &edge.node.evaluations {
                let pos = buf
                    .binary_search_by(|&probe| utils::cmp_f64(probe, *eval))
                    .unwrap_or_else(|e| e);
                if pos < k {
                    if buf.len() >= k {
                        buf.pop();
                    }
                    buf.insert(pos, *eval);
                }
            }
        }

        if let Some(threshold) = buf.pop() {
            for (_action, edge) in self.children.iter_mut() {
                edge.node.tag = Some(
                    edge.node
                        .evaluations
                        .iter()
                        .filter(|eval| **eval <= threshold)
                        .count() as f64
                        / k as f64,
                );

                edge.node.compute_top(k);
            }
        }
    }
}

struct TreeInfo<'a> {
    nodes: Vec<&'a Node>,
    edges: Vec<(usize, usize, &'a Edge)>,
}

type Nd<'a> = (usize, &'a Node);

type Ed<'a> = &'a (usize, usize, &'a Edge);

impl<'a> dot::GraphWalk<'a, Nd<'a>, Ed<'a>> for TreeInfo<'a> {
    fn nodes(&'a self) -> dot::Nodes<'a, Nd<'a>> {
        self.nodes.iter().cloned().enumerate().collect()
    }

    fn edges(&'a self) -> dot::Edges<'a, Ed<'a>> {
        self.edges.iter().collect()
    }

    fn source(&'a self, edge: &Ed<'a>) -> Nd<'a> {
        (edge.0, self.nodes[edge.0])
    }

    fn target(&'a self, edge: &Ed<'a>) -> Nd<'a> {
        (edge.1, self.nodes[edge.1])
    }
}

impl<'a> dot::Labeller<'a, Nd<'a>, Ed<'a>> for TreeInfo<'a> {
    fn graph_id(&'a self) -> dot::Id<'a> {
        dot::Id::new("telamon").unwrap()
    }

    fn node_id(&'a self, n: &Nd<'a>) -> dot::Id<'a> {
        dot::Id::new(format!("N{}", n.0)).unwrap()
    }

    fn node_label(&self, n: &Nd<'a>) -> dot::LabelText<'_> {
        dot::LabelText::label(format!(
            "#{}: {} (best: {:7.03e}, avg: {:7.03e}, avglog: {:7.03e}{})",
            n.1.id,
            n.1.evaluations.len(),
            n.1.evaluations
                .iter()
                .cloned()
                .min_by(|lhs, rhs| utils::cmp_f64(*lhs, *rhs))
                .unwrap_or(std::f64::INFINITY),
            n.1.evaluations.iter().sum::<f64>() / n.1.evaluations.len() as f64,
            n.1.evaluations.iter().map(|x| x.ln()).sum::<f64>()
                / n.1.evaluations.len() as f64,
            if let Some(tag) = n.1.tag {
                format!(", top10: {}%", tag * 100.)
            } else {
                "".into()
            },
        ))
    }

    fn edge_label(&self, e: &Ed<'a>) -> dot::LabelText<'_> {
        dot::LabelText::label(format!("{:?}", e.2.action))
    }
}

impl Node {
    fn info(&self, max_depth: Option<u32>, min_evals: Option<u32>) -> TreeInfo<'_> {
        let mut worklist = vec![(self, 0)];
        let mut nodes = vec![];
        let mut edges = vec![];

        while let Some((node, depth)) = worklist.pop() {
            nodes.push((node as *const Node, node));

            if max_depth.map(|max_depth| depth < max_depth).unwrap_or(true)
                && min_evals
                    .map(|min_evals| node.evaluations.len() as u32 > min_evals)
                    .unwrap_or(true)
            {
                for (_action, edge) in node.children.iter() {
                    edges.push((node as *const Node, &*edge.node as *const Node, edge));

                    worklist.push((&edge.node, depth + 1));
                }
            }
        }

        let mut nodeindex: HashMap<*const Node, usize> = HashMap::new();
        for (index, (nid, _)) in nodes.iter().enumerate() {
            nodeindex.insert(*nid, index);
        }

        TreeInfo {
            nodes: nodes.into_iter().map(|(_nid, info)| info).collect(),
            edges: edges
                .into_iter()
                .map(|(from, to, info)| (nodeindex[&from], nodeindex[&to], info))
                .collect(),
        }
    }
}

/// Recursively walks the tree defined by `children` and records an evaluation value of `eval`
/// along the path.
fn record<II>(children: &mut HashMap<ActionEx, Edge>, actions: II, eval: f64, id: usize)
where
    II: IntoIterator<Item = ActionEx>,
{
    let mut it = actions.into_iter();

    if let Some(action) = it.next() {
        let edge = children.entry(action.clone()).or_insert_with(|| Edge {
            action: action.clone(),
            node: Box::new(Node {
                children: Default::default(),
                evaluations: vec![],
                id,
                tag: None,
            }),
        });

        edge.node.evaluations.push(eval);
        record(&mut edge.node.children, it, eval, id)
    }
}

#[derive(Debug, StructOpt)]
#[structopt(name = "parse_event_log")]
struct Opt {
    #[structopt(
        parse(from_os_str),
        short = "i",
        long = "input",
        default_value = "eventlog.tfrecord.gz"
    )]
    eventlog: PathBuf,

    #[structopt(long = "topk", default_value = "10")]
    topk: usize,

    /// Maximum depth after which nodes should be hidden in the graph output.  Nodes at a depth
    /// larger than max_depth are not displayed.
    #[structopt(long = "max-depth")]
    max_depth: Option<u32>,

    /// Minimum number of evaluations below which children should be hidden in the graph output.
    /// Nodes with less than `min_evals` evaluations are displayed, but not their children.
    #[structopt(long = "min-evals")]
    min_evals: Option<u32>,
}

fn main() -> io::Result<()> {
    let opt = Opt::from_args();

    let mut root = Node {
        children: Default::default(),
        evaluations: vec![],
        id: 0,
        tag: None,
    };

    let mut evals = Vec::new();

    for (id, record_bytes) in EventLog::open(&opt.eventlog)?.records().enumerate() {
        match bincode::deserialize(&record_bytes?).unwrap() {
            TreeEvent::Evaluation { actions, score }
            | TreeEvent::EvaluationV2 { actions, score, .. } => {
                root.evaluations.push(score);

                let actions = {
                    let mut actions = actions.iter().cloned().collect::<Vec<_>>();
                    actions.reverse();
                    actions
                };

                record(&mut root.children, actions, score, id);

                evals.push(score);
            }
            TreeEvent::DeadEnd { .. } => (),
        }
    }

    println!("Computing top{} for all nodes...", opt.topk);
    root.compute_top(opt.topk);

    // Print the graph
    println!(
        "Writing graph to {}...",
        format!("graph-top{}.dot", opt.topk)
    );
    {
        let mut f = File::create(format!("graph-top{}.dot", opt.topk)).unwrap();
        dot::render(&root.info(opt.max_depth, opt.min_evals), &mut f).unwrap();
    }

    // Print the csv
    println!("Writing out.csv...");
    {
        let mut f = File::create("out.csv")?;
        let mut writer = csv::Writer::from_writer(&mut f);
        writer.write_record(&["Id", "Time"]).unwrap();
        for (id, eval) in evals.iter().enumerate() {
            writer
                .write_record(&[id.to_string(), eval.to_string()])
                .unwrap();
        }
        writer.flush()?;
    }

    Ok(())
}
