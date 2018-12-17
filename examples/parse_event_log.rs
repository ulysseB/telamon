extern crate bincode;
extern crate csv;
extern crate dot;
extern crate flate2;
extern crate structopt;

extern crate telamon;
extern crate telamon_utils as utils;

use std::collections::HashMap;

use std::ffi::OsStr;
use std::fs::File;
use std::io::{self, Read};
use std::path::PathBuf;
use telamon::explorer::{choice::ActionEx, TreeEvent};
use utils::tfrecord::{ReadError, RecordReader};

use flate2::read::{GzDecoder, ZlibDecoder};
use structopt::StructOpt;

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

type Nd<'a> = (usize, &'a &'a Node);

type Ed<'a> = &'a (usize, usize, &'a Edge);

impl<'a> dot::GraphWalk<'a, Nd<'a>, Ed<'a>> for TreeInfo<'a> {
    fn nodes(&'a self) -> dot::Nodes<'a, Nd<'a>> {
        self.nodes.iter().enumerate().collect()
    }

    fn edges(&'a self) -> dot::Edges<'a, Ed<'a>> {
        self.edges.iter().collect()
    }

    fn source(&'a self, edge: &Ed<'a>) -> Nd<'a> {
        (edge.0, &self.nodes[edge.0])
    }

    fn target(&'a self, edge: &Ed<'a>) -> Nd<'a> {
        (edge.1, &self.nodes[edge.1])
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
    fn info(&self) -> TreeInfo<'_> {
        let mut worklist = vec![(self, 0)];
        let mut nodes = vec![];
        let mut edges = vec![];

        let max_depth = None;
        let min_evals = Some(1000);

        while let Some((node, depth)) = worklist.pop() {
            nodes.push((node as *const Node, node));

            if max_depth.map(|max_depth| depth < max_depth).unwrap_or(true)
                && min_evals
                    .map(|min_evals| node.evaluations.len() > min_evals)
                    .unwrap_or(true)
            {
                for (action, edge) in node.children.iter() {
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

fn dig<II>(children: &mut HashMap<ActionEx, Edge>, actions: II, eval: f64, id: usize)
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
                id: id,
                tag: None,
            }),
        });

        edge.node.evaluations.push(eval);
        dig(&mut edge.node.children, it, eval, id)
    }
}

#[derive(Debug, StructOpt)]
#[structopt(name = "print_event_log")]
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
}

impl Opt {
    fn open_eventlog(&self) -> io::Result<Box<dyn Read>> {
        let raw_file = File::open(&self.eventlog)?;
        Ok(match self.eventlog.extension().and_then(OsStr::to_str) {
            Some("gz") => Box::new(GzDecoder::new(raw_file)),
            Some("zz") => Box::new(ZlibDecoder::new(raw_file)),
            _ => Box::new(raw_file),
        })
    }
}

fn main() -> Result<(), ReadError> {
    let opt = Opt::from_args();

    let mut f = opt.open_eventlog()?;
    let mut root = Node {
        children: Default::default(),
        evaluations: vec![],
        id: 0,
        tag: None,
    };

    let mut evals = Vec::new();

    for (id, record_bytes) in f.records().enumerate() {
        let evt: TreeEvent = bincode::deserialize(&record_bytes?).unwrap();
        let score = evt.score().unwrap_or(std::f64::INFINITY);
        root.evaluations.push(score);

        let actions = {
            let mut actions = evt.actions().cloned().collect::<Vec<_>>();
            actions.reverse();
            actions
        };

        dig(&mut root.children, actions, score, id);

        evals.push(score);
    }

    println!("Computing top{} for all nodes...", opt.topk);
    root.compute_top(opt.topk);

    // Print the graph
    println!(
        "Writing graph to {}...",
        format!("graph-top{}.dot", opt.topk)
    );
    {
        let mut f = std::fs::File::create(format!("graph-top{}.dot", opt.topk)).unwrap();
        dot::render(&root.info(), &mut f).unwrap();
    }

    // Print the csv
    println!("Writing out.csv...");
    {
        let mut f = std::fs::File::create("out.csv")?;
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
