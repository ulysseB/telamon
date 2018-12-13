extern crate bincode;
extern crate dot;
extern crate telamon;
extern crate telamon_utils as utils;

use std::collections::HashMap;

use std::io;
use std::io::Seek;
use telamon::explorer::{choice::ActionEx, TreeEvent};
use utils::tfrecord::{ReadError, RecordReader};

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

fn main() -> Result<(), ReadError> {
    let mut f = std::fs::File::open("eventlog.tfrecord")?;
    let mut root = Node {
        children: Default::default(),
        evaluations: vec![],
        id: 0,
        tag: None,
    };

    let mut offset;
    let mut id = 0;
    loop {
        id += 1;

        offset = f.seek(io::SeekFrom::Current(0))?;

        match f.read_record() {
            Ok(record) => match bincode::deserialize(&record).unwrap() {
                TreeEvent::Evaluation { actions, score } => {
                    root.evaluations.push(score);
                    let actions = actions.to_vec();
                    dig(&mut root.children, actions.iter().rev().cloned(), score, id);

                    // println!("{:?}", actions.to_vec());
                }
            },
            Err(err) => {
                // If we reached eof and no bytes were read, we were
                // at the end of a well-formed file and we can safely
                // exit. Otherwise, we propagate the error.
                if let ReadError::IOError(ref error) = err {
                    if error.kind() == io::ErrorKind::UnexpectedEof
                        && offset == f.seek(io::SeekFrom::Current(0))?
                    {
                        break;
                    }
                }
                return Err(err);
            }
        }
    }

    root.compute_top(10);

    {
        let mut f = std::fs::File::create("graph.dot").unwrap();
        dot::render(&root.info(), &mut f).unwrap();
    }

    Ok(())
}
