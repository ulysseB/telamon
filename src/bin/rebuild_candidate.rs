use std::io;
use std::path::PathBuf;

use serde_json;
use structopt::StructOpt;

use telamon::explorer::{eventlog::EventLog, mcts};
use telamon::offline_analysis::tree::CandidateTree;

#[derive(Debug, StructOpt)]
#[structopt(name = "rebuild_candidate")]
struct Opt {
    #[structopt(
        parse(from_os_str),
        short = "i",
        long = "input",
        default_value = "eventlog.tfrecord.gz"
    )]
    eventlog: PathBuf,

    id: usize,
}

fn main() -> io::Result<()> {
    let opt = Opt::from_args();

    let mut nevals = 0;
    let mut tree = CandidateTree::new();

    for record_bytes in EventLog::open(&opt.eventlog)?.records() {
        match bincode::deserialize(&record_bytes?).unwrap() {
            mcts::Message::Node {
                id,
                parent,
                mut children,
                bound,
                discovery_time,
            } => tree.extend(id, discovery_time, parent, bound, &mut children),
            mcts::Message::Trace { .. } => (),
            mcts::Message::Evaluation { id, .. } => {
                if nevals == opt.id {
                    for action in tree.get_node(id).actions() {
                        println!("{:?}", action);
                    }

                    break;
                }

                nevals += 1;
            }
        }
    }

    Ok(())
}
