use std::io;
use std::path::PathBuf;

use serde_json;
use structopt::StructOpt;

use telamon::explorer::{choice::ActionEx as Action, eventlog::EventLog, mcts};
use telamon::offline_analysis::tree::CandidateTree;

#[derive(StructOpt)]
struct Rebuild {
    /// Path to the eventlog to rebuild from
    #[structopt(
        parse(from_os_str),
        short = "i",
        long = "input",
        default_value = "eventlog.tfrecord.gz"
    )]
    eventlog: PathBuf,

    /// Identifier of the candidate node to rebuild.  This corresponds to the ID indicated in
    /// `watch.log`.
    id: usize,
}

impl Rebuild {
    fn find_candidate(&self) -> io::Result<(mcts::NodeId, f64, Vec<Action>)> {
        let mut nevals = 0;
        let mut tree = CandidateTree::new();

        for record_bytes in EventLog::open(&self.eventlog)?.records() {
            match bincode::deserialize(&record_bytes?)
                .map_err(|err| io::Error::new(io::ErrorKind::Other, err))?
            {
                mcts::Message::Node {
                    id,
                    parent,
                    mut children,
                    bound,
                    discovery_time,
                } => tree.extend(id, discovery_time, parent, bound, &mut children),
                mcts::Message::Trace { .. } => (),
                mcts::Message::Evaluation { id, value, .. } => {
                    if let Some(value) = value {
                        if nevals == self.id {
                            return Ok((id, value, tree.get_node(id).actions()));
                        }

                        nevals += 1;
                    }
                }
            }
        }

        Err(io::Error::new(
            io::ErrorKind::Other,
            "Unable to find candidate",
        ))
    }

    fn run(&self, _args: &Opt) -> io::Result<()> {
        let (id, score, actions) = self.find_candidate()?;

        println!("Found candidate {} (score: {})", id, score);
        println!("{}", serde_json::to_string_pretty(&actions)?);

        Ok(())
    }
}

#[derive(StructOpt)]
enum Command {
    #[structopt(name = "rebuild")]
    Rebuild(Rebuild),
}

#[derive(StructOpt)]
#[structopt(name = "telamon")]
struct Opt {
    #[structopt(subcommand)]
    command: Command,
}

fn main() -> io::Result<()> {
    let args = Opt::from_args();
    env_logger::init();

    match &args.command {
        Command::Rebuild(rebuild) => rebuild.run(&args),
    }
}
