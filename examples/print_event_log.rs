use std::io;
use std::path::PathBuf;

use serde_json;
use structopt::StructOpt;

use telamon::explorer::{eventlog::EventLog, mcts};

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
}

fn main() -> io::Result<()> {
    let opt = Opt::from_args();

    for record_bytes in EventLog::open(&opt.eventlog)?.records() {
        let message: mcts::Message = bincode::deserialize(&record_bytes?).unwrap();
        println!("{}", serde_json::to_string(&message).unwrap());
    }

    Ok(())
}
