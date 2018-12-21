use std::io;
use std::io::Seek;
use telamon::explorer::TreeEvent;
use utils::tfrecord::{ReadError, RecordReader};

fn main() -> Result<(), ReadError> {
    let mut f = std::fs::File::open("eventlog.tfrecord")?;

    for record_bytes in (&mut f).records() {
        match bincode::deserialize(&record_bytes?).unwrap() {
            TreeEvent::Evaluation {
                actions,
                score: _score,
            } => println!("{:?}", actions.into_iter().collect::<Vec<_>>()),
        }
    }

    Ok(())
}
