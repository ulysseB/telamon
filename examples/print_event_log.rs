/// This example shows how to iterate over the events stored in an eventlog.
extern crate bincode;
extern crate telamon;
extern crate telamon_utils as utils;

use std::io;
use std::io::Seek;
use telamon::explorer::TreeEvent;
use utils::tfrecord::{ReadError, RecordReader};

fn main() -> Result<(), ReadError> {
    let mut f = std::fs::File::open("eventlog.tfrecord")?;

    for record_bytes in (&mut f).records() {
        let evt: TreeEvent = bincode::deserialize(&record_bytes?).unwrap();
        println!("{:?}", evt.actions.actions().collect::<Vec<_>>());
    }

    Ok(())
}
