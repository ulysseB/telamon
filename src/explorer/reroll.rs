use bincode;
use std;
use std::io;
use std::io::Seek;
use utils::tfrecord::{ReadError, RecordReader};

use device::Context;
use explorer::{choice, Candidate, SearchSpace, TreeEvent};
use model::bound;

pub fn candidate_from_actions<'a>(
    actions: Vec<choice::ActionEx>,
    context: &Context,
    space: SearchSpace<'a>,
) -> Candidate<'a> {
    let bound = bound(&space, context);
    actions
        .into_iter()
        .fold(Candidate::new(space, bound), |cand, action| {
            unwrap!(cand.apply_decision(context, action))
        })
    //unimplemented!()
}

struct ActionIter {
    log_file: std::fs::File,
}

impl Iterator for ActionIter {
    type Item = Vec<choice::ActionEx>;

    fn next(&mut self) -> Option<Self::Item> {
        let offset = self
            .log_file
            .seek(io::SeekFrom::Current(0))
            .expect("Seek Error");
        match self.log_file.read_record() {
            Ok(record) => match bincode::deserialize(&record).unwrap() {
                TreeEvent::Evaluation {
                    actions,
                    score: _score,
                } => Some(actions.to_vec()),
            },
            Err(err) => {
                // If we reached eof and no bytes were read, we were
                // at the end of a well-formed file and we can safely
                // exit. Otherwise, we propagate the error.
                if let ReadError::IOError(ref error) = err {
                    if error.kind() == io::ErrorKind::UnexpectedEof
                        && offset == self.log_file.seek(io::SeekFrom::Current(0)).unwrap()
                    {
                        None
                    } else {
                        panic!(format!("Read Error {:?} in event file", err))
                    }
                } else {
                    panic!(format!("Error {:?} in event file", err))
                }
            }
        }
    }
}

pub fn actions_from_log(filename: &str) -> impl Iterator<Item = Vec<choice::ActionEx>> {
    let f = std::fs::File::open(filename)
        .expect(&format!("Could not open log file {}", filename));
    ActionIter { log_file: f }
}
