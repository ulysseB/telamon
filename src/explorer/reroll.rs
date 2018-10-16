use bincode;
use std;
use std::io;
use std::io::Seek;
use utils::tfrecord::{ReadError, RecordReader};

use explorer::{choice, Candidate, SearchSpace, TreeEvent};

fn candidate_from_actions<'a>(space: SearchSpace<'a>, actions: Vec<choice::ActionEx>) -> Candidate<'a> {
    unimplemented!()
}

struct ActionIter{log_file: std::fs::File}

impl Iterator for ActionIter {
    type Item = Result<Vec<choice::ActionEx>, ReadError>;

    fn next(&mut self) -> Option<Self::Item> {
        let offset = self.log_file.seek(io::SeekFrom::Current(0))?;
        match self.log_file.read_record() {
            Ok(record) => match bincode::deserialize(&record).unwrap() {
                TreeEvent::Evaluation {
                    actions,
                    score: _score,
                } => Ok(actions.to_vec()),
            },
            Err(err) => {
                // If we reached eof and no bytes were read, we were
                // at the end of a well-formed file and we can safely
                // exit. Otherwise, we propagate the error.
                if let ReadError::IOError(ref error) = err {
                    if error.kind() == io::ErrorKind::UnexpectedEof
                        && offset == self.log_file.seek(io::SeekFrom::Current(0))?
                        {
                            None
                        }
                }
                Err(err)
            }
        }
    }
}

fn actions_from_log(filename: &str) -> Result<impl Iterator<Item = Vec<choice::ActionEx>>, ReadError> {
    let mut f = std::fs::File::open(filename)?;
    Ok(ActionIter{log_file : f})
}
