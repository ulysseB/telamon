use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::sync::mpsc;
use std::time::Duration;

use crate::explorer::config::Config;
use crate::explorer::eventlog::EventLog;
use crate::explorer::monitor;
use bincode;
use failure::Fail;
use serde::{Deserialize, Serialize};

use utils::tfrecord;

#[derive(Serialize, Deserialize)]
pub enum LogMessage<E> {
    Event(E),
    NewBest {
        score: f64,
        cpt: usize,
        timestamp: Duration,
    },
    Finished(monitor::TerminationReason),
}

#[derive(Debug, Fail)]
pub enum LogError {
    #[fail(display = "{}", _0)]
    IOError(#[cause] ::std::io::Error),
    #[fail(display = "event serialization failed")]
    SerializationError(#[cause] bincode::Error),
    #[fail(display = "{}", _0)]
    RecvError(mpsc::RecvError),
}

impl From<::std::io::Error> for LogError {
    fn from(error: ::std::io::Error) -> LogError {
        LogError::IOError(error)
    }
}

impl From<bincode::Error> for LogError {
    fn from(error: bincode::Error) -> LogError {
        LogError::SerializationError(error)
    }
}

impl From<mpsc::RecvError> for LogError {
    fn from(error: mpsc::RecvError) -> LogError {
        LogError::RecvError(error)
    }
}

pub fn log<E: Send + Serialize>(
    config: &Config,
    recv: mpsc::Receiver<LogMessage<E>>,
) -> Result<(), LogError> {
    let mut record_writer = init_eventlog(config)?;
    let mut write_buffer = init_log(config)?;
    while let Ok(message) = recv.recv() {
        match message {
            LogMessage::Event(event) => {
                record_writer.write_record(&bincode::serialize(&event)?)?;
            }
            LogMessage::NewBest {
                score,
                cpt,
                timestamp,
            } => {
                log_monitor(score, cpt, timestamp, &mut write_buffer);
            }
            LogMessage::Finished(reason) => {
                writeln!(write_buffer, "search stopped because {}", reason)?;
            }
        }
    }
    write_buffer.flush()?;
    record_writer
        .into_inner()
        .map_err(io::Error::from)?
        .finish()?
        .flush()?;
    Ok(())
}

fn init_eventlog(config: &Config) -> io::Result<tfrecord::Writer<EventLog>> {
    Ok(EventLog::create(&config.event_log)?)
}

fn init_log(config: &Config) -> io::Result<BufWriter<File>> {
    let mut output_file = File::create(&config.log_file)?;
    writeln!(output_file, "LOGGER\n{}", config)?;
    Ok(BufWriter::new(output_file))
}

fn log_monitor(
    score: f64,
    cpt: usize,
    timestamp: Duration,
    write_buffer: &mut BufWriter<File>,
) {
    let t_s = timestamp.as_secs();
    let n_seconds = t_s % 60;
    let n_minutes = (t_s / 60) % 60;
    let n_hours = t_s / 3600;
    let message = format!(
        "New best candidate, score: {:.4e}ns, timestamp: {}h {}m {}s, \
         {} candidates evaluated\n",
        score, n_hours, n_minutes, n_seconds, cpt
    );
    write_buffer.write_all(message.as_bytes()).unwrap();
}
