use explorer::config::Config;
use explorer::monitor;
use serde::ser::Serialize;
use bincode;
use std::fs::File;
use std::io;
use std::io::{BufWriter, Write};
use std::sync::mpsc;
use std::time::Duration;
use utils::tfrecord;
use utils::tfrecord::RecordWriter;

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
    #[fail(display = "tfrecord serialization failed")]
    TFRecordError(#[cause] tfrecord::WriteError),
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

impl From<tfrecord::WriteError> for LogError {
    fn from(error: tfrecord::WriteError) -> LogError {
        LogError::TFRecordError(error)
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
    let mut record_writer = File::create(&config.event_log)?;
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
    record_writer.flush()?;
    write_buffer.flush()?;
    return Ok(());
}

fn init_log(config: &Config) -> io::Result<BufWriter<File>> {
    let mut output_file = File::create(&config.log_file)?;
    write!(output_file, "LOGGER\n{}\n", config)?;
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
    unwrap!(write_buffer.write_all(message.as_bytes()));
}
