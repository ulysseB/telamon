
use std::sync::mpsc;
use std::fs::File;
use std::io::{Write, BufWriter};
use std::time::Instant;
use explorer::config::Config;
use crossbeam;
use std::time::Duration;

pub enum LogMessage {
    Evaluator{score: f64, cpt: usize},
    Explorer,
    Monitor{score: f64, cpt: usize, timestamp: Duration},
}


pub fn log(config: &Config, recv: mpsc::Receiver<LogMessage>) {
    let mut write_buffer = init_log(config);
    while let Ok(message) = recv.recv() {
        match message {
            LogMessage::Monitor{score, cpt, timestamp} =>{
                log_monitor(score, cpt, timestamp, &mut write_buffer);
            }
            // For now the evaluator is the only one to send logs, so we just ignore any other
            // types of message
            _ => { }
        }
        write_buffer.flush().unwrap();
    }
}

fn init_log(config: &Config) -> BufWriter<File> {
    let mut output_file = unwrap!(File::create(&config.log_file));
    unwrap!(write!(output_file, "LOGGER\n{}\n", config));
    BufWriter::new(output_file)
}

fn log_monitor(score: f64, cpt: usize, timestamp: Duration, write_buffer: &mut BufWriter<File>) {
    let t_s = timestamp.as_secs();
    let n_seconds = t_s % 60;
    let n_minutes = (t_s / 60) % 60;
    let n_hours = t_s / 3600;
    let message = format!("New best candidate, score: {:.4e}ns, timestamp: {}h {}m {}s, \
                         {} candidates evaluated\n",
                         score, n_hours, n_minutes, n_seconds, cpt);
    unwrap!(write_buffer.write_all(message.as_bytes()));
}
