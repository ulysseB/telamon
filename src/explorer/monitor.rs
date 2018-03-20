//! This file exposes a single function, monitor, that is launched in a special thread and pulls
//! the evaluations results, store them and then updates the Store accordingly.
use explorer::store::Store;
use explorer::candidate::Candidate;
use explorer::config::Config;
use std::sync::mpsc;
use search_space::SearchSpace;
use std::time::Instant;
use explorer::logger::LogMessage;
use std;
use std::time::Duration;
use futures::stream;
use futures::prelude::*;
use futures::executor::LocalPool;
use tokio_timer_futures2::*;

pub type MonitorMessage<'a, T: Store<'a>> = (Candidate<'a>, f64, usize, T::PayLoad);

enum CommEnd {
    TimedOut,
    Other,
}
impl<T> From<TimeoutError<T>> for CommEnd {
    fn from(_: TimeoutError<T>) -> Self {
        CommEnd::TimedOut
    }
}

/// This function is an interface supposed to make a connection between the Store and the
/// evaluator. Retrieve evaluations, retains the results and update the store accordingly.
pub fn monitor<'a, T>(config: &Config, candidate_store: &T, 
                      recv: mpsc::Receiver<MonitorMessage<'a, T>>,
                     log_sender: mpsc::SyncSender<LogMessage>) -> Option<SearchSpace<'a>> where T: Store<'a> {
    warn!("Monitor waiting for evaluation results");
    let t0 = Instant::now();
    let mut results_stack = vec![];
    let mut best_cand = None;
    while let Ok((cand, eval, cpt, payload)) = recv.recv() {
        let t = Instant::now() - t0;
        warn!("Got a new evaluation, bound: {:.4e} score: {:.4e}, current best: {:.4e}",
              cand.bound.value(), eval, best_cand.as_ref().map_or(std::f64::INFINITY, |best:
                                                                  &(Candidate, f64)| best.1 ));
        results_stack.push(eval);
        best_cand = match best_cand {
            Some((old_cand, score)) => {
                if eval < score {
                    warn!("Got a new best candidate, score: {:.3e}", eval);
                    candidate_store.update_cut(eval);
                    let log_message = LogMessage::Monitor{score: eval, cpt, timestamp:t};
                    log_sender.send(log_message).unwrap();
                    Some((cand, eval)) 
                }
                else { Some((old_cand, score)) }
            }
            None => {
                warn!("Got the first candidate, score: {:.3e}", eval);
                candidate_store.update_cut(eval);
                let log_message = LogMessage::Monitor{score: eval, cpt, timestamp:t};
                log_sender.send(log_message).unwrap();
                Some((cand, eval))
            }
        };
        candidate_store.commit_evaluation(config, payload, eval);
    }
    best_cand.map(|x| x.0.space)
}

fn get_future_timeout<F>(config: &Config, future: F)
    -> Timeout<F> where F: Future{
    let timer = Timer::default();
    let time  = Duration::from_secs(config.timeout.unwrap() * 60);
    timer.timeout(future.map_err(|_| CommEnd::Other), time)
}

fn stop_search(config: &Config) -> bool { false }
