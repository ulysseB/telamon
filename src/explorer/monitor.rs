//! This file exposes a single function, monitor, that is launched in a special thread and pulls
//! the evaluations results, store them and then updates the Store accordingly.
use explorer::candidate::Candidate;
use explorer::config::Config;
use explorer::logger::LogMessage;
use explorer::store::Store;
use futures::channel;
use futures::prelude::*;
use futures::executor::block_on;
use std::time::{Duration, Instant};
use std;
use std::sync;
use tokio_timer::*;

pub type MonitorMessage<'a, T> = (Candidate<'a>, f64, usize, <T as Store<'a>>::PayLoad);

enum CommEnd { TimedOut, Other }

impl<T> From<TimeoutError<T>> for CommEnd {
    fn from(_: TimeoutError<T>) -> Self {
        CommEnd::TimedOut
    }
}

/// This function is an interface supposed to make a connection between the Store and the
/// evaluator. Retrieve evaluations, retains the results and update the store accordingly.
pub fn monitor<'a, T>(config: &Config, candidate_store: &T,
                      recv: channel::mpsc::Receiver<MonitorMessage<'a, T>>,
                     log_sender: sync::mpsc::SyncSender<LogMessage>)
    -> Option<Candidate<'a>> where T: Store<'a>
{
    warn!("Monitor waiting for evaluation results");
    let t0 = Instant::now();
    let mut best_cand: Option<(Candidate, f64)> = None;
    let res = if let Some(_) = config.timeout {
        block_until_timeout(
            config, recv, t0, candidate_store, &log_sender, &mut best_cand)
    } else {
        block_unbounded( config, recv, t0, candidate_store, &log_sender, &mut best_cand)
    };
    match res {
        Err(CommEnd::TimedOut) => {
            candidate_store.update_cut(0.0);
            log_sender.send(LogMessage::Timeout).unwrap();
            warn!("Timeout expired")
        }
        Err(CommEnd::Other) =>
            panic!("an error occured in the monitor while witing for nes candidates"),
        Ok(_) => warn!("No candidates to try anymore"),
    }
    best_cand.map(|x| x.0)
}

/// Depending on the value of the evaluation we just did, computes the new cut value for the store
/// Can be 0 if we decide to stop the search
fn get_new_cut(config: &Config, eval: f64) -> f64 {
    if let Some(bound) = config.stop_bound {
        if eval < bound { return 0.;}
    }
    if let Some(ratio) = config.distance_to_best {
        ( 1. - ratio / 100.) * eval
    } else { eval}
}


/// Given a receiver channel, constructs a future that returns whenever the tree has
/// been fully explored.
fn block_unbounded<'a, T>(config: &Config,
                          receiver: channel::mpsc::Receiver<MonitorMessage<'a, T>>,
                          t0: Instant,
                          candidate_store: &T,
                          log_sender: &sync::mpsc::SyncSender<LogMessage>,
                          best_cand: &mut Option<(Candidate<'a>, f64)>)
    -> Result<(), CommEnd> where T: Store<'a>
{
    block_on(receiver.for_each(move |message| {
        handle_message::<T>(config, message, t0, candidate_store, log_sender, best_cand);
        Ok(())
    }).map_err(|_| CommEnd::Other)).map(|_| ())
}

/// Given a receiver channel, builds a future that returns whenever the tree has been fully
/// explored or the timeout has been reached
fn block_until_timeout<'a, 'b, T>(config: &'b Config,
                                 receiver: channel::mpsc::Receiver<MonitorMessage<'a, T>>,
                                 t0: Instant,
                                 candidate_store: &'b T,
                                 log_sender: &'b sync::mpsc::SyncSender<LogMessage>,
                                 best_cand: &'b mut Option<(Candidate<'a>, f64)>)
    -> Result<(), CommEnd> where T: Store<'a>
{
    let timer = configure_timer();
    //TODO Find a clean way to get a timeout that never returns - or no timeout at all
    let timeout = config.timeout.unwrap_or(10000);
    let time = Duration::from_secs(timeout as u64 * 60);
    block_on(timer.timeout(receiver.for_each(move |message| {
        handle_message::<T>(config, message, t0, candidate_store, log_sender, best_cand);
        Ok(())
    }).map_err(|_| CommEnd::Other), time).map(|_| ()))
}

/// All work that has to be done on reception of a message, meaning updating the best cand if
/// needed, logging, committing back to candidate_store
fn handle_message<'a, T>(config: &Config,
                         message: MonitorMessage<'a, T>,
                         t0: Instant,
                         candidate_store: &T,
                         log_sender: &sync::mpsc::SyncSender<LogMessage>,
                         best_cand: &mut Option<(Candidate<'a>, f64)>) where T: Store<'a> {
    let (cand, eval, cpt, payload) = message;
    let t = Instant::now() - t0;
    warn!("Got a new evaluation, bound: {:.4e} score: {:.4e}, current best: {:.4e}",
          cand.bound.value(), eval, best_cand.as_ref().map_or(std::f64::INFINITY, |best:
                                                              &(Candidate, f64)| best.1 ));
    let change = best_cand.as_ref().map(|&(_, time)| time > eval).unwrap_or(true);
    if change {
        warn!("Got a new best candidate, score: {:.3e}, {}", eval, cand);
        candidate_store.update_cut(get_new_cut(config, eval));
        let log_message = LogMessage::NewBest{score: eval, cpt, timestamp:t};
        log_sender.send(log_message).unwrap();
        *best_cand = Some((cand, eval));
    }
    candidate_store.commit_evaluation(payload, eval);
}

/// Builds and returns a timer that suits our needs - that is, which timeout can be set to at least
/// a few tens of hours
fn configure_timer() -> Timer {
    let builder = wheel();
    let builder = builder.tick_duration(Duration::new(10, 0));
    let builder = builder.num_slots(usize::pow(2, 16));
    builder.build()
}
