//! This file exposes a single function, monitor, that is launched in a special
//! thread and pulls the evaluations results, store them and then updates the
//! Store accordingly.
use explorer::candidate::Candidate;
use explorer::config::Config;
use explorer::logger::LogMessage;
use explorer::store::Store;
use futures::channel;
use futures::executor::block_on;
use futures::prelude::*;
use std;
use std::sync;
use std::time::{Duration, Instant};
use tokio_timer::*;

pub type MonitorMessage<'a, T> = (Candidate<'a>, f64, <T as Store<'a>>::PayLoad);

/// Indicates why the exploration was terminated.
#[derive(Serialize, Deserialize)]
pub enum TerminationReason {
    /// The maximal number of evaluation was reached.
    MaxEvaluations,
    /// The timeout was reached.
    Timeout,
}

impl<T> From<TimeoutError<T>> for TerminationReason {
    fn from(_: TimeoutError<T>) -> Self {
        TerminationReason::Timeout
    }
}

impl std::fmt::Display for TerminationReason {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            TerminationReason::MaxEvaluations => {
                write!(f, "the maximum number of evaluations was reached")
            }
            TerminationReason::Timeout => {
                write!(f, "the maximum exploration time was reached")
            }
        }
    }
}

struct Status<'a> {
    best_candidate: Option<(Candidate<'a>, f64)>,
    num_evaluations: usize,
}

impl<'a> Default for Status<'a> {
    fn default() -> Self {
        Status {
            best_candidate: None,
            num_evaluations: 0,
        }
    }
}

/// This function is an interface supposed to make a connection between the
/// Store and the evaluator. Retrieve evaluations, retains the results and
/// update the store accordingly.
pub fn monitor<'a, T, E>(
    config: &Config,
    candidate_store: &T,
    recv: channel::mpsc::Receiver<MonitorMessage<'a, T>>,
    log_sender: sync::mpsc::SyncSender<LogMessage<E>>,
) -> Option<Candidate<'a>>
where
    T: Store<'a>,
{
    warn!("Monitor waiting for evaluation results");
    let t0 = Instant::now();
    let mut status = Status::default();
    let res = if let Some(_) = config.timeout {
        block_until_timeout(config, recv, t0, candidate_store, &log_sender, &mut status)
    } else {
        block_unbounded(config, recv, t0, candidate_store, &log_sender, &mut status)
    };
    let duration = t0.elapsed();
    let duration_secs = duration.as_secs() as f64 + duration.subsec_nanos() as f64 * 1e-9;
    warn!(
        "Exploration finished in {}s with {} candidates evaluated (avg {} candidate/s).",
        duration_secs,
        status.num_evaluations,
        status.num_evaluations as f64 / duration_secs
    );
    match res {
        Ok(_) => warn!("No candidates to try anymore"),
        Err(reason) => {
            warn!("exploration stopped because {}", reason);
            candidate_store.stop_exploration();
            unwrap!(log_sender.send(LogMessage::Finished(reason)));
        }
    }
    status.best_candidate.map(|x| x.0)
}

/// Depending on the value of the evaluation we just did, computes the new cut
/// value for the store Can be 0 if we decide to stop the search
fn get_new_cut(config: &Config, eval: f64) -> f64 {
    if let Some(bound) = config.stop_bound {
        if eval < bound {
            return 0.;
        }
    }
    if let Some(ratio) = config.distance_to_best {
        (1. - ratio / 100.) * eval
    } else {
        eval
    }
}

/// Given a receiver channel, constructs a future that returns whenever the
/// tree has been fully explored.
fn block_unbounded<'a, T, E>(
    config: &Config,
    receiver: channel::mpsc::Receiver<MonitorMessage<'a, T>>,
    t0: Instant,
    candidate_store: &T,
    log_sender: &sync::mpsc::SyncSender<LogMessage<E>>,
    status: &mut Status<'a>,
) -> Result<(), TerminationReason>
where
    T: Store<'a>,
{
    block_on(
        receiver
            .map_err(|n| n.never_into())
            .for_each(move |message| {
                handle_message::<T, E>(
                    config,
                    message,
                    t0,
                    candidate_store,
                    log_sender,
                    status,
                )
            }),
    ).map(|_| ())
}

/// Given a receiver channel, builds a future that returns whenever the tree
/// has been fully explored or the timeout has been reached
fn block_until_timeout<'a, 'b, T, E>(
    config: &'b Config,
    receiver: channel::mpsc::Receiver<MonitorMessage<'a, T>>,
    t0: Instant,
    candidate_store: &'b T,
    log_sender: &'b sync::mpsc::SyncSender<LogMessage<E>>,
    status: &'b mut Status<'a>,
) -> Result<(), TerminationReason>
where
    T: Store<'a>,
{
    let timer = configure_timer();
    //TODO Find a clean way to get a timeout that never returns - or no timeout at
    // all
    let timeout = config.timeout.unwrap_or(10000);
    let time = Duration::from_secs(timeout as u64 * 60);
    block_on(
        timer.timeout(
            receiver
                .map_err(|n| n.never_into())
                .for_each(move |message| {
                    handle_message::<T, E>(
                        config,
                        message,
                        t0,
                        candidate_store,
                        log_sender,
                        status,
                    )
                })
                .map(|_| ()),
            time,
        ),
    )
}

/// All work that has to be done on reception of a message, meaning updating
/// the best cand if needed, logging, committing back to candidate_store
fn handle_message<'a, T, E>(
    config: &Config,
    message: MonitorMessage<'a, T>,
    start_time: Instant,
    candidate_store: &T,
    log_sender: &sync::mpsc::SyncSender<LogMessage<E>>,
    status: &mut Status<'a>,
) -> Result<(), TerminationReason>
where
    T: Store<'a>,
{
    let (cand, eval, payload) = message;

    let wall = start_time.elapsed();
    warn!("Got a new evaluation after {}, bound: {:.4e} score: {:.4e}, current best: {:.4e}",
          status.num_evaluations,
          cand.bound.value(),
          eval,
          status.best_candidate.as_ref().map_or(std::f64::INFINITY, |best:
                                                &(Candidate, f64)| best.1 ));
    candidate_store.commit_evaluation(&cand.actions, payload, eval);

    let change = status
        .best_candidate
        .as_ref()
        .map(|&(_, time)| time > eval)
        .unwrap_or(true);
    if change {
        warn!("Got a new best candidate, score: {:.3e}, {}", eval, cand);
        candidate_store.update_cut(get_new_cut(config, eval));
        let log_message = LogMessage::NewBest {
            score: eval,
            cpt: status.num_evaluations,
            timestamp: wall,
        };
        unwrap!(log_sender.send(log_message));
        status.best_candidate = Some((cand, eval));
    }

    // Note that it is possible that we actually didn't make an
    // evaluation here, because the evaluator may return an infinite
    // runtime for some of the candidates in its queue when a new best
    // candidate was found. In this case, `eval` is be infinite, and
    // we don't count the corresponding evaluation towards the number
    // of evaluations performed (a sequential, non-parallel
    // implementation of the search algorithm would not have selected
    // this candidate since it would get cut).
    if !eval.is_infinite() {
        status.num_evaluations += 1;
        if let Some(max_evaluations) = config.max_evaluations {
            if status.num_evaluations >= max_evaluations {
                return Err(TerminationReason::MaxEvaluations);
            }
        }
    }

    Ok(())
}

/// Builds and returns a timer that suits our needs - that is, which timeout
/// can be set to at least a few tens of hours
fn configure_timer() -> Timer {
    let builder = wheel();
    let builder = builder.tick_duration(Duration::new(10, 0));
    let builder = builder.num_slots(usize::pow(2, 16));
    builder.build()
}
