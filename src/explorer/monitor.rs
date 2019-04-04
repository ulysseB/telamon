//! This file exposes a single function, monitor, that is launched in a special
//! thread and pulls the evaluations results, store them and then updates the
//! Store accordingly.
use crate::device::Context;
use crate::explorer::candidate::Candidate;
use crate::explorer::config::Config;
use crate::explorer::logger::LogMessage;
use crate::explorer::store::Store;
use futures::prelude::*;
use futures::{executor, future, task, Async};
use log::warn;
use serde::{Deserialize, Serialize};
use std::sync::{
    self,
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::{Duration, Instant};
use std::{self, thread};
use utils::unwrap;

pub type MonitorMessage<'a, T> = (Candidate<'a>, f64, <T as Store<'a>>::PayLoad);

/// Indicates why the exploration was terminated.
#[derive(Serialize, Deserialize)]
pub enum TerminationReason {
    /// The maximal number of evaluation was reached.
    MaxEvaluations,
    /// The timeout was reached.
    Timeout,
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
    context: &dyn Context,
    candidate_store: &T,
    recv: futures::sync::mpsc::Receiver<MonitorMessage<'a, T>>,
    log_sender: sync::mpsc::SyncSender<LogMessage<E>>,
) -> Option<Candidate<'a>>
where
    T: Store<'a>,
{
    warn!("Monitor waiting for evaluation results");
    let t0 = Instant::now();
    let mut status = Status::default();
    let log_sender_ref = &log_sender;
    let status_mut = &mut status;
    let fut = recv.map_err(|()| unreachable!()).for_each(move |message| {
        handle_message(
            config,
            context,
            message,
            t0,
            candidate_store,
            log_sender_ref,
            status_mut,
        )
    });
    let res = if let Some(timeout_mins) = config.timeout {
        executor::spawn(
            fut.select(timeout(Duration::from_secs(timeout_mins * 60)))
                .map(|((), _)| ())
                .map_err(|(err, _)| err),
        )
        .wait_future()
    } else {
        executor::spawn(fut).wait_future()
    };

    let duration = t0.elapsed();
    let duration_secs =
        duration.as_secs() as f64 + f64::from(duration.subsec_nanos()) * 1e-9;
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

/// All work that has to be done on reception of a message, meaning updating
/// the best cand if needed, logging, committing back to candidate_store
fn handle_message<'a, T, E>(
    config: &Config,
    context: &dyn Context,
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

        match config.output_path(format!("best_{}", status.num_evaluations)) {
            Ok(output_path) => cand
                .space
                .dump_code(context, output_path)
                .unwrap_or_else(|err| warn!("Error while dumping candidate: {}", err)),
            Err(err) => warn!("Error while dumping candidate: {}", err),
        }

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

struct TimeoutWorker {
    running: Arc<AtomicBool>,
    thread: thread::Thread,
}

impl Drop for TimeoutWorker {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        self.thread.unpark();
    }
}

fn timeout(duration: Duration) -> Box<dyn Future<Item = (), Error = TerminationReason>> {
    let start_time = std::time::Instant::now();

    let mut worker = None;
    Box::new(future::poll_fn(move || {
        if start_time.elapsed() > duration {
            Err(TerminationReason::Timeout)
        } else {
            if worker.is_none() {
                let running = Arc::new(AtomicBool::new(true));
                let task = task::current();
                let thread_running = Arc::clone(&running);
                let thread = thread::spawn(move || loop {
                    if !thread_running.load(Ordering::Relaxed) {
                        break;
                    }

                    let elapsed = start_time.elapsed();
                    if elapsed < duration {
                        thread::park_timeout(duration - elapsed);
                    } else {
                        task.notify();
                    }
                })
                .thread()
                .clone();

                worker = Some(TimeoutWorker { running, thread });
            }

            Ok(Async::NotReady)
        }
    }))
}
