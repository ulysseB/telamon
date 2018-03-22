//! This file exposes a single function, monitor, that is launched in a special thread and pulls
//! the evaluations results, store them and then updates the Store accordingly.
use explorer::store::Store;
use explorer::candidate::Candidate;
use explorer::config::Config;
use std::sync;
use futures::channel;
use search_space::SearchSpace;
use std::time::Instant;
use explorer::logger::LogMessage;
use std;
use std::time::Duration;
use futures::stream;
use futures::prelude::*;
use futures::executor::block_on;
use tokio_timer::*;

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
                      recv: channel::mpsc::Receiver<MonitorMessage<'a, T>>,
                     log_sender: sync::mpsc::SyncSender<LogMessage>) 
    //-> Option<SearchSpace<'a>> where T: Store<'a>,  <T as Store<'a>>::PayLoad: 'a
    //-> Option<SearchSpace<'a>> where T: Store<'a>, T:'a
    -> Option<SearchSpace<'a>> where T: Store<'a>
{
    warn!("Monitor waiting for evaluation results");
    let t0 = Instant::now();
    let mut best_cand: Option<(Candidate, f64)> = None;
    match block_on(
        get_future_timeout(config, recv, t0, candidate_store, &log_sender, &mut best_cand)
        ) {
        Err(CommEnd::TimedOut) => {candidate_store.update_cut(0.0); warn!("Timeout expired")}
        Err(CommEnd::Other) => warn!("No candidates to try anymore"),
        Ok(_) => warn!("What ???? An Error is supposed to happen at some point")
    }
    best_cand.map(|x| x.0.space)
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
        warn!("Got a new best candidate, score: {:.3e}", eval);
        candidate_store.update_cut(eval);
        let log_message = LogMessage::Monitor{score: eval, cpt, timestamp:t};
        log_sender.send(log_message).unwrap();
        *best_cand = Some((cand, eval));
    }
    candidate_store.commit_evaluation(config, payload, eval);

}

//fn get_future<'a, T, U, V>(config: &Config, 
//                             receiver: channel::mpsc::Receiver<MonitorMessage<'a, T>>)
//    -> Box<Future<Item=(), Error=()>>  where T: Store<'a>, <T as Store<'a>>::PayLoad: 'a {
//    //timer.timeout(receiver.for_each(|message| {handle_message::<T>(message); Ok(())})
//    if let Some(timeout) = config.timeout {
//        Box::new(get_future_timeout(config, receiver))
//    } else { Box::new(receiver.for_each(|message| {Ok(())})
//        .map_err(|_| CommEnd::Other))
//    }
//}

fn get_future_timeout<'a, 'b, T>(config: &'b Config, 
                             receiver: channel::mpsc::Receiver<MonitorMessage<'a, T>>,
                             t0: Instant,
                             candidate_store: &'b T,
                             log_sender: &'b sync::mpsc::SyncSender<LogMessage>,
                             best_cand: &'b mut Option<(Candidate<'a>, f64)>
                             )
    -> impl Future<Error=CommEnd> + 'b where T: Store<'a>, <T as Store<'a>>::PayLoad: 'b, 'a: 'b {
    let timer = configure_timer();
    let timeout = config.timeout.unwrap_or(1);
    println!("TIMEOUT: {}min", timeout);
    let time = Duration::from_secs(timeout as u64 * 60);
    timer.timeout(receiver.for_each(move |message| {handle_message::<T>(config,
                                                                   message,
                                                                   t0,
                                                                   candidate_store,
                                                                   log_sender,
                                                                   best_cand);
    Ok(())})
                  .map_err(|_| CommEnd::Other), time)
}

fn configure_timer() -> Timer {
    let builder = wheel();
    let builder = builder.tick_duration(Duration::new(1,0));
    let builder = builder.num_slots(usize::pow(2, 16));
    builder.build()
}

fn stop_search(config: &Config) -> bool { false }
