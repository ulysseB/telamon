//! exploration of the search space.
mod candidate;
mod parallel_list;
mod bandit_arm;
mod store;
mod monitor;
mod logger;

pub mod config;
pub mod choice;
pub mod local_selection;

pub use self::config::{Config, SearchAlgorithm};
pub use self::candidate::Candidate;

use self::choice::fix_order;
use self::monitor::{monitor, MonitorMessage};
use self::parallel_list::ParallelCandidateList;
use self::store::Store;

use boxfnonce::SendBoxFnOnce;
use crossbeam;
use device::{Context, EvalMode};
use model::bound;
use search_space::SearchSpace;
use std::sync;
use futures::prelude::*;
use futures::{channel, SinkExt};
use futures::executor::block_on;

// TODO(cc_perf): To improve performances, the following should be considered:
// * choices should be ranked once and then reused for multiple steps.
// * empty and unitary choices should be applied a soon as they are detected.
// * illegal actions should be forbidden by applying their inverse as soon as possible.
// * avoid one copy of the candidate by reusing previous one when applying a choice might
//   be beneficial.


/// Entry point of the exploration. This function returns the best candidate that it has found in
/// the given time (or at whatever point we decided to stop the search - potentially after an
/// exhaustive search)
pub fn find_best<'a>(config: &Config, 
                     context: &Context,
                     search_space: Vec<SearchSpace<'a>>) -> Option<SearchSpace<'a>> { 
    let candidates = search_space.into_iter().map(|space| {
        let bound = bound(&space, context);
        Candidate::new(space, bound)
    }).collect();
    find_best_ex(config, context, candidates).map(|c| c.space)
}

/// Same as `find_best`, but allows to specify pre-existing actions and also returns the
/// actionsfor the best candidate.
pub fn find_best_ex<'a>(config: &Config, 
                        context: &Context,
                        candidates: Vec<Candidate<'a>>) -> Option<Candidate<'a>> { 
    match config.algorithm {
        config::SearchAlgorithm::MultiArmedBandit(ref band_config) => {
            let tree = bandit_arm::Tree::new(candidates, band_config);
            launch_search(config, tree, context)
        }
        config::SearchAlgorithm::BoundOrder => {
            let candidate_list = ParallelCandidateList::new(config.num_workers);
            for candidate in candidates { candidate_list.insert(candidate); }
            launch_search(config, candidate_list, context)
        }
    }
}

/// Launch all threads needed for the search. wait for each one of them to finish. Monitor is
/// supposed to return the best candidate found
fn launch_search<'a, T>(config: &Config, candidate_store: T, context: &Context) 
    -> Option<Candidate<'a>> where T: Store<'a>
{
    let (monitor_sender, monitor_receiver) = channel::mpsc::channel(100);
    let (log_sender, log_receiver) = sync::mpsc::sync_channel(100);
    let maybe_candidate = crossbeam::scope( |scope| {
        unwrap!(scope.builder().name("Telamon - Logger".to_string())
            .spawn( || logger::log(config, log_receiver)));
        let best_cand_opt = scope.builder().name("Telamon - Monitor".to_string()).
            spawn(|| monitor(config, &candidate_store, monitor_receiver, log_sender));
        explore_space(config, &candidate_store, monitor_sender, context);
        unwrap!(best_cand_opt)
    }).join();
    // At this point all threads have ended and nobody is going to be
    // exploring the candidate store anymore, so the stats printer
    // should have a consistent view on the tree.
    candidate_store.print_stats();
    maybe_candidate
}

/// Defines the work that explorer threads will do in a closure that will be passed to
/// context.async_eval. Also defines a callback that will be executed by the evaluator
fn explore_space<'a, T>(config: &Config, 
                        candidate_store: &T, 
                        eval_sender: channel::mpsc::Sender<MonitorMessage<'a, T>>, 
                        context: &Context) where T: Store<'a> 
{
    context.async_eval(config.num_workers, EvalMode::FindBest, &|evaluator| {
        while let Some((cand, payload)) = candidate_store.explore(context) {
            let space = fix_order(cand.space);
            let eval_sender = eval_sender.clone();
            let callback = move |leaf, eval| {
                if let Err(err) = block_on(eval_sender.send((leaf, eval, payload))
                                           .map(|_| ()))
                { warn!("Got disconnected , {:?}", err);}
            };
            evaluator.add_kernel(Candidate {space, .. cand }, SendBoxFnOnce::from(callback));
        }
    });
} 


/// Explores the full search space.
pub fn gen_space<F, G>(context: &Context, space: SearchSpace, mut on_node: F, mut on_leaf: G)
        where F: FnMut(&Candidate), G: FnMut(&Candidate) {
    let perf_bound = bound(&space, context);
    let mut stack = vec![Candidate::new(space, perf_bound)];
    let mut total = 0;

    info!("Beginning exploration");
    while let Some(candidate) = stack.pop() {
        total += 1;
        if total % 10 == 0 { warn!("{} candidates", total); }
        let choice_opt = choice::list(&candidate.space).next();
        if let Some(choice) = choice_opt {
            on_node(&candidate);
            stack.extend(candidate.apply_choice(context, choice));
        } else {
            let space = fix_order(candidate.space);
            on_leaf(&Candidate { space, .. candidate });
        }
    }
    info!("{} candidates explored", total);
}
