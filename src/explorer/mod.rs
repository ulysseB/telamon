//! Exploration of the search space.
mod candidate;
mod parallel_list;
pub mod choice;
mod bandit_arm;
mod config;
mod store;
mod monitor;
mod logger;

use self::monitor::monitor;
use self::monitor::MonitorMessage;

use crossbeam;

pub use self::candidate::Candidate;

use self::parallel_list::ParallelCandidateList;
use self::choice::{fix_order};
use device::Context;
use model::bound;
use search_space::SearchSpace;
use std::sync::mpsc;

use self::bandit_arm::{SafeTree, SearchTree};
pub use self::config::{Config, SearchAlgorithm};
use self::store::Store;

// TODO(cc_perf): To improve performances, the following should be considered:
// * choices should be ranked once and then reused for multiple steps.
// * empty and unitary choices should be applied a soon as they are detected.
// * illegal actions should be forbidden by applying their inverse as soon as possible.
// * avoid one copy of the candidate by reusing previous one when applying a choice might
//   be beneficial.


pub fn find_best<'a, 'b>(config: &Config, 
                         context: &'b Context<'b>, 
                         search_space: Vec<SearchSpace<'a>>) -> Option<SearchSpace<'a>> { 
    match config.algorithm {
        config::SearchAlgorithm::MultiArmedBandit(ref band_config) => {
            let new_candidates = search_space.into_iter().map(|space| {
                let bound = bound(&space, context);
                Candidate::new(space, bound)
            }).collect();
            let root = SearchTree::new(new_candidates, context);
            let safe_tree = SafeTree::new(root, band_config);
            launch_search(config, safe_tree, context)
        }
        config::SearchAlgorithm::BoundOrder => {
            let candidate_list = ParallelCandidateList::new(config.num_workers);
            for space in search_space {
                let bound = bound(&space, context);
                candidate_list.insert(Candidate::new(space, bound));
            }
            launch_search(config, candidate_list, context)
        }
    }
}

/// Launch all threads needed for the search. wait for each one of them to finish. Monitor is
/// supposed to return the best candidate found
fn launch_search<'a, T>(config: &Config, candidate_store: T, context: &Context) 
    -> Option<SearchSpace<'a>> where T: Store<'a>  
{
    let (monitor_sender, monitor_receiver) = mpsc::sync_channel(100);
    let (log_sender, log_receiver) = mpsc::sync_channel(100);
    crossbeam::scope( |scope| {
        scope.spawn( || logger::log(config, log_receiver));
        let best_cand_opt = scope.spawn(|| monitor(config, &candidate_store, monitor_receiver, log_sender));
        explore_space(config, &candidate_store, monitor_sender, context);
        best_cand_opt
    }).join()
}

fn explore_space<'a, T>(config: &Config, 
                        candidate_store: &T, 
                        eval_sender: mpsc::SyncSender<MonitorMessage<'a, T>>, 
                        context: &Context) where T: Store<'a> 
{
    context.async_eval(config.num_workers, &|evaluator| {
        while let Some((cand, payload)) = candidate_store.explore(config, context) {
            let space = fix_order(cand.space);
            let eval_sender = eval_sender.clone();
            let callback = move |leaf, eval, cpt| {
                eval_sender.send((leaf, eval, cpt, payload)).unwrap();
            };
            evaluator.add_kernel(Candidate {space, .. cand }, Box::new(callback));
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
