//! exploration of the search space.
mod candidate;
mod logger;
mod monitor;
mod parallel_list;
mod store;

pub mod bandit_arm;
pub mod choice;
pub mod config;
pub mod eventlog;
pub mod local_selection;
pub mod mcts;

pub use self::candidate::Candidate;
pub use self::config::{BanditConfig, Config, SearchAlgorithm};
pub use self::logger::LogMessage;

use self::choice::fix_order;
use self::monitor::{monitor, MonitorMessage};
use self::parallel_list::ParallelCandidateList;
use self::store::Store;

use crate::device::{Context, EvalMode};
use crate::model::bound;
use crate::search_space::SearchSpace;

use crossbeam;
use futures::executor;
use futures::prelude::*;
use log::{error, info, warn};
use std::sync::{
    self,
    atomic::{AtomicUsize, Ordering},
    mpsc, Mutex,
};
use utils::unwrap;

pub type CheckResultFn<'a> =
    dyn Fn(&Candidate, &dyn Context) -> Result<(), String> + Sync + 'a;

// TODO(cc_perf): To improve performances, the following should be considered:
// * choices should be ranked once and then reused for multiple steps.
// * empty and unitary choices should be applied a soon as they are detected.
// * illegal actions should be forbidden by applying their inverse as soon as possible.
// * avoid one copy of the candidate by reusing previous one when applying a choice might
//   be beneficial.

/// Entry point of the exploration. This function returns the best candidate that it has found in
/// the given time (or at whatever point we decided to stop the search - potentially after an
/// exhaustive search)
pub fn find_best(
    config: &Config,
    context: &dyn Context,
    search_space: Vec<SearchSpace>,
    check_result_fn: Option<&CheckResultFn<'_>>,
) -> Option<SearchSpace> {
    find_best_ex(
        config,
        context,
        search_space
            .into_iter()
            .map(|s| {
                let bound = bound(&s, context);
                Candidate::new(s, bound)
            })
            .collect(),
        check_result_fn,
    )
    .map(|c| c.space)
}

struct MctsBuilder<'a> {
    space: SearchSpace,
    config: &'a Config,
    bandit_config: &'a BanditConfig,
    context: &'a dyn Context,
    check_result_fn: Option<&'a CheckResultFn<'a>>,
}

impl<'a> MctsBuilder<'a> {
    fn search<N, E>(
        self,
        tree_policy: Box<dyn mcts::TreePolicy<N, E>>,
        default_policy: Box<dyn mcts::TreePolicy<N, E>>,
    ) -> Option<Candidate>
    where
        N: Sync + Send + std::fmt::Debug + Default,
        E: Sync + Send + std::fmt::Debug + Default,
    {
        let MctsBuilder {
            space,
            config,
            bandit_config,
            context,
            check_result_fn,
        } = self;

        crossbeam::scope(|scope| {
            let (log_sender, log_receiver) = mpsc::sync_channel(100);
            unwrap!(scope
                .builder()
                .name("Telamon - Logger".to_string())
                .spawn(|_| unwrap!(logger::log(config, log_receiver))));

            let store = mcts::MctsStore::new(
                space,
                context,
                bandit_config,
                tree_policy,
                default_policy,
                log_sender.clone(),
            );

            unwrap!(scope
                .builder()
                .name("Telamon - Search".to_string())
                .spawn(move |_| launch_search(
                    config,
                    store,
                    context,
                    log_sender,
                    check_result_fn
                ))
                .unwrap()
                .join())
        })
        .unwrap()
    }
}

struct TreeBuilder<'l> {
    candidates: Vec<Candidate>,
    config: &'l Config,
    bandit_config: &'l BanditConfig,
    context: &'l dyn Context,
    check_result_fn: Option<&'l CheckResultFn<'l>>,
}

impl<'l> TreeBuilder<'l> {
    fn build<P: bandit_arm::TreePolicy>(self, policy: P) -> Option<Candidate>
    where
        P: 'l + Send + Sync,
        P::EdgeStats: Send + Sync,
    {
        let TreeBuilder {
            candidates,
            config,
            bandit_config,
            context,
            check_result_fn,
        } = self;

        crossbeam::scope(|scope| {
            let (log_sender, log_receiver) = sync::mpsc::sync_channel(100);
            unwrap!(scope
                .builder()
                .name("Telamon - Logger".to_string())
                .spawn(|_| (unwrap!(logger::log(config, log_receiver)))));

            let tree = bandit_arm::Tree::new(
                candidates,
                bandit_config,
                policy,
                log_sender.clone(),
            );

            unwrap!(scope
                .builder()
                .name("Telamon - Search".to_string())
                .spawn(move |_| launch_search(
                    config,
                    tree,
                    context,
                    log_sender,
                    check_result_fn
                ))
                .unwrap()
                .join())
        })
        .unwrap()
    }
}

/// Same as `find_best`, but allows to specify pre-existing actions and also returns the
/// actions for the best candidate.
pub fn find_best_ex(
    config: &Config,
    context: &dyn Context,
    candidates: Vec<Candidate>,
    check_result_fn: Option<&CheckResultFn<'_>>,
) -> Option<Candidate> {
    match config.algorithm {
        config::SearchAlgorithm::MultiArmedBandit(ref bandit_config) => {
            let builder = TreeBuilder {
                candidates,
                config,
                bandit_config,
                context,
                check_result_fn,
            };
            match &bandit_config.tree_policy {
                self::config::TreePolicy::UCT(uct_config) => {
                    builder.build(bandit_arm::UCTPolicy::from(uct_config.clone()))
                }
                self::config::TreePolicy::TAG(tag_config) => {
                    builder.build(bandit_arm::TAGPolicy::from(tag_config.clone()))
                }
                self::config::TreePolicy::Bound => {
                    builder.build(self::config::NewNodeOrder::Bound)
                }
                self::config::TreePolicy::WeightedRandom => {
                    builder.build(self::config::NewNodeOrder::WeightedRandom)
                }
                self::config::TreePolicy::RoundRobin => panic!(
                    "Round-robin policy not supported with legacy 'bandit' implementation.  Use `'mcts'` instead."
                ),
            }
        }
        config::SearchAlgorithm::Mcts(ref bandit_config) => {
            assert!(candidates.len() == 1);

            let builder = MctsBuilder {
                space: candidates.into_iter().next().unwrap().space,
                config,
                bandit_config,
                context,
                check_result_fn,
            };

            let default_policy = Box::new(bandit_config.new_nodes_order);

            match &bandit_config.tree_policy {
                config::TreePolicy::UCT(uct_config) => builder
                    .search::<(), mcts::UCTStats>(
                        Box::new(mcts::UCTPolicy::from(uct_config.clone())),
                        default_policy,
                    ),
                config::TreePolicy::TAG(tag_config) => builder
                    .search::<(), mcts::TAGStats>(
                        Box::new(mcts::TAGPolicy::from(tag_config.clone())),
                        default_policy,
                    ),
                config::TreePolicy::Bound => builder.search::<(), ()>(
                    Box::new(config::NewNodeOrder::Bound),
                    default_policy,
                ),
                config::TreePolicy::WeightedRandom => builder.search::<(), ()>(
                    Box::new(config::NewNodeOrder::WeightedRandom),
                    default_policy,
                ),
                config::TreePolicy::RoundRobin => builder
                    .search::<(), mcts::CommonStats>(
                        Box::new(mcts::RoundRobinPolicy),
                        default_policy,
                    ),
            }
        }
        config::SearchAlgorithm::BoundOrder => crossbeam::scope(|scope| {
            let (log_sender, log_receiver) = sync::mpsc::sync_channel(100);
            unwrap!(scope
                .builder()
                .name("Telamon - Logger".to_string())
                .spawn(|_| (unwrap!(logger::log(config, log_receiver)))));

            let candidate_list = ParallelCandidateList::new(config.num_workers);
            candidate_list.insert_many(candidates);
            unwrap!(scope
                .builder()
                .name("Telamon - Search".to_string())
                .spawn(move |_| launch_search(
                    config,
                    candidate_list,
                    context,
                    log_sender,
                    check_result_fn
                ))
                .unwrap()
                .join())
        })
        .unwrap(),
    }
}

/// Launch all threads needed for the search. wait for each one of them to finish. Monitor is
/// supposed to return the best candidate found
fn launch_search<T: Store>(
    config: &Config,
    candidate_store: T,
    context: &dyn Context,
    log_sender: sync::mpsc::SyncSender<LogMessage<T::Event>>,
    check_result_fn: Option<&CheckResultFn<'_>>,
) -> Option<Candidate> {
    let (monitor_sender, monitor_receiver) = futures::sync::mpsc::channel(100);
    let maybe_candidate = crossbeam::scope(|scope| {
        let best_cand_opt = scope
            .builder()
            .name("Telamon - Monitor".to_string())
            .spawn(|_| {
                monitor(
                    config,
                    context,
                    &candidate_store,
                    monitor_receiver,
                    log_sender,
                )
            })
            .unwrap();
        explore_space(
            config,
            &candidate_store,
            monitor_sender,
            context,
            check_result_fn,
        );
        unwrap!(best_cand_opt.join())
    })
    .unwrap();
    // At this point all threads have ended and nobody is going to be
    // exploring the candidate store anymore, so the stats printer
    // should have a consistent view on the tree.
    candidate_store.print_stats();
    maybe_candidate
}

/// Defines the work that explorer threads will do in a closure that will be passed to
/// context.async_eval. Also defines a callback that will be executed by the evaluator
fn explore_space<T>(
    config: &Config,
    candidate_store: &T,
    eval_sender: futures::sync::mpsc::Sender<MonitorMessage<T>>,
    context: &dyn Context,
    check_result_fn: Option<&CheckResultFn<'_>>,
) where
    T: Store,
{
    let best_mutex = &Mutex::new(None);
    let n_evals = &AtomicUsize::new(0);
    let stabilizer = &context.stabilizer().skip_bad_candidates(true);

    context.async_eval(config.num_workers, EvalMode::FindBest, &|evaluator| {
        while let Some((cand, payload)) = candidate_store.explore(context) {
            let space = fix_order(cand.space);
            let eval_sender = eval_sender.clone();
            evaluator.add_kernel(Candidate { space, ..cand }, move |leaf, compiled| {
                let mut best = best_mutex.lock().unwrap();
                let n_evals = n_evals.fetch_add(1, Ordering::SeqCst);

                let mut eval = unwrap!(
                    stabilizer
                        .wrap(compiled)
                        .bound(Some(leaf.bound.value()))
                        .best(*best)
                        .evaluate(),
                    "evaluation failed for actions {:?}, with kernel {}",
                    leaf.actions,
                    compiled
                );

                if let Some(check_result_fn) = check_result_fn {
                    if eval.is_finite()
                        && (config.check_all || best.is_none() || Some(eval) < *best)
                    {
                        // The values computed by the kernel are kept in the context, so we
                        // need to do this *now* before the evaluator runs any other version of
                        // the kernel.
                        if let Err(err) = check_result_fn(&leaf, context) {
                            error!(
                                "Invalid results (score {:.4e}ns) at #{} for {}: {}",
                                eval, n_evals, leaf, err
                            );

                            config
                                .output_path(format!("error_{}", n_evals))
                                .and_then(|path| leaf.dump_to(path, context, eval, &err))
                                .unwrap_or_else(|err| {
                                    error!("Error while dumping candidate: {}", err)
                                });

                            eval = std::f64::INFINITY;
                        }
                    }
                }

                // Only update best if the check passed!
                if eval.is_finite() && (best.is_none() || Some(eval) < *best) {
                    *best = Some(eval);
                }

                if let Err(err) =
                    executor::spawn(eval_sender.send((leaf, eval, payload)).map(|_| ()))
                        .wait_future()
                {
                    warn!("Got disconnected , {:?}", err);
                }
            });
        }
    });
}

/// Explores the full search space.
pub fn gen_space<F, G>(
    context: &dyn Context,
    space: SearchSpace,
    mut on_node: F,
    mut on_leaf: G,
) where
    F: FnMut(&Candidate),
    G: FnMut(&Candidate),
{
    let perf_bound = bound(&space, context);
    let mut stack = vec![Candidate::new(space, perf_bound)];
    let mut total = 0;

    info!("Beginning exploration");
    while let Some(candidate) = stack.pop() {
        total += 1;
        if total % 10 == 0 {
            warn!("{} candidates", total);
        }
        let choice_opt = choice::default_list(&candidate.space).next();
        if let Some(choice) = choice_opt {
            on_node(&candidate);
            stack.extend(candidate.apply_choice(context, choice));
        } else {
            let space = fix_order(candidate.space);
            on_leaf(&Candidate { space, ..candidate });
        }
    }
    info!("{} candidates explored", total);
}
