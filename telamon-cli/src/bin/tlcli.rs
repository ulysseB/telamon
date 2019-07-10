use std::borrow::Cow;
use std::io;
use std::path::PathBuf;
use std::sync::{atomic, Arc, Mutex};

use serde_json;
use structopt::StructOpt;

use telamon::device;
use telamon::explorer::{
    self,
    choice::{ActionEx as Action, Choice},
    config,
    eventlog::EventLog,
    mcts, Candidate,
};
use telamon::model::Bound;
use telamon::offline_analysis::tree::CandidateTree;
use telamon::search_space::SearchSpace;
use telamon_kernels::{linalg, Kernel, KernelBuilder};

/// Compute bounds.csv
#[derive(StructOpt)]
struct Bounds {
    #[structopt(long = "order")]
    order: Option<config::ChoiceOrdering>,

    #[structopt(long = "num-runs", default_value = "500")]
    num_runs: usize,
}

/// Ignore candidates with a too big bound in tests.
const CUT: f64 = 2e8f64;

impl Bounds {
    fn next_choice(&self, space: &SearchSpace) -> Option<Choice> {
        if let Some(order) = &self.order {
            explorer::choice::list(order, space).next()
        } else {
            explorer::choice::default_list(space).next()
        }
    }

    /// Descend along a path in the search tree and stores the bounds encountered on the way.
    fn random_descent(
        &self,
        candidates: &[Candidate],
        context: &dyn device::Context,
    ) -> Option<(Candidate, Vec<Bound>)> {
        let order = explorer::config::NewNodeOrder::Random;
        let mut candidates = Cow::Borrowed(candidates);
        let mut bounds = Vec::new();
        loop {
            let idx = if let Some(idx) = order.pick_candidate(&candidates, CUT) {
                idx
            } else {
                break None;
            };
            bounds.push(candidates[idx].bound.clone());
            let choice_opt = self.next_choice(&candidates[idx].space);
            if let Some(choice) = choice_opt {
                let new_nodes = candidates[idx]
                    .apply_choice(context, choice)
                    .into_iter()
                    .filter(|x| x.bound.value() < CUT)
                    .collect::<Vec<_>>();
                candidates = std::borrow::Cow::Owned(new_nodes);
            } else {
                break Some((
                    match candidates {
                        Cow::Borrowed(candidates) => candidates[idx].clone(),
                        Cow::Owned(mut candidates) => candidates.swap_remove(idx),
                    },
                    bounds,
                ));
            }
        }
    }

    fn test_bound(
        &self,
        candidates: Vec<Candidate>,
        context: &dyn device::Context,
    ) -> Vec<(f64, Vec<f64>)> {
        let leaves = Mutex::new(Vec::new());
        let num_tested = atomic::AtomicUsize::new(0);
        let stabilizer = &context.stabilizer();
        context.async_eval(
            1, // TODO: num_cpus
            device::EvalMode::TestBound,
            &|evaluator| loop {
                if num_tested.fetch_add(1, atomic::Ordering::SeqCst) >= self.num_runs {
                    if num_tested.fetch_sub(1, atomic::Ordering::SeqCst) > self.num_runs {
                        break;
                    }
                }

                if let Some((leaf, mut bounds)) =
                    self.random_descent(&candidates, context)
                {
                    evaluator.add_kernel(leaf, {
                        let leaves = &leaves;
                        move |leaf, kernel| {
                            let bound = leaf.bound.clone();
                            let runtime = stabilizer
                                .wrap(kernel)
                                .bound(Some(bound.value()))
                                .evaluate()
                                .unwrap();
                            let mut leaves = leaves.lock().unwrap();
                            bounds.push(bound);
                            leaves.push((
                                runtime,
                                bounds.into_iter().map(|bound| bound.value()).collect(),
                            ));
                        }
                    });
                } else {
                    num_tested.fetch_sub(1, atomic::Ordering::SeqCst);
                }
            },
        );
        leaves.into_inner().unwrap()
    }

    fn run(&self, _args: &Opt) -> io::Result<()> {
        let executor = telamon_cuda::Executor::init();
        let mut context = telamon_cuda::Context::new(&executor);
        let params = linalg::FusedMMP::new(256, 256, 32);

        let (signature, kernel, context) = KernelBuilder::default()
            .build::<linalg::FusedMM<f32>, telamon_cuda::Context>(
                params.clone(),
                &mut context,
            );
        let signature = Arc::new(signature);
        let expected = kernel.get_expected_output(context);

        for (runtime, bounds) in
            self.test_bound(kernel.build_body(signature, context), context)
        {
            print!("matmul,{}", runtime);
            for bound in bounds {
                print!(",{}", bound);
            }
            print!("\n");
        }

        Ok(())
    }
}

#[derive(StructOpt)]
struct Rebuild {
    /// Path to the eventlog to rebuild from
    #[structopt(
        parse(from_os_str),
        short = "i",
        long = "input",
        default_value = "eventlog.tfrecord.gz"
    )]
    eventlog: PathBuf,

    /// Identifier of the candidate node to rebuild.  This corresponds to the ID indicated in
    /// `watch.log`.
    id: usize,
}

impl Rebuild {
    fn find_candidate(&self) -> io::Result<(mcts::NodeId, f64, Vec<Action>)> {
        let mut nevals = 0;
        let mut tree = CandidateTree::new();

        for record_bytes in EventLog::open(&self.eventlog)?.records() {
            match bincode::deserialize(&record_bytes?)
                .map_err(|err| io::Error::new(io::ErrorKind::Other, err))?
            {
                mcts::Message::Node {
                    id,
                    parent,
                    mut children,
                    bound,
                    discovery_time,
                } => tree.extend(id, discovery_time, parent, bound, &mut children),
                mcts::Message::Trace { .. } => (),
                mcts::Message::Evaluation { id, value, .. } => {
                    if let Some(value) = value {
                        if nevals == self.id {
                            return Ok((id, value, tree.get_node(id).actions()));
                        }

                        nevals += 1;
                    }
                }
            }
        }

        Err(io::Error::new(
            io::ErrorKind::Other,
            "Unable to find candidate",
        ))
    }

    fn run(&self, _args: &Opt) -> io::Result<()> {
        let (id, score, actions) = self.find_candidate()?;

        println!("Found candidate {} (score: {})", id, score);
        println!("{}", serde_json::to_string_pretty(&actions)?);

        Ok(())
    }
}

#[derive(StructOpt)]
struct Stats {
    /// Path to the eventlog to compute stats for
    #[structopt(
        parse(from_os_str),
        short = "i",
        long = "input",
        default_value = "eventlog.tfrecord.gz"
    )]
    eventlog: PathBuf,
}

impl Stats {
    fn run(&self, _args: &Opt) -> io::Result<()> {
        let (mut nimpl, mut ndead) = (0, 0);
        for record_bytes in EventLog::open(&self.eventlog)?.records() {
            match bincode::deserialize(&record_bytes?)
                .map_err(|err| io::Error::new(io::ErrorKind::Other, err))?
            {
                mcts::Message::Node { .. } => (),
                mcts::Message::Trace { events, .. } => match events.last() {
                    Some(mcts::Timed {
                        value: mcts::Event::Implementation,
                        ..
                    }) => nimpl += 1,
                    _ => ndead += 1,
                },
                mcts::Message::Evaluation { .. } => (),
            }
        }

        println!("Implementations: {}", nimpl);
        println!("Deadends: {}", ndead);

        Ok(())
    }
}

#[derive(StructOpt)]
enum Command {
    #[structopt(name = "rebuild")]
    Rebuild(Rebuild),

    #[structopt(name = "bounds")]
    Bounds(Bounds),

    #[structopt(name = "stats")]
    Stats(Stats),
}

#[derive(StructOpt)]
#[structopt(name = "telamon")]
struct Opt {
    #[structopt(subcommand)]
    command: Command,
}

fn main() -> io::Result<()> {
    let args = Opt::from_args();
    env_logger::init();

    match &args.command {
        Command::Rebuild(rebuild) => rebuild.run(&args),
        Command::Bounds(bounds) => bounds.run(&args),
        Command::Stats(stats) => stats.run(&args),
    }
}
