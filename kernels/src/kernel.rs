//! Abstracts kernels so we can build generic methods to test them.
use itertools::Itertools;
use num_cpus;
use rayon::prelude::*;
use statistics;
use std;
use std::ffi::OsStr;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::sync::{atomic, Mutex};
use telamon::explorer::config::{ChoiceGroup, ChoiceOrdering};
use telamon::explorer::{
    local_selection, reroll_last_candidate, Candidate, Config, TreeEvent,
};
use telamon::helper::{Builder, SignatureBuilder};
use telamon::model::Bound;
use telamon::search_space::SearchSpace;
use telamon::{codegen, device, explorer, ir};
use utils::*;

use bincode;
use flate2::write::{GzEncoder, ZlibEncoder};
use flate2::Compression;
use serde::ser::Serialize;

use utils::tfrecord;
use utils::tfrecord::RecordWriter;
/// Ignore candidates with a too big bound in tests.
const CUT: f64 = 2e8f64;
/// Maximal number of deadends to accept before failing.
// TODO(cleanup): tune MAX_DEADEND_RATIO
//const MAX_DEADEND_RATIO: usize = 20;
const MAX_DEADEND_RATIO: f32 = 0.95;

/// A kernel that can be compiled, benchmarked and used for correctness tests.
pub trait Kernel<'a>: Sized {
    /// The input parameters of the kernel.
    type Parameters: Clone;
    /// The values to expect as output.
    type ExpectedOutput;

    /// The name of the function computed by the kernel.
    fn name() -> &'static str;

    /// Builds the signature of the kernel in the builder and returns an object that
    /// stores enough information to later build the kernel body and check its result.
    /// The `is_generic` flag indicates if th sizes should be instantiated.
    fn build_signature<AM>(
        parameters: Self::Parameters,
        builder: &mut SignatureBuilder<AM>,
    ) -> Self
    where
        AM: device::ArgMap + device::Context + 'a;

    /// Builder the kernel body in the given builder. This builder should be based on the
    /// signature created by `build_signature`.
    fn build_body<'b>(
        &self,
        signature: &'b ir::Signature,
        ctx: &'b device::Context,
    ) -> Vec<Candidate<'b>>;

    /// Computes the expected output.
    fn get_expected_output(&self, &device::Context) -> Self::ExpectedOutput;

    /// Ensures the generated code performs the correct operation.
    fn check_result(
        &self,
        expected: &Self::ExpectedOutput,
        context: &device::Context,
    ) -> Result<(), String>;

    fn reroll_last_cand<AM>(params: Self::Parameters, context: &'a mut AM)
    where
        AM: device::ArgMap + device::Context + 'a,
    {
        let kernel;
        let signature = {
            let mut builder = SignatureBuilder::new(Self::name(), context);
            builder.set_random_fill(true);
            kernel = Self::build_signature(params, &mut builder);
            builder.get()
        };
        let builder = Builder::new(&signature, context.device());
        let expected_output = kernel.get_expected_output(context);
        let space = builder.get();
        unwrap!(reroll_last_candidate("eventlog.tfrecord", context, space));
        if let Err(err) = kernel.check_result(&expected_output, context) {
            panic!("incorrect output for kernel {}: {}", Self::name(), err)
        }
    }

    /// Generates, executes and tests the output of candidates for the kernel.
    fn test_correctness<AM>(params: Self::Parameters, num_tests: usize, context: &mut AM)
    where
        AM: device::ArgMap + device::Context + 'a,
    {
        let kernel;
        let signature = {
            let mut builder = SignatureBuilder::new(Self::name(), context);
            builder.set_random_fill(true);
            kernel = Self::build_signature(params, &mut builder);
            builder.get()
        };
        let expected_output = kernel.get_expected_output(context);
        let candidates = kernel.build_body(&signature, context);
        let mut num_deadends = 0;
        let mut num_runs = 0;
        let config = Config::default();
        let mut record_writer = unwrap!(init_eventlog(&config));
        while num_runs < num_tests {
            let order = explorer::config::NewNodeOrder::WeightedRandom;
            let ordering = explorer::config::ChoiceOrdering::default();
            let bounds = candidates.iter().map(|c| c.bound.value()).enumerate();
            let candidate_idx = local_selection::pick_index(order, bounds, CUT);
            let candidate = candidates[unwrap!(candidate_idx)].clone();
            let leaf =
                local_selection::descend(&ordering, order, context, candidate, CUT);
            if let Some(leaf) = leaf {
                let device_fn = codegen::Function::build(&leaf.space);
                let event = TreeEvent::Evaluation {
                    actions: Sequence::Vec(leaf.actions.iter().cloned().collect()),
                    score: 1.,
                };
                unwrap!(record_writer.write_record(&unwrap!(bincode::serialize(&event))));
                unwrap!(
                    context.evaluate(&device_fn, device::EvalMode::FindBest),
                    "evaluation failed for kernel {}, with actions {:?}",
                    Self::name(),
                    leaf.actions
                );
                if let Err(err) = kernel.check_result(&expected_output, context) {
                    panic!(
                        "incorrect output for kernel {}, with actions {:?}: {}",
                        Self::name(),
                        leaf.actions,
                        err
                    )
                }
                num_runs += 1;
            } else {
                num_deadends += 1;
                if num_deadends as f32 / ((1 + num_deadends + num_runs) as f32)
                    >= MAX_DEADEND_RATIO
                {
                    panic!("too many dead-ends for kernel {}, {} deadends for {} successful runs", Self::name(), num_deadends, num_runs)
                }
            }
        }
        println!("{} deadends encountered", num_deadends);
        unwrap!(unwrap!(record_writer.finish_box()).flush());
    }

    fn find_cut_depth<AM>(params: Self::Parameters, cut: f64, context: &mut AM)
    where
        AM: device::ArgMap + device::Context + 'a,
    {
        let kernel;
        let signature = {
            let mut builder = SignatureBuilder::new(Self::name(), context);
            builder.set_random_fill(true);
            kernel = Self::build_signature(params, &mut builder);
            builder.get()
        };
        //let ordering = explorer::config::ChoiceOrdering::default();
        //let ordering = vec![ChoiceGroup::,];
        let ordering = ChoiceOrdering::new(vec![
            ChoiceGroup::DimKind,
            ChoiceGroup::Size,
            ChoiceGroup::LowerLayout,
            ChoiceGroup::InstFlag,
            ChoiceGroup::Order,
            ChoiceGroup::MemSpace,
            ChoiceGroup::DimMap,
        ]);
        let candidates = kernel.build_body(&signature, context);
        let depth_opt =
            local_selection::parallel_first_cut(&ordering, context, candidates, cut);
        println!("KERNEL {}", Self::name());
        if let Some(depth) = depth_opt {
            println!("Possibility of cut at depth {} for cut {}", depth, cut);
        } else {
            println!("cut too low, no suitable candidate found");
        }
    }

    /// Tests the correctness of the bound of kernels and returns the list of tested leafs
    /// along with the actual evaluation time.
    fn test_bound<AM>(
        params: Self::Parameters,
        num_tests: usize,
        random_fill: bool,
        context: &mut AM,
    ) -> Vec<BoundSample>
    where
        AM: device::ArgMap + device::Context + 'a,
    {
        let kernel;
        let signature = {
            let mut builder = SignatureBuilder::new(Self::name(), context);
            builder.set_random_fill(random_fill);
            kernel = Self::build_signature(params, &mut builder);
            builder.get()
        };
        let candidates = kernel.build_body(&signature, context);
        let leaves = Mutex::new(Vec::new());
        let num_tested = atomic::AtomicUsize::new(0);
        context.async_eval(
            num_cpus::get(),
            device::EvalMode::TestBound,
            &|evaluator| loop {
                if num_tested.fetch_add(1, atomic::Ordering::SeqCst) >= num_tests {
                    if num_tested.fetch_sub(1, atomic::Ordering::SeqCst) > num_tests {
                        break;
                    }
                }
                if let Some((leaf, bounds)) = descend_check_bounds(&candidates, context) {
                    let leaves = &leaves;
                    evaluator.add_kernel(
                        leaf,
                        (move |leaf: Candidate, runtime: f64| {
                            let bound = leaf.bound.clone();
                            let mut leaves = unwrap!(leaves.lock());
                            let mut actions = leaf.actions.iter().cloned().collect_vec();
                            actions.reverse();
                            for (idx, partial_bound) in bounds.iter().enumerate() {
                                assert!(
                                    partial_bound.value() <= bound.value() * 1.01,
                                    "invalid inner bound: {} < {}, kernel {}, \
                                     actions {:?} then {:?}",
                                    partial_bound,
                                    bound,
                                    Self::name(),
                                    &actions[..idx],
                                    &actions[idx..]
                                );
                            }
                            info!("new evaluation: {:.2e}ns, bound {}", runtime, bound);
                            leaves.push(BoundSample {
                                actions,
                                bound,
                                runtime,
                            });
                        }).into(),
                    );
                } else {
                    num_tested.fetch_sub(1, atomic::Ordering::SeqCst);
                }
            },
        );
        unwrap!(leaves.into_inner())
    }

    /// Runs the search and benchmarks the resulting candidate.
    fn benchmark<AM>(
        config: &explorer::Config,
        params: Self::Parameters,
        num_samples: usize,
        random_fill: bool,
        context: &mut AM,
    ) -> Vec<f64>
    where
        AM: device::ArgMap + device::Context + 'a,
    {
        let kernel;
        let signature = {
            let mut builder = SignatureBuilder::new(Self::name(), context);
            builder.set_random_fill(random_fill);
            kernel = Self::build_signature(params, &mut builder);
            builder.get()
        };
        let search_space = kernel.build_body(&signature, context);
        let best = unwrap!(
            explorer::find_best_ex(config, context, search_space),
            "no candidates found for kernel {}",
            Self::name()
        );
        let best_fn = codegen::Function::build(&best.space);
        context.benchmark(&best_fn, num_samples)
    }

    /// Computes the probability of encountering a dead-end when descending in the search
    /// tree.
    fn deadend_ratio<AM>(
        params: Self::Parameters,
        num_samples: usize,
        context: &mut AM,
    ) -> f64
    where
        AM: device::ArgMap + device::Context + 'a,
    {
        let kernel;
        let signature = {
            let mut builder = SignatureBuilder::new(Self::name(), context);
            kernel = Self::build_signature(params, &mut builder);
            builder.get()
        };
        let candidates = kernel.build_body(&signature, context);
        let num_deadends = (0..num_samples)
            .into_par_iter()
            .filter(|_| {
                let order = explorer::config::NewNodeOrder::WeightedRandom;
                let ordering = explorer::config::ChoiceOrdering::default();
                let inf = std::f64::INFINITY;
                let bounds = candidates.iter().map(|c| c.bound.value()).enumerate();
                let candidate_idx = local_selection::pick_index(order, bounds, inf);
                let candidate = candidates[unwrap!(candidate_idx)].clone();
                local_selection::descend(&ordering, order, context, candidate, inf)
                    .is_none()
            }).count();
        num_deadends as f64 / num_samples as f64
    }
}

/// Descend along a path in the search tree and stores the bounds encountered on the way.
fn descend_check_bounds<'a>(
    candidates: &[Candidate<'a>],
    context: &device::Context,
) -> Option<(Candidate<'a>, Vec<Bound>)> {
    let order = explorer::config::NewNodeOrder::WeightedRandom;
    let mut candidates = std::borrow::Cow::Borrowed(candidates);
    let mut bounds = Vec::new();
    loop {
        let idx = if let Some(idx) = {
            let idx_bounds = candidates.iter().map(|c| c.bound.value()).enumerate();
            local_selection::pick_index(order, idx_bounds, CUT)
        } {
            idx
        } else {
            return None;
        };
        bounds.push(candidates[idx].bound.clone());
        let choice_opt = explorer::choice::default_list(&candidates[idx].space).next();
        if let Some(choice) = choice_opt {
            let new_nodes = candidates[idx]
                .apply_choice(context, choice)
                .into_iter()
                .filter(|x| x.bound.value() < CUT)
                .collect_vec();
            candidates = std::borrow::Cow::Owned(new_nodes);
        } else {
            return Some((candidates[idx].clone(), bounds));
        }
    }
}

/// A sample of the accuracy of bounds.
pub struct BoundSample {
    actions: Vec<explorer::choice::ActionEx>,
    bound: Bound,
    runtime: f64,
}

impl BoundSample {
    /// Returns the ratio between the bound and the actual evaluation.
    fn ratio(&self) -> f64 {
        self.runtime / self.bound.value()
    }
}

impl std::fmt::Display for BoundSample {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{:.2}x, {:.2e}ns vs {}, for actions {:?}",
            self.ratio(),
            self.runtime,
            self.bound,
            self.actions
        )
    }
}

/// Prints an analysis of the bounds computed by the lower bound model.
pub fn analyze_bounds(mut bounds: Vec<BoundSample>) {
    const NUM_QUANTILES: usize = 5;
    bounds.sort_by(|x, y| cmp_f64(x.ratio(), y.ratio()));
    let num_errors = bounds.iter().take_while(|b| b.ratio() < 1.).count();
    if num_errors > 0 {
        let error_ratio = num_errors as f64 / bounds.len() as f64;
        let error_ratio = statistics::estimate_ratio(error_ratio, bounds.len());
        println!("ratio of errors {}, for example: ", error_ratio);
        let num_printed = std::cmp::min(NUM_QUANTILES, num_errors);
        for i in 0..num_printed {
            let index = i * num_errors / num_printed;
            println!("{}% worst error: {}", i * 100 / num_printed, bounds[index]);
        }
    }
    if num_errors < bounds.len() {
        let num_bounds = bounds.len() - num_errors;
        let num_quantiles = std::cmp::min(NUM_QUANTILES, num_bounds);
        for i in 0..num_quantiles {
            let index = (i + 1) * (num_bounds / num_quantiles) - 1;
            println!(
                "{}% worst: {}",
                (i + 1) * 100 / num_quantiles,
                bounds[num_errors + index]
            );
        }
    }
}

fn init_eventlog(config: &Config) -> io::Result<Box<dyn RecordWriter<Writer = File>>> {
    let path = Path::new(&config.event_log);
    let raw_file = File::create(&path)?;
    Ok(match path.extension().and_then(OsStr::to_str) {
        Some("gz") => Box::new(GzEncoder::new(raw_file, Compression::default())),
        Some("zz") => Box::new(ZlibEncoder::new(raw_file, Compression::default())),
        _ => Box::new(raw_file),
    })
}

fn init_log(config: &Config) -> io::Result<BufWriter<File>> {
    let mut output_file = File::create(&config.log_file)?;
    writeln!(output_file, "LOGGER\n{}", config)?;
    Ok(BufWriter::new(output_file))
}
