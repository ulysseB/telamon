//! Benchmarks the variance of measures on the GPU.
extern crate env_logger;
extern crate futures;
extern crate itertools;
#[macro_use]
extern crate log;
extern crate telamon;
extern crate telamon_kernels;
#[macro_use]
extern crate telamon_utils as utils;

use futures::Future;
use futures::sync::oneshot;
use itertools::Itertools;
use telamon::{device, explorer, helper};
use telamon::device::{cuda, Context};
use telamon::explorer::local_selection;
use telamon_kernels::{linalg, Kernel};
use std::sync::Mutex;
use std::time::Duration;
use utils::*;

/// The number of candidates to try.
const NUM_TESTS: usize = 20;
/// The number of time each candidate should be evaluated, in cpu-bound mode.
const NUM_SLOW_SAMPLES: usize = 20;
/// The number of time each candidate should be evaluated, in gpu-bound mode.
const NUM_FAST_SAMPLES: usize = 20;
/// Maximal legal relative difference between evaluation times.
const MAX_RELATIVE_DIFF: f64 = 0.01;
/// The time to wait between two slow evaluations.
const SLEEP_TIME: Duration = Duration::from_millis(500);
/// The maximal bound for kernels to consider.
const CUT: f64 = 1e8;

fn main() {
    env_logger::init();
    let executor = cuda::Executor::init();
    let small_malmul = linalg::MatMulP::new(128, 128, 128);
    let big_malmul = linalg::MatMulP::new(256, 256, 256);
    accuracy::<linalg::MatMul<f32>>(&small_malmul, "matmul_128", &executor);
    accuracy::<linalg::MatMul<f32>>(&big_malmul, "matmul_256", &executor);
}

fn accuracy<'a, K>(params: &K::Parameters,
                   name: &str,
                   executor: &'a cuda::Executor) where K: Kernel<'a> {
    let mut context = cuda::Context::new(executor);
    info!("Generating {} candidates", NUM_TESTS);
    let (kernel, signature) = {
        let mut builder =helper:: SignatureBuilder::new(name, &mut context);
        let kernel = K::build_signature(params.clone(), &mut builder);
        (kernel, builder.get())
    };
    let candidates = kernel.build_body(&signature, &context);
    let candidates = std::iter::repeat(()).flat_map(|()| {
         let order = explorer::config::NewNodeOrder::WeightedRandom;
         let bounds = candidates.iter().map(|c| c.bound.value()).enumerate();
         let candidate_idx = local_selection::pick_index(order, bounds, CUT);
         let candidate = candidates[unwrap!(candidate_idx)].clone();
         local_selection::descend(order, &context, candidate, CUT)
    }).take(NUM_TESTS).collect_vec();
    info!("Evaluating candidates, simulating a GPU-bound exploration");
    let fast_evals = run_evaluations(&candidates, &context, NUM_FAST_SAMPLES, None);
    info!("Evaluating candidates, simulating a CPU-bound exploration");
    let sleep = Some(SLEEP_TIME);
    let slow_evals = run_evaluations(&candidates, &context, NUM_SLOW_SAMPLES, sleep);
    info!("Evaluation finished, analysing results");
    let results = candidates.iter().zip_eq(slow_evals.into_iter().zip_eq(fast_evals));
    let mut all_diffs = 0.;
    let mut all_cpu_gpu_diffs = 0.;
    for (_, (cpu_bound_times, gpu_bound_times)) in results {
        let (cpu_med, cpu_diff) = analyse_runtimes(cpu_bound_times, name, "cpu");
        let (gpu_med, gpu_diff) = analyse_runtimes(gpu_bound_times, name, "gpu");
        all_diffs += cpu_diff;
        all_diffs += gpu_diff;
        let diff_factor = 2.*(cpu_med - gpu_med).abs()/(cpu_med + gpu_med);
        all_cpu_gpu_diffs += diff_factor;
        if diff_factor > MAX_RELATIVE_DIFF {
            println!("annormal difference between cpu-bound {:.2e} and gpu-bound {:.2e} \
                      evaluations ({:.2e}x) in {}", cpu_med, gpu_med, diff_factor, name)
        }
    }
    let diff = all_diffs / (2 * NUM_TESTS) as f64; 
    let cpu_gpu_diff = all_cpu_gpu_diffs / NUM_TESTS as f64;
    println!("average delta between the fastest and slowest evaluation: {:.2e}x", diff);
    println!("average delta cpu-bound and gpu-bound evaluations: {:.2e}x", cpu_gpu_diff);
}

/// Analyses the regularity of execution times and returns the median and the relative
/// difference between the fastest and the slowest evaluation. Fails if the relative
/// difference is too big.
fn analyse_runtimes(mut runtimes: Vec<f64>, name: &str, bound: &str) -> (f64, f64) {
    runtimes.sort_by(|&x, &y| cmp_f64(x, y));
    let median = runtimes[runtimes.len()/2];
    let diff = (runtimes[runtimes.len()-1] - runtimes[0])/median;
    if diff > MAX_RELATIVE_DIFF {
        let min = 1. - runtimes[0]/median;
        let max = runtimes[runtimes.len()-1]/median - 1.;
        println!("noisy {}-bound evaluations {:.2e}ns (-{:.2e}x, +{:.2e}x) in {}",
                  bound, median, min, max, name);
    }
    (median, diff)
}

/// Runs a series of evaluation, sleeping the given duration betwen each evaluation.
fn run_evaluations(candidates: &[explorer::Candidate],
                   context: &Context,
                   num_samples: usize,
                   sleep: Option<Duration>) -> Vec<Vec<f64>> {
    let runtimes = candidates.iter().map(|_| Mutex::new(vec![])).collect_vec();
    context.async_eval(1, device::EvalMode::TestEval, &|evaluator| {
        for (candidate, results) in candidates.iter().zip_eq(&runtimes) {
            for _ in 0..num_samples {
                if let Some(duration) = sleep {
                    let (sender, waiter) = oneshot::channel();
                    evaluator.add_kernel(candidate.clone(), (move |_, runtime| {
                        unwrap!(sender.send(()));
                        unwrap!(results.lock()).push(runtime);
                    }).into());
                    unwrap!(waiter.wait());
                    std::thread::sleep(duration);
                } else {
                    evaluator.add_kernel(candidate.clone(), (move |_, runtime| {
                        unwrap!(results.lock()).push(runtime);
                    }).into());
                }
            }
        }
    });
    runtimes.into_iter().map(|lock| unwrap!(lock.into_inner())).collect()
}
