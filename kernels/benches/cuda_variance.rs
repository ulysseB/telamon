//! Benchmarks the variance of measures on the GPU.
extern crate env_logger;
#[macro_use]
extern crate failure;
extern crate itertools;
#[macro_use]
extern crate log;
extern crate telamon;
extern crate telamon_kernels;
#[macro_use]
extern crate telamon_utils as utils;

use itertools::Itertools;
use telamon::{codegen, device, explorer, helper};
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
const NUM_FAST_SAMPLES: usize = 50;
/// Maximal legal difference between evaluation times.
const MAX_DIFF: f64 = 0.01;
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
    accuracy::<linalg::MatMul<f32>>(&big_malmul, "matmul_1024", &executor);
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
    let fast_evals = candidates.iter().map(|_| Mutex::new(vec![])).collect_vec();
    context.async_eval(1, device::EvalMode::TestEval, &|evaluator| {
        for (candidate, results) in candidates.iter().zip_eq(&fast_evals) {
            for _ in 0..NUM_FAST_SAMPLES {
                evaluator.add_kernel(candidate.clone(), (move |_, runtime| {
                    unwrap!(results.lock()).push(runtime);
                }).into());
            }
        }
    });
    info!("Evaluating candidates, simulating a CPU-bound exploration");
    let slow_evals = candidates.iter().map(|candidate| {
        let code = codegen::Function::build(&candidate.space);
        (0..NUM_SLOW_SAMPLES).map(|_| {
            std::thread::sleep(SLEEP_TIME);
            unwrap!(context.evaluate(&code, device::EvalMode::TestEval))
        }).collect_vec()
    }).collect_vec();
    info!("Evaluation finished, analysing results");
    let results = candidates.iter().zip_eq(slow_evals.into_iter().zip_eq(fast_evals));
    for (_, (cpu_bound_times, gpu_bound_times)) in results {
        let cpu = analyse_runtimes(cpu_bound_times).unwrap_or_else(|err| {
            println!("{} in cpu-bound {}", err, name);
            err.median
        });
        let gpu_bound_times = unwrap!(gpu_bound_times.into_inner());
        let gpu = analyse_runtimes(gpu_bound_times).unwrap_or_else(|err| {
            println!("{} in cpu-bound {}", err, name);
            err.median
        });
        let diff_factor = 2.*(cpu-gpu).abs()/(cpu+gpu);
        if diff_factor > MAX_DIFF {
            println!("annormal difference between cpu-bound {:.2e} and gpu-bound {:.2e} \
                      evaluations ({:.2e} factor) in {}", cpu, gpu, diff_factor, name)
        }
    }
}

/// Analyses the regularity of execution times and returns the median. Fails if the
/// relative difference between the fastest and the slowest evaluation is too big.
fn analyse_runtimes(mut runtimes: Vec<f64>) -> Result<f64, NoisyError> {
    runtimes.sort_by(|&x, &y| cmp_f64(x, y));
    let median = runtimes[runtimes.len()/2];
    let diff = (runtimes[runtimes.len()-1] - runtimes[0])/median;
    if diff > MAX_DIFF {
        return Err(NoisyError {
            median,
            min: 1. - runtimes[0]/median,
            max: runtimes[runtimes.len()-1]/median - 1.,
        });
    }
    Ok(median)
}

/// Error raised when the fastest and slowest evaluation are too far appart.
#[derive(Debug, Fail)]
#[fail(display="noisy evaluations ({:.2e}ns -{:.2e}, +{:.2e}%)", median, min, max)]
struct NoisyError { median: f64, min: f64, max: f64 }
