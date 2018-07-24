//! Benchmarks the variance of measures on the GPU.
extern crate env_logger;
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
const NUM_TESTS: usize = 10;
/// The number of time each candidate should be evaluated, in cpu-bound mode.
const NUM_SLOW_SAMPLES: usize = 10;
/// The number of time each candidate should be evaluated, in gpu-bound mode.
const NUM_FAST_SAMPLES: usize = 50;
/// The time to wait between two slow evaluations.
const SLEEP_TIME: Duration = Duration::from_millis(500);
/// The maximal bound for kernels to consider.
const CUT: f64 = 1e8;

fn main() {
    env_logger::init();
    let executor = cuda::Executor::init();
    let small_malmul = linalg::MatMulP::new(128, 128, 128);
    let big_malmul = linalg::MatMulP::new(1024, 1024, 1024);
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
    context.async_eval(1, device::EvalMode::FindBest, &|evaluator| {
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
            unwrap!(context.evaluate(&code, device::EvalMode::FindBest))
        }).collect_vec()
    }).collect_vec();
    info!("Evaluation finished, analysing results");
    let results = candidates.iter().zip_eq(slow_evals.into_iter().zip_eq(fast_evals));
    for (_, (cpu_bound_times, gpu_bound_times)) in results {
        println!("-------");
        analyse_runtimes(cpu_bound_times);
        analyse_runtimes(unwrap!(gpu_bound_times.into_inner()));
    }
}

/// Analyses the regularity of execution times.
fn analyse_runtimes(mut runtimes: Vec<f64>) {
    runtimes.sort_by(|&x, &y| cmp_f64(x, y));
    let len = runtimes.len();
    println!("[{}, {}, {}]", runtimes[0], runtimes[len/2], runtimes[len-1]);

}
