//! Measures the amount of deadends in the search space.
extern crate env_logger;
extern crate telamon;
extern crate telamon_kernels;

use telamon::device::cuda;
use telamon_kernels::{Kernel, linalg, statistics};

fn main() {
    env_logger::init();
    let executor = &cuda::Executor::init();
    benchmark::<linalg::Axpy<f32>>(1<<25, 1000, executor);
    benchmark::<linalg::MatVec<f32>>((1<<13, 1<<13), 1000, executor);
    benchmark::<linalg::Gesummv<f32>>((1<<13, 1<<13), 1000, executor);
    benchmark::<linalg::MatMul<f32>>((1<<10, 1<<10, 1<<10), 500, executor);
    benchmark::<linalg::Doitgen<f32>>((1<<7, 1<<7, 1<<7), 500, executor);
}

fn benchmark<'a, K: Kernel<'a>>(params: K::Parameters,
                                num_sample: usize,
                                executor: &'a cuda::Executor) {
    let mut context = cuda::Context::new(executor);
    let ratio = K::deadend_ratio(params, num_sample, &mut context);
    let estimate = statistics::estimate_ratio(ratio, num_sample);
    println!("{} deadend ratio: {}", K::name(), estimate);
}
