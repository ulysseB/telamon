//! Measures the amount of deadends in the search space.
extern crate env_logger;
extern crate telamon;
extern crate telamon_kernels;

use telamon::device::cuda;
use telamon_kernels::{linalg, statistics, Kernel};

fn main() {
    env_logger::init();
    let executor = &cuda::Executor::init();
    benchmark::<linalg::Axpy<f32>>((1 << 25, true), 1000, executor);
    benchmark::<linalg::MatVec<f32>>((1 << 13, 1 << 13, true), 1000, executor);
    benchmark::<linalg::Gesummv<f32>>((1 << 13, 1 << 13, true), 1000, executor);
    let params = linalg::MatMulP::new(1024, 1024, 1024);
    benchmark::<linalg::MatMul<f32>>(params, 500, executor);
}

fn benchmark<'a, K: Kernel<'a>>(
    params: K::Parameters,
    num_sample: usize,
    executor: &'a cuda::Executor,
)
{
    let mut context = cuda::Context::new(executor);
    let ratio = K::deadend_ratio(params, num_sample, &mut context);
    let estimate = statistics::estimate_ratio(ratio, num_sample);
    println!("{} deadend ratio: {}", K::name(), estimate);
}
