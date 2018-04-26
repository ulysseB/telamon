//! Benchmarks the accuracy of bounds on CUDA GPUs.
extern crate env_logger;
extern crate telamon;
extern crate telamon_kernels;

use telamon::device::cuda;
use telamon_kernels::{Kernel, analyze_bounds, linalg};

fn main() {
    env_logger::init();
    let executor = &cuda::Executor::init();
    //benchmark::<linalg::Axpy<f32>>(1<<25, 500, executor);
    //benchmark::<linalg::MatVec<f32>>((1<<13, 1<<13), 500, executor);
    //benchmark::<linalg::Gesummv<f32>>((1<<13, 1<<13), 500, executor);
    benchmark::<linalg::MatMul<f32>>((1<<10, 1<<10, 1<<10), 500, executor);
    //benchmark::<linalg::Doitgen<f32>>((1<<7, 1<<7, 1<<7), 500, executor);
}

fn benchmark<'a, K>(params: K::Parameters,
                    num_runs: usize,
                    executor: &'a cuda::Executor) where K: Kernel<'a> {
    let mut context = cuda::Context::new(executor);
    let bounds = K::test_bound(params, num_runs, &mut context);
    println!("bounds for kernel {}", K::name());
    analyze_bounds(bounds);
}
