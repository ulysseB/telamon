//! Benchmarks the accuracy of bounds on CUDA GPUs.
extern crate env_logger;
extern crate telamon;
extern crate telamon_kernels;

use telamon::device::cuda;
use telamon_kernels::{linalg, Kernel};

fn main() {
    env_logger::init();
    let executor = &cuda::Executor::init();
    let params = linalg::MatMulP::new(1024, 1024, 1024);
    find_cut::<linalg::MatMul<f32>>(params, 500, executor);
    let params = linalg::MatMulP::new(128, 1024, 1024).transpose_b();
    find_cut::<linalg::MatMul<f32>>(params, 500, executor);
    let params = linalg::BatchMMP::new(500, 26, 26, 72)
        .transpose_b()
        .static_sizes();
    find_cut::<linalg::BatchMM<f32>>(params, 500, executor);
    let params = linalg::BatchMMP::new(512, 32, 32, 64).static_sizes();
    find_cut::<linalg::BatchMM<f32>>(params, 500, executor);
}

fn find_cut<'a, K>(params: K::Parameters, num_runs: usize, executor: &'a cuda::Executor)
where
    K: Kernel<'a>,
{
    let mut context = cuda::Context::new(executor);
    K::find_cut_depth(params, 1.0e8, &mut context);
}
