//! Benchmarks the accuracy of bounds on CUDA GPUs.
use telamon::helper::MemInit;
use telamon_cuda as cuda;
use telamon_kernels::{analyze_bounds, linalg, Kernel};

fn main() {
    env_logger::init();
    let executor = &cuda::Executor::init();
    /*benchmark::<linalg::Axpy<f32>>((1 << 25, true), 500, executor);
    benchmark::<linalg::MatVec<f32>>((1 << 13, 1 << 13, true), 500, executor);
    benchmark::<linalg::Gesummv<f32>>((1 << 13, 1 << 13, true), 500, executor);*/
    let params = linalg::MatMulP::new(1024, 1024, 1024);
    benchmark::<linalg::MatMul<f32>>(params, 500, executor);
    let params = linalg::MatMulP::new(128, 1024, 1024).transpose_b();
    benchmark::<linalg::MatMul<f32>>(params, 500, executor);
    let params = linalg::BatchMMP::new(500, 26, 26, 72)
        .transpose_b()
        .static_sizes();
    benchmark::<linalg::BatchMM<f32>>(params, 500, executor);
    let params = linalg::BatchMMP::new(512, 32, 32, 64).static_sizes();
    benchmark::<linalg::BatchMM<f32>>(params, 500, executor);
}

fn benchmark<'a, K>(params: K::Parameters, num_runs: usize, executor: &'a cuda::Executor)
where
    K: Kernel<'a>,
{
    let mut context = cuda::Context::new(executor);
    let bounds = K::test_bound(params, num_runs, MemInit::RandomFill, &mut context);
    println!("bounds for kernel {}", K::name());
    analyze_bounds(bounds);
}
