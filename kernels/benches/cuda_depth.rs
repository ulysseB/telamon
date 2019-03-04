use telamon_cuda as cuda;
use telamon_kernels::{linalg, Kernel};

fn main() {
    env_logger::init();
    let executor = &cuda::Executor::init();
    let params = linalg::MatMulP::new(1024, 1024, 1024);
    benchmark::<linalg::MatMul<f32>>(params, 500, executor);
}

fn benchmark<'a, K: Kernel<'a>>(
    params: K::Parameters,
    num_sample: usize,
    executor: &'a cuda::Executor,
) {
    let mut context = cuda::Context::new(executor);
    let estimate = K::path_depth(params, num_sample, &mut context);
    println!("{} path depth: {}", K::name(), estimate);
}
