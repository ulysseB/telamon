//! Benchmarks the accuracy of bounds on CUDA GPUs.
extern crate env_logger;
extern crate telamon;
extern crate telamon_kernels;

use telamon::device::cuda;
use telamon_kernels::{linalg, Kernel};

macro_rules! test_cut {
    ($name:ident, $kernel:ty, $params:expr) => {
        fn $name() {
            let _ = env_logger::try_init();
            let executor = cuda::Executor::init();
            let mut context = cuda::Context::new(&executor);
            <$kernel>::find_cut_depth($params, 1.0e4, &mut context);
        }
    };
}

test_cut!(axpy, linalg::Axpy<f32>, (1 << 15, true));
test_cut!(mv, linalg::MatVec<f32>, (1 << 4, 1 << 2, true));
test_cut!(gesummv, linalg::Gesummv<f32>, (1 << 4, 1 << 4, true));
test_cut!(
    matmul,
    linalg::MatMul<f32>,
    linalg::MatMulP::new(16, 16, 16)
);

fn main() {
    axpy();
    mv();
    gesummv();
    matmul();
}
