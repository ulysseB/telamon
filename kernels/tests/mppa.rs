#![cfg(feature = "mppa")]
use telamon_mppa as mppa;
use telamon_kernels::{linalg, Kernel};

macro_rules! test_output {
    ($name:ident, $kernel:ty, $num_tests:expr, $params:expr) => {
        #[test]
        fn $name() {
            let _ = env_logger::try_init();
            let mut context = mppa::Context::new();
            <$kernel>::test_correctness($params, $num_tests, &mut context);
        }
    };
}

test_output!(axpy, linalg::Axpy<f32>, 100, (1 << 16, true));
test_output!(mv, linalg::MatVec<f32>, 100, (1 << 4, 1 << 2, true));
test_output!(gesummv, linalg::Gesummv<f32>, 100, (1 << 4, 1 << 4, true));
test_output!(
    matmul,
    linalg::MatMul<f32>,
    100,
    linalg::MatMulP::new(16, 16, 16)
);
