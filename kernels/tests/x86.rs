extern crate env_logger;
extern crate telamon;
extern crate telamon_kernels;

use telamon::device::x86;
use telamon_kernels::{linalg, Kernel};

macro_rules! test_output {
    ($name:ident, $kernel:ty, $num_tests:expr, $params:expr) => {
        #[test]
        fn $name() {
            let _ = env_logger::try_init();
            let mut context = x86::Context::default();
            <$kernel>::test_correctness($params, $num_tests, &mut context);
        }
    };
}

macro_rules! reroll_last {
    ($name:ident, $kernel:ty, $params:expr) => {
        fn $name() {
            let _ = env_logger::try_init();
            let mut context = x86::Context::default();
            <$kernel>::reroll_last_cand($params, &mut context);
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
reroll_last!(reroll_axpy, linalg::Axpy<f32>, (1 << 16, true));
reroll_last!(reroll_mv, linalg::MatVec<f32>, (1 << 4, 1 << 2, true));
reroll_last!(reroll_gesummv, linalg::Gesummv<f32>, (1 << 4, 1 << 4, true));
reroll_last!(
    reroll_matmul,
    linalg::MatMul<f32>,
    linalg::MatMulP::new(16, 16, 16)
);

fn main() {
    reroll_axpy();
}
