extern crate env_logger;
extern crate telamon;
extern crate telamon_kernels;

use telamon::device::x86;
use telamon_kernels::{Kernel, linalg};

macro_rules! test_output {
    ($name:ident, $kernel:ty, $num_tests:expr, $params:expr) => {
        #[test]
        fn $name() {
            let _ = env_logger::try_init();
            let mut context = x86::Context::new();
            <$kernel>::test_correctness($params, $num_tests, &mut context);
        }
    }
}

test_output!(axpy, linalg::Axpy<f32>, 100, 1 << 16);
test_output!(mv, linalg::MatVec<f32>, 100, (1<<4, 1<<2));
//test_output!(gesummv, linalg::Gesummv<f32>, 100, (1<<4, 1<<4));
//test_output!(matmul, linalg::MatMul<f32>, 100, (1<<4, 1<<4, 1<<4));
//test_output!(doitgen, linalg::Doitgen<f32>, 100, (1<<4, 1<<4, 1<<4));
