#![cfg(feature="cuda")]
extern crate env_logger;
extern crate telamon;
extern crate telamon_kernels;

use telamon::device::cuda;
use telamon_kernels::{Kernel, linalg};

macro_rules! test_output {
    ($name:ident, $kernel:ty, $num_tests:expr, $params:expr) => {
        #[test]
        fn $name() {
            let _ = env_logger::init();
            let executor = cuda::Executor::init();
            let mut context = cuda::Context::new(&executor);
            <$kernel>::test_correctness($params, $num_tests, &mut context);
        }
    }

    // FIXME: generate both fast and slow tests with the same macro
}

test_output!(axpy_output, linalg::Axpy<f32>, 1000, 1 << 26);
