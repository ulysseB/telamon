use telamon_kernels::{linalg, Kernel};
use telamon_x86 as x86;

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

macro_rules! test_dump {
    ($name:ident, $kernel:ty, $params:expr) => {
        #[test]
        fn $name() {
            let _ = env_logger::try_init();
            let mut context = x86::Context::default();
            let path = format!("log_to_test/{}.log", stringify!($name));
            <$kernel>::execute_log($params, &mut context, &path);
        }
    };
}

test_output!(correct_axpy, linalg::Axpy<f32>, 100, (1 << 16, true));
test_output!(correct_mv, linalg::MatVec<f32>, 100, (1 << 4, 1 << 2, true));
test_output!(
    correct_gesummv,
    linalg::Gesummv<f32>,
    100,
    (1 << 4, 1 << 4, true)
);
test_output!(
    test_matmul,
    linalg::MatMul<f32>,
    100,
    linalg::MatMulP::new(16, 16, 16)
);

test_dump!(axpy, linalg::Axpy<f32>, (1 << 16, true));
test_dump!(mv, linalg::MatVec<f32>, (1 << 4, 1 << 2, true));
test_dump!(gesummv, linalg::Gesummv<f32>, (1 << 4, 1 << 4, true));
test_dump!(
    matmul,
    linalg::MatMul<f32>,
    linalg::MatMulP::new(16, 16, 16)
);
