use telamon_kernels::{linalg, Kernel};
use telamon_x86 as x86;

macro_rules! test_dump {
    ($name:ident, $kernel:ty, $params:expr) => {
        #[test]
        fn $name() {
            let _ = env_logger::try_init();
            let mut context = x86::Context::default();
            let path = format!("kernel_dump/x86/{}.json", stringify!($name));
            let mut file = std::fs::File::open(&path).unwrap();
            <$kernel>::execute_dump(&mut context, &mut file);
        }
    };
}

test_dump!(axpy, linalg::Axpy<f32>, (1 << 16, true));
test_dump!(mv, linalg::MatVec<f32>, (1 << 4, 1 << 2, true));
test_dump!(gesummv, linalg::Gesummv<f32>, (1 << 4, 1 << 4, true));
test_dump!(
    matmul,
    linalg::FusedMM<f32>,
    linalg::FusedMMP::new(16, 16, 16)
);
