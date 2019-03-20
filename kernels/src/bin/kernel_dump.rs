use telamon_kernels::{linalg, Kernel};
use telamon_x86 as x86;

macro_rules! kernel_dump {
    ($kernel:ty, $num_tests:expr, $params:expr) => {
        let _ = env_logger::try_init();
        let mut context = x86::Context::default();
        <$kernel>::generate_dump($params, &mut context);
    };
}

fn main() {
    kernel_dump!(linalg::Axpy<f32>, 100, (1 << 16, true));
    kernel_dump!(linalg::MatVec<f32>, 100, (1 << 4, 1 << 2, true));
    kernel_dump!(linalg::Gesummv<f32>, 100, (1 << 4, 1 << 4, true));
    kernel_dump!(linalg::MatMul<f32>, 100, linalg::MatMulP::new(16, 16, 16));
}
