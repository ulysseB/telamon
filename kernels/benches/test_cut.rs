//! Benchmarks the accuracy of bounds on CUDA GPUs.
extern crate env_logger;
extern crate serde_json;
extern crate telamon;
extern crate telamon_kernels;
extern crate telamon_utils;
extern crate xdg;

use telamon::device::{cuda, fake::FakeContext};
use telamon_kernels::{linalg, Kernel};
use telamon_utils::*;

/// Returns the name of the configuration file.
pub fn get_config_path() -> std::path::PathBuf {
    let xdg_dirs = unwrap!(xdg::BaseDirectories::with_prefix("telamon"));
    // We use `place_config_file` instead of `find_config_file` to avoid returning
    // a system-wide files (e.g. in /usr/share/telamon/cuda_gpus.json) on which the user
    // doesn't have write permissions.
    let path = xdg_dirs.place_config_file("cuda_gpus.json");
    unwrap!(path, "cannot create configuration directory")
}

macro_rules! test_cut {
    ($name:ident, $kernel:ty, $params:expr) => {
        fn $name() {
            let _ = env_logger::try_init();
            //let gpu = unwrap!(cuda::characterize::get_gpu_desc_from_file());
            let gpu: cuda::Gpu = unwrap!(serde_json::from_reader(&unwrap!(std::fs::File::open(get_config_path()))));
            let mut context = FakeContext::new(gpu);
            <$kernel>::find_cut_depth($params, 4.0e4, &mut context);
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
