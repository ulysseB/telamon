use telamon_kernels::{linalg, Kernel};
use std::env;
use telamon_x86 as x86;

macro_rules! dispatch_kernel {
    ($name:expr, $func_call:ident, $( $arg:expr),* ) => {
        match $name {
            "axpy" => linalg::Axpy::<f32>::$func_call($($arg),*),
            "mv" => linalg::MatVec::<f32>::$func_call($($arg),*),
            "gesummv" => linalg::Gesummv::<f32>::$func_call($($arg),*),
            "matmul" => linalg::MatMul::<f32>::$func_call($($arg),*),
            _ => panic!("Valid kernel names are: axpy, mv, gesummv, matmul"),
        }
    };
}

fn main() {
    let args: Vec<String> = env::args().collect();
    assert!(args.len() == 3);
    let kernel_name = &args[1];
    let binary_path = &args[2];


    let _ = env_logger::try_init();
    let mut context = x86::Context::default();

    let mut file = std::fs::File::open(binary_path).expect("Invalid dump path");
    dispatch_kernel!(kernel_name as &str, execute_dump, &mut context, &mut file);
}
