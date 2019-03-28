use std::env;
use telamon_kernels::{linalg, Kernel};
use telamon_x86 as x86;

fn main() {
    let args: Vec<String> = env::args().collect();
    assert!(args.len() == 3);
    let kernel_name = &args[1];
    let binary_path = &args[2];

    let _ = env_logger::try_init();
    let mut context = x86::Context::default();

    let mut file = std::fs::File::open(binary_path).expect("Invalid dump path");
    match kernel_name as &str {
        "axpy" => linalg::Axpy::<f32>::execute_dump(&mut context, &mut file),
        "mv" => linalg::MatVec::<f32>::execute_dump(&mut context, &mut file),
        "gesummv" => linalg::Gesummv::<f32>::execute_dump(&mut context, &mut file),
        "matmul" => linalg::MatMul::<f32>::execute_dump(&mut context, &mut file),
        _ => panic!("Valid kernel names are: axpy, mv, gesummv, matmul"),
    }
}
