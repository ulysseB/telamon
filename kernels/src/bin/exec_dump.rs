use std::env;
use std::io::Read;
use telamon::device::{ArgMap, Context};
use telamon_kernels::{linalg, Kernel};

fn dispatch_exec<'a, C: Context + ArgMap<'a>, R: Read>(
    file: &mut R,
    kernel_name: &str,
    context: &mut C,
) {
    match kernel_name as &str {
        "axpy" => linalg::Axpy::<f32>::execute_dump(context, file),
        "mv" => linalg::MatVec::<f32>::execute_dump(context, file),
        "gesummv" => linalg::Gesummv::<f32>::execute_dump(context, file),
        "matmul" => linalg::FusedMM::<f32>::execute_dump(context, file),
        _ => panic!("Valid kernel names are: axpy, mv, gesummv, matmul"),
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    assert!(args.len() == 3);
    let kernel_name = &args[1];
    let binary_path = &args[2];

    let _ = env_logger::try_init();
    let mut file = std::fs::File::open(binary_path).expect("Invalid dump path");
    #[cfg(feature = "mppa")]
    {
        let mut context = telamon_mppa::Context::default();
        dispatch_exec(&mut file, kernel_name, &mut context);
    }
    #[cfg(not(feature = "mppa"))]
    {
        let mut context = telamon_x86::Context::default();
        dispatch_exec(&mut file, kernel_name, &mut context);
    }
}
