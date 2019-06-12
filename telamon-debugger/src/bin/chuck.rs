use std::sync::Arc;

use telamon::device::Context;
use telamon::explorer;
use telamon::ir;
use telamon::model::bound;
use telamon::search_space::{Action, DimKind};
use telamon_kernels::{linalg, Kernel, KernelBuilder};

fn main() {
    env_logger::init();

    let executor = telamon_cuda::Executor::init();
    let mut context = telamon_cuda::Context::new(&executor);
    let params = linalg::FusedMMP::new(256, 256, 256);

    let (signature, kernel, context) = KernelBuilder::default()
        .build::<linalg::FusedMM<f32>, telamon_cuda::Context>(
            params.clone(),
            &mut context,
        );
    let signature = Arc::new(signature);

    let stabilizer = &context.stabilizer();
    let mut config = explorer::Config::default();
    config.output_dir = "/tmp".to_string();
    config.max_evaluations = Some(10);

    let mut candidate = kernel
        .build_body(signature, context)
        .swap_remove(0)
        .space
        .prioritized();
    candidate
        .apply_decisions(vec![
            Action::DimKind(ir::DimId(11), DimKind::UNROLL),
            Action::DimKind(ir::DimId(1), DimKind::LOOP),
        ])
        .unwrap();

    let bounds = (0..100)
        .map(|_| bound(&candidate, context))
        .collect::<Vec<_>>();
    println!("Bounds: {:?}", bounds.len());
    // println!("Expr: {}", bounds[0].lol);
}
