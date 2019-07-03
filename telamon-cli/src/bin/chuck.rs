use std::fs;
use std::sync::Arc;

use telamon::device::Context;
use telamon::explorer::{self, choice::ActionEx as Action};
use telamon::ir;
use telamon::model::bound;
use telamon::search_space::{DimKind, NumDomain, NumericSet};
use telamon_kernels::{linalg, Kernel, KernelBuilder};

fn main() {
    env_logger::init();

    let executor = telamon_cuda::Executor::init();
    let mut context = telamon_cuda::Context::new(&executor);
    let params = linalg::FusedMMP::new(256, 256, 32);

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

    let mut candidate = kernel.build_body(signature, context).swap_remove(0).space;
    // .prioritized();

    let replay: Vec<_> =
        serde_json::from_reader(fs::File::open("/tmp/slow.json").unwrap()).unwrap();
    for action in &replay {
        match action {
            Action::Action(action) => candidate
                .apply_decisions(vec![action.clone()])
                .unwrap_or_else(|()| panic!("unable to apply decision {:?}", action)),
            Action::LowerLayout {
                mem,
                st_dims,
                ld_dims,
            } => candidate
                .lower_layout(*mem, st_dims, ld_dims)
                .expect("lower layout"),
        }
    }

    eprintln!("__STARTXX__");
    for _ in 0..10 {
        let start = std::time::Instant::now();
        let bound = bound(&candidate, context);
        let duration = start.elapsed();
        println!("Bound: {:?} (in {:?})", bound, duration);
    }

    /*
    let bounds = (0..100)
        .map(|_| bound(&candidate, context))
        .collect::<Vec<_>>();
        */
    //println!("Bounds: {:?}", bounds.len());
    // println!("Expr: {}", bounds[0].lol);
}
