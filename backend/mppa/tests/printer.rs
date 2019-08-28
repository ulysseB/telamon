use rpds::List;
use std::fs;
use telamon::codegen;
use telamon::explorer::choice::ActionEx;
use telamon_kernels::{linalg, Kernel, KernelBuilder};
use telamon_mppa as mppa;

#[test]
fn print_candidates() {
    env_logger::try_init().unwrap();
    let dump = fs::read_to_string("dumps/implementation_code.json").unwrap();

    let mut context = mppa::Context::new();
    let params = linalg::FusedMMP::new(16, 16, 16);

    let (signature, kernel, context) = KernelBuilder::new()
        .build::<linalg::FusedMM<f32>, mppa::Context>(params, &mut context);

    // build_body for the kernel always returns a single candidate
    let candidate = kernel.build_body(signature.into(), context).remove(0);

    let pairs: Vec<(List<ActionEx>, String)> = serde_json::from_str(&dump).unwrap();

    for (actions, expected_code) in pairs.iter() {
        let implementation = actions
            .iter()
            .fold(Ok(candidate.clone()), |accu_cand, action| {
                accu_cand?.apply_decision(context, action.clone())
            })
            .unwrap();

        let function = codegen::Function::build(&implementation.space);
        let mut namegen = mppa::NameGenerator::default();
        let mut name_map = codegen::NameMap::new(&function, &mut namegen);
        let mut printer = mppa::printer::MppaPrinter::default();

        let generated_code = printer.wrapper_function(&function, &mut name_map, 1);

        assert!(
            *expected_code == generated_code,
            "Expected code:\n{}\nBut generated code was:\n{}",
            expected_code,
            generated_code
        );
    }
}
