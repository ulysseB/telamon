//! Tests the accuracy of the performance model. A pattern can be passed as argument to
//! specify the tests to run.
#![feature(stmt_expr_attributes)]
extern crate env_logger;
extern crate telamon;
extern crate regex;
#[macro_use]
extern crate log;
#[macro_use]
extern crate telamon_utils as utils;

mod latency;
mod memory;
mod tests;

use telamon::device::Context;
use telamon::helper;
use telamon::model::bound;
use telamon::search_space::Action;
use regex::Regex;

// TODO(test_model): RAM bandwidth
// TODO(test_model): L2 bandwidth
// TODO(test_model): L1 bandwidth
// TODO(test_model): memory instruction latency.

/// A tests of the accuracy of the performance model.
trait PerfModelTest {
    /// Returns the name of the tests.
    fn name() -> &'static str;

    /// Generates the base of the function to evaluate.
    fn gen_signature(builder: &mut helper::SignatureBuilder);

    /// Generates the function to evaluate.
    fn gen_function(builder: &mut helper::Builder) -> Self;

    /// Returns the actions to apply to the function to constrain it.
    fn get_actions(&self) -> Vec<Action> { vec![] }
}

/// Runs a test.
fn run<T: PerfModelTest>(pattern: &Regex) {
    if !pattern.is_match(T::name()) { return; }
    let executor = telamon::device::cuda::Executor::init();
    let mut context = telamon::device::cuda::Context::new(&executor);
    let base = {
        let mut base_builder = helper::SignatureBuilder::new(T::name(), &mut context);
        T::gen_signature(&mut base_builder);
        base_builder.get()
    };
    let mut builder = helper::Builder::new(&base, context.device());
    let state = T::gen_function(&mut builder);
    let actions = T::get_actions(&state);

    let mut early_model_perf = None;
    if actions.len() != 0 {
        early_model_perf = Some(bound(&builder.get_clone(), &context).value());
        for action in actions {
            builder.action(action);
        }
    }
    let fun = builder.get();
    let model_perf = bound(&fun, &context);
    let dev_fun = telamon::codegen::Function::build(&fun);
    let run_perf = unwrap!(context.evaluate(&dev_fun));

    if let Some(early_model_perf) = early_model_perf {
        info!("bound: {}", model_perf);
        let model_diff = model_perf.value() - run_perf;
        let model_ratio = - model_diff / run_perf;
        let early_diff = early_model_perf - run_perf;
        let early_ratio = - early_diff / run_perf;
        println!("{}: real {:.4e}ns, model {:+.2e} (x{:.2e}), early {:+.2e} (x{:.2e})",
                 T::name(), run_perf, model_diff, model_ratio, early_diff, early_ratio);
    } else {
        info!("bound: {}", model_perf);
        let model_diff = model_perf.value() - run_perf;
        let model_ratio = - model_diff / run_perf;
        println!("{}: real {:.4e}ns, model {:+.2e} (x{:.2e})",
                 T::name(), run_perf, model_diff, model_ratio);
    }
}

fn main() {
    let _ = env_logger::init();
    let arg = std::env::args().nth(1).unwrap_or(String::new());
    let pattern = match Regex::new(&arg) {
        Ok(x) => x,
        Err(e) => {
            println!("Invalid pattern: {}", e);
            std::process::exit(1);
        }
    };

    run::<latency::EmptyLoop>(&pattern);
    run::<latency::TwoEmptyLoop>(&pattern);
    run::<latency::InstChain>(&pattern);
    run::<latency::LongInstChain>(&pattern);
    run::<latency::UnrollReduction>(&pattern);
    run::<latency::OrderedLoops>(&pattern);
    run::<latency::OrderedThreadDims>(&pattern);
    run::<latency::DimMap>(&pattern);
    run::<latency::OperandPositionSlow>(&pattern);
    run::<latency::OperandPositionFast>(&pattern);
    run::<memory::L1LinesPressure>(&pattern);
    run::<memory::L2LinesPressure>(&pattern);
    run::<memory::SharedLoad>(&pattern);
    run::<memory::VectorSharedLoad>(&pattern);
    run::<memory::SharedReplay>(&pattern);
    run::<memory::VectorSharedReplay>(&pattern);
    run::<tests::Test0>(&pattern);
    run::<tests::Test1>(&pattern);
    run::<tests::Test2>(&pattern);
}
