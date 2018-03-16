extern crate pbr;
extern crate env_logger;
extern crate telamon;
extern crate crossbeam;
#[macro_use]
extern crate lazy_static;

#[macro_use]
mod bencher;

use bencher::*;
use telamon::{device, explorer, helper};
use telamon::device::Context;

fn main() {
    benchmark!(sequential_evaluation, 100);
    benchmark!(parallel_evaluation, 512, 8);
}

/// Evaluates a kernel.
fn sequential_evaluation(bencher: &Bencher) {
    let ref mut context = device::mppa::Context::new();
    let signature = helper::SignatureBuilder::new("empty", context).get();
    let builder = helper::Builder::new(&signature, context.device());
    let fun = builder.get();
    let dev_fun = device::Function::build(&fun);
    context.evaluate(&dev_fun);
    bencher.bench(|| context.evaluate(&dev_fun))
}

/// Executes a kernel multiple times in parallel.
fn parallel_evaluation(bencher: &Bencher, threads: usize) {
    let ref mut context = device::mppa::Context::new();
    let signature = helper::SignatureBuilder::new("empty", context).get();
    let builder = helper::Builder::new(&signature, context.device());
    let fun = builder.get();
    let dev_fun = device::Function::build(&fun);
    context.evaluate(&dev_fun);
    let per_thread = bencher.num_samples()/threads + 1;
    bencher.bench_single(|progress| {
        context.async_eval(&|_, _| progress.incr(), &mut |eval| {
            crossbeam::scope(|scope| for _ in 0..threads {
                let fun = fun.clone();
                scope.spawn(move || {
                    for _ in 0..per_thread {
                        let candidate = explorer::Candidate {
                            space: fun.clone(),
                            bound: 0.0,
                            depth: 1,
                        };
                        eval.add_kernel(candidate);
                    }
                });
            });
        })
    })
}
