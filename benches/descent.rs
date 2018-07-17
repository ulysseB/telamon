//! Benchmarks how fast it is to decend in the search tree.
#[macro_use]
extern crate criterion;
extern crate env_logger;
extern crate telamon;
extern crate rand;
#[macro_use]
extern crate lazy_static;

use criterion::Criterion;
use telamon::explorer;
use telamon::explorer::choice::ActionEx;
use rand::Rng;

mod common;

/// Configure the bencher.
fn config_criterion() -> Criterion {
    Criterion::default()
        .sample_size(50)
        .configure_from_args()
}

/// Benchmarks full descents in the search tree.
fn mm_descent(c: &mut Criterion) {
    let _ = env_logger::try_init();
    c.bench_function("mm descent", |b| b.iter(|| {
        let mut space = common::MM.clone();
        while let Some(mut choice) = {
            let choice = explorer::choice::list(&space).next();
            choice
        } {
           let id = rand::thread_rng().gen_range(0, choice.len());
           let res = match choice.swap_remove(id) {
                ActionEx::TileSizes(..) => panic!(),
                ActionEx::Action(action) => space.apply_decisions(vec![action]),
                ActionEx::LowerLayout { mem, ref st_dims, ref ld_dims } =>
                    space.lower_layout(mem, st_dims.clone(), ld_dims.clone()),
           };
           if res.is_err() { return; }
        }
    }));
}

/// Benchmarks full descents in the search tree, with a copy at each level.
fn mm_descent_copy(c: &mut Criterion) {
    let _ = env_logger::try_init();
    c.bench_function("mm descent with copy", |b| b.iter(|| {
        let mut spaces = vec![];
        let mut space = common::MM.clone();
        while let Some(mut choice) = {
            let choice = explorer::choice::list(&space).next();
            choice
        } {
            spaces.push(space.clone());
            let id = rand::thread_rng().gen_range(0, choice.len());
            let res = match choice.swap_remove(id) {
                ActionEx::TileSizes(..) => panic!(),
                ActionEx::Action(action) => space.apply_decisions(vec![action]),
                ActionEx::LowerLayout { mem, ref st_dims, ref ld_dims } =>
                    space.lower_layout(mem, st_dims.clone(), ld_dims.clone()),
            };
            if res.is_err() { return; }
        }
    }));
}

criterion_group! {
    name = benches;
    config = config_criterion();
    targets = mm_descent, mm_descent_copy
}

criterion_main!(benches);
