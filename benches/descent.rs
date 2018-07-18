//! Benchmarks how fast it is to decend in the search tree.
#[macro_use]
extern crate criterion;
extern crate env_logger;
#[macro_use]
extern crate lazy_static;

use criterion::Criterion;

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
    c.bench_function("mm descent without copies", |b| b.iter(|| {
        common::descend_without_copies(common::MM.clone());
    }));
}

/// Benchmarks full descents in the search tree, with a copy at each level.
fn mm_descent_copy(c: &mut Criterion) {
    let _ = env_logger::try_init();
    c.bench_function("mm descent with copy", |b| b.iter(|| {
        common::descend_with_copies(common::MM.clone());
    }));
}

criterion_group! {
    name = benches;
    config = config_criterion();
    targets = mm_descent, mm_descent_copy
}

criterion_main!(benches);
