#[macro_use]
extern crate criterion;
extern crate telamon_gen;

use criterion::Criterion;

use telamon_gen::lexer;

use std::ffi::OsStr;
use std::fs;

fn criterion_benchmark(c: &mut Criterion) {
    let entries = fs::read_dir("cc_tests/src/").unwrap();
    for entry in entries {
        if let Ok(entry) = entry {
            if entry.path().extension().eq(&Some(OsStr::new("exh"))) {
                let path = entry.path();
                let mut input = fs::File::open(&path).unwrap();
                let mut name = String::from("lexer ");
                name.push_str(path.file_stem().unwrap().to_str().unwrap());

                c.bench_function(&name, move |b| {
                    b.iter(|| lexer::Lexer::from_input(&mut input))
                });
            }
        }
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
