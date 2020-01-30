//! Rust script to compile non-rust files.
use std::path::Path;

fn add_dependency(dep: &Path) {
    println!("cargo:rerun-if-changed={}", dep.display());
}

fn main() {
    let exh_file = "src/search_space/choices.exh";
    let out_dir = std::env::var_os("OUT_DIR").unwrap();
    for file in glob::glob("src/search_space/*.exh").unwrap() {
        add_dependency(&file.unwrap());
    }

    let exh_out = Path::new(&out_dir).join("choices.rs");
    telamon_gen::process_file(
        &Path::new(exh_file),
        &exh_out,
        cfg!(feature = "format_exh"),
    )
    .unwrap_or_else(|err| {
        eprintln!("could not compile EXH file: {}", err);
        std::process::exit(-1);
    });
}
