//! Rust script to compile non-rust files.
extern crate cc;
extern crate failure;
extern crate glob;
extern crate telamon_gen;

use std::env;
use std::path::{Path, PathBuf};

/// Orders to Cargo to link a library.
fn add_lib(lib: &str) {
    println!("cargo:rustc-link-lib={}", lib);
}

fn add_dependency(dep: &Path) {
    println!("cargo:rerun-if-changed={}", dep.display());
}

/// Compiles and links the cuda wrapper and libraries.
fn compile_link_cuda() {
    let mut builder = cc::Build::new();

    // If CUDA_HOME is defined, use the cuda headers and libraries from there.
    if let Some(cuda_home) = env::var_os("CUDA_HOME").map(PathBuf::from) {
        println!(
            "cargo:rustc-link-search=native={}",
            cuda_home.join("lib64").display()
        );
        println!(
            "cargo:rustc-link-search=native={}",
            cuda_home
                .join("extras")
                .join("CUPTI")
                .join("lib64")
                .display()
        );

        builder
            .include(cuda_home.join("include"))
            .include(cuda_home.join("extras").join("CUPTI").join("include"));
    }

    builder
        .flag("-Werror")
        .file("src/device/cuda/api/wrapper.c")
        .compile("libdevice_cuda_wrapper.a");
    add_dependency(Path::new("src/device/cuda/api/wrapper.c"));
    add_lib("cuda");
    add_lib("curand");
    add_lib("cupti");
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
    if cfg!(feature = "cuda") {
        compile_link_cuda();
    }

    if cfg!(feature = "mppa") {
        add_lib("telajax");
        add_lib("OpenCL");
    }
}
