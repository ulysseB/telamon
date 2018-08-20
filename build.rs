//! Rust script to compile non-rust files.
extern crate cc;
extern crate telamon_gen;

use std::path::Path;

/// Orders to Cargo to link a library.
fn add_lib(lib: &str) {
    println!("cargo:rustc-link-lib={}", lib);
}

fn add_dependency(dep: &str) {
    println!("cargo:rerun-if-changed={}", dep);
}

/// Compiles and links the cuda wrapper and libraries.
fn compile_link_cuda() {
    cc::Build::new()
        .flag("-Werror")
        .file("src/device/cuda/api/wrapper.c")
        .compile("libdevice_cuda_wrapper.a");
    add_dependency("src/device/cuda/api/wrapper.c");
    add_lib("cuda");
    add_lib("curand");
    add_lib("cupti");
}

fn main() {
    let exh_file = "src/search_space/choices.exh";
    let out_dir = std::env::var_os("OUT_DIR").unwrap();

    add_dependency(exh_file);
    let exh_out = Path::new(&out_dir).join("choices.rs");
    telamon_gen::process_file(
        &Path::new(exh_file),
        &exh_out,
        cfg!(feature = "format_exh"),
    ).unwrap();
    if cfg!(feature = "cuda") {
        compile_link_cuda();
    }

    if cfg!(feature = "mppa") {
        add_lib("telajax");
        add_lib("OpenCL");
    }
}
