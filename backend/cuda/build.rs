//! Rust script to compile non-rust files.
use cc;
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
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
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
        .file("src/api/wrapper.c")
        .compile("libdevice_cuda_wrapper.a");
    add_dependency(Path::new("src/api/wrapper.c"));
    add_lib("cuda");
    add_lib("curand");
    add_lib("cupti");
}

fn main() {
    if cfg!(feature = "real_gpu") {
        compile_link_cuda();
    }
}
