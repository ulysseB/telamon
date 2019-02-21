//! Rust script to compile non-rust files.
use std::fs;
use std::env;
use std::path::{Path, PathBuf};

/// Orders to Cargo to link a library.
fn add_lib(lib: &str) {
    println!("cargo:rustc-link-lib={}", lib);
}

fn add_dependency(dep: &Path) {
    println!("cargo:rerun-if-changed={}", dep.display());
}

fn compile_and_link_telajax() {
    let dst = cmake::Config::new("telajax")
        .define("OPENCL_ROOT", "/usr/local/k1tools/")
        .build();
    println!("cargo:rustc-link-search={}", dst.join("lib64/").display());
    println!("cargo:rustc-link-search=/usr/local/k1tools/lib64/");
}

fn add_telajax_dependencies() {
    for f in fs::read_dir("telajax/src").unwrap() {
        //let filepath = format!("{}", f.unwrap().path().display());
        add_dependency(&f.unwrap().path());
    }
}

fn bind_telajax() {
    let binding = bindgen::Builder::default()
        .header("telajax/include/telajax.h")
        .clang_arg("-I/usr/local/k1tools/include/")
        .whitelist_function("telajax_device_init")
        .whitelist_function("telajax_is_initialized")
        .whitelist_function("telajax_device_finalize")
        .whitelist_function("telajax_is_finalized")
        .whitelist_function("telajax_device_waitall")
        .whitelist_function("telajax_device_mem_alloc")
        .whitelist_function("telajax_device_mem_write")
        .whitelist_function("telajax_device_mem_read")
        .whitelist_function("telajax_device_mem_release")
        .whitelist_function("telajax_wrapper_build")
        .whitelist_function("telajax_wrapper_release")
        .whitelist_function("telajax_kernel_build")
        .whitelist_function("telajax_kernel_set_dim")
        .whitelist_function("telajax_kernel_release")
        .whitelist_function("telajax_kernel_set_args")
        .whitelist_function("telajax_kernel_enqueue")
        .whitelist_function("telajax_event_set_callback")
        .whitelist_function("telajax_event_wait")
        .whitelist_function("telajax_event_release")
        .generate()
        .expect("unable to generate bindings");
    let out_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    binding
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Could not write to file");
}

fn main() {
    compile_and_link_telajax();
    bind_telajax();
    add_lib("telajax");
    add_lib("OpenCL");
    add_telajax_dependencies();
}
