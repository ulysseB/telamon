use std::env;
use std::path::PathBuf;

fn main() {
    let out_path = PathBuf::from(env::var_os("OUT_DIR").unwrap());

    // Get includes and libraries from CUDA_HOME, if defined
    println!("cargo:rerun-if-env-changed=CUDA_HOME");

    let mut clang_args = Vec::new();
    if let Some(cuda_home) = env::var_os("CUDA_HOME").map(PathBuf::from) {
        clang_args.push(format!("-I{}", cuda_home.join("include").display()));

        println!(
            "cargo:rustc-link-search=native={}",
            cuda_home.join("lib64").display()
        );

        if cfg!(feature = "cupti") {
            clang_args.push(format!(
                "-I{}",
                cuda_home
                    .join("extras")
                    .join("CUPTI")
                    .join("include")
                    .display()
            ));

            println!(
                "cargo:rustc-link-search=native={}",
                cuda_home
                    .join("extras")
                    .join("CUPTI")
                    .join("lib64")
                    .display(),
            );
        }
    }

    bindgen::Builder::default()
        .header_contents("library_types.h", "#include <library_types.h>")
        .clang_args(clang_args.iter())
        .prepend_enum_name(false)
        .whitelist_recursively(false)
        .whitelist_type("cudaDataType")
        .whitelist_type("cudaDataType_t")
        .whitelist_type("libraryPropertyType")
        .whitelist_type("libraryPropertyType_t")
        .generate()
        .expect("Unable to generate library_types.h bindings")
        .write_to_file(out_path.join("library_types.rs"))
        .expect("Couldn't write library_types.h bindings");

    bindgen::Builder::default()
        .header_contents("vector_types.h", "#include <vector_types.h>")
        .clang_args(clang_args.iter())
        .prepend_enum_name(false)
        .whitelist_recursively(false)
        .whitelist_type("u?(char|short|int|long|longlong)[1-4]")
        .whitelist_type("(float|double)[1-4]")
        .whitelist_type("dim3")
        .generate()
        .expect("Unable to generate vector_types.h bindings")
        .write_to_file(out_path.join("vector_types.rs"))
        .expect("Couldn't write vector_types.h bindings");

    bindgen::Builder::default()
        .header_contents("driver_types.h", "#include <driver_types.h>")
        .clang_args(clang_args.iter())
        .prepend_enum_name(false)
        .whitelist_recursively(false)
        .whitelist_var("cuda.*")
        .whitelist_type("cuda.*")
        .whitelist_type("CU.*")
        .generate()
        .expect("Unable to generate driver_types.h bindings")
        .write_to_file(out_path.join("driver_types.rs"))
        .expect("Couldn't write driver_types.h bindings");

    bindgen::Builder::default()
        .header_contents("cuComplex.h", "#include <cuComplex.h>")
        .clang_args(clang_args.iter())
        .prepend_enum_name(false)
        .whitelist_recursively(false)
        .whitelist_type("cu(Float|Double)?Complex")
        .generate()
        .expect("Unable to generate cuComplex.h bindings")
        .write_to_file(out_path.join("cuComplex.rs"))
        .expect("Couldn't write cuComplex.h bindings");

    if cfg!(any(feature = "cuda", feature = "cupti")) {
        bindgen::Builder::default()
            .header_contents("cuda_types.h", "#include <cuda.h>")
            .clang_args(clang_args.iter())
            .prepend_enum_name(false)
            .whitelist_recursively(false)
            .whitelist_type("cu.*")
            .whitelist_type("CU.*")
            // Those are defined in driver_types.h
            .blacklist_type("CUuuid")
            .blacklist_type("CUuuid_st")
            .generate()
            .expect("Unable to generate cuda.h type bindings")
            .write_to_file(out_path.join("cuda_types.rs"))
            .expect("Couldn't write cuda.h type bindings");
    }

    if cfg!(feature = "cuda") {
        println!("cargo:rustc-link-lib=cuda");

        bindgen::Builder::default()
            .header_contents("cuda_types.h", "#include <cuda.h>")
            .clang_args(clang_args.iter())
            .prepend_enum_name(false)
            .whitelist_recursively(false)
            .whitelist_function("cu.*")
            .generate()
            .expect("Unable to generate cuda.h bindings")
            .write_to_file(out_path.join("cuda.rs"))
            .expect("Couldn't write cuda.h bindings");
    }

    if cfg!(feature = "cupti") {
        println!("cargo:rustc-link-lib=cupti");

        bindgen::Builder::default()
            .header_contents("cupti.h", "#include <cupti.h>")
            .clang_args(clang_args.iter())
            .prepend_enum_name(false)
            .whitelist_recursively(false)
            .whitelist_type("CUpti.*")
            .whitelist_function("cupti.*")
            // Those hit a packing/align issue; marking them as opaque for the time being.
            // https://github.com/rust-lang/rust-bindgen/issues/1538
            .opaque_type("CUpti_ActivityUnifiedMemoryCounterConfig")
            .opaque_type("CUpti_ActivityAutoBoostState")
            .opaque_type("CUpti_ActivityPCSamplingConfig")
            .opaque_type("CUpti_ActivityContext")
            .opaque_type("CUpti_ActivityPCSampling")
            .opaque_type("CUpti_ActivityPCSampling2")
            .opaque_type("CUpti_ActivityCudaEvent")
            .opaque_type("CUpti_ActivityStream")
            .opaque_type("CUpti_ActivityInstructionCorrelation")
            .opaque_type("CUpti_ActivityPcie")
            .opaque_type("CUpti_Activity")
            .generate()
            .expect("Unable to generate CUPTI bindings")
            .write_to_file(out_path.join("cupti.rs"))
            .expect("Couldn't write CUPTI bindings");
    }

    if cfg!(feature = "curand") {
        println!("cargo:rustc-link-lib=curand");

        bindgen::Builder::default()
            .header_contents("curand.h", "#include <curand.h>")
            .clang_args(clang_args.iter())
            .prepend_enum_name(false)
            .whitelist_recursively(false)
            .whitelist_type("curand.*")
            .whitelist_function("curand.*")
            .generate()
            .expect("Unable to generate cuRAND bindings")
            .write_to_file(out_path.join("curand.rs"))
            .expect("Couldn't write cuRAND bindings");
    }

    if cfg!(feature = "cudnn") {
        println!("cargo:rustc-link-lib=cudnn");

        bindgen::Builder::default()
            .header_contents("cudnn.h", "#include <cudnn.h>")
            .clang_args(clang_args.iter())
            .prepend_enum_name(false)
            .whitelist_recursively(false)
            .whitelist_type("cudnn.*")
            .whitelist_function("cudnn.*")
            .generate()
            .expect("Unable to generate cuDNN bindings")
            .write_to_file(out_path.join("cudnn.rs"))
            .expect("Couldn't write cuDNN bindings");
    }

    if cfg!(any(feature = "cublas", feature = "cublas-lt")) {
        bindgen::Builder::default()
            .header_contents("cublas_types.h", "#include <cublas.h>")
            .clang_args(clang_args.iter())
            .prepend_enum_name(false)
            .whitelist_recursively(false)
            .whitelist_type("cublas.*")
            .generate()
            .expect("Unable to generate cuBLAS type bindings")
            .write_to_file(out_path.join("cublas_types.rs"))
            .expect("Couldn't write cuBLAS type bindings");
    }

    if cfg!(feature = "cublas") {
        println!("cargo:rustc-link-lib=cublas");

        bindgen::Builder::default()
            .header_contents("cublas.h", "#include <cublas.h>")
            .clang_args(clang_args.iter())
            .prepend_enum_name(false)
            .whitelist_recursively(false)
            .whitelist_function("cublas.*")
            .generate()
            .expect("Unable to generate cuBLAS bindings")
            .write_to_file(out_path.join("cublas.rs"))
            .expect("Couldn't write cuBLAS bindings");

        // CUDA 4+
        bindgen::Builder::default()
            .header_contents("cublas_v2.h", "#include <cublas_v2.h>")
            .clang_args(clang_args.iter())
            .prepend_enum_name(false)
            .whitelist_recursively(false)
            .whitelist_function("cublas.*")
            .generate()
            .expect("Unable to generate cuBLAS v2 bindings")
            .write_to_file(out_path.join("cublas_v2.rs"))
            .expect("Couldn't write cuBLAS v2 bindings");
    }

    if cfg!(feature = "cublas-lt") {
        println!("cargo:rustc-link-lib=cublasLt");
        // CUDA 10+
        bindgen::Builder::default()
            .header_contents(
                "cublasLt.h",
                // FIXME: typedef needed because of broken cublasLt.h in CUDA 10
                "
typedef enum cudaDataType_t cudaDataType_t;

#include <cublasLt.h>
",
            )
            .clang_args(clang_args.iter())
            .prepend_enum_name(false)
            .whitelist_recursively(false)
            .whitelist_type("cublasLt.*")
            .whitelist_function("cublasLt.*")
            .generate()
            .expect("Unable to generate cuBLASLt bindings")
            .write_to_file(out_path.join("cublasLt.rs"))
            .expect("Couldn't write cuBLASLt bindings");
    }
}
