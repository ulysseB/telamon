fn main() {
    if cfg!(feature = "mppa") {
        println!("cargo:rustc-link-lib=telajax");
        println!("cargo:rustc-link-lib=OpenCL");
    }
}
