extern crate telamon_gen;
extern crate env_logger;

use std::path::Path;

fn main() {
    let _ = ::env_logger::init();
    let out_dir = std::env::var_os("OUT_DIR").unwrap();
    for entry in std::fs::read_dir("src/").unwrap() {
        let entry = entry.unwrap();
        if !entry.file_type().unwrap().is_file() { continue; }
        let src_path = entry.path();
        if !src_path.extension().map(|s| s == "exh").unwrap_or(false) { continue; }
        let file_name = src_path.file_name().unwrap();
        println!("cargo:rerun-if-changed={}", file_name.to_str().unwrap());
        let dst_path = Path::new(&out_dir).join(&file_name).with_extension("rs");
        telamon_gen::process_file(&src_path, &dst_path, !cfg!(feature="noformat_exh"));
    }
}
