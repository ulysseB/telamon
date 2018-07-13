extern crate cbindgen;

use std::env;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let config = {
        let mut c: cbindgen::Config = Default::default();
        c.include_guard = Some("TELAMON_CAPI_H".to_owned());
        c.language = cbindgen::Language::C;
        c
    };
    cbindgen::generate_with_config(&crate_dir, config)
        .expect("Could not generate C header")
        .write_to_file("include/telamon.h");
}
