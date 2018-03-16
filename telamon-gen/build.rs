extern crate lalrpop;

/// Adds a dependency to the build script.
fn add_dependency(dep: &str) { println!("cargo:rerun-if-changed={}", dep); }

fn main() {
    // Compile the parser.
    add_dependency("src/parser.lalrpop");
    lalrpop::Configuration::new().use_cargo_dir_conventions().process().unwrap();
}
