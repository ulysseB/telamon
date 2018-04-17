extern crate cc;
extern crate lalrpop;

use std::process::Command;

/// Adds a dependency to the build script.
fn add_dependency(dep: &str) { println!("cargo:rerun-if-changed={}", dep); }

fn main() {
    // Compile the lexer.
    Command::new("flex")
            .arg("-oexh.c")
            .arg("src/exh.l")
            .status()
            .expect("failed to execute Flex's process");

    cc::Build::new()
            .file("exh.c")
            .compile("exh.a");

    // Compile the parser.
    add_dependency("src/parser.lalrpop");
    lalrpop::Configuration::new().use_cargo_dir_conventions().process().unwrap();
}
