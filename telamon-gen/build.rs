extern crate cc;
extern crate lalrpop;


/// Adds a dependency to the build script.
fn add_dependency(dep: &str) { println!("cargo:rerun-if-changed={}", dep); }

fn main() {
    // Compile the lexer.(`LEX="flex" cargo build --features "lex"`)
    #[cfg(feature = "lex")]
    {
        use std::{env,process::Command};

        let bin = env::var("LEX").unwrap_or(String::from("flex"));

        Command::new(bin)
                .arg("-oexh.c")
                .arg("src/exh.l")
                .status()
                .expect("failed to execute Flex's process");
    }

    cc::Build::new()
            .file("exh.c")
            .flag("-Wno-unused-parameter")
            .flag("-Wno-unused-variable")
            .flag_if_supported("-Wno-unused-function")
            .compile("exh.a");

    // Compile the parser.
    add_dependency("src/parser.lalrpop");
    lalrpop::Configuration::new().use_cargo_dir_conventions().process().unwrap();
}
