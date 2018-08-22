extern crate env_logger;
/// Tool that generates constraints from stdin to stdout.
extern crate telamon_gen;

use std::path::Path;
use std::process;

fn main() {
    env_logger::init();
    if let Err(process_error) = telamon_gen::process(
        Some(&mut std::io::stdin()),
        &mut std::io::stdout(),
        true,
        &Path::new("exh"),
    ) {
        eprintln!("error: {}", process_error);
        process::exit(-1);
    }
}
