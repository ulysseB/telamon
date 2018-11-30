extern crate getopts;
extern crate toml;

extern crate telamon;

use std::env;
use std::io::{Read, Write};

use getopts::Options;

/// Expand a configuration.
///
/// # Arguments
///
///  - input: Path to the input file to use.  If `None`, the default configuration is loaded.  If
///    "-", the configuration is read from the standard input.
///
///  - output: Path to the output file.  If "-", the configuration is printed to the standard
///    output.
fn expand_config(input: Option<&str>, output: &str) {
    let input_str = input
        .map(|input| {
            let mut input_str = String::new();
            match input {
                "-" => {
                    let stdin = std::io::stdin();
                    stdin.lock().read_to_string(&mut input_str).unwrap();
                }
                _ => {
                    std::fs::File::open(input)
                        .unwrap()
                        .read_to_string(&mut input_str)
                        .unwrap();
                }
            };
            input_str
        }).unwrap_or_else(|| "".to_string());

    let config: telamon::explorer::config::Config = toml::from_str(&input_str).unwrap();
    let output_str = toml::to_string(&config).unwrap();

    match output {
        "-" => print!("{}", output_str),
        _ => {
            write!(std::fs::File::create(output).unwrap(), "{}", output_str);
        }
    }
}

fn print_usage(program: &str, opts: Options) {
    let brief = format!(
        "Usage: {} PATH [options]

Loads an explorer configuration file and outputs it with the default values added.  This can also
be used as a simple validator of an explorer configuration file by discarding the output.
",
        program
    );
    print!("{}", opts.usage(&brief));
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let program = args[0].clone();

    let mut opts = Options::new();
    opts.optflag("h", "help", "Prints help information");
    opts.optopt("o", "output", "Path to the output file", "PATH");

    let matches = match opts.parse(&args[1..]) {
        Ok(m) => m,
        Err(f) => panic!(f.to_string()),
    };

    if matches.opt_present("h") {
        print_usage(&program, opts);
        return;
    }

    let output = matches.opt_str("o").unwrap_or_else(|| "-".to_string());
    let input = if !matches.free.is_empty() {
        Some(&matches.free[0][..])
    } else {
        None
    };

    expand_config(input, &output);
}
