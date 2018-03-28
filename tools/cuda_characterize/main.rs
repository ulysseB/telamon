#![feature(iterator_step_by)]
extern crate env_logger;
extern crate telamon;
extern crate getopts;
extern crate itertools;
#[macro_use]
extern crate log;
extern crate prettytable;
extern crate rustc_serialize;
extern crate telamon_utils as utils;

mod instruction;
mod gen;
mod gpu;
mod math;
mod table;

use telamon::device::cuda::{Executor, Gpu, PerfCounter};
use getopts::Options;
use itertools::Itertools;
use table::Table;
use rustc_serialize::json;
use std::io::{Read, Write};

/// Creates an empty `Table` to hold the given performance counters.
fn create_table(parameters: &[&str], counters: &[PerfCounter]) -> Table<u64> {
    let header = parameters.iter().map(|x| x.to_string())
        .chain(counters.iter().map(|x| x.to_string())).collect_vec();
    Table::new(header)
}

/// Returns the list of existing GPU description.
fn gpu_list() -> Result<Vec<Gpu>, String> {
    let mut file = match std::fs::File::open("data/cuda_gpus.json") {
        Ok(f) => f,
        Err(x) => return Err(x.to_string()),
    };
    let mut string = String::new();
    match file.read_to_string(&mut string) {
        Ok(_) => (),
        Err(x) => return Err(x.to_string()),
    };
    match json::decode(&string) {
        Ok(x) => Ok(x),
        Err(x) => Err(x.to_string()),
    }
}

/// Asks for a confirmation on the command line.
fn ask_confirmation(msg: &str) -> bool {
    let mut stdout = std::io::stdout();
    write!(stdout, "{} (y/N): ", msg).unwrap();
    stdout.flush().unwrap();
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
    match input.trim() {
        "y" => true,
        _ => false,
    }
}

fn main() {
    env_logger::init();
    let mut opts = Options::new();
    opts.optflag("h", "help", "Print the help menu.");
    opts.optflag("w", "write", "Write the gpu description to the gpu description file.");
    opts.optflag("y", "yes",
                 "Do not ask for confirmation before writing the gpu description.");
    let args = std::env::args().collect_vec();
    let opt_matches = match opts.parse(&args[1..]) {
        Ok(x) => x,
        Err(x) => {
            println!("{} Use '--help' to display a list of valid options.", x);
            std::process::exit(-1)
        },
    };
    if opt_matches.opt_present("h") {
        let brief = opts.short_usage(&args[0]);
        println!("{}", opts.usage(&brief));
        return;
    }
    let write_desc = opt_matches.opt_present("w");
    let ask_before_write = !opt_matches.opt_present("y");

    let executor = Executor::init();
    let mut gpu = gpu::functional_desc(&executor);
    info!("GPU name: {}", executor.device_name());
    gpu::performance_desc(&executor, &mut gpu);

    let ok = write_desc && (!ask_before_write || {
        println!("{}", json::as_pretty_json(&gpu));
        ask_confirmation("Write the GPU description ?")
    });
    if ok {
        let mut gpu_list = match gpu_list() {
            Ok(x) => x.into_iter().filter(|x| x.name != gpu.name).collect_vec(),
            Err(s) => {
                println!("Unable to load the existing GPU descriptions: {}", s);
                if ask_confirmation("Overwrite existing descriptions ?") {
                    vec![Gpu { name: "dummy_cuda_gpu".to_string(), .. gpu.clone() }]
                } else {
                    std::process::exit(0)
                }
            },
        };
        gpu_list.push(gpu);
        let mut file = std::fs::File::create("data/cuda_gpus.json").unwrap();
        write!(file, "{}", json::as_pretty_json(&gpu_list)).unwrap();
    } else {
        println!("{}", json::as_pretty_json(&gpu));
    }
    //instruction::print_smx_bandwidth(&gpu, &executor);
    //instruction::print_smx_store_bandwidth(&gpu, &executor);
}
