extern crate env_logger;
extern crate itertools;
extern crate telamon;
extern crate getopts;

use itertools::Itertools;
use getopts::Options;
use telamon::device::cuda;

#[derive(Default)]
struct Config {
    write_to_file: bool,
}

impl Config {
    fn parse() -> Self {
        let mut config = Self::default();
        let mut opts = Options::new();
        opts.optflag("h", "help", "Print the help menu.");
        opts.optflag("w", "write", "Write the gpu description to the description file.");
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
            std::process::exit(0);
        }
        config.write_to_file = opt_matches.opt_present("w");
        config
    }
}

fn main() {
    env_logger::init();
    let config = Config::parse();
    let executor = cuda::Executor::init();
    let gpu = cuda::characterize::characterize(&executor);

    if config.write_to_file {
        unimplemented!(); // FIXME
        /*let mut gpu_list = match gpu_list() {
            Ok(x) => x.into_iter().filter(|x| x.name != gpu.name).collect_vec(),
            Err(s) => {
                println!("Unable to load the existing GPU descriptions: {}", s);
            },
            };
            gpu_list.push(gpu);
            let mut file = std::fs::File::create("data/cuda_gpus.json").unwrap();
            write!(file, "{}", json::as_pretty_json(&gpu_list)).unwrap();
        } else {
            println!("{}", json::as_pretty_json(&gpu));
        }*/
    }
    //instruction::print_smx_bandwidth(&gpu, &executor);
    //instruction::print_smx_store_bandwidth(&gpu, &executor);*/
}
