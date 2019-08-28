///! Generates a list of implementations for a matrix multiplication
///! kernel and writes a list with the lists of actions defining the
///! implementations to a dump file
use log::{debug, warn};
use std::fs;
use std::path::PathBuf;
use structopt::StructOpt;
use telamon::codegen;
use telamon::explorer::{self, local_selection};
use telamon_kernels::{linalg, Kernel, KernelBuilder};
use telamon_mppa as mppa;

#[derive(Debug, StructOpt)]
#[structopt(name = "dump_generated_code")]
struct Opt {
    #[structopt(
        parse(from_os_str),
        short = "o",
        long = "output",
        default_value = "dumps/implementation_code.json"
    )]
    /// Output file
    output: PathBuf,

    #[structopt(short = "n", long = "num-implementations", default_value = "20")]
    /// Number of implementations to be generated
    num_implementations: usize,

    #[structopt(short = "c", long = "cut", default_value = "2e8")]
    /// Minimum cut that must be met when considering candidates
    cut: f64,

    #[structopt(short = "f", long = "force")]
    /// Overwrite an existing dump file
    force: bool,
}

fn main() {
    let opt = Opt::from_args();
    env_logger::try_init().unwrap();

    if !opt.force && opt.output.exists() {
        warn!(
            "Skipping generation of {}, dump already exists",
            opt.output.to_str().unwrap()
        );
    } else {
        let file = fs::File::create(&opt.output).unwrap();
        let mut action_list = vec![];
        let mut context = mppa::Context::new();
        let params = linalg::FusedMMP::new(16, 16, 16);

        let (signature, kernel, context) = KernelBuilder::new()
            .build::<linalg::FusedMM<f32>, mppa::Context>(params, &mut context);

        let candidates = kernel.build_body(signature.into(), context);
        let ordering = explorer::config::ChoiceOrdering::default();
        let mut impls_generated = 0;

        while impls_generated < opt.num_implementations {
            debug!(
                "Trying to generate implementation {}/{}",
                impls_generated + 1,
                opt.num_implementations
            );

            let order = explorer::config::NewNodeOrder::WeightedRandom;
            let candidate_idx = order.pick_candidate(&candidates, opt.cut).unwrap();
            let candidate = candidates[candidate_idx].clone();
            let implementation =
                local_selection::descend(&ordering, order, context, candidate, opt.cut);

            if let Some(implementation) = implementation {
                debug!("Found implementation");

                debug!("Invoking backend for code generation");
                let function = codegen::Function::build(&implementation.space);
                let mut namegen = mppa::NameGenerator::default();
                let mut name_map = codegen::NameMap::new(&function, &mut namegen);
                let mut printer = mppa::printer::MppaPrinter::default();
                let kernel_code = printer.wrapper_function(&function, &mut name_map, 1);

                action_list.push((implementation.actions.reverse(), kernel_code));
                impls_generated += 1;
            } else {
                debug!("Deadend encountered");
            }
        }

        debug!("Writing dump file");
        serde_json::to_writer(&file, &action_list).unwrap();
    }
}
