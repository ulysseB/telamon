//! Benchmarks the exploration on CUDA gpus.
use std::io::Write;

use telamon::{codegen, explorer};
use telamon_cli::{Bench, CommonOpt, KernelParam, Platform};
use telamon_kernels::statistics::estimate_mean;

use structopt::StructOpt;

#[derive(StructOpt)]
struct Opt {
    #[structopt(flatten)]
    common: CommonOpt,

    #[structopt(short = "r", long = "repeat", default_value = "10")]
    repeat: usize,

    #[structopt(short = "k", long = "kernel")]
    kernels: Vec<KernelParam>,

    #[structopt(long = "platform", default_value = "cuda")]
    platform: Platform,

    /// Number of times to run the generated code to evaluate its performance.
    #[structopt(long = "num-code-runs", default_value = "40")]
    num_code_runs: usize,
}

fn main() {
    env_logger::init();
    let args = Opt::from_args();

    let builder = args.platform.to_builder();
    let mut config = args.common.config().unwrap().clone();
    let output_base = std::path::Path::new(&config.output_dir).to_owned();

    for idx in 0..args.repeat {
        for kernel in &args.kernels {
            config.output_dir = output_base
                .join(kernel.to_string())
                .join(idx.to_string())
                .to_str()
                .unwrap()
                .to_string();

            let mut context = builder.build_context();
            let (bundle, context) = context.kernel_bundle(kernel);

            let best = explorer::find_best_ex(
                &config,
                context,
                bundle.candidates,
                Some(&bundle.check_fn),
            )
            .unwrap_or_else(|| panic!("no candidates found for kernel {}", kernel));

            let best_fn = codegen::Function::build(&best.space);
            let runtime = context.benchmark(&best_fn, args.num_code_runs);

            let ref_runtime = Bench::default()
                .runs(args.num_code_runs)
                .benchmark_fn(&bundle.reference_fn);

            let mut f =
                std::fs::File::create(config.output_path("benchmark.txt").unwrap())
                    .unwrap();
            writeln!(f, "runtimes: {:?}", runtime).unwrap();
            let mean = estimate_mean(runtime, 0.95, "ns");
            let ref_mean = estimate_mean(ref_runtime, 0.95, "ns");
            writeln!(
                f,
                "{}: {}, reference: {}, speedup: {:.2}",
                kernel,
                mean,
                ref_mean,
                ref_mean.value / mean.value
            )
            .unwrap();
        }
    }
}
