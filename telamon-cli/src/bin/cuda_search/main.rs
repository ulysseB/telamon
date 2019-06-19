//! Benchmarks the exploration on CUDA gpus.
use std::io::{self, Write};

use telamon::device::{ArgMap, Context};
use telamon::explorer::config::Config;
use telamon::helper::MemInit;
use telamon_cli::{Bench, CommonOpt, ContextBuilder, CublasHandle, Reference};
use telamon_kernels::statistics::estimate_mean;
use telamon_kernels::{linalg, Kernel};

use structopt::StructOpt;

/// The number of times to run the generated code to evaluate its performance.
const NUM_CODE_RUNS: usize = 40;
/// Search timeout in minutes.
const TIMEOUT: u64 = 240;

/// Benchamrks a kernel against a reference implementation.
fn benchmark<'a, K, REF, CB>(
    mut config: Config,
    params: K::Parameters,
    executor: CB,
    reference: &REF,
    // output_dir: String,
) where
    K: Kernel<'a>,
    CB: ContextBuilder<'a>,
    REF: Reference<'a, K, Context = CB::Context>,
{
    config.timeout.get_or_insert(TIMEOUT);
    // config.output_dir = output_dir;
    //config.distance_to_best.get_or_insert(20.);

    let mut context = executor.build_context();
    let runtime = K::benchmark(
        &config,
        params.clone(),
        NUM_CODE_RUNS,
        MemInit::RandomFill,
        &mut context,
    );
    let ref_runtime = Bench::default()
        .warmup(4)
        .runs(NUM_CODE_RUNS)
        .benchmark_fn(|| reference.eval_reference(&params, &context));
    let mut f =
        std::fs::File::create(config.output_path("benchmark.txt").unwrap()).unwrap();
    writeln!(f, "runtimes: {:?}", runtime).unwrap();
    let mean = estimate_mean(runtime, 0.95, "ns");
    let ref_mean = estimate_mean(ref_runtime, 0.95, "ns");
    writeln!(
        f,
        "{}: {}, reference: {}, speedup: {:.2}",
        K::name(),
        mean,
        ref_mean,
        ref_mean.value / mean.value
    )
    .unwrap();
}

#[derive(StructOpt)]
struct Opt {
    #[strucopt(flatten)]
    common: CommonOpt,
}

trait BenchRun<'a, B, R> {
    fn run(self, config: &Config, builder: B, reference: &R);
}

struct Benchmark<'a, K>
where
    K: Kernel<'a>,
{
    params: K::Parameters,
    name: String,
    iteration: usize,
}

impl<'a, K> Benchmark<'a, K>
where
    K: Kernel<'a>,
{
    fn new(params: K::Parameters, name: String, iteration: usize) -> Self {
        Benchmark {
            params,
            name,
            iteration,
        }
    }

    fn output_dir(&self) -> String {
        format!("{}/{}", self.name, self.iteration)
    }
}

impl<'a, K, B, R> BenchRun<'a, B, R> for Benchmark<'a, K>
where
    K: Kernel<'a>,
    B: ContextBuilder<'a>,
    R: Reference<'a, K, Context = B::Context>,
{
    fn run(self, config: &Config, builder: B, reference: &R) {
        let mut config = config.clone();
        config.output_dir = std::path::Path::new(&config.output_dir)
            .join(self.output_dir())
            .to_str()
            .unwrap()
            .to_string();
        benchmark::<K, _, _>(config.clone(), self.params, builder, reference)
    }
}

macro_rules! benchmark {
    (Sgemm($m:literal, $n:literal, $k:literal)[$iter:expr]) => {{
        self::Benchmark::<'_, linalg::FusedMM<'_, f32>>::new(
            linalg::FusedMMP::new($m, $n, $k),
            format!("Sgemm_{}_{}_{}", $m, $n, $k),
            $iter,
        )
    }};

    (BatchMM($b:literal, $m:literal, $n:literal, $k:literal)[$iter:expr]) => {{
        self::Benchmark::<'_, linalg::BatchMM<'_, f32>>::new(
            lnialg::BatchMMP::new($b, $m, $n, $k),
            format!("BatchMM_{}_{}_{}_{}", $b, $m, $n, $k),
            $iter,
        )
    }};
}

fn main() {
    env_logger::init();
    let args = Opt::from_args();

    let executor = telamon_cuda::Executor::init();
    let reference = CublasHandle::new();

    let config = args.common.config().unwrap();

    // Repeat 10 times (!)
    for idx in 0..10 {
        /*
        benchmark!(Axpy(26)[idx]).run(&config, &executor, &cublas)
        benchmark::<linalg::Axpy<f32>, _>(
            (1 << 26, true),
            &executor,
            |params, ctx| saxpy_reference(&cublas_handle, params, ctx),
            format!("Saxpy_2p26/{}", idx),
        );
        */

        benchmark!(Sgemm(256, 256, 32)[idx]).run(&config, &executor, &reference);

        /* TODO(bclement): Fix this.
        let params = linalg::FusedMMP::new(1024, 1024, 1024).stride_a(32);
        benchmark::<linalg::FusedMM<f32>, _>(params, &executor, |_, _| {
            7.1e8 // Obtained from a cuda program.
        });
        */

        benchmark!(Sgemm(1024, 1024, 1024)[idx]).run(&config, &executor, &reference);

        /*
        let mut params = linalg::FusedMMP::new(4096, 4096, 4096);
        benchmark::<linalg::FusedMM<f32>, _>(
            params,
            &executor,
            |params, ctx| matmul_reference(&cublas_handle, params, ctx),
            format!("Sgemm_4096_4096_4096/{}", idx),
        );

        let params = linalg::BatchMMP::new(512, 32, 32, 64)
            .static_sizes()
            .reuse_b();
        benchmark::<linalg::BatchMM<f32>, _>(
            params,
            &executor,
            |params, ctx| batchmm_reference(&cublas_handle, params, ctx),
            format!("BatchMMP_512_32_32_64_SS_rb/{}", idx),
        );

        let params = linalg::BatchMMP::new(512, 32, 32, 64).static_sizes();
        benchmark::<linalg::BatchMM<f32>, _>(
            params,
            &executor,
            |params, ctx| batchmm_reference(&cublas_handle, params, ctx),
            format!("BatchMMP_512_32_32_64_SS/{}", idx),
        );

        let params = linalg::BatchMMP::new(512, 32, 32, 64);
        benchmark::<linalg::BatchMM<f32>, _>(
            params,
            &executor,
            |params, ctx| batchmm_reference(&cublas_handle, params, ctx),
            format!("BatchMMP_512_32_32_64/{}", idx),
        );

        benchmark::<linalg::MatVec<f32>, _>(
            (1 << 13, 1 << 13, true),
            &executor,
            |params, ctx| matvec_reference(&cublas_handle, params, ctx),
            format!("Sgemv_2p13_2p13_generic/{}", idx),
        );

        benchmark::<linalg::Gesummv<f32>, _>(
            (1 << 13, 1 << 13, true),
            &executor,
            |params, ctx| gesummv_reference(&cublas_handle, params, ctx),
            format!("Gesummv_2p13_2p13_generic/{}", idx),
        );
        */
    }

    /* OLD BENCHES
    // 1.5
    benchmark::<linalg::Axpy<f32>, _>((1 << 25, true), &executor, |params, ctx| {
        saxpy_reference(&cublas_handle, params, ctx)
    });
    // 2.82 without tb
    let params = linalg::FusedMMP::new(128, 256, 32).static_sizes(); //.transpose_b();
    benchmark::<linalg::FusedMM<f32>, _>(params, &executor, |params, ctx| {
        matmul_reference(&cublas_handle, params, ctx)
    });
    // 66x
    let params = linalg::FusedMMP::new(1024, 1024, 1024).stride_a(32);
    benchmark::<linalg::FusedMM<f32>, _>(params, &executor, |_, _| {
        7.1e8 // Obtained from a cuda program.
    });
    // 0.52/4H
    let params = linalg::FusedMMP::new(128, 1024, 1024)
        .static_sizes()
        .transpose_a();
    benchmark::<linalg::FusedMM<f32>, _>(params, &executor, |params, ctx| {
        matmul_reference(&cublas_handle, params, ctx)
    });
    // 0.41, 0.53 with TA+Static
    let params = linalg::FusedMMP::new(128, 16384, 4096)
        .static_sizes()
        .transpose_a();
    benchmark::<linalg::FusedMM<f32>, _>(params, &executor, |params, ctx| {
        matmul_reference(&cublas_handle, params, ctx)
    });
    // 0.87 in 2.38 hours/4H
    let mut params = linalg::FusedMMP::new(1024, 1024, 1024);
    params.m_tiling = Some(telamon::helper::TilingPattern::new_fixed(&[32, 4]));
    params.n_tiling = Some(telamon::helper::TilingPattern::new_fixed(&[32, 4]));
    params.k_tiling = Some(telamon::helper::TilingPattern::new_fixed(&[32]));
    benchmark::<linalg::FusedMM<f32>, _>(params, &executor, |params, ctx| {
        matmul_reference(&cublas_handle, params, ctx)
    });
    // 1.66 if reuseb + static sizes
    let params = linalg::BatchMMP::new(512, 32, 32, 64)
        .static_sizes()
        .reuse_b();
    benchmark::<linalg::BatchMM<f32>, _>(params, &executor, |params, ctx| {
        batchmm_reference(&cublas_handle, params, ctx)
    });
    // 0.94 if not transposed in 20min
    let params = linalg::BatchMMP::new(512, 32, 32, 64).static_sizes();
    benchmark::<linalg::BatchMM<f32>, _>(params, &executor, |params, ctx| {
        batchmm_reference(&cublas_handle, params, ctx)
    });
    // 0.60 if not transposed in 20min
    let params = linalg::BatchMMP::new(500, 26, 26, 72).static_sizes();
    benchmark::<linalg::BatchMM<f32>, _>(params, &executor, |params, ctx| {
        batchmm_reference(&cublas_handle, params, ctx)
    });
    // 0.55 perf, with exhaustive search
    /*benchmark::<linalg::MatVec<f32>, _>((1<<13, 1<<13, true), &executor, |params, ctx| {
        matvec_reference(&cublas_handle, params, ctx)
    });
    // 0.31 perf, with exhaustive search
    benchmark::<linalg::Gesummv<f32>, _>((1<<13, 1<<13, true), &executor, |params, ctx| {
        gesummv_reference(&cublas_handle, params, ctx)
    });*/
    // FIXME: add more input sizes for benchmarks
    // - non-powers of 2
    // - repeat B
     */
}
