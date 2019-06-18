//! Benchmarks the exploration on CUDA gpus.
use std::io::{self, Write};

use telamon::device::{ArgMap, Context};
use telamon::explorer::config::Config;
use telamon::helper::MemInit;
use telamon_kernels::statistics::estimate_mean;
use telamon_kernels::{linalg, Kernel};

use structopt::StructOpt;

trait ContextBuilder<'a> {
    type Context: ArgMap<'a>;

    fn build_context(&self) -> Self::Context;
}

#[cfg(feature = "cuda")]
impl<'a> ContextBuilder<'a> for &'a telamon_cuda::Executor {
    type Context = telamon_cuda::Context<'a>;

    fn build_context(&self) -> Self::Context {
        telamon_cuda::Context::new(self)
    }
}

trait Reference<'a, K>
where
    K: Kernel<'a>,
{
    type Context: Context + 'a;

    fn eval_reference(&self, params: &K::Parameters, context: &Self::Context) -> f64;
}

#[cfg(feature = "cuda")]
mod cuda_reference {
    use cuda_sys::cublas::*;
    use cuda_sys::cuda::*;
    use telamon_cuda as cuda;
    use telamon_kernels::linalg;

    use super::Reference;

    /// Checks the cublas status and panics if an error occured.
    fn check_cublas(status: cublasStatus_t) {
        if status != cublasStatus_t::SUCCESS {
            panic!("error in cublas: {:?}", status);
        }
    }

    /// Checks a cuda status and panics if an error occured.
    fn check_cuda(status: CUresult) {
        if status != cudaError_t::CUDA_SUCCESS {
            panic!("error in cuda: {:?}", status)
        }
    }

    pub struct CublasHandle(cublasHandle_t);

    impl CublasHandle {
        /// Initialize a new handle.
        pub fn new() -> Self {
            unsafe {
                let mut handle = std::mem::uninitialized();
                check_cublas(cublasCreate_v2(&mut handle));
                CublasHandle(handle)
            }
        }
    }

    impl Drop for CublasHandle {
        fn drop(&mut self) {
            unsafe {
                check_cublas(cublasDestroy_v2(self.0));
            }
        }
    }

    /// Evaluates the runtime of a cuda function with events.
    unsafe fn time_cuda<F: FnOnce()>(f: F) -> f64 {
        let mut start = std::mem::uninitialized();
        let mut stop = std::mem::uninitialized();
        check_cuda(cuEventCreate(
            &mut start,
            CUevent_flags_enum::CU_EVENT_DEFAULT as _,
        ));
        check_cuda(cuEventCreate(
            &mut stop,
            CUevent_flags_enum::CU_EVENT_DEFAULT as _,
        ));
        check_cuda(cuCtxSynchronize());
        check_cuda(cuEventRecord(start, std::ptr::null_mut()));
        f();
        check_cuda(cuEventRecord(stop, std::ptr::null_mut()));
        check_cuda(cuEventSynchronize(stop));
        let mut time = 0f32;
        check_cuda(cuEventElapsedTime(&mut time, start, stop));
        check_cuda(cuEventDestroy_v2(start));
        check_cuda(cuEventDestroy_v2(stop));
        time as f64 * 1.0e6f64
    }

    unsafe fn get_array<T>(name: &str, context: &cuda::Context) -> *mut T {
        let ptr: *const *mut T = std::mem::transmute(context.get_param(name).raw_ptr());
        *ptr
    }

    const CUBLAS_N: cublasOperation_t = cublasOperation_t_CUBLAS_OP_N;
    const CUBLAS_T: cublasOperation_t = cublasOperation_t_CUBLAS_OP_T;

    /// Reference implementation for the `Axpy` kernel.
    fn saxpy_reference(
        handle: &CublasHandle,
        &(n, _): &(i32, bool),
        context: &cuda::Context,
    ) -> f64 {
        let n = n as libc::c_int;
        let alpha = context.get_param("alpha").raw_ptr() as *const f32;
        unsafe {
            let x = get_array("x", context);
            let y = get_array("y", context);
            time_cuda(|| check_cublas(cublasSaxpy_v2(handle.0, n, alpha, x, 1, y, 1)))
        }
    }

    /// Reference implementation for the matrix-vector multiplication.
    fn matvec_reference(
        handle: &CublasHandle,
        &(m, n, _): &(i32, i32, bool),
        context: &cuda::Context,
    ) -> f64 {
        let m = m as libc::c_int;
        let n = n as libc::c_int;
        unsafe {
            let x = get_array("x", context);
            let a = get_array("a", context);
            let y = get_array("y", context);
            time_cuda(|| {
                let op = cublasOperation_t_CUBLAS_OP_T;
                check_cublas(cublasSgemv_v2(
                    handle.0, op, n, m, &2., a, n, x, 1, &3., y, 1,
                ))
            })
        }
    }

    /// Reference implementation for the matrix-matrix multiplication.
    fn matmul_reference(
        handle: &CublasHandle,
        params: &linalg::FusedMMP,
        context: &cuda::Context,
    ) -> f64 {
        let m = params.m as libc::c_int;
        let n = params.n as libc::c_int;
        let k = params.k as libc::c_int;
        assert!(params.a_stride == 1);
        unsafe {
            let a = get_array("a", context);
            let b = get_array("b", context);
            let c = get_array("c", context);
            let (op_a, lda) = if params.transpose_a {
                (CUBLAS_T, m)
            } else {
                (CUBLAS_N, k)
            };
            let (op_b, ldb) = if params.transpose_b {
                (CUBLAS_T, k)
            } else {
                (CUBLAS_N, n)
            };
            time_cuda(|| {
                check_cublas(cublasSgemm_v2(
                    handle.0, op_b, op_a, n, m, k, &2., b, ldb, a, lda, &3., c, n,
                ));
            })
        }
    }

    /// Reference implementation for the matrix-matrix multiplication.
    fn batchmm_reference(
        handle: &CublasHandle,
        params: &linalg::BatchMMP,
        context: &cuda::Context,
    ) -> f64 {
        let m = params.m as libc::c_int;
        let n = params.n as libc::c_int;
        let k = params.k as libc::c_int;
        let batch = params.batch as libc::c_int;
        unsafe {
            let a = get_array("a", context);
            let b = get_array("b", context);
            let c = get_array("c", context);
            let (op_a, lda) = if params.transpose_a {
                (CUBLAS_T, m)
            } else {
                (CUBLAS_N, k)
            };
            let (op_b, ldb) = if params.transpose_b {
                (CUBLAS_T, k)
            } else {
                (CUBLAS_N, n)
            };
            let stride_a = (m * k) as libc::c_long;
            let stride_b = if params.batch_b { n * k } else { 0 } as libc::c_long;
            let stride_c = (m * n) as libc::c_long;
            time_cuda(|| {
                check_cublas(cublasSgemmStridedBatched(
                    handle.0, op_b, op_a, n, m, k, &2., b, ldb, stride_b, a, lda,
                    stride_a, &3., c, n, stride_c, batch,
                ));
            })
        }
    }

    /// Reference implementation for the `Gesummv` params.
    fn gesummv_reference(
        handle: &CublasHandle,
        &(m, n, _): &(i32, i32, bool),
        context: &cuda::Context,
    ) -> f64 {
        let m = m as libc::c_int;
        let n = n as libc::c_int;
        unsafe {
            let a = get_array("a", context);
            let b = get_array("b", context);
            let x = get_array("x", context);
            let y = get_array("y", context);
            time_cuda(|| {
                let op = cublasOperation_t_CUBLAS_OP_T;
                check_cublas(cublasSgemv_v2(
                    handle.0, op, n, m, &3.1, a, n, x, 1, &0., y, 1,
                ));
                check_cublas(cublasSgemv_v2(
                    handle.0, op, n, m, &4.1, b, n, x, 1, &1., y, 1,
                ));
            })
        }
    }

    impl<'a> Reference<'a, linalg::Axpy<'a, f32>> for CublasHandle {
        type Context = cuda::Context<'a>;

        fn eval_reference(&self, params: &(i32, bool), context: &Self::Context) -> f64 {
            saxpy_reference(self, params, context)
        }
    }

    impl<'a> Reference<'a, linalg::MatVec<'a, f32>> for CublasHandle {
        type Context = cuda::Context<'a>;

        fn eval_reference(
            &self,
            params: &(i32, i32, bool),
            context: &Self::Context,
        ) -> f64 {
            matvec_reference(self, params, context)
        }
    }

    impl<'a> Reference<'a, linalg::FusedMM<'a, f32>> for CublasHandle {
        type Context = cuda::Context<'a>;

        fn eval_reference(
            &self,
            params: &linalg::FusedMMP,
            context: &Self::Context,
        ) -> f64 {
            matmul_reference(self, params, context)
        }
    }

    impl<'a> Reference<'a, linalg::BatchMM<'a, f32>> for CublasHandle {
        type Context = cuda::Context<'a>;

        fn eval_reference(
            &self,
            params: &linalg::BatchMMP,
            context: &Self::Context,
        ) -> f64 {
            batchmm_reference(self, params, context)
        }
    }

    impl<'a> Reference<'a, linalg::Gesummv<'a, f32>> for CublasHandle {
        type Context = cuda::Context<'a>;

        fn eval_reference(
            &self,
            params: &(i32, i32, bool),
            context: &Self::Context,
        ) -> f64 {
            gesummv_reference(self, params, context)
        }
    }
}

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
    for _ in 0..4 {
        reference.eval_reference(&params, &context);
    }
    let ref_runtime = (0..NUM_CODE_RUNS)
        .map(|_| reference.eval_reference(&params, &context))
        .collect();
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

use std::fmt;
use std::path::PathBuf;

#[derive(Debug, Clone)]
enum DeviceKind {
    X86,
    Cuda,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ParseDeviceError {
    _priv: (),
}

impl fmt::Display for ParseDeviceError {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "invalid device kind")
    }
}

impl std::error::Error for ParseDeviceError {}

impl std::str::FromStr for DeviceKind {
    type Err = ParseDeviceError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim() {
            "x86" => Ok(DeviceKind::X86),
            "cuda" => Ok(DeviceKind::Cuda),
            _ => Err(ParseDeviceError { _priv: () }),
        }
    }
}

#[derive(StructOpt)]
struct Opt {
    /// Path to the configuration file to use.
    ///
    /// Configuration file must be in TOML format.
    #[structopt(parse(from_os_str), long = "config")]
    config_path: Option<PathBuf>,

    #[structopt(long = "device", short = "d")]
    device: DeviceKind,
}

impl Opt {
    fn config(&self) -> io::Result<Config> {
        if let Some(config_path) = &self.config_path {
            Config::from_path(config_path)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e))
        } else {
            Ok(Config::default())
        }
    }
}

trait Bench<'a, B, R> {
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

impl<'a, K, B, R> Bench<'a, B, R> for Benchmark<'a, K>
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
    let reference = cuda_reference::CublasHandle::new();

    let config = args.config().unwrap();

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
