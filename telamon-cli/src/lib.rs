use std::error::Error;
use std::ffi::OsStr;
use std::path::PathBuf;
use std::sync::Arc;
use std::{fmt, fs, io};

use structopt::StructOpt;

use telamon::device::{ArgMap, Context};
use telamon::explorer::{choice::ActionEx as Action, config::Config, Candidate};
use telamon_kernels::{linalg, Kernel, KernelBuilder};

#[derive(StructOpt)]
pub struct CommonOpt {
    /// Path to the configuration file to use.
    ///
    /// Configuration file must be in TOML format.
    #[structopt(parse(from_os_str), long = "config")]
    config_path: Option<PathBuf>,
}

impl CommonOpt {
    pub fn config(&self) -> io::Result<Config> {
        if let Some(config_path) = &self.config_path {
            Config::from_path(config_path)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e))
        } else {
            Ok(Config::default())
        }
    }
}

pub trait ContextBuilder<'a> {
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

pub trait Reference<'a, K>
where
    K: Kernel<'a>,
{
    type Context: Context + 'a;

    fn eval_reference(&self, params: &K::Parameters, context: &Self::Context) -> f64;
}

#[derive(Debug, Clone)]
pub struct Bench {
    warmup: usize,
    runs: usize,
}

impl Default for Bench {
    fn default() -> Self {
        Bench {
            warmup: 4,
            runs: 40,
        }
    }
}

impl Bench {
    pub fn warmup(mut self, warmup: usize) -> Self {
        self.warmup = warmup;
        self
    }

    pub fn runs(mut self, runs: usize) -> Self {
        self.runs = runs;
        self
    }

    pub fn benchmark_fn<F>(&self, f: F) -> Vec<f64>
    where
        F: Fn() -> f64,
    {
        for _ in 0..self.warmup {
            f();
        }

        (0..self.runs).map(|_| f()).collect()
    }
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

    /// Reference implementation for `Gesummv`.
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

    /// Reference implementation for `ResNetCell`.
    fn resnetcell_reference(
        handle: &CublasHandle,
        params: &linalg::ResNetCellP,
        context: &cuda::Context,
    ) -> f64 {
        panic!("NOT IMPLEMENTED!");
    }

    /// Reference implementation for the transformer cell.
    fn transformercell_reference(
        handle: &CublasHandle,
        params: &linalg::TransformerCellP,
        context: &cuda::Context,
    ) -> f64 {
        panic!("NOT IMPLEMENTED!");
    }

    /// Reference implementation for the first operations of the
    /// transformer cell.
    fn transformercelltophalf_reference(
        handle: &CublasHandle,
        params: &linalg::TransformerCellTopHalfP,
        context: &cuda::Context,
    ) -> f64 {
        panic!("NOT IMPLEMENTED");
    }

    /// Reference implementation for the last operations of the
    /// transformer cell.
    fn transformercellbottomhalf_reference(
        handle: &CublasHandle,
        params: &linalg::TransformerCellBottomHalfP,
        context: &cuda::Context,
    ) -> f64 {
        panic!("NOT IMPLEMENTED");
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

    impl<'a> Reference<'a, linalg::ResNetCell<'a, f32>> for CublasHandle {
        type Context = cuda::Context<'a>;

        fn eval_reference(
            &self,
            params: &linalg::ResNetCellP,
            context: &Self::Context,
        ) -> f64 {
            resnetcell_reference(self, params, context)
        }
    }

    impl<'a> Reference<'a, linalg::TransformerCell<'a, f32>> for CublasHandle {
        type Context = cuda::Context<'a>;

        fn eval_reference(
            &self,
            params: &linalg::TransformerCellP,
            context: &Self::Context,
        ) -> f64 {
            transformercell_reference(self, params, context)
        }
    }

    impl<'a> Reference<'a, linalg::TransformerCellTopHalf<'a, f32>> for CublasHandle {
        type Context = cuda::Context<'a>;

        fn eval_reference(
            &self,
            params: &linalg::TransformerCellTopHalfP,
            context: &Self::Context,
        ) -> f64 {
            transformercelltophalf_reference(self, params, context)
        }
    }

    impl<'a> Reference<'a, linalg::TransformerCellBottomHalf<'a, f32>> for CublasHandle {
        type Context = cuda::Context<'a>;

        fn eval_reference(
            &self,
            params: &linalg::TransformerCellBottomHalfP,
            context: &Self::Context,
        ) -> f64 {
            transformercellbottomhalf_reference(self, params, context)
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda_reference::CublasHandle;

/// Helper enum to create the supported kernel parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelParam {
    Axpy {
        n: i32,
    },
    MatVec {
        m: i32,
        n: i32,
    },
    Gesummv {
        m: i32,
        n: i32,
    },
    Gemm {
        m: i32,
        n: i32,
        k: i32,
    },
    BatchMM {
        b: i32,
        m: i32,
        n: i32,
        k: i32,
    },
    ResNetCell {
        m: i32,
        n: i32,
        k: i32,
        activation_fun: Option<linalg::compose::ActivationFunction>,
    },
    TransformerCell {
        m: i32,
        n: i32,
        p: i32,
        r: i32,
    },
    TransformerCellTopHalf {
        m: i32,
        n: i32,
        p: i32,
    },
    TransformerCellBottomHalf {
        m: i32,
        n: i32,
        r: i32,
    },
}

impl KernelParam {
    /// Build the kernel in a given context, and return a list of candidates.
    pub fn build<'a, 'b, C>(&self, context: &'b mut C) -> (Vec<Candidate>, &'b C)
    where
        C: Context + ArgMap<'a>,
        'a: 'b,
    {
        fn build<'a, 'b, K, C>(
            params: K::Parameters,
            context: &'b mut C,
        ) -> (Vec<Candidate>, &'b C)
        where
            K: Kernel<'a> + 'b,
            C: Context + ArgMap<'a>,
        {
            let (signature, kernel, context) =
                KernelBuilder::default().build::<K, C>(params, context);
            let signature = Arc::new(signature);
            (kernel.build_body(signature, context), context)
        }

        match *self {
            KernelParam::Axpy { n } => {
                build::<'_, '_, linalg::Axpy<'_, f32>, C>((n, true), context)
            }
            KernelParam::MatVec { m, n } => {
                build::<'_, '_, linalg::MatVec<'_, f32>, C>((m, n, true), context)
            }
            KernelParam::Gesummv { m, n } => {
                build::<'_, '_, linalg::Gesummv<'_, f32>, C>((m, n, true), context)
            }
            KernelParam::Gemm { m, n, k } => {
                build::<'_, '_, linalg::FusedMM<'_, f32>, C>(
                    linalg::FusedMMP::new(m, n, k),
                    context,
                )
            }
            KernelParam::BatchMM { b, m, n, k } => {
                build::<'_, '_, linalg::BatchMM<'_, f32>, C>(
                    linalg::BatchMMP::new(b, m, n, k),
                    context,
                )
            }
            KernelParam::ResNetCell {
                m,
                n,
                k,
                activation_fun,
            } => build::<'_, '_, linalg::ResNetCell<'_, f32>, C>(
                linalg::ResNetCellP::new(m, n, k).activation_fun(activation_fun),
                context,
            ),
            KernelParam::TransformerCell { m, n, p, r } => {
                build::<'_, '_, linalg::TransformerCell<'_, f32>, C>(
                    linalg::TransformerCellP::new(m, n, p, r),
                    context,
                )
            }
            KernelParam::TransformerCellTopHalf { m, n, p } => {
                build::<'_, '_, linalg::TransformerCellTopHalf<'_, f32>, C>(
                    linalg::TransformerCellTopHalfP::new(m, n, p),
                    context,
                )
            }
            KernelParam::TransformerCellBottomHalf { m, n, r } => {
                build::<'_, '_, linalg::TransformerCellBottomHalf<'_, f32>, C>(
                    linalg::TransformerCellBottomHalfP::new(m, n, r),
                    context,
                )
            }
        }
    }
}

impl fmt::Display for KernelParam {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            KernelParam::Axpy { n } => write!(fmt, "axpy_{}", n),
            KernelParam::MatVec { m, n } => write!(fmt, "matvec_{}_{}", m, n),
            KernelParam::Gesummv { m, n } => write!(fmt, "gesummv_{}_{}", m, n),
            KernelParam::Gemm { m, n, k } => write!(fmt, "matmul_{}_{}_{}", m, n, k),
            KernelParam::BatchMM { b, m, n, k } => {
                write!(fmt, "batchmm_{}_{}_{}_{}", b, m, n, k)
            }
            KernelParam::ResNetCell {
                m,
                n,
                k,
                activation_fun,
            } => write!(
                fmt,
                "resnetcell_{}_{}_{}_{}",
                m,
                n,
                k,
                linalg::compose::ActivationFunction::opt_to_display(activation_fun)
            ),
            KernelParam::TransformerCell { m, n, p, r } => {
                write!(fmt, "transformercell_{}_{}_{}_{}", m, n, p, r)
            }
            KernelParam::TransformerCellTopHalf { m, n, p } => {
                write!(fmt, "transformercelltophalf_{}_{}_{}", m, n, p)
            }
            KernelParam::TransformerCellBottomHalf { m, n, r } => {
                write!(fmt, "transformercellbottomhalf_{}_{}_{}", m, n, r)
            }
        }
    }
}

/// An error which can be returned when parsing a kernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseKernelError {
    kind: KernelErrorKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelErrorKind {
    /// Value being parsed is empty.
    ///
    /// This variant will be constructed when parsing an empty string.
    Empty,

    /// Invalid kernel name provided.
    InvalidName,

    /// Kernel name is too short and a parameter was missing
    MissingParameter,

    /// Kernel name is too long and has extra parameters.
    UnexpectedParameter,

    /// A non-integer value was found where an integer value was expected.
    IntError(std::num::ParseIntError),

    /// An invalid activation function was found.
    ActivationFunctionError(linalg::compose::ActivationFunctionParseError),
}

impl ParseKernelError {
    /// Outputs the detailed cause of parsing a kernel failing
    pub fn kind(&self) -> &KernelErrorKind {
        &self.kind
    }
}

impl fmt::Display for ParseKernelError {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            KernelErrorKind::Empty => {
                fmt.write_str("cannot parse kernel from empty string")
            }
            KernelErrorKind::InvalidName => fmt.write_str("invalid kernel name"),
            KernelErrorKind::MissingParameter => {
                fmt.write_str("missing kernel parameter")
            }
            KernelErrorKind::UnexpectedParameter => {
                fmt.write_str("extraneous unexpected kernel parameter")
            }
            KernelErrorKind::IntError(error) => fmt::Display::fmt(error, fmt),
            KernelErrorKind::ActivationFunctionError(error) => {
                fmt::Display::fmt(error, fmt)
            }
        }
    }
}

impl Error for ParseKernelError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match &self.kind {
            KernelErrorKind::IntError(error) => Some(error),
            _ => None,
        }
    }
}

impl From<std::num::ParseIntError> for ParseKernelError {
    fn from(error: std::num::ParseIntError) -> ParseKernelError {
        ParseKernelError {
            kind: KernelErrorKind::IntError(error),
        }
    }
}

impl From<telamon_kernels::compose::ActivationFunctionParseError> for ParseKernelError {
    fn from(
        error: telamon_kernels::compose::ActivationFunctionParseError,
    ) -> ParseKernelError {
        ParseKernelError {
            kind: KernelErrorKind::ActivationFunctionError(error),
        }
    }
}

impl std::str::FromStr for KernelParam {
    type Err = ParseKernelError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        use KernelParam::*;

        fn parse_i32(s: &str) -> Result<i32, std::num::ParseIntError> {
            if let Some(pos) = s.find("p") {
                let (base, exp) = s.split_at(pos);
                Ok(base.parse::<i32>()?.pow(exp[1..].parse::<u32>()?))
            } else {
                s.parse::<i32>()
            }
        }

        fn next_part<'a, I>(parts: &mut I) -> Result<&'a str, ParseKernelError>
        where
            I: Iterator<Item = &'a str>,
        {
            parts.next().ok_or(ParseKernelError {
                kind: KernelErrorKind::MissingParameter,
            })
        }

        let mut parts = s.split('_');
        let name = next_part(&mut parts)?;

        let result = match name {
            "axpy" => {
                let n = parse_i32(next_part(&mut parts)?)?;
                Axpy { n }
            }
            "matvec" => {
                let m = parse_i32(next_part(&mut parts)?)?;
                let n = parse_i32(next_part(&mut parts)?)?;
                MatVec { m, n }
            }
            "gesummv" => {
                let m = parse_i32(next_part(&mut parts)?)?;
                let n = parse_i32(next_part(&mut parts)?)?;
                Gesummv { m, n }
            }
            "matmul" => {
                let m = parse_i32(next_part(&mut parts)?)?;
                let n = parse_i32(next_part(&mut parts)?)?;
                let k = parse_i32(next_part(&mut parts)?)?;
                Gemm { m, n, k }
            }
            "batchmm" => {
                let b = parse_i32(next_part(&mut parts)?)?;
                let m = parse_i32(next_part(&mut parts)?)?;
                let n = parse_i32(next_part(&mut parts)?)?;
                let k = parse_i32(next_part(&mut parts)?)?;
                BatchMM { b, m, n, k }
            }
            "resnetcell" => {
                let m = parse_i32(next_part(&mut parts)?)?;
                let n = parse_i32(next_part(&mut parts)?)?;
                let k = parse_i32(next_part(&mut parts)?)?;
                let activation_fun =
                    linalg::compose::ActivationFunction::opt_from_string(next_part(
                        &mut parts,
                    )?)?;
                ResNetCell {
                    m,
                    n,
                    k,
                    activation_fun,
                }
            }
            "transformercell" => {
                let m = parse_i32(next_part(&mut parts)?)?;
                let n = parse_i32(next_part(&mut parts)?)?;
                let p = parse_i32(next_part(&mut parts)?)?;
                let r = parse_i32(next_part(&mut parts)?)?;
                TransformerCell { m, n, p, r }
            }
            "transformercelltophalf" => {
                let m = parse_i32(next_part(&mut parts)?)?;
                let n = parse_i32(next_part(&mut parts)?)?;
                let p = parse_i32(next_part(&mut parts)?)?;
                TransformerCellTopHalf { m, n, p }
            }
            "transformercellbottomhalf" => {
                let m = parse_i32(next_part(&mut parts)?)?;
                let n = parse_i32(next_part(&mut parts)?)?;
                let r = parse_i32(next_part(&mut parts)?)?;
                TransformerCellBottomHalf { m, n, r }
            }
            _ => {
                return Err(ParseKernelError {
                    kind: KernelErrorKind::InvalidName,
                })
            }
        };

        if parts.next().is_some() {
            Err(ParseKernelError {
                kind: KernelErrorKind::UnexpectedParameter,
            })
        } else {
            Ok(result)
        }
    }
}

/// Path to a replay file.
///
/// Replay files are .json files containing a serialized representation of actions to apply.  They
/// can be generated by the debugger or the replay tests.
///
/// This is a thin wrapper around a `PathBuf` which provides convenience functions to load the
/// actual actions.
#[derive(Debug)]
pub struct ReplayPath(PathBuf);

impl From<&'_ OsStr> for ReplayPath {
    fn from(os_str: &'_ OsStr) -> ReplayPath {
        ReplayPath(os_str.into())
    }
}

impl ReplayPath {
    /// Load the replay and returns the corresponding actions.
    ///
    /// If no replay path was provided, an empty vector is returned.
    pub fn load(&self) -> io::Result<Vec<Action>> {
        Ok(serde_json::from_reader(fs::File::open(&self.0)?)?)
    }
}
