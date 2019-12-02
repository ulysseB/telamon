#![deny(bare_trait_objects, unused_lifetimes)]
#![allow(clippy::many_single_char_names)]

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

    /// Search timeout (in minutes)
    ///
    /// If provided, overrides the timeout from the configuration file.
    #[structopt(long = "timeout")]
    timeout: Option<u64>,
}

impl CommonOpt {
    pub fn config(&self) -> io::Result<Config> {
        let mut config = if let Some(config_path) = &self.config_path {
            Config::from_path(config_path)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e))
        } else {
            Ok(Config::default())
        }?;

        config.timeout = config.timeout.or(self.timeout);
        Ok(config)
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
        if status != CUBLAS_STATUS_SUCCESS {
            panic!("error in cublas: {:?}", status);
        }
    }

    /// Checks a cuda status and panics if an error occured.
    fn check_cuda(status: CUresult) {
        if status != CUDA_SUCCESS {
            panic!("error in cuda: {:?}", status)
        }
    }

    pub struct CudaHandle {
        cublas_raw: cublasHandle_t,
        cudnn: cudnn::CudnnHandle,
    }

    #[allow(clippy::new_without_default)]
    impl CudaHandle {
        /// Initialize a new handle.
        pub fn new() -> Self {
            unsafe {
                let mut cublas_raw = std::mem::uninitialized();
                check_cublas(cublasCreate_v2(&mut cublas_raw));
                CudaHandle {
                    cublas_raw,
                    cudnn: cudnn::CudnnHandle::new().unwrap(),
                }
            }
        }
    }

    impl Drop for CudaHandle {
        fn drop(&mut self) {
            unsafe {
                check_cublas(cublasDestroy_v2(self.cublas_raw));
            }
        }
    }

    /// Evaluates the runtime of a cuda function with events.
    unsafe fn time_cuda<F: FnOnce()>(f: F) -> f64 {
        let mut start = std::mem::uninitialized();
        let mut stop = std::mem::uninitialized();
        check_cuda(cuEventCreate(&mut start, CU_EVENT_DEFAULT));
        check_cuda(cuEventCreate(&mut stop, CU_EVENT_DEFAULT));
        check_cuda(cuCtxSynchronize());
        check_cuda(cuEventRecord(start, std::ptr::null_mut()));
        f();
        check_cuda(cuEventRecord(stop, std::ptr::null_mut()));
        check_cuda(cuEventSynchronize(stop));
        let mut time = 0f32;
        check_cuda(cuEventElapsedTime(&mut time, start, stop));
        check_cuda(cuEventDestroy_v2(start));
        check_cuda(cuEventDestroy_v2(stop));
        f64::from(time) * 1.0e6f64
    }

    unsafe fn get_array<T>(name: &str, context: &cuda::Context) -> *mut T {
        *(context.get_param(name).raw_ptr() as *const *mut T)
    }

    /// Reference implementation for the `Axpy` kernel.
    fn saxpy_reference(
        handle: &CudaHandle,
        (n, _): (i32, bool),
        context: &cuda::Context,
    ) -> f64 {
        let n = n as libc::c_int;
        let alpha = context.get_param("alpha").raw_ptr() as *const f32;
        unsafe {
            let x = get_array("x", context);
            let y = get_array("y", context);
            time_cuda(|| {
                check_cublas(cublasSaxpy_v2(handle.cublas_raw, n, alpha, x, 1, y, 1))
            })
        }
    }

    /// Reference implementation for the matrix-vector multiplication.
    fn matvec_reference(
        handle: &CudaHandle,
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
                let op = CUBLAS_OP_T;
                let cublas = handle.cublas_raw;
                check_cublas(cublasSgemv_v2(cublas, op, n, m, &2., a, n, x, 1, &3., y, 1))
            })
        }
    }

    /// Reference implementation for the matrix-matrix multiplication.
    fn matmul_reference(
        handle: &CudaHandle,
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
                (CUBLAS_OP_T, m)
            } else {
                (CUBLAS_OP_N, k)
            };
            let (op_b, ldb) = if params.transpose_b {
                (CUBLAS_OP_T, k)
            } else {
                (CUBLAS_OP_N, n)
            };
            time_cuda(|| {
                let cublas = handle.cublas_raw;
                check_cublas(cublasSgemm_v2(
                    cublas, op_b, op_a, n, m, k, &1., b, ldb, a, lda, &0., c, n,
                ));
            })
        }
    }

    fn conv2d_reference(
        handle: &CudaHandle,
        params: &linalg::Conv2dP,
        context: &cuda::Context,
    ) -> cudnn::Result<f64> {
        let input_desc = cudnn::TensorDescriptor::new_4d(
            cudnn::TensorFormat::Nchw,
            cudnn::DataType::Float,
            params.batch,
            params.in_channels,
            params.in_height,
            params.in_width,
        )?;

        let filter_desc = cudnn::FilterDescriptor::new_4d(
            cudnn::DataType::Float,
            cudnn::TensorFormat::Nchw,
            params.out_channels,
            params.in_channels,
            params.filter_height,
            params.filter_width,
        )?;

        let conv_desc = cudnn::ConvolutionDescriptor::new_2d(
            params.pad_h(),
            params.pad_w(),
            1,
            1,
            1,
            1,
            cudnn::ConvolutionMode::CrossCorrelation,
            cudnn::DataType::Float,
        )?;

        let output_desc = cudnn::TensorDescriptor::new_4d(
            cudnn::TensorFormat::Nchw,
            cudnn::DataType::Float,
            params.batch,
            params.out_channels,
            params.out_height(),
            params.out_width(),
        )?;

        let time = unsafe {
            let input = get_array("input", context);
            let filter = get_array("filter", context);
            let output = get_array("output", context);

            time_cuda(|| {
                handle
                    .cudnn
                    .convolution_forward_implicit_gemm(
                        &1f32 as *const f32 as *const _,
                        &input_desc,
                        input,
                        &filter_desc,
                        filter,
                        &conv_desc,
                        &0f32 as *const f32 as *const _,
                        &output_desc,
                        output,
                    )
                    .unwrap()
            })
        };

        Ok(time)
    }

    /// Reference implementation for the matrix-matrix multiplication.
    fn batchmm_reference(
        handle: &CudaHandle,
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
                (CUBLAS_OP_T, m)
            } else {
                (CUBLAS_OP_N, k)
            };
            let (op_b, ldb) = if params.transpose_b {
                (CUBLAS_OP_T, k)
            } else {
                (CUBLAS_OP_N, n)
            };
            let stride_a = libc::c_long::from(m * k);
            let stride_b = libc::c_long::from(if params.batch_b { n * k } else { 0 });
            let stride_c = libc::c_long::from(m * n);
            time_cuda(|| {
                let cublas = handle.cublas_raw;
                check_cublas(cublasSgemmStridedBatched(
                    cublas, op_b, op_a, n, m, k, &1., b, ldb, stride_b, a, lda, stride_a,
                    &0., c, n, stride_c, batch,
                ));
            })
        }
    }

    /// Reference implementation for `Gesummv`.
    fn gesummv_reference(
        handle: &CudaHandle,
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
                let op = CUBLAS_OP_T;
                let cublas = handle.cublas_raw;
                check_cublas(cublasSgemv_v2(
                    cublas, op, n, m, &3.1, a, n, x, 1, &0., y, 1,
                ));
                check_cublas(cublasSgemv_v2(
                    cublas, op, n, m, &4.1, b, n, x, 1, &1., y, 1,
                ));
            })
        }
    }

    impl<'a> Reference<'a, linalg::Axpy<'a, f32>> for CudaHandle {
        type Context = cuda::Context<'a>;

        fn eval_reference(&self, params: &(i32, bool), context: &Self::Context) -> f64 {
            saxpy_reference(self, *params, context)
        }
    }

    impl<'a> Reference<'a, linalg::MatVec<'a, f32>> for CudaHandle {
        type Context = cuda::Context<'a>;

        fn eval_reference(
            &self,
            params: &(i32, i32, bool),
            context: &Self::Context,
        ) -> f64 {
            matvec_reference(self, params, context)
        }
    }

    impl<'a> Reference<'a, linalg::FusedMM<'a, f32>> for CudaHandle {
        type Context = cuda::Context<'a>;

        fn eval_reference(
            &self,
            params: &linalg::FusedMMP,
            context: &Self::Context,
        ) -> f64 {
            matmul_reference(self, params, context)
        }
    }

    impl<'a> Reference<'a, linalg::Conv2d<'a, f32>> for CudaHandle {
        type Context = cuda::Context<'a>;

        fn eval_reference(
            &self,
            params: &linalg::Conv2dP,
            context: &Self::Context,
        ) -> f64 {
            conv2d_reference(self, params, context).unwrap()
        }
    }

    impl<'a> Reference<'a, linalg::BatchMM<'a, f32>> for CudaHandle {
        type Context = cuda::Context<'a>;

        fn eval_reference(
            &self,
            params: &linalg::BatchMMP,
            context: &Self::Context,
        ) -> f64 {
            batchmm_reference(self, params, context)
        }
    }

    impl<'a> Reference<'a, linalg::Gesummv<'a, f32>> for CudaHandle {
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

#[cfg(feature = "cuda")]
pub use cuda_reference::CudaHandle;

#[cfg(feature = "x86")]
mod x86_reference {
    use log::warn;
    use telamon_kernels::linalg;

    use super::Reference;

    #[derive(Default)]
    pub struct X86Reference {
        _priv: (),
    }

    impl<'a> Reference<'a, linalg::Axpy<'a, f32>> for X86Reference {
        type Context = telamon_x86::Context;

        fn eval_reference(&self, _params: &(i32, bool), _context: &Self::Context) -> f64 {
            warn!("x86 reference is not implemented");
            1.
        }
    }

    impl<'a> Reference<'a, linalg::MatVec<'a, f32>> for X86Reference {
        type Context = telamon_x86::Context;

        fn eval_reference(
            &self,
            _params: &(i32, i32, bool),
            _context: &Self::Context,
        ) -> f64 {
            warn!("x86 reference is not implemented");
            1.
        }
    }

    impl<'a> Reference<'a, linalg::Gesummv<'a, f32>> for X86Reference {
        type Context = telamon_x86::Context;

        fn eval_reference(
            &self,
            _params: &(i32, i32, bool),
            _context: &Self::Context,
        ) -> f64 {
            warn!("x86 reference is not implemented");
            1.
        }
    }

    impl<'a> Reference<'a, linalg::FusedMM<'a, f32>> for X86Reference {
        type Context = telamon_x86::Context;

        fn eval_reference(
            &self,
            _params: &linalg::FusedMMP,
            _context: &Self::Context,
        ) -> f64 {
            warn!("x86 reference is not implemented");
            1.
        }
    }

    impl<'a> Reference<'a, linalg::BatchMM<'a, f32>> for X86Reference {
        type Context = telamon_x86::Context;

        fn eval_reference(
            &self,
            _params: &linalg::BatchMMP,
            _context: &Self::Context,
        ) -> f64 {
            warn!("x86 reference is not implemented");
            1.
        }
    }
}

#[cfg(feature = "x86")]
pub use x86_reference::X86Reference;

/// A wrapper type containing a (list of) candidates; a checking function to ensure that an
/// implementation's output is valid, and a reference function to compare to.
pub struct KernelBundle<'a> {
    pub candidates: Vec<Candidate>,
    pub check_fn: Box<dyn Fn(&dyn Context) -> Result<(), String> + Sync + 'a>,
    pub reference_fn: Box<dyn Fn() -> f64 + 'a>,
}

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
        ta: bool,
        tb: bool,
    },
    Conv2d {
        n: i32,
        c: i32,
        h: i32,
        w: i32,
        k: i32,
        r: i32,
        s: i32,
    },
    BatchMM {
        b: i32,
        m: i32,
        n: i32,
        k: i32,
    },
}

impl KernelParam {
    /// Build the kernel in a given context, and returns a list of candidates along with a
    /// correction checking function and a reference function.
    pub fn to_bundle<'a, 'b, C, R>(
        &self,
        context: &'b mut C,
        reference: R,
    ) -> (KernelBundle<'b>, &'b C)
    where
        C: Context + ArgMap<'a>,
        R: Reference<'a, linalg::Axpy<'a, f32>, Context = C>
            + Reference<'a, linalg::MatVec<'a, f32>, Context = C>
            + Reference<'a, linalg::FusedMM<'a, f32>, Context = C>
            + Reference<'a, linalg::BatchMM<'a, f32>, Context = C>
            + Reference<'a, linalg::Gesummv<'a, f32>, Context = C>
            + Reference<'a, linalg::Conv2d<'a, f32>, Context = C>
            + 'b,
        'a: 'b,
    {
        struct Builder<'b, C, R> {
            context: &'b mut C,
            reference: R,
        }

        impl<'b, C, R> Builder<'b, C, R> where {
            fn build<'a, K>(self, params: K::Parameters) -> (KernelBundle<'b>, &'b C)
            where
                K: Kernel<'a> + 'b,
                K::Parameters: 'b,
                C: Context + ArgMap<'a>,
                R: Reference<'a, K, Context = C> + 'b,
            {
                let (signature, kernel, context) =
                    KernelBuilder::default().build::<K, C>(params.clone(), self.context);
                let signature = Arc::new(signature);
                let expected = kernel.get_expected_output(context);
                let candidates = kernel.build_body(signature, context);
                let check_fn =
                    move |context: &dyn Context| kernel.check_result(&expected, context);
                let reference = self.reference;
                let reference_fn = move || {
                    Reference::<'_, K>::eval_reference(&reference, &params, context)
                };

                (
                    KernelBundle {
                        candidates,
                        check_fn: Box::new(check_fn),
                        reference_fn: Box::new(reference_fn),
                    },
                    context,
                )
            }
        }

        let builder = Builder { context, reference };
        match *self {
            KernelParam::Axpy { n } => {
                builder.build::<'_, linalg::Axpy<'_, f32>>((n, true))
            }
            KernelParam::MatVec { m, n } => {
                builder.build::<'_, linalg::MatVec<'_, f32>>((m, n, true))
            }
            KernelParam::Gesummv { m, n } => {
                builder.build::<'_, linalg::Gesummv<'_, f32>>((m, n, true))
            }
            KernelParam::Gemm { m, n, k, ta, tb } => {
                let mut params = linalg::FusedMMP::new(m, n, k);
                if ta {
                    params = params.transpose_a();
                }
                if tb {
                    params = params.transpose_b();
                }
                builder.build::<'_, linalg::FusedMM<'_, f32>>(params)
            }
            KernelParam::Conv2d {
                n,
                c,
                h,
                w,
                k,
                r,
                s,
            } => {
                let params = linalg::Conv2dP {
                    batch: n,
                    in_channels: c,
                    in_height: h,
                    in_width: w,
                    out_channels: k,
                    filter_height: r,
                    filter_width: s,
                };
                builder.build::<'_, linalg::Conv2d<'_, f32>>(params)
            }
            KernelParam::BatchMM { b, m, n, k } => builder
                .build::<'_, linalg::BatchMM<'_, f32>>(linalg::BatchMMP::new(b, m, n, k)),
        }
    }
}

impl fmt::Display for KernelParam {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            KernelParam::Axpy { n } => write!(fmt, "axpy_{}", n),
            KernelParam::MatVec { m, n } => write!(fmt, "matvec_{}_{}", m, n),
            KernelParam::Gesummv { m, n } => write!(fmt, "gesummv_{}_{}", m, n),
            KernelParam::Gemm { m, n, k, ta, tb } => write!(
                fmt,
                "matmul_{}_{}_{}_{}{}",
                m,
                n,
                k,
                if ta { "AT" } else { "A" },
                if tb { "BT" } else { "B" }
            ),
            KernelParam::Conv2d {
                n,
                c,
                h,
                w,
                k,
                r,
                s,
            } => write!(
                fmt,
                "conv2d_{}_{}_{}_{}_{}_{}_{}_VALID",
                n, c, h, w, k, r, s
            ),
            KernelParam::BatchMM { b, m, n, k } => {
                write!(fmt, "batchmm_{}_{}_{}_{}", b, m, n, k)
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

impl std::str::FromStr for KernelParam {
    type Err = ParseKernelError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        use KernelParam::*;

        fn parse_i32(s: &str) -> Result<i32, std::num::ParseIntError> {
            if let Some(pos) = s.find('p') {
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

                let (ta, tb) = match next_part(&mut parts) {
                    Ok("AB") | Err(_) => (false, false),
                    Ok("ATB") => (true, false),
                    Ok("ABT") => (false, true),
                    Ok("ATBT") => (true, true),
                    Ok(_) => {
                        return Err(ParseKernelError {
                            kind: KernelErrorKind::InvalidName,
                        })
                    }
                };

                Gemm { m, n, k, ta, tb }
            }
            "conv2d" => {
                let n = parse_i32(next_part(&mut parts)?)?;
                let c = parse_i32(next_part(&mut parts)?)?;
                let h = parse_i32(next_part(&mut parts)?)?;
                let w = parse_i32(next_part(&mut parts)?)?;
                let k = parse_i32(next_part(&mut parts)?)?;
                let r = parse_i32(next_part(&mut parts)?)?;
                let s = parse_i32(next_part(&mut parts)?)?;
                let _ = match next_part(&mut parts) {
                    Ok("VALID") => (),
                    Err(_) => (),
                    _ => {
                        return Err(ParseKernelError {
                            kind: KernelErrorKind::InvalidName,
                        })
                    }
                };
                Conv2d {
                    n,
                    c,
                    h,
                    w,
                    k,
                    r,
                    s,
                }
            }
            "batchmm" => {
                let b = parse_i32(next_part(&mut parts)?)?;
                let m = parse_i32(next_part(&mut parts)?)?;
                let n = parse_i32(next_part(&mut parts)?)?;
                let k = parse_i32(next_part(&mut parts)?)?;
                BatchMM { b, m, n, k }
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

/// Available platforms for running kernels on.
#[derive(Copy, Clone, Debug)]
pub enum Platform {
    X86,
    Cuda,
    __Unsupported,
}

impl std::str::FromStr for Platform {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "x86" => Platform::X86,
            "cuda" => Platform::Cuda,
            _ => return Err(format!("invalid platform: {}", s)),
        })
    }
}

impl Platform {
    /// Convert the platform into the appropriate context builder.  This initializes any internal
    /// ressources of the platform; for instance, requesting a Cuda context builder will setup the
    /// connection to the GPU.
    pub fn to_builder(self) -> PlatformContextBuilder {
        match self {
            #[cfg(feature = "x86")]
            Platform::X86 => PlatformContextBuilder::X86,
            #[cfg(feature = "cuda")]
            Platform::Cuda => {
                PlatformContextBuilder::Cuda(telamon_cuda::Executor::init())
            }
            _ => panic!("platform is not supported"),
        }
    }
}

pub enum PlatformContextBuilder {
    #[cfg(feature = "x86")]
    X86,
    #[cfg(feature = "cuda")]
    Cuda(telamon_cuda::Executor),
}

impl PlatformContextBuilder {
    /// Create a new context for this platform.
    ///
    /// There can be multiple concurrent contexts on the same platform.
    pub fn build_context(&self) -> PlatformContext<'_> {
        match self {
            #[cfg(feature = "x86")]
            PlatformContextBuilder::X86 => PlatformContext::X86(
                telamon_x86::Context::default(),
                std::marker::PhantomData,
            ),
            #[cfg(feature = "cuda")]
            PlatformContextBuilder::Cuda(executor) => {
                PlatformContext::Cuda(telamon_cuda::Context::new(executor))
            }
        }
    }
}

/// An abstraction over multiple platform's contexts.
pub enum PlatformContext<'a> {
    #[cfg(feature = "x86")]
    X86(telamon_x86::Context, std::marker::PhantomData<&'a ()>),
    #[cfg(feature = "cuda")]
    Cuda(telamon_cuda::Context<'a>),
}

impl<'a> PlatformContext<'a> {
    /// Create a kernel bundle, complete with checking and reference function, for the given kernel
    /// parameters.  Note that all platforms may not support all kernels.
    pub fn kernel_bundle(
        &mut self,
        kernel: &KernelParam,
    ) -> (KernelBundle<'_>, &dyn Context) {
        match self {
            #[cfg(feature = "x86")]
            PlatformContext::X86(context, _) => {
                let (bundle, context) =
                    kernel.to_bundle(context, X86Reference::default());
                (bundle, context as &dyn Context)
            }
            #[cfg(feature = "cuda")]
            PlatformContext::Cuda(context) => {
                let (bundle, context) = kernel.to_bundle(context, CudaHandle::new());
                (bundle, context as &dyn Context)
            }
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

impl From<&'_ str> for ReplayPath {
    fn from(s: &'_ str) -> ReplayPath {
        ReplayPath(s.into())
    }
}

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

    pub fn display(&self) -> std::path::Display<'_> {
        self.0.display()
    }
}
