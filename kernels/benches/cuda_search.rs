//! Benchmarks the exploration on CUDA gpus.
extern crate env_logger;
extern crate cuda_sys;
extern crate libc;
extern crate num;
extern crate telamon;
extern crate telamon_kernels;
#[macro_use]
extern crate telamon_utils as utils;

use cuda_sys::cublas::*;
use cuda_sys::cuda::*;
use telamon::explorer::config::Config;
use telamon::device::cuda;
use telamon_kernels::{linalg, Kernel};
use telamon_kernels::statistics::estimate_mean;

fn main() {
    env_logger::init();
    let executor = cuda::Executor::init();
    let cublas_handle = CublasHandle::new();
    // 1.5
    benchmark::<linalg::Axpy<f32>, _>((1 << 25, true), &executor, |params, ctx| {
        saxpy_reference(&cublas_handle, params, ctx)
    });
    // 2.82 without tb*/
    let params = linalg::MatMulP::new(128, 256, 32).static_sizes();//.transpose_b();
    benchmark::<linalg::MatMul<f32>, _>(params, &executor, |params, ctx| {
        matmul_reference(&cublas_handle, params, ctx)
    });
    // 66x
    let params = linalg::MatMulP::new(1024, 1024, 1024).stride_a(32);
    benchmark::<linalg::MatMul<f32>, _>(params, &executor, |_, _| {
        7.1e8 // Obtained from a cuda program.
    });
    // 0.52/4H
    let params = linalg::MatMulP::new(128, 1024, 1024).static_sizes().transpose_a();
    benchmark::<linalg::MatMul<f32>, _>(params, &executor, |params, ctx| {
        matmul_reference(&cublas_handle, params, ctx)
    });
    // 0.41, 0.53 with TA+Static
    let params = linalg::MatMulP::new(128, 16384, 4096).static_sizes().transpose_a();
    benchmark::<linalg::MatMul<f32>, _>(params, &executor, |params, ctx| {
        matmul_reference(&cublas_handle, params, ctx)
    });
    // 0.87 in 2.38 hours/4H
    let params = linalg::MatMulP::new(1024, 1024, 1024).static_sizes();
    benchmark::<linalg::MatMul<f32>, _>(params, &executor, |params, ctx| {
        matmul_reference(&cublas_handle, params, ctx)
    });
    // 1.66 if reuseb + static sizes
    let params = linalg::BatchMMP::new(512, 32, 32, 64).static_sizes().reuse_b();
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
}

/// The number of times to run the generated code to evaluate its performance.
const NUM_CODE_RUNS: usize = 40;
/// Search timeout in minutes.
const TIMEOUT: u64 = 240;

/// Benchamrks a kernel against a reference implementation.
fn benchmark<'a, K, REF>(params: K::Parameters,
                         executor: &'a cuda::Executor,
                         mut reference: REF)
    where K: Kernel<'a>, REF: FnMut(&K::Parameters, &cuda::Context) -> f64
{
    let mut config = Config::read_from_file();
    config.timeout.get_or_insert(TIMEOUT);
    //config.distance_to_best.get_or_insert(20.);

    let mut context = cuda::Context::new(executor);
    let runtime = K::benchmark(&config, params.clone(), NUM_CODE_RUNS, &mut context);
    for _ in 0..4 { reference(&params, &context); }
    let ref_runtime = (0..NUM_CODE_RUNS).map(|_| reference(&params, &context)).collect();
    println!("runtimes: {:?}", runtime);
    let mean = estimate_mean(runtime, 0.95, "ns");
    let ref_mean = estimate_mean(ref_runtime, 0.95, "ns");
    println!("{}: {}, reference: {}, speedup: {:.2}",
             K::name(), mean, ref_mean, ref_mean.value/mean.value)
}

/// Checks the cublas status and panics if an error occured.
fn check_cublas(status: cublasStatus_t) {
    if status != cublasStatus_t::SUCCESS {
        panic!("error in cublas: {:?}", status);
    }
}

/// Checks a cuda status and panics if an error occured.
fn check_cuda(status: CUresult) {
    if status != cudaError_t::CUDA_SUCCESS { panic!("error in cuda: {:?}", status) }
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
        unsafe { check_cublas(cublasDestroy_v2(self.0)); }
    }
}

/// Evaluates the runtime of a cuda function with events.
unsafe fn time_cuda<F: FnOnce()>(f: F) -> f64 {
    let mut start = std::mem::uninitialized();
    let mut stop = std::mem::uninitialized();
    check_cuda(cuEventCreate(&mut start, CUevent_flags_enum::CU_EVENT_DEFAULT as _));
    check_cuda(cuEventCreate(&mut stop, CUevent_flags_enum::CU_EVENT_DEFAULT as _));
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
fn saxpy_reference(handle: &CublasHandle,
                   &(n, _): &(i32, bool),
                   context: &cuda::Context) -> f64 {
    let n = n as libc::c_int;
    let alpha = context.get_param("alpha").raw_ptr() as *const f32;
    unsafe {
        let x = get_array("x", context);
        let y = get_array("y", context);
        time_cuda(|| check_cublas(cublasSaxpy_v2(handle.0, n, alpha, x, 1, y, 1)))
    }
}

/// Reference implementation for the matrix-vector multiplication.
fn matvec_reference(handle: &CublasHandle,
                    &(m, n, _): &(i32, i32, bool),
                    context: &cuda::Context) -> f64 {

    let m = m as libc::c_int;
    let n = n as libc::c_int;
    unsafe {
        let x = get_array("x", context);
        let a = get_array("a", context);
        let y = get_array("y", context);
        time_cuda(|| {
            let op = cublasOperation_t_CUBLAS_OP_T;
            check_cublas(cublasSgemv_v2(handle.0, op, n, m, &2., a, n, x, 1, &3., y, 1))
        })
    }
}

/// Reference implementation for the matrix-matrix multiplication.
fn matmul_reference(handle: &CublasHandle,
                    params: &linalg::MatMulP,
                    context: &cuda::Context) -> f64 {
    let m = params.m as libc::c_int;
    let n = params.n as libc::c_int;
    let k = params.k as libc::c_int;
    assert!(params.a_stride == 1);
    unsafe {
        let a = get_array("a", context);
        let b = get_array("b", context);
        let c = get_array("c", context);
        let (op_a, lda) = if params.transpose_a { (CUBLAS_T, m) } else { (CUBLAS_N, k) };
        let (op_b, ldb) = if params.transpose_b { (CUBLAS_T, k) } else { (CUBLAS_N, n) };
        time_cuda(|| {
            check_cublas(cublasSgemm_v2(
                    handle.0, op_b, op_a, n, m, k, &2., b, ldb, a, lda, &3., c, n));
        })
    }
}

/// Reference implementation for the matrix-matrix multiplication.
fn batchmm_reference(handle: &CublasHandle,
                     params: &linalg::BatchMMP,
                     context: &cuda::Context) -> f64 {
    let m = params.m as libc::c_int;
    let n = params.n as libc::c_int;
    let k = params.k as libc::c_int;
    let batch = params.batch as libc::c_int;
    unsafe {
        let a = get_array("a", context);
        let b = get_array("b", context);
        let c = get_array("c", context);
        let (op_a, lda) = if params.transpose_a { (CUBLAS_T, m) } else { (CUBLAS_N, k) };
        let (op_b, ldb) = if params.transpose_b { (CUBLAS_T, k) } else { (CUBLAS_N, n) };
        let stride_a = (m*k) as libc::c_long;
        let stride_b = if params.batch_b {  n*k } else { 0 } as libc::c_long;
        let stride_c = (m*n) as libc::c_long;
        time_cuda(|| {
            check_cublas(cublasSgemmStridedBatched(
                    handle.0, op_b, op_a, n, m, k, &2.,
                    b, ldb, stride_b,
                    a, lda, stride_a, &3.,
                    c, n, stride_c, batch));
        })
    }
}

/// Reference implementation for the `Gesummv` params.
fn gesummv_reference(handle: &CublasHandle,
                     &(m, n, _): &(i32, i32, bool),
                     context: &cuda::Context) -> f64 {
    let m = m as libc::c_int;
    let n = n as libc::c_int;
    unsafe {
        let a = get_array("a", context);
        let b = get_array("b", context);
        let x = get_array("x", context);
        let y = get_array("y", context);
        time_cuda(|| {
            let op = cublasOperation_t_CUBLAS_OP_T;
            check_cublas(cublasSgemv_v2(handle.0, op, n, m, &3.1, a, n, x, 1, &0., y, 1));
            check_cublas(cublasSgemv_v2(handle.0, op, n, m, &4.1, b, n, x, 1, &1., y, 1));
        })
    }
}
