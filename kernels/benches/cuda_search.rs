//! Benchmarks the exploration on CUDA gpus.
extern crate env_logger;
extern crate cuda_sys;
extern crate libc;
extern crate telamon;
extern crate telamon_kernels;
#[macro_use]
extern crate telamon_utils as utils;

use cuda_sys::cublas::*;
use cuda_sys::cuda::*;
use telamon::explorer::config::Config;
use telamon::device::{Context, cuda};
use telamon_kernels::{linalg, Kernel};
use telamon_kernels::statistics::estimate_mean;

fn main() {
    env_logger::init();
    let executor = cuda::Executor::init();
    let cublas_handle = CublasHandle::new();
    benchmark::<linalg::Axpy<f32>, _>(1 << 25, &executor, |params, ctx| {
        saxpy_reference(&cublas_handle, params, ctx)
    });
    benchmark::<linalg::MatMul<f32>, _>((1<<10, 1<<10, 1<<10), &executor, |params, ctx| {
        matmul_reference(&cublas_handle, params, ctx)
    });
    // FIXME: 0.5 perf, with exhaustive search
    benchmark::<linalg::MatVec<f32>, _>((1<<13, 1<<13), &executor, |params, ctx| {
        matvec_reference(&cublas_handle, params, ctx)
    });
    // FIXME: 0.28 perf, with exhaustive search
    benchmark::<linalg::Gesummv<f32>, _>((1<<13, 1<<13), &executor, |params, ctx| {
        gesummv_reference(&cublas_handle, params, ctx)
    });
    // FIXME: a bit too fast, increase problem size ?
    benchmark::<linalg::Doitgen<f32>, _>((1<<7, 1<<7, 1<<7), &executor, |params, ctx| {
        doitgen_reference(&cublas_handle, params, ctx)
    });
    // FIXME: add more input sizes for benchmarks
}

/// The number of times to run the generated code to evaluate its performance.
const NUM_CODE_RUNS: usize = 40;
/// Search timeout in minutes.
const TIMEOUT: u64 = 120;

/// Benchamrks a kernel against a reference implementation.
fn benchmark<'a, K, REF>(params: K::Parameters,
                         executor: &'a cuda::Executor,
                         mut reference: REF)
    where K: Kernel<'a>, REF: FnMut(K::Parameters, &cuda::Context) -> f64
{
    let mut config = Config::read_from_file();
    config.timeout.get_or_insert(TIMEOUT);
    config.distance_to_best.get_or_insert(20.);

    let mut context = cuda::Context::new(executor);
    let runtime = K::benchmark(&config, params, NUM_CODE_RUNS, false, &mut context);
    for _ in 0..4 { reference(params, &context); }
    let ref_runtime = (0..NUM_CODE_RUNS).map(|_| reference(params, &context)).collect();
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

/// Reference implementation for the `Axpy` kernel.
fn saxpy_reference(handle: &CublasHandle,
                   n: i32,
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
                    (m, n): (i32, i32),
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
                    (m, n, k): (i32, i32, i32),
                    context: &cuda::Context) -> f64 {
    let m = m as libc::c_int;
    let n = n as libc::c_int;
    let k = k as libc::c_int;
    unsafe {
        let a = get_array("a", context);
        let b = get_array("b", context);
        let c = get_array("c", context);
        time_cuda(|| {
            let op = cublasOperation_t_CUBLAS_OP_N;
            check_cublas(cublasSgemm_v2(
                    handle.0, op, op, n, m, k, &2., b, n, a, k, &3., c, n));
        })
    }
}

/// Reference implementation for the `Gesummv` params.
fn gesummv_reference(handle: &CublasHandle,
                     (m, n): (i32, i32),
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

/// Reference implementation for the `Doitgen` params.
fn doitgen_reference(handle: &CublasHandle,
                     (p, q, r): (i32, i32, i32),
                     context: &cuda::Context) -> f64 {
    let p = p as libc::c_int;
    let q = q as libc::c_int;
    let r = r as libc::c_int;
    unsafe {
        let a = get_array("a", context);
        let x = get_array("x", context);
        let b = get_array("b", context);
        time_cuda(|| {
            let op = cublasOperation_t_CUBLAS_OP_N;
            check_cublas(cublasSgemm_v2(
                    handle.0, op, op, p, r*q, p, &1., x, p, a, p, &0., b, p))
        })
    }
}
