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
    benchmark::<linalg::Axpy<f32>, _>(1 << 25, &executor, |context| {
        saxpy_reference(&cublas_handle, context)
    });
}

/// The number of times to run the generated code to evaluate its performance.
const NUM_CODE_RUNS: usize = 40;

/// Benchamrks a kernel against a reference implementation.
fn benchmark<'a, K, REF>(params: K::Parameters,
                         executor: &'a cuda::Executor,
                         mut reference: REF)
    where K: Kernel<'a>, REF: FnMut(&cuda::Context) -> f64
{
    let mut config = Config::default();
    config.timeout.get_or_insert(120);
    config.distance_to_best.get_or_insert(0.2);

    let mut context = cuda::Context::new(executor);
    let runtime = K::benchmark(&config, params, NUM_CODE_RUNS, &mut context);
    for _ in 0..4 { reference(&context); }
    let ref_runtime = (0..NUM_CODE_RUNS).map(|_| reference(&context)).collect();
    let mean = estimate_mean(runtime, 0.95);
    let ref_mean = estimate_mean(ref_runtime, 0.95);
    println!("{}: {}, reference: {}, speedup: {:.2}",
             K::name(), mean, ref_mean, ref_mean.mean/mean.mean)
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
    time as f64 * 10e6f64
}

unsafe fn get_array<T>(name: &str, context: &cuda::Context) -> *mut T {
    let ptr: *const *mut T = std::mem::transmute(context.get_param(name).raw_ptr());
    *ptr
}

/// Reference implementation for the `Axpy` kernel.
fn saxpy_reference(handle: &CublasHandle, context: &cuda::Context) -> f64 {
    let n = unwrap!(context.param_as_size("n")) as libc::c_int;
    let alpha = context.get_param("alpha").raw_ptr() as *const f32;
    unsafe {
        let x = get_array("x", context);
        let y = get_array("y", context);
        time_cuda(|| check_cublas(cublasSaxpy_v2(handle.0, n, alpha, x, 1, y, 1)))
    }
}

/// Reference implementation for the matrix-vector multiplication.
fn saxpy_reference(handle: &CublasHandle, context: &cuda::Context) -> f64 {
    let n = unwrap!(context.param_as_size("n")) as libc::c_int;
    let alpha = context.get_param("alpha").raw_ptr() as *const f32;
    unsafe {
        let x = get_array("x", context);
        let y = get_array("y", context);
        time_cuda(|| check_cublas(cublasSaxpy_v2(handle.0, n, alpha, x, 1, y, 1)))
    }
}
