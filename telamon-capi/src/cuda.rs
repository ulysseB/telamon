//! C API wrappers for manipulating Telamon CUDA context through FFI.
use context::Context;
use std;
use telamon::device;

use failure;

pub struct CudaExecutor(device::cuda::Executor);

/// Create a new CUDA executor.
#[no_mangle]
pub extern "C" fn telamon_cuda_executor_new() -> *mut CudaExecutor {
    unwrap_or_exit!(
        device::cuda::Executor::try_init()
            .map(|executor| Box::into_raw(Box::new(CudaExecutor(executor))))
            .map_err(failure::Error::from),
        null
    )
}

/// Create a CUDA context from an executor.  The context keeps a pointer to the executor and must
/// be freed (using `telamon_context_free`) before freeing the executor.
#[no_mangle]
pub unsafe extern "C" fn telamon_cuda_context_new(
    executor: *const CudaExecutor,
) -> *mut Context {
    exit_if_null!(executor, null);

    Box::into_raw(Box::new(Context::new(device::cuda::Context::new(
        &(*executor).0,
    ))))
}

/// Destroys a CUDA executor.
#[no_mangle]
pub unsafe extern "C" fn telamon_cuda_executor_free(executor: *mut CudaExecutor) {
    std::mem::drop(Box::from_raw(executor));
}
