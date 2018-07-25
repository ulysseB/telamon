//! C API wrappers for manipulating Telamon CUDA context through FFI.
use env_logger;
use std;
use telamon::device::cuda;

/// Opaque type that bundles a `telamon::device::cuda::Context` and its executor.
pub struct CudaContext {
    executor: cuda::Executor,
    context: std::mem::ManuallyDrop<cuda::Context<'static>>,
}

/// Returns a pointer to a CUDA context. The CUDA context is not the same than
/// `telamon::device::cuda::Context` and should only be used with the C API functions.
#[no_mangle]
pub unsafe extern "C" fn telamon_cuda_create_context() -> *mut CudaContext {
    let _ = env_logger::try_init();
    let mut c_context = Box::new(CudaContext {
        executor: cuda::Executor::init(),
        context: std::mem::uninitialized(),
    });
    let context = std::mem::transmute(cuda::Context::new(&c_context.executor));
    std::ptr::write(&mut c_context.context, std::mem::ManuallyDrop::new(context));
    Box::into_raw(c_context)
}

/// Destroys a CUDA context.
#[no_mangle]
pub unsafe extern "C" fn telamon_cuda_destroy_context(ctx: *mut CudaContext) {
    let mut ctx = Box::from_raw(ctx);
    std::mem::ManuallyDrop::drop(&mut ctx.context);
}

#[test]
fn create_destroy_context() {
    unsafe {
        let ctx = telamon_cuda_create_context();
        telamon_cuda_destroy_context(ctx);
    }
}
