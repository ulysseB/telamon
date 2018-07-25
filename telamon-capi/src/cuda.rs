//! C API wrappers for manipulating Telamon CUDA context through FFI.
use Context;
use env_logger;
use libc;
use std;
use telamon::device::{cuda, ArgMap};
use telamon::ir;

/// Opaque type that contains the mapping of kernel parameters to actual values.
pub struct CudaEnvironment {
    executor: cuda::Executor,
    context: std::mem::ManuallyDrop<cuda::Context<'static>>,
    context_ref: Context,
}

/// Returns a pointer to a CUDA environement. The caller is responsible for deallocating
/// the pointer by calling `telamon_cuda_destroy_environment`.
#[no_mangle]
pub unsafe extern "C" fn telamon_cuda_create_environment() -> *mut CudaEnvironment {
    let _ = env_logger::try_init();
    let mut env = Box::new(CudaEnvironment {
        executor: cuda::Executor::init(),
        context: std::mem::uninitialized(),
        context_ref: std::mem::uninitialized(),
    });
    let context = std::mem::transmute(cuda::Context::new(&env.executor));
    std::ptr::write(&mut env.context, std::mem::ManuallyDrop::new(context));
    std::ptr::write(&mut env.context_ref, Context(&*env.context));
    Box::into_raw(env)
}

/// Destroys a CUDA context.
#[no_mangle]
pub unsafe extern "C" fn telamon_cuda_destroy_context(env: *mut CudaEnvironment) {
    let mut env = Box::from_raw(env);
    std::mem::ManuallyDrop::drop(&mut env.context);
}

/// Returns a pointer to a view of environment generic to all target devices.
#[no_mangle]
pub unsafe extern "C" fn telamon_cuda_get_context(env: *const CudaEnvironment)
    -> *const Context
{
        &(*env).context_ref
}

/// Allocates and binds an array to the given parameter. `size` is given in bytes.
///
/// The allocated array is managed by the context and doesn't need to be explicitely
/// destroyed.
#[no_mangle]
pub unsafe extern "C" fn telamon_cuda_bind_array(env: *mut CudaEnvironment,
                                                 param: *const ir::Parameter,
                                                 size: libc::size_t) {
    (*env).context.bind_array::<i8>(&*param, size);
}

/// Binds an `int8_t` to a parameter.
#[no_mangle]
pub unsafe extern "C" fn telamon_cuda_bind_int8(env: *mut CudaEnvironment,
                                                param: *const ir::Parameter,
                                                value: libc::int8_t) {
    (*env).context.bind_scalar::<i8>(&*param, value);
}

/// Binds an `int16_t` to a parameter.
#[no_mangle]
pub unsafe extern "C" fn telamon_cuda_bind_int16(env: *mut CudaEnvironment,
                                                param: *const ir::Parameter,
                                                value: libc::int16_t) {
    (*env).context.bind_scalar::<i16>(&*param, value);
}

/// Binds an `int32_t` to a parameter.
#[no_mangle]
pub unsafe extern "C" fn telamon_cuda_bind_int32(env: *mut CudaEnvironment,
                                                param: *const ir::Parameter,
                                                value: libc::int32_t) {
    (*env).context.bind_scalar::<i32>(&*param, value);
}

/// Binds an `int64_t` to a parameter.
#[no_mangle]
pub unsafe extern "C" fn telamon_cuda_bind_int64(env: *mut CudaEnvironment,
                                                param: *const ir::Parameter,
                                                value: libc::int64_t) {
    (*env).context.bind_scalar::<i64>(&*param, value);
}

/// Binds a `float` to a parameter.
#[no_mangle]
pub unsafe extern "C" fn telamon_cuda_bind_float(env: *mut CudaEnvironment,
                                                param: *const ir::Parameter,
                                                value: libc::c_float) {
    (*env).context.bind_scalar::<f32>(&*param, value);
}

/// Binds a `double` to a parameter.
#[no_mangle]
pub unsafe extern "C" fn telamon_cuda_bind_double(env: *mut CudaEnvironment,
                                                param: *const ir::Parameter,
                                                value: libc::c_double) {
    (*env).context.bind_scalar::<f64>(&*param, value);
}
