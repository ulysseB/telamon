//! C API wrappers for manipulating Telamon CUDA context through FFI.
use Context;
use Device;
use env_logger;
use libc;
use std;
use telamon::device::{cuda, ArgMap};
use telamon::ir;

/// Opaque type that contains the mapping of kernel parameters to actual values.
pub struct CudaEnvironment {
    executor: cuda::Executor,
    // `ManuallyDrop` is necessary to control the order in wich `drop` is called on the
    // members of the struct. `context` references `executor` and thus needs
    // to be dropped before `executor`.
    context: std::mem::ManuallyDrop<cuda::Context<'static>>,
    context_ref: Context,
    device_ref: Device,
}

/// Returns a pointer to a CUDA environement. The caller is responsible for deallocating
/// the pointer by calling `telamon_cuda_destroy_environment`.
#[no_mangle]
pub unsafe extern "C" fn telamon_cuda_environment_new() -> *mut CudaEnvironment {
    let _ = env_logger::try_init();
    // `CudaEnvironment` is a self-referential struct: it contains references to iteself.
    // We must thus intialize it step by step, so that members holding references are
    // created after referenced members. Initially, only `executor` is created and the
    // rest is left initialized. We then manually write the remaining fields.
    let mut env = Box::new(CudaEnvironment {
        executor: cuda::Executor::init(),
        context: std::mem::uninitialized(),
        context_ref: std::mem::uninitialized(),
        device_ref: std::mem::uninitialized(),
    });
    let context = std::mem::transmute(cuda::Context::new(&env.executor));
    std::ptr::write(&mut env.context, std::mem::ManuallyDrop::new(context));
    std::ptr::write(&mut env.context_ref, Context(&*env.context));
    std::ptr::write(&mut env.device_ref, Device((*env.context_ref.0).device()));
    Box::into_raw(env)
}

/// Destroys a CUDA context.
#[no_mangle]
pub unsafe extern "C" fn telamon_cuda_environment_free(env: *mut CudaEnvironment) {
    let mut env = Box::from_raw(env);
    std::mem::ManuallyDrop::drop(&mut env.context);
}

/// Returns a pointer to ithe evaluation context.
#[no_mangle]
pub unsafe extern "C" fn telamon_cuda_get_context(env: *const CudaEnvironment)
    -> *const Context
{
        &(*env).context_ref
}

/// Returns a pointer to the description of the target device.
#[no_mangle]
pub unsafe extern "C" fn telamon_cuda_get_device(env: *const CudaEnvironment)
    -> *const Device
{
    &(*env).device_ref
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
