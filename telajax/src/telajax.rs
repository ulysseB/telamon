//! Wrapper around telajax that sequentialize all calls with a mutex
/// Needed because of the fact that Kalray OpenCL implementation is actually not thread-safe (some
/// segmentation faults can occur when using it in parallel). As soon as this bug is fixed, this
/// module will become unnecessary
use lazy_static::lazy_static;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

lazy_static! {
    static ref MEM_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());
}

pub(crate) unsafe fn device_init(
    argc: ::std::os::raw::c_int,
    argv: *mut *mut ::std::os::raw::c_char,
    error: *mut ::std::os::raw::c_int,
) -> device_t {
    let _ = MEM_MUTEX.lock();
    telajax_device_init(argc, argv, error)
}
/// Return 1 if telajax is already initialized, 0 otherwise
pub(crate) unsafe fn is_initialized() -> ::std::os::raw::c_int {
    let _ = MEM_MUTEX.lock();
    telajax_is_initialized()
}

/// Wait for termination of all on-flight kernels and finalize telajax
/// @param[in]  device  Device
/// @return 0 on success, -1 otherwise
pub(crate) unsafe fn device_finalize(device: *mut device_t) -> std::os::raw::c_int {
    let _ = MEM_MUTEX.lock();
    telajax_device_finalize(device)
}
/// Return 1 if telajax is already finalized, 0 otherwise
pub(crate) unsafe fn is_finalized() -> ::std::os::raw::c_int {
    let _ = MEM_MUTEX.lock();
    telajax_is_finalized()
}
/// Wait for termination of all on-flight kernels on the device
/// @param[in]  device  Device
/// @return 0 on success, -1 otherwise
pub(crate) unsafe fn device_waitall(device: *mut device_t) -> ::std::os::raw::c_int {
    let _ = MEM_MUTEX.lock();
    telajax_device_waitall(device)
}
/// Allocate a buffer on device
/// @param[in]   size         Malloc size in bytes
/// @param[in]   mem_flags    Memory flags :
/// TELAJAX_MEM_READ_WRITE
/// TELAJAX_MEM_WRITE_ONLY
/// TELAJAX_MEM_READ_ONLY
/// @param[in]   device       Device
/// @param[out]  error        Error code : 0 on success, -1 otherwise
/// @return mem_t
pub(crate) unsafe fn device_mem_alloc(
    size: usize,
    mem_flags: mem_flags_t,
    device: *mut device_t,
    error: *mut ::std::os::raw::c_int,
) -> mem_t {
    let _ = MEM_MUTEX.lock();
    telajax_device_mem_alloc(size, mem_flags, device, error)
}

/// Initialize the device memory from a host memory buffer
/// @param[in]  device        Device
/// @param[in]  device_mem    Device memory
/// @param[in]  host_mem      Host memory pointer
/// @param[in]  size          Size in bytes to transfer
/// @param[in]  num_events_in_wait_list  Number of event in wait list.
/// Executed immediately if zero.
/// @param[in]  event_wait_list  List of event to wait before doing the
/// memory operation. If not NULL, length of
/// event_wait_list must be >= num_events_in_wait_list
/// @param[out]  event  Event of the operation. If NULL, call is blocking.
/// If not NULL, the call is non-blocking and returns immediately.
/// One must later call telajax_event_wait() to wait for termination.
/// @return 0 on success, -1 otherwise
pub(crate) unsafe fn device_mem_write(
    device: *mut device_t,
    device_mem: mem_t,
    host_mem: *mut ::std::os::raw::c_void,
    size: usize,
    num_events_in_wait_list: ::std::os::raw::c_uint,
    event_wait_list: *const event_t,
    event: *mut event_t,
) -> ::std::os::raw::c_int {
    let _ = MEM_MUTEX.lock();
    telajax_device_mem_write(
        device,
        device_mem,
        host_mem,
        size,
        num_events_in_wait_list,
        event_wait_list,
        event,
    )
}

/// Read the device memory into a host memory buffer
/// @param[in]  device        Device
/// @param[in]  device_mem    Device memory
/// @param[in]  host_mem      Host memory pointer
/// @param[in]  size          Size in bytes to transfer
/// @param[in]  num_events_in_wait_list  Number of event in wait list.
/// Executed immediately if zero.
/// @param[in]  event_wait_list  List of event to wait before doing the
/// memory operation. If not NULL, length of
/// event_wait_list must be >= num_events_in_wait_list
/// @param[out]  event  Event of the operation. If NULL, call is blocking.
/// If not NULL, the call is non-blocking and returns immediately.
/// One must later call telajax_event_wait() to wait for termination.
/// @return 0 on success, -1 otherwise
pub(crate) unsafe fn device_mem_read(
    device: *mut device_t,
    device_mem: mem_t,
    host_mem: *mut ::std::os::raw::c_void,
    size: usize,
    num_events_in_wait_list: ::std::os::raw::c_uint,
    event_wait_list: *const event_t,
    event: *mut event_t,
) -> ::std::os::raw::c_int {
    let _ = MEM_MUTEX.lock();
    telajax_device_mem_read(
        device,
        device_mem,
        host_mem,
        size,
        num_events_in_wait_list,
        event_wait_list,
        event,
    )
}

/// Release a buffer on device
/// @param[in]  device_mem   Device memory returned by mem_alloc
/// @return 0 on success, -1 otherwise
pub(crate) unsafe fn device_mem_release(device_mem: mem_t) -> ::std::os::raw::c_int {
    let _ = MEM_MUTEX.lock();
    telajax_device_mem_release(device_mem)
}
/// Build up a wrapper from an OpenCL wrapper code
/// @param[in]  kernel_ocl_name     OpenCL wrapper name
/// @param[in]  kernel_ocl_wrapper  OpenCL wrapper code
/// @param[in]  options      Compilation options of wrapper (ignored if NULL)
/// @param[in]  device       Device on which kernel will be built for.
/// This identifies the hardware and uses the correct
/// compilation steps (and optimization).
/// @param[out] error        Error code : 0 on success, -1 otherwise
/// @return    wrapper_t
pub(crate) unsafe fn wrapper_build(
    kernel_ocl_name: *const ::std::os::raw::c_char,
    kernel_ocl_wrapper: *const ::std::os::raw::c_char,
    options: *const ::std::os::raw::c_char,
    device: *mut device_t,
    error: *mut ::std::os::raw::c_int,
) -> wrapper_t {
    let _ = MEM_MUTEX.lock();
    telajax_wrapper_build(kernel_ocl_name, kernel_ocl_wrapper, options, device, error)
}

/// Release a wrapper
/// @param[out] wrapper       Wrapper to release
/// @return 0 on success, -1 otherwise
pub(crate) unsafe fn wrapper_release(wrapper: *mut wrapper_t) -> ::std::os::raw::c_int {
    let _ = MEM_MUTEX.lock();
    telajax_wrapper_release(wrapper)
}

/// Build up a kernel from a code source
/// @param[in]  kernel_code  String containing C code (ignored if NULL)
/// @param[in]  cflags       Compilation flags of kernel (ignored if NULL)
/// @param[in]  lflags       Link flags of kernel (ignored if NULL)
/// @param[in]  wrapper      OpenCL wrapper
/// @param[in]  device       Device on which kernel will be built for.
/// This identifies the hardware and uses the correct
/// compilation steps (and optimization).
/// @param[out] error        Error code : 0 on success, -1 otherwise
/// @return    kernel_t
pub(crate) unsafe fn kernel_build(
    kernel_code: *const ::std::os::raw::c_char,
    cflags: *const ::std::os::raw::c_char,
    lflags: *const ::std::os::raw::c_char,
    wrapper: *const wrapper_t,
    device: *mut device_t,
    error: *mut ::std::os::raw::c_int,
) -> kernel_t {
    let _ = MEM_MUTEX.lock();
    telajax_kernel_build(kernel_code, cflags, lflags, wrapper, device, error)
}

/// Set work_dim and globalSize, localSize of a kernel
/// @param[in]  work_dim     Number of dimension (=< 3)
/// @param[in]  globalSize   Number of global work-item in each dimension
/// @param[in]  localSize    Number of local  work-item in each dimension
/// @param[in]  kernel       Kernel
/// @return 0 on success, -1 otherwise
pub(crate) unsafe fn kernel_set_dim(
    work_dim: ::std::os::raw::c_int,
    globalSize: *const usize,
    localSize: *const usize,
    kernel: *mut kernel_t,
) -> ::std::os::raw::c_int {
    let _ = MEM_MUTEX.lock();
    telajax_kernel_set_dim(work_dim, globalSize, localSize, kernel)
}
/// Release a kernel
/// @details    User should manage to call this function inside the callback or
/// explicitly in application to avoid possible memory leak.
/// @param[out] kernel       Kernel to release
/// @return 0 on success, -1 otherwise
pub(crate) unsafe fn kernel_release(kernel: *mut kernel_t) -> ::std::os::raw::c_int {
    let _ = MEM_MUTEX.lock();
    telajax_kernel_release(kernel)
}

/// Set arguments for a kernel
/// @param[in]  num_args    Number of arguments of the kernel in args[]
/// @param[in]  args_size   Array of size of each argument in args[]
/// @param[in]  args        Array of arguments
/// @param[in]  kernel      Associated kernel
/// @return 0 on success, -1 otherwise
pub(crate) unsafe fn kernel_set_args(
    num_args: ::std::os::raw::c_int,
    args_size: *mut usize,
    args: *mut *mut ::std::os::raw::c_void,
    kernel: *mut kernel_t,
) -> ::std::os::raw::c_int {
    let _ = MEM_MUTEX.lock();
    telajax_kernel_set_args(num_args, args_size, args, kernel)
}

/// Enqueue the kernel to the device
/// @param[in]  kernel   Kernel to enqueue on device
/// @param[in]  device   Device
/// @param[out] event    If not NULL, return an event to wait on later
/// @return 0 on success, -1 otherwise
pub(crate) unsafe fn kernel_enqueue(
    kernel: *mut kernel_t,
    device: *mut device_t,
    event: *mut event_t,
) -> ::std::os::raw::c_int {
    let _ = MEM_MUTEX.lock();
    telajax_kernel_enqueue(kernel, device, event)
}

/// Attach a callback to a event
/// @param[in]  pfn_event_notify  Pointer to the callback function
/// @param[in]  user_data         Pointer to the argument of callback
/// @param[in]  event             Associated event
/// @return 0 on success, -1 otherwise
pub(crate) unsafe fn event_set_callback(
    pfn_event_notify: ::std::option::Option<
        unsafe extern "C" fn(
            event: cl_event,
            event_command_exec_status: cl_int,
            user_data: *mut ::std::os::raw::c_void,
        ),
    >,
    user_data: *mut ::std::os::raw::c_void,
    event: event_t,
) -> ::std::os::raw::c_int {
    let _ = MEM_MUTEX.lock();
    telajax_event_set_callback(pfn_event_notify, user_data, event)
}
/// Wait for event
/// @param[in]  nb_events  Number of event in list
/// @param[in]  event_list Event list
/// @return 0 on success, -1 otherwise
pub(crate) unsafe fn event_wait(
    nb_events: ::std::os::raw::c_int,
    event_list: *const event_t,
) -> ::std::os::raw::c_int {
    let _ = MEM_MUTEX.lock();
    telajax_event_wait(nb_events, event_list)
}
/// Release an event
/// @param[in]  event Event to release
/// @return 0 on success, -1 otherwise
pub(crate) unsafe fn event_release(event: event_t) -> ::std::os::raw::c_int {
    let _ = MEM_MUTEX.lock();
    telajax_event_release(event)
}
