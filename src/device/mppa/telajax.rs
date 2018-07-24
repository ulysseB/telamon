//! Unsafe wrapper around  library from Kalray.
#![allow(dead_code)]
use libc;
use parking_lot;
use std::ffi::CStr;
use std::sync::Arc;
use std;
use device::{self, ArrayArgument};

lazy_static! {
    static ref DEVICE: std::sync::Mutex<Device> = std::sync::Mutex::new(Device::init());
}

unsafe impl Sync for Device {}
unsafe impl Send for Device {}

/// A Telajax device.
#[repr(C)]
struct DeviceInner {
    platform_id: *mut libc::c_void,
    device_id: *mut libc::c_void,
    context: *mut libc::c_void,
    queue: *mut libc::c_void,
}

/// Buffer in MPPA RAM.
pub struct Buffer<'a> {
    pub mem: std::sync::RwLock<Mem>,
    pub executor: &'a Device,
}


impl<'a> Buffer<'a> {
    pub fn new(executor: *mut Device, len: usize) -> Self {
        let mem_block = executor.alloc(len);
        Buffer{
            mem: RwLock::new(mem),
            executor: unsafe {executor as &Device},
        }
    }
}
impl<'a> device::ArrayArgument for Buffer<'a> {
    fn read_i8(&self) -> Vec<i8> {
        let mem_block = unwrap!(self.mem.read());
        let mut read_buffer = vec![0; mem_block.len()];
        self.executor.read_buffer::<i8>(&mem_block, &mut read_buffer, &[]);
        read_buffer
    }

    fn write_i8(&self, slice: &[i8]) {
        let mut mem_block = unwrap!(self.mem.write());
        self.executor.write_buffer::<i8>(slice, &mut mem_block, &[]);
    }
}
/// A Telajax execution context.
pub struct Device {
    inner: DeviceInner,
    rwlock: Box<parking_lot::RwLock<()>>,
}



impl Device {
    /// Returns a reference to the `Device`. Guarantees unique access to the device.
    pub fn get() -> std::sync::MutexGuard<'static, Device> { DEVICE.lock().unwrap() }

    /// Initializes the device.
    fn init() -> Self {
        let mut error = 0;
        let device = unsafe { Device {
            inner: telajax_device_init(0, [].as_ptr(), &mut error),
            rwlock: Box::new(parking_lot::RwLock::default()),
        }};
        assert_eq!(error, 0);
        device
    }

    pub fn allocate_array<'a>(&'a self, len: usize) -> Buffer<'a> {
        let mem_block = self.alloc(len);
        Buffer {
            mem : std::sync::RwLock::new(mem_block),
            executor: self,
        }
    }

    /// Build a wrapper for a kernel.
    pub fn build_wrapper(&self, name: &CStr, code: &CStr) -> Wrapper {
        let mut error = 0;
        let flags: &'static CStr = Default::default();
        let wrapper = unsafe {
            telajax_wrapper_build(name.as_ptr(), code.as_ptr(), flags.as_ptr(),
                                  &self.inner, &mut error)
        };
        assert_eq!(error, 0);
        wrapper
    }

    /// Compiles a kernel.
    pub fn build_kernel(&self, code: &CStr, cflags: &CStr, lflags: &CStr,
                        wrapper: &Wrapper) -> Kernel {
        // FIXME: disable -O3
        // TODO(cc_perf): precompile headers
        // TODO(cc_perf): use double pipes in telajax
        // TODO(cc_perf): avoid collisions in kernel evaluations
        let mut error = 0;
        let kernel = unsafe {
            telajax_kernel_build(code.as_ptr(), cflags.as_ptr(), lflags.as_ptr(),
                                 wrapper, &self.inner, &mut error)
        };
        assert_eq!(error, 0);
        kernel
    }

    /// Asynchronously executes a `Kernel`.
    pub fn enqueue_kernel(&self, kernel: &Kernel) -> Event {
        unsafe {
            let mut event_ptr = std::mem::uninitialized();
            assert_eq!(telajax_kernel_enqueue(kernel, &self.inner, &mut event_ptr), 0);
            Event(event_ptr)
        }
    }

    /// Executes a `Kernel`.
    pub fn execute_kernel(&self, kernel: &Kernel) {
        unsafe {
            let mut event: Event = std::mem::uninitialized();
            assert_eq!(telajax_kernel_enqueue(kernel, &self.inner, &mut event.0), 0);
            assert_eq!(telajax_event_wait(event.0), 0);
        }

    }

    /// Waits until all kernels have completed their execution.
    pub fn wait_all(&self) {
        let _ = self.rwlock.write();
        unsafe { assert_eq!(telajax_device_waitall(&self.inner), 0); }
    }

    /// Allocates a memory buffer.
    pub fn alloc(&mut self, size: usize) -> Mem {
        let mut error = 0;
        let mem = unsafe {
            telajax_device_mem_alloc(size, 1 << 0, &mut self.inner, &mut error)
        };
        assert_eq!(error, 0);
        Mem { ptr: mem, len: size }
    }

    /// Asynchronously copies a buffer to the device.
    pub fn async_write_buffer<T: Copy>(&self,
                                       data: &[T],
                                       mem: &mut Mem,
                                       wait_events: &[Event]) -> Event {
        let size = data.len() * std::mem::size_of::<T>();
        assert!(size <= mem.len);
        let data_ptr = data.as_ptr() as *const libc::c_void;
        let wait_n = wait_events.len() as libc::c_uint;
        let wait_ptr = wait_events.as_ptr() as *const libc::c_void;
        unsafe {
            let mut event: Event = std::mem::uninitialized();
            let res = telajax_device_mem_write(
                &self.inner, mem.ptr, data_ptr, size, wait_n, wait_ptr, &mut event.0);
            assert_eq!(res, 0);
            event
        }
    }

    /// Copies a buffer to the device.
    pub fn write_buffer<T: Copy>(&self,
                                 data: &[T],
                                 mem: &mut Mem,
                                 wait_events: &[Event]) {
        let size = data.len() * std::mem::size_of::<T>();
        assert!(size <= mem.len);
        let data_ptr = data.as_ptr() as *const libc::c_void;
        let wait_n = wait_events.len() as libc::c_uint;
        let wait_ptr = wait_events.as_ptr() as *const libc::c_void;
        unsafe {
            let null_mut = std::ptr::null_mut();
            let res = telajax_device_mem_write(
                &self.inner, mem.ptr, data_ptr, size, wait_n, wait_ptr, null_mut);
            assert_eq!(res, 0);
        }
    }

    /// Asynchronously copies a buffer from the device.
    pub fn async_read_buffer<T: Copy>(&self,
                                      mem: &Mem,
                                      data: &mut [T],
                                      wait_events: &[Event]) -> Event {
        let size = data.len() * std::mem::size_of::<T>();
        assert!(size <= mem.len);
        let data_ptr = data.as_ptr() as *mut libc::c_void;
        let wait_n = wait_events.len() as libc::c_uint;
        let wait_ptr = wait_events.as_ptr() as *const libc::c_void;
        unsafe {
            let mut event: Event = std::mem::uninitialized();
            let res = telajax_device_mem_read(
                &self.inner, mem.ptr, data_ptr, size, wait_n, wait_ptr, &mut event.0);
            assert_eq!(res, 0);
            event
        }
    }

    /// Copies a buffer from the device.
    pub fn read_buffer<T: Copy>(&self, mem: &Mem, data: &mut [T], wait_events: &[Event]) {
        let size = data.len() * std::mem::size_of::<T>();
        assert!(size <= mem.len);
        let data_ptr = data.as_ptr() as *mut libc::c_void;
        unsafe {
            let null_mut = std::ptr::null_mut();
            let wait_n = wait_events.len() as libc::c_uint;
            let wait_ptr = wait_events.as_ptr() as *const libc::c_void;
            let res = telajax_device_mem_read(
                &self.inner, mem.ptr, data_ptr, size, wait_n, wait_ptr, null_mut);
            assert_eq!(res, 0);
        }
    }

    /// Set a callback to call when an event is triggered.
    pub fn set_event_callback<F>(&self, event: &Event, closure: F)
            where F: FnOnce() + Send {
        let callback_data = CallbackData { closure, rwlock: &*self.rwlock };
        let data_ptr = Box::into_raw(Box::new(callback_data));
        let callback = callback_wrapper::<F>;
        unsafe {
            self.rwlock.raw_read();
            let data_ptr = data_ptr as *mut libc::c_void;
            telajax_event_set_callback(callback, data_ptr, event.0);
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        let _ = self.rwlock.write();
        unsafe { telajax_device_finalize(&mut self.inner); }
        debug!("MPPA device finalized");
    }
}

/// A wrapper to call a kernel.
#[repr(C)]
pub struct Wrapper {
    name: *mut libc::c_char,
    program: *mut libc::c_void,
}

unsafe impl Send for Wrapper {}
unsafe impl Sync for Wrapper {}

impl Drop for Wrapper {
    fn drop(&mut self) {
        unsafe { assert_eq!(telajax_wrapper_release(self), 0); }
    }
}

/// A kernel to be executed on the device.
#[repr(C)]
pub struct Kernel {
    program: *mut libc::c_void,
    kernel: *mut libc::c_void,
    work_dim: libc::c_int,
    global_size: [libc::size_t; 3],
    local_size: [libc::size_t; 3],
}

impl Kernel {
    /// Sets the arguments of the `Kernel`.
    pub fn set_args(&mut self, sizes: &[usize], args: &[*const libc::c_void]) {
        assert_eq!(sizes.len(), args.len());
        let num_arg = sizes.len() as i32;
        let sizes_ptr = sizes.as_ptr();
        unsafe {
            assert_eq!(telajax_kernel_set_args(num_arg, sizes_ptr, args.as_ptr(), self), 0);
        }
    }

    /// Sets the number of clusters that must execute the `Kernel`.
    pub fn set_num_clusters(&mut self, num: usize) {
        assert!(num <= 16);
        unsafe {
            assert_eq!(telajax_kernel_set_dim(1, [1].as_ptr(), [num].as_ptr(), self), 0);
        }
    }
}

unsafe impl Send for Kernel {}

impl Drop for Kernel {
    fn drop(&mut self) {
        unsafe { assert_eq!(telajax_kernel_release(self), 0); }
    }
}

/// A buffer allocated on the device.
pub struct Mem {
    ptr: *mut libc::c_void,
    len: usize,
}

impl Mem {
    pub fn raw_ptr(&self) -> *const libc::c_void {
        &self.ptr as *const _ as *const libc::c_void
    }
    
    pub fn len(&self) -> usize {
        self.len
    }
}

unsafe impl Sync for Mem {}
unsafe impl Send for Mem {}

impl Drop for Mem {
    fn drop(&mut self) {
        unsafe { assert_eq!(telajax_device_mem_release(self.ptr), 0); }
    }
}

/// An event triggered at the end of a memory operation or kernel execution.
#[repr(C)]
pub struct Event(*mut libc::c_void);

impl Drop for Event {
    fn drop(&mut self) {
        unsafe { telajax_event_release(self.0); }
    }
}

/// Calls the closure passed in data.
unsafe extern fn callback_wrapper<F: FnOnce()>(_: *const libc::c_void,
                                               _: i32,
                                               data: *mut libc::c_void) {
    let data = Box::from_raw(data as *mut CallbackData<F>);
    let ref lock = *data.rwlock;
    (data.closure)();
    lock.raw_unlock_read();
}

type Callback = unsafe extern fn(*const libc::c_void, i32, *mut libc::c_void);

struct CallbackData<F: FnOnce()> {
    closure: F,
    rwlock: *const parking_lot::RwLock<()>,
}

extern "C" {
    fn telajax_device_init(argc: libc::c_int,
                           argv: *const *const libc::c_char,
                           error: *mut libc::c_int) -> DeviceInner;

    fn telajax_device_finalize(device: *mut DeviceInner);

    fn telajax_device_waitall(device: *const DeviceInner) -> libc::c_int;

    fn telajax_device_mem_alloc(size: libc::size_t,
                                mem_flags: u64,
                                device: *mut DeviceInner,
                                error: *mut libc::c_int) -> *mut libc::c_void;

    fn telajax_device_mem_write(device: *const DeviceInner,
                                device_mem: *mut libc::c_void,
                                host_mem: *const libc::c_void,
                                size: libc::size_t,
                                num_events_wait: libc::c_uint,
                                events_wait: *const libc::c_void,
                                event: *mut *mut libc::c_void) -> libc::c_int;

    fn telajax_device_mem_read(device: *const DeviceInner,
                               device_mem: *const libc::c_void,
                               host_mem: *mut libc::c_void,
                               size: libc::size_t,
                               num_events_wait: libc::c_uint,
                               events_wait: *const libc::c_void,
                               event: *mut *mut libc::c_void) -> libc::c_int;

    fn telajax_device_mem_release(mem: *mut libc::c_void) -> libc::c_int;

    fn telajax_wrapper_build(kernel_ocl_name: *const libc::c_char,
                             kernel_ocl_wrapper: *const libc::c_char,
                             options: *const libc::c_char,
                             device: *const DeviceInner,
                             error: *mut libc::c_int) -> Wrapper;

    fn telajax_wrapper_release(wrapper: *mut Wrapper) -> libc::c_int;

    fn telajax_kernel_build(kernel_code: *const libc::c_char,
	                        cflags: *const libc::c_char,
                            lflags: *const libc::c_char,
                            wrapper: *const Wrapper,
                            device: *const DeviceInner,
                            error: *mut libc::c_int) -> Kernel;

    fn telajax_kernel_set_dim(work_dim: libc::c_int,
                              global_size: *const libc::size_t,
                              local_size: *const libc::size_t,
                              kernel: *mut Kernel) -> libc::c_int;

    fn telajax_kernel_release(kernel: *mut Kernel) -> libc::c_int;

    fn telajax_kernel_set_args(num_args: libc::c_int,
                               args_size: *const libc::size_t,
                               args: *const *const libc::c_void,
                               kernel: *mut Kernel) -> libc::c_int;

    fn telajax_kernel_enqueue(kernel: *const Kernel,
                              device: *const DeviceInner,
                              event: *mut *mut libc::c_void) -> libc::c_int;

    fn telajax_event_set_callback(callback: Callback,
                                  data: *mut libc::c_void,
                                  event: *mut libc::c_void);

    fn telajax_event_wait(event: *mut libc::c_void) -> libc::c_int;

    fn telajax_event_release(event: *mut libc::c_void) -> libc::c_int;
}
