//! Allows the execution of kernels on the GPU.
use device::cuda::api::*;
use device::cuda::api::wrapper::*;
use libc;
use std::ffi::CStr;
use std::sync::Mutex;

lazy_static! {
    static ref JIT_SPAWNER: Mutex<DaemonSpawner> = Mutex::new(DaemonSpawner::new());
}

/// Interface with a CUDA device.
pub struct Executor {
    context: *mut CudaContext,
}

impl Executor {
    /// Initialize the `Executor`.
    pub fn init() -> Executor {
        // The daemon must be spawned before init_cuda is called.
        let _ = unwrap!(JIT_SPAWNER.lock());
        Executor { context: unsafe { init_cuda(0) } }
    }

    /// Spawns a `JITDaemon`.
    pub fn spawn_jit(&self) -> JITDaemon {
        unwrap!(JIT_SPAWNER.lock()).spawn_jit()
    }

    /// Compiles a PTX module.
    pub fn compile_ptx<'a>(&'a self, code: &str) -> Module<'a> {
        Module::new(unsafe { &*self.context as &'a _}, code)
    }

    /// Compiles a PTX module using a separate process.
    pub fn compile_remote<'a>(&'a self, jit: &mut JITDaemon, code: &str) -> Module<'a> {
        debug!("IN COMPILE REMOTE");
        jit.compile(unsafe { &*self.context as &'a _}, code)
    }

    /// Allocates an array on the CUDA device.
    pub fn allocate_array<T>(&self, len: usize) -> Array<T> {
        let context = unsafe { &*self.context as &_};
        Array::new(context, len)
    }

    /// Returns the name of the device.
    pub fn device_name(&self) -> String {
        unsafe {
            let c_ptr = device_name(self.context);
            let string = unwrap!(CStr::from_ptr(c_ptr).to_str()).to_string();
            libc::free(c_ptr as *mut libc::c_void);
            string
        }
    }

    /// Creates a new set of performance counters.
    pub fn create_perf_counter_set<'a>(&'a self, counters: &[PerfCounter]
                                      ) -> PerfCounterSet<'a> {
        PerfCounterSet::new(unsafe { &*self.context as &'a _}, counters)
    }

    /// Returns the value of a CUDA device attribute.
    pub fn device_attribute(&self, attribute: DeviceAttribute) -> i32 {
        unsafe { device_attribute(self.context, attribute as u32) }
    }
}

impl Drop for Executor {
    fn drop(&mut self) {
        unsafe { free_cuda(self.context); }
    }
}

unsafe impl Sync for Executor {}
unsafe impl Send for Executor {}

/// Cuda device attributes. Not all alltributes are defined here, see the CUDA driver API
/// documentation at `CUdevice_attribute` for more.
pub enum DeviceAttribute {
    /// Maximum number of threads per block.
    MaxThreadPerBlock = 1,
    /// Maximum shared memory available per block in bytes.
    MaxSharedMemoryPerBlock = 8,
    /// Wrap size in threads.
    WrapSize = 10,
    /// Typical clock frequency in kilohertz.
    ClockRate = 13,
    /// Number of SMX on a device.
    SmxCount = 16,
    /// Peak memory clock rate in kilohertz.
    MemoryClockRate = 36,
    /// Width on the memory bus in bits.
    GlobalMemoryBusWidth = 37,
    /// Size of the L2 cache in bytes.
    L2CacheSize = 38,
    /// Major compute capability version number.
    ComputeCapabilityMajor = 75,
    /// Minor compute capability version number.
    ComputeCapabilityMinor = 76,
    /// Device supports caching globals in L1.
    GlobalL1CacheSupported = 79,
    /// Maximum shared memory available per multiprocessor in bytes.
    MaxSharedMemoryPerSmx = 81,
}
