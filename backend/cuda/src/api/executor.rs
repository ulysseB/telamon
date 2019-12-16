//! Allows the execution of kernels on the GPU.
use crate::api::wrapper::*;
use crate::api::*;
use lazy_static::lazy_static;
use libc;
use std::ffi::CStr;
use std::sync::Arc;
use std::sync::Mutex;
use utils::*;

lazy_static! {
    static ref JIT_SPAWNER: Mutex<DaemonSpawner> = Mutex::new(DaemonSpawner::new());
}

struct RawExecutor {
    context: *mut CudaContext,
}

impl RawExecutor {
    fn try_new() -> Result<RawExecutor, InitError> {
        // The daemon must be spawned before init_cuda is called.
        let _ = unwrap!(JIT_SPAWNER.lock());
        Ok(RawExecutor {
            context: unsafe { init_cuda(0) },
        })
    }
}

impl Drop for RawExecutor {
    fn drop(&mut self) {
        unsafe { free_cuda(self.context) }
    }
}

unsafe impl Sync for RawExecutor {}
unsafe impl Send for RawExecutor {}

/// Interface with a CUDA device.
#[derive(Clone)]
pub struct Executor {
    inner: Arc<RawExecutor>,
}

impl Executor {
    pub(super) fn raw(&self) -> &CudaContext {
        unsafe { &*self.inner.context }
    }

    /// Tries to initialize the `Executor` and panics if it fails.
    pub fn init() -> Executor {
        unwrap!(Self::try_init())
    }

    /// Initializes the `Executor`.
    pub fn try_init() -> Result<Executor, InitError> {
        let raw_executor = RawExecutor::try_new()?;
        Ok(Executor {
            inner: Arc::new(raw_executor),
        })
    }

    /// Spawns a `JITDaemon`.
    pub fn spawn_jit(&self, opt_level: usize) -> JITDaemon {
        unwrap!(JIT_SPAWNER.lock()).spawn_jit(opt_level)
    }

    /// Compiles a PTX module.
    pub fn compile_ptx<'a>(&'a self, code: &str, opt_level: usize) -> Module<'a> {
        Module::new(self.raw(), code, opt_level)
    }

    /// Compiles a PTX module using a separate process.
    pub fn compile_remote<'a>(&'a self, jit: &mut JITDaemon, code: &str) -> Module<'a> {
        jit.compile(self.raw(), code)
    }

    /// Allocates an array on the CUDA device.
    pub fn allocate_array<T>(&self, len: usize) -> Array<T> {
        Array::new(self.clone(), len)
    }

    /// Returns the name of the device.
    pub fn device_name(&self) -> String {
        unsafe {
            let c_ptr = device_name(self.raw());
            let string = unwrap!(CStr::from_ptr(c_ptr).to_str()).to_string();
            libc::free(c_ptr as *mut libc::c_void);
            string
        }
    }

    /// Creates a new set of performance counters.
    pub fn create_perf_counter_set<'a>(
        &'a self,
        counters: &[PerfCounter],
    ) -> PerfCounterSet<'a> {
        PerfCounterSet::new(self.raw(), counters)
    }

    /// Returns the value of a CUDA device attribute.
    pub fn device_attribute(&self, attribute: DeviceAttribute) -> i32 {
        unsafe { device_attribute(self.raw(), attribute as u32) }
    }
}

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
