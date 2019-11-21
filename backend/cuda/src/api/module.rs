//! Interface with CUDA Modules and Kernels.
use crate::api::wrapper::*;
#[cfg(feature = "real_gpu")]
use crate::api::PerfCounterSet;
use itertools::Itertools;
use libc;
use log::*;
use std::ffi::CString;
use telamon::device;
use utils::*;

/// A CUDA module.
pub struct Module<'a> {
    module: *mut CudaModule,
    context: &'a CudaContext,
}

impl<'a> Module<'a> {
    /// Creates a new `Module`.
    pub fn new(context: &'a CudaContext, code: &str, opt_level: usize) -> Self {
        debug!("compiling... {}", code);
        let c_str = unwrap!(CString::new(code));
        let module = unsafe {
            let cubin_obj =
                compile_ptx_to_cubin(context, c_str.as_ptr(), code.len(), opt_level);
            let module = load_cubin(context, cubin_obj.data as *const libc::c_void);
            free_cubin_object(cubin_obj);
            module
        };
        Module { module, context }
    }

    /// Creates a `Module` from a cubin image.
    pub fn from_cubin(context: &'a CudaContext, image: &[u8]) -> Self {
        let module =
            unsafe { load_cubin(context, image.as_ptr() as *const libc::c_void) };
        Module { module, context }
    }

    /// Returns the `Kernel` with the given name.
    pub fn kernel<'b>(&'b self, name: &str) -> Kernel<'a>
    where
        'a: 'b,
    {
        let name_c_str = unwrap!(CString::new(name));
        let function =
            unsafe { get_function(self.context, self.module, name_c_str.as_ptr()) };
        Kernel {
            function,
            context: self.context,
        }
    }
}

impl<'a> Drop for Module<'a> {
    fn drop(&mut self) {
        unsafe {
            free_module(self.module);
        }
    }
}

unsafe impl<'a> Sync for Module<'a> {}
unsafe impl<'a> Send for Module<'a> {}

/// A CUDA kernel, ready to execute.
pub struct Kernel<'a> {
    function: *mut CudaFunction,
    context: &'a CudaContext,
}

impl<'a> Kernel<'a> {
    /// Executes the `Kernel` and returns the execution time in number of cycles.
    pub fn execute(
        &self,
        blocks: &[u32; 3],
        threads: &[u32; 3],
        args: &[&dyn Argument],
    ) -> Result<u64, ()> {
        unsafe {
            let arg_raw_ptrs = args.iter().map(|x| x.raw_ptr()).collect_vec();
            let mut out = 0;
            let ret = launch_kernel(
                self.context,
                self.function,
                blocks.as_ptr(),
                threads.as_ptr(),
                arg_raw_ptrs.as_ptr(),
                &mut out,
            );
            if ret == 0 {
                Ok(out)
            } else {
                Err(())
            }
        }
    }

    /// Instruments the kernel with the given performance counters.
    #[cfg(feature = "real_gpu")]
    pub fn instrument(
        &self,
        blocks: &[u32; 3],
        threads: &[u32; 3],
        args: &[&dyn Argument],
        counters: &PerfCounterSet,
    ) -> Vec<u64> {
        counters.instrument(unsafe { &*self.function }, blocks, threads, args)
    }

    /// Runs a kernel and returns the number of nanoseconds it takes to execute,
    /// measured using cuda event rather than hardware counters.
    pub fn time_real_conds(
        &self,
        blocks: &[u32; 3],
        threads: &[u32; 3],
        args: &[&dyn Argument],
    ) -> f64 {
        unsafe {
            let arg_raw_ptrs = args.iter().map(|x| x.raw_ptr()).collect_vec();
            time_with_events(
                self.context,
                self.function,
                blocks.as_ptr(),
                threads.as_ptr(),
                arg_raw_ptrs.as_ptr(),
            )
        }
    }

    /// Indicates the number of active block of threads per multiprocessors.
    pub fn blocks_per_smx(&self, threads: &[u32; 3]) -> u32 {
        let block_size = threads.iter().product::<u32>();
        unsafe { max_active_blocks_per_smx(self.function, block_size, 0) }
    }
}

impl<'a> Drop for Kernel<'a> {
    fn drop(&mut self) {
        unsafe {
            libc::free(self.function as *mut libc::c_void);
        }
    }
}

unsafe impl<'a> Sync for Kernel<'a> {}
unsafe impl<'a> Send for Kernel<'a> {}

/// An object that can be passed to a CUDA kernel.
pub trait Argument: Sync + Send {
    /// Returns a pointer to the object.
    fn raw_ptr(&self) -> *const libc::c_void;
    /// Returns the argument value if it can represent a size.
    fn as_size(&self) -> Option<u32> {
        None
    }
}

impl Argument for Box<dyn device::ScalarArgument> {
    fn raw_ptr(&self) -> *const libc::c_void {
        device::ScalarArgument::raw_ptr(self.as_ref())
    }

    fn as_size(&self) -> Option<u32> {
        device::ScalarArgument::as_size(self.as_ref())
    }
}
