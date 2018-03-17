//! Interface with CUDA Modules and Kernels.
use device::Argument;
use device::cuda::api::PerfCounterSet;
use device::cuda::api::wrapper::*;
use itertools::Itertools;
use libc;
use std::ffi::CString;

/// A CUDA module.
pub struct Module<'a> {
    module: *mut CudaModule,
    context: &'a CudaContext,
}

impl<'a> Module<'a> {
    /// Creates a new `Module`.
    pub fn new(context: &'a CudaContext, code: &str) -> Self {
        debug!("compiling... {}", code);
        let c_str = unwrap!(CString::new(code));
        let module = unsafe { compile_ptx(context, c_str.as_ptr()) };
        Module { module, context }
    }

    /// Creates a `Module` from a cubin image.
    pub fn from_cubin(context: &'a CudaContext, image: &[u8]) -> Self {
        let module = unsafe {
            load_cubin(context, image.as_ptr() as *const libc::c_void)
        };
        Module { module, context }
    }

    /// Returns the `Kernel` with the given name.
    pub fn kernel<'b>(&'b self, name: &str) -> Kernel<'a>  where 'a: 'b {
        let name_c_str = unwrap!(CString::new(name));
        let function = unsafe {
            get_function(self.context, self.module, name_c_str.as_ptr())
        };
        Kernel { function, context: self.context }
    }
}

impl<'a> Drop for Module<'a> {
    fn drop(&mut self) {
        unsafe { free_module(self.module); }
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
    pub fn execute(&self, blocks: &[u32; 3], threads: &[u32; 3], args: &[&Argument])
            -> Result<u64, ()> {
        unsafe {
            let arg_raw_ptrs = args.iter().map(|x| x.raw_ptr()).collect_vec();
            let mut out = 0;
            let ret = launch_kernel(
                self.context,
                self.function,
                blocks.as_ptr(),
                threads.as_ptr(),
                arg_raw_ptrs.as_ptr(),
                &mut out);
            if ret == 0 { Ok(out) } else { Err(()) }
        }
    }

    /// Instruments the kernel with the given performance counters.
    pub fn instrument(&self, blocks: &[u32; 3], threads: &[u32; 3], args: &[&Argument],
                      counters: &PerfCounterSet) -> Vec<u64> {
        counters.instrument( unsafe { &*self.function }, blocks, threads, args)
    }

    /// Indicates the number of active block of threads per multiprocessors.
    pub fn blocks_per_smx(&self, threads: &[u32; 3]) -> u32 {
        let block_size = threads.iter().product::<u32>();
        unsafe { max_active_blocks_per_smx(self.function, block_size, 0) }
    }
}

impl<'a> Drop for Kernel<'a> {
    fn drop(&mut self) {
        unsafe { libc::free(self.function as *mut libc::c_void); }
    }
}

unsafe impl<'a> Sync for Kernel<'a> {}
unsafe impl<'a> Send for Kernel<'a> {}
