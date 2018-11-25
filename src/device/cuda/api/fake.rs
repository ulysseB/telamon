//! Provides a fake implementation of the CUDA API so we can compile it even when cuda
//! is not installed. This allows us to reference the CUDA-specific types without cuda
//! installed and to run tests on functions that do not rely on the executor.

use device;
use device::cuda::api;
use std::marker::PhantomData;

/// An argument that can be passed to the executor.
pub trait Argument: Send + Sync {
    /// Returns the argument value if it can represent a size.
    fn as_size(&self) -> Option<u32> {
        None
    }
}

impl<T> Argument for T
where
    T: device::ScalarArgument,
{
    fn as_size(&self) -> Option<u32> {
        device::ScalarArgument::as_size(self)
    }
}

impl Argument for Box<dyn device::ScalarArgument> {
    fn as_size(&self) -> Option<u32> {
        device::ScalarArgument::as_size(self.as_ref())
    }
}

/// An array on the CUDA device.
#[derive(Clone)]
pub struct Array<'a, T> {
    a: PhantomData<&'a ()>,
    t: PhantomData<T>,
}

impl<'a, T> device::ArrayArgument for Array<'a, T>
where
    T: device::ScalarArgument,
{
    fn read_i8(&self) -> Vec<i8> {
        panic!("no instance of Array should exist")
    }

    fn write_i8(&self, _: &[i8]) {
        panic!("no instance of Array should exist")
    }
}

impl<'a, T> Argument for Array<'a, T> where T: device::ScalarArgument {}

/// Interface with a CUDA device.
pub enum Executor {}

impl Executor {
    /// Initializes the `Executor`.
    pub fn try_init() -> Result<Executor, api::InitError> {
        Err(api::InitError::NeedsCudaFeature)
    }

    /// Spawns a `JITDaemon`.
    pub fn spawn_jit(&self, _: usize) -> JITDaemon {
        match *self {}
    }

    /// Allocates an array on the CUDA device.
    pub fn allocate_array<T>(&self, _: usize) -> Array<T> {
        match *self {}
    }

    /// Returns the name of the device.
    pub fn device_name(&self) -> String {
        match *self {}
    }

    /// Compiles a PTX module.
    pub fn compile_ptx<'a>(&'a self, _: &str, _: usize) -> Module<'a> {
        match *self {}
    }

    /// Compiles a PTX module using a separate process.
    pub fn compile_remote<'a>(&'a self, _: &mut JITDaemon, _: &str) -> Module<'a> {
        match *self {}
    }
}

/// A process that compiles PTX in a separate process.
pub enum JITDaemon {}

/// A CUDA module.
pub struct Module<'a> {
    executor: &'a Executor,
}

impl<'a> Module<'a> {
    /// Returns the `Kernel` with the given name.
    pub fn kernel<'b>(&'b self, _: &str) -> Kernel<'a>
    where
        'a: 'b,
    {
        match *self.executor {}
    }
}

/// A CUDA kernel, ready to execute.
pub struct Kernel<'a> {
    executor: &'a Executor,
}

impl<'a> Kernel<'a> {
    /// Executes the `Kernel` and returns the execution time in number of cycles.
    pub fn execute(
        &self,
        _: &[u32; 3],
        _: &[u32; 3],
        _: &[&Argument],
    ) -> Result<u64, ()> {
        match *self.executor {}
    }

    /// Runs a kernel and returns the number of nanoseconds it takes to execute,
    /// measured using cuda event rather than hardware counters.
    pub fn time_real_conds(&self, _: &[u32; 3], _: &[u32; 3], _: &[&Argument]) -> f64 {
        match *self.executor {}
    }

    /// Indicates the number of active block of threads per multiprocessors.
    pub fn blocks_per_smx(&self, _: &[u32; 3]) -> u32 {
        match *self.executor {}
    }
}
