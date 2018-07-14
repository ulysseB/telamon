//! Provides a fake implementation of the CUDA API so we can compile it even when cuda
//! is not installed. This allows us to reference the CUDA-specific types without cuda
//! installed and to run tests on functions that do not rely on the executor.

/// Fake wrapper for the CUDA API.
pub mod api {
    use device;
    use std::marker::PhantomData;

    /// An argument that can be passed to the executor.
    pub trait Argument: Send + Sync {
        /// Returns the argument value if it can represent a size.
        fn as_size(&self) -> Option<u32> { None }
    }

    impl<T> Argument for T where T: device::ScalarArgument {
        fn as_size(&self) -> Option<u32> {
            device::ScalarArgument::as_size(self)
        }
    }

    /// An array on the CUDA device.
    #[derive(Clone)]
    pub struct Array<'a, T> { a: PhantomData<&'a ()>, t: PhantomData<T> }

    impl<'a, T> device::ArrayArgument for Array<'a, T> where T: device::ScalarArgument {
        fn read_i8(&self) -> Vec<i8> { panic!("no instance of Array should exist") }

        fn write_i8(&self, bytes: &[i8]) { panic!("no instance of Array should exist") }
    }

    impl<'a, T> Argument for Array<'a, T> where T: device::ScalarArgument { }

    /// Interface with a CUDA device.
    #[derive(Debug)] // FIXME: remove the derive
    pub enum Executor { }

    impl Executor {
        /// Initialize the `Executor`.
        pub fn init() -> Executor {
            panic!("CUDA support was not enable for this build, please recompile with \
                    --features=cuda")
        }

        /// Spawns a `JITDaemon`.
        pub fn spawn_jit(&self, _: usize) -> JITDaemon { match *self { } }

        /// Allocates an array on the CUDA device.
        pub fn allocate_array<T>(&self, _: usize) -> Array<T> { match *self { } }

        /// Returns the name of the device.
        pub fn device_name(&self) -> String { match *self { } }
    }

    /// A process that compiles PTX in a separate process.
    pub enum JITDaemon { }
}

/// IR instance compiled into a CUDA kernel.
// FIXME: amybe we can use the real kernels instead
pub mod kernel {
    use device::{self, cuda};

    /// An IR instance compiled into a CUDA kernel.
    #[allow(dead_code)]
    pub struct Kernel<'a, 'b> {
        executor: &'a cuda::api::Executor,
        function: &'b device::Function<'b>,
    }

    impl<'a, 'b> Kernel<'a, 'b> {
        /// Compiles a device function.
        pub fn compile(_: &'b device::Function<'b>,
                       _: &cuda::Gpu,
                       executor: &'a cuda::api::Executor,
                       _: usize) -> Self {
            match *executor { }
        }

        /// Compiles a device function, using a separate process.
        pub fn compile_remote(_: &'b device::Function<'b>,
                              _: &cuda::Gpu,
                              executor: &'a cuda::Executor,
                              _: &mut cuda::JITDaemon) -> Self {
            match *executor { }
        }

        /// Runs a kernel and returns the number of cycles it takes to execute in cycles.
        pub fn evaluate(&self, _: &cuda::Context) -> Result<u64, ()> {
            match *self.executor { }
        }

        /// Runs a kernel and returns the number of cycles it takes to execute in nanoseconds,
        /// measured using cuda event rather than hardware counters.
        pub fn evaluate_real(&self, _: &cuda::Context, _: usize) -> Vec<f64> {
            match *self.executor { }
        }

        /// Generates a Thunk than can then be run on the GPU.
        pub fn gen_thunk(self, _: &'a cuda::Context) -> Thunk<'a> {
            match *self.executor { }
        }
    }

    /// A kernel ready to execute.
    #[derive(Debug)]
    pub struct Thunk<'a> { executor: &'a cuda::Executor }

    impl<'a> Thunk<'a> {
        /// Executes the kernel and returns the number of cycles it took to execute.
        pub fn execute(&self) -> Result<u64, ()> {
            match *self.executor { }
        }
    }
}
