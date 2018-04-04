//! Abstracts kernels so we can build generic methods to test them.
use telamon::{device, ir};
use telamon::helper::{Builder, SignatureBuilder};

/// A kernel that can be compiled, benchmarked and used for correctness tests.
pub trait Kernel {
    /// The input parameters of the kernel.
    type Parameters;

    /// Builds the signature of the kernel in the builder and returns an object that
    /// stores enough information to later build the kernel body and check its result.
    /// The `is_generic` flag indicates if th sizes should be instantiated.
    fn build_signature(parameters: Self::Parameters,
                       is_generic: bool,
                       builder: &mut SignatureBuilder) -> Self;

    /// Builder the kernel body in the given builder. This builder should be based on the
    /// signature created by `build_signature`.
    fn build_body(&self, builder: &mut Builder);

    /// Ensures the generated code performs the correct operation.
    fn check_result(&self, context: &device::Context) -> Result<(), String>;
}

/// A kernel that can be compiled on CUDA GPUs.
#[cfg(features="cuda")]
trait CudaKernel: Kernel {
    /// Returns the execution time of the kernel using the reference implementation on
    /// CUDA.
    fn benchmark_cuda(&self) -> f64;
}
