use libc::{c_int, c_uint, size_t, uint32_t};
use std::marker::PhantomData;

use telamon::device;
use telamon::helper::TilingPattern;

use telamon_kernels::{self as kernels, linalg, MemInit, SignedKernel};

use super::context;
use super::error::TelamonStatus;

/// A kernel which can be used to create search spaces.
pub struct Kernel(Box<kernels::DynKernel<'static>>);

/// Frees a kernel.
#[no_mangle]
pub unsafe extern "C" fn telamon_kernel_free(kernel: *mut Kernel) {
    std::mem::drop(Box::from_raw(kernel))
}

/// The parameters for a kernel.  This can be used to build the kernel on a specific device using
/// `telamon_kernel_parameters_build_kernel`.
pub struct KernelParameters(Box<dyn KernelParametersTrait>);

// Helper trait to define kernel parameters.  This allows to build type-erased kernel builders.
trait KernelParametersTrait {
    fn erase<'a, K, C>(
        sigcontext: (SignedKernel<'a, K>, &'a C),
    ) -> (Box<kernels::DynKernel<'a>>, &'a dyn device::Context)
    where
        Self: Sized,
        K: kernels::Kernel<'a> + 'a,
        C: device::Context + 'a,
    {
        let (sig, context) = sigcontext;
        (Box::new(sig), context)
    }

    fn build<'a, 'c: 'a>(
        &self,
        builder: &kernels::KernelBuilder<'c>,
        context: &'a mut Box<dyn context::DContext<'c> + 'c>,
    ) -> (Box<kernels::DynKernel<'c>>, &'a dyn device::Context);
}

// Parameters for a matrix multiplication kernel
struct MatMulParameters<S: kernels::Scalar> {
    inner: linalg::MatMulP,
    _type: PhantomData<S>,
}

impl<S: kernels::Scalar> MatMulParameters<S> {
    fn new(inner: linalg::MatMulP) -> Self {
        MatMulParameters {
            inner,
            _type: PhantomData,
        }
    }
}

impl<S: kernels::Scalar> KernelParametersTrait for MatMulParameters<S> {
    fn build<'a, 'c: 'a>(
        &self,
        builder: &kernels::KernelBuilder<'c>,
        context: &'a mut Box<dyn context::DContext<'c> + 'c>,
    ) -> (Box<kernels::DynKernel<'c>>, &'a dyn device::Context) {
        let (sig, context) =
            builder.build::<linalg::MatMul<S>, _>(self.inner.clone(), context);
        (Box::new(sig), context)
    }
}

/// Build the kernel in a given `context`.
///
/// # Safety
///
/// If the call succeeds, `telamon_kernel_build` does not take ownership of any of its argument;
/// however it requires that `context` is kept alive and unused until both `*context_out` and
/// `*kernel_out` are freed.
///
/// If the call fails, `*kernel_out` and `*context_out` are set to invalid values which must not be
/// used or freed in any way.
#[no_mangle]
pub unsafe extern "C" fn telamon_kernel_parameters_build_kernel(
    params: *const KernelParameters,
    context: *mut context::Context,
    kernel_out: *mut *mut Kernel,
    context_out: *mut *mut context::ContextRef,
) -> TelamonStatus {
    let builder = kernels::KernelBuilder::default().mem_init(MemInit::RandomFill);
    let (kernel, context) = (*params).0.build(&builder, (*context).as_inner_mut());

    *kernel_out = Box::into_raw(Box::new(Kernel(kernel)));
    *context_out = Box::into_raw(Box::new(context::ContextRef(context)));

    TelamonStatus::Ok
}

/// Helper function to create a TilingPattern from a buffer of u32
/// values without transferring ownership (it performs a copy).
/// Returns None when data is null.
unsafe fn c_tiling_pattern(data: *const u32, len: usize) -> Option<TilingPattern> {
    if data.is_null() {
        None
    } else {
        Some(std::slice::from_raw_parts(data, len).into())
    }
}

/// Instanciate a new kernel for matrix-matrix multiplication. The
/// caller is responsible for deallocating the returned pointer using
/// kernel_free. The tile_m, tile_n and tile_k parameters are read
/// from during the call, but no pointer to the corresponding data is
/// kept afterwards.
#[no_mangle]
pub unsafe extern "C" fn telamon_kernel_parameters_matmul_new(
    m: c_int,
    n: c_int,
    k: c_int,
    a_stride: c_uint,
    transpose_a: c_int,
    transpose_b: c_int,
    generic: c_int,
    tile_m: *const uint32_t,
    tile_m_len: size_t,
    tile_n: *const uint32_t,
    tile_n_len: size_t,
    tile_k: *const uint32_t,
    tile_k_len: size_t,
) -> *mut KernelParameters {
    Box::into_raw(Box::new(KernelParameters(Box::new(
        MatMulParameters::<f32>::new(linalg::MatMulP {
            m: m as i32,
            n: n as i32,
            k: k as i32,
            a_stride: a_stride as u32,
            transpose_a: transpose_a == 1,
            transpose_b: transpose_b == 1,
            generic: generic == 1,
            m_tiling: c_tiling_pattern(tile_m, tile_m_len),
            n_tiling: c_tiling_pattern(tile_n, tile_n_len),
            k_tiling: c_tiling_pattern(tile_k, tile_k_len),
        }),
    ))))
}

/// Deallocates kernel parameters created through one of the `telamon_kernel_parameters_*_new`
/// functions.
#[no_mangle]
pub unsafe extern "C" fn telamon_kernel_parameters_free(params: *mut KernelParameters) {
    std::mem::drop(Box::from_raw(params));
}
