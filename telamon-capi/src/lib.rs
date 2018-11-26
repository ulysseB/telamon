//! C API wrappers for calling Telamon through FFI.
//!
//! The goal of the C API is to provide thin wrappers over existing Rust
//! functionality, and only provides some cosmetic improvements to try and
//! provide a somewhat idiomatic C interface.
extern crate env_logger;
extern crate libc;
extern crate num;
extern crate telamon;
extern crate telamon_kernels;
#[macro_use]
extern crate telamon_utils;
#[macro_use]
extern crate failure;

#[macro_use]
pub mod error;

pub mod cuda;
pub mod explorer;
pub mod ir;
pub mod search_space;

pub mod context;

use libc::{c_char, c_int, c_uint, size_t, uint32_t};

use telamon::device;
use telamon::explorer::config::Config;
use telamon::helper::TilingPattern;
pub use telamon_kernels::{
    linalg, DynKernel, Kernel, KernelBuilder, MemInit, SignedKernel,
};

use error::TelamonStatus;

/// Initializes Telamon.  This must be called before calling any other APIs.
#[no_mangle]
pub extern "C" fn telamon_init() {
    let _ = env_logger::try_init();
}

/// Supported kernels.
#[derive(Clone)]
pub enum KernelParameters {
    /// A matrix-matrix multiplication kernel.
    MatMul(linalg::MatMulP),
}

impl KernelParameters {
    fn erase<'a, K, C>(
        (sig, context): (SignedKernel<'a, K>, &'a C),
    ) -> (Box<DynKernel<'a>>, &'a dyn device::Context)
    where
        K: Kernel<'a> + 'a,
        C: device::Context + 'a,
    {
        (Box::new(sig), context)
    }

    fn build<'a, C: device::ArgMap + device::Context + 'a>(
        &self,
        context: &'a mut C,
    ) -> (Box<DynKernel<'a>>, &'a dyn device::Context) {
        let builder = KernelBuilder::default().mem_init(MemInit::RandomFill);

        match self {
            KernelParameters::MatMul(params) => Self::erase(
                builder.build::<linalg::MatMul<f32>, _>(params.clone(), context),
            ),
        }
    }

    /// Runs the search for a best candidate.
    fn optimize_kernel<'a, C: device::ArgMap<'a> + device::Context>(
        &self,
        config: &Config,
        context: &mut C,
    ) -> Vec<f64> {
        let (sig, context) = self.build(context);
        sig.benchmark(context, config, 0)
    }
}

#[no_mangle]
pub unsafe extern "C" fn telamon_kernel_build(
    kernel: *const KernelParameters,
    context: *mut context::Context,
    kernel_out: *mut *mut DynKernel<'static>,
    context_out: *mut *mut context::ContextRef,
) -> TelamonStatus {
    let (kernel, context) = (&*kernel).build(&mut *context);

    *kernel_out = Box::into_raw(kernel);
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
pub unsafe extern "C" fn telamon_kernel_matmul_new(
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
    Box::into_raw(Box::new(KernelParameters::MatMul(linalg::MatMulP {
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
    })))
}

/// Deallocates kernel parameters created through one of the `kernel_*_new`
/// functions. The `params` pointer becomes invalid and must not be used again
/// after calling `kernel_free`.
#[no_mangle]
pub unsafe extern "C" fn telamon_kernel_free(params: *mut KernelParameters) -> () {
    std::mem::drop(Box::from_raw(params));
}
