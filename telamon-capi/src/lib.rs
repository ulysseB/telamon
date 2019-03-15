//! C API wrappers for calling Telamon through FFI.
//!
//! The goal of the C API is to provide thin wrappers over existing Rust
//! functionality, and only provides some cosmetic improvements to try and
//! provide a somewhat idiomatic C interface.

#[macro_use]
pub mod error;

pub mod explorer;
pub mod ir;
pub mod search_space;

use libc::{c_char, c_int, c_uint, size_t, uint32_t};
use telamon::device;
use telamon::device::x86;
use telamon::explorer::config::Config;
use telamon::helper::TilingPattern;
pub use telamon_kernels::{linalg, Kernel};

// Pointers to `device::Context` and `device::Device` are not C-like pointers.
// Instead, they are fat pointers containing both a regular pointer to the
// object and a pointer to the vtable. Thus, we define wrappers to encapsulate
// the pointers in an opaque type and we return pointers to the wrappers to C
// users.

/// Description of the evaluation context. In particular, in contains the
/// mapping between argument names and argument values.
pub struct Context(pub(crate) *const device::Context);

/// Description of the targeted device.
pub struct Device(*const device::Device);

/// Initializes the logger.
#[no_mangle]
pub extern "C" fn env_logger_try_init() {
    let _ = env_logger::try_init();
}

/// Supported device types for running kernels.
#[repr(C)]
pub enum DeviceId {
    X86,
    Cuda,
}

/// Supported kernels.
#[derive(Clone)]
pub enum KernelParameters {
    /// A matrix-matrix multiplication kernel.
    MatMul(linalg::MatMulP),
}

impl KernelParameters {
    /// Runs the search for a best candidate.
    fn optimize_kernel<'a, C: device::ArgMap<'a> + device::Context>(
        &self,
        config: &Config,
        context: &mut C,
    ) {
        match self {
            KernelParameters::MatMul(params) => {
                linalg::MatMul::<f32>::benchmark(
                    config,
                    params.clone(),
                    0,
                    true,
                    context,
                );
            }
        }
    }
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
pub unsafe extern "C" fn kernel_matmul_new(
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
pub unsafe extern "C" fn kernel_free(params: *mut KernelParameters) {
    std::mem::drop(Box::from_raw(params));
}

/// Optimize a kernel on a given device. `config_data` points to a JSON-encoded
/// string of length `config_len` containing the configuration parameters for
/// the explorer.
#[no_mangle]
pub unsafe extern "C" fn kernel_optimize(
    params: *mut KernelParameters,
    device: DeviceId,
    config_data: *const c_char,
    config_len: size_t,
) -> bool {
    let config = {
        let config_str = {
            let slice = std::slice::from_raw_parts(config_data as *const u8, config_len);
            std::str::from_utf8(slice).expect("Invalid configuration string")
        };
        Config::from_json(config_str)
    };
    match device {
        DeviceId::X86 => (*params).optimize_kernel(&config, &mut x86::Context::default()),
        DeviceId::Cuda => {
            #[cfg(feature = "cuda")]
            {
                let executor = ::telamon_cuda::Executor::init();
                let mut context = ::telamon_cuda::Context::new(&executor);
                (*params).optimize_kernel(&config, &mut context);
            }
            #[cfg(not(feature = "cuda"))]
            return false;
        }
    };
    true
}
