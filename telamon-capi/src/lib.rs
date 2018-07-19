//! C API wrappers for calling Telamon through FFI.
//!
//! The goal of the C API is to provide thin wrappers over existing Rust
//! functionality, and only provides some cosmetic improvements to try and
//! provide a somewhat idiomatic C interface.

extern crate env_logger;
extern crate libc;
extern crate telamon;
extern crate telamon_kernels;

use libc::{c_char, c_int, c_uint, size_t, uint32_t};
use telamon::device;
#[cfg(feature = "cuda")]
use telamon::device::cuda;
use telamon::device::x86;
use telamon::explorer::config::Config;
pub use telamon_kernels::{linalg, Kernel};

/// Initializes the logger.
#[no_mangle]
pub extern "C" fn env_logger_try_init() {
    let _ = env_logger::try_init();
}

/// Supported device types for running kernels.
#[repr(C)]
pub enum Device {
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
    fn optimize_kernel<C: device::ArgMap + device::Context>(
        &self,
        config: &Config,
        context: &mut C,
    ) {
        match self {
            KernelParameters::MatMul(params) => {
                linalg::MatMul::<f32>::benchmark(config, params.clone(), 0, context);
            }
        }
    }
}

/// Helper function to create a Rust vector from a C array (pointer
/// and size) without transfering ownership (it performs a
/// copy). Returns None when data is null.
unsafe fn c_vec<T: Clone>(data: *const T, len: usize) -> Option<Vec<T>> {
    if data.is_null() {
        None
    } else {
        Some(std::slice::from_raw_parts(data, len).to_vec())
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
        m_tiling: c_vec(tile_m, tile_m_len),
        n_tiling: c_vec(tile_n, tile_n_len),
        k_tiling: c_vec(tile_k, tile_k_len),
    })))
}

/// Deallocates kernel parameters created through one of the `kernel_*_new`
/// functions. The `params` pointer becomes invalid and must not be used again
/// after calling `kernel_free`.
#[no_mangle]
pub unsafe extern "C" fn kernel_free(params: *mut KernelParameters) -> () {
    std::mem::drop(Box::from_raw(params));
}

/// Optimize a kernel on a given device. `config_data` points to a JSON-encoded
/// string of length `config_len` containing the configuration parameters for
/// the explorer.
#[no_mangle]
pub unsafe extern "C" fn kernel_optimize(
    params: *mut KernelParameters,
    device: Device,
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
    let _bench_result = match device {
        Device::X86 => (*params).optimize_kernel(&config, &mut x86::Context::new()),
        Device::Cuda => {
            #[cfg(feature = "cuda")]
            {
                let executor = cuda::Executor::init();
                (*params).optimize_kernel(&config, &mut cuda::Context::new(&executor));
            }
            #[cfg(not(feature = "cuda"))]
            return false;
        }
    };
    true
}
