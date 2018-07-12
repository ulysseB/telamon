extern crate libc;
extern crate telamon;
extern crate telamon_kernels;

use libc::{c_char, c_int, c_uint, size_t};
use telamon::device;
#[cfg(feature = "cuda")]
use telamon::device::cuda;
use telamon::device::x86;
use telamon::explorer::config::Config;
pub use telamon_kernels::{linalg, Kernel};

#[repr(C)]
pub enum Device {
    X86,
    Cuda,
}

pub enum KernelParameters {
    MatMul(linalg::MatMulP),
}

impl KernelParameters {
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

#[no_mangle]
pub extern "C" fn kernel_matmul_new(
    m: c_int,
    n: c_int,
    k: c_int,
    a_stride: c_uint,
    transpose_a: c_int,
    transpose_b: c_int,
    generic: c_int,
) -> *mut KernelParameters {
    Box::into_raw(Box::new(KernelParameters::MatMul(linalg::MatMulP {
        m: m as i32,
        n: n as i32,
        k: k as i32,
        a_stride: a_stride as u32,
        transpose_a: transpose_a == 1,
        transpose_b: transpose_b == 1,
        generic: generic == 1,
    })))
}

#[no_mangle]
pub unsafe extern "C" fn kernel_free(params: *mut KernelParameters) -> () {
    std::mem::drop(Box::from_raw(params));
}

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
