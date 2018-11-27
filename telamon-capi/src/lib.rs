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
pub mod kernels;

/// Initializes Telamon.  This must be called before calling any other APIs.
#[no_mangle]
pub extern "C" fn telamon_init() {
    let _ = env_logger::try_init();
}
