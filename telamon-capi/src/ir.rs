//! C API wrappers to create Telamon Kernels.
use libc;
use std;
use telamon::ir;

/// Creates a function signature that must be deallocated with
/// `telamon_ir_signature_free`.
#[no_mangle]
pub unsafe extern "C" fn telamon_ir_signature_new(name: *const libc::c_char)
    -> *mut ir::Signature
{
    let name = unwrap!(std::ffi::CStr::from_ptr(name).to_str());
    Box::into_raw(Box::new(ir::Signature::new(name.to_string())))
}

/// Deallocates a signature created with `telamon_ir_signature_new`.
#[no_mangle]
pub unsafe extern "C" fn telamon_ir_signature_free(signature: *mut ir::Signature) {
    std::mem::drop(Box::from_raw(signature));
}

/// Returns the parameter at the given position.
#[no_mangle]
pub unsafe extern "C" fn telamon_ir_signature_param(
    signature: *const ir::Signature,
    index: libc::size_t
) -> *const ir::Parameter {
    &(*signature).params[index]
}

// FIXME: parameter creation + parameter fetching
// FIXME: function creation
// FIXME: size creation
// FIXME: dimension creation
// FIXME: operand creation
// FIXME: instruction creation
// FIXME: induction variable creation
// FIXME: access pattern creation
