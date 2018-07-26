//! C API wrappers to create Telamon Kernels.
use Device;
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

/// Adds a scalar parameter to the function signature.
#[no_mangle]
pub unsafe extern "C" fn telamon_ir_signature_add_scalar(
    signature: *mut ir::Signature,
    name: *const libc::c_char,
    t: *const ir::Type
) {
    let name = unwrap!(std::ffi::CStr::from_ptr(name).to_str());
    (*signature).add_scalar(name.to_string(), *t);
}

/// Adds an array parameter to the function signature.
#[no_mangle]
pub unsafe extern "C" fn telamon_ir_signature_add_array(
    signature: *mut ir::Signature,
    name: *const libc::c_char
) {
    // FIXME: retrun the memory Id
    let name = unwrap!(std::ffi::CStr::from_ptr(name).to_str());
    (*signature).add_array(name.to_string());
}

/// Creates an integer type that must be freed with `telamon_ir_type_free`.
#[no_mangle]
pub unsafe extern "C" fn telamon_ir_type_new_int(
    num_bits: libc::uint16_t,
) -> *mut ir::Type {
    Box::into_raw(Box::new(ir::Type::I(num_bits)))
}

/// Creates a floating-point type that must be freed with `telamon_ir_type_free`.
#[no_mangle]
pub unsafe extern "C" fn telamon_ir_type_new_float(
    num_bits: libc::uint16_t,
) -> *mut ir::Type {
    Box::into_raw(Box::new(ir::Type::F(num_bits)))
}

/// Frees a type allocated with `telamon_ir_type_new_int` or `telamon_ir_type_new_float`.
#[no_mangle]
pub unsafe extern "C" fn telamon_ir_type_free(t: *mut ir::Type) {
    std::mem::drop(Box::from_raw(t));
}

/// Opaque type that abstracts away the lifetime parameter of `ir::Function` so that
/// cbindgen generates the bindings.
pub struct Function(ir::Function<'static>);

/// Creates a function to optimize. The function must be freed with
/// `telamon_ir_function_free`. `signature` and `device` must outlive the function.
#[no_mangle]
pub unsafe extern "C" fn telamon_ir_function_new(
    signature: *const ir::Signature,
    device: *const Device,
) -> *mut Function {
    Box::into_raw(Box::new(Function(ir::Function::new(&*signature, &*(*device).0))))
}

/// Frees a function allocated with `telamon_ir_function_new`.
#[no_mangle]
pub unsafe extern "C" fn telamon_ir_function_free(function: *mut Function) {
    std::mem::drop(Box::from_raw(function));
}

// FIXME
#[no_mangle]
pub unsafe extern "C" fn telamon_ir_function_add_inst(
    function: *mut Function,
    operator: *mut ir::Operator,
    dimensions: *const ir::dim::Id,
    num_dimensions: libc::size_t,
) {
    unimplemented!() // FIXME: return the instruction ID.
}

// FIXME
#[no_mangle]
pub unsafe extern "C" fn telamon_ir_function_add_dimension(
    function: *mut Function,
    size: *mut Size,
) {
    unimplemented!() // FIXME: return the dimension ID.
}

/// Opaque type that abstracts away the lifetime parameter of `ir::Size` so cbindgen
/// can generate bindings.
pub struct Size(ir::Size<'static>);

// FIXME:
#[no_mangle]
pub unsafe extern "C" fn telamon_ir_size_new(
    const_factor: libc::uint32_t,
    const_divisor: libc::uint32_t,
    parameter_factors: *const ir::Parameter,
    num_parameter_factors: libc::size_t,
) -> *mut Size {
    unimplemented!() // FIXME
}

// FIXME: size free ?
// FIXME: operator creation
// FIXME: operand creation
// FIXME: induction variable creation
// FIXME: access pattern creation
