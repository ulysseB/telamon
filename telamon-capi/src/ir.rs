//! C API wrappers to create Telamon Kernels.
use Device;
use libc;
use num::rational::Ratio;
use std;
use telamon::ir;

pub use telamon::ir::op::Rounding;

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

/// Adds an instruction performing the given operator in the given dimensions to the
/// function. Takes ownership of the operator but does not keeps any reference to
/// `dimensions`.
#[no_mangle]
pub unsafe extern "C" fn telamon_ir_function_add_instruction(
    function: *mut Function,
    operator: *mut Operator,
    dimensions: *const ir::dim::Id,
    num_dimensions: libc::size_t,
) {
    // FIXME: return the instruction ID
    let dimensions = std::slice::from_raw_parts(dimensions, num_dimensions);
    let dim_set = dimensions.iter().cloned().collect();
    (*function).0.add_inst((Box::from_raw(operator)).0, dim_set);
}

/// Adds a dimension of the given size to the function. Takes ownership of `size`.
#[no_mangle]
pub unsafe extern "C" fn telamon_ir_function_add_dimension(
    function: *mut Function,
    size: *mut Size,
) {
    // FIXME: return the dimension ID
    (*function).0.add_dim(Box::from_raw(size).0);
}

/// Opaque type that abstracts away the lifetime parameter of `ir::Size` so cbindgen
/// can generate bindings.
pub struct Size(ir::Size<'static>);

/// Create a size equal to:
/// ```
/// const_factor * paramr_factors[0] * .. * param_factors[num_params-1] / const_divisor
/// ```
/// The size must be freed calling `telamon_ir_size_free` or passed to a function that
/// takes its ownership.
#[no_mangle]
pub unsafe extern "C" fn telamon_ir_size_new(
    const_factor: libc::uint32_t,
    const_divisor: libc::uint32_t,
    param_factors: *const *const ir::Parameter,
    num_params: libc::size_t,
) -> *mut Size {
    let parameters = std::slice::from_raw_parts(param_factors, num_params)
        .iter().map(|&ptr| &*ptr).collect();
    let size = ir::Size::new(const_factor, parameters, const_divisor);
    Box::into_raw(Box::new(Size(size)))
}

/// Frees a size allocated with `telamon_ir_size_new`.
#[no_mangle]
pub unsafe extern "C" fn telamon_ir_size_free(size: *mut Size) {
    std::mem::drop(Box::from_raw(size));
}

/// Opaque type that abstracts away the lifetime parameter of `ir::Operand` so that
/// cbindgen can generate bindings.
pub struct Operand(ir::Operand<'static>);

/// Create a constant integer operand. The provided type must be an integer type.
#[no_mangle]
pub unsafe extern "C" fn telamon_ir_operand_new_int(
    t: *const ir::Type,
    value: libc::int64_t,
) -> *mut Operand {
    ir::TypeError::check_integer(*t);
    let type_len = unwrap!((*t).len_byte()) as u16;
    let operand = ir::Operand::new_int(value.into(), 8*type_len);
    Box::into_raw(Box::new(Operand(operand)))
}

/// Creates a constant floating point operand. The provided type must be an float type.
#[no_mangle]
pub unsafe extern "C" fn telamon_ir_operand_new_float(
    t: *const ir::Type,
    value: libc::c_double,
) -> *mut Operand {
    ir::TypeError::check_float(*t);
    let type_len = unwrap!((*t).len_byte()) as u16;
    let value = unwrap!(Ratio::from_float(value));
    let operand = ir::Operand::new_float(value, type_len);
    Box::into_raw(Box::new(Operand(operand)))
}

/// Creates an operand that fetches the value of a parameter. The created operand holds
/// a reference to `parameter`.
#[no_mangle]
pub unsafe extern "C" fn telamon_ir_operand_new_parameter(
    parameter: *const ir::Parameter,
) -> *mut Operand {
    let operand = ir::Operand::Param(&*parameter);
    Box::into_raw(Box::new(Operand(operand)))
}

/// Opaque type that abstracts away the lifetime parameter of `ir::Operator` so that
/// cbindgen can generate bindings.
pub struct Operator(ir::Operator<'static>);

/// Creates a `mov` operator. Takes ownership of `operand`.
#[no_mangle]
pub unsafe extern "C" fn telamon_ir_operator_new_mov(
    operand: *mut Operand
) -> *mut Operator {
    let operator = ir::Operator::Mov(Box::from_raw(operand).0);
    Box::into_raw(Box::new(Operator(operator)))
}

/// Creates a binary operator. Takes ownership of the operands.
#[no_mangle]
pub unsafe extern "C" fn telamon_ir_operator_new_binop(
    binop: ir::BinOp,
    lhs: *mut Operand,
    rhs: *mut Operand,
    rounding: ir::op::Rounding,
) -> *mut Operator {
    let lhs = Box::from_raw(lhs).0;
    let rhs = Box::from_raw(rhs).0;
    let operator = ir::Operator::BinOp(binop, lhs, rhs, rounding);
    Box::into_raw(Box::new(Operator(operator)))
}

/// Creates a `mul` operator. The return type can either be the operands type or, if the
/// multplication operates on integers, a type twice the size of the input. Takes
/// ownership of both `lhs` and `rhs`. No references to `return_type` is hold after the
/// function returns.
#[no_mangle]
pub unsafe extern "C" fn telamon_ir_operator_new_mul(
    lhs: *mut Operand,
    rhs: *mut Operand,
    rounding: ir::op::Rounding,
    return_type: *const ir::Type,
) -> *mut Operator {
    let lhs = Box::from_raw(lhs).0;
    let rhs = Box::from_raw(rhs).0;
    let operator = ir::Operator::Mul(lhs, rhs, rounding, *return_type);
    Box::into_raw(Box::new(Operator(operator)))
}

/// Creates a `mad` operator, that computes `mul_lhs * mul_rhs + add_rhs`. If the operator
/// operates on integer, the type of `add_rhs` can either be the type of both `mul_lhs`
/// and `mul_rhs` or an integer type having twice the size of the multiplied types. Takes
/// ownership of `mul_lhs`, `mul_rhs` and `add_rhs`.
#[no_mangle]
pub unsafe extern "C" fn telamon_ir_operator_new_mad(
    mul_lhs: *mut Operand,
    mul_rhs: *mut Operand,
    add_rhs: *mut Operand,
    rounding: ir::op::Rounding,
) -> *mut Operator {
    let mul_lhs = Box::from_raw(mul_lhs).0;
    let mul_rhs = Box::from_raw(mul_rhs).0;
    let add_rhs = Box::from_raw(add_rhs).0;
    let operator = ir::Operator::Mad(mul_lhs, mul_rhs, add_rhs, rounding);
    Box::into_raw(Box::new(Operator(operator)))
}

/// Creates a `cast` operator. Takes ownership of `operand`. No reference to `return_type`
/// is hold after the function returns.
#[no_mangle]
pub unsafe extern "C" fn telamon_ir_operator_new_cast(
    operand: *mut Operand,
    return_type: *const ir::Type,
) -> *mut Operator {
    let operand = Box::from_raw(operand).0;
    let operator = ir::Operator::Cast(operand, *return_type);
    Box::into_raw(Box::new(Operator(operator)))
}

// FIXME: doc
#[no_mangle]
pub unsafe extern "C" fn telamon_ir_operator_new_tensor_load(
    base_address: *mut Operand,
    strided_dims: *const ir::dim::Id,
    strides: *const Size, // FIXME: should we take ownership of sizes
    loaded_type: *const ir::Type,
) -> *mut Operator {
    unimplemented!() // FIXME
}

// FIXME: operator ld, st
// FIXME: operand creation: inst, index, reduce
