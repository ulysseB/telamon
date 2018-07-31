#ifndef TELAMON_CAPI_H
#define TELAMON_CAPI_H

/* DO NOT MODIFY THIS MANUALLY!
 * This file has been generated usin cbindgen.
 *
 * To generate this file:
 *  1. Get the latest cbindgen using `cargo install --force cbindgen --git https://github.com/Elarnon/cbindgen`
 *  2. Run `rustup run nightly cbindgen telamon-capi -o telamon-capi/include/telamon.h`
 */

#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

/*
 * Represents binary arithmetic operators.
 */
typedef enum {
    /*
     * Adds two operands.
     */
    Add,
    /*
     * Substracts two operands.
     */
    Sub,
    /*
     * Divides two operands,
     */
    Div,
} BinOp;

/*
 * Supported device types for running kernels.
 */
typedef enum {
    X86,
    Cuda,
} DeviceId;

/*
 * The rounding mode of an arithmetic operation.
 */
typedef enum {
    /*
     * No rounding occurs.
     */
    Exact,
    /*
     * Rounds toward the nearest number.
     */
    Nearest,
    /*
     * Rounds toward zero.
     */
    Zero,
    /*
     * Rounds toward positive infinite.
     */
    Positive,
    /*
     * Rounds toward negative infinite.
     */
    Negative,
} Rounding;

/*
 * Indicates if a telamon function exited correctly.
 */
typedef enum {
    TelamonStatusOk,
    TelamonStatusFail,
} TelamonStatus;

/*
 * Description of the evaluation context. In particular, in contains the mapping between
 * argument names and argument values.
 */
typedef struct Context Context;

/*
 * Opaque type that contains the mapping of kernel parameters to actual values.
 */
typedef struct CudaEnvironment CudaEnvironment;

/*
 * Description of the targeted device.
 */
typedef struct Device Device;

/*
 * A function ready to execute on a device, derived from a constrained IR instance.
 */
typedef struct Function Function;

/*
 * Supported kernels.
 */
typedef struct KernelParameters KernelParameters;

/*
 * Opaque type that abstracts away the lifetime parameter of `ir::Operand` so that
 * cbindgen can generate bindings.
 */
typedef struct Operand Operand;

/*
 * Opaque type that abstracts away the lifetime parameter of `ir::Operator` so that
 * cbindgen can generate bindings.
 */
typedef struct Operator Operator;

/*
 * Represents an argument of a function.
 */
typedef struct Parameter Parameter;

/*
 * Holds the signature of a function.
 */
typedef struct Signature Signature;

/*
 * The size of an iteration dimension. The size is of the form:
 * `(factor * dividend_0 * dividend_1 * ...)) / divisor`
 * where the reminder of the division is null.
 */
typedef struct Size Size;

/*
 * Values and intructions types.
 */
typedef struct Type Type;

/*
 * Provides a unique identifier for iteration dimensions.
 */
typedef struct {
    uint32_t _0;
} DimId;

/*
 * Uniquely identifies an instruction.
 */
typedef struct {
    uint32_t _0;
} InstId;

/*
 * Uniquely identifies a block.
 */
typedef enum {
    Internal,
    External,
} MemId_Tag;

typedef struct {
    uint32_t _0;
} Internal_Body;

typedef struct {
    uint32_t _0;
} External_Body;

typedef struct {
    MemId_Tag tag;
    union {
        Internal_Body internal;
        External_Body external;
    };
} MemId;

/*
 * Initializes the logger.
 */
void env_logger_try_init(void);

/*
 * Deallocates kernel parameters created through one of the `kernel_*_new`
 * functions. The `params` pointer becomes invalid and must not be used again
 * after calling `kernel_free`.
 */
void kernel_free(KernelParameters *params);

/*
 * Instanciate a new kernel for matrix-matrix multiplication. The
 * caller is responsible for deallocating the returned pointer using
 * kernel_free. The tile_m, tile_n and tile_k parameters are read
 * from during the call, but no pointer to the corresponding data is
 * kept afterwards.
 */
KernelParameters *kernel_matmul_new(int m,
                                    int n,
                                    int k,
                                    unsigned int a_stride,
                                    int transpose_a,
                                    int transpose_b,
                                    int generic,
                                    const uint32_t *tile_m,
                                    size_t tile_m_len,
                                    const uint32_t *tile_n,
                                    size_t tile_n_len,
                                    const uint32_t *tile_k,
                                    size_t tile_k_len);

/*
 * Optimize a kernel on a given device. `config_data` points to a JSON-encoded
 * string of length `config_len` containing the configuration parameters for
 * the explorer.
 */
bool kernel_optimize(KernelParameters *params,
                     DeviceId device,
                     const char *config_data,
                     size_t config_len);

/*
 * Allocates and binds an array to the given parameter. `size` is given in bytes.
 *
 * The allocated array is managed by the context and doesn't need to be explicitely
 * destroyed.
 */
void telamon_cuda_bind_array(CudaEnvironment *env, const Parameter *param, size_t size);

/*
 * Binds a `double` to a parameter.
 */
void telamon_cuda_bind_double(CudaEnvironment *env, const Parameter *param, double value);

/*
 * Binds a `float` to a parameter.
 */
void telamon_cuda_bind_float(CudaEnvironment *env, const Parameter *param, float value);

/*
 * Binds an `int16_t` to a parameter.
 */
void telamon_cuda_bind_int16(CudaEnvironment *env, const Parameter *param, int16_t value);

/*
 * Binds an `int32_t` to a parameter.
 */
void telamon_cuda_bind_int32(CudaEnvironment *env, const Parameter *param, int32_t value);

/*
 * Binds an `int64_t` to a parameter.
 */
void telamon_cuda_bind_int64(CudaEnvironment *env, const Parameter *param, int64_t value);

/*
 * Binds an `int8_t` to a parameter.
 */
void telamon_cuda_bind_int8(CudaEnvironment *env, const Parameter *param, int8_t value);

/*
 * Destroys a CUDA context.
 */
void telamon_cuda_environment_free(CudaEnvironment *env);

/*
 * Returns a pointer to a CUDA environement. The caller is responsible for deallocating
 * the pointer by calling `telamon_cuda_destroy_environment`.
 */
CudaEnvironment *telamon_cuda_environment_new(void);

/*
 * Returns a pointer to ithe evaluation context.
 */
const Context *telamon_cuda_get_context(const CudaEnvironment *env);

/*
 * Returns a pointer to the description of the target device.
 */
const Device *telamon_cuda_get_device(const CudaEnvironment *env);

/*
 * Adds a dimension of the given size to the function. Takes ownership of `size` and
 * writes the unique identifier of the dimension in `dim_id`. Returns `TelamonStatusOk`
 * except if an error occurs.
 */
TelamonStatus telamon_ir_function_add_dimension(Function *function, Size *size, DimId *dim_id);

/*
 * Adds an instruction performing the given operator in the given dimensions to the
 * function. Writes the unique identifier of the instruction in `inst_id`. Returns
 * `TelamonStatusOk` except if an error occurs. Takes ownership of the operator
 * but does not keeps any reference to `dimensions`.
 */
TelamonStatus telamon_ir_function_add_instruction(Function *function,
                                                  Operator *operator,
                                                  const DimId *dimensions,
                                                  uintptr_t num_dimensions,
                                                  InstId *inst_id);

/*
 * Frees a function allocated with `telamon_ir_function_new`.
 */
void telamon_ir_function_free(Function *function);

/*
 * Creates a function to optimize. The function must be freed with
 * `telamon_ir_function_free`. `signature` and `device` must outlive the function.
 */
Function *telamon_ir_function_new(const Signature *signature, const Device *device);

/*
 * Creates a constant floating point operand. The provided type must be a float type.
 * Returns `null` if an error is encountered.
 */
Operand *telamon_ir_operand_new_float(const Type *t, double value);

/*
 * Creates an operand that returns the current index on a dimension.
 */
Operand *telamon_ir_operand_new_index(DimId dim);

/*
 * Creates an operand that references the value of an instruction. The value of the
 * instruction is transmitted point-to-point between the source dimensions (`src_dims`,
 * in which the instruction is produced) and destination dimensions (`dst_dims`, in which
 * the operand is used). `num_mapped_dims` indicates the number of dimensions in
 * `src_dims` and in `dst_dims`. If `allow_tmp_mem` is non-zero, Telamon can allocate
 * memory to transfer data between the two loop nests. Otherwise, it makes sure the data
 * can be stored in registers (for example by fusing or unrolling loops).
 */
Operand *telamon_ir_operand_new_inst(const Function *function,
                                     InstId inst,
                                     const DimId *src_dims,
                                     const DimId *dst_dims,
                                     uintptr_t num_mapped_dims,
                                     int allow_tmp_mem);

/*
 * Create a constant integer operand. The provided type must be an integer type.
 * Returns `null` if an error is encountered.
 */
Operand *telamon_ir_operand_new_int(const Type *t, int64_t value);

/*
 * Creates an operand that fetches the value of a parameter. The created operand holds
 * a reference to `parameter`.
 */
Operand *telamon_ir_operand_new_parameter(const Parameter *parameter);

/*
 * Creates an operand that take the value of `init_inst` the first time is is encountered
 * and then reuse the value produced by the instruction using the operand, effectivelly
 * creating a reduction. The value is is transmitted point-to-point between the source
 * dimensions (`src_dims`, in which `init_inst` is produced) and destination dimensions
 * (`dst_dims`, in which the operand is used). `num_mapped_dims` indicates the number of
 * dimensions in `src_dims` and in `dst_dims`. `reduction_dims` indicates on which
 * dimensions the reduction occurs: values are not reused accross other dimensions.
 */
Operand *telamon_ir_operand_new_reduction(const Function *function,
                                          InstId init_inst,
                                          const DimId *src_dims,
                                          const DimId *dst_dims,
                                          uintptr_t num_mapped_dims,
                                          const DimId *reduction_dims,
                                          uintptr_t num_reduction_dims);

/*
 * Creates a binary operator. Takes ownership of the operands.
 */
Operator *telamon_ir_operator_new_binop(BinOp binop, Operand *lhs, Operand *rhs, Rounding rounding);

/*
 * Creates a `cast` operator. Takes ownership of `operand`. No reference to `return_type`
 * is hold after the function returns.
 */
Operator *telamon_ir_operator_new_cast(Operand *operand, const Type *return_type);

/*
 * Creates a `mad` operator, that computes `mul_lhs * mul_rhs + add_rhs`. If the operator
 * operates on integer, the type of `add_rhs` can either be the type of both `mul_lhs`
 * and `mul_rhs` or an integer type having twice the size of the multiplied types. Takes
 * ownership of `mul_lhs`, `mul_rhs` and `add_rhs`.
 */
Operator *telamon_ir_operator_new_mad(Operand *mul_lhs,
                                      Operand *mul_rhs,
                                      Operand *add_rhs,
                                      Rounding rounding);

/*
 * Creates a `mov` operator. Takes ownership of `operand`.
 */
Operator *telamon_ir_operator_new_mov(Operand *operand);

/*
 * Creates a `mul` operator. The return type can either be the operands type or, if the
 * multplication operates on integers, a type twice the size of the input. Takes
 * ownership of both `lhs` and `rhs`. No references to `return_type` is hold after the
 * function returns.
 */
Operator *telamon_ir_operator_new_mul(Operand *lhs,
                                      Operand *rhs,
                                      Rounding rounding,
                                      const Type *return_type);

/*
 * Creates an operator that loads a tensor stored in memory. Takes the ownership of
 * `base_address` and creates copies of `strided_dims`, `strides` and `loaded_type`.
 * This function also adds the necessary address computation code to `function`.
 */
Operator *telamon_ir_operator_new_tensor_load(Function *function,
                                              MemId array_id,
                                              Operand *base_address,
                                              const DimId *strided_dims,
                                              const Size *strides,
                                              uintptr_t num_strided_dims,
                                              const Type *loaded_type);

/*
 * Creates an operator that stores a tensor in memory. Takes the ownership of
 * `base_address` and `value` and creates copies of `strided_dims`, `strides` and
 * `loaded_type`. This function also adds the necessary address computation code to
 * `function`.
 */
Operator *telamon_ir_operator_new_tensor_store(Function *function,
                                               MemId array_id,
                                               Operand *base_address,
                                               const DimId *strided_dims,
                                               const Size *strides,
                                               uintptr_t num_strided_dims,
                                               Operand *value);

/*
 * Adds an array parameter to the function signature.
 */
MemId telamon_ir_signature_add_array(Signature *signature, const char *name);

/*
 * Adds a scalar parameter to the function signature.
 */
void telamon_ir_signature_add_scalar(Signature *signature, const char *name, const Type *t);

/*
 * Deallocates a signature created with `telamon_ir_signature_new`.
 */
void telamon_ir_signature_free(Signature *signature);

/*
 * Creates a function signature that must be deallocated with
 * `telamon_ir_signature_free`.
 */
Signature *telamon_ir_signature_new(const char *name);

/*
 * Returns the parameter at the given position.
 */
const Parameter *telamon_ir_signature_param(const Signature *signature, uintptr_t index);

/*
 * Frees a size allocated with `telamon_ir_size_new`.
 */
void telamon_ir_size_free(Size *size);

/*
 * Create a size equal to:
 * ```
 * const_factor * param_factors[0] * .. * param_factors[num_params-1] / const_divisor
 * ```
 * The size must be freed calling `telamon_ir_size_free` or passed to a function that
 * takes its ownership.
 */
Size *telamon_ir_size_new(uint32_t const_factor,
                          uint32_t const_divisor,
                          const Parameter *const *param_factors,
                          uintptr_t num_params);

/*
 * Prints the error message in a string. Returns `null` if no error was present. The
 * caller is responsible for freeing the string with `free`.
 */
char *telamon_ir_strerror(void);

/*
 * Frees a type allocated with `telamon_ir_type_new_int` or `telamon_ir_type_new_float`.
 */
void telamon_ir_type_free(Type *t);

/*
 * Creates a floating point type that must be freed with `telamon_ir_type_free`.
 */
Type *telamon_ir_type_new_float(uint16_t num_bits);

/*
 * Creates an integer type that must be freed with `telamon_ir_type_free`.
 */
Type *telamon_ir_type_new_int(uint16_t num_bits);

#endif /* TELAMON_CAPI_H */
