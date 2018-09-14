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
    BinOp_Add,
    /*
     * Substracts two operands.
     */
    BinOp_Sub,
    /*
     * Divides two operands,
     */
    BinOp_Div,
} BinOp;

/*
 * Supported device types for running kernels.
 */
typedef enum {
    DeviceId_X86,
    DeviceId_Cuda,
} DeviceId;

/*
 * The rounding mode of an arithmetic operation.
 */
typedef enum {
    /*
     * No rounding occurs.
     */
    Rounding_Exact,
    /*
     * Rounds toward the nearest number.
     */
    Rounding_Nearest,
    /*
     * Rounds toward zero.
     */
    Rounding_Zero,
    /*
     * Rounds toward positive infinite.
     */
    Rounding_Positive,
    /*
     * Rounds toward negative infinite.
     */
    Rounding_Negative,
} Rounding;

/*
 * Indicates if a telamon function exited correctly.
 */
typedef enum {
    TelamonStatus_Ok,
    TelamonStatus_Fail,
} TelamonStatus;

/*
 * Stores the configuration of the exploration.
 */
typedef struct Config Config;

/*
 * Description of the evaluation context. In particular, in contains the
 * mapping between argument names and argument values.
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
 * A size whose exact value is not yet decided.
 */
typedef struct PartialSize PartialSize;

/*
 * A partially specified implementation.
 */
typedef struct SearchSpace SearchSpace;

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

typedef struct String String;

/*
 * Values and intructions types.
 */
typedef struct Type Type;

/*
 * Provides a unique identifier for iteration dimensions.
 */
typedef struct {
    uint32_t id;
} DimId;

/*
 * Provides a unique identifier for logic dimensions.
 */
typedef struct {
    uint32_t _0;
} LogicalDimId;

/*
 * Uniquely identifies an instruction.
 */
typedef struct {
    uint32_t id;
} InstId;

/*
 * Uniquely identifies a block.
 */
typedef enum {
    MemId_Internal,
    MemId_External,
} MemId_Tag;

typedef struct {
    uint32_t id;
} MemId_Internal_Body;

typedef struct {
    uint32_t id;
} MemId_External_Body;

typedef struct {
    MemId_Tag tag;
    union {
        MemId_Internal_Body internal;
        MemId_External_Body external;
    };
} MemId;

typedef struct {
    uint8_t bits;
} Bool;

typedef struct {
    uint16_t enabled_values;
} NumericSet;

/*
 * Specifies how iteration dimensions are implemented.
 */
typedef struct {
    uint8_t bits;
} DimKind;

/*
 * Indicates where a memory block is located.
 */
typedef struct {
    uint8_t bits;
} MemSpace;

/*
 * Specifies the version of an instruction to use.
 */
typedef struct {
    uint8_t bits;
} InstFlag;

/*
 * Provides a unique identifer for a basic block.
 */
typedef enum {
    StmtId_Inst,
    StmtId_Dim,
} StmtId_Tag;

typedef struct {
    InstId id;
} StmtId_Inst_Body;

typedef struct {
    DimId id;
} StmtId_Dim_Body;

typedef struct {
    StmtId_Tag tag;
    union {
        StmtId_Inst_Body inst;
        StmtId_Dim_Body dim;
    };
} StmtId;

/*
 * Defines how two basic blocks are ordered.
 */
typedef struct {
    uint8_t bits;
} Order;

/*
 * Specifies the valid mappings between two dimensions.
 */
typedef struct {
    uint8_t bits;
} DimMapping;

/*
 * Indicates how are thread dimensions mapped on the GPU.
 */
typedef struct {
    uint8_t bits;
} ThreadMapping;

/*
 * Abstracts integer choices by a range, but only store `min`.
 */
typedef struct {
    uint32_t min;
} HalfRange;

typedef struct {
    uint32_t id;
} InternalId;

/*
 * Abstracts integer choices by a range.
 */
typedef struct {
    uint32_t min;
    uint32_t max;
} Range;

/*
 * A decision to apply to the domain.
 */
typedef enum {
    Action_IsIterationDim,
    Action_IsThreadDim,
    Action_Size,
    Action_DimKind,
    Action_MemSpace,
    Action_InstFlag,
    Action_Order,
    Action_DimMapping,
    Action_ThreadMapping,
    Action_NumThreads,
    Action_NumThreadDims,
    Action_IncrementUnrollFactor,
    Action_UnrollFactor,
    Action_IncrementNumBlockDims,
    Action_NumBlockDims,
    Action_NumNestedInst,
    Action_IncrementMemSize,
    Action_MemSize,
    Action_SharedMemUsed,
    Action_IncrementTilingFactor,
    Action_TilingFactor,
    Action_IsIterationDimClassCounter,
    Action_IsThreadDimClassCounter,
} Action_Tag;

typedef struct {
    InstId inst;
    DimId dim;
    Bool domain;
} Action_IsIterationDim_Body;

typedef struct {
    DimId dim;
    Bool domain;
} Action_IsThreadDim_Body;

typedef struct {
    DimId dim;
    NumericSet domain;
} Action_Size_Body;

typedef struct {
    DimId dim;
    DimKind domain;
} Action_DimKind_Body;

typedef struct {
    MemId mem;
    MemSpace domain;
} Action_MemSpace_Body;

typedef struct {
    InstId inst;
    InstFlag domain;
} Action_InstFlag_Body;

typedef struct {
    StmtId lhs;
    StmtId rhs;
    Order domain;
} Action_Order_Body;

typedef struct {
    DimId lhs;
    DimId rhs;
    DimMapping domain;
} Action_DimMapping_Body;

typedef struct {
    DimId lhs;
    DimId rhs;
    ThreadMapping domain;
} Action_ThreadMapping_Body;

typedef struct {
    HalfRange domain;
} Action_NumThreads_Body;

typedef struct {
    HalfRange domain;
} Action_NumThreadDims_Body;

typedef struct {
    InstId inst;
    DimId dim;
    Bool domain;
} Action_IncrementUnrollFactor_Body;

typedef struct {
    InstId inst;
    HalfRange domain;
} Action_UnrollFactor_Body;

typedef struct {
    InstId inst;
    DimId dim;
    Bool domain;
} Action_IncrementNumBlockDims_Body;

typedef struct {
    InstId inst;
    HalfRange domain;
} Action_NumBlockDims_Body;

typedef struct {
    DimId dim;
    HalfRange domain;
} Action_NumNestedInst_Body;

typedef struct {
    InternalId mem;
    DimId lhs;
    DimId rhs;
    Bool domain;
} Action_IncrementMemSize_Body;

typedef struct {
    InternalId mem;
    HalfRange domain;
} Action_MemSize_Body;

typedef struct {
    HalfRange domain;
} Action_SharedMemUsed_Body;

typedef struct {
    LogicalDimId logical;
    DimId dim;
    Bool domain;
} Action_IncrementTilingFactor_Body;

typedef struct {
    LogicalDimId logical;
    Range domain;
} Action_TilingFactor_Body;

typedef struct {
    InstId inst;
    DimId dim;
    Range domain;
} Action_IsIterationDimClassCounter_Body;

typedef struct {
    DimId dim;
    Range domain;
} Action_IsThreadDimClassCounter_Body;

typedef struct {
    Action_Tag tag;
    union {
        Action_IsIterationDim_Body is_iteration_dim;
        Action_IsThreadDim_Body is_thread_dim;
        Action_Size_Body size;
        Action_DimKind_Body dim_kind;
        Action_MemSpace_Body mem_space;
        Action_InstFlag_Body inst_flag;
        Action_Order_Body order;
        Action_DimMapping_Body dim_mapping;
        Action_ThreadMapping_Body thread_mapping;
        Action_NumThreads_Body num_threads;
        Action_NumThreadDims_Body num_thread_dims;
        Action_IncrementUnrollFactor_Body increment_unroll_factor;
        Action_UnrollFactor_Body unroll_factor;
        Action_IncrementNumBlockDims_Body increment_num_block_dims;
        Action_NumBlockDims_Body num_block_dims;
        Action_NumNestedInst_Body num_nested_inst;
        Action_IncrementMemSize_Body increment_mem_size;
        Action_MemSize_Body mem_size;
        Action_SharedMemUsed_Body shared_mem_used;
        Action_IncrementTilingFactor_Body increment_tiling_factor;
        Action_TilingFactor_Body tiling_factor;
        Action_IsIterationDimClassCounter_Body is_iteration_dim_class_counter;
        Action_IsThreadDimClassCounter_Body is_thread_dim_class_counter;
    };
} Action;

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
 * Frees an explorer configuration.
 */
void telamon_config_free(Config *config);

/*
 * Allocate a new explorer configuration object with suitable
 * defaults.
 *
 * The resulting config object must be freed using
 * `telamon_config_free`.
 */
Config *telamon_config_new(void);

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
 * Run the exploration according to the configuration.
 *
 * Does not take ownership of any of its arguments. The caller is
 * responsible for freeing them after `telamon_explore_all` returns.
 *
 * # Safety
 *
 * * `config` and `context` must point to valid objects of their
 * respective types.
 * * `num_search_spaces` must be non-zero.
 * * `search_space` must point to a sequence of at least
 * `num_search_spaces` valid `SearchSpace` objects.
 */
SearchSpace *telamon_explore(const Config *config,
                             const Context *context,
                             uintptr_t num_search_spaces,
                             const SearchSpace *search_space);

/*
 * Returns the size of a dimension.
 */
PartialSize *telamon_ir_dimension_size(const Function *function, DimId dim);

/*
 * Adds a logical dimension of the given size to the function. In practice, this creates a
 * dimension for each tiling level plus one. Takes ownership of `size` and writes the unique
 * identifier of the logical dimension in `logical_id`. Writes the ids of the dimensions, from the
 * outermost to the innermost, in `dim_ids`. `dim_ids` must be at least of size `num_tiles + 1`.
 * Returns `Ok` except if an error occurs.
 */
TelamonStatus telamon_ir_function_add_dimensions(Function *function,
                                                 Size *size,
                                                 const uint32_t *tile_sizes,
                                                 uintptr_t num_tiles,
                                                 LogicalDimId *logical_id,
                                                 DimId *dim_ids);

/*
 * Adds an instruction performing the given operator in the given dimensions to the
 * function. Writes the unique identifier of the instruction in `inst_id`. Returns
 * `Ok` except if an error occurs. Takes ownership of the operator
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
                                              const PartialSize *strides,
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
                                               const PartialSize *strides,
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
 * Converts an `ir::Size` into an `ir::PartialSize`.
 */
PartialSize *telamon_ir_size_into_partial(Size *size);

/*
 * Multiplies `lhs` by `rhs`.
 */
void telamon_ir_size_mul(PartialSize *lhs, const PartialSize *rhs);

/*
 * Create a size equal to:
 * ```
 * const_factor * param_factors[0] * .. * param_factors[num_params-1]
 * ```
 * The size must be freed calling `telamon_ir_size_free` or passed to a function that
 * takes its ownership.
 */
Size *telamon_ir_size_new(uint32_t const_factor,
                          const Parameter *const *param_factors,
                          uintptr_t num_params);

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

/*
 * Apply a sequence of actions to a search space.
 *
 * # Safety
 *
 * * `search_space` must be a valid pointer containing a valid
 * `SearchSpace` value.
 * * `num_actions` must be non-zero.
 * * `actions` must point to a sequence of at least `num_actions`
 * valid `Action` values.
 */
TelamonStatus telamon_search_space_apply(SearchSpace *search_space,
                                         uintptr_t num_actions,
                                         const Action *actions);

/*
 * Frees a search space instance allocated through
 * `telamon_search_space_new`.
 *
 * # Safety
 *
 * `search_space` must point to a `SearchSpace` object created by
 * `telamon_search_space_new` which has not yet been freed.
 */
TelamonStatus telamon_search_space_free(SearchSpace *search_space);

/*
 * Creates a new search space from an IR function. The caller stays
 * is responsible for freeing the instance and action pointers; the
 * created search space does not keep references to them.
 *
 * Must be freed using `telamon_search_space_free`.
 *
 * # Safety
 *
 * * `ir_instance` must point to a valid `Function` value.
 * * `actions` must point to a sequence of at least `num_actions`
 * valid `Action` values, unless `num_actions` is 0 in which case
 * `actions` is not used.
 */
SearchSpace *telamon_search_space_new(const Function *ir_instance,
                                      uintptr_t num_actions,
                                      const Action *actions);

/*
 * Prints the error message in a string. Returns `null` if no error was
 * present. The caller is responsible for freeing the string with `free`.
 */
char *telamon_strerror(void);

/*
 * Copy a C string pointer into a Rust String object. Use this to set
 * string-valued configuration options.
 *
 * # Safety
 *
 * `dst` must point to a valid Rust String object and `src` must
 * point to a NULL-terminated C string.
 */
TelamonStatus telamon_string_copy(String *dst, const char *src);

#endif /* TELAMON_CAPI_H */
