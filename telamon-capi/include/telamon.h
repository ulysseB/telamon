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
 * Indicates how to choose between nodes of the search tree when no children have been
 * evaluated.
 */
typedef enum {
    /*
     * Consider the nodes in the order given by the search space API.
     */
    Api,
    /*
     * Consider the nodes in a random order.
     */
    Random,
    /*
     * Consider the nodes with the lowest bound first.
     */
    Bound,
    /*
     * Consider the nodes with a probability proportional to the distance between the
     * cut and the bound.
     */
    WeightedRandom,
} NewNodeOrder;

/*
 * Indicates how to choose between nodes of the search tree with at least one descendent
 * evaluated.
 */
typedef enum {
    /*
     * Use the weights from the bandit algorithm.
     */
    Bandit,
    /*
     * Take the candidate with the best bound.
     */
    Bound,
    /*
     * Consider the nodes with a probability proportional to the distance between the
     * cut and the bound.
     */
    WeightedRandom,
} OldNodeOrder;

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
 * Provides a unique identifer for a basic block.
 */
typedef struct BBId BBId;

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

typedef struct InternalId InternalId;

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

typedef struct Option_f64 Option_f64;

typedef struct Option_u64 Option_u64;

typedef struct Option_usize Option_usize;

/*
 * Represents an argument of a function.
 */
typedef struct Parameter Parameter;

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
 * Configuration parameters specific to the multi-armed bandit algorithm.
 */
typedef struct {
    /*
     * Indicates how to select between nodes of the search tree when none of their
     * children have been evaluated.
     */
    NewNodeOrder new_nodes_order;
    /*
     * Indicates how to choose between nodes with at least one children evaluated.
     */
    OldNodeOrder old_nodes_order;
    /*
     * The number of best execution times to remember.
     */
    uintptr_t threshold;
    /*
     * The biggest delta is, the more focused on the previous best candidates the
     * exploration is.
     */
    double delta;
    /*
     * If true, does not expand tree until end - instead, starts a montecarlo descend after each
     * expansion of a node
     */
    bool monte_carlo;
} BanditConfig;

/*
 * Exploration algorithm to use.
 */
typedef enum {
    /*
     * Evaluate all the candidates that cannot be pruned.
     */
    BoundOrder,
    /*
     * Use a multi-armed bandit algorithm.
     */
    MultiArmedBandit,
} SearchAlgorithm_Tag;

typedef struct {
    BanditConfig _0;
} MultiArmedBandit_Body;

typedef struct {
    SearchAlgorithm_Tag tag;
    union {
        MultiArmedBandit_Body multi_armed_bandit;
    };
} SearchAlgorithm;

/*
 * Stores the configuration of the exploration.
 */
typedef struct {
    /*
     * Name of the file in wich to store the logs.
     */
    String log_file;
    /*
     * Name of the file in which to store the binary event log.
     */
    String event_log;
    /*
     * Number of exploration threads.
     */
    uintptr_t num_workers;
    /*
     * Indicates the search must be stopped if a candidate with an execution time better
     * than the bound (in ns) is found.
     */
    Option_f64 stop_bound;
    /*
     * Indicates the search must be stopped after the given number of minutes.
     */
    Option_u64 timeout;
    /*
     * Indicates the search must be stopped after the given number of
     * candidates have been evaluated.
     */
    Option_usize max_evaluations;
    /*
     * A percentage cut indicate that we only care to find a candidate that is in a
     * certain range above the best Therefore, if cut_under is 20%, we can discard any
     * candidate whose bound is above 80% of the current best.
     */
    Option_f64 distance_to_best;
    /*
     * Exploration algorithm to use. Needs to be last for TOML serialization, because it is a table.
     */
    SearchAlgorithm algorithm;
} Config;

/*
 * Provides a unique identifier for iteration dimensions.
 */
typedef struct {
    uint32_t id;
} DimId;

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
    Internal,
    External,
} MemId_Tag;

typedef struct {
    uint32_t id;
} Internal_Body;

typedef struct {
    uint32_t id;
} External_Body;

typedef struct {
    MemId_Tag tag;
    union {
        Internal_Body internal;
        External_Body external;
    };
} MemId;

typedef struct {
    uint8_t bits;
} Bool;

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
    IsIterationDim,
    IsThreadDim,
    DimKind,
    MemSpace,
    InstFlag,
    Order,
    DimMapping,
    ThreadMapping,
    NumThreads,
    NumThreadDims,
    IncrementUnrollFactor,
    UnrollFactor,
    IncrementNumBlockDims,
    NumBlockDims,
    NumNestedInst,
    IncrementMemSize,
    MemSize,
    SharedMemUsed,
    IsIterationDimClassCounter,
    IsThreadDimClassCounter,
} Action_Tag;

typedef struct {
    InstId inst;
    DimId dim;
    Bool domain;
} IsIterationDim_Body;

typedef struct {
    DimId dim;
    Bool domain;
} IsThreadDim_Body;

typedef struct {
    DimId dim;
    DimKind domain;
} DimKind_Body;

typedef struct {
    MemId mem;
    MemSpace domain;
} MemSpace_Body;

typedef struct {
    InstId inst;
    InstFlag domain;
} InstFlag_Body;

typedef struct {
    BBId lhs;
    BBId rhs;
    Order domain;
} Order_Body;

typedef struct {
    DimId lhs;
    DimId rhs;
    DimMapping domain;
} DimMapping_Body;

typedef struct {
    DimId lhs;
    DimId rhs;
    ThreadMapping domain;
} ThreadMapping_Body;

typedef struct {
    HalfRange domain;
} NumThreads_Body;

typedef struct {
    HalfRange domain;
} NumThreadDims_Body;

typedef struct {
    InstId inst;
    DimId dim;
    Bool domain;
} IncrementUnrollFactor_Body;

typedef struct {
    InstId inst;
    HalfRange domain;
} UnrollFactor_Body;

typedef struct {
    InstId inst;
    DimId dim;
    Bool domain;
} IncrementNumBlockDims_Body;

typedef struct {
    InstId inst;
    HalfRange domain;
} NumBlockDims_Body;

typedef struct {
    DimId dim;
    HalfRange domain;
} NumNestedInst_Body;

typedef struct {
    InternalId mem;
    DimId lhs;
    DimId rhs;
    Bool domain;
} IncrementMemSize_Body;

typedef struct {
    InternalId mem;
    HalfRange domain;
} MemSize_Body;

typedef struct {
    HalfRange domain;
} SharedMemUsed_Body;

typedef struct {
    InstId inst;
    DimId dim;
    Range domain;
} IsIterationDimClassCounter_Body;

typedef struct {
    DimId dim;
    Range domain;
} IsThreadDimClassCounter_Body;

typedef struct {
    Action_Tag tag;
    union {
        IsIterationDim_Body is_iteration_dim;
        IsThreadDim_Body is_thread_dim;
        DimKind_Body dim_kind;
        MemSpace_Body mem_space;
        InstFlag_Body inst_flag;
        Order_Body order;
        DimMapping_Body dim_mapping;
        ThreadMapping_Body thread_mapping;
        NumThreads_Body num_threads;
        NumThreadDims_Body num_thread_dims;
        IncrementUnrollFactor_Body increment_unroll_factor;
        UnrollFactor_Body unroll_factor;
        IncrementNumBlockDims_Body increment_num_block_dims;
        NumBlockDims_Body num_block_dims;
        NumNestedInst_Body num_nested_inst;
        IncrementMemSize_Body increment_mem_size;
        MemSize_Body mem_size;
        SharedMemUsed_Body shared_mem_used;
        IsIterationDimClassCounter_Body is_iteration_dim_class_counter;
        IsThreadDimClassCounter_Body is_thread_dim_class_counter;
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
TelamonStatus telamon_config_free(Config *config);

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
