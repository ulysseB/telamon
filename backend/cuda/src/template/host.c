#include <cuda.h>
#include <stdint.h>
#include <stdio.h>

#define ERROR_BUFF_SIZE 500

static const char* ptx = "{ptx_code}";
static size_t ptx_len = {ptx_len};

int32_t int32_t_ceil_log2(int32_t val) {{
  val -= 1;
  int32_t log = 1;
  while (val >>= 1)
    ++log;
  return log;
}}

void int32_t_magic(int32_t divisor, int32_t *magic, int32_t *shift) {{
  int32_t l = int32_t_ceil_log2(divisor);
  l = l > 1 ? l : 1;

  uint64_t m = 1 + ((uint64_t)1 << (32 + l - 1)) / (uint64_t)divisor;
  *magic = (int32_t)(m - ((uint64_t)1 << 32));
  *shift = l - 1;
}}

// Checks the result of a CUDA dirver API call and exits in case of error.
static void check_cuda(CUresult err, const char* file, const int line) {{
  if (err != CUDA_SUCCESS) {{
    const char* err_name;
    const char* err_desc;
    cuGetErrorName(err, &err_name);
    cuGetErrorString(err, &err_desc);
    fprintf(stderr, "CUDA driver API error %s: %s, line: %i, file: %s",
        err_name, err_desc, line, file);
    exit(-1);
  }}
}}

#define CHECK_CUDA(err) check_cuda(err, __FILE__, __LINE__)

// Compiles the generated PTX code and stores it in the given module and function.
//
// NB: We use cuLinkCreate / cuLinkAddData / cuLinkComplete for the compilation
// process to match what is done during the search.  For currently unknown
// reasons, passing the PTX directly to cuModuleLoadDataEx does not generate
// the same code and can have wildly different performance characteristics.
void cuda_compile_{name}(CUmodule *module, CUfunction *function) {{
  char error_buff[ERROR_BUFF_SIZE+1];
  CUjit_option options[] = 
    {{CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES}};
  void* option_values[] = {{(void*)error_buff, (void*)ERROR_BUFF_SIZE}};

  CUlinkState state;
  CUresult err = cuLinkCreate(2, options, option_values, &state);
  if (option_values[1]) {{ fprintf(stderr, "%s\n", error_buff); }}
  CHECK_CUDA(err);

  CHECK_CUDA(cuLinkAddData(state, CU_JIT_INPUT_PTX, (void *)ptx, ptx_len, NULL, 0, NULL, NULL));

  void* cubin_data;;
  size_t cubin_size;
  CHECK_CUDA(cuLinkComplete(state, &cubin_data, &cubin_size));

  CHECK_CUDA(cuModuleLoadData(module, cubin_data));
  CHECK_CUDA(cuModuleGetFunction(function, *module, "{name}"));

  CHECK_CUDA(cuLinkDestroy(state));
}}

// Executes the generated kernel and returns the execution time in milliseconds.
float cuda_execute_{name}(CUfunction function, {extern_params}) {{
  {extra_def}
  void* params[] = {param_vec};
  CUevent start, stop;
  CHECK_CUDA(cuEventCreate(&start, CU_EVENT_DEFAULT));
  CHECK_CUDA(cuEventCreate(&stop, CU_EVENT_DEFAULT));
  CHECK_CUDA(cuEventRecord(start, 0));
  CHECK_CUDA(cuLaunchKernel(
        function, {b_dim_x}, {b_dim_y}, {b_dim_z}, {t_dim_x}, {t_dim_y}, {t_dim_z},
        0, NULL, params, NULL));
  CHECK_CUDA(cuEventRecord(stop, 0));
  CHECK_CUDA(cuEventSynchronize(stop));
  float ms = -1;
  CHECK_CUDA(cuEventElapsedTime(&ms, start, stop));
  CHECK_CUDA(cuEventDestroy(start));
  CHECK_CUDA(cuEventDestroy(stop));
  {extra_cleanup}
  return ms; 
}}

// Compiles and executes the function and returns its execution time.
float cuda_compile_executes_{name}({extern_params}) {{
  CUmodule module;
  CUfunction function;
  cuda_compile_{name}(&module, &function);
  float ms = cuda_execute_{name}(function, {extern_param_names});
  CHECK_CUDA(cuModuleUnload(module));
  return ms;
}}
