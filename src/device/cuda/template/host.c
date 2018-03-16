#include <cuda.h>
#include <stdint.h>
#include <stdio.h>

#define ERROR_BUFF_SIZE 500

static const char* ptx = "{ptx_code}";

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
void cuda_compile_{name}(CUmodule *module, CUfunction *function) {{
  char error_buff[ERROR_BUFF_SIZE+1];
  CUjit_option options[] = 
    {{CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES}};
  void* option_values[] = {{(void*)error_buff, (void*)ERROR_BUFF_SIZE}};
  CUresult err = cuModuleLoadDataEx(module, ptx, 2, options, option_values);
  if (option_values[1]) {{ fprintf(stderr, "%s\n", error_buff); }}
  CHECK_CUDA(err);
  CHECK_CUDA(cuModuleGetFunction(function, *module, "{name}"));
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
