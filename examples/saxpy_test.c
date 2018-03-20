// Provides an interface with the CUDA driver API.

#include <cuda.h>
#include <curand.h>
#include <cublas_v2.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#define ERROR_BUFF_SIZE 500

float cuda_compile_executes_saxpy(int32_t n, float a, CUdeviceptr x, CUdeviceptr y);

// Checks the result of a CUDA dirver API call and exits in case of error.
static void check_cuda(CUresult err, const char* file, const int line) {
  if (err != CUDA_SUCCESS) {
    const char* err_name;
    const char* err_desc;
    cuGetErrorName(err, &err_name);
    cuGetErrorString(err, &err_desc);
    fprintf(stderr, "CUDA driver API error %s: %s, line: %i, file: %s\n",
        err_name, err_desc, line, file);
    exit(-1);
  }
}

#define CHECK_CUDA(err) check_cuda(err, __FILE__, __LINE__)

// Checks the result of a CURAND API call and exits in case of error.
static void check_curand(curandStatus_t err, const char* file, const int line) {
  if (err != CURAND_STATUS_SUCCESS) {
    fprintf(stderr, "CURAND error line: %i, file: %s", line, file);
    exit(-1);
  }
}

#define CHECK_CURAND(err) check_curand(err, __FILE__, __LINE__)

// Represents a CUDA execution context.
typedef struct CudaContext {
  CUdevice device;
  CUcontext ctx;
  curandGenerator_t rng;
} CudaContext;

// Initalize a CUDA execution context. This context may only be used from the thread that
// created it (https://devtalk.nvidia.com/default/topic/519087/cuda-context-and-threading/).
CudaContext init_cuda(uint64_t seed) {
  CudaContext context;
  CHECK_CUDA(cuInit(0));
  CHECK_CUDA(cuDeviceGet(&context.device, 0));
  CHECK_CUDA(cuCtxCreate(&context.ctx, CU_CTX_SCHED_AUTO, context.device));
  CHECK_CURAND(curandCreateGenerator(&context.rng, CURAND_RNG_PSEUDO_DEFAULT));
  CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(context.rng, seed));
  return context;
}

// Frees a CUDA execution context.
void free_cuda(CudaContext* context) {
  CHECK_CURAND(curandDestroyGenerator(context->rng));
  CHECK_CUDA(cuCtxDestroy(context->ctx));
}

// Fills a float array with random data.
CUdeviceptr random_array(CudaContext* ctx, uint64_t size, float mean, float stddev) {
  CUdeviceptr array;
  CHECK_CUDA(cuMemAlloc(&array, size * sizeof(float)));
  CHECK_CURAND(curandGenerateNormal(ctx->rng, (float*)array, size, mean, stddev));
  return array;
}

// Copies an array to an other on the device.
CUdeviceptr copy_array(CUdeviceptr src, uint64_t size) {
  CUdeviceptr array;
  CHECK_CUDA(cuMemAlloc(&array, size));
  CHECK_CUDA(cuMemcpyDtoD(array, src, size));
  return array;
}

float compare_arrays(CUdeviceptr lhs, CUdeviceptr rhs, uint64_t size) {
  float* lhs_host = (float*)malloc(size * sizeof(float));
  float* rhs_host = (float*)malloc(size * sizeof(float));
  CHECK_CUDA(cuMemcpyDtoH(lhs_host, lhs, size * sizeof(float)));
  CHECK_CUDA(cuMemcpyDtoH(rhs_host, rhs, size * sizeof(float)));
  float diff = 0;
  for(uint64_t i=0; i<size; ++i) {
    float d = 2*fabs(lhs_host[i]-rhs_host[i])/(fabs(lhs_host[i])+fabs(rhs_host[i]));
    if (d > 1e-6f) {
      fprintf(stderr, "%f expected, got %f", rhs_host[i], lhs_host[i]);
      exit(-1);
    } 
    if (d > diff) { diff = d; }
  } 
  return diff;
}

int main() {
  int N = 1024 * 1024 *64;
  float a = 1.234;

  CudaContext context = init_cuda(0);
  CUdeviceptr x = random_array(&context, N, 0, 1);
  CUdeviceptr y = random_array(&context, N, 0, 1);
  CUdeviceptr y_ref = copy_array(y, N*sizeof(float));
  
  float t = cuda_compile_executes_saxpy(N, a, x, y); 

  cublasHandle_t h;
  cublasCreate(&h);
  
  CUevent cublas_start, cublas_stop;
  float cublas_t = -1;
  CHECK_CUDA(cuEventCreate(&cublas_start, CU_EVENT_DEFAULT));
  CHECK_CUDA(cuEventCreate(&cublas_stop, CU_EVENT_DEFAULT));
  CHECK_CUDA(cuCtxSynchronize());
  CHECK_CUDA(cuEventRecord(cublas_start, 0));
  cublasSaxpy(h, N, &a, (float*)x, 1, (float*)y_ref, 1); 
  cublasDestroy(h);
  CHECK_CUDA(cuEventRecord(cublas_stop, 0));
  CHECK_CUDA(cuEventSynchronize(cublas_stop));
  CHECK_CUDA(cuEventElapsedTime(&cublas_t, cublas_start, cublas_stop));
  CHECK_CUDA(cuEventDestroy(cublas_start));
  CHECK_CUDA(cuEventDestroy(cublas_stop));

  float diff = compare_arrays(y, y_ref, N);

  CHECK_CUDA(cuMemFree(x));
  CHECK_CUDA(cuMemFree(y));
  CHECK_CUDA(cuMemFree(y_ref));
  free_cuda(&context);

  printf("exec: %fms, cublas: %fms, diff: %f\n", t, cublas_t, diff);
}
