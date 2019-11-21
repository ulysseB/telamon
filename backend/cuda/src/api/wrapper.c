// Provides an interface with the CUDA driver API.

#include <cuda.h>
#include <cupti.h>
#include <curand.h>
#include <stdint.h>
#include <stdio.h>

#define ERROR_BUFF_SIZE 500

// Checks the result of a CUDA dirver API call and exits in case of error.
int check_cuda(CUresult err, const char* file, const int line) {
  if (err != CUDA_SUCCESS) {
    const char* err_name;
    const char* err_desc;
    cuGetErrorName(err, &err_name);
    cuGetErrorString(err, &err_desc);
    fprintf(stderr, "CUDA driver API error %s: %s, line: %i, file: %s\n",
        err_name, err_desc, line, file);
    return -1;
  } else {
    return 0;
  }
}

#define CHECK_CUDA(err) if (check_cuda(err, __FILE__, __LINE__) != 0) { return -1; }
#define HARD_CHECK_CUDA(err) if (check_cuda(err, __FILE__, __LINE__) != 0) { exit(-1); }

// Checks the result of a CURAND API call and exits in case of error.
void check_curand(curandStatus_t err, const char* file, const int line) {
  if (err != CURAND_STATUS_SUCCESS) {
    fprintf(stderr, "CURAND error line: %i, file: %s\n", line, file);
    exit(-1);
  }
}

#define CHECK_CURAND(err) check_curand(err, __FILE__, __LINE__)

// Checks the results of a CUPTI API call and exits in case of error.
void check_cupti(CUptiResult err, const char* file, const int line) {
  if (err != CUPTI_SUCCESS) {
    const char* err_name;
    cuptiGetResultString(err, &err_name);
    fprintf(stderr, "CUPTI error: %s, line: %i, file: %s\n", err_name, line, file);
    exit(-1);
  }
}

#define CHECK_CUPTI(err) check_cupti(err, __FILE__, __LINE__)

// Represents a CUDA execution context.
typedef struct CudaContext {
  CUdevice device;
  CUcontext ctx;
  curandGenerator_t rng;
  CUpti_EventGroup num_cycle_event;
} CudaContext;

// Holds informations on an event group.
typedef struct EventGroupInfo {
  uint32_t num_event;
  uint32_t total_num_instance;
  uint32_t actual_num_instance;
} EventGroupInfo;

// Holds a set of sets of events and its associated EventGroupInfo.
typedef struct EventSets {
  CUpti_EventGroupSets* sets;
  EventGroupInfo* groups_info;
  // The maximum number of num_event * num_instance in a group.
  uint32_t max_num_value_per_group;
} EventSets;

// Initalize a CUDA execution context. This context may only be used from the thread that
// created it (https://devtalk.nvidia.com/default/topic/519087/cuda-context-and-threading/).
CudaContext* init_cuda(uint64_t seed) {
  CudaContext* context = malloc(sizeof(CudaContext));
  HARD_CHECK_CUDA(cuInit(0));
  HARD_CHECK_CUDA(cuDeviceGet(&context->device, 0));
  HARD_CHECK_CUDA(cuCtxCreate(&context->ctx, CU_CTX_SCHED_AUTO, context->device));
  CHECK_CURAND(curandCreateGenerator(&context->rng, CURAND_RNG_PSEUDO_DEFAULT));
  CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(context->rng, seed));
  CHECK_CUPTI(cuptiEventGroupCreate(context->ctx, &context->num_cycle_event, 0));
  CUpti_EventID event;
  CHECK_CUPTI(cuptiEventGetIdFromName(context->device, "elapsed_cycles_sm", &event));
  CHECK_CUPTI(cuptiEventGroupAddEvent(context->num_cycle_event, event));
  return context;
}

// Frees a CUDA execution context.
void free_cuda(CudaContext* context) {
  HARD_CHECK_CUDA(cuCtxSetCurrent(context->ctx));
  CHECK_CUPTI(cuptiEventGroupDestroy(context->num_cycle_event));
  CHECK_CURAND(curandDestroyGenerator(context->rng));
  HARD_CHECK_CUDA(cuCtxDestroy(context->ctx));
  free(context);
}

// Returns the name of the CUDA device
char* device_name(const CudaContext* context) {
  char* name = malloc(20*sizeof(char));
  HARD_CHECK_CUDA(cuDeviceGetName(name, 20, context->device));
  return name;
}

// Returns the value of a CUDA attributes.
int32_t device_attribute(const CudaContext* context, uint32_t attr) {
  int res;
  HARD_CHECK_CUDA(cuDeviceGetAttribute(&res, (CUdevice_attribute)attr, context->device));
  return res;
}

#define NUM_JIT_OPTIONS 3

CUjit_option jit_options[] = {
  CU_JIT_ERROR_LOG_BUFFER,
  CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
  CU_JIT_OPTIMIZATION_LEVEL,
};

// Must be freed with free_cubin_object.
typedef struct CubinObject {
  CUlinkState* state;
  uint8_t* data;
  size_t data_size;
} CubinObject;

// Compiles a PTX string into a cubin object.  The options passed to
// cuLinkCreate can be used to set the optimization level.
//
// The cubin object should be loaded through `load_cubin`, which goes through
// to cuModuleLoadData.
CubinObject compile_ptx_to_cubin(CudaContext* ctx,
                                 char* ptx_code,
                                 size_t ptx_size,
                                 size_t opt_lvl) {
  HARD_CHECK_CUDA(cuCtxSetCurrent(ctx->ctx));
  char error_buff[ERROR_BUFF_SIZE+1];
  void* option_values[] = {
    (void*)error_buff,
    (void*)ERROR_BUFF_SIZE,
    (void*)opt_lvl,
  };
  CubinObject object;
  object.state = malloc(sizeof(CUlinkState));
  HARD_CHECK_CUDA(cuLinkCreate(NUM_JIT_OPTIONS, jit_options, option_values, object.state));
  CUresult err = cuLinkAddData(
      *object.state, CU_JIT_INPUT_PTX, ptx_code, ptx_size, NULL, 0, NULL, NULL);
  if (option_values[1]) {
    fprintf(stderr, "%s%s\n", ptx_code, error_buff);
  }
  HARD_CHECK_CUDA(err);
  HARD_CHECK_CUDA(cuLinkComplete(*object.state, (void**)&object.data, &object.data_size));
  return object;
}

void free_cubin_object(CubinObject object) {
  HARD_CHECK_CUDA(cuLinkDestroy(*object.state));
  free(object.state);
}

/// Loads a cubin object.
CUmodule* load_cubin(CudaContext* ctx, const void* image) {
  HARD_CHECK_CUDA(cuCtxSetCurrent(ctx->ctx));
  CUmodule* module = malloc(sizeof(CUmodule));
  HARD_CHECK_CUDA(cuModuleLoadData(module, image));
  return module;
}

// Releases the memory used by a CUDA module.
void free_module(CUmodule* module) {
  HARD_CHECK_CUDA(cuModuleUnload(*module));
  free(module);
}

// Retrieve a function from a CUDA module.
CUfunction* get_function(CudaContext* ctx, CUmodule* module, const char* name) {
  HARD_CHECK_CUDA(cuCtxSetCurrent(ctx->ctx));
  CUfunction* function = malloc(sizeof(CUfunction));
  HARD_CHECK_CUDA(cuModuleGetFunction(function, *module, name));
  HARD_CHECK_CUDA(cuFuncSetCacheConfig(*function, CU_FUNC_CACHE_PREFER_SHARED));
  return function;
}

// Reads the values of performance counters in a CUpti_EventGroup.
static void read_events(CUpti_EventGroup group, int32_t num_event,
    uint32_t max_num_values, uint32_t* event_ids, uint64_t* event_values) {
  size_t size_indexes = num_event * sizeof(CUpti_EventID);
  size_t size_values = max_num_values * sizeof(uint64_t);
  size_t unused;
  CHECK_CUPTI(cuptiEventGroupReadAllEvents(group, CUPTI_EVENT_READ_FLAG_NONE,
        &size_values, event_values, &size_indexes, event_ids, &unused));
}

// Launches a kernel and stores its execution time in milliseconds in `out`. Returns 0
// upon success.
int32_t launch_kernel(CudaContext* context, CUfunction* function, uint32_t* blocks,
    uint32_t* threads, void** params, uint64_t* out) {
  HARD_CHECK_CUDA(cuCtxSetCurrent(context->ctx));
  CHECK_CUPTI(cuptiEventGroupEnable(context->num_cycle_event));
  CHECK_CUDA(cuLaunchKernel(*function, blocks[0], blocks[1], blocks[2], threads[0],
        threads[1], threads[2], 0, NULL, params, NULL));
  CHECK_CUDA(cuCtxSynchronize());
  CUpti_EventID unused;
  read_events(context->num_cycle_event, 1, 1, &unused, out);
  CHECK_CUPTI(cuptiEventGroupDisable(context->num_cycle_event));
  return 0;
}

// Time a kernel using events. Returns the time in nanoseconds.
double time_with_events(CudaContext* context, CUfunction* function, uint32_t* blocks,
    uint32_t* threads, void** params) {
  HARD_CHECK_CUDA(cuCtxSetCurrent(context->ctx));

  CUevent start, stop;
  HARD_CHECK_CUDA(cuEventCreate(&start, CU_EVENT_DEFAULT));
  HARD_CHECK_CUDA(cuEventCreate(&stop, CU_EVENT_DEFAULT));

  HARD_CHECK_CUDA(cuEventRecord(start, 0));
  HARD_CHECK_CUDA(cuLaunchKernel(*function, blocks[0], blocks[1], blocks[2], threads[0],
        threads[1], threads[2], 0, NULL, params, NULL));
  HARD_CHECK_CUDA(cuEventRecord(stop, 0));
  HARD_CHECK_CUDA(cuEventSynchronize(stop));

  float ms = -1;
  HARD_CHECK_CUDA(cuEventElapsedTime(&ms, start, stop));
  HARD_CHECK_CUDA(cuEventDestroy(start));
  HARD_CHECK_CUDA(cuEventDestroy(stop));

  return ((double)ms) * 1e6;
}

// Runs a kernel multiple times to gather a set of performance counter values.
void instrument_kernel(CudaContext* ctx, CUfunction* function, uint32_t* blocks,
    uint32_t* threads, void** params, EventSets* events, CUpti_EventID* event_ids,
    uint64_t* event_values) {
  HARD_CHECK_CUDA(cuCtxSetCurrent(ctx->ctx));
  uint32_t i;
  uint32_t num_group = 0;
  uint32_t total_num_event = 0;
  uint64_t* values = malloc(sizeof(uint64_t) * events->max_num_value_per_group);
  for (i=0; i<events->sets->numSets; ++i) {
    CHECK_CUPTI(cuptiEventGroupSetEnable(&events->sets->sets[i]));
    HARD_CHECK_CUDA(cuLaunchKernel(*function, blocks[0], blocks[1], blocks[2], threads[0],
          threads[1], threads[2], 0, NULL, params, NULL));
    HARD_CHECK_CUDA(cuCtxSynchronize());
    uint32_t j;
    for (j=0; j<events->sets->sets[i].numEventGroups; ++j) {
      CUpti_EventGroup group = events->sets->sets[i].eventGroups[j];
      EventGroupInfo *info = &events->groups_info[num_group];
      read_events(group, info->num_event, events->max_num_value_per_group,
          &event_ids[total_num_event], values);
      uint32_t k;
      for (k=0; k<info->num_event; ++k) {
        uint64_t sum = 0;
        uint32_t instance;
        for (instance = 0; instance<info->actual_num_instance; ++instance) {
          sum += values[instance * info->num_event + k];
        }
        sum = (sum * info->total_num_instance) / info->actual_num_instance;
        event_values[total_num_event] = sum;
        ++total_num_event;
      }
      ++num_group;
    }
    CHECK_CUPTI(cuptiEventGroupSetDisable(&events->sets->sets[i]));
  }
  free(values);
}

// Allocates an array.
CUdeviceptr* allocate_array(CudaContext* ctx, int64_t size) {
  HARD_CHECK_CUDA(cuCtxSetCurrent(ctx->ctx));
  CUdeviceptr* array = malloc(sizeof(CUdeviceptr));
  HARD_CHECK_CUDA(cuMemAlloc(array, size));
  return array;
}

// Deallocate an array. Must be done before the module is deallocated.
void free_array(CudaContext* ctx, CUdeviceptr* array) {
  HARD_CHECK_CUDA(cuCtxSetCurrent(ctx->ctx));
  HARD_CHECK_CUDA(cuMemFree(*array));
  free(array);
}

// Copies an array from the device to the host.
void copy_DtoH(CudaContext* ctx, CUdeviceptr* src, void* dst, uint64_t size) {
  HARD_CHECK_CUDA(cuCtxSetCurrent(ctx->ctx));
  HARD_CHECK_CUDA(cuMemcpyDtoH(dst, *src, size));
}

// Copies an array from the device to the host.
void copy_HtoD(CudaContext* ctx, void* src, CUdeviceptr* dst, uint64_t size) {
  HARD_CHECK_CUDA(cuCtxSetCurrent(ctx->ctx));
  HARD_CHECK_CUDA(cuMemcpyHtoD(*dst, src, size));
}

// Copies an array to an other on the device.
void copy_DtoD(CudaContext* ctx, CUdeviceptr* src, CUdeviceptr* dst, uint64_t size) {
  HARD_CHECK_CUDA(cuCtxSetCurrent(ctx->ctx));
  HARD_CHECK_CUDA(cuMemcpyDtoD(*dst, *src, size));
}

// Fills a float array with random data.
void randomize_float_array(CudaContext* ctx, CUdeviceptr* dst, uint64_t size,
    float mean, float stddev) {
  HARD_CHECK_CUDA(cuCtxSetCurrent(ctx->ctx));
  CHECK_CURAND(curandGenerateNormal(ctx->rng, (float*)*dst, size, mean, stddev));
}

// Creates a new CuptiEventGroupSets and stores the IDs of the events in 
EventSets* create_cuptiEventGroupSets(CudaContext* ctx, uint32_t num_event,
    char** event_names, uint32_t* eventIDs) {
  HARD_CHECK_CUDA(cuCtxSetCurrent(ctx->ctx));
  EventSets* sets = malloc(sizeof(EventSets));
  uint32_t i, j;
  // Convert event names to IDs.
  for (i=0; i< num_event; ++i) {
    CHECK_CUPTI(cuptiEventGetIdFromName(ctx->device, event_names[i], &eventIDs[i]));
  }
  // Create the event set.
  size_t eventIDs_bytes = num_event * sizeof(CUpti_EventID);
  CHECK_CUPTI(cuptiEventGroupSetsCreate(ctx->ctx, eventIDs_bytes, eventIDs, &sets->sets)); 
  // Count the number of group to allocate the EventGroupInfo array.
  uint32_t num_group = 0;
  for (i=0; i<sets->sets->numSets; ++i) {
    num_group += sets->sets->sets[i].numEventGroups;
  }
  sets->groups_info = malloc(num_group * sizeof(EventGroupInfo));
  // Fill the EventGroupInfo array.
  num_group = 0;
  uint32_t max_num_value_per_group = 0;
  for (i=0; i<sets->sets->numSets; ++i) {
    for (j=0; j<sets->sets->sets[i].numEventGroups; ++j) {
      CUpti_EventGroup group = sets->sets->sets[i].eventGroups[j];
      uint32_t one = 1;
      CHECK_CUPTI(cuptiEventGroupSetAttribute(group,
            CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES, sizeof(one), &one));
      CUpti_EventDomainID domain;
      size_t size = sizeof(domain);
      CHECK_CUPTI(cuptiEventGroupGetAttribute(group,
            CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID, &size, &domain));
      uint32_t* num_events = &sets->groups_info[num_group].num_event;
      size = sizeof(*num_events);
      CHECK_CUPTI(cuptiEventGroupGetAttribute(group,
            CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS, &size, num_events));
      uint32_t* total_instances = &sets->groups_info[num_group].total_num_instance;
      size = sizeof(*total_instances);
      CHECK_CUPTI(cuptiDeviceGetEventDomainAttribute(ctx->device, domain,
            CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT, &size, total_instances));
      uint32_t* counted_instances = &sets->groups_info[num_group].actual_num_instance;
      size = sizeof(*counted_instances);
      CHECK_CUPTI(cuptiDeviceGetEventDomainAttribute(ctx->device, domain,
            CUPTI_EVENT_DOMAIN_ATTR_INSTANCE_COUNT, &size, counted_instances));
      if (*counted_instances != *total_instances) {
        fprintf(stderr, "WARNING: not all SMX may be profiled. Performance counters "
            "will be scaled but might be inaccurate\n");
      }
      uint32_t num_value = *counted_instances * *num_events;
      if (num_value > max_num_value_per_group) { max_num_value_per_group = num_value; }
      ++num_group;
    }
  }
  sets->max_num_value_per_group = max_num_value_per_group;
  return sets;
}

/// Frees a CuptiEventGroupSets
void free_cuptiEventGroupSets(CudaContext* ctx, EventSets* sets) {
  HARD_CHECK_CUDA(cuCtxSetCurrent(ctx->ctx));
  CHECK_CUPTI(cuptiEventGroupSetsDestroy(sets->sets));
  free(sets->groups_info);
  free(sets);
}

/// Computes the number of simultaneously active blocks in an SMX.
uint32_t max_active_blocks_per_smx(CUfunction* func, uint32_t block_size,
    size_t dynamic_smem_size) {
  int out;
  HARD_CHECK_CUDA(cuOccupancyMaxActiveBlocksPerMultiprocessor(
        &out, *func, block_size, dynamic_smem_size));
  return out;
}
