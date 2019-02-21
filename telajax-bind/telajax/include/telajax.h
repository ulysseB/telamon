/* Copyright (c) 2017 Kalray

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#ifndef __TELAJAX_H
#define __TELAJAX_H

#ifdef __cplusplus
extern "C" {
#endif

#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <sys/stat.h>
#include <errno.h>

#include <CL/cl.h>     // for binding Telajax-Opencl

const char* get_ocl_error(int err);

// =============================================================================
// Type definitions
// =============================================================================

// A self-containing device_t
// Asumption : one platform, one device, one context and one command queue
typedef struct device_s
{
	cl_platform_id   _platform;
	cl_device_id     _device_id;
	cl_context       _context;
	cl_command_queue _queue;
} device_t;

#define TELAJAX_MEM_READ_WRITE      CL_MEM_READ_WRITE
#define TELAJAX_MEM_WRITE_ONLY      CL_MEM_WRITE_ONLY
#define TELAJAX_MEM_READ_ONLY       CL_MEM_READ_ONLY

typedef cl_mem_flags mem_flags_t;

typedef cl_mem mem_t;

typedef struct kernel_s
{
	cl_program _program;
	cl_kernel  _kernel;
	int        _work_dim;
	size_t     _globalSize[3];
	size_t     _localSize[3];
} kernel_t;

typedef struct wrapper_s
{
	char*      _name;
	cl_program _program;
} wrapper_t;

typedef cl_event event_t;

// =============================================================================
// Device Init and finalize and sync
// =============================================================================

/**
 * Initialization of device
 * @param[in]  argc     For general purpose
 * @param[in]  argv     For general purpose
 * @param[out] error    Error code : 0 on success, -1 otherwise
 * @return device_t
 */
device_t telajax_device_init(int argc, char** argv, int* error);

/**
 * Return 1 if telajax is already initialized, 0 otherwise
 */
int telajax_is_initialized();

/**
 * Wait for termination of all on-flight kernels and finalize telajax
 * @param[in]  device  Device
 * @return 0 on success, -1 otherwise
 */
int telajax_device_finalize(device_t* device);

/**
 * Return 1 if telajax is already finalized, 0 otherwise
 */
int telajax_is_finalized();

/**
 * Wait for termination of all on-flight kernels on the device
 * @param[in]  device  Device
 * @return 0 on success, -1 otherwise
 */
int telajax_device_waitall(device_t* device);

// =============================================================================
// Device memory operations
// =============================================================================

/**
 * Allocate a buffer on device
 * @param[in]   size         Malloc size in bytes
 * @param[in]   mem_flags    Memory flags :
 *                           TELAJAX_MEM_READ_WRITE
 *                           TELAJAX_MEM_WRITE_ONLY
 *                           TELAJAX_MEM_READ_ONLY
 * @param[in]   device       Device
 * @param[out]  error        Error code : 0 on success, -1 otherwise
 * @return mem_t
 */
mem_t telajax_device_mem_alloc(size_t size, mem_flags_t mem_flags, device_t* device, int* error);

/**
 * Initialize the device memory from a host memory buffer
 * @param[in]  device        Device
 * @param[in]  device_mem    Device memory
 * @param[in]  host_mem      Host memory pointer
 * @param[in]  size          Size in bytes to transfer
 * @param[in]  num_events_in_wait_list  Number of event in wait list.
 *                           Executed immediately if zero.
 * @param[in]  event_wait_list  List of event to wait before doing the
 *                              memory operation. If not NULL, length of
 *                              event_wait_list must be >= num_events_in_wait_list
 * @param[out]  event  Event of the operation. If NULL, call is blocking.
 *                     If not NULL, the call is non-blocking and returns immediately.
 *                     One must later call telajax_event_wait() to wait for termination.
 * @return 0 on success, -1 otherwise
 */
int telajax_device_mem_write(device_t* device, mem_t device_mem, void* host_mem, size_t size,
	const unsigned int num_events_in_wait_list, const event_t* event_wait_list,
	event_t *event);

/**
 * Read the device memory into a host memory buffer
 * @param[in]  device        Device
 * @param[in]  device_mem    Device memory
 * @param[in]  host_mem      Host memory pointer
 * @param[in]  size          Size in bytes to transfer
 * @param[in]  num_events_in_wait_list  Number of event in wait list.
 *                           Executed immediately if zero.
 * @param[in]  event_wait_list  List of event to wait before doing the
 *                              memory operation. If not NULL, length of
 *                              event_wait_list must be >= num_events_in_wait_list
 * @param[out]  event  Event of the operation. If NULL, call is blocking.
 *                     If not NULL, the call is non-blocking and returns immediately.
 *                     One must later call telajax_event_wait() to wait for termination.
 * @return 0 on success, -1 otherwise
 */
int telajax_device_mem_read(device_t* device, mem_t device_mem, void* host_mem, size_t size,
	const unsigned int num_events_in_wait_list, const event_t* event_wait_list,
	event_t *event);

/**
 * Release a buffer on device
 * @param[in]  device_mem   Device memory returned by mem_alloc
 * @return 0 on success, -1 otherwise
 */
int telajax_device_mem_release(mem_t device_mem);

// =============================================================================
// Kernel operations
// =============================================================================

/**
 * Build up a wrapper from an OpenCL wrapper code
 * @param[in]  kernel_ocl_name     OpenCL wrapper name
 * @param[in]  kernel_ocl_wrapper  OpenCL wrapper code
 * @param[in]  options      Compilation options of wrapper (ignored if NULL)
 * @param[in]  device       Device on which kernel will be built for.
 *                          This identifies the hardware and uses the correct
 *                          compilation steps (and optimization).
 * @param[out] error        Error code : 0 on success, -1 otherwise
 * @return    wrapper_t
 */
wrapper_t telajax_wrapper_build(
	const char* kernel_ocl_name,
	const char* kernel_ocl_wrapper,
	const char* options,
	device_t* device, int* error);

/**
 * Release a wrapper
 * @param[out] wrapper       Wrapper to release
 * @return 0 on success, -1 otherwise
 */
int telajax_wrapper_release(wrapper_t* wrapper);

/**
 * Build up a kernel from a code source
 * @param[in]  kernel_code  String containing C code (ignored if NULL)
 * @param[in]  cflags       Compilation flags of kernel (ignored if NULL)
 * @param[in]  lflags       Link flags of kernel (ignored if NULL)
 * @param[in]  wrapper      OpenCL wrapper
 * @param[in]  device       Device on which kernel will be built for.
 *                          This identifies the hardware and uses the correct
 *                          compilation steps (and optimization).
 * @param[out] error        Error code : 0 on success, -1 otherwise
 * @return    kernel_t
 */
kernel_t telajax_kernel_build(
	const char* kernel_code,
	const char* cflags, const char* lflags,
	const wrapper_t* wrapper,
	device_t* device, int* error);

/**
 * Set work_dim and globalSize, localSize of a kernel
 * @param[in]  work_dim     Number of dimension (=< 3)
 * @param[in]  globalSize   Number of global work-item in each dimension
 * @param[in]  localSize    Number of local  work-item in each dimension
 * @param[in]  kernel       Kernel
 * @return 0 on success, -1 otherwise
 */
int telajax_kernel_set_dim(
	int work_dim, const size_t* globalSize, const size_t* localSize, kernel_t* kernel);

/**
 * Release a kernel
 * @details    User should manage to call this function inside the callback or
 *             explicitly in application to avoid possible memory leak.
 * @param[out] kernel       Kernel to release
 * @return 0 on success, -1 otherwise
 */
int telajax_kernel_release(kernel_t* kernel);

/**
 * Set arguments for a kernel
 * @param[in]  num_args    Number of arguments of the kernel in args[]
 * @param[in]  args_size   Array of size of each argument in args[]
 * @param[in]  args        Array of arguments
 * @param[in]  kernel      Associated kernel
 * @return 0 on success, -1 otherwise
 */
int telajax_kernel_set_args(int num_args, size_t* args_size, void** args, kernel_t* kernel);

/**
 * Enqueue the kernel to the device
 * @param[in]  kernel   Kernel to enqueue on device
 * @param[in]  device   Device
 * @param[out] event    If not NULL, return an event to wait on later
 * @return 0 on success, -1 otherwise
 */
int telajax_kernel_enqueue(kernel_t* kernel, device_t* device, event_t* event);

// =============================================================================
// Event operations
// =============================================================================

/**
 * Attach a callback to a event
 * @param[in]  pfn_event_notify  Pointer to the callback function
 * @param[in]  user_data         Pointer to the argument of callback
 * @param[in]  event             Associated event
 * @return 0 on success, -1 otherwise
 */
int telajax_event_set_callback(
	void (CL_CALLBACK *pfn_event_notify)(
		cl_event event,
		cl_int event_command_exec_status,
		void* user_data
	),
	void* user_data, event_t event);

/**
 * Wait for event
 * @param[in]  nb_events  Number of event in list
 * @param[in]  event_list Event list
 * @return 0 on success, -1 otherwise
 */
int telajax_event_wait(int nb_events, const event_t *event_list);

/**
 * Release an event
 * @param[in]  event Event to release
 * @return 0 on success, -1 otherwise
 */
int telajax_event_release(event_t event);

#ifdef __cplusplus
}
#endif

#endif // __TELAJAX_H
