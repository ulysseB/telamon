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

#include <stdio.h>
#include <telajax.h>

#define VEC_LENGTH 16

const char* kernel_ocl_wrapper = "\n" \
"void _vec_add(int n, __global float* x, __global float* y, unsigned long* timestamp_in, unsigned long* timestamp_out);\n" \
"\n" \
"__kernel void vec_add(int n, __global float* x, __global float* y){\n" \
"	unsigned long timestamp_in, timestamp_out;\n" \
"	_vec_add(n, x, y, &timestamp_in, &timestamp_out);\n" \
"	printf(\"[OCL] Exec time of _vec_add is %d cycles\\n\", timestamp_out - timestamp_in);\n" \
"}\n" \
"\n";


const char* kernel_code = "\n" \
"#include <stdio.h> \n " \
"\n" \
"void _vec_add(int n, float* x, float* y, unsigned long* timestamp_in, unsigned long* timestamp_out){ \n" \
"	\n" \
"	*timestamp_in = __k1_read_dsu_timestamp(); \n" \
"	for(int i = 0; i < n; i++){ \n" \
"		y[i] += x[i]; \n" \
"	} \n" \
"	*timestamp_out = __k1_read_dsu_timestamp(); \n" \
"	\n" \
"} \n" ;

void pfn_event_notify(
		cl_event event,
		cl_int event_command_exec_status,
		void* exec_time)
{
	if(exec_time){
		cl_ulong time_start = 0, time_end = 0;
		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
			sizeof(time_start), &time_start, NULL);
		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
			sizeof(time_end), &time_end, NULL);
		*((unsigned long long*)exec_time) = (unsigned long long) (time_end - time_start);
	}
}

int main()
{
	int vec_length = VEC_LENGTH;
	int err = 0;
	unsigned long long exec_time = 0;
	unsigned long long exec_time_read = 0;

	// Initialize device for Telajax
	device_t device = telajax_device_init(0, NULL, &err);
	assert(!err);

	// Create host buffers
	float* host_x = (float*) malloc(VEC_LENGTH * sizeof(float));
	float* host_y = (float*) malloc(VEC_LENGTH * sizeof(float));

	for(int i = 0; i < VEC_LENGTH; i++){
		host_x[i] = i; host_y[i] = i;
	}

	// Allocate device memory and copy
	mem_t dev_x = telajax_device_mem_alloc(VEC_LENGTH * sizeof(float),
		TELAJAX_MEM_READ_ONLY, &device, &err);
	assert(!err);
	mem_t dev_y = telajax_device_mem_alloc(VEC_LENGTH * sizeof(float),
		TELAJAX_MEM_READ_WRITE, &device, &err);
	assert(!err);

	telajax_device_mem_write(&device, dev_x, host_x, VEC_LENGTH * sizeof(float),
		0, NULL, NULL);
	telajax_device_mem_write(&device, dev_y, host_y, VEC_LENGTH * sizeof(float),
		0, NULL, NULL);

	// Build wrapper on device
	wrapper_t wrapper = telajax_wrapper_build("vec_add", kernel_ocl_wrapper,
		NULL, &device, &err);
	assert(!err);

	// Build kernel on device
	kernel_t vec_add_kernel = telajax_kernel_build(
		kernel_code,
		" -std=c99 ",       // cflags
		"",                 // lflags
		&wrapper,           // wrapper
		&device, &err);
	assert(!err);

	// Set kernel args
	size_t args_size[3] = {sizeof(int), sizeof(mem_t), sizeof(mem_t)};
	void*  args[3]      = {&vec_length,    &dev_x,        &dev_y};
	telajax_kernel_set_args(3, args_size, args, &vec_add_kernel);

	// Enqueue kernel
	event_t kernel_event;
	telajax_kernel_enqueue(&vec_add_kernel, &device, &kernel_event);
	telajax_event_set_callback(pfn_event_notify, (void*)&exec_time, kernel_event);

	// may read vector y from device to check if correct
	event_t read_event;
	telajax_device_mem_read(&device, dev_y, host_y, VEC_LENGTH * sizeof(float),
		1, &kernel_event, &read_event);
	telajax_event_set_callback(pfn_event_notify, (void*)&exec_time_read, read_event);

	// Wait
	telajax_event_wait(1, &read_event);
	telajax_device_waitall(&device);

	telajax_event_release(kernel_event);
	telajax_event_release(read_event);

	printf("Vector Y is : ");
	for(int i = 0; i < VEC_LENGTH; i++){
		printf("%.2f %s", host_y[i], (i == VEC_LENGTH-1) ? "\n" : "");
	}

	printf("Exec_time      = %llu ns\n", exec_time);
	printf("Exec_time_read = %llu ns\n", exec_time_read);

	// release kernel
	telajax_kernel_release(&vec_add_kernel);

	// release wrapper
	telajax_wrapper_release(&wrapper);

	// cleanup host buffers
	free(host_x);
	free(host_y);

	// cleanup device buffers
	telajax_device_mem_release(dev_x);
	telajax_device_mem_release(dev_y);

	// Finalize Telajax
	telajax_device_finalize(&device);

	return 0;
}
