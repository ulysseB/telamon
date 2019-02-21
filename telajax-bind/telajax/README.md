# Telajax - An OpenCL-based library to run C code on device

## Basis

This is a fork from hominhquan/telajax
Original work can be found at https://github.com/hominhquan/telajax/tree/da8892fbbc5b7d6acdf08357afd3ca9e9051b9ca

The idea is to be able to compile standard C functions into an elf object file,
then link to the OpenCL runtime as an ordinary `cl_program` object.
This program will be later linked (`clLinkProgram()`) to a second `cl_program`,
obtained from building/compiling an OpenCL wrapper
(`clBuildProgram()`/`clCompileProgram()`).

Telajax API has a lot of similarities to OpenCL. We designed it to run quickly
standard C code on OpenCL-enabled devices, on which we are able to compile and
provide binary code which performs better than the OpenCL compiler does.

## Build & Test

``` sh
$ mkdir -p build
$ cd build
$ cmake -DCMAKE_INSTALL_PREFIX=<prefix> -DOPENCL_ROOT=<path/to/opencl/lib> ..
$ make
$ make check
```

## Install

``` sh
$ make install
```

## Example

Here is an example of fast prototyping of `vec_add` using Telajax :

``` c
#include <stdio.h>
#include <telajax.h>

#define VEC_LENGTH 16

const char* kernel_ocl_wrapper = "\n" \
"void _vec_add(int n, __global float* x, __global float* y);\n" \
"\n" \
"__kernel void vec_add(int n, __global float* x, __global float* y){\n" \
"	_vec_add(n, x, y);\n" \
"}\n" \
"\n";


const char* kernel_code = "\n" \
"#include <stdio.h> \n " \
"\n" \
"void _vec_add(int n, float* x, float* y){ \n" \
"	\n" \
"	for(int i = 0; i < n; i++){ \n" \
"		y[i] += x[i]; \n" \
"	} \n" \
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

	telajax_device_mem_write(&device, dev_x, host_x, VEC_LENGTH * sizeof(float));
	telajax_device_mem_write(&device, dev_y, host_y, VEC_LENGTH * sizeof(float));

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
	telajax_kernel_enqueue(&vec_add_kernel, &device);
	telajax_kernel_set_callback(pfn_event_notify, (void*)&exec_time, &vec_add_kernel);

	// Wait for kernel termination (and callback is executed in backgroud)
	telajax_device_waitall(&device);

	// may read vector y from device to check if correct
	telajax_device_mem_read(&device, dev_y, host_y, VEC_LENGTH * sizeof(float));

	printf("Vector Y is : ");
	for(int i = 0; i < VEC_LENGTH; i++){
		printf("%.2f %s", host_y[i], (i == VEC_LENGTH-1) ? "\n" : "");
	}

	printf("Exec_time = %llu ns\n", exec_time);

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
```
