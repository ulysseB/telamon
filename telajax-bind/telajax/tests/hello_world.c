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
"void _hello_world();\n" \
"\n" \
"__kernel void hello_world(){\n" \
"	printf(\"Hello world from %d\\n\", get_global_id(0)); \n" \
"	_hello_world(); \n" \
"}\n" \
"\n";


const char* kernel_code = "\n" \
"#include <stdio.h> \n " \
"\n" \
"void _hello_world(){ \n" \
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
	int err = 0;
	unsigned long long exec_time = 0;

	// Initialize device for Telajax
	device_t device = telajax_device_init(0, NULL, &err);
	assert(!err);

	// Build wrapper on device
	wrapper_t wrapper = telajax_wrapper_build("hello_world", kernel_ocl_wrapper,
		NULL, &device, &err);
	assert(!err);

	// Build kernel on device
	kernel_t helloworld_kernel = telajax_kernel_build(
		kernel_code,
		" -std=c99 ",       // cflags
		"",                 // lflags
		&wrapper,           // wrapper
		&device, &err);
	assert(!err);

	// Enqueue kernel
	event_t kernel_event;
	telajax_kernel_enqueue(&helloworld_kernel, &device, &kernel_event);
	telajax_event_set_callback(pfn_event_notify, (void*)&exec_time, kernel_event);

	// Wait for kernel termination (and callback is executed in backgroud)
	telajax_device_waitall(&device);

	telajax_event_release(kernel_event);

	printf("Exec_time = %llu ns\n", exec_time);

	// release kernel
	telajax_kernel_release(&helloworld_kernel);

	// release wrapper
	telajax_wrapper_release(&wrapper);

	// Finalize Telajax
	telajax_device_finalize(&device);

	return 0;
}
