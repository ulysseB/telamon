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

#include "telajax.h"

/*

OpenCL kernel wrapper :

void _vec_add(int n, __global float* x, __global float* y);

__kernel vec_add(int n, __global float* x, __global float* y)
{
	_vec_add(n, x, y);
}


GCC kernel code :

#include <stdio.h>

void _vec_add(int n, float* x, float* y)
{
	for(int i = 0; i < n; i++){
		y[i] += x[i];
	}
}

*/

char* telajax_cachedir = NULL;
char* telajax_compiler = NULL;

// This code is appended in the head of any C kernel_code
const char* telajax_extra_code_prefix = "\n" \
"\n" \
"#include <stdlib.h> \n" \
"#include <stdio.h> \n" \
"#include <vbsp.h> \n" \
"#include <utask.h> \n" \
"#include <sys/time.h> \n" \
"#include <math.h> \n" \
"#include <mppa_power.h> \n" \
"#include <mppa_rpc.h> \n" \
"#include <mppa_async.h> \n" \
"#include <mppa_remote.h> \n" \
;

const char* default_cflags = "" \
" -mhypervisor" \
" -march=k1b" \
" -O3" \
" -fpic" \
" -ffunction-sections -fdata-sections -fno-common" \
" -Wa,--check-resources" \
;

#define RANDOM_STRING_LENGTH 8
#define FILE_PATH_LENGTH     128
#define COMMAND_LENGTH       1024

kernel_t
telajax_kernel_build(
	const char* kernel_code,
	const char* cflags, const char* lflags,
	const wrapper_t* wrapper,
	device_t* device, int* error)
{
	int err = 0;
	char cmd[COMMAND_LENGTH];
	char rand_string[RANDOM_STRING_LENGTH+1];
	char rand_file_path_obj[FILE_PATH_LENGTH];
	kernel_t kernel_res;
	FILE *fp, *fpopen;

	if(kernel_code == NULL || wrapper == NULL){
		err = -1; goto ERROR;
	}

	// get a random string
	snprintf(cmd, COMMAND_LENGTH,
		"head -c 64 /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w %d | head -n 1",
		RANDOM_STRING_LENGTH);
	fpopen = popen(cmd, "r");
	if (fpopen == NULL) {
		printf("Failed to generate a random string\n");
		err = -1; goto ERROR;
	}
	fgets(rand_string, RANDOM_STRING_LENGTH+1, fpopen);
	rand_string[RANDOM_STRING_LENGTH] = '\0';
	pclose(fpopen);

	snprintf(rand_file_path_obj, FILE_PATH_LENGTH, "%s/%s.o", telajax_cachedir, rand_string);

	// Compile the C kernel_code to a .o
	snprintf(cmd, COMMAND_LENGTH,
		" %s "               // telajax_compiler
		" %s "               // default_cflags
		" %s "               // cflags
		" -c "
		" -o %s "            // rand_file_path_obj
		" %s "               // lflags
		" -xc -"          // wait for input pipe here
		, telajax_compiler
		, default_cflags
		, cflags
		, rand_file_path_obj
		, lflags
		);
	fpopen = popen(cmd, "w");
	if (fpopen == NULL) {
		printf("Failed to open pipe to gcc\n");
		err = -1; goto ERROR;
	}
	fwrite(telajax_extra_code_prefix, strlen(telajax_extra_code_prefix), 1, fpopen);
	fwrite(kernel_code, strlen(kernel_code), 1, fpopen);
	pclose(fpopen);

	//  declare two cl_program
	cl_program input_programs[2];

	input_programs[0] = wrapper->_program;

	// create elf program and link
	// read the .o
	fp = fopen(rand_file_path_obj, "r");
	fseek(fp, 0, SEEK_END);
	size_t size_ftell = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	char *kernel_elf_source = (char*)malloc(size_ftell);
	fread(kernel_elf_source, size_ftell, 1, fp);
	fclose(fp);

	input_programs[1] = clCreateProgramWithBinary(device->_context, 1,
		&(device->_device_id), &size_ftell,
		(const unsigned char **) &kernel_elf_source, NULL, &err);
	assert(input_programs[1]);
	free(kernel_elf_source);

	cl_program program_final = clLinkProgram(device->_context, 1, &(device->_device_id),
		NULL, 2, input_programs, NULL, NULL, &err);
	assert(!err);
	assert(program_final);

	// release input_programs[1]
	clReleaseProgram(input_programs[1]);

	// build the OpenCL kernel associated with program_final
	cl_kernel kernel_final = clCreateKernel(program_final, wrapper->_name, &err);
	assert(kernel_final);
	assert(!err);

	// delete rand_file_path_src and rand_file_path_obj
	remove(rand_file_path_obj);

	// setup kernel_res and return
	kernel_res._program       = program_final;
	kernel_res._kernel        = kernel_final;
	kernel_res._work_dim      = 3;
	kernel_res._globalSize[0] = 1;
	kernel_res._globalSize[1] = 1;
	kernel_res._globalSize[2] = 1;
	kernel_res._localSize[0]  = 1;
	kernel_res._localSize[1]  = 1;
	kernel_res._localSize[2]  = 1;

ERROR:
	if(error) *error = err;
	return kernel_res;
}
