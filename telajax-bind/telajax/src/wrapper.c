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
wrapper_t
telajax_wrapper_build(
	const char* kernel_ocl_name,
	const char* kernel_ocl_wrapper,
	const char* options,
	device_t* device, int* error)
{
	int err = 0;

	wrapper_t wrapper;
	asprintf(&(wrapper._name), "%s", kernel_ocl_name);

	// build wrapper program
	wrapper._program = clCreateProgramWithSource(device->_context, 1,
				(const char **) &kernel_ocl_wrapper, NULL, &err);
	assert(wrapper._program);
  if (err) {
    printf("Error in create program: %s\n", get_ocl_error(err));
    goto ERROR;
  }

	// FIXME : The most rationale function here is clCompileProgram(), but
	// currently not implemented in pocl
	err = clBuildProgram(wrapper._program, 0, NULL, options, NULL, NULL);
  if (err) {
    printf("Error in build program: %s\n", get_ocl_error(err));
    goto ERROR;
  }
	//assert(!err);

	// FIXME : This is to force pocl runtime to compile wrapper.
	// That's why we may need clCompileProgram() to do so, instead of
	// clBuildProgram() + clGetProgramInfo().
	size_t dummy;
	err = clGetProgramInfo(wrapper._program, CL_PROGRAM_BINARY_SIZES,
		sizeof(dummy), &dummy, NULL);
  if (err) {
    printf("Error in get program info: %s\n", get_ocl_error(err));
    goto ERROR;
  }
	//assert(!err);

ERROR:
	if(error != NULL) {
    *error = err;
  }
	return wrapper;
}

/**
 * Release a wrapper
 * @param[out] wrapper       Wrapper to release
 * @return 0 on success, -1 otherwise
 */
int
telajax_wrapper_release(wrapper_t* wrapper)
{
	free(wrapper->_name);
	clReleaseProgram(wrapper->_program);
	return 0;
}
