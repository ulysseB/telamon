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

extern char* telajax_cachedir;
extern char* telajax_compiler;

static int telajax_finalized = 0;

/**
 * Return 1 if telajax is already finalized, 0 otherwise
 */
int
telajax_is_finalized()
{
	return telajax_finalized;
}

/**
 * Wait for termination of all on-flight kernels and finalize telajax
 * @return 0 on success, -1 otherwise
 */
int
telajax_device_finalize(device_t* device)
{
	int finalized = __sync_val_compare_and_swap(&telajax_finalized, 0, 1);

	if(finalized == 0){
		clFinish(device->_queue);

		clReleaseCommandQueue(device->_queue);
		clReleaseContext(device->_context);
		clReleaseDevice(device->_device_id);

		if(telajax_cachedir != NULL) free(telajax_cachedir);
		if(telajax_compiler != NULL) free(telajax_compiler);
	}

	return 0;
}

