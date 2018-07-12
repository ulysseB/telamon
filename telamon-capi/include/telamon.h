#ifndef TELAMON_CAPI_H
#define TELAMON_CAPI_H

#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

typedef enum {
  X86,
  Cuda,
} Device;

typedef struct KernelParameters KernelParameters;

void kernel_free(KernelParameters *params);

KernelParameters *kernel_matmul_new(int m,
                                    int n,
                                    int k,
                                    unsigned int a_stride,
                                    int transpose_a,
                                    int transpose_b,
                                    int generic);

bool kernel_optimize(KernelParameters *params,
                     Device device,
                     const char *config_data,
                     size_t config_len);

#endif /* TELAMON_CAPI_H */
