#ifndef TELAMON_CAPI_H
#define TELAMON_CAPI_H

/* DO NOT MODIFY THIS MANUALLY!
 * This file has been generated usin cbindgen.
 *
 * To generate this file:
 *  1. Get the latest cbindgen using `cargo install --force cbindgen --git https://github.com/Elarnon/cbindgen`
 *  2. Run `rustup run nightly cbindgen telamon-capi -o telamon-capi/include/telamon.h`
 */

#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

/*
 * Supported device types for running kernels.
 */
typedef enum {
    X86,
    Cuda,
} Device;

/*
 * Supported kernels.
 */
typedef struct KernelParameters KernelParameters;

/*
 * Initializes the logger.
 */
void env_logger_try_init(void);

/*
 * Deallocates kernel parameters created through one of the `kernel_*_new`
 * functions. The `params` pointer becomes invalid and must not be used again
 * after calling `kernel_free`.
 */
void kernel_free(KernelParameters *params);

/*
 * Instanciate a new kernel for matrix-matrix multiplication. The
 * caller is responsible for deallocating the returned pointer using
 * kernel_free. The tile_m, tile_n and tile_k parameters are read
 * from during the call, but no pointer to the corresponding data is
 * kept afterwards.
 */
KernelParameters *kernel_matmul_new(int m,
                                    int n,
                                    int k,
                                    unsigned int a_stride,
                                    int transpose_a,
                                    int transpose_b,
                                    int generic,
                                    const uint32_t *tile_m,
                                    size_t tile_m_len,
                                    const uint32_t *tile_n,
                                    size_t tile_n_len,
                                    const uint32_t *tile_k,
                                    size_t tile_k_len);

/*
 * Optimize a kernel on a given device. `config_data` points to a JSON-encoded
 * string of length `config_len` containing the configuration parameters for
 * the explorer.
 */
bool kernel_optimize(KernelParameters *params,
                     Device device,
                     const char *config_data,
                     size_t config_len);

#endif /* TELAMON_CAPI_H */
