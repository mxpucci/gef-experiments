#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Maximum compressed output size (bytes) for n f64 values. */
size_t buff_max_compressed_size(size_t n);

/**
 * Compress n f64 values.
 *
 * scale  – multiply each value by this before rounding to i64.
 *          Use 10^decimals for lossless fixed-precision data.
 *
 * Returns bytes written on success, or -1 on error.
 * output must be at least buff_max_compressed_size(n) bytes.
 */
int64_t buff_compress_f64(const double *input, size_t n,
                          uint8_t *output, size_t output_capacity,
                          int64_t scale);

/**
 * Decompress n f64 values from BUFF-compressed data.
 *
 * Returns 0 on success, -1 on error.
 * output must hold at least n doubles (n * 8 bytes).
 */
int32_t buff_decompress_f64(const uint8_t *input, size_t input_len,
                             double *output, size_t n);

#ifdef __cplusplus
}
#endif
