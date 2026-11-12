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
 * scale selects the authors' bounded decimal precision (10^decimals).
 * This is not a bitwise-lossless binary64 codec.
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

/**
 * Extract a single f64 value at position idx from a BUFF-compressed block
 * without decoding the whole block.  Reads ~ceil(fixed_len/8) bytes (one
 * byte per byte-plane) plus a few trailing bits. Uses the authors'
 * selective decoder, including its small result/intermediate allocations.
 *
 * Returns 0 on success (writes the value to *output), -1 on error.
 */
int32_t buff_extract_f64(const uint8_t *input, size_t input_len,
                         size_t idx, double *output);
int32_t buff_extract_range_f64(const uint8_t *input, size_t input_len,
                               size_t first, size_t count, double *output);

#ifdef __cplusplus
}
#endif
