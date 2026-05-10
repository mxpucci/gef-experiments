#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "buff_ffi.h"

namespace buff {

struct CompressedBlock {
    std::vector<uint8_t> data;
    size_t               count{};
};

inline CompressedBlock compress_block(const double *values, size_t n, int64_t scale) {
    CompressedBlock blk;
    blk.count = n;
    blk.data.resize(buff_max_compressed_size(n));

    int64_t written = buff_compress_f64(values, n, blk.data.data(),
                                        blk.data.size(), scale);
    if (written < 0)
        throw std::runtime_error("BUFF: compression failed");

    blk.data.resize(static_cast<size_t>(written));
    return blk;
}

inline void decompress_block(const CompressedBlock &blk, double *output) {
    int32_t ret = buff_decompress_f64(blk.data.data(), blk.data.size(),
                                      output, blk.count);
    if (ret != 0)
        throw std::runtime_error("BUFF: decompression failed");
}

// O(1) single-value extraction: reads ~ceil(fixed_len/8) bytes from the
// block instead of decoding it whole.  Use for random access; for sequential
// scans / range queries decompress_block is faster.
inline double get_value_at(const CompressedBlock &blk, size_t idx) {
    double out;
    int32_t ret = buff_extract_f64(blk.data.data(), blk.data.size(), idx, &out);
    if (ret != 0)
        throw std::runtime_error("BUFF: extract failed");
    return out;
}

} // namespace buff
