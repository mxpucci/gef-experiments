#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
namespace gef_compressors { namespace fastpfor {
// Official FastPFor library (lemire/FastPFor), 64-bit CompositeCodec<FastPFor<4>, VariableByte>.
// The library has no single-value access, so the harness measures it on independent blocks.
// FOR: values - block minimum. Delta: zigzag(v_i - v_{i-1}) after the block's first value.
std::vector<uint8_t> compress_for_i64(const std::vector<int64_t>& values);
std::vector<int64_t> decompress_for_i64(const std::vector<uint8_t>& bytes, size_t count);
std::vector<uint8_t> compress_delta_i64(const std::vector<int64_t>& values);
std::vector<int64_t> decompress_delta_i64(const std::vector<uint8_t>& bytes, size_t count);
}}
