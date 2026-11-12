#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
namespace gef_compressors { namespace flea {
// Values are already scaled integers; there is no lossy double conversion.
std::vector<uint8_t> compress_i64(const std::vector<int64_t>& values);
std::vector<int64_t> decompress_i64(const std::vector<uint8_t>& bytes, size_t count);
}}
