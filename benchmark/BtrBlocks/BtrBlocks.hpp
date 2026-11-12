#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
namespace gef_compressors { namespace btrblocks {
// Official BtrBlocks (maxi-k/btrblocks), default configuration and scheme selection,
// one DOUBLE column chunk per block (its INTEGER columns are 32-bit only).
// BtrBlocks has no single-value access, so the harness measures it on independent blocks.
std::vector<uint8_t> compress(const std::vector<double>& values);
std::vector<double> decompress(const std::vector<uint8_t>& bytes, size_t count);
}}
