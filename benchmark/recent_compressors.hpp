#pragma once
#include "benchmark_common.hpp"
#include <string>
#include <vector>

// Independent translation unit: compiled without fast-math, including when GEF
// enables fast-math on the main benchmark target.
std::vector<std::string> recent_compressor_names();
bool recent_compressor_uses_integers(const std::string& name);
bool is_recent_compressor(const std::string& name);
struct NativeAccessOptions {
    size_t fauna_batch_size=65536;
};
BenchmarkResult benchmark_recent_compressor(const std::string& name,
    const BenchmarkData& data, const std::vector<size_t>& ranges, size_t block_size,
    const NativeAccessOptions& native=NativeAccessOptions{});
