// GEF benchmark translation unit.
//
// GEF (the codec under study) is built with the flags of the `gef` CMake
// target, including -ffast-math. It lives in its own translation unit so that
// competitor codecs in lossless_benchmark.cpp can be compiled with strict IEEE
// arithmetic without changing GEF's own build.
//
// NOTE: When running multiple GEF compressors sequentially with SIMD and OpenMP enabled,
// you may encounter a bug where subsequent compressors produce astronomically large
// compressed sizes (compression ratio > 100x). This manifests on Linux with AVX-512.
//
// WORKAROUND: Use LosslessBenchmarkFullNoSIMD which is compiled with:
//   -DGEF_DISABLE_SIMD=1 -DGEF_DISABLE_OPENMP=1
// These flags force sequential partition construction and scalar gap computation.

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

// gef.hpp relies on headers that the benchmark used to include before it.
#include <sdsl/bit_vectors.hpp>
#include "benchmark_common.hpp"
#include "range_queries.hpp"
#include "gef/gef.hpp"
#include "gef/gef_fp.hpp"

namespace {

// ============================================================================
// GEF Wrapper and Benchmark
// ============================================================================

template<typename GEFType, typename T = int64_t>
BenchmarkResult benchmark_gef(const std::string &compressor_name,
                              const BenchmarkData &bench_data,
                              const std::vector<size_t> &range_sizes) {
    BenchmarkResult result;
    result.compressor = compressor_name;
    result.dataset = bench_data.filename;
    
    // Use shifted data
    const auto& data = bench_data.shifted_data;
                   
    result.num_values = data.size();
    result.uncompressed_bits = bench_data.uncompressed_bits;
    
    // Compression
    const size_t gef_ram0 = gef_live_heap_bytes();
    auto t1 = std::chrono::high_resolution_clock::now();
    GEFType compressor(data);
    compiler_barrier();
    auto t2 = std::chrono::high_resolution_clock::now();
    auto compression_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    
    result.compressed_bits = compressor.size_in_bytes() * 8;
    result.compression_ratio = static_cast<double>(result.compressed_bits) / result.uncompressed_bits;
    result.ram_bytes = gef_heap_delta(gef_ram0);
    result.compression_throughput_mbs = (data.size() * sizeof(T) / 1024.0 / 1024.0) / (compression_time_ns / 1e9);
    
    // Full decompression
    std::vector<T> decompressed(data.size());
    volatile T decomp_checksum = 0;
    t1 = std::chrono::high_resolution_clock::now();
    compressor.get_elements(0, data.size(), decompressed);
    decomp_checksum += decompressed[0];
    compiler_barrier();
    t2 = std::chrono::high_resolution_clock::now();
    do_not_optimize(decompressed);
    do_not_optimize(decomp_checksum);
    
    auto decompression_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.decompression_throughput_mbs = (data.size() * sizeof(T) / 1024.0 / 1024.0) / (decompression_time_ns / 1e9);
    
    check_roundtrip(result, data, decompressed);
    finish_roundtrip_validation(result);
    
    // Random access
    const size_t num_ra_queries = bench_data.random_indices.size();
    
    t1 = std::chrono::high_resolution_clock::now();
    volatile T ra_sum = 0;
    for (auto idx : bench_data.random_indices) {
        ra_sum += compressor[idx];
    }
    compiler_barrier();
    t2 = std::chrono::high_resolution_clock::now();
    do_not_optimize(ra_sum);
    auto ra_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.random_access_ns = static_cast<double>(ra_time_ns) / num_ra_queries;
    result.random_access_mbs = (num_ra_queries * sizeof(T) / 1024.0 / 1024.0) / (ra_time_ns / 1e9);
    
    // Range queries (range_queries.hpp)
    run_range_queries(result, bench_data, range_sizes, data.size(), decompressed.data(),
        [&](size_t start, size_t length, std::vector<T> &out) {
            compressor.get_elements(start, length, out);
        });
    
    return result;
}

} // namespace

bool is_gef_compressor(const std::string &name) {
    return name == "rle_gef" || name == "u_gef_approximate" || name == "u_gef_optimal" ||
           name == "b_gef_approximate" || name == "b_gef_optimal" ||
           name == "b_star_gef_approximate" || name == "b_star_gef_optimal";
}

BenchmarkResult benchmark_gef_compressor(const std::string &name, const BenchmarkData &bench_data,
                                         const std::vector<size_t> &range_sizes) {
    if (name == "rle_gef")
        return benchmark_gef<gef::RLE_GEF<int64_t, GEF_UNIFORM_PARTITION_SIZE>, int64_t>(name, bench_data, range_sizes);
    if (name == "u_gef_approximate")
        return benchmark_gef<gef::U_GEF_APPROXIMATE<int64_t, GEF_UNIFORM_PARTITION_SIZE>, int64_t>(name, bench_data, range_sizes);
    if (name == "u_gef_optimal")
        return benchmark_gef<gef::U_GEF<int64_t, GEF_UNIFORM_PARTITION_SIZE>, int64_t>(name, bench_data, range_sizes);
    if (name == "b_gef_approximate")
        return benchmark_gef<gef::B_GEF_APPROXIMATE<int64_t, GEF_UNIFORM_PARTITION_SIZE>, int64_t>(name, bench_data, range_sizes);
    if (name == "b_gef_optimal")
        return benchmark_gef<gef::B_GEF<int64_t, GEF_UNIFORM_PARTITION_SIZE>, int64_t>(name, bench_data, range_sizes);
    if (name == "b_star_gef_approximate")
        return benchmark_gef<gef::B_STAR_GEF_APPROXIMATE<int64_t, GEF_UNIFORM_PARTITION_SIZE>, int64_t>(name, bench_data, range_sizes);
    if (name == "b_star_gef_optimal")
        return benchmark_gef<gef::B_STAR_GEF<int64_t, GEF_UNIFORM_PARTITION_SIZE>, int64_t>(name, bench_data, range_sizes);
    throw std::invalid_argument("unknown GEF compressor: " + name);
}

namespace {
// Floating-point forks of GEF (gef/gef_fp.hpp): the same encoded structure over scaled
// integers, with double input and double output. Compression converts each partition
// into a reused buffer as it is built; decoding converts in registers before each
// value's only store; a point access converts the one value. Every decoded double is
// checked bit for bit against the input (bitwise_mismatches).
template<typename FPType>
BenchmarkResult benchmark_gef_f64(const std::string &name, const BenchmarkData &bench_data,
                                  const std::vector<size_t> &range_sizes) {
    using Clock = std::chrono::high_resolution_clock;
    BenchmarkResult result;
    result.compressor = name;
    result.dataset = bench_data.filename;
    const auto &data = bench_data.double_data;
    const size_t n = data.size();
    result.num_values = n;
    result.uncompressed_bits = bench_data.uncompressed_bits;
    result.access_layout = "native";
    const double mib = n * sizeof(double) / 1048576.0;
    auto seconds = [](Clock::time_point a, Clock::time_point b) { return std::max(std::chrono::duration<double>(b - a).count(), 1e-9); };

    const size_t gef_ram0 = gef_live_heap_bytes();
    auto t1 = Clock::now();
    FPType compressor(data.data(), n, bench_data.decimals);
    compiler_barrier();
    auto t2 = Clock::now();
    result.compressed_bits = compressor.size_in_bytes() * 8;
    result.compression_ratio = double(result.compressed_bits) / result.uncompressed_bits;
    result.ram_bytes = gef_heap_delta(gef_ram0);
    result.compression_throughput_mbs = mib / seconds(t1, t2);

    std::vector<double> decompressed(n);
    t1 = Clock::now();
    compressor.get_doubles(0, n, decompressed.data());
    compiler_barrier();
    t2 = Clock::now();
    do_not_optimize(decompressed);
    result.decompression_throughput_mbs = mib / seconds(t1, t2);
    check_roundtrip(result, data, decompressed);
    finish_roundtrip_validation(result);

    const size_t q = bench_data.random_indices.size();
    volatile double ra_sum = 0;
    t1 = Clock::now();
    for (auto idx : bench_data.random_indices) ra_sum += compressor.at_double(idx);
    compiler_barrier();
    t2 = Clock::now();
    do_not_optimize(ra_sum);
    result.random_access_ns = q ? seconds(t1, t2) * 1e9 / q : 0;
    result.random_access_mbs = q * sizeof(double) / 1048576.0 / seconds(t1, t2);

    // Range queries return doubles as well (range_queries.hpp).
    run_range_queries(result, bench_data, range_sizes, n, decompressed.data(),
        [&](size_t start, size_t length, std::vector<double> &out) { compressor.get_doubles(start, length, out.data()); });
    return result;
}
} // namespace

bool is_gef_f64_compressor(const std::string &name) {
    return name == "rle_gef_f64" || name == "b_star_gef_approximate_f64";
}
BenchmarkResult benchmark_gef_f64_compressor(const std::string &name, const BenchmarkData &bench_data,
                                             const std::vector<size_t> &range_sizes) {
    if (name == "rle_gef_f64")
        return benchmark_gef_f64<gef::RLE_GEF_FP<int64_t, GEF_UNIFORM_PARTITION_SIZE>>(name, bench_data, range_sizes);
    if (name == "b_star_gef_approximate_f64")
        return benchmark_gef_f64<gef::B_STAR_GEF_APPROXIMATE_FP<int64_t, GEF_UNIFORM_PARTITION_SIZE>>(name, bench_data, range_sizes);
    throw std::invalid_argument("unknown GEF f64 compressor: " + name);
}
