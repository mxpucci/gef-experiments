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
#include "gef/gef.hpp"

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
    auto t1 = std::chrono::high_resolution_clock::now();
    GEFType compressor(data);
    compiler_barrier();
    auto t2 = std::chrono::high_resolution_clock::now();
    auto compression_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    
    result.compressed_bits = compressor.size_in_bytes() * 8;
    result.compression_ratio = static_cast<double>(result.compressed_bits) / result.uncompressed_bits;
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
    
    // Range queries
    const size_t num_range_queries = bench_data.range_query_indices.begin()->second.size();
    for (auto range : range_sizes) {
        if (range >= data.size()) continue;
        
        const auto& range_indices = bench_data.range_query_indices.at(range);
        std::vector<T> out_buffer(range);
        volatile T range_checksum = 0;
        
        t1 = std::chrono::high_resolution_clock::now();
        for (auto start_idx : range_indices) {
            compressor.get_elements(start_idx, range, out_buffer);
            range_checksum += out_buffer[0];
            do_not_optimize(out_buffer);
        }
        compiler_barrier();
        t2 = std::chrono::high_resolution_clock::now();
        do_not_optimize(range_checksum);
        
        auto range_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
        double throughput = ((range * sizeof(T)) * num_range_queries / 1024.0 / 1024.0) / (range_time_ns / 1e9);
        result.range_query_throughputs.emplace_back(range, throughput);
    }
    
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
