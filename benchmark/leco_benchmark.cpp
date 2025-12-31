// LeCo benchmark - compiled with GCC 11 due to GCC 13 compatibility issues.
// (LeCo redefines std::is_integral<__uint128_t> which conflicts with GCC 13's libstdc++)

#include "benchmark_common.hpp"
#include "piecewise_fix_integer_template.h"

#include <chrono>
#include <cstring>
#include <iostream>

BenchmarkResult benchmark_leco(const BenchmarkData &bench_data,
                               const std::vector<size_t> &range_sizes,
                               size_t block_size) {
    using T = int64_t;
    
    BenchmarkResult result;
    result.compressor = "LeCo";
    result.dataset = bench_data.filename;
    
    const auto& data = bench_data.shifted_data;
    
    result.num_values = data.size();
    result.uncompressed_bits = bench_data.uncompressed_bits;
    
    const size_t n = data.size();
    const size_t num_blocks = (n / block_size) + (n % block_size != 0);
    
    Codecset::Leco_int<T> codec;
    codec.init(num_blocks, block_size);
    
    std::vector<uint8_t*> block_start_vec;
    size_t total_compressed_bits = 0;
    
    auto t1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_blocks; i++) {
        int block_length = block_size;
        if (i == num_blocks - 1) {
            block_length = n - (num_blocks - 1) * block_size;
        }

        uint8_t* descriptor = (uint8_t*)malloc(block_length * sizeof(T) * 4);
        uint8_t* res = descriptor;
        
        res = codec.encodeArray8_int(reinterpret_cast<const T*>(data.data() + (i * block_size)), block_length, descriptor, i);
        
        uint32_t segment_size = res - descriptor;
        uint8_t* exact_buf = (uint8_t*)realloc(descriptor, segment_size);
        if (exact_buf) descriptor = exact_buf;
        
        block_start_vec.push_back(descriptor);
        total_compressed_bits += segment_size * 8;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto compression_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    
    result.compressed_bits = total_compressed_bits;
    result.compression_ratio = static_cast<double>(result.compressed_bits) / result.uncompressed_bits;
    result.compression_throughput_mbs = (n * sizeof(T) / 1024.0 / 1024.0) / (compression_time_ns / 1e9);
    
    // Full decompression - with checksum to prevent optimization
    std::vector<T> decompressed(n);
    volatile T decomp_checksum = 0;  // volatile prevents optimization
    t1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_blocks; i++) {
        int block_length = block_size;
        if (i == num_blocks - 1) {
            block_length = n - (num_blocks - 1) * block_size;
        }
        codec.decodeArray8(block_start_vec[i], block_length, decompressed.data() + i * block_size, i);
        decomp_checksum += decompressed[i * block_size];  // Force computation
    }
    for (auto index : codec.mul_add_diff_set) {
        decompressed[index.first] += index.second;
    }
    asm volatile("" ::: "memory");  // Compiler barrier before timing ends
    t2 = std::chrono::high_resolution_clock::now();
    do_not_optimize(decompressed);
    do_not_optimize(decomp_checksum);
    auto decompression_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    
    result.decompression_throughput_mbs = (n * sizeof(T) / 1024.0 / 1024.0) / (decompression_time_ns / 1e9);
    
    for (size_t i = 0; i < n; ++i) {
        if (data[i] != decompressed[i]) {
            std::cerr << "LeCo decompression error at " << i << std::endl;
            break;
        }
    }
    
    // Random access - with volatile checksum to prevent optimization
    const size_t num_ra_queries = 10000;
    volatile T ra_sum = 0;  // volatile prevents optimization
    t1 = std::chrono::high_resolution_clock::now();
    size_t q_count = 0;
    for (auto idx : bench_data.random_indices) {
        if (q_count++ >= num_ra_queries) break;
        
        size_t ib = idx / block_size;
        size_t offset_in_block = idx % block_size;
        
        T val = codec.randomdecodeArray8(block_start_vec[ib], offset_in_block, nullptr, n);
        ra_sum += val;
    }
    asm volatile("" ::: "memory");  // Compiler barrier before timing ends
    t2 = std::chrono::high_resolution_clock::now();
    do_not_optimize(ra_sum);
    auto ra_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.random_access_ns = static_cast<double>(ra_time_ns) / num_ra_queries;
    result.random_access_mbs = (num_ra_queries * sizeof(T) / 1024.0 / 1024.0) / (ra_time_ns / 1e9);
    
    // Range queries - with volatile checksum to prevent optimization
    const size_t num_range_queries = 1000;
    for (auto range : range_sizes) {
        if (range >= n) continue;
        
        const auto& range_indices = bench_data.range_query_indices.at(range);
        std::vector<T> out_buffer(range);
        volatile T range_checksum = 0;  // volatile prevents optimization
        
        t1 = std::chrono::high_resolution_clock::now();
        size_t q_count_range = 0;
        for (auto start_idx : range_indices) {
            if (q_count_range++ >= num_range_queries) break;
            
            size_t start_block = start_idx / block_size;
            size_t end_block = (start_idx + range - 1) / block_size;
            
            size_t out_pos = 0;
            for (size_t ib = start_block; ib <= end_block; ++ib) {
                int block_length = block_size;
                if (ib == num_blocks - 1) {
                    block_length = n - (num_blocks - 1) * block_size;
                }
                
                std::vector<T> block_buffer(block_length);
                codec.decodeArray8(block_start_vec[ib], block_length, block_buffer.data(), ib);
                
                size_t block_start_global = ib * block_size;
                size_t skip = (ib == start_block) ? (start_idx - block_start_global) : 0;
                size_t take = std::min((size_t)block_length - skip, range - out_pos);
                
                memcpy(out_buffer.data() + out_pos, block_buffer.data() + skip, take * sizeof(T));
                out_pos += take;
            }
            range_checksum += out_buffer[0];  // Force computation
            do_not_optimize(out_buffer);
        }
        asm volatile("" ::: "memory");  // Compiler barrier before timing ends
        t2 = std::chrono::high_resolution_clock::now();
        do_not_optimize(range_checksum);
        
        auto range_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
        double throughput = ((range * sizeof(T)) * num_range_queries / 1024.0 / 1024.0) / (range_time_ns / 1e9);
        result.range_query_throughputs.emplace_back(range, throughput);
    }
    
    // Cleanup
    for(auto* ptr : block_start_vec) {
        free(ptr);
    }
    
    return result;
}

