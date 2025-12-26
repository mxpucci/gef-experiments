/**
 * Comprehensive Lossless Compressor Benchmark
 * 
 * Benchmarks all lossless compressors with the following metrics:
 * 1. Compression ratio
 * 2. Compression throughput (MB/s)
 * 3. Random access throughput (ns/query)
 * 4. Full decompression throughput (MB/s)
 * 5. Range queries throughput (MB/s) with parametric range
 * 
 * Compile with -DUSE_SQUASH to enable Squash-based compressors (lz4, zstd, etc.)
 */

#include <iostream>
#include <fstream>
#include <chrono>
#include <random>
#include <vector>
#include <string>
#include <filesystem>
#include <iomanip>
#include <functional>
#include <climits>

// SDSL includes
#include <sdsl/bit_vectors.hpp>
#include <sdsl/dac_vector.hpp>

// Squash compression library (optional)
#ifdef USE_SQUASH
#include <squash/squash.h>
#define HAS_SQUASH 1
#else
#define HAS_SQUASH 0
#endif

// Local includes
#include "NeaTS/NeaTS.hpp"
#include "NeaTS/algorithms.hpp"

// Streaming compressors
#include "Chimp/CompressorChimp.hpp"
#include "Chimp/DecompressorChimp.hpp"
#include "Chimp128/CompressorChimp128.hpp"
#include "Chimp128/DecompressorChimp128.hpp"
#include "TSXor/CompressorTSXor.hpp"
#include "TSXor/DecompressorTSXor.hpp"
#include "Gorilla/CompressorGorilla.hpp"
#include "Gorilla/DecompressorGorilla.hpp"
#include "Elf/CompressorElf.hpp"
#include "Elf/DecompressorElf.hpp"
#include "Camel/CompressorCamel.hpp"
#include "Camel/DecompressorCamel.hpp"
#include "Falcon/CompressorFalcon.hpp"
#include "Falcon/DecompressorFalcon.hpp"

// ============================================================================
// Utility functions
// ============================================================================

template<class T>
void do_not_optimize(T const &value) {
    asm volatile("" : : "r,m"(value) : "memory");
}

template<class T>
const auto to_bytes = [](auto &&x) -> std::array<uint8_t, sizeof(T)> {
    std::array<uint8_t, sizeof(T)> arrayOfByte{};
    for (size_t i = 0; i < sizeof(T); i++)
        arrayOfByte[(sizeof(T) - 1) - i] = (x >> (i * 8));
    return arrayOfByte;
};

std::vector<std::string> get_files(const std::string &path) {
    std::vector<std::string> files;
    for (const auto &entry : std::filesystem::directory_iterator(path)) {
        if (entry.path().extension() == ".bin")
            files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());
    return files;
}

std::string extract_filename(const std::string &path) {
    return std::filesystem::path(path).filename().string();
}

// ============================================================================
// Benchmark result structure
// ============================================================================

struct BenchmarkResult {
    std::string compressor;
    std::string dataset;
    size_t num_values;
    size_t uncompressed_bits;
    size_t compressed_bits;
    double compression_ratio;
    double compression_throughput_mbs;
    double decompression_throughput_mbs;
    double random_access_ns;
    std::vector<std::pair<size_t, double>> range_query_throughputs; // (range_size, MB/s)
    
    void print_header(std::ostream &out) const {
        out << "compressor,dataset,num_values,uncompressed_bits,compressed_bits,"
            << "compression_ratio,compression_throughput_mbs,decompression_throughput_mbs,"
            << "random_access_ns";
        for (const auto &[range, _] : range_query_throughputs) {
            out << ",range_" << range << "_mbs";
        }
        out << std::endl;
    }
    
    void print(std::ostream &out) const {
        out << std::fixed << std::setprecision(4);
        out << compressor << "," << extract_filename(dataset) << "," << num_values << ","
            << uncompressed_bits << "," << compressed_bits << ","
            << compression_ratio << "," << compression_throughput_mbs << ","
            << decompression_throughput_mbs << "," << random_access_ns;
        for (const auto &[_, throughput] : range_query_throughputs) {
            out << "," << throughput;
        }
        out << std::endl;
    }
};

// ============================================================================
// Random index generators
// ============================================================================

std::vector<size_t> generate_random_indices(size_t n, size_t num_queries, uint32_t seed = 2323) {
    std::mt19937 mt(seed);
    std::uniform_int_distribution<size_t> dist(0, n - 1);
    std::vector<size_t> indices(num_queries);
    for (auto &idx : indices) {
        idx = dist(mt);
    }
    return indices;
}

std::vector<size_t> generate_range_indices(size_t n, size_t range, size_t num_queries, uint32_t seed = 1234) {
    std::mt19937 mt(seed);
    if (range >= n) range = n - 1;
    std::uniform_int_distribution<size_t> dist(0, n - range - 1);
    std::vector<size_t> indices(num_queries);
    for (auto &idx : indices) {
        idx = dist(mt);
    }
    return indices;
}

// ============================================================================
// NeaTS Compressor Benchmark
// ============================================================================

template<typename T = int64_t>
BenchmarkResult benchmark_neats(const std::string &filename, 
                                const std::vector<size_t> &range_sizes,
                                uint8_t max_bpc = 32) {
    BenchmarkResult result;
    result.compressor = "NeaTS";
    result.dataset = filename;
    
    // Load data
    auto data = pfa::algorithm::io::read_data_binary<T, T>(filename, false);
    auto min_data = *std::min_element(data.begin(), data.end());
    min_data = min_data < 0 ? (min_data - 1) : -1;
    
    std::vector<T> processed_data(data.size());
    std::transform(data.begin(), data.end(), processed_data.begin(),
                   [min_data](T d) { return d - min_data; });
    
    result.num_values = data.size();
    result.uncompressed_bits = data.size() * sizeof(T) * 8;
    
    // Compression
    pfa::neats::compressor<uint32_t, T, double, float, double> compressor(max_bpc);
    
    auto t1 = std::chrono::high_resolution_clock::now();
    compressor.partitioning(processed_data.begin(), processed_data.end());
    auto t2 = std::chrono::high_resolution_clock::now();
    auto compression_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    
    result.compressed_bits = compressor.size_in_bits();
    result.compression_ratio = static_cast<double>(result.compressed_bits) / result.uncompressed_bits;
    result.compression_throughput_mbs = (data.size() * sizeof(T) / 1024 / 1024) / (compression_time_ns / 1e9);
    
    // Full decompression
    std::vector<T> decompressed(data.size());
    t1 = std::chrono::high_resolution_clock::now();
    compressor.simd_decompress(decompressed.data());
    t2 = std::chrono::high_resolution_clock::now();
    auto decompression_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    do_not_optimize(decompressed);
    
    result.decompression_throughput_mbs = (data.size() * sizeof(T) / 1024 / 1024) / (decompression_time_ns / 1e9);
    
    // Verify decompression
    for (size_t i = 0; i < data.size(); ++i) {
        if (processed_data[i] != decompressed[i]) {
            std::cerr << "NeaTS decompression error at " << i << std::endl;
            break;
        }
    }
    
    // Random access
    const size_t num_ra_queries = 1000000;
    auto indices = generate_random_indices(data.size(), num_ra_queries);
    
    t1 = std::chrono::high_resolution_clock::now();
    T sum = 0;
    for (auto idx : indices) {
        sum += compressor[idx];
    }
    t2 = std::chrono::high_resolution_clock::now();
    do_not_optimize(sum);
    auto ra_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.random_access_ns = static_cast<double>(ra_time_ns) / num_ra_queries;
    
    // Range queries
    const size_t num_range_queries = 10000;
    for (auto range : range_sizes) {
        if (range >= data.size()) continue;
        
        auto range_indices = generate_range_indices(data.size(), range, num_range_queries);
        std::vector<T> out_buffer(range);
        
        t1 = std::chrono::high_resolution_clock::now();
        for (auto start_idx : range_indices) {
            compressor.simd_scan(start_idx, start_idx + range, out_buffer.data());
            do_not_optimize(out_buffer);
        }
        t2 = std::chrono::high_resolution_clock::now();
        
        auto range_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
        double throughput = ((range * sizeof(T)) * num_range_queries / 1024 / 1024) / (range_time_ns / 1e9);
        result.range_query_throughputs.emplace_back(range, throughput);
    }
    
    return result;
}

// ============================================================================
// DAC (Direct Access Codes) Benchmark
// ============================================================================

template<typename T = int64_t>
BenchmarkResult benchmark_dac(const std::string &filename, 
                              const std::vector<size_t> &range_sizes) {
    BenchmarkResult result;
    result.compressor = "DAC";
    result.dataset = filename;
    
    // Load data
    auto data = pfa::algorithm::io::read_data_binary<T, T>(filename, false);
    auto min_data = *std::min_element(data.begin(), data.end());
    min_data = min_data < 0 ? (min_data - 1) : -1;
    
    std::vector<uint64_t> u_data(data.size());
    std::transform(data.begin(), data.end(), u_data.begin(),
                   [min_data](T x) { return static_cast<uint64_t>(x - min_data); });
    
    result.num_values = data.size();
    result.uncompressed_bits = data.size() * sizeof(T) * 8;
    
    // Compression
    auto t1 = std::chrono::high_resolution_clock::now();
    sdsl::dac_vector_dp<> dac_vector(u_data);
    auto t2 = std::chrono::high_resolution_clock::now();
    do_not_optimize(dac_vector);
    auto compression_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    
    result.compressed_bits = sdsl::size_in_bytes(dac_vector) * 8;
    result.compression_ratio = static_cast<double>(result.compressed_bits) / result.uncompressed_bits;
    result.compression_throughput_mbs = (data.size() * sizeof(T) / 1024 / 1024) / (compression_time_ns / 1e9);
    
    // Full decompression
    std::vector<uint64_t> decompressed(data.size());
    t1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < data.size(); ++i) {
        decompressed[i] = dac_vector[i];
    }
    t2 = std::chrono::high_resolution_clock::now();
    do_not_optimize(decompressed);
    auto decompression_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    
    result.decompression_throughput_mbs = (data.size() * sizeof(T) / 1024 / 1024) / (decompression_time_ns / 1e9);
    
    // Random access
    const size_t num_ra_queries = 1000000;
    auto indices = generate_random_indices(data.size(), num_ra_queries);
    
    t1 = std::chrono::high_resolution_clock::now();
    uint64_t sum = 0;
    for (auto idx : indices) {
        sum += dac_vector[idx];
    }
    t2 = std::chrono::high_resolution_clock::now();
    do_not_optimize(sum);
    auto ra_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.random_access_ns = static_cast<double>(ra_time_ns) / num_ra_queries;
    
    // Range queries
    const size_t num_range_queries = 10000;
    for (auto range : range_sizes) {
        if (range >= data.size()) continue;
        
        auto range_indices = generate_range_indices(data.size(), range, num_range_queries);
        std::vector<uint64_t> out_buffer(range);
        
        t1 = std::chrono::high_resolution_clock::now();
        for (auto start_idx : range_indices) {
            std::copy(dac_vector.begin() + start_idx, 
                      dac_vector.begin() + start_idx + range, 
                      out_buffer.begin());
            do_not_optimize(out_buffer);
        }
        t2 = std::chrono::high_resolution_clock::now();
        
        auto range_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
        double throughput = ((range * sizeof(T)) * num_range_queries / 1024 / 1024) / (range_time_ns / 1e9);
        result.range_query_throughputs.emplace_back(range, throughput);
    }
    
    return result;
}

// ============================================================================
// Streaming Compressor (Gorilla, Chimp, Chimp128) Benchmark Template
// These use BitStream as buffer
// ============================================================================

template<typename Compressor, typename Decompressor, typename T = double>
BenchmarkResult benchmark_bitstream_compressor(const std::string &compressor_name,
                                                const std::string &filename,
                                                const std::vector<size_t> &range_sizes,
                                                size_t block_size = 1000) {
    BenchmarkResult result;
    result.compressor = compressor_name;
    result.dataset = filename;
    
    // Load data
    auto data = pfa::algorithm::io::read_data_binary<T, T>(filename, false);
    result.num_values = data.size();
    result.uncompressed_bits = data.size() * sizeof(T) * 8;
    
    const size_t n = data.size();
    const size_t num_blocks = (n / block_size) + (n % block_size != 0);
    
    // Compression
    size_t total_compressed_bits = 0;
    std::vector<std::unique_ptr<Compressor>> compressed_blocks;
    
    auto t1 = std::chrono::high_resolution_clock::now();
    for (size_t ib = 0; ib < num_blocks; ++ib) {
        const size_t bs = std::min(block_size, n - ib * block_size);
        auto data_block = std::vector<T>(data.begin() + ib * block_size,
                                         data.begin() + ib * block_size + bs);
        
        auto cmpr = std::make_unique<Compressor>(*data_block.begin());
        for (auto it = data_block.begin() + 1; it < data_block.end(); ++it) {
            cmpr->addValue(*it);
        }
        cmpr->close();
        total_compressed_bits += cmpr->getSize();
        compressed_blocks.push_back(std::move(cmpr));
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto compression_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    
    result.compressed_bits = total_compressed_bits;
    result.compression_ratio = static_cast<double>(result.compressed_bits) / result.uncompressed_bits;
    result.compression_throughput_mbs = (n * sizeof(T) / 1024 / 1024) / (compression_time_ns / 1e9);
    
    // Full decompression
    std::vector<T> decompressed(n);
    t1 = std::chrono::high_resolution_clock::now();
    size_t offset = 0;
    for (size_t ib = 0; ib < num_blocks; ++ib) {
        const size_t bs = std::min(block_size, n - ib * block_size);
        auto dcmpr = Decompressor(compressed_blocks[ib]->getBuffer(), bs);
        decompressed[offset++] = dcmpr.storedValue;
        while (dcmpr.hasNext()) {
            decompressed[offset++] = dcmpr.storedValue;
        }
    }
    t2 = std::chrono::high_resolution_clock::now();
    do_not_optimize(decompressed);
    auto decompression_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    
    result.decompression_throughput_mbs = (n * sizeof(T) / 1024 / 1024) / (decompression_time_ns / 1e9);
    
    // Random access (requires block decompression)
    const size_t num_ra_queries = 100000; // Fewer queries since it's slower
    auto indices = generate_random_indices(n, num_ra_queries);
    
    t1 = std::chrono::high_resolution_clock::now();
    T sum = 0;
    for (auto idx : indices) {
        size_t ib = idx / block_size;
        size_t offset_in_block = idx % block_size;
        size_t bs = std::min(block_size, n - ib * block_size);
        
        auto dcmpr = Decompressor(compressed_blocks[ib]->getBuffer(), bs);
        size_t i = 0;
        while (i < offset_in_block && dcmpr.hasNext()) {
            ++i;
        }
        sum += dcmpr.storedValue;
    }
    t2 = std::chrono::high_resolution_clock::now();
    do_not_optimize(sum);
    auto ra_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.random_access_ns = static_cast<double>(ra_time_ns) / num_ra_queries;
    
    // Range queries (requires block-wise decompression)
    const size_t num_range_queries = 1000;
    for (auto range : range_sizes) {
        if (range >= n) continue;
        
        auto range_indices = generate_range_indices(n, range, num_range_queries);
        std::vector<T> out_buffer(range);
        
        t1 = std::chrono::high_resolution_clock::now();
        for (auto start_idx : range_indices) {
            size_t start_block = start_idx / block_size;
            size_t end_block = (start_idx + range - 1) / block_size;
            
            size_t out_pos = 0;
            for (size_t ib = start_block; ib <= end_block; ++ib) {
                size_t bs = std::min(block_size, n - ib * block_size);
                auto dcmpr = Decompressor(compressed_blocks[ib]->getBuffer(), bs);
                
                size_t block_start = ib * block_size;
                size_t i = 0;
                
                // Skip to start of range within block
                size_t skip_to = (ib == start_block) ? (start_idx - block_start) : 0;
                while (i < skip_to && dcmpr.hasNext()) {
                    ++i;
                }
                
                // Copy values
                size_t copy_end = std::min(block_start + bs, start_idx + range);
                while (block_start + i < copy_end) {
                    if (out_pos < range) {
                        out_buffer[out_pos++] = dcmpr.storedValue;
                    }
                    if (!dcmpr.hasNext()) break;
                    ++i;
                }
            }
            do_not_optimize(out_buffer);
        }
        t2 = std::chrono::high_resolution_clock::now();
        
        auto range_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
        double throughput = ((range * sizeof(T)) * num_range_queries / 1024 / 1024) / (range_time_ns / 1e9);
        result.range_query_throughputs.emplace_back(range, throughput);
    }
    
    return result;
}

// ============================================================================
// TSXor Benchmark (uses byte vector buffer instead of BitStream)
// ============================================================================

template<typename T = double>
BenchmarkResult benchmark_tsxor(const std::string &filename,
                                const std::vector<size_t> &range_sizes,
                                size_t block_size = 1000) {
    BenchmarkResult result;
    result.compressor = "TSXor";
    result.dataset = filename;
    
    // Load data
    auto data = pfa::algorithm::io::read_data_binary<T, T>(filename, false);
    result.num_values = data.size();
    result.uncompressed_bits = data.size() * sizeof(T) * 8;
    
    const size_t n = data.size();
    const size_t num_blocks = (n / block_size) + (n % block_size != 0);
    
    // Compression - store compressed bytes per block
    size_t total_compressed_bits = 0;
    std::vector<std::vector<uint8_t>> compressed_blocks(num_blocks);
    
    auto t1 = std::chrono::high_resolution_clock::now();
    for (size_t ib = 0; ib < num_blocks; ++ib) {
        const size_t bs = std::min(block_size, n - ib * block_size);
        auto data_block = std::vector<T>(data.begin() + ib * block_size,
                                         data.begin() + ib * block_size + bs);
        
        CompressorTSXor<T> cmpr(*data_block.begin());
        for (auto it = data_block.begin() + 1; it < data_block.end(); ++it) {
            cmpr.addValue(*it);
        }
        cmpr.close();
        total_compressed_bits += cmpr.getSize();
        compressed_blocks[ib] = cmpr.bytes; // TSXor uses bytes member
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto compression_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    
    result.compressed_bits = total_compressed_bits;
    result.compression_ratio = static_cast<double>(result.compressed_bits) / result.uncompressed_bits;
    result.compression_throughput_mbs = (n * sizeof(T) / 1024 / 1024) / (compression_time_ns / 1e9);
    
    // Full decompression
    std::vector<T> decompressed(n);
    t1 = std::chrono::high_resolution_clock::now();
    size_t offset = 0;
    for (size_t ib = 0; ib < num_blocks; ++ib) {
        const size_t bs = std::min(block_size, n - ib * block_size);
        DecompressorTSXor<T> dcmpr(compressed_blocks[ib], bs);
        decompressed[offset++] = dcmpr.storedValue;
        while (dcmpr.hasNext()) {
            decompressed[offset++] = dcmpr.storedValue;
        }
    }
    t2 = std::chrono::high_resolution_clock::now();
    do_not_optimize(decompressed);
    auto decompression_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    
    result.decompression_throughput_mbs = (n * sizeof(T) / 1024 / 1024) / (decompression_time_ns / 1e9);
    
    // Random access
    const size_t num_ra_queries = 100000;
    auto indices = generate_random_indices(n, num_ra_queries);
    
    t1 = std::chrono::high_resolution_clock::now();
    T sum = 0;
    for (auto idx : indices) {
        size_t ib = idx / block_size;
        size_t offset_in_block = idx % block_size;
        size_t bs = std::min(block_size, n - ib * block_size);
        
        DecompressorTSXor<T> dcmpr(compressed_blocks[ib], bs);
        size_t i = 0;
        while (i < offset_in_block && dcmpr.hasNext()) {
            ++i;
        }
        sum += dcmpr.storedValue;
    }
    t2 = std::chrono::high_resolution_clock::now();
    do_not_optimize(sum);
    auto ra_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.random_access_ns = static_cast<double>(ra_time_ns) / num_ra_queries;
    
    // Range queries
    const size_t num_range_queries = 1000;
    for (auto range : range_sizes) {
        if (range >= n) continue;
        
        auto range_indices = generate_range_indices(n, range, num_range_queries);
        std::vector<T> out_buffer(range);
        
        t1 = std::chrono::high_resolution_clock::now();
        for (auto start_idx : range_indices) {
            size_t start_block = start_idx / block_size;
            size_t end_block = (start_idx + range - 1) / block_size;
            
            size_t out_pos = 0;
            for (size_t ib = start_block; ib <= end_block; ++ib) {
                size_t bs = std::min(block_size, n - ib * block_size);
                DecompressorTSXor<T> dcmpr(compressed_blocks[ib], bs);
                
                size_t block_start = ib * block_size;
                size_t i = 0;
                
                size_t skip_to = (ib == start_block) ? (start_idx - block_start) : 0;
                while (i < skip_to && dcmpr.hasNext()) {
                    ++i;
                }
                
                size_t copy_end = std::min(block_start + bs, start_idx + range);
                while (block_start + i < copy_end) {
                    if (out_pos < range) {
                        out_buffer[out_pos++] = dcmpr.storedValue;
                    }
                    if (!dcmpr.hasNext()) break;
                    ++i;
                }
            }
            do_not_optimize(out_buffer);
        }
        t2 = std::chrono::high_resolution_clock::now();
        
        auto range_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
        double throughput = ((range * sizeof(T)) * num_range_queries / 1024 / 1024) / (range_time_ns / 1e9);
        result.range_query_throughputs.emplace_back(range, throughput);
    }
    
    return result;
}

// ============================================================================
// Falcon Benchmark (uses byte vector buffer)
// ============================================================================

template<typename T = double>
BenchmarkResult benchmark_falcon(const std::string &filename,
                                  const std::vector<size_t> &range_sizes) {
    BenchmarkResult result;
    result.compressor = "Falcon";
    result.dataset = filename;
    
    // Load data
    auto data = pfa::algorithm::io::read_data_binary<T, T>(filename, false);
    result.num_values = data.size();
    result.uncompressed_bits = data.size() * sizeof(T) * 8;
    
    const size_t n = data.size();
    
    // Compression
    auto t1 = std::chrono::high_resolution_clock::now();
    CompressorFalcon<T> cmpr(*data.begin());
    for (auto it = data.begin() + 1; it < data.end(); ++it) {
        cmpr.addValue(*it);
    }
    cmpr.close();
    auto t2 = std::chrono::high_resolution_clock::now();
    auto compression_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    
    result.compressed_bits = cmpr.getSize();
    result.compression_ratio = static_cast<double>(result.compressed_bits) / result.uncompressed_bits;
    result.compression_throughput_mbs = (n * sizeof(T) / 1024 / 1024) / (compression_time_ns / 1e9);
    
    // Full decompression
    std::vector<T> decompressed(n);
    auto compressed_bytes = cmpr.bytes;
    
    t1 = std::chrono::high_resolution_clock::now();
    DecompressorFalcon<T> dcmpr(compressed_bytes, n);
    size_t offset = 0;
    decompressed[offset++] = dcmpr.storedValue;
    while (dcmpr.hasNext()) {
        decompressed[offset++] = dcmpr.storedValue;
    }
    t2 = std::chrono::high_resolution_clock::now();
    do_not_optimize(decompressed);
    auto decompression_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    
    result.decompression_throughput_mbs = (n * sizeof(T) / 1024 / 1024) / (decompression_time_ns / 1e9);
    
    // Random access (requires full decompression - Falcon doesn't support random access)
    const size_t num_ra_queries = 100000;
    auto indices = generate_random_indices(n, num_ra_queries);
    
    t1 = std::chrono::high_resolution_clock::now();
    T sum = 0;
    for (auto idx : indices) {
        // Falcon doesn't support random access, so we measure from decompressed data
        DecompressorFalcon<T> dcmpr_ra(compressed_bytes, n);
        size_t i = 0;
        while (i < idx && dcmpr_ra.hasNext()) {
            ++i;
        }
        sum += dcmpr_ra.storedValue;
    }
    t2 = std::chrono::high_resolution_clock::now();
    do_not_optimize(sum);
    auto ra_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.random_access_ns = static_cast<double>(ra_time_ns) / num_ra_queries;
    
    // Range queries (requires decompression)
    const size_t num_range_queries = 1000;
    for (auto range : range_sizes) {
        if (range >= n) continue;
        
        auto range_indices = generate_range_indices(n, range, num_range_queries);
        std::vector<T> out_buffer(range);
        
        t1 = std::chrono::high_resolution_clock::now();
        for (auto start_idx : range_indices) {
            DecompressorFalcon<T> dcmpr_range(compressed_bytes, n);
            size_t i = 0;
            // Skip to start
            while (i < start_idx && dcmpr_range.hasNext()) {
                ++i;
            }
            // Copy range
            size_t out_pos = 0;
            out_buffer[out_pos++] = dcmpr_range.storedValue;
            while (out_pos < range && dcmpr_range.hasNext()) {
                out_buffer[out_pos++] = dcmpr_range.storedValue;
            }
            do_not_optimize(out_buffer);
        }
        t2 = std::chrono::high_resolution_clock::now();
        
        auto range_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
        double throughput = ((range * sizeof(T)) * num_range_queries / 1024 / 1024) / (range_time_ns / 1e9);
        result.range_query_throughputs.emplace_back(range, throughput);
    }
    
    return result;
}

// ============================================================================
// Squash-based Compressor Benchmark (LZ4, ZSTD, Brotli, XZ, Snappy)
// Only available when compiled with -DUSE_SQUASH
// ============================================================================

#if HAS_SQUASH
template<typename T = int64_t>
BenchmarkResult benchmark_squash(const std::string &compressor_name,
                                 const std::string &filename,
                                 const std::vector<size_t> &range_sizes,
                                 size_t block_size = 1000,
                                 int level = -1) {
    BenchmarkResult result;
    result.compressor = compressor_name;
    result.dataset = filename;
    
    SquashCodec *codec = squash_get_codec(compressor_name.c_str());
    if (codec == nullptr) {
        std::cerr << "Unable to find squash codec: " << compressor_name << std::endl;
        return result;
    }
    
    SquashOptions *opts = nullptr;
    if (level != -1) {
        char level_s[4];
        opts = squash_options_new(codec, NULL);
        squash_object_ref_sink(opts);
        snprintf(level_s, 4, "%d", level);
        squash_options_parse_option(opts, "level", level_s);
    }
    
    // Load data
    auto data = pfa::algorithm::io::read_data_binary<T, T>(filename, false);
    const size_t n = data.size();
    const size_t num_blocks = n / block_size + (n % block_size != 0);
    
    result.num_values = n;
    result.uncompressed_bits = n * sizeof(T) * 8;
    
    // Compression
    size_t total_compressed_bits = 0;
    std::vector<std::pair<uint8_t*, size_t>> compressed_blocks(num_blocks);
    
    auto t1 = std::chrono::high_resolution_clock::now();
    for (size_t ib = 0; ib < num_blocks; ++ib) {
        const size_t bs = std::min(block_size, n - ib * block_size);
        auto data_block = std::vector<T>(data.begin() + ib * block_size,
                                         data.begin() + ib * block_size + bs);
        
        std::vector<uint8_t> data_bytes;
        for (const auto &val : data_block) {
            auto bytes = to_bytes<T>(val);
            data_bytes.insert(data_bytes.end(), bytes.begin(), bytes.end());
        }
        
        size_t uncompressed_size = data_bytes.size();
        size_t compressed_size = squash_codec_get_max_compressed_size(codec, uncompressed_size);
        auto *compressed_data = (uint8_t*)malloc(compressed_size);
        
        squash_codec_compress_with_options(codec, &compressed_size, compressed_data,
                                     uncompressed_size, data_bytes.data(), opts);
        
        compressed_blocks[ib] = {compressed_data, compressed_size};
        total_compressed_bits += compressed_size * CHAR_BIT;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto compression_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    
    result.compressed_bits = total_compressed_bits;
    result.compression_ratio = static_cast<double>(result.compressed_bits) / result.uncompressed_bits;
    result.compression_throughput_mbs = (n * sizeof(T) / 1024 / 1024) / (compression_time_ns / 1e9);
    
    // Full decompression
    std::vector<T> decompressed(n);
    t1 = std::chrono::high_resolution_clock::now();
    for (size_t ib = 0; ib < num_blocks; ++ib) {
        const size_t bs = std::min(block_size, n - ib * block_size);
        size_t decompressed_size = bs * sizeof(T) + 1;
        auto *decompressed_bytes = (uint8_t*)malloc(decompressed_size);
        
        squash_codec_decompress_with_options(codec, &decompressed_size, decompressed_bytes,
                          compressed_blocks[ib].second, compressed_blocks[ib].first, opts);
        
        for (size_t i = 0; i < bs; ++i) {
            T value = 0;
            for (size_t j = 0; j < sizeof(T); ++j) {
                value = (value << 8) + decompressed_bytes[i * sizeof(T) + j];
            }
            decompressed[ib * block_size + i] = value;
        }
        free(decompressed_bytes);
    }
    t2 = std::chrono::high_resolution_clock::now();
    do_not_optimize(decompressed);
    auto decompression_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    
    result.decompression_throughput_mbs = (n * sizeof(T) / 1024 / 1024) / (decompression_time_ns / 1e9);
    
    // Random access
    const size_t num_ra_queries = 100000;
    auto indices = generate_random_indices(n, num_ra_queries);
    
    t1 = std::chrono::high_resolution_clock::now();
    T sum = 0;
    for (auto idx : indices) {
        size_t ib = idx / block_size;
        size_t offset = idx % block_size;
        size_t bs = std::min(block_size, n - ib * block_size);
        
        size_t decompressed_size = bs * sizeof(T) + 1;
        auto *decompressed_bytes = (uint8_t*)malloc(decompressed_size);
        
        squash_codec_decompress_with_options(codec, &decompressed_size, decompressed_bytes,
                          compressed_blocks[ib].second, compressed_blocks[ib].first, nullptr);
        
        T value = 0;
        for (size_t j = 0; j < sizeof(T); ++j) {
            value = (value << 8) + decompressed_bytes[offset * sizeof(T) + j];
        }
        sum += value;
        free(decompressed_bytes);
    }
    t2 = std::chrono::high_resolution_clock::now();
    do_not_optimize(sum);
    auto ra_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.random_access_ns = static_cast<double>(ra_time_ns) / num_ra_queries;
    
    // Range queries
    const size_t num_range_queries = 1000;
    for (auto range : range_sizes) {
        if (range >= n) continue;
        
        auto range_indices = generate_range_indices(n, range, num_range_queries);
        std::vector<T> out_buffer(range);
        
        t1 = std::chrono::high_resolution_clock::now();
        for (auto start_idx : range_indices) {
            size_t start_block = start_idx / block_size;
            size_t end_block = (start_idx + range - 1) / block_size;
            size_t buffer_blocks = end_block - start_block + 1;
            
            std::vector<uint8_t> decompressed_bytes(buffer_blocks * block_size * sizeof(T) + 1);
            size_t byte_offset = 0;
            
            for (size_t ib = start_block; ib <= end_block; ++ib) {
                size_t bs = std::min(block_size, n - ib * block_size);
                size_t decompressed_size = bs * sizeof(T) + 1;
                
                squash_codec_decompress_with_options(codec, &decompressed_size,
                                  decompressed_bytes.data() + byte_offset,
                                  compressed_blocks[ib].second, compressed_blocks[ib].first, nullptr);
                byte_offset += bs * sizeof(T);
            }
            
            size_t local_offset = (start_idx % block_size) * sizeof(T);
            for (size_t i = 0; i < range; ++i) {
                T value = 0;
                for (size_t j = 0; j < sizeof(T); ++j) {
                    value = (value << 8) + decompressed_bytes[local_offset + i * sizeof(T) + j];
                }
                out_buffer[i] = value;
            }
            do_not_optimize(out_buffer);
        }
        t2 = std::chrono::high_resolution_clock::now();
        
        auto range_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
        double throughput = ((range * sizeof(T)) * num_range_queries / 1024 / 1024) / (range_time_ns / 1e9);
        result.range_query_throughputs.emplace_back(range, throughput);
    }
    
    // Cleanup
    for (auto &block : compressed_blocks) {
        free(block.first);
    }
    if (opts != nullptr) {
        squash_object_unref(opts);
    }
    
    return result;
}
#endif // HAS_SQUASH

// ============================================================================
// Main benchmark runner
// ============================================================================

void print_usage(const char *prog_name) {
    std::cerr << "Usage: " << prog_name << " [options] <input_file_or_directory>" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  -o <file>      Output CSV file (default: stdout)" << std::endl;
    std::cerr << "  -c <list>      Comma-separated list of compressors (default: all)" << std::endl;
#if HAS_SQUASH
    std::cerr << "                 Available: neats,dac,gorilla,chimp,chimp128,tsxor,elf,camel,falcon,lz4,zstd,brotli,xz,snappy" << std::endl;
#else
    std::cerr << "                 Available: neats,dac,gorilla,chimp,chimp128,tsxor,elf,camel,falcon" << std::endl;
    std::cerr << "                 (Compile with -DUSE_SQUASH for lz4,zstd,brotli,xz,snappy)" << std::endl;
#endif
    std::cerr << "  -r <list>      Comma-separated list of range sizes (default: 10,100,1000,10000,100000)" << std::endl;
    std::cerr << "  -b <size>      Block size for block-based compressors (default: 1000)" << std::endl;
    std::cerr << "  -m <bpc>       Max bits per correction for NeaTS (default: 32)" << std::endl;
    std::cerr << "  -d             Data is double (default: int64)" << std::endl;
    std::cerr << "  -h             Show this help" << std::endl;
}

std::vector<std::string> split_string(const std::string &s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream stream(s);
    while (std::getline(stream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

int main(int argc, char *argv[]) {
    // Default parameters
    std::string output_file;
#if HAS_SQUASH
    std::vector<std::string> compressors = {"neats", "dac", "gorilla", "chimp", "chimp128", "tsxor",
                                            "elf", "camel", "falcon", "lz4", "zstd", "brotli", "snappy"};
#else
    std::vector<std::string> compressors = {"neats", "dac", "gorilla", "chimp", "chimp128", "tsxor",
                                            "elf", "camel", "falcon"};
#endif
    std::vector<size_t> range_sizes = {10, 100, 1000, 10000, 100000};
    size_t block_size = 1000;
    uint8_t max_bpc = 32;
    bool use_double = false;
    std::string input_path;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-o" && i + 1 < argc) {
            output_file = argv[++i];
        } else if (arg == "-c" && i + 1 < argc) {
            compressors = split_string(argv[++i], ',');
        } else if (arg == "-r" && i + 1 < argc) {
            auto range_strs = split_string(argv[++i], ',');
            range_sizes.clear();
            for (const auto &s : range_strs) {
                range_sizes.push_back(std::stoull(s));
            }
        } else if (arg == "-b" && i + 1 < argc) {
            block_size = std::stoull(argv[++i]);
        } else if (arg == "-m" && i + 1 < argc) {
            max_bpc = std::stoi(argv[++i]);
        } else if (arg == "-d") {
            use_double = true;
        } else if (arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg[0] != '-') {
            input_path = arg;
        }
    }
    
    if (input_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }
    
    // Get list of files to process
    std::vector<std::string> files;
    if (std::filesystem::is_directory(input_path)) {
        files = get_files(input_path);
    } else {
        files.push_back(input_path);
    }
    
    if (files.empty()) {
        std::cerr << "No .bin files found in " << input_path << std::endl;
        return 1;
    }
    
    // Setup output
    std::ostream *out = &std::cout;
    std::ofstream file_out;
    if (!output_file.empty()) {
        file_out.open(output_file);
        out = &file_out;
    }
    
    // Print header
    bool header_printed = false;
    
    // Run benchmarks
    for (const auto &filename : files) {
        std::cerr << "Processing: " << filename << std::endl;
        
        for (const auto &comp : compressors) {
            std::cerr << "  Running " << comp << "..." << std::flush;
            
            BenchmarkResult result;
            
            try {
                if (comp == "neats") {
                    result = benchmark_neats<int64_t>(filename, range_sizes, max_bpc);
                } else if (comp == "dac") {
                    result = benchmark_dac<int64_t>(filename, range_sizes);
                } else if (comp == "gorilla") {
                    result = benchmark_bitstream_compressor<CompressorGorilla<double>, DecompressorGorilla<double>, double>(
                        "Gorilla", filename, range_sizes, block_size);
                } else if (comp == "chimp") {
                    result = benchmark_bitstream_compressor<CompressorChimp<double>, DecompressorChimp<double>, double>(
                        "Chimp", filename, range_sizes, block_size);
                } else if (comp == "chimp128") {
                    result = benchmark_bitstream_compressor<CompressorChimp128<double>, DecompressorChimp128<double>, double>(
                        "Chimp128", filename, range_sizes, block_size);
                } else if (comp == "tsxor") {
                    result = benchmark_tsxor<double>(filename, range_sizes, block_size);
                } else if (comp == "elf") {
                    result = benchmark_bitstream_compressor<CompressorElf<double>, DecompressorElf<double>, double>(
                        "Elf", filename, range_sizes, block_size);
                } else if (comp == "camel") {
                    result = benchmark_bitstream_compressor<CompressorCamel<double>, DecompressorCamel<double>, double>(
                        "Camel", filename, range_sizes, block_size);
                } else if (comp == "falcon") {
                    result = benchmark_falcon<double>(filename, range_sizes);
#if HAS_SQUASH
                } else if (comp == "lz4" || comp == "zstd" || comp == "brotli" || 
                           comp == "xz" || comp == "snappy") {
                    result = benchmark_squash<int64_t>(comp, filename, range_sizes, block_size);
#endif
                } else {
                    std::cerr << " unknown compressor, skipping" << std::endl;
                    continue;
                }
                
                if (!header_printed) {
                    result.print_header(*out);
                    header_printed = true;
                }
                result.print(*out);
                std::cerr << " done" << std::endl;
                
            } catch (const std::exception &e) {
                std::cerr << " error: " << e.what() << std::endl;
            }
        }
    }
    
    if (file_out.is_open()) {
        file_out.close();
    }
    
    return 0;
}

