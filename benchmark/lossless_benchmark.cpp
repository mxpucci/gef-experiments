#include <bit>
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
#include <cmath>
#include <type_traits>

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
#include "benchmark_common.hpp"
#include <cstring>
#include "recent_compressors.hpp"

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

// ALP and LeCo benchmarks are in separate translation units (alp_benchmark.cpp, leco_benchmark.cpp)
// compiled with different compilers for compatibility. Declarations are in benchmark_common.hpp.

#ifdef NEATS_ENABLE_BUFF
#include "Buff/CompressorBuff.hpp"
#endif


// ============================================================================
// Utility functions
// ============================================================================

struct LoadedDataset {
    std::vector<int64_t> data;
    int64_t decimals;
};

LoadedDataset load_custom_dataset(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    uint64_t n_val = 0;
    in.read(reinterpret_cast<char*>(&n_val), 8);
    size_t n = static_cast<size_t>(n_val);
    
    // Check if file size matches the new format (N + X + Data)
    in.seekg(0, std::ios::end);
    size_t file_size = in.tellg();
    in.seekg(8, std::ios::beg); // Skip N
    
    // New format: 8 bytes N + 8 bytes X + N*8 bytes Data
    size_t expected_size_new = 8 + 8 + n * 8;
    // Old format: 8 bytes N + N*8 bytes Data (assuming 64-bit values)
    size_t expected_size_old = 8 + n * 8;
    
    int64_t x = 0;
    std::vector<int64_t> data(n);
    
    if (file_size == expected_size_new) {
        uint64_t x_val = 0;
        in.read(reinterpret_cast<char*>(&x_val), 8);
        x = static_cast<int64_t>(x_val);
    } else if (file_size == expected_size_old) {
        // Fallback to old format, assume x=0 and data follows immediately
        x = 0;
    } else {
        // Warning or error? Let's try to read data anyway if we can
        // Assuming old format structure for safety if unknown
        std::cerr << "Warning: File size " << file_size << " doesn't match expected new (" 
                  << expected_size_new << ") or old (" << expected_size_old << ") format." << std::endl;
    }
    
    char* ptr = reinterpret_cast<char*>(data.data());
    size_t total_bytes = n * 8;
    size_t bytes_read = 0;
    const size_t CHUNK_SIZE = 1024 * 1024 * 1024; // 1GB chunks

    while (bytes_read < total_bytes) {
        size_t to_read = std::min(CHUNK_SIZE, total_bytes - bytes_read);
        in.read(ptr + bytes_read, to_read);
        size_t read_this_time = in.gcount();
        bytes_read += read_this_time;
        
        if (read_this_time < to_read) {
            break; // EOF or error
        }
    }

    if (bytes_read != total_bytes) {
        std::cerr << "Warning: Read fewer bytes than expected. Expected " << total_bytes << ", got " << bytes_read << std::endl;
    }
    
    return {data, x};
}

// do_not_optimize is provided by `benchmark_common.hpp`

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
    return std::filesystem::path(path).stem().string();
}

// ============================================================================
// Benchmark result structure (definitions)
// ============================================================================

void BenchmarkResult::print_header(std::ostream &out) const {
    out << "compressor,dataset,num_values,uncompressed_bits,compressed_bits,"
        << "compression_ratio,compression_throughput_mbs,decompression_throughput_mbs,"
        << "random_access_ns,random_access_mbs";
    for (const auto &[range, _] : range_query_throughputs) {
        out << ",range_" << range << "_mbs";
    }
    out << ",input_conversion,input_changed_values,roundtrip_mismatches,validation_status"
        << ",access_layout,compression_unit_values,native_page_values,bitwise_mismatches";
    out << std::endl;
}

void BenchmarkResult::print(std::ostream &out) const {
    out << std::fixed << std::setprecision(4);
    out << compressor << "," << extract_filename(dataset) << "," << num_values << ","
        << uncompressed_bits << "," << compressed_bits << ","
        << compression_ratio << "," << compression_throughput_mbs << ","
        << decompression_throughput_mbs << "," << random_access_ns << "," << random_access_mbs;
    for (const auto &[_, throughput] : range_query_throughputs) {
        out << "," << throughput;
    }
    out << "," << input_conversion << "," << input_changed_values
        << "," << roundtrip_mismatches << "," << validation_status
        << "," << access_layout << "," << compression_unit_values << "," << native_page_values
        << "," << bitwise_mismatches;
    out << std::endl;
}

// ============================================================================
// Random index generators
// ============================================================================

std::vector<size_t> generate_random_indices(size_t n, size_t num_queries, uint64_t seed = 2323) {
    std::mt19937 mt(seed);
    std::uniform_int_distribution<size_t> dist(0, n - 1);
    std::vector<size_t> indices(num_queries);
    for (auto &idx : indices) {
        idx = dist(mt);
    }
    return indices;
}

std::vector<size_t> generate_range_indices(size_t n, size_t range, size_t num_queries, uint64_t seed = 1234) {
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

template<typename T, typename T1_coeff>
BenchmarkResult benchmark_neats_impl(const std::vector<T> &processed_data,
                                      const std::vector<size_t> &range_sizes,
                                      const std::vector<size_t> &random_indices,
                                      const std::map<size_t, std::vector<size_t>> &range_query_indices,
                                      uint8_t max_bpc,
                                      const std::string &dataset_name,
                                      size_t uncompressed_bits) {
    BenchmarkResult result;
    result.compressor = "NeaTS";
    result.dataset = dataset_name;
    result.num_values = processed_data.size();
    result.uncompressed_bits = uncompressed_bits;
    
    
    // Use uint64_t for x_t to prevent overflow of bit offsets for large datasets
    // T1_coeff is either float (for small values) or double (for large values)
    pfa::neats::compressor<uint64_t, T, double, T1_coeff, double> compressor(max_bpc);
    
    auto t1 = std::chrono::high_resolution_clock::now();
    compressor.partitioning(processed_data.begin(), processed_data.end());
    compiler_barrier();
    auto t2 = std::chrono::high_resolution_clock::now();
    auto compression_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    
    result.compressed_bits = compressor.size_in_bits();
    result.compression_ratio = static_cast<double>(result.compressed_bits) / result.uncompressed_bits;
    result.compression_throughput_mbs = (processed_data.size() * sizeof(T) / 1024.0 / 1024.0) / (compression_time_ns / 1e9);
    
    // Full decompression
    std::vector<T> decompressed(processed_data.size());
    volatile T decomp_checksum = 0;
    t1 = std::chrono::high_resolution_clock::now();
    compressor.simd_decompress(decompressed.data());
    decomp_checksum += decompressed[0];
    compiler_barrier();
    t2 = std::chrono::high_resolution_clock::now();
    do_not_optimize(decompressed);
    do_not_optimize(decomp_checksum);
    
    auto decompression_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    
    result.decompression_throughput_mbs = (processed_data.size() * sizeof(T) / 1024.0 / 1024.0) / (decompression_time_ns / 1e9);
    
    check_roundtrip(result, processed_data, decompressed);
    finish_roundtrip_validation(result);
    
    // Random access
    const size_t num_ra_queries = random_indices.size();
    
    t1 = std::chrono::high_resolution_clock::now();
    volatile T ra_sum = 0;
    for (auto idx : random_indices) {
        ra_sum += compressor[idx];
    }
    compiler_barrier();
    t2 = std::chrono::high_resolution_clock::now();
    do_not_optimize(ra_sum);
    auto ra_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.random_access_ns = static_cast<double>(ra_time_ns) / num_ra_queries;
    result.random_access_mbs = (num_ra_queries * sizeof(T) / 1024.0 / 1024.0) / (ra_time_ns / 1e9);
    
    // Range queries
    const size_t num_range_queries = range_query_indices.begin()->second.size();
    for (auto range : range_sizes) {
        if (range >= processed_data.size()) continue;
        
        const auto& range_indices = range_query_indices.at(range);
        std::vector<T> out_buffer(range);
        volatile T range_checksum = 0;
        
        t1 = std::chrono::high_resolution_clock::now();
        for (auto start_idx : range_indices) {
            compressor.simd_scan(start_idx, start_idx + range, out_buffer.data());
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

template<typename T = int64_t>
BenchmarkResult benchmark_neats(const BenchmarkData &bench_data, 
                                const std::vector<size_t> &range_sizes,
                                uint8_t max_bpc = 32) {
    // Use shifted data
    const auto& processed_data = bench_data.shifted_data;
    
    // Determine the maximum absolute value to decide coefficient precision
    // A 32-bit float has ~7 significant decimal digits of precision
    // Use double when max value exceeds 10^6 (1 million) to avoid precision loss
    // in slope/intercept calculations with very large values
    constexpr T FLOAT_PRECISION_THRESHOLD = static_cast<T>(1000000); // 10^6 (1 million)
    
    T max_val = *std::max_element(processed_data.begin(), processed_data.end());
    
    if (max_val > FLOAT_PRECISION_THRESHOLD) {
        // Use double precision for coefficients when values are large
        return benchmark_neats_impl<T, double>(processed_data, range_sizes, 
                                                bench_data.random_indices, bench_data.range_query_indices,
                                                max_bpc, bench_data.filename, bench_data.uncompressed_bits);
    } else {
        // Use float precision for coefficients when values are small (better compression)
        return benchmark_neats_impl<T, float>(processed_data, range_sizes,
                                               bench_data.random_indices, bench_data.range_query_indices,
                                               max_bpc, bench_data.filename, bench_data.uncompressed_bits);
    }
}
// ============================================================================
// DAC (Direct Access Codes) Benchmark
// ============================================================================

template<typename T = int64_t>
BenchmarkResult benchmark_dac(const BenchmarkData &bench_data, 
                              const std::vector<size_t> &range_sizes) {
    BenchmarkResult result;
    result.compressor = "DAC";
    result.dataset = bench_data.filename;
    
    // Use shifted data which is already non-negative
    const auto& data = bench_data.shifted_data;
    
    result.num_values = data.size();
    result.uncompressed_bits = bench_data.uncompressed_bits;
    
    std::vector<uint64_t> u_data(data.size());
    std::transform(data.begin(), data.end(), u_data.begin(),
                   [](int64_t x) { return static_cast<uint64_t>(x); });
    
    // Compression
    auto t1 = std::chrono::high_resolution_clock::now();
    sdsl::dac_vector_dp<> dac_vector(u_data);
    compiler_barrier();
    auto t2 = std::chrono::high_resolution_clock::now();
    do_not_optimize(dac_vector);
    auto compression_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    
    result.compressed_bits = sdsl::size_in_bytes(dac_vector) * 8;
    result.compression_ratio = static_cast<double>(result.compressed_bits) / result.uncompressed_bits;
    result.compression_throughput_mbs = (data.size() * sizeof(T) / 1024.0 / 1024.0) / (compression_time_ns / 1e9);
    
    // Full decompression
    std::vector<uint64_t> decompressed(data.size());
    volatile uint64_t decomp_checksum = 0;
    t1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < data.size(); ++i) {
        decompressed[i] = dac_vector[i];
    }
    decomp_checksum += decompressed[0];
    compiler_barrier();
    t2 = std::chrono::high_resolution_clock::now();
    do_not_optimize(decompressed);
    do_not_optimize(decomp_checksum);
    auto decompression_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    
    result.decompression_throughput_mbs = (data.size() * sizeof(T) / 1024.0 / 1024.0) / (decompression_time_ns / 1e9);
    
    check_roundtrip(result, u_data, decompressed);
    finish_roundtrip_validation(result);
    
    // Random access
    const size_t num_ra_queries = bench_data.random_indices.size();
    
    t1 = std::chrono::high_resolution_clock::now();
    volatile uint64_t ra_sum = 0;
    for (auto idx : bench_data.random_indices) {
        ra_sum += dac_vector[idx];
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
        std::vector<uint64_t> out_buffer(range);
        volatile uint64_t range_checksum = 0;
        
        t1 = std::chrono::high_resolution_clock::now();
        for (auto start_idx : range_indices) {
            std::copy(dac_vector.begin() + start_idx, 
                      dac_vector.begin() + start_idx + range, 
                      out_buffer.begin());
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

// ============================================================================
// Streaming Compressor (Gorilla, Chimp, Chimp128) Benchmark Template
// These use BitStream as buffer
// ============================================================================

template<typename Compressor, typename Decompressor, typename T = double>
BenchmarkResult benchmark_bitstream_compressor(const std::string &compressor_name,
                                                const BenchmarkData &bench_data,
                                                const std::vector<size_t> &range_sizes,
                                                size_t block_size = 1000) {
    BenchmarkResult result;
    result.compressor = compressor_name;
    result.dataset = bench_data.filename;
    
    // Use double data
    const auto& data = bench_data.double_data;
    
    result.num_values = data.size();
    result.uncompressed_bits = bench_data.uncompressed_bits;
    
    const size_t n = data.size();
    const size_t num_blocks = (n / block_size) + (n % block_size != 0);
    
    // Compression
    size_t total_compressed_bits = 0;
    std::vector<std::unique_ptr<Compressor>> compressed_blocks;
    
    auto t1 = std::chrono::high_resolution_clock::now();
    for (size_t ib = 0; ib < num_blocks; ++ib) {
        const size_t bs = std::min(block_size, n - ib * block_size);
        // Using iterators directly from the main vector
        auto start_it = data.begin() + ib * block_size;
        auto end_it = start_it + bs;
        
        auto cmpr = std::make_unique<Compressor>(*start_it);

        for (auto it = start_it + 1; it < end_it; ++it) {
            cmpr->addValue(*it);
        }
        cmpr->close();
        total_compressed_bits += cmpr->getSize();
        compressed_blocks.push_back(std::move(cmpr));
    }
    compiler_barrier();
    auto t2 = std::chrono::high_resolution_clock::now();
    auto compression_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    
    result.compressed_bits = total_compressed_bits;
    result.compression_ratio = static_cast<double>(result.compressed_bits) / result.uncompressed_bits;
    result.compression_throughput_mbs = (n * sizeof(T) / 1024.0 / 1024.0) / (compression_time_ns / 1e9);
    
    // Full decompression
    std::vector<T> decompressed(n);
    volatile T decomp_checksum = 0;
    t1 = std::chrono::high_resolution_clock::now();
    size_t offset = 0;
    for (size_t ib = 0; ib < num_blocks; ++ib) {
        const size_t bs = std::min(block_size, n - ib * block_size);
        auto dcmpr = Decompressor(compressed_blocks[ib]->getBuffer(), bs);
        decompressed[offset++] = dcmpr.storedValue;
        for (size_t k = 1; k < bs; ++k) {
            if (dcmpr.hasNext()) {
                decompressed[offset++] = dcmpr.storedValue;
            } else {
                break;
            }
        }
        decomp_checksum += decompressed[ib * block_size];
    }
    compiler_barrier();
    t2 = std::chrono::high_resolution_clock::now();
    do_not_optimize(decompressed);
    do_not_optimize(decomp_checksum);
    auto decompression_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    
    result.decompression_throughput_mbs = (n * sizeof(T) / 1024.0 / 1024.0) / (decompression_time_ns / 1e9);
    
    check_roundtrip(result, data, decompressed);
    finish_roundtrip_validation(result);
    result.access_layout = "independent_blocks";
    result.compression_unit_values = block_size;

    // Random access (requires block decompression)
    const size_t num_ra_queries = 100000; // Fewer queries since it's slower
    // Use subset of random indices if we generated more for others?
    // The original code used 100,000 for this, but 1,000,000 for others.
    // bench_data.random_indices has 1,000,000. We can just take the first 100,000.
    
    t1 = std::chrono::high_resolution_clock::now();
    volatile T ra_sum = 0;
    size_t query_count = 0;
    for (auto idx : bench_data.random_indices) {
        if (query_count++ >= num_ra_queries) break;
        
        size_t ib = idx / block_size;
        size_t offset_in_block = idx % block_size;
        size_t bs = std::min(block_size, n - ib * block_size);
        
        auto dcmpr = Decompressor(compressed_blocks[ib]->getBuffer(), bs);
        size_t i = 0;
        while (i < offset_in_block && dcmpr.hasNext()) {
            ++i;
        }
        ra_sum += dcmpr.storedValue;
    }
    compiler_barrier();
    t2 = std::chrono::high_resolution_clock::now();
    do_not_optimize(ra_sum);
    auto ra_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.random_access_ns = static_cast<double>(ra_time_ns) / num_ra_queries;
    result.random_access_mbs = (num_ra_queries * sizeof(T) / 1024.0 / 1024.0) / (ra_time_ns / 1e9);
    
    // Range queries (requires block-wise decompression)
    const size_t num_range_queries = 1000;
    
    for (auto range : range_sizes) {
        if (range >= n) continue;
        
        const auto& range_indices = bench_data.range_query_indices.at(range);
        std::vector<T> out_buffer(range);
        volatile T range_checksum = 0;
        
        t1 = std::chrono::high_resolution_clock::now();
        size_t q_count = 0;
        for (auto start_idx : range_indices) {
            if (q_count++ >= num_range_queries) break;
            
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
                    if (block_start + i + 1 < copy_end) {
                         if (!dcmpr.hasNext()) break;
                    }
                    ++i;
                }
            }
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

// ============================================================================
// TSXor Benchmark (uses byte vector buffer instead of BitStream)
// ============================================================================

template<typename T = double>
BenchmarkResult benchmark_tsxor(const BenchmarkData &bench_data,
                                const std::vector<size_t> &range_sizes,
                                size_t block_size = 1000) {
    BenchmarkResult result;
    result.compressor = "TSXor";
    result.dataset = bench_data.filename;
    
    // Use double data
    const auto& data = bench_data.double_data;
    
    result.num_values = data.size();
    result.uncompressed_bits = bench_data.uncompressed_bits;
    
    const size_t n = data.size();
    const size_t num_blocks = (n / block_size) + (n % block_size != 0);
    
    // Compression - store compressed bytes per block
    size_t total_compressed_bits = 0;
    std::vector<std::vector<uint8_t>> compressed_blocks(num_blocks);
    
    auto t1 = std::chrono::high_resolution_clock::now();
    for (size_t ib = 0; ib < num_blocks; ++ib) {
        const size_t bs = std::min(block_size, n - ib * block_size);
        // Using iterators directly from the main vector
        auto start_it = data.begin() + ib * block_size;
        auto end_it = start_it + bs;
        
        CompressorTSXor<T> cmpr(*start_it);
        for (auto it = start_it + 1; it < end_it; ++it) {
            cmpr.addValue(*it);
        }
        cmpr.close();
        total_compressed_bits += cmpr.getSize();
        compressed_blocks[ib] = cmpr.bytes; // TSXor uses bytes member
    }
    compiler_barrier();
    auto t2 = std::chrono::high_resolution_clock::now();
    auto compression_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();

    result.compressed_bits = total_compressed_bits;
    result.compression_ratio = static_cast<double>(result.compressed_bits) / result.uncompressed_bits;
    result.compression_throughput_mbs = (n * sizeof(T) / 1024.0 / 1024.0) / (compression_time_ns / 1e9);
    
    // Full decompression
    std::vector<T> decompressed(n);
    volatile T decomp_checksum = 0;
    t1 = std::chrono::high_resolution_clock::now();
    size_t offset = 0;
    for (size_t ib = 0; ib < num_blocks; ++ib) {
        const size_t bs = std::min(block_size, n - ib * block_size);
        DecompressorTSXor<T> dcmpr(compressed_blocks[ib], bs);
        decompressed[offset++] = dcmpr.storedValue;
        for (size_t k = 1; k < bs; ++k) {
            if (dcmpr.hasNext()) {
                decompressed[offset++] = dcmpr.storedValue;
            } else {
                break;
            }
        }
        decomp_checksum += decompressed[ib * block_size];
    }
    compiler_barrier();
    t2 = std::chrono::high_resolution_clock::now();
    do_not_optimize(decompressed);
    do_not_optimize(decomp_checksum);
    auto decompression_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    
    result.decompression_throughput_mbs = (n * sizeof(T) / 1024.0 / 1024.0) / (decompression_time_ns / 1e9);
    
    check_roundtrip(result, data, decompressed);
    finish_roundtrip_validation(result);
    result.access_layout = "independent_blocks";
    result.compression_unit_values = block_size;
    
    // Random access
    const size_t num_ra_queries = 100000;
    
    t1 = std::chrono::high_resolution_clock::now();
    volatile T ra_sum = 0;
    size_t q_count = 0;
    for (auto idx : bench_data.random_indices) {
        if (q_count++ >= num_ra_queries) break;
        
        size_t ib = idx / block_size;
        size_t offset_in_block = idx % block_size;
        size_t bs = std::min(block_size, n - ib * block_size);
        
        DecompressorTSXor<T> dcmpr(compressed_blocks[ib], bs);
        size_t i = 0;
        while (i < offset_in_block && dcmpr.hasNext()) {
            ++i;
        }
        ra_sum += dcmpr.storedValue;
    }
    compiler_barrier();
    t2 = std::chrono::high_resolution_clock::now();
    do_not_optimize(ra_sum);
    auto ra_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.random_access_ns = static_cast<double>(ra_time_ns) / num_ra_queries;
    result.random_access_mbs = (num_ra_queries * sizeof(T) / 1024.0 / 1024.0) / (ra_time_ns / 1e9);
    
    // Range queries
    const size_t num_range_queries = 1000;
    for (auto range : range_sizes) {
        if (range >= n) continue;
        
        const auto& range_indices = bench_data.range_query_indices.at(range);
        std::vector<T> out_buffer(range);
        volatile T range_checksum = 0;
        
        t1 = std::chrono::high_resolution_clock::now();
        size_t q_count_range = 0;
        for (auto start_idx : range_indices) {
            if (q_count_range++ >= num_range_queries) break;
            
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
                    if (block_start + i + 1 < copy_end) {
                        if (!dcmpr.hasNext()) break;
                    }
                    ++i;
                }
            }
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

// ============================================================================
// BUFF Benchmark
// ============================================================================

#ifdef NEATS_ENABLE_BUFF
template<typename T = double>
BenchmarkResult benchmark_buff(const BenchmarkData &bench_data,
                               const std::vector<size_t> &range_sizes) {
    BenchmarkResult result;
    result.compressor="BUFF"; result.dataset=bench_data.filename;
    const auto &data=bench_data.double_data;
    const size_t n=data.size();
    result.num_values=n; result.uncompressed_bits=bench_data.uncompressed_bits;
    result.access_layout="native_columns"; result.compression_unit_values=n;
    int64_t scale=1;
    for(int64_t d=0;d<bench_data.decimals;++d) scale*=10;
    using Clock=std::chrono::high_resolution_clock;
    auto elapsed=[](auto start){return std::chrono::duration<double>(Clock::now()-start).count();};
    const double mib=n*sizeof(T)/1024.0/1024.0;
    auto start=Clock::now();
    // One original BUFF column for the complete input. No external 1k wrapper.
    auto column=buff::compress_block(data.data(),n,scale);
    compiler_barrier();
    result.compression_throughput_mbs=mib/elapsed(start);
    result.compressed_bits=column.data.size()*8;
    result.compression_ratio=double(result.compressed_bits)/result.uncompressed_bits;
    std::vector<T> decoded(n);
    start=Clock::now();
    buff::decompress_block(column,decoded.data());
    do_not_optimize(decoded); compiler_barrier();
    result.decompression_throughput_mbs=mib/elapsed(start);
    // Shared 1e-9 policy. BUFF truncates to 2^-dec_len fractional bits, so most
    // values differ by up to 0.5/10^decimals and the row reports failed (warning only).
    check_roundtrip(result,data,decoded);
    finish_roundtrip_validation(result);
    size_t bound_failures=0;
    const double tolerance=0.5/static_cast<double>(scale);
    for(size_t i=0;i<n;++i) bound_failures+=std::abs(data[i]-decoded[i])>tolerance;
    if(bound_failures)
        std::cerr << "Warning: BUFF: " << bound_failures << "/" << n
                  << " values exceed BUFF's precision bound 0.5/10^decimals.\n";
    const size_t nq=std::min<size_t>(10000,bench_data.random_indices.size());
    volatile T sum=0;
    start=Clock::now();
    for(size_t q=0;q<nq;++q) sum+=buff::get_value_at(column,bench_data.random_indices[q]);
    do_not_optimize(sum); compiler_barrier();
    const double point_seconds=elapsed(start);
    result.random_access_ns=point_seconds*1e9/nq;
    result.random_access_mbs=nq*sizeof(T)/1024.0/1024.0/point_seconds;
    size_t query_mismatches=0;
    for(size_t q=0;q<std::min<size_t>(nq,200);++q){
        const auto i=bench_data.random_indices[q];T value=buff::get_value_at(column,i);
        query_mismatches+=roundtrip_value_mismatch(decoded[i],value);
    }
    for(size_t range:range_sizes){
        if(range>=n)continue;
        const auto &indices=bench_data.range_query_indices.at(range);
        const size_t queries=std::min<size_t>(1000,indices.size());
        std::vector<T> output(range);
        start=Clock::now();
        for(size_t q=0;q<queries;++q){
            buff::get_range(column,indices[q],range,output.data());do_not_optimize(output);
        }
        compiler_barrier();
        result.range_query_throughputs.emplace_back(range,queries*range*sizeof(T)/1024.0/1024.0/elapsed(start));
        for(size_t q=0;q<std::min<size_t>(queries,3);++q){
            buff::get_range(column,indices[q],range,output.data());
            for(size_t k=0;k<range;++k) query_mismatches+=roundtrip_value_mismatch(decoded[indices[q]+k],output[k]);
        }
    }
    // Point/range reads use the same 1e-9 rule against the full decode (which is
    // itself checked against the input above), so this isolates the unpatched
    // upstream selective-decoder bug (docs/buff-leco-native.txt). Warning only.
    if(query_mismatches){result.validation_status="failed";
        std::cerr << "Warning: BUFF selective decode: " << query_mismatches
                  << " point/range values differ from full decode by > 1e-9; metrics retained.\n";}
    return result;
}
#endif // NEATS_ENABLE_BUFF

// LeCo benchmark is in leco_benchmark.cpp (compiled with GCC 11)
// Declaration is in benchmark_common.hpp

// ============================================================================
// Squash-based Compressor Benchmark (LZ4, ZSTD, Brotli, XZ, Snappy)
// Only available when compiled with -DUSE_SQUASH
// ============================================================================

#if HAS_SQUASH
template<typename T = int64_t>
BenchmarkResult benchmark_squash(const std::string &compressor_name,
                                 const BenchmarkData &bench_data,
                                 const std::vector<size_t> &range_sizes,
                                 size_t block_size = 1000,
                                 int64_t level = -1) {
    BenchmarkResult result;
    result.compressor = compressor_name;
    result.dataset = bench_data.filename;
    // Initialize num_values to 0 to indicate invalid/incomplete result by default
    result.num_values = 0;
    
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
        snprintf(level_s, 4, "%ld", level);
        squash_options_parse_option(opts, "level", level_s);
    }
    
    // Use raw data
    const auto& data = bench_data.raw_data;
    
    const size_t n = data.size();
    const size_t num_blocks = n / block_size + (n % block_size != 0);
    
    result.num_values = n;
    result.uncompressed_bits = bench_data.uncompressed_bits;
    
    // Compression
    size_t total_compressed_bits = 0;
    std::vector<std::pair<uint8_t*, size_t>> compressed_blocks(num_blocks);
    
    auto t1 = std::chrono::high_resolution_clock::now();
    for (size_t ib = 0; ib < num_blocks; ++ib) {
        const size_t bs = std::min(block_size, n - ib * block_size);
        auto start_it = data.begin() + ib * block_size;
        auto end_it = start_it + bs;
        
        std::vector<uint8_t> data_bytes;
        for (auto it = start_it; it != end_it; ++it) {
            auto bytes = to_bytes<T>(*it);
            data_bytes.insert(data_bytes.end(), bytes.begin(), bytes.end());
        }
        
        size_t uncompressed_size = data_bytes.size();
        size_t compressed_size = squash_codec_get_max_compressed_size(codec, uncompressed_size);
        auto *compressed_data = (uint8_t*)malloc(compressed_size);
        
        const SquashStatus status = squash_codec_compress_with_options(codec, &compressed_size, compressed_data,
                                     uncompressed_size, data_bytes.data(), opts);
        if (status != SQUASH_OK) {
            free(compressed_data);
            throw std::runtime_error(compressor_name + " compression failed: " + squash_status_to_string(status));
        }
        
        compressed_blocks[ib] = {compressed_data, compressed_size};
        total_compressed_bits += compressed_size * CHAR_BIT;
    }
    compiler_barrier();
    auto t2 = std::chrono::high_resolution_clock::now();
    auto compression_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    
    result.compressed_bits = total_compressed_bits;
    result.compression_ratio = static_cast<double>(result.compressed_bits) / result.uncompressed_bits;
    result.compression_throughput_mbs = (n * sizeof(T) / 1024.0 / 1024.0) / (compression_time_ns / 1e9);
    
    // Full decompression
    std::vector<T> decompressed(n);
    volatile T decomp_checksum = 0;
    t1 = std::chrono::high_resolution_clock::now();
    for (size_t ib = 0; ib < num_blocks; ++ib) {
        const size_t bs = std::min(block_size, n - ib * block_size);
        size_t decompressed_size = bs * sizeof(T) + 1;
        auto *decompressed_bytes = (uint8_t*)malloc(decompressed_size);
        
        const SquashStatus status = squash_codec_decompress_with_options(codec, &decompressed_size, decompressed_bytes,
                          compressed_blocks[ib].second, compressed_blocks[ib].first, opts);
        if (status != SQUASH_OK || decompressed_size != bs * sizeof(T)) {
            free(decompressed_bytes);
            throw std::runtime_error(compressor_name + " decompression failed: " + squash_status_to_string(status));
        }
        
        for (size_t i = 0; i < bs; ++i) {
            T value = 0;
            for (size_t j = 0; j < sizeof(T); ++j) {
                value = (value << 8) + decompressed_bytes[i * sizeof(T) + j];
            }
            decompressed[ib * block_size + i] = value;
        }
        decomp_checksum += decompressed[ib * block_size];
        free(decompressed_bytes);
    }
    compiler_barrier();
    t2 = std::chrono::high_resolution_clock::now();
    do_not_optimize(decompressed);
    do_not_optimize(decomp_checksum);
    auto decompression_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    
    result.decompression_throughput_mbs = (n * sizeof(T) / 1024.0 / 1024.0) / (decompression_time_ns / 1e9);
    
    check_roundtrip(result, data, decompressed);
    finish_roundtrip_validation(result);
    result.access_layout = "independent_blocks";
    result.compression_unit_values = block_size;
    
    // Random access
    const size_t num_ra_queries = 100000;
    
    t1 = std::chrono::high_resolution_clock::now();
    volatile T ra_sum = 0;
    size_t q_count = 0;
    for (auto idx : bench_data.random_indices) {
        if (q_count++ >= num_ra_queries) break;
        
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
        ra_sum += value;
        free(decompressed_bytes);
    }
    compiler_barrier();
    t2 = std::chrono::high_resolution_clock::now();
    do_not_optimize(ra_sum);
    auto ra_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.random_access_ns = static_cast<double>(ra_time_ns) / num_ra_queries;
    result.random_access_mbs = (num_ra_queries * sizeof(T) / 1024.0 / 1024.0) / (ra_time_ns / 1e9);
    
    // Range queries
    const size_t num_range_queries = 1000;
    for (auto range : range_sizes) {
        if (range >= n) continue;
        
        const auto& range_indices = bench_data.range_query_indices.at(range);
        std::vector<T> out_buffer(range);
        volatile T range_checksum = 0;
        
        t1 = std::chrono::high_resolution_clock::now();
        size_t q_count_range = 0;
        for (auto start_idx : range_indices) {
            if (q_count_range++ >= num_range_queries) break;
            
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
    std::cerr << "                 Available: neats,dac,rle_gef,u_gef_approximate,u_gef_optimal,b_gef_approximate,b_gef_optimal,b_star_gef_approximate,b_star_gef_optimal,gorilla,chimp,chimp128,tsxor,elf,camel,alp";
#if defined(NEATS_ENABLE_LECO)
    std::cerr << ",leco";
#endif
#if defined(NEATS_ENABLE_BUFF)
    std::cerr << ",buff";
#endif
    std::cerr << ",lz4,zstd,brotli,xz,snappy" << std::endl;
#else
    std::cerr << "                 Available: neats,dac,rle_gef,u_gef_approximate,u_gef_optimal,b_gef_approximate,b_gef_optimal,b_star_gef_approximate,b_star_gef_optimal,gorilla,chimp,chimp128,tsxor,elf,camel,alp";
#if defined(NEATS_ENABLE_BUFF)
    std::cerr << ",buff";
#endif
    std::cerr << std::endl;
    std::cerr << "                 (Compile with -DUSE_SQUASH for lz4,zstd,brotli,xz,snappy";
#if defined(NEATS_ENABLE_LECO)
    std::cerr << ",leco";
#endif
    std::cerr << ")" << std::endl;
#endif
    std::cerr << "                 Additional: ";
    for (const auto& name : recent_compressor_names()) std::cerr << name << " ";
    std::cerr << std::endl;
    std::cerr << "  -r <list>      Comma-separated list of range sizes (default: 10,100,1000,10000,100000)" << std::endl;
    std::cerr << "  -b <size>      Block size for block-based compressors (default: 1000)" << std::endl;
    std::cerr << "                 (Note: *_gef compressors use fixed UniformedPartitioner block size = " << GEF_UNIFORM_PARTITION_SIZE << ")" << std::endl;
    std::cerr << "  -m <bpc>       Max bits per correction for NeaTS (default: 32)" << std::endl;
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
    std::vector<std::string> compressors = {"neats", "dac", "rle_gef", "u_gef_approximate", "u_gef_optimal",
                                            "b_gef_approximate", "b_gef_optimal", "b_star_gef_approximate", "b_star_gef_optimal",
                                            "gorilla", "chimp", "chimp128", "tsxor",
                                            "elf", "camel", "alp",
#if defined(NEATS_ENABLE_LECO)
                                            "leco",
#endif
#if defined(NEATS_ENABLE_BUFF)
                                            "buff",
#endif
                                            "lz4", "zstd", "brotli", "xz", "snappy"};
#else
    std::vector<std::string> compressors = {"neats", "dac", "rle_gef", "u_gef_approximate", "u_gef_optimal",
                                            "b_gef_approximate", "b_gef_optimal", "b_star_gef_approximate", "b_star_gef_optimal",
                                            "gorilla", "chimp", "chimp128", "tsxor",
                                            "elf", "camel", "alp",
#if defined(NEATS_ENABLE_BUFF)
                                            "buff",
#endif
                                            };
#endif
    const auto recent = recent_compressor_names();
    compressors.insert(compressors.end(), recent.begin(), recent.end());
    std::vector<size_t> range_sizes = {10, 100, 1000, 10000, 100000};
    size_t block_size = 1000;
    NativeAccessOptions native_options;
    uint8_t max_bpc = 32;
    std::string input_path;
    size_t failures = 0;
    
    // Parse command line arguments
    for (int64_t i = 1; i < argc; ++i) {
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
        
        // Prepare data ONCE to avoid reloading and reprocessing for every compressor
        BenchmarkData bench_data;
        bench_data.filename = filename;
        
        try {
            auto loaded = load_custom_dataset(filename);
            bench_data.decimals = loaded.decimals;
            bench_data.uncompressed_bits = loaded.data.size() * sizeof(int64_t) * 8;
            
            size_t n = loaded.data.size();
            auto min_data = *std::min_element(loaded.data.begin(), loaded.data.end());
            bench_data.min_val = min_data < 0 ? (min_data - 1) : -1;
            
            // Move raw_data to bench_data (SQUASH compressors need this)
            bench_data.raw_data = std::move(loaded.data);
            
            // Prepare indices
            bench_data.random_indices = generate_random_indices(n, 1000000);
            
            // Generate range queries
            for(auto range : range_sizes) {
                 if (range < n) {
                     bench_data.range_query_indices[range] = generate_range_indices(n, range, 10000);
                 } else {
                     bench_data.range_query_indices[range] = {};
                 }
            }
        } catch (const std::exception &e) {
            std::cerr << "Error loading dataset " << filename << ": " << e.what() << std::endl;
            continue;
        }
        
        // Separate compressors by data type needed
        std::vector<std::string> raw_data_compressors;     // SQUASH: lz4, zstd, brotli, xz, snappy
        std::vector<std::string> shifted_data_compressors; // neats, dac, *_gef, leco
        std::vector<std::string> double_data_compressors;  // gorilla, chimp, etc.
        
        for (const auto &comp : compressors) {
            if (recent_compressor_uses_integers(comp)) {
                raw_data_compressors.push_back(comp);
                continue;
            }
#if HAS_SQUASH
            if (comp == "lz4" || comp == "zstd" || comp == "brotli" || 
                comp == "xz" || comp == "snappy") {
                raw_data_compressors.push_back(comp);
            } else
#endif
            if (comp == "neats" || comp == "dac" || comp == "rle_gef"
#if defined(NEATS_ENABLE_LECO)
                || comp == "leco"
#endif
                ||
                comp.find("_gef") != std::string::npos) {
                shifted_data_compressors.push_back(comp);
            } else if (comp == "alp") {
                double_data_compressors.push_back(comp);
            } else {
                double_data_compressors.push_back(comp);
            }
        }
        
        // Phase 0: Run compressors that need exact unshifted integers.
        for (const auto &comp : raw_data_compressors) {
            std::cerr << "  Running " << comp << "..." << std::flush;
            
            BenchmarkResult result;
            
            try {
                if (recent_compressor_uses_integers(comp)) {
                    result = benchmark_recent_compressor(comp, bench_data, range_sizes, block_size, native_options);
                }
#if HAS_SQUASH
                else result = benchmark_squash<int64_t>(comp, bench_data, range_sizes, block_size);
#endif
                
                if (!header_printed) {
                    result.print_header(*out);
                    header_printed = true;
                }
                result.print(*out);
                std::cerr << " done" << std::endl;
                
            } catch (const std::exception &e) {
                ++failures;
                std::cerr << " error: " << e.what() << std::endl;
            }
        }

        // Convert raw_data to shifted_data and free raw_data
        if (!shifted_data_compressors.empty() || !double_data_compressors.empty()) {
            if (!bench_data.raw_data.empty()) {
                std::cerr << "  Converting raw_data to shifted_data..." << std::flush;
                size_t n = bench_data.raw_data.size();
                bench_data.shifted_data.resize(n);
                std::transform(bench_data.raw_data.begin(), bench_data.raw_data.end(), 
                               bench_data.shifted_data.begin(),
                               [&bench_data](int64_t d) { return d - bench_data.min_val; });
                // Free raw_data
                std::vector<int64_t>().swap(bench_data.raw_data);
                std::cerr << " done" << std::endl;
            }
        }
        
        // Phase 1: Run compressors that need shifted_data
        for (const auto &comp : shifted_data_compressors) {
            std::cerr << "  Running " << comp << "..." << std::flush;
            
            BenchmarkResult result;
            
            try {
                if (comp == "neats") {
                    result = benchmark_neats<int64_t>(bench_data, range_sizes, max_bpc);
                } else if (comp == "dac") {
                    result = benchmark_dac<int64_t>(bench_data, range_sizes);
                }
                else if (is_gef_compressor(comp)) {
                    result = benchmark_gef_compressor(comp, bench_data, range_sizes);
                }
#if defined(NEATS_ENABLE_LECO)
                else if (comp == "leco") {
                    result = benchmark_leco(bench_data, range_sizes);
                }
#endif
                else {
                    ++failures;
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
                ++failures;
                std::cerr << " error: " << e.what() << std::endl;
            }
        }
        
        // Phase 2: Convert shifted_data to double_data if needed
        if (!double_data_compressors.empty() && !bench_data.shifted_data.empty()) {
            std::cerr << "  Converting shifted_data to double_data..." << std::flush;
            bench_data.convert_shifted_to_double();
            std::cerr << " done" << std::endl;
        }
        
        // Phase 3: Run compressors that need double_data
        for (const auto &comp : double_data_compressors) {
            std::cerr << "  Running " << comp << "..." << std::flush;
            
            BenchmarkResult result;
            
            try {
                if (is_recent_compressor(comp)) {
                    result = benchmark_recent_compressor(comp, bench_data, range_sizes, block_size, native_options);
                } else if (comp == "gorilla") {
                    result = benchmark_bitstream_compressor<CompressorGorilla<double>, DecompressorGorilla<double>, double>(
                        "Gorilla", bench_data, range_sizes, block_size);
                } else if (comp == "chimp") {
                    result = benchmark_bitstream_compressor<CompressorChimp<double>, DecompressorChimp<double>, double>(
                        "Chimp", bench_data, range_sizes, block_size);
                } else if (comp == "chimp128") {
                    result = benchmark_bitstream_compressor<CompressorChimp128<double>, DecompressorChimp128<double>, double>(
                        "Chimp128", bench_data, range_sizes, block_size);
                } else if (comp == "tsxor") {
                    result = benchmark_tsxor<double>(bench_data, range_sizes, block_size);
                } else if (comp == "elf") {
                    result = benchmark_bitstream_compressor<CompressorElf<double>, DecompressorElf<double>, double>(
                        "Elf", bench_data, range_sizes, block_size);
                } else if (comp == "camel") {
                    result = benchmark_camel(bench_data, range_sizes, block_size);
                } else if (comp == "alp") {
                    result = benchmark_alp(bench_data, range_sizes);
#ifdef NEATS_ENABLE_BUFF
                } else if (comp == "buff") {
                    result = benchmark_buff<double>(bench_data, range_sizes);
#endif
                } else {
                    ++failures;
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
                ++failures;
                std::cerr << " error: " << e.what() << std::endl;
            }
        }
    }
    
    if (file_out.is_open()) {
        file_out.close();
    }
    
    return failures ? 1 : 0;
}
