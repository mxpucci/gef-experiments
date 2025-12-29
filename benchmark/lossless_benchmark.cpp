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
    double random_access_mbs;
    std::vector<std::pair<size_t, double>> range_query_throughputs; // (range_size, MB/s)
    
    void print_header(std::ostream &out) const {
        out << "compressor,dataset,num_values,uncompressed_bits,compressed_bits,"
            << "compression_ratio,compression_throughput_mbs,decompression_throughput_mbs,"
            << "random_access_ns,random_access_mbs";
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
            << decompression_throughput_mbs << "," << random_access_ns << "," << random_access_mbs;
        for (const auto &[_, throughput] : range_query_throughputs) {
            out << "," << throughput;
        }
        out << std::endl;
    }
};

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

// Helper template to run NeaTS benchmark with specific coefficient precision
template<typename T, typename T1_coeff>
BenchmarkResult benchmark_neats_impl(const std::vector<T> &processed_data,
                                      const std::vector<size_t> &range_sizes,
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
    auto t2 = std::chrono::high_resolution_clock::now();
    auto compression_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    
    result.compressed_bits = compressor.size_in_bits();
    result.compression_ratio = static_cast<double>(result.compressed_bits) / result.uncompressed_bits;
    result.compression_throughput_mbs = (processed_data.size() * sizeof(T) / 1024.0 / 1024.0) / (compression_time_ns / 1e9);
    
    // Full decompression
    std::vector<T> decompressed(processed_data.size());
    t1 = std::chrono::high_resolution_clock::now();
    compressor.simd_decompress(decompressed.data());
    t2 = std::chrono::high_resolution_clock::now();
    auto decompression_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    do_not_optimize(decompressed);
    
    result.decompression_throughput_mbs = (processed_data.size() * sizeof(T) / 1024.0 / 1024.0) / (decompression_time_ns / 1e9);
    
    // Verify decompression
    for (size_t i = 0; i < processed_data.size(); ++i) {
        if (processed_data[i] != decompressed[i]) {
            std::cerr << "NeaTS decompression error at " << i << std::endl;
            break;
        }
    }
    
    // Random access
    const size_t num_ra_queries = 1000000;
    auto indices = generate_random_indices(processed_data.size(), num_ra_queries);
    
    t1 = std::chrono::high_resolution_clock::now();
    T sum = 0;
    for (auto idx : indices) {
        sum += compressor[idx];
    }
    t2 = std::chrono::high_resolution_clock::now();
    do_not_optimize(sum);
    auto ra_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.random_access_ns = static_cast<double>(ra_time_ns) / num_ra_queries;
    result.random_access_mbs = (num_ra_queries * sizeof(T) / 1024.0 / 1024.0) / (ra_time_ns / 1e9);
    
    // Range queries
    const size_t num_range_queries = 10000;
    for (auto range : range_sizes) {
        if (range >= processed_data.size()) continue;
        
        auto range_indices = generate_range_indices(processed_data.size(), range, num_range_queries);
        std::vector<T> out_buffer(range);
        
        t1 = std::chrono::high_resolution_clock::now();
        for (auto start_idx : range_indices) {
            compressor.simd_scan(start_idx, start_idx + range, out_buffer.data());
            do_not_optimize(out_buffer);
        }
        t2 = std::chrono::high_resolution_clock::now();
        
        auto range_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
        double throughput = ((range * sizeof(T)) * num_range_queries / 1024.0 / 1024.0) / (range_time_ns / 1e9);
        result.range_query_throughputs.emplace_back(range, throughput);
    }
    
    return result;
}

template<typename T = int64_t>
BenchmarkResult benchmark_neats(const std::string &filename, 
                                const std::vector<size_t> &range_sizes,
                                uint8_t max_bpc = 32) {
    // Load data
    auto loaded = load_custom_dataset(filename);
    auto& data = loaded.data;
    
    auto min_data = *std::min_element(data.begin(), data.end());
    min_data = min_data < 0 ? (min_data - 1) : -1;
    
    std::vector<T> processed_data(data.size());
    std::transform(data.begin(), data.end(), processed_data.begin(),
                   [min_data](int64_t d) { return static_cast<T>(d - min_data); });
    
    // Determine the maximum absolute value to decide coefficient precision
    // A 32-bit float has ~7 significant decimal digits of precision
    // Use double when max value exceeds 10^9 (1 billion) to avoid precision loss
    // in slope/intercept calculations with very large values
    constexpr T FLOAT_PRECISION_THRESHOLD = static_cast<T>(1000000000); // 10^9 (1 billion)
    
    T max_val = *std::max_element(processed_data.begin(), processed_data.end());
    size_t uncompressed_bits = data.size() * sizeof(T) * 8;
    
    if (max_val > FLOAT_PRECISION_THRESHOLD) {
        // Use double precision for coefficients when values are large
        return benchmark_neats_impl<T, double>(processed_data, range_sizes, max_bpc, 
                                                filename, uncompressed_bits);
    } else {
        // Use float precision for coefficients when values are small (better compression)
        return benchmark_neats_impl<T, float>(processed_data, range_sizes, max_bpc,
                                               filename, uncompressed_bits);
    }
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
    auto loaded = load_custom_dataset(filename);
    auto& data = loaded.data;
    // DAC works on the integers directly
    
    result.num_values = data.size();
    result.uncompressed_bits = data.size() * sizeof(T) * 8;
    
    auto min_data = *std::min_element(data.begin(), data.end());
    min_data = min_data < 0 ? (min_data - 1) : -1;
    
    std::vector<uint64_t> u_data(data.size());
    std::transform(data.begin(), data.end(), u_data.begin(),
                   [min_data](int64_t x) { return static_cast<uint64_t>(x - min_data); });
    
    // Compression
    auto t1 = std::chrono::high_resolution_clock::now();
    sdsl::dac_vector_dp<> dac_vector(u_data);
    auto t2 = std::chrono::high_resolution_clock::now();
    do_not_optimize(dac_vector);
    auto compression_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    
    result.compressed_bits = sdsl::size_in_bytes(dac_vector) * 8;
    result.compression_ratio = static_cast<double>(result.compressed_bits) / result.uncompressed_bits;
    result.compression_throughput_mbs = (data.size() * sizeof(T) / 1024.0 / 1024.0) / (compression_time_ns / 1e9);
    
    // Full decompression
    std::vector<uint64_t> decompressed(data.size());
    t1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < data.size(); ++i) {
        decompressed[i] = dac_vector[i];
    }
    t2 = std::chrono::high_resolution_clock::now();
    do_not_optimize(decompressed);
    auto decompression_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    
    result.decompression_throughput_mbs = (data.size() * sizeof(T) / 1024.0 / 1024.0) / (decompression_time_ns / 1e9);
    
    // Verify decompression
    for (size_t i = 0; i < data.size(); ++i) {
        if (u_data[i] != decompressed[i]) {
            std::cerr << "DAC decompression error at " << i << std::endl;
            break;
        }
    }
    
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
    result.random_access_mbs = (num_ra_queries * sizeof(T) / 1024.0 / 1024.0) / (ra_time_ns / 1e9);
    
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
                                                const std::string &filename,
                                                const std::vector<size_t> &range_sizes,
                                                size_t block_size = 1000) {
    BenchmarkResult result;
    result.compressor = compressor_name;
    result.dataset = filename;
    
    // Load data
    auto loaded = load_custom_dataset(filename);
    const auto& raw_data = loaded.data;
    double divisor = std::pow(10.0, loaded.decimals);
    
    std::vector<T> data(raw_data.size());
    for(size_t i=0; i<raw_data.size(); ++i) {
        data[i] = static_cast<T>(raw_data[i]) / divisor;
    }
    
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
        
        std::unique_ptr<Compressor> cmpr;
        if constexpr (std::is_same_v<Compressor, CompressorCamel<T>>) {
            cmpr = std::make_unique<Compressor>(*data_block.begin(), static_cast<int64_t>(loaded.decimals));
        } else {
            cmpr = std::make_unique<Compressor>(*data_block.begin());
        }

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
    result.compression_throughput_mbs = (n * sizeof(T) / 1024.0 / 1024.0) / (compression_time_ns / 1e9);
    
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
    
    result.decompression_throughput_mbs = (n * sizeof(T) / 1024.0 / 1024.0) / (decompression_time_ns / 1e9);
    
    // Verify decompression
    for (size_t i = 0; i < n; ++i) {
        bool match = (std::abs(data[i] - decompressed[i]) < 1e-9);

        if (!match) {
            std::cerr << compressor_name << " decompression error at " << i 
                      << " Expected: " << std::setprecision(20) << data[i] 
                      << " Got: " << decompressed[i] 
                      << " Diff: " << std::abs(data[i] - decompressed[i]) << std::endl;
            break;
        }
    }
    
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
    result.random_access_mbs = (num_ra_queries * sizeof(T) / 1024.0 / 1024.0) / (ra_time_ns / 1e9);
    
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
        double throughput = ((range * sizeof(T)) * num_range_queries / 1024.0 / 1024.0) / (range_time_ns / 1e9);
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
    auto loaded = load_custom_dataset(filename);
    const auto& raw_data = loaded.data;
    double divisor = std::pow(10.0, loaded.decimals);
    
    std::vector<T> data(raw_data.size());
    for(size_t i=0; i<raw_data.size(); ++i) {
        data[i] = static_cast<T>(raw_data[i]) / divisor;
    }
    
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
    result.compression_throughput_mbs = (n * sizeof(T) / 1024.0 / 1024.0) / (compression_time_ns / 1e9);
    
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
    
    result.decompression_throughput_mbs = (n * sizeof(T) / 1024.0 / 1024.0) / (decompression_time_ns / 1e9);
    
    // Verify decompression
    for (size_t i = 0; i < n; ++i) {
        if (std::abs(data[i] - decompressed[i]) >= 1e-9) {
            std::cerr << "TSXor decompression error at " << i << std::endl;
            break;
        }
    }
    
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
    result.random_access_mbs = (num_ra_queries * sizeof(T) / 1024.0 / 1024.0) / (ra_time_ns / 1e9);
    
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
        double throughput = ((range * sizeof(T)) * num_range_queries / 1024.0 / 1024.0) / (range_time_ns / 1e9);
        result.range_query_throughputs.emplace_back(range, throughput);
    }
    
    return result;
}

// ============================================================================
// Falcon Benchmark (uses byte vector buffer)
// ============================================================================
template<typename T = double>
BenchmarkResult benchmark_falcon(const std::string &filename,
                                 const std::vector<size_t> &range_sizes,
                                 size_t block_size = 1000) { // Default to 1000 to match your proposal
    BenchmarkResult result;
    result.compressor = "Falcon (Block-Based)";
    result.dataset = filename;
    
    // -------------------------------------------------------------------------
    // 1. Load Data
    // -------------------------------------------------------------------------
    auto loaded = load_custom_dataset(filename);
    const auto& raw_data = loaded.data;
    double divisor = std::pow(10.0, loaded.decimals);
    
    std::vector<T> data(raw_data.size());
    for(size_t i=0; i<raw_data.size(); ++i) {
        data[i] = static_cast<T>(raw_data[i]) / divisor;
    }
    
    result.num_values = data.size();
    result.uncompressed_bits = data.size() * sizeof(T) * 8;
    
    const size_t n = data.size();
    const size_t num_blocks = (n + block_size - 1) / block_size;
    
    // -------------------------------------------------------------------------
    // 2. Compression (Independent Blocks)
    // -------------------------------------------------------------------------
    size_t total_compressed_bits = 0;
    std::vector<std::vector<uint8_t>> compressed_blocks(num_blocks);
    
    auto t1 = std::chrono::high_resolution_clock::now();
    
    for (size_t ib = 0; ib < num_blocks; ++ib) {
        // Identify range for this block
        const size_t start_idx = ib * block_size;
        const size_t end_idx = std::min(start_idx + block_size, n);
        const size_t current_bs = end_idx - start_idx;
        
        // Initialize Compressor for JUST this block
        // Note: We pass current_bs so Falcon writes the correct header count
        CompressorFalcon<T> cmpr(data[start_idx], current_bs);
        
        for (size_t k = 1; k < current_bs; ++k) {
            cmpr.addValue(data[start_idx + k]);
        }
        cmpr.close();
        
        // Store results
        total_compressed_bits += cmpr.getSize();
        compressed_blocks[ib] = cmpr.getOut(); 
    }
    
    auto t2 = std::chrono::high_resolution_clock::now();
    auto compression_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    
    // -------------------------------------------------------------------------
    // Memory Estimation Correction
    // -------------------------------------------------------------------------
    // To fairly compare "RAM Usage" of the format, we must account for the Index.
    // We generated 'num_blocks' blocks. A real system needs an offset table (uint64_t)
    // to find where block 'i' lives.
    size_t index_overhead_bits = num_blocks * 64; 
    
    result.compressed_bits = total_compressed_bits + index_overhead_bits;
    result.compression_ratio = static_cast<double>(result.compressed_bits) / result.uncompressed_bits;
    result.compression_throughput_mbs = (n * sizeof(T) / 1024.0 / 1024.0) / (compression_time_ns / 1e9);
    
    // -------------------------------------------------------------------------
    // 3. Full Decompression (Sanity Check & Throughput)
    // -------------------------------------------------------------------------
    std::vector<T> decompressed(n);
    t1 = std::chrono::high_resolution_clock::now();
    
    size_t output_offset = 0;
    for (size_t ib = 0; ib < num_blocks; ++ib) {
        size_t expected_count = std::min(block_size, n - ib * block_size);
        
        // Decompress independent block
        DecompressorFalcon<T> dcmpr(compressed_blocks[ib], expected_count);
        
        // Falcon Decompressor initializes storedValue with the first item immediately
        decompressed[output_offset++] = dcmpr.storedValue;
        
        while (dcmpr.hasNext()) {
            decompressed[output_offset++] = dcmpr.storedValue;
        }
    }
    
    t2 = std::chrono::high_resolution_clock::now();
    do_not_optimize(decompressed);
    auto decompression_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.decompression_throughput_mbs = (n * sizeof(T) / 1024.0 / 1024.0) / (decompression_time_ns / 1e9);
    
    // Verification
    for (size_t i = 0; i < n; ++i) {
        if (std::abs(data[i] - decompressed[i]) >= 1e-9) {
            std::cerr << "Falcon (Chunked) decompression error at index " << i << std::endl;
            break;
        }
    }

    // -------------------------------------------------------------------------
    // 4. Random Access (Decompress ONLY the target block)
    // -------------------------------------------------------------------------
    const size_t num_ra_queries = 10000;
    auto indices = generate_random_indices(n, num_ra_queries);
    
    t1 = std::chrono::high_resolution_clock::now();
    T sum = 0;
    
    for (auto idx : indices) {
        size_t ib = idx / block_size;
        size_t offset_in_block = idx % block_size;
        size_t expected_count = std::min(block_size, n - ib * block_size);
        
        // Instantiate decompressor only for the required block
        DecompressorFalcon<T> dcmpr(compressed_blocks[ib], expected_count);
        
        // Scan to the specific offset
        size_t current_pos = 0;
        if (offset_in_block == 0) {
            sum += dcmpr.storedValue;
        } else {
            // hasNext() advances to the next value and updates storedValue
            while (current_pos < offset_in_block && dcmpr.hasNext()) {
                current_pos++;
            }
            sum += dcmpr.storedValue;
        }
    }
    
    t2 = std::chrono::high_resolution_clock::now();
    do_not_optimize(sum);
    
    auto ra_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.random_access_ns = static_cast<double>(ra_time_ns) / num_ra_queries;
    result.random_access_mbs = (num_ra_queries * sizeof(T) / 1024.0 / 1024.0) / (ra_time_ns / 1e9);

    // -------------------------------------------------------------------------
    // 5. Range Queries
    // -------------------------------------------------------------------------
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
            
            // We only decompress the blocks overlapping the range
            for (size_t ib = start_block; ib <= end_block; ++ib) {
                size_t expected_count = std::min(block_size, n - ib * block_size);
                DecompressorFalcon<T> dcmpr(compressed_blocks[ib], expected_count);
                
                size_t block_start_global = ib * block_size;
                
                // Determine how many items to skip in this block (only for the first block)
                size_t skip = (ib == start_block) ? (start_idx - block_start_global) : 0;
                
                size_t current_in_block = 0;
                
                // Fast skip
                // Note: storedValue is the 0-th element. 
                // We need to call hasNext 'skip' times to put 'storedValue' at index 'skip'
                while(current_in_block < skip && dcmpr.hasNext()) {
                    current_in_block++;
                }
                
                // Copy
                while (out_pos < range) {
                    out_buffer[out_pos++] = dcmpr.storedValue;
                    current_in_block++;
                    
                    // Boundary check: if we hit end of block or end of range, break
                    if (current_in_block >= expected_count) break;
                    if (!dcmpr.hasNext()) break; 
                }
            }
            do_not_optimize(out_buffer);
        }
        t2 = std::chrono::high_resolution_clock::now();
        
        auto range_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
        double throughput = ((range * sizeof(T)) * num_range_queries / 1024.0 / 1024.0) / (range_time_ns / 1e9);
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
                                 int64_t level = -1) {
    BenchmarkResult result;
    result.compressor = compressor_name;
    result.dataset = filename;
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
    
    // Load data
    auto loaded = load_custom_dataset(filename);
    auto& data = loaded.data;
    // Squash compresses raw bytes, so we can use integers directly or convert if we want
    // Assuming we want to compress the 64-bit values as is.
    
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
    result.compression_throughput_mbs = (n * sizeof(T) / 1024.0 / 1024.0) / (compression_time_ns / 1e9);
    
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
    
    result.decompression_throughput_mbs = (n * sizeof(T) / 1024.0 / 1024.0) / (decompression_time_ns / 1e9);
    
    // Verify decompression
    for (size_t i = 0; i < n; ++i) {
        if (data[i] != decompressed[i]) {
            std::cerr << compressor_name << " decompression error at " << i << std::endl;
            break;
        }
    }
    
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
    result.random_access_mbs = (num_ra_queries * sizeof(T) / 1024.0 / 1024.0) / (ra_time_ns / 1e9);
    
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
    std::cerr << "                 Available: neats,dac,gorilla,chimp,chimp128,tsxor,elf,camel,falcon,lz4,zstd,brotli,xz,snappy" << std::endl;
#else
    std::cerr << "                 Available: neats,dac,gorilla,chimp,chimp128,tsxor,elf,camel,falcon" << std::endl;
    std::cerr << "                 (Compile with -DUSE_SQUASH for lz4,zstd,brotli,xz,snappy)" << std::endl;
#endif
    std::cerr << "  -r <list>      Comma-separated list of range sizes (default: 10,100,1000,10000,100000)" << std::endl;
    std::cerr << "  -b <size>      Block size for block-based compressors (default: 1000)" << std::endl;
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
    std::vector<std::string> compressors = {"neats", "dac", "gorilla", "chimp", "chimp128", "tsxor",
                                            "elf", "camel", "falcon", "lz4", "zstd", "brotli", "snappy"};
#else
    std::vector<std::string> compressors = {"neats", "dac", "gorilla", "chimp", "chimp128", "tsxor",
                                            "elf", "camel", "falcon"};
#endif
    std::vector<size_t> range_sizes = {10, 100, 1000, 10000, 100000};
    size_t block_size = 1000;
    uint8_t max_bpc = 32;
    std::string input_path;
    
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
                
                if (result.num_values == 0 && comp != "falcon" && comp != "neats" && comp != "dac" && 
                    comp != "gorilla" && comp != "chimp" && comp != "chimp128" && 
                    comp != "tsxor" && comp != "elf" && comp != "camel") {
                   // If num_values is 0, it means the benchmark failed (e.g. missing squash codec)
                   // We don't check non-squash compressors because they don't set num_values=0 on failure in the same way 
                   // (they usually throw or crash, or set it correctly if they succeed)
                   // But for Squash, we explicitly set it to 0 on codec failure.
                   std::cerr << " skipping result output due to failure." << std::endl;
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

