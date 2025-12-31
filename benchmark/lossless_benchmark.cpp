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

// LeCo headers
#if defined(USE_SQUASH) && defined(NEATS_ENABLE_LECO)
// LeCo requires Eigen and other dependencies which we assume are available when SQUASH/Full benchmark is built
#include "piecewise_fix_integer_template.h"
#endif

// ALP headers (does not depend on Squash)
//
// NOTE: ALP's headers enable AVX-512-specific paths when __AVX512F__ is defined.
// In our CI toolchain (GCC), those AVX-512 paths currently fail to compile.
// We force ALP to use its scalar fallbacks by undefining AVX-512 macros for this TU.
// This keeps ALP functional/portable, at the cost of disabling AVX-512-accelerated paths.
#ifdef __AVX512F__
#undef __AVX512F__
#endif
#ifdef __AVX512BW__
#undef __AVX512BW__
#endif
#ifdef __AVX512DQ__
#undef __AVX512DQ__
#endif
#include "alp.hpp"

// GEF includes
// NOTE: When running multiple GEF compressors sequentially with SIMD and OpenMP enabled,
// you may encounter a bug where subsequent compressors produce astronomically large 
// compressed sizes (compression ratio > 100x). This manifests on Linux with AVX-512.
//
// WORKAROUND: Use LosslessBenchmarkFullNoSIMD which is compiled with:
//   -DGEF_DISABLE_SIMD=1 -DGEF_DISABLE_OPENMP=1
// These flags force sequential partition construction and scalar gap computation.
#include "gef/UniformedPartitioner.hpp"
#include "gef/RLE_GEF.hpp"
#include "gef/U_GEF.hpp"
#include "gef/B_GEF.hpp"
#include "gef/B_GEF_STAR.hpp"
#include "datastructures/SDSLBitVectorFactory.hpp"

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
    return std::filesystem::path(path).stem().string();
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

struct BenchmarkData {
    std::string filename;
    std::vector<int64_t> raw_data;
    std::vector<int64_t> shifted_data; 
    std::vector<double> double_data;
    int64_t decimals;
    int64_t min_val;
    size_t uncompressed_bits;
    
    // Indices
    std::vector<size_t> random_indices;
    // Map range size to indices
    std::map<size_t, std::vector<size_t>> range_query_indices;
    
    // Convert shifted_data to double_data and free shifted_data
    void convert_shifted_to_double() {
        if (shifted_data.empty()) return;
        
        int64_t divisor_int = 1;
        for (int64_t d = 0; d < decimals; ++d) divisor_int *= 10;
        double divisor = static_cast<double>(divisor_int);
        
        double_data.resize(shifted_data.size());
        for (size_t i = 0; i < shifted_data.size(); ++i) {
            // raw_data[i] = shifted_data[i] + min_val
            // double_data[i] = raw_data[i] / divisor
            double_data[i] = static_cast<double>(shifted_data[i] + min_val) / divisor;
        }
        
        // Free shifted_data
        shifted_data.clear();
        shifted_data.shrink_to_fit();
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
    
    for (size_t i = 0; i < processed_data.size(); ++i) {
        if (processed_data[i] != decompressed[i]) {
            std::cerr << "NeaTS decompression error at " << i << std::endl;
            break;
        }
    }
    
    // Random access
    const size_t num_ra_queries = random_indices.size();
    
    t1 = std::chrono::high_resolution_clock::now();
    T sum = 0;
    for (auto idx : random_indices) {
        sum += compressor[idx];
    }
    t2 = std::chrono::high_resolution_clock::now();
    do_not_optimize(sum);
    auto ra_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.random_access_ns = static_cast<double>(ra_time_ns) / num_ra_queries;
    result.random_access_mbs = (num_ra_queries * sizeof(T) / 1024.0 / 1024.0) / (ra_time_ns / 1e9);
    
    // Range queries
    const size_t num_range_queries = range_query_indices.begin()->second.size();
    for (auto range : range_sizes) {
        if (range >= processed_data.size()) continue;
        
        const auto& range_indices = range_query_indices.at(range);
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
// GEF Wrapper and Benchmark
// ============================================================================

static constexpr size_t GEF_UNIFORM_PARTITION_SIZE = 32000;

template<typename T>
struct RLE_GEF_Wrapper : public gef::RLE_GEF<T> {
    template<typename C>
    RLE_GEF_Wrapper(const C& data, std::shared_ptr<IBitVectorFactory> factory)
        : gef::RLE_GEF<T>(factory, data) {}
    
    RLE_GEF_Wrapper() : gef::RLE_GEF<T>() {}
    // RLE_GEF supports copy, so defaults are fine
};

template<typename T, gef::SplitPointStrategy Strategy>
struct U_GEF_Wrapper : public gef::U_GEF<T> {
    template<typename C>
    U_GEF_Wrapper(const C& data, std::shared_ptr<IBitVectorFactory> factory)
        : gef::U_GEF<T>(factory, data, Strategy) {}
    
    U_GEF_Wrapper() : gef::U_GEF<T>() {}
};

template<typename T, gef::SplitPointStrategy Strategy>
struct B_GEF_Wrapper : public gef::B_GEF<T> {
    template<typename C>
    B_GEF_Wrapper(const C& data, std::shared_ptr<IBitVectorFactory> factory)
        : gef::B_GEF<T>(factory, data, Strategy) {}
    
    B_GEF_Wrapper() : gef::B_GEF<T>() {}
};

template<typename T, gef::SplitPointStrategy Strategy>
struct B_GEF_STAR_Wrapper : public gef::B_GEF_STAR<T> {
    template<typename C>
    B_GEF_STAR_Wrapper(const C& data, std::shared_ptr<IBitVectorFactory> factory)
        : gef::B_GEF_STAR<T>(factory, data, Strategy) {}
    
    B_GEF_STAR_Wrapper() : gef::B_GEF_STAR<T>() {}
};

template<typename GEFType, typename T = int64_t>
BenchmarkResult benchmark_gef(const std::string &compressor_name,
                              const BenchmarkData &bench_data,
                              const std::vector<size_t> &range_sizes,
                              size_t block_size = 1000) {
    BenchmarkResult result;
    result.compressor = compressor_name;
    result.dataset = bench_data.filename;
    
    // Use shifted data
    const auto& data = bench_data.shifted_data;
                   
    result.num_values = data.size();
    result.uncompressed_bits = bench_data.uncompressed_bits;
    
    auto factory = std::make_shared<SDSLBitVectorFactory>();
    
    // Compression
    auto t1 = std::chrono::high_resolution_clock::now();
    // UniformedPartitioner(data, k, args...) -> Wrapper(view, factory)
    GEFType compressor(data, block_size, factory);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto compression_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    
    result.compressed_bits = compressor.size_in_bytes() * 8;
    result.compression_ratio = static_cast<double>(result.compressed_bits) / result.uncompressed_bits;
    result.compression_throughput_mbs = (data.size() * sizeof(T) / 1024.0 / 1024.0) / (compression_time_ns / 1e9);
    
    // Full decompression
    std::vector<T> decompressed(data.size());
    t1 = std::chrono::high_resolution_clock::now();
    compressor.get_elements(0, data.size(), decompressed);
    t2 = std::chrono::high_resolution_clock::now();
    do_not_optimize(decompressed);
    
    auto decompression_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.decompression_throughput_mbs = (data.size() * sizeof(T) / 1024.0 / 1024.0) / (decompression_time_ns / 1e9);
    
    for (size_t i = 0; i < data.size(); ++i) {
        if (data[i] != decompressed[i]) {
            std::cerr << compressor_name << " decompression error at " << i << std::endl;
            break;
        }
    }
    
    // Random access
    const size_t num_ra_queries = bench_data.random_indices.size();
    
    t1 = std::chrono::high_resolution_clock::now();
    T sum = 0;
    for (auto idx : bench_data.random_indices) {
        sum += compressor[idx];
    }
    t2 = std::chrono::high_resolution_clock::now();
    do_not_optimize(sum);
    auto ra_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.random_access_ns = static_cast<double>(ra_time_ns) / num_ra_queries;
    result.random_access_mbs = (num_ra_queries * sizeof(T) / 1024.0 / 1024.0) / (ra_time_ns / 1e9);
    
    // Range queries
    const size_t num_range_queries = bench_data.range_query_indices.begin()->second.size();
    for (auto range : range_sizes) {
        if (range >= data.size()) continue;
        
        const auto& range_indices = bench_data.range_query_indices.at(range);
        std::vector<T> out_buffer(range);
        
        t1 = std::chrono::high_resolution_clock::now();
        for (auto start_idx : range_indices) {
            compressor.get_elements(start_idx, range, out_buffer);
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
    
    for (size_t i = 0; i < data.size(); ++i) {
        if (u_data[i] != decompressed[i]) {
            std::cerr << "DAC decompression error at " << i << std::endl;
            break;
        }
    }
    
    // Random access
    const size_t num_ra_queries = bench_data.random_indices.size();
    
    t1 = std::chrono::high_resolution_clock::now();
    uint64_t sum = 0;
    for (auto idx : bench_data.random_indices) {
        sum += dac_vector[idx];
    }
    t2 = std::chrono::high_resolution_clock::now();
    do_not_optimize(sum);
    auto ra_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.random_access_ns = static_cast<double>(ra_time_ns) / num_ra_queries;
    result.random_access_mbs = (num_ra_queries * sizeof(T) / 1024.0 / 1024.0) / (ra_time_ns / 1e9);
    
    // Range queries
    const size_t num_range_queries = bench_data.range_query_indices.begin()->second.size();
    for (auto range : range_sizes) {
        if (range >= data.size()) continue;
        
        const auto& range_indices = bench_data.range_query_indices.at(range);
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
        
        std::unique_ptr<Compressor> cmpr;
        if constexpr (std::is_same_v<Compressor, CompressorCamel<T>>) {
            cmpr = std::make_unique<Compressor>(*start_it, static_cast<int64_t>(bench_data.decimals));
        } else {
            cmpr = std::make_unique<Compressor>(*start_it);
        }

        for (auto it = start_it + 1; it < end_it; ++it) {
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
        for (size_t k = 1; k < bs; ++k) {
            if (dcmpr.hasNext()) {
                decompressed[offset++] = dcmpr.storedValue;
            } else {
                break;
            }
        }
    }
    t2 = std::chrono::high_resolution_clock::now();
    do_not_optimize(decompressed);
    auto decompression_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    
    result.decompression_throughput_mbs = (n * sizeof(T) / 1024.0 / 1024.0) / (decompression_time_ns / 1e9);
    
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
    // Use subset of random indices if we generated more for others?
    // The original code used 100,000 for this, but 1,000,000 for others.
    // bench_data.random_indices has 1,000,000. We can just take the first 100,000.
    
    t1 = std::chrono::high_resolution_clock::now();
    T sum = 0;
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
        
        const auto& range_indices = bench_data.range_query_indices.at(range);
        std::vector<T> out_buffer(range);
        
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
        for (size_t k = 1; k < bs; ++k) {
            if (dcmpr.hasNext()) {
                decompressed[offset++] = dcmpr.storedValue;
            } else {
                break;
            }
        }
    }
    t2 = std::chrono::high_resolution_clock::now();
    do_not_optimize(decompressed);
    auto decompression_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    
    result.decompression_throughput_mbs = (n * sizeof(T) / 1024.0 / 1024.0) / (decompression_time_ns / 1e9);
    
    for (size_t i = 0; i < n; ++i) {
        if (std::abs(data[i] - decompressed[i]) >= 1e-9) {
            std::cerr << "TSXor decompression error at " << i << std::endl;
            break;
        }
    }
    
    // Random access
    const size_t num_ra_queries = 100000;
    
    t1 = std::chrono::high_resolution_clock::now();
    T sum = 0;
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
        
        const auto& range_indices = bench_data.range_query_indices.at(range);
        std::vector<T> out_buffer(range);
        
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
BenchmarkResult benchmark_falcon(const BenchmarkData &bench_data,
                                 const std::vector<size_t> &range_sizes,
                                 size_t block_size = 1000) { // Default to 1000 to match your proposal
    BenchmarkResult result;
    result.compressor = "Falcon (Block-Based)";
    result.dataset = bench_data.filename;
    
    // Use double data
    const auto& data = bench_data.double_data;
    
    result.num_values = data.size();
    result.uncompressed_bits = bench_data.uncompressed_bits;
    
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
    
    result.compressed_bits = total_compressed_bits;
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
        
        for (size_t k = 1; k < expected_count; ++k) {
            if (dcmpr.hasNext()) {
                decompressed[output_offset++] = dcmpr.storedValue;
            } else {
                break;
            }
        }
    }
    
    t2 = std::chrono::high_resolution_clock::now();
    do_not_optimize(decompressed);
    auto decompression_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.decompression_throughput_mbs = (n * sizeof(T) / 1024.0 / 1024.0) / (decompression_time_ns / 1e9);
    

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
    
    t1 = std::chrono::high_resolution_clock::now();
    T sum = 0;
    size_t q_count = 0;
    
    for (auto idx : bench_data.random_indices) {
        if (q_count++ >= num_ra_queries) break;

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
        
        const auto& range_indices = bench_data.range_query_indices.at(range);
        std::vector<T> out_buffer(range);
        
        t1 = std::chrono::high_resolution_clock::now();
        size_t q_count_range = 0;
        for (auto start_idx : range_indices) {
            if (q_count_range++ >= num_range_queries) break;
            
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
// LeCo Benchmark
// ============================================================================

#if defined(USE_SQUASH) && defined(NEATS_ENABLE_LECO)
template<typename T = int64_t>
BenchmarkResult benchmark_leco(const BenchmarkData &bench_data,
                               const std::vector<size_t> &range_sizes,
                               size_t block_size = 1000) {
    BenchmarkResult result;
    result.compressor = "LeCo";
    result.dataset = bench_data.filename;
    
    // Use shifted data for integer compression
    const auto& data = bench_data.shifted_data;
    
    result.num_values = data.size();
    result.uncompressed_bits = bench_data.uncompressed_bits;
    
    const size_t n = data.size();
    const size_t num_blocks = (n / block_size) + (n % block_size != 0);
    
    // Convert data to type T for LeCo (assumes T matches template)
    // Note: LeCo seems to take pointer to data
    
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
        
        // Cast data to expected type. T should be int64_t or uint64_t based on usage
        res = codec.encodeArray8_int(reinterpret_cast<const T*>(data.data() + (i * block_size)), block_length, descriptor, i);
        
        uint32_t segment_size = res - descriptor;
        // Realloc to fit exact size
        uint8_t* exact_buf = (uint8_t*)realloc(descriptor, segment_size);
        // If realloc moved it, fine. If it failed (unlikely for shrink), we handle it.
        // Actually realloc might fail if size is 0, but segment_size >= 1 usually.
        if (exact_buf) descriptor = exact_buf;
        
        block_start_vec.push_back(descriptor);
        total_compressed_bits += segment_size * 8;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto compression_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    
    result.compressed_bits = total_compressed_bits;
    result.compression_ratio = static_cast<double>(result.compressed_bits) / result.uncompressed_bits;
    result.compression_throughput_mbs = (n * sizeof(T) / 1024.0 / 1024.0) / (compression_time_ns / 1e9);
    
    // Full decompression
    std::vector<T> decompressed(n);
    t1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_blocks; i++) {
        int block_length = block_size;
        if (i == num_blocks - 1) {
            block_length = n - (num_blocks - 1) * block_size;
        }
        codec.decodeArray8(block_start_vec[i], block_length, decompressed.data() + i * block_size, i);
    }
    // Apply patches
    for (auto index : codec.mul_add_diff_set) {
        decompressed[index.first] += index.second;
    }
    
    t2 = std::chrono::high_resolution_clock::now();
    do_not_optimize(decompressed);
    auto decompression_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    
    result.decompression_throughput_mbs = (n * sizeof(T) / 1024.0 / 1024.0) / (decompression_time_ns / 1e9);
    
    for (size_t i = 0; i < n; ++i) {
        if (data[i] != decompressed[i]) {
            std::cerr << "LeCo decompression error at " << i << std::endl;
            break;
        }
    }
    
    // Random access
    const size_t num_ra_queries = 10000;
    t1 = std::chrono::high_resolution_clock::now();
    T sum = 0;
    size_t q_count = 0;
    for (auto idx : bench_data.random_indices) {
        if (q_count++ >= num_ra_queries) break;
        
        size_t ib = idx / block_size;
        size_t offset_in_block = idx % block_size;
        
        // randomdecodeArray8 expects in, to_find (offset), out (unused?), nvalue (total size?)
        // Signature: T randomdecodeArray8(const uint8_t *in, int to_find, uint32_t *out, size_t nvalue)
        // nvalue parameter in decodeArray8 seems unused or used for patching?
        // In randomdecodeArray8, nvalue is used as argument name but not used in body in the version I read?
        // Wait, in the file I read:
        // T randomdecodeArray8(..., size_t nvalue) { ... }
        // It seems nvalue is not used in the body I saw (maybe used in comments).
        // Let's pass n just in case.
        
        T val = codec.randomdecodeArray8(block_start_vec[ib], offset_in_block, nullptr, n);
        sum += val;
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
        
        const auto& range_indices = bench_data.range_query_indices.at(range);
        std::vector<T> out_buffer(range);
        
        t1 = std::chrono::high_resolution_clock::now();
        size_t q_count_range = 0;
        for (auto start_idx : range_indices) {
            if (q_count_range++ >= num_range_queries) break;
            
            size_t start_block = start_idx / block_size;
            size_t end_block = (start_idx + range - 1) / block_size;
            
            // LeCo doesn't have a specific range decode function exposed clearly in the template,
            // but we can use full decode of blocks or random access.
            // Since we want throughput, let's simulate by decoding blocks and copying.
            // Optimized range query might be possible but for now we do block decoding.
            
            size_t out_pos = 0;
            for (size_t ib = start_block; ib <= end_block; ++ib) {
                int block_length = block_size;
                if (ib == num_blocks - 1) {
                    block_length = n - (num_blocks - 1) * block_size;
                }
                
                // We have to decode the whole block because LeCo is delta/linear regression based
                // and might not support partial decode easily without seeking.
                // The filter_range functions exist but they are for filtering, not range extraction.
                // So we allocate a buffer for the block.
                std::vector<T> block_buffer(block_length);
                codec.decodeArray8(block_start_vec[ib], block_length, block_buffer.data(), ib);
                
                // Copy relevant part
                size_t block_start_global = ib * block_size;
                size_t skip = (ib == start_block) ? (start_idx - block_start_global) : 0;
                size_t take = std::min((size_t)block_length - skip, range - out_pos);
                
                memcpy(out_buffer.data() + out_pos, block_buffer.data() + skip, take * sizeof(T));
                out_pos += take;
            }
            do_not_optimize(out_buffer);
        }
        t2 = std::chrono::high_resolution_clock::now();
        
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
#endif


// ============================================================================
// ALP Benchmark (Adaptive Lossless Floating-Point Compression)
// https://github.com/cwida/ALP
// ============================================================================

namespace {
struct ALPBlock {
    alp::Scheme scheme = alp::Scheme::INVALID;
    uint16_t count = 0; // number of values in this block (<= alp::config::VECTOR_SIZE)

    // Common
    uint16_t exceptions_count = 0;

    // ALP (PDE)
    uint8_t bit_width = 0;
    uint8_t fac = 0;
    uint8_t exp = 0;
    int64_t base = 0;
    std::vector<uint64_t> packed_digits;          // FastLanes FFOR packed words
    std::vector<double> exceptions;              // exception values
    std::vector<uint16_t> exception_positions;   // exception positions within vector

    // ALP_RD
    uint8_t right_bit_width = 0;
    uint8_t left_bit_width = 0;
    uint8_t dict_size = 0;
    std::array<uint16_t, alp::config::MAX_RD_DICTIONARY_SIZE> dict {};
    std::vector<uint64_t> packed_right;          // packed right parts (u64 words)
    std::vector<uint16_t> packed_left;           // packed left parts (u16 words)
    std::vector<uint16_t> rd_exceptions;         // left-part exceptions (values)
    std::vector<uint16_t> rd_exception_positions;// positions within vector

    // RAW fallback for tail blocks
    std::vector<double> raw;
};

inline size_t words_u64_for_bw(uint8_t bw) {
    if (bw == 0) return 0;
    return (alp::config::VECTOR_SIZE * static_cast<size_t>(bw) + 63) / 64;
}

inline size_t words_u16_for_bw(uint8_t bw) {
    if (bw == 0) return 0;
    return (alp::config::VECTOR_SIZE * static_cast<size_t>(bw) + 15) / 16;
}
} // namespace

BenchmarkResult benchmark_alp(const BenchmarkData &bench_data,
                              const std::vector<size_t> &range_sizes) {
    BenchmarkResult result;
    result.compressor = "ALP";
    result.dataset = bench_data.filename;

    const auto& data = bench_data.double_data;
    const size_t n = data.size();

    result.num_values = n;
    result.uncompressed_bits = bench_data.uncompressed_bits;

    std::vector<ALPBlock> blocks;
    blocks.reserve((n + alp::config::VECTOR_SIZE - 1) / alp::config::VECTOR_SIZE);

    // Compression
    auto t1 = std::chrono::high_resolution_clock::now();
    {
        // Process full vectors in rowgroups of 100 vectors (ALP default)
        const size_t n_full = (n / alp::config::VECTOR_SIZE) * alp::config::VECTOR_SIZE;
        const size_t num_vectors = n_full / alp::config::VECTOR_SIZE;
        const size_t vectors_per_rg = alp::config::N_VECTORS_PER_ROWGROUP;

        std::array<double, alp::config::VECTOR_SIZE> sample_arr {};
        alp::state<double> stt;

        for (size_t vg = 0; vg < num_vectors; vg += vectors_per_rg) {
            const size_t rg_vectors = std::min(vectors_per_rg, num_vectors - vg);
            const size_t rg_offset_vals = vg * alp::config::VECTOR_SIZE;
            const size_t rg_vals = rg_vectors * alp::config::VECTOR_SIZE;

            // Initialize scheme for this rowgroup (ALP chooses ALP vs ALP_RD)
            alp::encoder<double>::init(data.data(), rg_offset_vals, rg_vals, sample_arr.data(), stt);
            if (stt.scheme == alp::Scheme::ALP_RD) {
                alp::rd_encoder<double>::init(data.data(), rg_offset_vals, rg_vals, sample_arr.data(), stt);
            }

            for (size_t v = 0; v < rg_vectors; ++v) {
                const double* vec = data.data() + rg_offset_vals + v * alp::config::VECTOR_SIZE;

                ALPBlock blk;
                blk.scheme = stt.scheme;
                blk.count = static_cast<uint16_t>(alp::config::VECTOR_SIZE);

                if (blk.scheme == alp::Scheme::ALP) {
                    alignas(64) std::array<int64_t, alp::config::VECTOR_SIZE> encoded {};
                    alignas(64) std::array<double, alp::config::VECTOR_SIZE> exceptions {};
                    alignas(64) std::array<uint16_t, alp::config::VECTOR_SIZE> positions {};
                    uint16_t exc_count = 0;

                    alp::encoder<double>::encode(vec, exceptions.data(), positions.data(), &exc_count, encoded.data(), stt);
                    blk.exceptions_count = exc_count;
                    blk.exceptions.assign(exceptions.begin(), exceptions.begin() + exc_count);
                    blk.exception_positions.assign(positions.begin(), positions.begin() + exc_count);
                    blk.fac = stt.fac;
                    blk.exp = stt.exp;

                    alp::bw_t bw = 0;
                    int64_t base = 0;
                    alp::encoder<double>::analyze_ffor(encoded.data(), bw, &base);
                    blk.bit_width = bw;
                    blk.base = base;

                    const size_t w64 = words_u64_for_bw(blk.bit_width);
                    blk.packed_digits.assign(w64, 0);
                    if (w64 > 0) {
                        ffor::ffor(reinterpret_cast<const uint64_t*>(encoded.data()),
                                   blk.packed_digits.data(),
                                   blk.bit_width,
                                   reinterpret_cast<const uint64_t*>(&blk.base));
                    }
                } else {
                    // ALP_RD
                    alignas(64) std::array<uint64_t, alp::config::VECTOR_SIZE> right_parts {};
                    alignas(64) std::array<uint16_t, alp::config::VECTOR_SIZE> left_parts {};
                    alignas(64) std::array<uint16_t, alp::config::VECTOR_SIZE> exc {};
                    alignas(64) std::array<uint16_t, alp::config::VECTOR_SIZE> exc_pos {};
                    uint16_t exc_count = 0;

                    alp::rd_encoder<double>::encode(vec, exc.data(), exc_pos.data(), &exc_count,
                                                    right_parts.data(), left_parts.data(), stt);
                    blk.exceptions_count = exc_count;
                    blk.rd_exceptions.assign(exc.begin(), exc.begin() + exc_count);
                    blk.rd_exception_positions.assign(exc_pos.begin(), exc_pos.begin() + exc_count);

                    blk.right_bit_width = static_cast<uint8_t>(stt.right_bit_width);
                    blk.left_bit_width = static_cast<uint8_t>(stt.left_bit_width);
                    blk.dict_size = static_cast<uint8_t>(stt.actual_dictionary_size);
                    for (size_t i = 0; i < alp::config::MAX_RD_DICTIONARY_SIZE; ++i) {
                        blk.dict[i] = stt.left_parts_dict[i];
                    }

                    const uint64_t zero64 = 0;
                    const uint16_t zero16 = 0;

                    const size_t wr = words_u64_for_bw(blk.right_bit_width);
                    blk.packed_right.assign(wr, 0);
                    if (wr > 0) {
                        ffor::ffor(right_parts.data(), blk.packed_right.data(), blk.right_bit_width, &zero64);
                    }

                    const size_t wl = words_u16_for_bw(blk.left_bit_width);
                    blk.packed_left.assign(wl, 0);
                    if (wl > 0) {
                        ffor::ffor(left_parts.data(), blk.packed_left.data(), blk.left_bit_width, &zero16);
                    }
                }

                blocks.push_back(std::move(blk));
            }
        }

        // Tail (store raw)
        if (n_full < n) {
            ALPBlock tail;
            tail.scheme = alp::Scheme::INVALID;
            tail.count = static_cast<uint16_t>(n - n_full);
            tail.raw.assign(data.begin() + n_full, data.end());
            blocks.push_back(std::move(tail));
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto compression_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();

    // Compressed size (bytes -> bits)
    size_t compressed_bytes = 0;
    for (const auto& blk : blocks) {
        // 1 byte scheme + 2 bytes count + 2 bytes exceptions_count
        compressed_bytes += 1 + 2 + 2;
        if (blk.scheme == alp::Scheme::ALP) {
            // bw + fac + exp + base
            compressed_bytes += 1 + 1 + 1 + sizeof(int64_t);
            compressed_bytes += blk.packed_digits.size() * sizeof(uint64_t);
            compressed_bytes += blk.exceptions.size() * sizeof(double);
            compressed_bytes += blk.exception_positions.size() * sizeof(uint16_t);
        } else if (blk.scheme == alp::Scheme::ALP_RD) {
            // right_bw + left_bw + dict_size + dictionary entries
            compressed_bytes += 1 + 1 + 1;
            compressed_bytes += static_cast<size_t>(blk.dict_size) * sizeof(uint16_t);
            compressed_bytes += blk.packed_right.size() * sizeof(uint64_t);
            compressed_bytes += blk.packed_left.size() * sizeof(uint16_t);
            compressed_bytes += blk.rd_exceptions.size() * sizeof(uint16_t);
            compressed_bytes += blk.rd_exception_positions.size() * sizeof(uint16_t);
        } else {
            compressed_bytes += blk.raw.size() * sizeof(double);
        }
    }

    result.compressed_bits = compressed_bytes * 8;
    result.compression_ratio = static_cast<double>(result.compressed_bits) / result.uncompressed_bits;
    result.compression_throughput_mbs = (n * sizeof(double) / 1024.0 / 1024.0) / (compression_time_ns / 1e9);

    // Full decompression
    std::vector<double> decompressed(n);
    t1 = std::chrono::high_resolution_clock::now();
    {
        size_t out_off = 0;
        for (const auto& blk : blocks) {
            if (blk.scheme == alp::Scheme::ALP) {
                alignas(64) std::array<double, alp::config::VECTOR_SIZE> out_vec {};
                generated::falp::fallback::scalar::falp(
                    blk.packed_digits.data(),
                    out_vec.data(),
                    blk.bit_width,
                    reinterpret_cast<const uint64_t*>(&blk.base),
                    blk.fac,
                    blk.exp
                );
                const alp::exp_c_t exc_c = static_cast<alp::exp_c_t>(blk.exceptions_count);
                alp::decoder<double>::patch_exceptions(
                    out_vec.data(),
                    blk.exceptions.data(),
                    reinterpret_cast<const alp::exp_p_t*>(blk.exception_positions.data()),
                    &exc_c
                );
                std::copy(out_vec.begin(), out_vec.begin() + blk.count, decompressed.begin() + out_off);
                out_off += blk.count;
            } else if (blk.scheme == alp::Scheme::ALP_RD) {
                alignas(64) std::array<uint64_t, alp::config::VECTOR_SIZE> right_out {};
                alignas(64) std::array<uint16_t, alp::config::VECTOR_SIZE> left_out {};
                const uint64_t zero64 = 0;
                const uint16_t zero16 = 0;
                if (!blk.packed_right.empty()) {
                    unffor::unffor(blk.packed_right.data(), right_out.data(), blk.right_bit_width, &zero64);
                }
                if (!blk.packed_left.empty()) {
                    unffor::unffor(blk.packed_left.data(), left_out.data(), blk.left_bit_width, &zero16);
                }
                alp::state<double> stt;
                stt.scheme = alp::Scheme::ALP_RD;
                stt.right_bit_width = blk.right_bit_width;
                stt.left_bit_width = blk.left_bit_width;
                stt.actual_dictionary_size = blk.dict_size;
                for (size_t i = 0; i < alp::config::MAX_RD_DICTIONARY_SIZE; ++i) {
                    stt.left_parts_dict[i] = blk.dict[i];
                }
                uint16_t exc_c = blk.exceptions_count;
                alignas(64) std::array<double, alp::config::VECTOR_SIZE> out_vec {};
                alp::rd_encoder<double>::decode(
                    out_vec.data(),
                    right_out.data(),
                    left_out.data(),
                    const_cast<uint16_t*>(blk.rd_exceptions.data()),
                    const_cast<uint16_t*>(blk.rd_exception_positions.data()),
                    &exc_c,
                    stt
                );
                std::copy(out_vec.begin(), out_vec.begin() + blk.count, decompressed.begin() + out_off);
                out_off += blk.count;
            } else {
                std::copy(blk.raw.begin(), blk.raw.end(), decompressed.begin() + out_off);
                out_off += blk.raw.size();
            }
        }
    }
    t2 = std::chrono::high_resolution_clock::now();
    do_not_optimize(decompressed);
    auto decompression_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.decompression_throughput_mbs = (n * sizeof(double) / 1024.0 / 1024.0) / (decompression_time_ns / 1e9);

    for (size_t i = 0; i < n; ++i) {
        if (data[i] != decompressed[i]) {
            std::cerr << "ALP decompression error at " << i << std::endl;
            break;
        }
    }

    // Random access (block decompression)
    const size_t num_ra_queries = 10000;
    t1 = std::chrono::high_resolution_clock::now();
    double sum = 0;
    size_t q_count = 0;
    for (auto idx : bench_data.random_indices) {
        if (q_count++ >= num_ra_queries) break;
        const size_t ib = idx / alp::config::VECTOR_SIZE;
        const size_t off = idx % alp::config::VECTOR_SIZE;
        const auto& blk = blocks[ib];
        alignas(64) std::array<double, alp::config::VECTOR_SIZE> out_vec {};

        if (blk.scheme == alp::Scheme::ALP) {
            generated::falp::fallback::scalar::falp(
                blk.packed_digits.data(),
                out_vec.data(),
                blk.bit_width,
                reinterpret_cast<const uint64_t*>(&blk.base),
                blk.fac,
                blk.exp
            );
            const alp::exp_c_t exc_c = static_cast<alp::exp_c_t>(blk.exceptions_count);
            alp::decoder<double>::patch_exceptions(
                out_vec.data(),
                blk.exceptions.data(),
                reinterpret_cast<const alp::exp_p_t*>(blk.exception_positions.data()),
                &exc_c
            );
            sum += out_vec[off];
        } else if (blk.scheme == alp::Scheme::ALP_RD) {
            alignas(64) std::array<uint64_t, alp::config::VECTOR_SIZE> right_out {};
            alignas(64) std::array<uint16_t, alp::config::VECTOR_SIZE> left_out {};
            const uint64_t zero64 = 0;
            const uint16_t zero16 = 0;
            if (!blk.packed_right.empty()) {
                    unffor::unffor(blk.packed_right.data(), right_out.data(), blk.right_bit_width, &zero64);
            }
            if (!blk.packed_left.empty()) {
                    unffor::unffor(blk.packed_left.data(), left_out.data(), blk.left_bit_width, &zero16);
            }
            alp::state<double> stt;
            stt.scheme = alp::Scheme::ALP_RD;
            stt.right_bit_width = blk.right_bit_width;
            stt.left_bit_width = blk.left_bit_width;
            stt.actual_dictionary_size = blk.dict_size;
            for (size_t i = 0; i < alp::config::MAX_RD_DICTIONARY_SIZE; ++i) {
                stt.left_parts_dict[i] = blk.dict[i];
            }
            uint16_t exc_c = blk.exceptions_count;
            alp::rd_encoder<double>::decode(
                out_vec.data(),
                right_out.data(),
                left_out.data(),
                const_cast<uint16_t*>(blk.rd_exceptions.data()),
                const_cast<uint16_t*>(blk.rd_exception_positions.data()),
                &exc_c,
                stt
            );
            sum += out_vec[off];
        } else {
            sum += blk.raw[off];
        }
    }
    t2 = std::chrono::high_resolution_clock::now();
    do_not_optimize(sum);
    auto ra_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.random_access_ns = static_cast<double>(ra_time_ns) / num_ra_queries;
    result.random_access_mbs = (num_ra_queries * sizeof(double) / 1024.0 / 1024.0) / (ra_time_ns / 1e9);

    // Range queries (block decompression)
    const size_t num_range_queries = 1000;
    for (auto range : range_sizes) {
        if (range >= n) continue;
        const auto& range_indices = bench_data.range_query_indices.at(range);
        std::vector<double> out_buffer(range);

        t1 = std::chrono::high_resolution_clock::now();
        size_t qcr = 0;
        for (auto start_idx : range_indices) {
            if (qcr++ >= num_range_queries) break;
            const size_t end_idx = start_idx + range;
            size_t cur = start_idx;
            size_t out_pos = 0;
            while (cur < end_idx) {
                const size_t ib = cur / alp::config::VECTOR_SIZE;
                const size_t in_off = cur % alp::config::VECTOR_SIZE;
                const auto& blk = blocks[ib];
                const size_t to_take = std::min(end_idx - cur, alp::config::VECTOR_SIZE - in_off);

                alignas(64) std::array<double, alp::config::VECTOR_SIZE> tmp {};
                if (blk.scheme == alp::Scheme::ALP) {
                    generated::falp::fallback::scalar::falp(
                        blk.packed_digits.data(),
                        tmp.data(),
                        blk.bit_width,
                        reinterpret_cast<const uint64_t*>(&blk.base),
                        blk.fac,
                        blk.exp
                    );
                    const alp::exp_c_t exc_c = static_cast<alp::exp_c_t>(blk.exceptions_count);
                    alp::decoder<double>::patch_exceptions(
                        tmp.data(),
                        blk.exceptions.data(),
                        reinterpret_cast<const alp::exp_p_t*>(blk.exception_positions.data()),
                        &exc_c
                    );
                } else if (blk.scheme == alp::Scheme::ALP_RD) {
                    alignas(64) std::array<uint64_t, alp::config::VECTOR_SIZE> right_out {};
                    alignas(64) std::array<uint16_t, alp::config::VECTOR_SIZE> left_out {};
                    const uint64_t zero64 = 0;
                    const uint16_t zero16 = 0;
                    if (!blk.packed_right.empty()) {
                        unffor::unffor(blk.packed_right.data(), right_out.data(), blk.right_bit_width, &zero64);
                    }
                    if (!blk.packed_left.empty()) {
                        unffor::unffor(blk.packed_left.data(), left_out.data(), blk.left_bit_width, &zero16);
                    }
                    alp::state<double> stt;
                    stt.scheme = alp::Scheme::ALP_RD;
                    stt.right_bit_width = blk.right_bit_width;
                    stt.left_bit_width = blk.left_bit_width;
                    stt.actual_dictionary_size = blk.dict_size;
                    for (size_t i = 0; i < alp::config::MAX_RD_DICTIONARY_SIZE; ++i) {
                        stt.left_parts_dict[i] = blk.dict[i];
                    }
                    uint16_t exc_c = blk.exceptions_count;
                    alp::rd_encoder<double>::decode(
                        tmp.data(),
                        right_out.data(),
                        left_out.data(),
                        const_cast<uint16_t*>(blk.rd_exceptions.data()),
                        const_cast<uint16_t*>(blk.rd_exception_positions.data()),
                        &exc_c,
                        stt
                    );
                } else {
                    std::copy(blk.raw.begin(), blk.raw.end(), tmp.begin());
                }

                memcpy(out_buffer.data() + out_pos, tmp.data() + in_off, to_take * sizeof(double));
                out_pos += to_take;
                cur += to_take;
            }
            do_not_optimize(out_buffer);
        }
        t2 = std::chrono::high_resolution_clock::now();
        auto range_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
        double throughput = ((range * sizeof(double)) * num_range_queries / 1024.0 / 1024.0) / (range_time_ns / 1e9);
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
    
    for (size_t i = 0; i < n; ++i) {
        if (data[i] != decompressed[i]) {
            std::cerr << compressor_name << " decompression error at " << i << std::endl;
            break;
        }
    }
    
    // Random access
    const size_t num_ra_queries = 100000;
    
    t1 = std::chrono::high_resolution_clock::now();
    T sum = 0;
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
        
        const auto& range_indices = bench_data.range_query_indices.at(range);
        std::vector<T> out_buffer(range);
        
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
    std::cerr << "                 Available: neats,dac,rle_gef,u_gef_approximate,u_gef_optimal,b_gef_approximate,b_gef_optimal,b_star_gef_approximate,b_star_gef_optimal,gorilla,chimp,chimp128,tsxor,elf,camel,falcon,alp";
#if defined(NEATS_ENABLE_LECO)
    std::cerr << ",leco";
#endif
    std::cerr << ",lz4,zstd,brotli,xz,snappy" << std::endl;
#else
    std::cerr << "                 Available: neats,dac,rle_gef,u_gef_approximate,u_gef_optimal,b_gef_approximate,b_gef_optimal,b_star_gef_approximate,b_star_gef_optimal,gorilla,chimp,chimp128,tsxor,elf,camel,falcon,alp" << std::endl;
    std::cerr << "                 (Compile with -DUSE_SQUASH for lz4,zstd,brotli,xz,snappy";
#if defined(NEATS_ENABLE_LECO)
    std::cerr << ",leco";
#endif
    std::cerr << ")" << std::endl;
#endif
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
                                            "elf", "camel", "falcon", "alp",
#if defined(NEATS_ENABLE_LECO)
                                            "leco",
#endif
                                            "lz4", "zstd", "brotli", "xz", "snappy"};
#else
    std::vector<std::string> compressors = {"neats", "dac", "rle_gef", "u_gef_approximate", "u_gef_optimal", 
                                            "b_gef_approximate", "b_gef_optimal", "b_star_gef_approximate", "b_star_gef_optimal",
                                            "gorilla", "chimp", "chimp128", "tsxor",
                                            "elf", "camel", "falcon", "alp"};
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
        
        // Phase 0: Run compressors that need raw_data (SQUASH compressors)
#if HAS_SQUASH
        for (const auto &comp : raw_data_compressors) {
            std::cerr << "  Running " << comp << "..." << std::flush;
            
            BenchmarkResult result;
            
            try {
                result = benchmark_squash<int64_t>(comp, bench_data, range_sizes, block_size);
                
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
#endif
        
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
#if defined(NEATS_ENABLE_LECO)
                else if (comp == "leco") {
                    result = benchmark_leco<int64_t>(bench_data, range_sizes, block_size);
                } else if (comp == "rle_gef") {
#else
                else if (comp == "rle_gef") {
#endif
                    using UP_RLE = gef::UniformedPartitioner<int64_t, RLE_GEF_Wrapper<int64_t>, std::shared_ptr<IBitVectorFactory>>;
                    result = benchmark_gef<UP_RLE, int64_t>("rle_gef", bench_data, range_sizes, GEF_UNIFORM_PARTITION_SIZE);
                } else if (comp == "u_gef_approximate") {
                    using UP_U = gef::UniformedPartitioner<int64_t, U_GEF_Wrapper<int64_t, gef::APPROXIMATE_SPLIT_POINT>, std::shared_ptr<IBitVectorFactory>>;
                    result = benchmark_gef<UP_U, int64_t>("u_gef_approximate", bench_data, range_sizes, GEF_UNIFORM_PARTITION_SIZE);
                } else if (comp == "u_gef_optimal") {
                    using UP_U = gef::UniformedPartitioner<int64_t, U_GEF_Wrapper<int64_t, gef::OPTIMAL_SPLIT_POINT>, std::shared_ptr<IBitVectorFactory>>;
                    result = benchmark_gef<UP_U, int64_t>("u_gef_optimal", bench_data, range_sizes, GEF_UNIFORM_PARTITION_SIZE);
                } else if (comp == "b_gef_approximate") {
                    using UP_B = gef::UniformedPartitioner<int64_t, B_GEF_Wrapper<int64_t, gef::APPROXIMATE_SPLIT_POINT>, std::shared_ptr<IBitVectorFactory>>;
                    result = benchmark_gef<UP_B, int64_t>("b_gef_approximate", bench_data, range_sizes, GEF_UNIFORM_PARTITION_SIZE);
                } else if (comp == "b_gef_optimal") {
                    using UP_B = gef::UniformedPartitioner<int64_t, B_GEF_Wrapper<int64_t, gef::OPTIMAL_SPLIT_POINT>, std::shared_ptr<IBitVectorFactory>>;
                    result = benchmark_gef<UP_B, int64_t>("b_gef_optimal", bench_data, range_sizes, GEF_UNIFORM_PARTITION_SIZE);
                } else if (comp == "b_star_gef_approximate") {
                    using UP_B_STAR = gef::UniformedPartitioner<int64_t, B_GEF_STAR_Wrapper<int64_t, gef::APPROXIMATE_SPLIT_POINT>, std::shared_ptr<IBitVectorFactory>>;
                    result = benchmark_gef<UP_B_STAR, int64_t>("b_star_gef_approximate", bench_data, range_sizes, GEF_UNIFORM_PARTITION_SIZE);
                } else if (comp == "b_star_gef_optimal") {
                    using UP_B_STAR = gef::UniformedPartitioner<int64_t, B_GEF_STAR_Wrapper<int64_t, gef::OPTIMAL_SPLIT_POINT>, std::shared_ptr<IBitVectorFactory>>;
                    result = benchmark_gef<UP_B_STAR, int64_t>("b_star_gef_optimal", bench_data, range_sizes, GEF_UNIFORM_PARTITION_SIZE);
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
                if (comp == "gorilla") {
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
                    result = benchmark_bitstream_compressor<CompressorCamel<double>, DecompressorCamel<double>, double>(
                        "Camel", bench_data, range_sizes, block_size);
                } else if (comp == "falcon") {
                    result = benchmark_falcon<double>(bench_data, range_sizes);
                } else if (comp == "alp") {
                    result = benchmark_alp(bench_data, range_sizes);
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
