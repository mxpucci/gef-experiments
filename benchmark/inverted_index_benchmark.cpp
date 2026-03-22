/**
 * Inverted Index Compressor Benchmark
 *
 * Benchmarks GEF variants against non-time-series compressors on
 * inverted index (posting list) datasets from the RoaringBitmap collection.
 *
 * Metrics:
 *   1. Compression ratio (bits per integer)
 *   2. Compression throughput (MB/s)
 *   3. Decompression throughput (MB/s)
 *   4. Random access latency (ns/query)
 *
 * Input: directory of .txt files, each containing comma-separated
 * sorted uint32_t document IDs (one posting list per file).
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "benchmark_common.hpp"

// x86 SIMD intrinsics must be included before GEF headers which use AVX2
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

// GEF
#include "gef/gef.hpp"

// FastPFOR
#include "codecfactory.h"
#include "deltautil.h"

// sux-rs FFI
#include "sux_rs_ffi.h"

// ============================================================================
// Dataset loading
// ============================================================================

struct PostingList {
    std::string name;
    std::vector<uint32_t> docids; // sorted document IDs
};

struct InvertedIndexDataset {
    std::string name;
    std::vector<PostingList> lists;
    size_t total_ints = 0;
};

PostingList load_posting_list(const std::string &filepath) {
    PostingList pl;
    pl.name = std::filesystem::path(filepath).stem().string();

    std::ifstream in(filepath);
    if (!in) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }

    std::string content((std::istreambuf_iterator<char>(in)),
                        std::istreambuf_iterator<char>());

    std::istringstream ss(content);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) {
            char *end;
            unsigned long val = std::strtoul(token.c_str(), &end, 10);
            if (end != token.c_str()) {
                pl.docids.push_back(static_cast<uint32_t>(val));
            }
        }
    }

    return pl;
}

InvertedIndexDataset load_dataset_directory(const std::string &dirpath) {
    InvertedIndexDataset dataset;
    dataset.name = std::filesystem::path(dirpath).filename().string();

    std::vector<std::string> files;
    for (const auto &entry : std::filesystem::directory_iterator(dirpath)) {
        if (entry.path().extension() == ".txt") {
            files.push_back(entry.path().string());
        }
    }
    std::sort(files.begin(), files.end());

    for (const auto &f : files) {
        auto pl = load_posting_list(f);
        if (!pl.docids.empty()) {
            dataset.total_ints += pl.docids.size();
            dataset.lists.push_back(std::move(pl));
        }
    }

    return dataset;
}

// ============================================================================
// Result structure
// ============================================================================

struct IIBenchmarkResult {
    std::string compressor;
    std::string dataset;
    size_t num_lists{};
    size_t total_ints{};
    size_t uncompressed_bits{};
    size_t compressed_bits{};
    double bits_per_int{};
    double compression_ratio{};
    double compression_throughput_mbs{};
    double decompression_throughput_mbs{};
    double random_access_ns{};

    static void print_header(std::ostream &out) {
        out << "compressor,dataset,num_lists,total_ints,uncompressed_bits,"
            << "compressed_bits,bits_per_int,compression_ratio,"
            << "compression_throughput_mbs,decompression_throughput_mbs,"
            << "random_access_ns" << std::endl;
    }

    void print(std::ostream &out) const {
        out << std::fixed << std::setprecision(4);
        out << compressor << "," << dataset << "," << num_lists << ","
            << total_ints << "," << uncompressed_bits << ","
            << compressed_bits << "," << bits_per_int << ","
            << compression_ratio << "," << compression_throughput_mbs << ","
            << decompression_throughput_mbs << "," << random_access_ns
            << std::endl;
    }
};

// ============================================================================
// GEF Benchmark (on posting lists)
// ============================================================================

static constexpr size_t GEF_UNIFORM_PARTITION_SIZE = 32000;

template <typename GEFType>
IIBenchmarkResult benchmark_gef_ii(const std::string &compressor_name,
                                   const InvertedIndexDataset &dataset) {
    IIBenchmarkResult result;
    result.compressor = compressor_name;
    result.dataset = dataset.name;
    result.num_lists = dataset.lists.size();
    result.total_ints = dataset.total_ints;
    result.uncompressed_bits = dataset.total_ints * 32;

    size_t total_compressed_bits = 0;
    double total_compression_ns = 0;
    double total_decompression_ns = 0;
    double total_ra_ns = 0;
    size_t total_ra_queries = 0;

    std::mt19937 rng(42);

    for (const auto &pl : dataset.lists) {
        if (pl.docids.size() < 2) continue;

        // Convert to int64_t for GEF
        std::vector<int64_t> data(pl.docids.begin(), pl.docids.end());

        // Compression
        auto t1 = std::chrono::high_resolution_clock::now();
        GEFType compressor(data);
        compiler_barrier();
        auto t2 = std::chrono::high_resolution_clock::now();
        total_compression_ns +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1)
                .count();

        total_compressed_bits += compressor.size_in_bytes() * 8;

        // Full decompression
        std::vector<int64_t> decompressed(data.size());
        t1 = std::chrono::high_resolution_clock::now();
        compressor.get_elements(0, data.size(), decompressed);
        compiler_barrier();
        t2 = std::chrono::high_resolution_clock::now();
        do_not_optimize(decompressed);
        total_decompression_ns +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1)
                .count();

        // Random access (sample up to 1000 per list)
        size_t n_queries = std::min<size_t>(1000, data.size());
        std::uniform_int_distribution<size_t> dist(0, data.size() - 1);
        std::vector<size_t> ra_indices(n_queries);
        for (auto &idx : ra_indices) idx = dist(rng);

        volatile int64_t ra_sum = 0;
        t1 = std::chrono::high_resolution_clock::now();
        for (auto idx : ra_indices) {
            ra_sum += compressor[idx];
        }
        compiler_barrier();
        t2 = std::chrono::high_resolution_clock::now();
        do_not_optimize(ra_sum);
        total_ra_ns +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1)
                .count();
        total_ra_queries += n_queries;
    }

    result.compressed_bits = total_compressed_bits;
    result.bits_per_int =
        static_cast<double>(total_compressed_bits) / dataset.total_ints;
    result.compression_ratio =
        static_cast<double>(total_compressed_bits) / result.uncompressed_bits;
    result.compression_throughput_mbs =
        (dataset.total_ints * sizeof(uint32_t) / 1024.0 / 1024.0) /
        (total_compression_ns / 1e9);
    result.decompression_throughput_mbs =
        (dataset.total_ints * sizeof(uint32_t) / 1024.0 / 1024.0) /
        (total_decompression_ns / 1e9);
    result.random_access_ns =
        total_ra_queries > 0
            ? static_cast<double>(total_ra_ns) / total_ra_queries
            : 0;

    return result;
}

// ============================================================================
// FastPFOR Benchmark (on posting lists, delta-encoded)
// ============================================================================

IIBenchmarkResult
benchmark_fastpfor_ii(const std::string &codec_name,
                      const InvertedIndexDataset &dataset) {
    IIBenchmarkResult result;
    result.compressor = "fastpfor_" + codec_name;
    result.dataset = dataset.name;
    result.num_lists = dataset.lists.size();
    result.total_ints = dataset.total_ints;
    result.uncompressed_bits = dataset.total_ints * 32;

    FastPForLib::CODECFactory factory;
    auto codec = factory.getFromName(codec_name);
    if (!codec) {
        std::cerr << "Unknown FastPFOR codec: " << codec_name << std::endl;
        return result;
    }

    size_t total_compressed_bits = 0;
    double total_compression_ns = 0;
    double total_decompression_ns = 0;
    double total_ra_ns = 0;
    size_t total_ra_queries = 0;

    std::mt19937 rng(42);

    for (const auto &pl : dataset.lists) {
        if (pl.docids.size() < 2) continue;

        size_t n = pl.docids.size();

        // Delta encode
        std::vector<uint32_t> delta_data(pl.docids.begin(), pl.docids.end());
        for (size_t i = n - 1; i > 0; --i) {
            delta_data[i] -= delta_data[i - 1];
        }

        // Compress
        std::vector<uint32_t> compressed(n + 1024);
        size_t compressed_size = compressed.size();

        auto t1 = std::chrono::high_resolution_clock::now();
        codec->encodeArray(delta_data.data(), n, compressed.data(),
                           compressed_size);
        compiler_barrier();
        auto t2 = std::chrono::high_resolution_clock::now();
        total_compression_ns +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1)
                .count();

        compressed.resize(compressed_size);
        total_compressed_bits += compressed_size * 32;

        // Decompress
        std::vector<uint32_t> recovered(n);
        size_t recovered_size = n;

        t1 = std::chrono::high_resolution_clock::now();
        codec->decodeArray(compressed.data(), compressed_size, recovered.data(),
                           recovered_size);
        compiler_barrier();
        t2 = std::chrono::high_resolution_clock::now();
        do_not_optimize(recovered);
        total_decompression_ns +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1)
                .count();

        // Random access: decompress full block then access
        // (FastPFOR doesn't support true random access)
        size_t n_queries = std::min<size_t>(1000, n);
        std::uniform_int_distribution<size_t> dist(0, n - 1);
        std::vector<size_t> ra_indices(n_queries);
        for (auto &idx : ra_indices) idx = dist(rng);

        volatile uint32_t ra_sum = 0;
        t1 = std::chrono::high_resolution_clock::now();
        // Decompress then look up
        std::vector<uint32_t> ra_buffer(n);
        size_t ra_recovered = n;
        codec->decodeArray(compressed.data(), compressed_size, ra_buffer.data(),
                           ra_recovered);
        // Undo delta to get original values
        for (size_t i = 1; i < ra_recovered; ++i) {
            ra_buffer[i] += ra_buffer[i - 1];
        }
        for (auto idx : ra_indices) {
            ra_sum += ra_buffer[idx];
        }
        compiler_barrier();
        t2 = std::chrono::high_resolution_clock::now();
        do_not_optimize(ra_sum);
        total_ra_ns +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1)
                .count();
        total_ra_queries += n_queries;
    }

    result.compressed_bits = total_compressed_bits;
    result.bits_per_int =
        static_cast<double>(total_compressed_bits) / dataset.total_ints;
    result.compression_ratio =
        static_cast<double>(total_compressed_bits) / result.uncompressed_bits;
    result.compression_throughput_mbs =
        (dataset.total_ints * sizeof(uint32_t) / 1024.0 / 1024.0) /
        (total_compression_ns / 1e9);
    result.decompression_throughput_mbs =
        (dataset.total_ints * sizeof(uint32_t) / 1024.0 / 1024.0) /
        (total_decompression_ns / 1e9);
    result.random_access_ns =
        total_ra_queries > 0
            ? static_cast<double>(total_ra_ns) / total_ra_queries
            : 0;

    return result;
}

// ============================================================================
// sux-rs CompIntList Benchmark (on posting list values directly)
// ============================================================================

IIBenchmarkResult
benchmark_comp_int_list(const InvertedIndexDataset &dataset) {
    IIBenchmarkResult result;
    result.compressor = "comp_int_list";
    result.dataset = dataset.name;
    result.num_lists = dataset.lists.size();
    result.total_ints = dataset.total_ints;
    result.uncompressed_bits = dataset.total_ints * 32;

    size_t total_compressed_bits = 0;
    double total_compression_ns = 0;
    double total_decompression_ns = 0;
    double total_ra_ns = 0;
    size_t total_ra_queries = 0;

    std::mt19937 rng(42);

    for (const auto &pl : dataset.lists) {
        if (pl.docids.size() < 2) continue;

        size_t n = pl.docids.size();

        // Convert to uint64_t
        std::vector<uint64_t> values(pl.docids.begin(), pl.docids.end());
        uint64_t min_val =
            *std::min_element(values.begin(), values.end());

        // Compress
        auto t1 = std::chrono::high_resolution_clock::now();
        auto *handle =
            comp_int_list_new(values.data(), values.size(), min_val);
        compiler_barrier();
        auto t2 = std::chrono::high_resolution_clock::now();
        total_compression_ns +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1)
                .count();

        if (!handle) {
            std::cerr << "comp_int_list_new failed for list " << pl.name
                      << std::endl;
            continue;
        }

        total_compressed_bits += comp_int_list_mem_size(handle) * 8;

        // Full decompression
        std::vector<uint64_t> decompressed(n);
        t1 = std::chrono::high_resolution_clock::now();
        comp_int_list_get_range(handle, 0, n, decompressed.data());
        compiler_barrier();
        t2 = std::chrono::high_resolution_clock::now();
        do_not_optimize(decompressed);
        total_decompression_ns +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1)
                .count();

        // Random access
        size_t n_queries = std::min<size_t>(1000, n);
        std::uniform_int_distribution<size_t> dist(0, n - 1);
        std::vector<size_t> ra_indices(n_queries);
        for (auto &idx : ra_indices) idx = dist(rng);

        volatile uint64_t ra_sum = 0;
        t1 = std::chrono::high_resolution_clock::now();
        for (auto idx : ra_indices) {
            ra_sum += comp_int_list_get(handle, idx);
        }
        compiler_barrier();
        t2 = std::chrono::high_resolution_clock::now();
        do_not_optimize(ra_sum);
        total_ra_ns +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1)
                .count();
        total_ra_queries += n_queries;

        comp_int_list_free(handle);
    }

    result.compressed_bits = total_compressed_bits;
    result.bits_per_int =
        static_cast<double>(total_compressed_bits) / dataset.total_ints;
    result.compression_ratio =
        static_cast<double>(total_compressed_bits) / result.uncompressed_bits;
    result.compression_throughput_mbs =
        (dataset.total_ints * sizeof(uint32_t) / 1024.0 / 1024.0) /
        (total_compression_ns / 1e9);
    result.decompression_throughput_mbs =
        (dataset.total_ints * sizeof(uint32_t) / 1024.0 / 1024.0) /
        (total_decompression_ns / 1e9);
    result.random_access_ns =
        total_ra_queries > 0
            ? static_cast<double>(total_ra_ns) / total_ra_queries
            : 0;

    return result;
}

// ============================================================================
// sux-rs PrefixSumIntList Benchmark (on gaps of posting lists)
// ============================================================================

IIBenchmarkResult
benchmark_prefix_sum_int_list(const InvertedIndexDataset &dataset) {
    IIBenchmarkResult result;
    result.compressor = "prefix_sum_int_list";
    result.dataset = dataset.name;
    result.num_lists = dataset.lists.size();
    result.total_ints = dataset.total_ints;
    result.uncompressed_bits = dataset.total_ints * 32;

    size_t total_compressed_bits = 0;
    double total_compression_ns = 0;
    double total_decompression_ns = 0;
    double total_ra_ns = 0;
    size_t total_ra_queries = 0;

    std::mt19937 rng(42);

    for (const auto &pl : dataset.lists) {
        if (pl.docids.size() < 2) continue;

        size_t n = pl.docids.size();

        // Compute gaps (delta encoding) as uint64_t
        std::vector<uint64_t> gaps(n);
        gaps[0] = pl.docids[0];
        for (size_t i = 1; i < n; ++i) {
            gaps[i] = pl.docids[i] - pl.docids[i - 1];
        }

        // Compress
        auto t1 = std::chrono::high_resolution_clock::now();
        auto *handle =
            prefix_sum_int_list_new(gaps.data(), gaps.size());
        compiler_barrier();
        auto t2 = std::chrono::high_resolution_clock::now();
        total_compression_ns +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1)
                .count();

        if (!handle) {
            std::cerr << "prefix_sum_int_list_new failed for list " << pl.name
                      << std::endl;
            continue;
        }

        total_compressed_bits += prefix_sum_int_list_mem_size(handle) * 8;

        // Full decompression
        std::vector<uint64_t> decompressed(n);
        t1 = std::chrono::high_resolution_clock::now();
        prefix_sum_int_list_get_range(handle, 0, n, decompressed.data());
        compiler_barrier();
        t2 = std::chrono::high_resolution_clock::now();
        do_not_optimize(decompressed);
        total_decompression_ns +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1)
                .count();

        // Random access
        size_t n_queries = std::min<size_t>(1000, n);
        std::uniform_int_distribution<size_t> dist(0, n - 1);
        std::vector<size_t> ra_indices(n_queries);
        for (auto &idx : ra_indices) idx = dist(rng);

        volatile uint64_t ra_sum = 0;
        t1 = std::chrono::high_resolution_clock::now();
        for (auto idx : ra_indices) {
            ra_sum += prefix_sum_int_list_get(handle, idx);
        }
        compiler_barrier();
        t2 = std::chrono::high_resolution_clock::now();
        do_not_optimize(ra_sum);
        total_ra_ns +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1)
                .count();
        total_ra_queries += n_queries;

        prefix_sum_int_list_free(handle);
    }

    result.compressed_bits = total_compressed_bits;
    result.bits_per_int =
        static_cast<double>(total_compressed_bits) / dataset.total_ints;
    result.compression_ratio =
        static_cast<double>(total_compressed_bits) / result.uncompressed_bits;
    result.compression_throughput_mbs =
        (dataset.total_ints * sizeof(uint32_t) / 1024.0 / 1024.0) /
        (total_compression_ns / 1e9);
    result.decompression_throughput_mbs =
        (dataset.total_ints * sizeof(uint32_t) / 1024.0 / 1024.0) /
        (total_decompression_ns / 1e9);
    result.random_access_ns =
        total_ra_queries > 0
            ? static_cast<double>(total_ra_ns) / total_ra_queries
            : 0;

    return result;
}

// ============================================================================
// Main
// ============================================================================

void print_usage(const char *prog_name) {
    std::cerr << "Usage: " << prog_name
              << " [options] <dataset_directory>" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  -o <file>  Output CSV file (default: stdout)" << std::endl;
    std::cerr << "  -c <list>  Comma-separated list of compressors"
              << std::endl;
    std::cerr << "             GEF: rle_gef, u_gef_approximate, u_gef_optimal,"
              << std::endl;
    std::cerr << "                  b_gef_approximate, b_gef_optimal,"
              << std::endl;
    std::cerr << "                  b_star_gef_approximate, b_star_gef_optimal"
              << std::endl;
    std::cerr << "             FastPFOR: fastpfor_simdfastpfor256, "
                 "fastpfor_optpfor,"
              << std::endl;
    std::cerr
        << "                       fastpfor_vbyte, fastpfor_streamvbyte,"
        << std::endl;
    std::cerr
        << "                       fastpfor_simple8b, fastpfor_simdbinarypacking"
        << std::endl;
    std::cerr << "             sux-rs: comp_int_list, prefix_sum_int_list"
              << std::endl;
    std::cerr << "  -h         Show this help" << std::endl;
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
    std::string output_file;
    std::vector<std::string> compressors = {
        "rle_gef",
        "u_gef_approximate",
        "u_gef_optimal",
        "b_gef_approximate",
        "b_gef_optimal",
        "b_star_gef_approximate",
        "b_star_gef_optimal",
        "fastpfor_simdfastpfor256",
        "fastpfor_optpfor",
        "fastpfor_vbyte",
        "fastpfor_streamvbyte",
        "fastpfor_simple8b",
        "fastpfor_simdbinarypacking",
        "comp_int_list",
        "prefix_sum_int_list",
    };
    std::string input_path;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-o" && i + 1 < argc) {
            output_file = argv[++i];
        } else if (arg == "-c" && i + 1 < argc) {
            compressors = split_string(argv[++i], ',');
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

    // Collect dataset directories: if input_path is a directory of directories,
    // each subdirectory is a dataset. If it directly contains .txt files, treat
    // it as a single dataset.
    std::vector<std::string> dataset_dirs;

    bool has_txt_files = false;
    bool has_subdirs = false;
    for (const auto &entry : std::filesystem::directory_iterator(input_path)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt")
            has_txt_files = true;
        if (entry.is_directory())
            has_subdirs = true;
    }

    if (has_txt_files) {
        dataset_dirs.push_back(input_path);
    } else if (has_subdirs) {
        for (const auto &entry :
             std::filesystem::directory_iterator(input_path)) {
            if (entry.is_directory()) {
                dataset_dirs.push_back(entry.path().string());
            }
        }
        std::sort(dataset_dirs.begin(), dataset_dirs.end());
    } else {
        std::cerr << "No .txt files or subdirectories found in " << input_path
                  << std::endl;
        return 1;
    }

    std::ostream *out = &std::cout;
    std::ofstream file_out;
    if (!output_file.empty()) {
        file_out.open(output_file);
        out = &file_out;
    }

    bool header_printed = false;

    for (const auto &dir : dataset_dirs) {
        std::cerr << "Loading dataset: " << dir << std::endl;

        InvertedIndexDataset dataset;
        try {
            dataset = load_dataset_directory(dir);
        } catch (const std::exception &e) {
            std::cerr << "  Error: " << e.what() << std::endl;
            continue;
        }

        std::cerr << "  Loaded " << dataset.lists.size() << " posting lists, "
                  << dataset.total_ints << " total integers" << std::endl;

        if (dataset.total_ints == 0) {
            std::cerr << "  Skipping empty dataset" << std::endl;
            continue;
        }

        for (const auto &comp : compressors) {
            std::cerr << "  Running " << comp << "..." << std::flush;

            IIBenchmarkResult result;

            try {
                if (comp == "rle_gef") {
                    using T =
                        gef::RLE_GEF<int64_t, GEF_UNIFORM_PARTITION_SIZE>;
                    result = benchmark_gef_ii<T>(comp, dataset);
                } else if (comp == "u_gef_approximate") {
                    using T = gef::U_GEF_APPROXIMATE<int64_t,
                                                     GEF_UNIFORM_PARTITION_SIZE>;
                    result = benchmark_gef_ii<T>(comp, dataset);
                } else if (comp == "u_gef_optimal") {
                    using T =
                        gef::U_GEF<int64_t, GEF_UNIFORM_PARTITION_SIZE>;
                    result = benchmark_gef_ii<T>(comp, dataset);
                } else if (comp == "b_gef_approximate") {
                    using T = gef::B_GEF_APPROXIMATE<int64_t,
                                                     GEF_UNIFORM_PARTITION_SIZE>;
                    result = benchmark_gef_ii<T>(comp, dataset);
                } else if (comp == "b_gef_optimal") {
                    using T =
                        gef::B_GEF<int64_t, GEF_UNIFORM_PARTITION_SIZE>;
                    result = benchmark_gef_ii<T>(comp, dataset);
                } else if (comp == "b_star_gef_approximate") {
                    using T =
                        gef::B_STAR_GEF_APPROXIMATE<int64_t,
                                                    GEF_UNIFORM_PARTITION_SIZE>;
                    result = benchmark_gef_ii<T>(comp, dataset);
                } else if (comp == "b_star_gef_optimal") {
                    using T = gef::B_STAR_GEF<int64_t,
                                              GEF_UNIFORM_PARTITION_SIZE>;
                    result = benchmark_gef_ii<T>(comp, dataset);
                } else if (comp.starts_with("fastpfor_")) {
                    std::string codec_name = comp.substr(9);
                    result = benchmark_fastpfor_ii(codec_name, dataset);
                } else if (comp == "comp_int_list") {
                    result = benchmark_comp_int_list(dataset);
                } else if (comp == "prefix_sum_int_list") {
                    result = benchmark_prefix_sum_int_list(dataset);
                } else {
                    std::cerr << " unknown compressor, skipping" << std::endl;
                    continue;
                }

                if (!header_printed) {
                    IIBenchmarkResult::print_header(*out);
                    header_printed = true;
                }
                result.print(*out);
                std::cerr << " done (bpi=" << std::fixed << std::setprecision(2)
                          << result.bits_per_int << ")" << std::endl;

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
