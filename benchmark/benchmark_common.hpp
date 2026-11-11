// Shared types/utilities used across benchmark translation units.
// This exists so we can compile some benchmarks (e.g., ALP) with a different compiler
// while still sharing the result/inputs structs.
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <atomic>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <ostream>
#include <type_traits>
#include <string>
#include <utility>
#include <vector>

// Range-query configuration shared by every codec (CLI: --range-*).
struct RangeQueryOptions {
    size_t queries = 10000;        // queries per range size, identical for every codec
    uint64_t seed = 1234;          // mt19937 seed of the shared random starts
    size_t trials = 1;             // timed repetitions of all queries; CSV reports the median
    size_t batch_bytes = 256 * 1024; // output arena per timed batch (see range_queries.hpp)
    bool verify = true;            // untimed check of every query against full decompression
};

// One timed trial of all queries of one range size (written by --range-trials-csv).
struct RangeTrial {
    size_t range{};
    size_t trial{};
    size_t queries{};
    size_t value_bytes{};          // bytes per returned value (8 for every current codec)
    uint64_t elapsed_ns{};
    size_t verified_queries{};
    size_t mismatched_queries{};
};

struct BenchmarkResult {
    std::string compressor;
    std::string dataset;
    size_t num_values{};
    size_t uncompressed_bits{};
    size_t compressed_bits{};
    double compression_ratio{};
    double compression_throughput_mbs{};
    double decompression_throughput_mbs{};
    double random_access_ns{};
    double random_access_mbs{};
    std::vector<std::pair<size_t, double>> range_query_throughputs; // (range_size, MB/s)
    std::string input_conversion{"none"};
    size_t input_changed_values{};
    size_t roundtrip_mismatches{};
    std::string validation_status{"not_checked"};
    std::string access_layout{"unspecified"};
    size_t compression_unit_values{};
    size_t native_page_values{};
    size_t bitwise_mismatches{}; // informational: values whose bits differ
    std::vector<RangeTrial> range_trials;  // not part of the main CSV
    size_t range_mismatched_queries{};     // not part of the main CSV

    void print_header(std::ostream &out) const;
    void print(std::ostream &out) const;
};

struct BenchmarkData {
    std::string filename;
    std::vector<int64_t> raw_data;
    std::vector<int64_t> shifted_data;
    std::vector<double> double_data;
    int64_t decimals{};
    int64_t min_val{};
    size_t uncompressed_bits{};

    // Indices
    std::vector<size_t> random_indices;
    // Map range size to indices
    std::map<size_t, std::vector<size_t>> range_query_indices;
    RangeQueryOptions range_options;

    // Defined in a strict-IEEE, non-LTO translation unit: callers may use
    // fast-math, but reciprocal multiplication changes the decimal input bits.
    void convert_shifted_to_double();
};

// Uniform round-trip policy for every codec. A decoded value counts as a
// mismatch (roundtrip_mismatches, validation_status=failed) when it is not
// bit-identical to the input AND |expected - decoded| > 1e-9; for integer
// codecs this is plain inequality. Mismatches only produce warnings: metrics
// are still emitted. bitwise_mismatches additionally counts every value whose
// binary64/int64 bits differ, including differences within the tolerance.
inline constexpr double ROUNDTRIP_TOLERANCE = 1e-9;

template<class T>
inline bool roundtrip_value_mismatch(const T &expected, const T &decoded) {
    if constexpr (std::is_floating_point_v<T>) {
        // NaN (or any non-finite difference) is never within the tolerance.
        const double difference = std::fabs(static_cast<double>(expected) - static_cast<double>(decoded));
        return !(difference <= ROUNDTRIP_TOLERANCE);
    } else {
        return expected != decoded;
    }
}

// Accumulates mismatches for decoded[0..n) against expected[0..n); call once
// per block with `first_index` = global index of element 0, then call
// finish_roundtrip_validation() once.
template<class T>
inline void check_roundtrip(BenchmarkResult &result, const T *expected, const T *decoded,
                            size_t n, size_t first_index = 0) {
    if (n == 0 || std::memcmp(expected, decoded, n * sizeof(T)) == 0) return;
    for (size_t i = 0; i < n; ++i) {
        if (std::memcmp(&expected[i], &decoded[i], sizeof(T)) == 0) continue;
        ++result.bitwise_mismatches;
        if (!roundtrip_value_mismatch(expected[i], decoded[i])) continue;
        if (result.roundtrip_mismatches++ == 0) {
            const auto precision = std::cerr.precision(17);
            std::cerr << "\nWarning: " << result.compressor << ": first round-trip mismatch at index "
                      << first_index + i << " (expected " << expected[i] << ", decoded " << decoded[i] << ")\n";
            std::cerr.precision(precision);
        }
    }
}

template<class T>
inline void check_roundtrip(BenchmarkResult &result, const std::vector<T> &expected,
                            const std::vector<T> &decoded) {
    if (expected.size() != decoded.size()) {
        std::cerr << "\nWarning: " << result.compressor << ": decoded " << decoded.size()
                  << " values, expected " << expected.size() << "\n";
        result.roundtrip_mismatches += expected.size() > decoded.size() ? expected.size() - decoded.size()
                                                                         : decoded.size() - expected.size();
    }
    check_roundtrip(result, expected.data(), decoded.data(), std::min(expected.size(), decoded.size()));
}

inline void finish_roundtrip_validation(BenchmarkResult &result) {
    result.validation_status = result.roundtrip_mismatches ? "failed" : "passed";
    if (result.roundtrip_mismatches)
        std::cerr << "\nWarning: " << result.compressor << " reconstructed " << result.roundtrip_mismatches
                  << "/" << result.num_values << " values with |error| > " << ROUNDTRIP_TOLERANCE
                  << "; emitting metrics with validation_status=failed.\n";
    else if (result.bitwise_mismatches)
        std::cerr << "\nNote: " << result.compressor << ": " << result.bitwise_mismatches
                  << " values differ bitwise but within " << ROUNDTRIP_TOLERANCE << "\n";
}

template<class T>
inline void do_not_optimize(T const &value) {
    // Prefer a portable implementation when building with strict toolchains.
    // GNU/Clang inline asm gives the strongest "use" semantics; otherwise fall back
    // to a compiler-only fence + volatile address-take.
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "g"(value) : "memory");
#else
    std::atomic_signal_fence(std::memory_order_seq_cst);
    auto const volatile* p = &value;
    (void)p;
    std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

// Portable compiler-only barrier to prevent reordering across timing points.
// Prefer this over inline `asm volatile("" ::: "memory")` in benchmarks to reduce
// toolchain-specific build issues (e.g., -Werror/-pedantic).
inline void compiler_barrier() {
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

// External benchmark functions (compiled with different compilers for compatibility)
// ALP: compiled with clang++ for AVX-512 support
BenchmarkResult benchmark_alp(const BenchmarkData &bench_data,
                              const std::vector<size_t> &range_sizes);

// LeCo: compiled with GCC 11 due to GCC 13 std::is_integral<__uint128_t> conflict
BenchmarkResult benchmark_leco(const BenchmarkData &bench_data,
                               const std::vector<size_t> &range_sizes);

// GEF: compiled in its own translation unit (gef_benchmark.cpp) with GEF's
// fast-math flags; competitor code in lossless_benchmark.cpp is strict IEEE.
inline constexpr size_t GEF_UNIFORM_PARTITION_SIZE = 32000;
bool is_gef_compressor(const std::string &name);
BenchmarkResult benchmark_gef_compressor(const std::string &name, const BenchmarkData &bench_data,
                                         const std::vector<size_t> &range_sizes);

BenchmarkResult benchmark_camel(const BenchmarkData &, const std::vector<size_t> &, size_t block_size = 1000);
#if defined(GEF_ENABLE_PEF)
// Partitioned Elias-Fano (ds2i) on the prefix sums of shifted_data; benchmark/pef_benchmark.cpp.
BenchmarkResult benchmark_pef(const BenchmarkData &, const std::vector<size_t> &);
#endif
#if defined(GEF_ENABLE_FASTLANES)
// FastLanes FFOR / transposed delta on 1,024-value vectors; benchmark/fastlanes_benchmark.cpp.
BenchmarkResult benchmark_fastlanes(const BenchmarkData &, const std::vector<size_t> &, bool delta);
#endif
