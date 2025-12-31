// Shared types/utilities used across benchmark translation units.
// This exists so we can compile some benchmarks (e.g., ALP) with a different compiler
// while still sharing the result/inputs structs.
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

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

template<class T>
inline void do_not_optimize(T const &value) {
    asm volatile("" : : "r,m"(value) : "memory");
}


