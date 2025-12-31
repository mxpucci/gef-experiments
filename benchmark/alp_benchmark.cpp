// Minimal ALP benchmark - just encode/decode the data and measure.
// No custom block management, no random access (ALP is a columnar codec).

#include "benchmark_common.hpp"
#include "alp.hpp"

#include <chrono>
#include <cstring>
#include <iostream>

BenchmarkResult benchmark_alp(const BenchmarkData &bench_data,
                              const std::vector<size_t> &range_sizes) {
    BenchmarkResult result;
    result.compressor = "ALP";
    result.dataset = bench_data.filename;

    const auto& data = bench_data.double_data;
    const size_t n = data.size();
    
    result.num_values = n;
    result.uncompressed_bits = bench_data.uncompressed_bits;

    // ALP works on vectors of 1024 doubles. We'll process full vectors only.
    constexpr size_t VEC = alp::config::VECTOR_SIZE;
    const size_t n_full = (n / VEC) * VEC;
    const size_t num_vecs = n_full / VEC;

    if (num_vecs == 0) {
        std::cerr << "ALP: dataset too small (< " << VEC << " values)" << std::endl;
        result.compression_ratio = 1.0;
        return result;
    }

    // Compressed storage: encoded integers + minimal metadata per vector
    std::vector<int64_t> encoded(n_full);
    std::vector<uint8_t> factors(num_vecs);
    std::vector<uint8_t> exponents(num_vecs);
    std::vector<uint16_t> exc_counts(num_vecs);
    std::vector<std::vector<double>> exceptions(num_vecs);
    std::vector<std::vector<uint16_t>> exc_positions(num_vecs);

    // Compression
    auto t1 = std::chrono::high_resolution_clock::now();
    {
        alp::state<double> stt;
        std::array<double, VEC> sample_buf{};

        for (size_t v = 0; v < num_vecs; ++v) {
            const double* vec_in = data.data() + v * VEC;
            int64_t* vec_out = encoded.data() + v * VEC;

            // Initialize encoder for this vector (simplified: one vector = one rowgroup)
            alp::encoder<double>::init(data.data(), v * VEC, VEC, sample_buf.data(), stt);

            // Skip ALP_RD for simplicity - just use standard ALP
            if (stt.scheme == alp::Scheme::ALP_RD) {
                stt.scheme = alp::Scheme::ALP;
                alp::encoder<double>::init(data.data(), v * VEC, VEC, sample_buf.data(), stt);
            }

            alignas(64) std::array<double, VEC> exc_buf{};
            alignas(64) std::array<uint16_t, VEC> pos_buf{};
            uint16_t exc_cnt = 0;

            alp::encoder<double>::encode(vec_in, exc_buf.data(), pos_buf.data(), &exc_cnt, vec_out, stt);

            factors[v] = stt.fac;
            exponents[v] = stt.exp;
            exc_counts[v] = exc_cnt;
            if (exc_cnt > 0) {
                exceptions[v].assign(exc_buf.begin(), exc_buf.begin() + exc_cnt);
                exc_positions[v].assign(pos_buf.begin(), pos_buf.begin() + exc_cnt);
            }
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto comp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();

    // Estimate compressed size (encoded ints + metadata)
    size_t comp_bytes = n_full * sizeof(int64_t);  // encoded integers
    comp_bytes += num_vecs * 2;                     // factors + exponents
    for (size_t v = 0; v < num_vecs; ++v) {
        comp_bytes += 2;  // exc_count
        comp_bytes += exceptions[v].size() * sizeof(double);
        comp_bytes += exc_positions[v].size() * sizeof(uint16_t);
    }

    result.compressed_bits = comp_bytes * 8;
    result.compression_ratio = static_cast<double>(result.compressed_bits) / result.uncompressed_bits;
    result.compression_throughput_mbs = (n * sizeof(double) / 1024.0 / 1024.0) / (comp_ns / 1e9);

    // Decompression
    std::vector<double> decompressed(n_full);
    t1 = std::chrono::high_resolution_clock::now();
    {
        for (size_t v = 0; v < num_vecs; ++v) {
            const int64_t* vec_in = encoded.data() + v * VEC;
            double* vec_out = decompressed.data() + v * VEC;

            alp::decoder<double>::decode(vec_in, factors[v], exponents[v], vec_out);

            if (exc_counts[v] > 0) {
                alp::exp_c_t ec = exc_counts[v];
                alp::decoder<double>::patch_exceptions(
                    vec_out,
                    exceptions[v].data(),
                    exc_positions[v].data(),
                    &ec
                );
            }
        }
    }
    t2 = std::chrono::high_resolution_clock::now();
    do_not_optimize(decompressed);
    auto decomp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.decompression_throughput_mbs = (n * sizeof(double) / 1024.0 / 1024.0) / (decomp_ns / 1e9);

    // Verify
    for (size_t i = 0; i < n_full; ++i) {
        if (data[i] != decompressed[i]) {
            std::cerr << "ALP verification failed at index " << i << std::endl;
            break;
        }
    }

    // Random access / range queries: N/A for columnar codec
    result.random_access_ns = 0;
    result.random_access_mbs = 0;
    for (auto r : range_sizes) {
        result.range_query_throughputs.emplace_back(r, 0.0);
    }

    return result;
}
