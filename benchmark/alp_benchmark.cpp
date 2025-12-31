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

    // Sample buffer - sized generously to avoid any overflow
    // ALP samples at most ROWGROUP_VECTOR_SAMPLES vectors, each with SAMPLES_PER_VECTOR samples
    // Add extra margin for safety
    constexpr size_t SAMPLE_BUF_SIZE = alp::config::ROWGROUP_SIZE;  // Use full rowgroup size for safety
    std::vector<double> sample_buf(SAMPLE_BUF_SIZE);

    // Compression
    auto t1 = std::chrono::high_resolution_clock::now();
    {
        // Process data in rowgroups
        constexpr size_t ROWGROUP_SIZE = alp::config::ROWGROUP_SIZE;  // 100 vectors = 102400 values
        const size_t num_rowgroups = (n_full + ROWGROUP_SIZE - 1) / ROWGROUP_SIZE;
        
        for (size_t rg = 0; rg < num_rowgroups; ++rg) {
            const size_t rg_start = rg * ROWGROUP_SIZE;
            const size_t rg_end = std::min(rg_start + ROWGROUP_SIZE, n_full);
            
            // Create fresh state for each rowgroup to avoid any state contamination
            alp::state<double> stt;
            
            // Initialize encoder once per rowgroup
            // Important: pass the total data size (n) and the starting offset (rg_start)
            alp::encoder<double>::init(data.data(), rg_start, n, sample_buf.data(), stt);
            
            // Calculate vector range for this rowgroup
            const size_t rg_vec_start = rg_start / VEC;
            const size_t rg_vec_end = rg_end / VEC;  // Only complete vectors
            
            // Skip ALP_RD or if no valid combinations found - fall back to storing raw values
            if (stt.scheme == alp::Scheme::ALP_RD || stt.best_k_combinations.empty()) {
                for (size_t v = rg_vec_start; v < rg_vec_end; ++v) {
                    const double* vec_in = data.data() + v * VEC;
                    int64_t* vec_out = encoded.data() + v * VEC;
                    
                    factors[v] = 0;
                    exponents[v] = 0;
                    exc_counts[v] = static_cast<uint16_t>(VEC);
                    exceptions[v].assign(vec_in, vec_in + VEC);
                    exc_positions[v].resize(VEC);
                    for (size_t i = 0; i < VEC; ++i) {
                        exc_positions[v][i] = static_cast<uint16_t>(i);
                        vec_out[i] = 0;  // placeholder
                    }
                }
                continue;
            }
            
            // Process each vector in this rowgroup with ALP encoding
            for (size_t v = rg_vec_start; v < rg_vec_end; ++v) {
                const double* vec_in = data.data() + v * VEC;
                int64_t* vec_out = encoded.data() + v * VEC;
                
                // Use aligned buffers for exceptions - sized for worst case
                alignas(64) double exc_buf[VEC] = {};
                alignas(64) uint16_t pos_buf[VEC] = {};
                uint16_t exc_cnt = 0;

                alp::encoder<double>::encode(vec_in, exc_buf, pos_buf, &exc_cnt, vec_out, stt);

                factors[v] = stt.fac;
                exponents[v] = stt.exp;
                exc_counts[v] = exc_cnt;
                if (exc_cnt > 0) {
                    exceptions[v].assign(exc_buf, exc_buf + exc_cnt);
                    exc_positions[v].assign(pos_buf, pos_buf + exc_cnt);
                }
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

            if (exc_counts[v] > 0 && !exceptions[v].empty()) {
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
