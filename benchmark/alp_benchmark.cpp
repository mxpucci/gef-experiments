// ALP benchmark - uses the ALP library for floating-point compression.

#include "benchmark_common.hpp"
#include "alp.hpp"

#include <chrono>
#include <cstring>
#include <iostream>

namespace {

// Helper to decode a single vector
inline void decode_vector(
    const int64_t* encoded_vec,
    uint8_t factor,
    uint8_t exponent,
    uint16_t exc_count,
    const double* exceptions,
    const uint16_t* exc_positions,
    double* out_vec,
    size_t VEC)
{
    alp::decoder<double>::decode(encoded_vec, factor, exponent, out_vec);
    
    if (exc_count > 0 && exceptions != nullptr) {
        alp::exp_c_t ec = exc_count;
        alp::decoder<double>::patch_exceptions(out_vec, exceptions, exc_positions, &ec);
    }
}

} // anonymous namespace

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

    // Sample buffer - sized for full rowgroup
    constexpr size_t SAMPLE_BUF_SIZE = alp::config::ROWGROUP_SIZE;
    std::vector<double> sample_buf(SAMPLE_BUF_SIZE);

    // Compression
    auto t1 = std::chrono::high_resolution_clock::now();
    {
        constexpr size_t ROWGROUP_SIZE = alp::config::ROWGROUP_SIZE;
        const size_t num_rowgroups = (n_full + ROWGROUP_SIZE - 1) / ROWGROUP_SIZE;
        
        for (size_t rg = 0; rg < num_rowgroups; ++rg) {
            const size_t rg_start = rg * ROWGROUP_SIZE;
            const size_t rg_end = std::min(rg_start + ROWGROUP_SIZE, n_full);
            
            alp::state<double> stt;
            alp::encoder<double>::init(data.data(), rg_start, n, sample_buf.data(), stt);
            
            const size_t rg_vec_start = rg_start / VEC;
            const size_t rg_vec_end = rg_end / VEC;
            
            // Fallback for ALP_RD or no valid combinations
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
                        vec_out[i] = 0;
                    }
                }
                continue;
            }
            
            for (size_t v = rg_vec_start; v < rg_vec_end; ++v) {
                const double* vec_in = data.data() + v * VEC;
                int64_t* vec_out = encoded.data() + v * VEC;
                
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

    // Estimate compressed size
    size_t comp_bytes = n_full * sizeof(int64_t);
    comp_bytes += num_vecs * 2;  // factors + exponents
    for (size_t v = 0; v < num_vecs; ++v) {
        comp_bytes += 2;  // exc_count
        comp_bytes += exceptions[v].size() * sizeof(double);
        comp_bytes += exc_positions[v].size() * sizeof(uint16_t);
    }

    result.compressed_bits = comp_bytes * 8;
    result.compression_ratio = static_cast<double>(result.compressed_bits) / result.uncompressed_bits;
    result.compression_throughput_mbs = (n * sizeof(double) / 1024.0 / 1024.0) / (comp_ns / 1e9);

    // Full decompression
    std::vector<double> decompressed(n_full);
    t1 = std::chrono::high_resolution_clock::now();
    for (size_t v = 0; v < num_vecs; ++v) {
        decode_vector(
            encoded.data() + v * VEC,
            factors[v], exponents[v],
            exc_counts[v],
            exceptions[v].empty() ? nullptr : exceptions[v].data(),
            exc_positions[v].empty() ? nullptr : exc_positions[v].data(),
            decompressed.data() + v * VEC,
            VEC
        );
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

    // Random access: decode the vector containing the queried index
    const size_t num_ra_queries = std::min(size_t(10000), bench_data.random_indices.size());
    alignas(64) double vec_buf[VEC];
    double sum = 0;
    
    t1 = std::chrono::high_resolution_clock::now();
    for (size_t q = 0; q < num_ra_queries; ++q) {
        size_t idx = bench_data.random_indices[q];
        if (idx >= n_full) continue;
        
        size_t vec_idx = idx / VEC;
        size_t offset = idx % VEC;
        
        decode_vector(
            encoded.data() + vec_idx * VEC,
            factors[vec_idx], exponents[vec_idx],
            exc_counts[vec_idx],
            exceptions[vec_idx].empty() ? nullptr : exceptions[vec_idx].data(),
            exc_positions[vec_idx].empty() ? nullptr : exc_positions[vec_idx].data(),
            vec_buf,
            VEC
        );
        sum += vec_buf[offset];
    }
    t2 = std::chrono::high_resolution_clock::now();
    do_not_optimize(sum);
    auto ra_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    
    result.random_access_ns = static_cast<double>(ra_ns) / num_ra_queries;
    result.random_access_mbs = (num_ra_queries * sizeof(double) / 1024.0 / 1024.0) / (ra_ns / 1e9);

    // Range queries: decode all vectors overlapping the range
    const size_t num_range_queries = 1000;
    for (auto range : range_sizes) {
        if (range >= n_full) {
            result.range_query_throughputs.emplace_back(range, 0.0);
            continue;
        }
        
        const auto& range_indices = bench_data.range_query_indices.at(range);
        std::vector<double> range_buf(range);
        
        t1 = std::chrono::high_resolution_clock::now();
        size_t q_count = 0;
        for (auto start_idx : range_indices) {
            if (q_count++ >= num_range_queries) break;
            if (start_idx + range > n_full) continue;
            
            size_t start_vec = start_idx / VEC;
            size_t end_vec = (start_idx + range - 1) / VEC;
            size_t out_pos = 0;
            
            for (size_t v = start_vec; v <= end_vec && v < num_vecs; ++v) {
                decode_vector(
                    encoded.data() + v * VEC,
                    factors[v], exponents[v],
                    exc_counts[v],
                    exceptions[v].empty() ? nullptr : exceptions[v].data(),
                    exc_positions[v].empty() ? nullptr : exc_positions[v].data(),
                    vec_buf,
                    VEC
                );
                
                size_t vec_start_global = v * VEC;
                size_t skip = (v == start_vec) ? (start_idx - vec_start_global) : 0;
                size_t take = std::min(VEC - skip, range - out_pos);
                
                memcpy(range_buf.data() + out_pos, vec_buf + skip, take * sizeof(double));
                out_pos += take;
            }
            do_not_optimize(range_buf);
        }
        t2 = std::chrono::high_resolution_clock::now();
        
        auto range_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
        double throughput = (range * sizeof(double) * q_count / 1024.0 / 1024.0) / (range_ns / 1e9);
        result.range_query_throughputs.emplace_back(range, throughput);
    }

    return result;
}
