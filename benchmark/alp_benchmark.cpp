// ALP benchmark - uses the ALP library for floating-point compression.

#include "benchmark_common.hpp"
#include "alp.hpp"

#include <chrono>
#include <cstring>
#include <iostream>
#include <vector>
#include <cstdint>

namespace {

// Metadata structure mirroring the authors' alp_m struct (but adapted for in-memory)
// See: lib/ALP/publication/source_code/bench_end_to_end/src/common/runtime/Import.cpp
struct VectorMetadata {
    uint8_t bit_width;
    uint8_t factor;
    uint8_t exponent;
    int64_t ffor_base;
    uint16_t exc_count;
    
    // Offsets into the respective data vectors
    size_t ffor_offset;
    size_t exc_val_offset;
    size_t exc_pos_offset;
};

// Helper to decode a single vector using separate data streams
// Uses the fused FALP function as per official ALP end-to-end benchmark
// See: lib/ALP/publication/source_code/bench_end_to_end/src/benchmarks/alp/queries/q1.cpp
inline void decode_vector(
    const VectorMetadata& meta,
    const uint64_t* ffor_stream,
    const double* exc_val_stream,
    const uint16_t* exc_pos_stream,
    double* out_vec)
{
    // 1. FALP: Fused bit-unpacking + ALP decode in one pass (as per official implementation)
    const uint64_t* packed_ptr = ffor_stream + meta.ffor_offset;
    uint64_t base_u64 = static_cast<uint64_t>(meta.ffor_base);
    
    generated::falp::fallback::scalar::falp(
        packed_ptr,
        out_vec,
        meta.bit_width,
        &base_u64,
        meta.factor,
        meta.exponent
    );
    
    // 2. Patch Exceptions (as per official implementation)
    if (meta.exc_count > 0) {
        const double* exceptions = exc_val_stream + meta.exc_val_offset;
        const uint16_t* positions = exc_pos_stream + meta.exc_pos_offset;
        
        alp::exp_c_t ec = meta.exc_count;
        alp::decoder<double>::patch_exceptions(out_vec, exceptions, positions, &ec);
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

    constexpr size_t VEC = alp::config::VECTOR_SIZE;
    const size_t n_full = (n / VEC) * VEC;
    const size_t num_vecs = n_full / VEC;

    if (num_vecs == 0) {
        std::cerr << "ALP: dataset too small (< " << VEC << " values)" << std::endl;
        result.compression_ratio = 1.0;
        return result;
    }

    // Faithful implementation: Split data into 4 components (Streams)
    // 1. Metadata Stream
    std::vector<VectorMetadata> metadata_stream;
    metadata_stream.reserve(num_vecs);

    // 2. FFOR Bitstream (Packed Integers)
    std::vector<uint64_t> ffor_stream;
    ffor_stream.reserve(n); // Conservative estimate

    // 3. Exception Values Stream
    std::vector<double> exc_val_stream;
    exc_val_stream.reserve(n / 10); 

    // 4. Exception Positions Stream
    std::vector<uint16_t> exc_pos_stream;
    exc_pos_stream.reserve(n / 10);

    // Sample buffer
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
            
            for (size_t v = rg_vec_start; v < rg_vec_end; ++v) {
                const double* vec_in = data.data() + v * VEC;
                
                alignas(64) int64_t temp_encoded[VEC]; 
                alignas(64) double exc_buf[VEC] = {};
                alignas(64) uint16_t pos_buf[VEC] = {};
                uint16_t exc_cnt = 0;
                
                alp::bw_t bw = 0;
                int64_t for_base = 0;
                uint8_t fac = 0;
                uint8_t exp = 0;

                if (stt.scheme == alp::Scheme::ALP_RD || stt.best_k_combinations.empty()) {
                    // Fallback
                    bw = 0;
                    for_base = 0;
                    fac = 0;
                    exp = 0;
                    exc_cnt = static_cast<uint16_t>(VEC);
                    std::copy(vec_in, vec_in + VEC, exc_buf);
                    for(uint16_t i=0; i<VEC; ++i) pos_buf[i] = i;
                } else {
                    alp::encoder<double>::encode(vec_in, exc_buf, pos_buf, &exc_cnt, temp_encoded, stt);
                    alp::encoder<double>::analyze_ffor(temp_encoded, bw, &for_base);
                    fac = stt.fac;
                    exp = stt.exp;
                }

                // Store Metadata
                VectorMetadata meta;
                meta.bit_width = bw;
                meta.factor = fac;
                meta.exponent = exp;
                meta.ffor_base = for_base;
                meta.exc_count = exc_cnt;
                meta.ffor_offset = ffor_stream.size(); // Index in 64-bit words
                meta.exc_val_offset = exc_val_stream.size();
                meta.exc_pos_offset = exc_pos_stream.size();
                metadata_stream.push_back(meta);

                // Store FFOR Data
                if (bw > 0) {
                    size_t ffor_words = (VEC * bw + 63) / 64;
                    size_t current_size = ffor_stream.size();
                    ffor_stream.resize(current_size + ffor_words);
                    ffor::ffor(temp_encoded, reinterpret_cast<int64_t*>(&ffor_stream[current_size]), bw, &for_base);
                }

                // Store Exceptions
                if (exc_cnt > 0) {
                    exc_val_stream.insert(exc_val_stream.end(), exc_buf, exc_buf + exc_cnt);
                    exc_pos_stream.insert(exc_pos_stream.end(), pos_buf, pos_buf + exc_cnt);
                }
            }
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto comp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();

    // Calculate compressed size (sum of all 4 streams)
    size_t comp_bits = 0;
    // 1. Metadata: 128 bits per vector (16 bytes conservative struct size)
    comp_bits += metadata_stream.size() * 128; 
    // 2. FFOR: 64 bits per word
    comp_bits += ffor_stream.size() * 64;
    // 3. Exceptions: 64 bits per value
    comp_bits += exc_val_stream.size() * 64;
    // 4. Positions: 16 bits per position
    comp_bits += exc_pos_stream.size() * 16;
    
    result.compressed_bits = comp_bits;
    result.compression_ratio = static_cast<double>(result.compressed_bits) / result.uncompressed_bits;
    result.compression_throughput_mbs = (n * sizeof(double) / 1024.0 / 1024.0) / (comp_ns / 1e9);

    // Full decompression - with checksum to prevent optimization
    std::vector<double> decompressed(n_full);
    volatile double decomp_checksum = 0;  // volatile prevents optimization
    t1 = std::chrono::high_resolution_clock::now();
    for (size_t v = 0; v < num_vecs; ++v) {
        decode_vector(
            metadata_stream[v],
            ffor_stream.data(),
            exc_val_stream.data(),
            exc_pos_stream.data(),
            decompressed.data() + v * VEC
        );
        // Force actual computation by reading result
        decomp_checksum += decompressed[v * VEC];
    }
    compiler_barrier();  // Compiler barrier before timing ends
    t2 = std::chrono::high_resolution_clock::now();
    do_not_optimize(decompressed);
    do_not_optimize(decomp_checksum);
    auto decomp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.decompression_throughput_mbs = (n * sizeof(double) / 1024.0 / 1024.0) / (decomp_ns / 1e9);

    // Verify correctness
    for (size_t i = 0; i < n_full; ++i) {
        if (data[i] != decompressed[i]) {
            std::cerr << "ALP verification failed at index " << i << std::endl;
            break;
        }
    }
    
    // Random access - with volatile checksum to prevent optimization
    const size_t num_ra_queries = std::min(size_t(10000), bench_data.random_indices.size());
    alignas(64) double vec_buf[VEC];
    volatile double ra_sum = 0;  // volatile prevents optimization
    
    t1 = std::chrono::high_resolution_clock::now();
    for (size_t q = 0; q < num_ra_queries; ++q) {
        size_t idx = bench_data.random_indices[q];
        if (idx >= n_full) continue;
        
        size_t vec_idx = idx / VEC;
        size_t offset = idx % VEC;
        
        decode_vector(
            metadata_stream[vec_idx],
            ffor_stream.data(),
            exc_val_stream.data(),
            exc_pos_stream.data(),
            vec_buf
        );
        ra_sum += vec_buf[offset];
    }
    compiler_barrier();  // Compiler barrier before timing ends
    t2 = std::chrono::high_resolution_clock::now();
    do_not_optimize(ra_sum);
    auto ra_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();

    if (num_ra_queries == 0) {
        result.random_access_ns = 0.0;
        result.random_access_mbs = 0.0;
    } else {
        result.random_access_ns = static_cast<double>(ra_ns) / num_ra_queries;
        result.random_access_mbs = (num_ra_queries * sizeof(double) / 1024.0 / 1024.0) / (ra_ns / 1e9);
    }

    // Random-access correctness spot-check (after timing, to avoid warming caches / predictors).
    // Compare point-decoded values to the original input.
    {
        const size_t check_queries = std::min<size_t>(200, num_ra_queries);
        for (size_t q = 0; q < check_queries; ++q) {
            const size_t idx = bench_data.random_indices[q];
            if (idx >= n_full) continue;

            const size_t vec_idx = idx / VEC;
            const size_t offset = idx % VEC;

            decode_vector(
                metadata_stream[vec_idx],
                ffor_stream.data(),
                exc_val_stream.data(),
                exc_pos_stream.data(),
                vec_buf
            );

            const double got = vec_buf[offset];
            const double expected = data[idx];
            if (got != expected) {
                std::cerr << "ALP random access mismatch at idx=" << idx
                          << " expected=" << expected << " got=" << got << std::endl;
                break;
            }
        }
    }

    // Range queries - with volatile checksum to prevent optimization
    const size_t num_range_queries = 1000;
    for (auto range : range_sizes) {
        if (range >= n_full) {
            result.range_query_throughputs.emplace_back(range, 0.0);
            continue;
        }
        
        const auto& range_indices = bench_data.range_query_indices.at(range);
        std::vector<double> range_buf(range);
        volatile double range_checksum = 0;  // volatile prevents optimization
        
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
                    metadata_stream[v],
                    ffor_stream.data(),
                    exc_val_stream.data(),
                    exc_pos_stream.data(),
                    vec_buf
                );
                
                size_t vec_start_global = v * VEC;
                size_t skip = (v == start_vec) ? (start_idx - vec_start_global) : 0;
                size_t take = std::min(VEC - skip, range - out_pos);
                
                memcpy(range_buf.data() + out_pos, vec_buf + skip, take * sizeof(double));
                out_pos += take;
            }
            range_checksum += range_buf[0];  // Force computation
            do_not_optimize(range_buf);
        }
        compiler_barrier();  // Compiler barrier before timing ends
        t2 = std::chrono::high_resolution_clock::now();
        do_not_optimize(range_checksum);
        
        auto range_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
        double throughput = (range * sizeof(double) * q_count / 1024.0 / 1024.0) / (range_ns / 1e9);
        result.range_query_throughputs.emplace_back(range, throughput);
    }

    return result;
}
