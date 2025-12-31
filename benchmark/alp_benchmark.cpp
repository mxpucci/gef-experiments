// ALP benchmark implementation compiled optionally with clang++ (hybrid toolchain).

#include "benchmark_common.hpp"

// ALP headers
// On GCC, ALP's AVX-512 header path currently fails to compile; keep AVX2 SIMD but disable AVX-512.
#if defined(__GNUC__) && !defined(__clang__)
  #ifdef __AVX512F__
    #undef __AVX512F__
  #endif
  #ifdef __AVX512BW__
    #undef __AVX512BW__
  #endif
  #ifdef __AVX512DQ__
    #undef __AVX512DQ__
  #endif
#endif

#include "alp.hpp"

#include <cstring>
#include <chrono>

namespace {

struct ALPBlock {
    alp::Scheme scheme = alp::Scheme::INVALID;
    uint16_t count = 0; // number of values in this block (<= alp::config::VECTOR_SIZE)

    uint16_t exceptions_count = 0;

    // ALP (PDE)
    uint8_t bit_width = 0;
    uint8_t fac = 0;
    uint8_t exp = 0;
    int64_t base = 0;
    std::vector<uint64_t> packed_digits;
    std::vector<double> exceptions;
    std::vector<uint16_t> exception_positions;

    // ALP_RD
    uint8_t right_bit_width = 0;
    uint8_t left_bit_width = 0;
    uint8_t dict_size = 0;
    std::array<uint16_t, alp::config::MAX_RD_DICTIONARY_SIZE> dict {};
    std::vector<uint64_t> packed_right;
    std::vector<uint16_t> packed_left;
    std::vector<uint16_t> rd_exceptions;
    std::vector<uint16_t> rd_exception_positions;

    // RAW tail
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

inline void falp_decode_best(const uint64_t* in,
                             double* out,
                             uint8_t bw,
                             const uint64_t* base_p,
                             uint8_t factor,
                             uint8_t exponent) {
    // Prefer best available SIMD kernel at compile time
#if defined(__AVX512F__)
    generated::falp::x86_64::avx512f::falp(in, out, bw, base_p, factor, exponent);
#elif defined(__AVX2__)
    generated::falp::x86_64::avx2::falp(in, out, bw, base_p, factor, exponent);
#elif defined(__SSE2__)
    generated::falp::x86_64::sse::falp(in, out, bw, base_p, factor, exponent);
#else
    generated::falp::fallback::scalar::falp(in, out, bw, base_p, factor, exponent);
#endif
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
        const size_t n_full = (n / alp::config::VECTOR_SIZE) * alp::config::VECTOR_SIZE;
        const size_t num_vectors = n_full / alp::config::VECTOR_SIZE;
        const size_t vectors_per_rg = alp::config::N_VECTORS_PER_ROWGROUP;

        std::array<double, alp::config::VECTOR_SIZE> sample_arr {};
        alp::state<double> stt;

        for (size_t vg = 0; vg < num_vectors; vg += vectors_per_rg) {
            const size_t rg_vectors = std::min(vectors_per_rg, num_vectors - vg);
            const size_t rg_offset_vals = vg * alp::config::VECTOR_SIZE;
            const size_t rg_vals = rg_vectors * alp::config::VECTOR_SIZE;

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

    // Compressed size estimate (bytes -> bits)
    size_t compressed_bytes = 0;
    for (const auto& blk : blocks) {
        compressed_bytes += 1 + 2 + 2;
        if (blk.scheme == alp::Scheme::ALP) {
            compressed_bytes += 1 + 1 + 1 + sizeof(int64_t);
            compressed_bytes += blk.packed_digits.size() * sizeof(uint64_t);
            compressed_bytes += blk.exceptions.size() * sizeof(double);
            compressed_bytes += blk.exception_positions.size() * sizeof(uint16_t);
        } else if (blk.scheme == alp::Scheme::ALP_RD) {
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
                falp_decode_best(blk.packed_digits.data(),
                                 out_vec.data(),
                                 blk.bit_width,
                                 reinterpret_cast<const uint64_t*>(&blk.base),
                                 blk.fac,
                                 blk.exp);
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
                    unffor::unffor(reinterpret_cast<const uint16_t*>(blk.packed_left.data()),
                                   left_out.data(),
                                   blk.left_bit_width,
                                   &zero16);
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

    // Random access + range queries: reuse existing logic by decompressing blocks on-demand
    // (kept identical to the previous in-TU implementation for now).
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
            falp_decode_best(blk.packed_digits.data(),
                             out_vec.data(),
                             blk.bit_width,
                             reinterpret_cast<const uint64_t*>(&blk.base),
                             blk.fac,
                             blk.exp);
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
                unffor::unffor(reinterpret_cast<const uint16_t*>(blk.packed_left.data()),
                               left_out.data(),
                               blk.left_bit_width,
                               &zero16);
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
                    falp_decode_best(blk.packed_digits.data(),
                                     tmp.data(),
                                     blk.bit_width,
                                     reinterpret_cast<const uint64_t*>(&blk.base),
                                     blk.fac,
                                     blk.exp);
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
                        unffor::unffor(reinterpret_cast<const uint16_t*>(blk.packed_left.data()),
                                       left_out.data(),
                                       blk.left_bit_width,
                                       &zero16);
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

                std::memcpy(out_buffer.data() + out_pos, tmp.data() + in_off, to_take * sizeof(double));
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


