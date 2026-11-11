// FastLanes (Afroozeh & Boncz, VLDB 2023), official generated kernels from
// cwida/FastLanes (libFastLanes), on 1,024-value vectors with native access.
//
//   fastlanes        FFOR: per vector, base = minimum, bit width of (max - min),
//                    fastlanes::generated::ffor / unffor.
//   fastlanes_delta  the paper's delta on the unified transposed layout:
//                    transpose_i -> unrsum -> FFOR; decode unffor -> rsum ->
//                    untranspose_i. The 16 lane bases are stored per vector.
//
// A partial last vector is padded with its last value (as for ALP); the padded
// vector is stored and counted. Size per vector: FFOR base (64 bits), bit width
// (8 bits), packed lanes (bw * 1024 bits), plus 16 lane bases (1,024 bits) for
// delta; plus 128 bits for n. A point access decodes the vector holding it.
#include "benchmark_common.hpp"
#include "range_queries.hpp"

#include "fls/ffor.hpp"
#include "fls/unffor.hpp"
#include "fls_gen/rsum/rsum.hpp"
#include "fls_gen/transpose/transpose.hpp"
#include "fls_gen/unrsum/unrsum.hpp"
#include "fls_gen/untranspose/untranspose.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
constexpr size_t V = 1024;
constexpr size_t LANES = 16; // 1024 bits of bases for 64-bit values (CFG::UNIFIED_TRANSPOSED::BASES_SIZE = 128 bytes)

struct Encoded {
    bool delta = false;
    size_t n = 0, vectors = 0;
    std::vector<uint64_t> base, packed, lane_bases;
    std::vector<uint8_t> bw;
    std::vector<size_t> offset; // word offset of each vector's packed lanes
    size_t bits() const {
        size_t b = 128;
        for (size_t v = 0; v < vectors; ++v) b += 64 + 8 + size_t(bw[v]) * V + (delta ? LANES * 64 : 0);
        return b;
    }
};

uint8_t width(uint64_t range) { return range ? uint8_t(64 - std::countl_zero(range)) : 0; }

Encoded encode(const std::vector<int64_t> &data, bool delta) {
    Encoded e;
    e.delta = delta;
    e.n = data.size();
    e.vectors = (e.n + V - 1) / V;
    e.base.resize(e.vectors);
    e.bw.resize(e.vectors);
    e.offset.resize(e.vectors + 1);
    if (delta) e.lane_bases.resize(e.vectors * LANES);
    e.packed.reserve(e.n); // grows as needed
    alignas(64) uint64_t in[V], t[V], d[V];
    for (size_t v = 0; v < e.vectors; ++v) {
        const size_t start = v * V, len = std::min(V, e.n - start);
        std::memcpy(in, data.data() + start, len * 8);
        for (size_t i = len; i < V; ++i) in[i] = in[len - 1];
        const uint64_t *src = in;
        if (delta) {
            generated::transpose::fallback::scalar::transpose_i(in, t);
            generated::unrsum::fallback::scalar::unrsum(t, d);
            std::memcpy(&e.lane_bases[v * LANES], t, LANES * 8);
            src = d;
        }
        int64_t mn = int64_t(src[0]), mx = int64_t(src[0]);
        for (size_t i = 1; i < V; ++i) { mn = std::min(mn, int64_t(src[i])); mx = std::max(mx, int64_t(src[i])); }
        const uint8_t bw = width(uint64_t(mx) - uint64_t(mn));
        e.base[v] = uint64_t(mn);
        e.bw[v] = bw;
        e.offset[v] = e.packed.size();
        e.packed.resize(e.packed.size() + size_t(bw) * V / 64);
        if (bw) fastlanes::generated::ffor::fallback::scalar::ffor(src, e.packed.data() + e.offset[v], bw, &e.base[v]);
    }
    e.offset[e.vectors] = e.packed.size();
    return e;
}

// Decodes vector v into out[0..1024).
inline void decode_vector(const Encoded &e, size_t v, uint64_t *out, uint64_t *scratch) {
    uint64_t *dst = e.delta ? scratch : out;
    if (e.bw[v]) fastlanes::generated::unffor::fallback::scalar::unffor(e.packed.data() + e.offset[v], dst, e.bw[v], &e.base[v]);
    else std::fill(dst, dst + V, e.base[v]);
    if (e.delta) {
        alignas(64) uint64_t t[V];
        generated::rsum::fallback::scalar::rsum(scratch, t, &e.lane_bases[v * LANES]);
        generated::untranspose::fallback::scalar::untranspose_i(t, out);
    }
}
} // namespace

BenchmarkResult benchmark_fastlanes(const BenchmarkData &bench_data, const std::vector<size_t> &range_sizes, bool delta) {
    using Clock = std::chrono::high_resolution_clock;
    auto seconds = [](Clock::time_point a, Clock::time_point b) {
        return std::max(std::chrono::duration<double>(b - a).count(), 1e-9);
    };
    BenchmarkResult result;
    result.compressor = delta ? "FastLanes_delta" : "FastLanes";
    result.dataset = bench_data.filename;
    const auto &data = bench_data.shifted_data;
    const size_t n = data.size();
    result.num_values = n;
    result.uncompressed_bits = bench_data.uncompressed_bits;
    result.access_layout = "native";
    result.native_page_values = V;
    const double mib = n * sizeof(int64_t) / 1048576.0;

    auto t1 = Clock::now();
    Encoded e = encode(data, delta);
    compiler_barrier();
    auto t2 = Clock::now();
    do_not_optimize(e);
    result.compressed_bits = e.bits();
    result.compression_ratio = double(result.compressed_bits) / result.uncompressed_bits;
    result.compression_throughput_mbs = mib / seconds(t1, t2);

    std::vector<int64_t> decompressed(e.vectors * V);
    alignas(64) uint64_t scratch[V];
    t1 = Clock::now();
    for (size_t v = 0; v < e.vectors; ++v)
        decode_vector(e, v, reinterpret_cast<uint64_t *>(decompressed.data()) + v * V, scratch);
    compiler_barrier();
    t2 = Clock::now();
    do_not_optimize(decompressed);
    result.decompression_throughput_mbs = mib / seconds(t1, t2);
    decompressed.resize(n);
    check_roundtrip(result, data, decompressed);
    finish_roundtrip_validation(result);

    const size_t queries = bench_data.random_indices.size();
    alignas(64) uint64_t vec[V];
    volatile int64_t ra_sum = 0;
    t1 = Clock::now();
    for (const size_t index : bench_data.random_indices) {
        decode_vector(e, index / V, vec, scratch);
        ra_sum += int64_t(vec[index % V]);
    }
    compiler_barrier();
    t2 = Clock::now();
    do_not_optimize(ra_sum);
    result.random_access_ns = queries ? seconds(t1, t2) * 1e9 / queries : 0;
    result.random_access_mbs = queries * sizeof(int64_t) / 1048576.0 / seconds(t1, t2);

    run_range_queries(result, bench_data, range_sizes, n, decompressed.data(),
        [&](size_t start, size_t length, std::vector<int64_t> &out) {
            size_t done = 0;
            while (done < length) {
                const size_t i = start + done, v = i / V, off = i % V, take = std::min(V - off, length - done);
                decode_vector(e, v, vec, scratch);
                std::memcpy(out.data() + done, vec + off, take * 8);
                done += take;
            }
        });
    return result;
}
