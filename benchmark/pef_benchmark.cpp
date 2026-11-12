// Partitioned Elias-Fano (Ottaviano & Venturini, SIGIR 2014), official ds2i
// implementation (lib/ds2i, ot/ds2i @ 2cf8237), with two 32-bit truncation
// fixes applied at build time (see CMakeLists.txt).
//
// Elias-Fano codes require a monotone sequence. As discussed in the paper for
// arbitrary sequences, the (positive) shifted values s_1..s_n are mapped to
// their prefix sums S_i = s_1 + ... + s_i, a strictly increasing sequence with
// universe S_n + 1, which ds2i encodes with partitioned_sequence<indexed_sequence>
// and the authors' default parameters (eps1 = 0.03, eps2 = 0.3, fix_cost = 64,
// overridable through the DS2I_* environment variables as in ds2i).
//
// Compression time includes computing the prefix sums. Size is the encoded bit
// vector plus 128 bits for n and the universe. A value is S_i - S_{i-1}: point
// access moves to i - 1 and reads the next element; decompression and range
// queries scan with next().
#include "benchmark_common.hpp"
#include "range_queries.hpp"

#include "partitioned_sequence.hpp"
#include <succinct/bit_vector.hpp>

#include <chrono>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
using PefSequence = ds2i::partitioned_sequence<ds2i::indexed_sequence>;

// Decodes values [start, start + count) from the prefix-sum enumerator into out.
template <class Out>
void pef_decode(PefSequence::enumerator &e, size_t start, size_t count, Out *out) {
    if (count == 0) return;
    uint64_t previous = start ? e.move(start - 1).second : 0;
    uint64_t current = start ? e.next().second : e.move(0).second;
    out[0] = static_cast<Out>(current - previous);
    for (size_t k = 1; k < count; ++k) {
        previous = current;
        current = e.next().second;
        out[k] = static_cast<Out>(current - previous);
    }
}
} // namespace

BenchmarkResult benchmark_pef(const BenchmarkData &bench_data, const std::vector<size_t> &range_sizes) {
    using Clock = std::chrono::high_resolution_clock;
    auto seconds = [](Clock::time_point a, Clock::time_point b) {
        return std::max(std::chrono::duration<double>(b - a).count(), 1e-9);
    };
    BenchmarkResult result;
    result.compressor = "PEF";
    result.dataset = bench_data.filename;
    const auto &data = bench_data.shifted_data; // every value >= 1
    const size_t n = data.size();
    result.num_values = n;
    result.uncompressed_bits = bench_data.uncompressed_bits;
    result.access_layout = "native";
    const double mib = n * sizeof(int64_t) / 1048576.0;

    // Compression: prefix sums + partitioned Elias-Fano construction.
    ds2i::global_parameters params;
    const size_t gef_ram0 = gef_live_heap_bytes();
    auto t1 = Clock::now();
    std::vector<uint64_t> prefix(n);
    uint64_t sum = 0;
    for (size_t i = 0; i < n; ++i) {
        const uint64_t value = static_cast<uint64_t>(data[i]);
        if (value > std::numeric_limits<uint64_t>::max() - sum)
            throw std::overflow_error("PEF: prefix sum exceeds 64 bits");
        sum += value;
        prefix[i] = sum;
    }
    const uint64_t universe = sum + 1;
    succinct::bit_vector_builder builder;
    PefSequence::write(builder, prefix.begin(), universe, n, params);
    succinct::bit_vector bits(&builder);
    compiler_barrier();
    auto t2 = Clock::now();
    std::vector<uint64_t>().swap(prefix);
    do_not_optimize(bits);
    result.compressed_bits = bits.size() + 2 * 64; // + n and universe
    result.compression_ratio = static_cast<double>(result.compressed_bits) / result.uncompressed_bits;
    result.ram_bytes = gef_heap_delta(gef_ram0);
    result.compression_throughput_mbs = mib / seconds(t1, t2);

    PefSequence::enumerator e(bits, 0, universe, n, params);

    // Full decompression.
    std::vector<int64_t> decompressed(n);
    t1 = Clock::now();
    pef_decode(e, 0, n, decompressed.data());
    compiler_barrier();
    t2 = Clock::now();
    do_not_optimize(decompressed);
    result.decompression_throughput_mbs = mib / seconds(t1, t2);
    check_roundtrip(result, data, decompressed);
    finish_roundtrip_validation(result);

    // Random access: each lookup reads S_{i-1} and S_i.
    const size_t queries = bench_data.random_indices.size();
    volatile int64_t ra_sum = 0;
    t1 = Clock::now();
    for (const size_t index : bench_data.random_indices) {
        int64_t value;
        pef_decode(e, index, 1, &value);
        ra_sum += value;
    }
    compiler_barrier();
    t2 = Clock::now();
    do_not_optimize(ra_sum);
    result.random_access_ns = queries ? seconds(t1, t2) * 1e9 / queries : 0;
    result.random_access_mbs = queries * sizeof(int64_t) / 1048576.0 / seconds(t1, t2);

    run_range_queries(result, bench_data, range_sizes, n, decompressed.data(),
        [&](size_t start, size_t length, std::vector<int64_t> &out) {
            pef_decode(e, start, length, out.data());
        });
    return result;
}
