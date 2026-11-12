// Camel benchmark: native C++ translation of the authors' Java codec
// (benchmark/Camel/camel.hpp), independent blocks (harness -b, default 1000 as in the
// authors' experiments, Java CamelDecompressor.DEFAULT_BLOCK_SIZE), final
// partial block kept. Size = bytes actually written per block.
#include "benchmark_common.hpp"
#include "range_queries.hpp"
#include "Camel/camel.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

BenchmarkResult benchmark_camel(const BenchmarkData& bench_data, const std::vector<size_t>& range_sizes,
                                size_t block_size) {
    const size_t kBlock = block_size ? block_size : 1000; // harness -b, default 1000
    using Clock = std::chrono::high_resolution_clock;
    auto seconds = [](Clock::time_point a, Clock::time_point b) {
        return std::max(std::chrono::duration<double>(b - a).count(), 1e-9);
    };
    BenchmarkResult r;
    r.compressor = "Camel";
    r.dataset = bench_data.filename;
    const auto& data = bench_data.double_data;
    const size_t n = data.size();
    r.num_values = n;
    r.uncompressed_bits = bench_data.uncompressed_bits;
    r.access_layout = "independent_blocks";
    r.compression_unit_values = kBlock;
    const double mib = n * sizeof(double) / 1048576.0;

    const camel::Config cfg = camel::configure(bench_data.decimals);
    std::cerr << "Camel configuration: decimal_max_count=" << cfg.decimal_max_count
              << " (count field " << cfg.count_bits() << " bits), integer_bits=" << cfg.integer_bits << '\n';

    const size_t num_blocks = (n + kBlock - 1) / kBlock;
    std::vector<std::string> blocks(num_blocks);
    size_t bytes = 0;
    const size_t gef_ram0 = gef_live_heap_bytes();
    auto t1 = Clock::now();
    for (size_t b = 0; b < num_blocks; ++b) {
        camel::Compressor c(cfg);
        for (size_t i = b * kBlock, end = std::min(n, i + kBlock); i < end; ++i) c.addValue(data[i]);
        blocks[b] = c.close();
        bytes += blocks[b].size();
    }
    compiler_barrier();
    auto t2 = Clock::now();
    r.compressed_bits = bytes * 8;
    r.compression_ratio = double(r.compressed_bits) / r.uncompressed_bits;
    r.ram_bytes = gef_heap_delta(gef_ram0);
    r.compression_throughput_mbs = mib / seconds(t1, t2);

    auto decoder = [&](size_t b) {
        return camel::Decompressor(reinterpret_cast<const uint8_t*>(blocks[b].data()), blocks[b].size(), cfg);
    };

    std::vector<double> decoded(n);
    t1 = Clock::now();
    for (size_t b = 0; b < num_blocks; ++b) {
        auto d = decoder(b);
        for (size_t i = b * kBlock, end = std::min(n, i + kBlock); i < end; ++i) decoded[i] = d.next();
    }
    compiler_barrier();
    t2 = Clock::now();
    do_not_optimize(decoded);
    r.decompression_throughput_mbs = mib / seconds(t1, t2);

    // Shared rule: |error| <= 1e-9 passes; failures are warnings, not fatal.
    size_t bitwise = 0;
    for (size_t i = 0; i < n; ++i) {
        bitwise += std::bit_cast<uint64_t>(data[i]) != std::bit_cast<uint64_t>(decoded[i]);
        r.roundtrip_mismatches += !(std::abs(data[i] - decoded[i]) <= 1e-9);
    }
    r.validation_status = r.roundtrip_mismatches ? "failed" : "passed";
    if (r.roundtrip_mismatches)
        std::cerr << "WARNING: Camel: " << r.roundtrip_mismatches << "/" << n
                  << " values differ by > 1e-9; emitting metrics with validation_status=failed\n";
    else if (bitwise)
        std::cerr << "Note: Camel: " << bitwise << " values differ bitwise but within 1e-9\n";

    // Random access: decode the containing block up to the requested offset.
    const size_t queries = std::min<size_t>(100000, bench_data.random_indices.size());
    volatile double sink = 0;
    t1 = Clock::now();
    for (size_t q = 0; q < queries; ++q) {
        const size_t idx = bench_data.random_indices[q];
        auto d = decoder(idx / kBlock);
        double v = 0;
        for (size_t k = 0; k <= idx % kBlock; ++k) v = d.next();
        sink = sink + v;
    }
    compiler_barrier();
    t2 = Clock::now();
    r.random_access_ns = queries ? seconds(t1, t2) * 1e9 / queries : 0;
    r.random_access_mbs = queries * sizeof(double) / 1048576.0 / seconds(t1, t2);

    // Range queries (range_queries.hpp): decode intersected blocks up to the end of the range.
    run_range_queries(r, bench_data, range_sizes, n, decoded.data(),
        [&](size_t start, size_t range, std::vector<double> &out) {
            const size_t end = start + range;
            size_t pos = 0;
            for (size_t b = start / kBlock; b <= (end - 1) / kBlock; ++b) {
                auto d = decoder(b);
                const size_t lo = std::max(start, b * kBlock), hi = std::min(end, (b + 1) * kBlock);
                for (size_t j = b * kBlock; j < lo; ++j) d.next();
                for (size_t j = lo; j < hi; ++j) out[pos++] = d.next();
            }
        });
    return r;
}
