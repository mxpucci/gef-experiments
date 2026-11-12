// Shared range-query measurement used by every codec adapter.
//
// Methodology (see docs/range-queries.md):
//   * every codec answers the SAME seeded starts (BenchmarkData::range_query_indices),
//     the same number of queries, and the same number of timing trials;
//   * each query writes `range` values into its own output slot. Slots come
//     from a preallocated arena that is reused across batches; before each
//     batch the slots are zero-filled OUTSIDE the timed region (NeaTS's scan
//     adds predictions into its destination, so it needs zeroed output; every
//     codec gets the same preparation so the timed work is comparable);
//   * only the queries are timed (one timer pair per batch of slots);
//   * correctness is checked after timing, for every query, against the codec's
//     own full decompression with the shared round-trip rule (bit-identical or
//     |difference| <= ROUNDTRIP_TOLERANCE); full decompression is checked
//     against the input separately by the same policy.
//
// C++17 only: also compiled by the clang-built ALP/LeCo translation units.
#pragma once

#include "benchmark_common.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

// Median of the per-trial throughputs (mean of the two middle values when even).
inline double range_median(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    return values.size() % 2 ? values[mid] : (values[mid - 1] + values[mid]) / 2.0;
}

// Timed batch execution of the shared range queries for one codec.
//   n          number of values in the column (ranges >= n are skipped, as before)
//   reference  codec's full decompression, indexable [0, n), same type as the output
//   query      callable (size_t start, size_t length, std::vector<T>& out); must write
//              out[0, length). `out.size() == length` and `out` is zero-filled.
// Appends (range, median MiB/s) to result.range_query_throughputs (legacy CSV unit)
// and one RangeTrial per trial to result.range_trials.
template<class T, class Query>
void run_range_queries(BenchmarkResult &result, const BenchmarkData &bench_data,
                       const std::vector<size_t> &range_sizes, size_t n, const T *reference,
                       Query &&query) {
    using Clock = std::chrono::steady_clock;
    const RangeQueryOptions &options = bench_data.range_options;
    const size_t trials = std::max<size_t>(1, options.trials);
    for (size_t range : range_sizes) {
        if (range == 0 || range >= n) continue;
        const auto found = bench_data.range_query_indices.find(range);
        if (found == bench_data.range_query_indices.end() || found->second.empty()) continue;
        const std::vector<size_t> &starts = found->second;
        const size_t queries = starts.size();
        for (size_t start : starts)
            if (start > n - range) throw std::out_of_range("range query start beyond column");

        const size_t slot_bytes = range * sizeof(T);
        const size_t slots = std::min(queries, std::max<size_t>(1, options.batch_bytes / slot_bytes));
        std::vector<std::vector<T>> arena(slots, std::vector<T>(range));
        auto prepare = [&](size_t count) {
            for (size_t s = 0; s < count; ++s) std::fill(arena[s].begin(), arena[s].end(), T{});
        };

        std::vector<double> throughputs;
        for (size_t trial = 0; trial < trials; ++trial) {
            uint64_t elapsed_ns = 0;
            for (size_t first = 0; first < queries; first += slots) {
                const size_t count = std::min(slots, queries - first);
                prepare(count);  // untimed
                compiler_barrier();
                const auto t1 = Clock::now();
                for (size_t s = 0; s < count; ++s) {
                    query(starts[first + s], range, arena[s]);
                    do_not_optimize(arena[s].data());  // results must be materialized
                }
                compiler_barrier();
                const auto t2 = Clock::now();
                elapsed_ns += static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count());
            }
            elapsed_ns = std::max<uint64_t>(elapsed_ns, 1);
            RangeTrial record;
            record.range = range;
            record.trial = trial;
            record.queries = queries;
            record.value_bytes = sizeof(T);
            record.elapsed_ns = elapsed_ns;
            result.range_trials.push_back(record);
            throughputs.push_back(double(queries) * range * sizeof(T) / 1048576.0 / (elapsed_ns / 1e9));
        }
        result.range_query_throughputs.emplace_back(range, range_median(throughputs));

        // Untimed correctness check of every query, with the same preparation.
        size_t verified = 0, mismatched = 0;
        if (options.verify && reference != nullptr) {
            for (size_t start : starts) {
                prepare(1);
                query(start, range, arena[0]);
                ++verified;
                const T *got = arena[0].data(), *want = reference + start;
                bool differs = false;
                if (std::memcmp(got, want, slot_bytes) != 0)
                    for (size_t i = 0; i < range && !differs; ++i)
                        differs = std::memcmp(&got[i], &want[i], sizeof(T)) != 0 &&
                                  roundtrip_value_mismatch(want[i], got[i]);
                if (differs) {
                    if (mismatched++ == 0)
                        std::cerr << "\nWarning: " << result.compressor << ": range query [" << start << ", "
                                  << start + range << ") differs from full decompression\n";
                }
            }
        }
        for (auto it = result.range_trials.end() - trials; it != result.range_trials.end(); ++it) {
            it->verified_queries = verified;
            it->mismatched_queries = mismatched;
        }
        if (mismatched) {
            result.range_mismatched_queries += mismatched;
            result.validation_status = "failed";
            std::cerr << "Warning: " << result.compressor << ": " << mismatched << "/" << verified
                      << " range queries of length " << range
                      << " differ from full decompression; emitting metrics with validation_status=failed.\n";
        }
    }
}
