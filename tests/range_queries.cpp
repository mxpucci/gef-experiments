// Shared range-query runner: same starts for every codec, zeroed slots,
// trials recorded, every query checked against the reference.
#include "../benchmark/range_queries.hpp"
#include <numeric>
#include <sstream>
#include <stdexcept>

static void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    std::ostringstream log;
    auto *previous = std::cerr.rdbuf(log.rdbuf());

    const size_t n = 1000;
    std::vector<int64_t> column(n);
    std::iota(column.begin(), column.end(), 5);
    BenchmarkData data;
    data.range_options.trials = 3;
    data.range_options.batch_bytes = 3 * 8 * 7;  // 3 slots of 7 values: several batches
    data.range_query_indices[7] = {0, 993, 10, 500, 993, 1, 2};
    data.range_query_indices[n] = {};

    // Additive kernel (like NeaTS simd_scan): correct only on zeroed output.
    std::vector<size_t> seen;
    BenchmarkResult additive{};
    additive.compressor = "additive";
    additive.validation_status = "passed";
    run_range_queries(additive, data, {7, n}, n, column.data(),
        [&](size_t start, size_t length, std::vector<int64_t> &out) {
            require(out.size() == length, "slot has the requested length");
            seen.push_back(start);
            for (size_t i = 0; i < length; ++i) out[i] += column[start + i];
        });
    require(additive.validation_status == "passed", "zeroed slots keep additive kernels correct");
    require(additive.range_query_throughputs.size() == 1, "ranges >= n are skipped");
    require(additive.range_trials.size() == 3, "one record per trial");
    for (const auto &t : additive.range_trials)
        require(t.queries == 7 && t.verified_queries == 7 && t.mismatched_queries == 0 && t.value_bytes == 8,
                "trial record");
    require(seen.size() == 3 * 7 + 7, "3 timed trials plus one verification pass");
    for (size_t q = 0; q < seen.size(); ++q)
        require(seen[q] == data.range_query_indices[7][q % 7], "shared starts, in order");

    // A wrong kernel is detected for every affected query.
    BenchmarkResult wrong{};
    wrong.compressor = "wrong";
    wrong.validation_status = "passed";
    run_range_queries(wrong, data, {7}, n, column.data(),
        [&](size_t start, size_t length, std::vector<int64_t> &out) {
            for (size_t i = 0; i < length; ++i) out[i] = column[start + i] + (start == 993);
        });
    require(wrong.validation_status == "failed", "mismatch fails validation");
    require(wrong.range_mismatched_queries == 2, "both queries at 993 counted");

    // Verification can be disabled; nothing is checked then.
    data.range_options.verify = false;
    BenchmarkResult unchecked{};
    unchecked.validation_status = "passed";
    run_range_queries(unchecked, data, {7}, n, column.data(),
        [&](size_t, size_t, std::vector<int64_t> &out) { out[0] = -1; });
    require(unchecked.validation_status == "passed" && unchecked.range_trials[0].verified_queries == 0,
            "--no-range-verify");

    // Doubles follow the shared round-trip rule: within 1e-9 passes, beyond fails.
    data.range_options.verify = true;
    std::vector<double> values(n);
    for (size_t i = 0; i < n; ++i) values[i] = 0.1 * i;
    for (double delta : {1e-12, 1e-6}) {
        BenchmarkResult doubles{};
        doubles.validation_status = "passed";
        run_range_queries(doubles, data, {7}, n, values.data(),
            [&](size_t start, size_t length, std::vector<double> &out) {
                for (size_t i = 0; i < length; ++i) out[i] = values[start + i] + delta;
            });
        require((doubles.validation_status == "passed") == (delta < 1e-9), "1e-9 tolerance for doubles");
    }

    require(range_median({3.0, 1.0, 2.0}) == 2.0 && range_median({1.0, 4.0}) == 2.5, "median");
    std::cerr.rdbuf(previous);
    return 0;
}
