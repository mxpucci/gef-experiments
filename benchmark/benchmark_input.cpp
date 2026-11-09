#include "benchmark_common.hpp"

#ifdef __FAST_MATH__
#error "Benchmark input conversion requires strict floating-point arithmetic"
#endif

void BenchmarkData::convert_shifted_to_double() {
    if (shifted_data.empty()) return;

    int64_t divisor_int = 1;
    for (int64_t d = 0; d < decimals; ++d) divisor_int *= 10;
    const double divisor = static_cast<double>(divisor_int);

    double_data.resize(shifted_data.size());
    for (size_t i = 0; i < shifted_data.size(); ++i) {
        // Restore the original scaled integer before correctly rounded division.
        double_data[i] = static_cast<double>(shifted_data[i] + min_val) / divisor;
    }
    shifted_data.clear();
    shifted_data.shrink_to_fit();
}
