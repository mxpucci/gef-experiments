// Shared round-trip policy: |error| > 1e-9 fails, bit differences are informational.
#include "../benchmark/benchmark_common.hpp"
#include <limits>
#include <sstream>
#include <stdexcept>

static void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    std::ostringstream log;
    auto *previous = std::cerr.rdbuf(log.rdbuf());
    const double nan = std::numeric_limits<double>::quiet_NaN();

    BenchmarkResult close{};
    close.compressor = "close";
    std::vector<double> expected{0.1, 1e6, -2.5, nan}, decoded{0.1 + 1e-12, 1e6, -2.5, nan};
    check_roundtrip(close, expected, decoded);
    finish_roundtrip_validation(close);

    BenchmarkResult far{};
    far.compressor = "far";
    std::vector<double> far_decoded{0.1 + 2e-9, 1e6, -2.5, 0.0};
    check_roundtrip(far, expected, far_decoded);
    finish_roundtrip_validation(far);

    BenchmarkResult ints{};
    ints.compressor = "ints";
    std::vector<int64_t> a{1, 2, 3}, b{1, 2, 4};
    check_roundtrip(ints, a, b);
    finish_roundtrip_validation(ints);
    std::cerr.rdbuf(previous);

    require(close.validation_status == "passed" && close.roundtrip_mismatches == 0 &&
            close.bitwise_mismatches == 1, "difference within 1e-9 must pass but be counted bitwise");
    require(far.validation_status == "failed" && far.roundtrip_mismatches == 2 &&
            far.bitwise_mismatches == 2, "difference above 1e-9 and NaN->0 must fail");
    require(ints.validation_status == "failed" && ints.roundtrip_mismatches == 1, "integers compare exactly");
    require(log.str().find("Warning: far") != std::string::npos, "mismatch must warn");
    std::cout << "round-trip policy ok\n";
}
