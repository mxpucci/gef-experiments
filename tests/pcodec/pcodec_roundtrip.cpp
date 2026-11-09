#include "Pcodec/Pcodec.hpp"
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

using namespace gef_compressors::pcodec;
static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
static void check_f64(const std::vector<double>& input) {
    const auto compressed = compress(input);
    const auto output = decompress(compressed, input.size());
    require(output.size() == input.size(), "f64 size mismatch");
    require(input.empty() || std::memcmp(input.data(), output.data(), input.size()*sizeof(double)) == 0,
            "f64 bit mismatch");
}
static void check_i64(const std::vector<std::int64_t>& input) {
    require(decompress_i64(compress_i64(input), input.size()) == input, "i64 mismatch");
}
template<class F> static void rejects(F f) {
    bool threw = false;
    try { f(); } catch (const std::runtime_error&) { threw = true; }
    require(threw, "malformed input or wrong count accepted");
}
int main() {
    check_f64({}); check_i64({}); check_f64({42.125}); check_i64({-42});
    check_f64(std::vector<double>(1000, -0.0));
    check_i64(std::vector<std::int64_t>(1000, -42));
    std::vector<double> corners;
    for (std::uint64_t bits : {0ULL, 0x8000000000000000ULL, 1ULL, 0x8000000000000001ULL,
         0x0010000000000000ULL, 0x7fefffffffffffffULL, 0xffefffffffffffffULL,
         0x7ff0000000000000ULL, 0xfff0000000000000ULL, 0x7ff8000000000001ULL,
         0xfff8000000001234ULL, 0x7ff0000000000001ULL}) {
        double value;
        std::memcpy(&value, &bits, sizeof(value));
        corners.push_back(value);
    }
    check_f64(corners);
    check_i64({INT64_MIN, INT64_MAX, 0, -1, 1});
    std::mt19937_64 rng(0x54f184a);
    std::vector<double> floats;
    std::vector<std::int64_t> integers;
    for (int i=0; i<4097; ++i) {
        std::uint64_t bits = rng();
        double value; std::int64_t integer;
        std::memcpy(&value, &bits, sizeof(value));
        std::memcpy(&integer, &bits, sizeof(integer));
        floats.push_back(value); integers.push_back(integer);
    }
    check_f64(floats); check_i64(integers);
    // Cross the default 2^18 chunk boundary and exercise decimal mode/delta choices.
    floats.clear(); integers.clear();
    for (int i=0; i<262144+17; ++i) {
        floats.push_back((i % 1000 - 500) / 100.0);
        integers.push_back(static_cast<std::int64_t>(i) * 123000000LL);
    }
    check_f64(floats); check_i64(integers);
    const auto bytes = compress({1.25, -9.5, 3.0, 1.25});
    rejects([&]{decompress(bytes, 3);});
    rejects([&]{decompress(bytes, 5);});
    rejects([&]{decompress_i64(bytes, 4);});
    for (std::size_t i=0; i<bytes.size(); ++i) {
        const std::vector<std::uint8_t> truncated(bytes.begin(), bytes.begin()+i);
        rejects([&]{decompress(truncated, 4);});
    }
    // Upstream simple_decompress ignores bytes after the standalone footer; the
    // benchmark always passes exactly the bytes returned by compression.
    std::cout << "Pcodec f64/i64 bit-exact round trips and malformed-input checks passed\n";
}
