#include "Fauna/Fauna.hpp"
#include <iostream>
#include <limits>
#include <random>

template<class T> void require_equal(const std::vector<T>& a, const std::vector<T>& b) {
    if (a.size() != b.size() || (!a.empty() && std::memcmp(a.data(), b.data(), a.size()*sizeof(T))))
        throw std::runtime_error("roundtrip mismatch");
}
// Swing lines are evaluated in f64, the same type as the values, so Fauna's
// model values may differ from the input by rounding; MacaqueV is bitwise exact.
static void require_close(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) throw std::runtime_error("roundtrip count mismatch");
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(&a[i], &b[i], 8) && !(std::fabs(a[i] - b[i]) <= 1e-9))
            throw std::runtime_error("roundtrip mismatch above 1e-9");
}
static void roundtrip_both(const std::vector<double>& input) {
    require_close(input, gef_compressors::fauna::decompress(gef_compressors::fauna::compress(input), input.size()));
    require_equal(input, gef_compressors::macaque::decompress(gef_compressors::macaque::compress(input), input.size()));
}
int main() {
    for (std::size_t n : {0U, 1U, 2U, 9U, 255U, 256U, 1000U, 8192U}) {
        for (int kind = 0; kind < 6; ++kind) {
            std::mt19937 gen(1908);
            std::vector<double> input;
            for (std::size_t i = 0; i < n; ++i) {
                double value = kind == 0 ? 7.5 : kind == 1 ? static_cast<double>(i) / 8 :
                    kind == 2 ? static_cast<double>(static_cast<int>(gen()%1000000) - 500000) / 128 :
                    kind == 3 ? static_cast<double>((i / 16) % 3) :
                    kind == 4 ? static_cast<double>(i%7) * 0.1 :
                    // Decimal-scaled values that float32 cannot represent.
                    static_cast<double>(static_cast<int>(gen()%2000000) - 1000000) / 1000;
                input.push_back(value);
            }
            roundtrip_both(input);
        }
    }
    // Values that differ from their float32 conversion are preserved bit for bit.
    roundtrip_both({0.1, 1.0e300, 16777217.0, 19705.4018, -273.15, 0.1, 0.1});
    std::mt19937_64 gen(123);
    std::vector<double> bits;
    for (unsigned i = 0; i < 10000; ++i) {
        std::uint64_t raw = gen(); double value; std::memcpy(&value, &raw, 8); bits.push_back(value);
    }
    // Sign flips with odd mantissas exercise the full 64-bit meaningful-bit block.
    bits.insert(bits.end(), {1.0000000000000002, -1.0000000000000004, 3.0000000000000004, -5.000000000000001});
    bits.insert(bits.end(), {0.0, -0.0, std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::denorm_min(),
        std::numeric_limits<double>::quiet_NaN()});
    auto encoded = gef_compressors::macaque::compress(bits);
    require_equal(bits, gef_compressors::macaque::decompress(encoded, bits.size()));
    std::vector<double> finite;
    // Magnitudes near f64::MAX overflow Fauna's f64 fitting (see docs/fauna.md).
    for (double v : bits) if (std::isfinite(v) && !(v == 0 && std::signbit(v)) && std::fabs(v) <= 0x1p1000) finite.push_back(v);
    auto finite_encoded = gef_compressors::fauna::compress(finite);
    require_close(finite, gef_compressors::fauna::decompress(finite_encoded, finite.size()));
    for (double extreme : {0x1p1000, -0x1p1000, std::numeric_limits<double>::min(),
                           std::numeric_limits<double>::denorm_min()}) {
        roundtrip_both(std::vector<double>(1000, extreme));
    }
    // Extreme alternating magnitudes still round-trip through both entries.
    {
        std::vector<double> swings;
        for (int i = 0; i < 1000; ++i) swings.push_back(i % 2 ? 0x1p1000 : -0x1p1000);
        roundtrip_both(swings);
    }
    // Fauna's f64 fitting can overflow near f64::MAX (documented, not patched);
    // standalone MacaqueV is pure XOR coding and preserves such values.
    require_equal(std::vector<double>{1.0, std::numeric_limits<double>::max()},
        gef_compressors::macaque::decompress(gef_compressors::macaque::compress({1.0, std::numeric_limits<double>::max()}), 2));
    bool rejected = false;
    try { gef_compressors::fauna::compress({-0.0}); } catch (const std::runtime_error&) { rejected = true; }
    if (!rejected) throw std::runtime_error("accepted unsupported negative zero");
    rejected = false;
    try { gef_compressors::macaque::decompress(encoded, bits.size()+1); } catch (const std::runtime_error&) { rejected = true; }
    if (!rejected) throw std::runtime_error("accepted incorrect count");
    std::cout << "Fauna/Macaque f64 roundtrip and eligibility tests passed\n";
}
