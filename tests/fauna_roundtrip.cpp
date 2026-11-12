#include "Fauna/Fauna.hpp"
#include <iostream>
#include <limits>
#include <random>

template<class T> void require_equal(const std::vector<T>& a, const std::vector<T>& b) {
    if (a.size() != b.size() || (!a.empty() && std::memcmp(a.data(), b.data(), a.size()*sizeof(T))))
        throw std::runtime_error("roundtrip mismatch");
}
int main() {
    for (std::size_t n : {0U, 1U, 2U, 9U, 255U, 256U, 1000U, 8192U}) {
        for (int kind = 0; kind < 5; ++kind) {
            std::mt19937 gen(1908);
            std::vector<float> input;
            for (std::size_t i = 0; i < n; ++i) {
                float value = kind == 0 ? 7.5f : kind == 1 ? static_cast<float>(i) / 8 :
                    kind == 2 ? static_cast<float>(static_cast<int>(gen()%1000000) - 500000) / 128 :
                    kind == 3 ? static_cast<float>((i / 16) % 3) : static_cast<float>(i%7) * 0.1f;
                input.push_back(value);
            }
            auto a = gef_compressors::fauna::compress_f32(input);
            require_equal(input, gef_compressors::fauna::decompress_f32(a, n));
            auto b = gef_compressors::macaque::compress_f32(input);
            require_equal(input, gef_compressors::macaque::decompress_f32(b, n));
        }
    }
    std::mt19937 gen(123);
    std::vector<float> bits;
    for (unsigned i = 0; i < 10000; ++i) {
        uint32_t raw = gen(); float value; std::memcpy(&value, &raw, 4); bits.push_back(value);
    }
    bits.insert(bits.end(), {0.0f, -0.0f, std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::denorm_min()});
    auto encoded = gef_compressors::macaque::compress_f32(bits);
    require_equal(bits, gef_compressors::macaque::decompress_f32(encoded, bits.size()));
    std::vector<float> finite;
    for (float v : bits) if (std::isfinite(v) && !(v == 0 && std::signbit(v))) finite.push_back(v);
    auto finite_encoded = gef_compressors::fauna::compress_f32(finite);
    require_equal(finite, gef_compressors::fauna::decompress_f32(finite_encoded, finite.size()));
    for (float extreme : {std::numeric_limits<float>::max(), std::numeric_limits<float>::min(),
                          std::numeric_limits<float>::denorm_min()}) {
        std::vector<float> run(1000, extreme);
        auto data = gef_compressors::fauna::compress_f32(run);
        require_equal(run, gef_compressors::fauna::decompress_f32(data, run.size()));
    }
    for (double v : {0.1, 1.0e300, 16777217.0}) {
        bool rejected = false;
        try { gef_compressors::fauna::compress({v}); } catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) throw std::runtime_error("silently narrowed f64 input");
    }
    bool rejected = false;
    try { gef_compressors::fauna::compress({-0.0}); } catch (const std::runtime_error&) { rejected = true; }
    if (!rejected) throw std::runtime_error("accepted unsupported negative zero");
    rejected = false;
    try { gef_compressors::macaque::decompress(encoded, bits.size()+1); } catch (const std::runtime_error&) { rejected = true; }
    if (!rejected) throw std::runtime_error("accepted incorrect count");
    std::cout << "Fauna/Macaque roundtrip and eligibility tests passed\n";
}
