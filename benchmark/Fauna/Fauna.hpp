#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
const char* gef_fauna_last_error();
int gef_fauna_compress(const float*, std::size_t, bool, std::uint8_t**, std::size_t*);
int gef_fauna_decompress(const std::uint8_t*, std::size_t, std::size_t, bool, float*);
void gef_fauna_free(std::uint8_t*, std::size_t);
}

namespace gef_compressors { namespace fauna_detail {
inline void check(int status) {
    if (status) throw std::runtime_error(std::string("Fauna/Macaque: ") + gef_fauna_last_error());
}
inline std::vector<std::uint8_t> compress32(const std::vector<float>& values, bool macaque) {
    std::uint8_t* bytes = nullptr;
    std::size_t size = 0;
    check(gef_fauna_compress(values.data(), values.size(), macaque, &bytes, &size));
    try {
        std::vector<std::uint8_t> result(bytes, bytes + size);
        gef_fauna_free(bytes, size);
        return result;
    } catch (...) { gef_fauna_free(bytes, size); throw; }
}
inline std::vector<float> decompress32(const std::vector<std::uint8_t>& bytes,
                                     std::size_t count, bool macaque) {
    std::vector<float> values(count);
    check(gef_fauna_decompress(bytes.data(), bytes.size(), count, macaque, values.data()));
    return values;
}
// A float64 benchmark must never silently round its input to the author's f32 type.
inline std::vector<float> exact_f32(const std::vector<double>& input) {
    std::vector<float> values;
    values.reserve(input.size());
    for (double value : input) {
        const float narrowed = static_cast<float>(value);
        const double restored = static_cast<double>(narrowed);
        if (std::memcmp(&value, &restored, sizeof(double)) != 0)
            throw std::invalid_argument("Fauna/Macaque upstream supports f32 only; input is not exactly representable");
        values.push_back(narrowed);
    }
    return values;
}
inline std::vector<double> expand(const std::vector<float>& values) {
    return std::vector<double>(values.begin(), values.end());
}
} }

namespace gef_compressors { namespace fauna {
inline std::vector<std::uint8_t> compress_f32(const std::vector<float>& values) {
    return fauna_detail::compress32(values, false);
}
inline std::vector<float> decompress_f32(const std::vector<std::uint8_t>& bytes, std::size_t count) {
    return fauna_detail::decompress32(bytes, count, false);
}
inline std::vector<std::uint8_t> compress(const std::vector<double>& values) {
    return compress_f32(fauna_detail::exact_f32(values));
}
inline std::vector<double> decompress(const std::vector<std::uint8_t>& bytes, std::size_t count) {
    return fauna_detail::expand(decompress_f32(bytes, count));
}
} }

namespace gef_compressors { namespace macaque {
inline std::vector<std::uint8_t> compress_f32(const std::vector<float>& values) {
    return fauna_detail::compress32(values, true);
}
inline std::vector<float> decompress_f32(const std::vector<std::uint8_t>& bytes, std::size_t count) {
    return fauna_detail::decompress32(bytes, count, true);
}
inline std::vector<std::uint8_t> compress(const std::vector<double>& values) {
    return compress_f32(fauna_detail::exact_f32(values));
}
inline std::vector<double> decompress(const std::vector<std::uint8_t>& bytes, std::size_t count) {
    return fauna_detail::expand(decompress_f32(bytes, count));
}
} }
