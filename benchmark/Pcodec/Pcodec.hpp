#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
int gef_pco_bound(std::size_t n, std::size_t* result);
int gef_pco_compress_f64(const double*, std::size_t, std::uint8_t*, std::size_t, std::size_t*);
int gef_pco_decompress_f64(const std::uint8_t*, std::size_t, double*, std::size_t);
int gef_pco_compress_i64(const std::int64_t*, std::size_t, std::uint8_t*, std::size_t, std::size_t*);
int gef_pco_decompress_i64(const std::uint8_t*, std::size_t, std::int64_t*, std::size_t);
}

namespace gef_compressors { namespace pcodec {
namespace detail {
inline void check(int code) {
    if (code != 0) throw std::runtime_error("Pcodec failed (invalid input, size mismatch, or upstream error; code " + std::to_string(code) + ")");
}
template<class T, class Encode>
std::vector<std::uint8_t> encode(const std::vector<T>& values, Encode fn) {
    std::size_t capacity = 0, written = 0;
    check(gef_pco_bound(values.size(), &capacity));
    std::vector<std::uint8_t> bytes(capacity);
    check(fn(values.data(), values.size(), bytes.data(), bytes.size(), &written));
    bytes.resize(written);
    return bytes;
}
}
// Complete official standalone .pco file; bytes.size() includes all metadata.
inline std::vector<std::uint8_t> compress(const std::vector<double>& values) {
    return detail::encode(values, gef_pco_compress_f64);
}
inline std::vector<double> decompress(const std::vector<std::uint8_t>& bytes, std::size_t count) {
    std::vector<double> values(count);
    detail::check(gef_pco_decompress_f64(bytes.data(), bytes.size(), values.data(), count));
    return values;
}
inline std::vector<std::uint8_t> compress_i64(const std::vector<std::int64_t>& values) {
    return detail::encode(values, gef_pco_compress_i64);
}
inline std::vector<std::int64_t> decompress_i64(const std::vector<std::uint8_t>& bytes, std::size_t count) {
    std::vector<std::int64_t> values(count);
    detail::check(gef_pco_decompress_i64(bytes.data(), bytes.size(), values.data(), count));
    return values;
}
}} // namespace gef_compressors::pcodec
