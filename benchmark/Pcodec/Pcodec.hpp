#pragma once
// Thin C++ wrapper over the official pco_c C bindings (pcodec v1.0.2).
#include <cpcodec.h>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace gef_compressors { namespace pcodec {
namespace detail {
// pco_c builds Rust slices from raw pointers, which must be non-null even when
// empty; empty std::vector::data() may be null.
template<class T> T* nonnull(T* p) { static T empty{}; return p ? p : &empty; }
inline void check(int code) {
    if (code != PcoSuccess) throw std::runtime_error("Pcodec failed (upstream error code " + std::to_string(code) + ")");
}
template<class T>
std::vector<std::uint8_t> encode(const std::vector<T>& values, unsigned char dtype) {
    const std::size_t capacity = pco_standalone_guarantee_file_size(values.size(), dtype);
    if (!capacity) throw std::runtime_error("Pcodec size bound failed");
    std::vector<std::uint8_t> bytes(capacity);
    std::size_t written = 0;
    // Null config = pco_c defaults: compression level 8, default paging.
    check(pco_standalone_simple_compress_into(nonnull(values.data()), values.size(), dtype, nullptr,
                                              bytes.data(), bytes.size(), &written));
    bytes.resize(written);
    return bytes;
}
template<class T>
std::vector<T> decode(const std::vector<std::uint8_t>& bytes, std::size_t count, unsigned char dtype) {
    std::vector<T> values(count);
    std::size_t written = 0;
    check(pco_standalone_simple_decompress_into(nonnull(bytes.data()), bytes.size(), dtype,
                                                nonnull(values.data()), count, &written));
    if (written != count) throw std::runtime_error("Pcodec decoded the wrong number of values");
    return values;
}
}
// Complete official standalone .pco file; bytes.size() includes all metadata.
inline std::vector<std::uint8_t> compress(const std::vector<double>& values) {
    return detail::encode(values, PCO_TYPE_F64);
}
inline std::vector<double> decompress(const std::vector<std::uint8_t>& bytes, std::size_t count) {
    return detail::decode<double>(bytes, count, PCO_TYPE_F64);
}
inline std::vector<std::uint8_t> compress_i64(const std::vector<std::int64_t>& values) {
    return detail::encode(values, PCO_TYPE_I64);
}
inline std::vector<std::int64_t> decompress_i64(const std::vector<std::uint8_t>& bytes, std::size_t count) {
    return detail::decode<std::int64_t>(bytes, count, PCO_TYPE_I64);
}
}} // namespace gef_compressors::pcodec
