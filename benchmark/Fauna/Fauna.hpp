#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// The Rust library is built from upstream with its `Value` alias widened from
// f32 to f64 (lib/fauna-ffi/upstream-f64.patch), so values cross the ABI as double.
extern "C" {
const char* gef_fauna_last_error();
int gef_fauna_compress(const double*, std::size_t, bool, std::uint8_t**, std::size_t*);
int gef_fauna_decompress(const std::uint8_t*, std::size_t, std::size_t, bool, double*);
void gef_fauna_free(std::uint8_t*, std::size_t);
}

namespace gef_compressors { namespace fauna_detail {
inline void check(int status) {
    if (status) throw std::runtime_error(std::string("Fauna/Macaque: ") + gef_fauna_last_error());
}
inline std::vector<std::uint8_t> compress64(const std::vector<double>& values, bool macaque) {
    std::uint8_t* bytes = nullptr;
    std::size_t size = 0;
    check(gef_fauna_compress(values.data(), values.size(), macaque, &bytes, &size));
    try {
        std::vector<std::uint8_t> result(bytes, bytes + size);
        gef_fauna_free(bytes, size);
        return result;
    } catch (...) { gef_fauna_free(bytes, size); throw; }
}
inline std::vector<double> decompress64(const std::vector<std::uint8_t>& bytes,
                                        std::size_t count, bool macaque) {
    std::vector<double> values(count);
    check(gef_fauna_decompress(bytes.data(), bytes.size(), count, macaque, values.data()));
    return values;
}
} }

namespace gef_compressors { namespace fauna {
inline std::vector<std::uint8_t> compress(const std::vector<double>& values) {
    return fauna_detail::compress64(values, false);
}
inline std::vector<double> decompress(const std::vector<std::uint8_t>& bytes, std::size_t count) {
    return fauna_detail::decompress64(bytes, count, false);
}
} }

namespace gef_compressors { namespace macaque {
inline std::vector<std::uint8_t> compress(const std::vector<double>& values) {
    return fauna_detail::compress64(values, true);
}
inline std::vector<double> decompress(const std::vector<std::uint8_t>& bytes, std::size_t count) {
    return fauna_detail::decompress64(bytes, count, true);
}
} }
