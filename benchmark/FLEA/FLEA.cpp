#include "FLEA.hpp"
#include <flea/encoder.hpp>
#include <mutex>
#include <stdexcept>
#include <string>
#ifdef __FAST_MATH__
#error FLEA adapter must be compiled without fast-math
#endif
namespace gef_compressors { namespace flea {
namespace {
// FFTWPlanCache owns scratch buffers, so the upstream API is not reentrant.
std::mutex& codec_mutex() { static std::mutex mutex; return mutex; }
constexpr size_t maximum_count = 1048576;
constexpr uint64_t maximum_l1 = uint64_t(1) << 50;
void check_count(size_t n) {
    if (n > maximum_count) throw std::invalid_argument("FLEA: blocks must contain at most 1048576 values");
}
void check_numeric_domain(const std::vector<int64_t>& values) {
    // The upstream FFT/quantizer uses double and unchecked signed arithmetic.
    // Bound its spectral magnitude and residuals far below int64 overflow.
    // This is an explicit eligibility restriction, never a substitute codec.
    uint64_t remaining = maximum_l1;
    for (auto value : values) {
        const uint64_t magnitude = value < 0 ? uint64_t(-(value + 1)) + 1 : uint64_t(value);
        if (magnitude > remaining)
            throw std::invalid_argument("FLEA: block sum(abs(values)) exceeds safe imported implementation limit 2^50");
        remaining -= magnitude;
    }
}
}
std::vector<uint8_t> compress_i64(const std::vector<int64_t>& values) {
    check_count(values.size());
    check_numeric_domain(values);
    const std::lock_guard<std::mutex> lock(codec_mutex());
    ::flea::FLEAEncoder encoder;
    return encoder.encode(values.data(), values.size());
}
std::vector<int64_t> decompress_i64(const std::vector<uint8_t>& bytes, size_t count) {
    check_count(count);
    if (bytes.size() < 4) throw std::invalid_argument("FLEA: truncated header");
    ::flea::BitReader header(bytes);
    const size_t stored_count = header.read(::flea::bit_width(::flea::MAX_LENGTH));
    if (stored_count != count) throw std::invalid_argument("FLEA: value count mismatch");
    if (count) {
        const bool padded = header.read(1);
        const size_t fft_count = padded ? header.read(::flea::bit_width(::flea::MAX_LENGTH)) : count;
        if (fft_count != ::flea::get_fft_length(count))
            throw std::invalid_argument("FLEA: invalid FFT padding length");
    }
    const std::lock_guard<std::mutex> lock(codec_mutex());
    ::flea::FLEAEncoder encoder;
    auto result = encoder.decode(bytes);
    if (result.size() != count) throw std::runtime_error("FLEA: decoded count mismatch");
    return result;
}
}}
