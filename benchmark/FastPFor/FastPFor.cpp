#include "FastPFor.hpp"
#include <compositecodec.h>
#include <fastpfor.h>
#include <variablebyte.h>
#include <algorithm>
#include <cstring>
#include <stdexcept>
namespace gef_compressors { namespace fastpfor {
namespace {
using Codec = FastPForLib::CompositeCodec<FastPForLib::FastPFor<4>, FastPForLib::VariableByte>;
Codec& codec() { thread_local Codec instance; return instance; }  // reuses upstream scratch buffers
// Layout: int64 header (block minimum or first value), then the codec's 32-bit words.
std::vector<uint8_t> encode(int64_t header, const std::vector<uint64_t>& input) {
    std::vector<uint32_t> words(2 * input.size() + 1024);
    size_t nwords = words.size();
    codec().encodeArray(input.data(), input.size(), words.data(), nwords);
    std::vector<uint8_t> out(sizeof header + nwords * 4);
    std::memcpy(out.data(), &header, sizeof header);
    std::memcpy(out.data() + sizeof header, words.data(), nwords * 4);
    return out;
}
int64_t decode(const std::vector<uint8_t>& bytes, size_t count, std::vector<uint64_t>& output) {
    if (bytes.size() < 8 || (bytes.size() - 8) % 4) throw std::invalid_argument("FastPFor: malformed block");
    int64_t header;
    std::memcpy(&header, bytes.data(), sizeof header);
    std::vector<uint32_t> words((bytes.size() - 8) / 4);
    std::memcpy(words.data(), bytes.data() + 8, words.size() * 4);
    output.resize(count + 1024);
    size_t n = output.size();
    codec().decodeArray(words.data(), words.size(), output.data(), n);
    output.resize(n);
    return header;
}
}
std::vector<uint8_t> compress_for_i64(const std::vector<int64_t>& values) {
    const int64_t base = values.empty() ? 0 : *std::min_element(values.begin(), values.end());
    std::vector<uint64_t> input(values.size());
    for (size_t i = 0; i < values.size(); ++i) input[i] = uint64_t(values[i]) - uint64_t(base);
    return encode(base, input);
}
std::vector<int64_t> decompress_for_i64(const std::vector<uint8_t>& bytes, size_t count) {
    std::vector<uint64_t> decoded;
    const int64_t base = decode(bytes, count, decoded);
    if (decoded.size() != count) throw std::runtime_error("FastPFor: wrong value count");
    std::vector<int64_t> out(count);
    for (size_t i = 0; i < count; ++i) out[i] = int64_t(decoded[i] + uint64_t(base));
    return out;
}
std::vector<uint8_t> compress_delta_i64(const std::vector<int64_t>& values) {
    std::vector<uint64_t> input(values.empty() ? 0 : values.size() - 1);
    for (size_t i = 1; i < values.size(); ++i) {
        const int64_t d = int64_t(uint64_t(values[i]) - uint64_t(values[i - 1]));
        input[i - 1] = (uint64_t(d) << 1) ^ uint64_t(d >> 63);
    }
    return encode(values.empty() ? 0 : values[0], input);
}
std::vector<int64_t> decompress_delta_i64(const std::vector<uint8_t>& bytes, size_t count) {
    std::vector<uint64_t> decoded;
    const int64_t first = decode(bytes, count ? count - 1 : 0, decoded);
    if (count && decoded.size() != count - 1) throw std::runtime_error("FastPFor: wrong value count");
    std::vector<int64_t> out(count);
    if (!count) return out;
    uint64_t current = uint64_t(first);
    out[0] = first;
    for (size_t i = 1; i < count; ++i) {
        const uint64_t z = decoded[i - 1];
        current += (z >> 1) ^ (~(z & 1) + 1);
        out[i] = int64_t(current);
    }
    return out;
}
}}
