#include "recent_compressors.hpp"
#include "DeXOR/DeXOR.hpp"
#include "ElfStar/ElfStar.hpp"
#include "SElfStar/SElfStar.hpp"
#include <memory>
#ifdef GEF_WITH_FAUNA
#include "Fauna/Fauna.hpp"
#endif
#ifdef GEF_WITH_PCODEC
#include "Pcodec/Pcodec.hpp"
#endif
#ifdef GEF_WITH_FLEA
#include "FLEA/FLEA.hpp"
#endif
#include <chrono>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>

namespace {
using Bytes = std::vector<uint8_t>;
template<class T> using Values = std::vector<T>;
template<class T> struct Codec {
    std::string name;
    std::function<Bytes(const Values<T>&)> encode;
    std::function<Values<T>(const Bytes&, size_t)> decode;
    std::function<Values<T>(const Bytes&, size_t, size_t)> decode_prefix;
};
std::vector<Codec<double>> codecs() {
    auto stream = std::make_shared<gef_compressors::selfstar::Encoder>();
    return {
        {"selfstar", [stream](const Values<double>& values) { return stream->compress_block(values); },
            gef_compressors::selfstar::decompress, gef_compressors::selfstar::decompress_prefix},
#ifdef GEF_WITH_FAUNA
        {"fauna", gef_compressors::fauna::compress, gef_compressors::fauna::decompress},
        {"macaque", gef_compressors::macaque::compress, gef_compressors::macaque::decompress},
#endif
        {"elfstar", gef_compressors::elfstar::compress, gef_compressors::elfstar::decompress, gef_compressors::elfstar::decompress_prefix},
        {"dexor", gef_compressors::dexor::compress, gef_compressors::dexor::decompress, gef_compressors::dexor::decompress_prefix},
#ifdef GEF_WITH_PCODEC
        {"pcodec", gef_compressors::pcodec::compress, gef_compressors::pcodec::decompress},
#endif
    };
}
std::vector<Codec<int64_t>> integer_codecs() {
    return {
#ifdef GEF_WITH_FLEA
        {"flea", gef_compressors::flea::compress_i64, gef_compressors::flea::decompress_i64},
#endif
#ifdef GEF_WITH_PCODEC
        {"pcodec_i64", gef_compressors::pcodec::compress_i64, gef_compressors::pcodec::decompress_i64},
#endif
    };
}
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point start) {
    return std::max(std::chrono::duration<double>(Clock::now() - start).count(), 1e-9);
}
template<class T> void check_equal(const Values<T>& expected, const Values<T>& actual) {
    if (expected.size() != actual.size() ||
        (!expected.empty() && std::memcmp(expected.data(), actual.data(), expected.size()*sizeof(T))))
        throw std::runtime_error("codec failed bit-exact round-trip validation");
}
}

std::vector<std::string> recent_compressor_names() {
    std::vector<std::string> names;
    for (const auto& c : codecs()) names.push_back(c.name);
    for (const auto& c : integer_codecs()) names.push_back(c.name);
    return names;
}
bool is_recent_compressor(const std::string& name) {
    for (const auto& c : codecs()) if (c.name == name) return true;
    return recent_compressor_uses_integers(name);
}
bool recent_compressor_uses_integers(const std::string& name) {
    for (const auto& c : integer_codecs()) if (c.name == name) return true;
    return false;
}
template<class T>
BenchmarkResult benchmark_blocks(const Codec<T>& codec, const Values<T>& values,
    const BenchmarkData& data, const std::vector<size_t>& ranges, size_t block_size) {
    if (!block_size) throw std::invalid_argument("block size must be positive");
    if (values.empty()) throw std::invalid_argument("empty benchmark dataset");
    const size_t n = values.size();
    const size_t nb = 1 + (n - 1) / block_size;
    std::vector<Bytes> blocks;
    blocks.reserve(nb);
    // Count the persisted block directory: N, block size, and nb+1 byte offsets.
    // Vector allocator/capacity overhead is not part of the compressed format.
    std::vector<uint64_t> offsets;
    offsets.reserve(nb + 1);
    offsets.push_back(0);
    BenchmarkResult result{};
    result.compressor = codec.name;
    result.dataset = data.filename;
    result.num_values = n;
    result.uncompressed_bits = data.uncompressed_bits;
    auto start = Clock::now();
    for (size_t b = 0; b < nb; ++b) {
        const size_t first = b * block_size;
        Values<T> input(values.begin() + first, values.begin() + std::min(n, first + block_size));
        blocks.push_back(codec.encode(input));
        offsets.push_back(offsets.back() + blocks.back().size());
    }
    const double encode_time = seconds(start);
    result.compressed_bits = (16 + offsets.size()*sizeof(uint64_t) + offsets.back())*8;
    result.compression_ratio = double(result.compressed_bits)/result.uncompressed_bits;
    const double mib = double(n)*sizeof(T)/(1024.0*1024.0);
    result.compression_throughput_mbs = mib/encode_time;
    // Validate outside timing. Never emit successful metrics for a lossy result.
    for (size_t b = 0; b < nb; ++b) {
        const size_t first = b*block_size, count = std::min(block_size, n-first);
        Values<T> expected(values.begin()+first, values.begin()+first+count);
        check_equal(expected, codec.decode(blocks[b], count));
    }
    start = Clock::now();
    for (size_t b = 0; b < nb; ++b) {
        auto output = codec.decode(blocks[b], std::min(block_size, n-b*block_size));
        do_not_optimize(output);
    }
    result.decompression_throughput_mbs = mib/seconds(start);
    // These are block accesses, not native random access. Streaming codecs
    // stop at the requested value, matching existing Gorilla/Chimp/Elf paths.
    const size_t nq = std::min<size_t>(100000, data.random_indices.size());
    start = Clock::now();
    for (size_t q = 0; q < nq; ++q) {
        const size_t index = data.random_indices[q];
        if (index >= n) throw std::out_of_range("point query index");
        const size_t b = index/block_size;
        const size_t count = std::min(block_size, n-b*block_size);
        auto output = codec.decode_prefix ? codec.decode_prefix(blocks[b], count, index%block_size+1)
                                          : codec.decode(blocks[b], count);
        do_not_optimize(output[index%block_size]);
    }
    const double access_time = seconds(start);
    result.random_access_ns = nq ? access_time*1e9/nq : 0;
    result.random_access_mbs = nq*sizeof(T)/(1024.0*1024.0)/access_time;
    for (size_t range : ranges) {
        if (range >= n || !range) continue;
        const auto& indices = data.range_query_indices.at(range);
        const size_t queries = std::min<size_t>(1000, indices.size());
        Values<T> output(range);
        start = Clock::now();
        for (size_t q = 0; q < queries; ++q) {
            const size_t begin = indices[q];
            if (begin > n-range) throw std::out_of_range("range query index");
            for (size_t b = begin/block_size; b <= (begin+range-1)/block_size; ++b) {
                const size_t count = std::min(block_size, n-b*block_size);
                const size_t needed = std::min(count, begin+range-b*block_size);
                auto decoded = codec.decode_prefix ? codec.decode_prefix(blocks[b], count, needed)
                                                   : codec.decode(blocks[b], count);
                const size_t lo = std::max(begin,b*block_size);
                const size_t hi = std::min(begin+range,b*block_size+decoded.size());
                std::copy(decoded.begin()+(lo-b*block_size), decoded.begin()+(hi-b*block_size),
                          output.begin()+(lo-begin));
            }
            do_not_optimize(output);
        }
        result.range_query_throughputs.emplace_back(range,
            double(queries)*range*sizeof(T)/(1024.0*1024.0)/seconds(start));
    }
    return result;
}

BenchmarkResult benchmark_recent_compressor(const std::string& name,
    const BenchmarkData& data, const std::vector<size_t>& ranges, size_t block_size) {
    for (const auto& codec : integer_codecs())
        if (codec.name == name) return benchmark_blocks(codec, data.raw_data, data, ranges, block_size);
    for (const auto& codec : codecs())
        if (codec.name == name) return benchmark_blocks(codec, data.double_data, data, ranges, block_size);
    throw std::invalid_argument("compressor not built: " + name);
}
