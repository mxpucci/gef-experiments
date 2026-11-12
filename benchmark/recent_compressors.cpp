#include "recent_compressors.hpp"
#include "range_queries.hpp"
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
#ifdef GEF_WITH_FASTPFOR
#include "FastPFor/FastPFor.hpp"
#endif
#ifdef GEF_WITH_BTRBLOCKS
#include "BtrBlocks/BtrBlocks.hpp"
#endif
#include <chrono>
#include <cstring>
#include <functional>
#include <iostream>
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
    bool float32_input = false;
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
#ifdef GEF_WITH_BTRBLOCKS
        {"btrblocks", gef_compressors::btrblocks::compress, gef_compressors::btrblocks::decompress},
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
#ifdef GEF_WITH_FASTPFOR
        {"fastpfor", gef_compressors::fastpfor::compress_for_i64, gef_compressors::fastpfor::decompress_for_i64},
        {"fastpfor_delta", gef_compressors::fastpfor::compress_delta_i64, gef_compressors::fastpfor::decompress_delta_i64},
#endif
    };
}
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point start) {
    return std::max(std::chrono::duration<double>(Clock::now() - start).count(), 1e-9);
}
// Shared round-trip policy (benchmark_common.hpp): |error| > 1e-9 fails,
// bit differences are also counted in bitwise_mismatches.
template<class T> void count_differences(BenchmarkResult& result, const Values<T>& expected,
    const Values<T>& actual, size_t first_index) {
    if (expected.size() != actual.size())
        throw std::runtime_error("codec decoded the wrong number of values; queries cannot be measured safely");
    check_roundtrip(result, expected.data(), actual.data(), expected.size(), first_index);
}
// Explicit benchmark preprocessing, kept separate from the strict codec API.
// Count changes before encoding and validate against exactly this converted input.
template<class T> size_t convert_to_float32(Values<T>& values) {
    size_t changed = 0;
    for (auto& value : values) {
        const T converted = static_cast<T>(static_cast<float>(value));
        changed += std::memcmp(&value, &converted, sizeof(T)) != 0;
        value = converted;
    }
    return changed;
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
    result.input_conversion = codec.float32_input ? "float32" : "none";
    result.access_layout = "external_blocks";
    result.compression_unit_values = block_size;
    const size_t gef_ram0 = gef_live_heap_bytes();
    auto start = Clock::now();
    for (size_t b = 0; b < nb; ++b) {
        const size_t first = b * block_size;
        Values<T> input(values.begin() + first, values.begin() + std::min(n, first + block_size));
        if (codec.float32_input) result.input_changed_values += convert_to_float32(input);
        blocks.push_back(codec.encode(input));
        offsets.push_back(offsets.back() + blocks.back().size());
    }
    const double encode_time = seconds(start);
    if (result.input_changed_values) {
        std::cerr << "\nWarning: " << codec.name << " converted " << result.input_changed_values
                  << "/" << n << " input values to float32; emitting metrics for the converted input. "
                  << "Original float64 bits are not preserved; see input_conversion and input_changed_values.\n";
    }
    result.compressed_bits = (16 + offsets.size()*sizeof(uint64_t) + offsets.back())*8;
    result.compression_ratio = double(result.compressed_bits)/result.uncompressed_bits;
    const double mib = double(n)*sizeof(T)/(1024.0*1024.0);
    for (auto& block : blocks) block.shrink_to_fit(); // untimed: harness container capacity
    result.ram_bytes = gef_heap_delta(gef_ram0);
    result.compression_throughput_mbs = mib/encode_time;
    // Validate outside timing. Value mismatches warn and remain visible in CSV;
    // malformed output/counts still fail the case instead of timing invalid queries.
    for (size_t b = 0; b < nb; ++b) {
        const size_t first = b*block_size, count = std::min(block_size, n-first);
        Values<T> expected(values.begin()+first, values.begin()+first+count);
        if (codec.float32_input) convert_to_float32(expected);
        count_differences(result, expected, codec.decode(blocks[b], count), first);
    }
    finish_roundtrip_validation(result);
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
    // Range queries (range_queries.hpp). Reference: the codec's own full
    // decompression, built untimed (the timed full decode above discards it).
    Values<T> reference;
    if (data.range_options.verify) {
        reference.resize(n);
        for (size_t b = 0; b < nb; ++b) {
            const size_t count = std::min(block_size, n-b*block_size);
            const auto decoded = codec.decode(blocks[b], count);
            std::copy(decoded.begin(), decoded.begin()+std::min(count, decoded.size()), reference.begin()+b*block_size);
        }
    }
    run_range_queries(result, data, ranges, n, reference.empty() ? nullptr : reference.data(),
        [&](size_t begin, size_t range, Values<T>& output) {
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
        });
    return result;
}

template<class T,class Factory>
BenchmarkResult benchmark_indexed(const std::string& name,const Values<T>& values,
    const BenchmarkData& data,const std::vector<size_t>& ranges,Factory make,
    bool float32_input,const std::string& layout,size_t unit_size,size_t page_size) {
    if(values.empty()) throw std::invalid_argument("empty benchmark dataset");
    const size_t n=values.size();
    BenchmarkResult result{};
    result.compressor=name;result.dataset=data.filename;result.num_values=n;
    result.uncompressed_bits=data.uncompressed_bits;
    result.input_conversion=float32_input?"float32":"none";
    result.access_layout=layout;result.compression_unit_values=unit_size;result.native_page_values=page_size;
    const size_t gef_ram0 = gef_live_heap_bytes();
    auto start=Clock::now();
    auto index=make();
    const double encode_time=seconds(start);
    if(index.size()!=n) throw std::runtime_error("indexed codec count mismatch");
    result.input_changed_values=index.changed_values();
    if(result.input_changed_values) {
        std::cerr << "\nWarning: " << name << " converted " << result.input_changed_values << "/" << n
                  << " input values to float32; emitting metrics for the converted input. "
                  << "Original float64 bits are not preserved; see input_conversion and input_changed_values.\n";
    }
    result.compressed_bits=index.size_in_bytes()*8;
    result.compression_ratio=double(result.compressed_bits)/result.uncompressed_bits;
    const double mib=double(n)*sizeof(T)/(1024.0*1024.0);
    result.ram_bytes = gef_heap_delta(gef_ram0);
    result.compression_throughput_mbs=mib/encode_time;
    // Whole-input validation is untimed. Temporary validation windows do NOT
    // define compressed storage or point-access granularity.
    for(size_t first=0;first<n;first+=65536) {
        const size_t count=std::min<size_t>(65536,n-first);
        Values<T> expected(values.begin()+first,values.begin()+first+count),decoded(count);
        if(float32_input) convert_to_float32(expected);
        index.read(first,count,decoded.data());
        count_differences(result,expected,decoded,first);
    }
    finish_roundtrip_validation(result);
    Values<T> full(n);  // kept as the range-query reference
    start=Clock::now();
    index.read(0,n,full.data());
    do_not_optimize(full);
    result.decompression_throughput_mbs=mib/seconds(start);
    const size_t nq=std::min<size_t>(100000,data.random_indices.size());
    start=Clock::now();
    for(size_t q=0;q<nq;++q) {
        T value{};
        index.read(data.random_indices[q],1,&value);
        do_not_optimize(value);
    }
    const double access_time=seconds(start);
    result.random_access_ns=nq?access_time*1e9/nq:0;
    result.random_access_mbs=nq*sizeof(T)/(1024.0*1024.0)/access_time;
    run_range_queries(result,data,ranges,n,full.data(),
        [&](size_t first,size_t range,Values<T>& output){index.read(first,range,output.data());});
    return result;
}

BenchmarkResult benchmark_recent_compressor(const std::string& name,
    const BenchmarkData& data, const std::vector<size_t>& ranges, size_t block_size,
    const NativeAccessOptions&) {
    for (const auto& codec : integer_codecs())
        if (codec.name == name) return benchmark_blocks(codec, data.raw_data, data, ranges, block_size);
    for (const auto& codec : codecs())
        if (codec.name == name) return benchmark_blocks(codec, data.double_data, data, ranges, block_size);
    throw std::invalid_argument("compressor not built: " + name);
}
