#include "BtrBlocks.hpp"
#include "btrblocks.hpp"
#include "compression/Datablock.hpp"
#include "storage/Chunk.hpp"
#include "extern/RoaringBitmap.hpp"
#include <cstring>
#include <mutex>
#include <stdexcept>
namespace gef_compressors { namespace btrblocks {
namespace {
void configure() { static std::once_flag once; std::call_once(once, [] { ::btrblocks::BtrBlocksConfig::configure(); }); }
constexpr size_t slack = 4096; // BtrBlocks decoders may read/write past the end (SIMD)
}
std::vector<uint8_t> compress(const std::vector<double>& values) {
    configure();
    const size_t n = values.size();
    auto data = std::unique_ptr<::btrblocks::u8[]>(new ::btrblocks::u8[n * sizeof(double)]);
    std::memcpy(data.get(), values.data(), n * sizeof(double));
    auto bitmap = std::unique_ptr<::btrblocks::BITMAP[]>(new ::btrblocks::BITMAP[n]);
    std::memset(bitmap.get(), 1, n * sizeof(::btrblocks::BITMAP));
    ::btrblocks::InputChunk chunk(std::move(data), std::move(bitmap), ::btrblocks::ColumnType::DOUBLE, n,
        n * sizeof(double));
    return ::btrblocks::Datablock::compress(chunk);
}
std::vector<double> decompress(const std::vector<uint8_t>& bytes, size_t count) {
    configure();
    std::vector<uint8_t> in(bytes.size() + slack, 0);
    std::memcpy(in.data(), bytes.data(), bytes.size());
    std::vector<double> out(count + slack);
    ::btrblocks::bitmap::BitmapWrapper* bitmap = nullptr;
    ::btrblocks::Datablock::decompress(in.data(), &bitmap, reinterpret_cast<::btrblocks::u8*>(out.data()));
    delete bitmap;
    out.resize(count);
    return out;
}
}}
