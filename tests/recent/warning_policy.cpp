// Exercise the actual generic benchmark with deliberately faulty decoders.
// Optional imported codec definitions are not enabled for this isolated target.
#include "../../benchmark/recent_compressors.cpp"
#include <sstream>

int main() {
    BenchmarkData data;
    data.filename = "injected_decoder";
    data.uncompressed_bits = 6*64;
    data.random_indices = {0,5};
    data.range_query_indices[2] = {1,4};
    Codec<int64_t> faulty{
        "fault_injection",
        [](const Values<int64_t>& values) {
            Bytes bytes(values.size()*sizeof(int64_t));
            std::memcpy(bytes.data(),values.data(),bytes.size());
            return bytes;
        },
        [](const Bytes& bytes, size_t count) {
            Values<int64_t> values(count);
            std::memcpy(values.data(),bytes.data(),bytes.size());
            values[0] ^= 1;
            return values;
        }, {}, false};
    std::ostringstream warning;
    auto* previous = std::cerr.rdbuf(warning.rdbuf());
    BenchmarkResult result;
    try {
        result = benchmark_blocks(faulty,Values<int64_t>{1,2,3,4,5,6},data,{2},3);
    } catch (...) {
        std::cerr.rdbuf(previous);
        throw;
    }
    std::cerr.rdbuf(previous);
    if (result.roundtrip_mismatches != 2 || result.validation_status != "failed" ||
        result.compressed_bits == 0 || result.compression_throughput_mbs <= 0 ||
        result.decompression_throughput_mbs <= 0 || result.random_access_ns <= 0 ||
        result.range_query_throughputs.size() != 1 ||
        warning.str().find("Warning:") == std::string::npos)
        throw std::runtime_error("value mismatches did not emit flagged timing metrics");

    faulty.decode = [](const Bytes&,size_t) { return Values<int64_t>{}; };
    bool rejected = false;
    try { benchmark_blocks(faulty,Values<int64_t>{1,2,3,4,5,6},data,{2},3); }
    catch (const std::runtime_error&) { rejected = true; }
    if (!rejected) throw std::runtime_error("malformed decode length was not rejected");
    std::cout << "Warnings retain real metrics; malformed lengths cannot reach queries\n";
}
