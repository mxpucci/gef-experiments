#include "recent_compressors.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
int main() {
    BenchmarkData data;
    data.filename="synthetic.bin";
    // Exactly representable in f32 as well as f64 for the Fauna implementations.
    for (size_t i=0;i<1031;++i) data.double_data.push_back(int(i%97)/4.0-8.0);
    for (size_t i=0;i<1031;++i) data.raw_data.push_back(int64_t(i%97)-48);
    data.uncompressed_bits=data.double_data.size()*64;
    data.random_indices={0,1,15,16,17,1000,1030};
    data.range_query_indices[3]={0,14,15,16,1000,1028};
    data.range_query_indices[35]={0,14,15,16,996};
    for (const auto& name : recent_compressor_names()) {
        auto result=benchmark_recent_compressor(name,data,{3,35},16);
        if (!result.compressed_bits || result.num_values != 1031 ||
            !std::isfinite(result.compression_ratio) ||
            result.range_query_throughputs.size()!=2)
            throw std::runtime_error("bad harness metrics: "+name);
        std::cout << name << " validated with short final block and crossing queries\n";
    }
    bool failed=false;
    try { benchmark_recent_compressor("absent",data,{},16); }
    catch(const std::invalid_argument&) { failed=true; }
    if(!failed) throw std::runtime_error("unknown codec accepted");
}
