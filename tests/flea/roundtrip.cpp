#include "FLEA/FLEA.hpp"
#include <flea/encoder.hpp>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
namespace codec = gef_compressors::flea;
void check(const std::vector<int64_t>& values) {
    auto encoded = codec::compress_i64(values);
    auto decoded = codec::decompress_i64(encoded, values.size());
    if (values != decoded) throw std::runtime_error("FLEA integer roundtrip mismatch");
    // Directly compare bytes against the unmodified authors' encoder API.
    ::flea::FLEAEncoder upstream;
    if (encoded != upstream.encode(values.data(), values.size()))
        throw std::runtime_error("FLEA upstream bitstream mismatch");
    if (upstream.decode(encoded) != values)
        throw std::runtime_error("FLEA upstream decode mismatch");
}
int main() {
    check({}); check({0}); check({1}); check({-1});
    check({0,1,-1,1,-1,0,123456789,-123456789});
    std::mt19937_64 random(0xf1ea);
    for (size_t n : {2,3,4,7,31,97,128,1000,1001,1024,2048,4097}) {
        check(std::vector<int64_t>(n,0));
        check(std::vector<int64_t>(n,123456789));
        check(std::vector<int64_t>(n,-123456789));
        std::vector<int64_t> values;
        for(size_t i=0;i<n;++i) values.push_back(int64_t(std::round(100000.0*std::sin(i*0.13))) + int64_t(random()%5)-2);
        check(values);
        for(auto& value: values) value = int64_t(random()%2000000001)-1000000000;
        check(values);
    }
    check({int64_t(1)<<50});
    check({-(int64_t(1)<<50)});
    for (const auto& values : {std::vector<int64_t>{INT64_MIN}, std::vector<int64_t>{INT64_MAX},
             std::vector<int64_t>{int64_t(1)<<50,1}}) {
        bool rejected=false;
        try { codec::compress_i64(values); } catch(const std::invalid_argument&) { rejected=true; }
        if(!rejected) throw std::runtime_error("FLEA unsafe integer range accepted");
    }
    auto bytes=codec::compress_i64({1,2,3});
    bool rejected=false;
    try { codec::decompress_i64(bytes,4); } catch(const std::invalid_argument&) { rejected=true; }
    if(!rejected) throw std::runtime_error("FLEA wrong count accepted");
    for(size_t n=0;n<4;++n) {
        rejected=false;
        try {codec::decompress_i64(std::vector<uint8_t>(bytes.begin(),bytes.begin()+n),3);}
        catch(const std::invalid_argument&) { rejected=true; }
        if(!rejected) throw std::runtime_error("FLEA truncated header accepted");
    }
    std::cout << "Official FLEA integer roundtrips and upstream parity passed\n";
}
