#include "../benchmark/ElfStar/ElfStar.hpp"
#include <iostream>
#include <random>
namespace es=gef_compressors::elfstar;
void require(bool b,const char* message) {if(!b)throw std::runtime_error(message);}
void check(const std::vector<double>& values) {
    auto bytes=es::compress(values);auto decoded=es::decompress(bytes,values.size());
    require(decoded.size()==values.size(),"length mismatch");
    size_t pn=std::min<size_t>(3,values.size());
    auto prefix=es::decompress_prefix(bytes,values.size(),pn);
    require(prefix.size()==pn,"prefix length mismatch");
    for(size_t i=0;i<pn;++i)require(es::detail::bits(prefix[i])==es::detail::bits(values[i]),"corner prefix mismatch");
    for(size_t i=0;i<values.size();++i)if(es::detail::bits(values[i])!=es::detail::bits(decoded[i]))throw std::runtime_error("roundtrip mismatch at "+std::to_string(i));
}
int main() {
    // Unmodified official Java ElfStarCompressor/ElfStarXORCompressor fixture.
    const std::vector<double> golden={1.1,2.2,3.3,3.3,3.3,-4.4,0.1,0.01,0.001,
        123.456,18.18,18.18,0.0,-0.0,42.0,42.000001};
    const std::string hex="8245a858b5947b34bc5b0ea2def64f60f62f6916e8b68f90020a910002dcb4dfa7e04ffc666666666666c33ffa44b99999999999aaa85801b9ffa185b076dc1bf876e982db3ccc70880645f0fc2e022a8d7047f7a00001";
    const auto encoded=es::compress(golden);
    require(encoded.size()==8+hex.size()/2,"official Java fixture length mismatch");
    for(size_t i=0;i<hex.size()/2;++i)require(encoded[8+i]==std::stoul(hex.substr(2*i,2),nullptr,16),"official Java bitstream mismatch");
    check(golden);
    for(size_t n=0;n<=golden.size();++n) {
        auto prefix=es::decompress_prefix(encoded,golden.size(),n);
        require(prefix.size()==n,"prefix length mismatch");
        for(size_t i=0;i<n;++i)require(es::detail::bits(prefix[i])==es::detail::bits(golden[i]),"golden prefix mismatch");
    }
    // A prefix must not validate or decode the unrequested tail.
    auto corrupt_tail=encoded;corrupt_tail.back()^=1;
    auto prefix=es::decompress_prefix(corrupt_tail,golden.size(),1);
    require(es::detail::bits(prefix[0])==es::detail::bits(golden[0]),"prefix inspected tail");
    bool corrupt_failed=false;
    try{es::decompress(corrupt_tail,golden.size());}catch(const std::runtime_error&){corrupt_failed=true;}
    require(corrupt_failed,"full decoder missed corrupt tail");
    bool prefix_failed=false;
    try{es::decompress_prefix(encoded,golden.size(),golden.size()+1);}catch(const std::invalid_argument&){prefix_failed=true;}
    require(prefix_failed,"oversized prefix accepted");
    prefix_failed=false;
    try{es::decompress_prefix(encoded,golden.size()+1,0);}catch(const std::runtime_error&){prefix_failed=true;}
    require(prefix_failed,"zero prefix missed count mismatch");
    check({});check({0});check({-0.0});check({0.0,-0.0,0.0});
    check({1.1,2.2,3.3,-4.4,0.1,0.01,0.001,123.456,1e-20,1e20});
    check(std::vector<double>(10000,0.1));
    std::vector<double> values;
    for(uint64_t bits:{UINT64_C(1),UINT64_C(0x8000000000000001),UINT64_C(0x7fefffffffffffff),
        UINT64_C(0x0010000000000000),UINT64_C(0x7ff0000000000000),UINT64_C(0xfff0000000000000),
        UINT64_C(0x7ff8000000000000),UINT64_C(0x7ff8000000000001),UINT64_C(0x7ff0000000000001)})values.push_back(es::detail::value(bits));
    check(values);
    std::mt19937_64 random(919);
    for(size_t trial=0;trial<100;++trial) {
        values.clear();for(size_t i=0;i<1000;++i)values.push_back(es::detail::value(random()));check(values);
        values.clear();for(size_t i=0;i<1000;++i)values.push_back(double(int(random()%200000)-100000)/1000.0);check(values);
    }
    for(int e=-320;e<=300;++e){values.clear();for(int i=1;i<30;++i)values.push_back(i*es::detail::power10(e));check(values);}
    auto bytes=es::compress({1.1,2.2,3.3});
    for(size_t n=0;n<bytes.size();++n){bool failed=false;try{es::decompress({bytes.begin(),bytes.begin()+n},3);}catch(const std::runtime_error&){failed=true;}require(failed,"truncation accepted");}
    bool failed=false;try{es::decompress(bytes,2);}catch(const std::runtime_error&){failed=true;}require(failed,"wrong count accepted");
    bytes.push_back(0);failed=false;try{es::decompress(bytes,3);}catch(const std::runtime_error&){failed=true;}require(failed,"trailing byte accepted");
    std::cout << "ElfStar round trips and malformed inputs passed\n";
}
