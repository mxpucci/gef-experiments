#include "../benchmark/Gorilla/CompressorGorilla.hpp"
#include "../benchmark/Gorilla/DecompressorGorilla.hpp"
#include <random>
#include <vector>
#include <cassert>
int main() {
    std::vector<uint64_t> values={0,0x8000000000000000ULL,1,0x8000000000000001ULL,
      0x7ff0000000000000ULL,0xfff0000000000000ULL,0x7ff8000000000042ULL,
      0x7ff0000000000001ULL,0xffffffffffffffffULL,0x3ff0000000000000ULL};
    std::mt19937_64 rng(42);for(int i=0;i<100000;i++)values.push_back(rng());
    for(size_t block:{size_t(1),size_t(2),size_t(1000)})for(size_t start=0;start<values.size();start+=block) {
        size_t end=std::min(start+block,values.size());
        CompressorGorilla<double> c(std::bit_cast<double>(values[start]));
        for(size_t i=start+1;i<end;i++)c.addValue(std::bit_cast<double>(values[i]));c.close();
        assert(c.getSize()==c.getBuffer().size()*8);
        DecompressorGorilla<double> d(c.getBuffer(),end-start);
        for(size_t i=start;i<end;i++) {if(i>start)assert(d.hasNext());assert(std::bit_cast<uint64_t>(d.storedValue)==values[i]);}
        assert(!d.hasNext());
    }
}
