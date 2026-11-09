#include "Pcodec/Indexed.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

static void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
template<class F> static void rejects(F action){bool bad=false;try{action();}catch(const std::exception&){bad=true;}require(bad,"invalid indexed input accepted");}
static uint64_t word(const std::vector<uint8_t>& bytes,size_t at){
    uint64_t value=0;for(size_t i=0;i<8;++i)value|=uint64_t(bytes.at(at+i))<<(8*i);return value;
}
static void append_word(std::vector<uint8_t>& bytes,uint64_t value){for(size_t i=0;i<8;++i)bytes.push_back(uint8_t(value>>(8*i)));}
template<class T,class Index> static void queries(Index& index,const std::vector<T>& expected){
    require(index.size()==expected.size(),"wrong index count");
    require(index.size_in_bytes()==index.bytes().size(),"index bytes not counted");
    std::vector<T> all(expected.size());index.read(0,all.size(),all.data());
    require(all.empty()||!std::memcmp(all.data(),expected.data(),all.size()*sizeof(T)),"full indexed reconstruction differs");
    index.read(expected.size(),0,nullptr);
    T value{};rejects([&]{index.read(expected.size(),1,&value);});
    rejects([&]{index.read(std::numeric_limits<size_t>::max(),1,&value);});
    if(expected.empty())return;
    std::mt19937_64 rng(713);
    for(size_t attempt=0;attempt<300;++attempt){
        size_t first=attempt<4?std::min(expected.size()-1,attempt*256):rng()%expected.size();
        index.read(first,1,&value);
        require(!std::memcmp(&value,&expected[first],sizeof(T)),"point query differs");
        size_t length=std::min<size_t>(expected.size()-first,1+rng()%2000);
        std::vector<T> range(length);index.read(first,length,range.data());
        require(!std::memcmp(range.data(),expected.data()+first,length*sizeof(T)),"range query differs");
    }
}
template<class T> static void pco_case(const std::vector<T>& input){
    using Index=gef_compressors::pcodec::Indexed<T>;
    std::vector<uint8_t> bytes;
    {
        Index index(input,4096,513);queries(index,input);bytes=index.bytes();
    }
    Index reopened(bytes);queries(reopened,input);
    auto truncated=bytes;truncated.pop_back();rejects([&]{Index bad(truncated);});
    rejects([&]{Index bad(input,0,513);});rejects([&]{Index bad(input,4096,0);});
    if(input.size()>5000){
        // An unrelated page may be damaged without being visited by a point
        // query into a later native page of the SAME compression chunk.
        const size_t chunks=word(bytes,24),pages=word(bytes,32);
        require(chunks>1&&pages>chunks,"test did not exercise native multi-page chunks");
        const size_t first=64+48*chunks;
        const size_t off=word(bytes,first+24),len=word(bytes,first+32);
        std::fill(bytes.begin()+off,bytes.begin()+off+len,0xff);
        const size_t target=word(bytes,first+40);
        Index selected(bytes);T result{};selected.read(target,1,&result);
        require(!std::memcmp(&result,&input[target],sizeof(T)),"query decoded an unrelated page");
    }
}
int main(){
    pco_case(std::vector<double>{});pco_case(std::vector<int64_t>{});
    pco_case(std::vector<double>{-0.0});pco_case(std::vector<int64_t>{INT64_MIN,INT64_MAX});
    std::mt19937_64 rng(71);std::vector<double> values;std::vector<int64_t> integers;
    for(size_t i=0;i<12291;++i){uint64_t bits=rng();double value;int64_t integer;std::memcpy(&value,&bits,8);std::memcpy(&integer,&bits,8);values.push_back(value);integers.push_back(integer);}
    pco_case(values);pco_case(integers);
    gef_compressors::pcodec::Indexed<double> typed(values,4096,513);
    rejects([&]{gef_compressors::pcodec::Indexed<int64_t> wrong(typed.bytes());});
    // Default Pco sizes also cross a native chunk boundary.
    values.resize(262144+19);for(size_t i=0;i<values.size();++i)values[i]=double(i%1000)/100;
    gef_compressors::pcodec::Indexed<double> defaults(values);queries(defaults,values);
    std::cout << "Native indexed access (Pcodec): f64/i64 bits, pages, chunk boundaries, serialization and bounds passed\n";
}
