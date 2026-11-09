#include "Fauna/Indexed.hpp"
#include <algorithm>
#include <cmath>
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
static void append_double(std::vector<uint8_t>& bytes,double value){uint64_t bits;std::memcpy(&bits,&value,8);append_word(bytes,bits);}
static void append_bytes(std::vector<uint8_t>& bytes,const std::vector<uint8_t>& data){append_word(bytes,data.size());bytes.insert(bytes.end(),data.begin(),data.end());}
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
static void fauna_case(const std::vector<double>& input,size_t batch){
    using Index=gef_compressors::fauna::Indexed;
    // Reference: full reconstruction by the authors' segment decoder for the
    // same fitting batches. Swing lines are evaluated in f64, the same type as
    // the values, so a model value may differ from the input by rounding.
    std::vector<double> expected;
    for(size_t first=0;first<input.size();first+=batch){
        size_t n=std::min(batch,input.size()-first);
        std::vector<double> values(input.begin()+first,input.begin()+first+n);
        auto original=gef_compressors::fauna::decompress(gef_compressors::fauna::compress(values),n);
        expected.insert(expected.end(),original.begin(),original.end());
    }
    for(size_t i=0;i<input.size();++i)require(std::fabs(expected[i]-input[i])<=1e-9,"Fauna f64 round trip outside 1e-9");
    std::vector<uint8_t> bytes;
    {
        Index index(input,batch);require(index.changed_values()==0,"f64 input must not be converted");queries(index,expected);bytes=index.bytes();
    }
    Index reopened(bytes);queries(reopened,expected);
    auto truncated=bytes;truncated.pop_back();rejects([&]{Index bad(truncated);});
    rejects([&]{Index bad(input,0);});
}
static void fauna_residual_fixture(bool swing){
    // Build a valid native model-plus-residual segment. Residual bits come
    // from the ORIGINAL Macaque encoder, with its initial raw value omitted.
    const size_t model_n=1024,n=model_n+3;
    std::vector<double> values(n,4.0);
    if(swing)for(size_t i=0;i<model_n;++i)values[i]=4.0+double(i)/4;
    for(size_t i=model_n;i<n;++i)values[i]=values[model_n-1]+double(i-model_n+1)/4;
    // Segment layout: model(1), first/last(8+8), min/max(8+8), then length-prefixed bytes.
    auto timestamps_frame=gef_compressors::fauna::compress(std::vector<double>(n,4.0));
    const size_t times_n=word(timestamps_frame,24+33);
    std::vector<uint8_t> times(timestamps_frame.begin()+24+41,timestamps_frame.begin()+24+41+times_n);
    std::vector<double> residual_values{values[model_n-1],values[model_n],values[model_n+1],values[model_n+2]};
    auto residual_frame=gef_compressors::macaque::compress(residual_values);
    std::vector<uint8_t> residuals(residual_frame.begin()+24,residual_frame.end());residuals.push_back(3);
    std::vector<uint8_t> model_values;
    if(swing){model_values.push_back(2);append_double(model_values,values[model_n-1]);}
    std::vector<uint8_t> segment{uint8_t(swing?1:0)};
    append_word(segment,0);append_word(segment,n-1);append_double(segment,4.0);append_double(segment,values.back());
    append_bytes(segment,times);append_bytes(segment,model_values);append_bytes(segment,residuals);
    std::vector<uint8_t> native{'G','F','F','A','U','0','0','2'};
    append_word(native,n);append_word(native,1);native.insert(native.end(),segment.begin(),segment.end());
    auto reference=gef_compressors::fauna::decompress(native,n);
    require(!std::memcmp(reference.data(),values.data(),n*8),"residual fixture does not match original decoder");
    std::vector<uint8_t> frame{'G','F','F','I','D','X','0','2'};
    for(uint64_t x:{uint64_t(n),uint64_t(1),uint64_t(65536),uint64_t(0),uint64_t(0),uint64_t(64),uint64_t(segment.size())})append_word(frame,x);
    frame.insert(frame.end(),segment.begin(),segment.end());
    gef_compressors::fauna::Indexed indexed(frame);
    const std::vector<double>& expected=values;queries(indexed,expected);
    for(size_t i=model_n-1;i<n;++i){double result;indexed.read(i,1,&result);require(result==expected[i],"model/residual boundary differs");}
}
int main(){
    std::mt19937_64 rng(71);std::vector<double> values;
    fauna_case({},65536);fauna_case({4.0},65536);fauna_case({0.1,0.2,19705.4018},65536);
    values.resize(131077);
    for(size_t i=0;i<values.size();++i){
        if(i<32768)values[i]=4.0;
        else if(i<65536)values[i]=double(i-32768)/8;
        else if(i<98304)values[i]=double(int(rng()%100000)-50000)/16;
        else values[i]=double(i%17)/4;
    }
    fauna_case(values,65536);
    fauna_case(std::vector<double>(values.begin(),values.begin()+4099),257);
    fauna_residual_fixture(false);fauna_residual_fixture(true);
    std::cout << "Native indexed access: original-grid agreement, segments, boundaries, serialization and bounds passed\n";
}
