from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[1]
p=root/'lib/Learn-to-Compress/headers/search_hyper.h'
s=p.read_text()
s=s.replace('#include "codecfactory.h"','').replace('#include <cassert>','').replace('using namespace Codecset;','')
# The published search helper is uint32-specific. Only bind it to the original
# uint64 codec/API and correct its byte-count denominator for that wider type.
s=s.replace('uint32_t * data','const uint64_t * data').replace('(4 * times * sample_size * 1.0)','(8 * times * sample_size * 1.0)')
header='''#pragma once
// Generated from the authors' search_hyper.h; search control flow unchanged.
#include <cassert>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <cstdlib>
#include <string>
#include <vector>
#include "piecewise_fix_integer_template.h"
namespace leco_native_tuning {
struct SampleCodec {
    Codecset::Leco_int<uint64_t> codec;
    void init(int blocks,int block_size,int){codec.init(blocks,block_size);}
    uint8_t* encodeArray8(const uint64_t* p,size_t n,uint8_t* out,size_t block){
        return codec.encodeArray8_int(p,n,out,block);
    }
};
using IntegerCODEC=SampleCodec;
struct CODECFactory {
    static SampleCodec* getFromName(const std::string&){
        static SampleCodec value;
        value=SampleCodec{};
        return &value;
    }
};
'''
footer='''
inline size_t choose(const uint64_t* data,size_t n){
    // Exact candidate grid, repetition count and sample rate from the authors'
    // experiments/search_hyper_regress.cpp. Tiny inputs cannot form a sample.
    int candidates[7]={200,400,800,1600,3200,6400,10000};
    if(n<20000)return std::min(n,size_t(200));
    auto result=pick_block_size(candidates,7,1,0.01,int(n),data,"leco_int_fix");
    if(result.select<0 || result.select>=7)
        throw std::runtime_error("LeCo upstream partition search found no candidate");
    return size_t(candidates[result.select]);
}
} // namespace leco_native_tuning
'''
(root/'benchmark/LeCo/NativeTuning.hpp').write_text(header+s+footer)
(root/'benchmark/LeCo/upstream-tuner.json').write_text(json.dumps({'commit':'28bf8bf3d0dd721baa2d6c714d8b454529af41aa','source':'headers/search_hyper.h','sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'changes':['Bind uint32 sampler to original uint64 codec through API adapter','Use 8-byte denominator for uint64 values','Use original 200..10000 candidate grid and 1% sample from search_hyper_regress.cpp']},indent=2)+'\n')
