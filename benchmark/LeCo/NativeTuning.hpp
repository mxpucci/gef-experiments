#pragma once
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





int randomint(int m)
{
    return rand() % m;
}

struct codec_vote{
    int select;
    double compression_rate;
};
codec_vote pick_block_size(int bsize [],int choice, int times, double sample_rate,int N, const uint64_t * data, std::string codec_name){
    
    int select_= -1;

    int delta = 0;
    double mini_size = 1.0;
    std::vector<int> pick_vec;
    double sample_size_percent = sample_rate;
    int sample_size = N * sample_size_percent;

    for (int p = 0; p < times; p++)
    {
        pick_vec.push_back(randomint(1 / sample_size_percent));
    }
    for(int j = 0; j < choice; ++j)
    {
        IntegerCODEC &codec = *CODECFactory::getFromName(codec_name);
        int block_size = bsize[j];
        int blocks = N / block_size;
        codec.init(blocks, block_size, delta);
        int totalsize = 0;
        
        for (int p = 0; p < times; p++)
        {
            int pick_block = pick_vec[p];
            for (int i = 0; i < sample_size / block_size; i++)
            {
                uint8_t *descriptor = (uint8_t *)malloc(block_size * sizeof(uint64_t)+1024);
                uint8_t *res = descriptor;
                res = codec.encodeArray8(data + pick_block * sample_size + (i * block_size), block_size, descriptor, i);
                totalsize += (res - descriptor);
                free(descriptor);
            }
            if (sample_size % block_size != 0)
            {
                int block_length = sample_size - (sample_size / block_size) * block_size;
                uint8_t *descriptor = (uint8_t *)malloc(block_length * sizeof(uint64_t)+1024);
                uint8_t *res = descriptor;
                res = codec.encodeArray8(data+ (sample_size / block_size) * block_size, block_length, descriptor, sample_size / block_size);
                totalsize += (res - descriptor);
                free(descriptor);
            }
        }
        double compressrate = totalsize * 1.0 / (8 * times * sample_size * 1.0);
        if (compressrate < mini_size)
        {
            mini_size = compressrate;
            select_ = j;
        }

    }
    assert (select_ != -1);
    codec_vote tmp;
    tmp.select = select_;
    tmp.compression_rate = mini_size;
    return tmp;
}



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
