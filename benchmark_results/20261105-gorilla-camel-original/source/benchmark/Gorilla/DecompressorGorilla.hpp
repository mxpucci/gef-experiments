#pragma once
#include <concepts>
#include "../../lib/beringei-value/value_codec.hpp"
template<typename T>
struct DecompressorGorilla {
    static_assert(std::same_as<T,double> || std::same_as<T,uint64_t>);
    std::string_view data;
    uint64_t bitpos=0, previous=0, leading=0, trailing=0;
    size_t n, i=0;
    T storedValue{};
    DecompressorGorilla(std::string_view bytes,size_t count):data(bytes),n(count) { if(n) nextValue(); }
    void nextValue() {
        storedValue=std::bit_cast<T>(gef_beringei::TimeSeriesStream::readNextValue(data,bitpos,previous,leading,trailing));
        ++i;
    }
    bool hasNext() { if(i>=n) return false; nextValue(); return true; }
};
