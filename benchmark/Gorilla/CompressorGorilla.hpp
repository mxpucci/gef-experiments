#pragma once
#include <concepts>
#include "../../lib/beringei-value/value_codec.hpp"
// Value-only Facebook Beringei codec; benchmark supplies independent block counts.
template<typename T>
struct CompressorGorilla {
    static_assert(std::same_as<T,double> || std::same_as<T,uint64_t>);
    gef_beringei::TimeSeriesStream stream;
    explicit CompressorGorilla(const T& value) { addValue(value); }
    void addValue(const T& value) { stream.appendValue(std::bit_cast<double>(value)); }
    void close() {} // Upstream bit writer already pads the last storage byte.
    size_t getSize() const { return stream.data_.size()*8; }
    const std::string& getBuffer() const { return stream.data_; }
};
