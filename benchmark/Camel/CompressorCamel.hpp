#pragma once

/**
 * Camel Compressor - Faithful C++ port from:
 * https://github.com/yoyo185644/camel
 * 
 * Ported from Camel.java
 * 
 * Key features:
 * - Separates integer and decimal parts
 * - Uses threshold-based encoding for decimals
 * - Delta encoding for integer parts
 */

#include <cstdint>
#include <cmath>
#include <vector>
#include <limits>
#include "../lib/BitStream.hpp"

template<typename T = double>
class CompressorCamel {
    static_assert(std::is_same_v<T, double>, "CompressorCamel only supports double");
    
    int64_t storedVal = 0;  // Stores integer part of previous value
    bool first = true;
    size_t size = 0;
    
    static constexpr int DECIMAL_MAX_COUNT = 3;
    
    // m value bits for different decimal counts
    static constexpr int mValueBits[] = {3, 5, 7, 10, 15};
    
    // Thresholds for decimal encoding
    static constexpr int64_t threshold[] = {5, 25, 125, 625};
    
    // Powers of 10
    static constexpr int64_t powers[] = {1L, 10L, 100L, 1000L, 10000L, 100000L};
    
    BitStream out{};
    
    // Calculate decimal count and decimal value
    void cal_decimal_count(double value, int& decimal_count, int64_t& decimal_value) {
        double factor = 1;
        decimal_count = 0;
        value = std::abs(value);
        double epsilon = 0.0000001;
        
        // Camel ultimately clamps to DECIMAL_MAX_COUNT, so avoid an unbounded loop
        // on values that do not have a short terminating decimal representation.
        while (decimal_count < DECIMAL_MAX_COUNT &&
               std::abs(value * factor - std::round(value * factor)) > epsilon) {
            factor *= 10.0;
            decimal_count++;
        }
        
        decimal_value = 0;
        if (decimal_count == 0) {
            decimal_count = 1;
        }
        
        if (decimal_count > 0 && decimal_count <= DECIMAL_MAX_COUNT) {
            decimal_value = static_cast<int64_t>(std::round(value * powers[decimal_count])) % powers[decimal_count];
        } else {
            decimal_value = static_cast<int64_t>(std::round(value * powers[DECIMAL_MAX_COUNT])) % powers[DECIMAL_MAX_COUNT];
            decimal_count = DECIMAL_MAX_COUNT;
        }
    }
    
    // Write first value
    size_t writeFirst(int64_t value) {
        first = false;
        storedVal = static_cast<int64_t>(static_cast<T>(*reinterpret_cast<double*>(&value)));
        out.append(value, 64);
        size += 64;
        return size;
    }
    
    // Compress decimal part
    size_t compressDecimalValue(int64_t decimal_value, int decimal_count) {
        if (decimal_count == 0) return size;
        
        out.append(decimal_count - 1, 2); // Save byte count: 00-1, 01-2, 10-3, 11-4
        size += 2;
        
        // Calculate m value
        int64_t thresh = threshold[decimal_count - 1];
        int m = static_cast<int>(decimal_value);
        size += 1;
        
        if (decimal_value - thresh >= 0) {
            // Flag: calculate m value
            out.push_back(true);
            m = static_cast<int>(decimal_value % thresh);
            
            // XOR operation for m
            int64_t xorVal = (Double_doubleToLongBits(static_cast<double>(decimal_value) / powers[decimal_count] + 1)) ^ 
                            Double_doubleToLongBits(static_cast<double>(m) / powers[decimal_count] + 1);
            
            // Save decimal_count bits of XOR
            out.append(static_cast<uint64_t>(xorVal) >> (52 - decimal_count), decimal_count);
            size += decimal_count;
        } else {
            out.push_back(false);
        }
        
        // Save m value based on decimal_count
        if (decimal_count == 1) {
            out.append(m, 3);
            size += 3;
        } else if (decimal_count == 2) {
            if (m < 8) {
                out.append(0, 1);
                out.append(m, 3);
                size += 4;
            } else {
                out.append(1, 1);
                out.append(m, 5);
                size += 6;
            }
        } else if (decimal_count == 3) {
            if (m < 2) {
                out.append(0, 2);
                out.append(m, 1);
                size += 3;
            } else if (m < 8) {
                out.append(1, 2);
                out.append(m, 3);
                size += 5;
            } else if (m < 32) {
                out.append(2, 2);
                out.append(m, 5);
                size += 7;
            } else {
                out.append(3, 2);
                out.append(m, mValueBits[decimal_count - 1]);
                size += 2 + mValueBits[decimal_count - 1];
            }
        } else {
            if (m < 16) {
                out.append(0, 2);
                out.append(m, 4);
                size += 6;
            } else if (m < 64) {
                out.append(1, 2);
                out.append(m, 6);
                size += 8;
            } else if (m < 256) {
                out.append(2, 2);
                out.append(m, 8);
                size += 10;
            } else {
                out.append(3, 2);
                out.append(m, mValueBits[decimal_count - 1]);
                size += 2 + mValueBits[decimal_count - 1];
            }
        }
        
        return size;
    }
    
    // Compress integer part
    size_t compressIntegerValue(int64_t int_value, int intSignal) {
        int64_t diff = int_value - storedVal;
        
        // Write sign bit to distinguish -0 and +0
        out.append(intSignal, 1);
        size += 1;
        
        if (diff >= -1 && diff <= 1) {
            out.append(diff + 1, 2); // Map -1 to 0, 0 to 1, 1 to 2
            size += 2;
        } else {
            out.append(3, 2); // 11
            size += 2;
            if (diff < 0) {
                out.push_back(false);
                diff = -diff;
            } else {
                out.push_back(true);
            }
            size += 1;

            // Extended size selector (2 bits):
            // 00: 3 bits (2..7)
            // 01: 16 bits
            // 10: 32 bits
            // 11: 64 bits
            if (diff >= 2 && diff < 8) {
                out.append(0, 2);
                out.append(static_cast<uint64_t>(diff), 3);
                size += 5;
            } else if (diff <= 0xFFFF) {
                out.append(1, 2);
                out.append(static_cast<uint64_t>(diff), 16);
                size += 18;
            } else if (diff <= 0xFFFFFFFFULL) {
                out.append(2, 2);
                out.append(static_cast<uint64_t>(diff), 32);
                size += 34;
            } else {
                out.append(3, 2);
                out.append(static_cast<uint64_t>(diff), 64);
                size += 66;
            }
        }
        
        storedVal = int_value;
        return size;
    }
    
    // Compress a value (after first)
    size_t compressValue(double value) {
        int intSignal = value < 0 ? 0 : 1;
        size = compressIntegerValue(static_cast<int64_t>(value), intSignal);
        
        int decimal_count;
        int64_t decimal_value;
        cal_decimal_count(value, decimal_count, decimal_value);
        size = compressDecimalValue(decimal_value, decimal_count);
        
        return size;
    }
    
    static int64_t Double_doubleToLongBits(double d) {
        return *reinterpret_cast<int64_t*>(&d);
    }
    
public:
    T storedValue = 0;
    
    explicit CompressorCamel(const T& value) {
        addValue(value);
    }
    
    void addValue(const T& value) {
        if (first) {
            int decimal_count;
            int64_t decimal_value;
            cal_decimal_count(value, decimal_count, decimal_value);
            
            double adjustedValue = (value < 0 ? -1 : 1) * 
                (static_cast<double>(static_cast<int64_t>(std::abs(value)) * powers[decimal_count] + decimal_value)) / 
                powers[decimal_count];
            
            writeFirst(Double_doubleToLongBits(adjustedValue));
        } else {
            compressValue(value);
        }
    }
    
    void close() {
        out.push_back(false);
        out.close();
    }
    
    size_t getSize() const { return size; }
    BitStream getBuffer() { return out; }
};
