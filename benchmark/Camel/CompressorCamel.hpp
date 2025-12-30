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
#include <algorithm>
#include "../lib/BitStream.hpp"

template<typename T = double>
class CompressorCamel {
    static_assert(std::is_same_v<T, double>, "CompressorCamel only supports double");
    
    int64_t storedVal = 0;  // Stores integer part of previous value
    bool first = true;
    size_t size = 0;
    
    static constexpr int DECIMAL_MAX_COUNT = 18;
    
    // m value bits for different decimal counts (calculated as ceil(log2(5^k)))
    static constexpr int mValueBits[] = {
        3, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28, 31, 33, 35, 38, 40, 42
    };
    
    // Thresholds for decimal encoding (5^k)
    static constexpr int64_t threshold[] = {
        5, 25, 125, 625, 3125, 15625, 78125, 390625, 1953125, 9765625, 
        48828125, 244140625, 1220703125, 6103515625, 30517578125, 
        152587890625, 762939453125, 3814697265625
    };
    
    // Powers of 10
    static constexpr int64_t powers[] = {
        1L, 10L, 100L, 1000L, 10000L, 100000L, 1000000L, 10000000L, 100000000L, 
        1000000000L, 10000000000L, 100000000000L, 1000000000000L, 
        10000000000000L, 100000000000000L, 1000000000000000L, 
        10000000000000000L, 100000000000000000L, 1000000000000000000L
    };
    
    BitStream out{};
    
    static inline int64_t doubleToBits(double d) {
        return *reinterpret_cast<int64_t*>(&d);
    }

    static inline double bitsToDouble(int64_t b) {
        return *reinterpret_cast<double*>(&b);
    }

    inline bool fitsInInt64(double d) const {
        if (!std::isfinite(d)) return false;
        return d >= static_cast<double>(std::numeric_limits<int64_t>::min()) &&
               d <= static_cast<double>(std::numeric_limits<int64_t>::max());
    }

    // Calculate decimal count and decimal value
    void cal_decimal_count(double value, int& decimal_count, int64_t& decimal_value) {
        double factor = 1;
        decimal_count = 0;
        value = std::abs(value);
        double epsilon = 0.0000001;
        
        // Find minimal 10^k scale that makes value an integer (up to MAX)
        // We use a tighter check for higher precision support
        while (decimal_count < max_precision) {
            double scaled = value * factor;
            if (std::abs(scaled - std::round(scaled)) < epsilon) {
                // Double check if we can stop earlier? 
                // Actually the loop condition assumes we continue UNTIL it's integer.
                // But epsilon check is tricky for large numbers.
                break;
            }
            factor *= 10.0;
            decimal_count++;
        }
        
        // If we reached max and it's still not integer, we just take the max.
        
        decimal_value = 0;
        if (decimal_count == 0) {
            decimal_count = 1;
        }
        
        if (decimal_count > 0 && decimal_count <= max_precision) {
            // Safe modulo arithmetic
            double scaled = std::round(value * powers[decimal_count]);
            // decimal_value is the fractional part scaled: (scaled % powers[decimal_count])
            // Using fmod for large numbers
            double rem = std::fmod(scaled, static_cast<double>(powers[decimal_count]));
            decimal_value = static_cast<int64_t>(rem);
        } else {
            decimal_value = static_cast<int64_t>(std::round(value * powers[max_precision])) % powers[max_precision];
            decimal_count = max_precision;
        }
    }
    
    // Write first value
    size_t writeFirst(int64_t value) {
        first = false;
        double d = bitsToDouble(value);
        storedVal = fitsInInt64(d) ? static_cast<int64_t>(d) : 0;
        out.append(value, 64);
        size += 64;
        return size;
    }
    
    // Compress decimal part
    size_t compressDecimalValue(int64_t decimal_value, int decimal_count) {
        if (decimal_count == 0) return size;
        
        // Extended header for decimal count
        // Original: 2 bits (0..3). We need up to 18.
        // We will use a prefix code extension or just fixed bits?
        // To be compatible with "Camel" spirit, we should keep small counts small.
        // But since we broke compatibility with the original Java (by extending counts),
        // we can implement a variable length scheme.
        // 00 -> 1 (original 0)
        // 01 -> 2 (original 1)
        // 10 -> 3 (original 2)
        // 11 -> escape/extended
        
        if (decimal_count <= 3) {
            out.append(decimal_count - 1, 2); 
            size += 2;
        } else {
            out.append(3, 2); // 11
            size += 2;
            // Write remaining value (decimal_count - 4) in 4 bits (supports up to 4+15=19)
            out.append(decimal_count - 4, 4);
            size += 4;
        }
        
        // Calculate m value
        int64_t thresh = threshold[decimal_count - 1];
        int64_t m = decimal_value;
        size += 1;
        
        if (decimal_value - thresh >= 0) {
            // Flag: calculate m value
            out.push_back(true);
            m = decimal_value % thresh;
            
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
            if (m < 8) { out.append(0, 1); out.append(m, 3); size += 4; } 
            else { out.append(1, 1); out.append(m, 5); size += 6; }
        } else if (decimal_count == 3) {
            if (m < 2) { out.append(0, 2); out.append(m, 1); size += 3; }
            else if (m < 8) { out.append(1, 2); out.append(m, 3); size += 5; }
            else if (m < 32) { out.append(2, 2); out.append(m, 5); size += 7; }
            else { out.append(3, 2); out.append(m, mValueBits[decimal_count - 1]); size += 2 + mValueBits[decimal_count - 1]; }
        } else {
            // Generic fallback for higher counts
            if (m < 16) { out.append(0, 2); out.append(m, 4); size += 6; }
            else if (m < 64) { out.append(1, 2); out.append(m, 6); size += 8; }
            else if (m < 256) { out.append(2, 2); out.append(m, 8); size += 10; }
            else { 
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

        out.append(intSignal, 1);
        size += 1;
        
        if (diff >= -1 && diff <= 1) {
            out.append(diff + 1, 2); 
            size += 2;
        } else {
            out.append(3, 2); 
            size += 2;
            if (diff < 0) {
                out.push_back(false);
                diff = -diff;
            } else {
                out.push_back(true);
            }
            size += 1;

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
        // 1. Compute model representation
        int decimal_count;
        int64_t decimal_value;
        cal_decimal_count(value, decimal_count, decimal_value);
        
        // Correctly handle values close to the next integer (e.g. 17.999...)
        // where cal_decimal_count determines they are integers (count=0/1, val=0)
        // but simple casting would truncate (e.g. to 17).
        int64_t intPart;
        double abs_val = std::abs(value);
        
        // Use the same epsilon as in cal_decimal_count
        if (std::abs(abs_val - std::round(abs_val)) < 0.0000001) {
             // It's effectively an integer, so round to nearest
             intPart = static_cast<int64_t>(std::round(value));
        } else {
             intPart = static_cast<int64_t>(value);
        }

        // Normal Camel path: integer+decimal encoding.
        int intSignal = std::signbit(value) ? 0 : 1;
        size = compressIntegerValue(intPart, intSignal);
        size = compressDecimalValue(decimal_value, decimal_count);
        
        return size;
    }
    
    static int64_t Double_doubleToLongBits(double d) {
        return *reinterpret_cast<int64_t*>(&d);
    }
    
public:
    T storedValue = 0;
    int max_precision = DECIMAL_MAX_COUNT;
    
    explicit CompressorCamel(const T& value, int max_prec = DECIMAL_MAX_COUNT) {
        if (max_prec >= 0 && max_prec < DECIMAL_MAX_COUNT) {
            max_precision = max_prec;
        }
        addValue(value);
    }
    
    void addValue(const T& value) {
        if (first) {
            writeFirst(Double_doubleToLongBits(value));
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
