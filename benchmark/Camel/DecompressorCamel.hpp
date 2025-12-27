#pragma once

/**
 * Camel Decompressor - Faithful C++ port from:
 * https://github.com/yoyo185644/camel
 */

#include <cstdint>
#include <cmath>
#include <limits>
#include <algorithm>
#include "../lib/BitStream.hpp"

template<typename T = double>
class DecompressorCamel {
    static_assert(std::is_same_v<T, double>, "DecompressorCamel only supports double");
    
    BitStream in;
    int64_t storedIntVal = 0;
    bool first = true;
    bool endOfStream = false;
    size_t n;
    size_t i = 0;
    
    static constexpr int DECIMAL_MAX_COUNT = 18;
    
    // m value bits for different decimal counts
    static constexpr int mValueBits[] = {
        3, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28, 31, 33, 35, 38, 40, 42
    };
    
    // Thresholds
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
    
    static double Long_longBitsToDouble(int64_t bits) {
        return *reinterpret_cast<double*>(&bits);
    }

    static bool fitsInInt64(double d) {
        if (!std::isfinite(d)) return false;
        return d >= static_cast<double>(std::numeric_limits<int64_t>::min()) &&
               d <= static_cast<double>(std::numeric_limits<int64_t>::max());
    }
    
    T readFirst() {
        first = false;
        int64_t value = in.get(64);
        double d = Long_longBitsToDouble(value);
        storedIntVal = static_cast<int64_t>(d);
        return static_cast<T>(d);
    }
    
    // Decompress integer part
    int64_t decompressIntegerValue(int& intSignal) {
        intSignal = in.get(1);
        int diffCode = in.get(2);
        
        int64_t int_value;
        if (diffCode <= 2) {
            int diff = diffCode - 1; // 0->-1, 1->0, 2->1
            int_value = storedIntVal + diff;
        } else {
            int sign = in.get(1);
            int sizeSel = in.get(2); // must match CompressorCamel extended selector
            uint64_t diff_u = 0;
            if (sizeSel == 0) {
                diff_u = in.get(3);
            } else if (sizeSel == 1) {
                diff_u = in.get(16);
            } else if (sizeSel == 2) {
                diff_u = in.get(32);
            } else {
                diff_u = in.get(64);
            }
            int64_t diff = static_cast<int64_t>(diff_u);
            if (sign == 0) {
                diff = -diff;
            }
            int_value = storedIntVal + diff;
        }
        
        storedIntVal = int_value;
        return int_value;
    }
    
    // Decompress decimal part
    double decompressDecimalValue(int64_t int_value, int intSignal) {
        // Read decimal count
        int decimal_count;
        int prefix = in.get(2);
        if (prefix < 3) {
            decimal_count = prefix + 1;
        } else {
            // Extended mode (prefix 11)
            int extra = in.get(4);
            decimal_count = extra + 4;
        }
        
        // This should match compressDecimalValue logic
        int mFlag = in.get(1);
        
        int64_t decimal_value;
        int64_t m;
        
        if (mFlag == 1) {
            // Has XOR encoding
            // The "decimal_count bits of XOR" logic
            // Note: The original code does not store "m", it calculates m from decimal_value % thresh.
            // But here we need to read 'm' first? 
            // In compressDecimalValue:
            //   if (decimal_value - thresh >= 0) {
            //       out.push_back(true);
            //       m = decimal_value % thresh;
            //       // XOR stuff...
            //       out.append(xor, decimal_count);
            //   }
            //   
            //   // Save m value...
            
            // So we read XOR bits first, then we read m.
            int64_t xorBits = in.get(decimal_count);
            m = decompressMValue(decimal_count);
            
            int64_t thresh = threshold[decimal_count - 1];
            
            // Reconstruct decimal_value from m and XOR.
            // decimal_value = k * thresh + m
            // We need to find k such that the XOR condition matches.
            // This is a brute-force search in the original Java code too?
            // "Try to find original decimal_value" loop in my previous read suggests so.
            // Since k is small (powers[decimal_count] / thresh is roughly 2^decimal_count),
            // the loop runs 2^N times. For N=18 this is large (2^18 = 262k).
            // This is potentially slow for decompression.
            // But let's stick to the ported logic.
            
            decimal_value = m; // Default if not found
            for (int64_t k = 0; ; k++) {
                int64_t candidate = k * thresh + m;
                if (candidate >= powers[decimal_count]) break;
                
                double d1 = static_cast<double>(candidate) / powers[decimal_count] + 1;
                double d2 = static_cast<double>(m) / powers[decimal_count] + 1;
                int64_t candidateXor = (*reinterpret_cast<int64_t*>(&d1)) ^
                                       (*reinterpret_cast<int64_t*>(&d2));
                
                if ((static_cast<uint64_t>(candidateXor) >> (52 - decimal_count)) == static_cast<uint64_t>(xorBits)) {
                    decimal_value = candidate;
                    break;
                }
            }
        } else {
            m = decompressMValue(decimal_count);
            decimal_value = m;
        }
        
        double result = static_cast<double>(std::abs(int_value)) + 
                       static_cast<double>(decimal_value) / powers[decimal_count];
        
        if (intSignal == 0) {
            result = -result;
        }
        
        return result;
    }
    
    int64_t decompressMValue(int decimal_count) {
        int64_t m = 0;
        if (decimal_count == 1) {
            m = in.get(3);
        } else if (decimal_count == 2) {
            int flag = in.get(1);
            if (flag == 0) m = in.get(3);
            else m = in.get(5);
        } else if (decimal_count == 3) {
            int flag = in.get(2);
            if (flag == 0) m = in.get(1);
            else if (flag == 1) m = in.get(3);
            else if (flag == 2) m = in.get(5);
            else m = in.get(mValueBits[decimal_count - 1]);
        } else {
            // Generic logic for count > 3
            int flag = in.get(2);
            if (flag == 0) m = in.get(4);
            else if (flag == 1) m = in.get(6);
            else if (flag == 2) m = in.get(8);
            else m = in.get(mValueBits[decimal_count - 1]);
        }
        return m;
    }
    
    T nextValue() {
        // Removed rawFlag reading since we no longer support raw value escape and always encode.
        // int rawFlag = in.get(1);
        // if (rawFlag == 1) { ... }
        
        int intSignal;
        int64_t int_value = decompressIntegerValue(intSignal);
        return static_cast<T>(decompressDecimalValue(int_value, intSignal));
    }
    
public:
    T storedValue = 0;
    
    DecompressorCamel(const BitStream& bs, size_t nlines) : in(bs), n(nlines) {
        storedValue = readFirst();
        ++i;
        if (i > n) {
            endOfStream = true;
        }
    }
    
    bool hasNext() {
        if (!endOfStream) {
            storedValue = nextValue();
            ++i;
            if (i > n) {
                endOfStream = true;
            }
        }
        return !endOfStream;
    }
};
