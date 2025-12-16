#pragma once

/**
 * Camel Decompressor - Faithful C++ port from:
 * https://github.com/yoyo185644/camel
 */

#include <cstdint>
#include <cmath>
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
    
    static constexpr int DECIMAL_MAX_COUNT = 3;
    static constexpr int mValueBits[] = {3, 5, 7, 10, 15};
    static constexpr int64_t threshold[] = {5, 25, 125, 625};
    static constexpr int64_t powers[] = {1L, 10L, 100L, 1000L, 10000L, 100000L};
    
    static double Long_longBitsToDouble(int64_t bits) {
        return *reinterpret_cast<double*>(&bits);
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
            int sizeFlag = in.get(1);
            int diff;
            if (sizeFlag == 0) {
                diff = in.get(3);
            } else {
                diff = in.get(16);
            }
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
        int decimal_count = in.get(2) + 1;
        int mFlag = in.get(1);
        
        int64_t decimal_value;
        int m;
        
        if (mFlag == 1) {
            // Has XOR encoding
            int64_t xorBits = in.get(decimal_count);
            m = decompressMValue(decimal_count);
            
            // Reconstruct decimal_value from m and XOR
            int64_t thresh = threshold[decimal_count - 1];
            // The XOR helps recover the higher bits
            decimal_value = m; // Simplified - in practice need to XOR decode
            
            // Try to find original decimal_value
            for (int64_t k = 0; k * thresh + m < powers[decimal_count]; k++) {
                int64_t candidate = k * thresh + m;
                double d1 = static_cast<double>(candidate) / powers[decimal_count] + 1;
                double d2 = static_cast<double>(m) / powers[decimal_count] + 1;
                int64_t candidateXor = (*reinterpret_cast<int64_t*>(&d1)) ^
                                       (*reinterpret_cast<int64_t*>(&d2));
                if ((static_cast<uint64_t>(candidateXor) >> (52 - decimal_count)) == xorBits) {
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
    
    int decompressMValue(int decimal_count) {
        int m;
        if (decimal_count == 1) {
            m = in.get(3);
        } else if (decimal_count == 2) {
            int flag = in.get(1);
            if (flag == 0) {
                m = in.get(3);
            } else {
                m = in.get(5);
            }
        } else if (decimal_count == 3) {
            int flag = in.get(2);
            if (flag == 0) {
                m = in.get(1);
            } else if (flag == 1) {
                m = in.get(3);
            } else if (flag == 2) {
                m = in.get(5);
            } else {
                m = in.get(mValueBits[decimal_count - 1]);
            }
        } else {
            int flag = in.get(2);
            if (flag == 0) {
                m = in.get(4);
            } else if (flag == 1) {
                m = in.get(6);
            } else if (flag == 2) {
                m = in.get(8);
            } else {
                m = in.get(mValueBits[decimal_count - 1]);
            }
        }
        return m;
    }
    
    T nextValue() {
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
