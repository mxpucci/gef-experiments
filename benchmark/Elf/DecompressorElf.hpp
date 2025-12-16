#pragma once

/**
 * Elf Decompressor - Faithful C++ port from:
 * https://github.com/Spatio-Temporal-Lab/elf
 * 
 * Ported from:
 * - AbstractElfDecompressor.java
 * - ElfXORDecompressor.java
 */

#include <cstdint>
#include <cmath>
#include <limits>
#include "../lib/BitStream.hpp"
#include "CompressorElf.hpp"  // For Elf64Utils

// ElfXORDecompressor - ported from Java
template<typename T = double>
class ElfXORDecompressor {
    static_assert(std::is_same_v<T, double>, "ElfXORDecompressor only supports double");
    
    int64_t storedVal = 0;
    int storedLeadingZeros = INT32_MAX;
    int storedTrailingZeros = INT32_MAX;
    bool first = true;
    bool endOfStream = false;
    
    BitStream in;
    
    static constexpr int64_t END_SIGN = 0x7ff8000000000000LL; // Double.NaN bits
    static constexpr int16_t leadingRepresentation[] = {0, 8, 12, 16, 18, 20, 22, 24};
    
public:
    ElfXORDecompressor(const BitStream& bs) : in(bs) {}
    
    BitStream& getInputStream() { return in; }
    
    T readValue() {
        next();
        if (endOfStream) {
            return std::numeric_limits<T>::quiet_NaN();
        }
        return *reinterpret_cast<T*>(&storedVal);
    }
    
    bool isEndOfStream() const { return endOfStream; }
    
private:
    void next() {
        if (first) {
            first = false;
            int trailingZeros = in.get(7);
            if (trailingZeros < 64) {
                storedVal = ((in.get(63 - trailingZeros) << 1) + 1) << trailingZeros;
            } else {
                storedVal = 0;
            }
            if (storedVal == END_SIGN) {
                endOfStream = true;
            }
        } else {
            nextValue();
        }
    }
    
    void nextValue() {
        int64_t value;
        int centerBits, leadAndCenter;
        int flag = in.get(2);
        
        switch (flag) {
            case 3: {
                // case 11
                leadAndCenter = in.get(9);
                storedLeadingZeros = leadingRepresentation[leadAndCenter >> 6];
                centerBits = leadAndCenter & 0x3f;
                if (centerBits == 0) {
                    centerBits = 64;
                }
                storedTrailingZeros = 64 - storedLeadingZeros - centerBits;
                value = ((in.get(centerBits - 1) << 1) + 1) << storedTrailingZeros;
                value = storedVal ^ value;
                if (value == END_SIGN) {
                    endOfStream = true;
                } else {
                    storedVal = value;
                }
                break;
            }
            case 2: {
                // case 10
                leadAndCenter = in.get(7);
                storedLeadingZeros = leadingRepresentation[leadAndCenter >> 4];
                centerBits = leadAndCenter & 0xf;
                if (centerBits == 0) {
                    centerBits = 16;
                }
                storedTrailingZeros = 64 - storedLeadingZeros - centerBits;
                value = ((in.get(centerBits - 1) << 1) + 1) << storedTrailingZeros;
                value = storedVal ^ value;
                if (value == END_SIGN) {
                    endOfStream = true;
                } else {
                    storedVal = value;
                }
                break;
            }
            case 1:
                // case 01, same value as before
                break;
            default: {
                // case 00
                centerBits = 64 - storedLeadingZeros - storedTrailingZeros;
                value = in.get(centerBits) << storedTrailingZeros;
                value = storedVal ^ value;
                if (value == END_SIGN) {
                    endOfStream = true;
                } else {
                    storedVal = value;
                }
                break;
            }
        }
    }
};

// Main Elf Decompressor - ported from AbstractElfDecompressor + ElfDecompressor
template<typename T = double>
class DecompressorElf {
    static_assert(std::is_same_v<T, double>, "DecompressorElf only supports double");
    
    int lastBetaStar = INT32_MAX;
    ElfXORDecompressor<T> xorDecompressor;
    bool endOfStream = false;
    size_t n;
    size_t i = 0;
    
public:
    T storedValue = 0;
    
    DecompressorElf(const BitStream& bs, size_t nlines) 
        : xorDecompressor(bs), n(nlines) {
        // Read first value
        storedValue = nextValue();
        ++i;
        if (i > n || std::isnan(storedValue)) {
            endOfStream = true;
        }
    }
    
    bool hasNext() {
        if (!endOfStream) {
            T val = nextValue();
            if (std::isnan(val) || xorDecompressor.isEndOfStream()) {
                endOfStream = true;
                return false;
            }
            storedValue = val;
            ++i;
            if (i > n) {
                endOfStream = true;
            }
        }
        return !endOfStream;
    }
    
private:
    T nextValue() {
        T v;
        int firstBit = xorDecompressor.getInputStream().get(1);
        
        if (firstBit == 0) {
            v = recoverVByBetaStar(); // case 0
        } else {
            int secondBit = xorDecompressor.getInputStream().get(1);
            if (secondBit == 0) {
                v = xorDecompressor.readValue(); // case 10
            } else {
                lastBetaStar = xorDecompressor.getInputStream().get(4); // case 11
                v = recoverVByBetaStar();
            }
        }
        return v;
    }
    
    T recoverVByBetaStar() {
        T v;
        T vPrime = xorDecompressor.readValue();
        if (std::isnan(vPrime)) {
            return vPrime;
        }
        int sp = Elf64Utils::getSP(std::abs(vPrime));
        if (lastBetaStar == 0) {
            v = Elf64Utils::get10iN(-sp - 1);
            if (vPrime < 0) {
                v = -v;
            }
        } else {
            int alpha = lastBetaStar - sp - 1;
            v = Elf64Utils::roundUp(vPrime, alpha);
        }
        return v;
    }
};
