#pragma once

/**
 * Elf (Erasing-based Lossless Floating-Point Compression) - VLDB 2023
 * 
 * Faithful C++ port of the original Java implementation from:
 * https://github.com/Spatio-Temporal-Lab/elf
 * 
 * This implementation follows the exact logic from:
 * - AbstractElfCompressor.java
 * - ElfXORCompressor.java
 * - Elf64Utils.java
 */

#include <cstdint>
#include <cmath>
#include <limits>
#include <vector>
#include "../lib/BitStream.hpp"

// Elf64Utils - ported from Java
class Elf64Utils {
public:
    // αlog_2(10) lookup table
    static constexpr int f[] = {0, 4, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40, 44, 47, 50, 54, 57, 60, 64, 67};
    
    static constexpr double map10iP[] = {
        1.0, 1.0E1, 1.0E2, 1.0E3, 1.0E4, 1.0E5, 1.0E6, 1.0E7,
        1.0E8, 1.0E9, 1.0E10, 1.0E11, 1.0E12, 1.0E13, 1.0E14,
        1.0E15, 1.0E16, 1.0E17, 1.0E18, 1.0E19, 1.0E20
    };
    
    static constexpr double map10iN[] = {
        1.0, 1.0E-1, 1.0E-2, 1.0E-3, 1.0E-4, 1.0E-5, 1.0E-6, 1.0E-7,
        1.0E-8, 1.0E-9, 1.0E-10, 1.0E-11, 1.0E-12, 1.0E-13, 1.0E-14,
        1.0E-15, 1.0E-16, 1.0E-17, 1.0E-18, 1.0E-19, 1.0E-20
    };
    
    static constexpr int64_t mapSPGreater1[] = {1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000};
    
    static constexpr double mapSPLess1[] = {
        1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001,
        0.000000001, 0.0000000001
    };
    
    static constexpr double LOG_2_10 = 3.321928094887362; // Math.log(10) / Math.log(2)
    
    static int getFAlpha(int alpha) {
        if (alpha < 0) {
            return 0; // Should not happen in valid use
        }
        if (alpha >= 21) {
            return static_cast<int>(std::ceil(alpha * LOG_2_10));
        }
        return f[alpha];
    }
    
    static double get10iP(int i) {
        if (i < 0) return 1.0;
        if (i >= 21) {
            return std::pow(10.0, i);
        }
        return map10iP[i];
    }
    
    static double get10iN(int i) {
        if (i < 0) return 1.0;
        if (i >= 21) {
            return std::pow(10.0, -i);
        }
        return map10iN[i];
    }
    
    static int getSP(double v) {
        if (v == 0) return -324; // Handle 0 safely
        if (v >= 1) {
            for (int i = 0; i < 9; i++) {
                if (v < mapSPGreater1[i + 1]) {
                    return i;
                }
            }
        } else {
            for (int i = 1; i < 11; i++) {
                if (v >= mapSPLess1[i]) {
                    return -i;
                }
            }
        }
        return static_cast<int>(std::floor(std::log10(v)));
    }
    
    // Returns [sp, is10iNFlag]
    static void getSPAnd10iNFlag(double v, int& sp, int& flag10iN) {
        flag10iN = 0;
        if (v >= 1) {
            for (int i = 0; i < 9; i++) {
                if (v < mapSPGreater1[i + 1]) {
                    sp = i;
                    return;
                }
            }
        } else {
            for (int i = 1; i < 11; i++) {
                if (v >= mapSPLess1[i]) {
                    sp = -i;
                    flag10iN = (v == mapSPLess1[i]) ? 1 : 0;
                    return;
                }
            }
        }
        double log10v = std::log10(v);
        sp = static_cast<int>(std::floor(log10v));
        flag10iN = (log10v == static_cast<int64_t>(log10v)) ? 1 : 0;
    }
    
    static int getSignificantCount(double v, int sp, int lastBetaStar) {
        int i;
        if (lastBetaStar != INT32_MAX && lastBetaStar != 0) {
            i = std::max(lastBetaStar - sp - 1, 1);
        } else if (lastBetaStar == INT32_MAX) {
            i = 17 - sp - 1;
        } else if (sp >= 0) {
            i = 1;
        } else {
            i = -sp;
        }
        
        double temp = v * get10iP(i);
        if (std::isinf(temp) || std::abs(temp) > 9.22e18) return 17;
        int64_t tempLong = static_cast<int64_t>(temp);
        while (tempLong != temp) {
            i++;
            if (i > 308) return 17; // Prevent infinite loop/overflow
            double scale = get10iP(i);
            if (std::isinf(scale)) return 17;
            
            temp = v * scale;
            if (std::abs(temp) > 9.22e18) return 17;
            tempLong = static_cast<int64_t>(temp);
        }
        
        // Bug fix check from original
        if (temp / get10iP(i) != v) {
            return 17;
        } else {
            while (i > 0 && tempLong % 10 == 0) {
                i--;
                tempLong = tempLong / 10;
            }
            return sp + i + 1;
        }
    }
    
    // Returns [alpha, betaStar]
    static void getAlphaAndBetaStar(double v, int lastBetaStar, int& alpha, int& betaStar) {
        if (v < 0) v = -v;
        int sp, flag10iN;
        getSPAnd10iNFlag(v, sp, flag10iN);
        int beta = getSignificantCount(v, sp, lastBetaStar);
        alpha = beta - sp - 1;
        betaStar = (flag10iN == 1) ? 0 : beta;
    }
    
    static double roundUp(double v, int alpha) {
        double scale = get10iP(alpha);
        if (v < 0) {
            return std::floor(v * scale) / scale;
        } else {
            return std::ceil(v * scale) / scale;
        }
    }
};

// ElfXORCompressor - ported from Java
template<typename T = double>
class ElfXORCompressor {
    static_assert(std::is_same_v<T, double>, "ElfXORCompressor only supports double");
    
    int storedLeadingZeros = INT32_MAX;
    int storedTrailingZeros = INT32_MAX;
    int64_t storedVal = 0;
    bool first = true;
    size_t size = 0;
    
    static constexpr int64_t END_SIGN = 0x7ff8000000000000LL; // Double.NaN bits
    
public:
    static constexpr int16_t leadingRepresentation[] = {
        0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 2, 2, 2, 2,
        3, 3, 4, 4, 5, 5, 6, 6,
        7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7
    };
    
    static constexpr int16_t leadingRound[] = {
        0, 0, 0, 0, 0, 0, 0, 0,
        8, 8, 8, 8, 12, 12, 12, 12,
        16, 16, 18, 18, 20, 20, 22, 22,
        24, 24, 24, 24, 24, 24, 24, 24,
        24, 24, 24, 24, 24, 24, 24, 24,
        24, 24, 24, 24, 24, 24, 24, 24,
        24, 24, 24, 24, 24, 24, 24, 24,
        24, 24, 24, 24, 24, 24, 24, 24
    };
    
    BitStream out{};
    
    size_t addValue(int64_t value) {
        if (first) {
            return writeFirst(value);
        } else {
            return compressValue(value);
        }
    }
    
    size_t writeFirst(int64_t value) {
        first = false;
        storedVal = value;
        int trailingZeros = 64;
        if (value != 0) {
            trailingZeros = __builtin_ctzll(value);
        }
        out.append(trailingZeros, 7);
        if (trailingZeros < 64) {
            if (63 - trailingZeros > 0) {
                out.append(static_cast<uint64_t>(storedVal) >> (trailingZeros + 1), 63 - trailingZeros);
            }
            size += 70 - trailingZeros;
            return 70 - trailingZeros;
        } else {
            size += 7;
            return 7;
        }
    }
    
    size_t compressValue(int64_t value) {
        size_t thisSize = 0;
        int64_t xorVal = storedVal ^ value;
        
        if (xorVal == 0) {
            // case 01
            out.append(1, 2);
            size += 2;
            thisSize += 2;
        } else {
            int leadingZeros = leadingRound[__builtin_clzll(xorVal)];
            int trailingZeros = __builtin_ctzll(xorVal);
            
            if (leadingZeros == storedLeadingZeros && trailingZeros >= storedTrailingZeros) {
                // case 00
                int centerBits = 64 - storedLeadingZeros - storedTrailingZeros;
                int len = 2 + centerBits;
                if (len > 64) {
                    out.append(0, 2);
                    out.append(static_cast<uint64_t>(xorVal) >> storedTrailingZeros, centerBits);
                } else {
                    out.append(static_cast<uint64_t>(xorVal) >> storedTrailingZeros, len);
                }
                size += len;
                thisSize += len;
            } else {
                storedLeadingZeros = leadingZeros;
                storedTrailingZeros = trailingZeros;
                int centerBits = 64 - storedLeadingZeros - storedTrailingZeros;
                
                if (centerBits <= 16) {
                    // case 10
                    int header = (((0x2 << 3) | leadingRepresentation[storedLeadingZeros]) << 4) | (centerBits & 0xf);
                    out.append(header, 9);
                    if (centerBits > 1) {
                        out.append(static_cast<uint64_t>(xorVal) >> (storedTrailingZeros + 1), centerBits - 1);
                    }
                    size += 8 + centerBits;
                    thisSize += 8 + centerBits;
                } else {
                    // case 11
                    int header = (((0x3 << 3) | leadingRepresentation[storedLeadingZeros]) << 6) | (centerBits & 0x3f);
                    out.append(header, 11);
                    if (centerBits > 1) {
                        out.append(static_cast<uint64_t>(xorVal) >> (storedTrailingZeros + 1), centerBits - 1);
                    }
                    size += 10 + centerBits;
                    thisSize += 10 + centerBits;
                }
            }
            storedVal = value;
        }
        return thisSize;
    }
    
    void close() {
        addValue(END_SIGN);
        out.push_back(false);
        out.close();
    }
    
    size_t getSize() const { return size; }
    BitStream& getOutputStream() { return out; }
    BitStream getBuffer() { return out; }
};

// Main Elf Compressor - ported from AbstractElfCompressor + ElfCompressor
template<typename T = double>
class CompressorElf {
    static_assert(std::is_same_v<T, double>, "CompressorElf only supports double");
    
    size_t size = 0;
    int lastBetaStar = INT32_MAX;
    ElfXORCompressor<T> xorCompressor;
    
public:
    T storedValue = 0;
    
    explicit CompressorElf(const T& value) {
        addValue(value);
    }
    
    void addValue(const T& v) {
        int64_t vLong = *reinterpret_cast<const int64_t*>(&v);
        int64_t vPrimeLong;
        
        if (v == 0.0 || std::isinf(v)) {
            size += writeInt(2, 2); // case 10
            vPrimeLong = vLong;
        } else if (std::isnan(v)) {
            size += writeInt(2, 2); // case 10
            vPrimeLong = 0x7ff8000000000000LL;
        } else {
            // C1: v is a normal or subnormal
            int alpha, betaStar;
            Elf64Utils::getAlphaAndBetaStar(v, lastBetaStar, alpha, betaStar);
            int e = ((vLong >> 52) & 0x7ff);
            int gAlpha = Elf64Utils::getFAlpha(alpha) + e - 1023;
            int eraseBits = 52 - gAlpha;
            int64_t mask;
            if (eraseBits < 0) {
                mask = 0xffffffffffffffffLL;
            } else if (eraseBits >= 64) {
                mask = 0;
            } else {
                mask = 0xffffffffffffffffLL << eraseBits;
            }
            int64_t delta = (~mask) & vLong;
            
            if (delta != 0 && eraseBits > 4) { // C2
                if (betaStar == lastBetaStar) {
                    size += writeBit(false); // case 0
                } else {
                    size += writeInt(betaStar | 0x30, 6); // case 11, 2 + 4 = 6
                    lastBetaStar = betaStar;
                }
                vPrimeLong = mask & vLong;
            } else {
                size += writeInt(2, 2); // case 10
                vPrimeLong = vLong;
            }
        }
        size += xorCompressor.addValue(vPrimeLong);
    }
    
    size_t writeInt(int n, int len) {
        xorCompressor.getOutputStream().append(n, len);
        return len;
    }
    
    size_t writeBit(bool bit) {
        xorCompressor.getOutputStream().push_back(bit);
        return 1;
    }
    
    void close() {
        writeInt(2, 2); // end marker
        xorCompressor.close();
    }
    
    size_t getSize() const { return size; }
    BitStream getBuffer() { return xorCompressor.getBuffer(); }
};
