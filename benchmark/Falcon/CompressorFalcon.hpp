#pragma once

/**
 * Falcon Compressor - Header-only C++ port from:
 * https://github.com/Spatio-Temporal-Lab/Falcon
 * 
 * This is a faithful port of the original CPU implementation from:
 * - src/cpu/Falcon_basic_compressor.cpp
 * - src/utils/output_bit_stream.cc
 * 
 * Key features (from original):
 * - Block-based compression (1025 elements per block)
 * - Decimal place calculation for integer conversion
 * - Delta encoding with zigzag encoding
 * - Bit-plane transpose with sparse/dense encoding
 */

#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <memory>

// Portable byte swap for big-endian conversion
#if defined(__APPLE__)
#include <libkern/OSByteOrder.h>
#define htobe32_portable(x) OSSwapHostToBigInt32(x)
#define be32toh_portable(x) OSSwapBigToHostInt32(x)
#elif defined(__linux__)
#include <endian.h>
#define htobe32_portable(x) htobe32(x)
#define be32toh_portable(x) be32toh(x)
#else
// Fallback for other platforms
inline uint32_t htobe32_portable(uint32_t x) {
    return ((x & 0xFF) << 24) | ((x & 0xFF00) << 8) | ((x >> 8) & 0xFF00) | ((x >> 24) & 0xFF);
}
inline uint32_t be32toh_portable(uint32_t x) {
    return htobe32_portable(x);
}
#endif

namespace falcon {

// Array class from original Falcon (src/utils/array.h)
template<typename T>
class Array {
public:
    Array() = default;
    
    explicit Array(int length) : length_(length) {
        data_ = std::make_unique<T[]>(length_);
    }
    
    Array(const Array& other) : length_(other.length_) {
        data_ = std::make_unique<T[]>(length_);
        std::copy(other.begin(), other.end(), begin());
    }
    
    Array& operator=(const Array& right) {
        length_ = right.length_;
        data_ = std::make_unique<T[]>(right.length_);
        std::copy(right.begin(), right.end(), begin());
        return *this;
    }
    
    Array(Array&& other) noexcept = default;
    Array& operator=(Array&& other) noexcept = default;
    
    T& operator[](int index) const { return data_[index]; }
    T* begin() const { return data_.get(); }
    T* end() const { return data_.get() + length_; }
    int length() const { return length_; }
    
private:
    int length_ = 0;
    std::unique_ptr<T[]> data_ = nullptr;
};

// OutputBitStream from original Falcon (src/utils/output_bit_stream.cc)
class OutputBitStream {
public:
    explicit OutputBitStream(uint32_t buffer_size) {
        data_ = Array<uint32_t>(buffer_size / 4 + 1);
        buffer_ = 0;
        cursor_ = 0;
        bit_in_buffer_ = 0;
    }
    
    uint32_t Write(uint64_t content, uint32_t len) {
        if (len > 64) {
            std::cerr << "Error: Attempt to write more than 64 bits." << std::endl;
            return 0;
        }
        
        if (cursor_ >= static_cast<uint32_t>(data_.length())) {
            int newsize = cursor_ + (len + 7) / 8;
            if (newsize > data_.length()) {
                Array<uint32_t> newData(newsize);
                std::copy(data_.begin(), data_.end(), newData.begin());
                data_ = std::move(newData);
            }
        }
        
        content <<= (64 - len);
        buffer_ |= (content >> bit_in_buffer_);
        bit_in_buffer_ += len;
        
        if (bit_in_buffer_ >= 32) {
            data_[cursor_++] = (buffer_ >> 32);
            buffer_ <<= 32;
            bit_in_buffer_ -= 32;
        }
        return len;
    }
    
    uint32_t WriteLong(uint64_t content, uint64_t len) {
        if (len == 0) return 0;
        if (len > 32) {
            Write(content >> (len - 32), 32);
            Write(content, len - 32);
            return len;
        }
        return Write(content, len);
    }
    
    uint32_t WriteInt(uint32_t content, uint32_t len) {
        return Write(static_cast<uint64_t>(content), len);
    }
    
    uint32_t WriteBit(bool bit) {
        return Write(static_cast<uint64_t>(bit), 1);
    }
    
    uint32_t WriteByte(uint8_t byte) {
        return Write(static_cast<uint64_t>(byte), 8);
    }
    
    Array<uint8_t> GetBuffer(uint32_t len) {
        Array<uint8_t> ret(len);
        for (auto& blk : data_) blk = htobe32_portable(blk);
        std::memcpy(ret.begin(), data_.begin(), len);
        return ret;
    }
    
    void Flush() {
        if (bit_in_buffer_) {
            data_[cursor_++] = buffer_ >> 32;
            buffer_ = 0;
            bit_in_buffer_ = 0;
        }
    }
    
    void Refresh() {
        cursor_ = 0;
        bit_in_buffer_ = 0;
        buffer_ = 0;
    }
    
    uint32_t GetBufferSize() const { return cursor_; }
    
private:
    Array<uint32_t> data_;
    uint32_t cursor_;
    uint32_t bit_in_buffer_;
    uint64_t buffer_;
};

} // namespace falcon

// Main Falcon Compressor - ported from Falcon_basic_compressor.cpp
template<typename T = double>
class CompressorFalcon {
    static_assert(std::is_same_v<T, double>, "CompressorFalcon only supports double");
    
public:
    static constexpr size_t BLOCK_SIZE = 1025;
    
private:
    static constexpr double POW_NUM = (1L << 51) + (1L << 52);
    
    static constexpr double pow10_table[17] = {
        1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0,
        10000000.0, 100000000.0, 1000000000.0, 10000000000.0,
        100000000000.0, 1000000000000.0, 10000000000000.0,
        100000000000000.0, 1000000000000000.0, 10000000000000000.0
    };
    
    std::vector<uint8_t> output;
    std::vector<T> buffer;
    size_t totalValues = 0;
    
    // From original: zigzag_encode
    static uint64_t zigzag_encode(int64_t value) {
        return (value << 1) ^ (value >> (sizeof(int64_t) * 8 - 1));
    }
    
    // From original: encodeDoubleWithSignLast
    static int64_t encodeDoubleWithSignLast(double x) {
        union { double d; int64_t u; } val;
        val.d = x;
        return (val.u << 1) ^ (val.u >> (sizeof(int64_t) * 8 - 1));
    }
    
    // From original: getDecimalPlaces
    static int getDecimalPlaces(double value, int sp) {
        double trac = value + POW_NUM - POW_NUM;
        double temp = value;
        
        int digits = 0;
        double td = 1;
        double deltaBound = std::abs(value) * std::pow(2.0, -52);
        
        while (std::abs(temp - trac) >= deltaBound * td && digits < 16 - sp - 1) {
            digits++;
            if (digits < 17) {
                td = pow10_table[digits];
            } else {
                td = std::pow(10.0, digits);
            }
            temp = value * td;
            if (std::isinf(temp)) { // Check for overflow
                digits = 16 - sp; // Force fallback
                break;
            }
            trac = temp + POW_NUM - POW_NUM;
        }
        
        if (std::round(temp) / td != value) {
            digits = 23;
        }
        return digits;
    }
    
    // From original: sampleBlock
    void sampleBlock(const std::vector<T>& block, std::vector<int64_t>& longs,
                     int64_t& firstValue, int& maxDecimalPlaces, int& isOk, int& maxBeta) {
        int maxSp = -99;
        maxDecimalPlaces = 0;
        
        for (const T& val : block) {
            if (val == 0) {
                maxSp = std::max(maxSp, 0); // Treat 0 as having sp 0
                continue;
            }
            double log10v = std::log10(std::abs(val));
            int sp = static_cast<int>(std::floor(log10v));
            maxSp = std::max(maxSp, sp);
            
            int decimalPlaces = getDecimalPlaces(val, sp);
            maxDecimalPlaces = std::max(maxDecimalPlaces, decimalPlaces);
        }
        
        maxBeta = maxSp + maxDecimalPlaces + 1;
        
        if (maxBeta > 15 || maxDecimalPlaces > 15) {
            isOk = 0;
            firstValue = encodeDoubleWithSignLast(block[0]);
            for (const T& val : block) {
                longs.push_back(encodeDoubleWithSignLast(val));
            }
        } else {
            isOk = 1;
            double multiplier = (maxDecimalPlaces < 17) ? pow10_table[maxDecimalPlaces] : std::pow(10.0, maxDecimalPlaces);
            firstValue = static_cast<int64_t>(std::round(block[0] * multiplier));
            for (const T& val : block) {
                longs.push_back(static_cast<int64_t>(std::round(val * multiplier)));
            }
        }
    }
    
    // From original: compressBlock
    void compressBlock(const std::vector<T>& block, falcon::OutputBitStream& bitStream, int& bitSize) {
        std::vector<int64_t> longs;
        int64_t firstValue;
        int maxDecimalPlaces = 0;
        int maxBeta = 0;
        int isOk;
        
        sampleBlock(block, longs, firstValue, maxDecimalPlaces, isOk, maxBeta);
        size_t currentBlockSize = block.size();
        
        // Delta encoding with zigzag
        std::vector<uint64_t> deltas(currentBlockSize - 1);
        int64_t prevQuant = firstValue;
        uint64_t maxDelta = 0;
        
        for (size_t i = 0; i < currentBlockSize - 1; i++) {
            int64_t currQuant = longs[i + 1];
            int64_t deltaValue = currQuant - prevQuant;
            deltas[i] = zigzag_encode(deltaValue);
            maxDelta = std::max(maxDelta, deltas[i]);
            prevQuant = currQuant;
        }
        
        // Calculate bitCount
        int bitCount = (maxDelta > 0) ? (64 - __builtin_clzll(maxDelta)) : 1;
        bitCount = std::min(bitCount, 64);
        
        // Bit transpose
        int numByte = (currentBlockSize - 1 + 7) / 8;
        std::vector<std::vector<uint8_t>> result_matrix(bitCount, std::vector<uint8_t>(numByte, 0));
        
        for (int i = 0; i < bitCount; ++i) {
            for (size_t j = 0; j < currentBlockSize - 1; ++j) {
                int byteIndex = j / 8;
                int bitIndex = j % 8;
                uint8_t bitVal = ((deltas[j] >> (bitCount - 1 - i)) & 1);
                result_matrix[i][byteIndex] |= bitVal << (7 - bitIndex);
            }
        }
        
        // Sparse/dense decision
        uint64_t flag1 = 0;
        std::vector<std::vector<uint8_t>> flag2_matrix(bitCount);
        
        for (int i = 0; i < bitCount; i++) {
            int flag2Size = (numByte + 7) / 8;
            flag2_matrix[i].resize(flag2Size, 0);
            
            int b0 = 0, b1 = 0;
            for (int j = 0; j < numByte; j++) {
                if (result_matrix[i][j] == 0) {
                    b0++;
                } else {
                    b1++;
                    int flag2_byte_idx = j / 8;
                    int flag2_bit_idx = j % 8;
                    flag2_matrix[i][flag2_byte_idx] |= (1 << flag2_bit_idx);
                }
            }
            
            uint64_t is_sparse = (uint64_t)(((numByte + 7) / 8 + b1) < numByte);
            if (is_sparse) {
                flag1 |= (1ULL << i);
            } else {
                flag1 &= ~(1ULL << i);
            }
        }
        
        // Calculate bitSize
        bitSize = 64 + 64 + 8 + 8 + 8 + 64;
        
        for (int i = 0; i < bitCount; i++) {
            if ((flag1 & (1ULL << i)) != 0) {
                int flag2Size = (numByte + 7) / 8;
                int nonZeroCount = 0;
                for (int j = 0; j < numByte; j++) {
                    if (result_matrix[i][j] != 0) nonZeroCount++;
                }
                bitSize += (flag2Size + nonZeroCount) * 8;
            } else {
                bitSize += numByte * 8;
            }
        }
        
        // Write data
        bitStream.WriteLong(bitSize, 64);
        bitStream.WriteLong(firstValue, 64);
        bitStream.WriteInt(maxDecimalPlaces, 8);
        bitStream.WriteInt(maxBeta, 8);
        bitStream.WriteInt(bitCount, 8);
        bitStream.WriteLong(flag1, 64);
        
        for (int i = 0; i < bitCount; i++) {
            if ((flag1 & (1ULL << i)) != 0) {
                int flag2Size = (numByte + 7) / 8;
                for (int j = 0; j < flag2Size; j++) {
                    bitStream.WriteByte(flag2_matrix[i][j]);
                }
                for (int j = 0; j < numByte; j++) {
                    if (result_matrix[i][j] != 0) {
                        bitStream.WriteByte(result_matrix[i][j]);
                    }
                }
            } else {
                for (int j = 0; j < numByte; j++) {
                    bitStream.WriteByte(result_matrix[i][j]);
                }
            }
        }
    }
    
public:
    T storedValue = 0;
    std::vector<uint8_t> bytes;  // For benchmark compatibility
    
    explicit CompressorFalcon(const T& value) {
        buffer.push_back(value);
        storedValue = value;
        totalValues = 1;
    }
    
    void addValue(const T& value) {
        buffer.push_back(value);
        storedValue = value;
        totalValues++;
    }
    
    void close() {
        // Compress all buffered data
        falcon::OutputBitStream bitStream(BLOCK_SIZE * 8);
        
        // Write total count
        bitStream.Write(totalValues, 64);
        bitStream.Flush();
        falcon::Array<uint8_t> header = bitStream.GetBuffer(8);
        for (int j = 0; j < header.length(); j++) {
            bytes.push_back(header[j]);
        }
        bitStream.Refresh();
        
        // Process blocks
        for (size_t i = 0; i < buffer.size(); i += BLOCK_SIZE) {
            int perBlockBitSize = 0;
            size_t currentBlockSize = std::min(BLOCK_SIZE, buffer.size() - i);
            std::vector<T> block(buffer.begin() + i, buffer.begin() + i + currentBlockSize);
            
            compressBlock(block, bitStream, perBlockBitSize);
            bitStream.Flush();
            falcon::Array<uint8_t> blockData = bitStream.GetBuffer((perBlockBitSize + 31) / 32 * 4);
            
            for (int j = 0; j < blockData.length(); j++) {
                bytes.push_back(blockData[j]);
            }
            bitStream.Refresh();
        }
    }
    
    size_t getSize() const {
        return bytes.size() * 8;
    }
    
    std::vector<uint8_t> getOut() const {
        return bytes;
    }
};

// Static member initialization
template<typename T>
constexpr double CompressorFalcon<T>::pow10_table[17];
