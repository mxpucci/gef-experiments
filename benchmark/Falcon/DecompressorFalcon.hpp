#pragma once

/**
 * Falcon Decompressor - Header-only C++ port from:
 * https://github.com/Spatio-Temporal-Lab/Falcon
 * 
 * This is a faithful port of the original CPU implementation from:
 * - src/cpu/Falcon_basic_decompressor.cpp
 * - src/utils/input_bit_stream.cc
 */

#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstring>
#include <memory>

// Portable byte swap
#if defined(__APPLE__)
#include <libkern/OSByteOrder.h>
#define be32toh_portable(x) OSSwapBigToHostInt32(x)
#elif defined(__linux__)
#include <endian.h>
#define be32toh_portable(x) be32toh(x)
#else
inline uint32_t be32toh_portable(uint32_t x) {
    return ((x & 0xFF) << 24) | ((x & 0xFF00) << 8) | ((x >> 8) & 0xFF00) | ((x >> 24) & 0xFF);
}
#endif

namespace falcon {

// InputBitStream from original Falcon (src/utils/input_bit_stream.cc)
class InputBitStream {
public:
    InputBitStream() = default;
    
    void SetBuffer(const std::vector<uint8_t>& new_buffer) {
        data_ = Array<uint32_t>(std::ceil(static_cast<double>(new_buffer.size()) / sizeof(uint32_t)));
        std::memcpy(data_.begin(), new_buffer.data(), new_buffer.size());
        for (auto& blk : data_) blk = be32toh_portable(blk);
        buffer_ = (static_cast<uint64_t>(data_[0])) << 32;
        cursor_ = 1;
        bit_in_buffer_ = 32;
    }
    
    uint64_t ReadLong(size_t len) {
        if (len == 0) return 0;
        uint64_t ret = 0;
        if (len > 32) {
            ret = Peek(32);
            Forward(32);
            ret <<= len - 32;
            len -= 32;
        }
        ret |= Peek(len);
        Forward(len);
        return ret;
    }
    
    uint32_t ReadInt(size_t len) {
        if (len == 0) return 0;
        uint32_t ret = Peek(len);
        Forward(len);
        return ret;
    }
    
    uint8_t ReadByte(size_t len) {
        if (len == 0) return 0;
        uint8_t ret = Peek(len);
        Forward(len);
        return ret;
    }
    
private:
    uint64_t Peek(size_t len) {
        return buffer_ >> (64 - len);
    }
    
    void Forward(size_t len) {
        bit_in_buffer_ -= len;
        buffer_ <<= len;
        if (bit_in_buffer_ < 32) {
            if (cursor_ < static_cast<uint64_t>(data_.length())) {
                auto next = static_cast<uint64_t>(data_[cursor_]);
                buffer_ |= (next << (32 - bit_in_buffer_));
                bit_in_buffer_ += 32;
                cursor_++;
            } else {
                bit_in_buffer_ = 64;
            }
        }
    }
    
    Array<uint32_t> data_;
    uint64_t buffer_ = 0;
    uint64_t cursor_ = 0;
    uint64_t bit_in_buffer_ = 0;
};

} // namespace falcon

// Main Falcon Decompressor - ported from Falcon_basic_decompressor.cpp
template<typename T = double>
class DecompressorFalcon {
    static_assert(std::is_same_v<T, double>, "DecompressorFalcon only supports double");
    
    static constexpr size_t BLOCK_SIZE = 1025;
    
    std::vector<T> decompressedValues;
    size_t valueIndex = 0;
    bool endOfStream = false;
    size_t n;
    size_t i = 0;
    
    // From original: zigzag_decode
    static int64_t zigzag_decode(uint64_t value) {
        return (value >> 1) ^ -(static_cast<int64_t>(value & 1));
    }
    
    // From original: decodeDoubleWithSignLast
    static double decodeDoubleWithSignLast(uint64_t value) {
        uint64_t original = (value >> 1) ^ -(static_cast<int64_t>(value & 1));
        union { uint64_t u; double d; } val;
        val.u = original;
        return val.d;
    }
    
    // From original: decompressBlock
    void decompressBlock(falcon::InputBitStream& bitStream, std::vector<int64_t>& originalData,
                         int& totalBitsRead, size_t blockSize, int& maxDecimalPlaces, int& isOk) {
        int64_t firstValue = bitStream.ReadLong(64);
        maxDecimalPlaces = bitStream.ReadInt(8);
        int maxBeta = bitStream.ReadInt(8);
        int bitCount = bitStream.ReadInt(8);
        uint64_t flag1 = bitStream.ReadLong(64);
        
        isOk = (maxBeta <= 15 && maxDecimalPlaces <= 15) ? 1 : 0;
        
        if (bitCount == 0 || bitCount > 64) {
            return;
        }
        
        int numByte = (blockSize - 1 + 7) / 8;
        int flag2Size = (numByte + 7) / 8;
        
        std::vector<std::vector<uint8_t>> result_matrix(bitCount, std::vector<uint8_t>(numByte, 0));
        
        for (int col = 0; col < bitCount; col++) {
            if ((flag1 & (1ULL << col)) != 0) {
                std::vector<uint8_t> flag2(flag2Size);
                for (int j = 0; j < flag2Size; j++) {
                    flag2[j] = bitStream.ReadByte(8);
                }
                
                for (int j = 0; j < numByte; j++) {
                    int flag2_byte_idx = j / 8;
                    int flag2_bit_idx = j % 8;
                    if (flag2[flag2_byte_idx] & (1 << flag2_bit_idx)) {
                        result_matrix[col][j] = bitStream.ReadByte(8);
                    } else {
                        result_matrix[col][j] = 0;
                    }
                }
            } else {
                for (int j = 0; j < numByte; j++) {
                    result_matrix[col][j] = bitStream.ReadByte(8);
                }
            }
        }
        
        std::vector<uint64_t> deltas(blockSize - 1, 0);
        
        for (size_t j = 0; j < blockSize - 1; j++) {
            for (int col = 0; col < bitCount; col++) {
                int byteIndex = j / 8;
                int bitIndex = j % 8;
                uint8_t bitVal = (result_matrix[col][byteIndex] >> (7 - bitIndex)) & 1;
                deltas[j] |= (static_cast<uint64_t>(bitVal) << (bitCount - 1 - col));
            }
        }
        
        originalData.clear();
        originalData.push_back(firstValue);
        
        int64_t prevValue = firstValue;
        for (size_t j = 0; j < blockSize - 1; j++) {
            int64_t deltaDecoded = zigzag_decode(deltas[j]);
            int64_t currentValue = prevValue + deltaDecoded;
            originalData.push_back(currentValue);
            prevValue = currentValue;
        }
    }
    
    void decompressAll(const std::vector<uint8_t>& input) {
        if (input.size() < 8) return;
        
        falcon::InputBitStream bitStream;
        bitStream.SetBuffer(input);
        
        int64_t totalValues = bitStream.ReadLong(64);
        size_t numBlocks = (totalValues + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        for (size_t blk = 0; blk < numBlocks; blk++) {
            size_t currentBlockSize = std::min(BLOCK_SIZE, static_cast<size_t>(totalValues) - blk * BLOCK_SIZE);
            std::vector<int64_t> integers;
            int totalBitsRead = bitStream.ReadLong(64);
            int maxDecimalPlaces = 0;
            int isOk;
            
            decompressBlock(bitStream, integers, totalBitsRead, currentBlockSize, maxDecimalPlaces, isOk);
            
            if (isOk == 0) {
                for (int64_t intValue : integers) {
                    double d = decodeDoubleWithSignLast(static_cast<uint64_t>(intValue));
                    decompressedValues.push_back(d);
                }
            } else {
                double divisor = std::pow(10.0, maxDecimalPlaces);
                for (int64_t intValue : integers) {
                    double value = static_cast<double>(intValue) / divisor;
                    decompressedValues.push_back(value);
                }
            }
            
            // Skip padding
            int paddingBits = (totalBitsRead + 31) / 32 * 32 - totalBitsRead;
            if (paddingBits > 0) {
                bitStream.ReadLong(paddingBits);
            }
        }
    }
    
public:
    T storedValue = 0;
    
    DecompressorFalcon(const std::vector<uint8_t>& bytes, size_t nlines) : n(nlines) {
        decompressAll(bytes);
        
        if (!decompressedValues.empty()) {
            storedValue = decompressedValues[0];
            valueIndex = 0;
            ++i;
        }
        
        if (i > n || decompressedValues.empty()) {
            endOfStream = true;
        }
    }
    
    bool hasNext() {
        if (!endOfStream) {
            valueIndex++;
            if (valueIndex < decompressedValues.size()) {
                storedValue = decompressedValues[valueIndex];
                ++i;
                if (i > n) {
                    endOfStream = true;
                }
            } else {
                endOfStream = true;
            }
        }
        return !endOfStream;
    }
};
