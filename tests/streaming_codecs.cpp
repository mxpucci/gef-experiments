// Round trips of the in-harness streaming codecs, compiled with the same strict
// floating-point flags as lossless_benchmark.cpp.
#include <array>
#include <bit>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "../benchmark/Chimp/CompressorChimp.hpp"
#include "../benchmark/Chimp/DecompressorChimp.hpp"
#include "../benchmark/Chimp128/CompressorChimp128.hpp"
#include "../benchmark/Chimp128/DecompressorChimp128.hpp"
#include "../benchmark/TSXor/CompressorTSXor.hpp"
#include "../benchmark/TSXor/DecompressorTSXor.hpp"
#include "../benchmark/Elf/CompressorElf.hpp"
#include "../benchmark/Elf/DecompressorElf.hpp"

template<class C, class D>
static void round_trip(const char *name, const std::vector<double> &values) {
    for (size_t block : {size_t(1), size_t(2), size_t(1000)}) {
        for (size_t start = 0; start < values.size(); start += block) {
            const size_t count = std::min(block, values.size() - start);
            C compressor(values[start]);
            for (size_t i = 1; i < count; ++i) compressor.addValue(values[start + i]);
            compressor.close();
            D decompressor(compressor.getBuffer(), count);
            for (size_t i = 0; i < count; ++i) {
                if ((i && !decompressor.hasNext()) ||
                    std::bit_cast<uint64_t>(decompressor.storedValue) != std::bit_cast<uint64_t>(values[start + i]))
                    throw std::runtime_error(std::string(name) + " round trip failed at " +
                                             std::to_string(start + i) + " (block " + std::to_string(block) + ")");
            }
        }
    }
}

int main() {
    std::mt19937_64 rng(42);
    std::vector<double> bits, decimals;
    for (int i = 0; i < 20000; ++i) {
        uint64_t b;
        do b = rng(); while (((b >> 52) & 0x7ff) == 0x7ff);  // finite values only
        bits.push_back(std::bit_cast<double>(b));
        // Fixed-decimal inputs as produced by benchmark_input.cpp.
        const int64_t digits[] = {1, 10, 100, 1000, 10000, 100000};
        const int64_t d = digits[i % 6];
        decimals.push_back(static_cast<double>(static_cast<int64_t>(rng() % (200 * d)) - 100 * d) / static_cast<double>(d));
    }
    for (auto *data : {&bits, &decimals}) {
        round_trip<CompressorChimp<double>, DecompressorChimp<double>>("Chimp", *data);
        round_trip<CompressorChimp128<double>, DecompressorChimp128<double>>("Chimp128", *data);
        round_trip<CompressorTSXor<double>, DecompressorTSXor<double>>("TSXor", *data);
    }
    // Elf is checked on fixed-decimal inputs, its documented domain.
    round_trip<CompressorElf<double>, DecompressorElf<double>>("Elf", decimals);
    std::cout << "streaming codecs round trip\n";
}
