#include "benchmark_common.hpp"
#include <cstring>
#include <iostream>
// Expected bits from correctly rounded decimal strings; no fast-math arithmetic.
struct Case { int64_t raw, decimals; uint64_t expected; };
int main() {
    const Case cases[] = {
        {3, 0, UINT64_C(0x4008000000000000)},
        {7, 0, UINT64_C(0x401c000000000000)},
        {-3, 0, UINT64_C(0xc008000000000000)},
        {123456789, 0, UINT64_C(0x419d6f3454000000)},
        {3, 1, UINT64_C(0x3fd3333333333333)},
        {7, 1, UINT64_C(0x3fe6666666666666)},
        {-3, 1, UINT64_C(0xbfd3333333333333)},
        {123456789, 1, UINT64_C(0x41678c29dccccccd)},
        {3, 2, UINT64_C(0x3f9eb851eb851eb8)},
        {7, 2, UINT64_C(0x3fb1eb851eb851ec)},
        {-3, 2, UINT64_C(0xbf9eb851eb851eb8)},
        {123456789, 2, UINT64_C(0x4132d687e3d70a3d)},
        {3, 3, UINT64_C(0x3f689374bc6a7efa)},
        {7, 3, UINT64_C(0x3f7cac083126e979)},
        {-3, 3, UINT64_C(0xbf689374bc6a7efa)},
        {123456789, 3, UINT64_C(0x40fe240c9fbe76c9)},
        {3, 4, UINT64_C(0x3f33a92a30553261)},
        {7, 4, UINT64_C(0x3f46f0068db8bac7)},
        {-3, 4, UINT64_C(0xbf33a92a30553261)},
        {123456789, 4, UINT64_C(0x40c81cd6e631f8a1)},
        {3, 5, UINT64_C(0x3eff75104d551d69)},
        {7, 5, UINT64_C(0x3f12599ed7c6fbd2)},
        {-3, 5, UINT64_C(0xbeff75104d551d69)},
        {123456789, 5, UINT64_C(0x40934a4584f4c6e7)},
        {3, 7, UINT64_C(0x3e9421f5f40d8376)},
        {7, 7, UINT64_C(0x3ea77cf44765195f)},
        {-3, 7, UINT64_C(0xbe9421f5f40d8376)},
        {123456789, 7, UINT64_C(0x4028b0fcd324d5a2)},
        {3, 9, UINT64_C(0x3e29c511dc3a41df)},
        {7, 9, UINT64_C(0x3e3e1094d643f784)},
        {-3, 9, UINT64_C(0xbe29c511dc3a41df)},
        {123456789, 9, UINT64_C(0x3fbf9add3739635f)},
    };
    size_t checked = 0;
    for (const auto& c : cases) {
        BenchmarkData data;
        data.decimals = c.decimals;
        data.min_val = -100;
        data.shifted_data.assign(1031, c.raw - data.min_val);
        data.convert_shifted_to_double();
        if (!data.shifted_data.empty() || data.double_data.size() != 1031) return 1;
        for (double value : data.double_data) {
            uint64_t bits;
            std::memcpy(&bits, &value, sizeof bits);
            if (bits != c.expected) {
                std::cerr << "Wrong input bits: raw=" << c.raw << " decimals="
                          << c.decimals << " expected=" << std::hex << c.expected
                          << " actual=" << bits << '\n';
                return 1;
            }
            ++checked;
        }
    }
    std::cout << "Strict decimal input: " << checked << " bitwise checks passed\n";
}
