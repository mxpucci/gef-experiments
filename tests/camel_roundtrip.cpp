// Camel native port checks:
//  1. byte-for-byte equality with the authors' Java (DECIMAL_MAX_COUNT=4,
//     16-bit integer differences) on a fixed sequence, including Java's
//     silent truncation of a > 16-bit integer jump (golden bytes and decoded
//     values produced by tools/camel/CamelDump.java);
//  2. round trips with the benchmark's per-dataset configuration for random
//     decimal series with 1..12 decimals, 1000-value blocks and tails: exact
//     up to 4 decimals, within the benchmark's 1e-9 tolerance beyond.
#include "../benchmark/Camel/camel.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

static std::vector<uint8_t> from_hex(const char* h) {
    std::vector<uint8_t> out;
    for (; h[0] && h[1]; h += 2) out.push_back(static_cast<uint8_t>(std::stoi(std::string(h, 2), nullptr, 16)));
    return out;
}

static std::string encode(const std::vector<double>& v, size_t s, size_t e, camel::Config cfg) {
    camel::Compressor c(cfg);
    for (size_t i = s; i < e; ++i) c.addValue(v[i]);
    return c.close();
}

int main() {
    const std::vector<double> seq = {12.5, 12.75, -3.25, -0.5, 0.0, 7.1234, 7.1, 100000.3,
                                     -70000.02, 1.0, 2.999, 3.0001, -1.5, -2.0, 65.4321, 65.0};
    const std::vector<double> java_decoded = {12.5, 12.75, -3.25, -0.5, 0.0, 7.1234, 7.1, 34464.3,
                                              -4464.02, 1.0, 2.999, 3.0001, -1.5, -2.0, 65.4321, 65.0};
    // CamelDump.java output: u32 length 0x2c, then the block bytes.
    const auto golden = from_hex("4029000000000000af068007b41ccc503df8f30d07f0d321b6604117c45c40d7ff360590c003e0087db8ee80");
    camel::Config java_cfg; // DECIMAL_MAX_COUNT=4, 16-bit differences
    const std::string bytes = encode(seq, 0, seq.size(), java_cfg);
    assert(bytes.size() == 0x2c && bytes.size() == golden.size());
    for (size_t i = 0; i < golden.size(); ++i) assert(static_cast<uint8_t>(bytes[i]) == golden[i]);
    {
        camel::Decompressor d(golden.data(), golden.size(), java_cfg);
        for (double expected : java_decoded) assert(std::bit_cast<uint64_t>(d.next()) == std::bit_cast<uint64_t>(expected));
    }
    {
        auto cfg = camel::configure(4);
        assert(cfg.integer_bits == 16 && cfg.decimal_max_count == 4);
        cfg.integer_bits = 18; // the README's per-dataset knob makes the jump fit
        const std::string b = encode(seq, 0, seq.size(), cfg);
        camel::Decompressor d(reinterpret_cast<const uint8_t*>(b.data()), b.size(), cfg);
        for (double expected : seq) assert(std::bit_cast<uint64_t>(d.next()) == std::bit_cast<uint64_t>(expected));
    }

    std::mt19937_64 rng(7);
    size_t inexact = 0;
    for (int decimals = 1; decimals <= 12; ++decimals) {
        int64_t scale = 1;
        for (int i = 0; i < decimals; ++i) scale *= 10;
        for (int64_t magnitude : {int64_t{10}, int64_t{1000}, int64_t{1000000}}) {
            const int64_t limit = std::min<int64_t>(magnitude * scale, int64_t{1} << 52);
            std::uniform_int_distribution<int64_t> start(-limit, limit), step(-scale * 3, scale * 3);
            std::vector<double> v;
            int64_t x = start(rng);
            for (int i = 0; i < 2537; ++i) {
                x = std::clamp<int64_t>(x + step(rng), -limit, limit);
                if (i % 97 == 0) x = start(rng); // large jumps
                v.push_back(static_cast<double>(x) / static_cast<double>(scale));
            }
            auto cfg = camel::configure(decimals);
            cfg.integer_bits = 22; // exercise the codec, not Java's 16-bit truncation
            for (size_t s = 0; s < v.size(); s += 1000) {
                const size_t e = std::min(v.size(), s + 1000);
                const std::string b = encode(v, s, e, cfg);
                camel::Decompressor d(reinterpret_cast<const uint8_t*>(b.data()), b.size(), cfg);
                for (size_t i = s; i < e; ++i) {
                    const double got = d.next();
                    const bool exact = std::bit_cast<uint64_t>(got) == std::bit_cast<uint64_t>(v[i]);
                    inexact += !exact;
                    // The Java reconstruction (I*10^l + dec*10^l) / 10^l can be off by an
                    // ulp once I*10^l nears 2^53; the benchmark's tolerance is 1e-9.
                    if ((decimals <= 4 && !exact) || !(std::abs(got - v[i]) < 1e-9)) {
                        std::fprintf(stderr, "decimals=%d magnitude=%lld index=%zu: %.17g -> %.17g\n", decimals,
                                     (long long)magnitude, i, v[i], got);
                        return 1;
                    }
                }
            }
        }
    }
    std::printf("camel_roundtrip: ok (%zu values within 1e-9 but not bitwise equal)\n", inexact);
}
