// Camel parity/validation tool (not a timing benchmark).
//   camel_dump <dataset.bin> [--doubles out.f64] [--blocks out.blocks]
//              [--decoded out.f64] [--decimal-max-count k] [--integer-bits b]
//              [--rescale-to k]
// Loads a GEF dataset (u64 n, optional u64 decimals, int64 values) exactly as
// LosslessBenchmark does (raw / 10^decimals, correctly rounded), encodes
// independent 1000-value blocks with benchmark/Camel/camel.hpp using the
// benchmark's per-dataset configuration (overridable) and prints size and
// round-trip statistics. --blocks/--decoded use the same layout as
// tools/camel/CamelDump.java, so the two outputs can be compared with cmp.
// --rescale-to k (k < dataset decimals) feeds raw / 10^k instead, i.e. the
// values multiplied by 10^(decimals-k) so that at most k decimals remain
// (a lossless, exactly invertible reinterpretation of the same integers);
// used only to compare against the full-precision configuration.
#include "../benchmark/Camel/camel.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static void put64(std::ofstream& f, uint64_t v) {
    for (int i = 0; i < 8; ++i) f.put(static_cast<char>(v >> (8 * i)));
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: camel_dump dataset.bin [--doubles f] [--blocks f] [--decoded f] "
                     "[--decimal-max-count k] [--integer-bits b] [--rescale-to k]\n";
        return 2;
    }
    std::string doubles_path, blocks_path, decoded_path;
    int dmc = 0, ibits = 0, rescale = -1;
    for (int i = 2; i + 1 < argc; i += 2) {
        std::string k = argv[i], v = argv[i + 1];
        if (k == "--doubles") doubles_path = v;
        else if (k == "--blocks") blocks_path = v;
        else if (k == "--decoded") decoded_path = v;
        else if (k == "--decimal-max-count") dmc = std::stoi(v);
        else if (k == "--integer-bits") ibits = std::stoi(v);
        else if (k == "--rescale-to") rescale = std::stoi(v);
        else { std::cerr << "unknown option " << k << '\n'; return 2; }
    }
    std::ifstream in(argv[1], std::ios::binary);
    if (!in) { std::cerr << "cannot open " << argv[1] << '\n'; return 1; }
    uint64_t n = 0;
    in.read(reinterpret_cast<char*>(&n), 8);
    in.seekg(0, std::ios::end);
    const uint64_t file_size = static_cast<uint64_t>(in.tellg());
    int64_t decimals = 0;
    in.seekg(8);
    if (file_size == 16 + 8 * n) in.read(reinterpret_cast<char*>(&decimals), 8);
    else if (file_size != 8 + 8 * n) { std::cerr << "unexpected file size\n"; return 1; }
    std::vector<int64_t> raw(n);
    in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(8 * n));
    if (rescale >= 0 && rescale < decimals) decimals = rescale;
    int64_t div = 1;
    for (int64_t d = 0; d < decimals; ++d) div *= 10;
    std::vector<double> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = static_cast<double>(raw[i]) / static_cast<double>(div);

    const size_t block = 1000;
    camel::Config cfg = camel::configure(decimals);
    if (dmc) cfg.decimal_max_count = dmc;
    if (ibits) cfg.integer_bits = ibits;

    if (!doubles_path.empty()) {
        std::ofstream f(doubles_path, std::ios::binary);
        put64(f, n);
        for (double x : v) put64(f, std::bit_cast<uint64_t>(x));
    }
    std::ofstream blocks_out, decoded_out;
    if (!blocks_path.empty()) blocks_out.open(blocks_path, std::ios::binary);
    if (!decoded_path.empty()) decoded_out.open(decoded_path, std::ios::binary);

    uint64_t bytes = 0, bit_mismatch = 0, tol_fail = 0;
    double max_err = 0;
    for (size_t s = 0; s < n; s += block) {
        const size_t e = std::min<size_t>(n, s + block);
        camel::Compressor c(cfg);
        for (size_t i = s; i < e; ++i) c.addValue(v[i]);
        const std::string& b = c.close();
        bytes += b.size();
        if (blocks_out) {
            const uint32_t len = static_cast<uint32_t>(b.size());
            for (int i = 0; i < 4; ++i) blocks_out.put(static_cast<char>(len >> (8 * i)));
            blocks_out.write(b.data(), static_cast<std::streamsize>(b.size()));
        }
        camel::Decompressor d(reinterpret_cast<const uint8_t*>(b.data()), b.size(), cfg);
        for (size_t i = s; i < e; ++i) {
            const double x = d.next();
            if (decoded_out) put64(decoded_out, std::bit_cast<uint64_t>(x));
            bit_mismatch += std::bit_cast<uint64_t>(x) != std::bit_cast<uint64_t>(v[i]);
            const double err = std::abs(x - v[i]);
            if (!(err < 1e-9)) ++tol_fail;
            if (err > max_err || std::isnan(err)) max_err = err;
        }
    }
    std::printf("dataset=%s n=%llu decimals=%lld decimal_max_count=%d count_bits=%d integer_bits=%d "
                "bytes=%llu bits=%llu ratio=%.6f bitwise_mismatches=%llu tolerance_failures=%llu max_abs_err=%.3g\n",
                argv[1], (unsigned long long)n, (long long)decimals, cfg.decimal_max_count, cfg.count_bits(),
                cfg.integer_bits, (unsigned long long)bytes, (unsigned long long)(bytes * 8),
                n ? double(bytes * 8) / double(n * 64) : 0.0, (unsigned long long)bit_mismatch,
                (unsigned long long)tol_fail, max_err);
    return tol_fail ? 3 : 0;
}
