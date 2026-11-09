// Camel (Yao et al., "Camel: Efficient Compression of Floating-Point Time
// Series", SIGMOD 2025): C++ translation of the authors' Java codec
//   https://github.com/yoyo185644/camel @ c4b55c2226163b0bfb6c330aa78b7c0a97d872b3
//   src/main/java/yyy/ts/compress/camel/{Camel,CamelDecompressor}.java
// The paper authors publish no C/C++ code, so this file translates the Java
// statement by statement (same double arithmetic, same bit layout, same
// Java cast/rounding semantics). With decimal_max_count <= 4 and
// integer_bits == 16 the byte stream is identical to Java's Camel with
// DECIMAL_MAX_COUNT = decimal_max_count (verified with tools/camel/CamelDump.java vs tests/camel_dump).
//
// The only deviation from upstream, needed because the Java cannot represent
// more than 4 decimals (lossless precision, per the benchmark's policy):
//  * decimal_max_count > 4: the Java writes (l-1) in 2 bits, so it cannot
//    represent more than 4 decimals. The field widens to
//    ceil(log2(decimal_max_count)) bits, and for l > 4 the dxor' integer
//    (m in the Java) uses the paper's rule for l > 2 (Algorithm 2 lines
//    19-22 / Algorithm 5 lines 7-10): max = ceil(log2(5^l)) bits, a 2-bit
//    selector and widths floor((i+1)*max/4), i = 0..3. Java's l <= 4 tables
//    are kept verbatim.
// integer_bits is the Java's 16-bit integer-difference width (a parameter in
// the README / paper Sec. 4.4.1, used here only for Java parity tests). The
// benchmark keeps 16: larger jumps are truncated exactly as in the Java and
// show up as logged round-trip mismatches (LAT: 1 value), not patched.
// Configuration is fixed per stream (as in Java) and is not stored.
#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#ifdef __FAST_MATH__
#error "Camel requires strict IEEE arithmetic (the Java codec is bit-exact)"
#endif

namespace camel {

struct Config {
    int decimal_max_count = 4; // Camel.DECIMAL_MAX_COUNT (Java default 3)
    int integer_bits = 16;     // width of large integer differences

    int count_bits() const { // Java: 2 bits, values 00-1 01-2 10-3 11-4
        int b = 2;
        while ((1 << b) < decimal_max_count) ++b;
        return b;
    }
    void check() const {
        if (decimal_max_count < 1 || decimal_max_count > 18)
            throw std::invalid_argument("Camel: decimal_max_count must be in [1,18]");
        if (integer_bits < 3 || integer_bits > 31)
            throw std::invalid_argument("Camel: integer_bits must be in [3,31]");
    }
};

namespace detail {

inline constexpr int64_t power10(int e) {
    int64_t p = 1;
    for (int i = 0; i < e; ++i) p *= 10;
    return p;
}
inline constexpr int64_t power5(int e) {
    int64_t p = 1;
    for (int i = 0; i < e; ++i) p *= 5;
    return p;
}

// Java (int) cast of a double: truncation, NaN -> 0, saturation.
inline int32_t java_d2i(double v) {
    if (std::isnan(v)) return 0;
    if (v >= 2147483647.0) return std::numeric_limits<int32_t>::max();
    if (v <= -2147483648.0) return std::numeric_limits<int32_t>::min();
    return static_cast<int32_t>(v);
}

// java.lang.Math.round(double) (Java 7+): floor(a + 1/2) evaluated exactly,
// NaN -> 0, saturating at the long range.
inline int64_t java_round(double a) {
    if (std::isnan(a)) return 0;
    if (a >= 9223372036854775807.0) return std::numeric_limits<int64_t>::max();
    if (a <= -9223372036854775808.0) return std::numeric_limits<int64_t>::min();
    const double f = std::floor(a);
    // a - floor(a) is exact; for |a| >= 2^52, a is an integer already.
    return static_cast<int64_t>(f) + (a - f >= 0.5 ? 1 : 0);
}

// Width of the dxor' (m) field for decimal count l and 2-bit selector index.
inline int m_width(int l, int index) {
    static constexpr int l3[4] = {1, 3, 5, 7};  // Java, decimal_count == 3
    static constexpr int l4[4] = {4, 6, 8, 10}; // Java, decimal_count == 4
    if (l == 3) return l3[index];
    if (l == 4) return l4[index];
    // Paper, l > 2: max = ceil(log2(2^-l * 10^l)) = bit length of 5^l - 1.
    const int max = std::bit_width(static_cast<uint64_t>(power5(l) - 1));
    return (index + 1) * max / 4;
}

class BitWriter { // gr.aueb.delorean.chimp.OutputBitStream (MSB first)
public:
    std::string bytes;
    void write(uint64_t value, int len) { // low len bits of value
        for (int i = len - 1; i >= 0; --i) {
            if (free_ == 8) bytes.push_back(0);
            --free_;
            if ((value >> i) & 1) bytes.back() |= static_cast<char>(1u << free_);
            if (free_ == 0) free_ = 8;
        }
    }
    uint64_t written_bits() const { return bytes.size() * 8 - (free_ == 8 ? 0 : free_); }

private:
    int free_ = 8;
};

class BitReader { // gr.aueb.delorean.chimp.InputBitStream
public:
    BitReader(const uint8_t* data, size_t size) : data_(data), size_(size) {}
    uint64_t read(int len) {
        uint64_t x = 0;
        for (int i = 0; i < len; ++i) {
            const size_t byte = pos_ >> 3;
            if (byte >= size_) throw std::runtime_error("Camel: read past end of stream");
            x = (x << 1) | ((data_[byte] >> (7 - (pos_ & 7))) & 1u);
            ++pos_;
        }
        return x;
    }

private:
    const uint8_t* data_;
    size_t size_;
    size_t pos_ = 0;
};

} // namespace detail

// yyy.ts.compress.camel.Camel
class Compressor {
public:
    explicit Compressor(Config config = {}) : cfg_(config) { cfg_.check(); }

    void addValue(double value) { // Camel.addValue(double)
        if (first_) {
            const auto [decimal_count, decimal_value] = cal_decimal_count(value);
            const int64_t p = detail::power10(decimal_count);
            value = (value < 0 ? -1 : 1) *
                    static_cast<double>(static_cast<int64_t>(detail::java_d2i(std::fabs(value))) * p + decimal_value) /
                    static_cast<double>(p);
            writeFirst(std::bit_cast<uint64_t>(value));
        } else {
            compressValue(value);
        }
    }

    // Camel.close(): one 0 bit, then OutputBitStream.flush() pads the byte.
    const std::string& close() {
        out_.write(0, 1);
        return out_.bytes;
    }
    const std::string& bytes() const { return out_.bytes; }

    struct Decimal {
        int count;
        int64_t value;
    };
    Decimal cal_decimal_count(double value) const { // Camel.cal_decimal_count
        double factor = 1;
        int decimal_count = 0;
        value = std::fabs(value);
        const double epsilon = 0.0000001;
        while (std::fabs(value * factor - static_cast<double>(detail::java_round(value * factor))) > epsilon) {
            factor *= 10.0;
            decimal_count++;
        }
        int64_t decimal_value = 0;
        if (decimal_count == 0) decimal_count = 1;
        if (decimal_count > 0 && decimal_count <= cfg_.decimal_max_count) {
            const int64_t p = detail::power10(decimal_count);
            decimal_value = detail::java_round(value * static_cast<double>(p)) % p;
        } else {
            const int64_t p = detail::power10(cfg_.decimal_max_count);
            decimal_value = detail::java_round(value * static_cast<double>(p)) % p;
            decimal_count = cfg_.decimal_max_count;
        }
        return {decimal_count, decimal_value};
    }

private:
    void writeFirst(uint64_t value) {
        first_ = false;
        storedVal_ = detail::java_d2i(std::bit_cast<double>(value));
        out_.write(value, 64);
    }

    void compressValue(double value) {
        const int intSignal = value < 0 ? 0 : 1;
        compressIntegerValue(detail::java_d2i(value), intSignal);
        const auto [decimal_count, decimal_value] = cal_decimal_count(value);
        compressDecimalValue(decimal_value, decimal_count);
    }

    void compressDecimalValue(int64_t decimal_value, int decimal_count) {
        if (decimal_count == 0) return;
        out_.write(static_cast<uint64_t>(decimal_count - 1), cfg_.count_bits());
        const int64_t thread = detail::power5(decimal_count); // threshold[] = {5,25,125,625}
        const int64_t p = detail::power10(decimal_count);
        int64_t m = decimal_value; // Java: int
        if (decimal_value - thread >= 0) {
            out_.write(1, 1);
            m = decimal_value % thread;
            const uint64_t xr =
                std::bit_cast<uint64_t>(static_cast<double>(decimal_value) / static_cast<double>(p) + 1) ^
                std::bit_cast<uint64_t>(static_cast<double>(m) / static_cast<double>(p) + 1);
            out_.write(xr >> (52 - decimal_count), decimal_count);
        } else {
            out_.write(0, 1);
        }
        const uint64_t mu = static_cast<uint64_t>(m);
        if (decimal_count == 1) {
            out_.write(mu, 3);
        } else if (decimal_count == 2) {
            if (m < 8) {
                out_.write(0, 1);
                out_.write(mu, 3);
            } else {
                out_.write(1, 1);
                out_.write(mu, 5);
            }
        } else {
            int index = 0;
            while (index < 3 && m >= (int64_t{1} << detail::m_width(decimal_count, index))) ++index;
            out_.write(static_cast<uint64_t>(index), 2);
            out_.write(mu, detail::m_width(decimal_count, index));
        }
    }

    void compressIntegerValue(int64_t int_value, int intSignal) {
        int32_t diff = static_cast<int32_t>(int_value - storedVal_);
        out_.write(static_cast<uint64_t>(intSignal), 1);
        if (diff >= -1 && diff <= 1) {
            out_.write(static_cast<uint64_t>(diff + 1), 2);
        } else {
            out_.write(3, 2);
            if (diff < 0) {
                out_.write(0, 1);
                diff = -diff;
            } else {
                out_.write(1, 1);
            }
            if (diff >= 2 && diff < 8) {
                out_.write(0, 1);
                out_.write(static_cast<uint64_t>(diff), 3);
            } else {
                out_.write(1, 1);
                out_.write(static_cast<uint64_t>(static_cast<uint32_t>(diff)), cfg_.integer_bits);
            }
        }
        storedVal_ = int_value;
    }

    Config cfg_;
    detail::BitWriter out_;
    int64_t storedVal_ = 0;
    bool first_ = true;
};

// yyy.ts.compress.camel.CamelDecompressor (the caller supplies the value
// count; Java stops at DEFAULT_BLOCK_SIZE = 1000).
class Decompressor {
public:
    Decompressor(const uint8_t* data, size_t size, Config config = {}) : cfg_(config), in_(data, size) {
        cfg_.check();
    }

    double next() {
        if (first_) {
            first_ = false;
            const double fistVal = std::bit_cast<double>(in_.read(64));
            storedVal_ = detail::java_d2i(fistVal);
            return fistVal;
        }
        return nextValue();
    }

private:
    double nextValue() {
        const int intSignal = static_cast<int>(in_.read(1));
        const int integerNum = static_cast<int>(in_.read(2));
        int64_t diffVal;
        if (integerNum < 3) {
            diffVal = integerNum - 1;
        } else {
            const int diffSymbol = static_cast<int>(in_.read(1));
            const int range = static_cast<int>(in_.read(1));
            diffVal = static_cast<int64_t>(in_.read(range == 0 ? 3 : cfg_.integer_bits));
            diffVal = (diffSymbol == 0 ? -1 : 1) * diffVal;
        }
        storedVal_ = diffVal + storedVal_;
        int64_t sign = storedVal_ >= 0 ? 1 : 0;
        if (storedVal_ == 0 && intSignal == 0) sign = 0;
        const int64_t intAbs = storedVal_ < 0 ? -storedVal_ : storedVal_;

        // readDecimal()
        const int decimal_count = static_cast<int>(in_.read(cfg_.count_bits())) + 1;
        const int isM = static_cast<int>(in_.read(1));
        uint64_t xorString = 0;
        if (isM == 1) xorString = in_.read(decimal_count) << (52 - decimal_count);
        int64_t m_int;
        if (decimal_count <= 1) {
            m_int = static_cast<int64_t>(in_.read(3));
        } else if (decimal_count == 2) {
            m_int = static_cast<int64_t>(in_.read(in_.read(1) == 0 ? 3 : 5));
        } else {
            const int index = static_cast<int>(in_.read(2));
            m_int = static_cast<int64_t>(in_.read(detail::m_width(decimal_count, index)));
        }
        const int64_t p = detail::power10(decimal_count);
        const double scale = static_cast<double>(p);
        double decimalVal;
        if (isM == 1) {
            const double m = static_cast<double>(m_int) / scale + 1;
            decimalVal = std::bit_cast<double>(xorString ^ std::bit_cast<uint64_t>(m)) - 1;
        } else {
            decimalVal = static_cast<double>(m_int) / static_cast<double>(p);
        }
        decimalVal = static_cast<double>(detail::java_round(decimalVal * scale)) / scale;

        // nextValue(): long*long + double*long, then / long.
        return static_cast<double>(sign == 0 ? -1 : 1) *
               ((static_cast<double>(intAbs * p) + decimalVal * static_cast<double>(p)) / static_cast<double>(p));
    }

    Config cfg_;
    detail::BitReader in_;
    int64_t storedVal_ = 0;
    bool first_ = true;
};

// Benchmark configuration: DECIMAL_MAX_COUNT = the dataset's decimal count,
// at least 4 (the Java benchmark configuration; identical streams for <= 4
// decimals); integer-difference width = the Java's 16.
inline Config configure(int64_t decimals) {
    Config cfg;
    cfg.decimal_max_count = static_cast<int>(decimals < 4 ? 4 : (decimals > 18 ? 18 : decimals));
    return cfg;
}

} // namespace camel
