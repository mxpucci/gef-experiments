#pragma once
// C++ port of SuDIS-ZJU/DeXOR's Native double encoder/decoder (MIT).
// Upstream: 06128b3df57cc1e8c53681231e235e22eb508c8d. See LICENSE and docs/dexor.md.
//
// The encoder reproduces upstream's bitstream, including its tolerance-based
// acceptance of decimal encodings (DoubleDeXOREncoder.java:113), so decoding is
// NOT guaranteed bit-exact. Java semantics (saturating double->long casts,
// Math.round, Double.toString) are replicated explicitly. Deviations are
// limited to inputs on which upstream throws (array index out of range): those
// values use DeXOR's own exception encoding instead of aborting compression.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(__FAST_MATH__)
#error "DeXOR requires IEEE floating-point semantics; compile without -ffast-math"
#endif

namespace gef_compressors { namespace dexor {
namespace detail {
inline uint64_t bits(double x) { uint64_t u; std::memcpy(&u, &x, 8); return u; }
inline double value(uint64_t u) { double x; std::memcpy(&x, &u, 8); return x; }
static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559,
              "DeXOR requires IEEE binary64");

// ---- DeXORTools.java ------------------------------------------------------
constexpr double equal_eps = 1e-23;
constexpr double integer_eps = 1e-6;
constexpr double P10[] = {
    1e-23,1e-22,1e-21,1e-20,1e-19,1e-18,1e-17,1e-16,1e-15,1e-14,
    1e-13,1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,
    1e-2,1e-1,1,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9,1e10,1e11,
    1e12,1e13,1e14,1e15,1e16,1e17,1e18,1e19,1e20,1e21,1e22,1e23};
constexpr unsigned cost[] = {0,4,7,10,14,17,20,24,27,30,34,37,40,44,47,50,54,57,60,64,67,70,74,77};
// Thrown where upstream Java throws ArrayIndexOutOfBoundsException.
struct OutOfRange {};
inline double getP10(int pow) {
    if (pow < -23 || pow > 23) throw OutOfRange{};
    return P10[pow + 23];
}
// Java (long) cast of a double: NaN -> 0, saturating at the long range.
inline int64_t java_long(double x) {
    if (std::isnan(x)) return 0;
    if (x >= 0x1p63) return std::numeric_limits<int64_t>::max();
    if (x <= -0x1p63) return std::numeric_limits<int64_t>::min();
    return static_cast<int64_t>(x);
}
// Java Math.round(double): floor(x + 1/2) computed exactly, then (long).
inline int64_t java_round(double x) {
    if (std::isnan(x)) return 0;
    const double f = std::floor(x);
    return java_long(x - f >= 0.5 ? f + 1.0 : f);
}
inline int comp(double a, double b, double eps) {
    const double delta = a - b;
    if (delta >= eps) return 1;
    if (delta <= -eps) return -1;
    return 0;
}
inline int comp(double a, double b) {
    const double delta = a - b;
    if (delta > equal_eps) return 1;
    if (delta < -equal_eps) return -1;
    return 0;
}
inline bool isInt(double v, double eps = equal_eps) {
    return comp(v, static_cast<double>(java_round(v)), eps) == 0;
}
inline bool isEnd(double v, int end) {
    const double alpha = v / getP10(end);
    const double beta = v / getP10(end - 1);
    return isInt(alpha) && !isInt(beta);
}
// Length and '.' index of Java (JDK 19+) Double.toString(v): the shortest
// decimal that rounds to v (ties: closest); when one digit suffices, the
// closest decimal of one or two digits. Rendered plain for 1e-3 <= |v| < 1e7,
// otherwise as d.dddE[-]n. Only reached for finite nonzero v.
inline int java_to_string_end(double v) {
    char buf[64];
    int precision = 0;
    for (; precision < 17; ++precision) {
        std::snprintf(buf, sizeof buf, "%.*e", precision, v);
        if (std::strtod(buf, nullptr) == v) break;
    }
    if (precision == 0) std::snprintf(buf, sizeof buf, "%.1e", v);
    std::string digits;
    const char* p = buf;
    if (*p == '-') ++p;
    for (; *p && *p != 'e'; ++p) if (*p >= '0' && *p <= '9') digits.push_back(*p);
    const int exponent = std::atoi(p + 1);
    while (digits.size() > 1 && digits.back() == '0') digits.pop_back();
    const int sign = v < 0 ? 1 : 0;
    const double a = std::abs(v);
    int length, index;
    if (a >= 1e-3 && a < 1e7) {
        const int n = static_cast<int>(digits.size());
        const int integer = exponent >= 0 ? exponent + 1 : 1;
        const int fraction = exponent >= 0 ? std::max(1, n - exponent - 1) : -exponent - 1 + n;
        index = sign + integer;
        length = index + 1 + fraction;
    } else {
        const int fraction = std::max<int>(1, static_cast<int>(digits.size()) - 1);
        const std::string e = std::to_string(exponent);
        index = sign + 1;
        length = index + 1 + fraction + 1 + static_cast<int>(e.size());
    }
    return index - (length - 1);
}
inline int getEnd_HP(double v, int last_end) {
    if (isEnd(v, last_end)) return last_end;
    return java_to_string_end(v);
}
inline int getEnd(double v, int last_end) {
    if (comp(v, 0, equal_eps) == 0) return 0;
    if (last_end < -12) return getEnd_HP(v, last_end);
    int q = last_end;
    double vq = v / getP10(q);
    if (isInt(vq, integer_eps)) {
        vq = v / getP10(q + 1);
        while (isInt(vq, integer_eps)) { ++q; vq = v / getP10(q + 1); }
        return q;
    }
    --q;
    vq = v / getP10(q);
    while (!isInt(vq, integer_eps)) { --q; vq = v / getP10(q); }
    return q;
}
inline int64_t truncate(double v) {
    if (isInt(v)) return java_round(v);
    if (v > equal_eps) return java_long(std::floor(v + integer_eps));
    if (v < -equal_eps) return java_long(std::ceil(v - integer_eps));
    return 0;
}

// ---- bit I/O (utils/StreamWriter.java semantics: MSB first, zero padding) --
class Writer {
    unsigned used_ = 0;
public:
    std::vector<uint8_t> bytes;
    // Writes the low n bits of x, like StreamWriter.write(long, n).
    void put(uint64_t x, unsigned n) {
        while (n) {
            if (!used_) bytes.push_back(0);
            const unsigned k = std::min(n, 8 - used_);
            bytes.back() |= static_cast<uint8_t>(((x >> (n-k)) & ((1u<<k)-1)) << (8-used_-k));
            used_ = (used_ + k) % 8;
            n -= k;
        }
    }
};
class Reader {
    const std::vector<uint8_t>& bytes_;
    size_t byte_ = 0;
    unsigned used_ = 0;
public:
    explicit Reader(const std::vector<uint8_t>& bytes) : bytes_(bytes) {}
    uint64_t get(unsigned n) {
        uint64_t x = 0;
        while (n) {
            if (byte_ >= bytes_.size()) throw std::runtime_error("Truncated DeXOR stream");
            const unsigned k = std::min(n, 8 - used_);
            x = (x << k) | ((bytes_[byte_] >> (8-used_-k)) & ((1u<<k)-1));
            used_ += k;
            if (used_ == 8) { ++byte_; used_ = 0; }
            n -= k;
        }
        return x;
    }
    void finish() const {
        if (!used_) {
            if (byte_ != bytes_.size()) throw std::runtime_error("Trailing DeXOR data");
        } else if (byte_ + 1 != bytes_.size() ||
                   (bytes_[byte_] & ((1u << (8-used_))-1))) {
            throw std::runtime_error("Invalid DeXOR padding");
        }
    }
};

constexpr int rho = 8;
constexpr int P2[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};

// ---- DoubleDeXOREncoder.java (Native) -------------------------------------
struct Encoder {
    Writer out;
    double previous_value = 0;
    int previous_q = 0, previous_delta = 0;
    int64_t previous_exp = 1023;
    int EL = 1, contract_step = 0;

    void ExceptionHandle(double v) {
        const uint64_t lv = bits(v);
        const int64_t exp = static_cast<int64_t>((lv >> 52) & 0x7ff);
        const int64_t delta = exp - previous_exp;
        const int bias = P2[EL - 1] - 1;
        if (delta >= -bias && delta <= bias) {
            out.put(static_cast<uint64_t>(delta + bias), EL);
            out.put(lv >> 63, 1);
            out.put(lv, 52);
            if (EL > 1) {
                const int su_bias = P2[EL - 2] - 1;
                contract_step = (delta >= -su_bias && delta <= su_bias) ? contract_step + 1 : 0;
                if (contract_step == rho) { --EL; contract_step = 0; }
            }
        } else {
            out.put(static_cast<uint64_t>(P2[EL] - 1), EL);
            out.put(lv, 64);
            contract_step = 0;
            if (EL < 10) ++EL;
        }
        previous_exp = exp;
    }
    // Returns false where upstream would throw before writing anything.
    bool Decimal_XOR(double v) {
        int q, delta = 0;
        double alpha = 0, pow;
        int64_t beta;
        try {
            q = getEnd(v, previous_q);
            while (delta < 16) {
                const double p = getP10(q + delta);
                const int64_t a = truncate(v / p);
                const int64_t b = truncate(previous_value / p);
                if (a == b) { alpha = static_cast<double>(a) * p; break; }
                ++delta;
            }
            pow = getP10(q);
        } catch (const OutOfRange&) {
            return false;
        }
        const double residual = v - alpha;
        beta = java_round(residual / pow);
        if (delta >= 16 || comp(alpha + static_cast<double>(beta) * pow, v, pow) != 0) {
            out.put(3, 2); // Exception 11
            ExceptionHandle(v);
            return true;
        }
        // Upstream writes q+20 into a 5-bit field unchecked; a q outside [-20,11]
        // wraps and desynchronizes its own decoder (the rest of the block is
        // undecodable). Only such values use the exception encoding instead.
        if (q < -20 || q > 11) {
            out.put(3, 2);
            ExceptionHandle(v);
            return true;
        }
        // Math.abs; like upstream, only the low cost[delta] bits are written.
        const uint64_t magnitude = beta < 0 ? 0 - static_cast<uint64_t>(beta) : static_cast<uint64_t>(beta);
        beta = static_cast<int64_t>(magnitude);
        const bool flag = q == previous_q;
        if (flag && delta == previous_delta) {
            out.put(2, 2);
        } else {
            out.put(flag ? 1 : 0, 2);
            if (!flag) { out.put(static_cast<uint64_t>(q + 20), 5); previous_q = q; }
            out.put(static_cast<uint64_t>(delta), 4);
            previous_delta = delta;
        }
        if (comp(alpha, 0) == 0) out.put(v > 0 ? 1 : 0, 1);
        out.put(static_cast<uint64_t>(beta), cost[delta]);
        previous_value = v;
        return true;
    }
    void encode(double v) {
        if (!Decimal_XOR(v)) { out.put(3, 2); ExceptionHandle(v); }
    }
};

// ---- DoubleDeXORDecoder.java (Native) -------------------------------------
struct Decoder {
    Reader& in;
    double previous_value = 0, previous_alpha = 0;
    int previous_q = 0, previous_delta = 0;
    int64_t previous_exp = 1023;
    int EL = 1, contract_step = 0;
    explicit Decoder(Reader& r) : in(r) {}

    double ExceptionDecode() {
        const int bias = P2[EL - 1] - 1;
        const int64_t delta = static_cast<int64_t>(in.get(EL)) - bias;
        uint64_t lv;
        if (delta >= -bias && delta <= bias) {
            previous_exp += delta;
            if (previous_exp < 0 || previous_exp > 2047) throw std::runtime_error("Invalid DeXOR exponent");
            lv = (in.get(1) << 63) | (static_cast<uint64_t>(previous_exp) << 52) | in.get(52);
            if (EL > 1) {
                const int su_bias = P2[EL - 2] - 1;
                contract_step = (delta >= -su_bias && delta <= su_bias) ? contract_step + 1 : 0;
                if (contract_step == rho) { --EL; contract_step = 0; }
            }
        } else {
            lv = in.get(64);
            previous_exp = static_cast<int64_t>((lv >> 52) & 0x7ff);
            // Upstream resets contract_step only while EL < 10, but its encoder
            // always resets it; resetting here keeps the two state machines equal.
            contract_step = 0;
            if (EL < 10) ++EL;
        }
        return value(lv);
    }
    double decodeDouble() {
        const unsigned con = static_cast<unsigned>(in.get(2));
        if (con == 3) return ExceptionDecode();
        if (con == 0 || con == 1) {
            if (con == 0) previous_q = static_cast<int>(in.get(5)) - 20;
            previous_delta = static_cast<int>(in.get(4));
            double pow;
            try { pow = getP10(previous_q + previous_delta); }
            catch (const OutOfRange&) { throw std::runtime_error("Invalid DeXOR coordinates"); }
            previous_alpha = static_cast<double>(truncate(previous_value / pow)) * pow;
        }
        int64_t sign = previous_alpha > 0 ? 1 : -1;
        if (comp(previous_alpha, 0) == 0) sign = in.get(1) ? 1 : -1;
        const int64_t beta_star = sign * static_cast<int64_t>(in.get(cost[previous_delta]));
        double p;
        try { p = getP10(previous_q); }
        catch (const OutOfRange&) { throw std::runtime_error("Invalid DeXOR coordinates"); }
        previous_value = previous_alpha + static_cast<double>(beta_star) * p;
        return previous_value;
    }
};
} // namespace detail

// The first eight bytes store the value count in big-endian order. Everything
// after that is the default DeXOR bitstream (rho=8, buffer_bits=0, no skip).
inline std::vector<uint8_t> compress(const std::vector<double>& input) {
    detail::Encoder encoder;
    encoder.out.put(input.size(), 64);
    for (double x : input) encoder.encode(x);
    return std::move(encoder.out.bytes);
}
// Decode only the requested prefix, without reading or reconstructing the tail.
// A proper prefix validates its consumed portion, not the remaining framing.
inline std::vector<double> decompress_prefix(const std::vector<uint8_t>& input,
                                             size_t total_count, size_t prefix_count) {
    if (prefix_count > total_count) throw std::invalid_argument("DeXOR prefix exceeds count");
    detail::Reader in(input);
    if (in.get(64) != total_count) throw std::runtime_error("DeXOR count mismatch");
    // Every encoded value consumes at least two control bits.
    if (input.size() < 8 || prefix_count / 4 > input.size()-8)
        throw std::runtime_error("Invalid DeXOR count");
    std::vector<double> output;
    output.reserve(prefix_count);
    detail::Decoder decoder(in);
    for (size_t i = 0; i < prefix_count; ++i) output.push_back(decoder.decodeDouble());
    if (prefix_count == total_count) in.finish();
    return output;
}
inline std::vector<double> decompress(const std::vector<uint8_t>& input, size_t count) {
    return decompress_prefix(input, count, count);
}
}} // namespace gef_compressors::dexor
