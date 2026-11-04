#pragma once
// C++ port of SuDIS-ZJU/DeXOR's Native double encoder/decoder (MIT).
// Upstream: 06128b3df57cc1e8c53681231e235e22eb508c8d. See LICENSE and docs/dexor.md.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
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
constexpr double powers[] = {
    1e-23,1e-22,1e-21,1e-20,1e-19,1e-18,1e-17,1e-16,1e-15,1e-14,
    1e-13,1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,
    1e-2,1e-1,1,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9,1e10,1e11,
    1e12,1e13,1e14,1e15,1e16,1e17,1e18,1e19,1e20,1e21,1e22,1e23};
constexpr unsigned cost[] = {0,4,7,10,14,17,20,24,27,30,34,37,40,44,47,50};
inline double p10(int q) { return powers[q + 23]; }
inline bool near_zero(double x) { return std::abs(x) <= 1e-23; }
// Java Math.round breaks ties toward positive infinity, unlike std::round.
inline double java_round(double x) {
    const double f = std::floor(x);
    return x - f >= 0.5 ? f + 1.0 : f;
}
inline bool is_int(double x, double eps) {
    return std::isfinite(x) && std::abs(x - java_round(x)) < eps;
}
inline bool truncate(double x, int64_t& result) {
    if (!std::isfinite(x)) return false;
    double t = is_int(x, 1e-23) ? java_round(x) :
        (x > 1e-23 ? std::floor(x + 1e-6) :
         x < -1e-23 ? std::ceil(x - 1e-6) : 0.0);
    if (t < -0x1p63 || t >= 0x1p63) return false;
    result = static_cast<int64_t>(t);
    return true;
}
// Bounded form of the paper's locality-aware tail-coordinate search.
inline bool tail(double x, int previous, int& q) {
    if (std::abs(x) < 1e-23) { q = 0; return true; }
    q = previous;
    if (is_int(x / p10(q), 1e-6)) {
        while (q < 11 && is_int(x / p10(q + 1), 1e-6)) ++q;
        return q != 11 || !is_int(x / p10(12), 1e-6);
    }
    while (q > -20) {
        --q;
        if (is_int(x / p10(q), 1e-6)) return true;
    }
    return false;
}

class Writer {
    unsigned used_ = 0;
public:
    std::vector<uint8_t> bytes;
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

struct State {
    double previous = 0, alpha = 0;
    int q = 0, delta = 0, exponent = 1023, width = 1, contractions = 0;
    void contract(int difference) {
        if (width <= 1) return;
        const int small_bias = (1 << (width-2))-1;
        contractions = std::abs(difference) <= small_bias ? contractions+1 : 0;
        if (contractions == 8) { --width; contractions = 0; }
    }
    void exception(Writer& out, uint64_t raw) {
        const int next_exponent = static_cast<int>((raw >> 52) & 2047);
        const int difference = next_exponent - exponent;
        const int bias = (1 << (width-1))-1;
        if (std::abs(difference) <= bias) {
            out.put(static_cast<uint64_t>(difference+bias), static_cast<unsigned>(width));
            out.put(raw >> 63, 1);
            out.put(raw, 52);
            contract(difference);
        } else {
            out.put((1u << width)-1, static_cast<unsigned>(width));
            out.put(raw, 64);
            contractions = 0;
            if (width < 10) ++width; // official implementation's cap
        }
        exponent = next_exponent;
    }
    double exception(Reader& in) {
        const int bias = (1 << (width-1))-1;
        const int difference = static_cast<int>(in.get(static_cast<unsigned>(width)))-bias;
        uint64_t raw;
        if (std::abs(difference) <= bias) {
            exponent += difference;
            if (exponent < 0 || exponent > 2047) throw std::runtime_error("Invalid DeXOR exponent");
            const uint64_t sign = in.get(1);
            raw = (sign << 63) | (static_cast<uint64_t>(exponent) << 52) | in.get(52);
            contract(difference);
        } else {
            raw = in.get(64);
            exponent = static_cast<int>((raw >> 52) & 2047);
            contractions = 0;
            if (width < 10) ++width;
        }
        return value(raw);
    }
    bool decimal(Writer& out, double x) {
        if (!std::isfinite(x)) return false;
        int next_q;
        if (!tail(x, q, next_q)) return false;
        int next_delta = 0;
        double prefix = 0;
        for (; next_delta < 16; ++next_delta) {
            if (next_q + next_delta > 23) return false;
            const double power = p10(next_q + next_delta);
            int64_t a, b;
            if (!truncate(x / power, a) || !truncate(previous / power, b)) return false;
            if (a == b) { prefix = static_cast<double>(a) * power; break; }
        }
        if (next_delta == 16) return false;
        const double power = p10(next_q);
        const double rounded = java_round((x-prefix) / power);
        if (!std::isfinite(rounded) || std::abs(rounded) >= 0x1p63) return false;
        const uint64_t suffix = static_cast<uint64_t>(std::abs(rounded));
        if (suffix >= (uint64_t(1) << cost[next_delta])) return false;
        const bool reuse = q == next_q && delta == next_delta;
        // Match decoder arithmetic exactly, including cached-prefix reuse.
        const double decoded_prefix = reuse ? alpha : prefix;
        const int sign = near_zero(decoded_prefix) ? (x > 0 ? 1 : -1) :
                         (decoded_prefix > 0 ? 1 : -1);
        const double decoded_suffix = static_cast<double>(suffix) * sign * power;
        const double reconstructed = decoded_prefix + decoded_suffix;
        if (bits(reconstructed) != bits(x)) return false;
        if (reuse) out.put(2, 2);
        else {
            out.put(next_q == q ? 1 : 0, 2);
            if (next_q != q) out.put(static_cast<unsigned>(next_q+20), 5);
            out.put(static_cast<unsigned>(next_delta), 4);
            alpha = prefix;
            q = next_q;
            delta = next_delta;
        }
        if (near_zero(decoded_prefix)) out.put(x > 0, 1);
        out.put(suffix, cost[next_delta]);
        previous = x;
        return true;
    }
    double decimal(Reader& in, unsigned control) {
        if (control != 2) {
            if (control == 0) q = static_cast<int>(in.get(5))-20;
            delta = static_cast<int>(in.get(4));
            if (q + delta > 23) throw std::runtime_error("Invalid DeXOR coordinates");
            const double power = p10(q+delta);
            int64_t a;
            if (!truncate(previous / power, a)) throw std::runtime_error("Invalid DeXOR prefix");
            alpha = static_cast<double>(a) * power;
        }
        const int sign = near_zero(alpha) ? (in.get(1) ? 1 : -1) : (alpha > 0 ? 1 : -1);
        const double suffix = static_cast<double>(in.get(cost[delta])) * sign * p10(q);
        previous = alpha + suffix;
        return previous;
    }
};
} // namespace detail

// The first eight bytes store the value count in big-endian order. Everything
// after that is the default DeXOR bitstream (rho=8, buffer_bits=0, no skip).
inline std::vector<uint8_t> compress(const std::vector<double>& input) {
    detail::Writer out;
    out.put(input.size(), 64);
    detail::State state;
    for (double x : input) {
        if (!state.decimal(out, x)) {
            out.put(3, 2);
            state.exception(out, detail::bits(x));
        }
    }
    return std::move(out.bytes);
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
    detail::State state;
    for (size_t i = 0; i < prefix_count; ++i) {
        const auto control = static_cast<unsigned>(in.get(2));
        output.push_back(control == 3 ? state.exception(in) : state.decimal(in, control));
    }
    if (prefix_count == total_count) in.finish();
    return output;
}
inline std::vector<double> decompress(const std::vector<uint8_t>& input, size_t count) {
    return decompress_prefix(input, count, count);
}
}} // namespace gef_compressors::dexor
