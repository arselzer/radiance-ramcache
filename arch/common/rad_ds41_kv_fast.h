/* Optional integer-only reader for the EXACT stored DS41 producer-output codec.
 * rad_ds41_kv_codec.h remains the independent implementation and default reader.
 * No floating point operation, BF16 requantization, NaN recreation or changed page layout.
 *
 * E4M3 scales1..126 times FP4 magnitudes are all BF16-normal and have <=8 significant bits.
 * A single nonzero CLZ therefore supplies the exact exponent/significand without a loop.
 * Pow2 products need no CLZ: FP4 magnitudes2..7 advance by64 in BF16's positive encoding.
 * Only magnitude1 at source scale1 is subnormal (0x0040); stored254 explicitly flushes that
 * product to signed zero. Values past BF16's finite exponent saturate to signed infinity.
 * Scale0 always returns the literal uint16 pattern, including signaling NaNs, without FP math.
 */
#pragma once
#include "rad_ds41_kv_codec.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define RAD_DS41_KVF_INLINE __host__ __device__ __forceinline__
#define RAD_DS41_KVF_UNROLL _Pragma("unroll")
#else
#define RAD_DS41_KVF_INLINE inline
#define RAD_DS41_KVF_UNROLL
#endif

namespace rad_ds41_kv_fast {
using Kind = rad_ds41_kv_codec::Kind;
static_assert(sizeof(unsigned) == 4, "DS41 integer reader expects 32-bit unsigned CLZ");

RAD_DS41_KVF_INLINE constexpr int group_size(Kind k) {
    return k == Kind::Fp4E4M3G16 ? 16 : k == Kind::Fp4Pow2G32 ? 32 : 0;
}
RAD_DS41_KVF_INLINE constexpr bool scale_valid(Kind k, uint8_t s) {
    return s && (k == Kind::Fp4E4M3G16 ? s <= 126 : k == Kind::Fp4Pow2G32 && s <= 254);
}
RAD_DS41_KVF_INLINE uint16_t u16(const uint8_t* p) {
    return (uint16_t)((uint16_t)p[0] | (uint16_t)p[1] << 8);
}
RAD_DS41_KVF_INLINE uint8_t code_at(const uint8_t* p, int i) {
    return (uint8_t)((p[i >> 1] >> (4 * (i & 1))) & 15);
}
RAD_DS41_KVF_INLINE unsigned twice_fp4(unsigned m) {
    return m < 2 ? m : (2u + (m & 1u)) << ((m >> 1) - 1);
}
/* coeff/power are invariant across a normal E4 group, so decode8 prepares them once. */
RAD_DS41_KVF_INLINE uint16_t e4_bits(unsigned coeff, int power, uint8_t code) {
    const unsigned n = twice_fp4(code & 7u) * coeff;
    const uint16_t sign = (uint16_t)((code & 8u) << 12);
    if (!n) return sign;
    const unsigned top = 31u - (unsigned)__builtin_clz(n); // n<=180, top<=7; never CLZ(0)
    const unsigned exponent = (unsigned)((int)top + power + 127);
    return (uint16_t)(sign | (exponent << 7) | ((n << (7u - top)) & 127u));
}
RAD_DS41_KVF_INLINE uint16_t pow2_bits(unsigned source, bool half_ftz, uint8_t code) {
    const unsigned m = code & 7u;
    const uint16_t sign = (uint16_t)((code & 8u) << 12);
    if (!m || (half_ftz && m == 1)) return sign;
    if (source == 1 && m == 1) return (uint16_t)(sign | 0x0040u);
    const int offset = m == 1 ? -128 : ((int)m - 2) * 64;
    const unsigned bits = (unsigned)((int)(source << 7) + offset);
    return (uint16_t)(sign | (bits >= 0x7F80u ? 0x7F80u : bits));
}
template<Kind K>
RAD_DS41_KVF_INLINE uint16_t normal_bits(uint8_t scale, uint8_t code) {
    if constexpr (K == Kind::Fp4E4M3G16) {
        const unsigned e = scale >> 3, m = scale & 7u;
        return e4_bits(e ? m + 8u : m, e ? (int)e - 11 : -10, code);
    } else {
        return pow2_bits(scale == 254 ? 1u : scale, scale == 254, code);
    }
}
RAD_DS41_KVF_INLINE bool decode_normal_bits(Kind k, uint8_t scale, uint8_t code, uint16_t* out) {
    if (!out || code > 15 || !scale_valid(k, scale)) return false;
    *out = k == Kind::Fp4E4M3G16 ? normal_bits<Kind::Fp4E4M3G16>(scale, code)
                                : normal_bits<Kind::Fp4Pow2G32>(scale, code);
    return true;
}
template<Kind K>
RAD_DS41_KVF_INLINE uint16_t group_bits(uint8_t scale, const uint8_t* payload, int i) {
    if (!scale) return u16(payload + 2 * ((payload[4 + (i >> 3)] >> (i & 7)) & 1));
    return normal_bits<K>(scale, code_at(payload, i));
}
RAD_DS41_KVF_INLINE bool decode_group_value(Kind k, uint8_t scale, const uint8_t* payload,
                                           int i, uint16_t* out) {
    const int G = rad_ds41_kv_fast::group_size(k);
    if (!payload || !out || i < 0 || i >= G || (scale && !scale_valid(k, scale))) return false;
    *out = k == Kind::Fp4E4M3G16 ? group_bits<Kind::Fp4E4M3G16>(scale, payload, i)
                                : group_bits<Kind::Fp4Pow2G32>(scale, payload, i);
    return true;
}
RAD_DS41_KVF_INLINE bool valid_width(Kind k, int64_t n) {
    const int G = rad_ds41_kv_fast::group_size(k); return G && n > 0 && n % G == 0;
}
RAD_DS41_KVF_INLINE bool decode_row(Kind k, const uint8_t* row, int64_t n, uint16_t* out) {
    if (!row || !out || !rad_ds41_kv_fast::valid_width(k, n)) return false;
    const int G = rad_ds41_kv_fast::group_size(k);
    // Match reference's all-metadata-first contract: no partial output on invalid late scale.
    for (int64_t g = 0; g < n / G; ++g)
        if (row[n / 2 + g] && !scale_valid(k, row[n / 2 + g])) return false;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t g = i / G;
        const uint8_t* p = row + g * (G / 2);
        const uint8_t s = row[n / 2 + g];
        out[i] = k == Kind::Fp4E4M3G16 ? group_bits<Kind::Fp4E4M3G16>(s, p, (int)(i % G))
                                      : group_bits<Kind::Fp4Pow2G32>(s, p, (int)(i % G));
    }
    return true;
}
/* Width/format are compile-time native reader geometry. Bounds/metadata still fail closed. */
template<Kind K, int Width>
RAD_DS41_KVF_INLINE uint16_t decode_row_bits(const uint8_t* row, int col) {
    constexpr int G = K == Kind::Fp4E4M3G16 ? 16 : 32;
    static_assert((K == Kind::Fp4E4M3G16 || K == Kind::Fp4Pow2G32) && Width > 0 && Width % G == 0,
                  "packed DS41 reader geometry");
    if (!row || col < 0 || col >= Width) __builtin_trap();
    const int g = col / G;
    const uint8_t scale = row[Width / 2 + g];
    if (scale && !scale_valid(K, scale)) __builtin_trap();
    return group_bits<K>(scale, row + g * (G / 2), col % G);
}
/* Eight adjacent values share a scale in every current vector-load site. Generic crossings retain
 * exact scalar semantics. No aligned loads: index row pitch68 need only byte alignment. */
template<Kind K, int Width>
RAD_DS41_KVF_INLINE void decode_row8_bits(const uint8_t* row, int col, uint16_t* out) {
    constexpr int G = K == Kind::Fp4E4M3G16 ? 16 : 32;
    static_assert((K == Kind::Fp4E4M3G16 || K == Kind::Fp4Pow2G32) && Width > 0 && Width % G == 0,
                  "packed DS41 reader geometry");
    if (!row || !out || col < 0 || col > Width - 8) __builtin_trap();
    const int local = col % G;
    if (local + 8 > G) {
        RAD_DS41_KVF_UNROLL
        for (int i = 0; i < 8; ++i) out[i] = decode_row_bits<K, Width>(row, col + i);
        return;
    }
    const int g = col / G;
    const uint8_t* payload = row + g * (G / 2);
    const uint8_t scale = row[Width / 2 + g];
    if (scale && !scale_valid(K, scale)) __builtin_trap();
    if (!scale) {
        const uint16_t a = u16(payload), b = u16(payload + 2);
        if (!(local & 7)) {
            const uint8_t mask = payload[4 + (local >> 3)];
            RAD_DS41_KVF_UNROLL
            for (int i = 0; i < 8; ++i) out[i] = (mask >> i) & 1 ? b : a;
        } else {
            RAD_DS41_KVF_UNROLL
            for (int i = 0; i < 8; ++i)
                out[i] = (payload[4 + ((local + i) >> 3)] >> ((local + i) & 7)) & 1 ? b : a;
        }
    } else if constexpr (K == Kind::Fp4E4M3G16) {
        const unsigned e = scale >> 3, m = scale & 7u;
        const unsigned coeff = e ? m + 8u : m;
        const int power = e ? (int)e - 11 : -10;
        RAD_DS41_KVF_UNROLL
        for (int i = 0; i < 8; ++i) out[i] = e4_bits(coeff, power, code_at(payload, local + i));
    } else {
        const unsigned source = scale == 254 ? 1u : scale;
        const bool ftz = scale == 254;
        RAD_DS41_KVF_UNROLL
        for (int i = 0; i < 8; ++i) out[i] = pow2_bits(source, ftz, code_at(payload, local + i));
    }
}
} // namespace rad_ds41_kv_fast
#undef RAD_DS41_KVF_INLINE
#undef RAD_DS41_KVF_UNROLL
