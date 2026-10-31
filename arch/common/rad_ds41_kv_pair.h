/* Default-off adjacent-value sharing for the existing packed-fast BF16 reconstruction.
 * No format, arithmetic or alignment change. Ordinary/index/QK/WMMA paths do not use this.
 * The caller retains its null/kDsZeroKey guard before passing a physical packed row. */
#pragma once
#include "rad_ds41_kv_fast.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define RAD_DS41_KVP_INLINE __host__ __device__ __forceinline__
#else
#define RAD_DS41_KVP_INLINE inline
#endif

namespace rad_ds41_kv_pair {
using Kind = rad_ds41_kv_codec::Kind;
namespace fast = rad_ds41_kv_fast;

/* Strict, read-once host admission. Kept pure so the startup/dispatch contract is testable. */
inline int parse_mode(const char* value) {
    if (!value || (value[0] == '0' && value[1] == 0)) return 0;
    return value[0] == '1' && value[1] == 0 ? 1 : -1;
}
inline bool dependencies(int mode, bool packed, bool packed_fast, bool window_packed) {
    return mode == 0 || (mode == 1 && packed && packed_fast && !window_packed);
}
inline bool normal_candidate(int mode, uint32_t kind, bool packed_fast, int64_t M, int64_t H) {
    return mode == 1 && kind == 1 && packed_fast && M == 1 && H == 16;
}

/* Checked counterpart for byte oracles. Reject before writing on invalid geometry/metadata.
 * Odd requests (including a group crossing) retain the old scalar decode semantics. Native
 * direct PV's even columns use one scale, payload byte and group-invariant decode preparation. */
template<Kind K, int Width>
RAD_DS41_KVP_INLINE bool decode_row2(const uint8_t* row, int col, uint32_t* out) {
    constexpr int G = K == Kind::Fp4E4M3G16 ? 16 : 32;
    static_assert((K == Kind::Fp4E4M3G16 || K == Kind::Fp4Pow2G32) && Width > 0 && Width % G == 0,
                  "packed DS41 pair geometry");
    if (!row || !out || col < 0 || col > Width - 2) return false;
    const int g = col / G;
    const uint8_t scale = row[Width / 2 + g];
    if (scale && !fast::scale_valid(K, scale)) return false;
    if (col & 1) {
        const uint8_t next_scale = row[Width / 2 + (col + 1) / G];
        if (next_scale && !fast::scale_valid(K, next_scale)) return false;
        *out = (uint32_t)fast::decode_row_bits<K, Width>(row, col) |
               ((uint32_t)fast::decode_row_bits<K, Width>(row, col + 1) << 16);
        return true;
    }
    const int local = col % G;
    const uint8_t* payload = row + g * (G / 2);
    uint16_t lo, hi;
    if (!scale) {
        const uint16_t a = fast::u16(payload), b = fast::u16(payload + 2);
        const unsigned selector = payload[4 + (local >> 3)] >> (local & 7);
        lo = selector & 1 ? b : a;
        hi = selector & 2 ? b : a;
    } else {
        const uint8_t codes = payload[local >> 1];
        if constexpr (K == Kind::Fp4E4M3G16) {
            const unsigned e = scale >> 3, m = scale & 7u;
            const unsigned coeff = e ? m + 8u : m;
            const int power = e ? (int)e - 11 : -10;
            lo = fast::e4_bits(coeff, power, codes & 15u);
            hi = fast::e4_bits(coeff, power, codes >> 4);
        } else {
            const unsigned source = scale == 254 ? 1u : scale;
            const bool ftz = scale == 254;
            lo = fast::pow2_bits(source, ftz, codes & 15u);
            hi = fast::pow2_bits(source, ftz, codes >> 4);
        }
    }
    *out = (uint32_t)lo | ((uint32_t)hi << 16);
    return true;
}
template<Kind K, int Width>
RAD_DS41_KVP_INLINE uint32_t decode_row2_bits(const uint8_t* row, int col) {
    uint32_t out;
    if (!decode_row2<K, Width>(row, col, &out)) __builtin_trap();
    return out;
}
} // namespace rad_ds41_kv_pair
#undef RAD_DS41_KVP_INLINE
