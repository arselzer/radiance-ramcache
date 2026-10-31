/* DS41's exact packed producer-output domain. No quantizer or engine ABI assumptions.
 *
 * Planar row: [width/2 nibble-payload bytes][width/G scale bytes], no row padding. Element 2j
 * uses the low nibble, 2j+1 the high. Each group's payload is G/2 bytes. Normal positive scales
 * use the ORIGINAL producer codes. Scale 0 selects a literal two-pattern group: uint16 patterns
 * 0 and 1 in little endian at bytes 0..3, then G selector bits (low bit first) at byte 4. The
 * remaining payload bytes are zero. A zero-filled physical row therefore decodes to BF16 +0.
 *
 * Pow2 scale bytes are NOT UE8M0. The producer's raw exponent 255 means +Inf and must be encoded
 * as a captured literal pair; never recreate NaN payloads on the host. Stored byte 254 instead
 * denotes source scale 1 (2^-126) with its q=+/-0.5 product flushed to signed zero. A finite BF16
 * producer cannot choose exponent 254: amax*f32(1/6) < 2^126, hence ceil_log2 <=126, bias <=253.
 *
 * Finite products have at most eight significant bits. Integer decoding preserves their BF16
 * signs, subnormals and overflow. pack_captured_group verifies EVERY decoded uint16 against the
 * actual old producer output, tries the explicit FTZ tag, then the literal pair. Raw means the
 * caller must retain the original BF16 bytes; no output is written and nothing is requantized.
 * An active packed cache must first prove that its producer domain never returns Raw.
 */
#pragma once
#include <cstdint>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define RAD_DS41_KV_HD __host__ __device__
#else
#define RAD_DS41_KV_HD
#endif

namespace rad_ds41_kv_codec {

enum class Kind : uint32_t { Plain = 0, Fp4E4M3G16 = 1, Fp4Pow2G32 = 2 };
using Format = Kind;
enum class PackResult : uint8_t { Exact = 0, Raw = 1 };
constexpr uint8_t kPairScale = 0, kPow2HalfFtzScale = 254;

RAD_DS41_KV_HD constexpr int group_size(Kind k) {
    return k == Kind::Fp4E4M3G16 ? 16 : k == Kind::Fp4Pow2G32 ? 32 : 0;
}
RAD_DS41_KV_HD constexpr int group_size(uint32_t k) { return group_size((Kind)k); }
RAD_DS41_KV_HD constexpr bool valid_width(Kind k, int64_t n) {
    return group_size(k) > 0 && n > 0 && n % group_size(k) == 0;
}
RAD_DS41_KV_HD constexpr bool valid_width(uint32_t k, int64_t n) { return valid_width((Kind)k, n); }
RAD_DS41_KV_HD constexpr int64_t row_used_bytes(Kind k, int64_t n) {
    return valid_width(k, n) ? n / 2 + n / group_size(k) : 0;
}
RAD_DS41_KV_HD constexpr int64_t row_used_bytes(uint32_t k, int64_t n) { return row_used_bytes((Kind)k, n); }
RAD_DS41_KV_HD constexpr int64_t row_bytes(Kind k, int64_t n) { return row_used_bytes(k, n); }
RAD_DS41_KV_HD constexpr int64_t row_bytes(uint32_t k, int64_t n) { return row_bytes((Kind)k, n); }
RAD_DS41_KV_HD constexpr int64_t payload_offset(Kind k, int64_t group) { return group * (group_size(k) / 2); }
RAD_DS41_KV_HD constexpr int64_t payload_offset(uint32_t k, int64_t g) { return payload_offset((Kind)k, g); }
RAD_DS41_KV_HD constexpr int64_t scale_offset(Kind, int64_t n, int64_t group) { return n / 2 + group; }
RAD_DS41_KV_HD constexpr int64_t scale_offset(uint32_t k, int64_t n, int64_t g) { return scale_offset((Kind)k, n, g); }

RAD_DS41_KV_HD inline uint16_t load_u16_le(const uint8_t* p) { return (uint16_t)(p[0] | (uint16_t)p[1] << 8); }
RAD_DS41_KV_HD inline void store_u16_le(uint8_t* p, uint16_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }
RAD_DS41_KV_HD inline uint8_t nibble(const uint8_t* p, int i) { return (uint8_t)((p[i / 2] >> (4 * (i & 1))) & 15); }

/* The old ds_e2m1_rt result, as a bit pattern. Code 8 is -0, not a duplicate +0. */
RAD_DS41_KV_HD inline uint32_t fp4_fp32_bits(uint8_t code) {
    const uint32_t mag[] = { 0, 0x3F000000u, 0x3F800000u, 0x3FC00000u,
                             0x40000000u, 0x40400000u, 0x40800000u, 0x40C00000u };
    return mag[code & 7] | ((uint32_t)(code & 8) << 28);
}
RAD_DS41_KV_HD inline bool code_from_fp32_bits(uint32_t q, uint8_t* out) {
    if (!out) return false;
    for (uint8_t c = 0; c < 16; ++c) if (fp4_fp32_bits(c) == q) { *out = c; return true; }
    return false;
}

/* Raw ORIGINAL scale semantics, including index 255=+Inf. This is not stored-scale decoding:
 * source index 254 is an ordinary power of two but outside the closed finite producer domain. */
RAD_DS41_KV_HD inline bool source_scale_fp32_bits(Kind k, uint8_t scale, uint32_t* out) {
    if (!out) return false;
    if (k == Kind::Fp4Pow2G32) { *out = (uint32_t)scale << 23; return true; }
    if (k != Kind::Fp4E4M3G16 || scale > 126) return false;
    const uint32_t e = scale >> 3, m = scale & 7;
    if (e) { *out = ((e + 120) << 23) | (m << 20); return true; }
    if (!m) { *out = 0; return true; }
    int top = 0;
    for (uint32_t x = m; x > 1; x >>= 1) ++top;
    *out = ((uint32_t)(118 + top) << 23) | ((m << (23 - top)) & 0x7FFFFFu);
    return true;
}
RAD_DS41_KV_HD inline bool source_scale_fp32_bits(uint32_t k, uint8_t s, uint32_t* out) {
    return source_scale_fp32_bits((Kind)k, s, out);
}

/* n*2^power, n<=180. All finite normal results have <=8 significant bits. The subnormal path
 * uses round-to-nearest-even explicitly; preserving the zero sign never requires floating point. */
RAD_DS41_KV_HD inline uint16_t dyadic_bf16(unsigned n, int power, bool negative) {
    const uint16_t sign = negative ? 0x8000 : 0;
    if (!n) return sign;
    int top = 0;
    for (unsigned x = n; x > 1; x >>= 1) ++top;
    const int exponent = top + power;
    if (exponent > 127) return (uint16_t)(sign | 0x7F80);
    if (exponent >= -126)
        return (uint16_t)(sign | ((exponent + 127) << 7) | ((n << (7 - top)) & 127));
    const int shift = power + 133;
    unsigned bits = 0;
    if (shift >= 0) bits = n << shift;
    else if (-shift < 32) {
        const unsigned right = (unsigned)-shift, half = 1u << (right - 1);
        bits = n >> right;
        const unsigned rem = n & ((1u << right) - 1);
        if (rem > half || (rem == half && (bits & 1))) ++bits;
    }
    return (uint16_t)(sign | bits);
}
RAD_DS41_KV_HD constexpr bool normal_scale_valid(Kind k, uint8_t s) {
    return s > 0 && (k == Kind::Fp4E4M3G16 ? s <= 126 : k == Kind::Fp4Pow2G32 && s <= 254);
}
RAD_DS41_KV_HD inline bool decode_normal_bits(Kind k, uint8_t scale, uint8_t code, uint16_t* out) {
    if (!out || code > 15 || !normal_scale_valid(k, scale)) return false;
    const unsigned twice[] = { 0, 1, 2, 3, 4, 6, 8, 12 };
    const bool sign = (code & 8) != 0;
    unsigned n = twice[code & 7];
    if (k == Kind::Fp4E4M3G16) {
        const unsigned e = scale >> 3, m = scale & 7;
        *out = dyadic_bf16(n * (e ? m + 8 : m), e ? (int)e - 11 : -10, sign);
    } else if (scale == kPow2HalfFtzScale) {
        *out = dyadic_bf16((code & 7) == 1 ? 0 : n, -127, sign);
    } else *out = dyadic_bf16(n, (int)scale - 128, sign);
    return true;
}
RAD_DS41_KV_HD inline bool decode_group_value(Kind k, uint8_t scale, const uint8_t* payload,
                                              int i, uint16_t* out) {
    if (!payload || !out || i < 0 || i >= group_size(k)) return false;
    if (scale == kPairScale) {
        *out = load_u16_le(payload + 2 * ((payload[4 + i / 8] >> (i & 7)) & 1));
        return true;
    }
    return decode_normal_bits(k, scale, nibble(payload, i), out);
}
RAD_DS41_KV_HD inline bool decode_group_value(uint32_t k, uint8_t s, const uint8_t* p, int i, uint16_t* out) {
    return decode_group_value((Kind)k, s, p, i, out);
}
RAD_DS41_KV_HD inline bool decode_row(Kind k, const uint8_t* row, int64_t width, uint16_t* out) {
    if (!row || !out || !valid_width(k, width)) return false;
    const int G = group_size(k);
    for (int64_t g = 0; g < width / G; ++g) {
        const uint8_t s = row[scale_offset(k, width, g)];
        if (s && !normal_scale_valid(k, s)) return false; // refuse before writing any output
    }
    for (int64_t i = 0; i < width; ++i)
        if (!decode_group_value(k, row[scale_offset(k, width, i / G)],
                                row + payload_offset(k, i / G), (int)(i % G), out + i)) return false;
    return true;
}
RAD_DS41_KV_HD inline bool decode_row(uint32_t k, const uint8_t* row, int64_t n, uint16_t* out) {
    return decode_row((Kind)k, row, n, out);
}
/* Fast scalar reader for an already-validated page contract. Invalid metadata traps instead of
 * silently manufacturing a value. The checked APIs above are available to validators/oracles. */
RAD_DS41_KV_HD inline uint16_t decode_row_bits(Kind k, const uint8_t* row, int64_t width, int64_t col) {
    uint16_t out = 0;
    const int G = group_size(k);
    if (!row || !valid_width(k, width) || col < 0 || col >= width ||
        !decode_group_value(k, row[scale_offset(k, width, col / G)], row + payload_offset(k, col / G),
                            (int)(col % G), &out)) __builtin_trap();
    return out;
}
RAD_DS41_KV_HD inline uint16_t decode_row_bits(uint32_t k, const uint8_t* row, int64_t n, int64_t col) {
    return decode_row_bits((Kind)k, row, n, col);
}

RAD_DS41_KV_HD inline PackResult pack_captured_group(Kind k, uint8_t source_scale, const uint8_t* codes,
                                                    const uint16_t* current, uint8_t* payload,
                                                    uint8_t* stored_scale) {
    const int G = group_size(k);
    if (!G || !current || !payload || !stored_scale) return PackResult::Raw;
    // Source index 254 is NOT the FTZ descriptor. Only source 1 can select the stored 254 tag.
    const int attempts = k == Kind::Fp4Pow2G32 && source_scale == 1 ? 2 : 1;
    for (int attempt = 0; codes && attempt < attempts; ++attempt) {
        const uint8_t s = attempt ? kPow2HalfFtzScale : source_scale;
        bool exact = !(k == Kind::Fp4Pow2G32 && source_scale >= 254);
        for (int i = 0; i < G && exact; ++i) {
            uint16_t decoded = 0;
            exact = decode_normal_bits(k, s, codes[i], &decoded) && decoded == current[i];
        }
        if (exact) {
            for (int i = 0; i < G / 2; ++i) payload[i] = (uint8_t)(codes[2 * i] | (codes[2 * i + 1] << 4));
            *stored_scale = s;
            return PackResult::Exact;
        }
    }
    uint16_t a = current[0], b = a;
    bool second = false;
    for (int i = 1; i < G; ++i) if (current[i] != a) {
        if (!second) { b = current[i]; second = true; }
        else if (current[i] != b) return PackResult::Raw;
    }
    for (int i = 0; i < G / 2; ++i) payload[i] = 0;
    store_u16_le(payload, a); store_u16_le(payload + 2, b);
    for (int i = 0; i < G; ++i) if (current[i] != a) payload[4 + i / 8] |= (uint8_t)(1u << (i & 7));
    *stored_scale = kPairScale;
    return PackResult::Exact;
}
RAD_DS41_KV_HD inline PackResult pack_captured_group(uint32_t k, uint8_t s, const uint8_t* codes,
                                                    const uint16_t* current, uint8_t* p, uint8_t* out_s) {
    return pack_captured_group((Kind)k, s, codes, current, p, out_s);
}

} // namespace rad_ds41_kv_codec
namespace rad { namespace ds41 { namespace kvcodec = ::rad_ds41_kv_codec; } }
#undef RAD_DS41_KV_HD
