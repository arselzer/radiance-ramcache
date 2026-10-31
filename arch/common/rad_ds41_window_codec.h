/* Standalone candidate exact FP8 window codec. NO engine/page enable is implied.
 * Capture ORIGINAL ds_to_e4m3/clamp codes and UE exponent, never requantize BF16.
 * Native block_scale(kFp8) floors amax at1e-4: finite scales are105..247.
 * Thus nonzero finite products are normal BF16/F32 (min2^-31); no FTZ tag.
 * Scale0 is an inline literal BF16 pair+32 selectors; zeroed pages decode+0.
 * EVERY pack compares actual uint16 producer output. Raw is explicit and leaves
 * output untouched. Active use needs a card proof of zeroRaw, especially NaNs.
 */
#pragma once
#include <cstdint>
#if defined(__HIPCC__) || defined(__CUDACC__)
#define RAD_DS41_W_HD __host__ __device__
#else
#define RAD_DS41_W_HD
#endif
namespace rad_ds41_window_codec {
// Reserved standalone candidate only; existing core format admission still rejects3.
enum class Kind : uint32_t { Fp8Pow2G32 = 3 };
enum class PackResult : uint8_t { Exact = 0, Raw = 1 };
constexpr int kGroup = 32;
constexpr uint8_t kPairScale = 0, kMinFiniteScale = 105, kMaxFiniteScale = 247;
RAD_DS41_W_HD constexpr bool valid_width(int64_t n) { return n > 0 && n % kGroup == 0; }
RAD_DS41_W_HD constexpr int64_t row_bytes(int64_t n) {
    return valid_width(n) && n <= INT64_MAX - n / kGroup ? n + n / kGroup : 0;
}
RAD_DS41_W_HD constexpr int64_t payload_offset(int64_t group) { return group * kGroup; }
RAD_DS41_W_HD constexpr int64_t scale_offset(int64_t n, int64_t group) { return n + group; }
RAD_DS41_W_HD inline uint16_t load_u16_le(const uint8_t* p) {
    return uint16_t(p[0] | uint16_t(p[1]) << 8);
}
RAD_DS41_W_HD inline void store_u16_le(uint8_t* p, uint16_t u) {
    p[0] = uint8_t(u); p[1] = uint8_t(u >> 8);
}
RAD_DS41_W_HD constexpr bool normal_scale(uint8_t s) {
    return s >= kMinFiniteScale && s <= kMaxFiniteScale;
}
/* Finite E4M3 is m*2^-9 (subnormal) or(8+m)*2^(e-10). Its product
 * with2^(scale-127) has<=4 significant bits, so BF16 stores are exact
 * except overflow. NaN bits below are a CANDIDATE product contract; actual
 * card NaNs must pass pack+oracle byte checks, never CPU float equality. */
RAD_DS41_W_HD inline bool decode_normal_bits(uint8_t scale, uint8_t code, uint16_t* out) {
    if (!out || !normal_scale(scale)) return false;
    const uint16_t sign = code & 0x80 ? 0x8000 : 0;
    const int e = (code >> 3) & 15, m = code & 7;
    if (e == 15 && m == 7) { *out = uint16_t(sign | 0x7FC0); return true; }
    unsigned n = e ? unsigned(8 + m) : unsigned(m);
    if (!n) { *out = sign; return true; }
    const int power = (e ? e - 10 : -9) + int(scale) - 127;
    int top = 0;
    for (unsigned x = n; x > 1; x >>= 1) ++top;
    const int exponent = top + power;
    if (exponent > 127) *out = uint16_t(sign | 0x7F80);
    else *out = uint16_t(sign | ((exponent + 127) << 7) | ((n << (7 - top)) & 127));
    return true;
}
RAD_DS41_W_HD inline bool decode_group_bits(const uint8_t* payload, uint8_t scale,
                                           int col, uint16_t* out) {
    if (!payload || !out || col < 0 || col >= kGroup) return false;
    if (!scale) {
        const int select = (payload[4 + col / 8] >> (col & 7)) & 1;
        *out = load_u16_le(payload + 2 * select); return true;
    }
    return decode_normal_bits(scale, payload[col], out);
}
RAD_DS41_W_HD inline uint16_t decode_row_bits(const uint8_t* row, int64_t n, int64_t col) {
    uint16_t out = 0;
    if (!row || !valid_width(n) || col < 0 || col >= n) return out;
    const int64_t group = col / kGroup;
    (void)decode_group_bits(row + payload_offset(group), row[scale_offset(n, group)], int(col % kGroup), &out);
    return out;
}
RAD_DS41_W_HD inline bool decode_row(const uint8_t* row, int64_t n, uint16_t* out) {
    if (!row || !out || !row_bytes(n)) return false;
    for (int64_t g = 0; g < n / kGroup; ++g)
        if (row[n + g] && !normal_scale(row[n + g])) return false;
    for (int64_t c = 0; c < n; ++c) out[c] = decode_row_bits(row, n, c);
    return true;
}
RAD_DS41_W_HD inline PackResult pack_captured_group(uint8_t source_scale, const uint8_t* codes,
                                                   const uint16_t* actual, uint8_t* payload,
                                                   uint8_t* stored_scale) {
    if (!codes || !actual || !payload || !stored_scale ||
        (!normal_scale(source_scale) && source_scale != 255)) return PackResult::Raw;
    bool exact = normal_scale(source_scale);
    for (int i = 0; i < kGroup && exact; ++i) {
        uint16_t decoded = 0;
        exact = decode_normal_bits(source_scale, codes[i], &decoded) && decoded == actual[i];
    }
    if (exact) {
        for (int i = 0; i < kGroup; ++i) payload[i] = codes[i];
        *stored_scale = source_scale;
        return PackResult::Exact;
    }
    uint16_t a = actual[0], b = a;
    bool second = false;
    for (int i = 1; i < kGroup; ++i) if (actual[i] != a) {
        if (!second) { b = actual[i]; second = true; }
        else if (actual[i] != b) return PackResult::Raw;
    }
    for (int i = 0; i < kGroup; ++i) payload[i] = 0;
    store_u16_le(payload, a); store_u16_le(payload + 2, b);
    for (int i = 0; i < kGroup; ++i)
        if (actual[i] != a) payload[4 + i / 8] |= uint8_t(1u << (i & 7));
    *stored_scale = kPairScale;
    return PackResult::Exact;
}
} // namespace rad_ds41_window_codec
#undef RAD_DS41_W_HD
