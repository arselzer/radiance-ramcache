/* Shared HOST/DEVICE comparison cases, never an implementation dependency of the fast reader.
 * The unchanged codec is the independent reference. Every comparison is uint16/byte equality. */
#pragma once
#include "../../arch/common/rad_ds41_kv_codec.h"
#include "../../arch/common/rad_ds41_kv_fast.h"
#if defined(__HIPCC__) || defined(__CUDACC__)
#define DS41_KVF_ORACLE __host__ __device__
#else
#define DS41_KVF_ORACLE
#endif
namespace ds41_kv_fast_oracle {
namespace ref = rad_ds41_kv_codec;
namespace fast = rad_ds41_kv_fast;
DS41_KVF_ORACLE inline uint32_t next(uint32_t& seed) {
    seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return seed;
}
DS41_KVF_ORACLE inline unsigned normal_case(unsigned at) {
    const auto k = (ref::Kind)((at >> 16) + 1);
    const uint8_t scale = (uint8_t)(at >> 8), code = (uint8_t)at;
    uint16_t a = 0xA55A, b = 0xA55A;
    const bool ar = ref::decode_normal_bits(k, scale, code, &a);
    const bool br = fast::decode_normal_bits(k, scale, code, &b);
    return ar != br || a != b || (!ar && (a != 0xA55A || b != 0xA55A));
}
DS41_KVF_ORACLE inline unsigned pair_case(uint32_t kind, unsigned at) {
    const auto k = (ref::Kind)kind;
    const int G = ref::group_size(k);
    uint8_t payload[16]; uint32_t seed = 0xA36193D5u ^ (at * 0x9E3779B9u);
    for (int i = 0; i < 16; ++i) payload[i] = (uint8_t)next(seed);
    // Every uint16 pattern appears as each literal. Selectors/padding are varied independently.
    const uint16_t p0 = (uint16_t)at, p1 = (uint16_t)(at ^ 0xA55Au);
    payload[0] = (uint8_t)p0; payload[1] = (uint8_t)(p0 >> 8);
    payload[2] = (uint8_t)p1; payload[3] = (uint8_t)(p1 >> 8);
    payload[4] = (uint8_t)((payload[4] & ~3u) | 2u); // Always select BOTH literals at i0/i1.
    unsigned bad = 0;
    for (int i = -1; i <= G; ++i) {
        uint16_t a = 0xBEEF, b = 0xBEEF;
        const bool ar = ref::decode_group_value(k, 0, payload, i, &a);
        const bool br = fast::decode_group_value(k, 0, payload, i, &b);
        bad += ar != br || a != b;
        if (i >= 0 && i < G) {
            const uint16_t want = (payload[4 + i / 8] >> (i & 7)) & 1 ? p1 : p0;
            bad += a != want || b != want;
        } else bad += a != 0xBEEF || b != 0xBEEF;
    }
    return bad;
}
template<ref::Kind K, int Width>
DS41_KVF_ORACLE inline unsigned row_case(unsigned at) {
    constexpr int G = K == ref::Kind::Fp4E4M3G16 ? 16 : 32;
    constexpr int Bytes = Width / 2 + Width / G;
    uint8_t guarded[Bytes + 2]; uint16_t a[Width], b[Width];
    uint32_t seed = 0x7F4A7C15u ^ ((at + 1) * 0x9E3779B9u);
    guarded[0] = guarded[Bytes + 1] = 0xA5;
    uint8_t* row = guarded + 1; // Deliberately unaligned; native index rows also pitch68.
    for (int i = 0; i < Bytes; ++i) row[i] = (uint8_t)next(seed);
    for (int g = 0; g < Width / G; ++g) {
        uint8_t scale = (uint8_t)(1 + next(seed) % (K == ref::Kind::Fp4E4M3G16 ? 126 : 253));
        if ((g + at) % 3 == 0) scale = 0;
        if (K == ref::Kind::Fp4Pow2G32 && (g + at) % 7 == 0) scale = 254;
        row[Width / 2 + g] = scale;
    }
    if (at % 11 == 0) row[Bytes - 1] = K == ref::Kind::Fp4E4M3G16 ? 127 : 255;
    if (at == 1) for (int i = 0; i < Bytes; ++i) row[i] = 0;
    for (int i = 0; i < Width; ++i) a[i] = b[i] = 0xBEEF;
    const bool ar = ref::decode_row(K, row, Width, a), br = fast::decode_row(K, row, Width, b);
    unsigned bad = ar != br;
    for (int i = 0; i < Width; ++i) bad += a[i] != b[i] || (!ar && (a[i] != 0xBEEF || b[i] != 0xBEEF));
    if (ar && br) {
        for (int i = 0; i < Width; ++i)
            bad += fast::decode_row_bits<K, Width>(row, i) != ref::decode_row_bits(K, row, Width, i);
        // Every starting column includes unaligned selector reads and both group-boundary crossings.
        for (int col = 0; col <= Width - 8; ++col) {
            uint16_t out[10]; for (int i = 0; i < 10; ++i) out[i] = 0xA55A;
            fast::decode_row8_bits<K, Width>(row, col, out + 1);
            bad += out[0] != 0xA55A || out[9] != 0xA55A;
            for (int i = 0; i < 8; ++i) bad += out[i + 1] != a[col + i];
        }
    }
    bad += guarded[0] != 0xA5 || guarded[Bytes + 1] != 0xA5;
    return bad;
}
DS41_KVF_ORACLE inline unsigned metadata_cases() {
    unsigned bad = 0; uint8_t payload[16] = {}; uint16_t a = 0xA55A, b = 0xA55A;
    for (uint32_t ki = 0; ki < 5; ++ki) {
        const auto k = (ref::Kind)(ki == 4 ? 255 : ki);
        for (int scale = 0; scale < 256; ++scale) for (int i = -1; i <= 33; ++i) {
            a = b = 0xA55A;
            const bool ar = ref::decode_group_value(k, (uint8_t)scale, payload, i, &a);
            const bool br = fast::decode_group_value(k, (uint8_t)scale, payload, i, &b);
            bad += ar != br || a != b;
        }
        bad += ref::decode_normal_bits(k, 1, 0, nullptr) != fast::decode_normal_bits(k, 1, 0, nullptr);
        bad += ref::decode_group_value(k, 0, nullptr, 0, &a) != fast::decode_group_value(k, 0, nullptr, 0, &b);
        bad += ref::decode_group_value(k, 0, payload, 0, nullptr) != fast::decode_group_value(k, 0, payload, 0, nullptr);
        uint8_t row[288] = {}; uint16_t out[512];
        for (int width = -1; width <= 65; ++width) {
            for (int j = 0; j < 512; ++j) out[j] = 0xBEEF;
            const bool ar = ref::decode_row(k, row, width, out);
            uint16_t other[512]; for (int j = 0; j < 512; ++j) other[j] = 0xBEEF;
            const bool br = fast::decode_row(k, row, width, other);
            bad += ar != br;
            for (int j = 0; j < 512; ++j) bad += out[j] != other[j];
        }
        bad += ref::decode_row(k, nullptr, 32, out) != fast::decode_row(k, nullptr, 32, out);
        bad += ref::decode_row(k, row, 32, nullptr) != fast::decode_row(k, row, 32, nullptr);
    }
    return bad;
}
} // namespace ds41_kv_fast_oracle
#undef DS41_KVF_ORACLE
