/* Test-only old reference and bounded corpus. NEVER included by serving code.
 * old_nearest is the literal old ds_e2m1_rt loop with portable bit access only.
 */
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include "../../arch/common/rad_ds41_fp4_classifier.h"
#if defined(__HIPCC__)
#define DS41_FP4_ORACLE_HD __host__ __device__
#else
#define DS41_FP4_ORACLE_HD
#endif
namespace ds41_fp4_oracle {
DS41_FP4_ORACLE_HD inline uint32_t bits(float x) {
#if defined(__HIP_DEVICE_COMPILE__)
    return __float_as_uint(x);
#else
    uint32_t u; std::memcpy(&u, &x, 4); return u;
#endif
}
DS41_FP4_ORACLE_HD inline float value(uint32_t u) {
#if defined(__HIP_DEVICE_COMPILE__)
    return __uint_as_float(u);
#else
    float x; std::memcpy(&x, &u, 4); return x;
#endif
}
DS41_FP4_ORACLE_HD inline float old_nearest(float v) {
    if (v != v) return 0.0f;
    const bool neg = (bits(v) >> 31) != 0;
    const float a = fabsf(v);
    const float mag[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
    float bm = 0.0f;
#if defined(__HIP_DEVICE_COMPILE__)
#pragma unroll
#endif
    for (int c = 1; c < 8; ++c) {
        const float d = fabsf(mag[c] - a), db = fabsf(bm - a);
        if (d < db || (d == db && (c & 1) == 0)) bm = mag[c];
    }
    return neg ? -bm : bm;
}
DS41_FP4_ORACLE_HD inline float prototype(float v) {
    rad_ds41_fp4_classifier::Result out;
    return rad_ds41_fp4_classifier::classify_bits(bits(v), &out) ? value(out.value_bits) : old_nearest(v);
}
DS41_FP4_ORACLE_HD inline uint16_t bf16(float f) {
    const uint32_t u = bits(f);
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) return uint16_t((u >> 16) | 0x40u);
    return uint16_t((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
inline std::vector<uint32_t> corpus() {
    std::vector<uint32_t> out; out.reserve(220000);
    // Every BF16 input image, including NaN payload/sign, infinities and huge values.
    for (uint32_t h = 0; h < 65536; ++h) out.push_back(h << 16);
    const uint32_t boundaries[] = {0x3E800000,0x3F400000,0x3FA00000,0x3FE00000,0x40200000,0x40600000,0x40A00000};
    for (uint32_t b : boundaries) for (int d = -4096; d <= 4096; ++d) {
        out.push_back(uint32_t(int64_t(b) + d));out.push_back(uint32_t(int64_t(b) + d) | 0x80000000u);
    }
    // Stratify every exponent and mantissa, instead of testing only BF16 lattice points.
    for (uint32_t e = 0; e < 256; ++e) for (uint32_t j = 0; j < 64; ++j) {
        const uint32_t u = (e << 23) | ((j * 0x42109u) & 0x7FFFFFu);
        out.push_back(u);out.push_back(u | 0x80000000u);
    }
    for (uint32_t u : {1u,0x7Fu,0x7FFFu,0x7FFFFFu,0x800000u,0x40C00001u,0x7F7FFFFFu,0x7F800001u,0x7FC00001u,0x7FFFFFFFu}) {
        out.push_back(u);out.push_back(u | 0x80000000u);
    }
    return out;
}
} // namespace ds41_fp4_oracle
#undef DS41_FP4_ORACLE_HD
