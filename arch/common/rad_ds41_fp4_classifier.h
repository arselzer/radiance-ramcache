/* Exact pointwise prototype for ds_e2m1_rt's FINITE |x| <= 6 domain.
 * No environment, FP arithmetic, scale, storage or engine integration. A false
 * result MUST run the original seven-distance loop; it never means saturation.
 */
#pragma once
#include <cstdint>
#if defined(__HIPCC__) || defined(__CUDACC__)
#define RAD_DS41_FP4_HD __host__ __device__
#else
#define RAD_DS41_FP4_HD
#endif
namespace rad_ds41_fp4_classifier {
struct Result {
    uint32_t value_bits = 0;
    uint8_t code = 0; // original E2M1 nibble, including sign bit3
};
RAD_DS41_FP4_HD constexpr bool admitted(uint32_t u) {
    return (u & 0x7FFFFFFFu) <= 0x40C00000u; // closed finite interval [-6,6]
}
RAD_DS41_FP4_HD constexpr bool classify_bits(uint32_t u, Result* out) {
    if (!out || !admitted(u)) return false;
    const uint32_t a = u & 0x7FFFFFFFu;
    // Midpoints .25,.75,1.25,1.75,2.5,3.5,5; ties choose the EVEN code.
    const uint8_t c = a <= 0x3E800000u ? 0 : a < 0x3F400000u ? 1 :
                      a <= 0x3FA00000u ? 2 : a < 0x3FE00000u ? 3 :
                      a <= 0x40200000u ? 4 : a < 0x40600000u ? 5 :
                      a <= 0x40A00000u ? 6 : 7;
    const uint32_t magnitude = c == 0 ? 0u : c == 1 ? 0x3F000000u :
        c == 2 ? 0x3F800000u : c == 3 ? 0x3FC00000u : c == 4 ? 0x40000000u :
        c == 5 ? 0x40400000u : c == 6 ? 0x40800000u : 0x40C00000u;
    out->value_bits = magnitude | (u & 0x80000000u);
    out->code = uint8_t(c | ((u >> 28) & 8u));
    return true;
}
} // namespace rad_ds41_fp4_classifier
#undef RAD_DS41_FP4_HD
