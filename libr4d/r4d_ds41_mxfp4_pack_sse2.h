#pragma once
// Private CPU packing only. No quantization, dequantization or GPU instructions.
#include "rad_abi.h"
#include <cstdint>
#include <cstdlib>
#include <limits>
#if defined(__SSE2__) && (defined(__x86_64__) || defined(__i386__))
#include <emmintrin.h>
#define DS41_MXFP4_PACK_HAS_SSE2 1
#else
#define DS41_MXFP4_PACK_HAS_SSE2 0
#endif
namespace ds41_mxfp4_pack_sse2 {
inline bool opt_in(const char* s) { return s && s[0] == '1' && s[1] == '\0'; }
inline bool enabled() {
    static const bool on = opt_in(std::getenv("RADIANCE_EXPERIMENTAL_DS41_MXFP4_PACK_SSE2"));
    return DS41_MXFP4_PACK_HAS_SSE2 && on;
}
inline bool disjoint(const void* a, uint64_t an, const void* b, uint64_t bn) {
    const auto x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
    const auto hi = std::numeric_limits<uintptr_t>::max();
    if (!a || !b || an > hi - x || bn > hi - y) return false;
    return x + an <= y || y + bn <= x;
}
inline bool eligible(const RadTensor* p, int np, int64_t rows, int64_t nb,
                     void* out, int64_t cap) {
    if (!enabled() || !p || np != 2 || rows <= 0 || nb <= 0 || cap < 0 ||
        uint64_t(rows) > uint64_t(INT64_MAX) / 32 / uint64_t(nb)) return false;
    const uint64_t blocks = uint64_t(rows) * uint64_t(nb), codes = blocks * 16,
                   scales = blocks, bytes = blocks * 17;
    if (uint64_t(cap) < bytes || p[0].dtype != RAD_FP4E2M1 || p[1].dtype != RAD_E8M0)
        return false;
    for (int i = 0; i < 2; ++i) {
        const int64_t cols = nb * (i == 0 ? 32 : 1);
        if (p[i].rank != 2 || p[i].shape[0] != rows || p[i].shape[1] != cols ||
            p[i].stride[0] != cols || p[i].stride[1] != 1) return false;
    }
    // Whole-plane ranges protect later rows too; legacy behavior is retained for aliases.
    return disjoint(p[0].data, codes, out, bytes) && disjoint(p[1].data, scales, out, bytes);
}
inline void pack(const uint8_t* const* row, int64_t nb, void* dst) {
#if DS41_MXFP4_PACK_HAS_SSE2
    const __m128i mask = _mm_set1_epi8(15);
    auto* out = static_cast<uint8_t*>(dst);
    for (int64_t b = 0; b < nb; ++b) {
        const __m128i in = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row[0] + b * 16));
        const __m128i low = _mm_and_si128(in, mask);
        const __m128i high = _mm_and_si128(_mm_srli_epi16(in, 4), mask);
        const __m128i first = _mm_unpacklo_epi8(low, high);
        const __m128i last = _mm_unpackhi_epi8(low, high);
        out[b * 17] = row[1][b];
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out + b * 17 + 1),
                        _mm_or_si128(first, _mm_slli_epi16(last, 4)));
    }
#else
    (void)row; (void)nb; (void)dst; // Never selected on unsupported architectures.
#endif
}
#ifdef RADIANCE_DS41_PACK_HOST_TEST
inline thread_local uint64_t fast_callbacks = 0, fast_rows = 0;
inline void note(int64_t rows) { ++fast_callbacks; fast_rows += uint64_t(rows); }
#endif
} // namespace ds41_mxfp4_pack_sse2
