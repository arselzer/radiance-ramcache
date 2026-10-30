/* router_score.h -- the router predictor's arithmetic (router_predictor.h), apart so its test and
 * the offline evaluator (tools/ds41-host-experts/router_score_eval.cpp) run the same code.
 *
 * ---- bf16 (the predictor's own scoring) -----------------------------------------------------------
 *
 * Each of a layer's 384 experts scores dot(w, x) over its router row w (bf16, 10 KB) and the
 * stand-in input x (f32), FMA in two accumulators; the key is the router's: sqrt(softplus(dot))
 * plus the correction bias. One prediction reads 384 rows, 3.93 MB, from DRAM while the pool
 * streams its experts: ~597 us a prediction on rank 0 (ab-20261029T161035Z), latency-bound
 * (~0.15 ms of FMA work).
 *
 * ---- int8 / int4 (RADIANCE_EXPERIMENTAL_DS41_PREFETCH_SCORE) --------------------------------------
 *
 * A copy of each row quantized with one scale (an expert's): int8 w ~ s q, q in [-127, 127],
 * s = max|w| / 127; int4 w ~ s (n - 8), n in [0, 15], s = c max|w| / 7 with the clip c in
 * kU4Clip that gives the row the least squared error (u4_sq_err_*; the AVX2 form sums its lanes in
 * f32, so a near tie between two clips may go the other way than the reference's: either is a
 * valid scale). The input is quantized once a prediction to
 * int16, x ~ sx xq, sx = max|x| / 32767, and the dots are integer (vpmaddwd), exact:
 *
 *   int8  logit ~ s sx (sum q xq)              1.97 MB a prediction
 *   int4  logit ~ s sx (sum n xq - 8 sum xq)    0.98 MB a prediction
 *
 * The keys rank the experts; the best R (..._PREFETCH_RESCORE, 12) are scored again in bf16 and the
 * best k of those (by their bf16 keys) are the prediction. So whenever bf16's own best k are among
 * the R, the prediction is bf16's, in bf16's order; the predictor's check (..._SCORE_CHECK) counts
 * how often they are. Predictions only steer reads ahead: no output depends on them.
 *
 * int4's nibbles: a row is groups of 32 elements, 16 bytes a group; byte j holds element j in its
 * low nibble and element 16 + j in its high one, so a mask and a shift give each half in order.
 *
 * Bounds (int32 lanes; the sums leave them as int64): a vpmaddwd lane adds two products, at most
 * 2 x 127 x 32767 = 8.3e6 (int8) or 2 x 15 x 32767 = 9.8e5 (int4), and each of the four
 * accumulators takes one of every four 16-element steps: 80 steps a 5120-element row, so at most
 * 6.7e8 < 2^31 (int8). A row may be up to 8192 elements (kMaxDotElems) that way. */
#pragma once
#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace rad {
namespace rscore {

/* The longest row the integer dots take within the int32 bound above: 8192 elements give each
 * accumulator 128 steps, at most 128 x 8.3e6 = 1.07e9 a lane (int8). */
constexpr int kMaxDotElems = 8192;
/* int4's clips of max|w| tried for each row's scale (the least squared error wins). */
constexpr float kU4Clip[] = { 1.0f, 0.9f, 0.8f, 0.7f, 0.6f, 0.5f };

inline uint16_t f32_to_bf16_rne(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fff + ((u >> 16) & 1);
    return (uint16_t)(u >> 16);
}

inline float bf16_to_f32(uint16_t h) {
    const uint32_t u = (uint32_t)h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

/* The router's key for a logit: sqrt(softplus(logit)) + the correction bias (it only selects). */
inline float key_of(float logit, float corr) {
    const float sp = logit > 0 ? logit + std::log1p(std::exp(-logit)) : std::log1p(std::exp(logit));
    return std::sqrt(sp) + corr;
}

/* dot(w bf16, x f32) for n a multiple of 8; FMA is fine here, the result only ranks experts. */
__attribute__((target("avx2,fma"))) inline float dot_bf16_avx2(const uint16_t* w, const float* x, int n) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    int k = 0;
    for (; k + 16 <= n; k += 16) {
        const __m256 w0 = _mm256_castsi256_ps(_mm256_slli_epi32(
            _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(w + k))), 16));
        const __m256 w1 = _mm256_castsi256_ps(_mm256_slli_epi32(
            _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(w + k + 8))), 16));
        a0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x + k), a0);
        a1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x + k + 8), a1);
    }
    for (; k + 8 <= n; k += 8) {
        const __m256 w0 = _mm256_castsi256_ps(_mm256_slli_epi32(
            _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(w + k))), 16));
        a0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x + k), a0);
    }
    a0 = _mm256_add_ps(a0, a1);
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(a0), _mm256_extractf128_ps(a0, 1));
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

inline float dot_bf16_ref(const uint16_t* w, const float* x, int n) {
    float a = 0;
    for (int k = 0; k < n; ++k) a += bf16_to_f32(w[k]) * x[k];
    return a;
}

/* ---- quantized rows ------------------------------------------------------------------------- */

/* Row w[n] into q[n]; returns its scale (0 for a row of zeros, or one not finite: q all 0). */
inline float quantize_row_i8(const float* w, int n, int8_t* q) {
    float amax = 0.0f;
    for (int k = 0; k < n; ++k) amax = std::max(amax, std::fabs(w[k]));
    if (!(amax > 0.0f) || !std::isfinite(amax)) {
        std::memset(q, 0, (size_t)n);
        return 0.0f;
    }
    const float inv = 127.0f / amax;
    for (int k = 0; k < n; ++k)
        q[k] = (int8_t)std::clamp(std::lrint(w[k] * inv), -127L, 127L);
    return amax / 127.0f;
}

/* The squared error of row w[n] at int4 scale s (n - 8 in [-8, 7], rounded to nearest even). */
inline double u4_sq_err_ref(const float* w, int n, float s) {
    const float inv = 1.0f / s;
    double err = 0;
    for (int k = 0; k < n; ++k) {
        const float v = (float)std::clamp(std::lrint(w[k] * inv), -8L, 7L);
        const double d = (double)w[k] - (double)v * s;
        err += d * d;
    }
    return err;
}

/* The same, eight lanes at a time (n a multiple of 8); f32 lanes, summed in f64. */
__attribute__((target("avx2,fma"))) inline double u4_sq_err_avx2(const float* w, int n, float s) {
    const __m256 inv = _mm256_set1_ps(1.0f / s), sv = _mm256_set1_ps(s);
    const __m256 lo = _mm256_set1_ps(-8.0f), hi = _mm256_set1_ps(7.0f);
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    int k = 0;
    for (; k + 16 <= n; k += 16) {
        const __m256 x0 = _mm256_loadu_ps(w + k), x1 = _mm256_loadu_ps(w + k + 8);
        const __m256 q0 = _mm256_min_ps(_mm256_max_ps(_mm256_round_ps(_mm256_mul_ps(x0, inv),
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), lo), hi);
        const __m256 q1 = _mm256_min_ps(_mm256_max_ps(_mm256_round_ps(_mm256_mul_ps(x1, inv),
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), lo), hi);
        const __m256 d0 = _mm256_fnmadd_ps(q0, sv, x0), d1 = _mm256_fnmadd_ps(q1, sv, x1);
        a0 = _mm256_fmadd_ps(d0, d0, a0);
        a1 = _mm256_fmadd_ps(d1, d1, a1);
    }
    alignas(32) float t[8];
    _mm256_store_ps(t, _mm256_add_ps(a0, a1));
    double err = 0;
    for (float v : t) err += v;
    if (k < n) err += u4_sq_err_ref(w + k, n - k, s);
    return err;
}

/* Row w[n] (n a multiple of 32) into q[n / 2], nibbles as above; returns its scale. */
inline float quantize_row_u4(const float* w, int n, uint8_t* q) {
    float amax = 0.0f;
    for (int k = 0; k < n; ++k) amax = std::max(amax, std::fabs(w[k]));
    if (!(amax > 0.0f) || !std::isfinite(amax)) {
        std::memset(q, 0x88, (size_t)n / 2);   /* n = 8: w = 0 */
        return 0.0f;
    }
    static const bool avx2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    float best = 0.0f;
    double best_err = INFINITY;
    for (float c : kU4Clip) {
        const float s = amax * c / 7.0f;
        const double err = avx2 ? u4_sq_err_avx2(w, n, s) : u4_sq_err_ref(w, n, s);
        if (err < best_err) { best_err = err; best = s; }
    }
    const float inv = 1.0f / best;
    for (int g = 0; g < n / 32; ++g)
        for (int j = 0; j < 16; ++j) {
            const long lo = std::clamp(std::lrint(w[g * 32 + j] * inv), -8L, 7L) + 8;
            const long hi = std::clamp(std::lrint(w[g * 32 + 16 + j] * inv), -8L, 7L) + 8;
            q[g * 16 + j] = (uint8_t)(lo | (hi << 4));
        }
    return best;
}

/* Element k of an int4 row, as n - 8 (the quantized value). */
inline int u4_at(const uint8_t* q, int k) {
    const uint8_t b = q[(k / 32) * 16 + (k % 32) % 16];
    return ((k % 32) < 16 ? (b & 15) : (b >> 4)) - 8;
}

/* The input x[n] into xq[n]: returns sx (x ~ sx xq; 0 when x is all zeros), *sum = sum of xq. A
 * value that is not finite counts as 0. The reference form; quantize_x16 below takes the AVX2 one. */
inline float quantize_x16_ref(const float* x, int n, int16_t* xq, int64_t* sum) {
    float amax = 0.0f;
    for (int k = 0; k < n; ++k)
        if (std::isfinite(x[k])) amax = std::max(amax, std::fabs(x[k]));
    int64_t s = 0;
    if (!(amax > 0.0f)) {
        std::memset(xq, 0, sizeof(int16_t) * (size_t)n);
        *sum = 0;
        return 0.0f;
    }
    const float inv = 32767.0f / amax;
    for (int k = 0; k < n; ++k) {
        const long v = std::isfinite(x[k]) ? std::clamp(std::lrint(x[k] * inv), -32767L, 32767L) : 0L;
        xq[k] = (int16_t)v;
        s += v;
    }
    *sum = s;
    return amax / 32767.0f;
}

/* quantize_x16_ref's values eight lanes at a time (n a multiple of 16): the same max over the
 * finite |x| (a lane not finite is masked to 0 in both passes), _mm256_cvtps_epi32 rounding to
 * nearest even as lrint does in the default mode, the same clamp; the lanes' sums in int32, which
 * n <= 65536 keeps under 2^31 (each lane adds n / 16 pairs of at most 32767). A scalar libm lrint a
 * value cost ~16 us a prediction of 5120 against ~0.7 (the predictor review). */
__attribute__((target("avx2"))) inline float quantize_x16_avx2(const float* x, int n, int16_t* xq, int64_t* sum) {
    const __m256 abs = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256 inf = _mm256_set1_ps(INFINITY);
    __m256 m = _mm256_setzero_ps();
    for (int k = 0; k < n; k += 8) {
        const __m256 a = _mm256_and_ps(_mm256_loadu_ps(x + k), abs);
        m = _mm256_max_ps(m, _mm256_and_ps(a, _mm256_cmp_ps(a, inf, _CMP_LT_OQ)));
    }
    alignas(32) float t[8];
    _mm256_store_ps(t, m);
    float amax = 0.0f;
    for (float v : t) amax = std::max(amax, v);
    if (!(amax > 0.0f)) {
        std::memset(xq, 0, sizeof(int16_t) * (size_t)n);
        *sum = 0;
        return 0.0f;
    }
    const __m256 inv = _mm256_set1_ps(32767.0f / amax);
    const __m256i hi = _mm256_set1_epi32(32767), lo = _mm256_set1_epi32(-32767);
    __m256i s = _mm256_setzero_si256();
    for (int k = 0; k < n; k += 16) {
        __m256 a0 = _mm256_loadu_ps(x + k), a1 = _mm256_loadu_ps(x + k + 8);
        a0 = _mm256_and_ps(a0, _mm256_cmp_ps(_mm256_and_ps(a0, abs), inf, _CMP_LT_OQ));
        a1 = _mm256_and_ps(a1, _mm256_cmp_ps(_mm256_and_ps(a1, abs), inf, _CMP_LT_OQ));
        __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(a0, inv)), i1 = _mm256_cvtps_epi32(_mm256_mul_ps(a1, inv));
        i0 = _mm256_max_epi32(_mm256_min_epi32(i0, hi), lo);
        i1 = _mm256_max_epi32(_mm256_min_epi32(i1, hi), lo);
        s = _mm256_add_epi32(s, _mm256_add_epi32(i0, i1));
        _mm256_storeu_si256((__m256i*)(xq + k), _mm256_permute4x64_epi64(_mm256_packs_epi32(i0, i1), 0xD8));
    }
    alignas(32) int32_t u[8];
    _mm256_store_si256((__m256i*)u, s);
    int64_t a = 0;
    for (int32_t v : u) a += v;
    *sum = a;
    return amax / 32767.0f;
}

/* quantize_x16_ref's result, by the AVX2 form where it applies. */
inline float quantize_x16(const float* x, int n, int16_t* xq, int64_t* sum) {
    static const bool avx2 = __builtin_cpu_supports("avx2");
    if (avx2 && n > 0 && n % 16 == 0 && n <= 65536) return quantize_x16_avx2(x, n, xq, sum);
    return quantize_x16_ref(x, n, xq, sum);
}

/* ---- integer dots: sum q[k] xq[k] (int8), sum n[k] xq[k] (int4, n the unsigned nibble) ------- */

inline int64_t dot_i8_ref(const int8_t* q, const int16_t* x, int n) {
    int64_t s = 0;
    for (int k = 0; k < n; ++k) s += (int64_t)q[k] * x[k];
    return s;
}

inline int64_t dot_u4_ref(const uint8_t* q, const int16_t* x, int n) {
    int64_t s = 0;
    for (int g = 0; g < n / 32; ++g)
        for (int j = 0; j < 16; ++j) {
            const uint8_t b = q[g * 16 + j];
            s += (int64_t)(b & 15) * x[g * 32 + j] + (int64_t)(b >> 4) * x[g * 32 + 16 + j];
        }
    return s;
}

__attribute__((target("avx2"))) inline int64_t hsum_epi32_i64(__m256i v) {
    const __m256i lo = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(v));
    const __m256i hi = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(v, 1));
    const __m256i s = _mm256_add_epi64(lo, hi);
    alignas(32) int64_t t[4];
    _mm256_store_si256((__m256i*)t, s);
    return t[0] + t[1] + t[2] + t[3];
}

/* n a multiple of 64, at most kMaxDotElems (the int32 bound); equal to dot_i8_ref. */
__attribute__((target("avx2"))) inline int64_t dot_i8_avx2(const int8_t* q, const int16_t* x, int n) {
    __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
    int k = 0;
    for (; k + 64 <= n; k += 64) {
        const __m256i w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(q + k)));
        const __m256i w1 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(q + k + 16)));
        const __m256i w2 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(q + k + 32)));
        const __m256i w3 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(q + k + 48)));
        a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(w0, _mm256_loadu_si256((const __m256i*)(x + k))));
        a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(w1, _mm256_loadu_si256((const __m256i*)(x + k + 16))));
        a2 = _mm256_add_epi32(a2, _mm256_madd_epi16(w2, _mm256_loadu_si256((const __m256i*)(x + k + 32))));
        a3 = _mm256_add_epi32(a3, _mm256_madd_epi16(w3, _mm256_loadu_si256((const __m256i*)(x + k + 48))));
    }
    int64_t s = hsum_epi32_i64(a0) + hsum_epi32_i64(a1) + hsum_epi32_i64(a2) + hsum_epi32_i64(a3);
    for (; k < n; ++k) s += (int64_t)q[k] * x[k];
    return s;
}

/* n a multiple of 64, at most kMaxDotElems; equal to dot_u4_ref. */
__attribute__((target("avx2"))) inline int64_t dot_u4_avx2(const uint8_t* q, const int16_t* x, int n) {
    const __m128i m = _mm_set1_epi8(0x0f);
    __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
    int k = 0;
    for (; k + 64 <= n; k += 64) {
        const __m128i v0 = _mm_loadu_si128((const __m128i*)(q + k / 2));
        const __m128i v1 = _mm_loadu_si128((const __m128i*)(q + k / 2 + 16));
        const __m128i l0 = _mm_and_si128(v0, m), h0 = _mm_and_si128(_mm_srli_epi16(v0, 4), m);
        const __m128i l1 = _mm_and_si128(v1, m), h1 = _mm_and_si128(_mm_srli_epi16(v1, 4), m);
        a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(_mm256_cvtepu8_epi16(l0),
                                                    _mm256_loadu_si256((const __m256i*)(x + k))));
        a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(_mm256_cvtepu8_epi16(h0),
                                                    _mm256_loadu_si256((const __m256i*)(x + k + 16))));
        a2 = _mm256_add_epi32(a2, _mm256_madd_epi16(_mm256_cvtepu8_epi16(l1),
                                                    _mm256_loadu_si256((const __m256i*)(x + k + 32))));
        a3 = _mm256_add_epi32(a3, _mm256_madd_epi16(_mm256_cvtepu8_epi16(h1),
                                                    _mm256_loadu_si256((const __m256i*)(x + k + 48))));
    }
    int64_t s = hsum_epi32_i64(a0) + hsum_epi32_i64(a1) + hsum_epi32_i64(a2) + hsum_epi32_i64(a3);
    if (k < n) s += dot_u4_ref(q + k / 2, x + k, n - k);   /* whole groups of 32 left */
    return s;
}

}  /* namespace rscore */
}  /* namespace rad */
