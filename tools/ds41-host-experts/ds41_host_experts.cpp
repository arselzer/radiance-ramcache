/* ds41_host_experts.cpp -- see ds41_host_experts.h. Build with -ffp-contract=off and without
 * -ffast-math: every sum below must stay one rounded product and one rounded add, in order, and
 * subnormals must not be flushed, as on the card. */
#include "ds41_host_experts.h"

#include <immintrin.h>

#include <cmath>
#include <cstring>
#include <thread>
#include <vector>

namespace ds41hx {
namespace {

/* r4d_ds41_mxfp4.hip kMxValues: e2m1 doubled. Code 8 is +0 there, so it is +0 here. */
constexpr float kKv[16] = { 0.f, 1.f, 2.f, 3.f, 4.f, 6.f, 8.f, 12.f,
                            0.f, -1.f, -2.f, -3.f, -4.f, -6.f, -8.f, -12.f };

inline float bits_f(uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; }
inline uint32_t f_bits(float f) { uint32_t b; std::memcpy(&b, &f, 4); return b; }

/* mx_e8m0_half: 2^(e - 128), the two subnormals (e = 0, 1) as their bit patterns. */
inline float e8m0_half(uint32_t e) { return bits_f(e < 2u ? (0x00200000u << e) : ((e - 1u) << 23)); }

/* One row of either layout. */
struct Row {
    const uint8_t* blk = nullptr;
    const uint8_t* codes = nullptr;
    const uint8_t* sc = nullptr;
};
inline Row row_of(const Mx& w, int n) {
    Row r;
    if (w.blocks) r.blk = w.blocks + (size_t)n * (size_t)(w.K / 32) * 17;
    else {
        r.codes = w.codes + (size_t)n * (size_t)(w.K / 2);
        r.sc = w.scales + (size_t)n * (size_t)(w.K / 32);
    }
    return r;
}
/* The code of value v (0..31) of block u, and the block's e8m0. */
inline int code(const Row& r, int u, int v) {
    if (r.blk) {
        const uint8_t* q = r.blk + (size_t)u * 17 + 1;
        return v < 16 ? (q[v] & 15) : (q[v - 16] >> 4);
    }
    const uint8_t c = r.codes[(size_t)u * 16 + (v >> 1)];
    return (v & 1) ? (c >> 4) : (c & 15);
}
inline uint32_t scale_of(const Row& r, int u) { return r.blk ? r.blk[(size_t)u * 17] : r.sc[u]; }

/* r4d_wave_sum over 32 lane partials: xor 16, then 8, 4, 2, 1. Only lane 0's result is kept, and
 * every lane's is the same because each pair is added in both orders, which is exact. */
inline float wave_sum(float (&p)[32]) {
    for (int o = 16; o; o >>= 1)
        for (int l = 0; l < o; ++l) p[l] = p[l] + p[l + o];
    return p[0];
}
/* ---- AVX2 -----------------------------------------------------------------------------------------
 *
 * One row at a time, its 32 lanes as four vectors of eight. In round r vector g, lane j, holds block
 * 32r + 8g + kLane[j], so a round reads 32 consecutive blocks and every lane still adds its own
 * blocks L, L + 32, ... in order, each block's values in order. Per value each lane computes
 * w = kv[code] * d (exact), p = x * w and sum + p: the reference's operations, eight lanes a vector.
 *
 * The codes of two groups of eight blocks are transposed together, so that one value of a group's
 * eight blocks is one load; x is transposed once a call to the same lanes (x_lanes). kLane takes a
 * group's even blocks, then its odd ones: the order in which e_pair and scales8 find the e8m0 bytes.
 *
 * A round whose 32 scales all lie in [2^-15, 2^-1] (e in [113, 127]: every block of the release's
 * experts but a few at e = 112) takes the fast path: there w = kv * d * 2 is built directly as fp16
 * (kv_pair16) and vcvtph2ps widens it exactly, and x is held halved, so p = (x / 2) * (2 w) is the
 * same real product as x * w, rounded once to the same f32. Halving a bf16 is exact. Any other round
 * takes the general path (general_round): int8 kv, then kv * d and x * w in f32, x doubled back.
 *
 * F16C is enabled for this section, for vcvtph2ps: every AVX2 CPU has it, and it adds no arithmetic
 * of its own. Neither FMA nor any contraction is enabled. */
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2,f16c")
#endif

constexpr int kLane[8] = { 0, 2, 4, 6, 1, 3, 5, 7 };
constexpr int kFastLo = 113, kFastHi = 127;   /* the fast path's e8m0 range */

/* The helpers are small functions and lambdas that must be inlined, or their constants go through
 * memory. */
#define HX_INLINE inline __attribute__((always_inline))

/* The value of byte b's low (h = 0) or high (h = 1) nibble. ggml: byte b < 16 holds values b and
 * 16 + b; planes: byte p holds values 2p and 2p + 1. */
template <bool kGgml>
constexpr int value_of(int b, int h) { return kGgml ? 16 * h + b : 2 * b + h; }

/* The codes of groups A and B transposed: `ca`, `cb` point at the code bytes of each group's block
 * 0, blocks S bytes apart; put(j, r) gets r_j, which holds in each half (A low, B high) byte 2j of
 * the group's eight blocks, then byte 2j + 1. */
template <bool kGgml, class Put>
HX_INLINE void transpose_pair(const uint8_t* ca, const uint8_t* cb, Put&& put) {
    constexpr size_t S = kGgml ? 17 : 16;
    auto ld = [&](int i) __attribute__((always_inline)) {
        return _mm256_loadu2_m128i(reinterpret_cast<const __m128i*>(cb + kLane[i] * S),
                                   reinterpret_cast<const __m128i*>(ca + kLane[i] * S));
    };
    const __m256i a0 = ld(0), a1 = ld(1), a2 = ld(2), a3 = ld(3), a4 = ld(4), a5 = ld(5), a6 = ld(6), a7 = ld(7);
    const __m256i t0 = _mm256_unpacklo_epi8(a0, a1), t1 = _mm256_unpackhi_epi8(a0, a1);
    const __m256i t2 = _mm256_unpacklo_epi8(a2, a3), t3 = _mm256_unpackhi_epi8(a2, a3);
    const __m256i t4 = _mm256_unpacklo_epi8(a4, a5), t5 = _mm256_unpackhi_epi8(a4, a5);
    const __m256i t6 = _mm256_unpacklo_epi8(a6, a7), t7 = _mm256_unpackhi_epi8(a6, a7);
    const __m256i s0 = _mm256_unpacklo_epi16(t0, t2), s1 = _mm256_unpackhi_epi16(t0, t2);
    const __m256i s2 = _mm256_unpacklo_epi16(t4, t6), s3 = _mm256_unpackhi_epi16(t4, t6);
    const __m256i s4 = _mm256_unpacklo_epi16(t1, t3), s5 = _mm256_unpackhi_epi16(t1, t3);
    const __m256i s6 = _mm256_unpacklo_epi16(t5, t7), s7 = _mm256_unpackhi_epi16(t5, t7);
    put(0, _mm256_unpacklo_epi32(s0, s2));
    put(1, _mm256_unpackhi_epi32(s0, s2));
    put(2, _mm256_unpacklo_epi32(s1, s3));
    put(3, _mm256_unpackhi_epi32(s1, s3));
    put(4, _mm256_unpacklo_epi32(s4, s6));
    put(5, _mm256_unpackhi_epi32(s4, s6));
    put(6, _mm256_unpacklo_epi32(s5, s7));
    put(7, _mm256_unpackhi_epi32(s5, s7));
}

/* The e8m0 bytes of a pair's sixteen blocks: A's eight in kLane order, twice, in the low half, B's
 * in the high. ggml: the bytes of a group sit at 0, 17, ..., 119 from `p`, so the even blocks' are
 * bytes 0, 2, 4, 6 of the low halves of four loads at +0, +32, +64, +96 and the odd ones' bytes
 * 1, 3, 5, 7 of their high halves; three word blends gather them, then the even halves and the odd
 * ones merge into bytes 0..7 of each half, in block order. planes: eight consecutive bytes. */
template <bool kGgml>
HX_INLINE __m256i e_pair(const uint8_t* pa, const uint8_t* pb) {
    const __m256i order = _mm256_setr_epi8(0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7,
                                           0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7);
    __m256i q;
    if (kGgml) {
        auto blend4 = [](const uint8_t* p) __attribute__((always_inline)) {
            const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
            const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 32));
            const __m256i c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 64));
            const __m256i d = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 96));
            return _mm256_blend_epi16(_mm256_blend_epi16(a, b, 0x02), _mm256_blend_epi16(c, d, 0x08), 0x0C);
        };
        const __m256i ta = blend4(pa), tb = blend4(pb);
        const __m256i odd = _mm256_setr_epi8(0, -1, 0, -1, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0,
                                             0, -1, 0, -1, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0);
        q = _mm256_blendv_epi8(_mm256_permute2x128_si256(ta, tb, 0x20), _mm256_permute2x128_si256(ta, tb, 0x31), odd);
    } else {
        q = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(pa))),
                                    _mm_loadl_epi64(reinterpret_cast<const __m128i*>(pb)), 1);
    }
    return _mm256_shuffle_epi8(q, order);
}

/* The fast path's weights: 2w = kv[code] * 2^s as fp16 with s = e - 127 in [-14, 0], exact. The fp16
 * of kv[code] has a zero low byte, so a table gives its high byte (sign, exponent, top mantissa
 * bit); adding 4s to that byte with signed saturation adds s to the exponent, which stays in
 * 1 .. 18, and codes 0 and 8, written as -0 (0x80), saturate to -0. A -0 weight instead of +0
 * changes no sum: x * w is then a zero, and a sum is never -0, so sum + p is sum. `s8`: 4s of A's
 * blocks in kLane order, twice (low half), and of B's (high half). w16[v * 16 + g * 8 + j]: value v
 * of group g (A 0, B 1), lane j. */
template <bool kGgml>
HX_INLINE void kv_pair16(const uint8_t* ca, const uint8_t* cb, __m256i s8, uint16_t* w16) {
    const __m256i lut = _mm256_setr_epi8(-128, 0x3C, 0x40, 0x42, 0x44, 0x46, 0x48, 0x4A,
                                         -128, -68, -64, -62, -60, -58, -56, -54,   /* 0x80, 0xBC .. 0xCA */
                                         -128, 0x3C, 0x40, 0x42, 0x44, 0x46, 0x48, 0x4A,
                                         -128, -68, -64, -62, -60, -58, -56, -54);
    const __m256i lo4 = _mm256_set1_epi8(15), zero = _mm256_setzero_si256();
    transpose_pair<kGgml>(ca, cb, [&](int j, __m256i r) __attribute__((always_inline)) {
        const __m256i lo = _mm256_adds_epi8(_mm256_shuffle_epi8(lut, _mm256_and_si256(r, lo4)), s8);
        const __m256i hi = _mm256_adds_epi8(_mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(r, 4), lo4)), s8);
        auto put = [&](int v, __m256i h) __attribute__((always_inline)) {   /* one value's 16 lanes */
            _mm256_store_si256(reinterpret_cast<__m256i*>(w16 + v * 16), h);
        };
        put(value_of<kGgml>(2 * j, 0), _mm256_unpacklo_epi8(zero, lo));
        put(value_of<kGgml>(2 * j + 1, 0), _mm256_unpackhi_epi8(zero, lo));
        put(value_of<kGgml>(2 * j, 1), _mm256_unpacklo_epi8(zero, hi));
        put(value_of<kGgml>(2 * j + 1, 1), _mm256_unpackhi_epi8(zero, hi));
    });
}

/* A fast round: for each value in order, each of the NG vectors adds (x / 2) * (2w). Two values a
 * loop, not unrolled further: the compiler would hoist every conversion of the round and spill. */
template <int G>
HX_INLINE void add16_one(__m256& acc, const uint16_t* w16, const float* xv) {
    const __m256 wv = _mm256_cvtph_ps(_mm_load_si128(reinterpret_cast<const __m128i*>(w16 + (G >> 1) * 512 + (G & 1) * 8)));
    acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_load_ps(xv + G * 8), wv));
}
template <int NG>
HX_INLINE void add_round16(__m256 (&acc)[4], const uint16_t* w16, const float* xr) {
#pragma GCC unroll 1
    for (int v = 0; v < 32; v += 2, w16 += 32, xr += 64) {
        add16_one<0>(acc[0], w16, xr);
        if (NG > 1) add16_one<1>(acc[1], w16, xr);
        if (NG > 2) add16_one<2>(acc[2], w16, xr);
        if (NG > 3) add16_one<3>(acc[3], w16, xr);
        add16_one<0>(acc[0], w16 + 16, xr + 32);
        if (NG > 1) add16_one<1>(acc[1], w16 + 16, xr + 32);
        if (NG > 2) add16_one<2>(acc[2], w16 + 16, xr + 32);
        if (NG > 3) add16_one<3>(acc[3], w16 + 16, xr + 32);
    }
}

/* ---- the general path: any scales, including the subnormal ones and those that overflow w ---- */

/* Where value v of a pair's first group sits in the pair's 512-byte int8 buffer; the second
 * group's is 16 bytes further. */
template <bool kGgml>
constexpr int kv_off(int v) {
    return kGgml ? (v >> 1) * 32 + (v & 1) * 8 : (v >> 2) * 64 + (v & 1) * 32 + ((v >> 1) & 1) * 8;
}

template <bool kGgml>
HX_INLINE void kv_pair(const uint8_t* ca, const uint8_t* cb, int8_t* kv) {
    const __m256i lut = _mm256_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
                                         0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    const __m256i lo4 = _mm256_set1_epi8(15);
    transpose_pair<kGgml>(ca, cb, [&](int j, __m256i r) __attribute__((always_inline)) {
        const __m256i lo = _mm256_shuffle_epi8(lut, _mm256_and_si256(r, lo4));
        const __m256i hi = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(r, 4), lo4));
        _mm256_store_si256(reinterpret_cast<__m256i*>(kv + kv_off<kGgml>(value_of<kGgml>(2 * j, 0))), lo);
        _mm256_store_si256(reinterpret_cast<__m256i*>(kv + kv_off<kGgml>(value_of<kGgml>(2 * j, 1))), hi);
    });
}

/* d = 2^(e - 128) of a group's eight blocks, in kLane order (e8m0_half, subnormals included). */
template <bool kGgml>
HX_INLINE __m256 scales8(const uint8_t* p) {
    const __m256i e = _mm256_cvtepu8_epi32(_mm256_castsi256_si128(e_pair<kGgml>(p, p)));
    const __m256i normal = _mm256_sub_epi32(_mm256_slli_epi32(e, 23), _mm256_set1_epi32(1 << 23));
    const __m256i sub = _mm256_sllv_epi32(_mm256_set1_epi32(0x00200000), e);
    return _mm256_castsi256_ps(_mm256_blendv_epi8(normal, sub, _mm256_cmpgt_epi32(_mm256_set1_epi32(2), e)));
}

template <int G>
HX_INLINE void add_one(__m256& acc, __m256 d, const int8_t* kv, const float* xv) {
    const __m128i b = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(kv + (G >> 1) * 512 + (G & 1) * 16));
    const __m256 wv = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b)), d);              /* exact */
    const __m256 xx = _mm256_mul_ps(_mm256_load_ps(xv + G * 8), _mm256_set1_ps(2.f));             /* exact */
    acc = _mm256_add_ps(acc, _mm256_mul_ps(xx, wv));
}
template <int NG>
HX_INLINE void add_step(__m256 (&acc)[4], const __m256 (&d)[4], const int8_t* kv, const float* xv) {
    add_one<0>(acc[0], d[0], kv, xv);
    if (NG > 1) add_one<1>(acc[1], d[1], kv, xv);
    if (NG > 2) add_one<2>(acc[2], d[2], kv, xv);
    if (NG > 3) add_one<3>(acc[3], d[3], kv, xv);
}
template <bool kGgml, int NG>
HX_INLINE void add_round(__m256 (&acc)[4], const __m256 (&d)[4], const int8_t* kv, const float* xr) {
    constexpr int VC = kGgml ? 2 : 4;   /* values a 32-byte chunk pair holds */
#pragma GCC unroll 1
    for (int c = 0; c < 32 / VC; ++c, kv += 32 * VC / 2, xr += 32 * VC) {
        add_step<NG>(acc, d, kv + kv_off<kGgml>(0), xr);
        add_step<NG>(acc, d, kv + kv_off<kGgml>(1), xr + 32);
        if (!kGgml) {
            add_step<NG>(acc, d, kv + kv_off<kGgml>(2), xr + 64);
            add_step<NG>(acc, d, kv + kv_off<kGgml>(3), xr + 96);
        }
    }
}

/* Out of line, so that its transposes do not merge with the fast path's (and spill). */
template <bool kGgml>
__attribute__((noinline)) void general_round(__m256 (&acc)[4], const uint8_t* const (&gc)[4], const uint8_t* const (&gs)[4],
                                             int ng, const float* xr) {
    alignas(32) int8_t kv[1024];
    kv_pair<kGgml>(gc[0], gc[1], kv);
    if (ng > 2) kv_pair<kGgml>(gc[2], gc[3], kv + 512);
    const __m256 d[4] = { scales8<kGgml>(gs[0]), scales8<kGgml>(gs[1]), scales8<kGgml>(gs[2]), scales8<kGgml>(gs[3]) };
    switch (ng) {
        case 1: add_round<kGgml, 1>(acc, d, kv, xr); break;
        case 2: add_round<kGgml, 2>(acc, d, kv, xr); break;
        case 3: add_round<kGgml, 3>(acc, d, kv, xr); break;
        default: add_round<kGgml, 4>(acc, d, kv, xr); break;
    }
}

/* ---- rounds and rows ---- */

/* xt[(r * 32 + v) * 32 + g * 8 + j] = x / 2 of value v of block 32r + 8g + kLane[j], 0 past the row. */
inline void x_lanes(const uint16_t* x, int UK, float* xt) {
    const int R = (UK + 31) / 32;
    const __m256i zero = _mm256_setzero_si256();
    const __m256 half = _mm256_set1_ps(0.5f);
    for (int r = 0; r < R; ++r)
        for (int g = 0; g < 4; ++g) {
            const int u0 = 32 * r + 8 * g;
            float* dst = xt + (size_t)r * 1024 + g * 8;
            if (u0 + 8 > UK) {
                for (int v = 0; v < 32; ++v)
                    for (int j = 0; j < 8; ++j) {
                        const int u = u0 + kLane[j];
                        dst[v * 32 + j] = u < UK ? bf16_to_f32(x[(size_t)u * 32 + v]) * 0.5f : 0.f;
                    }
                continue;
            }
            for (int h = 0; h < 2; ++h) {   /* values 16h .. 16h + 7 in the low halves, 16h + 8 .. in the high */
                __m256i a[8];
                for (int j = 0; j < 8; ++j)
                    a[j] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + (size_t)(u0 + kLane[j]) * 32 + 16 * h));
                const __m256i t0 = _mm256_unpacklo_epi16(a[0], a[1]), t1 = _mm256_unpackhi_epi16(a[0], a[1]);
                const __m256i t2 = _mm256_unpacklo_epi16(a[2], a[3]), t3 = _mm256_unpackhi_epi16(a[2], a[3]);
                const __m256i t4 = _mm256_unpacklo_epi16(a[4], a[5]), t5 = _mm256_unpackhi_epi16(a[4], a[5]);
                const __m256i t6 = _mm256_unpacklo_epi16(a[6], a[7]), t7 = _mm256_unpackhi_epi16(a[6], a[7]);
                const __m256i s0 = _mm256_unpacklo_epi32(t0, t2), s1 = _mm256_unpackhi_epi32(t0, t2);
                const __m256i s2 = _mm256_unpacklo_epi32(t4, t6), s3 = _mm256_unpackhi_epi32(t4, t6);
                const __m256i s4 = _mm256_unpacklo_epi32(t1, t3), s5 = _mm256_unpackhi_epi32(t1, t3);
                const __m256i s6 = _mm256_unpacklo_epi32(t5, t7), s7 = _mm256_unpackhi_epi32(t5, t7);
                /* o[m]: value 16h + m (low half) and 16h + 8 + m (high half) of the eight blocks */
                const __m256i o[8] = { _mm256_unpacklo_epi64(s0, s2), _mm256_unpackhi_epi64(s0, s2),
                                       _mm256_unpacklo_epi64(s1, s3), _mm256_unpackhi_epi64(s1, s3),
                                       _mm256_unpacklo_epi64(s4, s6), _mm256_unpackhi_epi64(s4, s6),
                                       _mm256_unpacklo_epi64(s5, s7), _mm256_unpackhi_epi64(s5, s7) };
                for (int m = 0; m < 8; ++m) {   /* bf16 to f32: the bits to the high half */
                    const __m256i lo = _mm256_unpacklo_epi16(zero, o[m]), hi = _mm256_unpackhi_epi16(zero, o[m]);
                    _mm256_store_ps(dst + (16 * h + m) * 32, _mm256_mul_ps(_mm256_castsi256_ps(_mm256_permute2x128_si256(lo, hi, 0x20)), half));
                    _mm256_store_ps(dst + (16 * h + 8 + m) * 32, _mm256_mul_ps(_mm256_castsi256_ps(_mm256_permute2x128_si256(lo, hi, 0x31)), half));
                }
            }
        }
}

/* Whether a round's e8m0 bytes (e_pair of groups 0 and 1, and of 2 and 3) all lie in the fast
 * path's window. */
HX_INLINE bool fast_round(__m256i e01, __m256i e23) {
    auto in_range = [](__m256i e) __attribute__((always_inline)) {
        return _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_max_epu8(e, _mm256_set1_epi8((char)kFastLo)), e),
                                _mm256_cmpeq_epi8(_mm256_min_epu8(e, _mm256_set1_epi8((char)kFastHi)), e));
    };
    return _mm256_movemask_epi8(_mm256_and_si256(in_range(e01), in_range(e23))) == -1;
}
/* 4s = 4 (e - 127) = 4e - 508: 4 (e + 1) mod 256 as a signed byte */
HX_INLINE __m256i s8_of(__m256i e) {
    const __m256i t = _mm256_add_epi8(e, _mm256_set1_epi8(1)), t2 = _mm256_add_epi8(t, t);
    return _mm256_add_epi8(t2, t2);
}

/* One round: groups at gc[g] (code bytes of block 0) and gs[g] (its e8m0 byte or scale), ng of
 * them with blocks (groups past ng repeat group 0 and are never added). */
template <bool kGgml>
__attribute__((noinline)) void add_any_round(__m256 (&acc)[4], const uint8_t* const (&gc)[4], const uint8_t* const (&gs)[4],
                                             int ng, const float* xr) {
    const __m256i e01 = e_pair<kGgml>(gs[0], gs[1]), e23 = ng > 2 ? e_pair<kGgml>(gs[2], gs[3]) : e01;
    if (!fast_round(e01, e23)) {
        general_round<kGgml>(acc, gc, gs, ng, xr);
        return;
    }
    alignas(32) uint16_t w16[1024];
    kv_pair16<kGgml>(gc[0], gc[1], s8_of(e01), w16);
    if (ng > 2) kv_pair16<kGgml>(gc[2], gc[3], s8_of(e23), w16 + 512);
    switch (ng) {
        case 1: add_round16<1>(acc, w16, xr); break;
        case 2: add_round16<2>(acc, w16, xr); break;
        case 3: add_round16<3>(acc, w16, xr); break;
        default: add_round16<4>(acc, w16, xr); break;
    }
}

/* The last, partial round of a row: blocks u0 .. UK - 1. A group the row ends inside is copied with
 * zero codes and e = 127 after the row's last block: those lanes add 0 * w = 0 (x is 0 there), which
 * leaves a sum as it is (a sum is never -0). */
template <bool kGgml>
void add_tail(__m256 (&acc)[4], const uint8_t* codes, const uint8_t* scl, int u0, int UK, const float* xr) {
    constexpr size_t BS = kGgml ? 17 : 16, SS = kGgml ? 17 : 1;
    const int nb = UK - u0, ng = (nb + 7) / 8;
    alignas(32) uint8_t pad[8 * 17 + 32];   /* ggml: 8 blocks and room for e_pair's loads */
    alignas(32) uint8_t pad_sc[8];
    const uint8_t* gc[4];
    const uint8_t* gs[4];
    for (int g = 0; g < ng; ++g) {
        const size_t ub = (size_t)u0 + 8 * g;
        const int live = nb - 8 * g < 8 ? nb - 8 * g : 8;
        gc[g] = codes + ub * BS;
        gs[g] = scl + ub * SS;
        if (live < 8) {
            std::memset(pad, 0, sizeof pad);
            std::memset(pad_sc, kFastHi, sizeof pad_sc);
            if (kGgml) {
                std::memcpy(pad, scl + ub * 17, (size_t)live * 17);
                for (int i = live; i < 8; ++i) pad[i * 17] = kFastHi;
                gc[g] = pad + 1;
                gs[g] = pad;
            } else {
                std::memcpy(pad, codes + ub * 16, (size_t)live * 16);
                std::memcpy(pad_sc, scl + ub, (size_t)live);
                gc[g] = pad;
                gs[g] = pad_sc;
            }
        }
    }
    for (int g = ng; g < 4; ++g) { gc[g] = gc[0]; gs[g] = gs[0]; }
    add_any_round<kGgml>(acc, gc, gs, ng, xr);
}

/* r4d_wave_sum of the 32 lanes, as wave_sum: xor 16 and 8 between the vectors, then 4, 2, 1 inside
 * vector 0, whose lane j is lane kLane[j]. Each pair adds the lower lane first. */
inline float wave_sum4(__m256 (&acc)[4]) {
    acc[0] = _mm256_add_ps(acc[0], acc[2]);
    acc[1] = _mm256_add_ps(acc[1], acc[3]);
    __m256 s = _mm256_add_ps(acc[0], acc[1]);
    s = _mm256_add_ps(s, _mm256_permute_ps(s, _MM_SHUFFLE(1, 0, 3, 2)));   /* lanes 0+4, 2+6, 1+5, 3+7 */
    s = _mm256_add_ps(s, _mm256_permute_ps(s, _MM_SHUFFLE(2, 3, 0, 1)));   /* 0+2, 1+3 */
    return _mm_cvtss_f32(_mm_add_ss(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1)));   /* 0+1 */
}

template <bool kGgml>
void gemv_avx2_t(const Mx& w, const uint16_t* x, uint16_t* y, int n0, int n1) {
    constexpr size_t BS = kGgml ? 17 : 16, SS = kGgml ? 17 : 1;
    const int UK = w.K / 32, R = (UK + 31) / 32, RF = UK / 32;   /* rounds, full rounds */
    struct alignas(32) X8 { float f[8]; };
    thread_local std::vector<X8> xbuf;
    xbuf.resize((size_t)R * 128);
    float* const xt = xbuf.data()->f;
    x_lanes(x, UK, xt);
    for (int n = n0; n < n1; ++n) {
        const uint8_t* const codes = kGgml ? w.blocks + (size_t)n * UK * 17 + 1 : w.codes + (size_t)n * (size_t)(w.K / 2);
        const uint8_t* const scl = kGgml ? codes - 1 : w.scales + (size_t)n * UK;
        __m256 acc[4] = { _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps() };
        for (int r = 0; r < RF; ++r) {
            const size_t u0 = (size_t)r * 32;
            const uint8_t* const gc[4] = { codes + u0 * BS, codes + (u0 + 8) * BS, codes + (u0 + 16) * BS, codes + (u0 + 24) * BS };
            const uint8_t* const gs[4] = { scl + u0 * SS, scl + (u0 + 8) * SS, scl + (u0 + 16) * SS, scl + (u0 + 24) * SS };
            add_any_round<kGgml>(acc, gc, gs, 4, xt + (size_t)r * 1024);
        }
        if (RF < R) add_tail<kGgml>(acc, codes, scl, 32 * RF, UK, xt + (size_t)RF * 1024);
        y[n] = bf16_rne_nan(wave_sum4(acc));
    }
}

/* ---- several token rows of one matrix ---- */

/* A full round for `rows` rows: on the fast path its codes are decoded once (kv_pair16) and added
 * to each row's sums as add_any_round adds them, so every row's sums see the same operations in the
 * same order. The general path decodes again for each row; its rounds are rare. */
template <bool kGgml>
__attribute__((noinline)) void add_round_rows(__m256 (*acc)[4], const uint8_t* const (&gc)[4], const uint8_t* const (&gs)[4],
                                              const float* const* xr, int rows) {
    const __m256i e01 = e_pair<kGgml>(gs[0], gs[1]), e23 = e_pair<kGgml>(gs[2], gs[3]);
    if (!fast_round(e01, e23)) {
        for (int i = 0; i < rows; ++i) general_round<kGgml>(acc[i], gc, gs, 4, xr[i]);
        return;
    }
    alignas(32) uint16_t w16[1024];
    kv_pair16<kGgml>(gc[0], gc[1], s8_of(e01), w16);
    kv_pair16<kGgml>(gc[2], gc[3], s8_of(e23), w16 + 512);
    for (int i = 0; i < rows; ++i) add_round16<4>(acc[i], w16, xr[i]);
}

/* gemv_avx2_t for 2 .. kGemvRows rows: x transposed once a row, the full rounds through
 * add_round_rows, each row's last, partial round (down's eight blocks) on its own. */
template <bool kGgml>
void gemv_avx2_rows_t(const Mx& w, const uint16_t* const* x, uint16_t* const* y, int rows, int n0, int n1) {
    constexpr size_t BS = kGgml ? 17 : 16, SS = kGgml ? 17 : 1;
    const int UK = w.K / 32, R = (UK + 31) / 32, RF = UK / 32;
    struct alignas(32) X8 { float f[8]; };
    thread_local std::vector<X8> xbuf;
    xbuf.resize((size_t)kGemvRows * R * 128);
    float* xt[kGemvRows];
    for (int i = 0; i < rows; ++i) {
        xt[i] = xbuf.data()->f + (size_t)i * R * 1024;
        x_lanes(x[i], UK, xt[i]);
    }
    for (int n = n0; n < n1; ++n) {
        const uint8_t* const codes = kGgml ? w.blocks + (size_t)n * UK * 17 + 1 : w.codes + (size_t)n * (size_t)(w.K / 2);
        const uint8_t* const scl = kGgml ? codes - 1 : w.scales + (size_t)n * UK;
        __m256 acc[kGemvRows][4];
        for (int i = 0; i < rows; ++i)
            for (int g = 0; g < 4; ++g) acc[i][g] = _mm256_setzero_ps();
        for (int r = 0; r < RF; ++r) {
            const size_t u0 = (size_t)r * 32;
            const uint8_t* const gc[4] = { codes + u0 * BS, codes + (u0 + 8) * BS, codes + (u0 + 16) * BS, codes + (u0 + 24) * BS };
            const uint8_t* const gs[4] = { scl + u0 * SS, scl + (u0 + 8) * SS, scl + (u0 + 16) * SS, scl + (u0 + 24) * SS };
            const float* xr[kGemvRows];
            for (int i = 0; i < rows; ++i) xr[i] = xt[i] + (size_t)r * 1024;
            add_round_rows<kGgml>(acc, gc, gs, xr, rows);
        }
        for (int i = 0; i < rows; ++i) {
            if (RF < R) add_tail<kGgml>(acc[i], codes, scl, 32 * RF, UK, xt[i] + (size_t)RF * 1024);
            y[i][n] = bf16_rne_nan(wave_sum4(acc[i]));
        }
    }
}

#undef HX_INLINE
#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

}  // namespace

uint16_t bf16_rne_nan(float f) {
    const uint32_t u = f_bits(f);
    const uint32_t r = u + 0x7fffu + ((u >> 16) & 1u), q = u | 0x400000u;
    return (uint16_t)((f != f ? q : r) >> 16);
}
uint16_t f32_to_bf16(float f) {
    const uint32_t u = f_bits(f);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
float bf16_to_f32(uint16_t h) { return bits_f((uint32_t)h << 16); }

void gemv_ref(const Mx& w, const uint16_t* x, uint16_t* y, int n0, int n1) {
    const int UK = w.K / 32;
    std::vector<float> xf((size_t)w.K);
    for (int k = 0; k < w.K; ++k) xf[(size_t)k] = bf16_to_f32(x[k]);
    for (int n = n0; n < n1; ++n) {
        const Row r = row_of(w, n);
        float lane[32];
        for (int l = 0; l < 32; ++l) {
            float acc = 0.f;
            for (int u = l; u < UK; u += 32) {
                const float d = e8m0_half(scale_of(r, u));
                for (int v = 0; v < 32; ++v) {
                    const float wv = kKv[code(r, u, v)] * d;
                    const float p = xf[(size_t)u * 32 + (size_t)v] * wv;
                    acc = acc + p;
                }
            }
            lane[l] = acc;
        }
        y[n] = bf16_rne_nan(wave_sum(lane));
    }
}

void gemv_avx2(const Mx& w, const uint16_t* x, uint16_t* y, int n0, int n1) {
    if (w.blocks) gemv_avx2_t<true>(w, x, y, n0, n1);
    else gemv_avx2_t<false>(w, x, y, n0, n1);
}

void gemv_avx2_rows(const Mx& w, const uint16_t* const* x, uint16_t* const* y, int rows, int n0, int n1) {
    for (int i = 0; i < rows; i += kGemvRows) {
        const int m = rows - i < kGemvRows ? rows - i : kGemvRows;
        if (m == 1) gemv_avx2(w, x[i], y[i], n0, n1);
        else if (w.blocks) gemv_avx2_rows_t<true>(w, x + i, y + i, m, n0, n1);
        else gemv_avx2_rows_t<false>(w, x + i, y + i, m, n0, n1);
    }
}

void swiglu(const float* silu, const uint16_t* gu, uint16_t* y, int n, float limit) {
    for (int i = 0; i < n; ++i) {
        const float u = std::fmin(std::fmax(bf16_to_f32(gu[n + i]), -limit), limit);
        y[i] = f32_to_bf16(silu[gu[i]] * u);
    }
}

void silu_table_host(float* table, float limit) {
    for (uint32_t h = 0; h < 65536; ++h) {
        const float g = std::fmin(bf16_to_f32((uint16_t)h), limit);
        table[h] = g / (1.0f + std::exp(-g));
    }
}

void expert_forward(const Expert& e, const float* silu, float limit, const uint16_t* x,
                    uint16_t* gate_up, uint16_t* act, uint16_t* down, int threads, bool avx2) {
    auto run = [&](const Mx& w, const uint16_t* in, uint16_t* out) {
        const int groups = w.N / 8, t = threads < 1 ? 1 : threads;
        auto part = [&](int i) {
            const int g0 = groups * i / t, g1 = groups * (i + 1) / t;
            if (avx2) gemv_avx2(w, in, out, g0 * 8, g1 * 8);
            else gemv_ref(w, in, out, g0 * 8, g1 * 8);
        };
        if (t == 1) { part(0); return; }
        std::vector<std::thread> pool;
        for (int i = 1; i < t; ++i) pool.emplace_back(part, i);
        part(0);
        for (auto& p : pool) p.join();
    };
    run(e.gu, x, gate_up);
    swiglu(silu, gate_up, act, e.dn.K, limit);
    run(e.dn, act, down);
}

}  // namespace ds41hx
