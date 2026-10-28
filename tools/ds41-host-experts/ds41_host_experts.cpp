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

/* The 16 code bytes of block u of eight rows, transposed: out[p] (p = 0..15) holds byte p of
 * rows 0..7 in its low eight bytes. The codes are returned as int8 values kv[nibble] ready for
 * conversion, in the block's value order: kvi[v * 8 + i] is value v of row i. */
inline void block_kv8(const Row (&rows)[8], int u, int8_t* kvi, float* d) {
    const __m128i lut = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    const __m128i lo4 = _mm_set1_epi8(15);
    __m128i a[8];
    for (int i = 0; i < 8; ++i) {
        const uint8_t* p = rows[i].blk ? rows[i].blk + (size_t)u * 17 + 1 : rows[i].codes + (size_t)u * 16;
        a[i] = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
        d[i] = e8m0_half(scale_of(rows[i], u));
    }
    const __m128i t0 = _mm_unpacklo_epi8(a[0], a[1]), t1 = _mm_unpackhi_epi8(a[0], a[1]);
    const __m128i t2 = _mm_unpacklo_epi8(a[2], a[3]), t3 = _mm_unpackhi_epi8(a[2], a[3]);
    const __m128i t4 = _mm_unpacklo_epi8(a[4], a[5]), t5 = _mm_unpackhi_epi8(a[4], a[5]);
    const __m128i t6 = _mm_unpacklo_epi8(a[6], a[7]), t7 = _mm_unpackhi_epi8(a[6], a[7]);
    const __m128i s0 = _mm_unpacklo_epi16(t0, t2), s1 = _mm_unpackhi_epi16(t0, t2);
    const __m128i s2 = _mm_unpacklo_epi16(t4, t6), s3 = _mm_unpackhi_epi16(t4, t6);
    const __m128i s4 = _mm_unpacklo_epi16(t1, t3), s5 = _mm_unpackhi_epi16(t1, t3);
    const __m128i s6 = _mm_unpacklo_epi16(t5, t7), s7 = _mm_unpackhi_epi16(t5, t7);
    /* r[j]: byte 2j of the eight rows in the low half, byte 2j + 1 in the high half. */
    const __m128i r[8] = { _mm_unpacklo_epi32(s0, s2), _mm_unpackhi_epi32(s0, s2),
                           _mm_unpacklo_epi32(s1, s3), _mm_unpackhi_epi32(s1, s3),
                           _mm_unpacklo_epi32(s4, s6), _mm_unpackhi_epi32(s4, s6),
                           _mm_unpacklo_epi32(s5, s7), _mm_unpackhi_epi32(s5, s7) };
    const bool ggml = rows[0].blk != nullptr;
    for (int j = 0; j < 8; ++j) {
        const __m128i lo = _mm_shuffle_epi8(lut, _mm_and_si128(r[j], lo4));
        const __m128i hi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(r[j], 4), lo4));
        const int p0 = 2 * j, p1 = 2 * j + 1;   /* the two byte positions in r[j] */
        if (ggml) {
            /* value p (< 16) is byte p's low nibble, value 16 + p its high nibble */
            _mm_storel_epi64(reinterpret_cast<__m128i*>(kvi + p0 * 8), lo);
            _mm_storel_epi64(reinterpret_cast<__m128i*>(kvi + p1 * 8), _mm_unpackhi_epi64(lo, lo));
            _mm_storel_epi64(reinterpret_cast<__m128i*>(kvi + (16 + p0) * 8), hi);
            _mm_storel_epi64(reinterpret_cast<__m128i*>(kvi + (16 + p1) * 8), _mm_unpackhi_epi64(hi, hi));
        } else {
            /* planes: value 2p is byte p's low nibble, value 2p + 1 its high nibble */
            _mm_storel_epi64(reinterpret_cast<__m128i*>(kvi + (2 * p0) * 8), lo);
            _mm_storel_epi64(reinterpret_cast<__m128i*>(kvi + (2 * p0 + 1) * 8), hi);
            _mm_storel_epi64(reinterpret_cast<__m128i*>(kvi + (2 * p1) * 8), _mm_unpackhi_epi64(lo, lo));
            _mm_storel_epi64(reinterpret_cast<__m128i*>(kvi + (2 * p1 + 1) * 8), _mm_unpackhi_epi64(hi, hi));
        }
    }
}

inline __m256 kv_to_w(const int8_t* kv8, __m256 d) {
    const __m128i b = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(kv8));
    return _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b)), d);   /* exact */
}

/* r4d_wave_sum over 32 lane partials: xor 16, then 8, 4, 2, 1. Only lane 0's result is kept, and
 * every lane's is the same because each pair is added in both orders, which is exact. */
inline float wave_sum(float (&p)[32]) {
    for (int o = 16; o; o >>= 1)
        for (int l = 0; l < o; ++l) p[l] = p[l] + p[l + o];
    return p[0];
}
inline __m256 wave_sum8(__m256 (&p)[32]) {
    for (int o = 16; o; o >>= 1)
        for (int l = 0; l < o; ++l) p[l] = _mm256_add_ps(p[l], p[l + o]);
    return p[0];
}

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
    const int UK = w.K / 32;
    thread_local std::vector<float> xf;
    xf.resize((size_t)w.K);
    for (int k = 0; k < w.K; ++k) xf[(size_t)k] = bf16_to_f32(x[k]);
    alignas(32) int8_t kvi[4][32 * 8];
    alignas(32) float dsc[4][8];
    for (int n = n0; n < n1; n += 8) {
        Row rows[8];
        for (int i = 0; i < 8; ++i) rows[i] = row_of(w, n + i);
        __m256 lane[32];
        /* Four adjacent lanes at once: four independent sums, so the adds do not wait on each
         * other. Each lane still sees its own blocks in increasing order. */
        for (int l0 = 0; l0 < 32; l0 += 4) {
            __m256 acc[4] = { _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
                              _mm256_setzero_ps() };
            for (int base = l0; base < UK; base += 32) {
                int live = 0;
                for (int i = 0; i < 4 && base + i < UK; ++i, ++live) block_kv8(rows, base + i, kvi[i], dsc[i]);
                if (live == 4) {
                    const __m256 d0 = _mm256_load_ps(dsc[0]), d1 = _mm256_load_ps(dsc[1]);
                    const __m256 d2 = _mm256_load_ps(dsc[2]), d3 = _mm256_load_ps(dsc[3]);
                    const float* x0 = &xf[(size_t)base * 32];
                    for (int v = 0; v < 32; ++v) {
                        acc[0] = _mm256_add_ps(acc[0], _mm256_mul_ps(_mm256_broadcast_ss(x0 + v), kv_to_w(kvi[0] + v * 8, d0)));
                        acc[1] = _mm256_add_ps(acc[1], _mm256_mul_ps(_mm256_broadcast_ss(x0 + 32 + v), kv_to_w(kvi[1] + v * 8, d1)));
                        acc[2] = _mm256_add_ps(acc[2], _mm256_mul_ps(_mm256_broadcast_ss(x0 + 64 + v), kv_to_w(kvi[2] + v * 8, d2)));
                        acc[3] = _mm256_add_ps(acc[3], _mm256_mul_ps(_mm256_broadcast_ss(x0 + 96 + v), kv_to_w(kvi[3] + v * 8, d3)));
                    }
                } else {
                    for (int i = 0; i < live; ++i) {
                        const __m256 d = _mm256_load_ps(dsc[i]);
                        const float* xi = &xf[(size_t)(base + i) * 32];
                        for (int v = 0; v < 32; ++v)
                            acc[i] = _mm256_add_ps(acc[i], _mm256_mul_ps(_mm256_broadcast_ss(xi + v), kv_to_w(kvi[i] + v * 8, d)));
                    }
                }
            }
            for (int i = 0; i < 4; ++i) lane[l0 + i] = acc[i];
        }
        alignas(32) float out[8];
        _mm256_store_ps(out, wave_sum8(lane));
        for (int i = 0; i < 8; ++i) y[n + i] = bf16_rne_nan(out[i]);
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
