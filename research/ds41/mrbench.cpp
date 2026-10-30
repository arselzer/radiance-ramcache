/* mrbench.cpp -- what a token row more costs in the CPU expert pool's tiles. T threads take the
 * 64-row tiles of E distinct experts (gate_up then down, streamed from DRAM) from a shared counter,
 * R token rows a tile, as HostExpertPool::tile does. Variant "now": one gemv_avx2 call a row (the
 * pool today); variant "rows": each round's codes decoded once for all rows (a prototype). Checks
 * that both give the same bits for every row first.
 *
 *   mrbench [experts=96] [threads=40] [reps=3] */
#include "ds41_host_experts.cpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2,f16c")
#endif
namespace ds41hx {
namespace {

constexpr int kMR = 4;

template <bool kGgml>
__attribute__((noinline)) void add_any_round_rows(__m256 (*acc)[4], const uint8_t* const (&gc)[4], const uint8_t* const (&gs)[4],
                                                  int ng, const float* const* xr, int rows) {
    const __m256i e01 = e_pair<kGgml>(gs[0], gs[1]), e23 = ng > 2 ? e_pair<kGgml>(gs[2], gs[3]) : e01;
    auto in_range = [](__m256i e) __attribute__((always_inline)) {
        return _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_max_epu8(e, _mm256_set1_epi8((char)kFastLo)), e),
                                _mm256_cmpeq_epi8(_mm256_min_epu8(e, _mm256_set1_epi8((char)kFastHi)), e));
    };
    if (_mm256_movemask_epi8(_mm256_and_si256(in_range(e01), in_range(e23))) != -1) {
        for (int i = 0; i < rows; ++i) general_round<kGgml>(acc[i], gc, gs, ng, xr[i]);
        return;
    }
    auto s8 = [](__m256i e) __attribute__((always_inline)) {
        const __m256i t = _mm256_add_epi8(e, _mm256_set1_epi8(1)), t2 = _mm256_add_epi8(t, t);
        return _mm256_add_epi8(t2, t2);
    };
    alignas(32) uint16_t w16[1024];
    kv_pair16<kGgml>(gc[0], gc[1], s8(e01), w16);
    if (ng > 2) kv_pair16<kGgml>(gc[2], gc[3], s8(e23), w16 + 512);
    for (int i = 0; i < rows; ++i) switch (ng) {
        case 1: add_round16<1>(acc[i], w16, xr[i]); break;
        case 2: add_round16<2>(acc[i], w16, xr[i]); break;
        case 3: add_round16<3>(acc[i], w16, xr[i]); break;
        default: add_round16<4>(acc[i], w16, xr[i]); break;
    }
}

/* Variant 2: each weight vector converted once for every row. Groups G and G + 1 together, so 2 * NR
 * sums are in flight; each sum still adds values 0..31 in order. */
#define MR_INLINE inline __attribute__((always_inline))
template <int NR, int G>
MR_INLINE void add16_rows_gg(__m256 (*acc)[4], const uint16_t* w16, const float* const* xr) {
    __m256 a[NR], b[NR];
    for (int i = 0; i < NR; ++i) { a[i] = acc[i][G]; b[i] = acc[i][G + 1]; }
    const uint16_t* wa = w16 + (G >> 1) * 512 + (G & 1) * 8;
    const uint16_t* wb = w16 + ((G + 1) >> 1) * 512 + ((G + 1) & 1) * 8;
#pragma GCC unroll 1
    for (int v = 0; v < 32; ++v) {
        const __m256 va = _mm256_cvtph_ps(_mm_load_si128(reinterpret_cast<const __m128i*>(wa + v * 16)));
        const __m256 vb = _mm256_cvtph_ps(_mm_load_si128(reinterpret_cast<const __m128i*>(wb + v * 16)));
        for (int i = 0; i < NR; ++i) {
            a[i] = _mm256_add_ps(a[i], _mm256_mul_ps(_mm256_load_ps(xr[i] + v * 32 + G * 8), va));
            b[i] = _mm256_add_ps(b[i], _mm256_mul_ps(_mm256_load_ps(xr[i] + v * 32 + (G + 1) * 8), vb));
        }
    }
    for (int i = 0; i < NR; ++i) { acc[i][G] = a[i]; acc[i][G + 1] = b[i]; }
}
template <int NR>
__attribute__((noinline)) void add_round16_rows(__m256 (*acc)[4], const uint16_t* w16, const float* const* xr) {
    add16_rows_gg<NR, 0>(acc, w16, xr);
    add16_rows_gg<NR, 2>(acc, w16, xr);
}

template <bool kGgml>
__attribute__((noinline)) void add_any_round_rows2(__m256 (*acc)[4], const uint8_t* const (&gc)[4], const uint8_t* const (&gs)[4],
                                                   const float* const* xr, int rows) {
    const __m256i e01 = e_pair<kGgml>(gs[0], gs[1]), e23 = e_pair<kGgml>(gs[2], gs[3]);
    auto in_range = [](__m256i e) __attribute__((always_inline)) {
        return _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_max_epu8(e, _mm256_set1_epi8((char)kFastLo)), e),
                                _mm256_cmpeq_epi8(_mm256_min_epu8(e, _mm256_set1_epi8((char)kFastHi)), e));
    };
    if (_mm256_movemask_epi8(_mm256_and_si256(in_range(e01), in_range(e23))) != -1) {
        for (int i = 0; i < rows; ++i) general_round<kGgml>(acc[i], gc, gs, 4, xr[i]);
        return;
    }
    auto s8 = [](__m256i e) __attribute__((always_inline)) {
        const __m256i t = _mm256_add_epi8(e, _mm256_set1_epi8(1)), t2 = _mm256_add_epi8(t, t);
        return _mm256_add_epi8(t2, t2);
    };
    alignas(32) uint16_t w16[1024];
    kv_pair16<kGgml>(gc[0], gc[1], s8(e01), w16);
    kv_pair16<kGgml>(gc[2], gc[3], s8(e23), w16 + 512);
    switch (rows) {
        case 1: add_round16_rows<1>(acc, w16, xr); break;
        case 2: add_round16_rows<2>(acc, w16, xr); break;
        case 3: add_round16_rows<3>(acc, w16, xr); break;
        default: add_round16_rows<4>(acc, w16, xr); break;
    }
}

int g_variant = 1;

template <bool kGgml>
void gemv_avx2_rows_t(const Mx& w, const uint16_t* const* x, uint16_t* const* y, int rows, int n0, int n1) {
    constexpr size_t BS = kGgml ? 17 : 16, SS = kGgml ? 17 : 1;
    const int UK = w.K / 32, R = (UK + 31) / 32, RF = UK / 32;
    struct alignas(32) X8 { float f[8]; };
    thread_local std::vector<X8> xbuf;
    xbuf.resize((size_t)kMR * R * 128);
    float* xt[kMR];
    for (int i = 0; i < rows; ++i) { xt[i] = xbuf.data()->f + (size_t)i * R * 1024; x_lanes(x[i], UK, xt[i]); }
    for (int n = n0; n < n1; ++n) {
        const uint8_t* const codes = kGgml ? w.blocks + (size_t)n * UK * 17 + 1 : w.codes + (size_t)n * (size_t)(w.K / 2);
        const uint8_t* const scl = kGgml ? codes - 1 : w.scales + (size_t)n * UK;
        __m256 acc[kMR][4];
        for (int i = 0; i < rows; ++i) for (int g = 0; g < 4; ++g) acc[i][g] = _mm256_setzero_ps();
        for (int r = 0; r < RF; ++r) {
            const size_t u0 = (size_t)r * 32;
            const uint8_t* const gc[4] = { codes + u0 * BS, codes + (u0 + 8) * BS, codes + (u0 + 16) * BS, codes + (u0 + 24) * BS };
            const uint8_t* const gs[4] = { scl + u0 * SS, scl + (u0 + 8) * SS, scl + (u0 + 16) * SS, scl + (u0 + 24) * SS };
            const float* xr[kMR];
            for (int i = 0; i < rows; ++i) xr[i] = xt[i] + (size_t)r * 1024;
            if (g_variant == 2) add_any_round_rows2<kGgml>(acc, gc, gs, xr, rows);
            else add_any_round_rows<kGgml>(acc, gc, gs, 4, xr, rows);
        }
        for (int i = 0; i < rows; ++i) {
            if (RF < R) add_tail<kGgml>(acc[i], codes, scl, 32 * RF, UK, xt[i] + (size_t)RF * 1024);
            y[i][n] = bf16_rne_nan(wave_sum4(acc[i]));
        }
    }
}

}  // namespace
void gemv_avx2_rows(const Mx& w, const uint16_t* const* x, uint16_t* const* y, int rows, int n0, int n1) {
    if (w.blocks) gemv_avx2_rows_t<true>(w, x, y, rows, n0, n1);
    else gemv_avx2_rows_t<false>(w, x, y, rows, n0, n1);
}
}  // namespace ds41hx
#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

using namespace ds41hx;
constexpr int kEmbd = 5120, kFf = 2304, kTile = 64, kGuTiles = 2 * kFf / kTile, kDnTiles = kEmbd / kTile;

static std::vector<uint8_t> make_blocks(int N, int K, std::mt19937_64& rng) {
    std::vector<uint8_t> b((size_t)N * (K / 32) * 17);
    for (size_t i = 0; i < b.size(); i += 17) {
        const uint64_t r = rng(), s = rng(), t = rng();
        b[i] = (uint8_t)(113 + r % 15);
        std::memcpy(&b[i + 1], &s, 8);
        std::memcpy(&b[i + 9], &t, 8);
    }
    return b;
}
static std::vector<uint16_t> make_x(int K, std::mt19937_64& rng) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<uint16_t> x(K);
    for (auto& h : x) { const float f = nd(rng); uint32_t u; std::memcpy(&u, &f, 4); h = (uint16_t)(u >> 16); }
    return x;
}

int main(int argc, char** argv) {
    const int E = argc > 1 ? std::atoi(argv[1]) : 96, T = argc > 2 ? std::atoi(argv[2]) : 40, reps = argc > 3 ? std::atoi(argv[3]) : 3;
    std::mt19937_64 rng(42);
    std::vector<std::vector<uint8_t>> gu(E), dn(E);
    for (int e = 0; e < E; ++e) { gu[e] = make_blocks(2 * kFf, kEmbd, rng); dn[e] = make_blocks(kEmbd, kFf, rng); }
    std::vector<uint16_t> xg[kMR], xd[kMR];
    for (int i = 0; i < kMR; ++i) { xg[i] = make_x(kEmbd, rng); xd[i] = make_x(kFf, rng); }
    auto mat = [&](int e, bool g) { Mx m; m.N = g ? 2 * kFf : kEmbd; m.K = g ? kEmbd : kFf; m.blocks = g ? gu[e].data() : dn[e].data(); return m; };

    /* bits: every row of a few experts, both variants, 4 rows */
    size_t bad = 0;
    for (int e = 0; e < 3 && e < E; ++e)
        for (int g = 0; g < 2; ++g) {
            const Mx m = mat(e, g);
            std::vector<uint16_t> a[kMR], b[kMR];
            const uint16_t* xs[kMR]; uint16_t* ys[kMR];
            for (int i = 0; i < kMR; ++i) { a[i].assign(m.N, 0); b[i].assign(m.N, 1); xs[i] = g ? xg[i].data() : xd[i].data(); ys[i] = b[i].data(); }
            for (int i = 0; i < kMR; ++i) gemv_avx2(m, xs[i], a[i].data(), 0, m.N);
            for (int v = 1; v <= 2; ++v)
                for (int nr = 1; nr <= kMR; ++nr) {
                    ds41hx::g_variant = v;
                    for (int i = 0; i < kMR; ++i) b[i].assign(m.N, 1);
                    gemv_avx2_rows(m, xs, ys, nr, 0, m.N);
                    for (int i = 0; i < nr; ++i) bad += std::memcmp(a[i].data(), b[i].data(), m.N * 2) != 0;
                }
        }
    std::printf("bits: %s (%zu row mismatches)\n", bad ? "DIFFER" : "identical", bad);
    if (bad) return 1;

    const int tiles_per = kGuTiles + kDnTiles, total = E * tiles_per;
    auto run = [&](int rows, bool proto) {
        std::atomic<int> next{0};
        auto body = [&]() {
            std::vector<uint16_t> out[kMR];
            for (int i = 0; i < kMR; ++i) out[i].resize(kEmbd);
            const uint16_t* xs[kMR]; uint16_t* ys[kMR];
            for (;;) {
                const int t = next.fetch_add(1, std::memory_order_relaxed);
                if (t >= total) return;
                const int e = t / tiles_per, k = t % tiles_per;
                const bool g = k < kGuTiles;
                const int kk = g ? k : k - kGuTiles;
                const Mx m = mat(e, g);
                for (int i = 0; i < rows; ++i) { xs[i] = g ? xg[i].data() : xd[i].data(); ys[i] = out[i].data(); }
                if (proto) gemv_avx2_rows(m, xs, ys, rows, kk * kTile, (kk + 1) * kTile);
                else for (int i = 0; i < rows; ++i) gemv_avx2(m, xs[i], ys[i], kk * kTile, (kk + 1) * kTile);
            }
        };
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<std::thread> th;
        for (int i = 0; i < T; ++i) th.emplace_back(body);
        for (auto& x : th) x.join();
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / E;
    };
    run(1, false);   /* touch everything once */
    double base = 0;
    for (int rows = 1; rows <= kMR; ++rows)
        for (int proto = 0; proto < 3; ++proto) {
            if (rows == 1 && proto == 1) continue;
            ds41hx::g_variant = proto == 2 ? 2 : 1;
            double best = 1e30;
            for (int r = 0; r < reps; ++r) best = std::min(best, run(rows, proto));
            if (rows == 1) base = best;
            std::printf("rows %d %-4s %.3f ms an expert (%.2f of one row; a row more %.2f)\n", rows, proto == 2 ? "cvt1" : proto ? "dec1" : "now",
                        best, best / base, rows > 1 ? (best / base - 1) / (rows - 1) : 0.0);
        }
    return 0;
}
