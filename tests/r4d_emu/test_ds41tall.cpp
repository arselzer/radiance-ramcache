/* tests/r4d_emu: DeepSeek-V4.1's kernels whose arithmetic follows the pass height, on the host SIMT
 * emulator, for RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL (libr4d/r4d_ds41_tall.h).
 *
 * The routed MXFP4 GEMM (moe_gemm_ggml type=mxfp4, r4d_ds41_mxfp4.hip): today a step of at most 16
 * tokens runs the decode band's kernel (moe_gemm_ggml_mxfp4_gemv) and a taller one the tiled row,
 * which sums each output in another order. Under the switch libr4d exports the twin
 * moe_gemm_ggml_mxfp4_gemv_tall (r4d_rows.cpp), the gemv at more tokens, so that a tall pass gives
 * the bits of the same tokens run as 16-token passes, and the CPU path's rows (the gemv's bits) stay
 * exact. This holds the twin's launch at T tokens to the gemv's launches over the same tokens cut into
 * passes of at most 16 (each with its own routing: sorted order and offsets as the scatter writes them
 * for that pass), (token, slot) row by row, byte for byte, for the gate_up (token rows) and the down
 * projection (sorted rows), at the expert shapes that take each of the kernel's row staging paths: a
 * whole K of 5120 for one row at a time (XLM 1, ds41's gate_up), three at K 2304 (ds41's down), eight
 * at a short K, and two weight rows a wave on a wide expert. Routings: tokens picking from a small
 * pool of the 384 experts (long runs, up to every token on one expert) and from all of them, with the
 * other ranks' slots dropped past the local ones as on an EP4 rank. Heights 17 to 48 and 64 (the
 * launch has no bound; the row's is libr4d's reach).
 *
 * The norms (r4d_ds41_elem.hip's fused mhc_collapse_rmsnorm, mhc_post_collapse_rmsnorm and
 * rmsnorm_rope_roundtrip, which take rmsnorm_bf16's width from r4d_act_threads): each row at pass
 * heights 1 to 48 against the same row launched alone and in launches of 16. Up to libr4d's reach
 * (32 without the switch, where the width stays 1024; kR4dDs41TallRows with it) every row must keep
 * its bytes; above it a row may differ, and the test counts how many did. */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"
#include "r4d_ds41_tall.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

extern const char* emu_last_kernel;

namespace {

constexpr int kNe = 96, kTop = 6;

RadTensor tens(void* p, uint32_t dt, std::vector<int64_t> shape) {
    RadTensor t{};
    t.data = p;
    t.dtype = dt;
    t.rank = (uint32_t)shape.size();
    int64_t st = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
        t.shape[i] = shape[i];
        t.stride[i] = st;
        st *= shape[i];
    }
    return t;
}
RadParam pint(const char* k, long long v) { return { k, RAD_P_INT, v, 0, nullptr, 0.0 }; }
RadParam pstr(const char* k, const char* v) { return { k, RAD_P_STR, 0, 0, v, 0.0 }; }

uint16_t bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

/* An expert's [N, K] weight in llama.cpp's MXFP4 blocks (17 bytes per 32 values): e8m0 around 2^-8,
 * every fifth block from the edges (the subnormal half-scales 0 and 1, large exponents). */
std::vector<uint8_t> make_expert(int N, int K, std::mt19937_64& rng) {
    static const uint8_t edge[] = { 0, 1, 2, 3, 100, 126, 127, 128, 129, 135, 140 };
    const size_t blocks = (size_t)N * (K / 32);
    std::vector<uint8_t> b(blocks * 17);
    for (size_t i = 0; i < blocks; ++i) {
        const uint64_t r = rng(), a = rng();
        b[i * 17] = r % 5 == 0 ? edge[(r >> 8) % sizeof edge] : (uint8_t)(116 + (r >> 16) % 12);
        for (int j = 0; j < 16; ++j) b[i * 17 + 1 + j] = (uint8_t)(a >> (4 * j));
    }
    return b;
}

/* One pass's routing on this rank, as moe_scatter writes it: each local expert's slots (flat index
 * token * top_k + slot, ascending) grouped by expert in ascending order, then -1 for the slots that
 * went to other ranks; off[e] where expert e's run starts, off[kNe] the local count. */
struct Routing {
    std::vector<int32_t> sorted, off;
    int local = 0, longest = 0;
};
Routing route(const std::vector<std::vector<int>>& picks, int t0, int t1) {
    Routing r;
    std::vector<std::vector<int32_t>> by(kNe);
    for (int t = t0; t < t1; ++t)
        for (int k = 0; k < kTop; ++k) {
            const int e = picks[(size_t)t][(size_t)k];
            if (e < kNe) by[(size_t)e].push_back((t - t0) * kTop + k);
        }
    r.off.assign(kNe + 1, 0);
    for (int e = 0; e < kNe; ++e) {
        r.off[(size_t)e + 1] = r.off[(size_t)e] + (int)by[(size_t)e].size();
        for (int32_t f : by[(size_t)e]) r.sorted.push_back(f);
        r.longest = std::max(r.longest, (int)by[(size_t)e].size());
    }
    r.local = (int)r.sorted.size();
    r.sorted.resize((size_t)(t1 - t0) * kTop, -1);
    return r;
}

typedef int (*Launch)(const RadArgs*, RadStream);

/* moe_gemm_ggml type=mxfp4 over a table of expert pointers: `AR` activation rows of K, the pass's
 * sorted order (T slots) and offsets, y [T, N]. */
int run(Launch f, const uint16_t* x, int AR, const uint8_t* const* table, int N, int K,
        const Routing& r, uint16_t* y, const char* order) {
    const int T = (int)r.sorted.size();
    RadTensor t[5] = { tens((void*)x, RAD_BF16, { AR, K }),
                       tens((void*)table, RAD_U8, { kNe, N, K / 32 * 17 }),
                       tens((void*)r.sorted.data(), RAD_I32, { T }),
                       tens((void*)r.off.data(), RAD_I32, { kNe + 1 }),
                       tens(y, RAD_BF16, { T, N }) };
    t[1].stride[0] = 0;   /* a table of expert pointers */
    RadParam p[6] = { pstr("type", "mxfp4"), pint("N", N), pint("K", K), pint("n_expert", kNe),
                      pint("top_k", kTop), pstr("a_order", order) };
    RadArgs a{};
    a.t = t;
    a.n_t = 5;
    a.p = p;
    a.n_p = 6;
    return f(&a, nullptr);
}

/* The sorted row of (token, slot) in a pass's routing, or -1 when the slot is another rank's. */
int row_of(const Routing& r, int tok, int k) {
    for (int i = 0; i < r.local; ++i)
        if (r.sorted[(size_t)i] == tok * kTop + k) return i;
    return -1;
}

/* An expert: gate_up [2 ff, embd] over token rows of embd, down [embd, ff] over sorted rows of ff;
 * `projs` the projections launched (1 the gate_up, 2 the down). */
struct Shape {
    const char* name;
    int ff, embd, projs;
};

/* T tokens picking from a pool of P of the 384 experts; the shape's projections at the tall height
 * and as passes of at most 16 tokens. Returns the number of differing (token, slot) rows. */
int check(const Shape& sh, int T, int P, uint32_t seed, int* rows_checked) {
    std::mt19937_64 rng(seed);
    std::vector<int> pool;
    while ((int)pool.size() < std::min(P, 4 * kNe)) {
        const int e = (int)(rng() % (4 * kNe));
        if (std::find(pool.begin(), pool.end(), e) == pool.end()) pool.push_back(e);
    }
    /* Every token takes a local expert of the pool first (pool[0] is made local), so a small pool
     * gives one expert a run of every token of the pass. */
    for (int& e : pool)
        if (e < kNe) { std::swap(e, pool[0]); break; }
    if (pool[0] >= kNe) pool[0] = (int)(rng() % kNe);
    std::vector<std::vector<int>> picks((size_t)T);
    for (int t = 0; t < T; ++t) {
        std::vector<int>& pk = picks[(size_t)t];
        pk.push_back(pool[0]);
        while ((int)pk.size() < kTop) {
            const int e = (int)pool.size() > kTop ? pool[(size_t)(rng() % pool.size())] : (int)(rng() % (4 * kNe));
            if (std::find(pk.begin(), pk.end(), e) == pk.end()) pk.push_back(e);
        }
        std::shuffle(pk.begin(), pk.end(), rng);
    }
    /* The experts, and the table: an expert with no rows points at expert 0's bytes, as a dropped
     * or aliased unit points at some readable weight. */
    std::vector<std::vector<uint8_t>> gu(kNe), dn(kNe);
    std::vector<const uint8_t*> tgu(kNe), tdn(kNe);
    const Routing all = route(picks, 0, T);
    for (int e = 0; e < kNe; ++e)
        if (all.off[(size_t)e + 1] > all.off[(size_t)e] || e == 0) {
            gu[(size_t)e] = make_expert(2 * sh.ff, sh.embd, rng);
            dn[(size_t)e] = make_expert(sh.embd, sh.ff, rng);
        }
    for (int e = 0; e < kNe; ++e) {
        tgu[(size_t)e] = gu[(size_t)e].empty() ? gu[0].data() : gu[(size_t)e].data();
        tdn[(size_t)e] = dn[(size_t)e].empty() ? dn[0].data() : dn[(size_t)e].data();
    }
    /* Token rows for the gate_up (bf16 near 0.5, a few signed zeros and subnormals), and a row per
     * (token, slot) for the down projection, the gate_up's output width halved as SwiGLU leaves it. */
    std::normal_distribution<float> nd(0.f, 0.5f);
    std::vector<uint16_t> x((size_t)T * sh.embd);
    for (auto& v : x) {
        v = bf16(nd(rng));
        if (rng() % 97 == 0) v = rng() % 2 ? 0x8000 : 0x0001;
    }
    std::vector<uint16_t> act((size_t)T * kTop * sh.ff);
    for (auto& v : act) v = bf16(0.25f * nd(rng));

    int bad = 0;
    for (int proj = 0; proj < 2; ++proj) {
        if (!((sh.projs >> proj) & 1)) continue;
        const bool g = proj == 0;
        const int N = g ? 2 * sh.ff : sh.embd, K = g ? sh.embd : sh.ff;
        const auto& table = g ? tgu : tdn;
        /* The tall pass: every token at once, on the twin. */
        std::vector<uint16_t> a_all, y_all((size_t)T * kTop * N, 0xAAAA);
        if (!g) {   /* sorted rows: each slot's own row */
            a_all.assign((size_t)T * kTop * K, 0);
            for (int i = 0; i < all.local; ++i)
                std::memcpy(&a_all[(size_t)i * K], &act[(size_t)all.sorted[(size_t)i] * K], (size_t)K * 2);
        }
        if (run(r4d_rad_moe_gemm_mxfp4_gemv_tall, g ? x.data() : a_all.data(), g ? T : T * kTop,
                table.data(), N, K, all, y_all.data(), g ? "token" : "sorted") != RAD_OK) {
            std::printf("FAIL %s T=%d P=%d %s: the tall launch refused\n", sh.name, T, P, g ? "gate_up" : "down");
            return 1 << 20;
        }
        if (std::strstr(emu_last_kernel, "mx_gemv_kernel") == nullptr) {
            std::printf("FAIL %s T=%d %s: the tall launch ran %s, not the gemv\n", sh.name, T,
                        g ? "gate_up" : "down", emu_last_kernel);
            return 1 << 20;
        }
        /* The same tokens as passes of at most 16, each with its own routing, on the gemv row. */
        for (int t0 = 0; t0 < T; t0 += 16) {
            const int t1 = std::min(T, t0 + 16), n = t1 - t0;
            const Routing rc = route(picks, t0, t1);
            std::vector<uint16_t> a_c, y_c((size_t)n * kTop * N, 0x5555);
            if (!g) {
                a_c.assign((size_t)n * kTop * K, 0);
                for (int i = 0; i < rc.local; ++i) {
                    const int f = rc.sorted[(size_t)i] + t0 * kTop;   /* the flat index of the whole pass */
                    std::memcpy(&a_c[(size_t)i * K], &act[(size_t)f * K], (size_t)K * 2);
                }
            }
            if (run(r4d_rad_moe_gemm_mxfp4_gemv, g ? x.data() + (size_t)t0 * sh.embd : a_c.data(),
                    g ? n : n * kTop, table.data(), N, K, rc, y_c.data(), g ? "token" : "sorted") != RAD_OK) {
                std::printf("FAIL %s T=%d: a 16-token launch refused\n", sh.name, T);
                return 1 << 20;
            }
            for (int t = t0; t < t1; ++t)
                for (int k = 0; k < kTop; ++k) {
                    const int ra = row_of(all, t, k), rb = row_of(rc, t - t0, k);
                    if ((ra < 0) != (rb < 0)) {
                        std::printf("FAIL %s: token %d slot %d is routed in one layout only\n", sh.name, t, k);
                        return 1 << 20;
                    }
                    if (ra < 0) continue;
                    ++*rows_checked;
                    const uint16_t* ya = &y_all[(size_t)ra * N];
                    const uint16_t* yb = &y_c[(size_t)rb * N];
                    if (std::memcmp(ya, yb, (size_t)N * 2) != 0) {
                        int first = 0;
                        while (ya[first] == yb[first]) ++first;
                        if (bad < 6)
                            std::printf("FAIL %s T=%d P=%d %s: token %d slot %d (tall row %d, 16-token row %d) "
                                        "differs first at column %d: 0x%04x tall, 0x%04x in passes of 16\n",
                                        sh.name, T, P, g ? "gate_up" : "down", t, k, ra, rb, first, ya[first],
                                        yb[first]);
                        ++bad;
                    }
                }
        }
        /* The other ranks' slots past the local ones are rows of zeros, as the gemv writes them. */
        for (size_t i = (size_t)all.local * N; i < y_all.size(); ++i)
            if (y_all[i] != 0) {
                std::printf("FAIL %s T=%d %s: a dropped slot's row holds 0x%04x\n", sh.name, T,
                            g ? "gate_up" : "down", y_all[i]);
                ++bad;
                break;
            }
    }
    return bad;
}

/* ---- the norms ---------------------------------------------------------------------------------- */

constexpr int kNormRows = 48;

/* One fused norm over kNormRows rows: `launch(r0, n)` runs rows [r0, r0 + n) as one pass and writes
 * their outputs, which `outs` (buffer, bytes a row) hold row-major. */
struct NormCase {
    const char* name;
    std::vector<std::pair<std::vector<uint8_t>*, size_t>> outs;
    std::function<int(int, int)> launch;
};

RadTensor rows_of(void* base, uint32_t dt, int64_t cols, int r0, int n) {
    const int es = dt == RAD_F32 || dt == RAD_I32 ? 4 : 2;
    return tens((char*)base + (size_t)r0 * cols * es, dt, { n, cols });
}
RadParam pf64(const char* k, double v) { return { k, RAD_P_F64, 0, 0, nullptr, v }; }

/* Every output row after the launches `runs`, outputs poisoned first; empty on a refused launch. */
std::vector<std::vector<uint8_t>> norm_run(NormCase& c, const std::vector<std::pair<int, int>>& runs) {
    for (auto& o : c.outs) std::memset(o.first->data(), 0xA5, o.first->size());
    for (const auto& r : runs)
        if (c.launch(r.first, r.second) != RAD_OK) return {};
    std::vector<std::vector<uint8_t>> rows(kNormRows);
    for (auto& o : c.outs)
        for (int m = 0; m < kNormRows; ++m)
            rows[(size_t)m].insert(rows[(size_t)m].end(), o.first->begin() + (size_t)m * o.second,
                                   o.first->begin() + (size_t)(m + 1) * o.second);
    return rows;
}

/* Heights 1..48 against rows alone and launches of 16. Returns the failures (a differing row at or
 * below `reach`); *above counts the rows that differed above it. */
int norm_heights(NormCase& c, int reach, int* above) {
    std::vector<std::pair<int, int>> one, sixteen;
    for (int m = 0; m < kNormRows; ++m) one.push_back({ m, 1 });
    for (int m = 0; m < kNormRows; m += 16) sixteen.push_back({ m, std::min(16, kNormRows - m) });
    const auto ref1 = norm_run(c, one), ref16 = norm_run(c, sixteen);
    if (ref1.empty() || ref16.empty()) {
        std::printf("FAIL %s: a launch was refused\n", c.name);
        return 1;
    }
    int fails = 0;
    for (int m = 0; m < kNormRows; ++m)
        if (ref1[(size_t)m] != ref16[(size_t)m]) {
            std::printf("FAIL %s: row %d differs between a launch of 1 and of 16\n", c.name, m);
            ++fails;
        }
    for (int M : { 1, 16, 17, 24, 25, 32, 33, 40, 48 }) {
        const auto got = norm_run(c, { { 0, M } });
        if (got.empty()) {
            std::printf("FAIL %s: the launch of %d rows was refused\n", c.name, M);
            ++fails;
            continue;
        }
        int d = 0;
        for (int m = 0; m < M; ++m) d += got[(size_t)m] != ref1[(size_t)m];
        if (!d) continue;
        if (M <= reach) {
            std::printf("FAIL %s: %d of %d rows differ in a pass of %d (the reach is %d)\n", c.name, d, M, M,
                        reach);
            ++fails;
        } else {
            *above += d;
        }
    }
    return fails;
}

int norms(int* above_out) {
    const auto t0 = std::chrono::steady_clock::now();
    const int reach = r4d_ds41_exact_tall() ? kR4dDs41TallRows : 32;
    const int n = 5120, hc = 4, rot = 64, P = 512;
    std::mt19937_64 rng(77);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto bf = [&](size_t count, float sd) {
        std::vector<uint8_t> v(count * 2);
        for (size_t i = 0; i < count; ++i) {
            const uint16_t b = bf16(sd * nd(rng));
            std::memcpy(&v[i * 2], &b, 2);
        }
        return v;
    };
    auto f32 = [&](size_t count, float sd, float mean, bool positive) {
        std::vector<float> v(count);
        for (auto& x : v) { x = mean + sd * nd(rng); if (positive) x = std::fabs(x) + 0.05f; }
        return v;
    };
    int fails = 0, above = 0;
    for (int trial = 0; trial < 3; ++trial) {
        /* mhc_collapse_rmsnorm: x = sum_j pre[j] h[j] (stored), y = rmsnorm(x). */
        {
            auto h = bf((size_t)kNormRows * hc * n, 1.f);
            auto pre = f32((size_t)kNormRows * hc, 0.4f, 0.f, true);
            auto w = f32((size_t)n, 0.2f, 1.f, false);
            std::vector<uint8_t> x((size_t)kNormRows * n * 2), y((size_t)kNormRows * n * 2);
            NormCase c{ "mhc_collapse_rmsnorm n 5120", { { &x, (size_t)n * 2 }, { &y, (size_t)n * 2 } }, nullptr };
            c.launch = [&](int r0, int m) {
                RadTensor t[5] = { rows_of(h.data(), RAD_BF16, hc * n, r0, m), rows_of(pre.data(), RAD_F32, hc, r0, m),
                                   rows_of(x.data(), RAD_BF16, n, r0, m), tens(w.data(), RAD_F32, { n }),
                                   rows_of(y.data(), RAD_BF16, n, r0, m) };
                RadParam p[4] = { pint("n", n), pint("hc", hc), pf64("eps", 1e-20), pstr("dtype", "bf16") };
                RadArgs a{};
                a.t = t; a.n_t = 5; a.p = p; a.n_p = 4;
                return r4d_rad_mhc_collapse_rmsnorm(&a, nullptr);
            };
            fails += norm_heights(c, reach, &above);
        }
        /* mhc_post_collapse_rmsnorm: out = post x + comb^T h, x' = sum_j pre[j] out[j], y = rmsnorm(x'). */
        {
            auto xi = bf((size_t)kNormRows * n, 1.f), h = bf((size_t)kNormRows * hc * n, 1.f);
            auto post = f32((size_t)kNormRows * hc, 0.5f, 0.f, true), comb = f32((size_t)kNormRows * hc * hc, 0.3f, 0.f, true);
            auto pre = f32((size_t)kNormRows * hc, 0.4f, 0.f, true), w = f32((size_t)n, 0.2f, 1.f, false);
            std::vector<uint8_t> out((size_t)kNormRows * hc * n * 2), x((size_t)kNormRows * n * 2),
                y((size_t)kNormRows * n * 2);
            NormCase c{ "mhc_post_collapse_rmsnorm n 5120",
                        { { &out, (size_t)hc * n * 2 }, { &x, (size_t)n * 2 }, { &y, (size_t)n * 2 } }, nullptr };
            c.launch = [&](int r0, int m) {
                RadTensor t[9] = { rows_of(xi.data(), RAD_BF16, n, r0, m), rows_of(h.data(), RAD_BF16, hc * n, r0, m),
                                   rows_of(post.data(), RAD_F32, hc, r0, m), rows_of(comb.data(), RAD_F32, hc * hc, r0, m),
                                   rows_of(out.data(), RAD_BF16, hc * n, r0, m), rows_of(pre.data(), RAD_F32, hc, r0, m),
                                   rows_of(x.data(), RAD_BF16, n, r0, m), tens(w.data(), RAD_F32, { n }),
                                   rows_of(y.data(), RAD_BF16, n, r0, m) };
                RadParam p[4] = { pint("n", n), pint("hc", hc), pf64("eps", 1e-20), pstr("dtype", "bf16") };
                RadArgs a{};
                a.t = t; a.n_t = 9; a.p = p; a.n_p = 4;
                return r4d_rad_mhc_post_collapse_rmsnorm(&a, nullptr);
            };
            fails += norm_heights(c, reach, &above);
        }
        /* rmsnorm_rope_roundtrip: the window key (512, E4M3) and the index key (128, E2M1). */
        for (int k = 0; k < 2; ++k) {
            const int kn = k ? 128 : 512;
            auto x = bf((size_t)kNormRows * kn, 1.f);
            auto w = f32((size_t)kn, 0.2f, 1.f, false);
            std::vector<float> cs((size_t)P * rot);
            for (int q = 0; q < P; ++q)
                for (int i = 0; i < rot / 2; ++i) {
                    const double ang = q * std::pow(10000.0, -2.0 * i / rot);
                    cs[(size_t)q * rot + i] = (float)std::cos(ang);
                    cs[(size_t)q * rot + rot / 2 + i] = (float)std::sin(ang);
                }
            std::vector<int32_t> pos(kNormRows);
            for (int m = 0; m < kNormRows; ++m) pos[(size_t)m] = 333 + m;
            std::vector<uint8_t> y((size_t)kNormRows * kn * 2);
            NormCase c{ k ? "rmsnorm_rope_roundtrip n 128 E2M1" : "rmsnorm_rope_roundtrip n 512 E4M3",
                        { { &y, (size_t)kn * 2 } }, nullptr };
            c.launch = [&](int r0, int m) {
                RadTensor t[5] = { rows_of(x.data(), RAD_BF16, kn, r0, m), tens(w.data(), RAD_F32, { kn }),
                                   rows_of(y.data(), RAD_BF16, kn, r0, m), tens(cs.data(), RAD_F32, { P, rot }),
                                   tens(pos.data() + r0, RAD_I32, { m }) };
                RadParam p[12] = { pint("n", kn), pf64("eps", 1e-20), pint("n_head", 1), pint("head_dim", kn),
                                   pint("rot", rot), pint("off", kn - rot), pstr("mode", "gptj"), pint("inverse", 0),
                                   pint("block", 32), pstr("format", k ? "fp4_e2m1" : "fp8_e4m3"),
                                   pstr("scale", "ue8m0"), pstr("dtype", "bf16") };
                RadArgs a{};
                a.t = t; a.n_t = 5; a.p = p; a.n_p = 12;
                return r4d_rad_rmsnorm_rope_roundtrip(&a, nullptr);
            };
            fails += norm_heights(c, reach, &above);
        }
    }
    std::printf("%s: norms at pass heights 1..48 against rows alone and launches of 16, every row's bytes "
                "kept up to %d rows (%s); %d rows differed above it (%.1f s)\n", fails ? "FAIL" : "PASS", reach,
                r4d_ds41_exact_tall() ? "RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL=1: libr4d's reach"
                                      : "no switch: the width r4d_act_threads keeps to 32 rows", above,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    *above_out = above;
    return fails;
}

}  // namespace

int main() {
    /* ds41's experts are ff 2304, embd 5120. Each shape keeps one of the two K and launches that
     * projection only: the gate_up's 5120 (one activation row staged at a time) or the down's 2304
     * (three), the other side cut short, which changes no output's sum (each is its own weight
     * row's), and whose projection would have the other's wide N (an emulated wave sum an output);
     * and a short K (eight rows staged), both projections, the gate_up wide enough for two weight
     * rows a wave. */
    const Shape shapes[] = {
        { "gate_up K 5120 (one row staged at a time)", 32, 5120, 1 },
        { "down K 2304 (three rows staged)", 2304, 64, 2 },
        { "K 512 (eight rows staged), two weight rows a wave", 512, 512, 3 },
    };
    /* Heights by shape: the gate_up's K takes the most (one row staged at a time); the others one
     * height over 16 and the CPU path's largest. */
    const std::vector<int> heights[3] = { { 17, 31, 48, 64 }, { 17, 48 }, { 33 } };
    int fails = 0, rows = 0, cases = 0;
    for (int si = 0; si < 3; ++si)
        for (int T : heights[si])
            for (int P : { 9, 384 }) {
                const Shape& sh = shapes[si];
                const auto t0 = std::chrono::steady_clock::now();
                const int r0 = rows;
                const int bad = check(sh, T, P, 1000u * (uint32_t)T + (uint32_t)P + 7u * (uint32_t)si, &rows);
                std::printf("  %s T=%d P=%d: %d rows, %.1f s\n", sh.name, T, P, rows - r0,
                            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
                ++cases;
                if (bad) {
                    std::printf("FAIL %s T=%d P=%d: %d differing rows\n", sh.name, T, P, bad);
                    ++fails;
                }
            }
    std::printf("%s: %d cases, %d (token, slot) rows compared, tall pass against passes of 16\n",
                fails ? "FAIL" : "PASS", cases, rows);
    int above = 0;
    fails += norms(&above);
    return fails ? 1 : 0;
}
