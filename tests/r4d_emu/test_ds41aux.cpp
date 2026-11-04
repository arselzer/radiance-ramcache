/* tests/r4d_emu: the aux expert worker's launches on the host SIMT emulator (core/place/aux_expert_worker.h;
 * work/aux/SPEC.md §15.5; on a card, ds41_mxfp4_device_check's `aux` mode).
 *
 * The worker relaunches a rank's decode chain for the rows of the units it holds: gate_up and down by the gemv_vp
 * row (moe_gemm_ggml_mxfp4_gemv_vp, r4d_ds41_mxfp4.hip) over packed weights (entries marked by bit 0),
 * swiglu_clamped between, over the slots tools/ds41-host-experts/ds41_aux.h's compact keeps for its experts: their
 * live slots in order, every other expert's range empty, T' rows instead of the pass's tokens * 6, reading the
 * pass's token rows (gate_up, token order) or its own SwiGLU's rows (down, sorted order). This holds, byte for byte:
 *
 *   projections  at M 1/4/8/16 for uniform routings (top-6 of 384, one EP4 rank's slots as moe_scatter writes
 *                them), every token on the rank's first and last experts ("edge"), and every slot local ("local");
 *                for the subsets one, every other, all and the last live expert, each compacted row equals the full
 *                gemv_vp launch's row (every byte, NaN payloads too: the same kernel on the same row) and the plain
 *                gemv's over block_mxfp4 (a NaN equal to a NaN). Shapes: gate_up's K 5120 and down's K 2304 with N
 *                cut short (an output is its own weight row's sum), two weight rows a wave (N 1024), and the
 *                live-tail geometry's small shapes;
 *   chain        gate_up [128, 512] -> swiglu_clamped -> down [96, 64]: the compacted chain's rows against the full
 *                chain's and the host experts' (ds41hx: gemv_ref, swiglu over silu_table_host).
 *
 * `r4d_emu_ds41aux tail` runs it with the live-tail ranges on (RADIANCE_EXPERIMENTAL_DS41_LIVE_TAIL=1, which libr4d
 * reads once a process): every launch at the live-tail geometry -- the full launches at the small shapes, and a
 * compacted launch that is one whole routing (gate_up: T' = M * 6, the "local" passes' "all" subset; down: T' a
 * multiple of 6) -- must take the tail branch (its gemv, then mx_tail_zero's kernel), and some compacted ones must.
 * Activations carry NaN, inf and subnormal values on some rows. Runs with the cards hidden. */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"
#include "r4d_ds41_live_tail_policy.h"
#include "ds41_aux.h"
#include "ds41_host_experts.h"
#include "ds41_p1.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

extern const char* emu_last_kernel;
extern int emu_grids;

namespace {

constexpr int kNe = 96, kTop = 6, kRanks = 4, kRank = 1;
constexpr float kLimit = 10.0f;
typedef int (*Launch)(const RadArgs*, RadStream);

bool g_tail = false;
int g_fail = 0, g_cases = 0, g_full_tail = 0, g_cmp_tail = 0, g_whole = 0;

uint8_t* aligned_bytes(size_t n) {
    const size_t a = (n + 63) / 64 * 64;
    void* p = std::aligned_alloc(64, a ? a : 64);
    if (!p) { std::printf("out of memory\n"); std::exit(2); }
    std::memset(p, 0xA5, a ? a : 64);
    return (uint8_t*)p;
}
uint16_t f2bf(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
bool nan16(uint16_t v) { return (v & 0x7f80u) == 0x7f80u && (v & 0x7fu); }
bool same16(uint16_t a, uint16_t b) { return a == b || (nan16(a) && nan16(b)); }

/* One expert weight from planes as the release stores them, in both forms the worker's card can hold. */
struct Expert {
    int N = 0, K = 0;
    uint8_t* block = nullptr;
    uint8_t* packed = nullptr;
    Expert() = default;
    Expert(const Expert&) = delete;
    Expert& operator=(const Expert&) = delete;
    ~Expert() { std::free(block); std::free(packed); }
    void make(int n, int k, std::mt19937_64& rng) {
        N = n;
        K = k;
        const int UK = K / 32, R = (UK + 31) / 32;
        std::vector<uint8_t> codes((size_t)N * (K / 2)), scales((size_t)N * UK);
        for (auto& c : codes) c = (uint8_t)rng();
        std::uniform_real_distribution<double> u01(0.0, 1.0);
        for (auto& s : scales) {
            const double r = u01(rng);
            s = r < 0.58 ? 120 : r < 0.995 ? 121 : r < 0.998 ? 119 : 122;
        }
        static const uint8_t edges[] = { 0, 1, 2, 3, 100, 127, 128, 135, 254 };
        for (int n0 = 0; n0 < N; ++n0)   /* a few raw rounds: a block far outside the window */
            for (int r = 0; r < R; ++r)
                if (u01(rng) < 0.08) {
                    const int t = r < UK / 32 ? 32 : UK % 32;
                    scales[(size_t)n0 * UK + 32 * r + (size_t)(rng() % (uint64_t)t)] = edges[rng() % sizeof edges];
                }
        block = aligned_bytes((size_t)N * UK * 17);
        for (int n0 = 0; n0 < N; ++n0)
            for (int u = 0; u < UK; ++u) {
                uint8_t* b = block + ((size_t)n0 * UK + u) * 17;
                const uint8_t* c = codes.data() + (size_t)n0 * (K / 2) + (size_t)u * 16;
                auto val = [&](int v) { return (c[v >> 1] >> (4 * (v & 1))) & 15; };
                b[0] = scales[(size_t)n0 * UK + u];
                for (int j = 0; j < 16; ++j) b[1 + j] = (uint8_t)(val(j) | (val(j + 16) << 4));
            }
        ds41hx::PackInfo plan;
        if (!ds41hx::pack_plan(scales.data(), N, K, &plan)) { std::printf("pack_plan refused a weight\n"); std::exit(2); }
        const int cap = std::min(plan.n_exc + 2, N * R);
        const size_t pb = ds41hx::packed_bytes(N, K, cap);
        packed = aligned_bytes(pb);
        ds41hx::Mx v;
        if (ds41hx::pack_planes(codes.data(), scales.data(), N, K, cap, plan, packed, pb) != ds41hx::kPackOk ||
            !ds41hx::packed_view(packed, pb, N, K, &v) || !ds41hx::packed_scales_equal(v, scales.data())) {
            std::printf("the packer refused a weight\n");
            std::exit(2);
        }
    }
};

/* A rank's 96 experts of one [N, K] projection; its tables: every entry block_mxfp4 (the plain row's, the CPU's
 * bits), or every entry packed and marked (the worker's slots). */
struct Bank {
    int N = 0, K = 0;
    std::vector<std::unique_ptr<Expert>> e;
    Bank(int n, int k, uint64_t seed) : N(n), K(k) {
        std::mt19937_64 rng(seed);
        for (int i = 0; i < kNe; ++i) {
            e.emplace_back(new Expert);
            e.back()->make(N, K, rng);
        }
    }
    std::vector<const void*> table(bool packed) const {
        std::vector<const void*> t((size_t)kNe);
        for (int i = 0; i < kNe; ++i)
            t[(size_t)i] = packed ? ds41p1::mark(e[(size_t)i]->packed, ds41p1::kMarkPacked) : e[(size_t)i]->block;
        return t;
    }
};

/* ---- routings: one pass on rank 1, as moe_scatter writes it ------------------------------------------ */
enum Kind { kUniform, kEdge, kLocal, kKinds };
const char* kind_name(int k) {
    static const char* n[] = { "uniform", "edge", "local" };
    return n[k];
}
struct Pass {
    std::vector<int32_t> sorted, off;   /* [M * 6] (-1 past the rank's slots), [97] */
    int local = 0;
};
Pass routing(int kind, int M, std::mt19937_64& rng) {
    for (;;) {
        std::vector<int> p((size_t)M * kTop);
        for (int t = 0; t < M; ++t) {
            int* pk = &p[(size_t)t * kTop];
            int n = 0;
            if (kind == kEdge) { pk[n++] = kRank * kNe; pk[n++] = kRank * kNe + kNe - 1; }
            while (n < kTop) {
                const int e = kind == kLocal ? kRank * kNe + (int)(rng() % kNe) : (int)(rng() % (kNe * kRanks));
                bool dup = false;
                for (int j = 0; j < n; ++j) dup |= pk[j] == e;
                if (!dup) pk[n++] = e;
            }
            std::shuffle(pk, pk + kTop, rng);
        }
        std::vector<std::vector<int32_t>> by(kNe);
        for (int t = 0; t < M; ++t)
            for (int k = 0; k < kTop; ++k) {
                const int e = p[(size_t)t * kTop + k];
                if (e / kNe == kRank) by[(size_t)(e % kNe)].push_back(t * kTop + k);
            }
        Pass r;
        r.off.assign(kNe + 1, 0);
        for (int e = 0; e < kNe; ++e) {
            r.off[(size_t)e + 1] = r.off[(size_t)e] + (int)by[(size_t)e].size();
            for (int32_t f : by[(size_t)e]) r.sorted.push_back(f);
        }
        r.local = (int)r.sorted.size();
        r.sorted.resize((size_t)M * kTop, -1);
        if (r.local) return r;   /* a pass with no slot here: drawn again */
    }
}

/* Activations: `rows` rows of K, NaN, inf and subnormal values on every 7th row. */
std::vector<uint16_t> make_x(int rows, int K, std::mt19937_64& rng) {
    std::normal_distribution<float> nd(0.f, 0.6f);
    static const uint16_t sp[] = { 0x7fc0, 0xffc0, 0x7f80, 0xff80, 0x0001, 0x8001, 0x0080, 0x8000 };
    std::vector<uint16_t> x((size_t)rows * K);
    for (int r = 0; r < rows; ++r)
        for (int k = 0; k < K; ++k) {
            uint16_t v = f2bf(nd(rng));
            if (r % 7 == 3 && rng() % 97 == 0) v = sp[rng() % (sizeof sp / 2)];
            x[(size_t)r * K + k] = v;
        }
    return x;
}

/* ---- launches -------------------------------------------------------------------------------------- */
RadTensor ten(void* p, uint32_t dt, std::vector<int64_t> shape) {
    RadTensor t{};
    t.data = p;
    t.dtype = dt;
    t.rank = (uint32_t)shape.size();
    int64_t st = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
        t.shape[i] = shape[(size_t)i];
        t.stride[i] = st;
        st *= shape[(size_t)i];
    }
    return t;
}
RadParam pint(const char* k, long long v) { RadParam p{}; p.key = k; p.kind = RAD_P_INT; p.ival = v; return p; }
RadParam pstr(const char* k, const char* v) { RadParam p{}; p.key = k; p.kind = RAD_P_STR; p.sval = v; return p; }
RadParam pf64(const char* k, double v) { RadParam p{}; p.key = k; p.kind = RAD_P_F64; p.dval = v; return p; }

/* One moe_gemm_ggml call into y: x [AR, K], the table, T sorted slots. Returns 0, or 1 after a FAIL line; *tail
 * whether it took the live-tail branch, which it must exactly when the ranges are on and the call is at the
 * live-tail geometry (mx_moe's own test: tokens AR in token order, T / 6 rounded up in sorted order). */
int moe(Launch f, const char* what, const uint16_t* x, int AR, int K, const std::vector<const void*>& table, int N,
        const int32_t* sorted, int T, const int32_t* off, bool sorted_order, uint16_t* y, bool* tail) {
    std::vector<const void*> tab = table;
    std::vector<int32_t> so(sorted, sorted + T), of(off, off + kNe + 1);
    RadTensor t[5] = { ten((void*)x, RAD_BF16, { AR, K }), ten(tab.data(), RAD_U8, { kNe, N, K / 32 * 17 }),
                       ten(so.data(), RAD_I32, { T }), ten(of.data(), RAD_I32, { kNe + 1 }),
                       ten(y, RAD_BF16, { T, N }) };
    t[1].stride[0] = 0;   /* a table of expert pointers */
    RadParam pr[6] = { pstr("type", "mxfp4"), pint("N", N), pint("K", K), pint("n_expert", kNe), pint("top_k", kTop),
                       pstr("a_order", sorted_order ? "sorted" : "token") };
    RadArgs a{};
    a.t = t;
    a.n_t = 5;
    a.p = pr;
    a.n_p = 6;
    const int g0 = emu_grids;
    const int rc = f(&a, nullptr);
    const int n = emu_grids - g0;
    const std::string last = emu_last_kernel ? emu_last_kernel : "?";
    const long long tokens = sorted_order ? (T + kTop - 1) / kTop : AR;
    *tail = g_tail && ds41_live_tail::geometry(tokens, N, K, kNe, kTop, T, AR, sorted_order);
    if (rc != RAD_OK) {
        std::printf("FAIL %s: the launch returned %d\n", what, rc);
        ++g_fail;
        return 1;
    }
    const bool zero_last = last.find("mx_zero_tail_kernel") != std::string::npos;
    if (*tail ? (n != 2 || !zero_last) : (n != 1 || zero_last)) {
        std::printf("FAIL %s: %d launches, the last %s (the live-tail branch %s)\n", what, n, last.c_str(),
                    *tail ? "expected" : "not expected");
        ++g_fail;
        return 1;
    }
    return 0;
}
int swiglu(uint16_t* gu, uint16_t* y, int rows, int n) {
    RadTensor t[2] = { ten(gu, RAD_BF16, { rows, 2 * n }), ten(y, RAD_BF16, { rows, n }) };
    RadParam p[2] = { pint("n", n), pf64("limit", kLimit) };
    RadArgs a{};
    a.t = t;
    a.n_t = 2;
    a.p = p;
    a.n_p = 2;
    return r4d_rad_swiglu_clamped(&a, nullptr);
}

/* The subsets of a pass's live experts the worker may hold: one, every other, all, the last. */
std::vector<std::vector<int>> subsets(const Pass& p) {
    std::vector<int> live;
    for (int e = 0; e < kNe; ++e)
        if (p.off[(size_t)e + 1] > p.off[(size_t)e]) live.push_back(e);
    std::vector<std::vector<int>> s(4);
    s[0] = { live.front() };
    for (size_t i = 0; i < live.size(); i += 2) s[1].push_back(live[i]);
    s[2] = live;
    s[3] = { live.back() };
    return s;
}
const char* subset_name(int i) {
    static const char* n[] = { "one", "every other", "all", "last" };
    return n[i];
}
bool compact(const Pass& p, int M, const std::vector<int>& held, ds41aux::Compact* c, const char* what) {
    uint8_t mask[kNe] = {};
    int32_t counts[kNe];
    for (int e : held) mask[e] = 1;
    for (int e = 0; e < kNe; ++e) counts[e] = p.off[(size_t)e + 1] - p.off[(size_t)e];
    const int rc = ds41aux::compact(p.sorted.data(), p.off.data(), M, kTop, kNe, mask, counts, kNe, c);
    if (rc != 0 || c->T < 1) {
        std::printf("FAIL %s: compact returned %d (T' %d)\n", what, rc, c->T);
        ++g_fail;
        return false;
    }
    return true;
}

/* ---- one projection -------------------------------------------------------------------------------- */
struct Shape {
    const char* name;
    int N, K;
    bool sorted;
    std::vector<int> Ms;
};
void projection(const Shape& sh, std::mt19937_64& rng, uint64_t seed) {
    const Bank b(sh.N, sh.K, seed);
    const auto blk = b.table(false), pk = b.table(true);
    const size_t N = (size_t)sh.N;
    for (int M : sh.Ms)
        for (int kind = 0; kind < kKinds; ++kind) {
            const Pass p = routing(kind, M, rng);
            const int T = M * kTop, L = p.local, AR = sh.sorted ? T : M;
            const std::vector<uint16_t> x = make_x(AR, sh.K, rng);
            char what[160];
            std::snprintf(what, sizeof what, "%s, M %d %s", sh.name, M, kind_name(kind));
            std::vector<uint16_t> yb((size_t)T * N, 0xAAAA), yp((size_t)T * N, 0xAAAA);
            bool tb = false, tp = false;
            if (moe(r4d_rad_moe_gemm_mxfp4_gemv, what, x.data(), AR, sh.K, blk, sh.N, p.sorted.data(), T, p.off.data(),
                    sh.sorted, yb.data(), &tb) ||
                moe(r4d_rad_moe_gemm_mxfp4_gemv_vp, what, x.data(), AR, sh.K, pk, sh.N, p.sorted.data(), T,
                    p.off.data(), sh.sorted, yp.data(), &tp))
                continue;
            g_full_tail += tb + tp;
            size_t d_full = 0;
            for (size_t i = 0; i < (size_t)L * N; ++i) d_full += !same16(yp[i], yb[i]);
            size_t d_cmp = 0, d_blk = 0;
            int n_tail = 0;
            std::string shapes;
            const auto ss = subsets(p);
            for (int si = 0; si < 4; ++si) {
                ds41aux::Compact c;
                if (!compact(p, M, ss[(size_t)si], &c, what)) continue;
                const int Tc = c.T;
                /* Sorted order: the worker's own SwiGLU rows, its compacted slots in order. */
                std::vector<uint16_t> xc;
                if (sh.sorted) {
                    xc.resize((size_t)Tc * sh.K);
                    for (int j = 0; j < Tc; ++j)
                        std::memcpy(&xc[(size_t)j * sh.K], &x[(size_t)c.row_of[j] * sh.K], (size_t)sh.K * 2);
                }
                std::vector<uint16_t> yc((size_t)Tc * N, 0xAAAA);
                bool tc = false;
                if (moe(r4d_rad_moe_gemm_mxfp4_gemv_vp, what, sh.sorted ? xc.data() : x.data(), sh.sorted ? Tc : M,
                        sh.K, pk, sh.N, c.sorted, Tc, c.off, sh.sorted, yc.data(), &tc))
                    continue;
                n_tail += tc;
                g_cmp_tail += tc;
                g_whole += Tc == T;
                for (int j = 0; j < Tc; ++j) {
                    const size_t i = (size_t)c.row_of[j];
                    for (size_t q = 0; q < N; ++q) {
                        d_cmp += yc[(size_t)j * N + q] != yp[i * N + q];
                        d_blk += !same16(yc[(size_t)j * N + q], yb[i * N + q]);
                    }
                }
                ++g_cases;
                shapes += (shapes.empty() ? "" : "/") + std::to_string(Tc);
            }
            const bool ok = !d_full && !d_cmp && !d_blk;
            if (!ok) ++g_fail;
            std::printf("%s %s: %d rows; gemv_vp %zu values differ from gemv; compacted T' %s of %d (%d live-tail): "
                        "%zu differ from gemv_vp's full launch, %zu from gemv's\n", ok ? "ok  " : "FAIL", what, L, d_full,
                        shapes.c_str(), T, n_tail, d_cmp, d_blk);
            std::fflush(stdout);
        }
}

/* ---- the chain ----------------------------------------------------------------------------------------- */
void chain(std::mt19937_64& rng) {
    constexpr int E = 512, F = 64, ND = 96;   /* gate_up [2F, E] = [128, 512] (a live-tail shape), down [ND, F] */
    const Bank g(2 * F, E, 0xa11c0de1), d(ND, F, 0xa11c0de2);
    const auto gb = g.table(false), gp = g.table(true), db = d.table(false), dp = d.table(true);
    std::vector<float> silu(65536);
    ds41hx::silu_table_host(silu.data(), kLimit);
    for (int M : { 1, 4, 8, 16 })
        for (int kind = 0; kind < kKinds; ++kind) {
            const Pass p = routing(kind, M, rng);
            const int T = M * kTop, L = p.local;
            const std::vector<uint16_t> x = make_x(M, E, rng);
            char what[96];
            std::snprintf(what, sizeof what, "chain, M %d %s", M, kind_name(kind));
            /* The full chains: plain gemv over block_mxfp4, gemv_vp over packed. */
            std::vector<uint16_t> G[2], A[2], Y[2];
            bool bad = false, t = false;
            for (int v = 0; v < 2 && !bad; ++v) {
                G[v].assign((size_t)T * 2 * F, 0xAAAA);
                A[v].assign((size_t)T * F, 0xAAAA);
                Y[v].assign((size_t)T * ND, 0xAAAA);
                const Launch f = v ? r4d_rad_moe_gemm_mxfp4_gemv_vp : r4d_rad_moe_gemm_mxfp4_gemv;
                bad = moe(f, what, x.data(), M, E, v ? gp : gb, 2 * F, p.sorted.data(), T, p.off.data(), false,
                          G[v].data(), &t) != 0;
                g_full_tail += t;
                if (!bad && swiglu(G[v].data(), A[v].data(), T, F) != RAD_OK) {
                    std::printf("FAIL %s: swiglu_clamped refused\n", what);
                    ++g_fail;
                    bad = true;
                }
                if (!bad) bad = moe(f, what, A[v].data(), T, F, v ? dp : db, ND, p.sorted.data(), T, p.off.data(), true,
                                    Y[v].data(), &t) != 0;
                g_full_tail += !bad && t;
            }
            if (bad) continue;
            size_t d_full = 0;
            for (int i = 0; i < L; ++i)
                for (int q = 0; q < ND; ++q) d_full += !same16(Y[1][(size_t)i * ND + q], Y[0][(size_t)i * ND + q]);
            size_t d_cmp = 0, d_host = 0;
            std::string shapes;
            const auto ss = subsets(p);
            for (int si = 0; si < 4; ++si) {
                ds41aux::Compact c;
                if (!compact(p, M, ss[(size_t)si], &c, what)) continue;
                const int Tc = c.T;
                std::vector<uint16_t> gc((size_t)Tc * 2 * F, 0xAAAA), ac((size_t)Tc * F, 0xAAAA), yc((size_t)Tc * ND, 0xAAAA);
                bool t1 = false, t2 = false;
                if (moe(r4d_rad_moe_gemm_mxfp4_gemv_vp, what, x.data(), M, E, gp, 2 * F, c.sorted, Tc, c.off, false,
                        gc.data(), &t1))
                    continue;
                if (swiglu(gc.data(), ac.data(), Tc, F) != RAD_OK) {
                    std::printf("FAIL %s: swiglu_clamped refused a compacted job\n", what);
                    ++g_fail;
                    continue;
                }
                if (moe(r4d_rad_moe_gemm_mxfp4_gemv_vp, what, ac.data(), Tc, F, dp, ND, c.sorted, Tc, c.off, true,
                        yc.data(), &t2))
                    continue;
                g_cmp_tail += t1 + t2;
                g_whole += Tc == T;
                for (int j = 0; j < Tc; ++j) {
                    const int i = c.row_of[j], f = c.sorted[j], tok = f / kTop;
                    int e = 0;
                    while (p.off[(size_t)e + 1] <= i) ++e;
                    for (int q = 0; q < 2 * F; ++q) d_cmp += gc[(size_t)j * 2 * F + q] != G[1][(size_t)i * 2 * F + q];
                    for (int q = 0; q < F; ++q) d_cmp += ac[(size_t)j * F + q] != A[1][(size_t)i * F + q];
                    for (int q = 0; q < ND; ++q) d_cmp += yc[(size_t)j * ND + q] != Y[1][(size_t)i * ND + q];
                    /* The host experts' chain for this row: the CPU path's bits. */
                    ds41hx::Mx wg, wd;
                    wg.N = 2 * F; wg.K = E; wg.blocks = g.e[(size_t)e]->block;
                    wd.N = ND; wd.K = F; wd.blocks = d.e[(size_t)e]->block;
                    uint16_t hg[2 * F], ha[F], hy[ND];
                    ds41hx::gemv_ref(wg, &x[(size_t)tok * E], hg, 0, 2 * F);
                    ds41hx::swiglu(silu.data(), hg, ha, F, kLimit);
                    ds41hx::gemv_ref(wd, ha, hy, 0, ND);
                    for (int q = 0; q < ND; ++q) d_host += !same16(hy[q], yc[(size_t)j * ND + q]);
                }
                ++g_cases;
                shapes += (shapes.empty() ? "" : "/") + std::to_string(Tc);
            }
            const bool ok = !d_full && !d_cmp && !d_host;
            if (!ok) ++g_fail;
            std::printf("%s %s: %d rows; full gemv_vp chain %zu values differ from gemv's; compacted T' %s of %d: %zu "
                        "differ from the full chain, %zu from the host experts\n", ok ? "ok  " : "FAIL", what, L, d_full,
                        shapes.c_str(), T, d_cmp, d_host);
            std::fflush(stdout);
        }
}

}  // namespace

int main(int argc, char** argv) {
    /* libr4d reads the switch once a process: each setting is its own run (r4d_emu_ds41aux, r4d_emu_ds41aux_tail). */
    g_tail = argc > 1 && !std::strcmp(argv[1], "tail");
    if (g_tail) setenv("RADIANCE_EXPERIMENTAL_DS41_LIVE_TAIL", "1", 1);
    else unsetenv("RADIANCE_EXPERIMENTAL_DS41_LIVE_TAIL");
    std::mt19937_64 rng(20261104);
    const Shape shapes[] = {
        { "gate_up's K 5120, N cut to 32", 32, 5120, false, { 1, 4, 8, 16 } },
        { "down's K 2304 (a partial round of 8), N cut to 24", 24, 2304, true, { 1, 4, 8, 16 } },
        { "two weight rows a wave (N 1024, K 256)", 1024, 256, false, { 1, 4 } },
        { "live-tail geometry [64, 256], token order", 64, 256, false, { 1, 4, 8, 16 } },
        { "live-tail geometry [128, 512], sorted order", 128, 512, true, { 1, 4, 8, 16 } },
    };
    uint64_t seed = 0x5eed0001;
    for (const Shape& sh : shapes) projection(sh, rng, seed++);
    chain(rng);
    std::printf("%d compacted launches (%d whole), %d live-tail full launches, %d live-tail compacted ones%s\n",
                g_cases, g_whole, g_full_tail, g_cmp_tail, g_tail ? " (the ranges on)" : "");
    if (g_tail && (g_full_tail == 0 || g_cmp_tail == 0)) {
        std::printf("FAIL the live-tail branch never ran for %s\n", g_full_tail ? "a compacted launch" : "a full launch");
        ++g_fail;
    }
    if (!g_tail && (g_full_tail || g_cmp_tail)) {
        std::printf("FAIL the live-tail branch ran with the ranges off\n");
        ++g_fail;
    }
    if (g_whole == 0) {
        std::printf("FAIL no compacted launch was one whole routing\n");
        ++g_fail;
    }
    if (g_fail) std::printf("FAIL r4d_emu_ds41aux%s: %d failure(s)\n", g_tail ? " tail" : "", g_fail);
    else std::printf("PASS r4d_emu_ds41aux%s\n", g_tail ? " tail" : "");
    return g_fail ? 1 : 0;
}
