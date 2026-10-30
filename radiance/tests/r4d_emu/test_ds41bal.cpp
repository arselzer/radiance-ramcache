/* tests/r4d_emu: RADIANCE_EXPERIMENTAL_DS41_MOE_BALANCE (libr4d/r4d_ds41_mxfp4.hip, mx_gemv_kernel) on the
 * host SIMT emulator.
 *
 * The routed MXFP4 GEMM's gemv kernel gives each live expert the same number of work items whatever its
 * rows, and mx_range hands an EP rank's last expert every slot routed to the other ranks ([off[ne], T),
 * about five sixths of a tall pass), one weight pass each at gate_up's XLM 1. Under the switch the items
 * follow each expert's row passes and a pass of dropped slots writes its zeros without reading weights.
 * This holds the balanced launch to the plain one byte for byte over the whole output (every row,
 * dropped ones included, poisoned before), through the checker export r4d_rad_moe_gemm_mxfp4_gemv_bal
 * (the switch forced per call): the gemv row at 1 to 16 tokens and the twin moe_gemm_ggml_mxfp4_gemv_tall
 * at 17 to 256, for the gate_up (token rows: XLM 1 at K 5120) and the down projection (sorted rows: XLM 3
 * at K 2304), and a short K with two weight rows a wave (XLM 8). Routings: a small pool of the 384 experts
 * (one expert on every token), all of them, a rank whose last expert has no rows of its own (it is live
 * through the dropped tail alone), every slot local (no tail), a pass with a dropped slot inside an
 * expert's run and token rows past the activations (both rows of zeros), and no local slot at all. */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
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

/* A pass's slots on this rank as moe_scatter writes them: each local expert's (token * top_k + slot)
 * ascending, grouped by expert, then -1 for the other ranks' slots; off[kNe] the local count. */
struct Routing {
    std::vector<int32_t> sorted, off;
    int local = 0;
};

enum Kind { kPool, kAll, kNoLast, kAllLocal, kHoles, kNone };
const char* kind_name(Kind k) {
    switch (k) {
    case kPool: return "pool of 9";
    case kAll: return "all 384";
    case kNoLast: return "last expert tail-only";
    case kAllLocal: return "every slot local";
    case kHoles: return "dropped slots inside runs, rows past the activations";
    default: return "no local slot";
    }
}

Routing make_routing(Kind kind, int T, std::mt19937_64& rng) {
    std::vector<int> pool;
    if (kind == kPool) {
        while ((int)pool.size() < 9) {
            const int e = (int)(rng() % (4 * kNe));
            if (std::find(pool.begin(), pool.end(), e) == pool.end()) pool.push_back(e);
        }
        pool[0] = (int)(rng() % kNe);   /* local, on every token */
    }
    std::vector<std::vector<int32_t>> by(kNe);
    for (int t = 0; t < T; ++t) {
        std::vector<int> pk;
        if (kind == kPool) pk.push_back(pool[0]);
        while ((int)pk.size() < kTop) {
            int e;
            if (kind == kPool) e = pool[(size_t)(rng() % pool.size())];
            else if (kind == kAllLocal) e = (int)(rng() % kNe);
            else if (kind == kNone) e = kNe + (int)(rng() % (3 * kNe));
            else e = (int)(rng() % (4 * kNe));
            if (kind == kNoLast && e == kNe - 1) continue;
            if (std::find(pk.begin(), pk.end(), e) == pk.end()) pk.push_back(e);
        }
        std::shuffle(pk.begin(), pk.end(), rng);
        for (int k = 0; k < kTop; ++k)
            if (pk[(size_t)k] < kNe) by[(size_t)pk[(size_t)k]].push_back(t * kTop + k);
    }
    Routing r;
    r.off.assign(kNe + 1, 0);
    for (int e = 0; e < kNe; ++e) {
        r.off[(size_t)e + 1] = r.off[(size_t)e] + (int)by[(size_t)e].size();
        for (int32_t f : by[(size_t)e]) r.sorted.push_back(f);
    }
    r.local = (int)r.sorted.size();
    r.sorted.resize((size_t)T * kTop, -1);
    if (kind == kHoles)   /* a few local slots dropped in place (their rows are zeros either way) */
        for (int i = 0; i < r.local; ++i)
            if (rng() % 7 == 0) r.sorted[(size_t)i] = -1;
    return r;
}

/* One launch through the checker export: `AR` activation rows of K, y [T, N] poisoned first. */
int run(int tall, int bal, const uint16_t* x, int AR, const uint8_t* const* table, int N, int K,
        const Routing& r, std::vector<uint16_t>& y, const char* order) {
    const int T = (int)r.sorted.size();
    y.assign((size_t)T * N, 0xAAAA);
    RadTensor t[5] = { tens((void*)x, RAD_BF16, { AR, K }),
                       tens((void*)table, RAD_U8, { kNe, N, K / 32 * 17 }),
                       tens((void*)r.sorted.data(), RAD_I32, { T }),
                       tens((void*)r.off.data(), RAD_I32, { kNe + 1 }),
                       tens(y.data(), RAD_BF16, { T, N }) };
    t[1].stride[0] = 0;
    RadParam p[6] = { pstr("type", "mxfp4"), pint("N", N), pint("K", K), pint("n_expert", kNe),
                      pint("top_k", kTop), pstr("a_order", order) };
    RadArgs a{};
    a.t = t;
    a.n_t = 5;
    a.p = p;
    a.n_p = 6;
    return r4d_rad_moe_gemm_mxfp4_gemv_bal(&a, nullptr, tall, bal);
}

struct Shape {
    const char* name;
    int N, K;
    bool sorted;   /* the down projection's rows: one a slot */
};

/* Returns the failures. */
int check(const Shape& sh, int T, Kind kind, uint32_t seed, long long* values) {
    std::mt19937_64 rng(seed);
    const Routing r = make_routing(kind, T, rng);
    std::vector<std::vector<uint8_t>> w(kNe);
    std::vector<const uint8_t*> table(kNe);
    w[0] = make_expert(sh.N, sh.K, rng);
    for (int e = 0; e < kNe; ++e) {
        if (e && r.off[(size_t)e + 1] > r.off[(size_t)e]) w[(size_t)e] = make_expert(sh.N, sh.K, rng);
        table[(size_t)e] = w[(size_t)e].empty() ? w[0].data() : w[(size_t)e].data();
    }
    /* Token rows (gate_up) or a row a slot (down); kHoles gives the gate_up fewer activation rows than
     * tokens, so the last tokens' slots read past `a` and are rows of zeros. */
    const int AR = sh.sorted ? T * kTop : (kind == kHoles && T > 2 ? T - 2 : T);
    std::normal_distribution<float> nd(0.f, 0.5f);
    std::vector<uint16_t> x((size_t)AR * sh.K);
    for (auto& v : x) {
        v = bf16(nd(rng));
        if (rng() % 97 == 0) v = rng() % 2 ? 0x8000 : 0x0001;
    }
    const int tall = T > 16 ? 1 : 0;
    std::vector<uint16_t> y0, y1;
    const char* order = sh.sorted ? "sorted" : "token";
    if (run(tall, 0, x.data(), AR, table.data(), sh.N, sh.K, r, y0, order) != RAD_OK ||
        run(tall, 1, x.data(), AR, table.data(), sh.N, sh.K, r, y1, order) != RAD_OK) {
        std::printf("FAIL %s T=%d %s: a launch was refused\n", sh.name, T, kind_name(kind));
        return 1;
    }
    if (!std::strstr(emu_last_kernel, "mx_gemv_kernel")) {
        std::printf("FAIL %s T=%d: the balanced launch ran %s, not the gemv\n", sh.name, T, emu_last_kernel);
        return 1;
    }
    *values += (long long)y0.size();
    int bad = 0;
    for (size_t i = 0; i < y0.size(); ++i)
        if (y0[i] != y1[i]) {
            if (bad < 4)
                std::printf("FAIL %s T=%d %s: row %zu column %zu: 0x%04x plain, 0x%04x balanced\n", sh.name, T,
                            kind_name(kind), i / sh.N, i % sh.N, y0[i], y1[i]);
            ++bad;
        }
    /* Every output was written (no poison left), and a dropped slot's row is zeros. */
    for (size_t i = 0; i < y1.size(); ++i)
        if (y1[i] == 0xAAAA) {
            std::printf("FAIL %s T=%d %s: row %zu column %zu was never written\n", sh.name, T, kind_name(kind),
                        i / sh.N, i % sh.N);
            ++bad;
            break;
        }
    for (int i = 0; i < (int)r.sorted.size(); ++i)
        if (r.sorted[(size_t)i] < 0)
            for (int n = 0; n < sh.N; ++n)
                if (y1[(size_t)i * sh.N + n] != 0) {
                    std::printf("FAIL %s T=%d %s: dropped slot %d holds 0x%04x\n", sh.name, T, kind_name(kind), i,
                                y1[(size_t)i * sh.N + n]);
                    return bad + 1;
                }
    return bad;
}

}  // namespace

int main() {
    /* ds41's gate_up is [4608, 5120] and its down [5120, 2304]; the emulator keeps one side short (each
     * output is its own weight row's sum, so N changes no sum), as test_ds41tall does. */
    const Shape shapes[] = {
        { "gate_up K 5120 (XLM 1)", 64, 5120, false },
        { "down K 2304 (XLM 3)", 64, 2304, true },
        { "K 512 (XLM 8), two weight rows a wave", 1024, 512, false },
    };
    const std::vector<int> heights[3] = { { 1, 6, 16, 17, 64, 171, 256 }, { 1, 16, 33, 100, 256 }, { 6, 48, 171 } };
    const Kind kinds[] = { kPool, kAll, kNoLast, kAllLocal, kHoles, kNone };
    int fails = 0, cases = 0;
    long long values = 0;
    for (int si = 0; si < 3; ++si)
        for (int T : heights[si])
            for (Kind k : kinds) {
                if (si == 2 && k != kPool && k != kAll && k != kHoles) continue;
                const auto t0 = std::chrono::steady_clock::now();
                const int bad = check(shapes[si], T, k, 7919u * (uint32_t)T + 31u * (uint32_t)k + (uint32_t)si, &values);
                std::printf("  %s T=%d %s: %s, %.1f s\n", shapes[si].name, T, kind_name(k), bad ? "DIFFERS" : "same bytes",
                            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
                ++cases;
                fails += bad != 0;
            }
    std::printf("%s: %d cases, %lld output values, the balanced gemv and twin against the plain kernel\n",
                fails ? "FAIL" : "PASS", cases, values);
    return fails ? 1 : 0;
}
