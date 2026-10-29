/* ds41_mxfp4_device_check.hip -- runs libr4d's own decode kernels (loaded from a built libr4d.so)
 * on one card and checks the host experts (ds41_host_experts) against them bit for bit.
 *
 *   ds41_mxfp4_device_check LIBR4D.so check [trials]   GEMV/SwiGLU/GEMV chain, one token, 1..3
 *                                                      live experts, live-tail ranges
 *   ds41_mxfp4_device_check LIBR4D.so swiglu TABLE      captures silu(g) for every bf16 gate into
 *                                                      TABLE, then proves swiglu_clamped against
 *                                                      the host for every (gate, up) bf16 pair
 *   ds41_mxfp4_device_check LIBR4D.so time [LIB2.so]   decode GEMV time per call at 1..3 live
 *                                                      experts; with LIB2, also its outputs vs
 *                                                      LIBR4D's, bit for bit
 *   ds41_mxfp4_device_check LIBR4D.so tall [T...]      RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL (set
 *                                                      here): the twin row moe_gemm_ggml_mxfp4_gemv_tall
 *                                                      in the plugin's table; its gate_up and down at
 *                                                      T tokens (17..96) against the gemv over the same
 *                                                      tokens in passes of 16 and against the host
 *                                                      kernels; attn_sparse_sink and index_score at
 *                                                      17..100 rows against the same rows one at a
 *                                                      time and in launches of 16. Every value equal
 *                                                      or FAIL.
 *
 * Needs a free card (HIP device 0 unless HIP_VISIBLE_DEVICES says otherwise). Uses the
 * live-tail ranges as the measured runs do (RADIANCE_EXPERIMENTAL_DS41_LIVE_TAIL=1), except `tall`,
 * which runs the routed GEMM as the A/B's servers do (without them). */
#include <hip/hip_runtime.h>
#include <dlfcn.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "rad_abi.h"
#include "ds41_host_experts.h"

using namespace ds41hx;

#define HC(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e_)); std::exit(2); } } while (0)

namespace {

constexpr int kEmbd = 5120, kFf = 2304, kNe = 96, kTop = 6;
constexpr float kLimit = 10.0f;
typedef int (*Launch)(const RadArgs*, RadStream);

struct Lib {
    Launch moe = nullptr, swiglu = nullptr;
    /* For `tall`: the twin's launch, the tiled row's (to show what the twin replaces), the sparse
     * ops, and the plugin's row table. */
    Launch moe_tall = nullptr, moe_tiled = nullptr, attn = nullptr, iscore = nullptr;
    /* For `sweep`: the other ops whose launcher reads the pass height. */
    Launch rmsnorm = nullptr, collapse_norm = nullptr, post_collapse_norm = nullptr, norm_rope = nullptr,
           mix = nullptr, gemm64 = nullptr, logits = nullptr;
    int (*count)(void) = nullptr;
    const RadKernelInfo* (*at)(int) = nullptr;
    explicit Lib(const char* path) {
        void* h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!h) { std::fprintf(stderr, "dlopen %s: %s\n", path, dlerror()); std::exit(2); }
        moe = (Launch)dlsym(h, "r4d_rad_moe_gemm_mxfp4_gemv");
        swiglu = (Launch)dlsym(h, "r4d_rad_swiglu_clamped");
        moe_tall = (Launch)dlsym(h, "r4d_rad_moe_gemm_mxfp4_gemv_tall");
        moe_tiled = (Launch)dlsym(h, "r4d_rad_moe_gemm_mxfp4_tiled");
        attn = (Launch)dlsym(h, "r4d_rad_ds41_attn_sparse_sink");
        iscore = (Launch)dlsym(h, "r4d_rad_ds41_index_score");
        rmsnorm = (Launch)dlsym(h, "r4d_rad_rmsnorm");
        collapse_norm = (Launch)dlsym(h, "r4d_rad_mhc_collapse_rmsnorm");
        post_collapse_norm = (Launch)dlsym(h, "r4d_rad_mhc_post_collapse_rmsnorm");
        norm_rope = (Launch)dlsym(h, "r4d_rad_rmsnorm_rope_roundtrip");
        mix = (Launch)dlsym(h, "r4d_rad_mhc_mix");
        gemm64 = (Launch)dlsym(h, "r4d_rad_gemm_bf16_m64");
        logits = (Launch)dlsym(h, "r4d_rad_logits_gemm");
        count = (int (*)(void))dlsym(h, "rad_kernel_count");
        at = (const RadKernelInfo* (*)(int))dlsym(h, "rad_kernel_at");
        if (!moe) { std::fprintf(stderr, "%s lacks r4d_rad_moe_gemm_mxfp4_gemv\n", path); std::exit(2); }
    }
};

RadTensor tensor(void* p, uint32_t dt, std::initializer_list<int64_t> shape) {
    RadTensor t{};
    t.data = p; t.dtype = dt; t.rank = (uint32_t)shape.size();
    int i = 0;
    for (int64_t s : shape) t.shape[i++] = s;
    int64_t st = 1;
    for (int k = (int)t.rank - 1; k >= 0; --k) { t.stride[k] = st; st *= t.shape[k]; }
    return t;
}
RadParam pi(const char* k, long long v) { RadParam p{}; p.key = k; p.kind = RAD_P_INT; p.ival = v; return p; }
RadParam ps(const char* k, const char* v) { RadParam p{}; p.key = k; p.kind = RAD_P_STR; p.sval = v; return p; }
RadParam pf(const char* k, double v) { RadParam p{}; p.key = k; p.kind = RAD_P_F64; p.dval = v; return p; }

/* One decode token's routes on this rank: `live` local experts in ascending order, one row each;
 * the other top-k slots went to other ranks (dropped, sorted = -1 past the live count). */
struct Route {
    std::vector<int32_t> sorted, off;
    std::vector<int> experts;
};
Route route(std::mt19937_64& rng, int live) {
    Route r;
    while ((int)r.experts.size() < live) {
        const int e = (int)(rng() % kNe);
        bool dup = false;
        for (int x : r.experts) dup |= x == e;
        if (!dup) r.experts.push_back(e);
    }
    std::sort(r.experts.begin(), r.experts.end());
    r.off.assign(kNe + 1, 0);
    for (int e = 0; e < kNe; ++e) {
        int c = 0;
        for (int x : r.experts) c += x == e;
        r.off[e + 1] = r.off[e] + c;
    }
    r.sorted.assign(kTop, -1);
    for (int i = 0; i < live; ++i) r.sorted[i] = i;   /* token 0, top-k slot i */
    return r;
}

/* Random expert in llama.cpp blocks, e8m0 around 2^-8, or from finite edges. */
std::vector<uint8_t> make_blocks(int N, int K, std::mt19937_64& rng, bool edges) {
    const int UK = K / 32;
    std::vector<uint8_t> b((size_t)N * UK * 17);
    static const uint8_t edge[] = { 0, 1, 2, 3, 60, 100, 126, 127, 128, 129, 135, 140 };
    for (size_t blk = 0; blk < (size_t)N * UK; ++blk) {
        const uint64_t r = rng(), a = rng();
        b[blk * 17] = edges && (r & 3) == 0 ? edge[(r >> 2) % sizeof edge] : (uint8_t)(116 + (r >> 8) % 12);
        for (int j = 0; j < 16; ++j) b[blk * 17 + 1 + j] = (uint8_t)(a >> (4 * j));
    }
    return b;
}

std::vector<uint16_t> make_x(int K, std::mt19937_64& rng, bool edges) {
    std::normal_distribution<float> nd(0.f, 0.5f);
    static const uint16_t special[] = { 0x0000, 0x8000, 0x0001, 0x8001, 0x0080, 0x3f80, 0x0040, 0x00ff, 0x4120 };
    std::vector<uint16_t> x((size_t)K);
    for (auto& v : x) {
        v = f32_to_bf16(nd(rng));
        if (edges && rng() % 7 == 0) v = special[rng() % (sizeof special / 2)];
    }
    return x;
}

struct Dev {
    void* p = nullptr;
    size_t n = 0;
    explicit Dev(size_t bytes) : n(bytes) { HC(hipMalloc(&p, bytes)); }
    ~Dev() { (void)hipFree(p); }
    void put(const void* h) { HC(hipMemcpy(p, h, n, hipMemcpyHostToDevice)); }
    void get(void* h) const { HC(hipMemcpy(h, p, n, hipMemcpyDeviceToHost)); }
};

/* moe_gemm_ggml type=mxfp4, gemv path, over a table of per-expert device pointers; T sorted rows
 * (one token's top_k by default). */
int run_moe(Launch f, hipStream_t s, void* x, int AR, void* table, int N, int K, void* sorted,
            void* off, void* y, const char* order, int T = kTop) {
    RadTensor t[5] = { tensor(x, RAD_BF16, { AR, K }),
                       tensor(table, RAD_U8, { kNe, N, K / 32 * 17 }),
                       tensor(sorted, RAD_I32, { T }), tensor(off, RAD_I32, { kNe + 1 }),
                       tensor(y, RAD_BF16, { T, N }) };
    t[1].stride[0] = 0;   /* a table of expert pointers */
    RadParam p[6] = { ps("type", "mxfp4"), pi("N", N), pi("K", K), pi("n_expert", kNe),
                      pi("top_k", kTop), ps("a_order", order) };
    RadArgs a{};
    a.t = t; a.n_t = 5; a.p = p; a.n_p = 6;
    return f(&a, (RadStream)s);
}

int run_swiglu(Launch f, hipStream_t s, void* gu, void* y, int rows, int n) {
    RadTensor t[2] = { tensor(gu, RAD_BF16, { rows, 2 * n }), tensor(y, RAD_BF16, { rows, n }) };
    RadParam p[2] = { pi("n", n), pf("limit", kLimit) };
    RadArgs a{};
    a.t = t; a.n_t = 2; a.p = p; a.n_p = 2;
    return f(&a, (RadStream)s);
}

__global__ void silu_capture(float* out, float limit) {
    const uint32_t h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= 65536) return;
    const float g = fminf(__uint_as_float(h << 16), limit);
    out[h] = g / (1.0f + expf(-g));
}

size_t count_diff(const uint16_t* a, const uint16_t* b, size_t n) {
    size_t d = 0;
    for (size_t i = 0; i < n; ++i) d += a[i] != b[i];
    return d;
}

std::vector<float> load_table(const char* path) {
    std::vector<float> t(65536);
    FILE* f = path ? std::fopen(path, "rb") : nullptr;
    if (!f || std::fread(t.data(), 4, 65536, f) != 65536) { std::fprintf(stderr, "cannot read silu table %s\n", path ? path : "(none)"); std::exit(2); }
    std::fclose(f);
    return t;
}

int check(const Lib& lib, int trials, const char* table_path) {
    const std::vector<float> silu = load_table(table_path);
    hipStream_t s;
    HC(hipStreamCreate(&s));
    std::mt19937_64 rng(2026);
    /* Every table entry must be readable: experts without rows point at a dummy. */
    const size_t gu_b = (size_t)2 * kFf * (kEmbd / 32) * 17, dn_b = (size_t)kEmbd * (kFf / 32) * 17;
    Dev dummy_gu(gu_b), dummy_dn(dn_b);
    HC(hipMemset(dummy_gu.p, 0, gu_b));
    HC(hipMemset(dummy_dn.p, 0, dn_b));
    Dev tab_gu(kNe * sizeof(void*)), tab_dn(kNe * sizeof(void*)), d_sorted(kTop * 4), d_off((kNe + 1) * 4);
    Dev d_x(kEmbd * 2), d_gu((size_t)kTop * 2 * kFf * 2), d_act((size_t)kTop * kFf * 2), d_dn((size_t)kTop * kEmbd * 2);
    size_t bad_gu = 0, bad_act = 0, bad_dn = 0, bad_chain = 0, rows = 0;
    for (int trial = 0; trial < trials; ++trial) {
        const bool edges = trial % 4 == 3;
        const int live = 1 + trial % 3;
        const Route r = route(rng, live);
        std::vector<std::vector<uint8_t>> hgu, hdn;
        std::vector<Dev*> dgu, ddn;
        std::vector<void*> tgu(kNe, dummy_gu.p), tdn(kNe, dummy_dn.p);
        for (int i = 0; i < live; ++i) {
            hgu.push_back(make_blocks(2 * kFf, kEmbd, rng, edges));
            hdn.push_back(make_blocks(kEmbd, kFf, rng, edges));
            dgu.push_back(new Dev(gu_b)); dgu.back()->put(hgu.back().data());
            ddn.push_back(new Dev(dn_b)); ddn.back()->put(hdn.back().data());
            tgu[(size_t)r.experts[(size_t)i]] = dgu.back()->p;
            tdn[(size_t)r.experts[(size_t)i]] = ddn.back()->p;
        }
        tab_gu.put(tgu.data()); tab_dn.put(tdn.data());
        d_sorted.put(r.sorted.data()); d_off.put(r.off.data());
        const auto x = make_x(kEmbd, rng, edges);
        d_x.put(x.data());
        if (run_moe(lib.moe, s, d_x.p, 1, tab_gu.p, 2 * kFf, kEmbd, d_sorted.p, d_off.p, d_gu.p, "token") < 0 ||
            run_swiglu(lib.swiglu, s, d_gu.p, d_act.p, kTop, kFf) < 0 ||
            run_moe(lib.moe, s, d_act.p, kTop, tab_dn.p, kEmbd, kFf, d_sorted.p, d_off.p, d_dn.p, "sorted") < 0) {
            std::fprintf(stderr, "a libr4d launch refused its arguments\n");
            return 2;
        }
        HC(hipStreamSynchronize(s));
        std::vector<uint16_t> g((size_t)kTop * 2 * kFf), a((size_t)kTop * kFf), dn((size_t)kTop * kEmbd);
        d_gu.get(g.data()); d_act.get(a.data()); d_dn.get(dn.data());
        for (int i = 0; i < live; ++i) {
            Mx GU; GU.N = 2 * kFf; GU.K = kEmbd; GU.blocks = hgu[(size_t)i].data();
            Mx DN; DN.N = kEmbd; DN.K = kFf; DN.blocks = hdn[(size_t)i].data();
            std::vector<uint16_t> hg(2 * kFf), ha(kFf), hd(kEmbd), cg(2 * kFf), ca(kFf), cd(kEmbd);
            /* Stage by stage, each from the card's own input... */
            gemv_avx2(GU, x.data(), hg.data(), 0, GU.N);
            swiglu(silu.data(), g.data() + (size_t)i * 2 * kFf, ha.data(), kFf, kLimit);
            gemv_avx2(DN, a.data() + (size_t)i * kFf, hd.data(), 0, DN.N);
            bad_gu += count_diff(hg.data(), g.data() + (size_t)i * 2 * kFf, 2 * kFf);
            bad_act += count_diff(ha.data(), a.data() + (size_t)i * kFf, kFf);
            bad_dn += count_diff(hd.data(), dn.data() + (size_t)i * kEmbd, kEmbd);
            /* ...and the whole chain on the host alone, scalar reference. */
            Expert e{ GU, DN };
            expert_forward(e, silu.data(), kLimit, x.data(), cg.data(), ca.data(), cd.data(), 1, false);
            bad_chain += count_diff(cd.data(), dn.data() + (size_t)i * kEmbd, kEmbd);
            ++rows;
        }
        for (auto* p : dgu) delete p;
        for (auto* p : ddn) delete p;
    }
    std::printf("%d trials, %zu expert rows: differing values gate_up %zu, swiglu %zu, down %zu, whole host chain %zu\n",
                trials, rows, bad_gu, bad_act, bad_dn, bad_chain);
    const bool ok = !bad_gu && !bad_act && !bad_dn && !bad_chain;
    std::printf(ok ? "PASS: host experts are bit-identical to libr4d's decode chain\n" : "FAIL\n");
    return ok ? 0 : 1;
}

int swiglu_proof(const Lib& lib, const char* table_path) {
    if (!lib.swiglu) { std::fprintf(stderr, "no r4d_rad_swiglu_clamped\n"); return 2; }
    hipStream_t s;
    HC(hipStreamCreate(&s));
    /* 1. silu(g) for every bf16 gate, from this card's expf (same compiler as the plugin). */
    Dev d_t(65536 * 4);
    silu_capture<<<256, 256, 0, s>>>((float*)d_t.p, kLimit);
    HC(hipStreamSynchronize(s));
    std::vector<float> table(65536);
    d_t.get(table.data());
    FILE* f = std::fopen(table_path, "wb");
    if (!f || std::fwrite(table.data(), 4, 65536, f) != 65536) { std::fprintf(stderr, "cannot write %s\n", table_path); return 2; }
    std::fclose(f);
    /* 2. Every distinct clamped up value: the bf16 patterns in [-limit, limit] plus NaN, which the
     *    clamp sends to -limit. Rows of 65536 gates each, up constant along a row. */
    std::vector<uint16_t> ups;
    for (uint32_t h = 0; h < 65536; ++h) {
        const float u = bf16_to_f32((uint16_t)h);
        if ((u >= -kLimit && u <= kLimit) || u != u) ups.push_back((uint16_t)h);
    }
    const int R = 256, n = 65536;
    Dev d_gu((size_t)R * 2 * n * 2), d_y((size_t)R * n * 2);
    std::vector<uint16_t> gu((size_t)R * 2 * n), y((size_t)R * n), want((size_t)n);
    /* Mismatches by the up value: a signaling NaN (exponent all ones, quiet bit 0x40 clear, payload
     * not zero), which the host's fmax and the card's clamp may treat differently, and the rest. A
     * gate_up row never holds one: the GEMV narrows through bf16_hi_rne_nan, which quiets NaNs. */
    size_t bad = 0, pairs = 0, bad_snan = 0, pairs_snan = 0, bad_other = 0;
    int shown_snan = 0, shown_other = 0;
    for (size_t base = 0; base < ups.size(); base += R) {
        const int rows = (int)std::min<size_t>(R, ups.size() - base);
        for (int r = 0; r < rows; ++r)
            for (int g = 0; g < n; ++g) { gu[(size_t)r * 2 * n + g] = (uint16_t)g; gu[(size_t)r * 2 * n + n + g] = ups[base + r]; }
        d_gu.put(gu.data());
        if (run_swiglu(lib.swiglu, s, d_gu.p, d_y.p, rows, n) < 0) { std::fprintf(stderr, "swiglu refused\n"); return 2; }
        HC(hipStreamSynchronize(s));
        d_y.get(y.data());
        for (int r = 0; r < rows; ++r) {
            const uint16_t up = ups[base + r];
            const bool snan = (up & 0x7f80) == 0x7f80 && (up & 0x7f) && !(up & 0x40);
            swiglu(table.data(), gu.data() + (size_t)r * 2 * n, want.data(), n, kLimit);
            const uint16_t* got = y.data() + (size_t)r * n;
            for (int g = 0; g < n; ++g) {
                if (want[(size_t)g] == got[g]) continue;
                ++bad;
                if (snan) {
                    ++bad_snan;
                    if (shown_snan++ < 4)
                        std::printf("  sNaN up:  gate 0x%04x up 0x%04x  host 0x%04x  card 0x%04x\n", g, up, want[(size_t)g], got[g]);
                } else {
                    ++bad_other;
                    if (shown_other++ < 16)
                        std::printf("  other:    gate 0x%04x up 0x%04x  host 0x%04x  card 0x%04x\n", g, up, want[(size_t)g], got[g]);
                }
            }
            pairs += (size_t)n;
            pairs_snan += snan ? (size_t)n : 0;
        }
    }
    std::printf("swiglu_clamped: %zu (gate, up) pairs, %zu differing: %zu with a signaling-NaN up (of %zu such "
                "pairs), %zu others; table written to %s\n", pairs, bad, bad_snan, pairs_snan, bad_other, table_path);
    std::printf(bad_other ? "FAIL\n"
                          : "PASS: host swiglu with the captured table equals libr4d's for every bf16 input but a "
                            "signaling-NaN up, which a gate_up row never holds\n");
    return bad_other ? 1 : 0;
}

/* Prefill-shaped calls at EP4: T tokens, each routed to top_k distinct experts of 4 x 96, this
 * rank's (0..95) slots grouped by expert, the rest dropped; gate_up over the tokens' rows (token
 * order), down over the sorted rows. Every live expert's weights in their own buffers. */
/* Tokens pick from a pool of P of the 384 experts (P 384: uniform routing; smaller: the skew of real
 * prompts, where popular experts get several of a chunk's tokens). With L2, the same calls run on a
 * second plugin too, and their outputs are compared value for value. */
int ptime(const Lib& L, const Lib* L2, int T, int P) {
    hipStream_t s;
    HC(hipStreamCreate(&s));
    std::mt19937_64 rng(4242 + (uint64_t)P);
    const size_t gu_b = (size_t)2 * kFf * (kEmbd / 32) * 17, dn_b = (size_t)kEmbd * (kFf / 32) * 17;
    std::vector<int> pool;
    while ((int)pool.size() < std::min(P, 4 * kNe)) {
        const int e = (int)(rng() % (4 * kNe));
        if (std::find(pool.begin(), pool.end(), e) == pool.end()) pool.push_back(e);
    }
    std::vector<std::vector<int32_t>> rows(kNe);
    for (int t = 0; t < T; ++t) {
        std::vector<int> pick;
        while ((int)pick.size() < std::min(kTop, (int)pool.size())) {
            const int e = pool[(size_t)(rng() % pool.size())];
            if (std::find(pick.begin(), pick.end(), e) == pick.end()) pick.push_back(e);
        }
        for (int k = 0; k < kTop; ++k)
            if (pick[(size_t)k] < kNe) rows[(size_t)pick[(size_t)k]].push_back(t * kTop + k);
    }
    std::vector<int32_t> sorted, off(kNe + 1, 0);
    int live = 0;
    for (int e = 0; e < kNe; ++e) {
        off[(size_t)e + 1] = off[(size_t)e] + (int)rows[(size_t)e].size();
        for (int32_t f : rows[(size_t)e]) sorted.push_back(f);
        live += !rows[(size_t)e].empty();
    }
    const int local = (int)sorted.size();
    sorted.resize((size_t)T * kTop, -1);
    std::vector<Dev*> gus, dns;
    std::vector<void*> tgu(kNe), tdn(kNe);
    for (int e = 0; e < kNe; ++e) {
        if (rows[(size_t)e].empty()) continue;
        const auto hg = make_blocks(2 * kFf, kEmbd, rng, false), hd = make_blocks(kEmbd, kFf, rng, false);
        gus.push_back(new Dev(gu_b)); gus.back()->put(hg.data());
        dns.push_back(new Dev(dn_b)); dns.back()->put(hd.data());
        tgu[(size_t)e] = gus.back()->p;
        tdn[(size_t)e] = dns.back()->p;
    }
    for (int e = 0; e < kNe; ++e) {
        if (!tgu[(size_t)e]) tgu[(size_t)e] = gus[0]->p;
        if (!tdn[(size_t)e]) tdn[(size_t)e] = dns[0]->p;
    }
    Dev tab_gu(kNe * sizeof(void*)), tab_dn(kNe * sizeof(void*)), d_sorted(sorted.size() * 4), d_off(off.size() * 4);
    tab_gu.put(tgu.data()); tab_dn.put(tdn.data()); d_sorted.put(sorted.data()); d_off.put(off.data());
    const int R = T * kTop;
    Dev d_x((size_t)T * kEmbd * 2), d_a((size_t)R * kFf * 2), d_ygu((size_t)R * 2 * kFf * 2), d_ydn((size_t)R * kEmbd * 2);
    std::vector<uint16_t> x((size_t)T * kEmbd), act((size_t)R * kFf);
    for (auto& v : x) v = f32_to_bf16(0.5f * (float)((int)(rng() % 200) - 100) / 100.f);
    for (auto& v : act) v = f32_to_bf16(0.1f * (float)((int)(rng() % 200) - 100) / 100.f);
    d_x.put(x.data()); d_a.put(act.data());
    hipEvent_t e0, e1;
    HC(hipEventCreate(&e0)); HC(hipEventCreate(&e1));
    int most = 0;
    for (const auto& r : rows) most = std::max(most, (int)r.size());
    size_t differ = 0;
    for (int which = 0; which < 2; ++which) {
        const bool gu = which == 0;
        const int N = gu ? 2 * kFf : kEmbd, K = gu ? kEmbd : kFf;
        Dev* y = gu ? &d_ygu : &d_ydn;
        std::vector<uint16_t> first((size_t)R * N), second((size_t)R * N);
        for (int lib = 0; lib < (L2 ? 2 : 1); ++lib) {
            const Lib& LL = lib ? *L2 : L;
            std::vector<float> ts;
            for (int it = 0; it < 25; ++it) {
                HC(hipMemset(y->p, 0, y->n));
                HC(hipEventRecord(e0, s));
                if (run_moe(LL.moe, s, gu ? d_x.p : d_a.p, gu ? T : R, gu ? tab_gu.p : tab_dn.p, N, K, d_sorted.p,
                            d_off.p, y->p, gu ? "token" : "sorted", R) < 0) {
                    std::fprintf(stderr, "moe_gemm_ggml refused the prefill shape\n");
                    return 3;
                }
                HC(hipEventRecord(e1, s));
                HC(hipEventSynchronize(e1));
                float ms = 0;
                HC(hipEventElapsedTime(&ms, e0, e1));
                if (it >= 5) ts.push_back(ms);
            }
            y->get((lib ? second : first).data());
            std::sort(ts.begin(), ts.end());
            const double med = ts[ts.size() / 2], bytes = (double)live * N * K / 32 * 17;
            std::printf("prefill T=%d pool %d: %s %s, %d live experts, %d local rows (most %d): median %.1f us, %.1f GB/s\n",
                        T, P, lib ? "lib2" : "lib1", gu ? "gate_up" : "down   ", live, local, most, 1e3 * med,
                        bytes / (med * 1e-3) / 1e9);
        }
        if (L2) differ += count_diff(first.data(), second.data(), first.size());
    }
    if (L2) std::printf("prefill T=%d pool %d: lib1 and lib2 outputs: %zu differing values\n", T, P, differ);
    for (auto* p : gus) delete p;
    for (auto* p : dns) delete p;
    return 0;
}

int timing(const Lib& a, const Lib* b) {
    hipStream_t s;
    HC(hipStreamCreate(&s));
    std::mt19937_64 rng(77);
    constexpr int kCopies = 24;   /* 24 x 12.5 MB of gate_up: well past the 64 MB cache */
    const size_t gu_b = (size_t)2 * kFf * (kEmbd / 32) * 17, dn_b = (size_t)kEmbd * (kFf / 32) * 17;
    std::vector<Dev*> gus, dns;
    for (int i = 0; i < kCopies; ++i) {
        const auto hg = make_blocks(2 * kFf, kEmbd, rng, false), hd = make_blocks(kEmbd, kFf, rng, false);
        gus.push_back(new Dev(gu_b)); gus.back()->put(hg.data());
        dns.push_back(new Dev(dn_b)); dns.back()->put(hd.data());
    }
    Dev tab(kNe * sizeof(void*)), d_sorted(kTop * 4), d_off((kNe + 1) * 4), d_x(kEmbd * 2), d_y1((size_t)kTop * kEmbd * 2),
        d_y2((size_t)kTop * kEmbd * 2), d_a((size_t)kTop * kFf * 2);
    const auto x = make_x(kEmbd, rng, false);
    d_x.put(x.data());
    std::vector<uint16_t> act((size_t)kTop * kFf);
    for (auto& v : act) v = f32_to_bf16(0.1f * (float)((int)(rng() % 200) - 100) / 100.f);
    d_a.put(act.data());
    hipEvent_t e0, e1;
    HC(hipEventCreate(&e0)); HC(hipEventCreate(&e1));
    size_t mism = 0;
    for (int live = 1; live <= 3; ++live)
        for (int which = 0; which < 2; ++which) {
            const bool gu = which == 0;
            const int N = gu ? 2 * kFf : kEmbd, K = gu ? kEmbd : kFf;
            const char* order = gu ? "token" : "sorted";
            void* in = gu ? d_x.p : d_a.p;
            const int AR = gu ? 1 : kTop;
            auto one = [&](const Lib& L, int it, void* y) {
                std::mt19937_64 rr(1000 + it);
                const Route r = route(rr, live);
                std::vector<void*> t(kNe, (gu ? gus : dns)[0]->p);
                for (int i = 0; i < live; ++i) t[(size_t)r.experts[(size_t)i]] = (gu ? gus : dns)[(size_t)((it * 3 + i) % kCopies)]->p;
                tab.put(t.data()); d_sorted.put(r.sorted.data()); d_off.put(r.off.data());
                HC(hipEventRecord(e0, s));
                if (run_moe(L.moe, s, in, AR, tab.p, N, K, d_sorted.p, d_off.p, y, order) < 0) std::exit(3);
                HC(hipEventRecord(e1, s));
                HC(hipEventSynchronize(e1));
                float ms = 0;
                HC(hipEventElapsedTime(&ms, e0, e1));
                return ms;
            };
            for (int lib = 0; lib < (b ? 2 : 1); ++lib) {
                const Lib& L = lib ? *b : a;
                for (int it = 0; it < 5; ++it) (void)one(L, it, d_y1.p);   /* warm-up */
                std::vector<float> t;
                for (int it = 0; it < 60; ++it) t.push_back(one(L, it, d_y1.p));
                std::sort(t.begin(), t.end());
                const double med = t[t.size() / 2], bytes = (double)live * (double)N * K / 32 * 17;
                std::printf("%s %s live=%d: median %.1f us, %.0f GB/s of weights\n", lib ? "lib2" : "lib1",
                            gu ? "gate_up" : "down   ", live, 1e3 * med, bytes / (med * 1e-3) / 1e9);
            }
            if (b)
                for (int it = 0; it < 20; ++it) {
                    (void)one(a, it, d_y1.p); (void)one(*b, it, d_y2.p);
                    std::vector<uint16_t> y1((size_t)kTop * N), y2((size_t)kTop * N);
                    HC(hipMemcpy(y1.data(), d_y1.p, y1.size() * 2, hipMemcpyDeviceToHost));
                    HC(hipMemcpy(y2.data(), d_y2.p, y2.size() * 2, hipMemcpyDeviceToHost));
                    mism += count_diff(y1.data(), y2.data(), y1.size());
                }
        }
    if (b) std::printf(mism ? "FAIL: lib2 differs from lib1 in %zu values\n" : "PASS: lib2 outputs bit-identical to lib1\n", mism);
    for (auto* p : gus) delete p;
    for (auto* p : dns) delete p;
    return mism ? 1 : 0;
}

/* ================================================================== tall (..._EXACT_TALL)
 *
 * A pass taller than 16 tokens must give a 16-token pass's bits on the three ops whose arithmetic
 * follows the pass height: the routed GEMM (the twin row, the gemv at any number of tokens), and the
 * sparse attention and the indexer score (their scalar forms up to 48 rows, and above under the
 * switch). Each is launched once at the tall height and again over the same rows cut as smaller
 * launches would cut them, and every output value compared. */

/* The plugin's twin row, as the core resolves it: present (the switch was set before the table was
 * first read), its launch the tall entry, the gemv's constraints with its bound on M raised past 16
 * (to the tallest pass libr4d keeps a 16-token pass's bits at), one priority below the gemv. Its
 * bound in *reach (0 when the row is missing). */
int tall_row_table(const Lib& L, long long* reach) {
    *reach = 0;
    if (!L.count || !L.at || !L.moe_tall) {
        std::printf("FAIL: the plugin has no row table or no r4d_rad_moe_gemm_mxfp4_gemv_tall\n");
        return 1;
    }
    const RadKernelInfo *gemv = nullptr, *tall = nullptr;
    for (int i = 0; i < L.count(); ++i) {
        const RadKernelInfo* k = L.at(i);
        if (!k || !k->name) continue;
        if (!std::strcmp(k->name, "moe_gemm_ggml_mxfp4_gemv")) gemv = k;
        if (!std::strcmp(k->name, "moe_gemm_ggml_mxfp4_gemv_tall")) tall = k;
    }
    if (!gemv || !tall) {
        std::printf("FAIL: row table: moe_gemm_ggml_mxfp4_gemv %s, moe_gemm_ggml_mxfp4_gemv_tall %s\n",
                    gemv ? "present" : "missing", tall ? "present" : "missing");
        return 1;
    }
    int m_le = 0;
    for (int i = 0; i < tall->n_constraints; ++i)
        if (!std::strcmp(tall->constraints[i].key, "M") && tall->constraints[i].op == RAD_C_LE) {
            ++m_le;
            *reach = tall->constraints[i].ival;
        }
    const bool ok = tall->launch == L.moe_tall && m_le == 1 && *reach > 16 && tall->priority == gemv->priority - 1 &&
                    tall->n_constraints == gemv->n_constraints && tall->layout == gemv->layout &&
                    tall->relayout == gemv->relayout;
    std::printf("%s: row table: %s, priority %d (the gemv's %d), %d constraints, M up to %lld\n",
                ok ? "PASS" : "FAIL", tall->name, tall->priority, gemv->priority, tall->n_constraints, *reach);
    return ok ? 0 : 1;
}

/* One rank's routing of T tokens: each picks top_k distinct experts of 4 x 96 from a pool of P
 * (with a popular local expert first in the pool, which a small P gives every token); this rank's
 * slots grouped by expert as moe_scatter writes them, the rest -1. */
struct TallRoute {
    std::vector<std::vector<int>> picks;
};
TallRoute tall_picks(std::mt19937_64& rng, int T, int P) {
    std::vector<int> pool;
    while ((int)pool.size() < std::min(P, 4 * kNe)) {
        const int e = (int)(rng() % (4 * kNe));
        if (std::find(pool.begin(), pool.end(), e) == pool.end()) pool.push_back(e);
    }
    for (int& e : pool)
        if (e < kNe) { std::swap(e, pool[0]); break; }
    if (pool[0] >= kNe) pool[0] = (int)(rng() % kNe);
    TallRoute r;
    r.picks.resize((size_t)T);
    for (int t = 0; t < T; ++t) {
        std::vector<int>& pk = r.picks[(size_t)t];
        if (P < 4 * kNe) pk.push_back(pool[0]);
        while ((int)pk.size() < kTop) {
            const int e = (int)pool.size() > kTop ? pool[(size_t)(rng() % pool.size())] : (int)(rng() % (4 * kNe));
            if (std::find(pk.begin(), pk.end(), e) == pk.end()) pk.push_back(e);
        }
        std::shuffle(pk.begin(), pk.end(), rng);
    }
    return r;
}
/* Tokens [t0, t1) of r as one pass: sorted (flat indices relative to t0), off, local count. */
void tall_pass(const TallRoute& r, int t0, int t1, std::vector<int32_t>* sorted, std::vector<int32_t>* off, int* local) {
    std::vector<std::vector<int32_t>> by(kNe);
    for (int t = t0; t < t1; ++t)
        for (int k = 0; k < kTop; ++k) {
            const int e = r.picks[(size_t)t][(size_t)k];
            if (e < kNe) by[(size_t)e].push_back((t - t0) * kTop + k);
        }
    sorted->clear();
    off->assign(kNe + 1, 0);
    for (int e = 0; e < kNe; ++e) {
        (*off)[(size_t)e + 1] = (*off)[(size_t)e] + (int)by[(size_t)e].size();
        for (int32_t f : by[(size_t)e]) sorted->push_back(f);
    }
    *local = (int)sorted->size();
    sorted->resize((size_t)(t1 - t0) * kTop, -1);
}

/* The routed GEMM at T tokens on the twin against the gemv in passes of 16 and against the host
 * kernels, both projections at ds41's shapes. Returns the differing values. */
size_t tall_moe(const Lib& L, hipStream_t s, int T, int P, size_t* rows_checked, size_t* tiled_differ) {
    std::mt19937_64 rng(7000 + 31 * (uint64_t)T + (uint64_t)P);
    const TallRoute r = tall_picks(rng, T, P);
    std::vector<int32_t> sorted, off;
    int local = 0;
    tall_pass(r, 0, T, &sorted, &off, &local);
    const size_t gu_b = (size_t)2 * kFf * (kEmbd / 32) * 17, dn_b = (size_t)kEmbd * (kFf / 32) * 17;
    std::vector<std::vector<uint8_t>> hgu(kNe), hdn(kNe);
    std::vector<Dev*> dev;
    std::vector<void*> tgu(kNe, nullptr), tdn(kNe, nullptr);
    for (int e = 0; e < kNe; ++e) {
        if (off[(size_t)e + 1] == off[(size_t)e] && e != 0) continue;
        hgu[(size_t)e] = make_blocks(2 * kFf, kEmbd, rng, e % 5 == 3);
        hdn[(size_t)e] = make_blocks(kEmbd, kFf, rng, e % 5 == 3);
        dev.push_back(new Dev(gu_b)); dev.back()->put(hgu[(size_t)e].data()); tgu[(size_t)e] = dev.back()->p;
        dev.push_back(new Dev(dn_b)); dev.back()->put(hdn[(size_t)e].data()); tdn[(size_t)e] = dev.back()->p;
    }
    for (int e = 0; e < kNe; ++e) {   /* an expert without rows points at a readable weight */
        if (!tgu[(size_t)e]) tgu[(size_t)e] = tgu[0];
        if (!tdn[(size_t)e]) tdn[(size_t)e] = tdn[0];
    }
    Dev tab_gu(kNe * sizeof(void*)), tab_dn(kNe * sizeof(void*));
    tab_gu.put(tgu.data()); tab_dn.put(tdn.data());
    std::vector<uint16_t> x((size_t)T * kEmbd), act((size_t)T * kTop * kFf);
    for (int t = 0; t < T; ++t) {
        const auto row = make_x(kEmbd, rng, t % 7 == 3);
        std::copy(row.begin(), row.end(), x.begin() + (size_t)t * kEmbd);
    }
    std::normal_distribution<float> nd(0.f, 0.3f);
    for (auto& v : act) v = f32_to_bf16(nd(rng));
    size_t differ = 0;
    for (int proj = 0; proj < 2; ++proj) {
        const bool g = proj == 0;
        const int N = g ? 2 * kFf : kEmbd, K = g ? kEmbd : kFf;
        void* table = g ? tab_gu.p : tab_dn.p;
        /* The tall pass. Sorted rows read their own (token, slot) row of act. */
        std::vector<uint16_t> in_all;
        if (g) in_all = x;
        else {
            in_all.assign((size_t)T * kTop * K, 0);
            for (int i = 0; i < local; ++i)
                std::memcpy(&in_all[(size_t)i * K], &act[(size_t)sorted[(size_t)i] * K], (size_t)K * 2);
        }
        Dev d_in(in_all.size() * 2), d_so(sorted.size() * 4), d_of(off.size() * 4), d_y((size_t)T * kTop * N * 2);
        d_in.put(in_all.data()); d_so.put(sorted.data()); d_of.put(off.data());
        HC(hipMemset(d_y.p, 0xAA, d_y.n));
        if (run_moe(L.moe_tall, s, d_in.p, g ? T : T * kTop, table, N, K, d_so.p, d_of.p, d_y.p,
                    g ? "token" : "sorted", T * kTop) < 0) {
            std::printf("FAIL: the tall launch refused T=%d\n", T);
            return (size_t)1 << 40;
        }
        HC(hipStreamSynchronize(s));
        std::vector<uint16_t> y_all((size_t)T * kTop * N);
        d_y.get(y_all.data());
        /* What the twin replaces: the tiled row at the same height (information only). */
        if (L.moe_tiled && T > 16) {
            HC(hipMemset(d_y.p, 0, d_y.n));
            if (run_moe(L.moe_tiled, s, d_in.p, g ? T : T * kTop, table, N, K, d_so.p, d_of.p, d_y.p,
                        g ? "token" : "sorted", T * kTop) >= 0) {
                HC(hipStreamSynchronize(s));
                std::vector<uint16_t> y_t((size_t)T * kTop * N);
                d_y.get(y_t.data());
                *tiled_differ += count_diff(y_t.data(), y_all.data(), (size_t)local * N);
            }
        }
        /* The same tokens in passes of at most 16, each with its own routing, on the gemv row. */
        for (int t0 = 0; t0 < T; t0 += 16) {
            const int t1 = std::min(T, t0 + 16), n = t1 - t0;
            std::vector<int32_t> so_c, of_c;
            int loc_c = 0;
            tall_pass(r, t0, t1, &so_c, &of_c, &loc_c);
            std::vector<uint16_t> in_c;
            if (g) in_c.assign(x.begin() + (size_t)t0 * kEmbd, x.begin() + (size_t)t1 * kEmbd);
            else {
                in_c.assign((size_t)n * kTop * K, 0);
                for (int i = 0; i < loc_c; ++i)
                    std::memcpy(&in_c[(size_t)i * K], &act[(size_t)(so_c[(size_t)i] + t0 * kTop) * K], (size_t)K * 2);
            }
            Dev c_in(in_c.size() * 2), c_so(so_c.size() * 4), c_of(of_c.size() * 4), c_y((size_t)n * kTop * N * 2);
            c_in.put(in_c.data()); c_so.put(so_c.data()); c_of.put(of_c.data());
            if (run_moe(L.moe, s, c_in.p, g ? n : n * kTop, table, N, K, c_so.p, c_of.p, c_y.p,
                        g ? "token" : "sorted", n * kTop) < 0) {
                std::printf("FAIL: a 16-token launch refused\n");
                return (size_t)1 << 40;
            }
            HC(hipStreamSynchronize(s));
            std::vector<uint16_t> y_c((size_t)n * kTop * N);
            c_y.get(y_c.data());
            for (int i = 0; i < loc_c; ++i) {
                const int f = so_c[(size_t)i] + t0 * kTop;
                int ra = -1;
                for (int j = 0; j < local && ra < 0; ++j)
                    if (sorted[(size_t)j] == f) ra = j;
                if (ra < 0) { std::printf("FAIL: slot %d routed in one layout only\n", f); return (size_t)1 << 40; }
                const size_t d = count_diff(&y_all[(size_t)ra * N], &y_c[(size_t)i * N], (size_t)N);
                if (d && differ < 4)
                    std::printf("  T=%d P=%d %s: token %d slot %d differs in %zu of %d values from its 16-token pass\n",
                                T, P, g ? "gate_up" : "down", f / kTop, f % kTop, d, N);
                differ += d;
                ++*rows_checked;
            }
        }
        /* The host kernels (the CPU path's bits) on every local row. */
        for (int i = 0; i < local; ++i) {
            const int f = sorted[(size_t)i], e = [&] { int ee = 0; while (off[(size_t)ee + 1] <= i) ++ee; return ee; }();
            Mx W; W.N = N; W.K = K; W.blocks = g ? hgu[(size_t)e].data() : hdn[(size_t)e].data();
            std::vector<uint16_t> hy((size_t)N);
            gemv_avx2(W, g ? &x[(size_t)(f / kTop) * kEmbd] : &act[(size_t)f * K], hy.data(), 0, N);
            const size_t d = count_diff(hy.data(), &y_all[(size_t)i * N], (size_t)N);
            if (d && differ < 4)
                std::printf("  T=%d P=%d %s: sorted row %d differs from the host kernel in %zu values\n", T, P,
                            g ? "gate_up" : "down", i, d);
            differ += d;
        }
    }
    for (auto* p : dev) delete p;
    return differ;
}

/* A one-sequence prefill chunk of M rows at positions P0.., as deepseek41 hands it to the sparse
 * ops: a window cache of bs-token blocks from position 0 (block table permuted), the compressed
 * cache at `ratio`, top-k selections per row (-1 for none), and the indexer's cache. */
struct SparseFix {
    static constexpr int D = 512, win = 128, topk = 512, ratio = 4, bs = 16, cbs = 16;
    static constexpr int IH = 32, ID = 128, IR = 4, ICBS = 16;   /* ds41's index_n_heads, on every rank */
    int H = 32;   /* attention heads a rank: deepseek41's n_head / 4 */
    int M = 0, P0 = 0, nwbt = 0, ncbt = 0, C = 0;
    std::vector<uint16_t> q, wc, cc, iq, iw, kc;
    std::vector<int32_t> wbt, cbt, sel, pos, ibt;
    std::vector<float> sink;
};
SparseFix sparse_fix(int M, int H, uint32_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.f, 0.5f);
    SparseFix f;
    f.H = H;
    f.M = M;
    f.P0 = 333;
    const int last = f.P0 + M - 1;
    f.nwbt = last / f.bs + 1;
    const int ncomp = (last + 1) / f.ratio;          /* compressed entries the chunk can select */
    f.ncbt = (ncomp * f.ratio) / f.cbs + 1;
    f.q.resize((size_t)M * f.H * f.D);
    for (auto& v : f.q) v = f32_to_bf16(nd(rng));
    f.wc.resize((size_t)f.nwbt * f.bs * f.D);
    for (auto& v : f.wc) v = f32_to_bf16(nd(rng));
    f.cc.resize((size_t)f.ncbt * f.cbs * (f.D / f.ratio));
    for (auto& v : f.cc) v = f32_to_bf16(nd(rng));
    f.wbt.resize((size_t)f.nwbt);
    for (int b = 0; b < f.nwbt; ++b) f.wbt[(size_t)b] = (b * 7 + 3) % f.nwbt;
    f.cbt.resize((size_t)f.ncbt);
    for (int b = 0; b < f.ncbt; ++b) f.cbt[(size_t)b] = (b * 5 + 1) % f.ncbt;
    f.pos.resize((size_t)M);
    for (int m = 0; m < M; ++m) f.pos[(size_t)m] = f.P0 + m;
    f.sel.assign((size_t)M * f.topk, -1);
    for (int m = 0; m < M; ++m) {
        const int reach = (f.pos[(size_t)m] + 1) / f.ratio;
        for (int i = 0; i < f.topk && i < reach; ++i)
            if (rng() % 9) f.sel[(size_t)m * f.topk + i] = (int32_t)(rng() % (uint64_t)reach);
    }
    f.sink.resize((size_t)f.H);
    for (auto& v : f.sink) v = nd(rng);
    f.C = (last + 1) / f.IR + 1;
    f.iq.resize((size_t)M * f.IH * f.ID);
    for (auto& v : f.iq) v = f32_to_bf16(nd(rng));
    f.iw.resize((size_t)M * f.IH);
    for (auto& v : f.iw) v = f32_to_bf16(nd(rng));
    const int nib = (f.C * f.IR) / f.ICBS + 1;
    f.kc.resize((size_t)nib * f.ICBS * (f.ID / f.IR));
    for (auto& v : f.kc) v = f32_to_bf16(nd(rng));
    f.ibt.resize((size_t)nib);
    for (int b = 0; b < nib; ++b) f.ibt[(size_t)b] = (b * 3 + 2) % nib;
    return f;
}

/* Rows [r0, r1) of the chunk as one launch of each sparse op (S = 1, cu {0, n}; the window table
 * rebased at the chunk's first = 0 by window_used = pos(last row) + 1). Outputs into out (bf16 rows
 * of H * D) and sc (f32 rows of C). */
int sparse_launch(const Lib& L, hipStream_t s, const SparseFix& f, int r0, int r1, std::vector<uint16_t>* out,
                  std::vector<float>* sc) {
    const int n = r1 - r0;
    const int32_t cu[2] = { 0, n }, used = f.pos[(size_t)r1 - 1] + 1;
    Dev q((size_t)n * f.H * f.D * 2), wc(f.wc.size() * 2), wbt(f.wbt.size() * 4), wu(4), cc(f.cc.size() * 2),
        cbt(f.cbt.size() * 4), sel((size_t)n * f.topk * 4), pos((size_t)n * 4), dcu(8), sink(f.sink.size() * 4),
        o((size_t)n * f.H * f.D * 2);
    q.put(&f.q[(size_t)r0 * f.H * f.D]); wc.put(f.wc.data()); wbt.put(f.wbt.data()); wu.put(&used);
    cc.put(f.cc.data()); cbt.put(f.cbt.data()); sel.put(&f.sel[(size_t)r0 * f.topk]);
    pos.put(&f.pos[(size_t)r0]); dcu.put(cu); sink.put(f.sink.data());
    HC(hipMemset(o.p, 0xAA, o.n));
    {
        RadTensor t[11] = { tensor(q.p, RAD_BF16, { n, f.H * f.D }),
                            tensor(wc.p, RAD_BF16, { (int64_t)f.nwbt, 1, f.bs, f.D }),
                            tensor(wbt.p, RAD_I32, { 1, f.nwbt }), tensor(wu.p, RAD_I32, { 1 }),
                            tensor(cc.p, RAD_BF16, { (int64_t)f.ncbt, 1, f.cbs, f.D / f.ratio }),
                            tensor(cbt.p, RAD_I32, { 1, f.ncbt }), tensor(sel.p, RAD_I32, { n, f.topk }),
                            tensor(pos.p, RAD_I32, { n }), tensor(dcu.p, RAD_I32, { 2 }),
                            tensor(sink.p, RAD_F32, { f.H }), tensor(o.p, RAD_BF16, { n, f.H * f.D }) };
        RadParam p[8] = { pi("M", n), pi("heads", f.H), pi("dim", f.D), pi("window", f.win), pi("ratio", f.ratio),
                          pi("topk", f.topk), pf("scale", 0.04419417382415922), ps("dtype", "bf16") };
        RadArgs a{};
        a.t = t; a.n_t = 11; a.p = p; a.n_p = 8;
        if (L.attn(&a, (RadStream)s) < 0) return -1;
    }
    const int nib = (int)f.ibt.size();
    Dev iq((size_t)n * f.IH * f.ID * 2), iw((size_t)n * f.IH * 2), kc(f.kc.size() * 2), ibt(f.ibt.size() * 4),
        isc((size_t)n * f.C * 4);
    iq.put(&f.iq[(size_t)r0 * f.IH * f.ID]); iw.put(&f.iw[(size_t)r0 * f.IH]); kc.put(f.kc.data());
    ibt.put(f.ibt.data());
    HC(hipMemset(isc.p, 0xAA, isc.n));
    {
        RadTensor t[7] = { tensor(iq.p, RAD_BF16, { n, f.IH * f.ID }), tensor(iw.p, RAD_BF16, { n, f.IH }),
                           tensor(kc.p, RAD_BF16, { (int64_t)nib, 1, f.ICBS, f.ID / f.IR }),
                           tensor(ibt.p, RAD_I32, { 1, nib }), tensor(pos.p, RAD_I32, { n }),
                           tensor(dcu.p, RAD_I32, { 2 }), tensor(isc.p, RAD_F32, { n, f.C }) };
        RadParam p[7] = { pi("M", n), pi("heads", f.IH), pi("dim", f.ID), pi("ratio", f.IR), pi("width", f.C),
                          pf("wscale", 0.125), ps("dtype", "bf16") };
        RadArgs a{};
        a.t = t; a.n_t = 7; a.p = p; a.n_p = 7;
        if (L.iscore(&a, (RadStream)s) < 0) return -2;
    }
    HC(hipStreamSynchronize(s));
    out->resize((size_t)n * f.H * f.D);
    o.get(out->data());
    sc->resize((size_t)n * f.C);
    isc.get(sc->data());
    return 0;
}

/* The chunk's rows at height M against the same rows one at a time (the decode form) and in
 * launches of 16. Returns the differing values (attention bf16 values, index score f32 bit patterns). */
size_t tall_sparse(const Lib& L, hipStream_t s, int M, int H, size_t* rows_checked) {
    const SparseFix f = sparse_fix(M, H, 9100u + (uint32_t)M + 1000u * (uint32_t)H);
    std::vector<uint16_t> out;
    std::vector<float> sc;
    if (sparse_launch(L, s, f, 0, M, &out, &sc) < 0) { std::printf("FAIL: a sparse launch refused M=%d\n", M); return (size_t)1 << 40; }
    size_t differ = 0;
    for (int cut : { 1, 16 })
        for (int r0 = 0; r0 < M; r0 += cut) {
            const int r1 = std::min(M, r0 + cut);
            std::vector<uint16_t> o2;
            std::vector<float> s2;
            if (sparse_launch(L, s, f, r0, r1, &o2, &s2) < 0) { std::printf("FAIL: a sparse launch refused\n"); return (size_t)1 << 40; }
            const size_t da = count_diff(&out[(size_t)r0 * f.H * f.D], o2.data(), o2.size());
            size_t ds = 0;
            for (size_t i = 0; i < s2.size(); ++i)
                ds += std::memcmp(&sc[(size_t)r0 * f.C + i], &s2[i], 4) != 0;
            if ((da || ds) && differ < 4)
                std::printf("  M=%d H=%d rows %d..%d (launches of %d): attention %zu, index score %zu differing values\n",
                            M, H, r0, r1 - 1, cut, da, ds);
            differ += da + ds;
            *rows_checked += (size_t)(r1 - r0);
        }
    return differ;
}

int tall_check(const Lib& L, const std::vector<int>& Ts) {
    long long reach = 0;
    int rc = tall_row_table(L, &reach);
    if (!L.moe_tall) return 1;
    hipStream_t s;
    HC(hipStreamCreate(&s));
    size_t moe_bad = 0, moe_rows = 0, tiled = 0;
    for (int T : Ts)
        for (int P : { 12, 4 * kNe }) {
            size_t rows = 0;
            const size_t d = tall_moe(L, s, T, P, &rows, &tiled);
            std::printf("moe_gemm_ggml T=%3d pool %3d: %zu (token, slot) rows against passes of 16 and the host: %zu "
                        "differing values\n", T, P, rows, d);
            moe_bad += d;
            moe_rows += rows;
        }
    std::printf("%s: the twin at %zu rows over %zu heights equals the gemv in passes of 16 and the host kernels "
                "(the tiled row at the same heights differs in %zu values)\n", moe_bad ? "FAIL" : "PASS", moe_rows,
                Ts.size(), tiled);
    rc |= moe_bad ? 1 : 0;
    if (!L.attn || !L.iscore) {
        std::printf("FAIL: the plugin lacks r4d_rad_ds41_attn_sparse_sink or r4d_rad_ds41_index_score\n");
        return 1;
    }
    size_t sp_bad = 0, sp_rows = 0;
    for (int H : { 16, 32 })
        for (int M : { 17, 24, 33, 40, 47, 48, 64, 100 }) {
            size_t rows = 0;
            const size_t d = tall_sparse(L, s, M, H, &rows);
            std::printf("attn_sparse_sink (%d heads) + index_score M=%3d: %zu rows against single rows and launches "
                        "of 16: %zu differing values\n", H, M, rows, d);
            sp_bad += d;
            sp_rows += rows;
        }
    std::printf("%s: sparse attention and index score at 17..100 rows equal their rows launched alone and 16 at a "
                "time (%zu rows)\n", sp_bad ? "FAIL" : "PASS", sp_rows);
    rc |= sp_bad ? 1 : 0;
    std::printf(rc ? "FAIL\n" : "PASS: tall passes give 16-token passes' bits on all three ops\n");
    return rc;
}

/* ================================================================== sweep (the height-dependent ops)
 *
 * Every other op of ds41's graph whose launcher picks its arithmetic, its width or its form from the
 * pass height, at the heights a step can carry, each row's bytes against the same row launched alone
 * (a decode pass) and in launches of 16 (today's chunking): the norms (rmsnorm at the widths ds41
 * declares it, the fused collapse, post-collapse and norm-rope-roundtrip norms), whose width
 * r4d_act_threads halves above 32 rows; mhc_mix, whose split form changes at 8 and 24 rows; the bf16
 * skinny GEMM (its MB) and the lm_head. A height at or below the reach must give every row's bytes
 * (FAIL otherwise): with the switch, the twin's bound in this plugin; without it 32, where the norms'
 * width still holds. Above the reach a norm may differ (counted, not a failure); any other op must
 * not. */
struct SweepOut {
    Dev* d;
    size_t row_bytes;
};
struct SweepOp {
    std::string name;
    bool norm = false;                         /* its width follows r4d_act_threads */
    std::vector<std::unique_ptr<Dev>> bufs;
    std::vector<SweepOut> outs;
    std::function<int(int r0, int n, hipStream_t s)> launch;   /* rows [r0, r0 + n) as one pass */
    Dev* buf(size_t bytes) {
        bufs.emplace_back(new Dev(bytes));
        return bufs.back().get();
    }
};

constexpr int kSweepRows = 48;

void fill_bf16(Dev* d, std::mt19937_64& rng, float sd, float mean = 0.f) {
    std::normal_distribution<float> nd(mean, sd);
    std::vector<uint16_t> h(d->n / 2);
    for (auto& v : h) v = f32_to_bf16(nd(rng));
    d->put(h.data());
}
void fill_f32(Dev* d, std::mt19937_64& rng, float sd, float mean = 0.f, bool positive = false) {
    std::normal_distribution<float> nd(mean, sd);
    std::vector<float> h(d->n / 4);
    for (auto& v : h) { v = nd(rng); if (positive) v = std::fabs(v) + 0.05f; }
    d->put(h.data());
}

/* A [rows, cols] operand view of `d` from row r0, rows `n`. */
RadTensor rows_of(Dev* d, uint32_t dt, int64_t cols, int r0, int n) {
    const int es = dt == RAD_F32 || dt == RAD_I32 ? 4 : 2;
    return tensor((char*)d->p + (size_t)r0 * cols * es, dt, { n, cols });
}

std::unique_ptr<SweepOp> sweep_rmsnorm(Launch f, int n, std::mt19937_64& rng) {
    auto o = std::make_unique<SweepOp>();
    o->name = "rmsnorm n=" + std::to_string(n);
    o->norm = true;
    Dev* x = o->buf((size_t)kSweepRows * n * 2);
    Dev* w = o->buf((size_t)n * 4);
    Dev* y = o->buf((size_t)kSweepRows * n * 2);
    fill_bf16(x, rng, 1.0f);
    fill_f32(w, rng, 0.2f, 1.0f);
    o->outs = { { y, (size_t)n * 2 } };
    o->launch = [=](int r0, int m, hipStream_t s) {
        RadTensor t[3] = { rows_of(x, RAD_BF16, n, r0, m), tensor(w->p, RAD_F32, { n }), rows_of(y, RAD_BF16, n, r0, m) };
        RadParam p[3] = { pi("n", n), pf("eps", 1e-20), ps("dtype", "bf16") };
        RadArgs a{};
        a.t = t; a.n_t = 3; a.p = p; a.n_p = 3;
        return f(&a, (RadStream)s);
    };
    return o;
}

std::unique_ptr<SweepOp> sweep_collapse_norm(Launch f, std::mt19937_64& rng) {
    const int n = kEmbd, hc = 4;
    auto o = std::make_unique<SweepOp>();
    o->name = "mhc_collapse_rmsnorm n=5120 hc=4";
    o->norm = true;
    Dev* h = o->buf((size_t)kSweepRows * hc * n * 2);
    Dev* pre = o->buf((size_t)kSweepRows * hc * 4);
    Dev* x = o->buf((size_t)kSweepRows * n * 2);
    Dev* w = o->buf((size_t)n * 4);
    Dev* y = o->buf((size_t)kSweepRows * n * 2);
    fill_bf16(h, rng, 1.0f);
    fill_f32(pre, rng, 0.4f, 0.0f, true);
    fill_f32(w, rng, 0.2f, 1.0f);
    o->outs = { { x, (size_t)n * 2 }, { y, (size_t)n * 2 } };
    o->launch = [=](int r0, int m, hipStream_t s) {
        RadTensor t[5] = { rows_of(h, RAD_BF16, hc * n, r0, m), rows_of(pre, RAD_F32, hc, r0, m),
                           rows_of(x, RAD_BF16, n, r0, m), tensor(w->p, RAD_F32, { n }),
                           rows_of(y, RAD_BF16, n, r0, m) };
        RadParam p[4] = { pi("n", n), pi("hc", hc), pf("eps", 1e-20), ps("dtype", "bf16") };
        RadArgs a{};
        a.t = t; a.n_t = 5; a.p = p; a.n_p = 4;
        return f(&a, (RadStream)s);
    };
    return o;
}

std::unique_ptr<SweepOp> sweep_post_collapse_norm(Launch f, std::mt19937_64& rng) {
    const int n = kEmbd, hc = 4;
    auto o = std::make_unique<SweepOp>();
    o->name = "mhc_post_collapse_rmsnorm n=5120 hc=4";
    o->norm = true;
    Dev* xi = o->buf((size_t)kSweepRows * n * 2);
    Dev* h = o->buf((size_t)kSweepRows * hc * n * 2);
    Dev* post = o->buf((size_t)kSweepRows * hc * 4);
    Dev* comb = o->buf((size_t)kSweepRows * hc * hc * 4);
    Dev* out = o->buf((size_t)kSweepRows * hc * n * 2);
    Dev* pre = o->buf((size_t)kSweepRows * hc * 4);
    Dev* x = o->buf((size_t)kSweepRows * n * 2);
    Dev* w = o->buf((size_t)n * 4);
    Dev* y = o->buf((size_t)kSweepRows * n * 2);
    fill_bf16(xi, rng, 1.0f);
    fill_bf16(h, rng, 1.0f);
    fill_f32(post, rng, 0.5f, 0.0f, true);
    fill_f32(comb, rng, 0.3f, 0.0f, true);
    fill_f32(pre, rng, 0.4f, 0.0f, true);
    fill_f32(w, rng, 0.2f, 1.0f);
    o->outs = { { out, (size_t)hc * n * 2 }, { x, (size_t)n * 2 }, { y, (size_t)n * 2 } };
    o->launch = [=](int r0, int m, hipStream_t s) {
        RadTensor t[9] = { rows_of(xi, RAD_BF16, n, r0, m), rows_of(h, RAD_BF16, hc * n, r0, m),
                           rows_of(post, RAD_F32, hc, r0, m), rows_of(comb, RAD_F32, hc * hc, r0, m),
                           rows_of(out, RAD_BF16, hc * n, r0, m), rows_of(pre, RAD_F32, hc, r0, m),
                           rows_of(x, RAD_BF16, n, r0, m), tensor(w->p, RAD_F32, { n }),
                           rows_of(y, RAD_BF16, n, r0, m) };
        RadParam p[4] = { pi("n", n), pi("hc", hc), pf("eps", 1e-20), ps("dtype", "bf16") };
        RadArgs a{};
        a.t = t; a.n_t = 9; a.p = p; a.n_p = 4;
        return f(&a, (RadStream)s);
    };
    return o;
}

/* rmsnorm, rope and the quant round trip of one head of n columns (ds41's window key: n 512, E4M3
 * with ue8m0 a 32-block; its index key: n 128, E2M1 with ue8m0 a 32-block). */
std::unique_ptr<SweepOp> sweep_norm_rope(Launch f, int n, const char* fmt, std::mt19937_64& rng) {
    const int rot = 64, P = 512;
    auto o = std::make_unique<SweepOp>();
    o->name = std::string("rmsnorm_rope_roundtrip n=") + std::to_string(n) + " " + fmt;
    o->norm = true;
    Dev* x = o->buf((size_t)kSweepRows * n * 2);
    Dev* w = o->buf((size_t)n * 4);
    Dev* y = o->buf((size_t)kSweepRows * n * 2);
    Dev* cs = o->buf((size_t)P * rot * 4);
    Dev* pos = o->buf((size_t)kSweepRows * 4);
    fill_bf16(x, rng, 1.0f);
    fill_f32(w, rng, 0.2f, 1.0f);
    {
        std::vector<float> t((size_t)P * rot);
        for (int q = 0; q < P; ++q)
            for (int i = 0; i < rot / 2; ++i) {
                const double a = q * std::pow(10000.0, -2.0 * i / rot);
                t[(size_t)q * rot + i] = (float)std::cos(a);
                t[(size_t)q * rot + rot / 2 + i] = (float)std::sin(a);
            }
        cs->put(t.data());
        std::vector<int32_t> ps_(kSweepRows);
        for (int m = 0; m < kSweepRows; ++m) ps_[(size_t)m] = 333 + m;
        pos->put(ps_.data());
    }
    o->outs = { { y, (size_t)n * 2 } };
    const std::string f_ = fmt;
    o->launch = [=](int r0, int m, hipStream_t s) {
        RadTensor t[5] = { rows_of(x, RAD_BF16, n, r0, m), tensor(w->p, RAD_F32, { n }),
                           rows_of(y, RAD_BF16, n, r0, m), tensor(cs->p, RAD_F32, { P, rot }),
                           tensor((char*)pos->p + (size_t)r0 * 4, RAD_I32, { m }) };
        RadParam p[13] = { pi("n", n), pf("eps", 1e-20), pi("n_head", 1), pi("head_dim", n), pi("rot", rot),
                           pi("off", n - rot), ps("mode", "gptj"), pi("inverse", 0), pi("block", 32),
                           ps("format", f_.c_str()), ps("scale", "ue8m0"), ps("dtype", "bf16"), pi("M", m) };
        RadArgs a{};
        a.t = t; a.n_t = 5; a.p = p; a.n_p = 13;
        return f(&a, (RadStream)s);
    };
    return o;
}

/* ds41's connection mix: hc 4 streams of 5120, 20 Sinkhorn iterations, with scratch (the split
 * forms the engine runs) or without (the fused form). */
std::unique_ptr<SweepOp> sweep_mix(Launch f, bool scratch, std::mt19937_64& rng) {
    const int n = kEmbd, hc = 4, mix = hc * (hc + 2), W = hc * n;
    auto o = std::make_unique<SweepOp>();
    o->name = std::string("mhc_mix n=5120 hc=4") + (scratch ? " (split, scratch)" : " (fused)");
    Dev* h = o->buf((size_t)kSweepRows * W * 2);
    Dev* fn = o->buf((size_t)mix * W * 4);
    Dev* sc = o->buf(3 * 4);
    Dev* bs = o->buf((size_t)mix * 4);
    Dev* pre = o->buf((size_t)kSweepRows * hc * 4);
    Dev* post = o->buf((size_t)kSweepRows * hc * 4);
    Dev* comb = o->buf((size_t)kSweepRows * hc * hc * 4);
    Dev* scr = o->buf(4u << 20);
    fill_bf16(h, rng, 1.0f);
    fill_f32(fn, rng, 0.01f);
    { const float v[3] = { 0.7f, 0.9f, 1.1f }; sc->put(v); }
    fill_f32(bs, rng, 0.5f);
    o->outs = { { pre, (size_t)hc * 4 }, { post, (size_t)hc * 4 }, { comb, (size_t)hc * hc * 4 } };
    o->launch = [=](int r0, int m, hipStream_t s) {
        RadTensor t[7] = { rows_of(h, RAD_BF16, W, r0, m), tensor(fn->p, RAD_F32, { mix, W }),
                           tensor(sc->p, RAD_F32, { 3 }), tensor(bs->p, RAD_F32, { mix }),
                           rows_of(pre, RAD_F32, hc, r0, m), rows_of(post, RAD_F32, hc, r0, m),
                           rows_of(comb, RAD_F32, hc * hc, r0, m) };
        RadParam p[6] = { pi("n", n), pi("hc", hc), pi("iterations", 20), pf("eps", 1e-20), pf("hc_eps", 1e-6),
                          ps("dtype", "bf16") };
        RadArgs a{};
        a.t = t; a.n_t = 7; a.p = p; a.n_p = 6;
        if (scratch) { a.scratch = scr->p; a.scratch_bytes = (int64_t)scr->n; }
        return f(&a, (RadStream)s);
    };
    return o;
}

/* gemm_nt's skinny row (gemm_nt_bf16f8 takes its split law): ds41's q_a (1280 x 5120) and kv
 * (512 x 5120) projections. */
std::unique_ptr<SweepOp> sweep_gemm(Launch f, int N, int K, std::mt19937_64& rng) {
    auto o = std::make_unique<SweepOp>();
    o->name = "gemm_bf16_nt_m64 N=" + std::to_string(N) + " K=" + std::to_string(K);
    Dev* A = o->buf((size_t)kSweepRows * K * 2);
    Dev* Wt = o->buf((size_t)N * K * 2);
    Dev* C = o->buf((size_t)kSweepRows * N * 2);
    fill_bf16(A, rng, 1.0f);
    fill_bf16(Wt, rng, 0.02f);
    o->outs = { { C, (size_t)N * 2 } };
    o->launch = [=](int r0, int m, hipStream_t s) {
        RadTensor t[3] = { rows_of(A, RAD_BF16, K, r0, m), tensor(Wt->p, RAD_BF16, { N, K }),
                           rows_of(C, RAD_BF16, N, r0, m) };
        RadParam p[4] = { pi("M", m), pi("N", N), pi("K", K), ps("dtype", "bf16") };
        RadArgs a{};
        a.t = t; a.n_t = 3; a.p = p; a.n_p = 4;
        return f(&a, (RadStream)s);
    };
    return o;
}

/* The lm_head over 1024 vocabulary columns (each rank holds 32320; a column's sum is its own). */
std::unique_ptr<SweepOp> sweep_logits(Launch f, std::mt19937_64& rng) {
    const int K = kEmbd, V = 1024;
    auto o = std::make_unique<SweepOp>();
    o->name = "logits_gemm n_embd=5120";
    Dev* x = o->buf((size_t)kSweepRows * K * 2);
    Dev* w = o->buf((size_t)V * K * 2);
    Dev* y = o->buf((size_t)kSweepRows * V * 4);
    fill_bf16(x, rng, 1.0f);
    fill_bf16(w, rng, 0.02f);
    o->outs = { { y, (size_t)V * 4 } };
    o->launch = [=](int r0, int m, hipStream_t s) {
        RadTensor t[3] = { rows_of(x, RAD_BF16, K, r0, m), tensor(w->p, RAD_BF16, { V, K }),
                           rows_of(y, RAD_F32, V, r0, m) };
        RadParam p[2] = { pi("n_vocab", V), pi("n_embd", K) };
        RadArgs a{};
        a.t = t; a.n_t = 3; a.p = p; a.n_p = 2;
        return f(&a, (RadStream)s);
    };
    return o;
}

/* Every output row of `o` after the launches `runs` (row ranges), outputs poisoned first. Empty on a
 * refused launch. */
std::vector<std::vector<uint8_t>> sweep_run(SweepOp& o, hipStream_t s, const std::vector<std::pair<int, int>>& runs) {
    for (const SweepOut& q : o.outs) HC(hipMemset(q.d->p, 0xA5, q.d->n));
    for (const auto& r : runs)
        if (o.launch(r.first, r.second, s) < 0) return {};
    HC(hipStreamSynchronize(s));
    std::vector<std::vector<uint8_t>> rows((size_t)kSweepRows);
    for (const SweepOut& q : o.outs) {
        std::vector<uint8_t> h(q.d->n);
        q.d->get(h.data());
        for (int m = 0; m < kSweepRows; ++m)
            rows[(size_t)m].insert(rows[(size_t)m].end(), h.begin() + (size_t)m * q.row_bytes,
                                   h.begin() + (size_t)(m + 1) * q.row_bytes);
    }
    return rows;
}

int sweep_check(const Lib& L, bool on, int trials) {
    long long reach = 32;   /* without the switch: the norms' 1024 threads hold to 32 rows */
    if (on) {
        long long r = 0;
        if (tall_row_table(L, &r) != 0) return 1;
        reach = r;
    }
    const std::vector<int> heights = { 1, 16, 17, 24, 25, 32, 33, 40, 48 };
    std::printf("sweep: RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL %s; heights up to %lld must give every row its "
                "bytes launched alone and in launches of 16; %d trials\n", on ? "1" : "unset", reach, trials);
    hipStream_t s;
    HC(hipStreamCreate(&s));
    int rc = 0;
    long long above = 0, above_rows = 0;   /* norm rows above the reach: differing, compared */
    for (int trial = 0; trial < trials; ++trial) {
        std::mt19937_64 rng(4100 + 17 * (uint64_t)trial);
        std::vector<std::unique_ptr<SweepOp>> ops;
        if (L.rmsnorm) for (int n : { 5120, 1280, 512, 128 }) ops.push_back(sweep_rmsnorm(L.rmsnorm, n, rng));
        if (L.collapse_norm) ops.push_back(sweep_collapse_norm(L.collapse_norm, rng));
        if (L.post_collapse_norm) ops.push_back(sweep_post_collapse_norm(L.post_collapse_norm, rng));
        if (L.norm_rope) {
            ops.push_back(sweep_norm_rope(L.norm_rope, 512, "fp8_e4m3", rng));
            ops.push_back(sweep_norm_rope(L.norm_rope, 128, "fp4_e2m1", rng));
        }
        if (L.mix) {
            ops.push_back(sweep_mix(L.mix, true, rng));
            if (trial == 0) ops.push_back(sweep_mix(L.mix, false, rng));
        }
        if (L.gemm64) {
            ops.push_back(sweep_gemm(L.gemm64, 1280, 5120, rng));
            ops.push_back(sweep_gemm(L.gemm64, 512, 5120, rng));
        }
        if (L.logits) ops.push_back(sweep_logits(L.logits, rng));
        if (trial == 0 && ops.size() < 13) {
            std::printf("FAIL: the plugin lacks some of r4d_rad_rmsnorm, _mhc_collapse_rmsnorm, "
                        "_mhc_post_collapse_rmsnorm, _rmsnorm_rope_roundtrip, _mhc_mix, _gemm_bf16_m64, _logits_gemm\n");
            rc = 1;
        }
        for (auto& op : ops) {
            std::vector<std::pair<int, int>> one, sixteen;
            for (int m = 0; m < kSweepRows; ++m) one.push_back({ m, 1 });
            for (int m = 0; m < kSweepRows; m += 16) sixteen.push_back({ m, std::min(16, kSweepRows - m) });
            const auto ref1 = sweep_run(*op, s, one), ref16 = sweep_run(*op, s, sixteen);
            if (ref1.empty() || ref16.empty()) {
                std::printf("FAIL: %s: a launch was refused\n", op->name.c_str());
                rc = 1;
                continue;
            }
            int d16 = 0;
            for (int m = 0; m < kSweepRows; ++m) d16 += ref1[(size_t)m] != ref16[(size_t)m];
            std::string line;
            bool bad = d16 != 0;
            for (int M : heights) {
                const auto got = sweep_run(*op, s, { { 0, M } });
                if (got.empty()) {
                    line += " " + std::to_string(M) + ":refused";
                    bad = true;
                    continue;
                }
                int d1 = 0, dc = 0;
                for (int m = 0; m < M; ++m) {
                    d1 += got[(size_t)m] != ref1[(size_t)m];
                    dc += got[(size_t)m] != ref16[(size_t)m];
                }
                line += " " + std::to_string(M) + ":" + (d1 || dc ? std::to_string(std::max(d1, dc)) + "rows" : "ok");
                if ((d1 || dc) && (M <= reach || !op->norm)) bad = true;
                if (M > reach && op->norm) {
                    above += std::max(d1, dc);
                    above_rows += M;
                }
            }
            if (trial == 0 || bad)
                std::printf("%s trial %d %-44s launches of 16 vs alone: %s;%s\n", bad ? "FAIL" : "    ", trial,
                            op->name.c_str(), d16 ? (std::to_string(d16) + " rows differ").c_str() : "ok",
                            line.c_str());
            rc |= bad ? 1 : 0;
        }
    }
    if (above_rows)
        std::printf("norm rows in passes above the reach (%lld): %lld of %lld differ from the same row alone\n",
                    reach, above, above_rows);
    std::printf(rc ? "FAIL: sweep\n"
                   : "PASS: sweep: every height up to the reach gives each row its bytes launched alone and in "
                     "launches of 16 (rows above it, if any differ, only in a norm)\n");
    return rc;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s LIBR4D.so check [trials] [table] | swiglu TABLE | time [LIB2.so] | ptime [T[:P]...] | tall [T...] | sweep [trials] | sweep0 [trials]\n", argv[0]); return 2; }
    /* `tall`: the switch before the plugin's row table and its sparse launchers first read it, and
     * the routed GEMM as the A/B's servers run it (no live tail); the other modes as the loop's
     * measured runs ran it. */
    if (!std::strcmp(argv[2], "tall") || !std::strcmp(argv[2], "sweep"))
        setenv("RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL", "1", 1);
    else if (!std::strcmp(argv[2], "sweep0")) unsetenv("RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL");
    else setenv("RADIANCE_EXPERIMENTAL_DS41_LIVE_TAIL", "1", 1);
    const Lib lib(argv[1]);
    if (!std::strcmp(argv[2], "check")) return check(lib, argc > 3 ? std::atoi(argv[3]) : 24, argc > 4 ? argv[4] : "silu_table.bin");
    if (!std::strcmp(argv[2], "swiglu")) return swiglu_proof(lib, argc > 3 ? argv[3] : "silu_table.bin");
    if (!std::strcmp(argv[2], "ptime")) {
        /* ptime T[:P] ...; PTIME_LIB2=<libr4d.so> also runs and compares a second plugin */
        const char* l2 = std::getenv("PTIME_LIB2");
        std::unique_ptr<Lib> b;
        if (l2 && *l2) b.reset(new Lib(l2));
        int rc = 0;
        for (int i = 3; i < argc; ++i) {
            int T = 16, P = 4 * kNe;
            if (std::sscanf(argv[i], "%d:%d", &T, &P) < 1) continue;
            rc |= ptime(lib, b.get(), T, P);
        }
        if (argc <= 3) rc = ptime(lib, b.get(), 16, 4 * kNe);
        return rc;
    }
    if (!std::strcmp(argv[2], "time")) {
        if (argc > 3) { const Lib b(argv[3]); return timing(lib, &b); }
        return timing(lib, nullptr);
    }
    if (!std::strcmp(argv[2], "tall")) {
        std::vector<int> Ts;
        for (int i = 3; i < argc; ++i) Ts.push_back(std::atoi(argv[i]));
        if (Ts.empty()) Ts = { 17, 24, 33, 40, 47, 48, 64, 96 };
        return tall_check(lib, Ts);
    }
    if (!std::strcmp(argv[2], "sweep") || !std::strcmp(argv[2], "sweep0"))
        return sweep_check(lib, argv[2][5] != '0', argc > 3 ? std::max(1, std::atoi(argv[3])) : 4);
    return 2;
}
