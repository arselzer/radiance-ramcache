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
 *   ds41_mxfp4_device_check LIBR4D.so draft [trials]   the DSpark draft bank's grouped GEMMs (32
 *                                                      experts a rank, top-3) in every mode of
 *                                                      RADIANCE_EXPERIMENTAL_DS41_DRAFT_GEMV (old
 *                                                      path, live tail, one weight pass, both) on
 *                                                      EP4 routings, experts in pinned host memory
 *                                                      and in VRAM, against the host experts bit
 *                                                      for bit, with the work counted; the served
 *                                                      entry under the switch; then each mode's
 *                                                      time a call at a 5-row query pass, experts
 *                                                      in pinned host memory as the bank's are
 *
 * Needs a free card (HIP device 0 unless HIP_VISIBLE_DEVICES says otherwise). Uses the
 * live-tail ranges as the measured runs do (RADIANCE_EXPERIMENTAL_DS41_LIVE_TAIL=1). */
#include <hip/hip_runtime.h>
#include <dlfcn.h>

#include <algorithm>
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

/* The draft switch's checker exports (r4d_ds41_mxfp4.hip): a forced mode with the probe's counts,
 * and the switch's call counts. */
typedef int (*DraftProbe)(const RadArgs*, RadStream, int gemv, int mode, unsigned long long* probe);
typedef void (*DraftCounts)(unsigned long long* out);

struct Lib {
    Launch moe = nullptr, swiglu = nullptr;
    DraftProbe draft = nullptr;
    DraftCounts counts = nullptr;
    explicit Lib(const char* path) {
        void* h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!h) { std::fprintf(stderr, "dlopen %s: %s\n", path, dlerror()); std::exit(2); }
        moe = (Launch)dlsym(h, "r4d_rad_moe_gemm_mxfp4_gemv");
        swiglu = (Launch)dlsym(h, "r4d_rad_swiglu_clamped");
        draft = (DraftProbe)dlsym(h, "r4d_rad_moe_gemm_mxfp4_draft_probe");
        counts = (DraftCounts)dlsym(h, "r4d_ds41_draft_gemv_counts");
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

/* ---- the DSpark draft bank's grouped GEMMs (RADIANCE_EXPERIMENTAL_DS41_DRAFT_GEMV) ---------------
 *
 * Its 128 experts at EP4 are 32 a rank, top-3; a query pass is 5 rows a sequence. The bank's experts
 * sit in pinned host memory the card reads in place (--draft-bank-expert-placement host), which is
 * where the switch's one weight pass pays: the old path reads a gate_up expert once per row (one
 * fits the LDS whole at K 5120) and the rank's last expert once more per slot routed elsewhere. */
constexpr int kDl = 32, kDt = 3;

/* Pinned host memory as the engine's host pool holds it (fine-grained, mapped, read in place by
 * the card), or VRAM. */
struct Buf {
    void* p = nullptr;
    void* dev = nullptr;
    size_t n = 0;
    bool host = false;
    Buf(size_t bytes, bool on_host) : n(bytes), host(on_host) {
        if (host) {
            HC(hipHostMalloc(&p, bytes, hipHostMallocMapped | hipHostMallocCoherent));
            HC(hipHostGetDevicePointer(&dev, p, 0));
        } else {
            HC(hipMalloc(&p, bytes));
            dev = p;
        }
    }
    ~Buf() { if (host) (void)hipHostFree(p); else (void)hipFree(p); }
    Buf(const Buf&) = delete;
    Buf& operator=(const Buf&) = delete;
    void put(const void* h) { if (host) std::memcpy(p, h, n); else HC(hipMemcpy(p, h, n, hipMemcpyHostToDevice)); }
};

/* moe_scatter on rank `rank` at EP4: its slots grouped by local expert in slot order, the rest -1. */
struct DRoute { int m = 0, T = 0; std::vector<int32_t> sorted, off; };
DRoute dscatter(const std::vector<int>& ids, int m, int rank) {
    DRoute R;
    R.m = m; R.T = m * kDt;
    R.off.assign(kDl + 1, 0);
    for (int e = 0; e < kDl; ++e) {
        R.off[(size_t)e] = (int)R.sorted.size();
        for (int i = 0; i < R.T; ++i)
            if (ids[(size_t)i] == rank * kDl + e) R.sorted.push_back(i);
    }
    R.off[kDl] = (int)R.sorted.size();
    R.sorted.resize((size_t)R.T, -1);
    return R;
}

/* Each token's top-3, distinct: no local slot; the last expert with rows of its own; the last
 * expert idle; one expert in every row; a few experts sharing the rows; spread. */
constexpr int kDPatterns = 6;
const char* const kDPatternName[kDPatterns] = { "no local slot", "last expert with its own rows", "last expert idle",
                                                "one expert in every row", "a few experts share the rows", "spread" };
std::vector<int> droute(int p, int m, int rank, std::mt19937_64& rng) {
    std::vector<int> ids((size_t)m * kDt, -1);
    const int base = rank * kDl, other = ((rank + 1) % 4) * kDl;
    static const int pool[] = { 3, 4, 31, 40, 70, 100 };
    for (int t = 0; t < m; ++t)
        for (int s = 0; s < kDt; ++s) {
            int e = -1;
            for (int tries = 0; tries < 1000; ++tries) {
                switch (p) {
                    case 0: e = (base + kDl + (int)(rng() % 96)) % 128; break;
                    case 1: e = s == 0 && t % 2 == 0 ? base + kDl - 1 : (int)(rng() % 128); break;
                    case 2: e = (int)(rng() % 128); if (e == base + kDl - 1) e = base; break;
                    case 3: e = s == 0 ? base + 7 : other + (int)(rng() % kDl); break;
                    case 4: e = (pool[rng() % 6] + base) % 128; break;
                    default: e = (int)(rng() % 128); break;
                }
                bool dup = false;
                for (int j = 0; j < s; ++j) dup |= ids[(size_t)t * kDt + j] == e;
                if (!dup) break;
            }
            ids[(size_t)t * kDt + s] = e;
        }
    return ids;
}

/* One grouped GEMM over a table of `kDl` expert pointers: the served entry (the switch decides) or
 * the checker's forced `mode` with the probe. */
int run_draft(const Lib& L, hipStream_t s, bool served, int mode, unsigned long long* probe, void* x, int AR,
              void* table, int N, int K, void* sorted, void* off, void* y, int T, const char* order) {
    RadTensor t[5] = { tensor(x, RAD_BF16, { AR, K }), tensor(table, RAD_U8, { kDl, N, K / 32 * 17 }),
                       tensor(sorted, RAD_I32, { T }), tensor(off, RAD_I32, { kDl + 1 }), tensor(y, RAD_BF16, { T, N }) };
    t[1].stride[0] = 0;   /* a table of expert pointers */
    RadParam p[6] = { ps("type", "mxfp4"), pi("N", N), pi("K", K), pi("n_expert", kDl), pi("top_k", kDt),
                      ps("a_order", order) };
    RadArgs a{};
    a.t = t; a.n_t = 5; a.p = p; a.n_p = 6;
    return served ? L.moe(&a, (RadStream)s) : L.draft(&a, (RadStream)s, 1, mode, probe);
}

/* The experts mx_range (old) or the live tail give rows, as row counts. */
std::vector<int> druns(const DRoute& R, bool tail) {
    std::vector<int> runs;
    for (int e = 0; e < kDl; ++e) {
        int a = e == 0 ? 0 : R.off[(size_t)e], b = tail || e < kDl - 1 ? R.off[(size_t)e + 1] : R.T;
        a = std::min(std::max(a, 0), R.T);
        b = std::min(std::max(b, a), R.T);
        if (b > a) runs.push_back(b - a);
    }
    return runs;
}

int draft_check(const Lib& L, int trials) {
    if (!L.draft || !L.counts) {
        std::fprintf(stderr, "this libr4d.so has no RADIANCE_EXPERIMENTAL_DS41_DRAFT_GEMV (r4d_rad_moe_gemm_mxfp4_draft_probe)\n");
        return 2;
    }
    hipStream_t s;
    HC(hipStreamCreate(&s));
    std::mt19937_64 rng(4141);
    const size_t gu_b = (size_t)2 * kFf * (kEmbd / 32) * 17, dn_b = (size_t)kEmbd * (kFf / 32) * 17;
    /* Distinct experts, each in both placements; a trial with more live experts than this shares. */
    constexpr int kPool = 24;
    std::vector<std::vector<uint8_t>> hgu, hdn;
    std::vector<std::unique_ptr<Buf>> pgu[2], pdn[2];   /* [0] pinned host memory, [1] VRAM */
    for (int i = 0; i < kPool; ++i) {
        hgu.push_back(make_blocks(2 * kFf, kEmbd, rng, i % 4 == 3));
        hdn.push_back(make_blocks(kEmbd, kFf, rng, i % 4 == 3));
        for (int h = 0; h < 2; ++h) {
            pgu[h].emplace_back(new Buf(gu_b, h == 0));
            pgu[h].back()->put(hgu.back().data());
            pdn[h].emplace_back(new Buf(dn_b, h == 0));
            pdn[h].back()->put(hdn.back().data());
        }
    }
    constexpr int kMaxM = 16, kMaxT = kMaxM * kDt;
    Dev tab(kDl * sizeof(void*)), d_sorted(kMaxT * 4), d_off((kDl + 1) * 4), d_x((size_t)kMaxM * kEmbd * 2),
        d_a((size_t)kMaxT * kFf * 2), d_ygu((size_t)kMaxT * 2 * kFf * 2), d_ydn((size_t)kMaxT * kEmbd * 2),
        d_probe(6 * sizeof(unsigned long long));
    const int sizes[] = { 1, 2, 5, 6, 8, 16 };
    size_t bad[2][4] = {}, bad_served = 0, bad_work = 0, outputs = 0;
    unsigned long long passes[2][4] = {};
    int served_calls = 0;
    for (int trial = 0; trial < trials; ++trial) {
        const int m = sizes[trial % 6], pat = (trial / 6) % kDPatterns, rank = trial % 4;
        const DRoute R = dscatter(droute(pat, m, rank, rng), m, rank);
        /* each local expert with rows, and the last expert always, its own matrix */
        std::vector<int> mat(kDl, 0);
        int next = 1 + trial;
        for (int e = 0; e < kDl; ++e)
            if (R.off[(size_t)e + 1] > R.off[(size_t)e] || e == kDl - 1) mat[(size_t)e] = next++ % kPool;
        std::vector<uint16_t> x((size_t)m * kEmbd), act((size_t)R.T * kFf);
        for (int r = 0; r < m; ++r) {
            const auto row = make_x(kEmbd, rng, trial % 4 == 3);
            std::copy(row.begin(), row.end(), x.begin() + (size_t)r * kEmbd);
        }
        for (auto& v : act) v = f32_to_bf16(0.1f * (float)((int)(rng() % 200) - 100) / 100.f);
        HC(hipMemcpy(d_x.p, x.data(), x.size() * 2, hipMemcpyHostToDevice));
        HC(hipMemcpy(d_a.p, act.data(), act.size() * 2, hipMemcpyHostToDevice));
        HC(hipMemcpy(d_sorted.p, R.sorted.data(), R.sorted.size() * 4, hipMemcpyHostToDevice));
        HC(hipMemcpy(d_off.p, R.off.data(), R.off.size() * 4, hipMemcpyHostToDevice));
        for (int q = 0; q < 2; ++q) {
            const bool gu = q == 0;
            const int N = gu ? 2 * kFf : kEmbd, K = gu ? kEmbd : kFf;
            Dev& dy = gu ? d_ygu : d_ydn;
            /* the host experts' GEMV for every live slot, zeros for a dropped one */
            std::vector<uint16_t> want((size_t)R.T * N, 0);
            for (int e = 0; e < kDl; ++e)
                for (int i = R.off[(size_t)e]; i < R.off[(size_t)e + 1]; ++i) {
                    if (R.sorted[(size_t)i] < 0) continue;
                    Mx W; W.N = N; W.K = K; W.blocks = (gu ? hgu : hdn)[(size_t)mat[(size_t)e]].data();
                    const uint16_t* in = gu ? x.data() + (size_t)(R.sorted[(size_t)i] / kDt) * kEmbd
                                            : act.data() + (size_t)i * kFf;
                    gemv_avx2(W, in, want.data() + (size_t)i * N, 0, N);
                }
            const int XM = std::min(m, 8), XLM = std::min(9216 / (K + K / 8), XM);
            std::vector<uint16_t> got((size_t)R.T * N);
            for (int h = 0; h < 2; ++h) {
                std::vector<void*> t(kDl);
                for (int e = 0; e < kDl; ++e) t[(size_t)e] = (gu ? pgu : pdn)[h][(size_t)mat[(size_t)e]]->dev;
                tab.put(t.data());
                for (int mode = 0; mode < 5; ++mode) {
                    /* modes 0..3 forced; 4 the served entry (pinned host memory only) */
                    const bool served = mode == 4;
                    if (served && h) continue;
                    unsigned long long c0[5] = {}, c1[5] = {}, pr[6] = {};
                    if (served) L.counts(c0);
                    HC(hipMemset(dy.p, 0x77, dy.n));
                    HC(hipMemset(d_probe.p, 0, d_probe.n));
                    if (run_draft(L, s, served, served ? 0 : mode, served ? nullptr : (unsigned long long*)d_probe.p,
                                  gu ? d_x.p : d_a.p, gu ? m : R.T, tab.p, N, K, d_sorted.p, d_off.p, dy.p, R.T,
                                  gu ? "token" : "sorted") < 0) {
                        std::fprintf(stderr, "the draft GEMV refused trial %d mode %d\n", trial, mode);
                        return 2;
                    }
                    HC(hipStreamSynchronize(s));
                    HC(hipMemcpy(got.data(), dy.p, got.size() * 2, hipMemcpyDeviceToHost));
                    const size_t d = count_diff(got.data(), want.data(), got.size());
                    outputs += got.size();
                    if (served) {
                        L.counts(c1);
                        bad_served += d;
                        ++served_calls;
                        if (c1[gu ? 0 : 1] != c0[gu ? 0 : 1] + 1) {
                            ++bad_work;
                            std::printf("  trial %d: the served %s call was not the switch's\n", trial, gu ? "gate_up" : "down");
                        }
                        continue;
                    }
                    bad[h][mode] += d;
                    if (d && bad[h][mode] == d)
                        std::printf("  %s, %s m %d rank %d (%s), mode %d: %zu values differ\n", h ? "VRAM" : "pinned host",
                                    gu ? "gate_up" : "down", m, rank, kDPatternName[pat], mode, d);
                    HC(hipMemcpy(pr, d_probe.p, sizeof pr, hipMemcpyDeviceToHost));
                    const bool tail = mode & 1, one = mode & 2;
                    const std::vector<int> runs = druns(R, tail);
                    const int step = !one && XLM >= 1 && XLM < XM ? XLM : XM;
                    unsigned long long owned = 0, np = 0;
                    for (int r : runs) { owned += (unsigned long long)r; np += (unsigned long long)((r + step - 1) / step); }
                    const unsigned long long zr = tail ? (unsigned long long)(R.T - R.off[kDl]) : 0;
                    if (pr[0] != owned || pr[2] != zr || pr[3] != zr * (unsigned long long)N || pr[5] != np) {
                        ++bad_work;
                        std::printf("  %s m %d rank %d (%s), mode %d: work owned %llu zero rows %llu passes %llu, want %llu %llu %llu\n",
                                    gu ? "gate_up" : "down", m, rank, kDPatternName[pat], mode, pr[0], pr[2], pr[5], owned, zr, np);
                    }
                    if (!h) passes[q][mode] += pr[5];
                }
            }
        }
    }
    std::printf("draft GEMV: %d trials, %zu outputs compared; differing values (old, tail, one pass, both): pinned host "
                "%zu %zu %zu %zu, VRAM %zu %zu %zu %zu; served entry %zu in %d calls; %zu work counts wrong\n",
                trials, outputs, bad[0][0], bad[0][1], bad[0][2], bad[0][3], bad[1][0], bad[1][1], bad[1][2], bad[1][3],
                bad_served, served_calls, bad_work);
    std::printf("draft GEMV weight passes over the trials (old, tail, one pass, both): gate_up %llu %llu %llu %llu, "
                "down %llu %llu %llu %llu\n", passes[0][0], passes[0][1], passes[0][2], passes[0][3], passes[1][0],
                passes[1][1], passes[1][2], passes[1][3]);
    size_t all_bad = bad_served + bad_work;
    for (auto& b : bad) for (size_t v : b) all_bad += v;
    std::printf(all_bad ? "FAIL\n" : "PASS: every draft GEMV mode is bit-identical to the host experts and does the work it should\n");

    /* Time a call, a 5-row query pass with `live` local experts (the rest of the 15 slots dropped),
     * the experts in pinned host memory as the bank's are and rotated over the pool, so no call
     * finds the last one's weights in the 64 MB cache. */
    std::printf("draft GEMV time a call, 5 query rows (15 slots), experts in pinned host memory; old / tail / one pass / "
                "both, and the live experts' weights over the best\n");
    static const int rows_of[6][5] = { {}, { 4 }, { 2, 2 }, { 2, 1, 1 }, { 1, 1, 1, 1 }, { 1, 1, 1, 1, 1 } };
    hipEvent_t e0, e1;
    HC(hipEventCreate(&e0)); HC(hipEventCreate(&e1));
    for (int live = 1; live <= 5; ++live) {
        /* Rank 0's experts 2, 9, 15, 22 and 27 take rows of distinct tokens (expert j tokens j,
         * j + 1, ..., each in its next free slot); every other slot goes to one of rank 1's, each
         * once, so a token's top-3 stay distinct. */
        static const int at[5] = { 2, 9, 15, 22, 27 };
        std::vector<int> ids(15, -1);
        for (int t = 0; t < 5; ++t)
            for (int sl = 0; sl < 3; ++sl) ids[(size_t)t * 3 + sl] = 32 + t * 3 + sl;
        int used[5] = {};
        for (int j = 0; j < live; ++j)
            for (int r = 0; r < rows_of[live][j]; ++r) {
                const int t = (j + r) % 5;
                ids[(size_t)t * 3 + (size_t)used[t]++] = at[j];
            }
        const DRoute R = dscatter(ids, 5, 0);
        HC(hipMemcpy(d_sorted.p, R.sorted.data(), R.sorted.size() * 4, hipMemcpyHostToDevice));
        HC(hipMemcpy(d_off.p, R.off.data(), R.off.size() * 4, hipMemcpyHostToDevice));
        for (int q = 0; q < 2; ++q) {
            const bool gu = q == 0;
            const int N = gu ? 2 * kFf : kEmbd, K = gu ? kEmbd : kFf;
            Dev& dy = gu ? d_ygu : d_ydn;
            double med[4] = {};
            for (int mode = 0; mode < 4; ++mode) {
                std::vector<float> ts;
                for (int it = 0; it < 35; ++it) {
                    std::vector<void*> t(kDl);
                    for (int e = 0; e < kDl; ++e) t[(size_t)e] = (gu ? pgu : pdn)[0][(size_t)((it * 6 + e) % kPool)]->dev;
                    tab.put(t.data());
                    HC(hipEventRecord(e0, s));
                    if (run_draft(L, s, false, mode, nullptr, gu ? d_x.p : d_a.p, gu ? 5 : 15, tab.p, N, K, d_sorted.p,
                                  d_off.p, dy.p, 15, gu ? "token" : "sorted") < 0)
                        return 3;
                    HC(hipEventRecord(e1, s));
                    HC(hipEventSynchronize(e1));
                    float ms = 0;
                    HC(hipEventElapsedTime(&ms, e0, e1));
                    if (it >= 5) ts.push_back(ms);
                }
                std::sort(ts.begin(), ts.end());
                med[mode] = ts[ts.size() / 2];
            }
            const double bytes = (double)live * N * (K / 32) * 17;
            const double best = std::min(std::min(med[0], med[1]), std::min(med[2], med[3]));
            std::printf("  %s live %d (%d rows): %8.1f %8.1f %8.1f %8.1f us   %.1f GB/s\n", gu ? "gate_up" : "down   ", live,
                        R.off[kDl], 1e3 * med[0], 1e3 * med[1], 1e3 * med[2], 1e3 * med[3], bytes / (best * 1e-3) / 1e9);
        }
    }
    return all_bad ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s LIBR4D.so check [trials] [table] | swiglu TABLE | time [LIB2.so] | ptime [T[:P]...] | draft [trials]\n", argv[0]); return 2; }
    setenv("RADIANCE_EXPERIMENTAL_DS41_LIVE_TAIL", "1", 1);
    /* The draft check's served entry runs under the switch, read once at the first draft-shaped call. */
    if (!std::strcmp(argv[2], "draft")) setenv("RADIANCE_EXPERIMENTAL_DS41_DRAFT_GEMV", "1", 1);
    const Lib lib(argv[1]);
    if (!std::strcmp(argv[2], "draft")) return draft_check(lib, argc > 3 ? std::atoi(argv[3]) : 36);
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
    return 2;
}
