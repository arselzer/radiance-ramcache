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
 *   ds41_mxfp4_device_check LIBR4D.so tall [T...]      RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL (set
 *                                                      here): the twin row moe_gemm_ggml_mxfp4_gemv_tall
 *                                                      in the plugin's table; its gate_up and down at
 *                                                      T tokens (17..96) against the gemv over the same
 *                                                      tokens in passes of 16 and against the host
 *                                                      kernels; attn_sparse_sink and index_score at
 *                                                      17..100 rows against the same rows one at a
 *                                                      time and in launches of 16. Every value equal
 *                                                      or FAIL.
 *   ds41_mxfp4_device_check LIBR4D.so lcp [C] [trials] RADIANCE_EXPERIMENTAL_DS41_LCP=C (default 1024,
 *                                                      set here; EXACT_TALL unset): `tall` at chunk
 *                                                      heights up to C (the twin's reach must cover
 *                                                      C), the sparse ops also as the suffix launches
 *                                                      a per-row consumer cut issues, then `sweep` at
 *                                                      heights up to 2048 with gemm_nt by the row the
 *                                                      table resolves at each height (m16 / m64 /
 *                                                      tiled), gemm_nt_f32 over the router's 384
 *                                                      columns and its 96-column slices, and
 *                                                      gemm_nt_bf16f8 at N 1280 and 4096 (its one
 *                                                      skinny row at every height).
 *   ds41_mxfp4_device_check LIBR4D.so bal [C] [ROUTES] RADIANCE_EXPERIMENTAL_DS41_MOE_BALANCE through the
 *                                                      checker export r4d_rad_moe_gemm_mxfp4_gemv_bal
 *                                                      (LCP=C set here, 1024 by default): the gemv row at
 *                                                      1..16 tokens and the twin at 17..C, ds41's gate_up
 *                                                      and down over 96 VRAM experts on an EP4 rank's
 *                                                      served slots (the other ranks' as -1 past the
 *                                                      local ones), skewed and uniform routings (and the
 *                                                      route file's layers, as ds41_p0_bench reads it,
 *                                                      when given): every output value of the balanced
 *                                                      launch equal to the plain one's, and both timed
 *                                                      (hipEvent, median of 3).
 *   ds41_mxfp4_device_check LIBR4D.so p1 [MMAX] [ROUTES] prefill Phase 1 (RADIANCE_EXPERIMENTAL_DS41_P1=1 and
 *                                                      LCP=2048, set here; tools/ds41-host-experts/ds41_p1.h):
 *                                                      the row moe_gemm_ggml_mxfp4_gemv_tall_p1 in the table,
 *                                                      reaching MMAX (4096), and its launcher at 17..MMAX
 *                                                      tokens over tables mixing block_mxfp4, packed and
 *                                                      skipped entries against the gemv in passes of 16 and
 *                                                      the host kernels, skipped rows untouched, the tail
 *                                                      zeros, other work splits and the stager's full/span
 *                                                      SwiGLU waves equal to one launch; pitched slices
 *                                                      preserve guards and NaN payloads; then "p1 time:" lines. Last line
 *                                                      "p1: PASS ..." or "p1: FAIL ...".
 *   ds41_mxfp4_device_check LIBR4D.so lsparse [C] [W:r...] the sparse ops at long context (LCP=C, 2048 by
 *                                                      default, set here): index_score, index_candidates,
 *                                                      index_topk (with and without keep) and attn_sparse_sink
 *                                                      on a 256-row chunk at the end of a context of index
 *                                                      width W at ratio r (32769:1 65537:2 131073:1 262145:2
 *                                                      524289:1 by default: --max-model-len 32768, 131072 and
 *                                                      524288), the chunk against its rows alone, 16 at a
 *                                                      time and as suffixes, top-k also against the host.
 *                                                      Last line "lsparse: PASS ..." or "lsparse: FAIL ...".
 *   ds41_mxfp4_device_check LIBR4D.so sweep [trials]   the height sweep under EXACT_TALL, or under
 *                                                      RADIANCE_EXPERIMENTAL_DS41_LCP when that is in
 *                                                      the environment (its heights); sweep0 without
 *                                                      either switch.
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
#include "ds41_p1.h"
#include "r4d_bf16f8_layout.h"
#include "r4d_ds41_tall.h"

using namespace ds41hx;

/* libr4d restates prefill Phase 1's constants (r4d_ds41_tall.h); they must agree with the shared header. */
constexpr bool p1_same_str(const char* a, const char* b) { return *a == *b && (!*a || p1_same_str(a + 1, b + 1)); }
static_assert(p1_same_str(ds41p1::kEnv, kR4dDs41P1Env), "the switch's name");
static_assert(p1_same_str(ds41p1::kTwinRow, kR4dDs41P1Row), "the twin row's name");
static_assert(ds41p1::kTwinMaxRows == kR4dDs41P1MaxRows, "the twin's bound");
static_assert(ds41p1::kMarkPacked == kR4dDs41MarkPacked && ds41p1::kMarkSkip == kR4dDs41MarkSkip, "the marks");

#define HC(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e_)); std::exit(2); } } while (0)

namespace {

constexpr int kEmbd = 5120, kFf = 2304, kNe = 96, kTop = 6;
/* `lcp`, or `sweep` with RADIANCE_EXPERIMENTAL_DS41_LCP in the environment: the chunk C (0 otherwise). */
int g_lcp = 0;
/* libr4d's R4D_DT_BF16F8 (r4d_args.h): gemm_nt_bf16f8's stored form. */
constexpr uint32_t R4D_BF16F8_DTYPE = RAD_DT_PLUGIN_BASE + 10;
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
    /* For `tall`: the twin's launch, the tiled row's (to show what the twin replaces), the sparse
     * ops, and the plugin's row table. */
    Launch moe_tall = nullptr, moe_tiled = nullptr, attn = nullptr, iscore = nullptr;
    /* For `lsparse`: the indexer's selection ops and index_topk's scratch hook. */
    Launch itopk = nullptr, icand = nullptr;
    int64_t (*itopk_scratch)(const RadArgs*) = nullptr;
    /* For `sweep`: the other ops whose launcher reads the pass height. */
    Launch rmsnorm = nullptr, collapse_norm = nullptr, post_collapse_norm = nullptr, norm_rope = nullptr,
           mix = nullptr, gemm64 = nullptr, logits = nullptr;
    /* For `lcp`: gemm_nt's other rows and the f32 projection (the router and the compressor), and
     * gemm_nt_bf16f8 (the FP8-stored dense weights, RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8). */
    Launch gemm16 = nullptr, gemm_tiled = nullptr, gemm_f32 = nullptr, gemm_f8 = nullptr;
    /* For `bal`: the checker export with RADIANCE_EXPERIMENTAL_DS41_MOE_BALANCE forced per call. */
    int (*moe_bal)(const RadArgs*, RadStream, int tall, int bal) = nullptr;
    /* For `p1`: prefill Phase 1's twin and its checker export with the work split forced. */
    Launch moe_p1 = nullptr;
    int (*moe_p1_opts)(const RadArgs*, RadStream, int, int, int, int) = nullptr;
    int (*count)(void) = nullptr;
    const RadKernelInfo* (*at)(int) = nullptr;
    explicit Lib(const char* path) {
        void* h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!h) { std::fprintf(stderr, "dlopen %s: %s\n", path, dlerror()); std::exit(2); }
        moe = (Launch)dlsym(h, "r4d_rad_moe_gemm_mxfp4_gemv");
        swiglu = (Launch)dlsym(h, "r4d_rad_swiglu_clamped");
        draft = (DraftProbe)dlsym(h, "r4d_rad_moe_gemm_mxfp4_draft_probe");
        counts = (DraftCounts)dlsym(h, "r4d_ds41_draft_gemv_counts");
        moe_tall = (Launch)dlsym(h, "r4d_rad_moe_gemm_mxfp4_gemv_tall");
        moe_tiled = (Launch)dlsym(h, "r4d_rad_moe_gemm_mxfp4_tiled");
        attn = (Launch)dlsym(h, "r4d_rad_ds41_attn_sparse_sink");
        iscore = (Launch)dlsym(h, "r4d_rad_ds41_index_score");
        itopk = (Launch)dlsym(h, "r4d_rad_ds41_index_topk");
        icand = (Launch)dlsym(h, "r4d_rad_ds41_index_candidates");
        itopk_scratch = (int64_t (*)(const RadArgs*))dlsym(h, "r4d_rad_ds41_index_topk_scratch");
        rmsnorm = (Launch)dlsym(h, "r4d_rad_rmsnorm");
        collapse_norm = (Launch)dlsym(h, "r4d_rad_mhc_collapse_rmsnorm");
        post_collapse_norm = (Launch)dlsym(h, "r4d_rad_mhc_post_collapse_rmsnorm");
        norm_rope = (Launch)dlsym(h, "r4d_rad_rmsnorm_rope_roundtrip");
        mix = (Launch)dlsym(h, "r4d_rad_mhc_mix");
        gemm64 = (Launch)dlsym(h, "r4d_rad_gemm_bf16_m64");
        logits = (Launch)dlsym(h, "r4d_rad_logits_gemm");
        gemm16 = (Launch)dlsym(h, "r4d_rad_gemm_bf16_m16");
        gemm_tiled = (Launch)dlsym(h, "r4d_rad_gemm_bf16_tiled");
        gemm_f32 = (Launch)dlsym(h, "r4d_rad_gemm_nt_f32");
        gemm_f8 = (Launch)dlsym(h, "r4d_rad_gemm_bf16f8_m64");
        moe_bal = (int (*)(const RadArgs*, RadStream, int, int))dlsym(h, "r4d_rad_moe_gemm_mxfp4_gemv_bal");
        moe_p1 = (Launch)dlsym(h, "r4d_rad_moe_gemm_mxfp4_gemv_tall_p1");
        moe_p1_opts = (int (*)(const RadArgs*, RadStream, int, int, int, int))dlsym(h, "r4d_rad_moe_gemm_mxfp4_gemv_tall_p1_opts");
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
                    tall->relayout == gemv->relayout && (!g_lcp || *reach >= g_lcp);
    std::printf("%s: row table: %s, priority %d (the gemv's %d), %d constraints, M up to %lld%s\n",
                ok ? "PASS" : "FAIL", tall->name, tall->priority, gemv->priority, tall->n_constraints, *reach,
                g_lcp ? (" (LCP: at least " + std::to_string(g_lcp) + ")").c_str() : "");
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
        if (L.moe_tiled && T > 16 && T <= 1024) {
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
        /* Keep the old host oracle on every local row through 1024 tokens. New larger heights
         * already compare every output with gemv passes of 16 above; independently sample the
         * first/middle/last host rows rather than repeat a multi-billion-op CPU sweep in a window. */
        for (int i = 0; i < local; ++i) {
            if (T > 1024 && i != 0 && i != local / 2 && i != local - 1) continue;
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
    /* The suffixes a per-row consumer cut launches: rows [r0, M) for r0 on the 16-row grid. */
    for (int r0 : { 16, M / 2 / 16 * 16, (M - 1) / 16 * 16 }) {
        if (r0 <= 0 || r0 >= M) continue;
        std::vector<uint16_t> o2;
        std::vector<float> s2;
        if (sparse_launch(L, s, f, r0, M, &o2, &s2) < 0) { std::printf("FAIL: a sparse launch refused\n"); return (size_t)1 << 40; }
        const size_t da = count_diff(&out[(size_t)r0 * f.H * f.D], o2.data(), o2.size());
        size_t ds = 0;
        for (size_t i = 0; i < s2.size(); ++i) ds += std::memcmp(&sc[(size_t)r0 * f.C + i], &s2[i], 4) != 0;
        if ((da || ds) && differ < 4)
            std::printf("  M=%d H=%d suffix %d..%d: attention %zu, index score %zu differing values\n", M, H, r0,
                        M - 1, da, ds);
        differ += da + ds;
    }
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
    std::vector<int> sp_heights = { 17, 24, 33, 40, 47, 48, 64, 100 };
    if (g_lcp)
        for (int M : { 49, 65, 128, 171, 256, 512, 1000, 1024 })
            if (M <= g_lcp) sp_heights.push_back(M);
    if (g_lcp > 2048) {
        for (int M : { 2048, 3072, 4096 }) if (M <= g_lcp) sp_heights.push_back(M);
        if (std::find(sp_heights.begin(), sp_heights.end(), g_lcp) == sp_heights.end())
            sp_heights.push_back(g_lcp);
    }
    for (int H : { 16, 32 })
        for (int M : sp_heights) {
            size_t rows = 0;
            const size_t d = tall_sparse(L, s, M, H, &rows);
            std::printf("attn_sparse_sink (%d heads) + index_score M=%3d: %zu rows against single rows and launches "
                        "of 16: %zu differing values\n", H, M, rows, d);
            sp_bad += d;
            sp_rows += rows;
        }
    std::printf("%s: sparse attention and index score at 17..%d rows equal their rows launched alone, 16 at a "
                "time and as suffixes (%zu rows)\n", sp_bad ? "FAIL" : "PASS", sp_heights.back(), sp_rows);
    rc |= sp_bad ? 1 : 0;
    std::printf(rc ? "FAIL\n" : "PASS: tall passes give 16-token passes' bits on all three ops\n");
    return rc;
}

/* ================================================================== lsparse (the sparse ops at long context)
 *
 * The indexer and the sparse attention at the index widths of a long --max-model-len: an index_score row
 * has max_ctx / ratio + 1 columns (deepseek41.cpp: ratio 2 at layers 2-19, ratio 1 from layer 20), so
 * 65,537 / 131,073 at --max-model-len 131072 and 262,145 / 524,289 at 524288, where `lcp` stops at a few
 * thousand. One sequence's chunk of M rows ends at the context's last position (every row's reach near
 * the width); with RADIANCE_EXPERIMENTAL_DS41_LCP set (the servers' Lc parts: the sparse ops' scalar forms
 * at every height) each op's rows launched as one chunk are held to the same rows launched alone (the
 * decode form: index_topk's split kernels over the scratch its hook asks for), 16 at a time and as the
 * suffixes a consumer cut issues, every op on the same inputs:
 *
 *   index_score        q, w and the index keys (one entry a block of `ratio` slots, as entry_store keeps
 *                      them; the block table permuted) -> f32 scores, every column's bits
 *   index_candidates   (ratio 1: the candidate layer's) blocks of 8 positions, the best 2048 a row ->
 *                      keep words
 *   index_topk         the best 512 positions a row in position order, -1 past them; without keep, and
 *                      at ratio 1 also with the candidates' keep (layers 24-36); its selections are also
 *                      held, on three rows a width, to a host top-k under (score desc, position asc)
 *   attn_sparse_sink   16 and 32 heads, the 128-position window and the 512 selected compressed entries
 *                      (the chunk's own top-k), bf16 outputs
 *
 * The chunk's selections must hold 512 positions a row (else the comparison would be vacuous). */
constexpr int kLsIH = 32, kLsID = 128, kLsTopk = 512, kLsCandBlock = 8, kLsCandBlocks = 2048;
constexpr int kLsD = 512, kLsWin = 128, kLsBs = 16;

uint64_t ls_mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}
/* Uniform bf16 values in [-amp, amp), a hash of (seed, index): fast at hundreds of millions. */
void ls_fill(std::vector<uint16_t>& v, uint64_t seed, float amp) {
    for (size_t i = 0; i < v.size(); ++i) {
        const float u = (float)(ls_mix(seed ^ (i * 0x632be59bd9b4e019ull)) >> 40) / 16777216.0f;
        v[i] = f32_to_bf16((2.0f * u - 1.0f) * amp);
    }
}
/* A permutation of [0, n) (block tables). */
std::vector<int32_t> ls_perm(long n, uint64_t seed) {
    std::vector<int32_t> p((size_t)n);
    for (long i = 0; i < n; ++i) p[(size_t)i] = (int32_t)i;
    std::mt19937_64 rng(seed);
    std::shuffle(p.begin(), p.end(), rng);
    return p;
}

struct LongSparse {
    int W = 0, r = 1, M = 0;
    long last = 0, P0 = 0;
    long nib = 0, ncb = 0, nwbt = 0, nwphys = 0, kwords = 0;
    std::vector<int32_t> pos;
    std::unique_ptr<Dev> iq, iw, kc, ibt, dpos, wc, wbt, cc, cbt, q16, q32, sink16, sink32, cu, used, scratch;
    int64_t scratch_bytes = 0;
};

/* The fixture of one width: the chunk's last row at the last position of a context of (W - 1) * r. */
bool ls_build(const Lib& L, LongSparse& f, int W, int r, int M, uint64_t seed) {
    f.W = W; f.r = r; f.M = M;
    f.last = (long)(W - 1) * r - 1;
    f.P0 = f.last - M + 1;
    f.pos.resize((size_t)M);
    for (int m = 0; m < M; ++m) f.pos[(size_t)m] = (int32_t)(f.P0 + m);
    /* The index keys and the compressed entries: one entry a block, block b of the table at entry b. */
    f.nib = W;
    f.ncb = W;
    {
        std::vector<uint16_t> v((size_t)M * kLsIH * kLsID);
        ls_fill(v, seed + 1, 0.5f);
        f.iq.reset(new Dev(v.size() * 2)); f.iq->put(v.data());
        v.assign((size_t)M * kLsIH, 0);
        ls_fill(v, seed + 2, 1.0f);
        f.iw.reset(new Dev(v.size() * 2)); f.iw->put(v.data());
        v.assign((size_t)f.nib * kLsID, 0);
        ls_fill(v, seed + 3, 0.5f);
        f.kc.reset(new Dev(v.size() * 2)); f.kc->put(v.data());
        v.assign((size_t)f.ncb * kLsD, 0);
        ls_fill(v, seed + 4, 0.5f);
        f.cc.reset(new Dev(v.size() * 2)); f.cc->put(v.data());
    }
    {
        const std::vector<int32_t> ib = ls_perm(f.nib, seed + 5), cb = ls_perm(f.ncb, seed + 6);
        f.ibt.reset(new Dev(ib.size() * 4)); f.ibt->put(ib.data());
        f.cbt.reset(new Dev(cb.size() * 4)); f.cbt->put(cb.data());
    }
    /* The window: its table covers positions from 0 (window_used = the launch's last position + 1); only
     * the blocks the chunk's windows reach hold keys (permuted), the rest name block 0 and are not read. */
    f.nwbt = f.last / kLsBs + 1;
    const long b_lo = std::max(0L, (f.P0 - kLsWin + 1) / kLsBs), b_hi = f.last / kLsBs;
    f.nwphys = b_hi - b_lo + 1;
    {
        std::vector<int32_t> wb((size_t)f.nwbt, 0);
        const std::vector<int32_t> p = ls_perm(f.nwphys, seed + 7);
        for (long b = b_lo; b <= b_hi; ++b) wb[(size_t)b] = p[(size_t)(b - b_lo)];
        f.wbt.reset(new Dev(wb.size() * 4)); f.wbt->put(wb.data());
        std::vector<uint16_t> v((size_t)f.nwphys * kLsBs * kLsD);
        ls_fill(v, seed + 8, 0.5f);
        f.wc.reset(new Dev(v.size() * 2)); f.wc->put(v.data());
    }
    for (int H : { 16, 32 }) {
        std::vector<uint16_t> v((size_t)M * H * kLsD);
        ls_fill(v, seed + 9 + (uint64_t)H, 0.5f);
        std::vector<float> s((size_t)H);
        for (int h = 0; h < H; ++h) s[(size_t)h] = (float)((int)(ls_mix(seed + 100 + (uint64_t)h) % 2001) - 1000) / 1000.0f;
        Dev* dq = new Dev(v.size() * 2);
        dq->put(v.data());
        Dev* ds = new Dev(s.size() * 4);
        ds->put(s.data());
        (H == 16 ? f.q16 : f.q32).reset(dq);
        (H == 16 ? f.sink16 : f.sink32).reset(ds);
    }
    f.dpos.reset(new Dev(f.pos.size() * 4)); f.dpos->put(f.pos.data());
    f.cu.reset(new Dev(8));
    f.used.reset(new Dev(4));
    f.kwords = (((long)W + kLsCandBlock - 1) / kLsCandBlock + 31) / 32;
    /* index_topk's scratch, as the core sizes it at declare: the hook at the band's bound of 16 rows. */
    {
        RadParam p[4] = { pi("M", 16), pi("width", W), pi("topk", kLsTopk), pi("ratio", r) };
        RadArgs a{};
        a.p = p; a.n_p = 4;
        f.scratch_bytes = L.itopk_scratch(&a);
        if (f.scratch_bytes <= 0) {
            std::printf("FAIL: lsparse W=%d: index_topk's scratch hook asks for %lld bytes (no split form for a long row?)\n",
                        W, (long long)f.scratch_bytes);
            return false;
        }
        f.scratch.reset(new Dev((size_t)f.scratch_bytes));
    }
    return true;
}

void ls_rows(LongSparse& f, int n, int r0) {
    const int32_t cu[2] = { 0, n }, used = f.pos[(size_t)(r0 + n - 1)] + 1;
    f.cu->put(cu);
    f.used->put(&used);
}

/* Rows [r0, r0 + n) of the chunk, one launch of each op into `out` (device; rows from 0). */
int ls_score(const Lib& L, hipStream_t s, LongSparse& f, int r0, int n, float* out) {
    ls_rows(f, n, r0);
    RadTensor t[7] = { tensor((uint16_t*)f.iq->p + (size_t)r0 * kLsIH * kLsID, RAD_BF16, { n, kLsIH * kLsID }),
                       tensor((uint16_t*)f.iw->p + (size_t)r0 * kLsIH, RAD_BF16, { n, kLsIH }),
                       tensor(f.kc->p, RAD_BF16, { f.nib, 1, f.r, kLsID / f.r }),
                       tensor(f.ibt->p, RAD_I32, { 1, f.nib }),
                       tensor((int32_t*)f.dpos->p + r0, RAD_I32, { n }), tensor(f.cu->p, RAD_I32, { 2 }),
                       tensor(out, RAD_F32, { n, f.W }) };
    RadParam p[7] = { pi("M", n), pi("heads", kLsIH), pi("dim", kLsID), pi("ratio", f.r), pi("width", f.W),
                      pf("wscale", 1.0 / std::sqrt((double)kLsID * (double)kLsIH)), ps("dtype", "bf16") };
    RadArgs a{};
    a.t = t; a.n_t = 7; a.p = p; a.n_p = 7;
    const int rc = L.iscore(&a, (RadStream)s);
    HC(hipStreamSynchronize(s));
    return rc;
}
int ls_cand(const Lib& L, hipStream_t s, LongSparse& f, int r0, int n, const float* score, int32_t* keep) {
    RadTensor t[3] = { tensor((void*)score, RAD_F32, { n, f.W }), tensor((int32_t*)f.dpos->p + r0, RAD_I32, { n }),
                       tensor(keep, RAD_I32, { n, f.kwords }) };
    RadParam p[3] = { pi("block", kLsCandBlock), pi("topk_blocks", kLsCandBlocks), pi("ratio", f.r) };
    RadArgs a{};
    a.t = t; a.n_t = 3; a.p = p; a.n_p = 3;
    const int rc = L.icand(&a, (RadStream)s);
    HC(hipStreamSynchronize(s));
    return rc;
}
int ls_topk(const Lib& L, hipStream_t s, LongSparse& f, int r0, int n, const float* score, const int32_t* keep,
            int32_t* sel) {
    RadTensor t[4] = { tensor((void*)score, RAD_F32, { n, f.W }),
                       keep ? tensor((void*)keep, RAD_I32, { n, f.kwords }) : RadTensor{},
                       tensor((int32_t*)f.dpos->p + r0, RAD_I32, { n }), tensor(sel, RAD_I32, { n, kLsTopk }) };
    std::vector<RadParam> p = { pi("M", n), pi("width", f.W), pi("topk", kLsTopk), pi("ratio", f.r) };
    if (keep) p.push_back(pi("block", kLsCandBlock));
    RadArgs a{};
    a.t = t; a.n_t = 4; a.p = p.data(); a.n_p = (int)p.size();
    a.scratch = f.scratch->p; a.scratch_bytes = f.scratch_bytes;
    const int rc = L.itopk(&a, (RadStream)s);
    HC(hipStreamSynchronize(s));
    return rc;
}
int ls_attn(const Lib& L, hipStream_t s, LongSparse& f, int H, int r0, int n, const int32_t* sel, uint16_t* out) {
    ls_rows(f, n, r0);
    const Dev& q = *(H == 16 ? f.q16 : f.q32);
    const Dev& sink = *(H == 16 ? f.sink16 : f.sink32);
    RadTensor t[11] = { tensor((uint16_t*)q.p + (size_t)r0 * H * kLsD, RAD_BF16, { n, (int64_t)H * kLsD }),
                        tensor(f.wc->p, RAD_BF16, { f.nwphys, 1, kLsBs, kLsD }), tensor(f.wbt->p, RAD_I32, { 1, f.nwbt }),
                        tensor(f.used->p, RAD_I32, { 1 }), tensor(f.cc->p, RAD_BF16, { f.ncb, 1, f.r, kLsD / f.r }),
                        tensor(f.cbt->p, RAD_I32, { 1, f.ncb }), tensor((void*)(sel + (size_t)r0 * kLsTopk), RAD_I32, { n, kLsTopk }),
                        tensor((int32_t*)f.dpos->p + r0, RAD_I32, { n }), tensor(f.cu->p, RAD_I32, { 2 }),
                        tensor(sink.p, RAD_F32, { H }), tensor(out, RAD_BF16, { n, (int64_t)H * kLsD }) };
    RadParam p[8] = { pi("M", n), pi("heads", H), pi("dim", kLsD), pi("window", kLsWin), pi("ratio", f.r),
                      pi("topk", kLsTopk), pf("scale", 0.04419417382415922), ps("dtype", "bf16") };
    RadArgs a{};
    a.t = t; a.n_t = 11; a.p = p; a.n_p = 8;
    const int rc = L.attn(&a, (RadStream)s);
    HC(hipStreamSynchronize(s));
    return rc;
}

/* The selection's key (r4d_ds41_sparse.hip ds_key; 0 for -inf and NaN, not candidates). */
uint32_t ls_key(float v) {
    if (!(v > -INFINITY)) return 0u;
    const float w = v + 0.0f;
    uint32_t u;
    std::memcpy(&u, &w, 4);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
/* The best `take` of keys (index i -> key, 0 = none) under (key desc, index asc), as a membership mask. */
std::vector<char> ls_best(const std::vector<uint32_t>& key, long take) {
    std::vector<int32_t> idx;
    for (size_t i = 0; i < key.size(); ++i) if (key[i]) idx.push_back((int32_t)i);
    const auto better = [&](int32_t a, int32_t b) { return key[(size_t)a] != key[(size_t)b] ? key[(size_t)a] > key[(size_t)b] : a < b; };
    if ((long)idx.size() > take) {
        std::nth_element(idx.begin(), idx.begin() + take, idx.end(), better);
        idx.resize((size_t)take);
    }
    std::vector<char> in(key.size(), 0);
    for (int32_t i : idx) in[(size_t)i] = 1;
    return in;
}
/* Row m's candidates (index_candidates) and selection (index_topk, with keep or without) on the host. */
std::vector<int32_t> ls_host_keep(const LongSparse& f, const float* row, int m) {
    const long nb = ((long)f.W + kLsCandBlock - 1) / kLsCandBlock;
    const long reach = ((long)f.pos[(size_t)m] + 1) / f.r;
    const long pin = (reach >= 1 && (reach - 1) / kLsCandBlock < nb) ? (reach - 1) / kLsCandBlock : -1;
    std::vector<uint32_t> key((size_t)nb);
    for (long b = 0; b < nb; ++b) {
        float bs = -INFINITY;
        for (long j = b * kLsCandBlock; j < std::min((long)f.W, (b + 1) * kLsCandBlock); ++j) bs = bs < row[j] ? row[j] : bs;
        if (b == pin) bs = INFINITY;
        key[(size_t)b] = ls_key(bs);
    }
    const std::vector<char> in = ls_best(key, kLsCandBlocks);
    std::vector<int32_t> words((size_t)f.kwords, 0);
    for (long b = 0; b < nb; ++b)
        if (in[(size_t)b]) words[(size_t)(b >> 5)] |= (int32_t)(1u << (b & 31));
    return words;
}
std::vector<int32_t> ls_host_topk(const LongSparse& f, const float* row, int m, const int32_t* keep) {
    long reach = ((long)f.pos[(size_t)m] + 1) / f.r;
    reach = std::max(0L, std::min(reach, (long)f.W));
    std::vector<uint32_t> key((size_t)reach);
    for (long j = 0; j < reach; ++j) {
        const long b = j / kLsCandBlock;
        key[(size_t)j] = keep && !(((uint32_t)keep[b >> 5] >> (b & 31)) & 1u) ? 0u : ls_key(row[j]);
    }
    const std::vector<char> in = ls_best(key, kLsTopk);
    std::vector<int32_t> sel;
    for (long j = 0; j < reach; ++j) if (in[(size_t)j]) sel.push_back((int32_t)j);
    sel.resize(kLsTopk, -1);
    return sel;
}

/* Every row range of the chunk the check launches apart from the whole: alone, 16 at a time, suffixes. */
std::vector<std::pair<int, int>> ls_ranges(int M) {
    std::vector<std::pair<int, int>> v;
    for (int r0 : { 16, M / 2 / 16 * 16, (M - 1) / 16 * 16 })
        if (r0 > 0 && r0 < M) v.push_back({ r0, M - r0 });
    for (int cut : { 1, 16 })
        for (int r0 = 0; r0 < M; r0 += cut) v.push_back({ r0, std::min(cut, M - r0) });
    return v;
}

/* One width: each op's whole chunk against its row ranges; the differing values (1 << 40 on a refusal). */
size_t ls_width(const Lib& L, hipStream_t s, int W, int r, int M, size_t* compared) {
    LongSparse f;
    if (!ls_build(L, f, W, r, M, 0x15a7c0deull + (uint64_t)W * 31 + (uint64_t)r)) return (size_t)1 << 40;
    const size_t plane = (size_t)M * W;
    Dev score(plane * 4), score2(plane * 4), keep((size_t)M * f.kwords * 4), keep2((size_t)M * f.kwords * 4),
        sel((size_t)M * kLsTopk * 4), selk((size_t)M * kLsTopk * 4), sel2((size_t)M * kLsTopk * 4),
        att((size_t)M * 32 * kLsD * 2), att2((size_t)M * 32 * kLsD * 2);
    const auto refused = [&](const char* op, int rc, int r0, int n) {
        std::printf("FAIL: lsparse W=%d r=%d: %s refused rows %d..%d (%d)\n", W, r, op, r0, r0 + n - 1, rc);
        return (size_t)1 << 40;
    };
    HC(hipMemset(score.p, 0xAA, score.n));
    int rc = ls_score(L, s, f, 0, M, (float*)score.p);
    if (rc < 0) return refused("index_score", rc, 0, M);
    std::vector<float> hs(plane);
    score.get(hs.data());
    /* The chunk's scores: finite below each row's reach, -inf from it. */
    size_t bad_shape = 0;
    for (int m = 0; m < M; ++m) {
        const long reach = std::min((long)W, ((long)f.pos[(size_t)m] + 1) / r);
        for (long j = 0; j < W; ++j) {
            const float v = hs[(size_t)m * W + j];
            bad_shape += j < reach ? !std::isfinite(v) : !(v == -INFINITY);
        }
    }
    if (bad_shape) std::printf("  W=%d r=%d: %zu scores not finite below a row's reach or not -inf from it\n", W, r, bad_shape);
    const bool cand = r == 1;
    std::vector<int32_t> hk, hsel((size_t)M * kLsTopk), hselk;
    if (cand) {
        HC(hipMemset(keep.p, 0xAA, keep.n));
        if ((rc = ls_cand(L, s, f, 0, M, (const float*)score.p, (int32_t*)keep.p)) < 0) return refused("index_candidates", rc, 0, M);
        hk.resize((size_t)M * f.kwords);
        keep.get(hk.data());
    }
    HC(hipMemset(sel.p, 0xAA, sel.n));
    if ((rc = ls_topk(L, s, f, 0, M, (const float*)score.p, nullptr, (int32_t*)sel.p)) < 0) return refused("index_topk", rc, 0, M);
    sel.get(hsel.data());
    if (cand) {
        HC(hipMemset(selk.p, 0xAA, selk.n));
        if ((rc = ls_topk(L, s, f, 0, M, (const float*)score.p, (const int32_t*)keep.p, (int32_t*)selk.p)) < 0)
            return refused("index_topk (keep)", rc, 0, M);
        hselk.resize((size_t)M * kLsTopk);
        selk.get(hselk.data());
    }
    /* Not vacuous: every row of the chunk selected 512 positions. */
    size_t short_rows = 0;
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < kLsTopk; ++k) {
            short_rows += hsel[(size_t)m * kLsTopk + k] < 0;
            if (cand) short_rows += hselk[(size_t)m * kLsTopk + k] < 0;
        }
    if (short_rows) std::printf("  W=%d r=%d: %zu unfilled selections in the chunk's top-k\n", W, r, short_rows);
    /* The host's selection on the first, middle and last rows. */
    size_t host_bad = 0;
    for (int m : { 0, M / 2, M - 1 }) {
        const float* row = &hs[(size_t)m * W];
        std::vector<int32_t> hkm;
        if (cand) {
            hkm = ls_host_keep(f, row, m);
            host_bad += std::memcmp(hkm.data(), &hk[(size_t)m * f.kwords], (size_t)f.kwords * 4) != 0;
        }
        const std::vector<int32_t> t0 = ls_host_topk(f, row, m, nullptr);
        host_bad += std::memcmp(t0.data(), &hsel[(size_t)m * kLsTopk], kLsTopk * 4) != 0;
        if (cand) {
            const std::vector<int32_t> t1 = ls_host_topk(f, row, m, &hk[(size_t)m * f.kwords]);
            host_bad += std::memcmp(t1.data(), &hselk[(size_t)m * kLsTopk], kLsTopk * 4) != 0;
        }
    }
    if (host_bad) std::printf("  W=%d r=%d: %zu of the sampled rows' selections or keep words differ from the host's\n", W, r, host_bad);
    std::vector<std::vector<uint16_t>> ha(2);
    for (int hi = 0; hi < 2; ++hi) {
        const int H = hi ? 32 : 16;
        HC(hipMemset(att.p, 0xAA, att.n));
        if ((rc = ls_attn(L, s, f, H, 0, M, (const int32_t*)sel.p, (uint16_t*)att.p)) < 0) return refused("attn_sparse_sink", rc, 0, M);
        ha[(size_t)hi].resize((size_t)M * H * kLsD);
        HC(hipMemcpy(ha[(size_t)hi].data(), att.p, ha[(size_t)hi].size() * 2, hipMemcpyDeviceToHost));
    }
    size_t d_score = 0, d_cand = 0, d_topk = 0, d_topkk = 0, d_attn = 0;
    std::vector<float> s2;
    std::vector<int32_t> i2;
    std::vector<uint16_t> a2;
    for (const auto& rg : ls_ranges(M)) {
        const int r0 = rg.first, n = rg.second;
        /* Each launch's rows start from 0xAA bytes, so a value it does not write cannot pass. */
        HC(hipMemset(score2.p, 0xAA, (size_t)n * W * 4));
        if ((rc = ls_score(L, s, f, r0, n, (float*)score2.p)) < 0) return refused("index_score", rc, r0, n);
        s2.resize((size_t)n * W);
        HC(hipMemcpy(s2.data(), score2.p, s2.size() * 4, hipMemcpyDeviceToHost));
        d_score += std::memcmp(s2.data(), &hs[(size_t)r0 * W], s2.size() * 4) ? [&] {
            size_t d = 0;
            for (size_t i = 0; i < s2.size(); ++i) d += std::memcmp(&s2[i], &hs[(size_t)r0 * W + i], 4) != 0;
            return d;
        }() : 0;
        /* The selections from the chunk's own scores and keep (each op on the same inputs). */
        const float* srow = (const float*)score.p + (size_t)r0 * W;
        if (cand) {
            HC(hipMemset(keep2.p, 0xAA, (size_t)n * f.kwords * 4));
            if ((rc = ls_cand(L, s, f, r0, n, srow, (int32_t*)keep2.p)) < 0) return refused("index_candidates", rc, r0, n);
            i2.resize((size_t)n * f.kwords);
            HC(hipMemcpy(i2.data(), keep2.p, i2.size() * 4, hipMemcpyDeviceToHost));
            for (size_t i = 0; i < i2.size(); ++i) d_cand += i2[i] != hk[(size_t)r0 * f.kwords + i];
        }
        HC(hipMemset(sel2.p, 0xAA, (size_t)n * kLsTopk * 4));
        if ((rc = ls_topk(L, s, f, r0, n, srow, nullptr, (int32_t*)sel2.p)) < 0) return refused("index_topk", rc, r0, n);
        i2.resize((size_t)n * kLsTopk);
        HC(hipMemcpy(i2.data(), sel2.p, i2.size() * 4, hipMemcpyDeviceToHost));
        for (size_t i = 0; i < i2.size(); ++i) d_topk += i2[i] != hsel[(size_t)r0 * kLsTopk + i];
        if (cand) {
            const int32_t* krow = (const int32_t*)keep.p + (size_t)r0 * f.kwords;
            HC(hipMemset(sel2.p, 0xAA, (size_t)n * kLsTopk * 4));
            if ((rc = ls_topk(L, s, f, r0, n, srow, krow, (int32_t*)sel2.p)) < 0) return refused("index_topk (keep)", rc, r0, n);
            HC(hipMemcpy(i2.data(), sel2.p, i2.size() * 4, hipMemcpyDeviceToHost));
            for (size_t i = 0; i < i2.size(); ++i) d_topkk += i2[i] != hselk[(size_t)r0 * kLsTopk + i];
        }
        for (int hi = 0; hi < 2; ++hi) {
            const int H = hi ? 32 : 16;
            HC(hipMemset(att2.p, 0xAA, (size_t)n * H * kLsD * 2));
            if ((rc = ls_attn(L, s, f, H, r0, n, (const int32_t*)sel.p, (uint16_t*)att2.p)) < 0) return refused("attn_sparse_sink", rc, r0, n);
            a2.resize((size_t)n * H * kLsD);
            HC(hipMemcpy(a2.data(), att2.p, a2.size() * 2, hipMemcpyDeviceToHost));
            d_attn += count_diff(&ha[(size_t)hi][(size_t)r0 * H * kLsD], a2.data(), a2.size());
        }
        *compared += (size_t)n;
    }
    std::printf("lsparse W=%d (ratio %d, positions %ld..%ld, %d rows): index_score %zu, %sindex_topk %zu%s, "
                "attn_sparse_sink (16, 32 heads) %zu differing values against rows alone, 16 at a time and as "
                "suffixes; host top-k %zu, unfilled %zu, score shape %zu\n",
                W, r, f.P0, f.last, M, d_score, cand ? ("index_candidates " + std::to_string(d_cand) + ", ").c_str() : "",
                d_topk, cand ? (" (with keep " + std::to_string(d_topkk) + ")").c_str() : "", d_attn, host_bad,
                short_rows, bad_shape);
    std::fflush(stdout);
    return d_score + d_cand + d_topk + d_topkk + d_attn + host_bad + short_rows + bad_shape;
}

/* `lsparse [C] [W:r ...]`: the widths (default the two ratios' at --max-model-len 32768, 131072 and
 * 524288's), M = 256 rows (the score plane's rows a run at 256K and above). */
int long_sparse_check(const Lib& L, const std::vector<std::pair<int, int>>& widths) {
    if (!L.attn || !L.iscore || !L.itopk || !L.icand || !L.itopk_scratch) {
        std::printf("lsparse: FAIL: the plugin lacks a sparse op's launcher (attn_sparse_sink, index_score, index_topk, "
                    "index_candidates or index_topk's scratch hook)\n");
        return 1;
    }
    hipStream_t s;
    HC(hipStreamCreate(&s));
    size_t bad = 0, rows = 0;
    std::string ws;
    for (const auto& w : widths) {
        bad += ls_width(L, s, w.first, w.second, 256, &rows);
        ws += (ws.empty() ? "" : " ") + std::to_string(w.first) + ":" + std::to_string(w.second);
    }
    HC(hipStreamDestroy(s));
    if (bad) std::printf("lsparse: FAIL: %zu differing values at widths %s\n", bad, ws.c_str());
    else std::printf("lsparse: PASS: index_score, index_candidates, index_topk and attn_sparse_sink at index widths %s "
                     "keep the chunk's bits in %zu rows launched alone, 16 at a time and as suffixes; host top-k equal\n",
                     ws.c_str(), rows);
    return bad ? 1 : 0;
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

/* The rows a sweep case holds: 48, or 2048 under LCP (sweep_check). */
int kSweepRows = 48;

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
    /* The split form's partials for every row of the case (libr4d's r4d_rad_mhc_mix_scratch: rows x
     * chunks of 1024 columns x (mix + 1) floats), as the engine sizes them at the band's top. */
    Dev* scr = o->buf(std::max<size_t>(4u << 20, (size_t)kSweepRows * ((W + 1023) / 1024) * (mix + 1) * 4));
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

/* gemm_nt at the row the table resolves at each height (LCP): gemm_bf16_nt_m16 up to 16 rows, the
 * tiled row above 64 where it serves (N >= 128, K a multiple of 64), gemm_bf16_nt_m64 elsewhere.
 * ds41's index projection (64 x 5120), its index key (128 x 512) and layer 20's compressor (512 x
 * 5120). */
std::unique_ptr<SweepOp> sweep_gemm_rows(const Lib& L, int N, int K, std::mt19937_64& rng) {
    auto o = std::make_unique<SweepOp>();
    o->name = "gemm_nt (m16/m64/tiled) N=" + std::to_string(N) + " K=" + std::to_string(K);
    Dev* A = o->buf((size_t)kSweepRows * K * 2);
    Dev* Wt = o->buf((size_t)N * K * 2);
    Dev* C = o->buf((size_t)kSweepRows * N * 2);
    fill_bf16(A, rng, 1.0f);
    fill_bf16(Wt, rng, 0.02f);
    o->outs = { { C, (size_t)N * 2 } };
    const Launch m16 = L.gemm16, m64 = L.gemm64, tiled = L.gemm_tiled;
    o->launch = [=](int r0, int m, hipStream_t s) {
        const Launch f = m <= 16 && m16 ? m16 : (m > 64 && N >= 128 && K % 64 == 0 && tiled ? tiled : m64);
        RadTensor t[3] = { rows_of(A, RAD_BF16, K, r0, m), tensor(Wt->p, RAD_BF16, { N, K }),
                           rows_of(C, RAD_BF16, N, r0, m) };
        RadParam p[4] = { pi("M", m), pi("N", N), pi("K", K), ps("dtype", "bf16") };
        RadArgs a{};
        a.t = t; a.n_t = 3; a.p = p; a.n_p = 4;
        return f(&a, (RadStream)s);
    };
    return o;
}

/* gemm_nt_bf16f8 (r4d_gemm_bf16f8_nt_m64.hip: the FP8-stored dense weights' one row at every M) over
 * its stored form (r4d_bf16f8_layout.h), written here from random E4M3 codes -- the NaN codes left
 * out, signed zeros and subnormals in -- and an E8M0 exponent a 32x32 block in [112, 120): ds41's
 * q_a (1280 x 5120) and wo_b-shaped (4096 x 1024) projections, N >= 128, where gemm_nt's bf16 rows
 * would go tiled above 64 rows while this one stays skinny (lcp review M1). */
std::unique_ptr<SweepOp> sweep_bf16f8(Launch f, int N, int K, std::mt19937_64& rng) {
    auto o = std::make_unique<SweepOp>();
    o->name = "gemm_nt_bf16f8 N=" + std::to_string(N) + " K=" + std::to_string(K);
    Dev* A = o->buf((size_t)kSweepRows * K * 2);
    const int64_t bytes = r4d_bf16f8_bytes(N, K);
    Dev* Wt = o->buf((size_t)bytes);
    Dev* C = o->buf((size_t)kSweepRows * N * 2);
    fill_bf16(A, rng, 1.0f);
    std::vector<uint8_t> h((size_t)bytes, 0);
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k) {
            uint8_t b = (uint8_t)(rng() % 256);
            if ((b & 0x7F) == 0x7F) b ^= 1;   /* NaN: never stored */
            h[(size_t)r4d_bf16f8_slot(K, n, k)] = b;
        }
    const int64_t off = r4d_bf16f8_scale_off(N, K), ldws = r4d_bf16f8_ldws(K);
    for (int64_t nb = 0; nb < (N + 31) / 32; ++nb)
        for (int64_t kb = 0; kb < K / 32; ++kb) h[(size_t)(off + nb * ldws + kb)] = (uint8_t)(112 + rng() % 8);
    Wt->put(h.data());
    o->outs = { { C, (size_t)N * 2 } };
    o->launch = [=](int r0, int m, hipStream_t s) {
        RadTensor t[3] = { rows_of(A, RAD_BF16, K, r0, m), tensor(Wt->p, R4D_BF16F8_DTYPE, { N, K }),
                           rows_of(C, RAD_BF16, N, r0, m) };
        RadParam p[4] = { pi("M", m), pi("N", N), pi("K", K), ps("dtype", "bf16") };
        RadArgs a{};
        a.t = t; a.n_t = 3; a.p = p; a.n_p = 4;
        return f(&a, (RadStream)s);
    };
    return o;
}

/* gemm_nt_f32 (bf16 x, f32 weight, f32 out): the router (384 x 5120), and with `parts` 4 the same
 * outputs as the four 96-column slices the ranks compute once the plugin splits it by columns
 * (rad_glm_tp.h SplitF32Proj, from 256 rows), each launch into its own columns; the compressor's
 * kv|gate projection (1024 x 5120). */
std::unique_ptr<SweepOp> sweep_f32(Launch f, int N, int K, int parts, std::mt19937_64& rng) {
    auto o = std::make_unique<SweepOp>();
    o->name = "gemm_nt_f32 N=" + std::to_string(N) + " K=" + std::to_string(K) +
              (parts > 1 ? " in " + std::to_string(parts) + " column slices" : "");
    Dev* x = o->buf((size_t)kSweepRows * K * 2);
    Dev* w = o->buf((size_t)N * K * 4);
    Dev* y = o->buf((size_t)kSweepRows * N * 4);
    fill_bf16(x, rng, 1.0f);
    fill_f32(w, rng, 0.05f);
    o->outs = { { y, (size_t)N * 4 } };
    o->launch = [=](int r0, int m, hipStream_t s) {
        const int cols = N / parts;
        for (int c = 0; c < parts; ++c) {
            RadTensor t[3] = { rows_of(x, RAD_BF16, K, r0, m),
                               tensor((char*)w->p + (size_t)c * cols * K * 4, RAD_F32, { cols, K }),
                               tensor((char*)y->p + ((size_t)r0 * N + (size_t)c * cols) * 4, RAD_F32, { m, cols }) };
            t[2].stride[0] = N;
            RadParam p[4] = { pi("M", m), pi("N", cols), pi("K", K), ps("dtype", "bf16") };
            RadArgs a{};
            a.t = t; a.n_t = 3; a.p = p; a.n_p = 4;
            const int rc = f(&a, (RadStream)s);
            if (rc < 0) return rc;
        }
        return 0;
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
    int top = 0;   /* the rows the runs cover: only those are read back */
    for (const auto& r : runs) {
        if (o.launch(r.first, r.second, s) < 0) return {};
        top = std::max(top, r.first + r.second);
    }
    HC(hipStreamSynchronize(s));
    std::vector<std::vector<uint8_t>> rows((size_t)top);
    for (const SweepOut& q : o.outs) {
        std::vector<uint8_t> h((size_t)top * q.row_bytes);
        HC(hipMemcpy(h.data(), q.d->p, h.size(), hipMemcpyDeviceToHost));
        for (int m = 0; m < top; ++m)
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
    std::vector<int> heights = { 1, 16, 17, 24, 25, 32, 33, 40, 48 };
    if (on && g_lcp) {
        heights = { 1, 16, 17, 32, 33, 48, 49, 64, 65, 100, 128, 170, 171, 255, 256, 257, 512, 1000, 1024, 2048 };
        kSweepRows = std::max(2048, g_lcp);
        if (g_lcp > 2048) {
            for (int M : { 2049, 3072, 4095, 4096 }) if (M <= g_lcp) heights.push_back(M);
            if (std::find(heights.begin(), heights.end(), g_lcp) == heights.end()) heights.push_back(g_lcp);
        }
    }
    std::printf("sweep: %s; heights up to %lld must give every row its bytes launched alone and in launches of 16; "
                "%d trials\n", !on ? "RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL unset"
                : g_lcp ? ("RADIANCE_EXPERIMENTAL_DS41_LCP=" + std::to_string(g_lcp)).c_str()
                        : "RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL 1", reach, trials);
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
        size_t want = 13;
        if (on && g_lcp) {
            if (L.gemm16 && L.gemm64 && L.gemm_tiled) {
                ops.push_back(sweep_gemm_rows(L, 64, 5120, rng));
                ops.push_back(sweep_gemm_rows(L, 128, 512, rng));
                ops.push_back(sweep_gemm_rows(L, 512, 5120, rng));
            }
            if (L.gemm_f32) {
                ops.push_back(sweep_f32(L.gemm_f32, 384, 5120, 1, rng));
                ops.push_back(sweep_f32(L.gemm_f32, 384, 5120, 4, rng));
                if (trial == 0) ops.push_back(sweep_f32(L.gemm_f32, 1024, 5120, 1, rng));
            }
            if (L.gemm_f8) {
                ops.push_back(sweep_bf16f8(L.gemm_f8, 1280, 5120, rng));
                ops.push_back(sweep_bf16f8(L.gemm_f8, 4096, 1024, rng));
            }
            want += trial == 0 ? 8 : 7;
            if (trial == 0 && (!L.gemm16 || !L.gemm_tiled || !L.gemm_f32 || !L.gemm_f8)) {
                std::printf("FAIL: the plugin lacks some of r4d_rad_gemm_bf16_m16, _gemm_bf16_tiled, _gemm_nt_f32, "
                            "_gemm_bf16f8_m64\n");
                rc = 1;
            }
        }
        if (trial == 0 && ops.size() < want) {
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


/* ================================================================== bal (RADIANCE_EXPERIMENTAL_DS41_MOE_BALANCE)
 *
 * The balanced kernel against the plain one through r4d_rad_moe_gemm_mxfp4_gemv_bal, byte for byte over every
 * output (dropped slots included, both poisoned first), at ds41's real expert shapes, and both timed. */
struct BalRoute {
    std::vector<int32_t> sorted, off;
    int local = 0, busiest = 0, live = 0;
};
/* One EP4 rank's slots of M tokens (picks(t) its top_k of 384), as moe_scatter writes them. */
BalRoute bal_route(const std::function<void(int, int*)>& picks, int M, int rank) {
    std::vector<std::vector<int32_t>> by(kNe);
    int pk[kTop];
    for (int t = 0; t < M; ++t) {
        picks(t, pk);
        for (int k = 0; k < kTop; ++k)
            if (pk[k] / kNe == rank) by[(size_t)(pk[k] % kNe)].push_back(t * kTop + k);
    }
    BalRoute r;
    r.off.assign(kNe + 1, 0);
    for (int e = 0; e < kNe; ++e) {
        r.off[(size_t)e + 1] = r.off[(size_t)e] + (int)by[(size_t)e].size();
        for (int32_t f : by[(size_t)e]) r.sorted.push_back(f);
        r.busiest = std::max(r.busiest, (int)by[(size_t)e].size());
        r.live += !by[(size_t)e].empty();
    }
    r.local = (int)r.sorted.size();
    r.sorted.resize((size_t)M * kTop, -1);
    return r;
}

double bal_median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? -1 : v[v.size() / 2];
}

int bal_check(const Lib& L, const char* routes_path) {
    if (!L.moe_bal) {
        std::printf("FAIL: the plugin has no r4d_rad_moe_gemm_mxfp4_gemv_bal (a libr4d without ..._MOE_BALANCE)\n");
        return 1;
    }
    long long reach = 0;
    int rc = tall_row_table(L, &reach);
    const int C = g_lcp;
    hipStream_t s;
    HC(hipStreamCreate(&s));
    std::mt19937_64 rng(4242);
    /* 96 experts of each projection, distinct bytes. */
    const size_t gu_b = (size_t)2 * kFf * (kEmbd / 32) * 17, dn_b = (size_t)kEmbd * (kFf / 32) * 17;
    Dev egu(gu_b * kNe), edn(dn_b * kNe);
    {
        std::vector<uint8_t> h(gu_b);
        for (int e = 0; e < kNe; ++e) {
            h = make_blocks(2 * kFf, kEmbd, rng, e % 7 == 0);
            HC(hipMemcpy((char*)egu.p + gu_b * e, h.data(), gu_b, hipMemcpyHostToDevice));
        }
        for (int e = 0; e < kNe; ++e) {
            h = make_blocks(kEmbd, kFf, rng, e % 7 == 0);
            HC(hipMemcpy((char*)edn.p + dn_b * e, h.data(), dn_b, hipMemcpyHostToDevice));
        }
    }
    std::vector<void*> tgu(kNe), tdn(kNe);
    for (int e = 0; e < kNe; ++e) { tgu[(size_t)e] = (char*)egu.p + gu_b * e; tdn[(size_t)e] = (char*)edn.p + dn_b * e; }
    Dev tab_gu(kNe * sizeof(void*)), tab_dn(kNe * sizeof(void*));
    tab_gu.put(tgu.data());
    tab_dn.put(tdn.data());
    const int Mmax = std::max(16, C);
    Dev x((size_t)Mmax * kEmbd * 2), act((size_t)Mmax * kTop * kFf * 2), so((size_t)Mmax * kTop * 4), of((kNe + 1) * 4);
    /* Keep redzones immediately around the active output shape, including when gate_up's N is
     * smaller than down's. A shared wrong extent must not pass old/new output equality. */
    constexpr size_t guard = 256;
    Dev y0((size_t)Mmax * kTop * kEmbd * 2 + 2 * guard), y1((size_t)Mmax * kTop * kEmbd * 2 + 2 * guard);
    {
        std::vector<uint16_t> h((size_t)Mmax * kTop * kFf);
        for (size_t i = 0; i < (size_t)Mmax * kEmbd; i += kEmbd) {
            auto r = make_x(kEmbd, rng, i == 0);
            std::copy(r.begin(), r.end(), h.begin() + (long)i);
        }
        HC(hipMemcpy(x.p, h.data(), (size_t)Mmax * kEmbd * 2, hipMemcpyHostToDevice));
        for (size_t i = 0; i < h.size(); i += kFf) {
            auto r = make_x(kFf, rng, false);
            std::copy(r.begin(), r.end(), h.begin() + (long)i);
        }
        act.put(h.data());
    }
    /* Skewed popularity: expert ranks shuffled over the 384, weight 1 / (1 + rank)^0.9. */
    std::vector<int> perm(4 * kNe);
    for (int i = 0; i < 4 * kNe; ++i) perm[(size_t)i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<double> wts(4 * kNe);
    for (int i = 0; i < 4 * kNe; ++i) wts[(size_t)perm[(size_t)i]] = 1.0 / std::pow(1.0 + i, 0.9);
    std::discrete_distribution<int> skew(wts.begin(), wts.end());
    auto pick6 = [&](bool skewed, int* pk) {
        int n = 0;
        while (n < kTop) {
            const int e = skewed ? skew(rng) : (int)(rng() % (4 * kNe));
            bool dup = false;
            for (int j = 0; j < n; ++j) dup |= pk[j] == e;
            if (!dup) pk[n++] = e;
        }
    };
    /* The route file (ds41_p0_bench's: i32 layers, rows, k, then i16 picks [layers][rows][k]), if given. */
    std::vector<int16_t> rp;
    int r_layers = 0, r_rows = 0, r_k = 0;
    if (routes_path) {
        FILE* f = std::fopen(routes_path, "rb");
        int32_t h[3];
        if (f && std::fread(h, 4, 3, f) == 3 && h[2] == kTop) {
            r_layers = h[0]; r_rows = h[1]; r_k = h[2];
            rp.resize((size_t)r_layers * r_rows * r_k);
            if (std::fread(rp.data(), 2, rp.size(), f) != rp.size()) rp.clear();
        }
        if (f) std::fclose(f);
        if (rp.empty()) std::printf("note: cannot read the routes %s; synthetic routings only\n", routes_path);
    }
    auto timed = [&](const std::function<int()>& launch, double* us) {
        std::vector<double> t;
        hipEvent_t a, b;
        HC(hipEventCreate(&a));
        HC(hipEventCreate(&b));
        if (launch() != RAD_OK) return false;   /* warm-up */
        for (int r = 0; r < 3; ++r) {
            HC(hipEventRecord(a, s));
            if (launch() != RAD_OK) return false;
            HC(hipEventRecord(b, s));
            HC(hipEventSynchronize(b));
            float ms = 0;
            HC(hipEventElapsedTime(&ms, a, b));
            t.push_back(ms * 1000.0);
        }
        (void)hipEventDestroy(a);
        (void)hipEventDestroy(b);
        *us = bal_median(t);
        return true;
    };
    size_t bad = 0, values = 0;
    double sum0 = 0, sum1 = 0;
    std::printf("bal: moe_gemm_ggml mxfp4 gate_up [%d, %d] and down [%d, %d], 96 VRAM experts, EP4 rank slots as served\n",
                2 * kFf, kEmbd, kEmbd, kFf);
    std::printf("%-14s %5s %5s %6s %5s %7s | %11s %11s | %11s %11s | %7s\n", "routing", "M", "rank", "local", "live",
                "busiest", "gu plain us", "gu bal us", "dn plain us", "dn bal us", "speedup");
    if (C > 2048)
        std::printf("bal: heights above M2048 are byte-only: one plain and one balanced launch per projection/profile; no timings\n");
    auto point = [&](const char* label, const BalRoute& r, int M, int rank) {
        HC(hipMemcpy(so.p, r.sorted.data(), r.sorted.size() * 4, hipMemcpyHostToDevice));
        of.put(r.off.data());
        const int T = M * kTop, tall = M > 16 ? 1 : 0;
        const bool byte_only = M > 2048;
        double us[4] = { 0, 0, 0, 0 };
        for (int proj = 0; proj < 2; ++proj) {
            const bool g = proj == 0;
            const int N = g ? 2 * kFf : kEmbd, K = g ? kEmbd : kFf;
            void* tab = g ? tab_gu.p : tab_dn.p;
            void* xin = g ? x.p : act.p;
            const int AR = g ? M : T;
            const char* order = g ? "token" : "sorted";
            const size_t n = (size_t)T * N;
            for (Dev* d : { &y0, &y1 }) {
                HC(hipMemset(d->p, 0xA5, guard));
                HC(hipMemset((char*)d->p + guard + n * 2, 0xA5, guard));
            }
            HC(hipMemset((char*)y0.p + guard, 0xAA, n * 2));
            HC(hipMemset((char*)y1.p + guard, 0x55, n * 2));
            for (int bal = 0; bal < 2; ++bal) {
                void* y = (char*)(bal ? y1.p : y0.p) + guard;
                auto launch = [&] {
                    RadTensor t[5] = { tensor(xin, RAD_BF16, { AR, K }), tensor(tab, RAD_U8, { kNe, N, K / 32 * 17 }),
                                       tensor(so.p, RAD_I32, { T }), tensor(of.p, RAD_I32, { kNe + 1 }),
                                       tensor(y, RAD_BF16, { T, N }) };
                    t[1].stride[0] = 0;
                    RadParam p[6] = { ps("type", "mxfp4"), pi("N", N), pi("K", K), pi("n_expert", kNe),
                                      pi("top_k", kTop), ps("a_order", order) };
                    RadArgs a{};
                    a.t = t; a.n_t = 5; a.p = p; a.n_p = 6;
                    return L.moe_bal(&a, (RadStream)s, tall, bal);
                };
                if (!(byte_only ? launch() == RAD_OK : timed(launch, &us[proj * 2 + bal]))) {
                    std::printf("FAIL %s M=%d: a launch was refused\n", label, M);
                    ++bad;
                    return;
                }
            }
            HC(hipStreamSynchronize(s));
            std::vector<uint16_t> h0(n), h1(n);
            HC(hipMemcpy(h0.data(), (char*)y0.p + guard, n * 2, hipMemcpyDeviceToHost));
            HC(hipMemcpy(h1.data(), (char*)y1.p + guard, n * 2, hipMemcpyDeviceToHost));
            const size_t d = count_diff(h0.data(), h1.data(), n);
            if (d) std::printf("FAIL %s M=%d rank %d %s: %zu of %zu values differ\n", label, M, rank, g ? "gate_up" : "down", d, n);
            bad += d;
            values += n;
            for (int bal = 0; bal < 2; ++bal) {
                const Dev& y = bal ? y1 : y0;
                std::vector<uint8_t> pre(guard), post(guard);
                HC(hipMemcpy(pre.data(), y.p, guard, hipMemcpyDeviceToHost));
                HC(hipMemcpy(post.data(), (char*)y.p + guard + n * 2, guard, hipMemcpyDeviceToHost));
                const size_t gd = (size_t)std::count_if(pre.begin(), pre.end(), [](uint8_t v) { return v != 0xA5; }) +
                                  (size_t)std::count_if(post.begin(), post.end(), [](uint8_t v) { return v != 0xA5; });
                if (gd) std::printf("FAIL %s M=%d rank %d %s %s: %zu output guard bytes changed\n", label, M,
                                    rank, g ? "gate_up" : "down", bal ? "balanced" : "plain", gd);
                bad += gd;
            }
        }
        if (byte_only) {
            std::printf("%-14s %5d %5d %6d %5d %7d | %11s %11s | %11s %11s | byte-only\n", label, M,
                        rank, r.local, r.live, r.busiest, "byte-only", "byte-only", "byte-only", "byte-only");
            std::fflush(stdout);
            return;
        }
        const double p0 = us[0] + us[2], p1 = us[1] + us[3];
        sum0 += p0;
        sum1 += p1;
        std::printf("%-14s %5d %5d %6d %5d %7d | %11.1f %11.1f | %11.1f %11.1f | %6.1fx\n", label, M, rank, r.local, r.live,
                    r.busiest, us[0], us[1], us[2], us[3], p1 > 0 ? p0 / p1 : 0.0);
        std::fflush(stdout);
    };
    std::vector<int> Ms = { 1, 6, 16 };
    for (int M : { 17, 64, 256, 512, 1024, 2048, 3072, 4096 })
        if (M <= C && M <= reach) Ms.push_back(M);
    /* An intermediate valid C (for example 4000) must be tested itself, not inferred from the
     * next lower fixed sweep height. Sorting keeps the reported maximum the actual last height. */
    if (C <= reach && std::find(Ms.begin(), Ms.end(), C) == Ms.end()) Ms.push_back(C);
    std::sort(Ms.begin(), Ms.end());
    for (int M : Ms) {
        for (int skewed = 1; skewed >= 0; --skewed) {
            std::vector<int> picks((size_t)M * kTop);
            for (int t = 0; t < M; ++t) pick6(skewed != 0, &picks[(size_t)t * kTop]);
            for (int rank : { 0, 3 }) {
                const BalRoute r = bal_route([&](int t, int* pk) { std::copy_n(&picks[(size_t)t * kTop], kTop, pk); }, M, rank);
                point(skewed ? "skewed" : "uniform", r, M, rank);
            }
        }
        if (!rp.empty() && M <= r_rows)
            for (int l : { 3, 12, 25, 36 }) {
                if (l >= r_layers) continue;
                char lab[32];
                std::snprintf(lab, sizeof lab, "real L%d", l);
                for (int rank : { 0, 2 }) {
                    const BalRoute r = bal_route([&](int t, int* pk) {
                        const int16_t* q = &rp[((size_t)l * r_rows + t) * r_k];
                        for (int k = 0; k < kTop; ++k) pk[k] = q[k];
                    }, M, rank);
                    point(lab, r, M, rank);
                }
            }
    }
    (void)hipStreamDestroy(s);
    std::printf("bal: %zu output values and output redzones compared; timed heights at M<=2048 only: summed time %.1f ms plain, %.1f ms balanced (%.1fx)\n", values,
                sum0 / 1000.0, sum1 / 1000.0, sum1 > 0 ? sum0 / sum1 : 0.0);
    if (bad) {
        std::printf("FAIL: bal: %zu output values or guard bytes differ between the balanced and the plain kernel\n", bad);
        return 1;
    }
    if (rc) std::printf("FAIL: bal: the row table check failed\n");
    else {
        /* Machine-readable coverage belongs to this successful run, not its requested C alone. */
        std::printf("bal: byte coverage max_M=%d requested_C=%d heights=", Ms.back(), C);
        for (size_t i = 0; i < Ms.size(); ++i) std::printf("%s%d", i ? "," : "", Ms[i]);
        std::printf("\nPASS: bal: the balanced gemv and twin give the plain kernel's bytes at %zu tested heights through M%d tokens (requested C%d; byte-only above M2048)\n",
                    Ms.size(), Ms.back(), C);
    }
    return rc;
}


/* ================================================================== p1 (prefill Phase 1's twin)
 *
 * RADIANCE_EXPERIMENTAL_DS41_P1=1 beside RADIANCE_EXPERIMENTAL_DS41_LCP=2048 (both set by main before the plugin
 * loads): the row moe_gemm_ggml_mxfp4_gemv_tall_p1 in the table in place of the LCP twin, reaching MMAX, and its
 * launcher's promises T1-T5 (tools/ds41-host-experts/ds41_p1.h) at ds41's shapes on 96 VRAM experts in both forms:
 *
 *   rows    at M in {17, 33, 48, 64, 100, 256, 1024, 2048, MMAX} tokens, real (ROUTES), skewed, uniform and
 *           hot routings of an EP4 rank, tables mixing block_mxfp4, packed and skipped entries (a skipped
 *           entry's pointer names unreadable memory): every computed row equal to the plain gemv over every
 *           expert in block_mxfp4 in passes of 16 tokens, sampled rows to the host kernels, skipped experts'
 *           rows untouched, the slots past off[96] zeros; NaN, inf and subnormal activations on some tokens;
 *   split   other work splits of the same call give the same bytes;
 *   waves   the stager's composition (main launch with a subset skipped, waves of 1-3 over it, swiglu_clamped
 *           relaunched over every row) equal to one launch, gate_up, SwiGLU and down;
 *   time    "p1 time:" lines: gate_up + down a call, us a local expert-row and a token row-layer, real routing
 *           at M 64 .. 2048, the forms (every entry block_mxfp4, every entry packed, a mix with a sixth skipped),
 *           against the served LCP twin, that twin cut to the local slots (Phase 0's twin-cut) and the
 *           balanced LCP twin (..._MOE_BALANCE); then a sweep of the work split at M 2048.
 * The last line is "p1: PASS ..." or "p1: FAIL ...". */
namespace {

typedef int (*P1Opts)(const RadArgs*, RadStream, int grid, int item_kib, int fill, int lds_kib);

/* One projection's 96 experts on the card in both forms, and their block bytes on the host. */
struct P1Bank {
    int N = 0, K = 0;
    size_t bb = 0;                        /* block bytes a weight */
    std::vector<std::vector<uint8_t>> host;
    std::unique_ptr<Dev> blocks;          /* 96 weights back to back */
    std::vector<std::unique_ptr<Dev>> packed;
    std::vector<void*> blk, pkd;          /* device addresses */
    int raw_rounds = 0;
};
void p1_bank(P1Bank* b, int N, int K, std::mt19937_64& rng) {
    b->N = N;
    b->K = K;
    const int UK = K / 32, R = (UK + 31) / 32;
    b->bb = (size_t)N * UK * 17;
    b->blocks.reset(new Dev(b->bb * kNe));
    b->host.resize(kNe);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    static const uint8_t edges[] = { 0, 1, 2, 3, 60, 100, 127, 128, 135, 140, 254 };
    for (int e = 0; e < kNe; ++e) {
        /* The release's planes: codes random, e8m0 mostly 120 / 121 as the release's experts have them, a few raw
         * rounds (a block outside any window of four), every 7th expert also e8m0 edges. */
        std::vector<uint8_t> codes((size_t)N * (K / 2)), scales((size_t)N * UK);
        for (size_t i = 0; i < codes.size(); i += 8) {
            const uint64_t r = rng();
            std::memcpy(&codes[i], &r, std::min<size_t>(8, codes.size() - i));
        }
        for (auto& s : scales) {
            const double r = u01(rng);
            s = r < 0.58 ? 120 : r < 0.995 ? 121 : r < 0.998 ? 119 : 122;
        }
        for (int n = 0; n < N; ++n)
            for (int r = 0; r < R; ++r)
                if (u01(rng) < (e % 7 == 3 ? 0.2 : 0.01)) {
                    const int t = r < UK / 32 ? 32 : UK % 32;
                    scales[(size_t)n * UK + 32 * r + (size_t)(rng() % (uint64_t)t)] = edges[rng() % sizeof edges];
                }
        std::vector<uint8_t>& h = b->host[(size_t)e];
        h.resize(b->bb);
        for (int n = 0; n < N; ++n)
            for (int u = 0; u < UK; ++u) {
                uint8_t* blk = &h[((size_t)n * UK + u) * 17];
                const uint8_t* c = &codes[(size_t)n * (K / 2) + (size_t)u * 16];
                auto val = [&](int v) { return (c[v >> 1] >> (4 * (v & 1))) & 15; };
                blk[0] = scales[(size_t)n * UK + u];
                for (int j = 0; j < 16; ++j) blk[1 + j] = (uint8_t)(val(j) | (val(j + 16) << 4));
            }
        HC(hipMemcpy((char*)b->blocks->p + b->bb * e, h.data(), b->bb, hipMemcpyHostToDevice));
        b->blk.push_back((char*)b->blocks->p + b->bb * e);
        PackInfo plan;
        if (!pack_plan(scales.data(), N, K, &plan)) { std::fprintf(stderr, "pack_plan refused expert %d\n", e); std::exit(2); }
        const int cap = std::min(plan.n_exc + 2, N * R);
        const size_t pb = packed_bytes(N, K, cap);
        std::vector<uint8_t> pk(pb);
        Mx v;
        if (pack_planes(codes.data(), scales.data(), N, K, cap, plan, pk.data(), pb) != kPackOk ||
            !packed_view(pk.data(), pb, N, K, &v) || !packed_scales_equal(v, scales.data())) {
            std::fprintf(stderr, "the packer refused expert %d\n", e);
            std::exit(2);
        }
        b->raw_rounds += plan.n_exc;
        b->packed.emplace_back(new Dev(pb));
        b->packed.back()->put(pk.data());
        b->pkd.push_back(b->packed.back()->p);
    }
}

/* A table's entries: 0 block_mxfp4, 1 packed, 2 skipped. */
std::vector<void*> p1_table(const P1Bank& b, const std::vector<int>& forms, const void* noread) {
    std::vector<void*> t(kNe);
    for (int e = 0; e < kNe; ++e)
        t[(size_t)e] = forms[(size_t)e] == 2   ? (void*)((uintptr_t)noread | ds41p1::kMarkSkip)
                       : forms[(size_t)e] == 1 ? (void*)((uintptr_t)b.pkd[(size_t)e] | ds41p1::kMarkPacked)
                                               : b.blk[(size_t)e];
    return t;
}

/* Token t's top-6 of 384 under a routing. */
struct P1Routing {
    std::vector<int> p;
    int rank = 0;
    std::string name;
};
std::vector<int16_t> g_p1_routes;
int g_p1_layers = 0, g_p1_rows = 0;
P1Routing p1_routing(const char* kind, int M, int rank, int layer, std::mt19937_64& rng) {
    P1Routing r;
    r.rank = rank;
    r.name = kind;
    r.p.resize((size_t)M * kTop);
    if (!std::strcmp(kind, "real")) {
        r.name = "real L" + std::to_string(layer);
        for (int t = 0; t < M; ++t) {   /* past the file's rows: the next layer's first rows */
            const int l = (layer + t / g_p1_rows) % g_p1_layers, row = t % g_p1_rows;
            for (int k = 0; k < kTop; ++k) r.p[(size_t)t * kTop + k] = g_p1_routes[((size_t)l * g_p1_rows + row) * kTop + k];
        }
        return r;
    }
    std::vector<double> w(4 * kNe);
    std::vector<int> perm(4 * kNe);
    for (int i = 0; i < 4 * kNe; ++i) perm[(size_t)i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);
    for (int i = 0; i < 4 * kNe; ++i) w[(size_t)perm[(size_t)i]] = 1.0 / std::pow(1.0 + i, 0.9);
    std::discrete_distribution<int> skew(w.begin(), w.end());
    const int hot = rank * kNe + (int)(rng() % kNe);
    for (int t = 0; t < M; ++t) {
        int* pk = &r.p[(size_t)t * kTop];
        int n = 0;
        if (!std::strcmp(kind, "hot") && rng() % 50) pk[n++] = hot;
        while (n < kTop) {
            const int e = !std::strcmp(kind, "skewed") ? skew(rng) : (int)(rng() % (4 * kNe));
            bool dup = false;
            for (int j = 0; j < n; ++j) dup |= pk[j] == e;
            if (!dup) pk[n++] = e;
        }
    }
    return r;
}
/* Tokens [t0, t1) as one pass on the rank, as moe_scatter writes it. */
struct P1Pass {
    std::vector<int32_t> sorted, off;
    int local = 0, live = 0, busiest = 0;
};
P1Pass p1_pass(const P1Routing& r, int t0, int t1) {
    std::vector<std::vector<int32_t>> by(kNe);
    for (int t = t0; t < t1; ++t)
        for (int k = 0; k < kTop; ++k) {
            const int e = r.p[(size_t)t * kTop + k];
            if (e / kNe == r.rank) by[(size_t)(e % kNe)].push_back((t - t0) * kTop + k);
        }
    P1Pass p;
    p.off.assign(kNe + 1, 0);
    for (int e = 0; e < kNe; ++e) {
        p.off[(size_t)e + 1] = p.off[(size_t)e] + (int)by[(size_t)e].size();
        for (int32_t f : by[(size_t)e]) p.sorted.push_back(f);
        p.live += !by[(size_t)e].empty();
        p.busiest = std::max(p.busiest, (int)by[(size_t)e].size());
    }
    p.local = (int)p.sorted.size();
    p.sorted.resize((size_t)(t1 - t0) * kTop, -1);
    return p;
}

bool p1_nan(uint16_t v) { return (v & 0x7f80u) == 0x7f80u && (v & 0x7fu); }
/* Values that differ, a NaN equal to any NaN (two kernels may carry different NaN payloads); the NaN pairs
 * counted apart. */
size_t p1_diff(const uint16_t* a, const uint16_t* b, size_t n, size_t* nans) {
    size_t d = 0;
    for (size_t i = 0; i < n; ++i)
        if (a[i] != b[i]) {
            if (p1_nan(a[i]) && p1_nan(b[i])) ++*nans;
            else ++d;
        }
    return d;
}

struct P1Ctx {
    const Lib* L = nullptr;
    P1Opts opts = nullptr;
    Launch p1 = nullptr;
    hipStream_t s{};
    P1Bank gu, dn;
    const void* noread = nullptr;
    size_t bad = 0, rows = 0, host_rows = 0, nan_pairs = 0;
    int cases = 0;
};

/* One moe_gemm_ggml launch over device buffers. */
int p1_launch(P1Ctx& c, Launch f, const int* split, void* x, int AR, int K, void* tab, int N, void* so, void* of,
              int T, void* y, bool sorted_order) {
    RadTensor t[5] = { tensor(x, RAD_BF16, { AR, K }), tensor(tab, RAD_U8, { kNe, N, K / 32 * 17 }),
                       tensor(so, RAD_I32, { T }), tensor(of, RAD_I32, { kNe + 1 }), tensor(y, RAD_BF16, { T, N }) };
    t[1].stride[0] = 0;
    RadParam p[6] = { ps("type", "mxfp4"), pi("N", N), pi("K", K), pi("n_expert", kNe), pi("top_k", kTop),
                      ps("a_order", sorted_order ? "sorted" : "token") };
    RadArgs a{};
    a.t = t; a.n_t = 5; a.p = p; a.n_p = 6;
    if (split) return c.opts(&a, (RadStream)c.s, split[0], split[1], split[2], split[3]);
    return f(&a, (RadStream)c.s);
}

/* The rows check of one (routing, M, forms) for both projections. */
void p1_rows(P1Ctx& c, const P1Routing& r, int M, std::mt19937_64& rng, bool splits) {
    ++c.cases;
    const P1Pass full = p1_pass(r, 0, M);
    const int T = M * kTop;
    /* Token rows for gate_up (NaN, inf and subnormal values on every 11th token), a row a slot for down. */
    std::vector<uint16_t> x((size_t)M * kEmbd), act((size_t)T * kFf);
    for (int t = 0; t < M; ++t) {
        const auto row = make_x(kEmbd, rng, t % 11 == 4);
        std::copy(row.begin(), row.end(), x.begin() + (size_t)t * kEmbd);
    }
    {
        std::normal_distribution<float> nd(0.f, 0.3f);
        for (auto& v : act) v = f32_to_bf16(nd(rng));
        static const uint16_t sp[] = { 0x7fc0, 0xff80, 0x7f80, 0x0001, 0x8001 };
        for (int i = 0; i < 64; ++i) act[(size_t)(rng() % act.size())] = sp[rng() % 5];
    }
    std::vector<int> forms(kNe);
    int nskip = 0, npk = 0;
    for (int e = 0; e < kNe; ++e) {
        const uint64_t q = rng() % 20;
        forms[(size_t)e] = q < 4 ? 2 : q < 10 ? 1 : 0;
        const bool live = full.off[(size_t)e + 1] > full.off[(size_t)e];
        nskip += live && forms[(size_t)e] == 2;
        npk += live && forms[(size_t)e] == 1;
    }
    std::vector<int> row_of((size_t)T, -1);
    for (int i = 0; i < full.local; ++i) row_of[(size_t)full.sorted[(size_t)i]] = i;
    std::vector<int> expert_of((size_t)std::max(1, full.local));
    for (int e = 0; e < kNe; ++e)
        for (int i = full.off[(size_t)e]; i < full.off[(size_t)e + 1]; ++i) expert_of[(size_t)i] = e;
    Dev d_so(full.sorted.size() * 4), d_of((kNe + 1) * 4), d_x(x.size() * 2);
    d_so.put(full.sorted.data());
    d_of.put(full.off.data());
    d_x.put(x.data());
    char label[96];
    std::snprintf(label, sizeof label, "M=%d %s rank %d", M, r.name.c_str(), r.rank);
    for (int proj = 0; proj < 2; ++proj) {
        const bool g = proj == 0;
        P1Bank& b = g ? c.gu : c.dn;
        const int N = b.N, K = b.K;
        std::vector<void*> tab = p1_table(b, forms, c.noread), blk = b.blk;
        Dev d_tab(kNe * sizeof(void*)), d_blk(kNe * sizeof(void*));
        d_tab.put(tab.data());
        d_blk.put(blk.data());
        std::unique_ptr<Dev> d_in;
        if (!g) {
            std::vector<uint16_t> in((size_t)T * K, 0);
            for (int i = 0; i < full.local; ++i)
                std::memcpy(&in[(size_t)i * K], &act[(size_t)full.sorted[(size_t)i] * K], (size_t)K * 2);
            d_in.reset(new Dev(in.size() * 2));
            d_in->put(in.data());
        }
        void* xin = g ? d_x.p : d_in->p;
        const int AR = g ? M : T;
        Dev d_y((size_t)T * N * 2);
        HC(hipMemset(d_y.p, 0xAA, d_y.n));
        if (p1_launch(c, c.p1, nullptr, xin, AR, K, d_tab.p, N, d_so.p, d_of.p, T, d_y.p, !g) != RAD_OK) {
            std::printf("FAIL %s %s: the P1 twin refused the call\n", label, g ? "gate_up" : "down");
            ++c.bad;
            return;
        }
        HC(hipStreamSynchronize(c.s));
        std::vector<uint16_t> y((size_t)T * N);
        d_y.get(y.data());
        /* The decode band: the plain gemv in passes of 16 over every expert in block_mxfp4. */
        size_t differ = 0, untouched = 0, tail = 0;
        for (int t0 = 0; t0 < M; t0 += 16) {
            const int t1 = std::min(M, t0 + 16), n = t1 - t0;
            const P1Pass pp = p1_pass(r, t0, t1);
            Dev c_so(pp.sorted.size() * 4), c_of((kNe + 1) * 4), c_y((size_t)n * kTop * N * 2);
            c_so.put(pp.sorted.data());
            c_of.put(pp.off.data());
            std::unique_ptr<Dev> c_in;
            void* cin = (char*)d_x.p + (size_t)t0 * kEmbd * 2;
            if (!g) {
                std::vector<uint16_t> in((size_t)n * kTop * K, 0);
                for (int i = 0; i < pp.local; ++i)
                    std::memcpy(&in[(size_t)i * K], &act[(size_t)(pp.sorted[(size_t)i] + t0 * kTop) * K], (size_t)K * 2);
                c_in.reset(new Dev(in.size() * 2));
                c_in->put(in.data());
                cin = c_in->p;
            }
            if (p1_launch(c, c.L->moe, nullptr, cin, g ? n : n * kTop, K, d_blk.p, N, c_so.p, c_of.p, n * kTop, c_y.p,
                          !g) != RAD_OK) {
                std::printf("FAIL %s: a 16-token gemv launch was refused\n", label);
                ++c.bad;
                return;
            }
            HC(hipStreamSynchronize(c.s));
            std::vector<uint16_t> yc((size_t)n * kTop * N);
            c_y.get(yc.data());
            for (int i = 0; i < pp.local; ++i) {
                const int f = pp.sorted[(size_t)i] + t0 * kTop, ra = row_of[(size_t)f], e = expert_of[(size_t)ra];
                const uint16_t* got = &y[(size_t)ra * N];
                if (forms[(size_t)e] == 2) {
                    for (int q = 0; q < N; ++q) untouched += got[q] != 0xAAAAu;
                    continue;
                }
                const size_t d = p1_diff(got, &yc[(size_t)i * N], (size_t)N, &c.nan_pairs);
                if (d && differ < 4)
                    std::printf("  %s %s: token %d slot %d (expert %d, %s) differs in %zu of %d values from its "
                                "16-token pass\n", label, g ? "gate_up" : "down", f / kTop, f % kTop, e,
                                forms[(size_t)e] ? "packed" : "block", d, N);
                differ += d;
                ++c.rows;
            }
        }
        for (size_t i = (size_t)full.local * N; i < y.size(); ++i) tail += y[i] != 0;
        /* The host kernels on a few computed rows. */
        size_t hdiff = 0;
        for (int q = 0; q < 6 && full.local; ++q) {
            const int i = (int)(rng() % (uint64_t)full.local), e = expert_of[(size_t)i];
            if (forms[(size_t)e] == 2) continue;
            Mx W;
            W.N = N; W.K = K; W.blocks = b.host[(size_t)e].data();
            std::vector<uint16_t> hy((size_t)N);
            const int f = full.sorted[(size_t)i];
            gemv_avx2(W, g ? &x[(size_t)(f / kTop) * kEmbd] : &act[(size_t)f * K], hy.data(), 0, N);
            hdiff += p1_diff(hy.data(), &y[(size_t)i * N], (size_t)N, &c.nan_pairs);
            ++c.host_rows;
        }
        /* Other work splits: the same bytes. */
        size_t sdiff = 0;
        int nsplit = 0;
        if (splits) {
            const int sps[][4] = { { 64, 48, 1, 0 }, { 1024, 4096, 32, 0 }, { 0, 0, 0, 36 }, { 2000, 20, 4, 0 } };
            for (const auto& sp : sps) {
                if (sp[3] && !g) continue;
                HC(hipMemset(d_y.p, 0xAA, d_y.n));
                if (p1_launch(c, nullptr, sp, xin, AR, K, d_tab.p, N, d_so.p, d_of.p, T, d_y.p, !g) != RAD_OK) {
                    std::printf("FAIL %s: a forced split was refused\n", label);
                    ++c.bad;
                    continue;
                }
                HC(hipStreamSynchronize(c.s));
                std::vector<uint16_t> y2((size_t)T * N);
                d_y.get(y2.data());
                sdiff += p1_diff(y.data(), y2.data(), y.size(), &c.nan_pairs);
                ++nsplit;
            }
        }
        std::printf("p1 rows %-16s %-7s local %5d live %2d (skipped %2d, packed %2d) busiest %4d: %zu differ from "
                    "16-token passes, %zu from the host, %zu skipped values written, %zu tail values not zero, %d "
                    "splits %zu differ\n", label, g ? "gate_up" : "down", full.local, full.live, nskip, npk,
                    full.busiest, differ, hdiff, untouched, tail, nsplit, sdiff);
        std::fflush(stdout);
        c.bad += differ + hdiff + untouched + tail + sdiff;
    }
}

/* The same stateless SwiGLU launch on a row view; pitches stay in elements. Empty live waves
 * enqueue no SwiGLU, as the stager's optional covering-span path does. */
int p1_swiglu_slice(Launch f, hipStream_t s, void* gu, void* y, int rows, int n, int gp, int yp, int lo, int hi) {
    if (lo < 0 || hi < lo || hi > rows) return RAD_E_SHAPE;
    if (lo == hi) return RAD_OK;
    RadTensor t[2] = { tensor((uint16_t*)gu + (int64_t)lo * gp, RAD_BF16, { hi - lo, 2 * n }),
                       tensor((uint16_t*)y + (int64_t)lo * yp, RAD_BF16, { hi - lo, n }) };
    t[0].stride[0] = gp; t[1].stride[0] = yp;
    RadParam p[3] = { pi("M", hi - lo), pi("n", n), pf("limit", kLimit) };
    RadArgs a{};
    a.t = t; a.n_t = 2; a.p = p; a.n_p = 3;
    return f(&a, (RadStream)s);
}

/* Pitch, guard and exact exceptional-value checks, including NaN payloads. This is small beside
 * the full expert banks and checks every intermediate view, rather than only the final chain. */
void p1_swiglu_views(P1Ctx& c) {
    ++c.cases;
    if (!c.L->swiglu) { ++c.bad; return; }
    constexpr int rows = 19, n = 259, gp = 528, yp = 272, guard = 16;
    std::vector<uint16_t> g((size_t)rows * gp + 2 * guard, 0x5AA5),
                          initial((size_t)rows * yp + 2 * guard, 0xA55A), gold = initial;
    const uint16_t edge[] = { 0x0000, 0x8000, 0x0001, 0x007F, 0x3F80, 0x7F80, 0xFF80, 0x7F81, 0x7FC1, 0xFFFF };
    for (int r = 0; r < rows; ++r)
        for (int j = 0; j < 2 * n; ++j)
            g[(size_t)guard + (size_t)r * gp + j] = j < 20 ? edge[(r + j) % 10]
                : (uint16_t)(r * 997 + j * 97);
    Dev dg(g.size() * 2), dy(initial.size() * 2);
    dg.put(g.data()); dy.put(initial.data());
    auto run = [&](int lo, int hi) {
        return p1_swiglu_slice(c.L->swiglu, c.s, (uint16_t*)dg.p + guard,
                               (uint16_t*)dy.p + guard, rows, n, gp, yp, lo, hi);
    };
    int rc = run(0, rows);
    HC(hipStreamSynchronize(c.s));
    dy.get(gold.data());
    dy.put(initial.data());
    std::vector<uint16_t> expected = initial, got = initial;
    size_t bad = 0;
    const int spans[][2] = { { 3, 6 }, { 12, 19 }, { 0, 3 }, { 6, 12 }, { 9, 9 }, { 2, 14 } };
    for (const auto& span : spans) {
        rc |= run(span[0], span[1]);
        HC(hipStreamSynchronize(c.s));
        for (int r = span[0]; r < span[1]; ++r)
            std::copy_n(gold.data() + guard + (size_t)r * yp, n,
                         expected.data() + guard + (size_t)r * yp);
        dy.get(got.data());
        bad += count_diff(got.data(), expected.data(), got.size());
    }
    std::vector<uint16_t> g_after(g.size());
    dg.get(g_after.data());
    bad += count_diff(g.data(), g_after.data(), g.size());
    std::printf("p1 SwiGLU views: pitched rows, gaps, empty/last/overlapping spans, NaN payloads and guards: "
                "%zu values differ%s\n", bad, rc ? " (a launch was refused)" : "");
    c.bad += bad + (rc ? 1 : 0);
}

/* The stager's full and covering-span wave composition at ds41's shapes against one launch over
 * every expert. Most live units are streamed, so this also covers a small resident set. */
void p1_waves(P1Ctx& c, const P1Routing& r, int M, std::mt19937_64& rng, bool small_cache = false) {
    ++c.cases;
    if (!c.L->swiglu) { std::printf("FAIL: the plugin has no r4d_rad_swiglu_clamped\n"); ++c.bad; return; }
    const P1Pass p = p1_pass(r, 0, M);
    const int T = M * kTop;
    std::vector<uint16_t> x((size_t)M * kEmbd);
    for (int t = 0; t < M; ++t) {
        const auto row = make_x(kEmbd, rng, false);
        std::copy(row.begin(), row.end(), x.begin() + (size_t)t * kEmbd);
    }
    Dev d_so(p.sorted.size() * 4), d_of((kNe + 1) * 4), d_x(x.size() * 2);
    d_so.put(p.sorted.data());
    d_of.put(p.off.data());
    d_x.put(x.data());
    std::vector<int> full_forms(kNe);
    for (int e = 0; e < kNe; ++e) full_forms[(size_t)e] = rng() % 3 == 0 ? 1 : 0;
    std::vector<int> sub;
    for (int e = 0; e < kNe; ++e) {
        const bool live = p.off[(size_t)e + 1] > p.off[(size_t)e];
        if (small_cache ? e != 0 : (live && rng() % 4 != 0) || (!live && rng() % 8 == 0)) sub.push_back(e);
    }
    std::shuffle(sub.begin(), sub.end(), rng);
    Dev G((size_t)T * 2 * kFf * 2), S((size_t)T * kFf * 2), D((size_t)T * kEmbd * 2);
    Dev G2(G.n), S2(S.n), D2(D.n), tg(kNe * sizeof(void*)), td(kNe * sizeof(void*));
    for (Dev* d : { &G, &S, &D, &G2, &S2, &D2 }) HC(hipMemset(d->p, 0xAA, d->n));
    auto chain = [&](const std::vector<int>& forms, Dev& g, Dev& sw, Dev& dn, int lo, int hi) {
        std::vector<void*> a = p1_table(c.gu, forms, c.noread), b = p1_table(c.dn, forms, c.noread);
        tg.put(a.data());
        td.put(b.data());
        int rc = p1_launch(c, c.p1, nullptr, d_x.p, M, kEmbd, tg.p, 2 * kFf, d_so.p, d_of.p, T, g.p, false);
        rc |= p1_swiglu_slice(c.L->swiglu, c.s, g.p, sw.p, T, kFf, 2 * kFf, kFf, lo, hi);
        rc |= p1_launch(c, c.p1, nullptr, sw.p, T, kFf, td.p, kEmbd, d_so.p, d_of.p, T, dn.p, true);
        HC(hipStreamSynchronize(c.s));   /* the tables are rewritten next */
        return rc;
    };
    const int ref_rc = chain(full_forms, G, S, D, 0, T);
    std::vector<int> main_forms = full_forms;
    for (int e : sub) main_forms[(size_t)e] = 2;
    for (bool spans : { false, true }) {
        if (spans) ++c.cases;
        for (Dev* d : { &G2, &S2, &D2 }) HC(hipMemset(d->p, 0xAA, d->n));
        int rc = ref_rc | chain(main_forms, G2, S2, D2, 0, T);
        int waves = 0;
        for (size_t w0 = 0, sz = 1; w0 < sub.size(); w0 += sz, sz = sz % 3 + 1, ++waves) {
            std::vector<int> wf(kNe, 2);
            int lo = T, hi = 0;
            for (size_t j = w0; j < std::min(sub.size(), w0 + sz); ++j) {
                const int e = sub[j];
                wf[(size_t)e] = full_forms[(size_t)e];
                if (p.off[(size_t)e + 1] > p.off[(size_t)e]) {
                    lo = std::min(lo, p.off[(size_t)e]);
                    hi = std::max(hi, p.off[(size_t)e + 1]);
                }
            }
            if (hi <= lo) lo = hi = 0;
            rc |= chain(wf, G2, S2, D2, spans ? lo : 0, spans ? hi : T);
        }
        size_t d[3] = { 0, 0, 0 };
        const Dev* a[3] = { &G, &S, &D };
        const Dev* b[3] = { &G2, &S2, &D2 };
        for (int i = 0; i < 3; ++i) {
            std::vector<uint16_t> ha(a[i]->n / 2), hb(b[i]->n / 2);
            a[i]->get(ha.data());
            b[i]->get(hb.data());
            d[i] = count_diff(ha.data(), hb.data(), ha.size());
        }
        std::printf("p1 waves M=%d %s rank %d, SwiGLU %s: %d local slots, %zu experts in %d waves of 1-3: "
                    "%zu gate_up, %zu swiglu, %zu down values differ from one launch%s\n", M, r.name.c_str(), r.rank,
                    spans ? "spans" : "full", p.local, sub.size(), waves,
                    d[0], d[1], d[2], rc ? " (a launch was refused)" : "");
        std::fflush(stdout);
        c.bad += d[0] + d[1] + d[2] + (rc ? 1 : 0);
    }
}

/* hipEvent time of `launch`, median of `reps` after a warm-up (us), or -1 when refused. */
double p1_time(P1Ctx& c, const std::function<int()>& launch, int reps) {
    if (launch() != RAD_OK) return -1;
    std::vector<double> t;
    hipEvent_t a, b;
    HC(hipEventCreate(&a));
    HC(hipEventCreate(&b));
    for (int i = 0; i < reps; ++i) {
        HC(hipEventRecord(a, c.s));
        if (launch() != RAD_OK) return -1;
        HC(hipEventRecord(b, c.s));
        HC(hipEventSynchronize(b));
        float ms = 0;
        HC(hipEventElapsedTime(&ms, a, b));
        t.push_back(ms * 1000.0);
    }
    (void)hipEventDestroy(a);
    (void)hipEventDestroy(b);
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

/* gate_up + down of one pass, timed: `kind` 0 the P1 twin (forms: 0 block, 1 packed, 2 a mix with a sixth
 * skipped), 1 the served LCP twin, 2 that twin cut to the local slots, 3 the balanced LCP twin. Returns us (gate_up
 * in *gu_us), -1 refused. */
double p1_pass_time(P1Ctx& c, const P1Pass& p, int M, int kind, int form, const int* split, double* gu_us, int reps) {
    const int T = kind == 2 ? std::max(1, p.local) : M * kTop;
    std::vector<int32_t> so(p.sorted.begin(), p.sorted.begin() + T);
    Dev d_so((size_t)T * 4), d_of((kNe + 1) * 4), d_x((size_t)M * kEmbd * 2), d_a((size_t)T * kFf * 2);
    Dev g((size_t)T * 2 * kFf * 2), dn((size_t)T * kEmbd * 2), tg(kNe * sizeof(void*)), td(kNe * sizeof(void*));
    d_so.put(so.data());
    d_of.put(p.off.data());
    HC(hipMemset(d_x.p, 0x3c, d_x.n));   /* finite activations: timing only */
    HC(hipMemset(d_a.p, 0x3c, d_a.n));
    std::vector<int> forms(kNe, kind == 0 ? (form == 1 ? 1 : 0) : 0);
    if (kind == 0 && form == 2)
        for (int e = 0; e < kNe; ++e) forms[(size_t)e] = e % 6 == 5 ? 2 : e % 3 == 1 ? 1 : 0;
    std::vector<void*> a = p1_table(c.gu, forms, c.noread), b = p1_table(c.dn, forms, c.noread);
    tg.put(a.data());
    td.put(b.data());
    auto run = [&](bool gate) -> int {
        void* x = gate ? d_x.p : d_a.p;
        const int AR = gate ? M : T, N = gate ? 2 * kFf : kEmbd, K = gate ? kEmbd : kFf;
        void* tab = gate ? tg.p : td.p;
        void* y = gate ? g.p : dn.p;
        if (kind == 0) return p1_launch(c, c.p1, split, x, AR, K, tab, N, d_so.p, d_of.p, T, y, !gate);
        if (kind == 3) {
            RadTensor t[5] = { tensor(x, RAD_BF16, { AR, K }), tensor(tab, RAD_U8, { kNe, N, K / 32 * 17 }),
                               tensor(d_so.p, RAD_I32, { T }), tensor(d_of.p, RAD_I32, { kNe + 1 }),
                               tensor(y, RAD_BF16, { T, N }) };
            t[1].stride[0] = 0;
            RadParam pr[6] = { ps("type", "mxfp4"), pi("N", N), pi("K", K), pi("n_expert", kNe), pi("top_k", kTop),
                               ps("a_order", gate ? "token" : "sorted") };
            RadArgs ra{};
            ra.t = t; ra.n_t = 5; ra.p = pr; ra.n_p = 6;
            return c.L->moe_bal ? c.L->moe_bal(&ra, (RadStream)c.s, 1, 1) : RAD_E_UNSUPPORTED;
        }
        return p1_launch(c, c.L->moe_tall, nullptr, x, AR, K, tab, N, d_so.p, d_of.p, T, y, !gate);
    };
    const double tg_us = p1_time(c, [&] { return run(true); }, reps);
    const double td_us = p1_time(c, [&] { return run(false); }, reps);
    if (tg_us < 0 || td_us < 0) return -1;
    *gu_us = tg_us;
    return tg_us + td_us;
}

int p1_check(const Lib& L, int mmax, const char* routes_path) {
    P1Ctx c;
    c.L = &L;
    int rc = 0;
    /* The row table: the P1 twin in place of the LCP twin, reaching MMAX, one priority below the gemv. */
    {
        const RadKernelInfo *gemv = nullptr, *twin = nullptr, *lcp_twin = nullptr;
        for (int i = 0; L.count && i < L.count(); ++i) {
            const RadKernelInfo* k = L.at(i);
            if (!k || !k->name) continue;
            if (!std::strcmp(k->name, ds41p1::kGemvRow)) gemv = k;
            if (!std::strcmp(k->name, ds41p1::kTwinRow)) twin = k;
            if (!std::strcmp(k->name, ds41p1::kLcpTwinRow)) lcp_twin = k;
        }
        long long reach = 0;
        int m_le = 0;
        for (int i = 0; twin && i < twin->n_constraints; ++i)
            if (!std::strcmp(twin->constraints[i].key, "M") && twin->constraints[i].op == RAD_C_LE) {
                ++m_le;
                reach = twin->constraints[i].ival;
            }
        const bool ok = gemv && twin && !lcp_twin && L.moe_p1 && twin->launch == L.moe_p1 && m_le == 1 &&
                        reach >= mmax && reach <= ds41p1::kTwinMaxRows && twin->priority == gemv->priority - 1 &&
                        twin->n_constraints == gemv->n_constraints && twin->layout == gemv->layout;
        std::printf("%s: row table: %s %s (reach %lld, priority %d, the gemv's %d), %s %s\n", ok ? "PASS" : "FAIL",
                    ds41p1::kTwinRow, twin ? "present" : "missing", reach, twin ? twin->priority : -1,
                    gemv ? gemv->priority : -1, ds41p1::kLcpTwinRow, lcp_twin ? "present (must not be)" : "absent");
        rc |= ok ? 0 : 1;
        if (!L.moe_p1 || !L.moe_p1_opts) {
            std::printf("p1: FAIL: the plugin lacks r4d_rad_moe_gemm_mxfp4_gemv_tall_p1 or its _opts export\n");
            return 1;
        }
    }
    c.p1 = L.moe_p1;
    c.opts = L.moe_p1_opts;
    HC(hipStreamCreate(&c.s));
    p1_swiglu_views(c);
    /* The weight under every skip mark: an address in the never-mapped first 64 KiB, so a read of it faults (T3:
     * the pointer under the mark need not be readable). */
    c.noread = (const void*)(uintptr_t)0x1000;
    if (routes_path) {
        FILE* f = std::fopen(routes_path, "rb");
        int32_t h[3];
        if (f && std::fread(h, 4, 3, f) == 3 && h[2] == kTop && h[0] > 0 && h[1] > 0) {
            g_p1_layers = h[0];
            g_p1_rows = h[1];
            g_p1_routes.resize((size_t)h[0] * h[1] * kTop);
            if (std::fread(g_p1_routes.data(), 2, g_p1_routes.size(), f) != g_p1_routes.size()) g_p1_routes.clear();
        }
        if (f) std::fclose(f);
        if (g_p1_routes.empty()) std::printf("note: cannot read the routes %s; synthetic routings only\n", routes_path);
    }
    const bool real = !g_p1_routes.empty();
    const auto t0 = std::chrono::steady_clock::now();
    std::mt19937_64 rng(1717);
    p1_bank(&c.gu, 2 * kFf, kEmbd, rng);
    p1_bank(&c.dn, kEmbd, kFf, rng);
    std::printf("p1: 96 experts of gate_up [%d, %d] and down [%d, %d] in VRAM, block_mxfp4 and packed (%d and %d raw "
                "rounds), made in %.1f s\n", 2 * kFf, kEmbd, kEmbd, kFf, c.gu.raw_rounds, c.dn.raw_rounds,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    std::fflush(stdout);
    std::vector<int> Ms = { 17, 33, 48, 64, 100, 256, 1024, 2048 };
    if (mmax > 2048) Ms.push_back(mmax);
    for (int M : Ms) {
        if (M > mmax) continue;
        if (real) {
            p1_rows(c, p1_routing("real", M, 0, 3 + M % 30, rng), M, rng, M == 256 || M == 1024);
            p1_rows(c, p1_routing("real", M, 2, 25, rng), M, rng, false);
        }
        p1_rows(c, p1_routing("skewed", M, 1, 0, rng), M, rng, !real && (M == 256 || M == 1024));
        if (M <= 256) {
            p1_rows(c, p1_routing("uniform", M, 3, 0, rng), M, rng, false);
            p1_rows(c, p1_routing("hot", M, 0, 0, rng), M, rng, M == 64);
        }
    }
    for (int M : { 64, 256, 2048 }) {
        if (M > mmax) continue;
        p1_waves(c, real ? p1_routing("real", M, 0, 12, rng) : p1_routing("skewed", M, 0, 0, rng), M, rng);
        if (M == 256) p1_waves(c, p1_routing("hot", M, 3, 0, rng), M, rng);
    }
    /* An explicit one-resident case: expert 0's ready rows lie before two distant streamed runs,
     * the last local expert included. The second routing has only other ranks' slots, so every
     * streamed wave is empty locally and the original tail zeros must survive unchanged. */
    for (bool no_local : { false, true }) {
        P1Routing edge;
        edge.rank = 0;
        edge.name = no_local ? "empty-local small-cache" : "last-expert small-cache";
        edge.p.resize(17 * kTop);
        const int ids[6] = { 0, 47, 95, 96, 192, 288 };
        for (int t = 0; t < 17; ++t)
            for (int k = 0; k < kTop; ++k) edge.p[(size_t)t * kTop + k] = no_local ? 96 + k : ids[k];
        p1_waves(c, edge, 17, rng, true);
    }
    std::printf("p1: %d cases, %zu (token, slot) rows against 16-token passes, %zu against the host kernels, %zu NaN "
                "pairs (any NaN equal), %zu bad values, %.0f s\n", c.cases, c.rows, c.host_rows, c.nan_pairs, c.bad,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    std::fflush(stdout);
    /* Timing. */
    std::printf("p1 time: gate_up + down a pass (hipEvent, median), us a local expert-row and a token row-layer\n");
    const char* fname[3] = { "block", "packed", "mix 1/6 skip" };
    const char* kname[4] = { "P1 twin", "LCP twin (served)", "LCP twin cut", "LCP twin balanced" };
    for (int M : { 64, 256, 1024, 2048 }) {
        if (M > mmax) continue;
        for (int lr = 0; lr < (real ? 3 : 1); ++lr) {
            const int layer = lr == 0 ? 12 : lr == 1 ? 25 : 36, rank = lr == 1 ? 2 : 0;
            const P1Routing r = real ? p1_routing("real", M, rank, layer, rng) : p1_routing("skewed", M, rank, 0, rng);
            const P1Pass p = p1_pass(r, 0, M);
            for (int kind = 0; kind < 4; ++kind)
                for (int form = 0; form < (kind == 0 ? 3 : 1); ++form) {
                    /* The served twin costs seconds a call at these heights: one layer, M <= 1024, one rep. */
                    if (kind == 1 && (M > 1024 || lr)) continue;
                    if ((kind == 2 || kind == 3) && lr) continue;
                    double gu = 0;
                    const double us = p1_pass_time(c, p, M, kind, form, nullptr, &gu, kind == 0 ? 5 : 1);
                    const int counted = kind == 0 && form == 2 ? p.local - [&] {
                        int sk = 0;
                        for (int e = 5; e < kNe; e += 6) sk += p.off[(size_t)e + 1] - p.off[(size_t)e];
                        return sk;
                    }() : p.local;
                    std::printf("p1 time: M %4d %-8s rank %d %-18s %-12s local %5d live %2d busiest %4d: gate_up %9.1f "
                                "down %9.1f us = %7.2f us an expert-row, %8.2f us a row-L\n", M, r.name.c_str(), rank,
                                kname[kind], kind == 0 ? fname[form] : "block", counted, p.live, p.busiest, gu, us - gu,
                                counted > 0 ? us / counted : 0.0, us / M);
                    std::fflush(stdout);
                }
        }
    }
    /* The work split (grid, item size, items a block, staged rows; 0 the default) at M 256, 1024 and 2048. */
    for (int M : { 256, 1024, 2048 }) {
        if (M > mmax) continue;
        const P1Routing r = real ? p1_routing("real", M, 0, 12, rng) : p1_routing("skewed", M, 0, 0, rng);
        const P1Pass p = p1_pass(r, 0, M);
        const int sps[][4] = { { 0, 0, 0, 0 },    { 256, 0, 0, 36 }, { 256, 0, 0, 0 },  { 1024, 0, 0, 0 },
                               { 2048, 0, 0, 0 }, { 0, 0, 0, 36 },   { 0, 96, 0, 0 },   { 0, 768, 0, 0 },
                               { 0, 0, 4, 0 },    { 0, 0, 16, 0 },   { 1024, 96, 0, 0 } };
        for (const auto& sp : sps) {
            double gu = 0;
            const double us = p1_pass_time(c, p, M, 0, 0, sp, &gu, 5);
            std::printf("p1 time split: M %d %s rank 0 grid %4d item %7d KiB fill %2d lds %2d KiB: gate_up %8.1f down "
                        "%8.1f us = %6.2f us an expert-row\n", M, r.name.c_str(), sp[0], sp[1], sp[2], sp[3], gu,
                        us - gu, p.local ? us / p.local : 0.0);
            std::fflush(stdout);
        }
    }
    (void)hipStreamDestroy(c.s);
    rc |= c.bad ? 1 : 0;
    if (rc)
        std::printf("p1: FAIL: %zu bad values in %d cases (rows, splits, waves; see FAIL and the counts above)%s\n",
                    c.bad, c.cases, c.bad ? "" : ", or the row table");
    else
        std::printf("p1: PASS: %s at M 17..%d: %zu rows equal to 16-token passes, skipped rows untouched, tail zeros, "
                    "splits and full/span SwiGLU waves of 1-3 equal to one launch (%d cases)\n", ds41p1::kTwinRow, Ms.back(), c.rows,
                    c.cases);
    return rc;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s LIBR4D.so check [trials] [table] | swiglu TABLE | time [LIB2.so] | ptime [T[:P]...] | draft [trials] | tall [T...] | lcp [C] [trials] | lsparse [C] [W:r...] | bal [C] [ROUTES] | p1 [MMAX] [ROUTES] | sweep [trials] | sweep0 [trials]\n", argv[0]); return 2; }
    /* `tall`: the switch before the plugin's row table and its sparse launchers first read it, and
     * the routed GEMM as the A/B's servers run it (no live tail); the other modes as the loop's
     * measured runs ran it. */
    /* `lcp [C]`: RADIANCE_EXPERIMENTAL_DS41_LCP=C and EXACT_TALL unset, as an Lc arm's server has them;
     * `sweep` keeps an LCP already in the environment (its heights) instead of setting EXACT_TALL. */
    /* Every mode but `p1` runs without prefill Phase 1's switch (its twin replaces the LCP twin `lcp` checks). */
    if (std::strcmp(argv[2], ds41p1::kCheckMode)) unsetenv(ds41p1::kEnv);
    if (!std::strcmp(argv[2], ds41p1::kCheckMode)) {
        /* `p1 [MMAX] [ROUTES]`: LCP=2048 and the switch, as a P1 arm's server has them. */
        setenv("RADIANCE_EXPERIMENTAL_DS41_LCP", "2048", 1);
        setenv(ds41p1::kEnv, "1", 1);
        unsetenv("RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL");
        unsetenv("RADIANCE_EXPERIMENTAL_DS41_LIVE_TAIL");
        unsetenv("RADIANCE_EXPERIMENTAL_DS41_MOE_BALANCE");
        g_lcp = 2048;
    } else if (!std::strcmp(argv[2], "lsparse")) {
        /* `lsparse [C]`: LCP=C (2048) and EXACT_TALL unset, as an Lc arm's server has them. */
        const int c = argc > 3 ? std::atoi(argv[3]) : 2048;
        if (c < 64 || c > 4096 || c % 16) { std::fprintf(stderr, "lsparse C: a multiple of 16 in 64 .. 4096\n"); return 2; }
        setenv("RADIANCE_EXPERIMENTAL_DS41_LCP", std::to_string(c).c_str(), 1);
        unsetenv("RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL");
        g_lcp = c;
    } else if (!std::strcmp(argv[2], "lcp") || !std::strcmp(argv[2], "bal")) {
        const int c = argc > 3 ? std::atoi(argv[3]) : 1024;
        if (c < 64 || c > 4096 || c % 16) { std::fprintf(stderr, "lcp C: a multiple of 16 in 64 .. 4096\n"); return 2; }
        setenv("RADIANCE_EXPERIMENTAL_DS41_LCP", std::to_string(c).c_str(), 1);
        unsetenv("RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL");
        g_lcp = c;
    } else if (!std::strcmp(argv[2], "sweep") && std::getenv("RADIANCE_EXPERIMENTAL_DS41_LCP") &&
               std::atoi(std::getenv("RADIANCE_EXPERIMENTAL_DS41_LCP")) >= 64) {
        g_lcp = std::atoi(std::getenv("RADIANCE_EXPERIMENTAL_DS41_LCP"));
        unsetenv("RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL");
    } else if (!std::strcmp(argv[2], "tall") || !std::strcmp(argv[2], "sweep"))
        setenv("RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL", "1", 1);
    else if (!std::strcmp(argv[2], "sweep0")) {
        unsetenv("RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL");
        unsetenv("RADIANCE_EXPERIMENTAL_DS41_LCP");
    }
    else setenv("RADIANCE_EXPERIMENTAL_DS41_LIVE_TAIL", "1", 1);
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
    if (!std::strcmp(argv[2], "lcp")) {
        std::vector<int> Ts = { 17, 48, 64, 100, 171, 256 };
        for (int T : { 512, 1024 })
            if (T <= g_lcp) Ts.push_back(T);
        if (g_lcp > 2048) {
            for (int T : { 2048, 3072, 4096 }) if (T <= g_lcp) Ts.push_back(T);
            if (std::find(Ts.begin(), Ts.end(), g_lcp) == Ts.end()) Ts.push_back(g_lcp);
            std::printf("lcp extended: every tall output against passes of 16; host expert oracle sampled "
                        "first/middle/last above 1024; dense/sparse sweep through %d\n", g_lcp);
        }
        int rc = tall_check(lib, Ts);
        rc |= sweep_check(lib, true, argc > 4 ? std::max(1, std::atoi(argv[4])) : 2);
        std::printf(rc ? "FAIL: lcp %d\n" : "PASS: lcp %d: tall passes and the height sweep keep 16-token passes' bits\n",
                    g_lcp);
        return rc;
    }
    if (!std::strcmp(argv[2], "bal")) return bal_check(lib, argc > 4 ? argv[4] : nullptr);
    if (!std::strcmp(argv[2], "lsparse")) {
        std::vector<std::pair<int, int>> ws;
        for (int i = 4; i < argc; ++i) {
            int w = 0, r = 0;
            if (std::sscanf(argv[i], "%d:%d", &w, &r) != 2 || w < 2 || (r != 1 && r != 2)) {
                std::fprintf(stderr, "lsparse W:r: a width of 2 or more at ratio 1 or 2, not %s\n", argv[i]);
                return 2;
            }
            ws.push_back({ w, r });
        }
        if (ws.empty()) ws = { { 32769, 1 }, { 65537, 2 }, { 131073, 1 }, { 262145, 2 }, { 524289, 1 } };
        return long_sparse_check(lib, ws);
    }
    if (!std::strcmp(argv[2], ds41p1::kCheckMode)) {
        const int mmax = argc > 3 ? std::atoi(argv[3]) : ds41p1::kTwinMaxRows;
        if (mmax < 17 || mmax > ds41p1::kTwinMaxRows) {
            std::fprintf(stderr, "p1 MMAX: 17 .. %d\n", ds41p1::kTwinMaxRows);
            return 2;
        }
        return p1_check(lib, mmax, argc > 4 ? argv[4] : nullptr);
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
