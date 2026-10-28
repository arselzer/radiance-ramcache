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
 *
 * Needs a free card (HIP device 0 unless HIP_VISIBLE_DEVICES says otherwise). Uses the
 * live-tail ranges as the measured runs do (RADIANCE_EXPERIMENTAL_DS41_LIVE_TAIL=1). */
#include <hip/hip_runtime.h>
#include <dlfcn.h>

#include <algorithm>
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
    explicit Lib(const char* path) {
        void* h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!h) { std::fprintf(stderr, "dlopen %s: %s\n", path, dlerror()); std::exit(2); }
        moe = (Launch)dlsym(h, "r4d_rad_moe_gemm_mxfp4_gemv");
        swiglu = (Launch)dlsym(h, "r4d_rad_swiglu_clamped");
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

/* moe_gemm_ggml type=mxfp4, gemv path, over a table of per-expert device pointers. */
int run_moe(Launch f, hipStream_t s, void* x, int AR, void* table, int N, int K, void* sorted,
            void* off, void* y, const char* order) {
    RadTensor t[5] = { tensor(x, RAD_BF16, { AR, K }),
                       tensor(table, RAD_U8, { kNe, N, K / 32 * 17 }),
                       tensor(sorted, RAD_I32, { kTop }), tensor(off, RAD_I32, { kNe + 1 }),
                       tensor(y, RAD_BF16, { kTop, N }) };
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
    size_t bad = 0, pairs = 0;
    for (size_t base = 0; base < ups.size(); base += R) {
        const int rows = (int)std::min<size_t>(R, ups.size() - base);
        for (int r = 0; r < rows; ++r)
            for (int g = 0; g < n; ++g) { gu[(size_t)r * 2 * n + g] = (uint16_t)g; gu[(size_t)r * 2 * n + n + g] = ups[base + r]; }
        d_gu.put(gu.data());
        if (run_swiglu(lib.swiglu, s, d_gu.p, d_y.p, rows, n) < 0) { std::fprintf(stderr, "swiglu refused\n"); return 2; }
        HC(hipStreamSynchronize(s));
        d_y.get(y.data());
        for (int r = 0; r < rows; ++r) {
            swiglu(table.data(), gu.data() + (size_t)r * 2 * n, want.data(), n, kLimit);
            bad += count_diff(want.data(), y.data() + (size_t)r * n, (size_t)n);
            pairs += (size_t)n;
        }
    }
    std::printf("swiglu_clamped: %zu (gate, up) pairs, %zu differing; table written to %s\n", pairs, bad, table_path);
    std::printf(bad ? "FAIL\n" : "PASS: host swiglu with the captured table equals libr4d's for every bf16 input\n");
    return bad ? 1 : 0;
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

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s LIBR4D.so check [trials] [table] | swiglu TABLE | time [LIB2.so]\n", argv[0]); return 2; }
    setenv("RADIANCE_EXPERIMENTAL_DS41_LIVE_TAIL", "1", 1);
    const Lib lib(argv[1]);
    if (!std::strcmp(argv[2], "check")) return check(lib, argc > 3 ? std::atoi(argv[3]) : 24, argc > 4 ? argv[4] : "silu_table.bin");
    if (!std::strcmp(argv[2], "swiglu")) return swiglu_proof(lib, argc > 3 ? argv[3] : "silu_table.bin");
    if (!std::strcmp(argv[2], "time")) {
        if (argc > 3) { const Lib b(argv[3]); return timing(lib, &b); }
        return timing(lib, nullptr);
    }
    return 2;
}
