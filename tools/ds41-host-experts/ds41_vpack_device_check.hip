/* ds41_vpack_device_check.hip -- RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED on a card: libr4d's
 * moe_gemm_ggml_mxfp4_gemv_vp over the packed-scale form against moe_gemm_ggml_mxfp4_gemv over block_mxfp4,
 * byte for byte, from a built libr4d.so (core/place/vram_packed.h; the host emulator's twin is
 * tests/r4d_emu/test_ds41vp.cpp).
 *
 *   ds41_vpack_device_check LIBR4D.so check [trials] [tail]
 *                                                      the decode chain's two GEMVs at the served shapes,
 *                                                      gate_up [4608, 5120] over token rows and down
 *                                                      [5120, 2304] over sorted rows, 1..3 live experts and
 *                                                      1, 2, 4, 16 tokens; each expert packed exactly as the
 *                                                      engine packs it (512 / 320 raw rounds of room) and
 *                                                      marked, beside block entries and a stray entry for the
 *                                                      experts without rows (an alias); scales as the release
 *                                                      has them, raw rounds anywhere, e8m0 edges, and the
 *                                                      research sample's real scales when its directory is
 *                                                      there (DS41_HX_SAMPLE or the default). Every value of
 *                                                      every row must be the plain row's bits; a damaged
 *                                                      header must give NaN rows. Exit 0 on PASS. `tail`:
 *                                                      both rows with the live-tail ranges on
 *                                                      (RADIANCE_EXPERIMENTAL_DS41_LIVE_TAIL=1, otherwise
 *                                                      off, as the A/B runs), every call at their geometry,
 *                                                      and the dropped slots' rows must be zeros in both.
 *   ds41_vpack_device_check LIBR4D.so time              decode GEMV time a call, plain over block_mxfp4,
 *                                                      gemv_vp over packed and gemv_vp over block_mxfp4 (a
 *                                                      table of unmarked entries: what the new kernel costs
 *                                                      the experts it does not pack), 1..3 live experts, 24
 *                                                      copies (past the 64 MB on-die cache); the outputs
 *                                                      compared too.
 *
 * When the library exports r4d_rad_moe_gemm_mxfp4_gemv_vp_variant, every gemv_vp kernel it has (the default
 * and r4d_rad_moe_gemm_mxfp4_gemv_vp_variants' names) is checked against the plain row in `check`, and timed in
 * `time`: each call alone (a host round trip around it, as above) and in bursts of 24 calls back to back on the
 * stream, every burst of every mode enqueued one after another with no host gap (their routings uploaded
 * first), so the card stays busy and at its clocks as in a decode step; the first round of bursts warms up and
 * is not counted. Two rows tell where the time goes: live=0 (no expert has a row: the launch, the live scan and
 * nothing else) and `hot` (live=1, every call on the same weight copy, so its bytes come from the on-die cache).
 *
 * Needs a free card (HIP device 0 unless HIP_VISIBLE_DEVICES says otherwise): run it in a GPU window, as
 * ab_window.sh runs ds41_mxfp4_device_check. Build (as build_hx.sh builds the other harness):
 *   g++ -O2 -std=c++17 -mavx2 -ffp-contract=off -fno-fast-math -fPIC -I. -c ds41_host_experts.cpp -o hx.o
 *   hipcc -O2 -std=c++17 --offload-arch=gfx1201 -I. -I../../abi -c ds41_vpack_device_check.hip -o vp.o
 *   hipcc vp.o hx.o -o ds41_vpack_device_check -ldl */
#include <hip/hip_runtime.h>
#include <dirent.h>
#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "rad_abi.h"
#include "ds41_host_experts.h"

#define HC(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e_)); std::exit(2); } } while (0)

namespace {

constexpr int kEmbd = 5120, kFf = 2304, kNe = 96, kTop = 6;
constexpr int kRaw[2] = { 512, 320 };   /* the engine's room, core/place/vram_packed.h */
typedef int (*Launch)(const RadArgs*, RadStream);

typedef int (*LaunchVar)(const RadArgs*, RadStream, int);
typedef int (*Variants)(const char**, int, int*);

/* The gemv_vp kernels a library has: [0] the row itself (its default), then the variant export's 1.. (their
 * names from r4d_rad_moe_gemm_mxfp4_gemv_vp_variants). */
constexpr int kKernels = 16;
char kKernelName[kKernels][48] = { "gemv_vp" };

struct Lib {
    Launch plain = nullptr, vp = nullptr;
    LaunchVar vpv = nullptr;   /* r4d_rad_moe_gemm_mxfp4_gemv_vp_variant, when the library has it */
    int nk = 1;
    explicit Lib(const char* path) {
        void* h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!h) { std::fprintf(stderr, "dlopen %s: %s\n", path, dlerror()); std::exit(2); }
        plain = (Launch)dlsym(h, "r4d_rad_moe_gemm_mxfp4_gemv");
        vp = (Launch)dlsym(h, "r4d_rad_moe_gemm_mxfp4_gemv_vp");
        vpv = (LaunchVar)dlsym(h, "r4d_rad_moe_gemm_mxfp4_gemv_vp_variant");
        const Variants var = (Variants)dlsym(h, "r4d_rad_moe_gemm_mxfp4_gemv_vp_variants");
        if (!plain || !vp) {
            std::fprintf(stderr, "%s lacks r4d_rad_moe_gemm_mxfp4_gemv%s\n", path, plain ? "_vp" : "");
            std::exit(2);
        }
        if (vpv && var) {
            const char* names[kKernels] = {};
            int deflt = -1;
            nk = std::min(var(names, kKernels, &deflt), kKernels);
            for (int k = 1; k < nk; ++k) std::snprintf(kKernelName[k], sizeof kKernelName[k], "gemv_vp[%s]", names[k]);
            std::printf("%s: gemv_vp runs variant %d (%s) by default; %d kernels checked\n", path, deflt,
                        deflt > 0 && deflt < nk ? names[deflt] : "?", nk);
        } else {
            vpv = nullptr;
        }
    }
    int kernels() const { return nk; }
    /* kernel -1 the plain row, 0 gemv_vp, 1.. its variants. */
    int launch(int kernel, const RadArgs* a, RadStream s) const {
        if (kernel < 0) return plain(a, s);
        if (kernel == 0) return vp(a, s);
        return vpv(a, s, kernel);
    }
};

struct Dev {
    void* p = nullptr;
    size_t n = 0;
    explicit Dev(size_t bytes) : n(bytes) { HC(hipMalloc(&p, bytes)); }
    ~Dev() { (void)hipFree(p); }
    void put(const void* h) { HC(hipMemcpy(p, h, n, hipMemcpyHostToDevice)); }
    void get(void* h) const { HC(hipMemcpy(h, p, n, hipMemcpyDeviceToHost)); }
};

/* One weight as the release stores it, and both forms on the card. */
struct Weight {
    int N = 0, K = 0, n_exc = 0;
    std::vector<uint8_t> codes, scales, block, packed;
    std::unique_ptr<Dev> d_block, d_packed;
    void make_block() {
        const int UK = K / 32;
        block.assign((size_t)N * UK * 17, 0);
        for (int n = 0; n < N; ++n)
            for (int u = 0; u < UK; ++u) {
                uint8_t* b = block.data() + ((size_t)n * UK + u) * 17;
                const uint8_t* c = codes.data() + (size_t)n * (K / 2) + (size_t)u * 16;
                auto val = [&](int v) { return (c[v >> 1] >> (4 * (v & 1))) & 15; };
                b[0] = scales[(size_t)n * UK + u];
                for (int j = 0; j < 16; ++j) b[1 + j] = (uint8_t)(val(j) | (val(j + 16) << 4));
            }
    }
    /* As the engine packs a VRAM resident (vram_pack_weight): room for `cap`, or the weight's own raw
     * rounds when it needs more (the engine would keep that one block_mxfp4). */
    bool make_packed(int cap) {
        ds41hx::PackInfo plan;
        if (!ds41hx::pack_plan(scales.data(), N, K, &plan)) return false;
        n_exc = plan.n_exc;
        cap = std::max(cap, plan.n_exc);
        packed.assign(ds41hx::packed_bytes(N, K, cap), 0);
        if (packed.empty() ||
            ds41hx::pack_planes(codes.data(), scales.data(), N, K, cap, plan, packed.data(), packed.size()) !=
                ds41hx::kPackOk)
            return false;
        ds41hx::Mx v;
        return ds41hx::packed_view(packed.data(), packed.size(), N, K, &v) &&
               ds41hx::packed_scales_equal(v, scales.data());
    }
    void upload() {
        d_block.reset(new Dev(block.size()));
        d_block->put(block.data());
        d_packed.reset(new Dev(packed.size()));
        d_packed->put(packed.data());
    }
};

void fill(Weight& w, int N, int K, std::mt19937_64& rng, double raw, double edge) {
    w.N = N;
    w.K = K;
    const int UK = K / 32, R = (UK + 31) / 32;
    w.codes.resize((size_t)N * (K / 2));
    w.scales.resize((size_t)N * UK);
    for (auto& c : w.codes) c = (uint8_t)rng();
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    for (auto& s : w.scales) {
        const double r = u01(rng);
        s = r < 0.58 ? 120 : r < 0.995 ? 121 : r < 0.998 ? 119 : 122;
    }
    static const uint8_t edges[] = { 0, 1, 2, 3, 100, 127, 128, 135, 254, 255 };
    for (int n = 0; n < N; ++n)
        for (int r = 0; r < R; ++r) {
            if (u01(rng) >= raw) continue;
            const int t = r < UK / 32 ? 32 : UK % 32;
            w.scales[(size_t)n * UK + 32 * r + (size_t)(rng() % (uint64_t)t)] = edges[rng() % sizeof edges];
        }
    for (auto& s : w.scales)
        if (u01(rng) < edge) s = edges[rng() % sizeof edges];
}

/* The research sample's real scales (and codes where it has them), as test_ds41vp reads it. */
bool read_file(const std::string& path, std::vector<uint8_t>* out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    out->resize(n > 0 ? (size_t)n : 0);
    const bool ok = n >= 0 && std::fread(out->data(), 1, out->size(), f) == out->size();
    std::fclose(f);
    return ok;
}
int load_sample(std::vector<std::unique_ptr<Weight>>* gus, std::vector<std::unique_ptr<Weight>>* dns,
                std::mt19937_64& rng, int most) {
    const char* e = std::getenv("DS41_HX_SAMPLE");
    const std::string dir = e && *e ? e : "/home/as/.claude/jobs/635603c1/tmp/research/sample";
    std::vector<std::pair<int, int>> ids;
    if (DIR* d = opendir(dir.c_str())) {
        while (dirent* ent = readdir(d)) {
            int L = -1, E = -1;
            char tail[32] = {};
            if (std::sscanf(ent->d_name, "L%d_E%d_w1.%31s", &L, &E, tail) == 3 && !std::strcmp(tail, "scale"))
                ids.emplace_back(L, E);
        }
        closedir(d);
    }
    std::sort(ids.begin(), ids.end());
    int got = 0;
    for (const auto& id : ids) {
        if (got >= most) break;
        char stem[64];
        std::snprintf(stem, sizeof stem, "/L%02d_E%03d_w", id.first, id.second);
        std::vector<uint8_t> s1, s2, s3, c1, c2, c3;
        const std::string b = dir + stem;
        if (!read_file(b + "1.scale", &s1) || !read_file(b + "2.scale", &s2) || !read_file(b + "3.scale", &s3) ||
            s1.size() != (size_t)kFf * (kEmbd / 32) || s3.size() != s1.size() ||
            s2.size() != (size_t)kEmbd * (kFf / 32))
            continue;
        read_file(b + "1.codes", &c1);
        read_file(b + "2.codes", &c2);
        read_file(b + "3.codes", &c3);
        auto rows = [&](Weight& w, int r0, int n, const std::vector<uint8_t>& sc, const std::vector<uint8_t>& cs) {
            const int UK = w.K / 32, CB = w.K / 2;
            std::memcpy(w.scales.data() + (size_t)r0 * UK, sc.data(), (size_t)n * UK);
            for (size_t i = 0; i < (size_t)n * CB; ++i)
                w.codes[(size_t)r0 * CB + i] = i < cs.size() ? cs[i] : (uint8_t)rng();
        };
        gus->emplace_back(new Weight());
        Weight& g = *gus->back();
        g.N = 2 * kFf; g.K = kEmbd;
        g.codes.resize((size_t)g.N * g.K / 2);
        g.scales.resize((size_t)g.N * (g.K / 32));
        rows(g, 0, kFf, s1, c1);
        rows(g, kFf, kFf, s3, c3);
        dns->emplace_back(new Weight());
        Weight& dn = *dns->back();
        dn.N = kEmbd; dn.K = kFf;
        dn.codes.resize((size_t)dn.N * dn.K / 2);
        dn.scales.resize((size_t)dn.N * (dn.K / 32));
        rows(dn, 0, kEmbd, s2, c2);
        ++got;
    }
    return got;
}

uint16_t f2bf(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
bool nan16(uint16_t v) { return (v & 0x7f80u) == 0x7f80u && (v & 0x7fu); }

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

int run_moe(const Lib& lib, int kernel, hipStream_t s, void* x, int AR, void* table, int N, int K, void* sorted,
            void* off, void* y, const char* order, int T) {
    RadTensor t[5] = { tensor(x, RAD_BF16, { AR, K }), tensor(table, RAD_U8, { kNe, N, K / 32 * 17 }),
                       tensor(sorted, RAD_I32, { T }), tensor(off, RAD_I32, { kNe + 1 }),
                       tensor(y, RAD_BF16, { T, N }) };
    t[1].stride[0] = 0;   /* a table of expert pointers */
    RadParam p[6] = { ps("type", "mxfp4"), pi("N", N), pi("K", K), pi("n_expert", kNe), pi("top_k", kTop),
                      ps("a_order", order) };
    RadArgs a{};
    a.t = t; a.n_t = 5; a.p = p; a.n_p = 6;
    return lib.launch(kernel, &a, (RadStream)s);
}

/* `tokens` tokens' routes on this rank: each token's top-6 of the 384, those in 0..95 this rank's rows,
 * the rest dropped (-1 past the routed rows); `live` the experts that got rows. */
struct Route {
    int T = 0;
    std::vector<int32_t> sorted, off;
    std::vector<int> live;
};
Route route(std::mt19937_64& rng, int tokens, const std::vector<int>& pool) {
    std::vector<std::vector<int32_t>> rows(kNe);
    for (int t = 0; t < tokens; ++t) {
        std::vector<int> pick = { pool[rng() % pool.size()] };
        while ((int)pick.size() < kTop) {
            const int e = (int)(rng() % (4 * kNe));
            if (std::find(pick.begin(), pick.end(), e) == pick.end()) pick.push_back(e);
        }
        for (int k = 0; k < kTop; ++k)
            if (pick[(size_t)k] < kNe) rows[(size_t)pick[(size_t)k]].push_back(t * kTop + k);
    }
    Route r;
    r.off.assign(kNe + 1, 0);
    for (int e = 0; e < kNe; ++e) {
        r.off[(size_t)e + 1] = r.off[(size_t)e] + (int)rows[(size_t)e].size();
        for (int32_t f : rows[(size_t)e]) r.sorted.push_back(f);
        if (!rows[(size_t)e].empty()) r.live.push_back(e);
    }
    r.T = tokens * kTop;
    r.sorted.resize((size_t)r.T, -1);
    return r;
}

/* `tail`: the live-tail ranges are on (main set the switch before the library's first call). */
int check(const Lib& lib, int trials, bool tail) {
    hipStream_t s;
    HC(hipStreamCreate(&s));
    std::mt19937_64 rng(4141);
    /* The experts: the sample's real ones first, then synthetic ones with raw rounds and edges. */
    std::vector<std::unique_ptr<Weight>> gus, dns;
    const int real = load_sample(&gus, &dns, rng, 6);
    while (gus.size() < 9) {
        const size_t i = gus.size();
        gus.emplace_back(new Weight());
        fill(*gus.back(), 2 * kFf, kEmbd, rng, i % 3 == 0 ? 0.002 : 0.0002, i % 4 == 3 ? 0.0005 : 0.0);
        dns.emplace_back(new Weight());
        fill(*dns.back(), kEmbd, kFf, rng, i % 3 == 0 ? 0.004 : 0.0002, i % 4 == 3 ? 0.0005 : 0.0);
    }
    int raw_gu = 0, raw_dn = 0;
    for (size_t i = 0; i < gus.size(); ++i) {
        gus[i]->make_block();
        dns[i]->make_block();
        if (!gus[i]->make_packed(kRaw[0]) || !dns[i]->make_packed(kRaw[1])) {
            std::fprintf(stderr, "expert %zu does not pack\n", i);
            return 2;
        }
        raw_gu = std::max(raw_gu, gus[i]->n_exc);
        raw_dn = std::max(raw_dn, dns[i]->n_exc);
        gus[i]->upload();
        dns[i]->upload();
    }
    std::printf("%zu experts (%d with the research sample's scales); at most %d raw rounds in a gate_up, %d in a "
                "down\n", gus.size(), real, raw_gu, raw_dn);
    /* A stray buffer for the experts without rows, as the engine's aliases point at a file buffer. */
    const size_t stray_b = (size_t)2 * kFf * (kEmbd / 32) * 17;
    Dev stray(stray_b);
    {
        std::vector<uint8_t> junk(stray_b);
        for (auto& b : junk) b = (uint8_t)rng();
        stray.put(junk.data());
    }
    Dev tab_plain(kNe * sizeof(void*)), tab_vp(kNe * sizeof(void*));
    const int nk = lib.kernels();
    size_t values = 0, differ = 0, nan_ok = 0, nan_bad = 0, calls = 0, dropped = 0, dropped_bad = 0;
    size_t kdiffer[kKernels] = {}, kvalues[kKernels] = {};
    const int token_counts[] = { 1, 2, 4, 16 };
    std::vector<int> pool;
    for (int e = 0; e < kNe; ++e) pool.push_back(e);
    for (int trial = 0; trial < trials; ++trial) {
        const int tokens = token_counts[trial % 4];
        const bool damage = trial % 8 == 7;
        for (int which = 0; which < 2; ++which) {
            const bool gu = which == 0;
            const int N = gu ? 2 * kFf : kEmbd, K = gu ? kEmbd : kFf;
            const Route r = route(rng, tokens, pool);
            std::vector<void*> tp(kNe, stray.p), tv(kNe, stray.p);
            Dev* damaged = nullptr;
            int damaged_e = -1;
            for (size_t i = 0; i < r.live.size(); ++i) {
                const int e = r.live[i];
                Weight& w = *(gu ? gus : dns)[(size_t)(trial * 3 + (int)i) % gus.size()];
                tp[(size_t)e] = w.d_block->p;
                /* Every third live expert stays a block entry in the marked table too. */
                const bool pk = (trial + (int)i) % 3 != 2;
                tv[(size_t)e] = pk ? (void*)((uintptr_t)w.d_packed->p | 1u) : w.d_block->p;
                if (pk && damage && damaged_e < 0) {
                    std::vector<uint8_t> bad = w.packed;
                    bad[0] ^= 0x10;
                    damaged = new Dev(bad.size());
                    damaged->put(bad.data());
                    tv[(size_t)e] = (void*)((uintptr_t)damaged->p | 1u);
                    damaged_e = e;
                }
            }
            tab_plain.put(tp.data());
            tab_vp.put(tv.data());
            const int AR = gu ? tokens : r.T;
            std::vector<uint16_t> x((size_t)AR * K);
            std::normal_distribution<float> nd(0.f, gu ? 0.5f : 0.1f);
            for (auto& v : x) v = f2bf(nd(rng));
            if (trial % 5 == 4)
                for (auto& v : x)
                    if (rng() % 101 == 0) v = (uint16_t)(rng() % 2 ? 0x7f80 : 0x0001);
            Dev d_x(x.size() * 2), d_sorted((size_t)r.T * 4), d_off((kNe + 1) * 4), y1((size_t)r.T * N * 2),
                y2((size_t)r.T * N * 2);
            d_x.put(x.data());
            d_sorted.put(r.sorted.data());
            d_off.put(r.off.data());
            const char* order = gu ? "token" : "sorted";
            /* Rows no kernel writes stay 0xffff: under `tail` the dropped slots' zeros are the tail kernel's. */
            HC(hipMemsetAsync(y1.p, 0xff, y1.n, s));
            if (run_moe(lib, -1, s, d_x.p, AR, tab_plain.p, N, K, d_sorted.p, d_off.p, y1.p, order, r.T) < 0) {
                std::fprintf(stderr, "a launch refused its arguments\n");
                return 2;
            }
            std::vector<uint16_t> a((size_t)r.T * N), b((size_t)r.T * N);
            HC(hipStreamSynchronize(s));
            y1.get(a.data());
            for (int k = 0; k < nk; ++k) {
                HC(hipMemsetAsync(y2.p, 0xff, y2.n, s));
                if (run_moe(lib, k, s, d_x.p, AR, tab_vp.p, N, K, d_sorted.p, d_off.p, y2.p, order, r.T) < 0) {
                    std::fprintf(stderr, "%s refused its arguments\n", kKernelName[k]);
                    return 2;
                }
                HC(hipStreamSynchronize(s));
                ++calls;
                y2.get(b.data());
                for (int i = 0; i < r.T; ++i) {
                    if (r.sorted[(size_t)i] < 0) {
                        /* A dropped slot (another rank's expert): written by neither row without the live-tail
                         * ranges, so not compared; with them, zeroed by both (mx_tail_zero). */
                        if (tail)
                            for (int n = 0; n < N; ++n) {
                                ++dropped;
                                dropped_bad += a[(size_t)i * N + n] != 0 || b[(size_t)i * N + n] != 0;
                            }
                        continue;
                    }
                    const bool dmg =
                        damaged_e >= 0 && i >= r.off[(size_t)damaged_e] && i < r.off[(size_t)damaged_e + 1];
                    for (int n = 0; n < N; ++n) {
                        const uint16_t p = a[(size_t)i * N + n], q = b[(size_t)i * N + n];
                        if (dmg) { if (nan16(q)) ++nan_ok; else ++nan_bad; continue; }
                        ++values;
                        ++kvalues[k];
                        if (p != q && !(nan16(p) && nan16(q))) {
                            if (differ < 8)
                                std::printf("  differ: %s trial %d %s row %d col %d plain %04x gemv_vp %04x\n",
                                            kKernelName[k], trial, gu ? "gate_up" : "down", i, n, p, q);
                            ++differ;
                            ++kdiffer[k];
                        }
                    }
                }
            }
            delete damaged;
        }
    }
    for (int k = 0; k < nk; ++k)
        std::printf("%s: %zu values compared with the plain row, %zu differ\n", kKernelName[k], kvalues[k],
                    kdiffer[k]);
    std::printf("%zu calls, %zu values compared: %zu differ; damaged headers: %zu values NaN, %zu not\n", calls,
                values, differ, nan_ok, nan_bad);
    if (tail)
        std::printf("live-tail ranges on: %zu dropped slots' values, %zu not zero in both rows\n", dropped,
                    dropped_bad);
    const bool ok = !differ && !nan_bad && values > 0 && !dropped_bad && (!tail || dropped > 0);
    if (ok)
        std::printf("PASS: gemv_vp over packed experts is bit-identical to the plain row over block_mxfp4%s\n",
                    tail ? " (live-tail ranges on)" : "");
    else
        std::printf("FAIL\n");
    return ok ? 0 : 1;
}

/* Timed call `c`'s routing: `live` distinct experts of the 96 for one token, the rest of the top-6 dropped, and the
 * table naming them (packed entries when `pk`, block_mxfp4 otherwise; every other entry a block copy). Call c's
 * experts are copies c * live .. + live - 1 (mod the copies), so consecutive calls of any modes walk every copy
 * before one is read again: 300 MB of gate_up and 150 MB of down between two reads of a copy, past the 64 MB
 * on-die cache, whichever modes are interleaved. */
struct Call {
    std::vector<void*> table;
    std::vector<int32_t> off, sorted;
};
Call make_call(const std::vector<std::unique_ptr<Weight>>& ws, int live, int it, bool pk, int fixed = -1) {
    std::mt19937_64 rr(1000 + it);
    std::vector<int> ex;
    while ((int)ex.size() < live) {
        const int e = (int)(rr() % kNe);
        if (std::find(ex.begin(), ex.end(), e) == ex.end()) ex.push_back(e);
    }
    std::sort(ex.begin(), ex.end());
    Call c;
    c.off.assign(kNe + 1, 0);
    c.sorted.assign(kTop, -1);
    for (int e = 0; e < kNe; ++e)
        c.off[(size_t)e + 1] = c.off[(size_t)e] + (std::find(ex.begin(), ex.end(), e) != ex.end());
    for (int i = 0; i < live; ++i) c.sorted[(size_t)i] = i;
    c.table.assign(kNe, ws[0]->d_block->p);
    for (int i = 0; i < live; ++i) {
        Weight& w = *ws[(size_t)(fixed >= 0 ? (fixed + i) % (int)ws.size() : (it * live + i) % (int)ws.size())];
        c.table[(size_t)ex[(size_t)i]] = pk ? (void*)((uintptr_t)w.d_packed->p | 1u) : w.d_block->p;
    }
    return c;
}

int timing(const Lib& lib) {
    hipStream_t s;
    HC(hipStreamCreate(&s));
    std::mt19937_64 rng(77);
    constexpr int kCopies = 24;   /* 24 x 12 MB of gate_up: past the 64 MB on-die cache */
    constexpr int kBurst = 24;    /* calls back to back on the stream, each its own table */
    std::vector<std::unique_ptr<Weight>> gus, dns;
    for (int i = 0; i < kCopies; ++i) {
        gus.emplace_back(new Weight());
        fill(*gus.back(), 2 * kFf, kEmbd, rng, 0.0002, 0);
        dns.emplace_back(new Weight());
        fill(*dns.back(), kEmbd, kFf, rng, 0.0002, 0);
        gus.back()->make_block();
        dns.back()->make_block();
        if (!gus.back()->make_packed(kRaw[0]) || !dns.back()->make_packed(kRaw[1])) return 2;
        gus.back()->upload();
        dns.back()->upload();
    }
    Dev tab(kNe * sizeof(void*)), d_sorted(kTop * 4), d_off((kNe + 1) * 4), d_x(kEmbd * 2),
        d_a((size_t)kTop * kFf * 2), y1((size_t)kTop * kEmbd * 2), y2((size_t)kTop * kEmbd * 2);
    std::vector<uint16_t> x(kEmbd), act((size_t)kTop * kFf);
    for (auto& v : x) v = f2bf(0.5f * (float)((int)(rng() % 200) - 100) / 100.f);
    for (auto& v : act) v = f2bf(0.1f * (float)((int)(rng() % 200) - 100) / 100.f);
    d_x.put(x.data());
    d_a.put(act.data());
    hipEvent_t e0, e1;
    HC(hipEventCreate(&e0));
    HC(hipEventCreate(&e1));
    const int nk = lib.kernels();
    /* Modes: 0 the plain row over block_mxfp4; then for each gemv_vp kernel k, 1 + 2k over packed entries and
     * 2 + 2k over the same block_mxfp4 entries unmarked (the experts it does not pack: staged, aliased, over
     * budget). */
    const int nmodes = 1 + 2 * nk;
    auto mode_kernel = [&](int mode) { return mode == 0 ? -1 : (mode - 1) / 2; };
    auto mode_pk = [&](int mode) { return mode > 0 && (mode - 1) % 2 == 0; };
    auto mode_name = [&](int mode) {
        static char buf[64];
        if (mode == 0) return (const char*)"plain/block   ";
        std::snprintf(buf, sizeof buf, "%s/%s", kKernelName[mode_kernel(mode)], mode_pk(mode) ? "packed" : "block ");
        return (const char*)buf;
    };
    size_t differ = 0;
    int cc = 0;   /* the call counter: every call reads copies no call has read for the last 23 */
    /* live 0..3, then (live 4) the hot row: live=1 on copy 0 every call, back to back only. */
    for (int live = 0; live <= 4; ++live)
        for (int which = 0; which < 2; ++which) {
            const bool hot = live == 4;
            const int nlive = hot ? 1 : live;
            const bool gu = which == 0;
            const int N = gu ? 2 * kFf : kEmbd, K = gu ? kEmbd : kFf;
            const auto& ws = gu ? gus : dns;
            auto one = [&](int mode, int it, void* y) {
                const Call c = make_call(ws, nlive, it, mode_pk(mode));
                tab.put(c.table.data());
                d_sorted.put(c.sorted.data());
                d_off.put(c.off.data());
                HC(hipEventRecord(e0, s));
                if (run_moe(lib, mode_kernel(mode), s, gu ? d_x.p : d_a.p, gu ? 1 : kTop, tab.p, N, K, d_sorted.p,
                            d_off.p, y, gu ? "token" : "sorted", kTop) < 0)
                    std::exit(3);
                HC(hipEventRecord(e1, s));
                HC(hipEventSynchronize(e1));
                float ms = 0;
                HC(hipEventElapsedTime(&ms, e0, e1));
                return ms;
            };
            std::vector<double> iso((size_t)nmodes, 0.0), bur((size_t)nmodes);
            /* Alone: warm-up, then 60 calls a mode, the modes interleaved call by call so drift falls on all. */
            if (!hot) {
                std::vector<std::vector<float>> t((size_t)nmodes);
                for (int it = 0; it < 5; ++it)
                    for (int mode = 0; mode < nmodes; ++mode) (void)one(mode, cc++, y1.p);
                for (int it = 0; it < 60; ++it)
                    for (int mode = 0; mode < nmodes; ++mode) t[(size_t)mode].push_back(one(mode, cc++, y1.p));
                for (int mode = 0; mode < nmodes; ++mode) {
                    std::sort(t[(size_t)mode].begin(), t[(size_t)mode].end());
                    iso[(size_t)mode] = t[(size_t)mode][t[(size_t)mode].size() / 2];
                }
            }
            /* Back to back: 8 rounds of one burst a mode (round 0 warms up), every call's routing uploaded first,
             * then all bursts enqueued with an event pair around each and nothing between them; the median of a
             * mode's 7 counted bursts. */
            constexpr int kRounds = 8;
            const int nb = kRounds * nmodes;
            const size_t tsz = kNe * sizeof(void*), osz = (kNe + 1) * 4, ssz = kTop * 4;
            Dev btab((size_t)nb * kBurst * tsz), boff((size_t)nb * kBurst * osz), bsort((size_t)nb * kBurst * ssz);
            {
                std::vector<uint8_t> ht((size_t)nb * kBurst * tsz), ho((size_t)nb * kBurst * osz),
                    hs((size_t)nb * kBurst * ssz);
                for (int bi = 0; bi < nb; ++bi)
                    for (int i = 0; i < kBurst; ++i) {
                        const size_t at = (size_t)bi * kBurst + i;
                        const Call c = make_call(ws, nlive, cc + bi * kBurst + i, mode_pk(bi % nmodes), hot ? 0 : -1);
                        std::memcpy(ht.data() + at * tsz, c.table.data(), tsz);
                        std::memcpy(ho.data() + at * osz, c.off.data(), osz);
                        std::memcpy(hs.data() + at * ssz, c.sorted.data(), ssz);
                    }
                btab.put(ht.data());
                boff.put(ho.data());
                bsort.put(hs.data());
            }
            cc += nb * kBurst;
            std::vector<hipEvent_t> ev((size_t)2 * nb);
            for (auto& e : ev) HC(hipEventCreate(&e));
            HC(hipStreamSynchronize(s));
            for (int bi = 0; bi < nb; ++bi) {
                HC(hipEventRecord(ev[(size_t)2 * bi], s));
                for (int i = 0; i < kBurst; ++i) {
                    const size_t at = (size_t)bi * kBurst + i;
                    if (run_moe(lib, mode_kernel(bi % nmodes), s, gu ? d_x.p : d_a.p, gu ? 1 : kTop,
                                (uint8_t*)btab.p + at * tsz, N, K, (uint8_t*)bsort.p + at * ssz,
                                (uint8_t*)boff.p + at * osz, y1.p, gu ? "token" : "sorted", kTop) < 0)
                        std::exit(3);
                }
                HC(hipEventRecord(ev[(size_t)2 * bi + 1], s));
            }
            HC(hipEventSynchronize(ev.back()));
            std::vector<std::vector<float>> b((size_t)nmodes);
            for (int bi = nmodes; bi < nb; ++bi) {
                float ms = 0;
                HC(hipEventElapsedTime(&ms, ev[(size_t)2 * bi], ev[(size_t)2 * bi + 1]));
                b[(size_t)(bi % nmodes)].push_back(ms / kBurst);
            }
            for (auto& e : ev) HC(hipEventDestroy(e));
            for (int mode = 0; mode < nmodes; ++mode) {
                std::sort(b[(size_t)mode].begin(), b[(size_t)mode].end());
                bur[(size_t)mode] = b[(size_t)mode][b[(size_t)mode].size() / 2];
            }
            for (int mode = 0; mode < nmodes; ++mode) {
                const double bytes = (double)nlive * (mode_pk(mode) ? (double)ws[0]->packed.size() : (double)N * K / 32 * 17);
                if (hot)
                    std::printf("%s %s hot   : back to back %.1f us a call, %.0f GB/s (one copy, from the on-die cache)\n",
                                mode_name(mode), gu ? "gate_up" : "down   ", 1e3 * bur[(size_t)mode],
                                bytes / (bur[(size_t)mode] * 1e-3) / 1e9);
                else
                    std::printf("%s %s live=%d: median %.1f us, %.0f GB/s of weights; back to back %.1f us a call, %.0f GB/s\n",
                                mode_name(mode), gu ? "gate_up" : "down   ", live, 1e3 * iso[(size_t)mode],
                                bytes / (iso[(size_t)mode] * 1e-3) / 1e9, 1e3 * bur[(size_t)mode],
                                bytes / (bur[(size_t)mode] * 1e-3) / 1e9);
            }
            if (live == 0 || hot) continue;   /* no rows to compare (live 0); the hot calls are the live=1 ones */
            for (int it = 0; it < 20; ++it)
                for (int mode = 1; mode < nmodes; ++mode) {
                    (void)one(0, cc, y1.p);
                    (void)one(mode, cc++, y2.p);
                    std::vector<uint16_t> a((size_t)kTop * N), bb((size_t)kTop * N);
                    HC(hipMemcpy(a.data(), y1.p, a.size() * 2, hipMemcpyDeviceToHost));
                    HC(hipMemcpy(bb.data(), y2.p, bb.size() * 2, hipMemcpyDeviceToHost));
                    for (int i = 0; i < live; ++i)
                        for (int n = 0; n < N; ++n) differ += a[(size_t)i * N + n] != bb[(size_t)i * N + n];
                }
        }
    std::printf(differ ? "FAIL: gemv_vp differs from the plain row in %zu values\n"
                       : "PASS: the timed calls' outputs are bit-identical\n", differ);
    return differ ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s LIBR4D.so check [trials] [tail] | time\n", argv[0]);
        return 2;
    }
    /* The rows as the A/B serves them, the live-tail ranges off (their own switch); `check N tail`, on. libr4d
     * reads the switch once a process, at its first call, so it is set here, before any. */
    const bool tail = !std::strcmp(argv[2], "check") && argc > 4 && !std::strcmp(argv[4], "tail");
    if (tail) setenv("RADIANCE_EXPERIMENTAL_DS41_LIVE_TAIL", "1", 1);
    else unsetenv("RADIANCE_EXPERIMENTAL_DS41_LIVE_TAIL");
    const Lib lib(argv[1]);
    if (!std::strcmp(argv[2], "check")) return check(lib, argc > 3 ? std::atoi(argv[3]) : 48, tail);
    if (!std::strcmp(argv[2], "time")) return timing(lib);
    std::fprintf(stderr, "unknown mode %s\n", argv[2]);
    return 2;
}
