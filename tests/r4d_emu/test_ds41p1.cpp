/* tests/r4d_emu: prefill Phase 1's twin, moe_gemm_ggml_mxfp4_gemv_tall_p1 (libr4d/r4d_ds41_mxfp4.hip, mx_moe's
 * p1 branch; the contract T1-T6 in tools/ds41-host-experts/ds41_p1.h), on the host SIMT emulator.
 *
 * Under RADIANCE_EXPERIMENTAL_DS41_P1 the core's card expert stager computes every routed expert of a large
 * prefill chunk on the card: the layer's own gate_up, clamped SwiGLU and down over the VRAM residents with every
 * non-resident expert's entry marked "skip", then the same three ops again in waves, each wave's table naming
 * ring slots (block_mxfp4, or the packed-scale form under the packed mark) for its experts and "skip" for every
 * other. The twin must give every row it writes the decode band's bits. This holds it, byte for byte:
 *
 *   rows      the twin at 17 .. 1024 tokens over a table mixing block_mxfp4, packed and skipped entries, against
 *             the plain gemv row over the same tokens in passes of 16 with every expert in block_mxfp4 (T1, T2);
 *             a skipped expert's rows keep the poison they had (T3: and its entry points at a page that cannot be
 *             read, so a read of its weight stops the test); the slots past off[96] are zeros (T4); exactly two
 *             launches, gemv_vp's kernel then the tail's zero fill;
 *   split     the same call with other work splits (grids, items of one row group, few and many items a block,
 *             one gate_up row a pass where the default stages two) gives the same bytes (T5 changes which block computes a pass, never a bit);
 *   waves     the stager's composition: a main launch with a subset skipped, then waves of 1, 2 and 3 experts
 *             over a partition of that subset (each wave's table skipping every other expert), the clamped
 *             SwiGLU relaunched over all rows after each gate_up, leaves the gate_up, SwiGLU and down outputs
 *             equal to one launch over every expert;
 *   cut       a layer the consumer skip's per-row cut leaves 1 .. 40 rows of a 64- or 2048-row pass: on the path
 *             the stager picks for its height (ds41p1::card_layer), the gate_up, SwiGLU and down equal the plain
 *             gemv's in passes of 16 from the layer's first row; at 16 rows or fewer the card path's marked
 *             composition on gemv_vp is shown to differ (the defect the per-layer decision removes).
 *
 * Routings, one EP4 rank's slots as moe_scatter writes them (top-6 of 384, the rank's 96, -1 past them): uniform,
 * skewed, one hot local expert on nearly every token, the rank's last expert without rows of its own (tail only),
 * no local slot at all, and the real routing of the 3,086-token prompt's first rows (Phase 0's routes file:
 * DS41_P1_ROUTES or work/p0/routes-long3086.bin, skipped with a note when absent). Activations carry NaN, inf
 * and subnormal values on some token rows; a NaN compares equal to a NaN. Shapes are ds41's K (5120 for gate_up,
 * two rows staged a pass by default, one under a forced 36 KiB; 2304 for down, three) with N cut short (each output is its own weight row's sum, so N
 * changes no sum), and a short K with two weight rows a wave. Runs with the cards hidden. */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"
#include "r4d_ds41_tall.h"
#include "ds41_p1.h"
#include "ds41_host_experts.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <sys/mman.h>
#include <vector>

extern const char* emu_last_kernel;
extern int emu_grids;

/* libr4d restates the contract's constants (r4d_ds41_tall.h); they must agree with the shared header. */
constexpr bool same_str(const char* a, const char* b) { return *a == *b && (!*a || same_str(a + 1, b + 1)); }
static_assert(same_str(ds41p1::kEnv, kR4dDs41P1Env), "the switch's name");
static_assert(same_str(ds41p1::kTwinRow, kR4dDs41P1Row), "the twin row's name");
static_assert(ds41p1::kTwinMaxRows == kR4dDs41P1MaxRows, "the twin's bound");
static_assert(ds41p1::kMarkPacked == kR4dDs41MarkPacked && ds41p1::kMarkSkip == kR4dDs41MarkSkip, "the marks");

namespace {

constexpr int kNe = 96, kTop = 6, kRanks = 4;
constexpr float kLimit = 10.0f;
typedef int (*Launch)(const RadArgs*, RadStream);

uint8_t* aligned_bytes(size_t n) {
    const size_t a = (n + 63) / 64 * 64;
    void* p = std::aligned_alloc(64, a ? a : 64);
    if (!p) { std::printf("out of memory\n"); std::exit(2); }
    std::memset(p, 0xA5, a ? a : 64);
    return (uint8_t*)p;
}

/* A page no one may read: the weight under every skip mark. */
const void* g_noread = nullptr;
const void* skip_entry() { return ds41p1::mark(g_noread, ds41p1::kMarkSkip); }

uint16_t f2bf(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
bool nan16(uint16_t v) { return (v & 0x7f80u) == 0x7f80u && (v & 0x7fu); }
bool same16(uint16_t a, uint16_t b) { return a == b || (nan16(a) && nan16(b)); }

/* One expert weight from the release's planes, in both forms the card can hold. */
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

/* 96 experts of one [N, K] projection, made once a shape. */
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
};

/* ---- routings ------------------------------------------------------------------------------------ */
enum Kind { kUniform, kSkewed, kHot, kTailOnly, kNoLocal, kReal, kKinds };
const char* kind_name(int k) {
    static const char* n[] = { "uniform", "skewed", "hot expert", "last expert tail-only", "no local slot", "real" };
    return n[k];
}

/* Token t's top-6 of 384. */
struct Picks {
    std::vector<int> p;   /* [tokens][6] */
    int rank = 0;
};
std::vector<int16_t> g_routes;
int g_r_layers = 0, g_r_rows = 0;

bool load_routes() {
    const char* path = std::getenv("DS41_P1_ROUTES");
    if (!path || !*path) path = "/home/as/ds41-claude-build/work/p0/routes-long3086.bin";
    FILE* f = std::fopen(path, "rb");
    int32_t h[3];
    bool ok = f && std::fread(h, 4, 3, f) == 3 && h[2] == kTop && h[0] > 0 && h[1] > 0;
    if (ok) {
        g_routes.resize((size_t)h[0] * h[1] * h[2]);
        ok = std::fread(g_routes.data(), 2, g_routes.size(), f) == g_routes.size();
        g_r_layers = h[0];
        g_r_rows = h[1];
    }
    if (f) std::fclose(f);
    if (!ok) {
        g_routes.clear();
        std::printf("note: no routes file (%s): the real routing is left out\n", path);
    }
    return ok;
}

Picks make_picks(int kind, int M, std::mt19937_64& rng) {
    Picks r;
    r.rank = (int)(rng() % kRanks);
    if (kind == kReal) r.rank = (int)(rng() % 2) * 2;   /* ranks 0 and 2: their cards' slot counts differ most */
    r.p.resize((size_t)M * kTop);
    std::vector<double> w(kNe * kRanks);
    std::vector<int> perm(kNe * kRanks);
    for (int i = 0; i < kNe * kRanks; ++i) perm[(size_t)i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);
    for (int i = 0; i < kNe * kRanks; ++i) w[(size_t)perm[(size_t)i]] = 1.0 / std::pow(1.0 + i, 0.9);
    std::discrete_distribution<int> skew(w.begin(), w.end());
    const int hot = r.rank * kNe + (int)(rng() % kNe);
    const int layer = g_r_layers ? (int)(rng() % (uint64_t)g_r_layers) : 0;
    for (int t = 0; t < M; ++t) {
        int* pk = &r.p[(size_t)t * kTop];
        int n = 0;
        if (kind == kReal) {
            /* Rows past the file's wrap into the next layer's first rows: still real routing. */
            const int l = (layer + t / g_r_rows) % g_r_layers, row = t % g_r_rows;
            for (int k = 0; k < kTop; ++k) pk[k] = g_routes[((size_t)l * g_r_rows + row) * kTop + k];
            continue;
        }
        if (kind == kHot && rng() % 50 != 0) pk[n++] = hot;
        while (n < kTop) {
            int e;
            if (kind == kSkewed) e = skew(rng);
            else if (kind == kNoLocal) e = ((r.rank + 1 + (int)(rng() % (kRanks - 1))) % kRanks) * kNe + (int)(rng() % kNe);
            else e = (int)(rng() % (kNe * kRanks));
            if (kind == kTailOnly && e == r.rank * kNe + kNe - 1) continue;
            bool dup = false;
            for (int j = 0; j < n; ++j) dup |= pk[j] == e;
            if (!dup) pk[n++] = e;
        }
        std::shuffle(pk, pk + kTop, rng);
    }
    return r;
}

/* Tokens [t0, t1) as one pass on the rank: sorted (flat indices relative to t0), off, the local count. */
struct Pass {
    std::vector<int32_t> sorted, off;
    int local = 0;
};
Pass make_pass(const Picks& r, int t0, int t1) {
    std::vector<std::vector<int32_t>> by(kNe);
    for (int t = t0; t < t1; ++t)
        for (int k = 0; k < kTop; ++k) {
            const int e = r.p[(size_t)t * kTop + k];
            if (e / kNe == r.rank) by[(size_t)(e % kNe)].push_back((t - t0) * kTop + k);
        }
    Pass p;
    p.off.assign(kNe + 1, 0);
    for (int e = 0; e < kNe; ++e) {
        p.off[(size_t)e + 1] = p.off[(size_t)e] + (int)by[(size_t)e].size();
        for (int32_t f : by[(size_t)e]) p.sorted.push_back(f);
    }
    p.local = (int)p.sorted.size();
    p.sorted.resize((size_t)(t1 - t0) * kTop, -1);
    return p;
}

/* ---- launches ------------------------------------------------------------------------------------ */
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

/* The split a P1 call is forced to (r4d_rad_moe_gemm_mxfp4_gemv_tall_p1_opts); all zero: the served launcher. */
struct Split {
    int grid = 0, item_kib = 0, fill = 0, lds_kib = 0;
    bool served() const { return !grid && !item_kib && !fill && !lds_kib; }
};

/* One moe_gemm_ggml call into y (not cleared here): x [AR, K], the table, the pass's routing. */
int moe(Launch f, const Split* sp, const uint16_t* x, int AR, int K, const std::vector<const void*>& table, int N,
        const Pass& p, bool sorted_order, uint16_t* y) {
    const int T = (int)p.sorted.size();
    std::vector<const void*> tab = table;
    std::vector<int32_t> so = p.sorted, of = p.off;
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
    if (sp && !sp->served())
        return r4d_rad_moe_gemm_mxfp4_gemv_tall_p1_opts(&a, nullptr, sp->grid, sp->item_kib, sp->fill, sp->lds_kib);
    return f(&a, nullptr);
}
int p1(const Split* sp, const uint16_t* x, int AR, int K, const std::vector<const void*>& table, int N, const Pass& p,
       bool sorted_order, uint16_t* y, std::string* why) {
    const int g0 = emu_grids;
    const int rc = moe(r4d_rad_moe_gemm_mxfp4_gemv_tall_p1, sp, x, AR, K, table, N, p, sorted_order, y);
    const int n = emu_grids - g0;
    const std::string last = emu_last_kernel ? emu_last_kernel : "?";
    if (rc != RAD_OK) { *why = "refused (" + std::to_string(rc) + ")"; return 1; }
    /* gemv_vp's kernel, then the tail's zero fill: two launches, the fill last. */
    if (n != 2 || last.find("mx_zero_tail_kernel") == std::string::npos) {
        *why = std::to_string(n) + " launches, the last " + last;
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

/* Activations: `rows` rows of K, with NaN, inf and subnormal values on some. */
std::vector<uint16_t> make_x(int rows, int K, std::mt19937_64& rng) {
    std::normal_distribution<float> nd(0.f, 0.6f);
    static const uint16_t sp[] = { 0x7fc0, 0xffc0, 0x7f80, 0xff80, 0x0001, 0x8001, 0x0080, 0x8000, 0x7f7f };
    std::vector<uint16_t> x((size_t)rows * K);
    for (int r = 0; r < rows; ++r)
        for (int k = 0; k < K; ++k) {
            uint16_t v = f2bf(nd(rng));
            if (r % 13 == 5 && rng() % 211 == 0) v = sp[rng() % (sizeof sp / 2)];
            x[(size_t)r * K + k] = v;
        }
    return x;
}

int g_fails = 0, g_cases = 0;
long long g_rows = 0;
void fail(const std::string& s) {
    if (g_fails < 40) std::printf("FAIL %s\n", s.c_str());
    ++g_fails;
}

/* ---- rows and split ------------------------------------------------------------------------------ */
struct Shape {
    const char* name;
    int N, K;
    bool sorted;   /* the down projection: one activation row a slot */
};

/* Per expert: 0 block_mxfp4, 1 packed, 2 skipped. */
std::vector<int> make_forms(const Pass& p, std::mt19937_64& rng) {
    std::vector<int> f((size_t)kNe);
    int skipped_live = 0, packed_live = 0;
    for (int e = 0; e < kNe; ++e) {
        const uint64_t r = rng() % 20;
        f[(size_t)e] = r < 4 ? 2 : r < 10 ? 1 : 0;
        const bool live = p.off[(size_t)e + 1] > p.off[(size_t)e];
        skipped_live += live && f[(size_t)e] == 2;
        packed_live += live && f[(size_t)e] == 1;
    }
    /* At least one live expert of each mark when there are live experts. */
    for (int e = 0; e < kNe && (!skipped_live || !packed_live); ++e)
        if (p.off[(size_t)e + 1] > p.off[(size_t)e] && f[(size_t)e] == 0) {
            if (!skipped_live) { f[(size_t)e] = 2; skipped_live = 1; }
            else { f[(size_t)e] = 1; packed_live = 1; }
        }
    return f;
}
std::vector<const void*> table_of(const Bank& b, const std::vector<int>& forms) {
    std::vector<const void*> t((size_t)kNe);
    for (int e = 0; e < kNe; ++e) {
        const Expert& x = *b.e[(size_t)e];
        t[(size_t)e] = forms[(size_t)e] == 2   ? skip_entry()
                       : forms[(size_t)e] == 1 ? ds41p1::mark(x.packed, ds41p1::kMarkPacked)
                                               : (const void*)x.block;
    }
    return t;
}

void rows_case(const Bank& b, const Shape& sh, int M, int kind, uint32_t seed, bool splits) {
    ++g_cases;
    const auto t0c = std::chrono::steady_clock::now();
    std::mt19937_64 rng(seed);
    const Picks pk = make_picks(kind, M, rng);
    const Pass full = make_pass(pk, 0, M);
    const int T = M * kTop, AR = sh.sorted ? T : M, N = sh.N, K = sh.K;
    /* Activations: token rows (gate_up), or a row for every (token, slot) (down, read by slot). */
    const std::vector<uint16_t> act = make_x(sh.sorted ? T : M, K, rng);
    std::vector<uint16_t> xin;
    if (sh.sorted) {
        xin.assign((size_t)T * K, 0);
        for (int i = 0; i < full.local; ++i)
            std::memcpy(&xin[(size_t)i * K], &act[(size_t)full.sorted[(size_t)i] * K], (size_t)K * 2);
    } else {
        xin = act;
    }
    const std::vector<int> forms = make_forms(full, rng);
    const std::vector<const void*> tab = table_of(b, forms);
    std::vector<const void*> blocks((size_t)kNe);
    for (int e = 0; e < kNe; ++e) blocks[(size_t)e] = b.e[(size_t)e]->block;
    std::string label = std::string(sh.name) + " M=" + std::to_string(M) + " " + kind_name(kind) + " rank " +
                        std::to_string(pk.rank);
    std::vector<uint16_t> y((size_t)T * N, 0xAAAAu);
    std::string why;
    if (p1(nullptr, xin.data(), AR, K, tab, N, full, sh.sorted, y.data(), &why)) {
        fail(label + ": the P1 twin: " + why);
        return;
    }
    /* The decode band: the plain gemv over the same tokens in passes of 16, every expert block_mxfp4. Each local
     * slot's row by its flat index. */
    std::vector<int> row_of((size_t)T, -1);
    for (int i = 0; i < full.local; ++i) row_of[(size_t)full.sorted[(size_t)i]] = i;
    size_t differ = 0, untouched_bad = 0, tail_bad = 0;
    int skipped_rows = 0, packed_rows = 0;
    for (int t0 = 0; t0 < M; t0 += 16) {
        const int t1 = std::min(M, t0 + 16), n = t1 - t0;
        const Pass pp = make_pass(pk, t0, t1);
        std::vector<uint16_t> in;
        if (sh.sorted) {
            in.assign((size_t)n * kTop * K, 0);
            for (int i = 0; i < pp.local; ++i)
                std::memcpy(&in[(size_t)i * K], &act[(size_t)(pp.sorted[(size_t)i] + t0 * kTop) * K], (size_t)K * 2);
        } else {
            in.assign(act.begin() + (size_t)t0 * K, act.begin() + (size_t)t1 * K);
        }
        std::vector<uint16_t> yc((size_t)n * kTop * N, 0x5555u);
        if (moe(r4d_rad_moe_gemm_mxfp4_gemv, nullptr, in.data(), sh.sorted ? n * kTop : n, K, blocks, N, pp, sh.sorted,
                yc.data()) != RAD_OK) {
            fail(label + ": a 16-token gemv launch was refused");
            return;
        }
        for (int i = 0; i < pp.local; ++i) {
            const int f = pp.sorted[(size_t)i] + t0 * kTop, ra = row_of[(size_t)f];
            int e = 0;
            while (full.off[(size_t)e + 1] <= ra) ++e;
            const uint16_t* got = &y[(size_t)ra * N];
            if (forms[(size_t)e] == 2) {
                ++skipped_rows;
                for (int c = 0; c < N; ++c) untouched_bad += got[c] != 0xAAAAu;
                continue;
            }
            packed_rows += forms[(size_t)e] == 1;
            size_t d = 0;
            for (int c = 0; c < N; ++c) d += !same16(got[c], yc[(size_t)i * N + c]);
            if (d && differ < 4)
                std::printf("  %s: token %d slot %d (expert %d, %s) differs in %zu of %d values from its 16-token "
                            "pass\n", label.c_str(), f / kTop, f % kTop, e, forms[(size_t)e] ? "packed" : "block", d, N);
            differ += d;
            ++g_rows;
        }
    }
    for (size_t i = (size_t)full.local * N; i < y.size(); ++i) tail_bad += y[i] != 0;
    if (differ) fail(label + ": " + std::to_string(differ) + " values differ from the 16-token gemv");
    if (untouched_bad) fail(label + ": " + std::to_string(untouched_bad) + " values of skipped experts' rows written");
    if (tail_bad) fail(label + ": " + std::to_string(tail_bad) + " values past the live end are not zero");
    /* Other work splits: the same bytes. */
    int nsplit = 0;
    if (splits) {
        const Split sps[] = { { 5, 1, 1, 0 }, { 64, 4, 64, 0 }, { 300, 1000, 2, 0 }, { 0, 0, 0, 36 }, { 0, 0, 0, 56 } };
        for (const Split& s2 : sps) {
            if (s2.lds_kib && sh.K != 5120) continue;   /* one gate_up row a pass (36 KiB), or two */
            std::vector<uint16_t> y2((size_t)T * N, 0xAAAAu);
            if (p1(&s2, xin.data(), AR, K, tab, N, full, sh.sorted, y2.data(), &why)) {
                fail(label + ": a forced split: " + why);
                continue;
            }
            ++nsplit;
            if (std::memcmp(y.data(), y2.data(), y.size() * 2) != 0)
                fail(label + ": split grid " + std::to_string(s2.grid) + " item " + std::to_string(s2.item_kib) +
                     " KiB fill " + std::to_string(s2.fill) + " lds " + std::to_string(s2.lds_kib) +
                     " KiB gives other bytes");
        }
    }
    std::printf("  %s: %d local slots (%d skipped, %d packed), %d splits, %.1f s\n", label.c_str(), full.local,
                skipped_rows, packed_rows, nsplit,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0c).count());
    std::fflush(stdout);
}

/* ---- the wave composition ------------------------------------------------------------------------- */
void waves_case(const Bank& gu, const Bank& dn, int M, int kind, uint32_t seed) {
    ++g_cases;
    const auto t0c = std::chrono::steady_clock::now();
    std::mt19937_64 rng(seed);
    const Picks pk = make_picks(kind, M, rng);
    const Pass p = make_pass(pk, 0, M);
    const int T = M * kTop, n = gu.N / 2, E = gu.K;
    const std::vector<uint16_t> x = make_x(M, E, rng);
    /* Every expert's entry as one launch has it: block or packed, by turns. */
    std::vector<const void*> fgu((size_t)kNe), fdn((size_t)kNe);
    for (int e = 0; e < kNe; ++e) {
        const bool pkd = rng() % 3 == 0;
        fgu[(size_t)e] = pkd ? ds41p1::mark(gu.e[(size_t)e]->packed, ds41p1::kMarkPacked) : gu.e[(size_t)e]->block;
        fdn[(size_t)e] = rng() % 3 == 0 ? ds41p1::mark(dn.e[(size_t)e]->packed, ds41p1::kMarkPacked)
                                        : (const void*)dn.e[(size_t)e]->block;
    }
    std::string label = std::string("waves E=") + std::to_string(E) + " n=" + std::to_string(n) + " M=" +
                        std::to_string(M) + " " + kind_name(kind);
    std::string why;
    /* One launch over every expert. */
    std::vector<uint16_t> G((size_t)T * 2 * n, 0xAAAAu), S((size_t)T * n, 0xAAAAu), D((size_t)T * dn.N, 0xAAAAu);
    if (p1(nullptr, x.data(), M, E, fgu, gu.N, p, false, G.data(), &why) || swiglu(G.data(), S.data(), T, n) != RAD_OK ||
        p1(nullptr, S.data(), T, n, fdn, dn.N, p, true, D.data(), &why)) {
        fail(label + ": the one-launch chain: " + why);
        return;
    }
    /* The stager's: the main launch with a subset skipped (about half of the live experts, and some without rows),
     * then waves over a partition of it. */
    std::vector<int> sub;
    std::vector<char> in_sub((size_t)kNe, 0);
    for (int e = 0; e < kNe; ++e) {
        const bool live = p.off[(size_t)e + 1] > p.off[(size_t)e];
        if ((live && rng() % 2 == 0) || (!live && rng() % 8 == 0)) {
            sub.push_back(e);
            in_sub[(size_t)e] = 1;
        }
    }
    std::shuffle(sub.begin(), sub.end(), rng);
    std::vector<uint16_t> G2((size_t)T * 2 * n, 0xAAAAu), S2((size_t)T * n, 0xAAAAu), D2((size_t)T * dn.N, 0xAAAAu);
    std::vector<const void*> mgu = fgu, mdn = fdn;
    for (int e : sub) mgu[(size_t)e] = mdn[(size_t)e] = skip_entry();
    if (p1(nullptr, x.data(), M, E, mgu, gu.N, p, false, G2.data(), &why) ||
        swiglu(G2.data(), S2.data(), T, n) != RAD_OK ||
        p1(nullptr, S2.data(), T, n, mdn, dn.N, p, true, D2.data(), &why)) {
        fail(label + ": the main launch: " + why);
        return;
    }
    int waves = 0;
    for (size_t w0 = 0, sz = 1; w0 < sub.size(); w0 += sz, sz = sz % 3 + 1) {
        std::vector<const void*> wgu((size_t)kNe, skip_entry()), wdn((size_t)kNe, skip_entry());
        for (size_t j = w0; j < std::min(sub.size(), w0 + sz); ++j) {
            wgu[(size_t)sub[j]] = fgu[(size_t)sub[j]];
            wdn[(size_t)sub[j]] = fdn[(size_t)sub[j]];
        }
        if (p1(nullptr, x.data(), M, E, wgu, gu.N, p, false, G2.data(), &why) ||
            swiglu(G2.data(), S2.data(), T, n) != RAD_OK ||
            p1(nullptr, S2.data(), T, n, wdn, dn.N, p, true, D2.data(), &why)) {
            fail(label + ": wave " + std::to_string(waves) + ": " + why);
            return;
        }
        ++waves;
    }
    auto cmp = [&](const std::vector<uint16_t>& a, const std::vector<uint16_t>& b2, const char* what) {
        size_t d = 0;
        for (size_t i = 0; i < a.size(); ++i) d += !same16(a[i], b2[i]);
        if (d) fail(label + ": " + what + ": " + std::to_string(d) + " values differ from one launch");
    };
    cmp(G, G2, "gate_up");
    cmp(S, S2, "swiglu");
    cmp(D, D2, "down");
    std::printf("  %s: %d local slots, %zu experts in %d waves, %.1f s\n", label.c_str(), p.local, sub.size(), waves,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0c).count());
    std::fflush(stdout);
}

/* ---- a layer the per-row cut leaves short --------------------------------------------------------------
 *
 * Under the consumer skip's per-row cut (core/sched/consumer_skip.h, consumer_row_starts) routed layer L of a pass
 * of M rows runs on its rows [row0(L), M) alone. In a prompt's last chunk a deep layer can be left with R <= 16 of
 * them: the plugin serves that call on the decode band (gemv, or gemv_vp under P1), which honours no skip mark. The
 * stager decides a layer at a time (ds41p1::card_layer, FileStager::p1_layer_on): R >= 17 the card path (the twin,
 * the non-residents skipped, then waves); R <= 16 the CPU path's layer, one gemv_vp launch over the plain table
 * (packed residents under the packed mark, every other entry plain; a non-resident's rows here stand for the CPU's,
 * which the host experts' own tests hold to the gemv). Either way the layer's gate_up, clamped SwiGLU and down give
 * the bytes of the plain gemv over the same rows in passes of 16 from row0 (the 16-row schedule's groups). For
 * R <= 16 it also shows the defect the per-layer decision removes: the card path's composition (non-residents
 * skip-marked, then waves) launched on gemv_vp reads each marked entry as an address and gives other bytes. */
struct Padded {   /* each expert's block copied with room past it: a skip-marked entry read as an address */
    std::vector<uint8_t*> p;
    explicit Padded(const Bank& b) {
        const size_t n = (size_t)b.N * (b.K / 32) * 17;
        for (int e = 0; e < kNe; ++e) {
            p.push_back(aligned_bytes(n + 64));
            std::memcpy(p.back(), b.e[(size_t)e]->block, n);
        }
    }
    ~Padded() { for (uint8_t* q : p) std::free(q); }
};

void cut_case(const Bank& gu, const Bank& dn, const Padded& pgu, const Padded& pdn, int M, int R, int kind,
              uint32_t seed) {
    ++g_cases;
    std::mt19937_64 rng(seed);
    const Picks pk = make_picks(kind, M, rng);
    const int r0 = M - R;
    const Pass p = make_pass(pk, r0, M);   /* the layer's own scatter, over its rows */
    const int T = R * kTop, n = gu.N / 2, E = gu.K;
    const std::vector<uint16_t> x = make_x(R, E, rng);
    const bool card = ds41p1::card_layer(R);
    std::string label = std::string("cut M=") + std::to_string(M) + " row0=" + std::to_string(r0) + " (" +
                        std::to_string(R) + " rows, " + (card ? "card path" : "CPU path") + ") " + kind_name(kind);
    if (card != (R >= 17)) {
        fail(label + ": ds41p1::card_layer disagrees with the gemv band's bound");
        return;
    }
    /* Residents (block or packed) and non-residents; at least one live non-resident. */
    std::vector<int> res((size_t)kNe);   /* 0 non-resident, 1 block resident, 2 packed resident */
    for (int e = 0; e < kNe; ++e) res[(size_t)e] = (int)(rng() % 3);
    bool live_nr = false;
    for (int e = 0; e < kNe; ++e) live_nr |= res[(size_t)e] == 0 && p.off[(size_t)e + 1] > p.off[(size_t)e];
    for (int e = 0; e < kNe && !live_nr; ++e)
        if (p.off[(size_t)e + 1] > p.off[(size_t)e]) { res[(size_t)e] = 0; live_nr = true; }
    /* And a live resident when two experts or more are live: every wave then rewrites its rows on gemv_vp. */
    int live = 0, live_res = 0;
    for (int e = 0; e < kNe; ++e)
        if (p.off[(size_t)e + 1] > p.off[(size_t)e]) { ++live; live_res += res[(size_t)e] != 0; }
    for (int e = kNe - 1; e >= 0 && live >= 2 && !live_res; --e)
        if (p.off[(size_t)e + 1] > p.off[(size_t)e] && !res[(size_t)e]) { res[(size_t)e] = 2; live_res = 1; }
    std::vector<const void*> fgu((size_t)kNe), fdn((size_t)kNe);
    for (int e = 0; e < kNe; ++e) {
        const bool pkd = res[(size_t)e] == 2;
        fgu[(size_t)e] = pkd ? ds41p1::mark(gu.e[(size_t)e]->packed, ds41p1::kMarkPacked) : gu.e[(size_t)e]->block;
        fdn[(size_t)e] = pkd ? ds41p1::mark(dn.e[(size_t)e]->packed, ds41p1::kMarkPacked) : dn.e[(size_t)e]->block;
    }
    std::string why;
    auto chain = [&](Launch f, const std::vector<const void*>& tg, const std::vector<const void*>& td,
                     std::vector<uint16_t>& G, std::vector<uint16_t>& S, std::vector<uint16_t>& D) {
        if (f == r4d_rad_moe_gemm_mxfp4_gemv_tall_p1)
            return p1(nullptr, x.data(), R, E, tg, gu.N, p, false, G.data(), &why) ||
                   swiglu(G.data(), S.data(), T, n) != RAD_OK ||
                   p1(nullptr, S.data(), T, n, td, dn.N, p, true, D.data(), &why);
        why = "a decode-band launch was refused";
        return moe(f, nullptr, x.data(), R, E, tg, gu.N, p, false, G.data()) != RAD_OK ||
               swiglu(G.data(), S.data(), T, n) != RAD_OK ||
               moe(f, nullptr, S.data(), T, n, td, dn.N, p, true, D.data()) != RAD_OK;
    };
    /* The stager's composition for a layer of R rows: the main launch with every non-resident skipped (`mark`
     * its entry), then waves of 1, 2, 3 non-residents, each skipping every other expert. */
    auto composed = [&](Launch f, auto mark, std::vector<uint16_t>& D) {
        std::vector<uint16_t> G((size_t)T * 2 * n, 0xAAAAu), S((size_t)T * n, 0xAAAAu);
        std::vector<const void*> mg = fgu, md = fdn;
        std::vector<int> nr;
        for (int e = 0; e < kNe; ++e)
            if (!res[(size_t)e]) {
                nr.push_back(e);
                mg[(size_t)e] = mark(true, e);
                md[(size_t)e] = mark(false, e);
            }
        std::shuffle(nr.begin(), nr.end(), rng);
        if (chain(f, mg, md, G, S, D)) return false;
        for (size_t w0 = 0, sz = 1; w0 < nr.size(); w0 += sz, sz = sz % 3 + 1) {
            std::vector<const void*> wg((size_t)kNe), wd((size_t)kNe);
            for (int e = 0; e < kNe; ++e) { wg[(size_t)e] = mark(true, e); wd[(size_t)e] = mark(false, e); }
            for (size_t j = w0; j < std::min(nr.size(), w0 + sz); ++j) {
                wg[(size_t)nr[j]] = fgu[(size_t)nr[j]];
                wd[(size_t)nr[j]] = fdn[(size_t)nr[j]];
            }
            if (chain(f, wg, wd, G, S, D)) return false;
        }
        return true;
    };
    std::vector<uint16_t> D((size_t)T * dn.N, 0xAAAAu);
    if (card) {
        if (!composed(r4d_rad_moe_gemm_mxfp4_gemv_tall_p1, [](bool, int) { return skip_entry(); }, D)) {
            fail(label + ": the card path: " + why);
            return;
        }
    } else {
        std::vector<uint16_t> G((size_t)T * 2 * n, 0xAAAAu), S((size_t)T * n, 0xAAAAu);
        if (chain(r4d_rad_moe_gemm_mxfp4_gemv_vp, fgu, fdn, G, S, D)) {
            fail(label + ": the CPU path's layer: " + why);
            return;
        }
    }
    /* The 16-row schedule: the plain gemv over [row0, M) in passes of 16 from row0, every expert block_mxfp4. */
    std::vector<const void*> bg((size_t)kNe), bd((size_t)kNe);
    for (int e = 0; e < kNe; ++e) { bg[(size_t)e] = gu.e[(size_t)e]->block; bd[(size_t)e] = dn.e[(size_t)e]->block; }
    std::vector<int> row_of((size_t)T, -1);
    for (int i = 0; i < p.local; ++i) row_of[(size_t)p.sorted[(size_t)i]] = i;
    std::vector<uint16_t> ref((size_t)T * dn.N, 0xAAAAu);
    for (int t0 = r0; t0 < M; t0 += 16) {
        const int t1 = std::min(M, t0 + 16), m = t1 - t0, Tm = m * kTop;
        const Pass pp = make_pass(pk, t0, t1);
        std::vector<uint16_t> G((size_t)Tm * 2 * n), S((size_t)Tm * n), Dm((size_t)Tm * dn.N);
        if (moe(r4d_rad_moe_gemm_mxfp4_gemv, nullptr, x.data() + (size_t)(t0 - r0) * E, m, E, bg, gu.N, pp, false,
                G.data()) != RAD_OK ||
            swiglu(G.data(), S.data(), Tm, n) != RAD_OK ||
            moe(r4d_rad_moe_gemm_mxfp4_gemv, nullptr, S.data(), Tm, n, bd, dn.N, pp, true, Dm.data()) != RAD_OK) {
            fail(label + ": a 16-token gemv launch was refused");
            return;
        }
        for (int i = 0; i < pp.local; ++i) {
            const int ra = row_of[(size_t)(pp.sorted[(size_t)i] + (t0 - r0) * kTop)];
            std::memcpy(&ref[(size_t)ra * dn.N], &Dm[(size_t)i * dn.N], (size_t)dn.N * 2);
        }
    }
    auto differ = [&](const std::vector<uint16_t>& a) {
        size_t d = 0;
        for (size_t i = 0; i < (size_t)p.local * dn.N; ++i) d += !same16(a[i], ref[i]);
        return d;
    };
    if (const size_t d = differ(D)) fail(label + ": " + std::to_string(d) + " down values differ from the 16-row passes");
    g_rows += p.local;
    /* The defect: at R <= 16 the card path's composition on the band's kernel. */
    size_t bad = 0;
    if (!card) {
        std::vector<uint16_t> D2((size_t)T * dn.N, 0xAAAAu);
        auto marked = [&](bool g, int e) {
            return ds41p1::mark(g ? pgu.p[(size_t)e] : pdn.p[(size_t)e], ds41p1::kMarkSkip);
        };
        if (!composed(r4d_rad_moe_gemm_mxfp4_gemv_vp, marked, D2)) {
            fail(label + ": the marked composition on gemv_vp: " + why);
            return;
        }
        bad = differ(D2);
        if (!bad && live >= 2)
            fail(label + ": the marked composition on gemv_vp gave the 16-row bytes: the case does not see the defect");
    }
    std::printf("  %s: %d local slots, equal to the 16-row passes%s\n", label.c_str(), p.local,
                card ? "" : (", the marked composition on gemv_vp differs in " + std::to_string(bad) + " values").c_str());
    std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
    /* `quick`: the smaller heights only (a sanitiser or a slow host). */
    const bool quick = argc > 1 && !std::strcmp(argv[1], "quick");
    void* page = mmap(nullptr, 1 << 16, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (page == MAP_FAILED) { std::printf("mmap failed\n"); return 2; }
    g_noread = page;
    const bool real = load_routes();
    const auto t0 = std::chrono::steady_clock::now();
    const Shape shapes[] = {
        { "gate_up K 5120", 32, 5120, false },
        { "down K 2304", 48, 2304, true },
        { "K 512, two weight rows a wave", 1024, 512, false },
    };
    const Bank b0(shapes[0].N, shapes[0].K, 101), b1(shapes[1].N, shapes[1].K, 202), b2(shapes[2].N, shapes[2].K, 303);
    const Bank* banks[] = { &b0, &b1, &b2 };
    for (int si = 0; si < 3; ++si) {
        std::vector<int> Ms = si == 2 ? std::vector<int>{ 17, 64 } : std::vector<int>{ 17, 64, 256, 1024 };
        if (quick) Ms = { 17, 64 };
        for (int M : Ms)
            for (int k = 0; k < kKinds; ++k) {
                if (k == kReal && !real) continue;
                /* The tallest height: the routings that differ most (real, skewed, hot). */
                if (M >= 1024 && k != kReal && k != kSkewed && k != kHot) continue;
                const bool splits = M <= 256 || k == kReal;
                rows_case(*banks[si], shapes[si], M, k, 9001u * (uint32_t)M + 77u * (uint32_t)k + (uint32_t)si,
                          splits);
            }
    }
    /* The wave composition: ds41's gate_up K with a short SwiGLU width, and a short K with a wide one. */
    const Bank wg0(64, 5120, 404), wd0(96, 32, 505), wg1(512, 512, 606), wd1(512, 256, 707);
    for (int M : quick ? std::vector<int>{ 17, 64 } : std::vector<int>{ 17, 64, 256 })
        for (int k : { kUniform, kHot, kTailOnly, kNoLocal, kReal }) {
            if (k == kReal && !real) continue;
            waves_case(wg0, wd0, M, k, 31337u * (uint32_t)M + (uint32_t)k);
            if (M <= 64) waves_case(wg1, wd1, M, k, 4242u * (uint32_t)M + (uint32_t)k);
        }
    /* A layer the per-row cut leaves short in an armed pass (FileStager::p1_layer_on). */
    {
        const Padded pg0(wg0), pd0(wd0);
        for (int M : { 64, 2048 })   /* at most 40 rows computed: quick too */
            for (int R : { 1, 6, 16, 17, 40 })
                for (int k : { kUniform, kHot, kReal }) {
                    if (k == kReal && !real) continue;
                    cut_case(wg0, wd0, pg0, pd0, M, R, k, 271u * (uint32_t)M + 13u * (uint32_t)R + (uint32_t)k);
                }
    }
    std::printf("%s: %d cases, %lld (token, slot) rows held to the 16-token gemv, %.0f s; P1 twin %s (M <= %d)\n",
                g_fails ? "FAIL" : "PASS", g_cases, g_rows,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), kR4dDs41P1Row,
                kR4dDs41P1MaxRows);
    munmap(page, 1 << 16);
    return g_fails ? 1 : 0;
}
