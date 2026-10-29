/* tests/r4d_emu: DeepSeek-V4.1's MXFP4 grouped GEMM (r4d_ds41_mxfp4.hip, moe_gemm_ggml type=mxfp4
 * over llama.cpp's 17-byte blocks) on the host SIMT emulator, at the DSpark draft bank's shape --
 * 32 experts a rank of 128, top-3 -- under RADIANCE_EXPERIMENTAL_DS41_DRAFT_GEMV's two changes:
 * the live tail (the slots routed to other ranks no longer run against the rank's last expert) and
 * one pass over each expert's weights (its rows restaged in K chunks a row group where they do not
 * fit the LDS whole). Every mode -- the old path, the tail, the one pass, both -- must write the
 * bytes of a host restatement of the decode GEMV's sequence for each output (lane = block mod 32,
 * its blocks in order, mx_unit's group order, one rounded product and one rounded sum a value,
 * r4d_wave_sum's butterfly, bf16 RNE), zeros in the rows of dropped slots, and nothing past a
 * row's N.
 *
 * Routings are moe_scatter's at EP4 (a rank's slots grouped by expert in slot order, the rest -1),
 * on each of the four ranks: no local slot, the last expert with and without rows of its own, one
 * expert in every query row, a few experts sharing the rows, spread; 1, 2, 5, 6, 8 and 16 query
 * rows (an expert with more than XM's 8 rows takes two passes in every mode) and 20 (the tiled
 * path, which takes the tail and not the pass, held to its own sequence: one sum over K in order,
 * not the GEMV's, from which it differs in a value now and then). gate_up reads token rows at K
 * 5120, where one row fits the LDS whole, down sorted rows at K 2304, where three do; N 1024 keeps
 * two weight rows a wave and N 64 one, a fraction of the real 4608 / 5120, which the served entry
 * runs once each. Row pitches padded, every other activation pitch unaligned (the 2-byte loads).
 *
 * The probe counts the work (r4d_rad_moe_gemm_mxfp4_draft_probe): with the tail the live experts
 * own off[32] rows and the zero-tail kernel writes the T - off[32] others; with one pass an expert
 * with up to XM rows has its weights read once, where the old path reads the last expert's once
 * per dropped slot at gate_up. The served entry, with the switch set before the first call, takes
 * both at the real shapes and counts its calls (r4d_ds41_draft_gemv_counts); the trunk's 96 /
 * top-6 shape is left as it was. */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"
#include "r4d_ds41_live_tail_policy.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

extern "C" int r4d_rad_moe_gemm_mxfp4_draft_probe(const RadArgs*, RadStream, int, int, unsigned long long*);
extern "C" void r4d_ds41_draft_gemv_counts(unsigned long long*);

namespace {

constexpr int kLocal = 32, kGlobal = 128, kTop = 3;
constexpr int kEmbd = 5120, kFf = 2304;

static_assert(ds41_live_tail::draft_eligible(5, 4608, 5120, 32, 3) &&
                  ds41_live_tail::draft_eligible(5, 5120, 2304, 32, 3),
              "the draft bank's gate_up and down are the switch's shape");
static_assert(!ds41_live_tail::draft_eligible(5, 4608, 5120, 96, 6) &&
                  !ds41_live_tail::eligible(5, 4608, 5120, 32, 3) &&
                  !ds41_live_tail::draft_eligible(5, 1024, 5120, 32, 3),
              "the switch and the trunk's live tail never share a shape");
static_assert(ds41_live_tail::draft_geometry(5, 4608, 5120, 32, 3, 15, 5, false) &&
                  ds41_live_tail::draft_geometry(5, 5120, 2304, 32, 3, 15, 15, true) &&
                  !ds41_live_tail::draft_geometry(5, 4608, 5120, 32, 3, 12, 5, false),
              "token rows are the tokens, sorted rows the slots, and the slots one whole routing");

const float kKv[16] = { 0.f, 1.f, 2.f, 3.f, 4.f, 6.f, 8.f, 12.f, 0.f, -1.f, -2.f, -3.f, -4.f, -6.f, -8.f, -12.f };

float bits_f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
uint32_t f_bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
float bf16f(uint16_t h) { return bits_f((uint32_t)h << 16); }
/* bf16_hi_rne_nan(f) >> 16: nearest even, a NaN kept and quieted */
uint16_t bf16_rne_nan(float f) {
    const uint32_t u = f_bits(f);
    return (uint16_t)((f != f ? (u | 0x400000u) : u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
/* ggml_e8m0_to_fp32_half: 2^(e - 128), e 0 and 1 as subnormals */
float e8m0_half(uint32_t e) { return bits_f(e < 2u ? (0x00200000u << e) : ((e - 1u) << 23)); }

/* The decode GEMV's sequence for one output: weight row `w` (K / 32 blocks) dotted with `x`. */
uint16_t ref_dot(const uint8_t* w, const uint16_t* x, int K) {
    float lane[32] = {};
    for (int U = 0; U < K / 32; ++U) {
        const uint8_t* b = w + (size_t)17 * U;
        const float d = e8m0_half(b[0]);
        float& acc = lane[U % 32];
        for (int G = 0; G < 4; ++G)
            for (int t = 0; t < 8; ++t) {
                const uint32_t by = b[1 + 8 * (G & 1) + t];
                const float wv = kKv[G >= 2 ? by >> 4 : by & 15u] * d;
                const float prod = bf16f(x[32 * U + 8 * G + t]) * wv;
                acc = acc + prod;
            }
    }
    for (int s = 16; s >= 1; s >>= 1) {
        float n[32];
        for (int L = 0; L < 32; ++L) n[L] = lane[L] + lane[L ^ s];
        std::memcpy(lane, n, sizeof n);
    }
    return bf16_rne_nan(lane[0]);
}

/* The tiled path's sequence (more than 16 rows): one sum over K in order, slab after slab, each
 * slab's 32 values in order. Not the GEMV's bits, by design; the same in every mode. */
uint16_t ref_dot_seq(const uint8_t* w, const uint16_t* x, int K) {
    float acc = 0.0f;
    for (int U = 0; U < K / 32; ++U) {
        const uint8_t* b = w + (size_t)17 * U;
        const float d = e8m0_half(b[0]);
        for (int v = 0; v < 32; ++v) {
            const uint32_t by = b[1 + v % 16];
            const float wv = kKv[v >= 16 ? by >> 4 : by & 15u] * d;
            const float prod = bf16f(x[32 * U + v]) * wv;
            acc = acc + prod;
        }
    }
    return bf16_rne_nan(acc);
}

/* A random expert in llama.cpp's blocks: e8m0 near 2^-8, or every few blocks an edge (the two
 * subnormal scales among them). */
std::vector<uint8_t> make_expert(int N, int K, std::mt19937_64& rng) {
    static const uint8_t edge[] = { 0, 1, 2, 3, 100, 126, 127, 128, 129, 140 };
    std::vector<uint8_t> b((size_t)N * (K / 32) * 17);
    for (size_t blk = 0; blk < b.size() / 17; ++blk) {
        const uint64_t r = rng(), a = rng();
        b[blk * 17] = (r & 15) == 0 ? edge[(r >> 4) % sizeof edge] : (uint8_t)(116 + (r >> 8) % 12);
        for (int j = 0; j < 16; ++j) b[blk * 17 + 1 + j] = (uint8_t)(a >> (4 * j));
    }
    return b;
}

std::vector<uint16_t> make_rows(int rows, int pitch, std::mt19937_64& rng) {
    static const uint16_t special[] = { 0x0000, 0x8000, 0x0001, 0x8001, 0x0080, 0x3f80, 0x00ff, 0x4120 };
    std::normal_distribution<float> nd(0.f, 0.5f);
    std::vector<uint16_t> x((size_t)rows * pitch);
    for (auto& v : x) {
        v = bf16_rne_nan(nd(rng));
        if (rng() % 13 == 0) v = special[rng() % (sizeof special / 2)];
    }
    return x;
}

/* moe_scatter on rank `rank`: its slots grouped by local expert in slot order, the rest -1. */
struct Routing {
    int m = 0, T = 0, rank = 0;
    std::vector<int32_t> sorted, off;
};
Routing scatter(const std::vector<int>& ids, int m, int rank) {
    Routing R;
    R.m = m; R.T = m * kTop; R.rank = rank;
    R.off.assign(kLocal + 1, 0);
    for (int e = 0; e < kLocal; ++e) {
        R.off[(size_t)e] = (int)R.sorted.size();
        for (int i = 0; i < R.T; ++i)
            if (ids[(size_t)i] == rank * kLocal + e) R.sorted.push_back(i);
    }
    R.off[kLocal] = (int)R.sorted.size();
    R.sorted.resize((size_t)R.T, -1);
    return R;
}

/* Each token's top-3: distinct global experts. `pick(token, slot)` proposes; a repeat is redrawn. */
enum Pattern { kNoLocal, kLastOwn, kLastIdle, kEveryRow, kFew, kSpread, kPatterns };
const char* const kPatternName[kPatterns] = { "no local slot", "last expert with its own rows",
                                              "last expert idle", "one expert in every row",
                                              "a few experts share the rows", "spread" };
std::vector<int> route(Pattern p, int m, int rank, std::mt19937_64& rng) {
    std::vector<int> ids((size_t)m * kTop, -1);
    const int base = rank * kLocal, other = ((rank + 1) % 4) * kLocal;
    for (int t = 0; t < m; ++t)
        for (int s = 0; s < kTop; ++s) {
            int e = -1;
            for (int tries = 0; tries < 1000; ++tries) {
                switch (p) {
                    case kNoLocal: e = (base + kLocal + (int)(rng() % (kGlobal - kLocal))) % kGlobal; break;
                    case kLastOwn: e = s == 0 && t % 2 == 0 ? base + kLocal - 1 : (int)(rng() % kGlobal); break;
                    case kLastIdle: {
                        e = (int)(rng() % kGlobal);
                        if (e == base + kLocal - 1) e = base;
                        break;
                    }
                    case kEveryRow: e = s == 0 ? base + 7 : other + (int)(rng() % kLocal); break;
                    case kFew: {
                        static const int pool[] = { 3, 4, 31, 40, 70, 100 };   /* rank 0's 3, 4, 31 */
                        e = (pool[rng() % 6] + base) % kGlobal;
                        break;
                    }
                    default: e = (int)(rng() % kGlobal); break;
                }
                bool dup = false;
                for (int j = 0; j < s; ++j) dup |= ids[(size_t)t * kTop + j] == e;
                if (!dup) break;
            }
            ids[(size_t)t * kTop + s] = e;
        }
    return ids;
}

RadTensor tensor(void* p, uint32_t dt, std::vector<int64_t> shape, std::vector<int64_t> stride) {
    RadTensor t{};
    t.data = p; t.dtype = dt; t.rank = (uint32_t)shape.size();
    for (size_t i = 0; i < shape.size(); ++i) { t.shape[i] = shape[i]; t.stride[i] = stride[i]; }
    return t;
}
RadParam pi(const char* k, long long v) { RadParam p{}; p.key = k; p.kind = RAD_P_INT; p.ival = v; return p; }
RadParam ps(const char* k, const char* v) { RadParam p{}; p.key = k; p.kind = RAD_P_STR; p.sval = v; return p; }

/* One projection's call: activation rows `x` (pitch xp), the experts' table, the routing, output
 * `y` (pitch yp). `served` runs the registered entry (the switch decides), else the checker's
 * forced `mode` with the probe. */
struct Call {
    const char* order;      /* "token" (gate_up) or "sorted" (down) */
    int N, K, ne, top;
    const uint16_t* x; int AR, xp;
    const uint8_t* const* table;
    const Routing* R;
    uint16_t* y; int yp;
};
int run(const Call& c, bool served, int mode, unsigned long long* probe, bool gemv = true) {
    RadTensor t[5] = { tensor((void*)c.x, RAD_BF16, { c.AR, c.K }, { c.xp, 1 }),
                       tensor((void*)c.table, RAD_U8, { c.ne, c.N, (int64_t)c.K / 32 * 17 },
                              { 0, (int64_t)c.K / 32 * 17, 1 }),
                       tensor((void*)c.R->sorted.data(), RAD_I32, { c.R->T }, { 1 }),
                       tensor((void*)c.R->off.data(), RAD_I32, { (int64_t)c.R->off.size() }, { 1 }),
                       tensor(c.y, RAD_BF16, { c.R->T, c.N }, { c.yp, 1 }) };
    const RadParam p[6] = { ps("type", "mxfp4"), pi("N", c.N), pi("K", c.K), pi("n_expert", c.ne),
                            pi("top_k", c.top), ps("a_order", c.order) };
    RadArgs a{};
    a.t = t; a.n_t = 5; a.p = p; a.n_p = 6;
    if (served) return r4d_rad_moe_gemm_mxfp4_gemv(&a, nullptr);
    return r4d_rad_moe_gemm_mxfp4_draft_probe(&a, nullptr, gemv ? 1 : 0, mode, probe);
}

/* The expected output: each slot's row through its expert, zeros for a dropped slot. `experts`
 * holds each local expert's matrix. Past 16 tokens the tiled path's sequence. */
std::vector<uint16_t> expect(const Call& c, const std::vector<const uint8_t*>& experts) {
    const Routing& R = *c.R;
    const bool tiled = R.m > 16;
    std::vector<uint16_t> y((size_t)R.T * c.N, 0);
    const int rb = c.K / 32 * 17;
    for (int e = 0; e < c.ne; ++e)
        for (int i = R.off[(size_t)e]; i < R.off[(size_t)e + 1]; ++i) {
            const int flat = R.sorted[(size_t)i];
            if (flat < 0) continue;
            const int row = !std::strcmp(c.order, "sorted") ? i : flat / c.top;
            for (int n = 0; n < c.N; ++n)
                y[(size_t)i * c.N + n] = (tiled ? ref_dot_seq : ref_dot)(experts[(size_t)e] + (size_t)n * rb,
                                                                         c.x + (size_t)row * c.xp, c.K);
        }
    return y;
}

constexpr uint16_t kPoison = 0x7777;
/* Rows [0, T) x [0, N) against `want`, the pitch's padding untouched. Returns the mismatches. */
size_t compare(const Call& c, const std::vector<uint16_t>& want, std::string* first) {
    size_t bad = 0;
    for (int i = 0; i < c.R->T; ++i)
        for (int n = 0; n < c.yp; ++n) {
            const uint16_t got = c.y[(size_t)i * c.yp + n];
            const uint16_t w = n < c.N ? want[(size_t)i * c.N + n] : kPoison;
            if (got == w) continue;
            if (!bad++ && first) {
                char b[160];
                std::snprintf(b, sizeof b, "row %d col %d: got 0x%04x want 0x%04x%s", i, n, got, w,
                              n >= c.N ? " (padding)" : c.R->sorted[(size_t)i] < 0 ? " (dropped slot)" : "");
                *first = b;
            }
        }
    return bad;
}

/* The experts mx_range / ds41_live_tail::range give rows, with their row counts. */
std::vector<int> live_runs(const Routing& R, bool tail) {
    std::vector<int> runs;
    for (int e = 0; e < kLocal; ++e) {
        int a = e == 0 ? 0 : R.off[(size_t)e], b = tail || e < kLocal - 1 ? R.off[(size_t)e + 1] : R.T;
        a = std::min(std::max(a, 0), R.T);
        b = std::min(std::max(b, a), R.T);
        if (b > a) runs.push_back(b - a);
    }
    return runs;
}

int g_fails = 0, g_checks = 0;
void fail(const std::string& what) {
    if (g_fails++ < 24) std::printf("FAIL %s\n", what.c_str());
}

}  // namespace

int main() {
    /* Before any call: the switch is read once, at the first draft-shaped one. The trunk's live
     * tail stays off, so its shape runs the old path. */
    setenv("RADIANCE_EXPERIMENTAL_DS41_DRAFT_GEMV", "1", 1);
    unsetenv("RADIANCE_EXPERIMENTAL_DS41_LIVE_TAIL");

    {
        const char* const no[] = { nullptr, "", "0", "3", "12", "1 ", "on" };
        for (const char* s : no)
            if (ds41_live_tail::draft_mode(s)) fail(std::string("draft_mode(\"") + (s ? s : "null") + "\") is not off");
        if (ds41_live_tail::draft_mode("1") != (ds41_live_tail::kTail | ds41_live_tail::kOnePass) ||
            ds41_live_tail::draft_mode("2") != ds41_live_tail::kTail)
            fail("draft_mode: 1 is the tail and the one pass, 2 the tail alone");
    }

    std::mt19937_64 rng(4141);
    const int sizes[] = { 1, 2, 5, 6, 8, 16, 20 };
    /* passes over the weights, gate_up and down, summed by mode over the gemv cases */
    unsigned long long passes[2][4] = {};
    int case_no = 0;
    for (int m : sizes)
        for (int p = 0; p < kPatterns; ++p) {
            if (m == 20 && p % 2) continue;   /* the tiled path: three patterns */
            /* N 1024 (two weight rows a wave) at up to 8 query rows: the old path computes every
             * dropped slot against the last expert, which at 16 rows and N 1024 is most of the run */
            const int rank = case_no % 4, N = case_no % 4 == 3 && m <= 8 ? 1024 : 64;
            ++case_no;
            const std::vector<int> ids = route((Pattern)p, m, rank, rng);
            const Routing R = scatter(ids, m, rank);
            /* every table entry readable (the old path reads the last expert for dropped slots
             * whether or not it has rows); experts without rows share one matrix */
            for (int q = 0; q < 2; ++q) {
                const bool gu = q == 0;
                const int K = gu ? kEmbd : kFf;
                std::vector<std::vector<uint8_t>> own;
                const std::vector<uint8_t> filler = make_expert(N, K, rng);
                std::vector<const uint8_t*> experts(kLocal, filler.data());
                for (int e = 0; e < kLocal; ++e)
                    if (R.off[(size_t)e + 1] > R.off[(size_t)e] || e == kLocal - 1) {
                        own.push_back(make_expert(N, K, rng));
                        experts[(size_t)e] = own.back().data();
                    }
                const int AR = gu ? m : R.T, xp = K + (case_no % 2 ? 3 : 8), yp = N + (q ? 5 : 8);
                const std::vector<uint16_t> x = make_rows(AR, xp, rng);
                std::vector<uint16_t> y((size_t)R.T * yp);
                Call c{ gu ? "token" : "sorted", N, K, kLocal, kTop, x.data(), AR, xp, experts.data(), &R, y.data(), yp };
                const std::vector<uint16_t> want = expect(c, experts);
                const bool gemv = m <= 16;
                const int XM = std::min(m, 8), XLM = std::min(9216 / (K + K / 8), XM);
                for (int mode = 0; mode < 4; ++mode) {
                    std::fill(y.begin(), y.end(), kPoison);
                    unsigned long long pr[6] = {};
                    char what[200];
                    std::snprintf(what, sizeof what, "%s m %d rank %d N %d %s, mode %d", gu ? "gate_up" : "down", m,
                                  rank, N, kPatternName[p], mode);
                    ++g_checks;
                    if (run(c, false, mode, pr, true) != 0) { fail(std::string(what) + ": refused"); continue; }
                    std::string first;
                    if (const size_t bad = compare(c, want, &first))
                        fail(std::string(what) + ": " + std::to_string(bad) + " values differ, first " + first);
                    /* the work: the tail's rows and zero rows, the passes */
                    const bool tail = mode & ds41_live_tail::kTail, one = mode & ds41_live_tail::kOnePass;
                    const std::vector<int> runs = live_runs(R, tail);
                    unsigned long long owned = 0, want_passes = 0;
                    const int step = !one && XLM >= 1 && XLM < XM ? XLM : XM;
                    for (int r : runs) { owned += (unsigned long long)r; want_passes += (unsigned long long)((r + step - 1) / step); }
                    const unsigned long long zero_rows = tail ? (unsigned long long)(R.T - R.off[kLocal]) : 0;
                    if (pr[0] != owned)
                        fail(std::string(what) + ": the live experts own " + std::to_string(pr[0]) + " rows, want " + std::to_string(owned));
                    if (tail && (pr[2] != zero_rows || pr[3] != zero_rows * (unsigned long long)N))
                        fail(std::string(what) + ": the zero tail wrote " + std::to_string(pr[2]) + " rows, want " + std::to_string(zero_rows));
                    if (!tail && pr[2]) fail(std::string(what) + ": a zero tail without the tail");
                    if (gemv && pr[5] != want_passes)
                        fail(std::string(what) + ": " + std::to_string(pr[5]) + " weight passes, want " + std::to_string(want_passes));
                    if (gemv && one) {
                        bool fits = true;
                        for (int r : runs) fits &= r <= XM;
                        if (fits && pr[5] != runs.size())
                            fail(std::string(what) + ": one pass read an expert's weights more than once");
                    }
                    if (gemv) passes[q][mode] += pr[5];
                }
            }
        }

    /* The served entry under the switch, at the real shapes: rank 1, five query rows -- two
     * experts with two rows each, one with one, the last expert idle, ten slots elsewhere. */
    {
        unsigned long long c0[5], c1[5];
        r4d_ds41_draft_gemv_counts(c0);
        const int m = 5, rank = 1;
        std::vector<int> ids = { 33, 70, 5, 34, 100, 6, 33, 64, 120, 35, 0, 90, 34, 1, 2 };
        const Routing R = scatter(ids, m, rank);
        std::vector<uint16_t> act;
        for (int q = 0; q < 2; ++q) {
            const bool gu = q == 0;
            const int N = gu ? 2 * kFf : kEmbd, K = gu ? kEmbd : kFf;
            std::vector<std::vector<uint8_t>> own;
            std::vector<const uint8_t*> experts(kLocal);
            const std::vector<uint8_t> filler = make_expert(N, K, rng);
            for (int e = 0; e < kLocal; ++e) {
                experts[(size_t)e] = filler.data();
                if (R.off[(size_t)e + 1] > R.off[(size_t)e]) { own.push_back(make_expert(N, K, rng)); experts[(size_t)e] = own.back().data(); }
            }
            const int AR = gu ? m : R.T, xp = K + 8, yp = N + 8;
            const std::vector<uint16_t> x = make_rows(AR, xp, rng);
            std::vector<uint16_t> y((size_t)R.T * yp, kPoison);
            Call c{ gu ? "token" : "sorted", N, K, kLocal, kTop, x.data(), AR, xp, experts.data(), &R, y.data(), yp };
            ++g_checks;
            const char* what = gu ? "served gate_up 4608 x 5120" : "served down 5120 x 2304";
            if (run(c, true, 0, nullptr) != 0) { fail(std::string(what) + ": refused"); continue; }
            std::string first;
            if (const size_t bad = compare(c, expect(c, experts), &first))
                fail(std::string(what) + ": " + std::to_string(bad) + " values differ, first " + first);
        }
        r4d_ds41_draft_gemv_counts(c1);
        if (c1[0] - c0[0] != 1 || c1[1] - c0[1] != 1 || c1[2] - c0[2] != 10 || c1[3] != c0[3] || c1[4] != c0[4])
            fail("the served draft calls were not counted as the switch's: gate_up +" + std::to_string(c1[0] - c0[0]) +
                 ", down +" + std::to_string(c1[1] - c0[1]) + ", rows +" + std::to_string(c1[2] - c0[2]));

        /* The trunk's shape (96 experts, top-6, one token on rank 0): the switch leaves it alone. */
        r4d_ds41_draft_gemv_counts(c0);
        std::vector<int32_t> sorted = { 2, 4, -1, -1, -1, -1 }, off(97, 0);
        for (int e = 0; e < 96; ++e) off[(size_t)e + 1] = off[(size_t)e] + (e == 10 || e == 95 ? 1 : 0);
        Routing T1;
        T1.m = 1; T1.T = 6; T1.sorted = sorted; T1.off = off;
        const int N = 2 * kFf, K = kEmbd;
        std::vector<std::vector<uint8_t>> own = { make_expert(N, K, rng), make_expert(N, K, rng) };
        std::vector<const uint8_t*> experts(96, own[0].data());
        experts[95] = own[1].data();
        const std::vector<uint16_t> x = make_rows(1, K, rng);
        std::vector<uint16_t> y((size_t)6 * N, kPoison);
        Call c{ "token", N, K, 96, 6, x.data(), 1, K, experts.data(), &T1, y.data(), N };
        ++g_checks;
        if (run(c, true, 0, nullptr) != 0) {
            fail("served trunk gate_up: refused");
        } else {
            std::string first;
            if (const size_t bad = compare(c, expect(c, experts), &first))
                fail("served trunk gate_up: " + std::to_string(bad) + " values differ, first " + first);
        }
        r4d_ds41_draft_gemv_counts(c1);
        for (int i = 0; i < 5; ++i)
            if (c1[i] != c0[i]) fail("the trunk's 96 / top-6 call was counted as a draft call");
    }

    std::printf("draft GEMV modes: %d cases, %d checks; weight passes gate_up old %llu, tail %llu, one pass %llu, "
                "both %llu; down %llu, %llu, %llu, %llu\n",
                case_no, g_checks, passes[0][0], passes[0][1], passes[0][2], passes[0][3], passes[1][0],
                passes[1][1], passes[1][2], passes[1][3]);
    if (g_fails)
        std::printf("FAIL: %d checks\n", g_fails);
    else
        std::printf("PASS: every mode writes the decode GEMV's bytes, and does the work it should\n");
    return g_fails ? 1 : 0;
}
