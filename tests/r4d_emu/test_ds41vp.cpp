/* tests/r4d_emu: moe_gemm_ggml_mxfp4_gemv_vp (r4d_ds41_mxfp4.hip) on the host SIMT emulator, held to the
 * plain gemv row bit for bit (RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED, core/place/vram_packed.h).
 *
 * Each case makes experts as the release stores them (a codes plane and an e8m0 plane) and calls both rows
 * over the same experts: the plain row over block_mxfp4 entries, gemv_vp over a table that mixes
 * block_mxfp4 entries with entries marked by bit 0 that name the packed-scale form, packed by
 * tools/ds41-host-experts' pack_planes -- the code the engine packs its VRAM residents with. Every output
 * must be the same bf16 bits, and gemv_vp over the block table alone must be the plain row too.
 *
 * The cases reach every way the kernel walks a row: whole K and K in chunks (rows past the LDS), one and
 * two weight rows a wave (N >= 1024), one to sixteen token rows (one pass, whole sub-passes), token and
 * sorted order, slots dropped to other ranks; full rounds and a last partial round (UK % 32 of 1, 8, 9 and
 * 31, and a row of the partial round alone), raw rounds at random positions of every round including the
 * partial one, none at all and nearly all, a form with room to spare (its sections moved); e8m0 edges (0,
 * 1, 2, 127, 254, 255) and NaN, inf and subnormal activations; the release's own scales at the served
 * shapes (the research sample, when its directory is there: DS41_HX_SAMPLE or the default). A marked
 * entry whose header is damaged must give NaN rows and leave every other expert's rows as they were.
 *
 * `r4d_emu_ds41vp tail` runs every case with the live-tail ranges on (RADIANCE_EXPERIMENTAL_DS41_LIVE_TAIL=1,
 * which libr4d reads once a process): the cases at their geometry (r4d_ds41_live_tail_policy.h: 96 experts
 * a rank, top-6, a served or a small checked shape) then take mx_moe's tail branch, gemv_vp's kernel over the
 * routed rows and mx_tail_zero's over the rows past them, and those rows must be zeros in every output. */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"
#include "r4d_ds41_live_tail_policy.h"
#include "ds41_host_experts.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <memory>
#include <random>
#include <string>
#include <vector>

extern const char* emu_last_kernel;
extern int emu_grids;

namespace {

/* `tail`: the live-tail ranges are on for the whole run; the cases whose launches took the tail branch. */
bool g_tail = false;
int g_tail_cases = 0;

uint8_t* aligned_bytes(size_t n) {
    const size_t a = (n + 63) / 64 * 64;
    void* p = std::aligned_alloc(64, a ? a : 64);
    if (!p) { std::printf("out of memory\n"); std::exit(2); }
    std::memset(p, 0xA5, a ? a : 64);
    return (uint8_t*)p;
}

/* One expert weight as the release stores it, and in both forms the card can hold. */
struct Expert {
    int N = 0, K = 0;
    std::vector<uint8_t> codes, scales;   /* [N, K / 2], value 2i in the low nibble; [N, K / 32] */
    uint8_t* block = nullptr;             /* block_mxfp4, 64-byte aligned */
    uint8_t* packed = nullptr;            /* the packed-scale form, 64-byte aligned */
    size_t packed_bytes = 0;
    int n_exc = 0;
    Expert() = default;
    Expert(const Expert&) = delete;
    Expert& operator=(const Expert&) = delete;
    ~Expert() { std::free(block); std::free(packed); }

    /* block_mxfp4 from the planes: e, then qs[j] = value j | value j + 16 << 4. */
    void make_block() {
        const int UK = K / 32;
        block = aligned_bytes((size_t)N * UK * 17);
        for (int n = 0; n < N; ++n)
            for (int u = 0; u < UK; ++u) {
                uint8_t* b = block + ((size_t)n * UK + u) * 17;
                const uint8_t* c = codes.data() + (size_t)n * (K / 2) + (size_t)u * 16;
                auto val = [&](int v) { return (c[v >> 1] >> (4 * (v & 1))) & 15; };
                b[0] = scales[(size_t)n * UK + u];
                for (int j = 0; j < 16; ++j) b[1 + j] = (uint8_t)(val(j) | (val(j + 16) << 4));
            }
    }
    /* The packed form with room for the weight's own raw rounds plus `extra`, checked as the engine checks
     * it at load. */
    bool make_packed(int extra) {
        ds41hx::PackInfo plan;
        if (!ds41hx::pack_plan(scales.data(), N, K, &plan)) return false;
        n_exc = plan.n_exc;
        const int UK = K / 32, R = (UK + 31) / 32;
        const int cap = std::min(plan.n_exc + extra, N * R);
        packed_bytes = ds41hx::packed_bytes(N, K, cap);
        if (!packed_bytes) return false;
        std::free(packed);
        packed = aligned_bytes(packed_bytes);
        if (ds41hx::pack_planes(codes.data(), scales.data(), N, K, cap, plan, packed, packed_bytes) !=
            ds41hx::kPackOk)
            return false;
        ds41hx::Mx v;
        return ds41hx::packed_view(packed, packed_bytes, N, K, &v) && ds41hx::packed_scales_equal(v, scales.data());
    }
};

/* Scales as the release has them (120 or 121, a few 119 or 122); in a share `raw` of the rounds one block
 * far outside the window (a raw round), and a share `edge` of all blocks at e8m0 edges. Codes random. */
void fill(Expert& x, std::mt19937_64& rng, double raw, double edge) {
    const int UK = x.K / 32, R = (UK + 31) / 32;
    x.codes.resize((size_t)x.N * (x.K / 2));
    x.scales.resize((size_t)x.N * UK);
    for (auto& c : x.codes) c = (uint8_t)rng();
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    for (auto& s : x.scales) {
        const double r = u01(rng);
        s = r < 0.58 ? 120 : r < 0.995 ? 121 : r < 0.998 ? 119 : 122;
    }
    static const uint8_t edges[] = { 0, 1, 2, 3, 100, 127, 128, 135, 254, 255 };
    for (int n = 0; n < x.N; ++n)
        for (int r = 0; r < R; ++r) {
            if (u01(rng) >= raw) continue;
            const int t = r < UK / 32 ? 32 : UK % 32;
            x.scales[(size_t)n * UK + 32 * r + (size_t)(rng() % (uint64_t)t)] = edges[rng() % sizeof edges];
        }
    for (auto& s : x.scales)
        if (u01(rng) < edge) s = edges[rng() % sizeof edges];
}

uint16_t f2bf(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

/* bf16 NaNs compare as one value: the host and the card may differ in a NaN's sign and payload. */
bool nan16(uint16_t v) { return (v & 0x7f80u) == 0x7f80u && (v & 0x7fu); }

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

typedef int (*Launch)(const RadArgs*, RadStream);

/* `tokens` token rows each picking top_k distinct experts of ne + other (the other ranks' experts, whose
 * slots this rank drops), this rank's rows in moe_scatter's order and -1 past them. */
struct Route {
    int T = 0;
    std::vector<int32_t> sorted, off;
};
Route route(std::mt19937_64& rng, int tokens, int ne, int other, int top_k) {
    std::vector<std::vector<int32_t>> rows((size_t)ne);
    for (int t = 0; t < tokens; ++t) {
        std::vector<int> pick = { (int)(rng() % (uint64_t)ne) };   /* at least one of this rank's */
        while ((int)pick.size() < top_k) {
            const int e = (int)(rng() % (uint64_t)(ne + other));
            if (std::find(pick.begin(), pick.end(), e) == pick.end()) pick.push_back(e);
        }
        for (int k = 0; k < top_k; ++k)
            if (pick[(size_t)k] < ne) rows[(size_t)pick[(size_t)k]].push_back(t * top_k + k);
    }
    Route r;
    r.off.assign((size_t)ne + 1, 0);
    for (int e = 0; e < ne; ++e) {
        r.off[(size_t)e + 1] = r.off[(size_t)e] + (int)rows[(size_t)e].size();
        for (int32_t f : rows[(size_t)e]) r.sorted.push_back(f);
    }
    r.T = tokens * top_k;
    r.sorted.resize((size_t)r.T, -1);
    return r;
}

int call(Launch f, const std::vector<uint16_t>& x, int AR, int K, std::vector<const void*>& table, int N,
         int top_k, const Route& r, const char* order, std::vector<uint16_t>* y) {
    const int ne = (int)table.size();
    y->assign((size_t)r.T * N, 0xDEADu);
    std::vector<int32_t> sorted = r.sorted, off = r.off;
    RadTensor t[5] = { ten((void*)x.data(), RAD_BF16, { AR, K }),
                       ten(table.data(), RAD_U8, { ne, N, K / 32 * 17 }),
                       ten(sorted.data(), RAD_I32, { r.T }), ten(off.data(), RAD_I32, { ne + 1 }),
                       ten(y->data(), RAD_BF16, { r.T, N }) };
    t[1].stride[0] = 0;   /* a table of expert pointers */
    RadParam p[6] = { pstr("type", "mxfp4"), pint("N", N), pint("K", K), pint("n_expert", ne),
                      pint("top_k", top_k), pstr("a_order", order) };
    RadArgs a{};
    a.t = t;
    a.n_t = 5;
    a.p = p;
    a.n_p = 6;
    return f(&a, nullptr);
}

struct Case {
    const char* name;
    int N, K, ne, other, top_k, tokens;
    bool sorted;
    double raw, edge;
    int pack_every;   /* every pack_every-th expert's entry marked packed (1: all) */
    int extra;        /* raw rounds of room past each weight's own */
    bool damage;      /* the first marked entry's header damaged */
    bool special_x;   /* NaN, inf and subnormal activations */
};

int fails = 0, cases = 0;

/* Both rows over `ex` (the experts this rank holds, entry e = ex[e]); 0 when every bit agrees. */
void check(const Case& c, std::mt19937_64& rng, const std::vector<Expert*>& ex) {
    ++cases;
    const int ne = (int)ex.size();
    std::vector<const void*> plain((size_t)ne), vp((size_t)ne);
    std::vector<std::unique_ptr<uint8_t, decltype(&std::free)>> damaged;
    int packed = 0, raw_rounds = 0, damaged_expert = -1;
    for (int e = 0; e < ne; ++e) {
        Expert& x = *ex[(size_t)e];
        plain[(size_t)e] = x.block;
        const bool pk = c.pack_every <= 1 || e % c.pack_every == 0;
        vp[(size_t)e] = pk ? (const void*)((uintptr_t)x.packed | 1u) : (const void*)x.block;
        packed += pk;
        raw_rounds += pk ? x.n_exc : 0;
        if (pk && c.damage && damaged_expert < 0) {
            uint8_t* copy = aligned_bytes(x.packed_bytes);
            std::memcpy(copy, x.packed, x.packed_bytes);
            copy[0] ^= 0x10;   /* the magic */
            damaged.emplace_back(copy, &std::free);
            vp[(size_t)e] = (const void*)((uintptr_t)copy | 1u);
            damaged_expert = e;
        }
    }
    const Route r = route(rng, c.tokens, ne, c.other, c.top_k);
    const int AR = c.sorted ? r.T : c.tokens;
    std::vector<uint16_t> xin((size_t)AR * c.K);
    std::normal_distribution<float> nd(0.f, 0.7f);
    for (auto& v : xin) v = f2bf(nd(rng));
    if (c.special_x) {
        static const uint16_t sp[] = { 0x7fc0, 0xffc0, 0x7f80, 0xff80, 0x0001, 0x8001, 0x0080, 0x8000, 0x7f7f };
        for (auto& v : xin)
            if (rng() % 97 == 0) v = sp[rng() % (sizeof sp / 2)];
    }
    const char* order = c.sorted ? "sorted" : "token";
    /* mx_moe's own test of the live-tail geometry, on the arguments `call` passes (tokens is a pass's token
     * rows in either order); with the ranges on, these launches take the tail branch. */
    const bool tail = g_tail && ds41_live_tail::geometry(c.tokens, c.N, c.K, ne, c.top_k, r.T, AR, c.sorted);
    std::vector<uint16_t> y_plain, y_vp, y_vp_block;
    int g0 = emu_grids;
    const int rc1 = call(r4d_rad_moe_gemm_mxfp4_gemv, xin, AR, c.K, plain, c.N, c.top_k, r, order, &y_plain);
    const int n1 = emu_grids - g0;
    const std::string k1 = emu_last_kernel ? emu_last_kernel : "?";
    g0 = emu_grids;
    const int rc2 = call(r4d_rad_moe_gemm_mxfp4_gemv_vp, xin, AR, c.K, vp, c.N, c.top_k, r, order, &y_vp);
    const int n2 = emu_grids - g0;
    const std::string k2 = emu_last_kernel ? emu_last_kernel : "?";
    g0 = emu_grids;
    const int rc3 = call(r4d_rad_moe_gemm_mxfp4_gemv_vp, xin, AR, c.K, plain, c.N, c.top_k, r, order, &y_vp_block);
    const int n3 = emu_grids - g0;
    const std::string k3 = emu_last_kernel ? emu_last_kernel : "?";
    if (rc1 != RAD_OK || rc2 != RAD_OK || rc3 != RAD_OK) {
        std::printf("FAIL %s: the launches returned %d, %d, %d\n", c.name, rc1, rc2, rc3);
        ++fails;
        return;
    }
    if (tail) {
        /* The tail branch: the gemv, then mx_tail_zero's kernel last, in each of the three calls. Which gemv
         * kernel ran is in the values: the plain kernel would read a marked entry's packed bytes as
         * block_mxfp4 and give other bits, and a damaged header NaN rows only through gemv_vp's. */
        const char* z = "mx_zero_tail_kernel";
        if (n1 != 2 || n2 != 2 || n3 != 2 || k1.find(z) == std::string::npos || k2.find(z) == std::string::npos ||
            k3.find(z) == std::string::npos) {
            std::printf("FAIL %s: the live-tail calls launched %d, %d, %d kernels, the last %s, %s, %s\n", c.name,
                        n1, n2, n3, k1.c_str(), k2.c_str(), k3.c_str());
            ++fails;
            return;
        }
        ++g_tail_cases;
    } else if (n1 != 1 || n2 != 1 || n3 != 1 || k2.find("false, true>") == std::string::npos ||
               k3.find("false, true>") == std::string::npos || k1.find("false, true>") != std::string::npos) {
        /* One launch each: gemv_vp's marked-table kernel for both of its tables, and the plain row's own. */
        std::printf("FAIL %s: the rows ran %s (%d launches), %s (%d) and %s (%d)\n", c.name, k1.c_str(), n1,
                    k2.c_str(), n2, k3.c_str(), n3);
        ++fails;
        return;
    }
    std::vector<uint8_t> damaged_row((size_t)r.T, 0);
    if (damaged_expert >= 0)
        for (int i = r.off[(size_t)damaged_expert]; i < r.off[(size_t)damaged_expert + 1]; ++i)
            damaged_row[(size_t)i] = 1;
    size_t diff = 0, diff_block = 0, nan_rows = 0, nan_bad = 0, live = 0, dead = 0, dead_bad = 0;
    for (int i = 0; i < r.T; ++i)
        for (int n = 0; n < c.N; ++n) {
            const size_t at = (size_t)i * c.N + n;
            const uint16_t a = y_plain[at], b = y_vp[at], d = y_vp_block[at];
            if (a != d && !(nan16(a) && nan16(d))) ++diff_block;
            /* A dropped slot (another rank's expert): past the routed rows, so the tail zeroes it. */
            if (r.sorted[(size_t)i] < 0) {
                ++dead;
                if (tail && (a | b | d)) ++dead_bad;
            }
            if (damaged_row[(size_t)i]) {
                if (nan16(b)) ++nan_rows;
                else ++nan_bad;
                continue;
            }
            live += r.sorted[(size_t)i] >= 0;
            if (a != b && !(nan16(a) && nan16(b))) {
                if (diff < 4) std::printf("  %s: row %d col %d plain %04x gemv_vp %04x\n", c.name, i, n, a, b);
                ++diff;
            }
        }
    const bool ok = !diff && !diff_block && !nan_bad && !dead_bad && (damaged_expert < 0 || nan_rows > 0) &&
                    live > 0;
    char what[160];
    if (tail)
        std::snprintf(what, sizeof what, "the live-tail branch (its gemv, then %zu dropped slots' values zeroed)",
                      dead);
    else
        std::snprintf(what, sizeof what, "the plain row's kernel %s", k1.c_str());
    std::printf("%s %s: N %d K %d, %d token(s) in %s order, %d of %d entries packed (%d raw rounds), %zu live "
                "values%s, %s\n",
                ok ? "pass" : "FAIL", c.name, c.N, c.K, c.tokens, order, packed, ne, raw_rounds, live,
                damaged_expert >= 0 ? " and the damaged entry's rows NaN" : "", what);
    if (!ok) {
        std::printf("  %zu values differ from the plain row, %zu over the block table, %zu of the damaged entry's "
                    "not NaN (%zu NaN), %zu live, %zu of the %zu dropped slots' values not zero\n", diff,
                    diff_block, nan_bad, nan_rows, live, dead_bad, dead);
        ++fails;
    }
}

void synthetic(const Case& c, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<std::unique_ptr<Expert>> own;
    std::vector<Expert*> ex;
    for (int e = 0; e < c.ne; ++e) {
        own.emplace_back(new Expert());
        Expert& x = *own.back();
        x.N = c.N;
        x.K = c.K;
        fill(x, rng, c.raw, c.edge);
        x.make_block();
        if (!x.make_packed(c.extra)) {
            std::printf("FAIL %s: expert %d does not pack\n", c.name, e);
            ++fails;
            return;
        }
        ex.push_back(&x);
    }
    check(c, rng, ex);
}

/* The research sample (tools/ds41-host-experts/ds41_host_experts_test.cpp, sample_ids): each expert's
 * three e8m0 planes whole and the first MiB of each codes plane, as L<layer>_E<expert>_w<1|2|3>.{scale,
 * codes}; gate_up is [w1; w3]. Codes past the sample's are random. */
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
void sample_rows(Expert& x, int r0, int rows, const std::vector<uint8_t>& sc, const std::vector<uint8_t>& codes,
                 std::mt19937_64& rng) {
    const int UK = x.K / 32, CB = x.K / 2;
    std::memcpy(x.scales.data() + (size_t)r0 * UK, sc.data(), (size_t)rows * UK);
    for (size_t i = 0; i < (size_t)rows * CB; ++i)
        x.codes[(size_t)r0 * CB + i] = i < codes.size() ? codes[i] : (uint8_t)rng();
}
int real_sample(std::mt19937_64& rng) {
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
    if (ids.empty()) {
        std::printf("note: no research sample in %s, so the release's own scales were not run\n", dir.c_str());
        return 0;
    }
    constexpr int kEmbd = 5120, kFf = 2304;
    /* The worst two by raw rounds and one plain: three experts at the served shapes, room 512 and 320 as the
     * engine's budget gives them (falling back to their own when they need more). */
    std::vector<std::unique_ptr<Expert>> gus, dns;
    for (const auto& id : ids) {
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
        gus.emplace_back(new Expert());
        Expert& g = *gus.back();
        g.N = 2 * kFf; g.K = kEmbd;
        g.codes.resize((size_t)g.N * g.K / 2);
        g.scales.resize((size_t)g.N * (g.K / 32));
        sample_rows(g, 0, kFf, s1, c1, rng);
        sample_rows(g, kFf, kFf, s3, c3, rng);
        dns.emplace_back(new Expert());
        Expert& d = *dns.back();
        d.N = kEmbd; d.K = kFf;
        d.codes.resize((size_t)d.N * d.K / 2);
        d.scales.resize((size_t)d.N * (d.K / 32));
        sample_rows(d, 0, kEmbd, s2, c2, rng);
        ds41hx::PackInfo pg, pd;
        ds41hx::pack_plan(g.scales.data(), g.N, g.K, &pg);
        ds41hx::pack_plan(d.scales.data(), d.N, d.K, &pd);
        g.n_exc = pg.n_exc;
        d.n_exc = pd.n_exc;
    }
    if (gus.empty()) return 0;
    std::vector<size_t> idx(gus.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
        return gus[a]->n_exc + dns[a]->n_exc > gus[b]->n_exc + dns[b]->n_exc;
    });
    std::vector<Expert*> g3, d3;
    for (size_t k = 0; k < idx.size() && g3.size() < 3; ++k) {
        const size_t i = k < 2 ? idx[k] : idx.back();
        Expert& g = *gus[i];
        Expert& d = *dns[i];
        g.make_block();
        d.make_block();
        if (!g.make_packed(std::max(0, 512 - g.n_exc)) || !d.make_packed(std::max(0, 320 - d.n_exc))) {
            std::printf("FAIL sample: L%02d E%03d does not pack\n", ids[i].first, ids[i].second);
            ++fails;
            return 0;
        }
        std::printf("sample L%02d E%03d: %d raw rounds in its gate_up, %d in its down\n", ids[i].first,
                    ids[i].second, g.n_exc, d.n_exc);
        g3.push_back(&g);
        d3.push_back(&d);
    }
    /* Decode: one token, top-6 over this rank's experts and three others' (the dropped slots). */
    const Case cg = { "release scales, gate_up", 2 * kFf, kEmbd, (int)g3.size(), 6, 6, 1, false, 0, 0, 1, 0,
                      false, false };
    check(cg, rng, g3);
    const Case cd = { "release scales, down", kEmbd, kFf, (int)d3.size(), 6, 6, 1, true, 0, 0, 1, 0, false,
                      false };
    check(cd, rng, d3);
    const Case cd2 = { "release scales, down, two tokens, one entry block", kEmbd, kFf, (int)d3.size(), 2, 4, 2,
                       true, 0, 0, 2, 0, false, false };
    check(cd2, rng, d3);
    /* A rank's whole table, 96 entries (the three experts behind every third), top-6 over all 384: the served
     * decode calls at the live-tail geometry, so under `tail` the tail branch at the served shapes. */
    std::vector<Expert*> g96, d96;
    for (int e = 0; e < 96; ++e) {
        g96.push_back(g3[(size_t)e % g3.size()]);
        d96.push_back(d3[(size_t)e % d3.size()]);
    }
    const Case cg96 = { "release scales, gate_up, a rank's 96 entries", 2 * kFf, kEmbd, 96, 288, 6, 1, false, 0, 0, 1,
                        0, false, false };
    check(cg96, rng, g96);
    const Case cd96 = { "release scales, down, a rank's 96 entries, every other one block", kEmbd, kFf, 96, 288, 6,
                        2, true, 0, 0, 2, 0, false, false };
    check(cd96, rng, d96);
    return (int)g3.size();
}

}  /* namespace */

int main(int argc, char** argv) {
    /* The rows as the A/B serves them, the live-tail ranges off (their own switch); `tail`, on. libr4d reads
     * the switch once a process, so each is its own run (ctest: r4d_emu_ds41vp, r4d_emu_ds41vp_tail). */
    g_tail = argc > 1 && !std::strcmp(argv[1], "tail");
    if (g_tail) setenv("RADIANCE_EXPERIMENTAL_DS41_LIVE_TAIL", "1", 1);
    else unsetenv("RADIANCE_EXPERIMENTAL_DS41_LIVE_TAIL");
    /* name, N, K, experts, others', top_k, tokens, sorted, raw, edge, pack every, extra room, damage, special x */
    const Case all[] = {
        { "one full round, decode", 64, 1024, 4, 4, 4, 1, false, 0.05, 0, 1, 0, false, false },
        { "partial round of 8 (down's K 2304)", 48, 2304, 4, 2, 3, 1, true, 0.3, 0, 1, 0, false, false },
        { "partial round of 1", 40, 1056, 3, 3, 3, 2, false, 0.5, 0, 1, 0, false, false },
        { "partial round of 9, room to spare", 40, 1312, 3, 3, 3, 3, true, 0.4, 0, 1, 7, false, false },
        { "partial round of 31", 24, 2016, 3, 1, 2, 2, false, 0.4, 0, 1, 0, false, false },
        { "the partial round alone", 33, 96, 4, 2, 3, 4, true, 0.6, 0, 1, 0, false, false },
        { "no raw rounds", 64, 2048, 4, 4, 4, 1, false, 0.0, 0, 1, 0, false, false },
        { "nearly every round raw", 32, 1024, 3, 3, 3, 2, false, 0.95, 0.02, 1, 0, false, false },
        { "e8m0 edges everywhere, special activations", 32, 1280, 3, 3, 3, 3, false, 0.3, 0.2, 1, 0, false,
          true },
        { "two weight rows a wave (N 1024)", 1024, 512, 3, 3, 3, 1, false, 0.2, 0, 1, 0, false, false },
        { "two weight rows a wave, three tokens", 1032, 320, 3, 3, 3, 3, true, 0.2, 0, 1, 0, false, false },
        { "sixteen tokens: two passes of whole rows", 48, 4096, 3, 1, 2, 16, false, 0.1, 0, 1, 0, false,
          false },
        { "sixteen sorted rows, every other entry block", 40, 2304, 4, 2, 3, 16, true, 0.2, 0, 2, 0, false,
          false },
        { "K in chunks (a row past the LDS)", 24, 9216, 2, 1, 2, 1, false, 0.2, 0, 1, 0, false, false },
        { "K in chunks with a partial round, 9 tokens", 16, 10272, 2, 1, 2, 9, false, 0.2, 0, 1, 3, false,
          false },
        { "every third entry block, dropped slots", 56, 1088, 6, 6, 4, 2, false, 0.3, 0, 3, 0, false, false },
        { "a damaged header gives NaN rows", 32, 1024, 4, 0, 4, 2, false, 0.2, 0, 1, 0, true, false },
        /* The live-tail geometry's small shapes: a rank's 96 experts, top-6 over the 384. */
        { "live-tail geometry [64, 256], decode", 64, 256, 96, 288, 6, 1, false, 0.3, 0, 1, 0, false, false },
        { "live-tail geometry [128, 512], four sorted rows, every other entry block", 128, 512, 96, 288, 6, 4,
          true, 0.2, 0, 2, 0, false, false },
        { "live-tail geometry, sixteen tokens and a damaged header", 64, 256, 96, 288, 6, 16, false, 0.2, 0, 1, 0,
          true, false },
    };
    uint64_t seed = 2026;
    for (const Case& c : all) synthetic(c, seed++);
    std::mt19937_64 rng(77);
    const int sampled = real_sample(rng);
    /* Under `tail` the branch must have run: the three small cases, and the two at the served shapes. */
    const int want_tail = 3 + (sampled ? 2 : 0);
    if (g_tail && g_tail_cases != want_tail) {
        std::printf("FAIL the live-tail branch ran in %d case(s), not %d\n", g_tail_cases, want_tail);
        ++fails;
    }
    char tl[96] = "";
    if (g_tail) std::snprintf(tl, sizeof tl, ", the live-tail ranges on: %d took the tail branch", g_tail_cases);
    std::printf("%d case(s), %d failed%s%s\n", cases, fails, tl,
                sampled ? "" : " (the release-scale cases did not run: no sample)");
    return fails ? 1 : 0;
}
