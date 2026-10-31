/* ds41_host_experts_test.cpp -- exactness checks and the throughput gate for ds41_host_experts.
 *
 *   ds41_host_experts_test check            scalar vs AVX2, ggml blocks vs planes vs packed: bit-identical
 *   ds41_host_experts_test packed [dir]     the packed form against the planes, bit for bit: real
 *                                           sampled experts (dir, default $DS41_HX_SAMPLE), a raw round
 *                                           at every block position, the general path, NaN rows, and
 *                                           the packing API's refusals
 *   ds41_host_experts_test bench [experts]  experts/s and weight GB/s at 1..20 pinned threads
 *   ds41_host_experts_test tiles [experts] [threads...]  as bench, 64-row tiles from a shared counter
 *   ds41_host_experts_test pkbench [experts] [threads...]  as tiles: block_mxfp4, packed, planes in turn
 *   ds41_host_experts_test hot [reps] [rows] [planes]  one matrix in cache, one thread (perf stat)
 *   ds41_host_experts_test rbench [experts] [threads] [block|packed|planes] [rows...]  the pool's shape
 *                                           at 1, 2, 4, 8, 16, 32 token rows an expert (up to 128):
 *                                           today's jobs of four rows against one job of every row
 *                                           (gemv_avx2_many) and the pool's even cuts at 32 and 64
 *                                           rows a job (..._MAX_ROWS)
 *   ds41_host_experts_test rbench-stream [experts] [threads] [packed|planes] [1]
 *                                           same working set, pool, tiles, warm-up and samples as
 *                                           rbench at one row; only old vs direct-consume GEMV.
 *   ds41_host_experts_test rhot [reps] [rows] [block|packed|planes] [n] [tile]  few and many rows, one
 *                                           matrix in cache, one thread, in 64-row tiles
 *   ds41_host_experts_test insbench [experts] [threads] [units] [batch]  the pool's packed GEMV, `batch`
 *                                           experts a burst, with a background insert beside it: none,
 *                                           memcpy's cached stores, copy_nt's streaming stores
 *                                           (..._INSERT_NT), or a spinning core
 *
 * The bench streams `experts` distinct DeepSeek-V4.1 experts (gate_up 4608 x 5120 and down
 * 5120 x 2304, 18.8 MB each) so the weights come from DRAM, as they would from the RAM cache. */
#include "ds41_host_experts.h"

#include <immintrin.h>
#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <functional>
#include <memory>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace ds41hx;

namespace {

constexpr int kEmbd = 5120, kFf = 2304;
constexpr float kLimit = 10.0f;   /* config.json swiglu_limit */

struct Mat {
    int N = 0, K = 0;
    std::vector<uint8_t> blocks, codes, scales;
    Mx ggml() const { Mx m; m.N = N; m.K = K; m.blocks = blocks.data(); return m; }
    Mx planes() const { Mx m; m.N = N; m.K = K; m.codes = codes.data(); m.scales = scales.data(); return m; }
};

int rounds_of(int N, int K) { return N * ((K / 32 + 31) / 32); }

/* Planes packed into a slot of cap raw rounds (cap < 0: every round, so any scales fit), and 64
 * guard bytes after the slot that packing must leave at 0xA5. rc: pack_planes' result. */
struct Packed {
    int N = 0, K = 0, rc = kPackInvalid;
    size_t bytes = 0;
    PackInfo plan;
    std::vector<uint8_t> slot;
    bool ok() const { return rc == kPackOk; }
    Mx view() const {
        Mx m;
        if (!ok() || !packed_view(slot.data(), bytes, N, K, &m)) {
            std::fprintf(stderr, "no packed view (rc %d)\n", rc);
            std::abort();
        }
        return m;
    }
    bool guard_ok() const {
        for (size_t i = bytes; i < slot.size(); ++i) if (slot[i] != 0xA5) return false;
        return true;
    }
};
Packed pack(const uint8_t* codes, const uint8_t* scales, int N, int K, int cap = -1) {
    Packed p;
    p.N = N;
    p.K = K;
    if (cap < 0) cap = rounds_of(N, K);
    p.bytes = packed_bytes(N, K, cap);
    p.slot.assign(p.bytes + 64, 0xA5);
    if (p.bytes && pack_plan(scales, N, K, &p.plan))
        p.rc = pack_planes(codes, scales, N, K, cap, p.plan, p.slot.data(), p.bytes);
    return p;
}
Packed pack(const Mat& m, int cap = -1) { return pack(m.codes.data(), m.scales.data(), m.N, m.K, cap); }

/* Block u of row n: its e8m0 and 32 codes, in both layouts. */
void put_block(Mat& m, int n, int u, uint8_t e, const uint8_t* c) {
    const int UK = m.K / 32;
    uint8_t* blk = &m.blocks[((size_t)n * UK + u) * 17];
    blk[0] = e;
    for (int j = 0; j < 16; ++j) blk[1 + j] = (uint8_t)(c[j] | (c[16 + j] << 4));
    m.scales[(size_t)n * UK + u] = e;
    for (int p = 0; p < 16; ++p) m.codes[(size_t)n * (m.K / 2) + (size_t)u * 16 + p] = (uint8_t)(c[2 * p] | (c[2 * p + 1] << 4));
}
/* Random codes; block u of row n's e8m0 is scale(n, u, a random word). */
template <class F>
Mat make_mat_with(int N, int K, uint64_t seed, F&& scale) {
    Mat m;
    m.N = N; m.K = K;
    const int UK = K / 32;
    m.blocks.resize((size_t)N * UK * 17);
    m.codes.resize((size_t)N * K / 2);
    m.scales.resize((size_t)N * UK);
    std::mt19937_64 rng(seed);
    for (int n = 0; n < N; ++n)
        for (int u = 0; u < UK; ++u) {
            const uint8_t e = scale(n, u, rng());
            uint8_t c[32];
            const uint64_t a = rng(), b = rng();
            for (int v = 0; v < 16; ++v) { c[v] = (uint8_t)((a >> (4 * v)) & 15); c[16 + v] = (uint8_t)((b >> (4 * v)) & 15); }
            put_block(m, n, u, e, c);
        }
    return m;
}
/* The release's experts' mix around base e0: e0 + 1 and e0 + 2 nearly always, e0 and e0 + 3 now
 * and then, and one block in `rare` (0: none) at one of the values the release has outside. */
uint8_t release_like(uint64_t r, int e0, int rare) {
    static const int8_t outside[] = { -7, -2, -1, 4, 5, 8 };
    if (rare && (r >> 40) % (uint64_t)rare == 0) return (uint8_t)(e0 + outside[(r >> 20) % sizeof outside]);
    const int p = (int)(r % 100);
    return (uint8_t)(e0 + (p < 2 ? 0 : p < 50 ? 1 : p < 98 ? 2 : 3));
}

/* How a matrix's e8m0 bytes are drawn:
 *   kReal    around 2^-8, 116 .. 127 (the release's experts: 112 .. 127, most at 120 and 121);
 *   kEdges   a quarter from the edges: the subnormal scales, tiny and large ones that stay finite
 *            through the sums. Scales near 255 overflow to inf and give NaN rows, whose sign and
 *            payload x86 and the card may legitimately produce differently;
 *   kWindow  uniform over the AVX2 kernel's fast path, 113 .. 127, both ends included;
 *   kMixed   kWindow but one block in 48 just outside it (111, 112, 128, 129), so that a row's
 *            rounds take both paths. */
enum Scales { kReal, kEdges, kWindow, kMixed };

/* Random codes, scales as above. */
Mat make_mat(int N, int K, uint64_t seed, Scales sc, bool with_planes = true) {
    Mat m;
    m.N = N; m.K = K;
    const int UK = K / 32;
    m.blocks.resize((size_t)N * UK * 17);
    if (with_planes) { m.codes.resize((size_t)N * K / 2); m.scales.resize((size_t)N * UK); }
    std::mt19937_64 rng(seed);
    static const uint8_t edge[] = { 0, 1, 2, 3, 60, 100, 126, 127, 128, 129, 135, 140 };
    static const uint8_t outside[] = { 111, 112, 128, 129 };
    for (int n = 0; n < N; ++n)
        for (int u = 0; u < UK; ++u) {
            const uint64_t r = rng();
            uint8_t e = (uint8_t)(116 + (r >> 8) % 12);
            if (sc == kEdges && (r & 3) == 0) e = edge[(r >> 2) % sizeof edge];
            if (sc == kWindow || sc == kMixed) e = (uint8_t)(113 + (r >> 8) % 15);
            if (sc == kMixed && (r & 63) % 48 == 0) e = outside[(r >> 6) % sizeof outside];
            uint8_t c[32];
            const uint64_t a = rng(), b = rng();
            for (int v = 0; v < 16; ++v) { c[v] = (uint8_t)((a >> (4 * v)) & 15); c[16 + v] = (uint8_t)((b >> (4 * v)) & 15); }
            uint8_t* blk = &m.blocks[((size_t)n * UK + u) * 17];
            blk[0] = e;
            for (int j = 0; j < 16; ++j) blk[1 + j] = (uint8_t)(c[j] | (c[16 + j] << 4));
            if (with_planes) {
                m.scales[(size_t)n * UK + u] = e;
                for (int p = 0; p < 16; ++p) m.codes[(size_t)n * (K / 2) + (size_t)u * 16 + p] = (uint8_t)(c[2 * p] | (c[2 * p + 1] << 4));
            }
        }
    return m;
}

/* Normal activations; adversarial: one in seven special (zeros, subnormals, ...); huge: one in
 * sixteen near 2^119, so that sums run near 2^120, and one bf16 max, whose products overflow in
 * the rows whose weight there is 1 or more (Inf rows; one infinite product a row, so no NaN). */
enum Acts { kNormal, kAdversarial, kHuge };
std::vector<uint16_t> make_x(int K, uint64_t seed, Acts act) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.f, 0.5f);
    std::vector<uint16_t> x((size_t)K);
    static const uint16_t special[] = { 0x0000, 0x8000, 0x0001, 0x8001, 0x0080, 0x3f80, 0x0040, 0x00ff, 0x4120 };
    static const uint16_t big[] = { 0x7b00, 0xfb00, 0x7a80, 0xfa80 };
    for (int k = 0; k < K; ++k) {
        x[(size_t)k] = f32_to_bf16(nd(rng));
        if (act == kAdversarial && rng() % 7 == 0) x[(size_t)k] = special[rng() % (sizeof special / 2)];
        if (act == kHuge && rng() % 16 == 0) x[(size_t)k] = big[rng() % (sizeof big / 2)];
    }
    if (act == kHuge) x[(size_t)K / 3] = 0x7f7f;
    return x;
}

size_t diff(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
    size_t d = 0;
    for (size_t i = 0; i < a.size(); ++i) d += a[i] != b[i];
    return d;
}

/* copy_nt against memcpy, and pack_planes writing the codes plane by non-temporal stores and in pieces
 * (PackCopy, the background inserts' ..._INSERT_NT and ..._INSERT_SLICE_KIB): every slot byte as one
 * memcpy writes it, at the release's shapes and an odd one, into a slot on and off a 64-byte line,
 * with before() called once a piece, the first included, for the piece's bytes. */
int packed_copy() {
    int failures = 0;
    std::mt19937_64 rng(4242u);
    std::vector<uint8_t> src(70000), a(70000 + 128), b(70000 + 128);
    for (auto& v : src) v = (uint8_t)rng();
    size_t dcopy = 0;
    for (size_t n : { (size_t)0, (size_t)1, (size_t)63, (size_t)64, (size_t)65, (size_t)127, (size_t)4096, (size_t)65537 })
        for (size_t d0 : { (size_t)0, (size_t)1, (size_t)8, (size_t)63 })
            for (size_t s0 : { (size_t)0, (size_t)3 }) {
                std::fill(a.begin(), a.end(), 0x5A);
                std::fill(b.begin(), b.end(), 0x5A);
                const size_t base = (size_t)(-(uintptr_t)a.data() & 63), bbase = (size_t)(-(uintptr_t)b.data() & 63);
                std::memcpy(a.data() + base + d0, src.data() + s0, n);
                copy_nt(b.data() + bbase + d0, src.data() + s0, n);
                dcopy += std::memcmp(a.data() + base, b.data() + bbase, 70000) != 0;
            }
    std::printf("%-26s copy_nt against memcpy, 8 sizes x 4 offsets x 2 sources: %zu differ\n", "copy_nt", dcopy);
    failures += dcopy != 0;
    struct Count { size_t calls = 0, bytes = 0, max = 0; };
    auto before = [](void* ctx, size_t bytes) {
        Count* c = static_cast<Count*>(ctx);
        ++c->calls;
        c->bytes += bytes;
        c->max = std::max(c->max, bytes);
    };
    auto mix = [](int, int, uint64_t r) { return release_like(r, 119, 3000); };
    struct Shape { int N, K; };
    for (const Shape sh : { Shape{ 2 * kFf, kEmbd }, Shape{ kEmbd, kFf }, Shape{ 64, 1056 } }) {
        const Mat m = make_mat_with(sh.N, sh.K, 77u + (uint64_t)sh.N, mix);
        const Packed want = pack(m);
        if (!want.ok()) { std::printf("packed copy %dx%d: FAIL: pack_planes returned %d\n", sh.N, sh.K, want.rc); ++failures; continue; }
        const size_t cb = (size_t)sh.N * (size_t)(sh.K / 2);
        size_t bad = 0, calls_off = 0;
        for (const bool nt : { true, false })
            for (const size_t slice : { (size_t)0, (size_t)1000, (size_t)4096, (size_t)256 << 10 })
                for (const size_t off : { (size_t)0, (size_t)8 }) {
                    std::vector<uint8_t> buf(want.bytes + 64 + 128, 0xA5);
                    uint8_t* const dst = buf.data() + (size_t)(-(uintptr_t)buf.data() & 63) + off;
                    Count cnt;
                    PackCopy cp;
                    cp.nt = nt;
                    cp.slice = slice;
                    cp.before = +before;
                    cp.ctx = &cnt;
                    const int rc = pack_planes(m.codes.data(), m.scales.data(), sh.N, sh.K, rounds_of(sh.N, sh.K), want.plan, dst,
                                               want.bytes, &cp);
                    bad += rc != kPackOk || std::memcmp(dst, want.slot.data(), want.bytes) != 0;
                    const size_t pieces = slice ? (cb + slice - 1) / slice : 1;
                    calls_off += cnt.calls != pieces || cnt.bytes != cb || cnt.max != (slice ? std::min(slice, cb) : cb);
                }
        std::printf("packed copy %4dx%-4d    by streaming stores and not, pieces of 0 / 1000 / 4 KiB / 256 KiB, slot on and "
                    "off a line: %zu slots differ, %zu piece counts off\n", sh.N, sh.K, bad, calls_off);
        failures += bad != 0 || calls_off != 0;
    }
    return failures;
}

int check() {
    int failures = 0;
    struct Case { const char* name; int N, K; Scales sc; Acts act; };
    const Case cases[] = {
        { "gate_up 4608x5120", 2 * kFf, kEmbd, kReal, kNormal }, { "down 5120x2304", kEmbd, kFf, kReal, kNormal },
        { "gate_up edges 256x5120", 256, kEmbd, kEdges, kAdversarial }, { "down edges 256x2304", 256, kFf, kEdges, kAdversarial },
        { "odd K 64x1056", 64, 1056, kEdges, kAdversarial },
        /* the fast path's scale window, its ends, and rounds of both paths in one row */
        { "gate_up window 256x5120", 256, kEmbd, kWindow, kAdversarial }, { "down window 256x2304", 256, kFf, kWindow, kAdversarial },
        { "gate_up mixed 256x5120", 256, kEmbd, kMixed, kAdversarial }, { "down mixed 256x2304", 256, kFf, kMixed, kAdversarial },
        { "gate_up window huge x", 64, kEmbd, kWindow, kHuge }, { "gate_up edges huge x", 64, kEmbd, kEdges, kHuge },
        /* a row's last round with 1 .. 4 groups, full or not, on either path; rows of one round */
        { "window K 1056 (33 blocks)", 64, 1056, kWindow, kAdversarial }, { "window K 1280 (40)", 64, 1280, kWindow, kAdversarial },
        { "window K 1536 (48)", 64, 1536, kWindow, kAdversarial }, { "window K 1664 (52)", 64, 1664, kWindow, kAdversarial },
        { "window K 2016 (63)", 64, 2016, kWindow, kAdversarial }, { "window K 256 (8)", 64, 256, kWindow, kAdversarial },
        { "window K 32 (1)", 64, 32, kWindow, kAdversarial }, { "edges K 1664 (52)", 64, 1664, kEdges, kAdversarial },
        { "edges K 2016 (63)", 64, 2016, kEdges, kAdversarial }, { "edges K 32 (1)", 64, 32, kEdges, kAdversarial },
        { "mixed K 1664 (52)", 64, 1664, kMixed, kAdversarial } };
    for (const Case& c : cases) {
        const Mat m = make_mat(c.N, c.K, 1234u + (uint64_t)c.N + 7u * (uint64_t)c.sc, c.sc);
        const auto x = make_x(c.K, 99u + (uint64_t)c.K + 3u * (uint64_t)c.act, c.act);
        std::vector<uint16_t> rg((size_t)c.N), rp((size_t)c.N), ag((size_t)c.N), ap((size_t)c.N);
        gemv_ref(m.ggml(), x.data(), rg.data(), 0, c.N);
        gemv_ref(m.planes(), x.data(), rp.data(), 0, c.N);
        gemv_avx2(m.ggml(), x.data(), ag.data(), 0, c.N);
        gemv_avx2(m.planes(), x.data(), ap.data(), 0, c.N);
        const size_t d1 = diff(rg, rp), d2 = diff(rg, ag), d3 = diff(rg, ap);
        size_t nan = 0, inf = 0;
        for (uint16_t h : rg) { const uint16_t e = h & 0x7f80; nan += e == 0x7f80 && (h & 0x7f); inf += e == 0x7f80 && !(h & 0x7f); }
        std::printf("%-26s ref planes %zu, avx2 blocks %zu, avx2 planes %zu differing rows of %d  (%zu NaN, %zu Inf)\n",
                    c.name, d1, d2, d3, c.N, nan, inf);
        failures += d1 || d2 || d3;
        /* The packed form of the same planes, in a slot that takes every round raw if need be: these
         * scales are spread wide, so most rounds are raw here (packed_test has the release's mix). */
        const Packed pk = pack(m);
        size_t d4 = 0, d5 = 0;
        bool rt = false;
        if (pk.ok()) {
            const Mx pm = pk.view();
            std::vector<uint16_t> rk((size_t)c.N), ak((size_t)c.N);
            gemv_ref(pm, x.data(), rk.data(), 0, c.N);
            gemv_avx2(pm, x.data(), ak.data(), 0, c.N);
            d4 = diff(rg, rk);
            d5 = diff(rg, ak);
            rt = pk.guard_ok() && packed_scales_equal(pm, m.scales.data());
        }
        std::printf("%-26s packed (%d of %d rounds raw, base %d): ref %zu, avx2 %zu differing rows; scales round trip %s\n",
                    c.name, pk.plan.n_exc, rounds_of(c.N, c.K), pk.plan.e0, d4, d5, rt ? "exact" : "FAILED");
        failures += !pk.ok() || d4 || d5 || !rt;
        /* Several token rows of the matrix, the case's x and more of its kind, each row as gemv_avx2
         * gives it alone, in every layout: gemv_avx2_few on 2 .. 5 rows (5: a call of four and one of
         * one), gemv_avx2_many on 1 .. 33 (33: a call of 32 and one of one; every remainder of its
         * register blocks of four rows) and on the pool's jobs of up to 64 rows (..._MAX_ROWS=64:
         * two calls of 32, and every split of a job between 33 and 64), also over output rows [8, N)
         * only, and gemv_avx2_rows, which takes one or the other, on 1 .. 33. */
        constexpr int kRowsMax = 64;
        std::vector<std::vector<uint16_t>> xr(kRowsMax);
        for (int i = 0; i < kRowsMax; ++i)
            xr[(size_t)i] = i ? make_x(c.K, 31u * (uint64_t)i + (uint64_t)c.K + 3u * (uint64_t)c.act, c.act) : x;
        size_t dfew = 0, dmany = 0, drows = 0, dlanes = 0;
        for (int lay = 0; lay < (pk.ok() ? 3 : 2); ++lay) {
            const Mx mx = lay == 2 ? pk.view() : lay ? m.planes() : m.ggml();
            std::vector<std::vector<uint16_t>> one(kRowsMax, std::vector<uint16_t>((size_t)c.N));
            const uint16_t* xs[kRowsMax];
            for (int i = 0; i < kRowsMax; ++i) {
                gemv_avx2(mx, xr[(size_t)i].data(), one[(size_t)i].data(), 0, c.N);
                xs[i] = xr[(size_t)i].data();
            }
            auto run = [&](void (*f)(const Mx&, const uint16_t* const*, uint16_t* const*, int, int, int), int rows) {
                std::vector<std::vector<uint16_t>> got((size_t)rows, std::vector<uint16_t>((size_t)c.N, 0xffff));
                uint16_t* ys[kRowsMax];
                for (int i = 0; i < rows; ++i) ys[i] = got[(size_t)i].data();
                f(mx, xs, ys, rows, 0, c.N);
                size_t d = 0;
                for (int i = 0; i < rows; ++i) d += diff(one[(size_t)i], got[(size_t)i]);
                return d;
            };
            /* Output rows [8, N) only: a block of eight output rows at the end and a packed cursor
             * that starts past row 0; rows [0, 8) must stay untouched. */
            auto run_from8 = [&](int rows) {
                std::vector<std::vector<uint16_t>> got((size_t)rows, std::vector<uint16_t>((size_t)c.N, 0xffff));
                uint16_t* ys[kRowsMax];
                for (int i = 0; i < rows; ++i) ys[i] = got[(size_t)i].data();
                gemv_avx2_many(mx, xs, ys, rows, 8, c.N);
                size_t d = 0;
                for (int i = 0; i < rows; ++i)
                    for (int n = 0; n < c.N; ++n)
                        d += got[(size_t)i][(size_t)n] != (n < 8 ? (uint16_t)0xffff : one[(size_t)i][(size_t)n]);
                return d;
            };
            /* gemv_avx2_few_lanes on few_lanes' lanes, laid out once and used for every 64-row tile
             * of the output rows, as the pool runs a job's tiles (..._HOST_EXPERTS_LEAN_TILES). */
            auto run_lanes = [&](int rows) {
                struct alignas(32) F8 { float f[8]; };
                std::vector<F8> lanes((size_t)rows * few_lanes_floats(c.K) / 8);
                few_lanes(xs, rows, c.K, lanes.data()->f);
                std::vector<std::vector<uint16_t>> got((size_t)rows, std::vector<uint16_t>((size_t)c.N, 0xffff));
                uint16_t* ys[kRowsMax];
                for (int i = 0; i < rows; ++i) ys[i] = got[(size_t)i].data();
                for (int n0 = 0; n0 < c.N; n0 += 64) gemv_avx2_few_lanes(mx, lanes.data()->f, ys, rows, n0, std::min(n0 + 64, c.N));
                size_t d = 0;
                for (int i = 0; i < rows; ++i) d += diff(one[(size_t)i], got[(size_t)i]);
                return d;
            };
            for (int rows = 2; rows <= 5; ++rows) dfew += run(gemv_avx2_few, rows);
            for (int rows = 1; rows <= 5; ++rows) dlanes += run_lanes(rows);
            for (int rows = 1; rows <= 33; ++rows) {
                dmany += run(gemv_avx2_many, rows);
                drows += run(gemv_avx2_rows, rows);
            }
            for (int rows : { 34, 40, 47, 48, 63, 64 }) {
                dmany += run(gemv_avx2_many, rows);
                drows += run(gemv_avx2_rows, rows);
            }
            for (int rows : { 1, 3, 4, 6, 32, 33, 64 }) dmany += run_from8(rows);
        }
        std::printf("%-26s every layout: avx2 few 2..5 rows %zu, few on laid-out lanes 1..5 %zu, many 1..33 and "
                    "34..64 %zu, rows 1..64 %zu differing values\n", c.name, dfew, dlanes, dmany, drows);
        failures += dfew || dlanes || dmany || drows;
    }
    /* A whole expert: gate_up, swiglu_clamped and down, scalar vs AVX2 at four threads. */
    const Mat gu = make_mat(2 * kFf, kEmbd, 7u, kReal), dn = make_mat(kEmbd, kFf, 8u, kReal);
    Expert e{ gu.ggml(), dn.ggml() };
    std::vector<float> silu(65536);
    silu_table_host(silu.data(), kLimit);
    const auto x = make_x(kEmbd, 5u, kNormal);
    std::vector<uint16_t> g1(2 * kFf), a1(kFf), d1(kEmbd), g2(2 * kFf), a2(kFf), d2(kEmbd);
    expert_forward(e, silu.data(), kLimit, x.data(), g1.data(), a1.data(), d1.data(), 1, false);
    expert_forward(e, silu.data(), kLimit, x.data(), g2.data(), a2.data(), d2.data(), 4, true);
    const size_t de = diff(g1, g2) + diff(a1, a2) + diff(d1, d2);
    std::printf("%-26s scalar vs AVX2 x4 threads: %zu differing values\n", "expert forward", de);
    failures += de != 0;
    /* swiglu_avx2 against swiglu: every gate pattern with every 61st up pattern and the special
     * ones (zeros, the limits, infinities, quiet NaNs, subnormals). A signalling-NaN up, which
     * the gate_up GEMV never writes, is the only input on which they may differ. */
    {
        std::vector<uint32_t> ups;
        for (uint32_t u = 0; u < 65536; u += 61) ups.push_back(u);
        for (uint32_t u : { 0x0000u, 0x8000u, 0x4120u, 0xc120u, 0x7f80u, 0xff80u, 0x7fc0u, 0xffc0u,
                            0x0001u, 0x8001u, 0x007fu, 0x4121u, 0xc121u, 0x411fu })
            ups.push_back(u);
        std::vector<uint16_t> row(2 * kFf), s1(kFf), s2(kFf);
        size_t dsw = 0, pairs = 0;
        for (uint32_t u : ups) {
            const bool snan = (u & 0x7f80u) == 0x7f80u && (u & 0x7fu) && !(u & 0x40u);
            if (snan) continue;
            for (uint32_t g0 = 0; g0 < 65536; g0 += kFf) {
                for (int i = 0; i < kFf; ++i) {
                    row[(size_t)i] = (uint16_t)((g0 + (uint32_t)i) & 0xffffu);
                    row[(size_t)(kFf + i)] = (uint16_t)u;
                }
                swiglu(silu.data(), row.data(), s1.data(), kFf, kLimit);
                swiglu_avx2(silu.data(), row.data(), s2.data(), kFf, kLimit);
                dsw += diff(s1, s2);
                pairs += kFf;
            }
        }
        std::printf("%-26s swiglu_avx2 vs swiglu over %zu gate/up pairs: %zu differing values\n", "swiglu",
                    pairs, dsw);
        failures += dsw != 0;
    }
    /* The same chain on packed weights with the release's scale mix: AVX2 at four threads and the
     * scalar reference on the packed form, against the scalar reference on the planes. */
    auto mix = [](int, int, uint64_t r) { return release_like(r, 119, 3000); };
    const Mat pgu = make_mat_with(2 * kFf, kEmbd, 17u, mix), pdn = make_mat_with(kEmbd, kFf, 18u, mix);
    const Packed kgu = pack(pgu), kdn = pack(pdn);
    size_t dk = 1;
    if (kgu.ok() && kdn.ok()) {
        Expert pe{ pgu.planes(), pdn.planes() }, ke{ kgu.view(), kdn.view() };
        std::vector<uint16_t> g3(2 * kFf), a3(kFf), d3(kEmbd), g4(2 * kFf), a4(kFf), d4(kEmbd), g5(2 * kFf), a5(kFf), d5(kEmbd);
        expert_forward(pe, silu.data(), kLimit, x.data(), g3.data(), a3.data(), d3.data(), 1, false);
        expert_forward(ke, silu.data(), kLimit, x.data(), g4.data(), a4.data(), d4.data(), 4, true);
        expert_forward(ke, silu.data(), kLimit, x.data(), g5.data(), a5.data(), d5.data(), 1, false);
        dk = diff(g3, g4) + diff(a3, a4) + diff(d3, d4) + diff(g3, g5) + diff(a3, a5) + diff(d3, d5);
    }
    std::printf("%-26s packed (gate_up %d, down %d rounds raw): %zu differing values\n", "expert forward packed",
                kgu.plan.n_exc, kdn.plan.n_exc, dk);
    failures += dk != 0;
    failures += packed_copy();
    std::printf(failures ? "FAIL: %d case(s) differ\n" : "PASS: every case bit-identical\n", failures);
    return failures ? 1 : 0;
}

/* ---- the packed form against the planes ------------------------------------------------------ */

/* y for x under w: gemv_ref (unless with_ref is false), gemv_avx2, gemv_avx2_few on four token rows
 * (x, then x reversed, halved and doubled) and gemv_avx2_many on nine (those, then x negated, rotated
 * by one and by seven values, reversed and negated, halved and rotated); `rows_off` counts the values
 * where a row differs from gemv_avx2 on that row alone. `rows` holds the four rows of each, few first. */
struct Outs { std::vector<uint16_t> ref, avx2, rows; size_t rows_off = 0; };
Outs run_all(const Mx& w, const std::vector<uint16_t>& x, bool with_ref) {
    const int N = w.N;
    constexpr int kMany = 9;
    Outs o;
    o.ref.assign((size_t)N, 0);
    o.avx2.assign((size_t)N, 0);
    o.rows.assign((size_t)N * 8, 0xffff);
    if (with_ref) gemv_ref(w, x.data(), o.ref.data(), 0, N);
    gemv_avx2(w, x.data(), o.avx2.data(), 0, N);
    std::vector<std::vector<uint16_t>> xv(kMany, x);
    std::reverse(xv[1].begin(), xv[1].end());
    for (auto& h : xv[2]) h = f32_to_bf16(bf16_to_f32(h) * 0.5f);
    for (auto& h : xv[3]) h = f32_to_bf16(bf16_to_f32(h) * 2.0f);
    for (auto& h : xv[4]) h ^= 0x8000;
    std::rotate(xv[5].begin(), xv[5].begin() + 1, xv[5].end());
    std::rotate(xv[6].begin(), xv[6].begin() + 7, xv[6].end());
    xv[7] = xv[1];
    for (auto& h : xv[7]) h ^= 0x8000;
    xv[8] = xv[2];
    std::rotate(xv[8].begin(), xv[8].begin() + 3, xv[8].end());
    const uint16_t* xs[kMany];
    for (int i = 0; i < kMany; ++i) xs[i] = xv[(size_t)i].data();
    std::vector<std::vector<uint16_t>> many(kMany, std::vector<uint16_t>((size_t)N, 0xffff));
    uint16_t* ys[kMany];
    for (int i = 0; i < kMany; ++i) ys[i] = i < 4 ? o.rows.data() + (size_t)i * N : many[(size_t)i].data();
    gemv_avx2_few(w, xs, ys, 4, 0, N);
    for (int i = 0; i < 4; ++i) ys[i] = many[(size_t)i].data();
    gemv_avx2_many(w, xs, ys, kMany, 0, N);
    for (int i = 0; i < 4; ++i) std::copy(many[(size_t)i].begin(), many[(size_t)i].end(), o.rows.begin() + (ptrdiff_t)(4 + i) * N);
    for (int i = 0; i < kMany; ++i) {
        std::vector<uint16_t> one((size_t)N);
        gemv_avx2(w, xs[i], one.data(), 0, N);
        if (i < 4)
            for (int n = 0; n < N; ++n) o.rows_off += one[(size_t)n] != o.rows[(size_t)i * N + (size_t)n];
        o.rows_off += diff(one, many[(size_t)i]);
    }
    return o;
}

/* pack_plan's answer the slow way: every base tried on every round. */
PackInfo brute_plan(const Mat& m) {
    const int UK = m.K / 32, R = (UK + 31) / 32;
    std::vector<int> lo((size_t)m.N * R, 255), hi((size_t)m.N * R, 0);
    for (int n = 0; n < m.N; ++n)
        for (int u = 0; u < UK; ++u) {
            const int e = m.scales[(size_t)n * UK + u], g = n * R + u / 32;
            lo[(size_t)g] = std::min(lo[(size_t)g], e);
            hi[(size_t)g] = std::max(hi[(size_t)g], e);
        }
    PackInfo best;
    best.n_exc = -1;
    for (int b = 0; b <= 252; ++b) {
        int raw = 0;
        for (size_t g = 0; g < lo.size(); ++g) raw += !(lo[g] >= b && hi[g] <= b + 3);
        if (best.n_exc < 0 || raw < best.n_exc) { best.e0 = b; best.n_exc = raw; }
    }
    return best;
}

/* Packed against planar for one matrix: every kernel path bit for bit (the scalar reference too
 * unless `with_ref` is false), the plan against brute_plan's, the scales' round trip, and the slot's
 * guard. want_raw >= 0: the raw rounds the case was built to have. Returns failures. */
int compare_packed(const char* name, const Mat& m, const std::vector<uint16_t>& x, bool with_ref = true,
                   int want_raw = -1) {
    const Packed pk = pack(m);
    if (!pk.ok()) {
        std::printf("%-34s FAIL: pack_planes returned %d\n", name, pk.rc);
        return 1;
    }
    const Mx pm = pk.view();
    const Outs a = run_all(m.planes(), x, with_ref), b = run_all(pm, x, with_ref);
    const size_t dref = diff(a.ref, b.ref), davx = diff(a.avx2, b.avx2), drows = diff(a.rows, b.rows) + a.rows_off + b.rows_off;
    const size_t dcross = with_ref ? diff(a.ref, b.avx2) : 0;
    const bool rt = packed_scales_equal(pm, m.scales.data()) && pk.guard_ok();
    const PackInfo bp = brute_plan(m);
    const bool plan = pk.plan.e0 == bp.e0 && pk.plan.n_exc == bp.n_exc && (want_raw < 0 || pk.plan.n_exc == want_raw);
    size_t nan = 0, inf = 0;
    for (uint16_t h : a.avx2) { const uint16_t e = h & 0x7f80; nan += e == 0x7f80 && (h & 0x7f); inf += e == 0x7f80 && !(h & 0x7f); }
    const bool bad = dref || davx || drows || dcross || !rt || !plan;
    std::printf("%-34s base %3d, %5d of %6d rounds raw: ref %zu, avx2 %zu, rows %zu%s differing; round trip %s%s  (%zu NaN, %zu Inf rows)%s\n",
                name, pk.plan.e0, pk.plan.n_exc, rounds_of(m.N, m.K), dref, davx, drows,
                with_ref ? (dcross ? ", ref/avx2 MISMATCH" : ", ref = avx2") : "", rt ? "exact" : "FAILED",
                plan ? "" : ", UNEXPECTED PLAN", nan, inf, bad ? "  FAIL" : "");
    return bad ? 1 : 0;
}

/* Every block position of a row's rounds raw once: row n < UK has one block, n, outside the window
 * (cycling through values on both paths and both sides); the rows after have none, then two. */
int packed_positions(int K, int e0) {
    const int UK = K / 32;
    const int N = (UK + 16 + 7) / 8 * 8;
    /* e0 + v for small v (both sides, the fast path and past it), and some absolute values: the
     * subnormal scales, the fast path's ends, large finite ones */
    static const int out[] = { -1, 4, -9, 9, 0, 140, 112, 128, 200, 1, -2, 5, 113, 127 };
    auto outside = [&](int i) -> uint8_t {
        const int v = out[i % (int)(sizeof out / sizeof out[0])];
        const bool absolute = v == 0 || v == 1 || v >= 100;
        const int e = absolute ? v : e0 + v;
        /* an absolute value inside the window would not make the round raw */
        return (uint8_t)(e >= e0 && e <= e0 + 3 ? e0 + 5 : e);
    };
    int raw = 0;
    std::vector<int> hit((size_t)N * UK, 0);
    for (int n = 0; n < N; ++n) {
        if (n < UK) hit[(size_t)n * UK + n] = 1 + n;
        else if (n >= UK + 8) { hit[(size_t)n * UK + (n % UK)] = 1 + n; hit[(size_t)n * UK + (n * 7 + 3) % UK] = 2 + n; }
    }
    for (int n = 0; n < N; ++n)
        for (int r = 0; r < (UK + 31) / 32; ++r) {
            bool any = false;
            for (int u = 32 * r; u < std::min(UK, 32 * r + 32); ++u) any = any || hit[(size_t)n * UK + u];
            raw += any;
        }
    const Mat m = make_mat_with(N, K, 4242u + (uint64_t)K, [&](int n, int u, uint64_t r) -> uint8_t {
        const int h = hit[(size_t)n * UK + u];
        return h ? outside(h) : (uint8_t)(e0 + (int)(r % 4));
    });
    char name[64];
    std::snprintf(name, sizeof name, "raw at every position K %d (%d)", K, UK);
    /* With rounds of 8+ blocks the window's base wins and the built raw rounds are the plan's; rounds
     * of a block or two can fit the outliers' own base better (brute_plan still holds the plan). */
    return compare_packed(name, m, make_x(K, 7u + (uint64_t)K, kAdversarial), true, UK >= 8 ? raw : -1);
}

/* The research sample (pack1bit.py's sample/): each expert's three e8m0 planes whole and the first
 * MiB of each codes plane, as L<layer>_E<expert>_w<1|2|3>.{scale,codes}. Gate_up is [w1; w3]. */
struct SampleExpert { int L = 0, E = 0; Mat gu, dn; };
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
/* Rows [r0, r0 + rows) of a weight from the sample's scales and (where they reach) codes; random
 * codes past them. */
void put_sample(Mat& m, int r0, int rows, const std::vector<uint8_t>& sc, const std::vector<uint8_t>& codes, uint64_t seed) {
    const int UK = m.K / 32, CB = m.K / 2;
    std::mt19937_64 rng(seed);
    for (int n = 0; n < rows; ++n)
        for (int u = 0; u < UK; ++u) {
            uint8_t c[32];
            const size_t at = (size_t)n * CB + (size_t)u * 16;
            if (at + 16 <= codes.size())
                for (int p = 0; p < 16; ++p) { c[2 * p] = codes[at + p] & 15; c[2 * p + 1] = codes[at + p] >> 4; }
            else {
                const uint64_t a = rng(), b = rng();
                for (int v = 0; v < 16; ++v) { c[v] = (uint8_t)((a >> (4 * v)) & 15); c[16 + v] = (uint8_t)((b >> (4 * v)) & 15); }
            }
            put_block(m, r0 + n, u, sc[(size_t)n * UK + u], c);
        }
}
std::vector<std::pair<int, int>> sample_ids(const std::string& dir) {
    std::vector<std::pair<int, int>> ids;
    DIR* d = opendir(dir.c_str());
    if (!d) return ids;
    while (dirent* e = readdir(d)) {
        int L = -1, E = -1;
        char tail[32] = {};
        if (std::sscanf(e->d_name, "L%d_E%d_w1.%31s", &L, &E, tail) == 3 && !std::strcmp(tail, "scale")) ids.emplace_back(L, E);
    }
    closedir(d);
    std::sort(ids.begin(), ids.end());
    return ids;
}
/* One sample expert as gate_up and down matrices, both layouts; false if its files are not whole. */
bool load_expert(const std::string& dir, int L, int E, SampleExpert* x) {
    char stem[64];
    std::snprintf(stem, sizeof stem, "/L%02d_E%03d_w", L, E);
    std::vector<uint8_t> s1, s2, s3, c1, c2, c3;
    const std::string b = dir + stem;
    if (!read_file(b + "1.scale", &s1) || !read_file(b + "2.scale", &s2) || !read_file(b + "3.scale", &s3) ||
        s1.size() != (size_t)kFf * (kEmbd / 32) || s3.size() != s1.size() || s2.size() != (size_t)kEmbd * (kFf / 32))
        return false;
    read_file(b + "1.codes", &c1);
    read_file(b + "2.codes", &c2);
    read_file(b + "3.codes", &c3);
    x->L = L;
    x->E = E;
    auto alloc = [](Mat& m, int N, int K) {
        m.N = N; m.K = K;
        m.blocks.assign((size_t)N * (K / 32) * 17, 0);
        m.codes.assign((size_t)N * K / 2, 0);
        m.scales.assign((size_t)N * (K / 32), 0);
    };
    alloc(x->gu, 2 * kFf, kEmbd);
    alloc(x->dn, kEmbd, kFf);
    const uint64_t seed = (uint64_t)L * 1000u + (uint64_t)E;
    put_sample(x->gu, 0, kFf, s1, c1, seed);
    put_sample(x->gu, kFf, kFf, s3, c3, seed + 1);
    put_sample(x->dn, 0, kEmbd, s2, c2, seed + 2);
    return true;
}
std::string sample_dir(int argc, char** argv, int at) {
    if (argc > at) return argv[at];
    const char* e = std::getenv("DS41_HX_SAMPLE");
    return e && *e ? e : "/home/as/.claude/jobs/635603c1/tmp/research/sample";
}

/* The packing API's refusals and the view's checks, each on its own copy of a slot. */
int packed_api() {
    int failures = 0;
    auto expect = [&](bool ok, const char* what) {
        if (!ok) { std::printf("  packing API: %s FAILED\n", what); ++failures; }
    };
    const Mat m = make_mat_with(64, kFf, 31337u, [](int, int, uint64_t r) { return release_like(r, 119, 40); });
    PackInfo plan;
    expect(pack_plan(m.scales.data(), m.N, m.K, &plan) && plan.e0 == 119 && plan.n_exc > 2, "plan of a release-like weight");
    const int raw = plan.n_exc;
    const Packed fit = pack(m, raw), over = pack(m, raw - 1);
    expect(fit.ok() && fit.bytes == packed_bytes(m.N, m.K, raw) && fit.guard_ok(), "a slot of exactly its raw rounds");
    expect(over.rc == kPackOverBudget, "one raw round over the cap: kPackOverBudget");
    std::vector<uint8_t> dst(fit.bytes + 64, 0xA5);
    PackInfo wrong = plan;
    wrong.e0 = plan.e0 + 1;
    expect(pack_planes(m.codes.data(), m.scales.data(), m.N, m.K, rounds_of(m.N, m.K), wrong, dst.data(), dst.size()) == kPackInvalid,
           "a plan with another base: kPackInvalid");
    wrong = plan;
    wrong.n_exc = plan.n_exc - 1;
    expect(pack_planes(m.codes.data(), m.scales.data(), m.N, m.K, rounds_of(m.N, m.K), wrong, dst.data(), dst.size()) == kPackInvalid,
           "a plan with another count: kPackInvalid");
    expect(pack_planes(m.codes.data(), m.scales.data(), m.N, m.K, raw, plan, dst.data(), fit.bytes - 1) == kPackInvalid,
           "a slot one byte short: kPackInvalid");
    expect(!pack_plan(m.scales.data(), 64, 100, &plan) && packed_bytes(64, 100, 0) == 0, "K not a multiple of 32");
    expect(packed_bytes(64, kFf, rounds_of(64, kFf) + 1) == 0 && packed_bytes(64, kFf, -1) == 0, "a cap outside 0 .. rounds");
    Mx v;
    expect(packed_view(fit.slot.data(), fit.bytes, m.N, m.K, &v) && v.n_exc == raw && v.e0 == 119, "the view");
    expect(!packed_view(fit.slot.data(), fit.bytes, m.N + 8, m.K, &v), "a view of another N");
    expect(!packed_view(fit.slot.data(), fit.bytes, m.N, kEmbd, &v), "a view of another K");
    expect(!packed_view(fit.slot.data(), fit.bytes - 1, m.N, m.K, &v), "a view of a short slot");
    expect(!packed_view(m.blocks.data(), m.blocks.size(), m.N, m.K, &v), "a view of block_mxfp4 bytes");
    {
        std::vector<uint8_t> s = fit.slot;
        s[0] ^= 1;
        expect(!packed_view(s.data(), fit.bytes, m.N, m.K, &v), "a view with a damaged magic");
    }
    /* The round trip catches a changed code, raw byte or index, and a disordered index. */
    const Mx good = fit.view();
    expect(packed_scales_equal(good, m.scales.data()), "the round trip of a good slot");
    auto damaged = [&](auto&& change) {
        std::vector<uint8_t> s = fit.slot;
        Mx w;
        if (!packed_view(s.data(), fit.bytes, m.N, m.K, &w)) return false;
        change(s.data(), w);
        return !packed_scales_equal(w, m.scales.data());
    };
    expect(damaged([](uint8_t*, Mx& w) {   /* a code of a widened round: row 0's round 0 if packed, else row 1's */
        uint8_t* p = const_cast<uint8_t*>(w.pk);
        const int R = (w.K / 32 + 31) / 32;
        for (int g = 0; g < w.N * R; ++g) {
            bool raw_round = false;
            for (int i = 0; i < w.n_exc; ++i) raw_round = raw_round || w.exc_idx[i] == (uint32_t)g;
            if (raw_round) continue;
            p[(size_t)(g / R) * pk_row_bytes(w.K) + (size_t)8 * (g % R)] ^= 4;
            return;
        }
    }), "a changed 2-bit code");
    expect(damaged([](uint8_t*, Mx& w) { const_cast<uint8_t*>(w.exc_raw)[3] ^= 1; }), "a changed raw byte");
    expect(damaged([](uint8_t*, Mx& w) { std::swap(const_cast<uint32_t*>(w.exc_idx)[0], const_cast<uint32_t*>(w.exc_idx)[1]); }),
           "a disordered index");
    expect(damaged([](uint8_t*, Mx& w) { const_cast<uint32_t*>(w.exc_idx)[w.n_exc - 1] += 1u << 20; }), "an index past the rounds");
    std::printf("packing API: %s\n", failures ? "FAIL" : "every refusal and check as specified");
    return failures;
}

int packed_test(const std::string& dir) {
    int failures = packed_api();
    /* The release's mix at full size, then windows that put widened rounds on the general path:
     * across the fast path's top (126 .. 129) and bottom (110 .. 113), and the subnormal scales. */
    struct Case { const char* name; int N, K, e0, rare; Acts act; };
    const Case cases[] = {
        { "release mix gate_up 4608x5120", 2 * kFf, kEmbd, 119, 3000, kNormal },
        { "release mix down 5120x2304", kEmbd, kFf, 119, 3000, kNormal },
        { "release mix down, adversarial x", 512, kFf, 119, 300, kAdversarial },
        { "window 126..129 gate_up", 256, kEmbd, 126, 200, kAdversarial },
        { "window 126..129 down", 256, kFf, 126, 200, kAdversarial },
        { "window 110..113 down", 256, kFf, 110, 200, kAdversarial },
        { "window 0..3 (subnormal) down", 256, kFf, 0, 0, kAdversarial },
        { "window 0..3 (subnormal) K 1664", 64, 1664, 0, 0, kAdversarial },
        { "window 119..122 K 1056 (33)", 64, 1056, 119, 50, kAdversarial },
        { "window 119..122 K 32 (1)", 64, 32, 119, 50, kAdversarial } };
    for (const Case& c : cases) {
        const Mat m = make_mat_with(c.N, c.K, 555u + (uint64_t)c.K + (uint64_t)c.e0, [&](int, int, uint64_t r) {
            return c.e0 == 0 ? (uint8_t)(r % 4) : release_like(r, c.e0, c.rare);
        });
        failures += compare_packed(c.name, m, make_x(c.K, 11u + (uint64_t)c.K, c.act));
    }
    /* Scales up to 2^127 (e8m0 252 .. 255) and huge x: Inf and NaN rows. The scalar reference and the
     * AVX2 kernel may give a NaN different signs or payloads, so packed is held to planar within each. */
    {
        const Mat m = make_mat_with(256, kFf, 99u, [](int, int, uint64_t r) { return (uint8_t)(252 + r % 4); });
        failures += compare_packed("window 252..255, huge x (NaN rows)", m, make_x(kFf, 5u, kHuge), false);
        const Mat g = make_mat_with(64, kEmbd, 98u, [](int, int u, uint64_t r) {
            return u % 37 == 5 ? (uint8_t)(248 + r % 4) : (uint8_t)(252 + r % 4);   /* raw rounds too */
        });
        failures += compare_packed("window 252..255 gate_up, huge x", g, make_x(kEmbd, 6u, kHuge), false);
    }
    for (int K : { kEmbd, kFf, 1056, 1664, 256, 32 }) failures += packed_positions(K, 119);
    failures += packed_positions(kFf, 126);
    /* Real experts: the sample's scales whole, its codes where they reach. */
    const auto ids = sample_ids(dir);
    int worst_gu = 0, worst_dn = 0, loaded = 0;
    for (const auto& id : ids) {
        SampleExpert s;
        if (!load_expert(dir, id.first, id.second, &s)) continue;
        ++loaded;
        char name[64];
        const auto x = make_x(kEmbd, 1000u + (uint64_t)s.E, kNormal), xd = make_x(kFf, 2000u + (uint64_t)s.E, kNormal);
        std::snprintf(name, sizeof name, "sample L%02d E%03d gate_up", s.L, s.E);
        failures += compare_packed(name, s.gu, x);
        std::snprintf(name, sizeof name, "sample L%02d E%03d down", s.L, s.E);
        failures += compare_packed(name, s.dn, xd);
        PackInfo a, b;
        pack_plan(s.gu.scales.data(), s.gu.N, s.gu.K, &a);
        pack_plan(s.dn.scales.data(), s.dn.N, s.dn.K, &b);
        worst_gu = std::max(worst_gu, a.n_exc);
        worst_dn = std::max(worst_dn, b.n_exc);
    }
    if (!loaded) {
        std::printf("no sample experts in %s: the real-weight cases did not run\n", dir.c_str());
        ++failures;
    } else {
        std::printf("sample: %d experts, at most %d raw rounds in a gate_up and %d in a down\n", loaded, worst_gu, worst_dn);
    }
    std::printf(failures ? "FAIL: %d packed case(s)\n" : "PASS: packed is planar bit for bit in every case\n", failures);
    return failures ? 1 : 0;
}

/* Physical cores round-robin over the four CCDs (0-5, 6-11, 12-17, 18-23 share an L3), then
 * their SMT siblings (24-47) in the same order. */
int core_for(int i) { return (i >= 24 ? 24 : 0) + ((i % 24) % 4) * 6 + ((i % 24) / 4) % 6; }

struct Pool {
    int n;
    std::vector<std::thread> threads;
    std::atomic<int> gen{ 0 }, done{ 0 };
    std::atomic<bool> stop{ false };
    std::function<void(int)> job;
    explicit Pool(int n_) : n(n_) {
        pin(0);
        for (int i = 1; i < n; ++i)
            threads.emplace_back([this, i] {
                pin(i);
                int seen = 0;
                for (;;) {
                    while (gen.load(std::memory_order_acquire) == seen && !stop.load()) _mm_pause();
                    if (stop.load()) return;
                    seen = gen.load();
                    job(i);
                    done.fetch_add(1, std::memory_order_acq_rel);
                }
            });
    }
    static void pin(int i) {
        cpu_set_t s;
        CPU_ZERO(&s);
        CPU_SET(core_for(i), &s);
        pthread_setaffinity_np(pthread_self(), sizeof s, &s);
    }
    void run(std::function<void(int)> f) {
        job = std::move(f);
        done.store(0);
        gen.fetch_add(1, std::memory_order_acq_rel);
        job(0);
        while (done.load(std::memory_order_acquire) < n - 1) _mm_pause();
    }
    ~Pool() { stop = true; for (auto& t : threads) t.join(); }
};

/* bench: each projection's rows split into one range a thread. tile > 0: into `tile`-row tiles that
 * the threads take in turn from a shared counter, as the engine's pool does. */
int bench(int experts, const std::vector<int>& counts, int tile = 0) {
    std::printf("building %d distinct experts (%.1f GB)...\n", experts, experts * 18.80064e6 / 1e9);
    std::vector<Mat> gu, dn;
    for (int i = 0; i < experts; ++i) {
        gu.push_back(make_mat(2 * kFf, kEmbd, 100u + (uint64_t)i, kReal, false));
        dn.push_back(make_mat(kEmbd, kFf, 900u + (uint64_t)i, kReal, false));
    }
    std::vector<float> silu(65536);
    silu_table_host(silu.data(), kLimit);
    const auto x = make_x(kEmbd, 3u, kNormal);
    std::vector<uint16_t> g(2 * kFf), a(kFf), d(kEmbd);
    const double bytes = (double)experts * (2.0 * kFf * kEmbd + (double)kEmbd * kFf) / 32 * 17;
    for (int t : counts) {
        Pool pool(t);
        std::atomic<int> next{ 0 };
        auto rows = [&](const Mx& m, const uint16_t* in, uint16_t* out, int k) {
            if (tile <= 0) {
                const int q = m.N / 8;
                gemv_avx2(m, in, out, q * k / t * 8, q * (k + 1) / t * 8);
                return;
            }
            for (int i; (i = next.fetch_add(1)) * tile < m.N;) gemv_avx2(m, in, out, i * tile, std::min(m.N, (i + 1) * tile));
        };
        auto pass = [&]() {
            for (int i = 0; i < experts; ++i) {
                const Mx G = gu[(size_t)i].ggml(), D = dn[(size_t)i].ggml();
                next = 0;
                pool.run([&](int k) { rows(G, x.data(), g.data(), k); });
                swiglu(silu.data(), g.data(), a.data(), kFf, kLimit);
                next = 0;
                pool.run([&](int k) { rows(D, a.data(), d.data(), k); });
            }
        };
        pass();   /* warm the code and the pages, not the caches: the set exceeds L3 */
        const auto t0 = std::chrono::steady_clock::now();
        pass();
        pass();
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("%2d threads%s: %7.3f ms per expert, %6.1f GB/s of weights\n", t, tile > 0 ? " (tiles)" : "",
                    1e3 * s / (2.0 * experts), 2.0 * bytes / s / 1e9);
    }
    return 0;
}

/* pkbench: the engine pool's shape (64-row tiles from a shared counter, gate_up's then down's)
 * over `experts` distinct experts held as block_mxfp4, packed and as the release's planes, timed
 * in turn (A/B/C, seven times) at each thread count. The experts' scales are the research
 * sample's real ones (cycled), or the release's mix when there is no sample; the codes are random.
 * Bytes a second count what each layout's kernel reads: block 17 a block, planes 16 + 1, packed 16
 * + its 2-bit codes (+ the few raw rounds). */
int pkbench(int experts, const std::vector<int>& counts, const std::string& dir) {
    std::printf("building %d distinct experts in three layouts (%.1f GB)...\n", experts, experts * 3 * 18.8e6 / 1e9);
    const auto ids = sample_ids(dir);
    std::vector<Mat> gu, dn;
    std::vector<Packed> pgu, pdn;
    int raw_gu = 0, raw_dn = 0;
    double packed_read = 0;   /* bytes the packed kernels read for one pass over the set */
    for (int i = 0; i < experts; ++i) {
        SampleExpert x;
        const bool real = !ids.empty() && load_expert(dir, ids[(size_t)i % ids.size()].first, ids[(size_t)i % ids.size()].second, &x);
        if (!real) {
            auto mix = [](int, int, uint64_t r) { return release_like(r, 119, 3000); };
            x.gu = make_mat_with(2 * kFf, kEmbd, 100u + (uint64_t)i, mix);
            x.dn = make_mat_with(kEmbd, kFf, 900u + (uint64_t)i, mix);
        }   /* a sample expert used again is a copy in its own memory: the set still streams from DRAM */
        pgu.push_back(pack(x.gu));
        pdn.push_back(pack(x.dn));
        if (!pgu.back().ok() || !pdn.back().ok()) { std::printf("packing failed\n"); return 1; }
        raw_gu = std::max(raw_gu, pgu.back().plan.n_exc);
        raw_dn = std::max(raw_dn, pdn.back().plan.n_exc);
        for (const Packed* q : { &pgu.back(), &pdn.back() })
            packed_read += (double)q->N * (q->K / 2) + (double)q->N * (double)pk_row_bytes(q->K) + 36.0 * q->plan.n_exc;
        gu.push_back(std::move(x.gu));
        dn.push_back(std::move(x.dn));
    }
    std::printf("scales: %s; at most %d raw rounds in a gate_up, %d in a down\n",
                ids.empty() ? "synthetic release mix" : "the research sample's", raw_gu, raw_dn);
    std::vector<float> silu(65536);
    silu_table_host(silu.data(), kLimit);
    const auto x = make_x(kEmbd, 3u, kNormal);
    const char* const name[3] = { "block_mxfp4", "packed", "planes" };
    std::vector<uint16_t> g[3], a[3], d[3];
    for (int l = 0; l < 3; ++l) { g[l].resize(2 * kFf); a[l].resize(kFf); d[l].resize(kEmbd); }
    const double block_read = 18.80064e6 * experts, read[3] = { block_read, packed_read, block_read };
    for (int t : counts) {
        Pool pool(t);
        std::atomic<int> next{ 0 };
        auto rows = [&](const Mx& m, const uint16_t* in, uint16_t* out) {
            for (int i; (i = next.fetch_add(1)) * 64 < m.N;) gemv_avx2(m, in, out, i * 64, std::min(m.N, (i + 1) * 64));
        };
        auto pass = [&](int lay) {
            for (int i = 0; i < experts; ++i) {
                const Mx G = lay == 1 ? pgu[(size_t)i].view() : lay == 2 ? gu[(size_t)i].planes() : gu[(size_t)i].ggml();
                const Mx D = lay == 1 ? pdn[(size_t)i].view() : lay == 2 ? dn[(size_t)i].planes() : dn[(size_t)i].ggml();
                next = 0;
                pool.run([&](int) { rows(G, x.data(), g[lay].data()); });
                swiglu(silu.data(), g[lay].data(), a[lay].data(), kFf, kLimit);
                next = 0;
                pool.run([&](int) { rows(D, a[lay].data(), d[lay].data()); });
            }
        };
        /* A sample: `passes` passes over the set, long enough (~0.1 s) for a steady rate. */
        const int passes = std::max(1, 320 / experts), samples = 7;
        auto timed = [&](int lay) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int p = 0; p < passes; ++p) pass(lay);
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / passes;
        };
        for (int lay = 0; lay < 3; ++lay) timed(lay);   /* warm the code and the pages, not the caches */
        std::vector<double> ts[3];
        for (int rep = 0; rep < samples; ++rep)
            for (int lay = 0; lay < 3; ++lay) ts[lay].push_back(timed(lay));
        double med[3];
        for (int lay = 0; lay < 3; ++lay) {
            std::sort(ts[lay].begin(), ts[lay].end());
            med[lay] = ts[lay][(size_t)samples / 2];
        }
        const bool same = !diff(d[0], d[1]) && !diff(d[0], d[2]) && !diff(g[0], g[1]) && !diff(g[0], g[2]);
        std::printf("%2d threads (tiles, %d samples of %d passes over %d experts), median ms per expert:\n", t, samples, passes, experts);
        for (int lay = 0; lay < 3; ++lay)
            std::printf("   %-11s %.4f (best %.4f, worst %.4f)  %+6.2f%% vs block_mxfp4  %5.1f GB/s read\n", name[lay],
                        1e3 * med[lay] / experts, 1e3 * ts[lay][0] / experts, 1e3 * ts[lay][(size_t)samples - 1] / experts,
                        100.0 * (med[lay] / med[0] - 1.0), read[lay] / med[lay] / 1e9);
        std::printf("   last expert's rows %s in the three layouts\n", same ? "identical" : "DIFFER");
    }
    return 0;
}

/* ---- insbench: a background insert beside the pool's streaming --------------------------------------
 *
 * The RAM cache's background inserts (RADIANCE_EXPERIMENTAL_DS41_INSERT_NT) against the pool's GEMV:
 * `threads` workers stream `experts` distinct packed experts in 64-row tiles, `batch` experts' tiles
 * from one shared counter at a time (a burst of four ranks' jobs: the DRAM-bound regime), while one
 * more thread packs `units` missed units' planes into slots of their own, back to back (the insert's
 * work, ~18 MB a unit): with memcpy's cached stores, with copy_nt's streaming stores, not at all
 * (asleep), or spinning without touching memory (the control for a busy core). Seven samples of each,
 * the order rotated each time; the median GEMV rate, the ms an insert took, and what the inserts cost
 * the GEMV in bytes per byte inserted. One unit's slots by the two copies are compared first. */
int insbench(int experts, int threads, int units, int batch) {
    batch = std::max(1, std::min(batch, experts));
    std::printf("building %d packed experts and %d missed units with slots (%.2f GB)...\n", experts, units,
                (experts * 18.0e6 + units * (17.7e6 + 1.1e6 + 18.0e6)) / 1e9);
    auto mix = [](int, int, uint64_t r) { return release_like(r, 119, 3000); };
    std::vector<Packed> pgu, pdn;
    double packed_read = 0;
    for (int i = 0; i < experts; ++i) {
        const Mat g = make_mat_with(2 * kFf, kEmbd, 100u + (uint64_t)i, mix), d = make_mat_with(kEmbd, kFf, 900u + (uint64_t)i, mix);
        pgu.push_back(pack(g, 512));   /* the engine's raw-round budgets */
        pdn.push_back(pack(d, 320));
        if (!pgu.back().ok() || !pdn.back().ok()) { std::printf("packing failed\n"); return 1; }
        for (const Packed* q : { &pgu.back(), &pdn.back() })
            packed_read += (double)q->N * (q->K / 2) + (double)q->N * (double)pk_row_bytes(q->K) + 36.0 * q->plan.n_exc;
    }
    struct Unit { Mat w[2]; PackInfo plan[2]; std::vector<uint8_t> slot[2]; };
    const int shape[2][3] = { { 2 * kFf, kEmbd, 512 }, { kEmbd, kFf, 320 } };   /* N, K, cap */
    std::vector<Unit> u((size_t)units);
    double unit_bytes = 0;
    for (int i = 0; i < units; ++i)
        for (int k = 0; k < 2; ++k) {
            Unit& x = u[(size_t)i];
            x.w[k] = make_mat_with(shape[k][0], shape[k][1], 5000u + 2u * (uint64_t)i + (uint64_t)k, mix);
            std::vector<uint8_t>().swap(x.w[k].blocks);
            if (!pack_plan(x.w[k].scales.data(), shape[k][0], shape[k][1], &x.plan[k]) || x.plan[k].n_exc > shape[k][2]) {
                std::printf("unit %d over its budget\n", i);
                return 1;
            }
            x.slot[k].assign(packed_bytes(shape[k][0], shape[k][1], shape[k][2]), 0);
            if (i == 0) unit_bytes += (double)x.slot[k].size();
        }
    auto insert = [&](Unit& x, bool nt) {
        PackCopy cp;
        cp.nt = true;
        for (int k = 0; k < 2; ++k)
            if (pack_planes(x.w[k].codes.data(), x.w[k].scales.data(), shape[k][0], shape[k][1], shape[k][2], x.plan[k],
                            x.slot[k].data(), x.slot[k].size(), nt ? &cp : nullptr) != kPackOk)
                return false;
        return true;
    };
    {
        Unit& x = u[0];
        std::vector<uint8_t> a[2];
        bool ok = insert(x, false);
        for (int k = 0; k < 2; ++k) a[k] = x.slot[k];
        ok = ok && insert(x, true);
        for (int k = 0; k < 2; ++k) ok = ok && a[k] == x.slot[k];
        std::printf("one unit packed by memcpy and by copy_nt: %s\n", ok ? "identical slots" : "SLOTS DIFFER");
        if (!ok) return 1;
    }
    std::vector<float> silu(65536);
    silu_table_host(silu.data(), kLimit);
    const auto xv = make_x(kEmbd, 3u, kNormal);
    std::vector<std::vector<uint16_t>> g((size_t)batch, std::vector<uint16_t>(2 * kFf)), act((size_t)batch, std::vector<uint16_t>(kFf)),
                                       d((size_t)batch, std::vector<uint16_t>(kEmbd));
    Pool pool(threads);
    /* The inserter's mode: 0 asleep between looks, 1 memcpy, 2 copy_nt, 3 spinning (no memory). */
    enum { kNone, kMemcpy, kNt, kSpin, kModes };
    std::atomic<int> mode{ kNone };
    std::atomic<bool> quit{ false }, busy{ false };
    std::atomic<uint64_t> ins_n{ 0 }, ins_ns{ 0 };
    std::thread inserter([&] {
        if (threads < 48) Pool::pin(threads);
        for (size_t i = 0; !quit.load();) {
            const int md = mode.load();
            if (md == kNone) {
                busy = false;
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                continue;
            }
            busy = true;
            if (md == kSpin) {
                for (int r = 0; r < 256; ++r) _mm_pause();
                continue;
            }
            const auto t0 = std::chrono::steady_clock::now();
            insert(u[i++ % (size_t)units], md == kNt);
            ins_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
            ++ins_n;
        }
    });
    std::atomic<int> next{ 0 };
    /* `nb` matrices' tiles of `per` 64-row tiles each, from one counter. */
    auto tiles = [&](const Mx* const* m, const uint16_t* const* in, uint16_t* const* out, int nb, int per) {
        for (int t; (t = next.fetch_add(1)) < nb * per;) {
            const int e = t / per, r = t % per;
            gemv_avx2(*m[e], in[e], out[e], r * 64, std::min(m[e]->N, (r + 1) * 64));
        }
    };
    const int passes = std::max(1, 640 / experts), samples = 7;
    auto timed = [&](int md) {   /* s a pass over the set; the inserts it saw, and their ns */
        mode = md;
        while (md != kNone && !busy.load()) _mm_pause();
        const uint64_t n0 = ins_n.load(), t0n = ins_ns.load();
        const auto t0 = std::chrono::steady_clock::now();
        for (int p = 0; p < passes; ++p)
            for (int i0 = 0; i0 < experts; i0 += batch) {
                const int nb = std::min(batch, experts - i0);
                std::vector<Mx> G, D;
                std::vector<const Mx*> gp, dp;
                std::vector<const uint16_t*> xin, ain;
                std::vector<uint16_t*> gout, dout;
                for (int b = 0; b < nb; ++b) {
                    G.push_back(pgu[(size_t)(i0 + b)].view());
                    D.push_back(pdn[(size_t)(i0 + b)].view());
                }
                for (int b = 0; b < nb; ++b) {
                    gp.push_back(&G[(size_t)b]);
                    dp.push_back(&D[(size_t)b]);
                    xin.push_back(xv.data());
                    ain.push_back(act[(size_t)b].data());
                    gout.push_back(g[(size_t)b].data());
                    dout.push_back(d[(size_t)b].data());
                }
                next = 0;
                pool.run([&](int) { tiles(gp.data(), xin.data(), gout.data(), nb, 2 * kFf / 64); });
                for (int b = 0; b < nb; ++b) swiglu(silu.data(), g[(size_t)b].data(), act[(size_t)b].data(), kFf, kLimit);
                next = 0;
                pool.run([&](int) { tiles(dp.data(), ain.data(), dout.data(), nb, kEmbd / 64); });
            }
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / passes;
        const std::array<double, 3> r{ s, (double)(ins_n.load() - n0), (double)(ins_ns.load() - t0n) };
        mode = kNone;
        while (busy.load()) _mm_pause();   /* the insert in flight ends before the next sample */
        return r;
    };
    for (int md = 0; md < kModes; ++md) timed(md);   /* warm the code and the pages */
    std::vector<std::array<double, 3>> ts[kModes];
    for (int rep = 0; rep < samples; ++rep)
        for (int j = 0; j < kModes; ++j) {
            const int md = (rep + j) % kModes;
            ts[md].push_back(timed(md));
        }
    quit = true;
    inserter.join();
    const char* const name[kModes] = { "no inserts", "memcpy", "copy_nt", "spin" };
    std::printf("%d threads, %d experts a burst, %d samples of %d passes over %d packed experts (%.0f MB a pass), an insert "
                "%.1f MB:\n", threads, batch, samples, passes, experts, packed_read / 1e6, unit_bytes / 1e6);
    double base = 0;
    for (int md = 0; md < kModes; ++md) {
        std::sort(ts[md].begin(), ts[md].end());
        const auto& m = ts[md][(size_t)samples / 2];
        double n = 0, ns = 0, s = 0;
        for (const auto& t : ts[md]) { n += t[1]; ns += t[2]; s += t[0] * passes; }
        const double gemv = packed_read / m[0], ins_rate = n * unit_bytes / s;
        if (md == kNone) base = gemv;
        std::printf("   %-10s GEMV %.4f ms an expert (best %.4f, worst %.4f), %5.1f GB/s, %+.2f%%", name[md], 1e3 * m[0] / experts,
                    1e3 * ts[md][0][0] / experts, 1e3 * ts[md][(size_t)samples - 1][0] / experts, gemv / 1e9,
                    100.0 * (gemv / base - 1.0));
        if (md == kMemcpy || md == kNt)
            std::printf("; inserts %.2f ms each, %.2f GB/s; %.2f GEMV bytes lost a byte inserted", n ? ns / n / 1e6 : 0.0,
                        ins_rate / 1e9, ins_rate > 0 ? (base - gemv) / ins_rate : 0.0);
        std::printf("\n");
    }
    return 0;
}

/* ---- rbench: what a token row more costs the pool ---------------------------------------------------
 *
 * `experts` distinct experts in one layout (block, packed or planes; the research sample's real scales
 * as in pkbench, cycled), streamed from DRAM, each with R token rows for each R of the list. The
 * pool's shape (core/place/host_expert_pool.cpp): an expert's rows as jobs of at most `per` rows, all
 * queued at once; workers claim 64-row tiles of the oldest job with one left, its gate_up tiles first;
 * the worker that ends a job's last gate_up tile runs its rows' swiglu and opens its down tiles.
 *   four  jobs of up to four rows through gemv_avx2_few (today's pool),
 *   rows  one job of all R rows through gemv_avx2_rows (gemv_avx2_many above kFewRows),
 *   many  one job of all R rows through gemv_avx2_many, at any R (in calls of 32),
 *   m32   the pool at ..._MAX_ROWS=32: ceil(R / 32) jobs as even as R allows, gemv_avx2_many,
 *   m64   the same at ..._MAX_ROWS=64 (large-chunk prefill's choice: a hot expert's weight tile read
 *         from DRAM once for 33 to 64 rows, where two 32-row jobs read it twice).
 * R up to kRbRows (128). Seven samples of each, interleaved; the median ms an expert, and what each row
 * past the first costs against four's one-row expert. Every variant's rows are checked against four's. */
constexpr int kRbRows = 128;
struct RJob {
    const Mx* gu = nullptr;
    const Mx* dn = nullptr;
    int rows = 0;
    const uint16_t* x[kRbRows] = {};
    uint16_t* y[kRbRows] = {};
    uint16_t* gate_up[kRbRows] = {};
    uint16_t* act[kRbRows] = {};
    std::atomic<int> next{ 0 }, gu_left{ 0 }, dn_left{ 0 };
    std::atomic<bool> open{ false };
};

int rbench(int experts, int threads, const char* layout, const std::vector<int>& row_counts, const std::string& dir,
           bool stream_mode = false) {
    if (stream_mode && (experts < 1 || experts > 256 || threads < 1 || threads > 256 || row_counts.empty() ||
        (std::strcmp(layout, "packed") && std::strcmp(layout, "planes")) ||
        std::any_of(row_counts.begin(), row_counts.end(), [](int r) { return r != 1; }))) {
        std::fprintf(stderr, "rbench-stream requires 1..256 experts/threads, packed|planes, and only row count 1\n");
        return 2;
    }
    const int lay = !std::strcmp(layout, "block") ? 0 : !std::strcmp(layout, "planes") ? 2 : 1;
    const char* const lname[3] = { "block_mxfp4", "packed", "planes" };
    std::printf("building %d distinct experts as %s (%.1f GB)...\n", experts, lname[lay], experts * 18.8e6 / 1e9);
    const auto ids = sample_ids(dir);
    std::vector<Mat> gu((size_t)experts), dn((size_t)experts);
    std::vector<Packed> pgu((size_t)experts), pdn((size_t)experts);
    std::vector<Mx> G((size_t)experts), D((size_t)experts);
    for (int i = 0; i < experts; ++i) {
        SampleExpert s;
        const bool real = !ids.empty() && load_expert(dir, ids[(size_t)i % ids.size()].first, ids[(size_t)i % ids.size()].second, &s);
        if (!real) {
            auto mix = [](int, int, uint64_t r) { return release_like(r, 119, 3000); };
            s.gu = make_mat_with(2 * kFf, kEmbd, 100u + (uint64_t)i, mix);
            s.dn = make_mat_with(kEmbd, kFf, 900u + (uint64_t)i, mix);
        }
        if (lay == 1) {
            pgu[(size_t)i] = pack(s.gu);
            pdn[(size_t)i] = pack(s.dn);
            if (!pgu[(size_t)i].ok() || !pdn[(size_t)i].ok()) { std::printf("packing failed\n"); return 1; }
            G[(size_t)i] = pgu[(size_t)i].view();
            D[(size_t)i] = pdn[(size_t)i].view();
            continue;
        }
        gu[(size_t)i] = std::move(s.gu);
        dn[(size_t)i] = std::move(s.dn);
        Mat& a = gu[(size_t)i];
        Mat& b = dn[(size_t)i];
        if (lay == 0) { std::vector<uint8_t>().swap(a.codes); std::vector<uint8_t>().swap(a.scales); std::vector<uint8_t>().swap(b.codes); std::vector<uint8_t>().swap(b.scales); }
        else { std::vector<uint8_t>().swap(a.blocks); std::vector<uint8_t>().swap(b.blocks); }
        G[(size_t)i] = lay == 0 ? a.ggml() : a.planes();
        D[(size_t)i] = lay == 0 ? b.ggml() : b.planes();
    }
    std::printf("scales: %s\n", ids.empty() ? "synthetic release mix" : "the research sample's");
    if (stream_mode) {
        size_t weight_bytes = 0;
        for (int i = 0; i < experts; ++i)
            weight_bytes += lay == 1 ? pgu[(size_t)i].bytes + pdn[(size_t)i].bytes
                                    : gu[(size_t)i].codes.size() + gu[(size_t)i].scales.size() +
                                      dn[(size_t)i].codes.size() + dn[(size_t)i].scales.size();
        std::printf("DS41HX_STREAM_BENCH working_set_bytes=%zu experts=%d rows=1 warmup=same_as_rbench samples=7 "
                    "timed_call_counters=0 cache_flush=0\n", weight_bytes, experts);
    }
    std::vector<float> silu(65536);
    silu_table_host(silu.data(), kLimit);
    int rmax = 1;
    for (int r : row_counts) rmax = std::max(rmax, std::min(r, kRbRows));
    std::vector<std::vector<uint16_t>> xs((size_t)rmax);
    for (int i = 0; i < rmax; ++i) xs[(size_t)i] = make_x(kEmbd, 500u + (uint64_t)i, kNormal);
    /* y, gate_up and act of every expert and row; four's y kept for the check */
    const size_t per_e = (size_t)rmax * kEmbd;
    std::vector<uint16_t> yb((size_t)experts * per_e), y4((size_t)experts * per_e);
    std::vector<uint16_t> gub((size_t)experts * rmax * 2 * kFf), acb((size_t)experts * rmax * kFf);
    Pool pool(threads);
    using Kern = void (*)(const Mx&, const uint16_t* const*, uint16_t* const*, int, int, int);
    constexpr int kV = 5;
    const int variants = stream_mode ? 2 : kV;
    const char* const vname[kV] = { "four", "rows", "many", "m32", "m64" };
    const Kern kern[kV] = { gemv_avx2_few, gemv_avx2_rows, gemv_avx2_many, gemv_avx2_many, gemv_avx2_many };
    double one_row = 0;   /* four's ms an expert at one row */
    bool stream_ok = true;
    for (int R : row_counts) {
        R = std::max(1, std::min(R, kRbRows));
        const double bytes = 18.80064e6;
        auto build = [&](int v, std::vector<std::unique_ptr<RJob>>& jobs) {
            jobs.clear();
            /* Rows a job: four; all R (rows, many); or the pool's even cut at 32 or 64. */
            const int per = v == 0 ? kFewRows : v == 3 ? 32 : v == 4 ? 64 : R;
            const int nj = (R + per - 1) / per;
            for (int e = 0; e < experts; ++e)
                for (int q = 0, r0 = 0; q < nj; ++q) {
                    auto j = std::make_unique<RJob>();
                    j->gu = &G[(size_t)e];
                    j->dn = &D[(size_t)e];
                    j->rows = v == 0 ? std::min(per, R - r0) : R / nj + (q < R % nj ? 1 : 0);
                    for (int i = 0; i < j->rows; ++i) {
                        const int r = r0 + i;
                        j->x[i] = xs[(size_t)r].data();
                        j->y[i] = yb.data() + (size_t)e * per_e + (size_t)r * kEmbd;
                        j->gate_up[i] = gub.data() + ((size_t)e * rmax + (size_t)r) * 2 * kFf;
                        j->act[i] = acb.data() + ((size_t)e * rmax + (size_t)r) * kFf;
                    }
                    r0 += j->rows;
                    jobs.push_back(std::move(j));
                }
        };
        std::vector<std::unique_ptr<RJob>> jobs[kV];
        for (int v = 0; v < variants; ++v) build(v, jobs[v]);
        std::atomic<size_t> proof_calls{ 0 }, proof_fast{ 0 }, proof_fallback_only{ 0 };
        auto run = [&](int v, bool proof = false) {
            std::vector<std::unique_ptr<RJob>>& js = jobs[v];
            for (auto& j : js) {
                j->next.store(0, std::memory_order_relaxed);
                j->gu_left.store(72, std::memory_order_relaxed);
                j->dn_left.store(80, std::memory_order_relaxed);
                j->open.store(false, std::memory_order_relaxed);
            }
            std::atomic<int> done{ 0 };
            const int nj = (int)js.size();
            const Kern k = kern[v];
            const auto t0 = std::chrono::steady_clock::now();
            pool.run([&](int) {
                while (done.load(std::memory_order_acquire) < nj) {
                    bool got = false;
                    for (int q = 0; q < nj && !got; ++q) {
                        RJob& j = *js[(size_t)q];
                        int t = j.next.load(std::memory_order_acquire);
                        while (t < 152 && (t < 72 || j.open.load(std::memory_order_acquire))) {
                            if (!j.next.compare_exchange_weak(t, t + 1, std::memory_order_acq_rel)) continue;
                            got = true;
                            break;
                        }
                        if (!got) continue;
                        auto project = [&](const Mx& w, const uint16_t* const* x, uint16_t* const* y, int n0, int n1) {
                            if (stream_mode && v == 1) {
                                const bool used = gemv_avx2_stream(w, x[0], y[0], n0, n1);
                                // Only the explicit untimed byte proof counts. No timed atomics.
                                if (proof) {
                                    proof_calls.fetch_add(1, std::memory_order_relaxed);
                                    (used ? proof_fast : proof_fallback_only).fetch_add(1, std::memory_order_relaxed);
                                }
                            } else k(w, x, y, j.rows, n0, n1);
                        };
                        if (t < 72) {
                            project(*j.gu, j.x, j.gate_up, t * 64, t * 64 + 64);
                            if (j.gu_left.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                                for (int i = 0; i < j.rows; ++i) swiglu_avx2(silu.data(), j.gate_up[i], j.act[i], kFf, kLimit);
                                j.open.store(true, std::memory_order_release);
                            }
                        } else {
                            const int d = t - 72;
                            project(*j.dn, j.act, j.y, d * 64, d * 64 + 64);
                            if (j.dn_left.fetch_sub(1, std::memory_order_acq_rel) == 1) done.fetch_add(1, std::memory_order_acq_rel);
                        }
                    }
                    if (!got) _mm_pause();
                }
            });
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        };
        if (stream_mode) {
            // Same buffers/weights/tiling, all three host-chain stages, before timing either arm.
            run(0);
            const auto old_gu = gub, old_act = acb, old_down = yb;
            run(1, true);
            const bool exact = old_gu == gub && old_act == acb && old_down == yb;
            const size_t calls = proof_calls.load(), fast = proof_fast.load(), fallback = proof_fallback_only.load();
            std::printf("DS41HX_STREAM_BENCH proof=untimed calls=%zu calls_with_fast=%zu calls_fallback_only=%zu "
                        "gu_act_down_equal=%d\n", calls, fast, fallback, exact ? 1 : 0);
            if (!exact || calls != (size_t)152 * experts || !fast || calls != fast + fallback) return 1;
        }
        const int passes = std::max(1, 64 / (experts * R) + 1), samples = 7;
        auto timed = [&](int v) {
            double s = 0;
            for (int p = 0; p < passes; ++p) s += run(v);
            return s / passes;
        };
        for (int v = 0; v < variants; ++v) timed(v);   /* warm the code and the pages, not the caches */
        std::vector<double> ts[kV];
        size_t bad[kV] = {};
        for (int rep = 0; rep < samples; ++rep)
            for (int v = 0; v < variants; ++v) {
                std::fill(yb.begin(), yb.end(), (uint16_t)0xffff);
                ts[v].push_back(timed(v));
                if (v == 0 && rep == 0) y4 = yb;
                else
                    for (int e = 0; e < experts; ++e)
                        for (size_t i = 0; i < (size_t)R * kEmbd; ++i) bad[v] += yb[(size_t)e * per_e + i] != y4[(size_t)e * per_e + i];
            }
        double med[kV];
        for (int v = 0; v < variants; ++v) {
            std::sort(ts[v].begin(), ts[v].end());
            med[v] = 1e3 * ts[v][(size_t)samples / 2] / experts;
        }
        if (R == 1 || one_row == 0) one_row = med[0];
        std::printf("%2d rows, %d threads, %d experts, %s: median ms an expert (best, worst), per row past the first:\n", R, threads,
                    experts, lname[lay]);
        for (int v = 0; v < variants; ++v) {
            const int reads = v == 0 ? (R + 3) / 4 : v == 3 ? (R + 31) / 32 : v == 4 ? (R + 63) / 64 : 1;
            const char* name = stream_mode ? (v == 0 ? "old" : "stream") : vname[v];
            std::printf("   %-4s %8.4f (%.4f, %.4f)  %.4f ms a row, %+.4f ms each row more%s  %5.1f GB/s  rows %s\n", name, med[v],
                        1e3 * ts[v][0] / experts, 1e3 * ts[v][(size_t)samples - 1] / experts, med[v] / R,
                        R > 1 ? (med[v] - one_row) / (R - 1) : 0.0, R > 1 ? "" : " (one row)", bytes * reads / (med[v] * 1e-3) / 1e9,
                        bad[v] ? "DIFFER" : "identical");
            if (stream_mode && bad[v]) stream_ok = false;
        }
    }
    if (stream_mode) {
        std::printf(stream_ok ? "PASS: rbench-stream full outputs equal old host; timings include unchanged pool/tile/SwiGLU work\n"
                              : "FAIL: rbench-stream output bytes differ\n");
        return stream_ok ? 0 : 1;
    }
    return 0;
}

/* AMD APERF: core cycles of this CPU (RDPRU, user mode), 0 where unsupported. */
uint64_t aperf() {
#if defined(__x86_64__)
    uint32_t lo, hi;
    asm volatile(".byte 0x0f, 0x01, 0xfd" : "=a"(lo), "=d"(hi) : "c"(1));   /* rdpru */
    return ((uint64_t)hi << 32) | lo;
#else
    return 0;
#endif
}

/* One matrix of `rows` x 5120 (default the gate_up's 4608 rows, 12.5 MB: in one CCD's L3) on one
 * pinned thread, `reps` times: the kernel's own rate, for perf stat. */
int hot(int reps, int rows, bool planes) {
    const Mat m = make_mat(rows, kEmbd, 77u, kReal, planes);
    const auto x = make_x(kEmbd, 3u, kNormal);
    std::vector<uint16_t> y((size_t)rows);
    Pool::pin(0);
    const Mx w = planes ? m.planes() : m.ggml();
    gemv_avx2(w, x.data(), y.data(), 0, rows);   /* warm */
    const auto t0 = std::chrono::steady_clock::now();
    const uint64_t c0 = aperf();
    for (int r = 0; r < reps; ++r) gemv_avx2(w, x.data(), y.data(), 0, rows);
    const uint64_t c1 = aperf();
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double weights = (double)reps * rows * kEmbd;
    std::printf("hot %s %dx%d x %d: %.3f ms a call, %.2f GB/s of weights, %.3f ns a weight, %.3f cycles a weight (%.2f GHz)\n",
                planes ? "planes" : "blocks", rows, kEmbd, reps, 1e3 * s / reps, weights / 32 * 17 / s / 1e9, 1e9 * s / weights,
                (double)(c1 - c0) / weights, (double)(c1 - c0) / s / 1e9);
    return 0;
}

/* rhot: gemv_avx2_few and gemv_avx2_many on `rows` token rows of one matrix of `n` x 5120 in one
 * layout (12.5 MB at 4608 rows: in one CCD's L3), one pinned thread, `reps` passes over it in calls of
 * `tile` output rows (64, the pool's; 0: one call): the kernels' own cost a row and weight. */
int rhot(int reps, int rows, const char* layout, int n, int tile) {
    if (tile <= 0 || tile > n) tile = n;
    rows = std::max(1, std::min(rows, kGemvRows));
    const int lay = !std::strcmp(layout, "block") ? 0 : !std::strcmp(layout, "planes") ? 2 : 1;
    auto mix = [](int, int, uint64_t r) { return release_like(r, 119, 3000); };
    const Mat m = make_mat_with(n, kEmbd, 77u, mix);
    const Packed p = lay == 1 ? pack(m) : Packed{};
    const Mx w = lay == 0 ? m.ggml() : lay == 2 ? m.planes() : p.view();
    std::vector<std::vector<uint16_t>> xv((size_t)rows), yv((size_t)rows, std::vector<uint16_t>((size_t)n));
    const uint16_t* xs[kGemvRows];
    uint16_t* ys[kGemvRows];
    for (int i = 0; i < rows; ++i) {
        xv[(size_t)i] = make_x(kEmbd, 3u + (uint64_t)i, kNormal);
        xs[i] = xv[(size_t)i].data();
        ys[i] = yv[(size_t)i].data();
    }
    Pool::pin(0);
    using Kern = void (*)(const Mx&, const uint16_t* const*, uint16_t* const*, int, int, int);
    const Kern kern[2] = { gemv_avx2_few, gemv_avx2_many };
    const char* const name[2] = { "few", "many" };
    for (int k = 0; k < 2; ++k) {
        auto pass = [&] { for (int t = 0; t < n; t += tile) kern[k](w, xs, ys, rows, t, std::min(n, t + tile)); };
        pass();   /* warm */
        const auto t0 = std::chrono::steady_clock::now();
        const uint64_t c0 = aperf();
        for (int r = 0; r < reps; ++r) pass();
        const uint64_t c1 = aperf();
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const double rw = (double)reps * rows * n * kEmbd;   /* row-weights */
        std::printf("rhot %-4s %s %dx%d in tiles of %d, %d rows x %d: %.3f ms a pass, %.4f ns and %.4f cycles a row-weight (%.2f GHz)\n",
                    name[k], layout, n, kEmbd, tile, rows, reps, 1e3 * s / reps, 1e9 * s / rw, (double)(c1 - c0) / rw,
                    (double)(c1 - c0) / s / 1e9);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && !std::strcmp(argv[1], "rbench-stream")) {
        std::vector<int> rows;
        for (int i = 5; i < argc; ++i) rows.push_back(std::atoi(argv[i]));
        if (rows.empty()) rows = { 1 };
        return rbench(argc > 2 ? std::atoi(argv[2]) : 32, argc > 3 ? std::atoi(argv[3]) : 40,
                      argc > 4 ? argv[4] : "packed", rows, sample_dir(0, argv, 1), true);
    }
    if (argc > 1 && !std::strcmp(argv[1], "rbench")) {
        /* rbench [experts] [threads] [block|packed|planes] [rows...] */
        std::vector<int> rows;
        for (int i = 5; i < argc; ++i) rows.push_back(std::atoi(argv[i]));
        if (rows.empty()) rows = { 1, 2, 4, 8, 16, 32 };
        return rbench(argc > 2 ? std::atoi(argv[2]) : 32, argc > 3 ? std::atoi(argv[3]) : 40, argc > 4 ? argv[4] : "packed", rows,
                      sample_dir(0, argv, 1));
    }
    if (argc > 1 && !std::strcmp(argv[1], "rhot"))   /* rhot [reps] [rows] [block|packed|planes] [n] [tile] */
        return rhot(argc > 2 ? std::atoi(argv[2]) : 20, argc > 3 ? std::atoi(argv[3]) : 32, argc > 4 ? argv[4] : "packed",
                    argc > 5 ? std::atoi(argv[5]) : 2 * kFf, argc > 6 ? std::atoi(argv[6]) : 64);
    if (argc > 1 && !std::strcmp(argv[1], "bench")) {
        /* bench [experts] [thread counts...]; counts past 24 use SMT siblings */
        std::vector<int> counts;
        for (int i = 3; i < argc; ++i) counts.push_back(std::atoi(argv[i]));
        if (counts.empty()) counts = { 1, 4, 8, 12, 16, 20 };
        return bench(argc > 2 ? std::atoi(argv[2]) : 32, counts);
    }
    if (argc > 1 && !std::strcmp(argv[1], "tiles")) {
        /* tiles [experts] [thread counts...]: as bench, but 64-row tiles from a shared counter */
        std::vector<int> counts;
        for (int i = 3; i < argc; ++i) counts.push_back(std::atoi(argv[i]));
        if (counts.empty()) counts = { 1, 4, 8, 12, 16, 20 };
        return bench(argc > 2 ? std::atoi(argv[2]) : 32, counts, 64);
    }
    if (argc > 1 && !std::strcmp(argv[1], "pkbench")) {
        /* pkbench [experts] [thread counts...]: tiles, block_mxfp4 against packed and planes */
        std::vector<int> counts;
        for (int i = 3; i < argc; ++i) counts.push_back(std::atoi(argv[i]));
        if (counts.empty()) counts = { 40 };
        return pkbench(argc > 2 ? std::atoi(argv[2]) : 32, counts, sample_dir(0, argv, 1));
    }
    if (argc > 1 && !std::strcmp(argv[1], "packed")) return packed_test(sample_dir(argc, argv, 2));
    if (argc > 1 && !std::strcmp(argv[1], "insbench"))   /* insbench [experts] [threads] [units] [batch] */
        return insbench(argc > 2 ? std::atoi(argv[2]) : 32, argc > 3 ? std::atoi(argv[3]) : 40, argc > 4 ? std::atoi(argv[4]) : 8,
                        argc > 5 ? std::atoi(argv[5]) : 4);
    if (argc > 1 && !std::strcmp(argv[1], "hot"))   /* hot [reps] [rows] [planes] */
        return hot(argc > 2 ? std::atoi(argv[2]) : 200, argc > 3 ? std::atoi(argv[3]) : 2 * kFf,
                   argc > 4 && !std::strcmp(argv[4], "planes"));
    return check();
}
