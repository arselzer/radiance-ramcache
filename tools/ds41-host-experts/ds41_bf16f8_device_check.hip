/* ds41_bf16f8_device_check.hip -- gemm_nt_bf16f8 on the card against gemm_nt, byte for byte.
 *
 * Runs a built libr4d.so's own rows (dlopen, found by name through rad_kernel_count /
 * rad_kernel_at): gemm_bf16f8_nt_m64 over the stored form its own layout hooks write, against
 * gemm_bf16_nt_m64 (and gemm_bf16_nt_tiled past 64 rows) over the bf16 plane. Every comparison is
 * of bits; any difference fails the run.
 *
 *   ds41_bf16f8_device_check LIBR4D.so all [MODEL.rad]   every part below (real with a model)
 *   ds41_bf16f8_device_check LIBR4D.so frag               the B registers themselves: the kernel's
 *                                                         expansion (libr4d/r4d_bf16f8_frag.h) of
 *                                                         all 254 E4M3 codes at all 237 exponents,
 *                                                         against r4d_bf16f8_expand, bit for bit
 *   ds41_bf16f8_device_check LIBR4D.so onehot             every (code, exponent) pair decoded
 *                                                         through both GEMMs, at every legal split
 *   ds41_bf16f8_device_check LIBR4D.so zeros              signed zeros and cancelling pairs through
 *                                                         the peel, the rounds and the tail
 *   ds41_bf16f8_device_check LIBR4D.so shapes             ds41's per-rank shapes and odd ones, across
 *                                                         rows, splits, row tiles a block, chain
 *   ds41_bf16f8_device_check LIBR4D.so real MODEL.rad     one container tensor per class, rank 0 and
 *                                                         rank 3 slices, at 1 and 16 rows
 *   ds41_bf16f8_device_check LIBR4D.so time               us and GB/s a call at one row, bf16 vs
 *                                                         stored form, weights rotated past the
 *                                                         64 MB on-die cache
 *
 * frag: one wave an exponent, its 32 lanes holding the 256 byte values as the kernel holds a k
 * step's codes, dump r4d_bf16f8_frag's four VGPRs and the f32 products behind them. Every non-NaN
 * code's bf16 must be r4d_bf16f8_expand's and its product's low half zero, +-0 and the subnormal
 * codes included. A GEMM output cannot show this: its sum starts at +0, which absorbs a -0.
 *
 * onehot: a [256, 1024] weight of 256 blocks whose exponents sweep 10..246 and whose elements
 * cycle through all 254 non-NaN codes, against one-hot activations (row m = 1.0 at k = 16q + m,
 * 64 launches cover K), so every output is one weight element: both rows must give it exactly
 * (a -0 may come out +0, the same in both).
 *
 * zeros: weights mostly signed zeros, the rest pairs of opposite values along K, against
 * activations in equal pairs with signed zeros, so most outputs cancel to zero; at shapes whose
 * slices start mid-block (the peel, whose first WMMA's accumulator the compiler knows is zero),
 * read exponents byte by byte, or end on a half block. Both rows must give the same bits.
 *
 * Needs a free card (HIP device 0 unless HIP_VISIBLE_DEVICES says otherwise): run it inside a GPU
 * window under both research locks, never beside the production container. Exit 0 when every
 * comparison is equal, 1 on any difference, 2 on a setup error. Prints a PASS or FAIL line per part
 * and one summary line. */
#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "rad_abi.h"
#include "rad_format.h"
#include "r4d_args.h"
#include "r4d_bf16f8_layout.h"
#include "r4d_bf16f8_frag.h"

#define HC(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e_)); std::exit(2); } } while (0)

namespace {

int g_fail = 0, g_runs = 0;

struct Lib {
    const RadKernelInfo* f8 = nullptr;     /* gemm_bf16f8_nt_m64 */
    const RadKernelInfo* m64 = nullptr;    /* gemm_bf16_nt_m64 */
    const RadKernelInfo* tiled = nullptr;  /* gemm_bf16_nt_tiled */
    explicit Lib(const char* path) {
        void* h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!h) { std::fprintf(stderr, "dlopen %s: %s\n", path, dlerror()); std::exit(2); }
        auto count = (int (*)(void))dlsym(h, "rad_kernel_count");
        auto at = (const RadKernelInfo* (*)(int))dlsym(h, "rad_kernel_at");
        if (!count || !at) { std::fprintf(stderr, "%s is not a kernel plugin\n", path); std::exit(2); }
        for (int i = 0; i < count(); ++i) {
            const RadKernelInfo* k = at(i);
            if (!k || !k->name) continue;
            if (!std::strcmp(k->name, "gemm_bf16f8_nt_m64")) f8 = k;
            if (!std::strcmp(k->name, "gemm_bf16_nt_m64")) m64 = k;
            if (!std::strcmp(k->name, "gemm_bf16_nt_tiled")) tiled = k;
        }
        if (!f8 || !m64 || !f8->layout || !f8->relayout) {
            std::fprintf(stderr, "%s lacks gemm_bf16f8_nt_m64 (with its hooks) or gemm_bf16_nt_m64\n", path);
            std::exit(2);
        }
    }
};

struct Dev {
    void* p = nullptr;
    size_t n = 0;
    explicit Dev(size_t bytes) : n(bytes) { HC(hipMalloc(&p, bytes ? bytes : 16)); }
    ~Dev() { (void)hipFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    void put(const void* h, size_t bytes) { HC(hipMemcpy(p, h, bytes, hipMemcpyHostToDevice)); }
    void get(void* h, size_t bytes) const { HC(hipMemcpy(h, p, bytes, hipMemcpyDeviceToHost)); }
};

RadTensor view(void* p, uint32_t dt, int64_t rows, int64_t cols, int64_t pitch) {
    RadTensor t{};
    t.data = p;
    t.dtype = dt;
    t.rank = 2;
    t.shape[0] = rows;
    t.shape[1] = cols;
    t.stride[0] = pitch;
    t.stride[1] = 1;
    return t;
}
RadParam pi(const char* k, long long v) { RadParam p{}; p.key = k; p.kind = RAD_P_INT; p.ival = v; return p; }
RadParam ps(const char* k, const char* v) { RadParam p{}; p.key = k; p.kind = RAD_P_STR; p.sval = v; return p; }

/* The stored form, by the library's own hooks; empty if the relayout refused. */
std::vector<uint8_t> relay(const Lib& lib, const uint16_t* w, int64_t N, int64_t K) {
    RadParam p[4] = { pi("M", 1), pi("N", N), pi("K", K), ps("dtype", "bf16") };
    const RadEncoding enc = rad_enc_plain(RAD_BF16);
    const int sel[1] = { 0 };
    RadTensor pl = view(const_cast<uint16_t*>(w), RAD_BF16, N, K, K);
    RadLayout L{};
    if (lib.f8->layout(p, 4, 1, &enc, sel, &pl, 1, &L) != RAD_OK || L.bytes != r4d_bf16f8_bytes(N, K)) {
        std::fprintf(stderr, "layout hook refused [%lld, %lld]\n", (long long)N, (long long)K);
        std::exit(2);
    }
    std::vector<uint8_t> b((size_t)L.bytes);
    const int rc = lib.f8->relayout(p, 4, 1, &enc, sel, &pl, 1, b.data(), L.bytes);
    if (rc != RAD_OK) {
        std::fprintf(stderr, "relayout refused [%lld, %lld]: %d\n", (long long)N, (long long)K, rc);
        b.clear();
    }
    return b;
}

/* One launch of row r: y[M, N] (pitch ldc) = a[M, K] (pitch lda) . b^T. sk / mb 0: the law. */
int gemm(const RadKernelInfo* r, void* a, void* b, uint32_t bdt, void* y, int64_t M, int64_t N,
         int64_t K, int64_t lda, int64_t ldc, int sk, int mb, int chain, hipStream_t s) {
    RadTensor t[3] = { view(a, RAD_BF16, M, K, lda), view(b, bdt, N, K, K), view(y, RAD_BF16, M, N, ldc) };
    std::vector<RadParam> p = { pi("M", M), pi("N", N), pi("K", K), ps("dtype", "bf16"), pi("chain", chain) };
    if (sk) p.push_back(pi("sk", sk));
    if (mb) p.push_back(pi("mb", mb));
    RadArgs args{};
    args.t = t;
    args.n_t = 3;
    args.p = p.data();
    args.n_p = (int)p.size();
    args.world_size = 1;
    return r->launch(&args, (RadStream)s);
}

/* An FP8-exact weight: random codes (signed zeros and subnormals among them) times an exponent a
 * 32x32 block in [lo, lo + span). `zeros`: even k mostly a signed zero, odd k the negation of its
 * even neighbour (the pair shares a block). */
std::vector<uint16_t> exact_weight(int64_t N, int64_t K, std::mt19937_64& rng, int lo = 100, int span = 20,
                                   bool zeros = false) {
    const int64_t kb = K / 32;
    std::vector<uint32_t> s((size_t)(((N + 31) / 32) * kb));
    for (auto& v : s) v = (uint32_t)(lo + (int)(rng() % (uint64_t)span));
    std::vector<uint16_t> w((size_t)(N * K));
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k) {
            uint16_t& x = w[(size_t)(n * K + k)];
            if (zeros && (k % 2)) {
                const uint16_t prev = w[(size_t)(n * K + k - 1)];
                x = (prev & 0x7FFF) ? (uint16_t)(prev ^ 0x8000) : (uint16_t)((rng() & 1) << 15);
                continue;
            }
            const uint64_t r = rng() % 64;
            uint32_t c;
            if (zeros) c = r < 40 ? (uint32_t)(rng() & 0x80) : (uint32_t)((rng() % 0x7E + 1) | (rng() & 0x80));
            else if (r == 0) c = 0x00;
            else if (r == 1) c = 0x80;
            else if (r < 5) c = (uint32_t)((rng() % 7 + 1) | (rng() & 0x80));
            else do c = (uint32_t)(rng() & 0xFF); while ((c & 0x7F) == 0x7F);
            x = r4d_bf16f8_expand(c, s[(size_t)((n / 32) * kb + k / 32)]);
        }
    return w;
}

uint16_t to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

std::vector<uint16_t> random_act(int64_t rows, int64_t cols, std::mt19937_64& rng) {
    std::vector<uint16_t> a((size_t)(rows * cols));
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : a) v = to_bf16(nd(rng));
    return a;
}

/* Runs both rows at (M, sk, mb, chain) and compares; the f8 row's output stays in y1. Returns
 * differing outputs, or -1 when a launch was refused. */
long long compare(const Lib& lib, const RadKernelInfo* ref, Dev& a, Dev& w16, Dev& w8, Dev& y0, Dev& y1,
                  int64_t M, int64_t N, int64_t K, int64_t lda, int64_t ldc, int sk, int mb, int chain,
                  hipStream_t s, long long* zero_out = nullptr) {
    const size_t yb = (size_t)(M * ldc) * 2;
    HC(hipMemsetAsync(y0.p, 0xA5, yb, s));
    HC(hipMemsetAsync(y1.p, 0x5A, yb, s));
    const int r0 = gemm(ref, a.p, w16.p, RAD_BF16, y0.p, M, N, K, lda, ldc, sk, mb, chain, s);
    const int r1 = gemm(lib.f8, a.p, w8.p, R4D_DT_BF16F8, y1.p, M, N, K, lda, ldc, sk, mb, chain, s);
    HC(hipStreamSynchronize(s));
    ++g_runs;
    if (r0 != RAD_OK || r1 != RAD_OK) return -1;
    std::vector<uint16_t> h0((size_t)(M * ldc)), h1((size_t)(M * ldc));
    y0.get(h0.data(), yb);
    y1.get(h1.data(), yb);
    long long d = 0;
    for (int64_t m = 0; m < M; ++m)
        for (int64_t n = 0; n < N; ++n) {
            const uint16_t g = h1[(size_t)(m * ldc + n)];
            d += h0[(size_t)(m * ldc + n)] != g;
            if (zero_out) *zero_out += (g & 0x7FFF) == 0;
        }
    /* The pitch's padding is the rows' to leave alone. */
    for (int64_t m = 0; m < M; ++m)
        for (int64_t n = N; n < ldc; ++n) d += h1[(size_t)(m * ldc + n)] != 0x5A5A;
    return d;
}

void report(const char* what, long long diff) {
    if (diff == 0) return;
    ++g_fail;
    if (diff < 0) std::printf("  FAIL %s: a launch was refused\n", what);
    else std::printf("  FAIL %s: %lld outputs differ\n", what, diff);
}

std::vector<int> legal_sks(int64_t K) {
    std::vector<int> v;
    for (int sk : { 1, 2, 4, 5, 8, 10, 16, 20 })
        if (K % (sk * 16) == 0) v.push_back(sk);
    return v;
}

/* ------------------------------------------------------------------ frag */
constexpr int kExps = R4D_BF16F8_SMAX - R4D_BF16F8_SMIN + 1;   /* 237 */

/* One wave an exponent. Lane l holds bytes 8l..8l+3 in d0 and 8l+4..8l+7 in d1, a k step's slots
 * 0..7 as the kernel's packet gives them, so the 32 lanes cover all 256 byte values. */
__global__ __launch_bounds__(32) void probe_frag(uint32_t* frag, float* prod) {
    const unsigned s = R4D_BF16F8_SMIN + blockIdx.x, l = threadIdx.x;
    const unsigned d0 = (8u * l) | ((8u * l + 1u) << 8) | ((8u * l + 2u) << 16) | ((8u * l + 3u) << 24);
    const unsigned d1 = d0 + 0x04040404u;
    const float f = r4d_bf16f8_pow2(s);
    const r4d_bf8_u4 r = __builtin_bit_cast(r4d_bf8_u4, r4d_bf16f8_frag(d0, d1, f));
    uint32_t* o = frag + ((size_t)blockIdx.x * 32 + l) * 4;
    o[0] = r.x;
    o[1] = r.y;
    o[2] = r.z;
    o[3] = r.w;
    float* p = prod + ((size_t)blockIdx.x * 32 + l) * 8;
    r4d_bf8_f2 q = r4d_bf16f8_products<false>(d0, f);
    p[0] = q.x; p[1] = q.y;
    q = r4d_bf16f8_products<true>(d0, f);
    p[2] = q.x; p[3] = q.y;
    q = r4d_bf16f8_products<false>(d1, f);
    p[4] = q.x; p[5] = q.y;
    q = r4d_bf16f8_products<true>(d1, f);
    p[6] = q.x; p[7] = q.y;
}

int frag() {
    const size_t nw = (size_t)kExps * 32;
    Dev df(nw * 4 * 4), dp(nw * 8 * 4);
    HC(hipMemset(df.p, 0xA5, df.n));
    HC(hipMemset(dp.p, 0xA5, dp.n));
    hipLaunchKernelGGL(probe_frag, dim3(kExps), dim3(32), 0, 0, (uint32_t*)df.p, (float*)dp.p);
    HC(hipGetLastError());
    HC(hipDeviceSynchronize());
    std::vector<uint32_t> hf(nw * 4);
    std::vector<float> hp(nw * 8);
    df.get(hf.data(), df.n);
    dp.get(hp.data(), dp.n);
    long long pairs = 0, bad_bits = 0, bad_low = 0, negz = 0, posz = 0, sub = 0;
    std::string first;
    for (int si = 0; si < kExps; ++si)
        for (unsigned l = 0; l < 32; ++l)
            for (unsigned e = 0; e < 8; ++e) {
                const unsigned code = 8 * l + e, s = (unsigned)(R4D_BF16F8_SMIN + si);
                if ((code & 0x7F) == 0x7F) continue;   /* NaN: never stored */
                const size_t wv = (size_t)si * 32 + l;
                const uint16_t got = (uint16_t)(hf[wv * 4 + e / 2] >> (16 * (e % 2)));
                uint32_t pu;
                std::memcpy(&pu, &hp[wv * 8 + e], 4);
                const uint16_t want = r4d_bf16f8_expand(code, s);
                ++pairs;
                negz += got == 0x8000;
                posz += got == 0x0000;
                sub += (code & 0x78) == 0 && (code & 0x07) != 0;
                const bool ok_bits = got == want && (uint16_t)(pu >> 16) == want;
                const bool ok_low = (pu & 0xFFFFu) == 0;
                bad_bits += !ok_bits;
                bad_low += !ok_low;
                if ((!ok_bits || !ok_low) && first.empty()) {
                    char b[160];
                    std::snprintf(b, sizeof b, "code 0x%02x at 2^(%u-127): register 0x%04x, product 0x%08x, want 0x%04x",
                                  code, s, got, pu, want);
                    first = b;
                }
            }
    if (bad_bits || bad_low || pairs != 254LL * kExps || negz != kExps || posz != kExps) {
        ++g_fail;
        std::printf("  FAIL frag: %lld of %lld (code, exponent) pairs expand to other bits, %lld products with a "
                    "nonzero low half, %lld -0 and %lld +0 (want %d each)%s%s\n",
                    bad_bits, pairs, bad_low, negz, posz, kExps, first.empty() ? "" : "; first: ", first.c_str());
        return 1;
    }
    std::printf("PASS frag: the kernel's expansion of all %lld (code, exponent) pairs is r4d_bf16f8_expand's, bit for "
                "bit, every product exact (low half zero); %lld -0, %lld +0, %lld subnormal codes among them\n",
                pairs, negz, posz, sub);
    return 0;
}

/* ------------------------------------------------------------------ onehot */
int onehot(const Lib& lib) {
    const int64_t N = 256, K = 1024, M = 16;
    std::vector<uint32_t> codes;
    for (uint32_t b = 0; b < 256; ++b)
        if ((b & 0x7F) != 0x7F) codes.push_back(b);
    std::vector<uint16_t> w((size_t)(N * K));
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k) {
            const int64_t blk = (n / 32) * (K / 32) + k / 32, i = (n % 32) * 32 + k % 32;
            const uint32_t s = (uint32_t)(R4D_BF16F8_SMIN + blk % kExps);
            w[(size_t)(n * K + k)] = r4d_bf16f8_expand(codes[(size_t)((i + blk) % 254)], s);
        }
    const auto blob = relay(lib, w.data(), N, K);
    if (blob.empty()) { ++g_fail; std::printf("  FAIL onehot: the relayout refused an exact weight\n"); return 1; }
    hipStream_t s;
    HC(hipStreamCreate(&s));
    Dev a((size_t)(M * K) * 2), w16(w.size() * 2), w8(blob.size()), y0((size_t)(M * N) * 2), y1((size_t)(M * N) * 2);
    w16.put(w.data(), w.size() * 2);
    w8.put(blob.data(), blob.size());
    long long bad = 0, wrong = 0, pairs = 0;
    for (int sk : legal_sks(K))
        for (int64_t q = 0; q < K / 16; ++q) {
            std::vector<uint16_t> ha((size_t)(M * K), 0);
            for (int64_t m = 0; m < M; ++m) ha[(size_t)(m * K + 16 * q + m)] = 0x3F80;   /* 1.0 */
            a.put(ha.data(), ha.size() * 2);
            const long long d = compare(lib, lib.m64, a, w16, w8, y0, y1, M, N, K, K, N, sk, 1, 0, s);
            if (d) { bad += d < 0 ? 1 : d; continue; }
            std::vector<uint16_t> got((size_t)(M * N));
            y1.get(got.data(), got.size() * 2);
            for (int64_t m = 0; m < M; ++m)
                for (int64_t n = 0; n < N; ++n) {
                    const uint16_t want = w[(size_t)(n * K + 16 * q + m)], g = got[(size_t)(m * N + n)];
                    wrong += !(g == want || ((g | want) & 0x7FFF) == 0);
                    ++pairs;
                }
        }
    HC(hipStreamDestroy(s));
    if (bad || wrong) {
        ++g_fail;
        std::printf("  FAIL onehot: %lld outputs differ between the rows, %lld from the weight element\n", bad, wrong);
        return 1;
    }
    std::printf("PASS onehot: %lld decoded elements (all 254 codes x 237 exponents) at splits", pairs);
    for (int sk : legal_sks(K)) std::printf(" %d", sk);
    std::printf(", equal to gemm_bf16_nt_m64's and to the weight\n");
    return 0;
}

/* ------------------------------------------------------------------ zeros */
int zeros(const Lib& lib) {
    struct Z { int64_t N, K; };
    /* the peel (K 96 at SK 2, K 160 at SK 10: slices from mid-block), byte-wise exponents, half-block
     * tails, and whole-dword rounds at KU 16 and 8 */
    const Z zs[] = { { 16, 96 }, { 48, 96 }, { 64, 160 }, { 32, 480 }, { 128, 512 }, { 80, 2560 } };
    std::mt19937_64 rng(2);
    hipStream_t s;
    HC(hipStreamCreate(&s));
    const int fails0 = g_fail;
    long long runs = 0, zero_out = 0;
    for (const Z& z : zs) {
        const int64_t N = z.N, K = z.K, Mmax = 64;
        const auto w = exact_weight(N, K, rng, 100, 20, true);
        const auto blob = relay(lib, w.data(), N, K);
        if (blob.empty()) { ++g_fail; std::printf("  FAIL zeros [%lld, %lld]: the relayout refused an exact weight\n", (long long)N, (long long)K); continue; }
        std::vector<uint16_t> ha((size_t)(Mmax * K));
        for (size_t i = 0; i < ha.size(); ++i) {
            ha[i] = to_bf16((float)((int)(rng() % 2001) - 1000) / 256.0f);
            if (rng() % 4 == 0) ha[i] = (uint16_t)((rng() & 1) << 15);
            if ((i % (size_t)K) % 2) ha[i] = ha[i - 1];   /* equal pairs: products cancel */
        }
        Dev a(ha.size() * 2), w16(w.size() * 2), w8(blob.size()), y0((size_t)(Mmax * N) * 2), y1((size_t)(Mmax * N) * 2);
        a.put(ha.data(), ha.size() * 2);
        w16.put(w.data(), w.size() * 2);
        w8.put(blob.data(), blob.size());
        for (int64_t M : { 1, 5, 16, 33, 64 })
            for (int sk : legal_sks(K))
                for (int mb : { 0, 1, 2, 4 })
                    for (int chain : { 0, 1 }) {
                        if (chain && mb) continue;
                        char what[160];
                        std::snprintf(what, sizeof what, "zeros [%lld, %lld] M %lld sk %d mb %d chain %d", (long long)N,
                                      (long long)K, (long long)M, sk, mb, chain);
                        report(what, compare(lib, lib.m64, a, w16, w8, y0, y1, M, N, K, K, N, sk, mb, chain, s, &zero_out));
                        ++runs;
                    }
    }
    HC(hipStreamDestroy(s));
    if (!zero_out) { ++g_fail; std::printf("  FAIL zeros: no output cancelled to zero; the part tests nothing\n"); }
    if (g_fail == fails0)
        std::printf("PASS zeros: %lld launch pairs over signed zeros and cancelling pairs, %lld zero outputs, every bit equal\n",
                    runs, zero_out);
    return g_fail != fails0;
}

/* ------------------------------------------------------------------ shapes */
struct Shape {
    const char* what;
    int64_t N, K, lda_pad, ldc_pad;
};

int shapes(const Lib& lib) {
    /* ds41's per-rank shapes at TP4 (replicated ones whole), then odd ones. */
    const Shape sh[] = {
        { "attn_q_a", 1280, 5120, 0, 0 },          { "attn_q_b (a rank)", 8192, 1280, 0, 0 },
        { "attn_kv", 512, 5120, 0, 0 },            { "attn_output_a.gK (pitched)", 1024, 4096, 4096, 1024 },
        { "attn_output_b (a rank)", 5120, 2048, 0, 0 }, { "ffn_gate_up_shexp (rank 0)", 1536, 5120, 0, 0 },
        { "ffn_gate_up_shexp (ranks 1-3)", 1024, 5120, 0, 0 }, { "ffn_down_shexp (rank 0)", 5120, 768, 0, 0 },
        { "ffn_down_shexp (ranks 1-3)", 5120, 512, 0, 0 }, { "indexer.attn_q_b", 4096, 1280, 0, 0 },
        { "engram_kv", 25600, 6144, 0, 0 },        { "engram_kv (a rank, ENGRAM_SPLIT)", 6400, 6144, 0, 0 },
        { "odd: the peel at SK 2", 16, 96, 0, 0 }, { "odd: byte-wise exponents at SK 10", 64, 160, 0, 0 },
        { "odd: N 48", 48, 1280, 0, 0 },
    };
    const int64_t Ms[] = { 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 48, 63, 64, 65, 100, 256 };
    std::mt19937_64 rng(4141);
    hipStream_t s;
    HC(hipStreamCreate(&s));
    int fails0 = g_fail;
    for (const Shape& x : sh) {
        const int64_t N = x.N, K = x.K, lda = K + x.lda_pad, ldc = N + x.ldc_pad;
        const auto w = exact_weight(N, K, rng);
        const auto blob = relay(lib, w.data(), N, K);
        if (blob.empty()) { ++g_fail; std::printf("  FAIL %s: the relayout refused an exact weight\n", x.what); continue; }
        Dev w16(w.size() * 2), w8(blob.size());
        w16.put(w.data(), w.size() * 2);
        w8.put(blob.data(), blob.size());
        const int64_t Mmax = 256;
        const auto ha = random_act(Mmax, lda, rng);
        Dev a(ha.size() * 2), y0((size_t)(Mmax * ldc) * 2), y1((size_t)(Mmax * ldc) * 2);
        a.put(ha.data(), ha.size() * 2);
        long long runs = 0;
        for (int64_t M : Ms) {
            /* the law at every height, chained and not; every split and row-tile count at 1, 16, 64 */
            const bool sweep = M == 1 || M == 16 || M == 64;
            std::vector<int> sks = { 0 };
            if (sweep) for (int k : legal_sks(K)) sks.push_back(k);
            for (int sk : sks)
                for (int mb : sweep ? std::vector<int>{ 0, 1, 2, 4 } : std::vector<int>{ 0 })
                    for (int chain : { 0, 1 }) {
                        if (chain && (sk || mb)) continue;
                        char what[200];
                        std::snprintf(what, sizeof what, "%s [%lld, %lld] M %lld sk %d mb %d chain %d", x.what,
                                      (long long)N, (long long)K, (long long)M, sk, mb, chain);
                        report(what, compare(lib, lib.m64, a, w16, w8, y0, y1, M, N, K, lda, ldc, sk, mb, chain, s));
                        ++runs;
                    }
            /* Past 64 rows gemm_nt resolves to the tiled row, byte-identical to m64 by contract. */
            if (M > 64 && lib.tiled && N >= 128 && K % 64 == 0) {
                char what[200];
                std::snprintf(what, sizeof what, "%s [%lld, %lld] M %lld against gemm_bf16_nt_tiled", x.what,
                              (long long)N, (long long)K, (long long)M);
                report(what, compare(lib, lib.tiled, a, w16, w8, y0, y1, M, N, K, lda, ldc, 0, 0, 0, s));
                ++runs;
            }
        }
        std::printf("  %-36s [%5lld, %5lld]%s: %lld comparisons\n", x.what, (long long)N, (long long)K,
                    x.lda_pad ? " pitched" : "", runs);
        std::fflush(stdout);
    }
    HC(hipStreamDestroy(s));
    if (g_fail == fails0) std::printf("PASS shapes: every output byte equal\n");
    return g_fail != fails0;
}

/* ------------------------------------------------------------------ real weights */
struct RealEntry { std::string name; int64_t rows = 0, cols = 0; uint64_t off = 0; };

std::vector<RealEntry> read_dir(const char* path, int fd) {
    RadFileHeader H{};
    if (::pread(fd, &H, sizeof H, 0) != (ssize_t)sizeof H || H.magic != RAD_MAGIC) {
        std::fprintf(stderr, "%s: not a .rad\n", path);
        std::exit(2);
    }
    std::vector<char> strs((size_t)H.str_bytes + 1, 0);
    std::vector<RadFileEntry> dir((size_t)H.dir_count);
    std::vector<RadFilePlane> planes((size_t)H.plane_count);
    if (::pread(fd, strs.data(), (size_t)H.str_bytes, (off_t)H.str_off) != (ssize_t)H.str_bytes ||
        ::pread(fd, dir.data(), dir.size() * sizeof(RadFileEntry), (off_t)H.dir_off) != (ssize_t)(dir.size() * sizeof(RadFileEntry)) ||
        ::pread(fd, planes.data(), planes.size() * sizeof(RadFilePlane), (off_t)H.plane_off) != (ssize_t)(planes.size() * sizeof(RadFilePlane))) {
        std::fprintf(stderr, "%s: cannot read the directory\n", path);
        std::exit(2);
    }
    std::vector<RealEntry> out;
    for (const auto& d : dir) {
        RealEntry e;
        e.name = d.name ? std::string(&strs[(size_t)d.name]) : "";
        if (d.rank != 2 || d.n_planes != 1 || d.plane_first >= planes.size()) continue;
        e.rows = d.shape[0];
        e.cols = d.shape[1];
        e.off = planes[(size_t)d.plane_first].offset;
        if (planes[(size_t)d.plane_first].bytes != (uint64_t)(e.rows * e.cols * 2)) continue;
        out.push_back(e);
    }
    return out;
}

int real(const Lib& lib, const char* model) {
    const int fd = ::open(model, O_RDONLY | O_CLOEXEC);
    if (fd < 0) { std::perror(model); return 2; }
    const auto dir = read_dir(model, fd);
    auto find = [&](const std::string& n) -> const RealEntry* {
        for (const auto& e : dir) if (e.name == n) return &e;
        return nullptr;
    };
    /* A class's tensor, as rank 0 and rank 3 hold it at TP4: rows [r0, r1) (then [r2, r3), a
     * second row part: gate|up's up half) x cols [c0, c1). */
    struct Slice { const char* name; int64_t r0, r1, c0, c1, r2 = 0, r3 = 0; };
    const Slice sl[] = {
        { "blk.3.attn_q_a.weight", 0, 1280, 0, 5120 },
        { "blk.3.attn_q_b.weight", 0, 8192, 0, 1280 },        { "blk.3.attn_q_b.weight", 24576, 32768, 0, 1280 },
        { "blk.3.attn_kv.weight", 0, 512, 0, 5120 },
        { "blk.3.attn_output_a.g0.weight", 0, 1024, 0, 4096 }, { "blk.3.attn_output_a.g7.weight", 0, 1024, 0, 4096 },
        { "blk.3.attn_output_b.weight", 0, 5120, 0, 2048 },   { "blk.3.attn_output_b.weight", 0, 5120, 6144, 8192 },
        { "blk.3.ffn_gate_up_shexp.weight", 0, 768, 0, 5120, 2304, 3072 },
        { "blk.3.ffn_gate_up_shexp.weight", 1792, 2304, 0, 5120, 4096, 4608 },
        { "blk.3.ffn_down_shexp.weight", 0, 5120, 0, 768 },   { "blk.3.ffn_down_shexp.weight", 0, 5120, 1792, 2304 },
        { "blk.2.indexer.attn_q_b.weight", 0, 4096, 0, 1280 },
        { "blk.14.engram_kv.weight", 0, 6400, 0, 6144 },      { "blk.14.engram_kv.weight", 19200, 25600, 0, 6144 },
    };
    std::mt19937_64 rng(77);
    hipStream_t s;
    HC(hipStreamCreate(&s));
    const int fails0 = g_fail;
    int done = 0;
    for (const Slice& x : sl) {
        const RealEntry* e = find(x.name);
        if (!e) { std::printf("  skip %s: not in the container\n", x.name); continue; }
        const int64_t n1 = x.r1 - x.r0, N = n1 + (x.r3 - x.r2), K = x.c1 - x.c0;
        std::vector<uint16_t> w((size_t)(N * K)), row((size_t)e->cols);
        for (int64_t r = 0; r < N; ++r) {
            const int64_t src = r < n1 ? x.r0 + r : x.r2 + (r - n1);
            const uint64_t off = e->off + (uint64_t)(src * e->cols * 2);
            if (::pread(fd, row.data(), row.size() * 2, (off_t)off) != (ssize_t)(row.size() * 2)) { std::perror("pread"); return 2; }
            std::memcpy(&w[(size_t)(r * K)], &row[(size_t)x.c0], (size_t)K * 2);
        }
        const auto blob = relay(lib, w.data(), N, K);
        char what[200];
        if (blob.empty()) {
            ++g_fail;
            std::printf("  FAIL %s rows [%lld, %lld) cols [%lld, %lld): the relayout refused it\n", x.name,
                        (long long)x.r0, (long long)x.r1, (long long)x.c0, (long long)x.c1);
            continue;
        }
        Dev w16(w.size() * 2), w8(blob.size());
        w16.put(w.data(), w.size() * 2);
        w8.put(blob.data(), blob.size());
        const auto ha = random_act(16, K, rng);
        Dev a(ha.size() * 2), y0((size_t)(16 * N) * 2), y1((size_t)(16 * N) * 2);
        a.put(ha.data(), ha.size() * 2);
        for (int64_t M : { 1, 16 }) {
            std::snprintf(what, sizeof what, "%s rows [%lld, %lld) cols [%lld, %lld) M %lld", x.name,
                          (long long)x.r0, (long long)x.r1, (long long)x.c0, (long long)x.c1, (long long)M);
            report(what, compare(lib, lib.m64, a, w16, w8, y0, y1, M, N, K, K, N, 0, 0, 0, s));
        }
        ++done;
    }
    HC(hipStreamDestroy(s));
    ::close(fd);
    if (g_fail == fails0) std::printf("PASS real: %d container slices, every output byte equal\n", done);
    return g_fail != fails0;
}

/* ------------------------------------------------------------------ time */
int timing(const Lib& lib) {
    const Shape sh[] = {
        { "attn_q_a", 1280, 5120, 0, 0 }, { "attn_q_b", 8192, 1280, 0, 0 }, { "attn_kv", 512, 5120, 0, 0 },
        { "attn_output_a.gK", 1024, 4096, 0, 0 }, { "attn_output_b", 5120, 2048, 0, 0 },
        { "gate_up_shexp r0", 1536, 5120, 0, 0 }, { "down_shexp r0", 5120, 768, 0, 0 },
        { "indexer.attn_q_b", 4096, 1280, 0, 0 }, { "engram_kv split", 6400, 6144, 0, 0 },
        { "engram_kv", 25600, 6144, 0, 0 },
    };
    std::mt19937_64 rng(9);
    hipStream_t s;
    HC(hipStreamCreate(&s));
    hipEvent_t e0, e1;
    HC(hipEventCreate(&e0));
    HC(hipEventCreate(&e1));
    std::printf("  %-20s %10s %10s %10s %10s %8s\n", "shape", "bf16 us", "GB/s", "f8 us", "GB/s", "speedup");
    for (const Shape& x : sh) {
        const int64_t N = x.N, K = x.K;
        const auto w = exact_weight(N, K, rng);
        const auto blob = relay(lib, w.data(), N, K);
        if (blob.empty()) { ++g_fail; continue; }
        /* Copies enough that the rotation reads well past the 64 MB on-die cache. */
        const size_t bytes16 = w.size() * 2;
        const int copies = (int)std::max<size_t>(2, std::min<size_t>(8, (size_t)(256u << 20) / bytes16 + 1));
        std::vector<Dev*> c16, c8;
        for (int i = 0; i < copies; ++i) {
            c16.push_back(new Dev(bytes16));
            c16.back()->put(w.data(), bytes16);
            c8.push_back(new Dev(blob.size()));
            c8.back()->put(blob.data(), blob.size());
        }
        const auto ha = random_act(1, K, rng);
        Dev a(ha.size() * 2), y((size_t)N * 2);
        a.put(ha.data(), ha.size() * 2);
        auto run = [&](bool f8) {
            const int reps = 200;
            for (int i = 0; i < 10; ++i)
                (void)gemm(f8 ? lib.f8 : lib.m64, a.p, (f8 ? c8 : c16)[(size_t)(i % copies)]->p,
                           f8 ? R4D_DT_BF16F8 : RAD_BF16, y.p, 1, N, K, K, N, 0, 0, 0, s);
            HC(hipEventRecord(e0, s));
            for (int i = 0; i < reps; ++i)
                (void)gemm(f8 ? lib.f8 : lib.m64, a.p, (f8 ? c8 : c16)[(size_t)(i % copies)]->p,
                           f8 ? R4D_DT_BF16F8 : RAD_BF16, y.p, 1, N, K, K, N, 0, 0, 0, s);
            HC(hipEventRecord(e1, s));
            HC(hipEventSynchronize(e1));
            float ms = 0;
            HC(hipEventElapsedTime(&ms, e0, e1));
            return 1000.0 * ms / reps;
        };
        const double t16 = run(false), t8 = run(true);
        std::printf("  %-20s %10.2f %10.1f %10.2f %10.1f %7.2fx\n", x.what, t16, (double)bytes16 / t16 / 1e3, t8,
                    (double)blob.size() / t8 / 1e3, t16 / t8);
        for (auto* p : c16) delete p;
        for (auto* p : c8) delete p;
    }
    HC(hipStreamDestroy(s));
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s LIBR4D.so all [MODEL.rad] | frag | onehot | zeros | shapes | real MODEL.rad | time\n",
                     argv[0]);
        return 2;
    }
    const std::string what = argv[2];
    const char* model = argc > 3 ? argv[3] : nullptr;
    static const char* const parts[] = { "all", "frag", "onehot", "zeros", "shapes", "real", "time" };
    if (std::none_of(std::begin(parts), std::end(parts), [&](const char* p) { return what == p; })) {
        std::fprintf(stderr, "unknown part '%s'\n", what.c_str());
        return 2;
    }
    if (what == "real" && !model) { std::fprintf(stderr, "real needs MODEL.rad\n"); return 2; }
    const Lib lib(argv[1]);
    const auto t0 = std::chrono::steady_clock::now();
    int rc = 0;
    if (what == "frag" || what == "all") rc |= frag();
    if (what == "onehot" || what == "all") rc |= onehot(lib);
    if (what == "zeros" || what == "all") rc |= zeros(lib);
    if (what == "shapes" || what == "all") rc |= shapes(lib);
    if ((what == "real" || what == "all") && model) rc |= real(lib, model);
    if (what == "time" || what == "all") rc |= timing(lib);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (g_fail || rc)
        std::printf("FAIL: %d of the checks differ (%d launch pairs, %.1f s)\n", g_fail, g_runs, secs);
    else
        std::printf("PASS: gemm_bf16f8_nt_m64 is byte-identical to gemm_bf16_nt_m64 in %d launch pairs (%.1f s)\n",
                    g_runs, secs);
    return (g_fail || rc) ? 1 : 0;
}
