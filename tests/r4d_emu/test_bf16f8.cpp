/* tests/r4d_emu: gemm_bf16f8_nt_m64 (r4d_gemm_bf16f8_nt_m64.hip) against gemm_bf16_nt_m64
 * (r4d_gemm_bf16_nt_m64.hip) on the host SIMT emulator.
 *
 * Both kernels' own sources run: the bf16 one over a bf16 plane, the bf16f8 one over the same
 * values stored as E4M3 codes and an E8M0 exponent a 32x32 block (r4d_bf16f8_layout.h's closed
 * form; tests/bf16f8_layout_test.cpp holds the relayout hook to it). Two comparisons a launch pair:
 *   - the operands of every WMMA, through the emulator's tap: each thread's sequence of A, B and C
 *     bits must be the bf16 kernel's, call for call. That is the kernel's claim itself (the same
 *     operands in the same order from the same zero), and unlike the outputs it sees a -0 in B,
 *     which vanishes into a sum;
 *   - the output bytes, and the pitch's padding left alone.
 * Covered: the peel of a slice that starts mid-block, the KU 16 and 8 rounds with whole-dword and
 * byte-wise exponents, the half-block tail, every row-tile count (MB 1..4), the chained form,
 * pitched activation and output rows (wo_a's column views), and weights of signed zeros and
 * cancelling pairs. Then the entry's refusals: a weight that is not [N, K] (its exponents would be
 * read at the wrong offsets), another dtype; and a bf16 plane, which it hands to the bf16 kernel.
 *
 * The fp8 decode is emulated exactly (v_cvt_pk_f32_fp8, emu.cpp); that the card's is exact too is
 * the device check's (tools/ds41-host-experts/ds41_bf16f8_device_check.hip, which dumps these
 * registers for every code and exponent). A product within rounding of an f64 reference shows the
 * comparison is not between two wrong answers. */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"
#include "r4d_args.h"
#include "r4d_bf16f8_layout.h"
#include "hip/hip_runtime.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <utility>
#include <vector>

extern "C" int r4d_rad_gemm_bf16_m64(const RadArgs*, RadStream);
extern "C" int r4d_rad_gemm_bf16f8_m64(const RadArgs*, RadStream);

namespace {

int g_fail = 0, g_cases = 0;
long long g_calls = 0, g_negzero_b = 0;

void fail(const std::string& what) {
    ++g_fail;
    std::fprintf(stderr, "  FAIL %s\n", what.c_str());
}

/* ------------------------------------------------------------------ the operand trace */

/* Per (block, thread): how many WMMAs it issued and an FNV-1a hash of their operands' bits in
 * order. Both kernels launch the same grid and block, so the keys are the same threads. */
struct Stream {
    long long calls = 0;
    uint64_t h = 0xcbf29ce484222325ull;
    void eat(uint32_t v) {
        for (int i = 0; i < 4; ++i) h = (h ^ ((v >> (8 * i)) & 0xffu)) * 0x100000001b3ull;
    }
};
using Key = std::pair<uint64_t, unsigned>;
std::map<Key, Stream>* g_trace = nullptr;
long long g_trace_negzero = 0;

void tap(const EmuIdx& blk, unsigned thread, const unsigned short* a, const unsigned short* b,
         const float* c) {
    const uint64_t bid = ((uint64_t)blk.z << 40) | ((uint64_t)blk.y << 20) | blk.x;
    Stream& s = (*g_trace)[Key(bid, thread)];
    ++s.calls;
    for (int e = 0; e < 8; ++e) {
        uint32_t cu;
        std::memcpy(&cu, &c[e], 4);
        s.eat(a[e]);
        s.eat((uint32_t)b[e] << 16);
        s.eat(cu);
        g_trace_negzero += b[e] == 0x8000;
    }
}

/* Runs `f` with the tap on, into `out`; returns the -0 B elements it saw. */
template <class F>
long long traced(std::map<Key, Stream>* out, F&& f) {
    out->clear();
    g_trace = out;
    g_trace_negzero = 0;
    emu_wmma_bf16_tap = tap;
    f();
    emu_wmma_bf16_tap = nullptr;
    g_trace = nullptr;
    return g_trace_negzero;
}

/* The first thread whose operand stream differs, or "" when every one is the same. */
std::string first_difference(const std::map<Key, Stream>& x, const std::map<Key, Stream>& y) {
    if (x.size() != y.size())
        return std::to_string(x.size()) + " vs " + std::to_string(y.size()) + " threads issued WMMAs";
    for (auto i = x.begin(), j = y.begin(); i != x.end(); ++i, ++j) {
        if (i->first != j->first) return "different threads issued WMMAs";
        if (i->second.calls != j->second.calls || i->second.h != j->second.h) {
            char b[200];
            std::snprintf(b, sizeof b, "block %llx thread %u: %lld calls vs %lld, operands %s",
                          (unsigned long long)i->first.first, i->first.second, i->second.calls,
                          j->second.calls, i->second.h == j->second.h ? "equal" : "differ");
            return b;
        }
    }
    return "";
}

/* ------------------------------------------------------------------ data */

/* The stored form from the closed form: each block's derived exponent, each element's code at its
 * slot. */
std::vector<uint8_t> store(const std::vector<uint16_t>& W, int64_t N, int64_t K) {
    std::vector<uint8_t> b((size_t)r4d_bf16f8_bytes(N, K), 0);
    const int64_t soff = r4d_bf16f8_scale_off(N, K), ldws = r4d_bf16f8_ldws(K);
    for (int64_t nb = 0; nb < (N + 31) / 32; ++nb)
        for (int64_t kb = 0; kb < K / 32; ++kb) {
            const int rows = N - nb * 32 < 32 ? (int)(N - nb * 32) : 32;
            b[(size_t)(soff + nb * ldws + kb)] =
                (uint8_t)r4d_bf16f8_block_exp(&W[(size_t)(nb * 32 * K + kb * 32)], K, rows);
        }
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k) {
            const uint32_t s = b[(size_t)(soff + (n / 32) * ldws + k / 32)];
            const int c = r4d_bf16f8_encode(W[(size_t)(n * K + k)], s);
            if (c < 0) { std::fprintf(stderr, "store: (%lld, %lld) not exact\n", (long long)n, (long long)k); std::abort(); }
            b[(size_t)r4d_bf16f8_slot(K, n, k)] = (uint8_t)c;
        }
    return b;
}

/* An FP8-exact weight: an exponent a block around 2^-20, so products and sums stay well inside
 * bf16, and random codes with signed zeros and subnormals among them. `zeros`: even k a signed zero
 * more often than not, odd k the negation of its even neighbour (the pair shares a block), so with
 * equal activation pairs every product cancels its neighbour's. */
std::vector<uint16_t> weight(int64_t N, int64_t K, std::mt19937_64& rng, bool zeros) {
    const int64_t kb = K / 32;
    std::vector<uint32_t> s((size_t)(((N + 31) / 32) * kb));
    for (auto& v : s) v = 100 + (uint32_t)(rng() % 12);
    std::vector<uint16_t> w((size_t)(N * K));
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k) {
            const uint32_t sb = s[(size_t)((n / 32) * kb + k / 32)];
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
            x = r4d_bf16f8_expand(c, sb);
        }
    return w;
}

float b2f(uint16_t h) {
    const uint32_t u = (uint32_t)h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

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

struct Case {
    int64_t N, K, M;
    bool sweep;     /* every legal split and MB; else the law's alone (and MB 0), chained too */
    int64_t lda_pad, ldc_pad;
    bool zeros;     /* weight() `zeros`, and activations in equal pairs with signed zeros */
};

void run_case(const Case& c, std::mt19937_64& rng) {
    const int64_t N = c.N, K = c.K, M = c.M, lda = K + c.lda_pad, ldc = N + c.ldc_pad;
    const auto W = weight(N, K, rng, c.zeros);
    const auto blob = store(W, N, K);
    std::vector<uint16_t> A((size_t)(M * lda));
    for (size_t i = 0; i < A.size(); ++i) {
        A[i] = rad_f32_to_bf16((float)((int)(rng() % 2001) - 1000) / 256.0f);
        if (c.zeros && rng() % 4 == 0) A[i] = (uint16_t)((rng() & 1) << 15);
        if (c.zeros && (i % (size_t)lda) % 2) A[i] = A[i - 1];
    }
    /* Buffers 16-byte aligned, as gemm_ops requires. */
    std::vector<uint64_t> Wb((blob.size() + 7) / 8 + 2), Wp((W.size() * 2 + 7) / 8 + 2);
    std::memcpy(Wb.data(), blob.data(), blob.size());
    std::memcpy(Wp.data(), W.data(), W.size() * 2);
    std::vector<uint16_t> y16((size_t)(M * ldc)), y8((size_t)(M * ldc));

    std::vector<int> sks = { 0 };
    std::vector<int> mbs = { 0 };
    if (c.sweep) {
        for (int sk : { 1, 2, 4, 5, 8, 10, 16, 20 })
            if (K % (sk * 16) == 0) sks.push_back(sk);
        mbs = { 0, 1, 2, 3, 4 };
    }
    std::map<Key, Stream> t16, t8;
    for (int sk : sks)
        for (int mb : mbs)
            for (int chain : { 0, 1 }) {
                if (c.sweep && chain && (sk || mb)) continue;   /* the chained form at the law */
                std::vector<RadParam> ps = { RAD_INT("M", M), RAD_INT("N", N), RAD_INT("K", K),
                                             RAD_STR("dtype", "bf16"), RAD_INT("chain", chain) };
                if (sk) ps.push_back(RAD_INT("sk", sk));
                if (mb) ps.push_back(RAD_INT("mb", mb));
                RadTensor a16t[3] = { view(A.data(), RAD_BF16, M, K, lda),
                                      view(Wp.data(), RAD_BF16, N, K, K),
                                      view(y16.data(), RAD_BF16, M, N, ldc) };
                RadTensor a8t[3] = { a16t[0], view(Wb.data(), R4D_DT_BF16F8, N, K, K),
                                     view(y8.data(), RAD_BF16, M, N, ldc) };
                RadArgs a16{}, a8{};
                a16.t = a16t; a16.n_t = 3; a16.p = ps.data(); a16.n_p = (int)ps.size();
                a8.t = a8t; a8.n_t = 3; a8.p = ps.data(); a8.n_p = (int)ps.size();
                a16.world_size = a8.world_size = 1;
                std::fill(y16.begin(), y16.end(), (uint16_t)0xA5A5);
                std::fill(y8.begin(), y8.end(), (uint16_t)0x5A5A);
                int r16 = 0, r8 = 0;
                traced(&t16, [&] { r16 = r4d_rad_gemm_bf16_m64(&a16, nullptr); });
                const long long nz = traced(&t8, [&] { r8 = r4d_rad_gemm_bf16f8_m64(&a8, nullptr); });
                char at[200];
                std::snprintf(at, sizeof at, "N=%lld K=%lld M=%lld lda=%lld ldc=%lld sk=%d mb=%d chain=%d%s",
                              (long long)N, (long long)K, (long long)M, (long long)lda, (long long)ldc,
                              sk, mb, chain, c.zeros ? " zeros" : "");
                ++g_cases;
                if (r16 != RAD_OK || r8 != RAD_OK) {
                    fail(std::string(at) + ": launch refused (" + std::to_string(r16) + ", " +
                         std::to_string(r8) + ")");
                    continue;
                }
                const std::string op = first_difference(t16, t8);
                if (!op.empty()) fail(std::string(at) + ": WMMA operands differ: " + op);
                if (t8.empty()) fail(std::string(at) + ": no WMMA traced");
                for (const auto& kv : t8) g_calls += kv.second.calls;
                g_negzero_b += nz;
                long long diff = 0, far = 0, zero_out = 0;
                for (int64_t m = 0; m < M; ++m)
                    for (int64_t n = 0; n < N; ++n) {
                        const uint16_t g16 = y16[(size_t)(m * ldc + n)], g8 = y8[(size_t)(m * ldc + n)];
                        diff += g16 != g8;
                        zero_out += (g8 & 0x7FFF) == 0;
                        if (sk || mb || chain) continue;
                        double ref = 0, mag = 0;
                        for (int64_t k = 0; k < K; ++k) {
                            const double p = (double)b2f(A[(size_t)(m * lda + k)]) * (double)b2f(W[(size_t)(n * K + k)]);
                            ref += p;
                            mag += std::fabs(p);
                        }
                        far += std::fabs((double)b2f(g8) - ref) > 1e-2 * mag + 1e-30;
                    }
                /* The pitch's padding is the kernels' to leave alone. */
                long long pad = 0;
                for (int64_t m = 0; m < M; ++m)
                    for (int64_t n = N; n < ldc; ++n) pad += y8[(size_t)(m * ldc + n)] != 0x5A5A;
                if (diff) fail(std::string(at) + ": " + std::to_string(diff) + " outputs differ");
                if (far) fail(std::string(at) + ": " + std::to_string(far) + " outputs far from the f64 product");
                if (pad) fail(std::string(at) + ": " + std::to_string(pad) + " padding values written");
                if (c.zeros && !zero_out) fail(std::string(at) + ": a zeros case gave no zero output");
            }
}

/* The entry's refusals, and its hand-off of a bf16 plane. */
void refusals(std::mt19937_64& rng) {
    const int64_t N = 64, K = 256, M = 3;
    const auto W = weight(N, K, rng, false);
    const auto blob = store(W, N, K);
    std::vector<uint64_t> Wb((blob.size() + 7) / 8 + 2), Wp((W.size() * 2 + 7) / 8 + 2);
    std::memcpy(Wb.data(), blob.data(), blob.size());
    std::memcpy(Wp.data(), W.data(), W.size() * 2);
    std::vector<uint16_t> A((size_t)(M * K)), y((size_t)(M * N), 0x1234), y16((size_t)(M * N), 0x4321);
    for (auto& v : A) v = rad_f32_to_bf16((float)((int)(rng() % 201) - 100) / 64.0f);
    auto call = [&](RadTensor b, int64_t yn, std::vector<uint16_t>& out) {
        std::vector<RadParam> ps = { RAD_INT("M", M), RAD_INT("N", yn), RAD_INT("K", K), RAD_STR("dtype", "bf16") };
        RadTensor t[3] = { view(A.data(), RAD_BF16, M, K, K), b, view(out.data(), RAD_BF16, M, yn, N) };
        RadArgs a{};
        a.t = t; a.n_t = 3; a.p = ps.data(); a.n_p = (int)ps.size(); a.world_size = 1;
        return r4d_rad_gemm_bf16f8_m64(&a, nullptr);
    };
    ++g_cases;
    /* An output narrower than the weight: the bf16 kernel would compute its first columns, this
     * one would read the exponents of another row. */
    const std::vector<uint16_t> before = y;
    if (call(view(Wb.data(), R4D_DT_BF16F8, N, K, K), N - 16, y) != RAD_E_SHAPE)
        fail("a [M, N-16] output over an [N, K] weight was not refused");
    if (call(view(Wb.data(), R4D_DT_BF16F8, N + 16, K, K), N, y) != RAD_E_SHAPE)
        fail("an [N+16, K] weight under an [M, N] output was not refused");
    if (y != before) fail("a refused call wrote its output");
    if (call(view(Wb.data(), RAD_F16, N, K, K), N, y) != RAD_E_DTYPE)
        fail("an f16 weight was not refused");
    /* A bf16 plane (a tool calling the row cold): gemm_bf16_nt_m64's bytes. */
    std::vector<RadParam> ps = { RAD_INT("M", M), RAD_INT("N", N), RAD_INT("K", K), RAD_STR("dtype", "bf16") };
    RadTensor t16[3] = { view(A.data(), RAD_BF16, M, K, K), view(Wp.data(), RAD_BF16, N, K, K),
                         view(y16.data(), RAD_BF16, M, N, N) };
    RadArgs a16{};
    a16.t = t16; a16.n_t = 3; a16.p = ps.data(); a16.n_p = (int)ps.size(); a16.world_size = 1;
    if (r4d_rad_gemm_bf16_m64(&a16, nullptr) != RAD_OK) fail("gemm_bf16_nt_m64 refused the reference");
    if (call(view(Wp.data(), RAD_BF16, N, K, K), N, y) != RAD_OK) fail("a bf16 plane was refused");
    if (y != y16) fail("a bf16 plane did not give gemm_bf16_nt_m64's bytes");
    std::fill(y.begin(), y.end(), (uint16_t)0x1234);
    if (call(view(Wb.data(), R4D_DT_BF16F8, N, K, K), N, y) != RAD_OK) fail("the stored form was refused");
    if (y != y16) fail("the stored form did not give gemm_bf16_nt_m64's bytes");
}

}  // namespace

int main() {
    std::mt19937_64 rng(20261029);
    const Case cases[] = {
        /* the peel (K 96 at SK 2; K 160 at SK 10), byte-wise exponents, N % 32 == 16 */
        { 16, 96, 1, true, 0, 0, false },
        { 48, 96, 5, true, 0, 0, false },
        { 64, 160, 17, true, 0, 0, false },
        /* whole-dword exponents at KU 16 and 8, three and four row tiles */
        { 32, 512, 33, true, 0, 0, false },
        { 80, 768, 64, true, 0, 0, false },
        { 48, 1280, 3, true, 0, 0, false },
        /* signed zeros and cancelling pairs, through the peel and the rounds */
        { 16, 96, 16, true, 0, 0, true },
        { 32, 160, 7, true, 0, 0, true },
        { 64, 512, 20, false, 0, 0, true },
        /* ds41 shapes at the law: attn_kv, a narrow q_a, the shared down at rank 1's K, past 64 rows */
        { 512, 5120, 1, false, 0, 0, false },
        { 256, 5120, 16, false, 0, 0, false },
        { 128, 512, 16, false, 0, 0, false },
        { 96, 2048, 70, false, 0, 0, false },
        /* pitched rows, as wo_a's group columns: A and C views of wider planes */
        { 64, 1024, 9, false, 1024, 64, false },
        { 32, 4096, 16, false, 4096, 32, false },
    };
    for (const Case& c : cases) run_case(c, rng);
    refusals(rng);
    if (g_negzero_b == 0) fail("no -0 reached a WMMA's B operand: the zero cases test nothing");
    std::printf("gemm_bf16f8_nt_m64 vs gemm_bf16_nt_m64 on the emulator: %d launch pairs, %lld lane WMMAs "
                "compared operand for operand (%lld -0 B elements), %d failures\n",
                g_cases, g_calls, g_negzero_b, g_fail);
    return g_fail ? 1 : 0;
}
