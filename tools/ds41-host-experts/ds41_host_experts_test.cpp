/* ds41_host_experts_test.cpp -- exactness checks and the throughput gate for ds41_host_experts.
 *
 *   ds41_host_experts_test check            scalar vs AVX2, ggml blocks vs planes: bit-identical
 *   ds41_host_experts_test bench [experts]  experts/s and weight GB/s at 1..20 pinned threads
 *
 * The bench streams `experts` distinct DeepSeek-V4.1 experts (gate_up 4608 x 5120 and down
 * 5120 x 2304, 18.8 MB each) so the weights come from DRAM, as they would from the RAM cache. */
#include "ds41_host_experts.h"

#include <immintrin.h>
#include <pthread.h>
#include <sched.h>

#include <atomic>
#include <functional>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
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

/* Random codes; e8m0 around 2^-8 (realistic) or drawn from the edges (adversarial). */
Mat make_mat(int N, int K, uint64_t seed, bool adversarial, bool with_planes = true) {
    Mat m;
    m.N = N; m.K = K;
    const int UK = K / 32;
    m.blocks.resize((size_t)N * UK * 17);
    if (with_planes) { m.codes.resize((size_t)N * K / 2); m.scales.resize((size_t)N * UK); }
    std::mt19937_64 rng(seed);
    static const uint8_t edge[] = { 0, 1, 2, 3, 100, 126, 127, 128, 129, 200, 252, 253, 254, 255 };
    for (int n = 0; n < N; ++n)
        for (int u = 0; u < UK; ++u) {
            const uint64_t r = rng();
            const uint8_t e = adversarial && (r & 3) == 0 ? edge[(r >> 2) % sizeof edge]
                                                          : (uint8_t)(116 + (r >> 8) % 12);
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

std::vector<uint16_t> make_x(int K, uint64_t seed, bool adversarial) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.f, 0.5f);
    std::vector<uint16_t> x((size_t)K);
    static const uint16_t special[] = { 0x0000, 0x8000, 0x0001, 0x8001, 0x0080, 0x7f7f, 0xff7f, 0x3f80, 0x0040, 0x00ff };
    for (int k = 0; k < K; ++k) {
        x[(size_t)k] = f32_to_bf16(nd(rng));
        if (adversarial && rng() % 7 == 0) x[(size_t)k] = special[rng() % (sizeof special / 2)];
    }
    return x;
}

size_t diff(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
    size_t d = 0;
    for (size_t i = 0; i < a.size(); ++i) d += a[i] != b[i];
    return d;
}

int check() {
    int failures = 0;
    struct Case { const char* name; int N, K; bool adv; };
    const Case cases[] = { { "gate_up 4608x5120", 2 * kFf, kEmbd, false }, { "down 5120x2304", kEmbd, kFf, false },
                           { "gate_up edges 256x5120", 256, kEmbd, true }, { "down edges 256x2304", 256, kFf, true },
                           { "odd K 64x1056", 64, 1056, true } };
    for (const Case& c : cases) {
        const Mat m = make_mat(c.N, c.K, 1234u + (uint64_t)c.N, c.adv);
        const auto x = make_x(c.K, 99u + (uint64_t)c.K, c.adv);
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
    }
    /* A whole expert: gate_up, swiglu_clamped and down, scalar vs AVX2 at four threads. */
    const Mat gu = make_mat(2 * kFf, kEmbd, 7u, false), dn = make_mat(kEmbd, kFf, 8u, false);
    Expert e{ gu.ggml(), dn.ggml() };
    std::vector<float> silu(65536);
    silu_table_host(silu.data(), kLimit);
    const auto x = make_x(kEmbd, 5u, false);
    std::vector<uint16_t> g1(2 * kFf), a1(kFf), d1(kEmbd), g2(2 * kFf), a2(kFf), d2(kEmbd);
    expert_forward(e, silu.data(), kLimit, x.data(), g1.data(), a1.data(), d1.data(), 1, false);
    expert_forward(e, silu.data(), kLimit, x.data(), g2.data(), a2.data(), d2.data(), 4, true);
    const size_t de = diff(g1, g2) + diff(a1, a2) + diff(d1, d2);
    std::printf("%-26s scalar vs AVX2 x4 threads: %zu differing values\n", "expert forward", de);
    failures += de != 0;
    std::printf(failures ? "FAIL: %d case(s) differ\n" : "PASS: every case bit-identical\n", failures);
    return failures ? 1 : 0;
}

/* Physical cores round-robin over the four CCDs (0-5, 6-11, 12-17, 18-23 share an L3). */
int core_for(int i) { return (i % 4) * 6 + (i / 4) % 6; }

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

int bench(int experts) {
    std::printf("building %d distinct experts (%.1f GB)...\n", experts, experts * 18.80064e6 / 1e9);
    std::vector<Mat> gu, dn;
    for (int i = 0; i < experts; ++i) {
        gu.push_back(make_mat(2 * kFf, kEmbd, 100u + (uint64_t)i, false, false));
        dn.push_back(make_mat(kEmbd, kFf, 900u + (uint64_t)i, false, false));
    }
    std::vector<float> silu(65536);
    silu_table_host(silu.data(), kLimit);
    const auto x = make_x(kEmbd, 3u, false);
    std::vector<uint16_t> g(2 * kFf), a(kFf), d(kEmbd);
    const double bytes = (double)experts * (2.0 * kFf * kEmbd + (double)kEmbd * kFf) / 32 * 17;
    for (int t : { 1, 4, 8, 12, 16, 20 }) {
        Pool pool(t);
        auto pass = [&]() {
            for (int i = 0; i < experts; ++i) {
                const Mx G = gu[(size_t)i].ggml(), D = dn[(size_t)i].ggml();
                pool.run([&](int k) { const int q = G.N / 8; gemv_avx2(G, x.data(), g.data(), q * k / t * 8, q * (k + 1) / t * 8); });
                swiglu(silu.data(), g.data(), a.data(), kFf, kLimit);
                pool.run([&](int k) { const int q = D.N / 8; gemv_avx2(D, a.data(), d.data(), q * k / t * 8, q * (k + 1) / t * 8); });
            }
        };
        pass();   /* warm the code and the pages, not the caches: the set exceeds L3 */
        const auto t0 = std::chrono::steady_clock::now();
        pass();
        pass();
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("%2d threads: %7.3f ms per expert, %6.1f GB/s of weights\n", t, 1e3 * s / (2.0 * experts),
                    2.0 * bytes / s / 1e9);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && !std::strcmp(argv[1], "bench")) return bench(argc > 2 ? std::atoi(argv[2]) : 32);
    return check();
}
