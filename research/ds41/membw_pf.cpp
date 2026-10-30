// membw_pf: aggregate DRAM read bandwidth at 40 and 48 threads, each streaming its own buffer
// (first touch on that thread), plain or with software prefetch PF bytes ahead (T0 or NTA), and
// with S interleaved streams a thread. Whether the GEMV kernels' 94 GB/s is the box's ceiling.
//   membw_pf [MiB a thread]
#include <immintrin.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

template <int S>
static double stream(const char* p, size_t per, int pf, bool nta) {
    __m256i acc = _mm256_setzero_si256();
    const size_t part = per / S;
    for (size_t k = 0; k < part; k += 128)
        for (int s = 0; s < S; ++s) {
            const char* b = p + s * part + k;
            if (pf) {
                if (nta) _mm_prefetch(b + pf, _MM_HINT_NTA);
                else _mm_prefetch(b + pf, _MM_HINT_T0);
                if (nta) _mm_prefetch(b + pf + 64, _MM_HINT_NTA);
                else _mm_prefetch(b + pf + 64, _MM_HINT_T0);
            }
            const __m256i* q = (const __m256i*)b;
            acc = _mm256_add_epi64(acc, _mm256_add_epi64(_mm256_add_epi64(_mm256_load_si256(q), _mm256_load_si256(q + 1)),
                                                         _mm256_add_epi64(_mm256_load_si256(q + 2), _mm256_load_si256(q + 3))));
        }
    return (double)_mm256_extract_epi64(acc, 0);
}

static double run(int n, size_t per, int pf, bool nta, int S) {
    std::vector<std::thread> ts; std::atomic<int> ready{0}; std::atomic<bool> go{false};
    std::vector<double> sink(n);
    std::vector<char*> buf(n);
    for (int i = 0; i < n; ++i) ts.emplace_back([&, i] {
        char* p = (char*)aligned_alloc(4096, per);
        for (size_t k = 0; k < per; k += 4096) p[k] = 1;
        buf[i] = p; ready++;
        while (!go.load()) {}
        for (int rep = 0; rep < 2; ++rep)
            sink[i] += S == 1 ? stream<1>(p, per, pf, nta) : S == 2 ? stream<2>(p, per, pf, nta) : stream<4>(p, per, pf, nta);
    });
    while (ready.load() < n) {}
    const auto t0 = std::chrono::steady_clock::now(); go = true;
    for (auto& t : ts) t.join();
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    for (auto p : buf) free(p);
    return 2.0 * per * n / s / 1e9;
}

int main(int argc, char** argv) {
    const size_t per = (argc > 1 ? atol(argv[1]) : 64) << 20;
    struct V { const char* name; int pf; bool nta; int S; };
    const V vs[] = { { "plain", 0, false, 1 }, { "pf T0 +512", 512, false, 1 }, { "pf T0 +1024", 1024, false, 1 },
                     { "pf T0 +2048", 2048, false, 1 }, { "pf NTA +1024", 1024, true, 1 },
                     { "2 streams", 0, false, 2 }, { "4 streams", 0, false, 4 }, { "4 streams pf +1024", 1024, false, 4 } };
    for (int n : { 40, 48 })
        for (const V& v : vs) {
            double best = 0;
            for (int r = 0; r < 3; ++r) { const double g = run(n, per, v.pf, v.nta, v.S); if (g > best) best = g; }
            std::printf("%2d threads  %-20s %6.1f GB/s (best of 3)\n", n, v.name, best);
        }
}
