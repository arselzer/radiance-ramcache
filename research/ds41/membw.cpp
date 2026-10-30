// Aggregate DRAM read bandwidth with N threads, each streaming its own buffer (first-touch).
// An upper bound for CPU expert GEMV, which streams weights once per token.
#include <immintrin.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>
int main(int argc, char** argv) {
    const size_t per = (argc > 1 ? atol(argv[1]) : 128) << 20;   // bytes per thread
    for (int n : {1, 4, 8, 12, 16, 20, 24}) {
        std::vector<std::thread> ts; std::atomic<int> ready{0}; std::atomic<bool> go{false};
        std::vector<double> sink(n);
        std::vector<char*> buf(n);
        auto t0 = std::chrono::steady_clock::now(); auto t1 = t0;
        for (int i = 0; i < n; ++i) ts.emplace_back([&, i] {
            char* p = (char*)aligned_alloc(4096, per);
            for (size_t k = 0; k < per; k += 4096) p[k] = 1;        // first touch on this thread
            buf[i] = p; ready++;
            while (!go.load()) {}
            __m256i acc = _mm256_setzero_si256();
            for (int rep = 0; rep < 2; ++rep)
                for (size_t k = 0; k < per; k += 128) {
                    const __m256i* q = (const __m256i*)(p + k);
                    acc = _mm256_add_epi64(acc, _mm256_add_epi64(_mm256_add_epi64(_mm256_load_si256(q), _mm256_load_si256(q + 1)),
                                                                 _mm256_add_epi64(_mm256_load_si256(q + 2), _mm256_load_si256(q + 3))));
                }
            sink[i] = (double)_mm256_extract_epi64(acc, 0);
        });
        while (ready.load() < n) {}
        t0 = std::chrono::steady_clock::now(); go = true;
        for (auto& t : ts) t.join();
        t1 = std::chrono::steady_clock::now();
        const double s = std::chrono::duration<double>(t1 - t0).count();
        printf("%2d threads: %6.1f GB/s\n", n, 2.0 * per * n / s / 1e9);
        for (auto p : buf) free(p);
    }
}
