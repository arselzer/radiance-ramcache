// membw_dur: aggregate DRAM read rate of N threads, each streaming its own buffer, for S seconds,
// printed every 0.5 s with the epoch time, to see what the cards' host->card copies take from the
// CPU's streaming when both run (bwmix/run.sh runs it beside ds41_p0_bench copy).
//   membw_dur [threads=40] [seconds=60] [MiB a thread=64]
#include <immintrin.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
    const int n = argc > 1 ? atoi(argv[1]) : 40;
    const double secs = argc > 2 ? atof(argv[2]) : 60;
    const size_t per = (size_t)(argc > 3 ? atol(argv[3]) : 64) << 20, piece = 1 << 20;
    std::vector<std::atomic<uint64_t>> bytes((size_t)n);
    std::vector<uint64_t> sink((size_t)n);
    std::atomic<bool> stop{false};
    std::atomic<int> ready{0};
    std::vector<std::thread> ts;
    for (int i = 0; i < n; ++i) ts.emplace_back([&, i] {
        char* p = (char*)aligned_alloc(4096, per);
        for (size_t k = 0; k < per; k += 4096) p[k] = 1;
        ready++;
        __m256i acc = _mm256_setzero_si256();
        while (!stop.load(std::memory_order_relaxed))
            for (size_t k = 0; k < per && !stop.load(std::memory_order_relaxed); k += piece) {
                for (size_t j = k; j < k + piece; j += 128) {
                    const __m256i* q = (const __m256i*)(p + j);
                    acc = _mm256_add_epi64(acc, _mm256_add_epi64(_mm256_add_epi64(_mm256_load_si256(q), _mm256_load_si256(q + 1)),
                                                                 _mm256_add_epi64(_mm256_load_si256(q + 2), _mm256_load_si256(q + 3))));
                }
                bytes[(size_t)i].fetch_add(piece, std::memory_order_relaxed);
            }
        sink[(size_t)i] = (uint64_t)_mm256_extract_epi64(acc, 0);
        free(p);
    });
    while (ready.load() < n) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    const auto t0 = std::chrono::steady_clock::now();
    uint64_t last = 0;
    double lt = 0;
    for (double t = 0; t < secs;) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        uint64_t b = 0;
        for (auto& x : bytes) b += x.load(std::memory_order_relaxed);
        const double ep = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
        std::printf("%.1f %6.1f s %6.1f GB/s\n", ep, t, (double)(b - last) / (t - lt) / 1e9);
        std::fflush(stdout);
        last = b;
        lt = t;
    }
    stop = true;
    for (auto& t : ts) t.join();
    uint64_t s = 0;
    for (uint64_t v : sink) s += v;
    return s == 42 ? 1 : 0;
}
