/* host_expert_pool_futex_test.cpp -- the CPU pool's futex wake-up (core/place/host_expert_pool.h,
 * RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_FUTEX=1) and its pre-wake (..._PREWAKE): every row still
 * equals the reference with four submitters at once; no wake-up is lost when submits from four
 * threads land while workers go to sleep and wake (a 20 us idle spin, so they sleep between nearly
 * every job; a lost wake-up leaves a ticket unfinished, which the deadline catches); a pre-wake
 * wakes three of the four workers, which sleep again after their window, a second call within it
 * wakes nobody, and pre-woken workers take a following submit's tiles. Its own binary: the pool
 * reads its switches once a process. The silu table is the host's. */
#include "rad_test.h"

#include "place/host_expert_pool.h"
#include "../tools/ds41-host-experts/ds41_host_experts.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <thread>
#include <vector>

using rad::HostExpertPool;

namespace {

constexpr int kEmbd = HostExpertPool::kEmbd, kFf = HostExpertPool::kFf;

/* block_mxfp4 rows with scales near 2^-8 and random codes, from a seed. */
std::vector<uint8_t> blocks(int N, int K, uint64_t seed) {
    std::vector<uint8_t> b((size_t)N * (K / 32) * 17);
    uint64_t s = seed * 0x9E3779B97F4A7C15ull + 1;
    for (size_t i = 0; i < b.size(); ++i) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        b[i] = (uint8_t)(i % 17 == 0 ? 116 + (s >> 40) % 12 : s >> 32);
    }
    return b;
}

std::vector<uint16_t> row(uint64_t seed) {
    std::vector<uint16_t> x(kEmbd);
    uint64_t s = seed + 77;
    for (int k = 0; k < kEmbd; ++k) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        const float v = ((float)(s >> 40) / (float)(1u << 24) - 0.5f) * 2.0f;
        x[(size_t)k] = ds41hx::f32_to_bf16(v);
    }
    return x;
}

struct Expert {
    std::vector<uint8_t> gu, dn;
};

std::vector<float> g_silu;

/* The one-row AVX2 kernel's rows (gemv_avx2, proven equal to gemv_ref by ds41_host_experts_test). */
std::vector<uint16_t> reference(const Expert& e, const std::vector<uint16_t>& x) {
    ds41hx::Expert ex;
    ex.gu.N = 2 * kFf; ex.gu.K = kEmbd; ex.gu.blocks = e.gu.data();
    ex.dn.N = kEmbd;   ex.dn.K = kFf;   ex.dn.blocks = e.dn.data();
    std::vector<uint16_t> g(2 * kFf), a(kFf), d(kEmbd);
    ds41hx::expert_forward(ex, g_silu.data(), 10.0f, x.data(), g.data(), a.data(), d.data(), 1, true);
    return d;
}

/* Four workers on the futex, idle spin 20 us, a pre-wake of three with a 50 ms window. An outside
 * setting of the switches wins (the cases then skip what they cannot check). */
HostExpertPool* pool() {
    static HostExpertPool* p = [] {
        g_silu.resize(65536);
        ds41hx::silu_table_host(g_silu.data(), 10.0f);
        const std::string path = radtest::tmpdir() + "/silu_host_futex.bin";
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f || std::fwrite(g_silu.data(), 4, 65536, f) != 65536) return (HostExpertPool*)nullptr;
        std::fclose(f);
        setenv("RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS", "4", 1);
        setenv("RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_SILU", path.c_str(), 1);
        setenv("RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_FUTEX", "1", 0);
        setenv("RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_PREWAKE", "3", 0);
        setenv("RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_PREWAKE_SPIN_US", "50000", 0);
        setenv("RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_SPIN_US", "20", 0);
        return HostExpertPool::get();
    }();
    return p;
}

/* Polls `ok` for up to `s` seconds. */
template <class F> bool eventually(F ok, int s = 5) {
    const auto t0 = std::chrono::steady_clock::now();
    while (!ok()) {
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(s)) return false;
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    return true;
}

}  // namespace

TEST(host_expert_pool_futex_four_submitters_match_the_reference) {
    HostExpertPool* p = pool();
    if (!p || !__builtin_cpu_supports("avx2")) { SKIP("no AVX2 or no pool"); return; }
    if (!p->futex()) { SKIP("..._FUTEX=0 in the environment"); return; }
    CHECK(p->report().find(" futex wake ") != std::string::npos);
    std::vector<Expert> ex(2);
    for (int i = 0; i < 2; ++i) {
        ex[(size_t)i].gu = blocks(2 * kFf, kEmbd, 110 + (uint64_t)i);
        ex[(size_t)i].dn = blocks(kEmbd, kFf, 120 + (uint64_t)i);
    }
    std::vector<std::vector<uint16_t>> xs, want(8);
    for (int t = 0; t < 4; ++t) xs.push_back(row(130 + (uint64_t)t));
    for (int r = 0; r < 4; ++r)
        for (int e = 0; e < 2; ++e) want[(size_t)(r * 2 + e)] = reference(ex[(size_t)e], xs[(size_t)r]);
    std::vector<std::vector<uint16_t>> got(8, std::vector<uint16_t>(kEmbd));
    for (int round = 0; round < 2; ++round) {
        std::vector<std::thread> th;
        for (int r = 0; r < 4; ++r)
            th.emplace_back([&, r] {
                HostExpertPool::Ticket t;
                HostExpertPool::Job j[2];
                for (int e = 0; e < 2; ++e) {
                    j[e].gu = ex[(size_t)e].gu.data();
                    j[e].dn = ex[(size_t)e].dn.data();
                    j[e].x = xs[(size_t)r].data();
                    j[e].y = got[(size_t)(r * 2 + e)].data();
                }
                p->reserve(2 * HostExpertPool::kExpertBytes);
                CHECK_EQ(p->submit(j, 2, &t), 0);
                p->wait(&t);
            });
        for (auto& t : th) t.join();
        for (size_t i = 0; i < want.size(); ++i) CHECK(got[i] == want[i]);
        for (auto& g : got) std::fill(g.begin(), g.end(), (uint16_t)0xffff);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));   /* the workers asleep again */
    }
    CHECK_EQ(p->backlog(), (int64_t)0);
}

/* Four threads submit one job at a time after a random pause of 0-400 us: the workers' 20 us spin
 * ends in most pauses, so each submit races workers on their way to sleep. Every ticket must finish
 * (a lost wake-up would leave its job queued with every worker asleep) and every row be right. */
TEST(host_expert_pool_futex_loses_no_wakeup) {
    HostExpertPool* p = pool();
    if (!p || !__builtin_cpu_supports("avx2")) { SKIP("no AVX2 or no pool"); return; }
    if (!p->futex()) { SKIP("..._FUTEX=0 in the environment"); return; }
    const Expert e{ blocks(2 * kFf, kEmbd, 140), blocks(kEmbd, kFf, 141) };
    const auto x = row(142);
    const auto want = reference(e, x);
    const HostExpertPool::Stats s0 = p->stats();
    constexpr int kThreads = 4, kRounds = 30;
    std::atomic<int> lost{ 0 }, wrong{ 0 };
    std::vector<std::thread> th;
    for (int r = 0; r < kThreads; ++r)
        th.emplace_back([&, r] {
            std::mt19937 rng(1234u + (unsigned)r);
            std::vector<uint16_t> y(kEmbd);
            for (int i = 0; i < kRounds; ++i) {
                std::this_thread::sleep_for(std::chrono::microseconds(rng() % 400));
                std::fill(y.begin(), y.end(), (uint16_t)0xffff);
                HostExpertPool::Job j;
                j.gu = e.gu.data(); j.dn = e.dn.data(); j.x = x.data(); j.y = y.data();
                j.urgent = (i % 3) == 0;
                HostExpertPool::Ticket t;
                p->reserve(HostExpertPool::kExpertBytes);
                if (p->submit(&j, 1, &t) != 0) { lost.fetch_add(1000); return; }
                /* A job takes a few ms; 3 s means a submit whose wake-up was lost and that only a
                 * later submit's wake rescued (one for good never returns: the test's timeout). */
                if (!eventually([&] { return p->done(&t); }, 3)) {
                    lost.fetch_add(1);
                    p->wait(&t);
                }
                if (y != want) wrong.fetch_add(1);
            }
        });
    for (auto& t : th) t.join();
    CHECK_EQ(lost.load(), 0);
    CHECK_EQ(wrong.load(), 0);
    const HostExpertPool::Stats s1 = p->stats();
    CHECK_EQ(s1.submits - s0.submits, (uint64_t)(kThreads * kRounds));
    CHECK_EQ(s1.first_n - s0.first_n, (uint64_t)(kThreads * kRounds));
    CHECK(s1.sleeps > s0.sleeps);           /* the workers did go to sleep between jobs */
    CHECK(s1.submit_woke > s0.submit_woke);   /* and submits woke them */
    CHECK_EQ(p->backlog(), (int64_t)0);
}

/* ..._PREWAKE on the futex: three of the four workers woken, all back asleep after their window
 * with no tile; a call within the window wakes nobody; after it, a pre-wake then a submit: a woken
 * worker takes some of the job's tiles. */
TEST(host_expert_pool_futex_prewake_wakes_a_subset_for_its_window) {
    HostExpertPool* p = pool();
    if (!p || !__builtin_cpu_supports("avx2")) { SKIP("no AVX2 or no pool"); return; }
    if (!p->futex() || p->prewake_n() != 3) { SKIP("the environment sets the futex or the pre-wake"); return; }
    constexpr int R = HostExpertPool::kWakeRouter, M = HostExpertPool::kWakeMiss;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));   /* every worker asleep */
    const HostExpertPool::Stats s0 = p->stats();
    const int woke = p->prewake(R);
    CHECK_EQ(woke, 3);
    CHECK_EQ(p->prewake(R), 0);   /* inside the window */
    CHECK(eventually([&] {
        const HostExpertPool::Stats s = p->stats();
        return s.pre_slept[R] + s.pre_used[R] >= s0.pre_slept[R] + s0.pre_used[R] + (uint64_t)woke;
    }));
    const HostExpertPool::Stats s1 = p->stats();
    CHECK_EQ(s1.pre_calls[R], s0.pre_calls[R] + 1);
    CHECK_EQ(s1.pre_woken[R], s0.pre_woken[R] + 3);
    CHECK_EQ(s1.pre_skipped[R], s0.pre_skipped[R] + 1);
    CHECK_EQ(s1.pre_used[R], s0.pre_used[R]);
    CHECK(s1.pre_slept[R] >= s0.pre_slept[R] + 3);
    std::this_thread::sleep_for(std::chrono::milliseconds(60));   /* past the window */
    const Expert e{ blocks(2 * kFf, kEmbd, 150), blocks(kEmbd, kFf, 151) };
    const auto x = row(152);
    std::vector<uint16_t> y(kEmbd, 0xffff);
    HostExpertPool::Job j;
    j.gu = e.gu.data(); j.dn = e.dn.data(); j.x = x.data(); j.y = y.data();
    CHECK_EQ(p->prewake(M), 3);
    HostExpertPool::Ticket t;
    p->reserve(HostExpertPool::kExpertBytes);
    CHECK_EQ(p->submit(&j, 1, &t), 0);
    p->wait(&t);
    CHECK(y == reference(e, x));
    CHECK(eventually([&] { return p->stats().pre_used[M] >= s1.pre_used[M] + 1; }));
    const std::string rep = p->report();
    CHECK(rep.find("router pre-wakes: ") != std::string::npos);
    CHECK(rep.find("miss pre-wakes: ") != std::string::npos);
    CHECK_EQ(p->backlog(), (int64_t)0);
}

RAD_TEST_MAIN()
