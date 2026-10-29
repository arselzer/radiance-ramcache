/* host_expert_pool_test.cpp -- the CPU pool for DeepSeek-V4.1 routed experts
 * (core/place/host_expert_pool.h): every row equals the scalar reference's (tools/ds41-host-experts,
 * gemv_ref) bit for bit with four submitters at once and more jobs than slots, and the backlog
 * returns to zero. The silu table here is the host's; the served one comes from the card. */
#include "rad_test.h"

#include "place/host_expert_pool.h"
#include "../tools/ds41-host-experts/ds41_host_experts.h"

#include <cstdio>
#include <cstdlib>
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

/* The pool reads its configuration once: four workers and the host's silu table. */
HostExpertPool* pool() {
    static HostExpertPool* p = [] {
        g_silu.resize(65536);
        ds41hx::silu_table_host(g_silu.data(), 10.0f);
        const std::string path = radtest::tmpdir() + "/silu_host.bin";
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f || std::fwrite(g_silu.data(), 4, 65536, f) != 65536) return (HostExpertPool*)nullptr;
        std::fclose(f);
        setenv("RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS", "4", 1);
        setenv("RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_SILU", path.c_str(), 1);
        return HostExpertPool::get();
    }();
    return p;
}

std::vector<uint16_t> reference(const Expert& e, const std::vector<uint16_t>& x) {
    ds41hx::Expert ex;
    ex.gu.N = 2 * kFf; ex.gu.K = kEmbd; ex.gu.blocks = e.gu.data();
    ex.dn.N = kEmbd;   ex.dn.K = kFf;   ex.dn.blocks = e.dn.data();
    std::vector<uint16_t> g(2 * kFf), a(kFf), d(kEmbd);
    ds41hx::expert_forward(ex, g_silu.data(), 10.0f, x.data(), g.data(), a.data(), d.data(), 1, false);
    return d;
}

}  // namespace

TEST(host_expert_pool_four_submitters_match_the_reference) {
    HostExpertPool* p = pool();
    if (!p || !__builtin_cpu_supports("avx2")) { SKIP("no AVX2 or no pool"); return; }
    std::vector<Expert> ex(3);
    for (int i = 0; i < 3; ++i) {
        ex[(size_t)i].gu = blocks(2 * kFf, kEmbd, 10 + (uint64_t)i);
        ex[(size_t)i].dn = blocks(kEmbd, kFf, 20 + (uint64_t)i);
    }
    std::vector<std::vector<uint16_t>> xs;
    for (int t = 0; t < 4; ++t) xs.push_back(row((uint64_t)t));
    std::vector<std::vector<uint16_t>> want(12);
    for (int r = 0; r < 4; ++r)
        for (int e = 0; e < 3; ++e) want[(size_t)(r * 3 + e)] = reference(ex[(size_t)e], xs[(size_t)r]);

    /* Four "ranks" submit their three experts on their own token row at once, twice. */
    std::vector<std::vector<uint16_t>> got(12, std::vector<uint16_t>(kEmbd));
    for (int round = 0; round < 2; ++round) {
        std::vector<std::thread> th;
        for (int r = 0; r < 4; ++r)
            th.emplace_back([&, r] {
                HostExpertPool::Ticket t;
                HostExpertPool::Job j[3];
                for (int e = 0; e < 3; ++e) {
                    j[e].gu = ex[(size_t)e].gu.data();
                    j[e].dn = ex[(size_t)e].dn.data();
                    j[e].x = xs[(size_t)r].data();
                    j[e].y = got[(size_t)(r * 3 + e)].data();
                }
                p->reserve(3 * HostExpertPool::kExpertBytes);
                CHECK_EQ(p->submit(j, 3, &t), 0);
                p->wait(&t);
                CHECK(p->done(&t));
            });
        for (auto& t : th) t.join();
        for (size_t i = 0; i < want.size(); ++i) CHECK(got[i] == want[i]);
        for (auto& g : got) std::fill(g.begin(), g.end(), (uint16_t)0xffff);
    }
    CHECK_EQ(p->backlog(), (int64_t)0);
}

TEST(host_expert_pool_more_jobs_than_slots) {
    HostExpertPool* p = pool();
    if (!p || !__builtin_cpu_supports("avx2")) { SKIP("no AVX2 or no pool"); return; }
    Expert e{ blocks(2 * kFf, kEmbd, 5), blocks(kEmbd, kFf, 6) };
    const auto x = row(9);
    const auto want = reference(e, x);
    constexpr int n = 160;   /* past the pool's 128 slots: submit waits for slots to free */
    std::vector<std::vector<uint16_t>> got(n, std::vector<uint16_t>(kEmbd));
    std::vector<HostExpertPool::Job> j(n);
    for (int i = 0; i < n; ++i) {
        j[(size_t)i].gu = e.gu.data();
        j[(size_t)i].dn = e.dn.data();
        j[(size_t)i].x = x.data();
        j[(size_t)i].y = got[(size_t)i].data();
    }
    HostExpertPool::Ticket t;
    p->reserve(n * HostExpertPool::kExpertBytes);
    CHECK_EQ(p->submit(j.data(), n, &t), 0);
    p->wait(&t);
    int bad = 0;
    for (const auto& g : got) bad += g != want;
    CHECK_EQ(bad, 0);
    CHECK_EQ(p->backlog(), (int64_t)0);
}

/* The release's planes of block_mxfp4 rows: codes [N, K/2] (element 2i in the low nibble) and e8m0
 * scales [N, K/32]. In a block, value j < 16 is the low nibble of code byte j, j + 16 its high one. */
static void planes_of(const std::vector<uint8_t>& b, int N, int K, std::vector<uint8_t>* codes,
                      std::vector<uint8_t>* scales) {
    const int nb = K / 32;
    codes->assign((size_t)N * K / 2, 0);
    scales->assign((size_t)N * nb, 0);
    for (int n = 0; n < N; ++n)
        for (int u = 0; u < nb; ++u) {
            const uint8_t* blk = &b[((size_t)n * nb + u) * 17];
            (*scales)[(size_t)n * nb + u] = blk[0];
            auto val = [&](int v) { return v < 16 ? blk[1 + v] & 15 : blk[1 + v - 16] >> 4; };
            for (int q = 0; q < 16; ++q)
                (*codes)[(size_t)n * (K / 2) + (size_t)u * 16 + q] =
                    (uint8_t)(val(2 * q) | (val(2 * q + 1) << 4));
        }
}

TEST(host_expert_pool_planes_and_urgent_jobs_match) {
    HostExpertPool* p = pool();
    if (!p || !__builtin_cpu_supports("avx2")) { SKIP("no AVX2 or no pool"); return; }
    Expert e{ blocks(2 * kFf, kEmbd, 31), blocks(kEmbd, kFf, 32) };
    std::vector<uint8_t> guc, gus, dnc, dns;
    planes_of(e.gu, 2 * kFf, kEmbd, &guc, &gus);
    planes_of(e.dn, kEmbd, kFf, &dnc, &dns);
    const auto x = row(17);
    const auto want = reference(e, x);
    std::vector<uint16_t> a(kEmbd), b(kEmbd, 0xffff);
    HostExpertPool::Job j[2];
    j[0].gu = e.gu.data(); j[0].dn = e.dn.data(); j[0].x = x.data(); j[0].y = a.data();
    j[1].gu = guc.data(); j[1].gu_scales = gus.data(); j[1].dn = dnc.data(); j[1].dn_scales = dns.data();
    j[1].x = x.data(); j[1].y = b.data(); j[1].urgent = true;
    HostExpertPool::Ticket t;
    p->reserve(2 * HostExpertPool::kExpertBytes);
    CHECK_EQ(p->submit(j, 2, &t), 0);
    p->wait(&t);
    CHECK(a == want);
    CHECK(b == want);
    /* One plane pointer without the other is refused. */
    HostExpertPool::Job bad = j[1];
    bad.dn_scales = nullptr;
    HostExpertPool::Ticket t2;
    CHECK(p->submit(&bad, 1, &t2) < 0);
    CHECK_EQ(p->backlog(), (int64_t)0);
}

TEST(host_expert_pool_multi_row_jobs_match) {
    HostExpertPool* p = pool();
    if (!p || !__builtin_cpu_supports("avx2")) { SKIP("no AVX2 or no pool"); return; }
    Expert e{ blocks(2 * kFf, kEmbd, 41), blocks(kEmbd, kFf, 42) };
    std::vector<uint8_t> guc, gus, dnc, dns;
    planes_of(e.gu, 2 * kFf, kEmbd, &guc, &gus);
    planes_of(e.dn, kEmbd, kFf, &dnc, &dns);
    std::vector<std::vector<uint16_t>> xs, want, got(6, std::vector<uint16_t>(kEmbd, 0xffff));
    for (int r = 0; r < 4; ++r) {
        xs.push_back(row(100 + (uint64_t)r));
        want.push_back(reference(e, xs.back()));
    }
    /* Four rows in one block-layout job, and rows 1 and 3 again in one planar urgent job. */
    HostExpertPool::Job j[2];
    j[0].gu = e.gu.data(); j[0].dn = e.dn.data();
    j[0].x = xs[0].data(); j[0].y = got[0].data();
    for (int r = 1; r < 4; ++r) { j[0].xs[r - 1] = xs[(size_t)r].data(); j[0].ys[r - 1] = got[(size_t)r].data(); }
    j[0].more = 3;
    j[1].gu = guc.data(); j[1].gu_scales = gus.data(); j[1].dn = dnc.data(); j[1].dn_scales = dns.data();
    j[1].x = xs[1].data(); j[1].y = got[4].data();
    j[1].xs[0] = xs[3].data(); j[1].ys[0] = got[5].data();
    j[1].more = 1; j[1].urgent = true;
    HostExpertPool::Ticket t;
    p->reserve(HostExpertPool::job_cost(4) + HostExpertPool::job_cost(2));
    CHECK_EQ(p->submit(j, 2, &t), 0);
    p->wait(&t);
    for (int r = 0; r < 4; ++r) CHECK(got[(size_t)r] == want[(size_t)r]);
    CHECK(got[4] == want[1]);
    CHECK(got[5] == want[3]);
    CHECK_EQ(p->backlog(), (int64_t)0);
    /* A missing extra row pointer, or too many rows, is refused. */
    HostExpertPool::Job bad = j[0];
    bad.xs[2] = nullptr;
    HostExpertPool::Ticket t2;
    CHECK(p->submit(&bad, 1, &t2) < 0);
    bad = j[0];
    bad.more = HostExpertPool::kMaxRows;
    CHECK(p->submit(&bad, 1, &t2) < 0);
    CHECK_EQ(HostExpertPool::cost_rows(9), 2 * HostExpertPool::job_cost(4) + HostExpertPool::job_cost(1));
}

/* block_mxfp4 rows with the release's scales: e8m0 120 or 121, now and then 119 or 122, and one
 * block in `rare` at 112, 117, 124 or 127 (raw rounds in the packed form). */
static std::vector<uint8_t> release_blocks(int N, int K, uint64_t seed, int rare) {
    std::vector<uint8_t> b = blocks(N, K, seed);
    uint64_t s = seed * 0xD1B54A32D192ED03ull + 7;
    static const uint8_t out[] = { 112, 117, 124, 127 };
    for (size_t i = 0; i < b.size(); i += 17) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        const int p = (int)((s >> 20) % 100);
        b[i] = (uint8_t)(p < 2 ? 119 : p < 50 ? 120 : p < 98 ? 121 : 122);
        if ((s >> 40) % (uint64_t)rare == 0) b[i] = out[(s >> 8) % 4];
    }
    return b;
}

/* The packed RAM cache's slots (RADIANCE_EXPERIMENTAL_DS41_PACKED_CACHE, ds41_host_experts.h): one-
 * and multi-row packed jobs give the reference's rows, beside block and planar jobs of the same
 * expert; a packed job whose slot does not describe its weight, is cut short, or also names scales
 * is refused. */
TEST(host_expert_pool_packed_jobs_match) {
    HostExpertPool* p = pool();
    if (!p || !__builtin_cpu_supports("avx2")) { SKIP("no AVX2 or no pool"); return; }
    Expert e{ release_blocks(2 * kFf, kEmbd, 61, 2000), release_blocks(kEmbd, kFf, 62, 2000) };
    std::vector<uint8_t> guc, gus, dnc, dns;
    planes_of(e.gu, 2 * kFf, kEmbd, &guc, &gus);
    planes_of(e.dn, kEmbd, kFf, &dnc, &dns);
    /* Slots that hold exactly their weight's raw rounds, as the engine sizes them for its budget. */
    auto slot = [](const std::vector<uint8_t>& codes, const std::vector<uint8_t>& scales, int N, int K,
                   ds41hx::PackInfo* plan) {
        std::vector<uint8_t> out;
        if (!ds41hx::pack_plan(scales.data(), N, K, plan)) return out;
        out.assign(ds41hx::packed_bytes(N, K, plan->n_exc), 0xEE);
        if (ds41hx::pack_planes(codes.data(), scales.data(), N, K, plan->n_exc, *plan, out.data(), out.size()) !=
            ds41hx::kPackOk)
            out.clear();
        return out;
    };
    ds41hx::PackInfo gp, dp;
    const std::vector<uint8_t> gk = slot(guc, gus, 2 * kFf, kEmbd, &gp), dk = slot(dnc, dns, kEmbd, kFf, &dp);
    REQUIRE(!gk.empty() && !dk.empty());
    CHECK(gp.n_exc > 0 && dp.n_exc > 0);   /* both widened and raw rounds */
    std::vector<std::vector<uint16_t>> xs, want, got(5, std::vector<uint16_t>(kEmbd, 0xffff));
    for (int r = 0; r < 3; ++r) {
        xs.push_back(row(200 + (uint64_t)r));
        want.push_back(reference(e, xs.back()));
    }
    HostExpertPool::Job j[3];
    j[0].gu = gk.data(); j[0].dn = dk.data(); j[0].packed = true;
    j[0].gu_bytes = (int64_t)gk.size(); j[0].dn_bytes = (int64_t)dk.size();
    j[0].x = xs[0].data(); j[0].y = got[0].data();
    j[1] = j[0];   /* three rows, urgent */
    j[1].x = xs[0].data(); j[1].y = got[1].data();
    j[1].xs[0] = xs[1].data(); j[1].ys[0] = got[2].data();
    j[1].xs[1] = xs[2].data(); j[1].ys[1] = got[3].data();
    j[1].more = 2; j[1].urgent = true;
    j[2].gu = e.gu.data(); j[2].dn = e.dn.data(); j[2].x = xs[2].data(); j[2].y = got[4].data();   /* blocks */
    HostExpertPool::Ticket t;
    p->reserve(HostExpertPool::job_cost(1) * 2 + HostExpertPool::job_cost(3));
    CHECK_EQ(p->submit(j, 3, &t), 0);
    p->wait(&t);
    CHECK(got[0] == want[0]);
    for (int r = 0; r < 3; ++r) CHECK(got[(size_t)r + 1] == want[(size_t)r]);
    CHECK(got[4] == want[2]);
    CHECK_EQ(p->backlog(), (int64_t)0);
    /* Refused: a slot a byte short, the weights' slots swapped (another shape), scales beside a packed
     * slot, and block bytes called packed. */
    HostExpertPool::Ticket t2;
    HostExpertPool::Job bad = j[0];
    bad.gu_bytes -= 1;
    CHECK(p->submit(&bad, 1, &t2) < 0);
    bad = j[0];
    std::swap(bad.gu, bad.dn);
    std::swap(bad.gu_bytes, bad.dn_bytes);
    CHECK(p->submit(&bad, 1, &t2) < 0);
    bad = j[0];
    bad.gu_scales = gus.data(); bad.dn_scales = dns.data();
    CHECK(p->submit(&bad, 1, &t2) < 0);
    bad = j[2];
    bad.packed = true; bad.gu_bytes = (int64_t)e.gu.size(); bad.dn_bytes = (int64_t)e.dn.size();
    CHECK(p->submit(&bad, 1, &t2) < 0);
    CHECK_EQ(t2.left.load(), 0);
    CHECK_EQ(p->backlog(), (int64_t)0);
}

TEST(host_expert_pool_refuses_incomplete_jobs) {
    HostExpertPool* p = pool();
    if (!p) { SKIP("no pool"); return; }
    HostExpertPool::Ticket t;
    HostExpertPool::Job j;   /* null pointers */
    CHECK(p->submit(&j, 1, &t) < 0);
    CHECK_EQ(t.left.load(), 0);
    CHECK_EQ(p->submit(nullptr, 0, &t), 0);
}

RAD_TEST_MAIN()
