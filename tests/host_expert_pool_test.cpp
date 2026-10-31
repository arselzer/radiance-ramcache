/* host_expert_pool_test.cpp -- the CPU pool for DeepSeek-V4.1 routed experts
 * (core/place/host_expert_pool.h): every row equals the scalar reference's (tools/ds41-host-experts,
 * gemv_ref) bit for bit with four submitters at once and more jobs than slots, and the backlog
 * returns to zero. The silu table here is the host's; the served one comes from the card.
 *
 * Jobs carry up to 64 token rows here (RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_MAX_ROWS, set before
 * the pool reads it unless the environment already sets it: run with ..._MAX_ROWS=4 for the default
 * limit, where the many-row cases skip).
 *
 * The workers sleep through the condition variable here (..._FUTEX unset; the futex path has
 * host_expert_pool_futex_test), with ..._PREWAKE=2 and a 50 ms pre-wake window: woken workers
 * spin their window and sleep again, a second call within the window wakes nobody, and pre-woken
 * workers take the next submit's tiles. */
#include <algorithm>
#include "rad_test.h"

#include "place/host_expert_pool.h"
#include "../tools/ds41-host-experts/ds41_host_experts.h"

#include <chrono>
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

/* Before anything reads it (max_rows() reads it once); an outside setting wins. */
const bool g_max_rows_set = setenv("RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_MAX_ROWS", "64", 0) == 0;
/* The pre-wake (read once, at the pool's start): two workers, a 50 ms window. */
const bool g_prewake_set = setenv("RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_PREWAKE", "2", 0) == 0 &&
                           setenv("RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_PREWAKE_SPIN_US", "50000", 0) == 0;

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

/* The same through the one-row AVX2 kernel (gemv_avx2, proven equal to gemv_ref by
 * ds41_host_experts_test check): for many rows within the CPU budget. */
std::vector<uint16_t> reference_avx2(const Expert& e, const std::vector<uint16_t>& x) {
    ds41hx::Expert ex;
    ex.gu.N = 2 * kFf; ex.gu.K = kEmbd; ex.gu.blocks = e.gu.data();
    ex.dn.N = kEmbd;   ex.dn.K = kFf;   ex.dn.blocks = e.dn.data();
    std::vector<uint16_t> g(2 * kFf), a(kFf), d(kEmbd);
    ds41hx::expert_forward(ex, g_silu.data(), 10.0f, x.data(), g.data(), a.data(), d.data(), 1, true);
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
    const int n = HostExpertPool::slots() + 44;   /* past the pool's slots: submit waits for slots to free */
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
    bad.more = HostExpertPool::max_rows();
    CHECK(p->submit(&bad, 1, &t2) < 0);
    /* An expert's rows as even jobs of at most max_rows(): nine rows are one job at 32 or 64, three of
     * three at four. */
    const int m = HostExpertPool::max_rows();
    CHECK_EQ(HostExpertPool::cost_rows(9), m >= 9 ? HostExpertPool::job_cost(9) : 3 * HostExpertPool::job_cost(3));
    CHECK_EQ(HostExpertPool::job_cost(4), HostExpertPool::kExpertBytes + 3 * (HostExpertPool::kExpertBytes / 2));
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
    const char* stream = std::getenv("RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_STREAM_DECODE");
    const bool requested_stream = stream && !std::strcmp(stream, "1");
    CHECK_EQ(p->stream_decode(), requested_stream);
    // Includes a native packed single-row job with full fast rounds; the flag must not be a no-op.
    CHECK_EQ(p->stream_decode_used(), requested_stream);
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

/* Jobs of many rows (the many-row kernel, ..._MAX_ROWS=32): 32 rows of block_mxfp4, 17 packed (urgent)
 * and 9 planar of one expert, submitted with a one-row and a four-row job beside them, from two
 * submitters at once: every row as the one-row kernel gives it; then one row past the limit is
 * refused and the backlog drains. */
TEST(host_expert_pool_many_row_jobs_match) {
    HostExpertPool* p = pool();
    if (!p || !__builtin_cpu_supports("avx2")) { SKIP("no AVX2 or no pool"); return; }
    CHECK(g_max_rows_set);
    if (HostExpertPool::max_rows() < 32) { SKIP("..._MAX_ROWS below 32 in the environment"); return; }
    Expert e{ release_blocks(2 * kFf, kEmbd, 71, 1500), release_blocks(kEmbd, kFf, 72, 1500) };
    std::vector<uint8_t> guc, gus, dnc, dns;
    planes_of(e.gu, 2 * kFf, kEmbd, &guc, &gus);
    planes_of(e.dn, kEmbd, kFf, &dnc, &dns);
    ds41hx::PackInfo gp, dp;
    REQUIRE(ds41hx::pack_plan(gus.data(), 2 * kFf, kEmbd, &gp) && ds41hx::pack_plan(dns.data(), kEmbd, kFf, &dp));
    std::vector<uint8_t> gk(ds41hx::packed_bytes(2 * kFf, kEmbd, gp.n_exc)), dk(ds41hx::packed_bytes(kEmbd, kFf, dp.n_exc));
    REQUIRE(ds41hx::pack_planes(guc.data(), gus.data(), 2 * kFf, kEmbd, gp.n_exc, gp, gk.data(), gk.size()) == ds41hx::kPackOk);
    REQUIRE(ds41hx::pack_planes(dnc.data(), dns.data(), kEmbd, kFf, dp.n_exc, dp, dk.data(), dk.size()) == ds41hx::kPackOk);
    constexpr int kRows = 32;
    std::vector<std::vector<uint16_t>> xs, want;
    for (int r = 0; r < kRows; ++r) {
        xs.push_back(row(300 + (uint64_t)r));
        want.push_back(reference_avx2(e, xs.back()));
    }
    /* job: layout (0 blocks, 1 packed, 2 planes), first row, rows, urgent */
    struct Spec { int lay, r0, rows; bool urgent; };
    const Spec spec[5] = { { 0, 0, 32, false }, { 1, 5, 17, true }, { 2, 20, 9, false }, { 1, 31, 1, false }, { 2, 3, 4, false } };
    std::vector<std::vector<std::vector<uint16_t>>> got(5);
    HostExpertPool::Job j[5];
    int64_t cost = 0;
    for (int k = 0; k < 5; ++k) {
        const Spec& sp = spec[k];
        got[(size_t)k].assign((size_t)sp.rows, std::vector<uint16_t>(kEmbd, 0xffff));
        HostExpertPool::Job& q = j[k];
        if (sp.lay == 0) { q.gu = e.gu.data(); q.dn = e.dn.data(); }
        else if (sp.lay == 1) {
            q.gu = gk.data(); q.dn = dk.data(); q.packed = true;
            q.gu_bytes = (int64_t)gk.size(); q.dn_bytes = (int64_t)dk.size();
        } else { q.gu = guc.data(); q.gu_scales = gus.data(); q.dn = dnc.data(); q.dn_scales = dns.data(); }
        q.x = xs[(size_t)sp.r0].data();
        q.y = got[(size_t)k][0].data();
        for (int i = 1; i < sp.rows; ++i) {
            q.xs[i - 1] = xs[(size_t)(sp.r0 + i)].data();
            q.ys[i - 1] = got[(size_t)k][(size_t)i].data();
        }
        q.more = sp.rows - 1;
        q.urgent = sp.urgent;
        cost += HostExpertPool::job_cost(sp.rows);
    }
    p->reserve(cost);
    HostExpertPool::Ticket t1, t2;
    std::thread other([&] { CHECK_EQ(p->submit(j + 3, 2, &t2), 0); p->wait(&t2); });
    CHECK_EQ(p->submit(j, 3, &t1), 0);
    p->wait(&t1);
    other.join();
    int bad = 0;
    for (int k = 0; k < 5; ++k)
        for (int i = 0; i < spec[k].rows; ++i) bad += got[(size_t)k][(size_t)i] != want[(size_t)(spec[k].r0 + i)];
    CHECK_EQ(bad, 0);
    CHECK_EQ(p->backlog(), (int64_t)0);
    HostExpertPool::Job over = j[0];
    over.more = HostExpertPool::max_rows();   /* one row past the limit */
    HostExpertPool::Ticket t3;
    CHECK(p->submit(&over, 1, &t3) < 0);
    CHECK_EQ(t3.left.load(), 0);
    /* 40 rows: one job at 64, two of 20 at 32 (even cuts, not 32 and 8). */
    CHECK_EQ(HostExpertPool::cost_rows(40), HostExpertPool::max_rows() >= 40 ? HostExpertPool::job_cost(40)
                                                                           : 2 * HostExpertPool::job_cost(20));
}

/* The even cuts (HostExpertPool::jobs_for / job_rows, which the file stager's hx_jobs_for follows):
 * ceil(n / max_rows()) jobs whose rows differ by at most one, the larger first, summing to n. */
TEST(host_expert_pool_cuts_an_experts_rows_evenly) {
    const int m = HostExpertPool::max_rows();
    for (int n = 0; n <= 4 * 64 + 3; ++n) {
        const int jobs = HostExpertPool::jobs_for(n);
        CHECK_EQ(jobs, n == 0 ? 0 : (n + m - 1) / m);
        int sum = 0, lo = 1 << 30, hi = 0, prev = 1 << 30;
        int64_t cost = 0;
        for (int k = 0; k < jobs; ++k) {
            const int r = HostExpertPool::job_rows(n, k);
            CHECK(r >= 1 && r <= m);
            CHECK(r <= prev);
            prev = r;
            sum += r;
            lo = std::min(lo, r);
            hi = std::max(hi, r);
            cost += HostExpertPool::job_cost(r);
        }
        CHECK_EQ(sum, n);
        if (jobs) CHECK(hi - lo <= 1);
        CHECK_EQ(HostExpertPool::cost_rows(n), cost);
    }
}

/* Jobs of up to 64 rows (two many-row calls a tile), from four submitters at once, more jobs in flight
 * than the pool has slots, two submitters in one order and two in the reverse: every row as the
 * one-row kernel gives it, whatever the order and the cut, no submitter stuck, the backlog drained.
 * Mostly one-row jobs (each still reads a whole expert), so the slots fill at a test's CPU cost. */
TEST(host_expert_pool_64_row_jobs_from_four_submitters) {
    HostExpertPool* p = pool();
    if (!p || !__builtin_cpu_supports("avx2")) { SKIP("no AVX2 or no pool"); return; }
    if (HostExpertPool::max_rows() < 64) { SKIP("..._MAX_ROWS below 64 in the environment"); return; }
    Expert e{ release_blocks(2 * kFf, kEmbd, 81, 1500), release_blocks(kEmbd, kFf, 82, 1500) };
    constexpr int kRows = 64;
    std::vector<std::vector<uint16_t>> xs, want;
    for (int r = 0; r < kRows; ++r) {
        xs.push_back(row(400 + (uint64_t)r));
        want.push_back(reference_avx2(e, xs.back()));
    }
    /* Each submitter: 66 jobs (264 in all, past the pool's 256 slots); every 22nd one of 64, 33, 17 or 5
     * rows (cycled, so each size is a job of each submitter's or another's), the rest one row; rows r0..
     * of the 64. */
    const int sizes[4] = { 64, 33, 17, 5 };
    constexpr int kSub = 4, kJobs = 66;
    static_assert(kSub * kJobs > 256, "more jobs than the pool's slots");
    auto rows_of = [&](int s2, int q) { return q % 22 == 0 ? sizes[(q / 22 + s2) % 4] : 1; };
    auto r0_of = [&](int s2, int q) { return (q * 7 + s2) % (kRows - rows_of(s2, q) + 1); };
    std::vector<std::vector<std::vector<uint16_t>>> got(kSub * kJobs);
    std::vector<std::vector<HostExpertPool::Job>> jobs(kSub);
    int64_t cost = 0;
    for (int s2 = 0; s2 < kSub; ++s2)
        for (int q = 0; q < kJobs; ++q) {
            const int idx = s2 * kJobs + q, rows = rows_of(s2, q), r0 = r0_of(s2, q);
            got[(size_t)idx].assign((size_t)rows, std::vector<uint16_t>(kEmbd, 0xffff));
            HostExpertPool::Job j;
            j.gu = e.gu.data();
            j.dn = e.dn.data();
            j.x = xs[(size_t)r0].data();
            j.y = got[(size_t)idx][0].data();
            for (int i = 1; i < rows; ++i) {
                j.xs[i - 1] = xs[(size_t)(r0 + i)].data();
                j.ys[i - 1] = got[(size_t)idx][(size_t)i].data();
            }
            j.more = rows - 1;
            j.urgent = (q % 9) == 0;
            cost += HostExpertPool::job_cost(rows);
            jobs[(size_t)s2].push_back(j);
        }
    for (int s2 = 1; s2 < kSub; s2 += 2) std::reverse(jobs[(size_t)s2].begin(), jobs[(size_t)s2].end());
    p->reserve(cost);
    std::vector<std::thread> th;
    for (int s2 = 0; s2 < kSub; ++s2)
        th.emplace_back([&, s2] {
            HostExpertPool::Ticket t;
            /* In bursts of 22, as a layer's units come. */
            for (int q = 0; q < kJobs; q += 22) CHECK_EQ(p->submit(jobs[(size_t)s2].data() + q, 22, &t), 0);
            p->wait(&t);
        });
    for (auto& t : th) t.join();
    int bad = 0;
    for (int s2 = 0; s2 < kSub; ++s2)
        for (int q = 0; q < kJobs; ++q) {
            const int idx = s2 * kJobs + q, rows = rows_of(s2, q), r0 = r0_of(s2, q);
            for (int i = 0; i < rows; ++i) bad += got[(size_t)idx][(size_t)i] != want[(size_t)(r0 + i)];
        }
    CHECK_EQ(bad, 0);
    CHECK_EQ(p->backlog(), (int64_t)0);
    fprintf(stderr, "    submit waited for a slot %llu times so far (%.1f ms)\n",
            (unsigned long long)p->submit_waits(), (double)p->submit_wait_ns() / 1e6);
}

/* Polls `ok` for up to `s` seconds. */
template <class F> static bool eventually(F ok, int s = 5) {
    const auto t0 = std::chrono::steady_clock::now();
    while (!ok()) {
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(s)) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

/* ..._PREWAKE through the condition variable: two of the four workers woken, both back asleep after
 * their window with no tile; a call inside the window wakes nobody; a submit right after a pre-wake
 * finds woken workers spinning, and the submit counters see it. */
TEST(host_expert_pool_prewake_wakes_a_subset_for_its_window) {
    HostExpertPool* p = pool();
    if (!p || !__builtin_cpu_supports("avx2")) { SKIP("no AVX2 or no pool"); return; }
    REQUIRE(g_prewake_set);
    if (p->prewake_n() != 2 || p->futex()) { SKIP("the environment sets the pre-wake or the futex"); return; }
    constexpr int R = HostExpertPool::kWakeRouter, M = HostExpertPool::kWakeMiss;
    /* Every worker asleep: well past the 200 us idle spin. */
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    const HostExpertPool::Stats s0 = p->stats();
    CHECK_EQ(p->prewake(R), 2);
    CHECK_EQ(p->prewake(M), 0);   /* inside the window: nobody */
    CHECK(eventually([&] { return p->stats().pre_slept[R] >= s0.pre_slept[R] + 2; }));
    HostExpertPool::Stats s1 = p->stats();
    CHECK_EQ(s1.pre_calls[R], s0.pre_calls[R] + 1);
    CHECK_EQ(s1.pre_woken[R], s0.pre_woken[R] + 2);
    CHECK_EQ(s1.pre_skipped[M], s0.pre_skipped[M] + 1);
    CHECK_EQ(s1.pre_used[R], s0.pre_used[R]);   /* no work came */
    CHECK(s1.pre_spin_ns[R] - s0.pre_spin_ns[R] >= 2ull * 45'000'000);   /* their window, each */
    /* A pre-wake, then a job: a woken worker takes some of its tiles. */
    std::this_thread::sleep_for(std::chrono::milliseconds(60));   /* past the window */
    Expert e{ blocks(2 * kFf, kEmbd, 81), blocks(kEmbd, kFf, 82) };
    const auto x = row(83);
    std::vector<uint16_t> y(kEmbd, 0xffff);
    HostExpertPool::Job j;
    j.gu = e.gu.data(); j.dn = e.dn.data(); j.x = x.data(); j.y = y.data();
    CHECK_EQ(p->prewake(M), 2);
    HostExpertPool::Ticket t;
    p->reserve(HostExpertPool::kExpertBytes);
    CHECK_EQ(p->submit(&j, 1, &t), 0);
    p->wait(&t);
    CHECK(y == reference_avx2(e, x));
    CHECK(eventually([&] { return p->stats().pre_used[M] >= s1.pre_used[M] + 1; }));
    const HostExpertPool::Stats s2 = p->stats();
    CHECK_EQ(s2.submits, s1.submits + 1);
    CHECK_EQ(s2.first_n, s1.first_n + 1);
    CHECK_EQ(s2.ramp_n, s1.ramp_n + 1);
    CHECK(p->report().find("router pre-wakes: ") != std::string::npos);
    CHECK(p->report().find("condition-variable wake") != std::string::npos);
    CHECK_EQ(p->backlog(), (int64_t)0);
}

/* RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_LEAN_TILES (the host_expert_pool_test_lean entry runs every
 * case with it): a job alone has its 152 tiles claimed through the active-slot bitmap, one slot looked
 * at a claim, each tile laying x out or reusing it, and a worker lays it out at most once a
 * projection; the rows are the reference's. Without the switch nothing is counted. */
TEST(host_expert_pool_lean_tiles_claim_through_the_bitmap_and_reuse_lanes) {
    HostExpertPool* p = pool();
    if (!p || !__builtin_cpu_supports("avx2")) { SKIP("no AVX2 or no pool"); return; }
    Expert e{ release_blocks(2 * kFf, kEmbd, 81, 2000), release_blocks(kEmbd, kFf, 82, 2000) };
    std::vector<uint8_t> guc, gus, dnc, dns;
    planes_of(e.gu, 2 * kFf, kEmbd, &guc, &gus);
    planes_of(e.dn, kEmbd, kFf, &dnc, &dns);
    std::vector<std::vector<uint16_t>> xs, want;
    for (int r = 0; r < 3; ++r) {
        xs.push_back(row(400 + (uint64_t)r));
        want.push_back(reference(e, xs.back()));
    }
    /* One row of blocks, then three rows of planes: the one-row and the multi-row kernels' lanes. */
    for (int rows : { 1, 3 }) {
        const HostExpertPool::LeanStats a = p->lean_stats();
        std::vector<std::vector<uint16_t>> got((size_t)rows, std::vector<uint16_t>(kEmbd, 0xffff));
        HostExpertPool::Job j;
        if (rows == 1) { j.gu = e.gu.data(); j.dn = e.dn.data(); }
        else { j.gu = guc.data(); j.gu_scales = gus.data(); j.dn = dnc.data(); j.dn_scales = dns.data(); }
        j.x = xs[0].data();
        j.y = got[0].data();
        for (int r = 1; r < rows; ++r) { j.xs[r - 1] = xs[(size_t)r].data(); j.ys[r - 1] = got[(size_t)r].data(); }
        j.more = rows - 1;
        HostExpertPool::Ticket t;
        p->reserve(HostExpertPool::job_cost(rows));
        CHECK_EQ(p->submit(&j, 1, &t), 0);
        p->wait(&t);
        for (int r = 0; r < rows; ++r) CHECK(got[(size_t)r] == want[(size_t)r]);
        const HostExpertPool::LeanStats b = p->lean_stats();
        if (!p->lean()) {
            CHECK_EQ(b.claims + b.scanned + b.lanes_built + b.lanes_reused, (uint64_t)0);
            continue;
        }
        CHECK_EQ(b.claims - a.claims, (uint64_t)152);
        CHECK_EQ(b.scanned - a.scanned, (uint64_t)152);
        CHECK_EQ(b.lanes_built - a.lanes_built + (b.lanes_reused - a.lanes_reused), (uint64_t)152);
        CHECK(b.lanes_built - a.lanes_built <= 2 * (uint64_t)p->threads());
    }
    CHECK_EQ(p->backlog(), (int64_t)0);
}

/* quiet_after_tickets (the sliced background inserts, ..._INSERT_SLICE_KIB): a wait_idle sleeper is
 * still woken when the pool drains, now after the draining job's ticket has counted down, never
 * left to its timeout. */
TEST(host_expert_pool_wait_idle_wakes_after_the_ticket) {
    HostExpertPool* p = pool();
    if (!p || !__builtin_cpu_supports("avx2")) { SKIP("no AVX2 or no pool"); return; }
    p->quiet_after_tickets();
    Expert e{ blocks(2 * kFf, kEmbd, 91), blocks(kEmbd, kFf, 92) };
    const auto x = row(93);
    const auto want = reference(e, x);
    for (int round = 0; round < 3; ++round) {
        std::vector<uint16_t> y(kEmbd, 0xffff);
        HostExpertPool::Job j;
        j.gu = e.gu.data();
        j.dn = e.dn.data();
        j.x = x.data();
        j.y = y.data();
        HostExpertPool::Ticket t;
        p->reserve(HostExpertPool::kExpertBytes);   /* the sleeper sees work from here */
        const auto t0 = std::chrono::steady_clock::now();
        std::thread sleeper([&] { p->wait_idle(4000000000LL); });
        CHECK_EQ(p->submit(&j, 1, &t), 0);
        p->wait(&t);
        sleeper.join();
        CHECK(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 2.0);
        CHECK(y == want);
        CHECK_EQ(p->backlog(), (int64_t)0);
    }
}

/* work() / wait_work (the sliced background inserts' gate, ..._INSERT_SLICE_KIB): a reservation alone
 * (a miss whose planes are being read) is not work, so it never holds the gate, while backlog() and
 * wait_idle still count it; a submitted job is work until its last tile, and a wait_work sleeper is
 * woken when it ends, after its ticket has counted down, never left to its timeout. */
TEST(host_expert_pool_work_counts_submitted_jobs_not_reservations) {
    HostExpertPool* p = pool();
    if (!p || !__builtin_cpu_supports("avx2")) { SKIP("no AVX2 or no pool"); return; }
    REQUIRE_EQ(p->work(), (int64_t)0);
    p->reserve(HostExpertPool::kExpertBytes);
    CHECK_EQ(p->work(), (int64_t)0);
    CHECK(p->backlog() > 0);
    auto t0 = std::chrono::steady_clock::now();
    p->wait_work(2000000000LL);   /* nothing submitted: back at once */
    CHECK(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 0.5);
    Expert e{ blocks(2 * kFf, kEmbd, 95), blocks(kEmbd, kFf, 96) };
    const auto x = row(97);
    const auto want = reference(e, x);
    for (int round = 0; round < 3; ++round) {
        std::vector<uint16_t> y(kEmbd, 0xffff);
        HostExpertPool::Job j;
        j.gu = e.gu.data();
        j.dn = e.dn.data();
        j.x = x.data();
        j.y = y.data();
        HostExpertPool::Ticket t;
        if (round) p->reserve(HostExpertPool::kExpertBytes);
        std::atomic<bool> woke{ false };
        t0 = std::chrono::steady_clock::now();
        CHECK_EQ(p->submit(&j, 1, &t), 0);
        std::thread sleeper([&] {
            p->wait_work(4000000000LL);
            woke = true;
            CHECK(p->done(&t));   /* woken after the ticket counted down */
        });
        p->wait(&t);
        sleeper.join();
        CHECK(woke.load());
        CHECK(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 2.0);
        CHECK(y == want);
        CHECK_EQ(p->work(), (int64_t)0);
        CHECK_EQ(p->backlog(), (int64_t)0);
    }
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
