/* hx_wake_stager_test.cpp -- the file stager's part of the CPU pool's pre-wake
 * (RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_PREWAKE, FileStager::hx_wake_*, stager.h) on the host
 * backend, through the stager's test friend. No fixture builds a DeepSeek-V4.1 EP4 program, so the
 * cases drive the pieces start_ep_routed and hx_early_decide call:
 *  - hx_wake_init finds each routed layer's router GEMV: the widest gemm reading its
 *    blk.L.glm_router.weight, not the TP-split part, not an on-die read, not another weight's gemm;
 *  - before_op records the router event before that op in a one-token pass only;
 *  - hx_wake_route waits for it and pre-wakes the pool when the counts are still out (here held
 *    behind a gate), and skips the pre-wake when they are in;
 *  - a miss read's pre-wake time is the median landing of the last reads of as many units, less
 *    the lead, from the eighth read on, apart for one unit and for more.
 * The pool is the process's (two workers, ..._PREWAKE=2); the silu table is never read here. */
#include "rad_test.h"

#include "rad_internal.h"
#include "device/device.h"
#include "place/host_expert_pool.h"
#include "place/stager.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#define REQUIRE_OK(expr) REQUIRE((expr) >= 0)

namespace rad {

/* FileStager's test friend (stager.h). Routed layer 0 is model layer 12; its gate_up is op 5. */
struct EpCountsHostProbe {
    Plan plan;
    Program prog;
    FileStager fs;

    EpCountsHostProbe(HostExpertPool* pool) {
        plan.rank = 0;
        plan.world_size = 4;
        fs.plan_ = &plan;
        StagedLayer L;
        L.layer = 12;
        L.first_op = 5;
        L.last_op = 6;
        fs.layers_ = { L };
        fs.hx_ops_.assign(1, FileStager::HxOps{});
        fs.hx_ops_[0].gu = 5;
        fs.hx_pool_ = pool;
        /* Weights: [1] the layer's router, [2] another layer's, [3] another weight of the layer. */
        prog.weights.resize(4);
        prog.weights[1].name = "blk.12.glm_router.weight";
        prog.weights[2].name = "blk.13.glm_router.weight";
        prog.weights[3].name = "blk.12.attn_q_a.weight";
        auto op = [&](const char* name, std::vector<rad_weight> w, long long N) {
            OpInfo o;
            o.op = name;
            o.weights = std::move(w);
            if (N) o.base.set_i("N", N);
            prog.ops.push_back(std::move(o));
        };
        op("prefetch", { 1 }, 0);              /* 0: an on-die read of the router */
        op("gemm_nt_f32", { 1 }, 96);          /* 1: the TP-split pass's part */
        op("gemm_nt_f32", { 1 }, 384);         /* 2: the router GEMV */
        op("gemm_nt_f32", { 3 }, 1536);        /* 3: another weight's gemm, wider */
        op("gemm_nt_f32", { 2 }, 384);         /* 4: the next layer's router (no routed layer here) */
        op("moe_gemm_ggml", {}, 4608);         /* 5: the gate_up */
        op("moe_gemm_ggml", {}, 5120);         /* 6: the down */
    }

    void pass(int64_t tok) {
        fs.pass_tok_ = tok;
        fs.routed_ = true;
        fs.pass_full_ = INT32_MAX;
        fs.hx_wake_li_ = -1;
    }

    void finds_the_router_gemv() {
        REQUIRE_OK(fs.hx_wake_init(prog));
        REQUIRE(fs.hx_wake_);
        REQUIRE(fs.hx_wake_ev_ != nullptr);
        REQUIRE_EQ(fs.hx_wake_op_.size(), prog.ops.size());
        for (size_t h = 0; h < prog.ops.size(); ++h) CHECK_EQ(fs.hx_wake_op_[h], h == 2 ? 0 : -1);
    }

    void records_before_the_router_in_one_token_passes() {
        REQUIRE_OK(fs.hx_wake_init(prog));
        RadStream s = nullptr;
        REQUIRE(rad_stream_create(&s, 0) >= 0);
        bool resync = false;
        pass(2);   /* a two-token pass: no event */
        CHECK_OK(fs.before_op(2, s, &resync));
        CHECK_EQ(fs.hx_wake_li_, -1);
        pass(1);
        CHECK_OK(fs.before_op(1, s, &resync));   /* the split part is not the router GEMV */
        CHECK_EQ(fs.hx_wake_li_, -1);
        CHECK_OK(fs.before_op(2, s, &resync));
        CHECK_EQ(fs.hx_wake_li_, 0);
        CHECK(!resync);
        /* A pass the consumer skip cut before the layer: no event. */
        pass(1);
        fs.pass_full_ = 12;
        CHECK_OK(fs.before_op(2, s, &resync));
        CHECK_EQ(fs.hx_wake_li_, -1);
        CHECK_OK(rad_stream_sync(s));
        rad_stream_destroy(s);
    }

    void waits_for_the_router_then_pre_wakes() {
        REQUIRE_OK(fs.hx_wake_init(prog));
        REQUIRE_OK(rad_event_create(&fs.counts_ev_));
        RadStream s = nullptr;
        REQUIRE(rad_stream_create(&s, 0) >= 0);
        bool resync = false;
        std::chrono::steady_clock::time_point woke{};
        /* Counts already in: no pre-wake, nothing waited for. */
        pass(1);
        CHECK_OK(fs.before_op(2, s, &resync));
        CHECK_OK(rad_event_record(fs.counts_ev_, s));
        CHECK_OK(rad_stream_sync(s));
        const HostExpertPool::Stats p0 = fs.hx_pool_->stats();
        CHECK_OK(fs.hx_wake_route(0, &woke));
        CHECK_EQ(fs.hx_stats_.wk_rt_ready, (uint64_t)1);
        CHECK_EQ(fs.hx_stats_.wk_rt, (uint64_t)0);
        CHECK_EQ(woke.time_since_epoch().count(), 0);
        CHECK_EQ(fs.hx_wake_li_, -1);   /* taken */
        /* Counts held behind a gate: the router's event (ahead of the gate) is waited for, then the
         * pool's pre-wake. */
        std::this_thread::sleep_for(std::chrono::milliseconds(50));   /* the workers asleep */
        pass(1);
        CHECK_OK(fs.before_op(2, s, &resync));
        void* gate = nullptr;
        const int g = rad_stream_gate(s, &gate);
        if (g == RAD_E_UNSUPPORTED) {
            CHECK_OK(rad_stream_sync(s));
            rad_stream_destroy(s);
            SKIP("no stream gates on this backend");
            return;
        }
        REQUIRE_OK(g);
        CHECK_OK(rad_event_record(fs.counts_ev_, s));
        CHECK_EQ(rad_event_query(fs.counts_ev_), 0);
        CHECK_OK(fs.hx_wake_route(0, &woke));
        CHECK_EQ(fs.hx_stats_.wk_rt, (uint64_t)1);
        CHECK(woke.time_since_epoch().count() != 0);
        const HostExpertPool::Stats p1 = fs.hx_pool_->stats();
        CHECK_EQ(p1.pre_calls[HostExpertPool::kWakeRouter] + p1.pre_skipped[HostExpertPool::kWakeRouter],
                 p0.pre_calls[HostExpertPool::kWakeRouter] + p0.pre_skipped[HostExpertPool::kWakeRouter] + 1);
        CHECK_EQ(rad_event_query(fs.counts_ev_), 0);   /* still behind the gate */
        CHECK_OK(rad_gate_open(gate));
        CHECK_OK(rad_event_sync(fs.counts_ev_));
        /* Another layer's decision does not take this layer's event. */
        pass(1);
        CHECK_OK(fs.before_op(2, s, &resync));
        CHECK_OK(fs.hx_wake_route(1, &woke));
        CHECK_EQ(fs.hx_stats_.wk_rt, (uint64_t)1);
        CHECK_EQ(fs.hx_wake_li_, 0);
        CHECK_OK(rad_stream_sync(s));
        rad_stream_destroy(s);
    }

    void times_a_miss_reads_pre_wake() {
        REQUIRE_OK(fs.hx_wake_init(prog));
        const int64_t lead = fs.hx_pool_->prewake_lead_ns();
        REQUIRE(lead > 0);
        CHECK_EQ(fs.hx_wake_at(1), (int64_t)-1);   /* nothing seen yet */
        Mover::PlaneReadTiming t;
        t.units = 1;
        /* Seven reads: not yet. The eighth: the median of the eight, less the lead. Landing is the
         * later leg's, the container's or the mirror's. */
        const int64_t land[8] = { 2'900'000, 2'700'000, 2'750'000, 5'500'000, 2'800'000, 2'650'000,
                                  2'720'000, 2'760'000 };
        for (int i = 0; i < 8; ++i) {
            CHECK_EQ(fs.hx_wake_at(1), (int64_t)-1);
            t.container_ns = i % 2 ? land[i] : land[i] - 100'000;
            t.mirror_ns = i % 2 ? land[i] - 200'000 : land[i];
            fs.hx_note_reads(t);
        }
        /* Sorted: 2.65 2.70 2.72 2.75 2.76 2.80 2.90 5.50; nth_element's n/2 = the fifth, 2.76 ms. */
        CHECK_EQ(fs.hx_wake_at(1), 2'760'000 - lead);
        CHECK_EQ(fs.hx_wake_at(3), (int64_t)-1);   /* reads of more units are timed apart */
        CHECK_EQ(fs.hx_wake_at(0), (int64_t)-1);
        /* The ring keeps the last 32: forty later reads of 3 ms move the median there. */
        for (int i = 0; i < 40; ++i) {
            t.container_ns = 3'000'000;
            t.mirror_ns = 0;
            fs.hx_note_reads(t);
        }
        CHECK_EQ(fs.hx_wake_at(1), 3'000'000 - lead);
        CHECK_EQ(fs.hx_stats_.mr[0].calls, (uint64_t)48);   /* the read counters as before */
    }
};

}  // namespace rad

using rad::EpCountsHostProbe;
using rad::HostExpertPool;

namespace {

/* Two workers with the pre-wake on; the silu table is zeros (no job runs here). */
HostExpertPool* pool() {
    static HostExpertPool* p = [] {
        const std::string path = radtest::tmpdir() + "/silu_zero.bin";
        FILE* f = std::fopen(path.c_str(), "wb");
        std::vector<float> z(65536, 0.0f);
        if (!f || std::fwrite(z.data(), 4, 65536, f) != 65536) return (HostExpertPool*)nullptr;
        std::fclose(f);
        setenv("RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS", "2", 1);
        setenv("RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_SILU", path.c_str(), 1);
        setenv("RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_PREWAKE", "2", 1);
        setenv("RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_PREWAKE_LEAD_US", "60", 1);
        return HostExpertPool::get();
    }();
    return p;
}

}  // namespace

TEST(the_router_gemv_is_the_widest_gemm_reading_the_layers_router) {
    HostExpertPool* p = pool();
    if (!p) { SKIP("no pool (no AVX2?)"); return; }
    EpCountsHostProbe e(p);
    e.finds_the_router_gemv();
}

TEST(the_router_event_is_recorded_before_the_router_in_one_token_passes_only) {
    HostExpertPool* p = pool();
    if (!p) { SKIP("no pool (no AVX2?)"); return; }
    EpCountsHostProbe e(p);
    e.records_before_the_router_in_one_token_passes();
}

TEST(the_decision_waits_for_the_router_then_pre_wakes_unless_the_counts_are_in) {
    HostExpertPool* p = pool();
    if (!p) { SKIP("no pool (no AVX2?)"); return; }
    EpCountsHostProbe e(p);
    e.waits_for_the_router_then_pre_wakes();
}

TEST(a_miss_reads_pre_wake_comes_at_the_median_landing_less_the_lead) {
    HostExpertPool* p = pool();
    if (!p) { SKIP("no pool (no AVX2?)"); return; }
    EpCountsHostProbe e(p);
    e.times_a_miss_reads_pre_wake();
}

RAD_TEST_MAIN()
