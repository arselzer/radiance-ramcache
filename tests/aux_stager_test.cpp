/* aux_stager_test.cpp -- the aux expert worker in the file stager and on the host backend
 * (core/place/aux_expert_worker.h, stager_aux.cpp; work/aux/SPEC.md §15.4), no card:
 *  - the pin set (device layer, host backend with five fake devices): pinned memory records every device
 *    unset, the set's four once it is set, alloc_pinned_on only its own; a different set after pinned memory
 *    exists is refused;
 *  - DeviceLane on the host backend with fake launch rows: the job's record, operands and copies; a step that
 *    fails leaves the job failed (wait answers < 0 without a query) and the lane dead; a late job (its queue held
 *    at a gate) is abandoned, keeps the lane busy until it completes, then backs off for the configured jobs;
 *  - the stager's hooks through its test friend and a fake lane, with real packed experts and the process's
 *    CPU pool: hx_submit compacts before the pool's submit (a count mismatch refuses with no job running and
 *    hx_drain then waits for none) and queues the lane's job after it; hx_finish places the rows on time, has
 *    the CPU compute them from the kept leases when the job is late, failed at submit or forced late (urgent
 *    jobs, rows equal to the CPU path's), keeps a job that finished while the pool took longer than the
 *    deadline, and in verify mode compares (one flipped bit: one row, one value; rows the CPU did not write
 *    are not compared); hx_drain abandons a job in flight and gives the leases back;
 *  - hx_choose's switch-off identity (no lane, a lane holding nothing, a lane declining) and its decline counts. */
#include "rad_test.h"

#include "rad_internal.h"
#include "device/device.h"
#include "place/host_expert_pool.h"
#include "place/stager.h"
#include "../tools/ds41-host-experts/ds41_host_experts.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#define REQUIRE_OK(expr) REQUIRE((expr) >= 0)

namespace {

/* Five fake devices, before the backend counts them (the first device call). */
const int g_devices = [] {
    ::setenv("RADIANCE_HOST_DEVICES", "5", 1);
    return 5;
}();

int64_t now_ns() {
    return (int64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

/* The process's CPU pool: two workers, the host's silu table at the trunk's limit. */
rad::HostExpertPool* pool() {
    static rad::HostExpertPool* p = []() -> rad::HostExpertPool* {
        if (!__builtin_cpu_supports("avx2")) return nullptr;
        const std::string path = radtest::tmpdir() + "/silu_aux.bin";
        std::vector<float> t(65536);
        ds41hx::silu_table_host(t.data(), 10.0f);
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f || std::fwrite(t.data(), 4, 65536, f) != 65536) return nullptr;
        std::fclose(f);
        ::setenv("RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS", "2", 1);
        ::setenv("RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_SILU", path.c_str(), 1);
        rad::HostExpertPool* got = rad::HostExpertPool::get();
        ::unsetenv("RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS");
        ::unsetenv("RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_SILU");
        return got;
    }();
    return p;
}

/* One expert in the packed RAM cache's form (scales as the release has them, no raw round). */
struct Packed {
    std::vector<uint8_t> gu, dn;
    bool make(uint32_t seed) {
        std::mt19937_64 rng(seed);
        auto one = [&](int N, int K, std::vector<uint8_t>* out) {
            std::vector<uint8_t> codes((size_t)N * (K / 2)), scales((size_t)N * (K / 32));
            for (auto& c : codes) c = (uint8_t)rng();
            for (auto& s : scales) s = (rng() % 100) < 58 ? 120 : 121;
            ds41hx::PackInfo plan;
            if (!ds41hx::pack_plan(scales.data(), N, K, &plan)) return false;
            const size_t bytes = ds41hx::packed_bytes(N, K, plan.n_exc + 4);
            out->assign(bytes + 64, 0);
            if (ds41hx::pack_planes(codes.data(), scales.data(), N, K, plan.n_exc + 4, plan, out->data(), bytes) !=
                ds41hx::kPackOk)
                return false;
            out->resize(bytes);
            ds41hx::Mx v;
            return ds41hx::packed_view(out->data(), bytes, N, K, &v);
        };
        return one(4608, 5120, &gu) && one(5120, 2304, &dn);
    }
};

/* A lane the tests drive: it records what it is asked and answers as told. */
struct FakeLane final : rad::aux::Lane {
    State st = kReady;
    int submit_rc = 0, wait_rc = 1;
    int submits = 0, waits = 0, abandons = 0, abandons_backoff = 0, declines = 0;
    int64_t last_deadline = 0, last_wait_at = 0;
    rad::aux::Job job;
    std::vector<int32_t> sorted, off;
    std::vector<uint16_t> out = std::vector<uint16_t>((size_t)ds41aux::kMaxSlots * ds41aux::kEmbd, 0x7777);
    State state() override { return st; }
    void declined() override { ++declines; }
    int submit(const rad::aux::Job& j) override {
        ++submits;
        job = j;
        sorted.assign(j.sorted, j.sorted + j.T);
        off.assign(j.off, j.off + ds41aux::kExperts + 1);
        return submit_rc;
    }
    int wait(int64_t deadline_ns) override {
        ++waits;
        last_deadline = deadline_ns;
        last_wait_at = now_ns();
        return wait_rc;
    }
    const uint16_t* rows() const override { return out.data(); }
    void abandon(bool backoff) override {
        ++abandons;
        abandons_backoff += backoff;
    }
    bool drain(int64_t) override { return true; }
    std::string why_dead() const override { return ""; }
};

}  // namespace

namespace rad {

/* FileStager's test friend (stager.h). One routed layer (model layer 3, gate_up op 1, SwiGLU op 2, down op 3)
 * with the CPU path's decode shape; units 0..95 are its local experts 0..95. */
struct AuxStagerProbe {
    Plan plan;
    Mover mover;
    FileStager fs;
    std::vector<int32_t> counts = std::vector<int32_t>(96, 0);
    aux::RankView view;
    std::vector<Packed> experts;
    std::vector<uint16_t> ref;   /* the CPU path's rows for the worker's slots, by compacted slot */

    AuxStagerProbe() {
        plan.rank = 0;
        plan.world_size = 4;
        plan.units.resize(96);
        for (int e = 0; e < 96; ++e) {
            plan.units[(size_t)e].expert = e;
            plan.units[(size_t)e].layer = 3;
        }
        fs.plan_ = &plan;
        fs.mover_ = &mover;
        fs.n_model_layers_ = 40;
        StagedLayer L;
        L.layer = 3;
        L.first_op = 1;
        L.last_op = 3;
        for (int e = 0; e < 96; ++e) L.units.push_back(e);
        fs.layers_ = { L };
        FileStager::HxOps o;
        o.gu = 1;
        o.sw = 2;
        o.n_expert = 96;
        o.top_k = 6;
        fs.hx_ops_ = { o };
        fs.ep_rows_ = 16;
        fs.hx_alloc_rows_ = 16;
        fs.hx_x_ = (uint16_t*)rad_dev_alloc(16 * 5120 * 2, RAD_MEM_HOST_PINNED);
        fs.hx_idx_ = (int32_t*)rad_dev_alloc((96 + 128) * 4, RAD_MEM_HOST_PINNED);
        fs.hx_out_ = (uint16_t*)rad_dev_alloc(96 * 5120 * 2, RAD_MEM_HOST_PINNED);
        fs.counts_host_ = counts.data();
        fs.counts_cap_ = 96;
        fs.hx_pool_ = pool();
        fs.hx_policy_ = FileStager::kHxAll;
        fs.hx_units_.reset(new Mover::HostUnit[FileStager::kHxUnits]);
        FileStager::AuxRows R;
        R.gu.assign(17, nullptr);
        R.sw.assign(97, nullptr);
        R.dn.assign(17, nullptr);
        R.ok = true;
        fs.aux_rows_ = { R };
        fs.aux_op_.assign(4, -1);
        view.resident.assign(96, 0);
        view.slot.assign(96, -1);
        view.gu_tab = { (const void*)0x1000 };
        view.dn_tab = { (const void*)0x2000 };
    }
    ~AuxStagerProbe() {
        fs.hx_drain();
        fs.aux_unbind();
        rad_dev_free(fs.hx_x_, RAD_MEM_HOST_PINNED);
        rad_dev_free(fs.hx_idx_, RAD_MEM_HOST_PINNED);
        rad_dev_free(fs.hx_out_, RAD_MEM_HOST_PINNED);
        fs.hx_x_ = nullptr;
        fs.hx_idx_ = nullptr;
        fs.hx_out_ = nullptr;
        fs.counts_host_ = nullptr;
        fs.counts_cap_ = 0;
        fs.hx_pool_ = nullptr;
        fs.mover_ = nullptr;
        fs.plan_ = nullptr;
    }

    /* A pass of AR rows: token t picks local experts e0..e5 = (base + t + 7k) mod 96 for k < picks (the rest go
     * to other ranks), as moe_scatter writes the sorted order and the counts. */
    void pass(int AR, int base, int picks) {
        fs.pass_tok_ = AR;
        fs.hx_li_ = 0;
        fs.hx_stage_ = 1;
        int32_t* const sorted = fs.hx_idx_;
        int32_t* const off = fs.hx_idx_ + fs.hx_sorted_max();
        std::vector<std::vector<int32_t>> by_e(96);
        for (int t = 0; t < AR; ++t)
            for (int k = 0; k < picks; ++k) by_e[(size_t)((base + t + 7 * k) % 96)].push_back(t * 6 + k);
        int at = 0;
        std::fill(counts.begin(), counts.end(), 0);
        for (int e = 0; e < 96; ++e) {
            off[e] = at;
            for (int32_t s : by_e[(size_t)e]) sorted[at++] = s;
            counts[(size_t)e] = (int32_t)by_e[(size_t)e].size();
        }
        off[96] = at;
        while (at < AR * 6) sorted[at++] = -1;
        std::mt19937 rng((uint32_t)(AR * 131 + base));
        for (int i = 0; i < AR * 5120; ++i) fs.hx_x_[i] = ds41hx::f32_to_bf16((float)((int)(rng() % 2001) - 1000) / 400.0f);
        std::memset(fs.hx_out_, 0xEE, 96 * 5120 * 2);
    }
    /* A lease as the packed RAM cache lends it, on experts_[k]'s bytes. */
    void lease(Mover::HostUnit* hu, int32_t u, const Packed& p) {
        hu->unit = u;
        hu->n = 2;
        hu->packed = true;
        hu->host[0] = p.gu.data();
        hu->host[1] = p.dn.data();
        hu->bytes[0] = HostExpertPool::kGuBytes;
        hu->bytes[1] = HostExpertPool::kDnBytes;
        hu->slot[0] = (int64_t)p.gu.size();
        hu->slot[1] = (int64_t)p.dn.size();
    }
    /* The worker's units: experts `w` resident and leased into aux_units_ (serve), as hx_choose leaves them. */
    void take(FakeLane* lane, const std::vector<int>& w, const aux::Config& cfg) {
        if (experts.size() < w.size()) experts.resize(w.size());
        for (size_t k = 0; k < w.size(); ++k)
            if (experts[k].gu.empty()) REQUIRE_OK(experts[k].make(1000 + (uint32_t)k) ? 0 : -1);
        for (int e : w) view.resident[(size_t)e] = 1;
        fs.aux_bind(lane, &view, cfg);
        fs.aux_take_ = true;
        fs.aux_list_.clear();
        fs.aux_n_ = 0;
        for (size_t k = 0; k < w.size(); ++k) {
            fs.aux_list_.push_back(w[k]);
            if (!cfg.verify) lease(&fs.aux_units_[fs.aux_n_++], w[k], experts[k]);
        }
    }
    /* The CPU path's rows for the compacted slots, through the pool directly (the reference). */
    bool reference() {
        const ds41aux::Compact& c = fs.aux_cmp_;
        ref.assign((size_t)c.T * 5120, 0);
        std::vector<HostExpertPool::Job> jobs;
        for (int j = 0; j < c.T; ++j) {
            int e = 0;
            while (e < 95 && c.off[e + 1] <= j) ++e;
            size_t k = 0;
            while (k < fs.aux_list_.size() && fs.aux_list_[k] != e) ++k;
            if (k >= experts.size()) return false;
            HostExpertPool::Job jb;
            jb.gu = experts[k].gu.data();
            jb.dn = experts[k].dn.data();
            jb.packed = true;
            jb.gu_bytes = (int64_t)experts[k].gu.size();
            jb.dn_bytes = (int64_t)experts[k].dn.size();
            jb.x = fs.hx_x_ + (int64_t)(c.sorted[j] / 6) * 5120;
            jb.y = ref.data() + (int64_t)j * 5120;
            jb.cost = HostExpertPool::job_cost(1);
            jobs.push_back(jb);
        }
        int64_t cost = 0;
        for (auto& jb : jobs) cost += jb.cost;
        HostExpertPool::Ticket t;
        fs.hx_pool_->reserve(cost);
        if (fs.hx_pool_->submit(jobs.data(), (int)jobs.size(), &t) < 0) return false;
        fs.hx_pool_->wait(&t);
        return true;
    }
    bool placed_rows_are_ref() const {
        const ds41aux::Compact& c = fs.aux_cmp_;
        for (int j = 0; j < c.T; ++j)
            if (std::memcmp(fs.hx_out_ + (int64_t)c.row_of[j] * 5120, ref.data() + (int64_t)j * 5120, 10240)) return false;
        return true;
    }
    /* hx_finish's order: the pool's ticket, then aux_finish. */
    int finish() {
        fs.hx_pool_->wait(&fs.hx_ticket_);
        return fs.aux_finish();
    }

    /* ---- the cases ---- */

    void serve_on_time_places_the_lanes_rows() {
        FakeLane lane;
        pass(4, 5, 4);
        take(&lane, { 5, 12, 20 }, aux::Config{});
        REQUIRE_OK(fs.hx_submit());
        CHECK_EQ(fs.hx_stage_, 2);
        CHECK_EQ(lane.submits, 1);
        const ds41aux::Compact& c = fs.aux_cmp_;
        CHECK_EQ(lane.job.AR, 4);
        CHECK_EQ(lane.job.T, c.T);
        CHECK(lane.job.x == fs.hx_x_);
        CHECK(lane.job.gu_w.data == (void*)0x1000 && lane.job.dn_w.data == (void*)0x2000);
        CHECK_EQ(lane.job.rank, 0);
        CHECK_EQ(lane.job.world, 4);
        CHECK_EQ(c.T, counts[5] + counts[12] + counts[20]);
        CHECK(lane.sorted == std::vector<int32_t>(c.sorted, c.sorted + c.T));
        /* hx_rows_: exactly the worker's sorted rows (no CPU unit here), sorted; none counted as the CPU's. */
        REQUIRE_EQ(fs.hx_rows_.size(), (size_t)c.T);
        CHECK(std::is_sorted(fs.hx_rows_.begin(), fs.hx_rows_.end()));
        CHECK_EQ(fs.hx_stats_.rows, (uint64_t)0);
        REQUIRE(reference());
        std::memcpy(lane.out.data(), ref.data(), ref.size() * 2);   /* the card's rows: the CPU path's */
        lane.wait_rc = 1;
        REQUIRE_OK(finish());
        CHECK(placed_rows_are_ref());
        CHECK_EQ(lane.waits, 1);
        CHECK_EQ(lane.abandons, 0);
        CHECK_EQ(fs.aux_stats_.placed, (uint64_t)c.T);
        CHECK_EQ(fs.aux_stats_.late, (uint64_t)0);
        CHECK_EQ(fs.aux_stats_.jobs, (uint64_t)1);
        CHECK_EQ(fs.aux_stats_.units, (uint64_t)3);
        CHECK_EQ(fs.aux_stats_.rt_n, (uint64_t)1);
    }

    void a_late_job_falls_back() {
        FakeLane lane;
        pass(8, 9, 3);
        take(&lane, { 9, 16 }, aux::Config{});
        REQUIRE_OK(fs.hx_submit());
        REQUIRE(reference());
        lane.wait_rc = 0;   /* still pending at the deadline: its rows (0x7777) are never read */
        REQUIRE_OK(finish());
        CHECK(placed_rows_are_ref());
        CHECK_EQ(lane.abandons, 1);
        CHECK_EQ(lane.abandons_backoff, 1);
        CHECK_EQ(fs.aux_stats_.late, (uint64_t)1);
        CHECK_EQ(fs.aux_stats_.placed, (uint64_t)0);
        CHECK_EQ(fs.aux_stats_.fallback_units, (uint64_t)2);
        CHECK_EQ(fs.aux_stats_.fallback_rows, (uint64_t)fs.aux_cmp_.T);
        CHECK(!fs.aux_fb_);
        CHECK_EQ(fs.aux_ticket_.left.load(), 0);
        bool urgent = !fs.aux_fb_jobs_.empty();
        for (const auto& jb : fs.aux_fb_jobs_) urgent = urgent && jb.urgent;
        CHECK(urgent);
        /* The fallback's rows are not the CPU's in the shutdown counts. */
        CHECK_EQ(fs.hx_stats_.rows, (uint64_t)0);
    }

    void a_failed_submit_falls_back() {
        FakeLane lane;
        pass(2, 30, 6);
        take(&lane, { 30, 37 }, aux::Config{});
        lane.submit_rc = RAD_E_DEVICE;   /* a step failed: never an error of hx_submit */
        REQUIRE_OK(fs.hx_submit());
        CHECK_EQ(fs.hx_stage_, 2);
        CHECK_EQ(fs.aux_job_, (int)FileStager::kAuxFailed);
        REQUIRE(reference());
        REQUIRE_OK(finish());
        CHECK_EQ(lane.waits, 0);   /* a failed job is never waited for */
        CHECK(placed_rows_are_ref());
        CHECK_EQ(fs.aux_stats_.failed, (uint64_t)1);
        CHECK_EQ(fs.aux_stats_.fallback_rows, (uint64_t)fs.aux_cmp_.T);
    }

    void a_forced_late_job_falls_back() {
        FakeLane lane;
        pass(3, 40, 5);
        aux::Config cfg;
        cfg.force_late = 1;   /* every job */
        take(&lane, { 40, 47 }, cfg);
        REQUIRE_OK(fs.hx_submit());
        CHECK_EQ(fs.aux_job_, (int)FileStager::kAuxForced);
        REQUIRE(reference());
        REQUIRE_OK(finish());
        CHECK_EQ(lane.waits, 0);
        CHECK_EQ(lane.abandons_backoff, 1);
        CHECK(placed_rows_are_ref());
        CHECK_EQ(fs.aux_stats_.forced_late, (uint64_t)1);
        CHECK_EQ(fs.aux_stats_.late, (uint64_t)1);
    }

    void a_pool_slower_than_the_deadline_keeps_a_finished_job() {
        FakeLane lane;
        pass(4, 50, 4);
        aux::Config cfg;
        cfg.deadline_us = 100;
        take(&lane, { 50, 57 }, cfg);
        REQUIRE_OK(fs.hx_submit());
        REQUIRE(reference());
        std::memcpy(lane.out.data(), ref.data(), ref.size() * 2);
        std::this_thread::sleep_for(std::chrono::milliseconds(3));   /* the pool took longer than the deadline */
        const int64_t t_pool = now_ns();
        lane.wait_rc = 1;
        REQUIRE_OK(finish());
        /* It was asked once, with a deadline after the pool's rows (not the submit's, long past). */
        CHECK_EQ(lane.waits, 1);
        CHECK(lane.last_deadline >= t_pool);
        CHECK(placed_rows_are_ref());
        CHECK_EQ(fs.aux_stats_.late, (uint64_t)0);
        CHECK_EQ(fs.aux_stats_.placed, (uint64_t)fs.aux_cmp_.T);
    }

    void verify_compares_and_counts() {
        FakeLane lane;
        pass(4, 60, 4);
        aux::Config cfg;
        cfg.verify = true;
        /* The worker holds 60, 67 and 74; the CPU computes 60 and 67 (its leases), not 74. */
        take(&lane, { 60, 67, 74 }, cfg);
        fs.hx_n_ = 0;
        lease(&fs.hx_units_[fs.hx_n_++], 60, experts[0]);
        lease(&fs.hx_units_[fs.hx_n_++], 67, experts[1]);
        REQUIRE_OK(fs.hx_submit());
        CHECK_EQ(lane.submits, 1);
        const ds41aux::Compact& c = fs.aux_cmp_;
        /* Verify appends nothing: hx_rows_ holds the CPU's rows only. */
        CHECK_EQ(fs.hx_rows_.size(), (size_t)(counts[60] + counts[67]));
        fs.hx_pool_->wait(&fs.hx_ticket_);
        /* The lane's rows: the CPU's, one bit flipped in one row of expert 67. */
        int flipped = -1;
        for (int j = 0; j < c.T; ++j) {
            std::memcpy(lane.out.data() + (int64_t)j * 5120, fs.hx_out_ + (int64_t)c.row_of[j] * 5120, 10240);
            if (flipped < 0 && j >= c.off[67] && j < c.off[68]) flipped = j;
        }
        REQUIRE(flipped >= 0);
        lane.out[(size_t)flipped * 5120 + 100] ^= 1;
        const std::vector<uint16_t> cpu(fs.hx_out_, fs.hx_out_ + 96 * 5120);
        REQUIRE_OK(fs.aux_finish());
        CHECK_EQ(fs.aux_stats_.verified, (uint64_t)(counts[60] + counts[67]));
        CHECK_EQ(fs.aux_stats_.differ_rows, (uint64_t)1);
        CHECK_EQ(fs.aux_stats_.differ_values, (uint64_t)1);
        CHECK_EQ(fs.aux_stats_.not_compared, (uint64_t)counts[74]);
        CHECK(std::memcmp(cpu.data(), fs.hx_out_, cpu.size() * 2) == 0);   /* the CPU's rows stay */
    }

    void a_count_mismatch_refuses_before_the_pool_submit() {
        FakeLane lane;
        pass(4, 70, 4);
        take(&lane, { 70, 77 }, aux::Config{});
        counts[77] += 1;   /* the scatter's count disagrees with the sorted order */
        CHECK_EQ(fs.hx_submit(), RAD_E_STATE);
        CHECK_EQ(fs.hx_stage_, 1);   /* nothing went to the pool */
        CHECK_EQ(fs.hx_ticket_.left.load(), 0);
        CHECK_EQ(lane.submits, 0);
        fs.hx_drain();               /* waits for nothing, gives the leases back */
        CHECK_EQ(fs.aux_n_, 0);
        CHECK_EQ(lane.abandons, 0);
        CHECK_EQ(fs.hx_stage_, 0);
    }

    void drain_abandons_a_job_in_flight() {
        FakeLane lane;
        pass(4, 80, 4);
        take(&lane, { 80, 87 }, aux::Config{});
        REQUIRE_OK(fs.hx_submit());
        CHECK_EQ(fs.aux_job_, (int)FileStager::kAuxPending);
        fs.hx_drain();
        CHECK_EQ(lane.abandons, 1);
        CHECK_EQ(lane.abandons_backoff, 0);   /* not late: a pass given up */
        CHECK_EQ(fs.aux_n_, 0);
        CHECK(fs.aux_list_.empty());
        CHECK_EQ(fs.aux_job_, (int)FileStager::kAuxNone);
        CHECK_EQ(fs.hx_stage_, 0);
    }

    /* hx_choose's state after one call, for comparing configurations. */
    struct Choice {
        std::vector<int32_t> slot, want;
        int n = 0, li = 0, stage = 0;
        uint64_t probed = 0, missed = 0, linked = 0;
        bool operator==(const Choice& o) const {
            return slot == o.slot && want == o.want && n == o.n && li == o.li && stage == o.stage &&
                   probed == o.probed && missed == o.missed && linked == o.linked;
        }
    };
    Choice choose(int64_t tok, const std::vector<int32_t>& want) {
        fs.pass_tok_ = tok;
        fs.hx_li_ = -1;
        fs.hx_stage_ = 0;
        fs.want_ = want;
        fs.hx_stats_ = FileStager::HostStats{};
        fs.hx_choose(0);
        Choice c;
        c.slot = fs.hx_slot_;
        c.want = fs.want_;
        c.n = fs.hx_n_;
        c.li = fs.hx_li_;
        c.stage = fs.hx_stage_;
        c.probed = fs.hx_stats_.probed;
        c.missed = fs.hx_stats_.missed;
        c.linked = fs.hx_stats_.linked;
        return c;
    }
    /* Without the RAM cache every selected unit is uncached (this mover has none), so the base sends them to
     * the copy link; the worker must change nothing of that whatever its state, and count what it declined. */
    void choose_identity_and_declines() {
        const std::vector<int32_t> want = { 3, 17, 40, 41, 90 };
        for (int e = 0; e < 96; ++e) counts[(size_t)e] = 0;
        for (int32_t u : want) counts[(size_t)u] = 1;
        const Choice base = choose(4, want);
        CHECK(base.want == want);
        CHECK_EQ(base.probed, (uint64_t)want.size());
        FakeLane lane;
        aux::Config cfg;
        /* A lane holding nothing. */
        fs.aux_bind(&lane, &view, cfg);
        fs.aux_pass_ = true;
        CHECK(choose(4, want) == base);
        CHECK_EQ(lane.submits, 0);
        /* Resident units, every way of declining, and a taken layer whose units are uncached. */
        view.resident[17] = view.resident[40] = 1;
        fs.aux_pass_ = false;   /* a draft pass */
        CHECK(choose(4, want) == base);
        fs.aux_pass_ = true;
        CHECK(choose(17, want) == base);   /* taller than 16 rows */
        lane.st = aux::Lane::kBusy;
        CHECK(choose(4, want) == base);
        lane.st = aux::Lane::kBackoff;
        CHECK(choose(4, want) == base);
        lane.st = aux::Lane::kDead;
        CHECK(choose(4, want) == base);
        lane.st = aux::Lane::kReady;
        fs.aux_rows_[0].ok = false;
        CHECK(choose(4, want) == base);
        fs.aux_rows_[0].ok = true;
        CHECK(choose(4, want) == base);   /* taken, but both uncached: the CPU's miss path as in the base */
        const aux::Stats& s = fs.aux_stats_;
        CHECK_EQ(s.declined_draft, (uint64_t)1);
        CHECK_EQ(s.declined_tall, (uint64_t)1);
        CHECK_EQ(s.declined_busy, (uint64_t)2);
        CHECK_EQ(lane.declines, 1);   /* the backoff counted down once */
        CHECK_EQ(s.declined_dead, (uint64_t)1);
        CHECK_EQ(s.declined_rows, (uint64_t)1);
        CHECK_EQ(s.uncached, (uint64_t)2);
        CHECK(!fs.aux_take_);
        CHECK_EQ(fs.aux_n_, 0);
        /* A layer without a resident selected unit counts nothing. */
        const aux::Stats before = s;
        lane.st = aux::Lane::kBusy;
        CHECK(choose(4, { 3, 90 }).probed == 2);
        CHECK_EQ(fs.aux_stats_.declined_busy, before.declined_busy);
        fs.aux_unbind();
        CHECK(choose(4, want) == base);
    }
};

}  // namespace rad

using rad::AuxStagerProbe;

TEST(pin_set_on_the_host_backend) {
    REQUIRE_EQ(rad_dev_count(), 5);
    void* p0 = rad_dev_alloc(4096, RAD_MEM_HOST_PINNED);
    REQUIRE(p0);
    CHECK_EQ(rad::host_pinned_devices(p0), (uint64_t)0x1f);   /* unset: every device */
    CHECK(!rad::dev_pin_restricted());
    CHECK_EQ(rad::dev_pin_devices({ 0, 1, 2, 3 }), RAD_E_STATE);   /* a pinned range exists, the set differs */
    CHECK_EQ(rad::dev_pin_devices({ 0, 1, 2, 3, 4 }), RAD_OK);    /* every device: no change */
    CHECK(!rad::dev_pin_restricted());
    rad_dev_free(p0, RAD_MEM_HOST_PINNED);
    CHECK_EQ(rad::dev_pin_devices({ 0, 9 }), RAD_E_INVAL);
    REQUIRE_EQ(rad::dev_pin_devices({ 0, 1, 2, 3 }), RAD_OK);
    CHECK(rad::dev_pin_restricted());
    void* p1 = rad_dev_alloc(4096, RAD_MEM_HOST_PINNED);
    void* p2 = rad::dev_alloc_pinned_on(4096, { 4 });
    REQUIRE(p1 && p2);
    CHECK_EQ(rad::host_pinned_devices(p1), (uint64_t)0x0f);
    CHECK_EQ(rad::host_pinned_devices(p2), (uint64_t)0x10);
    CHECK_EQ(rad::dev_pin_devices({ 3, 2, 1, 0 }), RAD_OK);   /* the same set again */
    CHECK_EQ(rad::dev_pin_devices({ 0, 1 }), RAD_E_STATE);
    void* d = rad_dev_alloc(4096, RAD_MEM_DEVICE);
    CHECK_EQ(rad::host_pinned_devices(d), (uint64_t)0);
    rad_dev_free(d, RAD_MEM_DEVICE);
    rad_dev_free(p1, RAD_MEM_HOST_PINNED);
    rad_dev_free(p2, RAD_MEM_HOST_PINNED);
    CHECK(!rad::dev_faulted(4));   /* the host backend has no faults to scope */
}

namespace {
struct LaunchLog {
    int calls[3] = {};
    int fail_at = -1;   /* 0 gate_up, 1 SwiGLU, 2 down: that launch returns an error */
    RadTensor t[3][5];
    int n_t[3] = {};
    const RadParam* p[3] = {};
    int rank = -1, world = -1;
} g_log;
int fake(int k, const RadArgs* a) {
    ++g_log.calls[k];
    g_log.n_t[k] = a->n_t;
    for (int i = 0; i < a->n_t && i < 5; ++i) g_log.t[k][i] = a->t[i];
    g_log.p[k] = a->p;
    g_log.rank = a->rank;
    g_log.world = a->world_size;
    if (g_log.fail_at == k) return RAD_E_DEVICE;
    if (k == 2) {   /* down: rows the D2H brings back */
        uint16_t* y = (uint16_t*)a->t[4].data;
        for (int64_t i = 0; i < a->t[4].shape[0] * a->t[4].shape[1]; ++i) y[i] = (uint16_t)(0x3c00 + i % 977);
    }
    return RAD_OK;
}
int fake_gu(const RadArgs* a, RadStream) { return fake(0, a); }
int fake_sw(const RadArgs* a, RadStream) { return fake(1, a); }
int fake_dn(const RadArgs* a, RadStream) { return fake(2, a); }
}  // namespace

TEST(device_lane_on_the_host_backend) {
    const ds41aux::Ws w = ds41aux::ws();
    std::vector<char> ws((size_t)w.bytes, 0);
    rad::aux::DeviceLane lane(4, 1, ws.data(), 3);
    std::string why;
    REQUIRE_OK(lane.init(&why));
    CHECK_EQ(rad::host_pinned_devices(lane.rows()), (uint64_t)0x10);   /* the extra card's only */
    CHECK_EQ(lane.state(), rad::aux::Lane::kReady);
    /* A job of 2 rows and 7 slots. */
    std::vector<uint16_t> x(2 * 5120);
    for (size_t i = 0; i < x.size(); ++i) x[i] = (uint16_t)(i * 3 + 1);
    int32_t sorted[7] = { 0, 6, 1, 7, 2, 8, 3 }, off[97];
    for (int e = 0; e <= 96; ++e) off[e] = e <= 10 ? 0 : 7;
    const RadParam params[1] = {};
    rad::aux::Job j;
    j.AR = 2;
    j.T = 7;
    j.x = x.data();
    j.sorted = sorted;
    j.off = off;
    j.gu = { fake_gu, params, 1, nullptr };
    j.sw = { fake_sw, params, 1, nullptr };
    j.dn = { fake_dn, params, 1, nullptr };
    j.gu_w.data = (void*)0x1000;
    j.gu_w.rank = 3;
    j.dn_w.data = (void*)0x2000;
    j.dn_w.rank = 3;
    j.rank = 1;
    j.world = 4;
    g_log = LaunchLog{};
    REQUIRE_OK(lane.submit(j));
    CHECK_EQ(lane.state(), rad::aux::Lane::kBusy);
    REQUIRE_EQ(lane.wait(now_ns() + 2'000'000'000LL), 1);
    /* The record landed in `in`; the operands are the workspace's; the rows came back. */
    CHECK(std::memcmp(ws.data() + w.sorted, sorted, sizeof sorted) == 0);
    CHECK(std::memcmp(ws.data() + w.off, off, sizeof off) == 0);
    CHECK(std::memcmp(ws.data() + w.x, x.data(), x.size() * 2) == 0);
    CHECK_EQ(g_log.calls[0] + g_log.calls[1] + g_log.calls[2], 3);
    CHECK_EQ(g_log.n_t[0], 5);
    CHECK_EQ(g_log.n_t[1], 2);
    CHECK_EQ(g_log.n_t[2], 5);
    CHECK(g_log.t[0][0].data == ws.data() + w.x && g_log.t[0][0].shape[0] == 2 && g_log.t[0][0].shape[1] == 5120);
    CHECK(g_log.t[0][1].data == (void*)0x1000);
    CHECK(g_log.t[0][2].data == ws.data() + w.sorted && g_log.t[0][2].shape[0] == 7 && g_log.t[0][2].dtype == RAD_I32);
    CHECK(g_log.t[0][3].data == ws.data() + w.off && g_log.t[0][3].shape[0] == 97);
    CHECK(g_log.t[0][4].data == ws.data() + w.gu_out && g_log.t[0][4].shape[0] == 7 && g_log.t[0][4].shape[1] == 4608);
    CHECK(g_log.t[1][0].data == ws.data() + w.gu_out && g_log.t[1][1].data == ws.data() + w.act);
    CHECK(g_log.t[1][1].shape[0] == 7 && g_log.t[1][1].shape[1] == 2304);
    CHECK(g_log.t[2][0].data == ws.data() + w.act && g_log.t[2][1].data == (void*)0x2000);
    CHECK(g_log.t[2][4].data == ws.data() + w.dn_out && g_log.t[2][4].shape[1] == 5120);
    CHECK(g_log.p[0] == params && g_log.rank == 1 && g_log.world == 4);
    bool rows_ok = true;
    for (int64_t i = 0; i < 7 * 5120; ++i) rows_ok = rows_ok && lane.rows()[i] == (uint16_t)(0x3c00 + i % 977);
    CHECK(rows_ok);
    CHECK_EQ(lane.state(), rad::aux::Lane::kReady);
    /* Late: the queue held at a gate. The job stays in flight past the deadline, the lane busy after the
     * abandon until it completes, then it declines three jobs. */
    void* gate = nullptr;
    REQUIRE_OK(rad_stream_gate(lane.stream(), &gate));
    REQUIRE_OK(lane.submit(j));
    CHECK_EQ(lane.wait(now_ns() + 2'000'000LL), 0);
    lane.abandon(true);
    CHECK_EQ(lane.state(), rad::aux::Lane::kBusy);
    REQUIRE_OK(rad_gate_open(gate));
    const int64_t end = now_ns() + 2'000'000'000LL;
    while (lane.state() == rad::aux::Lane::kBusy && now_ns() < end) std::this_thread::sleep_for(std::chrono::microseconds(100));
    CHECK_EQ(lane.state(), rad::aux::Lane::kBackoff);
    lane.declined();
    lane.declined();
    CHECK_EQ(lane.state(), rad::aux::Lane::kBackoff);
    lane.declined();
    CHECK_EQ(lane.state(), rad::aux::Lane::kReady);
    /* An abandon without backoff (a pass given up): ready once it completes. */
    REQUIRE_OK(lane.submit(j));
    lane.abandon(false);
    const int64_t end2 = now_ns() + 2'000'000'000LL;
    while (lane.state() == rad::aux::Lane::kBusy && now_ns() < end2) std::this_thread::sleep_for(std::chrono::microseconds(100));
    CHECK_EQ(lane.state(), rad::aux::Lane::kReady);
    CHECK(lane.drain(1'000'000'000LL));
}

TEST(device_lane_a_failed_step_fails_the_job_and_retires_the_lane) {
    for (int at = 0; at < 3; ++at) {
        const ds41aux::Ws w = ds41aux::ws();
        std::vector<char> ws((size_t)w.bytes, 0);
        rad::aux::DeviceLane lane(4, 2, ws.data(), 64);
        std::string why;
        REQUIRE_OK(lane.init(&why));
        std::vector<uint16_t> x(5120, 1);
        int32_t sorted[1] = { 0 }, off[97] = {};
        for (int e = 1; e <= 96; ++e) off[e] = 1;
        rad::aux::Job j;
        j.AR = 1;
        j.T = 1;
        j.x = x.data();
        j.sorted = sorted;
        j.off = off;
        j.gu = { fake_gu, nullptr, 0, nullptr };
        j.sw = { fake_sw, nullptr, 0, nullptr };
        j.dn = { fake_dn, nullptr, 0, nullptr };
        j.gu_w.data = (void*)0x1000;
        j.dn_w.data = (void*)0x2000;
        g_log = LaunchLog{};
        g_log.fail_at = at;
        CHECK(lane.submit(j) < 0);
        CHECK(lane.dead());
        CHECK(has(lane.why_dead(), at == 0 ? "gate_up" : at == 1 ? "SwiGLU" : "down"));
        CHECK(lane.wait(now_ns() + 1'000'000'000LL) < 0);   /* no query: the job is failed */
        CHECK_EQ(lane.state(), rad::aux::Lane::kDead);
        CHECK(lane.submit(j) < 0);
        CHECK(!lane.drain(1'000'000LL));
        lane.leak();   /* a dead lane is left to the process exit, as the worker leaves it */
    }
    /* A job outside the worker's shape is refused before anything is queued. */
    const ds41aux::Ws w = ds41aux::ws();
    std::vector<char> ws((size_t)w.bytes, 0);
    rad::aux::DeviceLane lane(4, 3, ws.data(), 64);
    std::string why;
    REQUIRE_OK(lane.init(&why));
    rad::aux::Job j;
    j.AR = 17;
    CHECK(lane.submit(j) < 0);
    CHECK(lane.dead());
    lane.leak();
}

TEST(serve_on_time_places_the_lanes_rows) {
    if (!pool()) SKIP("no AVX2 or no CPU pool");
    AuxStagerProbe p;
    p.serve_on_time_places_the_lanes_rows();
}

TEST(a_late_job_falls_back_to_the_cpu_from_the_kept_leases) {
    if (!pool()) SKIP("no AVX2 or no CPU pool");
    AuxStagerProbe p;
    p.a_late_job_falls_back();
}

TEST(a_failed_submit_falls_back_and_is_never_an_error) {
    if (!pool()) SKIP("no AVX2 or no CPU pool");
    AuxStagerProbe p;
    p.a_failed_submit_falls_back();
}

TEST(a_forced_late_job_falls_back) {
    if (!pool()) SKIP("no AVX2 or no CPU pool");
    AuxStagerProbe p;
    p.a_forced_late_job_falls_back();
}

TEST(a_pool_slower_than_the_deadline_keeps_a_finished_job) {
    if (!pool()) SKIP("no AVX2 or no CPU pool");
    AuxStagerProbe p;
    p.a_pool_slower_than_the_deadline_keeps_a_finished_job();
}

TEST(verify_compares_only_the_cpus_rows_and_counts_differences) {
    if (!pool()) SKIP("no AVX2 or no CPU pool");
    AuxStagerProbe p;
    p.verify_compares_and_counts();
}

TEST(a_count_mismatch_refuses_before_any_job_runs) {
    if (!pool()) SKIP("no AVX2 or no CPU pool");
    AuxStagerProbe p;
    p.a_count_mismatch_refuses_before_the_pool_submit();
}

TEST(hx_drain_abandons_a_job_in_flight) {
    if (!pool()) SKIP("no AVX2 or no CPU pool");
    AuxStagerProbe p;
    p.drain_abandons_a_job_in_flight();
}

TEST(hx_choose_switch_off_identity_and_declines) {
    if (!pool()) SKIP("no AVX2 or no CPU pool");
    AuxStagerProbe p;
    p.choose_identity_and_declines();
}

RAD_TEST_MAIN()
