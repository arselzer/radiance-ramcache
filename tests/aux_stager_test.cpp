/* aux_stager_test.cpp -- the aux expert worker in the file stager and on the host backend
 * (core/place/aux_expert_worker.h, stager_aux.cpp; work/aux/SPEC.md §15.4), no card:
 *  - the pin set (device layer, host backend with five fake devices): pinned memory records every device
 *    unset, the set's four once it is set, alloc_pinned_on only its own; a different set after pinned memory
 *    exists is refused;
 *  - DeviceLane on the host backend with fake launch rows: the job's record, operands and copies; a step that
 *    fails leaves the job failed (wait answers < 0 without a query) and the lane dead; a late job (its queue held
 *    at a gate) is abandoned, keeps the lane busy until it completes, then backs off for the configured jobs;
 *    teardown (work/aux/SPEC.md §7.5): a loader wait that timed out leaves its copies the lane's work (not
 *    drained until they complete), a failed record retires the lane, and an idle lane whose queue stopped is not
 *    drained either (the probe behind everything queued), each within its bound;
 *  - FileStager::aux_load and Worker::load_tables on the host backend's fifth device, from a packed RAM cache
 *    (Mover::borrow_host_unit's test hook): loaded, read back and their staging freed with the queue moving;
 *    with it held, each wait times out, the staging its copies use is not freed, the lane is not drained and the
 *    worker leaves the lanes and the arena to the process exit;
 *  - the stager's hooks through its test friend and a fake lane, with real packed experts and the process's
 *    CPU pool: hx_submit compacts before the pool's submit (a count mismatch refuses with no job running and
 *    hx_drain then waits for none) and queues the lane's job after it; hx_finish places the rows on time, has
 *    the CPU compute them from the kept leases when the job is late, failed at submit or forced late (urgent
 *    jobs, rows equal to the CPU path's), keeps a job that finished while the pool took longer than the
 *    deadline, and in verify mode compares (one flipped bit: one row, one value; rows the CPU did not write
 *    are not compared); hx_drain abandons a job in flight and gives the leases back;
 *  - hx_choose's switch-off identity (no lane, a lane holding nothing, a lane declining) and its decline counts;
 *  - hx_choose -> hx_submit -> hx_finish in serve mode with that cache: a resident cached unit leased into
 *    aux_units_ (slot code -3) with the base's admit, no pool reservation, kept off want_ and among the units
 *    start_ep_routed aliases (hx_cpu_units); a resident unit not cached left to the CPU's way; a layer of worker
 *    units only still runs (the job, the aliases); the gather's rows equal the same pass's without the worker;
 *  - the dynamic worker (work/aux-dyn/SPEC.md §7.4): with its switches unset no gate, no count, no hit array; a
 *    shut gate declines without asking the lane (declined_released, the base's rows), an open one is held from
 *    the decision to hx_finish (and hx_drain) and left; the online count (rows by candidate, passes the worker can
 *    take only, whatever the lane) and the hand-off (two in a row, another size refused); flips between passes. */
#include "rad_test.h"

#include "rad_internal.h"
#include "device/device.h"
#include "place/host_expert_pool.h"
#include "place/stager.h"
#include "../tools/ds41-host-experts/ds41_host_experts.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <deque>
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
    int submits = 0, waits = 0, abandons = 0, abandons_backoff = 0, declines = 0, states = 0;
    int64_t last_deadline = 0, last_wait_at = 0;
    rad::aux::Job job;
    std::vector<int32_t> sorted, off;
    std::vector<uint16_t> x;   /* the job's token rows, copied (its pointer may not outlive the call) */
    std::vector<uint16_t> out = std::vector<uint16_t>((size_t)ds41aux::kMaxSlots * ds41aux::kEmbd, 0x7777);
    /* A taller job (RADIANCE_EXPERIMENTAL_DS41_AUX_ROWS) gets room for its rows, 0x7777 too. */
    void room(int T) {
        if (out.size() < (size_t)T * ds41aux::kEmbd) out.resize((size_t)T * ds41aux::kEmbd, 0x7777);
    }
    State state() override {
        ++states;
        return st;
    }
    void declined() override { ++declines; }
    int submit(const rad::aux::Job& j) override {
        ++submits;
        room(j.T);
        job = j;
        sorted.assign(j.sorted, j.sorted + j.T);
        off.assign(j.off, j.off + ds41aux::kExperts + 1);
        x.assign(j.x, j.x + (size_t)j.AR * ds41aux::kEmbd);
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

    const int R;   /* the CPU path's rows (ep_rows_): 16, or a taller probe's (RADIANCE_EXPERIMENTAL_DS41_AUX_ROWS) */
    explicit AuxStagerProbe(int rows = 16) : R(rows) {
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
        fs.ep_rows_ = R;
        fs.hx_alloc_rows_ = R;
        fs.hx_x_ = (uint16_t*)rad_dev_alloc((int64_t)R * 5120 * 2, RAD_MEM_HOST_PINNED);
        fs.hx_idx_ = (int32_t*)rad_dev_alloc(((int64_t)R * 6 + 128) * 4, RAD_MEM_HOST_PINNED);
        fs.hx_out_ = (uint16_t*)rad_dev_alloc((int64_t)R * 6 * 5120 * 2, RAD_MEM_HOST_PINNED);
        fs.counts_host_ = counts.data();
        fs.counts_cap_ = 96;
        fs.hx_pool_ = pool();
        fs.hx_policy_ = FileStager::kHxAll;
        fs.hx_units_.reset(new Mover::HostUnit[FileStager::kHxUnits]);
        FileStager::AuxRows AR;
        AR.gu.assign((size_t)R + 1, nullptr);
        AR.sw.assign((size_t)R * 6 + 1, nullptr);
        AR.dn.assign((size_t)R + 1, nullptr);
        AR.ok = true;
        fs.aux_rows_ = { AR };
        fs.aux_op_.assign(4, -1);
        view.resident.assign(96, 0);
        view.slot.assign(96, -1);
        view.gu_tab = { (const void*)0x1000 };
        view.dn_tab = { (const void*)0x2000 };
    }
    ~AuxStagerProbe() {
        fs.hx_drain();
        fs.aux_unbind();
        mover.borrow_hook_ = {};
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
        std::memset(fs.hx_out_, 0xEE, (size_t)R * 6 * 5120 * 2);
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
    /* The packed RAM cache as hx_choose and aux_load borrow from it (Mover::borrow_host_unit's test hook): the
     * units in `units` lent on lent[k]'s bytes, every other unit not cached. */
    struct Cache {
        AuxStagerProbe* p = nullptr;
        std::vector<int32_t> units;
        std::vector<const Packed*> lent;
        int borrows = 0, touched = 0;
    } cache;
    static bool cache_borrow(void* ctx, int32_t u, Mover::HostUnit* hu, bool touch) {
        Cache* c = (Cache*)ctx;
        if (!hu || hu->n) return false;
        for (size_t k = 0; k < c->units.size(); ++k)
            if (c->units[k] == u) {
                ++c->borrows;
                c->touched += touch;
                c->p->lease(hu, u, *c->lent[k]);
                return true;
            }
        return false;
    }
    void cache_on(const std::vector<int32_t>& units, const std::vector<const Packed*>& lent) {
        cache.p = this;
        cache.units = units;
        cache.lent = lent;
        cache.borrows = cache.touched = 0;
        mover.borrow_hook_.context = &cache;
        mover.borrow_hook_.borrow = &cache_borrow;
    }
    /* The CPU path's rows of the compacted slots (aux_cmp_) into `y`, each from its unit's bytes in the cache
     * (unit u is expert u here), through the pool. */
    bool cpu_rows_of_compacted(uint16_t* y) {
        const ds41aux::Compact& c = fs.aux_cmp_;
        std::vector<HostExpertPool::Job> jobs;
        for (int j = 0; j < c.T; ++j) {
            int e = 0;
            while (e < 95 && c.off[e + 1] <= j) ++e;
            const Packed* p = nullptr;
            for (size_t k = 0; k < cache.units.size(); ++k)
                if (cache.units[k] == e) p = cache.lent[k];
            if (!p) return false;
            HostExpertPool::Job jb;
            jb.gu = p->gu.data();
            jb.dn = p->dn.data();
            jb.packed = true;
            jb.gu_bytes = (int64_t)p->gu.size();
            jb.dn_bytes = (int64_t)p->dn.size();
            jb.x = fs.hx_x_ + (int64_t)(c.sorted[j] / 6) * 5120;
            jb.y = y + (int64_t)j * 5120;
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

    /* RADIANCE_EXPERIMENTAL_DS41_AUX_ROWS (a probe of R rows, the switch at R): a pass of 17..R rows is compacted
     * and queued as a 16-row one is, the lane's rows placed at their sorted rows, and counted by height. */
    void tall_serve_places_the_lanes_rows(int AR) {
        FakeLane lane;
        pass(AR, 3, 5);
        aux::Config cfg;
        cfg.rows = R;
        const std::vector<int> w = { 3, 10, 17, 24, 50 };
        take(&lane, w, cfg);
        REQUIRE_OK(fs.hx_submit());
        const ds41aux::Compact& c = fs.aux_cmp_;
        CHECK_EQ(lane.submits, 1);
        CHECK_EQ(lane.job.AR, AR);
        CHECK_EQ(lane.job.T, c.T);
        int want = 0;
        for (int e : w) want += counts[(size_t)e];
        CHECK_EQ(c.T, want);
        REQUIRE(c.T > 0);
        REQUIRE_EQ(fs.hx_rows_.size(), (size_t)c.T);
        CHECK(lane.sorted == std::vector<int32_t>(c.sorted, c.sorted + c.T));
        REQUIRE(reference());
        std::memcpy(lane.out.data(), ref.data(), ref.size() * 2);
        lane.wait_rc = 1;
        REQUIRE_OK(finish());
        CHECK(placed_rows_are_ref());
        CHECK_EQ(fs.aux_stats_.placed, (uint64_t)c.T);
        CHECK_EQ(fs.aux_stats_.late, (uint64_t)0);
        const int b = ds41aux::height_bucket(AR);
        for (int k = 0; k < ds41aux::kHeightBuckets; ++k) {
            CHECK_EQ(fs.aux_stats_.jobs_h[k], (uint64_t)(k == b ? 1 : 0));
            CHECK_EQ(fs.aux_stats_.rows_h[k], (uint64_t)(k == b ? c.T : 0));
        }
    }
    /* A tall job still pending is asked with its submit + D * ceil(AR / 16) at the earliest, then its units go to
     * the CPU from the kept leases (the rows the CPU path's). */
    void tall_late_job_has_a_scaled_deadline_and_falls_back(int AR) {
        FakeLane lane;
        pass(AR, 7, 4);
        aux::Config cfg;
        cfg.rows = R;
        cfg.deadline_us = 1000;
        take(&lane, { 7, 14, 21 }, cfg);
        REQUIRE_OK(fs.hx_submit());
        REQUIRE(reference());
        lane.wait_rc = 0;
        const int64_t t0 = fs.aux_t0_ns_;
        REQUIRE_OK(finish());
        CHECK(lane.last_deadline >= t0 + ds41aux::deadline_us_for(1000, AR) * 1000);
        CHECK(ds41aux::deadline_us_for(1000, AR) > 1000);
        CHECK(placed_rows_are_ref());
        CHECK_EQ(fs.aux_stats_.late, (uint64_t)1);
        CHECK_EQ(lane.abandons_backoff, 1);
        CHECK_EQ(fs.aux_stats_.fallback_rows, (uint64_t)fs.aux_cmp_.T);
    }
    /* The bound: a pass of R rows is the worker's to take, R + 1 is declined as taller; with the switch at 16 (the
     * default) the same R-row pass is declined. */
    void tall_bound(int AR) {
        const std::vector<int32_t> want = { 3, 17, 40 };
        for (int e = 0; e < 96; ++e) counts[(size_t)e] = 0;
        for (int32_t u : want) counts[(size_t)u] = 1;
        view.resident[17] = 1;
        FakeLane lane;
        aux::Config cfg;
        cfg.rows = R;
        fs.aux_bind(&lane, &view, cfg);
        fs.aux_pass_ = true;
        (void)choose(AR, want);
        CHECK_EQ(fs.aux_stats_.declined_tall, (uint64_t)0);
        CHECK_EQ(fs.aux_stats_.uncached, (uint64_t)1);   /* taken, its unit not cached: the CPU's miss path */
        CHECK(lane.states >= 1);
        (void)choose(R + 1, want);   /* taller than the bound */
        CHECK_EQ(fs.aux_stats_.declined_tall, (uint64_t)1);
        fs.aux_unbind();
        fs.aux_bind(&lane, &view, aux::Config{});
        fs.aux_pass_ = true;
        (void)choose(AR, want);
        CHECK_EQ(fs.aux_stats_.declined_tall, (uint64_t)1);   /* a fresh bind: the default 16 declines it */
        fs.aux_unbind();
        view.resident[17] = 0;
    }

    /* The warm-up's tall job (ds41aux::kRowsEnv above 16): R token rows with one slot each on the unit's expert,
     * through the rows at R; refused when a row differs from the CPU pool's for the same lease, taken when they are
     * the pool's. */
    void tall_warm_checks_the_lanes_rows() {
        if (experts.empty()) experts.resize(1);
        if (experts[0].gu.empty()) REQUIRE_OK(experts[0].make(3000) ? 0 : -1);
        FakeLane lane;
        Mover::HostUnit hu;
        lease(&hu, 5, experts[0]);
        fs.aux_rows_max_ = R;
        std::string why;
        CHECK_EQ(fs.aux_warm_tall(&lane, 5, 0, &hu, (const void*)0x1000, (const void*)0x2000, 0, &why), RAD_E_STATE);
        CHECK(has(why, fmt("tall warm-up job (%d rows, unit 5, layer 3 expert 5)", R)));
        CHECK(has(why, "unlike the CPU pool's"));
        REQUIRE_EQ(lane.submits, 1);
        CHECK_EQ(lane.job.AR, R);
        CHECK_EQ(lane.job.T, R);
        CHECK(lane.job.gu_w.data == (void*)0x1000 && lane.job.dn_w.data == (void*)0x2000);
        bool order = (int)lane.sorted.size() == R;
        for (int i = 0; order && i < R; ++i) order = lane.sorted[(size_t)i] == i * 6;
        CHECK(order);
        CHECK(lane.off[5] == 0 && lane.off[6] == R && lane.off[96] == R);
        /* The pool's rows for the job's own token rows: taken. */
        std::vector<HostExpertPool::Job> jobs((size_t)R);
        int64_t cost = 0;
        for (int i = 0; i < R; ++i) {
            HostExpertPool::Job& jb = jobs[(size_t)i];
            jb.gu = experts[0].gu.data();
            jb.dn = experts[0].dn.data();
            jb.packed = true;
            jb.gu_bytes = (int64_t)experts[0].gu.size();
            jb.dn_bytes = (int64_t)experts[0].dn.size();
            jb.x = lane.x.data() + (int64_t)i * 5120;
            jb.y = lane.out.data() + (int64_t)i * 5120;
            jb.cost = HostExpertPool::job_cost(1);
            cost += jb.cost;
        }
        HostExpertPool::Ticket t;
        fs.hx_pool_->reserve(cost);
        REQUIRE_OK(fs.hx_pool_->submit(jobs.data(), R, &t));
        fs.hx_pool_->wait(&t);
        why.clear();
        CHECK_EQ(fs.aux_warm_tall(&lane, 5, 0, &hu, (const void*)0x1000, (const void*)0x2000, 0, &why), RAD_OK);
        CHECK(why.empty());
        lane.out[(size_t)(R - 1) * 5120 + 7] ^= 1;   /* the last row, one bit */
        CHECK_EQ(fs.aux_warm_tall(&lane, 5, 0, &hu, (const void*)0x1000, (const void*)0x2000, 0, &why), RAD_E_STATE);
        CHECK(has(why, fmt("row %d has 1 of 5120 values", R - 1)));
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

    /* One layer through hx_choose -> hx_submit -> hx_finish over the cache, with the worker bound (`lane`) or
     * not; the gather's operand y (24 rows, refilled first). What the case compares is left in the fields. */
    struct Layer {
        std::vector<int32_t> slot, want, keep, rows, aux_list, cpu_units;
        int64_t reserved = 0;
        int n = 0, aux_n = 0, stage = 0;
        uint64_t uncached = 0;
        std::vector<uint16_t> out, y;
    };
    void layer(const std::vector<int32_t>& want, FakeLane* lane, RadStream s, uint16_t* y, Layer* out) {
        Layer& L = *out;
        L = Layer{};
        pass(4, 5, 4);   /* tokens 0..3 pick experts 5 + t + 7k, k < 4: 5..8, 12..15, 19..22, 26..29, a row each */
        fs.hx_li_ = -1;
        fs.hx_stage_ = 0;
        fs.want_ = want;
        fs.hx_stats_ = FileStager::HostStats{};
        fs.aux_stats_ = aux::Stats{};
        if (lane) {
            fs.aux_bind(lane, &view, aux::Config{});
            fs.aux_pass_ = true;
        }
        fs.hx_choose(0);
        L.slot = fs.hx_slot_;
        L.want = fs.want_;
        L.reserved = fs.hx_reserved_;
        L.n = fs.hx_n_;
        L.aux_n = fs.aux_n_;
        L.aux_list = fs.aux_list_;
        L.stage = fs.hx_stage_;
        L.uncached = fs.aux_stats_.uncached;
        for (int k = 0; k < fs.hx_n_; ++k) L.cpu_units.push_back(fs.hx_units_[k].unit);
        if (fs.hx_stage_ != 1) return;
        fs.hx_cpu_units();   /* what start_ep_routed aliases */
        L.keep = fs.hx_keep_;
        REQUIRE_OK(fs.hx_submit());
        if (lane) REQUIRE(cpu_rows_of_compacted(lane->out.data()));   /* the card's rows: the CPU path's */
        L.rows = fs.hx_rows_;
        for (int i = 0; i < 24 * 5120; ++i) y[i] = 0x5555;
        fs.hx_confirmed_ = true;   /* hx_capture saw the gate_up read the rows */
        RadTensor Y{};
        Y.data = y;
        Y.dtype = RAD_BF16;
        Y.rank = 2;
        Y.shape[0] = 24;
        Y.shape[1] = 5120;
        Y.stride[0] = 5120;
        Y.stride[1] = 1;
        REQUIRE_OK(fs.hx_finish(s, &Y, 1));
        REQUIRE_OK(rad_stream_sync(s));
        L.out.assign(fs.hx_out_, fs.hx_out_ + 24 * 5120);
        L.y.assign(y, y + 24 * 5120);
        if (lane) fs.aux_unbind();
    }
    void serve_leases_through_hx_choose() {
        const std::vector<int32_t> sel = { 5, 6, 7, 8, 12, 13, 14, 15, 19, 20, 21, 22, 26, 27, 28, 29 };
        /* 5 and 13 cached; 12 and 20 cached and the worker's; 27 the worker's, not cached. */
        if (experts.size() < 4) experts.resize(4);
        for (size_t k = 0; k < 4; ++k)
            if (experts[k].gu.empty()) REQUIRE_OK(experts[k].make(2000 + (uint32_t)k) ? 0 : -1);
        cache_on({ 5, 12, 13, 20 }, { &experts[0], &experts[1], &experts[2], &experts[3] });
        view.resident[12] = view.resident[20] = view.resident[27] = 1;
        RadStream s = nullptr;
        RadEvent ev = nullptr;
        REQUIRE_OK(rad_stream_create(&s, 0));
        REQUIRE_OK(rad_event_create(&ev));
        fs.hx_out_ev_ = ev;
        uint16_t* y = (uint16_t*)rad_dev_alloc(24 * 5120 * 2, RAD_MEM_DEVICE);
        REQUIRE(y);
        const int64_t one = HostExpertPool::cost_rows(1);

        Layer base, w;
        layer(sel, nullptr, s, y, &base);
        CHECK_EQ(base.stage, 1);
        CHECK(base.cpu_units == std::vector<int32_t>({ 5, 12, 13, 20 }));
        CHECK_EQ(base.reserved, 4 * one);
        CHECK_EQ(cache.touched, cache.borrows);

        FakeLane lane;
        cache.borrows = cache.touched = 0;
        layer(sel, &lane, s, y, &w);
        /* The leases: 12 and 20 into aux_units_ (slot code -3), with the base's admit (touch), no reservation. */
        REQUIRE_EQ(w.slot.size(), sel.size());
        CHECK_EQ(w.slot[4], -3);   /* unit 12 */
        CHECK_EQ(w.slot[9], -3);   /* unit 20 */
        CHECK_EQ(w.slot[0], 0);    /* unit 5: the pool's */
        CHECK_EQ(w.slot[5], 1);    /* unit 13 */
        CHECK_EQ(w.slot[13], -1);  /* unit 27: the worker's, not cached, the copy link's as in the base */
        CHECK_EQ(w.aux_n, 2);
        CHECK(w.aux_list == std::vector<int32_t>({ 12, 20 }));
        CHECK(w.cpu_units == std::vector<int32_t>({ 5, 13 }));
        CHECK_EQ(w.reserved, 2 * one);
        CHECK_EQ(w.reserved, base.reserved - 2 * one);
        CHECK_EQ(cache.borrows, 4);
        CHECK_EQ(cache.touched, 4);
        CHECK_EQ(w.uncached, (uint64_t)1);
        /* Off want_ (the copy link stages the rest, as in the base) and among the aliased units. */
        CHECK(w.want == base.want);
        CHECK(std::find(w.want.begin(), w.want.end(), 12) == w.want.end());
        CHECK(std::find(w.want.begin(), w.want.end(), 20) == w.want.end());
        CHECK(std::find(w.want.begin(), w.want.end(), 27) != w.want.end());
        CHECK(w.keep == std::vector<int32_t>({ 5, 13, 12, 20 }));
        std::vector<int32_t> kb = base.keep, kw = w.keep;
        std::sort(kb.begin(), kb.end());
        std::sort(kw.begin(), kw.end());
        CHECK(kb == kw);
        /* The job, its rows placed, the leases back. */
        CHECK_EQ(lane.submits, 1);
        CHECK_EQ(lane.job.T, counts[12] + counts[20]);
        CHECK_EQ(lane.waits, 1);
        CHECK_EQ(fs.aux_stats_.placed, (uint64_t)lane.job.T);
        CHECK_EQ(fs.aux_n_, 0);
        CHECK(fs.aux_list_.empty());
        CHECK_EQ(fs.aux_units_[0].n, 0);
        CHECK_EQ(fs.aux_units_[1].n, 0);
        CHECK_EQ(fs.hx_stage_, 0);
        /* The gather's rows are the base's: the same rows, the same bytes in hx_out_ and in y. */
        CHECK(w.rows == base.rows);
        CHECK(w.out == base.out);
        CHECK(w.y == base.y);

        /* A layer of worker units only (12 and 20 cached, 27 not): no CPU job and no miss, yet the layer runs --
         * the job, the aliases -- where the base gives the pool 12 and 20. */
        const std::vector<int32_t> only = { 12, 20, 27 };
        Layer base2, w2;
        layer(only, nullptr, s, y, &base2);
        FakeLane lane2;
        layer(only, &lane2, s, y, &w2);
        CHECK_EQ(w2.stage, 1);
        CHECK_EQ(w2.n, 0);
        CHECK_EQ(w2.reserved, 0);
        CHECK(w2.want == std::vector<int32_t>({ 27 }));
        CHECK(w2.want == base2.want);
        CHECK(w2.keep == std::vector<int32_t>({ 12, 20 }));
        CHECK_EQ(lane2.submits, 1);
        CHECK(w2.rows == base2.rows);
        CHECK(w2.out == base2.out);
        CHECK(w2.y == base2.y);

        view.resident[12] = view.resident[20] = view.resident[27] = 0;
        mover.borrow_hook_ = {};
        fs.hx_out_rec_ = false;   /* its record completed (the stream synced) */
        fs.hx_out_ev_ = nullptr;
        rad_event_destroy(ev);
        rad_stream_destroy(s);
        rad_dev_free(y, RAD_MEM_DEVICE);
    }

    /* ---- the dynamic worker (work/aux-dyn/SPEC.md §7.4) ---- */

    /* No aux_set_dynamic (every new switch unset): no gate, no counting, the lane asked as the merged worker asks
     * it, the hit array never made. */
    void dynamic_switch_off_identity() {
        const std::vector<int32_t> want = { 3, 17, 40 };
        for (int e = 0; e < 96; ++e) counts[(size_t)e] = 0;
        for (int32_t u : want) counts[(size_t)u] = 2;
        FakeLane lane;
        fs.aux_bind(&lane, &view, aux::Config{});
        fs.aux_pass_ = true;
        view.resident[17] = 1;
        (void)choose(4, want);
        CHECK(fs.aux_gate_ == nullptr);
        CHECK(!fs.aux_online_);
        CHECK(!fs.aux_in_lane_);
        CHECK(fs.aux_hits_.empty());
        CHECK_EQ(fs.aux_hit_tok_, (uint64_t)0);
        CHECK_EQ(lane.states, 1);
        CHECK_EQ(fs.aux_stats_.declined_released, (uint64_t)0);
        std::vector<uint32_t> e(96, 0);
        CHECK_EQ(fs.aux_swap_hits(&e), (uint64_t)0);   /* not counting: nothing to hand off, nothing swapped */
        CHECK(e == std::vector<uint32_t>(96, 0));
        view.resident[17] = 0;
        fs.aux_unbind();
    }

    /* The gate shut (a release): hx_choose declines without asking the lane, counts the decline, holds no entry;
     * the gather's rows are the pass's without the worker. Open: the rank holds its entry from the decision to the
     * layer's last lane call (hx_finish), and leaves it. */
    void a_shut_gate_declines_without_the_lane() {
        const std::vector<int32_t> sel = { 5, 6, 7, 8, 12, 13, 14, 15, 19, 20, 21, 22, 26, 27, 28, 29 };
        if (experts.size() < 4) experts.resize(4);
        for (size_t k = 0; k < 4; ++k)
            if (experts[k].gu.empty()) REQUIRE_OK(experts[k].make(2000 + (uint32_t)k) ? 0 : -1);
        cache_on({ 5, 12, 13, 20 }, { &experts[0], &experts[1], &experts[2], &experts[3] });
        RadStream s = nullptr;
        RadEvent ev = nullptr;
        REQUIRE_OK(rad_stream_create(&s, 0));
        REQUIRE_OK(rad_event_create(&ev));
        fs.hx_out_ev_ = ev;
        uint16_t* y = (uint16_t*)rad_dev_alloc(24 * 5120 * 2, RAD_MEM_DEVICE);
        REQUIRE(y);
        Layer base;
        layer(sel, nullptr, s, y, &base);
        view.resident[12] = view.resident[20] = 1;
        aux::Gate gate;
        gate.closed.store(true);
        FakeLane lane;
        /* layer() binds the lane itself; the gate goes on after its bind, as aux_start does it. */
        pass(4, 5, 4);
        fs.hx_li_ = -1;
        fs.hx_stage_ = 0;
        fs.want_ = sel;
        fs.aux_bind(&lane, &view, aux::Config{});
        fs.aux_set_dynamic(&gate, 1, nullptr);
        fs.aux_pass_ = true;
        fs.hx_choose(0);
        CHECK(!fs.aux_take_);
        CHECK_EQ(lane.states, 0);   /* the lane never asked */
        CHECK_EQ(fs.aux_stats_.declined_released, (uint64_t)1);
        CHECK_EQ(gate.in_lane[1].load(), 0);
        CHECK(!fs.aux_in_lane_);
        fs.hx_drain();
        fs.aux_unbind();
        /* the same layer through hx_submit / hx_finish with the gate shut: the base's rows */
        Layer shut;
        pass(4, 5, 4);
        fs.hx_li_ = -1;
        fs.hx_stage_ = 0;
        fs.want_ = sel;
        fs.hx_stats_ = FileStager::HostStats{};
        fs.aux_bind(&lane, &view, aux::Config{});
        fs.aux_set_dynamic(&gate, 1, nullptr);
        fs.aux_pass_ = true;
        fs.hx_choose(0);
        REQUIRE_EQ(fs.hx_stage_, 1);
        fs.hx_cpu_units();
        REQUIRE_OK(fs.hx_submit());
        for (int i = 0; i < 24 * 5120; ++i) y[i] = 0x5555;
        fs.hx_confirmed_ = true;
        RadTensor Y{};
        Y.data = y;
        Y.dtype = RAD_BF16;
        Y.rank = 2;
        Y.shape[0] = 24;
        Y.shape[1] = 5120;
        Y.stride[0] = 5120;
        Y.stride[1] = 1;
        REQUIRE_OK(fs.hx_finish(s, &Y, 1));
        REQUIRE_OK(rad_stream_sync(s));
        shut.out.assign(fs.hx_out_, fs.hx_out_ + 24 * 5120);
        shut.y.assign(y, y + 24 * 5120);
        CHECK_EQ(lane.submits, 0);
        CHECK(shut.out == base.out);
        CHECK(shut.y == base.y);
        fs.aux_unbind();
        /* open: taken, the entry held through hx_submit, left by hx_finish */
        gate.closed.store(false);
        pass(4, 5, 4);
        fs.hx_li_ = -1;
        fs.hx_stage_ = 0;
        fs.want_ = sel;
        fs.aux_bind(&lane, &view, aux::Config{});
        fs.aux_set_dynamic(&gate, 1, nullptr);
        fs.aux_pass_ = true;
        fs.hx_choose(0);
        CHECK(fs.aux_take_);
        CHECK_EQ(gate.in_lane[1].load(), 1);
        fs.hx_cpu_units();
        REQUIRE_OK(fs.hx_submit());
        CHECK_EQ(gate.in_lane[1].load(), 1);   /* the job in flight: still the rank's */
        REQUIRE(cpu_rows_of_compacted(lane.out.data()));
        fs.hx_confirmed_ = true;
        REQUIRE_OK(fs.hx_finish(s, &Y, 1));
        REQUIRE_OK(rad_stream_sync(s));
        CHECK_EQ(gate.in_lane[1].load(), 0);
        CHECK(!fs.aux_in_lane_);
        CHECK_EQ(lane.submits, 1);
        CHECK(std::equal(fs.hx_out_, fs.hx_out_ + 24 * 5120, base.out.begin()));
        /* hx_drain leaves it too (a layer given up) */
        pass(4, 5, 4);
        fs.hx_li_ = -1;
        fs.hx_stage_ = 0;
        fs.want_ = sel;
        fs.hx_choose(0);
        CHECK_EQ(gate.in_lane[1].load(), 1);
        fs.hx_drain();
        CHECK_EQ(gate.in_lane[1].load(), 0);
        fs.aux_unbind();
        view.resident[12] = view.resident[20] = 0;
        mover.borrow_hook_ = {};
        fs.hx_out_rec_ = false;
        fs.hx_out_ev_ = nullptr;
        rad_event_destroy(ev);
        rad_stream_destroy(s);
        rad_dev_free(y, RAD_MEM_DEVICE);
    }

    /* The online set's count (B.2): rows routed to each selected candidate, in passes the worker could take only;
     * the hand-off swaps arrays of the plan's size (two in a row), and refuses another size. */
    void online_counting_and_the_hand_off() {
        for (int e = 0; e < 96; ++e) counts[(size_t)e] = 0;
        counts[3] = 2;
        counts[17] = 3;
        counts[40] = 1;
        std::vector<uint8_t> cand(96, 0);
        cand[17] = cand[40] = 1;
        FakeLane lane;
        lane.st = aux::Lane::kBusy;
        fs.aux_bind(&lane, &view, aux::Config{});
        fs.aux_set_dynamic(nullptr, 0, &cand);
        REQUIRE_EQ(fs.aux_hits_.size(), (size_t)96);
        fs.aux_pass_ = true;
        const std::vector<int32_t> want = { 3, 17, 40 };
        (void)choose(4, want);
        CHECK_EQ(fs.aux_hits_[17], 3u);
        CHECK_EQ(fs.aux_hits_[40], 1u);
        CHECK_EQ(fs.aux_hits_[3], 0u);   /* not a candidate */
        (void)choose(17, want);          /* taller than 16 rows: not counted */
        fs.aux_pass_ = false;            /* a draft or sliced pass */
        (void)choose(4, want);
        fs.aux_pass_ = true;
        fs.aux_rows_[0].ok = false;      /* a layer the worker cannot take */
        (void)choose(4, want);
        fs.aux_rows_[0].ok = true;
        CHECK_EQ(fs.aux_hits_[17], 3u);
        CHECK_EQ(fs.aux_hits_[40], 1u);
        /* counting goes on whatever the lane or the gate: the lane was busy throughout */
        (void)choose(4, want);
        CHECK_EQ(fs.aux_hits_[17], 6u);
        /* the hand-off: the stager's array for the inbox's zeroed one, its tokens with it */
        fs.aux_hit_tok_ = 5;
        std::vector<uint32_t> inbox(96, 0);
        CHECK_EQ(fs.aux_swap_hits(&inbox), (uint64_t)5);
        CHECK_EQ(inbox[17], 6u);
        CHECK_EQ(inbox[40], 2u);
        CHECK_EQ(fs.aux_hit_tok_, (uint64_t)0);
        CHECK(fs.aux_hits_ == std::vector<uint32_t>(96, 0));
        /* the control thread zeroes what it merged; the next hand-off swaps it back */
        std::fill(inbox.begin(), inbox.end(), 0u);
        (void)choose(4, want);
        fs.aux_hit_tok_ = 4;
        CHECK_EQ(fs.aux_swap_hits(&inbox), (uint64_t)4);
        CHECK_EQ(inbox[17], 3u);
        CHECK(fs.aux_hits_ == std::vector<uint32_t>(96, 0));
        std::vector<uint32_t> small(10, 0);
        CHECK_EQ(fs.aux_swap_hits(&small), (uint64_t)0);   /* another size: not swapped */
        CHECK_EQ(small.size(), (size_t)10);
        CHECK_EQ(fs.aux_hits_.size(), (size_t)96);
        fs.aux_unbind();
    }

    /* A boundary's flips (the view changing between passes): a unit flipped out is in no later job, one flipped in
     * is; the gather's rows stay the CPU path's. */
    void flips_follow_the_view() {
        const std::vector<int32_t> sel = { 5, 6, 7, 8, 12, 13, 14, 15, 19, 20, 21, 22, 26, 27, 28, 29 };
        if (experts.size() < 4) experts.resize(4);
        for (size_t k = 0; k < 4; ++k)
            if (experts[k].gu.empty()) REQUIRE_OK(experts[k].make(2000 + (uint32_t)k) ? 0 : -1);
        cache_on({ 5, 12, 13, 20 }, { &experts[0], &experts[1], &experts[2], &experts[3] });
        RadStream s = nullptr;
        RadEvent ev = nullptr;
        REQUIRE_OK(rad_stream_create(&s, 0));
        REQUIRE_OK(rad_event_create(&ev));
        fs.hx_out_ev_ = ev;
        uint16_t* y = (uint16_t*)rad_dev_alloc(24 * 5120 * 2, RAD_MEM_DEVICE);
        REQUIRE(y);
        Layer base, a, b;
        layer(sel, nullptr, s, y, &base);
        view.resident[12] = 1;
        FakeLane la;
        layer(sel, &la, s, y, &a);
        CHECK(a.aux_list == std::vector<int32_t>({ 12 }));
        view.resident[12] = 0;   /* flipped out */
        view.resident[20] = 1;   /* flipped in */
        FakeLane lb;
        layer(sel, &lb, s, y, &b);
        CHECK(b.aux_list == std::vector<int32_t>({ 20 }));
        CHECK(a.out == base.out && b.out == base.out);
        CHECK(a.y == base.y && b.y == base.y);
        view.resident[20] = 0;
        mover.borrow_hook_ = {};
        fs.hx_out_rec_ = false;
        fs.hx_out_ev_ = nullptr;
        rad_event_destroy(ev);
        rad_stream_destroy(s);
        rad_dev_free(y, RAD_MEM_DEVICE);
    }

    /* aux_load and Worker::load_tables on the host backend's fifth device (work/aux/SPEC.md §6.3, §7.5). */
    void a_load_frees_nothing_in_flight() {
        if (experts.empty()) experts.resize(1);
        if (experts[0].gu.empty()) REQUIRE_OK(experts[0].make(3000) ? 0 : -1);
        cache_on({ 12, 20 }, { &experts[0], &experts[0] });
        const ds41aux::Layout L = ds41aux::layout((int64_t)experts[0].gu.size(), (int64_t)experts[0].dn.size());
        auto make = [&](std::unique_ptr<aux::Worker>* out) {
            aux::Config cfg;
            cfg.on = true;
            cfg.cap_mib = 64;
            cfg.headroom_mib = 0;
            ds41aux::Card card;
            card.ord = 4;
            card.bdf = "0000:c3:00.0";
            out->reset(new aux::Worker(cfg, card, { 0, 1, 2, 3 }));
            std::string err;
            REQUIRE_OK((*out)->reserve(0, &err));
            REQUIRE_OK((*out)->build(1, 2, L, 0, &err));
        };
        /* Live allocations on the fifth device: pinned staging mapped there alone, and its device memory. */
        auto pinned = [] { return host_live_allocs(RAD_MEM_HOST_PINNED, 4); };
        auto device_allocs = [] { return host_live_allocs(RAD_MEM_DEVICE, 4); };
        std::string why;
        FileStager::AuxLoadStats st;

        /* The queue moving: both units loaded and read back, the tables too, every staging buffer freed; the
         * lanes drain and the arena goes. */
        std::unique_ptr<aux::Worker> w;
        make(&w);
        REQUIRE(w && w->n_slots() == 2);
        int64_t p0 = pinned();
        REQUIRE_OK(fs.aux_load(w.get(), 0, { 12, 20 }, 0, 2, &st, &why));
        CHECK_EQ(pinned(), p0);
        CHECK_EQ(st.loaded, 2);
        CHECK_EQ(st.cached, 2);
        const aux::RankView& v = w->view(0);
        CHECK(v.is_resident(12) && v.is_resident(20));
        CHECK_EQ(v.slot[12], 0);
        CHECK_EQ(v.slot[20], 1);
        CHECK(std::memcmp(w->slot(1), experts[0].gu.data(), experts[0].gu.size()) == 0);
        CHECK(std::memcmp(w->slot(1) + L.dn_off, experts[0].dn.data(), experts[0].dn.size()) == 0);
        CHECK(v.gu_tab_entries[12] == ds41p1::mark(w->slot(0), ds41p1::kMarkPacked));
        CHECK(v.dn_tab_entries[20] == ds41p1::mark(w->slot(1) + L.dn_off, ds41p1::kMarkPacked));
        REQUIRE_OK(w->load_tables(&why));
        CHECK_EQ(pinned(), p0);
        CHECK(w->view(0).gu_tab[0] == w->table(0, 0, 0));
        (void)w->finish(0, 0, 0);
        CHECK(w->drained());
        int64_t d0 = device_allocs();
        w.reset();
        CHECK_EQ(device_allocs(), d0 - 1);   /* the arena freed */

        /* The queue held at a gate (a stuck extra card): the waits time out, the copies stay queued, so neither
         * their bounce and read-back buffers nor the tables' staging is freed, lane 0 is not drained and the
         * worker leaves the lanes and the arena to the process exit. */
        make(&w);
        REQUIRE(w && w->n_slots() == 2);
        w->set_load_wait_ns(20'000'000LL);
        void* gate = nullptr;
        REQUIRE_OK(rad_stream_gate(w->device_lane(0)->stream(), &gate));
        p0 = pinned();
        CHECK(fs.aux_load(w.get(), 0, { 12, 20 }, 0, 2, &st, &why) < 0);
        CHECK(has(why, "did not complete in 0.02 s"));
        CHECK(has(why, "left to the process exit"));
        CHECK_EQ(pinned(), p0 + 2);
        CHECK_EQ(st.loaded, 0);
        CHECK_EQ(w->device_lane(0)->state(), aux::Lane::kBusy);   /* the copies: the lane's work in flight */
        CHECK(w->load_tables(&why) < 0);
        CHECK(has(why, "did not complete in 0.02 s"));
        CHECK_EQ(pinned(), p0 + 4);
        const int64_t t0 = now_ns();
        const std::string line = w->finish(0, 0, 0);
        CHECK(now_ns() - t0 < 10'000'000'000LL);   /* 2 s for lane 0, a probe for each idle one */
        CHECK(!w->drained());
        CHECK(has(line, "1 lane(s) not drained"));
        d0 = device_allocs();
        REQUIRE_OK(rad_gate_open(gate));   /* the queue moves on: its copies land in what was kept */
        w.reset();
        CHECK_EQ(device_allocs(), d0);     /* the arena kept */
        mover.borrow_hook_ = {};
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

/* A leaked lane's queue may still hold the record's copy into its workspace (submit queues it before the step
 * that fails), so the workspace must outlive the lane, as the worker's arena does (Worker::~Worker leaks both):
 * kept to the process exit here too. A stack buffer freed at the test's scope end was written after its free
 * now and then (aux_stager_test crashed once in 13 runs). */
std::vector<char>& leaked_workspace(size_t bytes) {
    static std::deque<std::vector<char>> kept;
    kept.emplace_back(bytes, 0);
    return kept.back();
}

/* RADIANCE_EXPERIMENTAL_DS41_AUX_ROWS: a lane for 64 rows takes a job of 40 rows and 200 slots into ws(64)'s
 * layout, and refuses one of 65 rows or 385 slots (a job's shape); a lane of the default 16 refuses 17 rows. */
TEST(device_lane_for_taller_jobs) {
    const ds41aux::Ws w = ds41aux::ws(64);
    std::vector<char>& ws = leaked_workspace((size_t)w.bytes);
    rad::aux::DeviceLane lane(4, 1, ws.data(), 3, 64);
    std::string why;
    REQUIRE_OK(lane.init(&why));
    std::vector<uint16_t> x((size_t)40 * 5120);
    for (size_t i = 0; i < x.size(); ++i) x[i] = (uint16_t)(i * 5 + 3);
    std::vector<int32_t> sorted(200), off(97);
    for (int i = 0; i < 200; ++i) sorted[(size_t)i] = (i % 40) * 6 + i / 40;
    for (int e = 0; e <= 96; ++e) off[(size_t)e] = e <= 2 ? 0 : 200;
    const RadParam params[1] = {};
    rad::aux::Job j;
    j.AR = 40;
    j.T = 200;
    j.x = x.data();
    j.sorted = sorted.data();
    j.off = off.data();
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
    REQUIRE_EQ(lane.wait(now_ns() + 2'000'000'000LL), 1);
    CHECK(std::memcmp(ws.data() + w.sorted, sorted.data(), 200 * 4) == 0);
    CHECK(std::memcmp(ws.data() + w.off, off.data(), 97 * 4) == 0);
    CHECK(std::memcmp(ws.data() + w.x, x.data(), x.size() * 2) == 0);
    CHECK(g_log.t[0][0].data == ws.data() + w.x && g_log.t[0][0].shape[0] == 40);
    CHECK(g_log.t[0][4].data == ws.data() + w.gu_out && g_log.t[0][4].shape[0] == 200);
    CHECK(g_log.t[1][1].data == ws.data() + w.act && g_log.t[1][1].shape[0] == 200);
    CHECK(g_log.t[2][4].data == ws.data() + w.dn_out && g_log.t[2][4].shape[0] == 200);
    bool rows_ok = true;
    for (int64_t i = 0; i < 200 * 5120; ++i) rows_ok = rows_ok && lane.rows()[i] == (uint16_t)(0x3c00 + i % 977);
    CHECK(rows_ok);
    CHECK(lane.drain(1'000'000'000LL));
    /* Past the lane's height or slots: refused before anything is queued, the lane retired. */
    for (int bad = 0; bad < 2; ++bad) {
        std::vector<char>& ws2 = leaked_workspace((size_t)w.bytes);
        rad::aux::DeviceLane l2(4, 1, ws2.data(), 3, 64);
        REQUIRE_OK(l2.init(&why));
        rad::aux::Job k = j;
        std::vector<int32_t> s2(385, 0);
        std::vector<uint16_t> x2((size_t)65 * 5120, 0);
        if (bad == 0) { k.AR = 65; k.x = x2.data(); }
        else { k.T = 385; k.sorted = s2.data(); }
        CHECK(l2.submit(k) < 0);
        CHECK(l2.dead());
        CHECK(has(l2.why_dead(), "a job's shape"));
        l2.leak();
    }
    const ds41aux::Ws w16 = ds41aux::ws();
    std::vector<char>& ws3 = leaked_workspace((size_t)w16.bytes);
    rad::aux::DeviceLane l3(4, 1, ws3.data(), 3);
    REQUIRE_OK(l3.init(&why));
    rad::aux::Job k = j;
    k.AR = 17;
    k.T = 17;
    CHECK(l3.submit(k) < 0);
    CHECK(has(l3.why_dead(), "a job's shape"));
    l3.leak();
}

/* The shutdown line: unchanged at the default 16 (no height part); at 256 rows it says so and counts every bucket up
 * to 256. */
TEST(the_shutdown_line_by_height) {
    rad::aux::Stats s;
    s.jobs = 5;
    s.jobs_h[0] = 3;
    s.rows_h[0] = 20;
    s.eligible_h[0] = 60;
    s.jobs_h[4] = 2;
    s.rows_h[4] = 300;
    s.eligible_h[4] = 900;
    const std::string a = rad::aux::stats_line(0, s), b = rad::aux::stats_line(0, s, false, 16);
    CHECK(a == b);
    CHECK(has(a, "passes of up to 16 rows"));
    CHECK(!has(a, "by height"));
    const std::string t = rad::aux::stats_line(2, s, false, 256);
    CHECK(has(t, "aux experts: rank 2: "));
    CHECK(has(t, "passes of up to 256 rows"));
    CHECK(has(t, "; by height: 1-16: 3 jobs, 20 of 60 rows; 17-32: 0 jobs, 0 of 0 rows;"));
    CHECK(has(t, "; 129-256: 2 jobs, 300 of 900 rows"));
    CHECK(!has(t, "257-"));
    CHECK(has(rad::aux::stats_line(2, s, false, 200), "; 129-200: 2 jobs, 300 of 900 rows"));
    /* The rest of the line is the default's, word for word, before the height part. */
    const std::string head = t.substr(0, t.find("; by height:"));
    std::string a2 = a;
    const size_t at = a2.find("passes of up to 16 rows");
    a2.replace(at, std::strlen("passes of up to 16 rows"), "passes of up to 256 rows");
    a2.replace(0, std::strlen("aux experts: rank 0: "), "aux experts: rank 2: ");
    CHECK(head == a2);
}

TEST(device_lane_a_failed_step_fails_the_job_and_retires_the_lane) {
    for (int at = 0; at < 3; ++at) {
        const ds41aux::Ws w = ds41aux::ws();
        std::vector<char>& ws = leaked_workspace((size_t)w.bytes);
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
    std::vector<char>& ws = leaked_workspace((size_t)w.bytes);
    rad::aux::DeviceLane lane(4, 3, ws.data(), 64);
    std::string why;
    REQUIRE_OK(lane.init(&why));
    rad::aux::Job j;
    j.AR = 17;
    CHECK(lane.submit(j) < 0);
    CHECK(lane.dead());
    lane.leak();
}

/* A loader wait that timed out (work/aux/SPEC.md §7.5): what it waited for is still queued, so the lane is not
 * idle and not drained -- within drain's bound -- until it completes. */
TEST(device_lane_a_load_wait_that_timed_out_keeps_the_lane_not_drained) {
    const ds41aux::Ws w = ds41aux::ws();
    std::vector<char> ws((size_t)w.bytes, 0);
    rad::aux::DeviceLane lane(4, 0, ws.data(), 64);
    std::string why;
    REQUIRE_OK(lane.init(&why));
    void* gate = nullptr;
    REQUIRE_OK(rad_stream_gate(lane.stream(), &gate));
    std::vector<char> src(4096, 7), dst(4096, 0);
    REQUIRE_OK(rad_memcpy_async(dst.data(), src.data(), (int64_t)src.size(), lane.stream()));
    CHECK_EQ(lane.sync_bounded(2'000'000LL), 0);
    CHECK(!lane.dead());
    CHECK_EQ(lane.state(), rad::aux::Lane::kBusy);
    int64_t t0 = now_ns();
    CHECK(!lane.drain(20'000'000LL));
    CHECK(now_ns() - t0 < 1'000'000'000LL);
    REQUIRE_OK(rad_gate_open(gate));
    CHECK(lane.drain(2'000'000'000LL));
    CHECK(dst == src);
    CHECK_EQ(lane.state(), rad::aux::Lane::kReady);   /* abandoned without a backoff: ready again */
}

/* An idle lane whose queue stopped (a card hung after the lane's last job, with no fault the host sees): the
 * probe drain() records behind everything does not complete, so the lane is not drained and nothing waits on
 * it without a bound; once the queue moves on, it drains. */
TEST(device_lane_an_idle_lane_whose_queue_stopped_is_not_drained) {
    const ds41aux::Ws w = ds41aux::ws();
    std::vector<char> ws((size_t)w.bytes, 0);
    rad::aux::DeviceLane lane(4, 1, ws.data(), 64);
    std::string why;
    REQUIRE_OK(lane.init(&why));
    CHECK(lane.drain(1'000'000'000LL));   /* idle on a moving queue */
    void* gate = nullptr;
    REQUIRE_OK(rad_stream_gate(lane.stream(), &gate));
    CHECK_EQ(lane.state(), rad::aux::Lane::kReady);   /* nothing of its own in flight */
    const int64_t t0 = now_ns();
    CHECK(!lane.drain(10'000'000LL));     /* at least 100 ms for the probe, then given up */
    const int64_t took = now_ns() - t0;
    CHECK(took >= 100'000'000LL && took < 1'000'000'000LL);
    REQUIRE_OK(rad_gate_open(gate));
    CHECK(lane.drain(2'000'000'000LL));
}

/* A loader wait whose event cannot be recorded behind copies already queued: the lane retires (dead, never
 * drained), so neither its buffers nor the arena are given back under those copies. */
TEST(device_lane_a_loader_record_that_fails_retires_the_lane) {
    const ds41aux::Ws w = ds41aux::ws();
    std::vector<char>& ws = leaked_workspace((size_t)w.bytes);
    rad::aux::DeviceLane lane(4, 2, ws.data(), 64);
    std::string why;
    REQUIRE_OK(lane.init(&why));
    void* gate = nullptr;
    REQUIRE_OK(rad_stream_gate(lane.stream(), &gate));
    /* The host backend refuses a submission onto a full stream behind a closed gate: fill it, again once the
     * stream has taken the gate itself off its ring (it waits there). */
    static char a = 1, b = 0;
    int queued = 0;
    for (int round = 0; round < 3; ++round) {
        while (queued < 4096 && rad_memcpy_async(&b, &a, 1, lane.stream()) >= 0) ++queued;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    REQUIRE(queued > 0 && queued < 4096);
    CHECK(lane.sync_bounded(1'000'000'000LL) < 0);
    CHECK(lane.dead());
    CHECK(has(lane.why_dead(), "a loader wait's event record"));
    CHECK(!lane.drain(1'000'000'000LL));
    REQUIRE_OK(rad_gate_open(gate));   /* the copies run; the lane stays dead */
    REQUIRE_OK(rad_stream_sync(lane.stream()));
    CHECK(!lane.drain(1'000'000'000LL));
    lane.leak();   /* a dead lane is left to the process exit, as the worker leaves it */
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

TEST(tall_passes_are_compacted_queued_and_placed) {
    if (!pool()) SKIP("no AVX2 or no CPU pool");
    for (int AR : { 17, 40, 64 }) {
        AuxStagerProbe p(64);
        p.tall_serve_places_the_lanes_rows(AR);
    }
    AuxStagerProbe p(128);
    p.tall_serve_places_the_lanes_rows(128);
}

TEST(a_late_tall_job_has_a_scaled_deadline_and_falls_back) {
    if (!pool()) SKIP("no AVX2 or no CPU pool");
    AuxStagerProbe p(64);
    p.tall_late_job_has_a_scaled_deadline_and_falls_back(48);
}

TEST(the_tall_warm_up_job_checks_every_row_against_the_cpu_pool) {
    if (!pool()) SKIP("no AVX2 or no CPU pool");
    AuxStagerProbe p(48);
    p.tall_warm_checks_the_lanes_rows();
}

TEST(tall_passes_up_to_the_bound_only) {
    if (!pool()) SKIP("no AVX2 or no CPU pool");
    AuxStagerProbe p(64);
    p.tall_bound(64);
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

TEST(hx_choose_serve_leases_resident_cached_units_without_a_reservation) {
    if (!pool()) SKIP("no AVX2 or no CPU pool");
    AuxStagerProbe p;
    p.serve_leases_through_hx_choose();
}

TEST(dynamic_switches_unset_change_nothing_in_hx_choose) {
    AuxStagerProbe p;
    p.dynamic_switch_off_identity();
}

TEST(a_shut_gate_declines_without_asking_the_lane) {
    if (!pool()) SKIP("no AVX2: the CPU pool cannot run");
    AuxStagerProbe p;
    p.a_shut_gate_declines_without_the_lane();
}

TEST(online_counting_and_the_hit_hand_off) {
    AuxStagerProbe p;
    p.online_counting_and_the_hand_off();
}

TEST(flips_between_passes_follow_the_view) {
    if (!pool()) SKIP("no AVX2: the CPU pool cannot run");
    AuxStagerProbe p;
    p.flips_follow_the_view();
}

TEST(aux_load_and_load_tables_free_nothing_their_queued_copies_use) {
    if (!pool()) SKIP("no AVX2 or no CPU pool");
    AuxStagerProbe p;
    p.a_load_frees_nothing_in_flight();
}

RAD_TEST_MAIN()
