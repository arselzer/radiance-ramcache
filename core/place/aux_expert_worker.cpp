/* aux_expert_worker.cpp -- see aux_expert_worker.h. */
#include "aux_expert_worker.h"

#include "device/device.h"
#include "rad_internal.h"

#include <immintrin.h>
#include <limits.h>
#include <stdlib.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <thread>

namespace rad {
namespace aux {

namespace {

int64_t now_ns() {
    return (int64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

RadTensor mk2(void* d, int64_t rows, int64_t cols) {
    RadTensor t{};
    t.data = d;
    t.dtype = RAD_BF16;
    t.rank = 2;
    t.shape[0] = rows;
    t.shape[1] = cols;
    t.stride[0] = cols;
    t.stride[1] = 1;
    return t;
}
RadTensor mk1(void* d, int64_t n) {
    RadTensor t{};
    t.data = d;
    t.dtype = RAD_I32;
    t.rank = 1;
    t.shape[0] = n;
    t.stride[0] = 1;
    return t;
}

}  // namespace

/* ------------------------------------------------------------------ discovery's I/O */

std::vector<ds41aux::Card> inventory(const std::string& sysfs) {
    std::vector<ds41aux::Card> cards;
    const int n = rad_dev_count();
    for (int i = 0; i < n; ++i) {
        RadDeviceProps p{};
        if (rad_dev_props(i, &p) < 0) continue;
        ds41aux::Card c;
        c.ord = i;
        c.name = p.name;
        c.arch = ds41aux::strip_arch(p.arch);
        c.integrated = dev_integrated(i);
        c.vram_bytes = p.vram_bytes;
        c.vram_free = p.vram_free;
        std::string bdf;
        if (ds41aux::parse_bdf(dev_pci_address(i), &bdf)) {
            c.bdf = bdf;
            char rp[PATH_MAX];
            const std::string path = sysfs + "/bus/pci/devices/" + bdf;
            if (::realpath(path.c_str(), rp)) ds41aux::parse_pci_path(rp, &c.complex, &c.port);
        }
        cards.push_back(c);
    }
    return cards;
}

std::string selector_bdf(const Config& cfg, const std::string& sysfs) {
    if (cfg.device.empty()) return std::string();
    if (!cfg.device_uuid) return cfg.device;
    std::string bdf;
    return ds41aux::resolve_uuid(sysfs, cfg.device, &bdf) ? bdf : std::string();
}

/* ------------------------------------------------------------------ the shutdown line */

std::string stats_line(int r, const Stats& s, bool released, int max_rows) {
    const double rt = s.rt_n ? (double)s.rt_sum_ns / (double)s.rt_n * 1e-3 : 0.0;
    const std::string rel = released ? fmt(", %llu while released", (unsigned long long)s.declined_released) : "";
    /* ds41aux::kRowsEnv above 16: the jobs, their rows and the rows the CPU path met, by pass height. */
    std::string by;
    if (max_rows > ds41aux::kMaxRows) {
        by = "; by height:";
        for (int b = 0; b < ds41aux::kHeightBuckets && ds41aux::height_lo(b) <= max_rows; ++b)
            by += fmt("%s %d-%d: %llu jobs, %llu of %llu rows", b ? ";" : "", ds41aux::height_lo(b),
                      std::min(ds41aux::height_hi(b), max_rows), (unsigned long long)s.jobs_h[b],
                      (unsigned long long)s.rows_h[b], (unsigned long long)s.eligible_h[b]);
    }
    return fmt("%s%d: %llu layer jobs, %llu units, %llu rows on the extra card of %llu rows the CPU path met in passes "
               "of up to %d rows; declined: %llu taller, %llu draft, %llu without usable kernel rows, %llu while the "
               "lane was busy or backing off, %llu after it failed%s; %llu worker units not cached (CPU miss path); "
               "waited %.1f ms beyond the CPU's rows, round trip %.1f us mean, %.1f us max; %llu deadline misses "
               "(%llu forced; %llu units, %llu rows recomputed on the CPU in %.1f ms), %llu launch failures; verify: "
               "%llu rows compared, %llu differ (%llu values), %llu not compared%s",
               ds41aux::kLogRank, r, (unsigned long long)s.jobs, (unsigned long long)s.units,
               (unsigned long long)s.placed, (unsigned long long)s.eligible_rows, max_rows,
               (unsigned long long)s.declined_tall, (unsigned long long)s.declined_draft,
               (unsigned long long)s.declined_rows, (unsigned long long)s.declined_busy,
               (unsigned long long)s.declined_dead, rel.c_str(), (unsigned long long)s.uncached, (double)s.wait_ns * 1e-6, rt,
               (double)s.rt_max_ns * 1e-3, (unsigned long long)s.late, (unsigned long long)s.forced_late,
               (unsigned long long)s.fallback_units, (unsigned long long)s.fallback_rows,
               (double)s.fallback_ns * 1e-6, (unsigned long long)s.failed, (unsigned long long)s.verified,
               (unsigned long long)s.differ_rows, (unsigned long long)s.differ_values,
               (unsigned long long)s.not_compared, by.c_str());
}

/* ------------------------------------------------------------------ the lane */

DeviceLane::DeviceLane(int device, int rank, char* ws, int64_t backoff, int rows)
    : dev_(device), rank_(rank), rows_(rows), ws_(ws), backoff_(backoff) {}

DeviceLane::~DeviceLane() {
    if (leaked_) return;   /* a job may still run on the queue: the process exit reclaims it all */
    if (ev_) rad_event_destroy(ev_);
    if (s_) rad_stream_destroy(s_);
    if (rec_) rad_dev_free(rec_, RAD_MEM_HOST_PINNED);
    if (out_) rad_dev_free(out_, RAD_MEM_HOST_PINNED);
}

int DeviceLane::init(std::string* why) {
    const ds41aux::Ws w = ds41aux::ws(rows_);
    if (w.bytes <= 0) {
        *why = fmt("rank %d's lane: no workspace for passes of %d rows", rank_, rows_);
        return RAD_E_INVAL;
    }
    if (rad_stream_create(&s_, 0) < 0) {
        s_ = nullptr;
        *why = fmt("rank %d's queue on device %d: %s", rank_, dev_, rad_dev_last_error());
        return RAD_E_DEVICE;
    }
    if (rad_event_create(&ev_) < 0) {
        ev_ = nullptr;
        *why = fmt("rank %d's event on device %d: %s", rank_, dev_, rad_dev_last_error());
        return RAD_E_DEVICE;
    }
    rec_ = (char*)dev_alloc_pinned_on(w.rec, { dev_ });
    out_ = (uint16_t*)dev_alloc_pinned_on(w.out, { dev_ });
    if (!rec_ || !out_) {
        *why = fmt("rank %d's pinned record and rows (device %d only): %s", rank_, dev_, rad_dev_last_error());
        return RAD_E_NOMEM;
    }
    return RAD_OK;
}

int DeviceLane::retire(const char* step, int rc, bool fault) {
    job_ok_ = false;
    fault_ = fault_ || fault;
    if (!dead_) {
        dead_ = true;
        why_dead_ = fmt("%s failed (%d): %s", step, rc, rad_dev_last_error());
        RAD_WARN("aux experts: rank %d's lane on device %d: %s; the lane takes no more jobs and the CPU computes "
                 "their rows", rank_, dev_, why_dead_.c_str());
    }
    return rc < 0 ? rc : RAD_E_DEVICE;
}

Lane::State DeviceLane::state() {
    if (dead_) return kDead;
    if (pending_) {
        if (!abandoned_) return kBusy;
        const int q = job_ok_ ? rad_event_query(ev_) : -1;
        if (q == 1) {
            pending_ = false;
            abandoned_ = false;
            if (backoff_after_) backoff_left_ = backoff_;
        } else if (q < 0 || dev_faulted(dev_)) {
            retire("an abandoned job", q < 0 ? q : RAD_E_DEVICE, true);
            return kDead;
        } else {
            return kBusy;
        }
    }
    return backoff_left_ > 0 ? kBackoff : kReady;
}

void DeviceLane::declined() {
    if (backoff_left_ > 0) --backoff_left_;
}

int DeviceLane::submit(const Job& j) {
    if (dead_) return RAD_E_STATE;
    if (pending_) return retire("a submit while a job was in flight", RAD_E_STATE);
    const ds41aux::Ws w = ds41aux::ws(rows_);
    if (j.AR < 1 || j.AR > rows_ || j.T < 1 || j.T > rows_ * ds41aux::kTopK || !j.x || !j.sorted || !j.off ||
        !j.gu.fn || !j.sw.fn || !j.dn.fn || !j.gu_w.data || !j.dn_w.data)
        return retire("a job's shape", RAD_E_INVAL);
    /* From here on the job is in flight: a step that fails leaves it failed (wait answers < 0). */
    pending_ = true;
    job_ok_ = false;
    abandoned_ = false;
    backoff_after_ = false;
    /* The record (the extra card's pinned memory only), then one copy of it onto the card: an H2D from
     * pinned memory acquires at system scope (a_memcpy_async), so no rad_dev_host_wrote, which would make
     * every rank queue's next dispatch acquire at system scope. */
    std::memcpy(rec_ + w.sorted, j.sorted, (size_t)j.T * 4);
    std::memcpy(rec_ + w.off, j.off, (size_t)(ds41aux::kExperts + 1) * 4);
    std::memcpy(rec_ + w.x, j.x, (size_t)j.AR * ds41aux::kEmbd * 2);
    int rc = rad_memcpy_async(ws_, rec_, w.x + (int64_t)j.AR * ds41aux::kEmbd * 2, s_);
    if (rc < 0) return retire("the record's copy to the card", rc);
    void* const x = ws_ + w.x;
    void* const so = ws_ + w.sorted;
    void* const of = ws_ + w.off;
    void* const gu = ws_ + w.gu_out;
    void* const act = ws_ + w.act;
    void* const dn = ws_ + w.dn_out;
    auto args = [&](const Launch& l, const RadTensor* t, int n_t) {
        RadArgs a{};
        a.t = t;
        a.n_t = n_t;
        a.p = l.p;
        a.n_p = l.n_p;
        a.instance = l.instance;
        a.rank = j.rank;
        a.world_size = j.world;
        return a;
    };
    {
        const RadTensor t[5] = { mk2(x, j.AR, ds41aux::kEmbd), j.gu_w, mk1(so, j.T), mk1(of, ds41aux::kExperts + 1),
                                 mk2(gu, j.T, ds41aux::kGuN) };
        const RadArgs a = args(j.gu, t, 5);
        if ((rc = j.gu.fn(&a, s_)) < 0) return retire("gate_up's launch", rc);
    }
    {
        const RadTensor t[2] = { mk2(gu, j.T, ds41aux::kGuN), mk2(act, j.T, ds41aux::kFf) };
        const RadArgs a = args(j.sw, t, 2);
        if ((rc = j.sw.fn(&a, s_)) < 0) return retire("the SwiGLU's launch", rc);
    }
    {
        const RadTensor t[5] = { mk2(act, j.T, ds41aux::kFf), j.dn_w, mk1(so, j.T), mk1(of, ds41aux::kExperts + 1),
                                 mk2(dn, j.T, ds41aux::kEmbd) };
        const RadArgs a = args(j.dn, t, 5);
        if ((rc = j.dn.fn(&a, s_)) < 0) return retire("down's launch", rc);
    }
    /* The rows back into pinned memory (a release at system scope), then the event: once a query sees it,
     * the rows are there (the record releases at system scope, the query acquires). */
    if ((rc = rad_memcpy_async(out_, dn, (int64_t)j.T * ds41aux::kEmbd * 2, s_)) < 0)
        return retire("the rows' copy back", rc);
    if ((rc = rad_event_record(ev_, s_)) < 0) return retire("the event's record", rc);
    job_ok_ = true;
    return RAD_OK;
}

int DeviceLane::wait(int64_t deadline_ns) {
    if (!pending_) return RAD_E_STATE;
    if (!job_ok_ || dead_) return RAD_E_DEVICE;
    const int64_t t0 = now_ns();
    for (int i = 0;; ++i) {
        const int q = rad_event_query(ev_);
        if (q == 1) {
            pending_ = false;
            return 1;
        }
        if (q < 0 || dev_faulted(dev_)) return retire("the job's completion", q < 0 ? q : RAD_E_DEVICE, true);
        const int64_t t = now_ns();
        if (t >= deadline_ns) return 0;
        /* As HostExpertPool::wait: a short spin, then yield. */
        if (t - t0 < 20000) _mm_pause();
        else std::this_thread::yield();
    }
}

void DeviceLane::abandon(bool backoff) {
    if (!pending_) return;
    abandoned_ = true;
    backoff_after_ = backoff;
}

bool DeviceLane::drain(int64_t timeout_ns) {
    if (dead_) return false;
    if (!s_) return true;     /* no queue was made: nothing ever ran on it */
    if (!ev_) return false;   /* a queue that cannot be probed is left to the process exit */
    const int64_t end = now_ns() + timeout_ns;
    auto poll = [&](int64_t until) {
        for (;;) {
            const int q = rad_event_query(ev_);
            if (q == 1) return 1;
            if (q < 0 || dev_faulted(dev_)) return -1;
            if (now_ns() >= until) return 0;
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
    };
    /* The job (or the loader's copies) in flight: its own event first. */
    if (pending_) {
        if (!job_ok_ || poll(end) != 1) return false;
        pending_ = false;
        abandoned_ = false;
    }
    /* Then a probe behind everything queued, an idle lane's too. The lane's last job completing says nothing
     * of the queue now: a card that hung since (another tenant, a scheduler hang) with no fault the host sees
     * would block rad_stream_destroy's host wait for ever, and only a fault ends it. The probe gets at least
     * 100 ms whatever the job took; one that does not complete leaves the lane (its queue still has work). */
    if (rad_event_record(ev_, s_) < 0) return false;
    if (poll(std::max<int64_t>(end, now_ns() + 100'000'000)) != 1) {
        pending_ = job_ok_ = abandoned_ = true;
        backoff_after_ = false;
        return false;
    }
    return true;
}

bool DeviceLane::drain_release(int64_t timeout_ns) {
    if (!dead_) return drain(timeout_ns);
    if (fault_ || !s_ || !ev_) return !s_;   /* a fault (or no event to probe with): never drained */
    /* Retired by a host-side refusal: whatever the queue took of the job (its record copy, a launch) completes
     * before a probe recorded behind it, so a completed probe means nothing is left that reads or writes the
     * arena. The lane stays retired: it takes no more jobs. */
    const int64_t end = now_ns() + std::max<int64_t>(timeout_ns, 100'000'000);
    if (rad_event_record(ev_, s_) < 0) {
        fault_ = true;
        return false;
    }
    for (;;) {
        const int q = rad_event_query(ev_);
        if (q == 1) {
            pending_ = abandoned_ = false;
            return true;
        }
        if (q < 0 || dev_faulted(dev_)) {
            fault_ = true;
            return false;
        }
        if (now_ns() >= end) return false;
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
}

int DeviceLane::sync_bounded(int64_t timeout_ns) {
    if (dead_) return RAD_E_DEVICE;
    /* The copies queued before this call are never seen complete unless the event says so. */
    const int rc = rad_event_record(ev_, s_);
    if (rc < 0) return retire("a loader wait's event record", rc);
    const int64_t end = now_ns() + timeout_ns;
    for (;;) {
        const int q = rad_event_query(ev_);
        if (q == 1) return 1;
        if (q < 0 || dev_faulted(dev_)) return retire("a loader wait", q < 0 ? q : RAD_E_DEVICE, true);
        if (now_ns() >= end) {
            /* Still queued: the lane is not idle. drain() polls this same event (2 s at teardown), and a lane it
             * does not see drain is left to the process exit with the arena its copies write. */
            pending_ = job_ok_ = abandoned_ = true;
            backoff_after_ = false;
            return 0;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
}

/* ------------------------------------------------------------------ the worker */

Worker::Worker(const Config& cfg, const ds41aux::Card& card, const std::vector<int>& ranks)
    : cfg_(cfg), card_(card), ranks_(ranks) {}

Worker::~Worker() {
    if (!finished_) (void)finish(0, 0, 0);
    if (!drained_) {
        /* A lane did not drain (finish() said so): its queue may still read the arena. */
        for (auto& l : lanes_)
            if (l) l->leak();
        lanes_.clear();
        return;
    }
    lanes_.clear();
    if (arena_) {
        /* The per-device free convention: the arena's own card bound (as ~Engine frees per rank). */
        (void)rad_dev_set(card_.ord);
        rad_dev_free(arena_, RAD_MEM_DEVICE);
        arena_ = nullptr;
        if (!ranks_.empty()) (void)rad_dev_set(ranks_[0]);
    }
}

int Worker::reserve(int restore_device, std::string* why) {
    if (arena_) return RAD_OK;   /* held already */
    arena_bytes_ = cfg_.cap_mib << 20;
    if (rad_dev_set(card_.ord) < 0) {
        *why = fmt("device %d could not be selected: %s", card_.ord, rad_dev_last_error());
        (void)rad_dev_set(restore_device);
        return RAD_E_DEVICE;
    }
    arena_ = (char*)rad_dev_alloc(arena_bytes_, RAD_MEM_DEVICE);
    if (!arena_) {
        *why = fmt("the arena of %lld MiB on device %d was refused: %s", (long long)cfg_.cap_mib, card_.ord,
                   rad_dev_last_error());
        (void)rad_dev_set(restore_device);
        return RAD_E_NOMEM;
    }
    RadDeviceProps p{};
    const int64_t free_after = rad_dev_props(card_.ord, &p) >= 0 ? p.vram_free : -1;
    free_after_.store(free_after, std::memory_order_relaxed);
    if (free_after < (cfg_.headroom_mib << 20)) {
        *why = fmt("after the arena of %lld MiB device %d has %s free, under the headroom of %lld MiB",
                   (long long)cfg_.cap_mib, card_.ord, humanb(free_after < 0 ? 0 : free_after).c_str(),
                   (long long)cfg_.headroom_mib);
        rad_dev_free(arena_, RAD_MEM_DEVICE);
        arena_ = nullptr;
        (void)rad_dev_set(restore_device);
        return RAD_E_NOMEM;
    }
    arena_present_.store(true, std::memory_order_release);
    (void)rad_dev_set(restore_device);
    return RAD_OK;
}

int Worker::build(int layers, int64_t slots, const ds41aux::Layout& L, int restore_device, std::string* why) {
    const int R = (int)ranks_.size();
    views_.assign((size_t)R, RankView{});
    for (int r = 0; r < R; ++r) {
        views_[(size_t)r].gu_tab.assign((size_t)(layers > 0 ? layers : 0), nullptr);
        views_[(size_t)r].dn_tab.assign((size_t)(layers > 0 ? layers : 0), nullptr);
    }
    const int rc = carve(layers, slots, L, why);
    if (rc < 0) return rc;
    return make_lanes(restore_device, why);
}

int Worker::carve(int layers, int64_t slots, const ds41aux::Layout& L, std::string* why) {
    const int R = (int)ranks_.size();
    const ds41aux::Ws w = ds41aux::ws((int)cfg_.rows);
    const int64_t tb = ds41aux::tables_bytes(R, layers);
    if (!arena_ || layers <= 0 || slots < 0 || tb + R * w.bytes + slots * L.stride > arena_bytes_) {
        *why = fmt("the arena of %s does not hold the tables (%s), %d lanes (%s) and %lld slots of %s",
                   humanb(arena_bytes_).c_str(), humanb(tb).c_str(), R, humanb(R * w.bytes).c_str(),
                   (long long)slots, humanb(L.stride).c_str());
        return RAD_E_INVAL;
    }
    layers_ = layers;
    layout_ = L;
    tables_ = arena_;
    ws_ = arena_ + tb;
    slots_ = ws_ + R * w.bytes;
    n_slots_ = slots;
    if (views_.size() != (size_t)R) views_.assign((size_t)R, RankView{});
    for (int r = 0; r < R; ++r) {
        views_[(size_t)r].gu_tab.resize((size_t)layers, nullptr);
        views_[(size_t)r].dn_tab.resize((size_t)layers, nullptr);
    }
    for (size_t r = 0; r < lanes_.size(); ++r)
        if (lanes_[r]) lanes_[r]->set_ws(ws_ + (int64_t)r * w.bytes);
    return RAD_OK;
}

void Worker::plan(int layers, int64_t slots, const ds41aux::Layout& L) {
    const int R = (int)ranks_.size();
    layers_ = layers;
    layout_ = L;
    n_slots_ = slots;
    views_.assign((size_t)R, RankView{});
    for (int r = 0; r < R; ++r) {
        views_[(size_t)r].gu_tab.assign((size_t)(layers > 0 ? layers : 0), nullptr);
        views_[(size_t)r].dn_tab.assign((size_t)(layers > 0 ? layers : 0), nullptr);
    }
}

int Worker::make_lanes(int restore_device, std::string* why) {
    const int R = (int)ranks_.size();
    const ds41aux::Ws w = ds41aux::ws((int)cfg_.rows);
    if (rad_dev_set(card_.ord) < 0) {
        *why = fmt("device %d could not be selected: %s", card_.ord, rad_dev_last_error());
        (void)rad_dev_set(restore_device);
        return RAD_E_DEVICE;
    }
    int rc = RAD_OK;
    for (int r = 0; r < R && rc >= 0; ++r) {
        lanes_.emplace_back(new DeviceLane(card_.ord, r, ws_ ? ws_ + r * w.bytes : nullptr, cfg_.backoff,
                                           (int)cfg_.rows));
        rc = lanes_.back()->init(why);
    }
    /* Later bring-up takes the current device (pinned memory's NUMA node, among others): rank 0's again. */
    (void)rad_dev_set(restore_device);
    return rc;
}

void Worker::release_arena(int restore_device) {
    for (auto& l : lanes_)
        if (l) l->set_ws(nullptr);
    if (arena_) {
        /* The per-device free convention: the arena's own card bound (as ~Worker). */
        (void)rad_dev_set(card_.ord);
        rad_dev_free(arena_, RAD_MEM_DEVICE);
        if (restore_device >= 0) (void)rad_dev_set(restore_device);
    }
    arena_ = tables_ = ws_ = slots_ = nullptr;
    free_after_.store(-1, std::memory_order_relaxed);
    arena_present_.store(false, std::memory_order_release);
}

char* Worker::table(int r, int li, int which) const {
    return tables_ + (((int64_t)r * layers_ + li) * 2 + which) * ds41aux::kExperts * 8;
}
Lane* Worker::lane(int r) const { return r >= 0 && (size_t)r < lanes_.size() ? lanes_[(size_t)r].get() : nullptr; }
DeviceLane* Worker::device_lane(int r) const {
    return r >= 0 && (size_t)r < lanes_.size() ? lanes_[(size_t)r].get() : nullptr;
}
int64_t Worker::units() const {
    int64_t n = 0;
    for (const RankView& v : views_) n += v.units;
    return n;
}

namespace {
/* The views' entries into a table image (load_tables, table_image). */
void image_from_views(const std::vector<RankView>& views, int layers, uint64_t* host) {
    for (size_t r = 0; r < views.size(); ++r)
        for (int li = 0; li < layers; ++li)
            for (int which = 0; which < 2; ++which) {
                const std::vector<const void*>& src = which ? views[r].dn_tab_entries : views[r].gu_tab_entries;
                uint64_t* dst = host + (((int64_t)r * layers + li) * 2 + which) * ds41aux::kExperts;
                for (int e = 0; e < ds41aux::kExperts; ++e) {
                    const size_t k = (size_t)li * ds41aux::kExperts + (size_t)e;
                    dst[e] = k < src.size() ? (uint64_t)(uintptr_t)src[k] : 0;
                }
            }
}
}  // namespace

std::vector<uint64_t> Worker::table_image() const {
    std::vector<uint64_t> img((size_t)(ds41aux::tables_bytes((int)ranks_.size(), layers_) / 8), 0);
    image_from_views(views_, layers_, img.data());
    return img;
}

int Worker::load_tables(std::string* why) {
    return load_tables_with([](const Worker* w, const void*, uint64_t* host) {
        image_from_views(w->views_, w->layers_, host);
    }, nullptr, why);
}

int Worker::load_tables_image(const uint64_t* image, std::string* why) {
    return load_tables_with([](const Worker* w, const void* ctx, uint64_t* host) {
        std::memcpy(host, ctx, (size_t)((int64_t)w->ranks_.size() * w->layers_ * 2 * ds41aux::kExperts * 8));
    }, image, why);
}

int Worker::load_tables_with(void (*fill)(const Worker*, const void*, uint64_t*), const void* ctx, std::string* why) {
    const int R = (int)ranks_.size();
    const int64_t bytes = ds41aux::tables_bytes(R, layers_);
    DeviceLane* l = device_lane(0);
    if (!l) {
        *why = "no lane";
        return RAD_E_STATE;
    }
    uint64_t* host = (uint64_t*)dev_alloc_pinned_on(bytes, { card_.ord });
    uint64_t* back = (uint64_t*)dev_alloc_pinned_on(bytes, { card_.ord });
    int rc = host && back ? RAD_OK : RAD_E_NOMEM;
    bool in_flight = false;   /* copies queued and not seen complete: host and back stay theirs */
    if (rc < 0) *why = fmt("the tables' pinned staging: %s", rad_dev_last_error());
    if (rc >= 0) {
        std::memset(host, 0, (size_t)bytes);
        fill(this, ctx, host);
        for (int r = 0; r < R; ++r)
            for (int li = 0; li < layers_; ++li)
                for (int which = 0; which < 2; ++which)
                    (which ? views_[(size_t)r].dn_tab : views_[(size_t)r].gu_tab)[(size_t)li] = table(r, li, which);
        std::memset(back, 0xa5, (size_t)bytes);
        int queued = 0;
        if ((rc = rad_memcpy_async(tables_, host, bytes, l->stream())) >= 0) {
            ++queued;
            if ((rc = rad_memcpy_async(back, tables_, bytes, l->stream())) >= 0) ++queued;
        }
        const std::string qerr = rc < 0 ? rad_dev_last_error() : "";
        /* A bounded wait for whatever was queued (both copies, or the first before the second was refused). */
        const int synced = queued ? l->sync_bounded(load_wait_ns_) : 1;
        in_flight = synced != 1;
        if (rc < 0) {
            *why = fmt("the tables' copies could not be queued: %s", qerr.c_str());
        } else if (synced != 1) {
            *why = synced == 0 ? fmt("the tables' copies did not complete in %g s", (double)load_wait_ns_ * 1e-9)
                               : std::string("the tables' copies failed");
            rc = RAD_E_DEVICE;
        } else if (std::memcmp(host, back, (size_t)bytes) != 0) {
            *why = "the tables read back differ from what was written";
            rc = RAD_E_STATE;
        } else {
            rc = RAD_OK;
        }
        if (in_flight) *why += " (their pinned staging is left to the process exit)";
    }
    /* Freed only once their copies were seen complete: a queue that still holds them would read and write
     * pages given back (work/aux/SPEC.md §7.5). */
    if (!in_flight) {
        if (host) rad_dev_free(host, RAD_MEM_HOST_PINNED);
        if (back) rad_dev_free(back, RAD_MEM_HOST_PINNED);
    }
    return rc;
}

int Worker::check_headroom(std::string* why) const {
    RadDeviceProps p{};
    const int64_t f = rad_dev_props(card_.ord, &p) >= 0 ? p.vram_free : -1;
    if (f >= (cfg_.headroom_mib << 20)) return RAD_OK;
    *why = fmt("device %d has %s free once the lanes loaded their code objects, under the headroom of %lld MiB",
               card_.ord, humanb(f < 0 ? 0 : f).c_str(), (long long)cfg_.headroom_mib);
    return RAD_E_NOMEM;
}

std::string Worker::finish(uint64_t jobs, uint64_t rows, uint64_t late) {
    finished_ = true;
    int stuck = 0;
    for (auto& l : lanes_)
        if (l && !(release_drain_ ? l->drain_release(2'000'000'000LL) : l->drain(2'000'000'000LL))) ++stuck;
    drained_ = stuck == 0 && !swap_undrained_;
    const int64_t held = units_held_.load(std::memory_order_relaxed);
    return fmt("%s%d (%s): %lld units held; %llu jobs, %llu rows over all ranks; %llu deadline misses; %s",
               ds41aux::kLogDevice, card_.ord, card_.bdf.c_str(), (long long)(held >= 0 ? held : units()),
               (unsigned long long)jobs, (unsigned long long)rows, (unsigned long long)late,
               drained_ ? "lanes drained at exit"
               : swap_undrained_ && !stuck
                   ? "the swap queue not drained in 2 s: its buffers and the arena are left to the process exit"
                   : fmt("%d lane(s) not drained in 2 s (or dead)%s: their queues, events and the arena are left to "
                         "the process exit", stuck, swap_undrained_ ? ", nor the swap queue" : "").c_str());
}

}  // namespace aux
}  // namespace rad
