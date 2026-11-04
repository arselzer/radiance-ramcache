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

std::string stats_line(int r, const Stats& s) {
    const double rt = s.rt_n ? (double)s.rt_sum_ns / (double)s.rt_n * 1e-3 : 0.0;
    return fmt("%s%d: %llu layer jobs, %llu units, %llu rows on the extra card of %llu rows the CPU path met in passes "
               "of up to %d rows; declined: %llu taller, %llu draft, %llu without usable kernel rows, %llu while the "
               "lane was busy or backing off, %llu after it failed; %llu worker units not cached (CPU miss path); "
               "waited %.1f ms beyond the CPU's rows, round trip %.1f us mean, %.1f us max; %llu deadline misses "
               "(%llu forced; %llu units, %llu rows recomputed on the CPU in %.1f ms), %llu launch failures; verify: "
               "%llu rows compared, %llu differ (%llu values), %llu not compared",
               ds41aux::kLogRank, r, (unsigned long long)s.jobs, (unsigned long long)s.units,
               (unsigned long long)s.placed, (unsigned long long)s.eligible_rows, ds41aux::kMaxRows,
               (unsigned long long)s.declined_tall, (unsigned long long)s.declined_draft,
               (unsigned long long)s.declined_rows, (unsigned long long)s.declined_busy,
               (unsigned long long)s.declined_dead, (unsigned long long)s.uncached, (double)s.wait_ns * 1e-6, rt,
               (double)s.rt_max_ns * 1e-3, (unsigned long long)s.late, (unsigned long long)s.forced_late,
               (unsigned long long)s.fallback_units, (unsigned long long)s.fallback_rows,
               (double)s.fallback_ns * 1e-6, (unsigned long long)s.failed, (unsigned long long)s.verified,
               (unsigned long long)s.differ_rows, (unsigned long long)s.differ_values,
               (unsigned long long)s.not_compared);
}

/* ------------------------------------------------------------------ the lane */

DeviceLane::DeviceLane(int device, int rank, char* ws, int64_t backoff)
    : dev_(device), rank_(rank), ws_(ws), backoff_(backoff) {}

DeviceLane::~DeviceLane() {
    if (leaked_) return;   /* a job may still run on the queue: the process exit reclaims it all */
    if (ev_) rad_event_destroy(ev_);
    if (s_) rad_stream_destroy(s_);
    if (rec_) rad_dev_free(rec_, RAD_MEM_HOST_PINNED);
    if (out_) rad_dev_free(out_, RAD_MEM_HOST_PINNED);
}

int DeviceLane::init(std::string* why) {
    const ds41aux::Ws w = ds41aux::ws();
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

int DeviceLane::retire(const char* step, int rc) {
    job_ok_ = false;
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
            retire("an abandoned job", q < 0 ? q : RAD_E_DEVICE);
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
    const ds41aux::Ws w = ds41aux::ws();
    if (j.AR < 1 || j.AR > ds41aux::kMaxRows || j.T < 1 || j.T > ds41aux::kMaxSlots || !j.x || !j.sorted || !j.off ||
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
        if (q < 0 || dev_faulted(dev_)) return retire("the job's completion", q < 0 ? q : RAD_E_DEVICE);
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
    if (!pending_) return true;
    if (!job_ok_) return false;
    const int64_t end = now_ns() + timeout_ns;
    for (;;) {
        const int q = rad_event_query(ev_);
        if (q == 1) {
            pending_ = false;
            return true;
        }
        if (q < 0 || dev_faulted(dev_) || now_ns() >= end) return false;
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
}

int DeviceLane::sync_bounded(int64_t timeout_ns) {
    if (dead_) return RAD_E_DEVICE;
    if (rad_event_record(ev_, s_) < 0) return RAD_E_DEVICE;
    const int64_t end = now_ns() + timeout_ns;
    for (;;) {
        const int q = rad_event_query(ev_);
        if (q == 1) return 1;
        if (q < 0 || dev_faulted(dev_)) return RAD_E_DEVICE;
        if (now_ns() >= end) return 0;
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
    free_after_ = rad_dev_props(card_.ord, &p) >= 0 ? p.vram_free : -1;
    if (free_after_ < (cfg_.headroom_mib << 20)) {
        *why = fmt("after the arena of %lld MiB device %d has %s free, under the headroom of %lld MiB",
                   (long long)cfg_.cap_mib, card_.ord, humanb(free_after_ < 0 ? 0 : free_after_).c_str(),
                   (long long)cfg_.headroom_mib);
        rad_dev_free(arena_, RAD_MEM_DEVICE);
        arena_ = nullptr;
        (void)rad_dev_set(restore_device);
        return RAD_E_NOMEM;
    }
    (void)rad_dev_set(restore_device);
    return RAD_OK;
}

int Worker::build(int layers, int64_t slots, const ds41aux::Layout& L, int restore_device, std::string* why) {
    const int R = (int)ranks_.size();
    const ds41aux::Ws w = ds41aux::ws();
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
    views_.assign((size_t)R, RankView{});
    for (int r = 0; r < R; ++r) {
        views_[(size_t)r].gu_tab.assign((size_t)layers, nullptr);
        views_[(size_t)r].dn_tab.assign((size_t)layers, nullptr);
    }
    if (rad_dev_set(card_.ord) < 0) {
        *why = fmt("device %d could not be selected: %s", card_.ord, rad_dev_last_error());
        (void)rad_dev_set(restore_device);
        return RAD_E_DEVICE;
    }
    int rc = RAD_OK;
    for (int r = 0; r < R && rc >= 0; ++r) {
        lanes_.emplace_back(new DeviceLane(card_.ord, r, ws_ + r * w.bytes, cfg_.backoff));
        rc = lanes_.back()->init(why);
    }
    /* Later bring-up takes the current device (pinned memory's NUMA node, among others): rank 0's again. */
    (void)rad_dev_set(restore_device);
    return rc;
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

int Worker::load_tables(std::string* why) {
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
    if (rc < 0) *why = fmt("the tables' pinned staging: %s", rad_dev_last_error());
    if (rc >= 0) {
        std::memset(host, 0, (size_t)bytes);
        for (int r = 0; r < R; ++r)
            for (int li = 0; li < layers_; ++li)
                for (int which = 0; which < 2; ++which) {
                    const std::vector<const void*>& src = which ? views_[(size_t)r].dn_tab_entries
                                                                : views_[(size_t)r].gu_tab_entries;
                    uint64_t* dst = host + (((int64_t)r * layers_ + li) * 2 + which) * ds41aux::kExperts;
                    for (int e = 0; e < ds41aux::kExperts; ++e) {
                        const size_t k = (size_t)li * ds41aux::kExperts + (size_t)e;
                        dst[e] = k < src.size() ? (uint64_t)(uintptr_t)src[k] : 0;
                    }
                    (which ? views_[(size_t)r].dn_tab : views_[(size_t)r].gu_tab)[(size_t)li] = table(r, li, which);
                }
        std::memset(back, 0xa5, (size_t)bytes);
        if ((rc = rad_memcpy_async(tables_, host, bytes, l->stream())) < 0 ||
            (rc = rad_memcpy_async(back, tables_, bytes, l->stream())) < 0)
            *why = fmt("the tables' copies: %s", rad_dev_last_error());
        else if ((rc = l->sync_bounded(10'000'000'000LL)) != 1) {
            *why = rc == 0 ? "the tables' copies did not complete in 10 s" : "the tables' copies failed";
            rc = RAD_E_DEVICE;
        } else if (std::memcmp(host, back, (size_t)bytes) != 0) {
            *why = "the tables read back differ from what was written";
            rc = RAD_E_STATE;
        } else {
            rc = RAD_OK;
        }
    }
    if (host) rad_dev_free(host, RAD_MEM_HOST_PINNED);
    if (back) rad_dev_free(back, RAD_MEM_HOST_PINNED);
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
        if (l && !l->drain(2'000'000'000LL)) ++stuck;
    drained_ = stuck == 0;
    return fmt("%s%d (%s): %lld units held; %llu jobs, %llu rows over all ranks; %llu deadline misses; %s",
               ds41aux::kLogDevice, card_.ord, card_.bdf.c_str(), (long long)units(), (unsigned long long)jobs,
               (unsigned long long)rows, (unsigned long long)late,
               drained_ ? "lanes drained at exit"
                        : fmt("%d lane(s) not drained in 2 s (or dead): their queues, events and the arena are left to "
                              "the process exit", stuck).c_str());
}

}  // namespace aux
}  // namespace rad
