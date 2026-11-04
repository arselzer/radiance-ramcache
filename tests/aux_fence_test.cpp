/* aux_fence_test.cpp -- the aux worker's control thread end to end on the host backend (work/aux-dyn/SPEC.md §7.3):
 * the real aux::Control, aux::Worker and DeviceLanes on the backend's fifth fake device, the probe on a fake sysfs,
 * the RAM cache and the stagers replaced by hooks. Four rank threads run passes in lockstep with a scheduler thread
 * that applies the step boundary; each job's fake gate_up launch copies, when the queue executes it, every listed
 * expert's two table entries and the first word of its slot's gate_up and down blocks, so a job that read a slot or
 * an entry being rewritten, or an arena already freed, shows. Meanwhile the online set swaps units (16 a token, every
 * copy read back) and the control file turns the worker off and on, releasing and reacquiring the arena; some jobs
 * are held late at a stream gate across those transitions.
 *  - every job's entries name the slot its view gave and every slot holds its unit's words: no swap, release or
 *    reacquire touched what a job could still read; the views change only at boundaries;
 *  - the control thread swapped, released and reacquired, read every copy back equal, and never went dead;
 *  - the gate: a release waits for a rank inside a job and completes once it leaves, the state file saying so;
 *  - online alone (no controller, no gate): swaps and flips with the lanes polled at the boundary. */
#include "rad_test.h"

#include "rad_internal.h"
#include "device/device.h"
#include "place/aux_control.h"
#include "../tools/ds41-host-experts/ds41_p1.h"

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <barrier>
#include <chrono>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

const int g_devices = [] {
    ::setenv("RADIANCE_HOST_DEVICES", "5", 1);
    return 5;
}();

constexpr int kR = 4, kLayers = 2, kExp = 20, kPlan = kLayers * kExp;
constexpr int64_t kGu = 4096, kDn = 4096;

/* A check whose failure ends the binary (in a rank thread or a constructor, where REQUIRE's return would hang the
 * lockstep or leave a half-built rig). */
#define MUST(cond)                                                                           \
    do {                                                                                     \
        if (!(cond)) {                                                                       \
            std::fprintf(stderr, "%s:%d: MUST(%s) failed\n", __FILE__, __LINE__, #cond);    \
            std::abort();                                                                    \
        }                                                                                    \
    } while (0)

bool contains(const std::string& s, const std::string& p) { return s.find(p) != std::string::npos; }

int64_t now_ns() {
    return (int64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}
uint64_t tag(int r, int u, int which) { return 0xA500000000000000ull | (uint64_t)r << 40 | (uint64_t)u << 8 | (uint64_t)which; }

void put(const std::string& path, const std::string& text) {
    const size_t k = path.rfind('/');
    const std::string dir = path.substr(0, k);
    for (size_t i = 1; i <= dir.size(); ++i)
        if (i == dir.size() || dir[i] == '/') ::mkdir(dir.substr(0, i).c_str(), 0777);
    std::ofstream f(path, std::ios::trunc);
    f << text;
}
std::string slurp(const std::string& path) {
    std::ifstream f(path);
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return s;
}

/* The units' packed bytes: a gate_up and a down block per (rank, unit), each starting with its tag. */
struct Bytes {
    std::vector<std::vector<uint8_t>> gu, dn;
    Bytes() {
        gu.resize(kR * kPlan);
        dn.resize(kR * kPlan);
        for (int r = 0; r < kR; ++r)
            for (int u = 0; u < kPlan; ++u) {
                std::vector<uint8_t>& g = gu[(size_t)(r * kPlan + u)];
                std::vector<uint8_t>& d = dn[(size_t)(r * kPlan + u)];
                g.assign(kGu, (uint8_t)(r * 7 + u));
                d.assign(kDn, (uint8_t)(r * 11 + u));
                const uint64_t t0 = tag(r, u, 0), t1 = tag(r, u, 1);
                std::memcpy(g.data(), &t0, 8);
                std::memcpy(d.data(), &t1, 8);
            }
    }
};
Bytes& bytes() {
    static Bytes b;
    return b;
}

/* What a job's fake gate_up launch copies when the lane's queue executes it. */
struct JobCtx {
    int      n = 0;
    int      e[ds41aux::kMaxSlots] = {};
    char*    slot[ds41aux::kMaxSlots] = {};
    uint64_t want_entry[ds41aux::kMaxSlots] = {};
    uint64_t want_tag[ds41aux::kMaxSlots] = {};
    uint64_t* got = nullptr;   /* pinned on the extra card: n x {gu entry, dn entry, gu tag, dn tag} */
    int      r = -1;
};
thread_local JobCtx* t_ctx = nullptr;
int fake_gu(const RadArgs* a, RadStream s) {
    JobCtx* c = t_ctx;
    const char* gu_tab = (const char*)a->t[1].data;
    for (int i = 0; i < c->n; ++i) {
        int rc = rad_memcpy_async(&c->got[i * 4 + 0], gu_tab + (int64_t)c->e[i] * 8, 8, s);
        if (rc >= 0) rc = rad_memcpy_async(&c->got[i * 4 + 2], c->slot[i], 8, s);
        if (rc < 0) return rc;
    }
    return RAD_OK;
}
int fake_sw(const RadArgs*, RadStream) { return RAD_OK; }
int fake_dn(const RadArgs* a, RadStream s) {
    JobCtx* c = t_ctx;
    const char* dn_tab = (const char*)a->t[1].data;
    const int64_t dn_off = ds41aux::layout(kGu, kDn).dn_off;
    for (int i = 0; i < c->n; ++i) {
        int rc = rad_memcpy_async(&c->got[i * 4 + 1], dn_tab + (int64_t)c->e[i] * 8, 8, s);
        if (rc >= 0) rc = rad_memcpy_async(&c->got[i * 4 + 3], c->slot[i] + dn_off, 8, s);
        if (rc < 0) return rc;
    }
    return RAD_OK;
}
/* A job's checks: n entries and tags as the view said at its submit. */
int job_errors(const JobCtx& c, std::string* first) {
    int bad = 0;
    for (int i = 0; i < c.n; ++i) {
        const bool ok = c.got[i * 4 + 0] == c.want_entry[i] && c.got[i * 4 + 2] == c.want_tag[i] &&
                        c.got[i * 4 + 3] == (c.want_tag[i] | 1);
        if (!ok && first->empty())
            *first = rad::fmt("rank %d expert %d: entry 0x%llx (want 0x%llx), tags 0x%llx / 0x%llx (want 0x%llx)", c.r,
                              c.e[i], (unsigned long long)c.got[i * 4 + 0], (unsigned long long)c.want_entry[i],
                              (unsigned long long)c.got[i * 4 + 2], (unsigned long long)c.got[i * 4 + 3],
                              (unsigned long long)c.want_tag[i]);
        bad += !ok;
    }
    return bad;
}

/* The fake card's sysfs: the extra card at 0000:c3:00.0 with 32 GiB, 1 GiB used. */
rad::aux::Roots fake_roots(const std::string& name) {
    rad::aux::Roots r;
    const std::string root = radtest::tmp_path(name);
    r.sysfs = root + "/sys";
    r.procfs = root + "/proc";
    put(r.sysfs + "/class/kfd/kfd/topology/nodes/5/properties",
        "gfx_target_version 120001\ndomain 0\nlocation_id " + std::to_string(0xc3 << 8) + "\nunique_id 170\n");
    put(r.sysfs + "/class/kfd/kfd/topology/nodes/5/gpu_id", "50005\n");
    put(r.sysfs + "/bus/pci/devices/0000:c3:00.0/mem_info_vram_total", std::to_string(32LL << 30) + "\n");
    put(r.sysfs + "/bus/pci/devices/0000:c3:00.0/mem_info_vram_used", std::to_string(1LL << 30) + "\n");
    put(r.procfs + "/" + std::to_string((long)::getpid()) + "/stat", "1 (t) S 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4242\n");
    return r;
}

/* The fixture: a worker with the startup set loaded, its control thread attached, the hooks' state. */
struct Rig {
    rad::aux::Config cfg;
    ds41aux::Card card;
    ds41aux::Layout L = ds41aux::layout(kGu, kDn);
    std::unique_ptr<rad::aux::Worker> w;
    std::unique_ptr<rad::aux::Control> ctl;
    std::vector<std::vector<uint32_t>> hits = std::vector<std::vector<uint32_t>>(kR, std::vector<uint32_t>(kPlan, 0));
    std::vector<uint64_t> tok = std::vector<uint64_t>(kR, 0);
    std::atomic<int> warmups{ 0 }, leases{ 0 };
    int64_t slots = 56, quota = 12;
    std::string ctl_path;

    explicit Rig(bool controller, const std::string& name) {
        cfg.on = true;
        cfg.cap_mib = 16;
        cfg.headroom_mib = 0;
        cfg.deadline_us = 2000;
        cfg.backoff = 0;
        if (controller) {
            ctl_path = radtest::tmp_path(name + "-ctl");
            cfg.control = ctl_path;
            cfg.poll_ms = 50;
            cfg.release_polls = 1;
            cfg.reacquire_s = 0;
            cfg.fill_mbps = 20000;
        }
        cfg.online = true;
        cfg.rate = 1600;
        cfg.omax = 64;
        cfg.halflife = 16;
        cfg.prior = 0;
        cfg.spare = 8;
        cfg.mbps = 20000;
        cfg.check = 1;
        card.ord = 4;
        card.bdf = "0000:c3:00.0";
        card.name = "fake";
        w.reset(new rad::aux::Worker(cfg, card, { 0, 1, 2, 3 }));
        std::string why;
        MUST((w->reserve(0, &why)) >= 0);
        MUST((w->build(kLayers, slots, L, 0, &why)) >= 0);
        /* The startup set: each rank's first `quota` candidates (unit order is the share order here). */
        for (int r = 0; r < kR; ++r) {
            rad::aux::RankView& v = w->view(r);
            v.resident.assign(kPlan, 0);
            v.slot.assign(kPlan, -1);
            v.gu_tab_entries.assign((size_t)kLayers * ds41aux::kExperts, nullptr);
            v.dn_tab_entries.assign((size_t)kLayers * ds41aux::kExperts, nullptr);
            for (int u = 0; u < quota; ++u) {
                const int64_t k = r * quota + u;
                char* s = w->slot(k);
                std::memcpy(s, bytes().gu[(size_t)(r * kPlan + u)].data(), kGu);
                std::memcpy(s + L.dn_off, bytes().dn[(size_t)(r * kPlan + u)].data(), kDn);
                v.resident[(size_t)u] = 1;
                v.slot[(size_t)u] = (int32_t)k;
                const size_t at = (size_t)(u / kExp) * ds41aux::kExperts + (size_t)(u % kExp);
                v.gu_tab_entries[at] = ds41p1::mark(s, ds41p1::kMarkPacked);
                v.dn_tab_entries[at] = ds41p1::mark(s + L.dn_off, ds41p1::kMarkPacked);
                ++v.units;
                v.share += (double)(kPlan - u) * 1e-3;
            }
        }
        MUST((w->load_tables(&why)) >= 0);
        ctl.reset(new rad::aux::Control(cfg, card, kR));
        if (controller) MUST((ctl->init_share(fake_roots(name), (long)::getpid(), &why)) >= 0);
        std::vector<rad::aux::RankUnits> units(kR);
        for (int r = 0; r < kR; ++r) {
            units[(size_t)r].n_plan_units = kPlan;
            units[(size_t)r].quota = quota;
            for (int u = 0; u < kPlan; ++u) {
                ds41aux::Cand c;
                c.rank = r;
                c.unit = u;
                c.share = (float)(kPlan - u) * 1e-3f;
                c.expert = u % kExp;
                c.layer = u / kExp;
                units[(size_t)r].cands.push_back(c);
                units[(size_t)r].li.push_back(u / kExp);
            }
        }
        rad::aux::Hooks h;
        h.source = [this](int r, int32_t u, rad::aux::Hooks::Source* s) {
            ++leases;
            s->gu = bytes().gu[(size_t)(r * kPlan + u)].data();
            s->dn = bytes().dn[(size_t)(r * kPlan + u)].data();
            s->gu_n = kGu;
            s->dn_n = kDn;
            return true;
        };
        h.release = [](int, rad::aux::Hooks::Source* s) { s->lease = nullptr; };
        h.warm = [this](int r, int32_t, int32_t li, const void* g, const void* d, std::string* why) {
            ++warmups;
            if (g != w->table(r, li, 0) || d != w->table(r, li, 1)) {
                *why = "the warm-up's tables are not the new arena's";
                return RAD_E_STATE;
            }
            return RAD_OK;
        };
        h.hit_tokens = [this](int r) { return tok[(size_t)r]; };
        h.swap_hits = [this](int r, std::vector<uint32_t>* e) {
            if (e->size() != hits[(size_t)r].size()) return (uint64_t)0;
            e->swap(hits[(size_t)r]);
            const uint64_t t = tok[(size_t)r];
            tok[(size_t)r] = 0;
            return t;
        };
        h.guard_tripped = [](int) { return false; };
        h.kfd_available = [] { return (int64_t)-1; };
        MUST((ctl->attach(w.get(), std::move(h), std::move(units), kLayers, slots, L, false, "", &why)) >= 0);
    }
    void mode(const char* m) { MUST(ds41aux::share::write_atomic(ctl_path, std::string(m) + "\n")); }
    std::string state() { return slurp(ctl_path + ".state"); }
};

/* Four rank threads and a scheduler in lockstep for `seconds`; `tick` runs on the scheduler between passes. */
struct Run {
    int errors = 0, jobs = 0, late = 0, declined = 0, view_changes = 0;
    std::string first;
};
Run run_passes(Rig& g, double seconds, uint32_t seed, const std::function<void(int64_t)>& tick) {
    Run out;
    std::mutex mu;
    std::atomic<bool> done{ false };
    std::barrier<> bar(kR + 1);
    struct Held { void* gate; int64_t open_at; };
    std::vector<Held> held;
    std::vector<std::unique_ptr<JobCtx>> lates;
    std::vector<uint64_t*> pinned;
    auto alloc_got = [&]() {
        uint64_t* p = (uint64_t*)rad::dev_alloc_pinned_on(ds41aux::kMaxSlots * 4 * 8, { 4 });
        std::lock_guard<std::mutex> lk(mu);
        pinned.push_back(p);
        return p;
    };
    std::vector<std::thread> ranks;
    for (int r = 0; r < kR; ++r)
        ranks.emplace_back([&, r] {
            std::mt19937 rng(seed * 31 + (uint32_t)r);
            std::vector<uint16_t> x(ds41aux::kEmbd, 0x3c00);
            int32_t sorted[ds41aux::kMaxSlots], off[ds41aux::kExperts + 1];
            rad::aux::Gate* gate = g.ctl->gate();
            for (;;) {
                bar.arrive_and_wait();   /* the boundary ran */
                if (done.load()) break;
                const rad::aux::RankView& v = g.w->view(r);
                const int64_t units0 = v.units;
                g.tok[(size_t)r] += 1;
                for (int li = 0; li < kLayers; ++li) {
                    std::vector<int> picked;
                    for (int k = 0; k < 3; ++k) {
                        const int u = li * kExp + (int)(rng() % (uint32_t)(r % 2 ? kExp : 8 + (seed % 12)));
                        if (std::find(picked.begin(), picked.end(), u) == picked.end()) picked.push_back(u);
                    }
                    for (int u : picked) g.hits[(size_t)r][(size_t)u] += 1;
                    std::vector<int> res;
                    for (int u : picked)
                        if (v.is_resident(u)) res.push_back(u);
                    if (res.empty()) continue;
                    if (gate && !gate->enter(r)) {
                        std::lock_guard<std::mutex> lk(mu);
                        ++out.declined;
                        continue;
                    }
                    rad::aux::Lane* lane = g.w->lane(r);
                    if (lane->state() == rad::aux::Lane::kReady) {
                        auto c = std::make_unique<JobCtx>();
                        c->r = r;
                        c->got = alloc_got();
                        std::sort(res.begin(), res.end(), [](int a, int b) { return a % kExp < b % kExp; });
                        for (int u : res) {
                            const int i = c->n++;
                            c->e[i] = u % kExp;
                            c->slot[i] = g.w->slot(v.slot[(size_t)u]);
                            c->want_entry[i] = (uint64_t)(uintptr_t)ds41p1::mark(c->slot[i], ds41p1::kMarkPacked);
                            c->want_tag[i] = tag(r, u, 0);
                        }
                        for (int i = 0; i < c->n; ++i) sorted[i] = i;
                        for (int e = 0; e <= ds41aux::kExperts; ++e) off[e] = 0;
                        rad::aux::Job j;
                        const RadParam params[1] = {};
                        j.AR = 1;
                        j.T = c->n;
                        j.x = x.data();
                        j.sorted = sorted;
                        j.off = off;
                        j.gu = { fake_gu, params, 1, nullptr };
                        j.sw = { fake_sw, params, 1, nullptr };
                        j.dn = { fake_dn, params, 1, nullptr };
                        j.gu_w.data = const_cast<void*>(v.gu_tab[(size_t)li]);
                        j.dn_w.data = const_cast<void*>(v.dn_tab[(size_t)li]);
                        j.rank = r;
                        j.world = kR;
                        void* sg = nullptr;
                        const bool hold = rng() % 40 == 0;
                        if (hold) MUST(rad_stream_gate(g.w->device_lane(r)->stream(), &sg) >= 0);
                        t_ctx = c.get();
                        MUST(lane->submit(j) >= 0);
                        t_ctx = nullptr;
                        const int got = lane->wait(now_ns() + (hold ? 100'000 : 2'000'000'000LL));
                        std::lock_guard<std::mutex> lk(mu);
                        ++out.jobs;
                        if (got == 1) {
                            out.errors += job_errors(*c, &out.first);
                        } else {
                            lane->abandon(false);
                            ++out.late;
                            lates.push_back(std::move(c));
                            if (sg) held.push_back({ sg, now_ns() + (int64_t)(rng() % 20'000'000) });
                        }
                    } else {
                        std::lock_guard<std::mutex> lk(mu);
                        ++out.declined;
                    }
                    if (gate) gate->leave(r);
                }
                if (v.units != units0) {
                    std::lock_guard<std::mutex> lk(mu);
                    ++out.view_changes;
                }
                bar.arrive_and_wait();   /* the pass done */
            }
        });
    const int64_t end = now_ns() + (int64_t)(seconds * 1e9);
    for (;;) {
        g.ctl->boundary(false);
        const int64_t t = now_ns();
        {
            std::lock_guard<std::mutex> lk(mu);
            for (size_t i = 0; i < held.size();)
                if (held[i].open_at <= t) {
                    MUST(rad_gate_open(held[i].gate) >= 0);
                    held[i] = held.back();
                    held.pop_back();
                } else {
                    ++i;
                }
        }
        if (tick) tick(t);
        if (t >= end) {
            done.store(true);
            bar.arrive_and_wait();
            break;
        }
        bar.arrive_and_wait();   /* the ranks run */
        bar.arrive_and_wait();   /* they are parked again */
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    for (auto& t : ranks) t.join();
    /* The control thread stops first: from here on only this thread touches the lanes. */
    g.ctl->stop();
    for (Held& h : held) MUST(rad_gate_open(h.gate) >= 0);
    /* The late jobs: their queues move on; each one's words were read when it ran. */
    const int64_t wend = now_ns() + 5'000'000'000LL;
    for (int r = 0; r < kR; ++r)
        while (g.w->lane(r)->state() == rad::aux::Lane::kBusy && now_ns() < wend)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    for (auto& c : lates) out.errors += job_errors(*c, &out.first);
    for (uint64_t* p : pinned) rad_dev_free(p, RAD_MEM_HOST_PINNED);
    return out;
}

}  // namespace

TEST(fence_swaps_releases_and_reacquires_never_touch_what_a_job_reads) {
    REQUIRE_EQ(rad_dev_count(), 5);
    REQUIRE(rad::dev_pin_devices({ 0, 1, 2, 3 }) >= 0);
    Rig g(/*controller=*/true, "fence1");
    g.mode("auto");
    g.ctl->start();
    int cycle = 0;
    int64_t next = now_ns() + 300'000'000LL;
    const Run run = run_passes(g, 4.0, 1, [&](int64_t t) {
        if (t < next) return;
        next = t + 300'000'000LL;
        g.mode(++cycle % 2 ? "off" : "on");
    });
    const rad::aux::Control::Snapshot s = g.ctl->snapshot();
    const std::string line = g.w->finish((uint64_t)run.jobs, 0, (uint64_t)run.late);
    for (const std::string& l : g.ctl->report()) std::fprintf(stderr, "%s\n", l.c_str());
    std::fprintf(stderr, "%s\njobs %d late %d declined %d view changes %d; swaps %llu fills %llu releases %llu "
                 "acquires %llu read back %llu (differ %llu); state %s\n", line.c_str(), run.jobs, run.late,
                 run.declined, run.view_changes, (unsigned long long)s.swaps, (unsigned long long)s.fills,
                 (unsigned long long)s.releases, (unsigned long long)s.acquires, (unsigned long long)s.readbacks,
                 (unsigned long long)s.differ, s.state.c_str());
    if (!run.first.empty()) std::fprintf(stderr, "first error: %s\n", run.first.c_str());
    CHECK_EQ(run.errors, 0);
    CHECK(run.jobs > 100);
    CHECK(run.late > 0);
    CHECK(s.swaps > 0);
    CHECK(s.releases >= 2);
    CHECK(s.acquires >= 2);
    CHECK(s.readbacks > 0);
    CHECK_EQ(s.differ, (uint64_t)0);
    CHECK(!contains(s.state, "dead"));
    CHECK(!s.online_stopped);
    CHECK(g.w->drained());
    CHECK(g.warmups.load() > 0);
    g.ctl->write_final("stopped", "the test ends");
    CHECK(contains(g.state(), "stopped "));
    g.ctl.reset();
    g.w.reset();
}

TEST(a_release_waits_for_a_rank_inside_its_lane) {
    Rig g(true, "fence2");
    g.mode("auto");
    g.ctl->start();
    rad::aux::Gate* gate = g.ctl->gate();
    REQUIRE(gate != nullptr);
    REQUIRE(gate->enter(2));   /* rank 2 inside a layer */
    g.mode("off");
    std::this_thread::sleep_for(std::chrono::milliseconds(400));
    CHECK(contains(g.state(), "draining "));   /* waiting for rank 2 */
    CHECK(gate->closed.load());
    CHECK(!gate->enter(1));                     /* a rank finding it shut takes no lane */
    gate->leave(2);
    const int64_t end = now_ns() + 3'000'000'000LL;
    while (!contains(g.state(), "released ") && now_ns() < end) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    CHECK(contains(g.state(), "released "));
    CHECK(contains(g.state(), "mode off (control file)"));
    /* auto again (reacquire after 0 s free): the gate opens at the next boundary */
    g.mode("auto");
    const int64_t end2 = now_ns() + 3'000'000'000LL;
    while (gate->closed.load() && now_ns() < end2) {
        g.ctl->boundary(false);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    CHECK(!gate->closed.load());
    g.ctl->stop();
    (void)g.w->finish(0, 0, 0);
    CHECK(g.w->drained());
    g.ctl.reset();
    g.w.reset();
}

TEST(online_alone_swaps_without_a_gate) {
    Rig g(/*controller=*/false, "fence3");
    CHECK(g.ctl->gate() == nullptr);
    g.ctl->start();
    const Run run = run_passes(g, 1.5, 7, {});
    const rad::aux::Control::Snapshot s = g.ctl->snapshot();
    (void)g.w->finish(0, 0, 0);
    if (!run.first.empty()) std::fprintf(stderr, "first error: %s\n", run.first.c_str());
    CHECK_EQ(run.errors, 0);
    CHECK(s.swaps > 0);
    CHECK_EQ(s.differ, (uint64_t)0);
    CHECK(run.view_changes == 0);
    CHECK(g.w->drained());
    g.ctl.reset();
    g.w.reset();
}

RAD_TEST_MAIN()
