/* p1_stager_test.cpp -- prefill Phase 1's stager on the host (place/card_expert_stager.h; the plan's
 * S-5), with the cards hidden.
 *
 * The scheduling (CardExpertStager) runs against a simulated card: two streams (the compute stream
 * and the mover's DMA lane), each a queue of waits, records, copies and waves that run in a random
 * order wherever the queues allow, events with record generations as HIP's are (a wait captures the
 * record it follows at the time it is queued), and the ring's slots as the units a copy last put
 * there (a copy in flight leaves its slot garbage until it ends). A wave checks, when it runs, that
 * every slot its table names holds the unit the table says; a release checks that the unit's copy
 * has run. So a missing free-event wait (a lapped slot overwritten under its reader), a copy-event
 * wait missing before a wave, a source let go early, a unit in no wave or in two, all show as
 * failures under some interleaving, and the stress cases run thousands of them.
 *
 * Also here: the tunables, the end-aligned chunks (sched/p1_chunk.h) over every prompt length to
 * 20,000, and the admission predicate (FileStager::p1_admits) against the sorted order. */
#include "rad_test.h"

#include "rad_internal.h"
#include "device/device.h"
#include "device/directio.h"
#include "place/card_expert_stager.h"
#include "place/stager.h"
#include "sched/consumer_skip.h"
#include "sched/p1_chunk.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace rad {

/* FileStager's private side, for the admission cases (stager.h names it a friend). */
struct EpCountsHostProbe {
    static void admission(FileStager& fs, int64_t C, int64_t pass_tok, int32_t admit_row, bool order) {
        fs.lcp_ = C;
        fs.ep_rows_ = C;
        fs.pass_tok_ = pass_tok;
        fs.lcp_admit_row_ = admit_row;
        fs.p1_order_ = order;
        fs.pass_row0_.clear();
        StagedLayer L;
        L.layer = 3;
        fs.layers_.assign(1, L);
        FileStager::HxOps o;
        o.gu = 1;
        o.n_expert = 96;
        o.top_k = 6;
        fs.hx_ops_.assign(1, o);
        if (!fs.hx_idx_) fs.hx_idx_ = (int32_t*)rad_dev_alloc((fs.hx_sorted_max() + 128) * 4, RAD_MEM_HOST_PINNED);
    }
    static int32_t* idx(FileStager& fs) { return fs.hx_idx_; }
    static int64_t sorted_max(const FileStager& fs) { return fs.hx_sorted_max(); }
    static bool admits(const FileStager& fs, int32_t e) { return fs.p1_admits(0, e); }
    /* A pass of pass_tok rows over n routed layers (routed layer li is model layer li), cut by row0 (by model
     * layer; empty: no cut), armed for P1 or not, at the threshold min_rows and chunk C. */
    static void cut(FileStager& fs, bool armed, int64_t pass_tok, const std::vector<int32_t>& row0, int32_t n,
                    int64_t min_rows = ds41p1::kMinArmRows, int64_t C = 2048) {
        fs.pass_tok_ = pass_tok;
        fs.p1_armed_ = armed;
        fs.p1_min_rows_ = min_rows;
        fs.ep_rows_ = C;
        fs.pass_row0_ = row0;
        fs.layers_.assign((size_t)n, StagedLayer{});
        for (int32_t li = 0; li < n; ++li) fs.layers_[(size_t)li].layer = li;
    }
    static int64_t rows(const FileStager& fs, int32_t li) { return fs.layer_rows(li); }
    static bool card(const FileStager& fs, int32_t li) { return fs.p1_layer_on(li); }
    static int64_t miss_need(const FileStager& fs, int32_t li) { return fs.hx_miss_need(li); }
    static int64_t cpu_rows(const FileStager& fs) { return fs.p1_cpu_rows(); }
    static void disarm(FileStager& fs) { fs.p1_armed_ = false; fs.layers_.clear(); fs.pass_row0_.clear(); }
    /* The host-pool source (FileStager::p1_pool_source) over a mover and its plan. */
    static void bind(FileStager& fs, Mover* m, const Plan* plan, const Program* prog) {
        fs.mover_ = m;
        fs.plan_ = plan;
        fs.ep_program_ = prog;
    }
    static bool pool_source(const FileStager& fs, int32_t u, p1::Source* out) { return fs.p1_pool_source(u, out); }
    static void unbind(FileStager& fs) {
        fs.mover_ = nullptr;
        fs.plan_ = nullptr;
        fs.ep_program_ = nullptr;
    }
};

}  // namespace rad

using namespace rad;
using namespace rad::p1;

namespace {

/* ---- the simulated card ------------------------------------------------------------------------- */
constexpr uint64_t kSrcBase = 0x7f0000000000ull;
/* A unit's fake source addresses: never read, they name the unit. */
const void* src_gu(int unit) { return (const void*)(kSrcBase + ((uint64_t)unit << 26)); }
const void* src_dn(int unit) { return (const void*)(kSrcBase + ((uint64_t)unit << 26) + (1u << 25)); }
int unit_of(const void* p) { return (int)(((uint64_t)p - kSrcBase) >> 26); }

struct Sim final : Device {
    enum T { Wait, Rec, CopyBeg, CopyEnd, Wave };
    struct Cmd {
        T        t = Wait;
        int      ev = -1;
        uint64_t gen = 0;
        int      slot = -1, unit = -1;
        int      layer = -1;
        std::vector<std::pair<int, int>> reads;   /* (slot, unit) */
        std::vector<int> es;                      /* experts the wave computes */
    };
    std::deque<Cmd> q[2];          /* 0 the compute stream, 1 the DMA lane */
    std::vector<uint64_t> rec, done;
    std::vector<uint8_t> host;
    std::vector<int> ring;
    std::mt19937 rng;
    CardExpertStager* st = nullptr;
    int  layer = -1;
    /* What the current layer's waves must compute: expert -> (unit, packed). */
    std::map<int, std::pair<int, bool>> expect;
    std::map<std::pair<int, int>, int> computed;   /* (layer, expert) -> waves that computed it */
    std::set<int> copied;                          /* units whose copy has run */
    std::set<int> queued;                          /* units whose copy was queued */
    std::map<std::pair<int, int>, int> handle_unit;
    std::set<std::pair<int, int>> released;
    std::vector<std::pair<int, int>> waves_seen;   /* (parity, w) */
    std::vector<int> copy_order;                   /* units, as their copies were queued */
    std::deque<std::pair<int, int>> drive_pending; /* (unit, expert) not yet packed */
    std::map<int, bool> drive_packed_form;
    int  next_handle = 1000;
    int  lag = 3;                  /* random steps a call lets the card take */
    int  max_hand = 2;             /* drive units handed over at most a wait */
    int  violations = 0;
    std::string first;

    explicit Sim(unsigned seed, int n_slots) : rng(seed) { ring.assign((size_t)n_slots, -1); }

    void bad(const std::string& w) {
        if (!violations++) first = w;
    }
    bool runnable(int s) const {
        if (q[s].empty()) return false;
        const Cmd& c = q[s].front();
        return c.t != Wait || done[(size_t)c.ev] >= c.gen;
    }
    void exec(int s) {
        Cmd c = q[s].front();
        q[s].pop_front();
        switch (c.t) {
        case Wait: break;
        case Rec: done[(size_t)c.ev] = std::max(done[(size_t)c.ev], c.gen); break;
        case CopyBeg: ring[(size_t)c.slot] = -2; break;
        case CopyEnd:
            ring[(size_t)c.slot] = c.unit;
            copied.insert(c.unit);
            break;
        case Wave:
            for (const auto& r : c.reads)
                if (ring[(size_t)r.first] != r.second)
                    bad("layer " + std::to_string(c.layer) + ": a wave read slot " + std::to_string(r.first) +
                        " holding " + std::to_string(ring[(size_t)r.first]) + " for unit " + std::to_string(r.second));
            for (int e : c.es) ++computed[{ c.layer, e }];
            break;
        }
    }
    bool step() {
        const bool a = runnable(0), b = runnable(1);
        if (!a && !b) return false;
        exec(a && b ? (int)(rng() & 1) : a ? 0 : 1);
        return true;
    }
    void progress() {
        const int k = (int)(rng() % (unsigned)(lag + 1));
        for (int i = 0; i < k && step(); ++i) {}
    }
    void drain() {
        while (step()) {}
        if (!q[0].empty() || !q[1].empty()) bad("the streams deadlocked");
    }

    int event_create(bool h) override {
        rec.push_back(0);
        done.push_back(0);
        host.push_back(h ? 1 : 0);
        return (int)rec.size() - 1;
    }
    int copy(int slot, const Source& src, int wait, int d) override {
        progress();
        const int u = unit_of(src.gu);
        if (unit_of(src.dn) != u || src.dn != src_dn(u)) bad("a copy's two weights name different units");
        if (!host[(size_t)d]) bad("a copy's done event cannot be queried");
        copy_order.push_back(u);
        queued.insert(u);
        Cmd c;
        if (wait >= 0) {
            c.t = Wait;
            c.ev = wait;
            c.gen = rec[(size_t)wait];
            q[1].push_back(c);
        }
        c = Cmd{};
        c.t = CopyBeg;
        c.slot = slot;
        q[1].push_back(c);
        c.t = CopyEnd;
        c.unit = u;
        q[1].push_back(c);
        c = Cmd{};
        c.t = Rec;
        c.ev = d;
        c.gen = ++rec[(size_t)d];
        q[1].push_back(c);
        return 0;
    }
    int compute_wait(int ev) override {
        Cmd c;
        c.t = Wait;
        c.ev = ev;
        c.gen = rec[(size_t)ev];
        q[0].push_back(c);
        return 0;
    }
    int compute_record(int ev) override {
        Cmd c;
        c.t = Rec;
        c.ev = ev;
        c.gen = ++rec[(size_t)ev];
        q[0].push_back(c);
        return 0;
    }
    int query(int ev) override {
        progress();
        if (!host[(size_t)ev]) { bad("a host query of a local event"); return -1; }
        return done[(size_t)ev] >= rec[(size_t)ev] ? 1 : 0;
    }
    int sync(int ev) override {
        if (!host[(size_t)ev]) { bad("a host sync of a local event"); return -1; }
        while (done[(size_t)ev] < rec[(size_t)ev])
            if (!step()) { bad("a sync that can never return"); return -1; }
        return 0;
    }
    int wave(int parity, int w, const void* const* gu, const void* const* dn, int n) override {
        progress();
        waves_seen.push_back({ parity, w });
        Cmd c;
        c.t = Wave;
        c.layer = layer;
        for (int e = 0; e < n; ++e) {
            const bool skip = ds41p1::is_skip(gu[e]);
            if (skip != ds41p1::is_skip(dn[e])) bad("gate_up and down disagree about a skip");
            if (skip) {
                if (ds41p1::is_packed(gu[e])) bad("an entry with both marks");
                continue;
            }
            const auto it = expect.find(e);
            if (it == expect.end()) { bad("a wave computes expert " + std::to_string(e) + ", not the ring's"); continue; }
            const int s = st->slot_of(gu[e]);
            if (s < 0) { bad("a wave entry outside the ring"); continue; }
            if ((const char*)ds41p1::unmark(dn[e]) != (const char*)ds41p1::unmark(gu[e]) + ds41p1::kDnOff)
                bad("down's entry is not its slot + kDnOff");
            if (ds41p1::is_packed(gu[e]) != it->second.second || ds41p1::is_packed(dn[e]) != it->second.second)
                bad("a wave's packed mark differs from its unit's form");
            if (((uintptr_t)ds41p1::unmark(gu[e]) & 63) != 0) bad("an unaligned slot");
            c.reads.push_back({ s, it->second.first });
            c.es.push_back(e);
        }
        q[0].push_back(c);
        return 0;
    }
    void release(Kind k, int32_t h) override {
        const auto key = std::make_pair((int)k, (int)h);
        const auto it = handle_unit.find(key);
        if (it == handle_unit.end()) { bad("a release of an unknown source"); return; }
        /* A unit never copied (its layer given up) may go at once; a copied one only after its copy ran. */
        if (queued.count(it->second) && !copied.count(it->second))
            bad("unit " + std::to_string(it->second) + "'s source let go before its copy ran");
        if (!released.insert(key).second) bad("a source let go twice");
    }
    int ready_cap = 1 << 20;   /* the packer's pinned ready slots: a handed unit holds one until released */
    int outstanding() const {
        int n = 0;
        for (const auto& kv : handle_unit)
            if (kv.first.first == (int)Kind::Drive && !released.count(kv.first)) ++n;
        return n;
    }
    int wait_drive() override {
        progress();
        if (drive_pending.empty()) return 0;
        const int room = ready_cap - outstanding();
        if (room <= 0) { bad("the packer starved: every pinned ready slot is held"); return -1; }
        const int n = std::min(room, 1 + (int)(rng() % (unsigned)max_hand));
        for (int i = 0; i < n && !drive_pending.empty(); ++i) {
            const size_t pick = rng() % drive_pending.size();
            const auto ue = drive_pending[pick];
            drive_pending.erase(drive_pending.begin() + (ptrdiff_t)pick);
            const int h = next_handle++;
            handle_unit[{ (int)Kind::Drive, h }] = ue.first;
            Source s{ src_gu(ue.first), ds41p1::kDnOff, src_dn(ue.first), 6266880, drive_packed_form[ue.first] };
            if (s.packed) {
                s.gu_bytes = 11999296;
                s.dn_bytes = 6001984;
            }
            if (st->drive_ready(ue.first, s, h) < 0) bad("drive_ready refused a unit the layer waits for");
        }
        return 1;
    }
};

/* One routed layer's units on the ring: (expert, rows, drive?, packed?, pinned in the host pool?). */
struct LUnit { int e, rows; bool drive, packed, pool = false; };

/* Runs a layer: adds every unit (unique unit ids from `*uid`), start, run. */
int run_layer(Sim& sim, CardExpertStager& st, int layer, const std::vector<LUnit>& us, int* uid) {
    sim.layer = layer;
    sim.expect.clear();
    if (st.begin_layer(layer & 1) < 0) return -1;
    for (const LUnit& x : us) {
        const int u = (*uid)++;
        sim.expect[x.e] = { u, x.packed };
        if (x.drive) {
            sim.drive_pending.push_back({ u, x.e });
            sim.drive_packed_form[u] = x.packed;
            if (st.add_drive(x.e, u, x.rows) < 0) return -1;
        } else if (x.pool) {
            Source s{ src_gu(u), ds41p1::kDnOff, src_dn(u), 6266880, false };
            if (st.add_pool(x.e, u, x.rows, s) < 0) return -1;
        } else {
            const int h = sim.next_handle++;
            sim.handle_unit[{ (int)Kind::Ram, h }] = u;
            Source s{ src_gu(u), x.packed ? 11999296 : ds41p1::kDnOff, src_dn(u), x.packed ? 6001984 : 6266880, x.packed };
            if (st.add_ram(x.e, u, x.rows, s, h) < 0) return -1;
        }
    }
    if (st.start() < 0) return -1;
    return st.run();
}

/* A random layer: n units over distinct experts, a share from the drive, some packed. */
std::vector<LUnit> random_layer(std::mt19937& rng, int n, int drive_pct, int packed_pct) {
    std::vector<int> es(96);
    for (int i = 0; i < 96; ++i) es[(size_t)i] = i;
    std::shuffle(es.begin(), es.end(), rng);
    std::vector<LUnit> us;
    for (int i = 0; i < n; ++i)
        us.push_back({ es[(size_t)i], 1 + (int)(rng() % 2000), (int)(rng() % 100) < drive_pct,
                       (int)(rng() % 100) < packed_pct });
    return us;
}

/* The ring and the alias base as the stager sees them: addresses only, never read. */
const char* const g_ring = (const char*)(uintptr_t)0x200000000000ull;
const char g_alias_mem[64] __attribute__((aligned(64))) = {};
const void* const g_alias = g_alias_mem;

}  // namespace

/* ---- the tunables ----------------------------------------------------------------------------------- */
TEST(p1_tunables_take_whole_numbers_in_range_and_end_or_start) {
    std::map<std::string, std::string> env;
    auto get = [&](const char* k) -> const char* {
        const auto it = env.find(k);
        return it == env.end() ? nullptr : it->second.c_str();
    };
    Tunables t;
    std::string why;
    CHECK(parse_tunables(get, &t, &why));
    CHECK_EQ(t.wave, ds41p1::kDefWave);
    CHECK_EQ(t.inflight, ds41p1::kDefInflight);
    CHECK_EQ(t.ready, ds41p1::kDefReady);
    CHECK_EQ(t.ahead, ds41p1::kDefAhead);
    CHECK_EQ(t.verify, 0);
    CHECK(t.align_end);
    env[ds41p1::kWaveEnv] = "4";
    env[ds41p1::kAheadEnv] = "0";
    env[ds41p1::kAlignEnv] = "start";
    CHECK(parse_tunables(get, &t, &why));
    CHECK_EQ(t.wave, 4);
    CHECK_EQ(t.ahead, 0);
    CHECK(!t.align_end);
    for (const char* bad : { "0", "17", "x", "-1", "2.5" }) {
        env[ds41p1::kWaveEnv] = bad;
        CHECK(!parse_tunables(get, &t, &why));
        CHECK(why.find(ds41p1::kWaveEnv) != std::string::npos);
    }
    env[ds41p1::kWaveEnv] = "2";
    env[ds41p1::kAlignEnv] = "middle";
    CHECK(!parse_tunables(get, &t, &why));
    CHECK(why.find("end or start") != std::string::npos);
    /* The threshold: 320 unset or empty, 17 to 4096 (the gemv rows below, the twin's bound above). */
    env.erase(ds41p1::kAlignEnv);
    CHECK(parse_tunables(get, &t, &why));
    CHECK_EQ(t.min_rows, 320);
    CHECK_EQ(t.min_rows, ds41p1::kDefMinRows);
    env[ds41p1::kMinRowsEnv] = "";
    CHECK(parse_tunables(get, &t, &why));
    CHECK_EQ(t.min_rows, 320);
    for (const char* ok : { "17", "320", "1024", "4096" }) {
        env[ds41p1::kMinRowsEnv] = ok;
        CHECK(parse_tunables(get, &t, &why));
        CHECK_EQ(t.min_rows, std::atoi(ok));
    }
    for (const char* bad : { "16", "0", "4097", "320x", "-320", " 320" }) {
        env[ds41p1::kMinRowsEnv] = bad;
        CHECK(!parse_tunables(get, &t, &why));
        CHECK(why.find(ds41p1::kMinRowsEnv) != std::string::npos);
        CHECK(why.find("17 to 4096") != std::string::npos);
    }
}

/* ---- end-aligned chunks ---------------------------------------------------------------------------- */
TEST(p1_end_aligned_chunks_fill_the_last_chunk_for_every_length_to_20000) {
    /* The plan's examples. */
    CHECK_EQ(end_aligned_chunk(0, 3086, 2048, 16), (int64_t)1040);
    CHECK_EQ(end_aligned_chunk(1040, 3086, 2048, 16), (int64_t)2046);
    CHECK_EQ(end_aligned_chunk(0, 6817, 2048, 16), (int64_t)688);
    CHECK_EQ(end_aligned_chunk(688, 6817, 2048, 16), (int64_t)2048);
    CHECK_EQ(end_aligned_chunk(2736, 6817, 2048, 16), (int64_t)2048);
    CHECK_EQ(end_aligned_chunk(4784, 6817, 2048, 16), (int64_t)2033);
    CHECK_EQ(end_aligned_chunk(0, 2048, 2048, 16), (int64_t)2048);
    CHECK_EQ(end_aligned_chunk(0, 17, 2048, 16), (int64_t)17);
    for (int64_t C : { 1024, 2048 })
        for (int64_t g : { 16, 64 })
            for (int64_t start : { 0, 5, 16, 1000, 2051 })
                for (int64_t P = 17; P <= 20000; ++P) {
                    const int64_t end = start + P;
                    std::vector<int64_t> ch;
                    for (int64_t at = start; at < end && ch.size() < 64;) {
                        const int64_t n = end_aligned_chunk(at, end, C, g);
                        if (n <= 0) break;
                        ch.push_back(n);
                        at += n;
                    }
                    int64_t sum = 0;
                    bool ok = !ch.empty();
                    for (size_t i = 0; i < ch.size(); ++i) {
                        sum += ch[i];
                        ok = ok && ch[i] >= 1 && ch[i] <= C;
                        if (i + 1 < ch.size()) ok = ok && (start + sum) % g == 0;   /* a non-final chunk ends on the grid */
                        if (start % g == 0 && i > 0 && i + 1 < ch.size()) ok = ok && ch[i] == C;   /* the middle ones are full */
                    }
                    /* From a grid position: no more chunks than start alignment, and the last as full as the
                     * grid allows. From off the grid (a resume that is not on it) the first chunk's end must
                     * reach the grid within C, which can cost one chunk more. */
                    const bool on = start % g == 0;
                    const int64_t n0 = (P + C - 1) / C;
                    ok = ok && sum == P && (int64_t)ch.size() >= n0 && (int64_t)ch.size() <= n0 + (on ? 0 : 1);
                    if (P > C && on) ok = ok && ch.back() > C - g;
                    if (!ok) {
                        CHECK(ok);
                        std::fprintf(stderr, "C %lld g %lld start %lld P %lld: %zu chunks\n", (long long)C, (long long)g,
                                     (long long)start, (long long)P, ch.size());
                        return;
                    }
                }
    CHECK(true);
}

/* ---- the scheduling ---------------------------------------------------------------------------------- */
TEST(p1_waves_take_ram_units_first_in_descending_rows_then_the_drive) {
    Sim sim(7, 12);
    CardExpertStager st;
    std::string why;
    sim.st = &st;
    REQUIRE_EQ(st.init(&sim, g_ring, 12, 2, 96, g_alias, 32, &why), 0);
    sim.lag = 0;   /* nothing runs until the host waits: the order is the host's */
    int uid = 1;
    std::vector<LUnit> us = { { 3, 10, false, false }, { 9, 700, true, false }, { 4, 300, false, true },
                              { 11, 5, false, false }, { 40, 900, false, false }, { 41, 50, true, true } };
    REQUIRE_EQ(run_layer(sim, st, 0, us, &uid), 0);
    REQUIRE_EQ(st.end_pass(), 0);
    sim.drain();
    /* Units 1..6 in the order above: RAM by rows 900 (5), 300 (3), 10 (1), 5 (4); then the drive's. */
    REQUIRE(sim.copy_order.size() == (size_t)6);
    CHECK_EQ(sim.copy_order[0], 5);
    CHECK_EQ(sim.copy_order[1], 3);
    CHECK_EQ(sim.copy_order[2], 1);
    CHECK_EQ(sim.copy_order[3], 4);
    CHECK(sim.copy_order[4] == 2 || sim.copy_order[4] == 6);
    CHECK_EQ(sim.violations, 0);
    if (sim.violations) std::fprintf(stderr, "%s\n", sim.first.c_str());
    for (const auto& kv : sim.expect) CHECK_EQ(sim.computed[std::make_pair(0, kv.first)], 1);
    CHECK_EQ(st.held(), 0);
    CHECK_EQ(st.stats().waves, (uint64_t)3);
}

TEST(p1_every_ring_unit_is_in_exactly_one_wave_and_the_rest_carry_the_skip_mark) {
    for (int wave = 1; wave <= 4; ++wave) {
        Sim sim(11 + wave, 12);
        CardExpertStager st;
        std::string why;
        sim.st = &st;
        REQUIRE_EQ(st.init(&sim, g_ring, 12, wave, 96, g_alias, 6, &why), 0);
        std::mt19937 rng(3 + wave);
        int uid = 1;
        for (int layer = 0; layer < 6; ++layer) {
            const auto us = random_layer(rng, 1 + (int)(rng() % 70), 40, 50);
            REQUIRE_EQ(run_layer(sim, st, layer, us, &uid), 0);
            for (const LUnit& x : us) CHECK(sim.expect.count(x.e));
        }
        REQUIRE_EQ(st.end_pass(), 0);
        sim.drain();
        CHECK_EQ(sim.violations, 0);
        if (sim.violations) std::fprintf(stderr, "wave %d: %s\n", wave, sim.first.c_str());
        for (const auto& kv : sim.computed) CHECK_EQ(kv.second, 1);
        CHECK_EQ(st.held(), 0);
        CHECK_EQ(sim.released.size(), sim.handle_unit.size());
    }
}

TEST(p1_wave_tables_alternate_parity_by_layer_and_count_waves_from_zero) {
    Sim sim(5, 4);
    CardExpertStager st;
    std::string why;
    sim.st = &st;
    REQUIRE_EQ(st.init(&sim, g_ring, 4, 2, 96, g_alias, 6, &why), 0);
    int uid = 1;
    std::mt19937 rng(9);
    std::vector<size_t> starts;
    std::vector<int> nunits;
    for (int layer = 0; layer < 5; ++layer) {
        starts.push_back(sim.waves_seen.size());
        const auto us = random_layer(rng, 1 + layer * 7, 30, 30);
        nunits.push_back((int)us.size());
        REQUIRE_EQ(run_layer(sim, st, layer, us, &uid), 0);
    }
    starts.push_back(sim.waves_seen.size());
    REQUIRE_EQ(st.end_pass(), 0);
    sim.drain();
    for (int layer = 0; layer < 5; ++layer) {
        const int n = (int)(starts[(size_t)layer + 1] - starts[(size_t)layer]);
        CHECK(n >= (nunits[(size_t)layer] + 1) / 2);
        for (int w = 0; w < n; ++w) {
            const auto pw = sim.waves_seen[starts[(size_t)layer] + (size_t)w];
            CHECK_EQ(pw.first, layer & 1);
            CHECK_EQ(pw.second, w);
        }
    }
    CHECK_EQ(sim.violations, 0);
}

TEST(p1_sources_go_only_after_their_copies_and_all_by_the_pass_end) {
    Sim sim(21, 4);
    CardExpertStager st;
    std::string why;
    sim.st = &st;
    REQUIRE_EQ(st.init(&sim, g_ring, 4, 1, 96, g_alias, 6, &why), 0);
    sim.lag = 1;
    int uid = 1;
    std::mt19937 rng(4);
    for (int layer = 0; layer < 8; ++layer) REQUIRE_EQ(run_layer(sim, st, layer, random_layer(rng, 20, 50, 50), &uid), 0);
    /* Before the pass ends, sources may still be held (their copies not yet seen). */
    REQUIRE_EQ(st.end_pass(), 0);
    CHECK_EQ(st.held(), 0);
    CHECK_EQ(sim.released.size(), sim.handle_unit.size());
    sim.drain();
    CHECK_EQ(sim.violations, 0);
    if (sim.violations) std::fprintf(stderr, "%s\n", sim.first.c_str());
}

TEST(p1_an_unfinished_layer_is_refused_at_the_pass_end_and_its_sources_go) {
    Sim sim(2, 4);
    CardExpertStager st;
    std::string why;
    sim.st = &st;
    REQUIRE_EQ(st.init(&sim, g_ring, 4, 2, 96, g_alias, 6, &why), 0);
    int uid = 1;
    REQUIRE_EQ(st.begin_layer(0), 0);
    sim.layer = 0;
    for (int e = 0; e < 6; ++e) {
        const int u = uid++, h = sim.next_handle++;
        sim.handle_unit[{ 0, h }] = u;
        sim.expect[e] = { u, false };
        Source s{ src_gu(u), ds41p1::kDnOff, src_dn(u), 6266880, false };
        REQUIRE_EQ(st.add_ram(e, u, 10, s, h), 0);
    }
    REQUIRE_EQ(st.start(), 0);   /* four copied, two still waiting for room */
    CHECK(st.begin_layer(1) < 0);  /* the layer before has not run */
    CHECK(st.end_pass() < 0);
    sim.drain();
    CHECK_EQ(st.held(), 0);
    CHECK_EQ(sim.released.size(), (size_t)6);
    CHECK_EQ(sim.violations, 0);
    /* And refusals at init and on bad units. */
    CardExpertStager st2;
    CHECK(st2.init(&sim, g_ring, 1, 1, 96, g_alias, 6, &why) < 0);
    CHECK(st2.init(&sim, g_ring + 8, 4, 1, 96, g_alias, 6, &why) < 0);
    CHECK(st2.init(&sim, g_ring, 4, 5, 96, g_alias, 6, &why) < 0);
    REQUIRE_EQ(st2.init(&sim, g_ring, 4, 2, 96, g_alias, 6, &why), 0);
    REQUIRE_EQ(st2.begin_layer(0), 0);
    Source big{ src_gu(1), ds41p1::kDnOff + 64, src_dn(1), 64, false };
    CHECK(st2.add_ram(1, 1, 1, big, 1) < 0);
    Source ok{ src_gu(1), 64, src_dn(1), 64, false };
    CHECK_EQ(st2.add_ram(1, 1, 1, ok, 1), 0);
    CHECK(st2.add_ram(1, 2, 1, ok, 2) < 0);   /* one unit an expert */
    CHECK(st2.drive_ready(77, ok, 3) < 0);    /* not a unit this layer waits for */
}

/* The critique's ring stress test: the smallest rings, waves of one, random completion, many layers. */
TEST(p1_ring_stress_two_slots_waves_of_one_random_completion) {
    int runs = 0;
    for (int n_slots : { 2, 3, 4, 12 })
        for (int wave = 1; wave <= std::min(3, n_slots); ++wave)
            for (unsigned seed = 1; seed <= 40; ++seed) {
                Sim sim(seed * 7919u + (unsigned)n_slots * 31u + (unsigned)wave, n_slots);
                sim.lag = 1 + (int)(seed % 6);
                sim.max_hand = 1 + (int)(seed % 3);
                CardExpertStager st;
                std::string why;
                sim.st = &st;
                const int ready = 1 + (int)(seed % 6);
                sim.ready_cap = ready;
                REQUIRE_EQ(st.init(&sim, g_ring, n_slots, wave, 96, g_alias, ready, &why), 0);
                std::mt19937 rng(seed);
                int uid = 1;
                for (int pass = 0; pass < 2; ++pass) {
                    for (int layer = 0; layer < 6; ++layer) {
                        const auto us = random_layer(rng, (int)(rng() % 40), (int)(rng() % 101), 50);
                        REQUIRE_EQ(run_layer(sim, st, pass * 100 + layer, us, &uid), 0);
                    }
                    REQUIRE_EQ(st.end_pass(), 0);
                    REQUIRE_EQ(st.held(), 0);
                }
                sim.drain();
                if (sim.violations) {
                    CHECK_EQ(sim.violations, 0);
                    std::fprintf(stderr, "slots %d wave %d seed %u: %s\n", n_slots, wave, seed, sim.first.c_str());
                    return;
                }
                for (const auto& kv : sim.computed)
                    if (kv.second != 1) {
                        CHECK_EQ(kv.second, 1);
                        return;
                    }
                CHECK_EQ(sim.released.size(), sim.handle_unit.size());
                ++runs;
            }
    CHECK_EQ(runs, (2 + 3 + 3 + 3) * 40);
}

/* ---- units the plan pinned in the host pool ---------------------------------------------------------------
 *
 * A pooled unit goes to the ring as a RAM-cache unit does (in the same descending-rows order, before the drive's),
 * but from its pool slot, read in place: its slot holds no source, nothing is let go for it (the simulated card
 * flags any release of an unknown source), and a packed form is refused (the pool holds the card's block form). */
TEST(p1_pool_units_are_copied_with_the_ram_units_and_never_let_go) {
    Sim sim(31, 12);
    CardExpertStager st;
    std::string why;
    sim.st = &st;
    REQUIRE_EQ(st.init(&sim, g_ring, 12, 2, 96, g_alias, 32, &why), 0);
    sim.lag = 0;
    REQUIRE_EQ(st.begin_layer(0), 0);
    sim.layer = 0;
    /* RAM unit 1 (rows 40), pool unit 2 (rows 600), drive unit 3 (rows 900), pool unit 4 (rows 5), RAM unit 5
     * (rows 300). */
    const int ram_h1 = sim.next_handle++, ram_h5 = sim.next_handle++;
    sim.handle_unit[{ (int)Kind::Ram, ram_h1 }] = 1;
    sim.handle_unit[{ (int)Kind::Ram, ram_h5 }] = 5;
    auto block = [](int u) { return Source{ src_gu(u), ds41p1::kDnOff, src_dn(u), 6266880, false }; };
    sim.expect[10] = { 1, false };
    sim.expect[11] = { 2, false };
    sim.expect[12] = { 3, false };
    sim.expect[13] = { 4, false };
    sim.expect[14] = { 5, false };
    REQUIRE_EQ(st.add_ram(10, 1, 40, block(1), ram_h1), 0);
    REQUIRE_EQ(st.add_pool(11, 2, 600, block(2)), 0);
    sim.drive_pending.push_back({ 3, 12 });
    sim.drive_packed_form[3] = false;
    REQUIRE_EQ(st.add_drive(12, 3, 900), 0);
    REQUIRE_EQ(st.add_pool(13, 4, 5, block(4)), 0);
    REQUIRE_EQ(st.add_ram(14, 5, 300, block(5), ram_h5), 0);
    Source packed{ src_gu(6), 11999296, src_dn(6), 6001984, true };
    CHECK(st.add_pool(15, 6, 7, packed) < 0);    /* not the pool's form */
    CHECK(st.add_pool(10, 7, 7, block(7)) < 0);  /* one unit an expert, whatever its kind */
    REQUIRE_EQ(st.start(), 0);
    REQUIRE_EQ(st.run(), 0);
    REQUIRE_EQ(st.end_pass(), 0);
    sim.drain();
    REQUIRE(sim.copy_order.size() == (size_t)5);
    CHECK_EQ(sim.copy_order[0], 2);   /* 600 rows */
    CHECK_EQ(sim.copy_order[1], 5);   /* 300 */
    CHECK_EQ(sim.copy_order[2], 1);   /* 40 */
    CHECK_EQ(sim.copy_order[3], 4);   /* 5 */
    CHECK_EQ(sim.copy_order[4], 3);   /* the drive's, once packed */
    for (int e = 10; e <= 14; ++e) CHECK_EQ(sim.computed[std::make_pair(0, e)], 1);
    CHECK_EQ(sim.violations, 0);
    if (sim.violations) std::fprintf(stderr, "%s\n", sim.first.c_str());
    CHECK_EQ(st.held(), 0);
    CHECK_EQ(st.stats().pool, (uint64_t)2);
    CHECK_EQ(st.stats().ram, (uint64_t)2);
    CHECK_EQ(st.stats().drive, (uint64_t)1);
    /* Released: the two RAM leases and the drive's ready slot, never a pool unit. */
    CHECK_EQ(sim.released.size(), (size_t)3);
    for (const auto& r : sim.released) CHECK(r.first != (int)Kind::Pool);
}

/* The host-pool source: a pooled unit's gate_up and down where the card's DMA reads them (its pool slot at each
 * weight's offset in the unit, told apart by their block sizes whatever their order), block form; none for a
 * VRAM resident or a weight not in the card's block_mxfp4. Mover::alias_pool_units publishes the pooled ones
 * skipped (the mark on the alias base) and leaves the rest; unstage_units publishes them at their pool slots
 * again, as the file stager's release of the layer does. */
TEST(p1_pool_source_reads_a_pooled_unit_in_place_and_its_skip_mark_comes_off_at_release) {
    const int64_t gu = ds41p1::kGuBlockBytes, dn = ds41p1::kDnBlockBytes;
    Program prog;
    prog.weights.resize(1);
    auto weight = [&](const std::string& name, int64_t bytes, int32_t e) {
        WeightInfo w;
        w.name = name;
        w.decl.dtype = RAD_U8;
        w.decl.rank = 1;
        w.decl.shape[0] = bytes;
        w.decl.access = RAD_ACCESS_CONDITIONAL;
        w.decl.shard = RAD_SHARD_NONE;
        w.decl.group.layer = 7;
        w.decl.group.expert = e;
        w.logical_bytes = w.stored_bytes = bytes;
        w.align = RAD_ALIGN_UNIT;
        w.layout_tag = "ggml_mxfp4";
        prog.weights.push_back(w);
        return (rad_weight)(prog.weights.size() - 1);
    };
    Plan plan;
    plan.rank = 0;
    plan.world_size = 1;
    plan.dynamic = false;
    SlabClass sc;
    sc.name = "expert";
    sc.stride = ds41p1::kSlotBytes;
    sc.n_units = 3;
    plan.classes.push_back(sc);
    /* Unit 0 pinned (gate_up first), unit 1 a static VRAM resident, unit 2 pinned with down first. */
    for (int e = 0; e < 3; ++e) {
        MoveUnit mu;
        mu.layer = 7;
        mu.expert = e;
        mu.slab_class = 0;
        mu.bytes = mu.slot_bytes = ds41p1::kSlotBytes;
        mu.access = RAD_ACCESS_CONDITIONAL;
        mu.tier = e == 1 ? Tier::VRAM : Tier::HostPinned;
        mu.site = e == 1 ? Site::Device : Site::DeviceZeroCopy;
        const std::string b = "blk.7.ffn_" + std::to_string(e);
        if (e < 2) {
            mu.weights = { weight(b + "_gate_up", gu, e), weight(b + "_down", dn, e) };
            mu.offset = { 0, gu };
        } else {
            mu.weights = { weight(b + "_down", dn, e), weight(b + "_gate_up", gu, e) };
            mu.offset = { 0, dn };
        }
        plan.units.push_back(mu);
    }
    plan.unit_of_weight.assign(prog.weights.size(), -1);
    for (size_t u = 0; u < plan.units.size(); ++u)
        for (rad_weight w : plan.units[u].weights) plan.unit_of_weight[w] = (int32_t)u;
    Mover m;
    MoverConfig mc;
    REQUIRE_EQ(m.init(prog, plan, mc), RAD_OK);
    FileStager fs;
    EpCountsHostProbe::bind(fs, &m, &plan, &prog);
    p1::Source src;
    REQUIRE(EpCountsHostProbe::pool_source(fs, 0, &src));
    const char* b0 = (const char*)m.unit_base(0);
    REQUIRE(b0 != nullptr);
    CHECK(src.gu == b0);
    CHECK_EQ(src.gu_bytes, gu);
    CHECK(src.dn == b0 + gu);
    CHECK_EQ(src.dn_bytes, dn);
    CHECK(!src.packed);
    REQUIRE(EpCountsHostProbe::pool_source(fs, 2, &src));
    const char* b2 = (const char*)m.unit_base(2);
    CHECK(src.gu == b2 + dn);   /* the gate_up is the unit's second weight */
    CHECK_EQ(src.gu_bytes, gu);
    CHECK(src.dn == b2);
    CHECK_EQ(src.dn_bytes, dn);
    CHECK(b2 != b0);
    CHECK(!EpCountsHostProbe::pool_source(fs, 1, &src));    /* a VRAM resident is read in place */
    CHECK(!EpCountsHostProbe::pool_source(fs, 3, &src));    /* no such unit */
    prog.weights[plan.units[2].weights[0]].layout_tag = "ggml_mxfp4_packed";
    CHECK(!EpCountsHostProbe::pool_source(fs, 2, &src));    /* not the card's block form */
    prog.weights[plan.units[2].weights[0]].layout_tag = "ggml_mxfp4";
    /* Published skipped for the layer, then at the pool slots again. */
    ResidencyTable* rt = m.residency();
    const char* res1 = (const char*)rt->lookup(plan.units[1].weights[0])->ptr;
    REQUIRE(res1 != nullptr);
    alignas(64) static char alias[64];
    std::vector<int32_t> staged;
    m.alias_pool_units({ 0, 1, 2 }, alias, &staged, ds41p1::kMarkSkip);
    REQUIRE(staged.size() == (size_t)2);
    CHECK_EQ(staged[0], 0);
    CHECK_EQ(staged[1], 2);
    for (int32_t u : { 0, 2 })
        for (rad_weight w : plan.units[(size_t)u].weights) {
            const void* p = rt->lookup(w)->ptr;
            CHECK(ds41p1::is_skip(p));
            CHECK(!ds41p1::is_packed(p));
            CHECK(ds41p1::unmark(p) == (const void*)alias);
        }
    CHECK((const char*)rt->lookup(plan.units[1].weights[0])->ptr == res1);   /* the resident left as it was */
    m.unstage_units(staged);
    for (int32_t u : { 0, 2 }) {
        const char* base = (const char*)m.unit_base(u);
        for (size_t k = 0; k < 2; ++k)
            CHECK((const char*)rt->lookup(plan.units[(size_t)u].weights[k])->ptr == base + plan.units[(size_t)u].offset[k]);
    }
    /* A misaligned alias base marks nothing. */
    staged.clear();
    m.alias_pool_units({ 0 }, alias + 8, &staged, ds41p1::kMarkSkip);
    CHECK(staged.empty());
    CHECK((const char*)rt->lookup(plan.units[0].weights[0])->ptr == b0);
    EpCountsHostProbe::unbind(fs);
}

TEST(p1_ring_stress_with_pool_units_beside_ram_and_drive_units) {
    int runs = 0;
    for (int n_slots : { 2, 4, 12 })
        for (unsigned seed = 1; seed <= 30; ++seed) {
            Sim sim(seed * 104729u + (unsigned)n_slots, n_slots);
            sim.lag = 1 + (int)(seed % 5);
            sim.max_hand = 1 + (int)(seed % 3);
            CardExpertStager st;
            std::string why;
            sim.st = &st;
            const int ready = 1 + (int)(seed % 6);
            sim.ready_cap = ready;
            REQUIRE_EQ(st.init(&sim, g_ring, n_slots, 1 + (int)(seed % std::min(3, n_slots)), 96, g_alias, ready, &why), 0);
            std::mt19937 rng(seed);
            int uid = 1;
            for (int layer = 0; layer < 6; ++layer) {
                auto us = random_layer(rng, (int)(rng() % 40), (int)(rng() % 60), 40);
                for (LUnit& x : us)
                    if (!x.drive && (int)(rng() % 100) < 25) { x.pool = true; x.packed = false; }
                REQUIRE_EQ(run_layer(sim, st, layer, us, &uid), 0);
            }
            REQUIRE_EQ(st.end_pass(), 0);
            REQUIRE_EQ(st.held(), 0);
            sim.drain();
            if (sim.violations) {
                CHECK_EQ(sim.violations, 0);
                std::fprintf(stderr, "slots %d seed %u: %s\n", n_slots, seed, sim.first.c_str());
                return;
            }
            for (const auto& kv : sim.computed)
                if (kv.second != 1) {
                    CHECK_EQ(kv.second, 1);
                    return;
                }
            CHECK_EQ(sim.released.size(), sim.handle_unit.size());
            ++runs;
        }
    CHECK_EQ(runs, 3 * 30);
}

/* ---- the threshold (..._P1_MIN_ROWS) ------------------------------------------------------------------------
 *
 * A routed layer's pass of at least the threshold's rows is the card path's; any shorter one, in an armed pass or
 * not, takes the CPU path exactly as without the switch, and the CPU path's miss bound (..._MISSES >= cpu_rows)
 * must then cover it. Over the end-aligned chunks of every prompt length to 6,000 at C = 1024 and 2048, under the
 * consumer cut, at thresholds 17 .. 4096: a pass arms only as a pure prefill pass of the threshold's rows or more;
 * an armed layer is the card path's exactly when its own rows reach the threshold (and 17); every layer the card
 * path does not take needs no more than cpu_rows(C, threshold) rows of the CPU path; a decode, mixed, draft or
 * speculative pass never arms. */
TEST(p1_threshold_arms_long_layer_passes_and_leaves_the_rest_to_the_cpu_path) {
    /* The CPU path's bound: everything below the threshold, at most the chunk, never fewer than 16 rows. */
    CHECK_EQ(ds41p1::cpu_rows(1024, 320), (int64_t)319);
    CHECK_EQ(ds41p1::cpu_rows(1024, 17), (int64_t)16);
    CHECK_EQ(ds41p1::cpu_rows(1024, 1), (int64_t)16);
    CHECK_EQ(ds41p1::cpu_rows(256, 320), (int64_t)256);
    CHECK_EQ(ds41p1::cpu_rows(2048, 4096), (int64_t)2048);
    CHECK_EQ(ds41p1::cpu_rows(2048, 1024), (int64_t)1023);
    for (int64_t r = 0; r <= 5000; ++r)
        for (int64_t m : { 17, 64, 320, 1024, 4096 }) {
            CHECK_EQ(ds41p1::card_layer(r, m), r >= 17 && r >= m);
            if (r >= 17 && r < m) CHECK(r <= ds41p1::cpu_rows(4096, m));
        }
    CHECK(!ds41p1::card_layer(16));
    CHECK(ds41p1::card_layer(17));
    /* Which passes arm. */
    RadBatch b{};
    b.phase = RAD_PHASE_PREFILL;
    for (int64_t n : { 1, 16, 17, 100, 319, 320, 321, 1024, 1025 }) {
        b.n_tok = n;
        CHECK_EQ(FileStager::p1_pass_arms(b, 320, 1024), n >= 320 && n <= 1024);
        CHECK_EQ(FileStager::p1_pass_arms(b, 17, 1024), n >= 17 && n <= 1024);
    }
    b.n_tok = 600;
    CHECK(FileStager::p1_pass_arms(b, 320, 1024));
    RadBatch mixed = b;
    mixed.n_tok_decode = 3;
    RadBatch decode = b;
    decode.phase = RAD_PHASE_DECODE;
    RadBatch draft = b;
    draft.draft_pass = -1;
    RadBatch spec = b;
    spec.n_spec = 5;
    for (const RadBatch* x : { &mixed, &decode, &draft, &spec }) CHECK(!FileStager::p1_pass_arms(*x, 320, 1024));

    const ConsumerShape c16{ 43, 20, 128, 16, 16 };
    FileStager fs;
    std::vector<int64_t> f((size_t)c16.n_layer);
    std::vector<int32_t> row0((size_t)c16.n_layer);
    for (int64_t C : { 1024, 2048 })
        for (int64_t thr : { 17, 64, 320, 1024, 2048, 4096 }) {
            int64_t card = 0, cpu = 0, armed_passes = 0, cpu_in_armed = 0, passes = 0;
            int64_t short_prompt_armed = 0;
            for (int64_t P = 1; P <= 6000; P += (P < 400 ? 1 : 13)) {
                REQUIRE(consumer_thresholds(c16, P, f.data()));
                for (int64_t s = 0, e = ds41p1::first_chunk_end_aligned(P, C, 16); s < P; s = e, e = std::min(P, e + C)) {
                    const int64_t n = e - s;
                    REQUIRE(consumer_row_starts(c16, f.data(), s, e, row0.data()));
                    b = RadBatch{};
                    b.phase = RAD_PHASE_PREFILL;
                    b.n_tok = n;
                    const bool armed = FileStager::p1_pass_arms(b, thr, C);
                    CHECK_EQ(armed, n >= std::max<int64_t>(thr, 17) && n <= C);
                    if (armed && P < thr) ++short_prompt_armed;
                    EpCountsHostProbe::cut(fs, armed, n, row0, c16.n_layer, thr, C);
                    CHECK_EQ(EpCountsHostProbe::cpu_rows(fs), ds41p1::cpu_rows(C, thr));
                    ++passes;
                    armed_passes += armed;
                    for (int32_t li = 0; li < c16.n_layer; ++li) {
                        const int64_t r = EpCountsHostProbe::rows(fs, li);
                        const bool want = armed && r >= thr && r >= 17;
                        CHECK_EQ(EpCountsHostProbe::card(fs, li), want);
                        if (want) { ++card; continue; }
                        /* Not the card path's: the CPU path must take it at ..._MISSES = cpu_rows (a layer with no
                         * row of its own needs none). */
                        ++cpu;
                        cpu_in_armed += armed && r > 0;
                        const int64_t need = EpCountsHostProbe::miss_need(fs, li);
                        if (r > 0 && need > ds41p1::cpu_rows(C, thr)) {
                            CHECK(need <= ds41p1::cpu_rows(C, thr));
                            std::fprintf(stderr, "C %lld thr %lld P %lld chunk [%lld, %lld) layer %d: %lld rows for the CPU path\n",
                                         (long long)C, (long long)thr, (long long)P, (long long)s, (long long)e, (int)li,
                                         (long long)need);
                            return;
                        }
                    }
                }
            }
            CHECK_EQ(short_prompt_armed, (int64_t)0);   /* a prompt below the threshold never arms */
            if (thr <= C) {
                CHECK(armed_passes > 0);
                CHECK(card > 0);
            } else {
                CHECK_EQ(armed_passes, (int64_t)0);    /* a threshold above the chunk: never the card path */
                CHECK_EQ(card, (int64_t)0);
            }
            if (thr == 320) CHECK(cpu_in_armed > 0);    /* the cut leaves deep layers of armed passes on the CPU */
            std::printf("  C %lld threshold %lld: %lld passes, %lld armed; %lld layer passes on the card path, %lld on the "
                        "CPU path (%lld of them in armed passes)\n", (long long)C, (long long)thr, (long long)passes,
                        (long long)armed_passes, (long long)card, (long long)cpu, (long long)cpu_in_armed);
        }
    EpCountsHostProbe::disarm(fs);
}

/* ---- the lent ring ------------------------------------------------------------------------------------------
 *
 * RingLender against a simulated host: loads land and copies pass at random host calls, the host's two reload
 * buffers bound the loads in flight (a RAM-cache hit needs none and lands at once), and failures are injected.
 * The host keeps, by slot, the card path generation of the last copy that passed into the slot (a copy queued
 * before a pass that armed the card path runs before the card path's own copies on the DMA lane, which then
 * overwrite the slot): a unit may be published only from a copy queued in the current generation, never while
 * a card-path pass runs, and every source is let go once, after its read and its copy. */
namespace {
struct FakeLend final : LendHost {
    struct S {
        int  src = 0;                  /* 0 none, 1 a reload buffer, 2 a RAM-cache lease */
        bool reading = false, copying = false, copy_done = false, published = false;
        uint64_t copy_gen = 0, content_gen = ~0ull;
        int  loads = 0, copies = 0;
    };
    std::mt19937 rng;
    std::vector<S> s;
    std::vector<uint8_t> ram;          /* slot k's unit hits the RAM cache in the card's form */
    std::deque<std::pair<int, bool>> landings;
    int bufs = 2, bufs_used = 0, land_pct = 50, copy_pct = 50;
    uint64_t gen = 0;                  /* card-path passes begun */
    std::set<int> fail_load_once, fail_copy_once, fail_query_once, refuse;
    int violations = 0;
    std::string first;

    FakeLend(unsigned seed, int n) : rng(seed) { s.assign((size_t)n, S{}); ram.assign((size_t)n, 0); }
    void bad(const std::string& w) {
        if (!violations++) first = w;
    }
    void step() {
        for (size_t k = 0; k < s.size(); ++k) {
            if (s[k].reading && (int)(rng() % 100) < land_pct) {
                s[k].reading = false;
                landings.push_back({ (int)k, !fail_load_once.erase((int)k) });
            }
            if (s[k].copying && !s[k].copy_done && (int)(rng() % 100) < copy_pct) s[k].copy_done = true;
        }
    }
    void settle_all() {   /* the background catches up: every read lands, every copy passes */
        for (size_t k = 0; k < s.size(); ++k) {
            if (s[k].reading) {
                s[k].reading = false;
                landings.push_back({ (int)k, !fail_load_once.erase((int)k) });
            }
            if (s[k].copying) s[k].copy_done = true;
        }
    }
    int load(int k) override {
        if (refuse.count(k)) return -1;
        if (s[(size_t)k].src) bad("a load of slot " + std::to_string(k) + " that holds a source");
        if (ram[(size_t)k]) {
            s[(size_t)k].src = 2;
            ++s[(size_t)k].loads;
            landings.push_back({ k, true });
            return 1;
        }
        if (bufs_used >= bufs) return 0;
        ++bufs_used;
        s[(size_t)k].src = 1;
        s[(size_t)k].reading = true;
        ++s[(size_t)k].loads;
        return 1;
    }
    int landed(bool* ok) override {
        step();
        if (landings.empty()) return -1;
        const auto d = landings.front();
        landings.pop_front();
        *ok = d.second;
        return d.first;
    }
    int copy(int k) override {
        S& x = s[(size_t)k];
        if (!x.src || x.reading) bad("a copy of slot " + std::to_string(k) + " without landed bytes");
        if (x.copying) bad("a second copy of slot " + std::to_string(k));
        if (fail_copy_once.erase(k)) return -1;
        x.copying = true;
        x.copy_done = false;
        x.copy_gen = gen;
        ++x.copies;
        return 0;
    }
    int copied(int k) override {
        S& x = s[(size_t)k];
        if (!x.copying) { bad("a query of slot " + std::to_string(k) + " with no copy"); return -1; }
        if (fail_query_once.erase(k)) return -1;
        step();
        return x.copy_done ? 1 : 0;
    }
    void drop(int k) override {
        S& x = s[(size_t)k];
        if (!x.src) bad("slot " + std::to_string(k) + " let go of nothing");
        if (x.reading) bad("slot " + std::to_string(k) + "'s buffer let go while its read runs");
        if (x.copying && x.copy_done) x.content_gen = x.copy_gen;
        if (x.src == 1) --bufs_used;
        x.src = 0;
        x.copying = false;
        x.copy_done = false;
    }
    void publish(int k, bool on) override {
        S& x = s[(size_t)k];
        if (on) {
            if (x.published) bad("slot " + std::to_string(k) + " published twice");
            if (x.content_gen != gen)
                bad("slot " + std::to_string(k) + " published from a copy of generation " + std::to_string(x.content_gen) +
                    " in generation " + std::to_string(gen) + " (the card path may have written it since)");
            x.published = true;
        } else {
            if (!x.published) bad("slot " + std::to_string(k) + " withdrawn unpublished");
            x.published = false;
            x.content_gen = ~0ull;   /* the card path takes the slot */
        }
    }
    int published() const {
        int n = 0;
        for (const S& x : s) n += x.published;
        return n;
    }
};

/* A pass of either kind through the lender, the host's checks around it. */
int lend_pass(RingLender& l, FakeLend& h, bool p1, int64_t now) {
    if (p1) ++h.gen;
    const int r = l.begin_pass(p1, now);
    if (p1 && h.published()) h.bad("a unit published while a card-path pass runs");
    if (l.residents() != h.published()) h.bad("the lender's residents and the host's published units differ");
    return r;
}
}  // namespace

TEST(p1_lend_returns_after_the_grace_and_takes_back_for_the_card_path) {
    FakeLend h(5, 6);
    h.land_pct = 100;
    h.copy_pct = 100;
    RingLender l;
    REQUIRE_EQ(l.init(&h, 6, 1000), 0);
    CHECK(l.lent());
    /* The first pass that does not arm the card path begins the reloads; every later one copies what landed
     * and publishes what passed, two buffers at a time. */
    int64_t t = 0;
    for (int i = 0; i < 20 && l.residents() < 6; ++i) lend_pass(l, h, false, t += 10);
    CHECK_EQ(l.residents(), 6);
    CHECK(!l.lent());
    CHECK_EQ(l.stats().returns, (uint64_t)1);
    CHECK_EQ(l.stats().reloaded, (uint64_t)6);
    CHECK_EQ(h.bufs_used, 0);
    CHECK_EQ(l.busy(), 0);
    /* A card-path pass takes every slot back. */
    CHECK_EQ(lend_pass(l, h, true, t += 10), 1);
    CHECK_EQ(l.residents(), 0);
    CHECK(l.lent());
    CHECK_EQ(l.stats().lends, (uint64_t)1);
    CHECK_EQ(l.stats().withdrawn, (uint64_t)6);
    l.end_p1(t += 500);
    const int64_t end = t;
    /* Back-to-back: inside the grace nothing reloads, a second card-path pass finds nothing to take. */
    lend_pass(l, h, false, end + 400);
    CHECK_EQ(lend_pass(l, h, true, end + 600), 0);
    l.end_p1(end + 700);
    lend_pass(l, h, false, end + 1699);
    CHECK(l.lent());
    CHECK_EQ(l.residents(), 0);
    for (const auto& x : h.s) CHECK_EQ(x.loads, 1);
    CHECK_EQ(l.stats().held_back, (uint64_t)2);
    /* The grace runs from the last card-path pass's end. */
    lend_pass(l, h, false, end + 1700);
    CHECK(!l.lent());
    CHECK_EQ(l.stats().returns, (uint64_t)2);
    for (int i = 0; i < 20 && l.residents() < 6; ++i) lend_pass(l, h, false, end + 1710 + i);
    CHECK_EQ(l.residents(), 6);
    for (const auto& x : h.s) CHECK_EQ(x.loads, 2);
    CHECK_EQ(h.violations, 0);
    if (h.violations) std::fprintf(stderr, "%s\n", h.first.c_str());
    /* A grace of 0: the pass after the card path's reloads at once. */
    FakeLend h0(6, 2);
    h0.land_pct = h0.copy_pct = 100;
    RingLender l0;
    REQUIRE_EQ(l0.init(&h0, 2, 0), 0);
    for (int i = 0; i < 5; ++i) lend_pass(l0, h0, false, i);
    CHECK_EQ(l0.residents(), 2);
    lend_pass(l0, h0, true, 10);
    l0.end_p1(11);
    lend_pass(l0, h0, false, 11);
    CHECK(!l0.lent());
    CHECK_EQ(h0.violations, 0);
    RingLender bad;
    CHECK(bad.init(nullptr, 2, 0) < 0);
    CHECK(bad.init(&h0, 0, 0) < 0);
    CHECK(bad.init(&h0, ds41p1::kMaxLend + 1, 0) < 0);
    CHECK(bad.init(&h0, 2, -1) < 0);
}

TEST(p1_lend_failed_reloads_leave_their_units_on_the_cpu_path_until_the_next_return) {
    FakeLend h(9, 5);
    h.land_pct = h.copy_pct = 100;
    h.fail_load_once = { 1 };     /* a read that fails */
    h.fail_copy_once = { 2 };     /* a copy that cannot be queued */
    h.fail_query_once = { 3 };    /* a copy whose event errs */
    h.refuse = { 4 };             /* a unit no load can take */
    RingLender l;
    REQUIRE_EQ(l.init(&h, 5, 100), 0);
    for (int i = 0; i < 30; ++i) lend_pass(l, h, false, i);
    CHECK_EQ(l.residents(), 1);
    CHECK(l.resident(0));
    for (int k = 1; k < 5; ++k) CHECK(!l.resident(k));
    CHECK_EQ(l.stats().failed, (uint64_t)4);
    CHECK_EQ(h.bufs_used, 0);
    CHECK_EQ(l.busy(), 0);
    for (int k = 1; k < 4; ++k) CHECK_EQ(h.s[(size_t)k].loads, 1);   /* no retry inside a return */
    /* The next return tries them again (the injected failures were once each; slot 4 is refused for good). */
    lend_pass(l, h, true, 100);
    l.end_p1(110);
    for (int i = 0; i < 30; ++i) lend_pass(l, h, false, 300 + i);
    CHECK_EQ(l.residents(), 4);
    CHECK(!l.resident(4));
    CHECK_EQ(h.violations, 0);
    if (h.violations) std::fprintf(stderr, "%s\n", h.first.c_str());
}

TEST(p1_lend_loads_and_copies_a_card_path_pass_overtakes_are_never_published) {
    FakeLend h(13, 3);
    h.land_pct = 0;
    h.copy_pct = 0;
    RingLender l;
    REQUIRE_EQ(l.init(&h, 3, 50), 0);
    h.ram[2] = 1;                 /* slot 2 a RAM-cache hit: lands at once, no buffer */
    lend_pass(l, h, false, 0);    /* slots 0, 1 reading (both buffers), 2 landed */
    CHECK_EQ(h.bufs_used, 2);
    lend_pass(l, h, false, 1);    /* slot 2's copy queued */
    REQUIRE(h.s[2].copying);
    h.s[0].reading = false;       /* slot 0's read lands */
    h.landings.push_back({ 0, true });
    /* Slot 0 landed (not yet seen), 1 reading, 2 copying: the card path overtakes all three. */
    CHECK_EQ(lend_pass(l, h, true, 2), 0);
    CHECK_EQ(l.residents(), 0);
    CHECK_EQ(h.bufs_used, 1);     /* slot 0's buffer given back at once; slot 1's read still runs */
    CHECK(h.s[2].copying);        /* its source stays until its copy passes */
    l.end_p1(3);
    h.settle_all();
    lend_pass(l, h, false, 10);   /* inside the grace: the overtaken ones only let go */
    CHECK_EQ(h.bufs_used, 0);
    CHECK_EQ(h.s[2].src, 0);
    CHECK_EQ(l.residents(), 0);
    CHECK_EQ(l.busy(), 0);
    CHECK_EQ(l.stats().cancelled, (uint64_t)3);
    /* After the grace each is reloaded from the start, and only those copies are published. */
    h.land_pct = h.copy_pct = 100;
    for (int i = 0; i < 20; ++i) lend_pass(l, h, false, 100 + i);
    CHECK_EQ(l.residents(), 3);
    for (const auto& x : h.s) CHECK_EQ(x.loads, 2);
    CHECK_EQ(h.violations, 0);
    if (h.violations) std::fprintf(stderr, "%s\n", h.first.c_str());
}

/* Decode beside the card path: a pass that does not arm it (decode, or a step that mixes prefill and decode rows,
 * which p1_pass_arms never arms) never takes residents back, and while the slots are lent such a pass sees the
 * units non-resident (the CPU path's) until a later pass publishes their reloads; residency changes only at a
 * pass's begin. */
TEST(p1_lend_decode_beside_the_card_path_sees_residency_change_only_between_passes) {
    RadBatch mixed{};
    mixed.phase = RAD_PHASE_PREFILL;
    mixed.n_tok = 700;
    mixed.n_tok_decode = 4;
    RadBatch decode{};
    decode.phase = RAD_PHASE_DECODE;
    decode.n_tok = 4;
    decode.n_tok_decode = 4;
    RadBatch chunk{};
    chunk.phase = RAD_PHASE_PREFILL;
    chunk.n_tok = 1024;
    FakeLend h(17, 4);
    h.land_pct = h.copy_pct = 100;
    RingLender l;
    REQUIRE_EQ(l.init(&h, 4, 300), 0);
    int64_t t = 0;
    for (int i = 0; i < 10; ++i) lend_pass(l, h, FileStager::p1_pass_arms(decode, 320, 1024), t += 5);
    CHECK_EQ(l.residents(), 4);
    lend_pass(l, h, FileStager::p1_pass_arms(mixed, 320, 1024), t += 5);
    CHECK_EQ(l.residents(), 4);   /* a mixed step keeps them */
    CHECK_EQ(lend_pass(l, h, FileStager::p1_pass_arms(chunk, 320, 1024), t += 5), 1);
    l.end_p1(t += 2000);
    /* Decode steps between two chunks of a long prompt: lent, nothing published, nothing read again. */
    for (int i = 0; i < 5; ++i) {
        const int before = h.published();
        lend_pass(l, h, FileStager::p1_pass_arms(decode, 320, 1024), t += 20);
        CHECK_EQ(h.published(), before);
        CHECK_EQ(h.published(), 0);
    }
    CHECK_EQ(lend_pass(l, h, FileStager::p1_pass_arms(chunk, 320, 1024), t += 5), 0);
    l.end_p1(t += 2000);
    for (const auto& x : h.s) CHECK_EQ(x.loads, 1);
    /* Then the decode steps after the prompt: held back for the grace, reloaded after it, each unit published at
     * a pass's begin once its copy passed. */
    int seen = 0;
    for (int i = 0; i < 40; ++i) {
        lend_pass(l, h, FileStager::p1_pass_arms(decode, 320, 1024), t += 20);
        CHECK(h.published() >= seen);
        seen = h.published();
    }
    CHECK_EQ(l.residents(), 4);
    CHECK_EQ(h.violations, 0);
    if (h.violations) std::fprintf(stderr, "%s\n", h.first.c_str());
}

TEST(p1_lend_stress_random_passes_landings_and_failures) {
    int runs = 0;
    for (unsigned seed = 1; seed <= 200; ++seed) {
        std::mt19937 rng(seed);
        const int n = 1 + (int)(rng() % 12);
        FakeLend h(seed * 7u + 1u, n);
        h.bufs = 1 + (int)(rng() % 3);
        h.land_pct = 10 + (int)(rng() % 90);
        h.copy_pct = 10 + (int)(rng() % 90);
        for (int k = 0; k < n; ++k) h.ram[(size_t)k] = rng() % 4 == 0;
        RingLender l;
        REQUIRE_EQ(l.init(&h, n, (int64_t)(rng() % 200)), 0);
        int64_t t = 0;
        bool in_p1 = false;
        for (int i = 0; i < 400; ++i) {
            t += 1 + (int64_t)(rng() % 40);
            if (rng() % 9 == 0) h.fail_load_once.insert((int)(rng() % (unsigned)n));
            if (rng() % 13 == 0) h.fail_copy_once.insert((int)(rng() % (unsigned)n));
            if (rng() % 17 == 0) h.fail_query_once.insert((int)(rng() % (unsigned)n));
            if (in_p1) l.end_p1(t);
            in_p1 = rng() % 5 == 0;
            lend_pass(l, h, in_p1, t);
            if (h.bufs_used > h.bufs) h.bad("more buffers in use than the host has");
        }
        /* One more card-path pass (a failed reload waits for the next return), then a long idle stretch of
         * decode passes: every unit is resident again, nothing held. */
        if (in_p1) l.end_p1(t);
        lend_pass(l, h, true, t += 1);
        l.end_p1(t += 1);
        h.fail_load_once.clear();
        h.fail_copy_once.clear();
        h.fail_query_once.clear();
        h.land_pct = h.copy_pct = 100;
        for (int i = 0; i < 4 * n + 20; ++i) lend_pass(l, h, false, t += 1000);
        if (l.residents() != n || l.busy() || h.bufs_used || h.violations) {
            CHECK_EQ(l.residents(), n);
            CHECK_EQ(l.busy(), 0);
            CHECK_EQ(h.bufs_used, 0);
            CHECK_EQ(h.violations, 0);
            std::fprintf(stderr, "seed %u: %s\n", seed, h.first.c_str());
            return;
        }
        ++runs;
    }
    CHECK_EQ(runs, 200);
}

/* The mover's side: a lent-ring resident is published at its ring slot (weight i at its offset), never aliased,
 * borrowed or read into a file buffer, counts as the slot's for the card path (file_unit_in_slab), and is a
 * file-tier unit again (published as null) once taken back. */
TEST(p1_lend_ring_residents_are_published_at_their_slots_and_aliased_never) {
    const std::string path = "p1_stager_test_ring.rad";
    {
        std::vector<uint8_t> img((size_t)(2 * RAD_ALIGN_UNIT), 7);
        FILE* fp = std::fopen(path.c_str(), "wb");
        REQUIRE(fp != nullptr);
        REQUIRE(std::fwrite(img.data(), 1, img.size(), fp) == img.size());
        std::fclose(fp);
        DirectFile probe;
        if (probe.open(path.c_str(), RAD_DIO_READ) < 0) {
            std::printf("  SKIP the working directory has no O_DIRECT\n");
            std::remove(path.c_str());
            return;
        }
    }
    const int64_t gu = ds41p1::kGuBlockBytes, dn = ds41p1::kDnBlockBytes;
    Program prog;
    prog.weights.resize(1);
    Plan plan;
    plan.rank = 0;
    plan.world_size = 1;
    plan.dynamic = false;
    SlabClass sc;
    sc.name = "expert";
    sc.stride = ds41p1::kSlotBytes;
    sc.n_units = 2;
    plan.classes.push_back(sc);
    for (int e = 0; e < 2; ++e) {
        MoveUnit mu;
        mu.layer = 3;
        mu.expert = e;
        mu.slab_class = 0;
        mu.bytes = mu.slot_bytes = ds41p1::kSlotBytes;
        mu.access = RAD_ACCESS_CONDITIONAL;
        mu.tier = Tier::SSD;
        mu.site = Site::DeviceStaged;
        for (int k = 0; k < 2; ++k) {
            WeightInfo w;
            w.name = "blk.3.ffn_" + std::string(k ? "down" : "gate_up") + "." + std::to_string(e);
            w.decl.dtype = RAD_U8;
            w.decl.rank = 1;
            w.decl.shape[0] = k ? dn : gu;
            w.decl.access = RAD_ACCESS_CONDITIONAL;
            w.decl.group.layer = 3;
            w.decl.group.expert = e;
            w.logical_bytes = w.stored_bytes = k ? dn : gu;
            w.align = RAD_ALIGN_UNIT;
            w.layout_tag = "ggml_mxfp4";
            w.file_offset = 0;
            prog.weights.push_back(w);
            mu.weights.push_back((rad_weight)(prog.weights.size() - 1));
        }
        mu.offset = { 0, gu };
        plan.units.push_back(mu);
    }
    plan.unit_of_weight.assign(prog.weights.size(), -1);
    for (size_t u = 0; u < plan.units.size(); ++u)
        for (rad_weight w : plan.units[u].weights) plan.unit_of_weight[w] = (int32_t)u;
    Mover m;
    MoverConfig mc;
    mc.container_path = path;
    mc.stage_slots = 0;
    REQUIRE_EQ(m.init(prog, plan, mc), RAD_OK);
    ResidencyTable* rt = m.residency();
    alignas(64) static char slot[64];
    alignas(64) static char alias[64];
    CHECK(rt->lookup(plan.units[0].weights[0])->ptr == nullptr);   /* a file-tier unit has no address */
    CHECK(!m.ring_resident(0));
    CHECK(!m.file_unit_in_slab(0));
    m.set_ring_resident(0, slot);
    CHECK(m.ring_resident(0));
    CHECK(m.file_unit_in_slab(0));
    CHECK(!m.ring_resident(1));
    CHECK((const char*)m.unit_base(0) == slot);
    CHECK((const char*)rt->lookup(plan.units[0].weights[0])->ptr == slot);
    CHECK((const char*)rt->lookup(plan.units[0].weights[1])->ptr == slot + gu);
    /* Aliases, marked or not, leave it where it is and take the other. */
    std::vector<int32_t> staged;
    m.alias_file_units({ 0, 1 }, alias, &staged);
    REQUIRE(staged.size() == (size_t)1);
    CHECK_EQ(staged[0], 1);
    CHECK((const char*)rt->lookup(plan.units[0].weights[0])->ptr == slot);
    CHECK((const char*)rt->lookup(plan.units[1].weights[0])->ptr == alias);
    staged.clear();
    m.alias_file_units({ 0 }, alias, &staged, ds41p1::kMarkSkip);
    CHECK(staged.empty());
    /* Releasing a layer's staged units republishes the resident at its slot, the rest as null. */
    m.unstage_units({ 0, 1 });
    CHECK((const char*)rt->lookup(plan.units[0].weights[1])->ptr == slot + gu);
    CHECK(rt->lookup(plan.units[1].weights[0])->ptr == nullptr);
    Mover::HostUnit hu;
    CHECK(!m.borrow_host_unit(0, &hu, false));
    /* Taken back: a file-tier unit again. */
    m.set_ring_resident(0, nullptr);
    CHECK(!m.ring_resident(0));
    CHECK(!m.file_unit_in_slab(0));
    CHECK(rt->lookup(plan.units[0].weights[0])->ptr == nullptr);
    CHECK(m.unit_base(0) == nullptr);
    staged.clear();
    m.alias_file_units({ 0 }, alias, &staged);
    CHECK_EQ(staged.size(), (size_t)1);
    std::remove(path.c_str());
}

/* ---- the admission predicate --------------------------------------------------------------------------- */
TEST(p1_admission_follows_the_sorted_order_past_the_admission_row) {
    const int64_t C = 256, T = C * 6;
    FileStager fs;
    std::mt19937 rng(17);
    for (int trial = 0; trial < 20; ++trial) {
        const int64_t rows = 17 + (int64_t)(rng() % (unsigned)(C - 16));
        const int32_t admit_row = (int32_t)(rng() % (unsigned)(rows + 8));
        EpCountsHostProbe::admission(fs, C, rows, admit_row, true);
        /* moe_scatter's order: each slot (token * 6 + k) of this rank's experts, grouped by expert in
         * token order; slots routed to other ranks past off[96]. */
        std::vector<std::vector<int32_t>> by(96);
        std::vector<int32_t> remote;
        for (int64_t t = 0; t < rows; ++t)
            for (int k = 0; k < 6; ++k) {
                const int g = (int)(rng() % 384);
                if (g < 96) by[(size_t)g].push_back((int32_t)(t * 6 + k));
                else remote.push_back((int32_t)(t * 6 + k));
            }
        int32_t* idx = EpCountsHostProbe::idx(fs);
        int32_t* off = idx + EpCountsHostProbe::sorted_max(fs);
        int32_t at = 0;
        for (int e = 0; e < 96; ++e) {
            off[e] = at;
            for (int32_t s : by[(size_t)e]) idx[at++] = s;
        }
        off[96] = at;
        for (int32_t s : remote) idx[at++] = s;
        REQUIRE(at == rows * 6 && at <= T);
        for (int e = 0; e < 96; ++e) {
            bool want = admit_row <= 0;
            for (int32_t s : by[(size_t)e]) want = want || (s / 6 >= admit_row && admit_row < rows);
            if (admit_row >= rows) want = false;
            CHECK_EQ(EpCountsHostProbe::admits(fs, e), want);
        }
        /* Without the order fetched, nothing is admitted (unless the row is 0: everything). */
        EpCountsHostProbe::admission(fs, C, rows, admit_row, false);
        for (int e = 0; e < 96; e += 7) CHECK_EQ(EpCountsHostProbe::admits(fs, e), admit_row <= 0);
    }
}

/* ---- a layer the per-row cut leaves short ------------------------------------------------------------------
 *
 * At the threshold 17 (..._P1_MIN_ROWS=17, the card path's bound before the threshold existed).
 * Under the consumer skip's per-row cut a routed layer of an armed pass runs on its rows [row0(L), n) alone, so a deep
 * layer of a prompt's last chunk can be left with 16 rows or fewer, which the plugin serves on the gemv rows (no skip
 * mark honoured). The card path is decided a layer at a time: such a layer takes the CPU path as a 16-row pass does
 * (no marks, no waves), its miss bound held to its own rows. Over the end-aligned chunks of every prompt length to
 * 6,000 at C = 2048 and 1024 (ds41's 43 layers, the cut from layer 20, groups of 16), every armed layer's choice is
 * its own rows' (ds41p1::card_layer), a disarmed pass's never the card path's, and short layers occur. The bytes of
 * such a layer on that path against 16-row passes: r4d_emu_ds41p1's cut cases. */
TEST(p1_card_path_is_decided_per_layer_under_the_per_row_cut) {
    const ConsumerShape c16{ 43, 20, 128, 16, 16 };
    FileStager fs;
    std::vector<int64_t> f((size_t)c16.n_layer);
    std::vector<int32_t> row0((size_t)c16.n_layer);
    int64_t short_layers = 0, card_layers = 0, chunks = 0;
    bool short_after_card = false;
    for (int64_t C : { 2048, 1024 })
        for (int64_t P = 17; P <= 6000; P += (P < 300 ? 1 : 7)) {
            REQUIRE(consumer_thresholds(c16, P, f.data()));
            for (int64_t s = 0, e = ds41p1::first_chunk_end_aligned(P, C, 16); s < P; s = e, e = std::min(P, e + C)) {
                const int64_t n = e - s;
                REQUIRE(consumer_row_starts(c16, f.data(), s, e, row0.data()));
                const bool armed = n >= ds41p1::kMinArmRows && n <= C;
                EpCountsHostProbe::cut(fs, armed, n, row0, c16.n_layer, ds41p1::kMinArmRows, C);
                ++chunks;
                bool seen_card = false;
                for (int32_t li = 0; li < c16.n_layer; ++li) {
                    const int64_t r = EpCountsHostProbe::rows(fs, li);
                    REQUIRE(r >= 0 && r <= n);
                    CHECK_EQ(r, n - row0[(size_t)li]);
                    const bool want = armed && r >= ds41p1::kMinArmRows;
                    CHECK_EQ(EpCountsHostProbe::card(fs, li), want);
                    CHECK_EQ(ds41p1::card_layer(r), r >= 17);
                    CHECK_EQ(EpCountsHostProbe::miss_need(fs, li), armed ? r : n);
                    if (armed && r > 0 && r < ds41p1::kMinArmRows) {
                        ++short_layers;
                        short_after_card = short_after_card || seen_card;
                    }
                    card_layers += want;
                    seen_card = seen_card || want;
                }
            }
        }
    /* Each length's last chunk leaves its deepest layers on the last rows' group: 16 or fewer. */
    CHECK(short_layers > 1000);
    CHECK(card_layers > 1000);
    CHECK(short_after_card);   /* the armed passes where card-path layers precede the short ones */
    /* Without the cut, an armed pass's every layer is the card path's. */
    EpCountsHostProbe::cut(fs, true, 2048, {}, c16.n_layer, ds41p1::kMinArmRows, 2048);
    for (int32_t li = 0; li < c16.n_layer; ++li) {
        CHECK(EpCountsHostProbe::card(fs, li));
        CHECK_EQ(EpCountsHostProbe::miss_need(fs, li), 2048);
    }
    std::printf("  %lld chunks: %lld armed layers of 16 rows or fewer on the CPU path, %lld on the card path\n",
                (long long)chunks, (long long)short_layers, (long long)card_layers);
    EpCountsHostProbe::disarm(fs);
}

RAD_TEST_MAIN()
