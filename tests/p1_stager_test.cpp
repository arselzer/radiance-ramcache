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
#include "place/card_expert_stager.h"
#include "place/stager.h"
#include "sched/consumer_skip.h"
#include "sched/p1_chunk.h"

#include <algorithm>
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
     * layer; empty: no cut), armed for P1 or not. */
    static void cut(FileStager& fs, bool armed, int64_t pass_tok, const std::vector<int32_t>& row0, int32_t n) {
        fs.pass_tok_ = pass_tok;
        fs.p1_armed_ = armed;
        fs.pass_row0_ = row0;
        fs.layers_.assign((size_t)n, StagedLayer{});
        for (int32_t li = 0; li < n; ++li) fs.layers_[(size_t)li].layer = li;
    }
    static int64_t rows(const FileStager& fs, int32_t li) { return fs.layer_rows(li); }
    static bool card(const FileStager& fs, int32_t li) { return fs.p1_layer_on(li); }
    static int64_t miss_need(const FileStager& fs, int32_t li) { return fs.hx_miss_need(li); }
    static void disarm(FileStager& fs) { fs.p1_armed_ = false; fs.layers_.clear(); fs.pass_row0_.clear(); }
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

/* One routed layer's units on the ring: (expert, rows, drive?, packed?). */
struct LUnit { int e, rows; bool drive, packed; };

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
                EpCountsHostProbe::cut(fs, armed, n, row0, c16.n_layer);
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
    EpCountsHostProbe::cut(fs, true, 2048, {}, c16.n_layer);
    for (int32_t li = 0; li < c16.n_layer; ++li) {
        CHECK(EpCountsHostProbe::card(fs, li));
        CHECK_EQ(EpCountsHostProbe::miss_need(fs, li), 2048);
    }
    std::printf("  %lld chunks: %lld armed layers of 16 rows or fewer on the CPU path, %lld on the card path\n",
                (long long)chunks, (long long)short_layers, (long long)card_layers);
    EpCountsHostProbe::disarm(fs);
}

RAD_TEST_MAIN()
