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
    static int64_t need_rows(FileStager& fs, int64_t lcp, int p1_state, bool lend) {
        fs.lcp_ = lcp;
        fs.p1_state_ = p1_state;
        fs.p1_lend_asked_ = lend;
        return fs.hx_need_rows();
    }
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
    /* The lent slots' heat (..._P1_LEND_HEAT): a stager over `plan` with the routed layers `layers` of a model
     * of n_model trunk layers, a counts row on the host and the heat on at the half-life `half_rows`. */
    static bool heat(FileStager& fs, const Plan* plan, const std::vector<StagedLayer>& layers, int32_t n_model,
                     int half_rows) {
        fs.plan_ = plan;
        fs.layers_ = layers;
        fs.n_model_layers_ = n_model;
        if (!fs.counts_host_) {
            fs.counts_host_ = (int32_t*)rad_dev_alloc(512 * 4, RAD_MEM_HOST_PINNED);
            fs.counts_cap_ = 512;
        }
        fs.p1_heat_ = std::make_unique<p1::LendHeat>();
        return fs.counts_host_ && fs.p1_heat_->init(plan->units.size(), 2, half_rows, 150) == 0;
    }
    static int32_t* counts(FileStager& fs) { return fs.counts_host_; }
    /* What begin_pass does for the heat with pass `b`; whether the pass is counted. */
    static bool heat_pass(FileStager& fs, const RadBatch& b) {
        fs.p1_heat_pass_ = FileStager::p1_heat_counts(b);
        if (fs.p1_heat_pass_) fs.p1_heat_->begin_pass(b.n_tok);
        return fs.p1_heat_pass_;
    }
    /* The count sites' line (start_ep_routed, hx_early_decide): layer li's counts are on the host. */
    static void heat_layer(FileStager& fs, int32_t li) {
        if (fs.p1_heat_pass_) fs.p1_heat_count(li);
    }
    static double heat_of(const FileStager& fs, int32_t u) { return fs.p1_heat_->heat(u); }
    static bool heat_on(const FileStager& fs) { return fs.p1_heat_ != nullptr || fs.p1_heat_pass_; }
    static void heat_off(FileStager& fs) {
        fs.p1_heat_.reset();
        fs.p1_heat_pass_ = false;
        fs.plan_ = nullptr;
        fs.layers_.clear();
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
    CHECK(!t.swiglu_span);
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

/* The lent slots' heat: off unless asked for (0, unset or empty), 1 to 8 retargets a pass; the half-life in
 * decode rows and the margin in percent have their defaults and ranges. */
TEST(p1_lend_heat_switch_is_default_off_and_its_tunables_keep_their_ranges) {
    std::map<std::string, std::string> env;
    auto get = [&](const char* k) -> const char* {
        const auto it = env.find(k);
        return it == env.end() ? nullptr : it->second.c_str();
    };
    Tunables t;
    std::string why;
    REQUIRE(parse_tunables(get, &t, &why));
    CHECK_EQ(t.lend_heat, 0);
    CHECK_EQ(t.lend_heat_half, 100);
    CHECK_EQ(t.lend_heat_half, ds41p1::kDefLendHeatHalfRows);
    CHECK_EQ(t.lend_heat_margin, 150);
    CHECK_EQ(t.lend_heat_margin, ds41p1::kDefLendHeatMarginPct);
    for (const char* off : { "", "0" }) {
        env[ds41p1::kLendHeatEnv] = off;
        REQUIRE(parse_tunables(get, &t, &why));
        CHECK_EQ(t.lend_heat, 0);
    }
    for (const char* ok : { "1", "2", "8" }) {
        env[ds41p1::kLendHeatEnv] = ok;
        REQUIRE(parse_tunables(get, &t, &why));
        CHECK_EQ(t.lend_heat, std::atoi(ok));
    }
    for (const char* bad : { "9", "-1", "on", "1.5", " 2" }) {
        env[ds41p1::kLendHeatEnv] = bad;
        CHECK(!parse_tunables(get, &t, &why));
        CHECK(why.find(ds41p1::kLendHeatEnv) != std::string::npos);
    }
    env[ds41p1::kLendHeatEnv] = "2";
    env[ds41p1::kLendHeatHalfEnv] = "400";
    env[ds41p1::kLendHeatMarginEnv] = "200";
    REQUIRE(parse_tunables(get, &t, &why));
    CHECK_EQ(t.lend_heat, 2);
    CHECK_EQ(t.lend_heat_half, 400);
    CHECK_EQ(t.lend_heat_margin, 200);
    for (const char* bad : { "0", "1000001", "x" }) {
        env[ds41p1::kLendHeatHalfEnv] = bad;
        CHECK(!parse_tunables(get, &t, &why));
        CHECK(why.find(ds41p1::kLendHeatHalfEnv) != std::string::npos);
    }
    env[ds41p1::kLendHeatHalfEnv] = "1000000";
    for (const char* bad : { "99", "1001", "1.5" }) {   /* below 100 a colder unit would take a slot */
        env[ds41p1::kLendHeatMarginEnv] = bad;
        CHECK(!parse_tunables(get, &t, &why));
        CHECK(why.find(ds41p1::kLendHeatMarginEnv) != std::string::npos);
    }
    env[ds41p1::kLendHeatMarginEnv] = "100";
    REQUIRE(parse_tunables(get, &t, &why));
    CHECK_EQ(t.lend_heat_half, 1000000);
    CHECK_EQ(t.lend_heat_margin, 100);
}

TEST(p1_swiglu_span_switch_is_default_off_and_accepts_only_zero_or_one) {
    std::string value;
    bool set = false;
    auto get = [&](const char* k) -> const char* {
        return set && !std::strcmp(k, kSwigluSpanEnv) ? value.c_str() : nullptr;
    };
    Tunables t;
    std::string why;
    REQUIRE(parse_tunables(get, &t, &why));
    CHECK(!t.swiglu_span);
    set = true;
    for (const char* v : { "", "0", "1" }) {
        value = v;
        REQUIRE(parse_tunables(get, &t, &why));
        CHECK_EQ(t.swiglu_span, value == "1");
    }
    for (const char* v : { "2", "01", "yes", "-1", "1 " }) {
        value = v;
        CHECK(!parse_tunables(get, &t, &why));
        CHECK(why.find(kSwigluSpanEnv) != std::string::npos);
    }
}

TEST(p1_swiglu_span_launch_guard_keeps_unknown_stateful_and_scratch_rows_full) {
    RadKernelInfo row{};
    row.name = "swiglu_clamped_bf16"; row.op = "swiglu_clamped"; row.domain = RAD_DOMAIN_DEVICE;
    const RadLaunchFn launch = +[](const RadArgs*, RadStream) -> int { return RAD_OK; };
    row.launch = launch;
    CHECK(swiglu_span_launch_ok(&row, launch));
    CHECK(!swiglu_span_launch_ok(nullptr, launch));
    CHECK(!swiglu_span_launch_ok(&row, nullptr));
    CHECK(!swiglu_span_launch_ok(&row, +[](const RadArgs*, RadStream) -> int { return RAD_E_INVAL; }));
    RadKernelInfo bad = row;
    bad.name = "another_swiglu";
    CHECK(!swiglu_span_launch_ok(&bad, launch));
    bad = row; bad.op = "silu_mul";
    CHECK(!swiglu_span_launch_ok(&bad, launch));
    bad = row; bad.domain = RAD_DOMAIN_HOST;
    CHECK(!swiglu_span_launch_ok(&bad, launch));
    bad = row;
    bad.init = +[](const RadParam*, int, int, int, void**) -> int { return RAD_OK; };
    CHECK(!swiglu_span_launch_ok(&bad, launch));
    bad = row;
    bad.scratch = +[](const RadArgs*) -> int64_t { return 0; };
    CHECK(!swiglu_span_launch_ok(&bad, launch));
}

TEST(p1_swiglu_wave_span_includes_resident_gaps_last_expert_and_excludes_tail) {
    const int32_t counts[6] = { 2, 0, 3, 4, 0, 2 };
    const void* skip = ds41p1::mark(g_alias, ds41p1::kMarkSkip);
    std::vector<const void*> gu(6, skip), dn(6, skip);
    auto live = [&](int e) { gu[(size_t)e] = dn[(size_t)e] = ds41p1::mark(g_alias, ds41p1::kMarkPacked); };
    SwigluSpan span;
    live(2);
    live(5);
    REQUIRE(wave_swiglu_span(counts, 6, gu.data(), dn.data(), 14, &span));
    CHECK_EQ(span.lo, int64_t(2));
    CHECK_EQ(span.hi, int64_t(11));   /* resident expert 3's gap included; three tail rows excluded */
    gu[2] = dn[2] = skip;
    REQUIRE(wave_swiglu_span(counts, 6, gu.data(), dn.data(), 14, &span));
    CHECK_EQ(span.lo, int64_t(9));
    CHECK_EQ(span.hi, int64_t(11));
    gu[5] = dn[5] = skip;
    live(1);   /* an unskipped expert with no rows is still an empty wave */
    REQUIRE(wave_swiglu_span(counts, 6, gu.data(), dn.data(), 14, &span));
    CHECK_EQ(span.lo, int64_t(0));
    CHECK_EQ(span.hi, int64_t(0));
    dn[1] = skip;
    CHECK(!wave_swiglu_span(counts, 6, gu.data(), dn.data(), 14, &span));
    gu[1] = skip;
    CHECK(!wave_swiglu_span(counts, 6, gu.data(), dn.data(), 10, &span));
    int32_t bad[6] = { 2, 0, -1, 4, 0, 2 };
    CHECK(!wave_swiglu_span(bad, 6, gu.data(), dn.data(), 14, &span));
}

TEST(p1_swiglu_span_views_keep_pitches_parameters_and_full_replay_fallbacks) {
    std::vector<uint16_t> g(17 * 10), y(17 * 7);
    RadTensor ts[2]{};
    for (int k = 0; k < 2; ++k) {
        ts[k].rank = 2;
        ts[k].dtype = RAD_BF16;
        ts[k].shape[0] = 17;
        ts[k].shape[1] = k ? 4 : 8;
        ts[k].stride[0] = k ? 7 : 10;
        ts[k].stride[1] = 1;
        ts[k].data = k ? (void*)y.data() : (void*)g.data();
    }
    RadParam ps[] = { RAD_INT("M", 17), RAD_INT("n", 4), RAD_F64("limit", 10), RAD_STR("dtype", "bf16") };
    RadArgs full{};
    full.t = ts; full.n_t = 2; full.p = ps; full.n_p = 4; full.rank = 2; full.world_size = 4;
    std::vector<RadTensor> view;
    std::vector<RadParam> p;
    RadArgs a{};
    REQUIRE(swiglu_span_view(&full, { 3, 16 }, &view, &p, &a));
    CHECK(view[0].data == g.data() + 3 * 10);
    CHECK(view[1].data == y.data() + 3 * 7);
    CHECK_EQ(view[0].stride[0], int64_t(10));
    CHECK_EQ(view[1].stride[0], int64_t(7));
    CHECK_EQ(view[0].shape[0], int64_t(13));
    CHECK_EQ(view[1].shape[1], int64_t(4));
    CHECK_EQ(p[0].ival, 13);
    CHECK_EQ(p[1].ival, 4);
    CHECK_EQ(p[2].dval, 10.0);
    CHECK(!std::strcmp(p[3].sval, "bf16"));
    CHECK_EQ(a.rank, 2); CHECK_EQ(a.world_size, 4);
    CHECK_EQ(ts[0].shape[0], int64_t(17));
    CHECK_EQ(ps[0].ival, 17);
    REQUIRE(swiglu_span_view(&full, { 0, 0 }, &view, &p, &a));
    CHECK_EQ(view[0].shape[0], int64_t(0));   /* caller skips, never launches M=0 */
    for (int bad = 0; bad < 9; ++bad) {
        RadTensor tbad[2] = { ts[0], ts[1] };
        RadArgs b = full;
        b.t = tbad;
        if (bad == 0) tbad[0].stride[1] = 2;
        if (bad == 1) tbad[1].stride[0] = 3;
        if (bad == 2) tbad[1].shape[0] = 16;
        if (bad == 3) tbad[0].dtype = RAD_F16;
        if (bad == 4) b.instance = (void*)(uintptr_t)1;
        if (bad == 5) b.scratch_bytes = 4;
        if (bad == 6) tbad[1].data = g.data();   /* overlapping operands */
        if (bad == 7) tbad[0].stride[0] = INT64_MAX;
        if (bad == 8) tbad[0].data = (char*)g.data() + 1;
        const void* before = view[0].data;
        CHECK(!swiglu_span_view(&b, { 3, 16 }, &view, &p, &a));
        CHECK(view[0].data == before);   /* failed view cannot alter the full replay */
    }
    CHECK(!swiglu_span_view(&full, { -1, 1 }, &view, &p, &a));
    CHECK(!swiglu_span_view(&full, { 16, 18 }, &view, &p, &a));
}

TEST(p1_swiglu_span_composition_preserves_ready_skipped_padding_and_tail_rows_in_many_waves) {
    std::array<int32_t, 96> counts;
    int rows = 5;   /* other ranks' tail */
    for (int e = 0; e < 96; ++e) { counts[(size_t)e] = e % 4; rows += e % 4; }
    constexpr int n = 4, gp = 12, yp = 7, guard = 8;
    std::vector<uint16_t> g((size_t)rows * gp + 2 * guard, 0x7777),
                          y((size_t)rows * yp + 2 * guard, 0xA55A), ref = y;
    RadTensor ts[2]{};
    for (int k = 0; k < 2; ++k) {
        ts[k].data = k ? (void*)(y.data() + guard) : (void*)(g.data() + guard);
        ts[k].dtype = RAD_BF16; ts[k].rank = 2;
        ts[k].shape[0] = rows; ts[k].shape[1] = k ? n : 2 * n;
        ts[k].stride[0] = k ? yp : gp; ts[k].stride[1] = 1;
    }
    RadParam ps[] = { RAD_INT("M", rows), RAD_INT("n", n), RAD_F64("limit", 10) };
    RadArgs full{};
    full.t = ts; full.n_t = 2; full.p = ps; full.n_p = 3;
    /* An integer row-local stand-in checks the actual view's byte addresses and untouched regions;
     * the device wave test separately holds SwiGLU's floating-point expression to its full launch. */
    auto transform = [&](const RadTensor& in, const RadTensor& out) {
        for (int64_t r = 0; r < out.shape[0]; ++r)
            for (int c = 0; c < n; ++c)
                ((uint16_t*)out.data)[r * out.stride[0] + c] =
                    (uint16_t)(((const uint16_t*)in.data)[r * in.stride[0] + c] ^ (uint16_t)(c * 37));
    };
    transform(ts[0], ts[1]);   /* initial resident launch also wrote skipped rows from their old G */
    const void* skip = ds41p1::mark(g_alias, ds41p1::kMarkSkip);
    for (int wave = 0; wave < 32; ++wave) {   /* only a few residents: every other expert is streamed */
        std::vector<const void*> gu(96, skip), dn(96, skip);
        int off = 0;
        for (int e = 0; e < 96; ++e) {
            if (e % 32 == wave) {
                gu[(size_t)e] = dn[(size_t)e] = ds41p1::mark(g_alias, e % 2 ? ds41p1::kMarkPacked : 0);
                for (int r = off; r < off + counts[(size_t)e]; ++r)
                    for (int c = 0; c < 2 * n; ++c) g[(size_t)guard + (size_t)r * gp + c] = (uint16_t)(1000 + e + c);
            }
            off += counts[(size_t)e];
        }
        ref = y;
        RadTensor expected = ts[1]; expected.data = ref.data() + guard;
        transform(ts[0], expected);
        SwigluSpan span;
        REQUIRE(wave_swiglu_span(counts.data(), 96, gu.data(), dn.data(), rows, &span));
        std::vector<RadTensor> view;
        std::vector<RadParam> p;
        RadArgs a{};
        REQUIRE(swiglu_span_view(&full, span, &view, &p, &a));
        if (span.hi > span.lo) transform(view[0], view[1]);
        CHECK(y == ref);   /* exact ready/skipped rows, padding and both guards after every wave */
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
    for (int64_t C : { 1024, 2048, 4096 })
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
    CHECK_EQ(ds41p1::cpu_rows(4096, 4096), (int64_t)4095);
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
    for (int64_t C : { 1024, 2048, 4096 })
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
 * a card-path pass runs, and every source is let go once, after its read and its copy. A pass that takes the
 * slots back fences every copy still queued (the lent arena's slots are written by the pass's own ops), and
 * fences nothing else. */
namespace {
struct FakeLend final : LendHost {
    struct S {
        int  src = 0;                  /* 0 none, 1 a reload buffer, 2 a RAM-cache lease */
        bool reading = false, copying = false, copy_done = false, published = false;
        uint64_t copy_gen = 0, content_gen = ~0ull, fenced_gen = ~0ull;
        int  loads = 0, copies = 0, fences = 0;
        /* The heat policy's retargets: the unit the slot is for now, the one its load in flight read, the one
         * its queued copy carries, the one whose bytes the slot holds, the one last published there; whether
         * the slot was fenced for a retarget, and whether its first copy since one is still to come; when
         * each step last happened (the host's call order). */
        int  unit = -1, load_unit = -1, copy_unit = -1, content_unit = -1, pub_unit = -1;
        bool rfenced = false, owes_fence = false;
        int  rfences = 0, renames = 0;
        uint64_t t_fence = 0, t_withdraw = 0, t_rename = 0, t_load = 0, t_copy = 0, t_publish = 0;
    };
    std::mt19937 rng;
    std::vector<S> s;
    std::vector<uint8_t> ram;          /* slot k's unit hits the RAM cache in the card's form */
    std::deque<std::pair<int, bool>> landings;
    int bufs = 2, bufs_used = 0, land_pct = 50, copy_pct = 50;
    uint64_t gen = 0;                  /* card-path passes begun */
    std::set<int> fail_load_once, fail_copy_once, fail_query_once, refuse, fail_rfence_once;
    std::map<int, int> next;           /* by slot: the unit the next retarget gives it */
    uint64_t seq = 0;                  /* host calls that matter to a retarget's order */
    int violations = 0;
    std::string first;

    FakeLend(unsigned seed, int n) : rng(seed) {
        s.assign((size_t)n, S{});
        ram.assign((size_t)n, 0);
        for (int k = 0; k < n; ++k) s[(size_t)k].unit = 100 + k;
    }
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
            s[(size_t)k].load_unit = s[(size_t)k].unit;
            s[(size_t)k].t_load = ++seq;
            landings.push_back({ k, true });
            return 1;
        }
        if (bufs_used >= bufs) return 0;
        ++bufs_used;
        s[(size_t)k].src = 1;
        s[(size_t)k].reading = true;
        ++s[(size_t)k].loads;
        s[(size_t)k].load_unit = s[(size_t)k].unit;
        s[(size_t)k].t_load = ++seq;
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
        if (x.load_unit != x.unit)
            bad("slot " + std::to_string(k) + "'s copy carries unit " + std::to_string(x.load_unit) + ", and the slot is unit " +
                std::to_string(x.unit) + "'s");
        /* The first copy into a slot since its retarget must come behind the fence the retarget placed. */
        if (x.owes_fence && !x.rfenced) bad("slot " + std::to_string(k) + " copied after a retarget with no fence before it");
        if (fail_copy_once.erase(k)) return -1;
        x.owes_fence = false;
        x.rfenced = false;
        x.copying = true;
        x.copy_done = false;
        x.copy_gen = gen;
        x.copy_unit = x.load_unit;
        x.t_copy = ++seq;
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
        if (x.copying && x.copy_done) {
            x.content_gen = x.copy_gen;
            x.content_unit = x.copy_unit;
        }
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
            if (x.content_unit != x.unit)
                bad("slot " + std::to_string(k) + " published as unit " + std::to_string(x.unit) + " holding unit " +
                    std::to_string(x.content_unit) + "'s bytes");
            x.published = true;
            x.pub_unit = x.unit;
            x.t_publish = ++seq;
        } else {
            if (!x.published) bad("slot " + std::to_string(k) + " withdrawn unpublished");
            x.published = false;
            x.content_gen = ~0ull;   /* the card path takes the slot (or, in a retarget, another unit does) */
            x.t_withdraw = ++seq;
        }
    }
    void fence(int k) override {
        S& x = s[(size_t)k];
        if (!x.copying) bad("slot " + std::to_string(k) + " fenced with no copy queued");
        x.fenced_gen = gen;
        ++x.fences;
    }
    /* The heat policy's retarget: fenced while its resident is still published and nothing of the slot is in
     * flight, renamed only once the resident is withdrawn. */
    int retarget_fence(int k) override {
        S& x = s[(size_t)k];
        if (fail_rfence_once.erase(k)) return -1;
        if (!x.published) bad("slot " + std::to_string(k) + " fenced for a retarget with no resident");
        if (x.src || x.reading || x.copying) bad("slot " + std::to_string(k) + " fenced for a retarget while it holds a source");
        x.rfenced = true;
        x.t_fence = ++seq;
        ++x.rfences;
        return 0;
    }
    void retarget(int k) override {
        S& x = s[(size_t)k];
        if (x.published) bad("slot " + std::to_string(k) + " given another unit while its resident is published");
        if (!x.rfenced) bad("slot " + std::to_string(k) + " given another unit with no fence");
        const auto it = next.find(k);
        if (it == next.end()) {
            bad("slot " + std::to_string(k) + " given no unit");
            return;
        }
        x.unit = it->second;
        next.erase(it);
        x.owes_fence = true;
        x.t_rename = ++seq;
        ++x.renames;
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
    std::vector<int> queued;
    for (size_t k = 0; k < h.s.size(); ++k)
        if (h.s[k].copying) queued.push_back((int)k);
    std::vector<int> fences0;
    for (const auto& x : h.s) fences0.push_back(x.fences);
    const int r = l.begin_pass(p1, now);
    if (p1 && h.published()) h.bad("a unit published while a card-path pass runs");
    if (l.residents() != h.published()) h.bad("the lender's residents and the host's published units differ");
    for (int k : queued)
        if (p1 && h.s[(size_t)k].fenced_gen != h.gen) h.bad("slot " + std::to_string(k) + "'s queued copy not fenced");
    for (size_t k = 0; k < h.s.size(); ++k)
        if (!p1 && h.s[k].fences != fences0[k]) h.bad("slot " + std::to_string(k) + " fenced by a pass that keeps the slots");
    return r;
}
}  // namespace

/* hx_init's bound: under P1 the CPU takes the layer passes below the threshold, and with the lent ring every
 * pass that does not arm P1 as well, or a decode step of more rows than the bound (or a mixed step) would stage
 * its misses into lent slots whose residents are still published (review of 01279cbe). */
TEST(p1_lend_makes_the_cpu_path_take_every_pass_that_does_not_arm) {
    FileStager fs;
    EpCountsHostProbe::cut(fs, false, 1, {}, 1, 320, 1024);
    CHECK_EQ(EpCountsHostProbe::need_rows(fs, 1024, 1, false), (int64_t)319);
    CHECK_EQ(EpCountsHostProbe::need_rows(fs, 1024, 1, true), (int64_t)1024);
    CHECK_EQ(EpCountsHostProbe::need_rows(fs, 1024, 0, false), (int64_t)1024);
    CHECK_EQ(EpCountsHostProbe::need_rows(fs, 0, 0, false), (int64_t)1024);
    EpCountsHostProbe::cut(fs, false, 1, {}, 1, 17, 1024);
    CHECK_EQ(EpCountsHostProbe::need_rows(fs, 1024, 1, false), (int64_t)16);
    CHECK_EQ(EpCountsHostProbe::need_rows(fs, 1024, 1, true), (int64_t)1024);
}

TEST(p1_lcp4096_keeps_full_miss_capacity_when_the_lent_arena_is_requested) {
    /* A smaller cache may miss every selected unit; lending never assumes those misses away. */
    FileStager fs;
    EpCountsHostProbe::cut(fs, false, 1, {}, 1, 320, 4096);
    CHECK_EQ(EpCountsHostProbe::need_rows(fs, 4096, 1, false), (int64_t)319);
    CHECK_EQ(EpCountsHostProbe::need_rows(fs, 4096, 1, true), (int64_t)4096);
    CHECK_EQ(EpCountsHostProbe::need_rows(fs, 4096, 0, false), (int64_t)4096);
    CHECK_EQ(EpCountsHostProbe::sorted_max(fs), (int64_t)6 * 4096);
}

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
    CHECK(bad.init(&h0, 2 * ds41p1::kMaxLend + 1, 0) < 0);
    CHECK_EQ(bad.init(&h0, 2 * ds41p1::kMaxLend, 0), 0);   /* the ring's and the arena's */
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
    CHECK_EQ(h.s[2].fences, 1);   /* and the pass's writes wait for it */
    CHECK_EQ(h.s[0].fences + h.s[1].fences, 0);
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
    /* Prompt lookup's verify steps (speculation beside the card path, core/place/ep_route_counts.h): a decode
     * pass of 2..16 rows of one sequence with speculative rows never arms, at any threshold the switch takes,
     * so it keeps the residents as a decode pass does and computes them in the plugin's own ops. */
    for (int64_t rows = 2; rows <= 16; ++rows) {
        RadBatch verify{};
        verify.phase = RAD_PHASE_DECODE;
        verify.n_tok = rows;
        verify.n_tok_decode = rows;
        verify.n_seq = verify.n_seq_decode = 1;
        verify.n_spec = (int32_t)rows - 1;
        for (int64_t thr : { 17, 64, 320, 1024 }) CHECK(!FileStager::p1_pass_arms(verify, thr, 1024));
        CHECK(!FileStager::p1_pass_arms(verify, ds41p1::kMinArmRows, 4096));
        const int before = h.published();
        lend_pass(l, h, FileStager::p1_pass_arms(verify, 320, 1024), t += 5);
        CHECK_EQ(l.residents(), 4);
        CHECK_EQ(h.published(), before);
    }
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

/* ---- the lent slots' heat (RADIANCE_EXPERIMENTAL_DS41_P1_LEND_HEAT) ----------------------------------------
 *
 * RingLender::retarget against the same simulated host, which now also keeps, by slot, the unit it is for, the
 * unit its load read, the unit its copy carries and the unit whose bytes it holds: a slot is fenced for a
 * retarget only while its resident is published and nothing of it is in flight, renamed only once the resident
 * is withdrawn, its first copy after that comes behind the fence, and it is published only holding the bytes of
 * the unit it is for. Then LendHeat alone (the decay, the choice, the hysteresis), the two together as
 * FileStager::p1_heat_policy drives them, and FileStager's count of a routed layer. */
namespace {
/* Decode passes until the lender is settled (every slot its resident's), at most `most`. */
bool lend_settle(RingLender& l, FakeLend& h, int64_t* t, int most = 400) {
    for (int i = 0; i < most && !l.settled(); ++i) lend_pass(l, h, false, *t += 10);
    return l.settled();
}
}  // namespace

TEST(p1_lend_heat_a_retarget_when_settled_fences_withdraws_reloads_copies_and_publishes_in_that_order) {
    FakeLend h(21, 4);
    h.land_pct = h.copy_pct = 100;
    RingLender l;
    REQUIRE_EQ(l.init(&h, 4, 1000), 0);
    CHECK(!l.settled());   /* lent: nothing loaded yet */
    int64_t t = 0;
    REQUIRE(lend_settle(l, h, &t));
    CHECK_EQ(l.residents(), 4);
    for (const auto& x : h.s) CHECK_EQ(x.loads, 1);
    const uint64_t reloaded0 = l.stats().reloaded;
    /* Slot 2's resident (unit 102) gives the slot to unit 900. */
    h.next[2] = 900;
    h.copy_pct = 0;
    const int k = 2;
    const uint64_t seq0 = h.seq;
    CHECK_EQ(l.retarget(&k, 1), 1);
    const FakeLend::S& x = h.s[2];
    CHECK(!x.published);   /* the old unit withdrawn: a file-tier unit again */
    CHECK(!l.resident(2));
    CHECK_EQ(l.residents(), 3);
    CHECK_EQ(h.published(), 3);
    CHECK(!l.settled());
    CHECK_EQ(x.unit, 900);
    CHECK_EQ(x.pub_unit, 102);   /* nothing of the new unit is published yet */
    CHECK_EQ(x.rfences, 1);
    CHECK_EQ(x.renames, 1);
    CHECK_EQ(x.fences, 0);       /* a take-back's fence is not the retarget's */
    CHECK(x.t_fence > seq0);
    CHECK(x.t_fence < x.t_withdraw);   /* the DMA lane waits before the resident goes */
    CHECK(x.t_withdraw < x.t_rename);
    CHECK(x.t_rename < x.t_load);      /* the load is the new unit's, begun at once (a buffer was free) */
    CHECK_EQ(x.loads, 2);
    CHECK_EQ(x.load_unit, 900);
    CHECK_EQ(x.copies, 1);             /* no copy yet: its read has not landed */
    CHECK_EQ(l.busy(), 1);
    CHECK_EQ(l.stats().retarget_withdrawn, (uint64_t)1);
    CHECK_EQ(l.stats().retargets, (uint64_t)0);
    CHECK_EQ(l.stats().withdrawn, (uint64_t)0);   /* the card path's count stays its own */
    /* The existing reload path does the rest: landed, copied on the DMA lane, published at a pass's begin once
     * the host has seen the copy pass. */
    lend_pass(l, h, false, t += 10);
    CHECK_EQ(x.copies, 2);
    CHECK(x.t_load < x.t_copy);
    CHECK(!x.published);
    CHECK(!l.settled());
    h.copy_pct = 100;
    lend_pass(l, h, false, t += 10);
    CHECK(x.published);
    CHECK(x.t_copy < x.t_publish);
    CHECK_EQ(x.pub_unit, 900);
    CHECK_EQ(x.content_unit, 900);
    CHECK(l.settled());
    CHECK_EQ(l.stats().retargets, (uint64_t)1);
    CHECK_EQ(l.stats().reloaded, reloaded0 + 1);
    CHECK_EQ(h.bufs_used, 0);
    for (int j : { 0, 1, 3 }) {   /* the other slots were never touched */
        CHECK_EQ(h.s[(size_t)j].loads, 1);
        CHECK_EQ(h.s[(size_t)j].pub_unit, 100 + j);
        CHECK_EQ(h.s[(size_t)j].rfences, 0);
        CHECK(h.s[(size_t)j].published);
    }
    /* Three at once with two reload buffers: the third waits wanted for a buffer, and all three come in. A slot
     * named twice or outside the lender is skipped. */
    const int ks[5] = { 0, 3, 1, 3, 9 };
    h.next[0] = 910;
    h.next[3] = 913;
    h.next[1] = 911;
    h.land_pct = 0;
    CHECK_EQ(l.retarget(ks, 5), 3);
    CHECK_EQ(l.residents(), 1);
    CHECK_EQ(h.bufs_used, 2);
    CHECK_EQ(h.s[0].loads, 2);
    CHECK_EQ(h.s[1].loads, 2);
    CHECK_EQ(h.s[3].loads, 1);   /* wanted: no buffer yet */
    CHECK(h.next.empty());
    h.land_pct = 100;
    REQUIRE(lend_settle(l, h, &t));
    CHECK_EQ(h.s[0].pub_unit, 910);
    CHECK_EQ(h.s[1].pub_unit, 911);
    CHECK_EQ(h.s[2].pub_unit, 900);
    CHECK_EQ(h.s[3].pub_unit, 913);
    CHECK_EQ(l.stats().retarget_withdrawn, (uint64_t)4);
    CHECK_EQ(l.stats().retargets, (uint64_t)4);
    for (uint64_t d : l.stats().declined) CHECK_EQ(d, (uint64_t)0);
    CHECK_EQ(h.violations, 0);
    if (h.violations) std::fprintf(stderr, "%s\n", h.first.c_str());
}

TEST(p1_lend_heat_a_retarget_is_refused_while_lent_in_the_grace_or_busy_and_changes_nothing) {
    FakeLend h(23, 3);
    h.land_pct = h.copy_pct = 100;
    RingLender l;
    REQUIRE_EQ(l.init(&h, 3, 500), 0);
    using D = RingLender::Decline;
    auto declined = [&](D d) { return l.stats().declined[(size_t)d]; };
    const int k1 = 1;
    h.next[1] = 901;
    /* Never returned yet: lent. */
    CHECK_EQ(l.retarget(&k1, 1), 0);
    CHECK_EQ(declined(D::Lent), (uint64_t)1);
    /* The first reloads in flight: busy. */
    int64_t t = 0;
    h.land_pct = 0;
    lend_pass(l, h, false, t += 10);
    CHECK(l.busy() > 0);
    CHECK_EQ(l.retarget(&k1, 1), 0);
    CHECK_EQ(declined(D::Busy), (uint64_t)1);
    h.land_pct = 100;
    REQUIRE(lend_settle(l, h, &t));
    /* A card-path pass takes the slots back: refused during it and through the grace. */
    lend_pass(l, h, true, t += 10);
    CHECK_EQ(l.retarget(&k1, 1), 0);
    CHECK_EQ(declined(D::Lent), (uint64_t)2);
    l.end_p1(t += 10);
    const int64_t end = t;
    lend_pass(l, h, false, end + 100);   /* inside the grace */
    CHECK(l.lent());
    CHECK_EQ(l.stats().held_back, (uint64_t)1);
    CHECK_EQ(l.retarget(&k1, 1), 0);
    CHECK_EQ(declined(D::Lent), (uint64_t)3);
    /* After the grace the reloads run: busy until every slot is its resident's again. */
    h.land_pct = 0;
    lend_pass(l, h, false, end + 500);
    CHECK(!l.lent());
    CHECK_EQ(l.retarget(&k1, 1), 0);
    CHECK_EQ(declined(D::Busy), (uint64_t)2);
    h.land_pct = 100;
    t = end + 500;
    REQUIRE(lend_settle(l, h, &t));
    /* The host cannot fence (no event behind the pass before this one): refused, the resident stays. */
    h.fail_rfence_once = { 1 };
    CHECK_EQ(l.retarget(&k1, 1), 0);
    CHECK_EQ(declined(D::Fence), (uint64_t)1);
    CHECK(l.settled());
    /* No refusal changed anything: every slot its first unit's, resident, never fenced or renamed. */
    for (int k = 0; k < 3; ++k) {
        CHECK_EQ(h.s[(size_t)k].unit, 100 + k);
        CHECK_EQ(h.s[(size_t)k].pub_unit, 100 + k);
        CHECK_EQ(h.s[(size_t)k].rfences, 0);
        CHECK_EQ(h.s[(size_t)k].renames, 0);
        CHECK(h.s[(size_t)k].published);
    }
    CHECK_EQ(l.stats().retarget_withdrawn, (uint64_t)0);
    /* A fence refused at the second slot of a call: the first is retargeted, the second stays. */
    const int two[2] = { 0, 2 };
    h.next[0] = 900;
    h.next[2] = 902;
    h.fail_rfence_once = { 2 };
    CHECK_EQ(l.retarget(two, 2), 1);
    CHECK_EQ(declined(D::Fence), (uint64_t)2);
    CHECK_EQ(h.s[0].unit, 900);
    CHECK_EQ(h.s[2].unit, 102);
    CHECK(h.s[2].published);
    h.next.erase(2);
    REQUIRE(lend_settle(l, h, &t));
    CHECK_EQ(h.s[0].pub_unit, 900);
    /* A reload that failed leaves its slot off until the next return: busy for the policy until then. */
    lend_pass(l, h, true, t += 10);
    l.end_p1(t += 10);
    h.fail_load_once = { 2 };
    t += 600;
    for (int i = 0; i < 20; ++i) lend_pass(l, h, false, t += 10);
    CHECK_EQ(l.residents(), 2);
    CHECK_EQ(l.busy(), 0);
    CHECK(!l.settled());
    const uint64_t busy0 = declined(D::Busy);
    CHECK_EQ(l.retarget(&k1, 1), 0);
    CHECK_EQ(declined(D::Busy), busy0 + 1);
    lend_pass(l, h, true, t += 10);
    l.end_p1(t += 10);
    t += 600;
    REQUIRE(lend_settle(l, h, &t));
    CHECK_EQ(l.retarget(&k1, 1), 1);   /* h.next[1] is still 901 */
    REQUIRE(lend_settle(l, h, &t));
    CHECK_EQ(h.s[1].pub_unit, 901);
    CHECK_EQ(h.violations, 0);
    if (h.violations) std::fprintf(stderr, "%s\n", h.first.c_str());
}

TEST(p1_lend_heat_a_card_path_pass_in_the_middle_of_a_retarget_cancels_cleanly_and_the_slot_recovers) {
    /* When the pass comes the retargeted slot is wanted (no buffer), reading, landed but not yet seen, or its
     * copy is queued. */
    for (int phase = 0; phase < 4; ++phase) {
        FakeLend h(31u + (unsigned)phase, 3);
        h.land_pct = h.copy_pct = 100;
        RingLender l;
        REQUIRE_EQ(l.init(&h, 3, 100), 0);
        int64_t t = 0;
        REQUIRE(lend_settle(l, h, &t));
        h.next[0] = 700;
        const int k = 0;
        h.land_pct = h.copy_pct = 0;
        if (phase == 0) h.bufs = 0;
        REQUIRE_EQ(l.retarget(&k, 1), 1);
        if (phase == 0) {
            CHECK_EQ(h.s[0].loads, 1);
            CHECK_EQ(l.busy(), 0);
        } else {
            CHECK(h.s[0].reading);
        }
        if (phase == 2) {
            h.s[0].reading = false;
            h.landings.push_back({ 0, true });
        }
        if (phase == 3) {
            h.land_pct = 100;
            lend_pass(l, h, false, t += 10);   /* landed, its copy queued and not passed */
            REQUIRE(h.s[0].copying);
            h.land_pct = 0;
        }
        const int fences0 = h.s[0].fences;
        /* The card path's pass: the two residents withdrawn, the retargeted slot's work overtaken. */
        CHECK_EQ(lend_pass(l, h, true, t += 10), 1);
        CHECK_EQ(l.residents(), 0);
        CHECK_EQ(h.published(), 0);
        CHECK_EQ(h.s[0].fences - fences0, phase == 3 ? 1 : 0);   /* only a queued copy is fenced */
        CHECK_EQ(l.stats().withdrawn, (uint64_t)2);
        l.end_p1(t += 10);
        h.bufs = 2;
        h.land_pct = h.copy_pct = 100;
        h.settle_all();
        lend_pass(l, h, false, t += 10);   /* inside the grace: the overtaken work only lets go */
        CHECK_EQ(l.residents(), 0);
        CHECK_EQ(h.published(), 0);
        CHECK_EQ(h.bufs_used, 0);
        CHECK_EQ(l.busy(), 0);
        CHECK_EQ(l.stats().cancelled, (uint64_t)(phase == 0 ? 0 : 1));
        /* After the grace every slot reloads, the retargeted one with its new unit, and the policy may go on. */
        t += 200;
        REQUIRE(lend_settle(l, h, &t));
        CHECK_EQ(h.s[0].pub_unit, 700);
        CHECK_EQ(h.s[0].content_unit, 700);
        CHECK_EQ(h.s[1].pub_unit, 101);
        CHECK_EQ(h.s[2].pub_unit, 102);
        CHECK_EQ(l.stats().retarget_withdrawn, (uint64_t)1);
        CHECK_EQ(l.stats().retargets, (uint64_t)1);
        h.next[1] = 701;
        const int k1 = 1;
        CHECK_EQ(l.retarget(&k1, 1), 1);
        REQUIRE(lend_settle(l, h, &t));
        CHECK_EQ(h.s[1].pub_unit, 701);
        CHECK_EQ(h.violations, 0);
        if (h.violations) std::fprintf(stderr, "phase %d: %s\n", phase, h.first.c_str());
    }
}

TEST(p1_lend_heat_1000_retargets_with_random_landings_and_card_path_passes_between_them) {
    std::mt19937 rng(77);
    const int n = 7;
    FakeLend h(78, n);
    RingLender l;
    REQUIRE_EQ(l.init(&h, n, 30), 0);
    std::vector<int> want((size_t)n);   /* the unit each slot was last given */
    for (int k = 0; k < n; ++k) want[(size_t)k] = 100 + k;
    int64_t t = 0;
    int next_unit = 1000, done = 0, refused = 0, bursts = 0;
    bool in_p1 = false;
    for (int pass = 0; pass < 200000 && done < 1000; ++pass) {
        t += 1 + (int64_t)(rng() % 20);
        h.land_pct = 20 + (int)(rng() % 81);
        h.copy_pct = 20 + (int)(rng() % 81);
        for (int k = 0; k < n; ++k) h.ram[(size_t)k] = rng() % 5 == 0;
        if (in_p1) l.end_p1(t);
        in_p1 = rng() % 23 == 0;
        bursts += in_p1;
        lend_pass(l, h, in_p1, t);
        /* Every pass asks for one to three slots, settled or not; a refusal changes nothing. */
        int ks[3];
        const int m = 1 + (int)(rng() % 3);
        std::set<int> named;
        for (int i = 0; i < m; ++i) {
            ks[i] = (int)(rng() % (unsigned)n);
            if (named.insert(ks[i]).second) h.next[ks[i]] = next_unit++;
        }
        const bool settled = l.settled();
        std::vector<int> before;
        for (const auto& x : h.s) before.push_back(x.unit);
        const int r = l.retarget(ks, m);
        if (!settled) {
            ++refused;
            if (r != 0) h.bad("a retarget went ahead while the lender was not settled");
            for (int k = 0; k < n; ++k)
                if (h.s[(size_t)k].unit != before[(size_t)k]) h.bad("a refused retarget renamed a slot");
            h.next.clear();
            continue;
        }
        if (r != (int)named.size()) h.bad("a settled lender retargeted " + std::to_string(r) + " of " + std::to_string(named.size()));
        for (int k : named) want[(size_t)k] = h.s[(size_t)k].unit;
        done += r;
        if (l.residents() != h.published()) h.bad("the lender's residents and the host's published units differ");
        if (h.bufs_used > h.bufs) h.bad("more buffers in use than the host has");
    }
    CHECK(done >= 1000);
    CHECK(refused > 0);
    CHECK(bursts > 0);
    if (in_p1) l.end_p1(t);
    h.land_pct = h.copy_pct = 100;
    t += 1000;
    REQUIRE(lend_settle(l, h, &t));
    for (int k = 0; k < n; ++k) {
        CHECK(h.s[(size_t)k].published);
        CHECK_EQ(h.s[(size_t)k].pub_unit, want[(size_t)k]);
        CHECK_EQ(h.s[(size_t)k].content_unit, want[(size_t)k]);
    }
    CHECK_EQ(l.stats().retarget_withdrawn, (uint64_t)done);
    CHECK_EQ(l.stats().retargets, (uint64_t)done);
    CHECK_EQ(l.stats().failed, (uint64_t)0);
    CHECK_EQ(h.bufs_used, 0);
    CHECK_EQ(l.busy(), 0);
    CHECK_EQ(h.violations, 0);
    if (h.violations) std::fprintf(stderr, "%s\n", h.first.c_str());
    std::fprintf(stderr, "  %d retargets, %d asks refused, %d card-path passes between them\n", done, refused, bursts);
}

TEST(p1_lend_heat_counts_decay_by_half_every_half_life_of_decode_rows) {
    LendHeat h;
    CHECK(h.init(8, 0, 100, 150) < 0);   /* S 0 is the switch off: no policy is made */
    CHECK(h.init(8, ds41p1::kMaxLendHeat + 1, 100, 150) < 0);
    CHECK(h.init(8, 2, 0, 150) < 0);
    CHECK(h.init(8, 2, 100, 99) < 0);
    REQUIRE_EQ(h.init(8, 2, 100, 150), 0);
    CHECK_EQ(h.max_pairs(), 2);
    CHECK(!h.dirty());
    CHECK(!h.warm());
    h.begin_pass(8);
    h.add(3, 8);
    CHECK(h.dirty());
    CHECK_NEAR(h.heat(3), 8.0, 1e-9);
    CHECK_EQ(h.heat(2), 0.0);
    h.add(-1, 5);   /* outside the plan, or no rows: nothing */
    h.add(8, 5);
    h.add(2, 0);
    h.add(2, -3);
    CHECK_EQ(h.heat(2), 0.0);
    CHECK_EQ(h.heat(8), 0.0);
    CHECK(!h.warm());
    /* 100 more decode rows halve it, as one pass or as many. */
    h.begin_pass(100);
    CHECK_NEAR(h.heat(3), 4.0, 1e-9);
    CHECK(h.warm());   /* a half-life of rows has been counted */
    for (int i = 0; i < 25; ++i) h.begin_pass(4);
    CHECK_NEAR(h.heat(3), 2.0, 1e-9);
    h.add(3, 6);       /* a count is worth its rows in the pass it comes in */
    CHECK_NEAR(h.heat(3), 8.0, 1e-9);
    h.begin_pass(0);   /* no rows: no time passes */
    h.begin_pass(-4);
    CHECK_NEAR(h.heat(3), 8.0, 1e-9);
    h.looked();
    CHECK(!h.dirty());
    h.again();
    CHECK(h.dirty());
    /* Far past where the scale is divided back, the heats are what the counts say: two rows a pass at a
     * half-life of one row sum to 4, one row to 2. */
    LendHeat g;
    REQUIRE_EQ(g.init(4, 1, 1, 150), 0);
    for (int i = 0; i < 5000; ++i) {
        g.begin_pass(1);
        g.add(0, 2);
        g.add(1, 1);
    }
    CHECK_NEAR(g.heat(0), 4.0, 1e-9);
    CHECK_NEAR(g.heat(1), 2.0, 1e-9);
    /* A pass of very many half-lives: finite, the older counts gone. */
    g.begin_pass(1000000);
    CHECK(g.heat(0) >= 0.0);
    CHECK(g.heat(0) < 1e-100);
    g.add(2, 3);
    CHECK_NEAR(g.heat(2), 3.0, 1e-9);
}

TEST(p1_lend_heat_equal_heat_never_swaps_and_a_swap_needs_the_margin_and_half_a_row) {
    LendHeat h;
    REQUIRE_EQ(h.init(16, 3, 100, 150), 0);
    /* No pass begins here, so every count stands as counted (whole numbers: the comparisons are exact). */
    const int32_t slot_unit[4] = { 0, 1, 2, 3 };
    const int32_t cand[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };   /* every lendable unit, the residents among them */
    std::vector<uint8_t> taken(16, 0);
    for (int u = 0; u < 4; ++u) taken[(size_t)u] = 1;
    LendHeat::Pair p[ds41p1::kMaxLendHeat];
    auto choose = [&] { return h.choose(slot_unit, 4, cand, 8, taken.data(), p); };
    CHECK_EQ(choose(), 0);   /* nothing counted: a unit never seen takes no slot from another never seen */
    for (int u = 0; u < 8; ++u) h.add(u, 10);
    for (int i = 0; i < 100; ++i) CHECK_EQ(choose(), 0);   /* equal heat: never, however often it is asked */
    h.add(5, 5);             /* 15 = 1.5 x 10: not above 1.5 x 10 + 0.5 */
    CHECK_EQ(choose(), 0);
    /* Exactly the margin and half a row is not above it: residents at 3, a candidate at 1.5 x 3 + 0.5 = 5. */
    LendHeat e;
    REQUIRE_EQ(e.init(16, 3, 100, 150), 0);
    for (int u = 0; u < 4; ++u) e.add(u, 3);
    e.add(6, 5);
    CHECK_EQ(e.choose(slot_unit, 4, cand, 8, taken.data(), p), 0);
    e.add(6, 1);
    REQUIRE_EQ(e.choose(slot_unit, 4, cand, 8, taken.data(), p), 1);
    CHECK_EQ(p[0].slot, 0);   /* equal residents: the lower slot */
    CHECK_EQ(p[0].unit, 6);
    /* A margin of 100%: still half a row. */
    LendHeat m;
    REQUIRE_EQ(m.init(16, 3, 100, 100), 0);
    for (int u = 0; u < 4; ++u) m.add(u, 3);
    m.add(6, 3);
    CHECK_EQ(m.choose(slot_unit, 4, cand, 8, taken.data(), p), 0);
    m.add(6, 1);
    CHECK_EQ(m.choose(slot_unit, 4, cand, 8, taken.data(), p), 1);
    /* The coldest slot takes the hottest candidate, the next coldest the next, while each pair clears the
     * margin; at most three here. Slots at 10, 2, 6, 30; candidates 4..7 at 40, 12, 4, 20. */
    LendHeat q;
    REQUIRE_EQ(q.init(16, 3, 100, 150), 0);
    const int rows[8] = { 10, 2, 6, 30, 40, 12, 4, 20 };
    for (int u = 0; u < 8; ++u) q.add(u, rows[u]);
    REQUIRE_EQ(q.choose(slot_unit, 4, cand, 8, taken.data(), p), 2);
    CHECK_EQ(p[0].slot, 1);
    CHECK_EQ(p[0].unit, 4);   /* 40 > 1.5 x 2 + 0.5 */
    CHECK_EQ(p[1].slot, 2);
    CHECK_EQ(p[1].unit, 7);   /* 20 > 1.5 x 6 + 0.5; then 12 is not above 1.5 x 10 + 0.5 */
    /* A unit that holds a slot is never chosen, whatever its heat (unit 4 here). */
    taken[4] = 1;
    REQUIRE_EQ(q.choose(slot_unit, 4, cand, 8, taken.data(), p), 2);
    CHECK_EQ(p[0].unit, 7);
    CHECK_EQ(p[1].unit, 5);
    taken[4] = 0;
    /* One retarget a pass: only the best pair. Equal candidates: the earlier one. */
    LendHeat one;
    REQUIRE_EQ(one.init(16, 1, 100, 150), 0);
    for (int u = 0; u < 8; ++u) one.add(u, rows[u]);
    one.add(7, 20);   /* units 4 and 7 both at 40 */
    REQUIRE_EQ(one.choose(slot_unit, 4, cand, 8, taken.data(), p), 1);
    CHECK_EQ(p[0].slot, 1);
    CHECK_EQ(p[0].unit, 4);
    /* A slot whose unit is not of this plan: nothing moves. */
    const int32_t odd[4] = { 0, 1, 99, 3 };
    CHECK_EQ(q.choose(odd, 4, cand, 8, taken.data(), p), 0);
    CHECK_EQ(q.choose(slot_unit, 0, cand, 8, taken.data(), p), 0);
}

/* The policy's turn (p1::heat_turn, the code FileStager::p1_heat_policy runs) over the fakes: behind the
 * lender's begin_pass, with the simulated host as its owner (a slot's unit is the host's; a retarget names the
 * slots' new units to the host and calls RingLender::retarget). The pass kinds, the draft bank's loan, the
 * fence's event and each unit's eligibility are the test's to set. */
namespace {
struct HeatSim {
    FakeLend h;
    RingLender l;
    LendHeat heat;
    std::vector<int32_t> cand;
    std::vector<uint8_t> taken;
    std::set<int32_t> ineligible;
    bool bank = false, fence_ready = true;
    int64_t t = 0;
    int moved = 0, most_a_pass = 0;
    HeatSim(unsigned seed, int n, int units, int s, int half) : h(seed, n) {
        h.land_pct = h.copy_pct = 100;
        (void)l.init(&h, n, 0);
        (void)heat.init((size_t)units, s, half, 150);
        taken.assign((size_t)units, 0);
        for (int k = 0; k < n; ++k) {
            h.s[(size_t)k].unit = k;
            taken[(size_t)k] = 1;
        }
        for (int u = 0; u < units; ++u) cand.push_back(u);
    }
    /* heat_turn's owner */
    int32_t slot_unit(int k) const { return h.s[(size_t)k].unit; }
    bool eligible(int32_t u) const { return !ineligible.count(u); }
    int retarget(const int* ks, const int32_t* units, int n) {
        for (int i = 0; i < n; ++i) h.next[ks[i]] = units[i];
        const int done = l.retarget(ks, n);
        h.next.clear();
        return done;
    }
    uint64_t declined(RingLender::Decline d) const { return l.stats().declined[(size_t)d]; }
    std::set<int32_t> held() const {
        std::set<int32_t> s;
        for (const auto& x : h.s) s.insert(x.unit);
        return s;
    }
    /* A pass: the lender's begin (p1: it takes the slots back), the policy's turn, then, for a decode pass
     * of `rows` rows, its counts: rows_by_unit[u] rows routed to unit u. */
    int pass(const std::vector<int>& rows_by_unit, int rows, bool p1 = false) {
        if (in_p1_) l.end_p1(t);
        in_p1_ = p1;
        lend_pass(l, h, p1, t += 10);
        const int done = heat_turn(l, heat, p1, bank, fence_ready, cand, taken, *this);
        moved += done;
        most_a_pass = std::max(most_a_pass, done);
        /* The taken marks are the units the slots are for, no more and no fewer. */
        size_t marks = 0;
        for (uint8_t m : taken) marks += m;
        if (marks != h.s.size()) h.bad("the taken marks and the slots differ in number");
        for (const auto& x : h.s)
            if (!taken[(size_t)x.unit]) h.bad("a slot's unit is not marked taken");
        if (p1) return done;
        heat.begin_pass(rows);
        for (size_t u = 0; u < rows_by_unit.size(); ++u) heat.add((int32_t)u, rows_by_unit[u]);
        return done;
    }

private:
    bool in_p1_ = false;
};
}  // namespace

TEST(p1_lend_heat_the_slots_follow_a_shift_in_routing_and_rest_while_it_is_steady) {
    /* Four slots hold units 0..3 of 12, and every unit takes one row a pass: equal heat, nothing ever moves. */
    HeatSim a(41, 4, 12, 2, 50);
    const std::vector<int> even(12, 1);
    for (int i = 0; i < 400; ++i) a.pass(even, 8);
    CHECK(a.l.settled());
    CHECK(a.heat.warm());
    CHECK_EQ(a.moved, 0);
    CHECK_EQ(a.l.stats().retarget_withdrawn, (uint64_t)0);
    /* The traffic shifts: units 8..11 take four rows a pass, the rest one. The slots follow, two a pass at
     * most, each slot once. */
    std::vector<int> shifted(12, 1);
    for (int u = 8; u < 12; ++u) shifted[(size_t)u] = 4;
    for (int i = 0; i < 400; ++i) a.pass(shifted, 8);
    CHECK_EQ(a.moved, 4);
    CHECK_EQ(a.most_a_pass, 2);
    CHECK(a.l.settled());
    CHECK(a.held() == (std::set<int32_t>{ 8, 9, 10, 11 }));
    for (size_t k = 0; k < 4; ++k) {
        CHECK(a.h.s[k].published);
        CHECK_EQ(a.h.s[k].pub_unit, a.h.s[k].unit);
    }
    CHECK_EQ(a.l.stats().retargets, (uint64_t)4);
    for (int u = 0; u < 12; ++u) CHECK_EQ((int)a.taken[(size_t)u], u >= 8 ? 1 : 0);
    /* Steady again: they stay (the withdrawn units' heat is below their residents'). */
    for (int i = 0; i < 400; ++i) a.pass(shifted, 8);
    CHECK_EQ(a.moved, 4);
    /* Nothing moves before a half-life of rows has been counted, however lopsided the first passes are. */
    HeatSim b(43, 2, 6, 2, 1000);
    std::vector<int> lop(6, 0);
    lop[4] = lop[5] = 8;
    for (int i = 0; i < 100; ++i) b.pass(lop, 8);   /* 800 rows */
    CHECK(b.l.settled());
    CHECK_EQ(b.moved, 0);
    for (int i = 0; i < 100; ++i) b.pass(lop, 8);
    CHECK_EQ(b.moved, 2);
    CHECK_EQ(a.h.violations, 0);
    CHECK_EQ(b.h.violations, 0);
    if (a.h.violations) std::fprintf(stderr, "%s\n", a.h.first.c_str());
    if (b.h.violations) std::fprintf(stderr, "%s\n", b.h.first.c_str());
}

/* The policy's turn does nothing, and counts why, on a pass that takes the slots back and through the grace,
 * beside a draft bank's loan, while a slot is busy, without the fence's event, and when the unit it chose can
 * hold no slot (which then leaves the candidates); each refusal leaves every slot as it was, and the turn
 * after the reason is gone goes ahead. */
TEST(p1_lend_heat_the_policy_declines_and_counts_why_and_goes_on_when_the_reason_is_gone) {
    using D = RingLender::Decline;
    /* Two slots hold units 0 and 1 of 8; units 5 and 6 take eight rows a pass (6 a little hotter), the rest none. */
    HeatSim s(51, 2, 8, 1, 16);
    std::vector<int> hot(8, 0);
    hot[5] = 8;
    hot[6] = 9;
    const std::vector<int> none(8, 0);
    const std::set<int32_t> first{ 0, 1 };
    /* Busy while the first reloads are in flight: nothing moves. */
    s.h.land_pct = 0;
    CHECK_EQ(s.pass(hot, 8), 0);
    CHECK_EQ(s.pass(hot, 8), 0);
    CHECK_EQ(s.declined(D::Busy), (uint64_t)2);
    CHECK_EQ(s.declined(D::Lent), (uint64_t)0);
    const uint64_t busy0 = s.declined(D::Busy);
    /* A draft bank's loan on the rank: declined, settled or not, whatever the heat says. */
    s.bank = true;
    s.h.land_pct = 100;
    for (int i = 0; i < 6; ++i) CHECK_EQ(s.pass(none, 8), 0);
    REQUIRE(s.l.settled());
    CHECK(s.heat.warm());
    CHECK_EQ(s.declined(D::Bank), (uint64_t)6);
    CHECK_EQ(s.pass(hot, 8), 0);
    CHECK_EQ(s.pass(hot, 8), 0);
    CHECK_EQ(s.declined(D::Bank), (uint64_t)8);
    CHECK(s.held() == first);
    s.bank = false;
    /* No event behind the pass before this one: declined, and asked again at the next pass without new counts. */
    s.fence_ready = false;
    CHECK_EQ(s.pass(none, 8), 0);
    CHECK_EQ(s.declined(D::Fence), (uint64_t)1);
    CHECK_EQ(s.pass(none, 8), 0);
    CHECK_EQ(s.declined(D::Fence), (uint64_t)2);
    CHECK(s.held() == first);
    CHECK_EQ(s.l.stats().retarget_withdrawn, (uint64_t)0);
    for (const auto& x : s.h.s) CHECK_EQ(x.rfences, 0);
    s.fence_ready = true;
    /* The hottest unit (6) can hold no slot: declined once, it leaves the candidates, and the next turn takes
     * the next hottest (5). */
    s.ineligible = { 6 };
    CHECK_EQ(s.pass(none, 8), 0);
    CHECK_EQ(s.declined(D::Candidate), (uint64_t)1);
    CHECK(std::find(s.cand.begin(), s.cand.end(), 6) == s.cand.end());
    CHECK(s.held() == first);
    CHECK_EQ(s.pass(none, 8), 1);
    CHECK(s.held() == (std::set<int32_t>{ 1, 5 }));   /* equal residents: the lower slot gave way */
    CHECK_EQ((int)s.taken[0], 0);
    CHECK_EQ((int)s.taken[5], 1);
    /* Busy while unit 5 loads; a pass that takes the slots back in the middle of it, and the grace after it. */
    s.h.land_pct = 0;
    CHECK_EQ(s.pass(hot, 8), 0);
    CHECK_EQ(s.declined(D::Busy), busy0 + 1);
    const uint64_t lent0 = s.declined(D::Lent);
    CHECK_EQ(s.pass(none, 0, true), 0);
    CHECK_EQ(s.declined(D::Lent), lent0 + 1);
    s.h.land_pct = 100;
    for (int i = 0; i < 8 && !s.l.settled(); ++i) (void)s.pass(none, 8);
    REQUIRE(s.l.settled());
    CHECK(s.held() == (std::set<int32_t>{ 1, 5 }));
    CHECK_EQ(s.h.s[0].pub_unit, 5);
    CHECK_EQ(s.l.stats().retargets, (uint64_t)1);
    CHECK_EQ(s.moved, 1);
    CHECK_EQ(s.h.violations, 0);
    if (s.h.violations) std::fprintf(stderr, "%s\n", s.h.first.c_str());
}

/* FileStager's side of the heat: which passes count (a trunk pass of decode rows only), and what a routed
 * layer adds when its counts are on the host: each file-tier unit's expert's rows, trunk layers only. */
TEST(p1_lend_heat_counts_pure_trunk_decode_passes_and_trunk_layers_only) {
    Plan plan;
    for (int e : { 5, 9, 40, 9 }) {
        MoveUnit mu;
        mu.expert = e;
        plan.units.push_back(mu);
    }
    StagedLayer trunk, head;   /* routed layer 0: model layer 3; routed layer 1: past the trunk's 43 layers */
    trunk.layer = 3;
    trunk.units = { 0, 1, 2 };
    head.layer = 43;
    head.units = { 3 };
    FileStager fs;
    CHECK(!EpCountsHostProbe::heat_on(fs));   /* without the switch: no heat, no counted pass */
    REQUIRE(EpCountsHostProbe::heat(fs, &plan, { trunk, head }, 43, 100));
    int32_t* const c = EpCountsHostProbe::counts(fs);
    std::fill(c, c + 96, 0);
    c[5] = 3;
    c[9] = 8;
    c[41] = 7;   /* an expert with no file-tier unit here */
    RadBatch decode{};
    decode.phase = RAD_PHASE_DECODE;
    decode.n_tok = decode.n_tok_decode = 8;
    decode.n_seq = decode.n_seq_decode = 8;
    REQUIRE(EpCountsHostProbe::heat_pass(fs, decode));
    EpCountsHostProbe::heat_layer(fs, 0);
    EpCountsHostProbe::heat_layer(fs, 1);
    CHECK_NEAR(EpCountsHostProbe::heat_of(fs, 0), 3.0, 1e-9);
    CHECK_NEAR(EpCountsHostProbe::heat_of(fs, 1), 8.0, 1e-9);
    CHECK_EQ(EpCountsHostProbe::heat_of(fs, 2), 0.0);   /* not routed to */
    CHECK_EQ(EpCountsHostProbe::heat_of(fs, 3), 0.0);   /* expert 9 of the draft head's layer: not the trunk's */
    /* Not counted: a draft pass, a step with prefill rows beside its decode rows, a prefill chunk, a decode
     * step some of whose rows are not decode rows, an empty one. Nothing ages either. */
    RadBatch draft = decode, mixed{}, chunk{}, part = decode, none{};
    draft.draft_pass = 2;
    mixed.phase = RAD_PHASE_MIXED;
    mixed.n_tok = 700;
    mixed.n_tok_decode = 4;
    chunk.phase = RAD_PHASE_PREFILL;
    chunk.n_tok = 1024;
    part.n_tok_decode = 4;
    none.phase = RAD_PHASE_DECODE;
    for (const RadBatch* b : { &draft, &mixed, &chunk, &part, &none }) {
        CHECK(!EpCountsHostProbe::heat_pass(fs, *b));
        EpCountsHostProbe::heat_layer(fs, 0);
    }
    CHECK_NEAR(EpCountsHostProbe::heat_of(fs, 0), 3.0, 1e-9);
    CHECK_NEAR(EpCountsHostProbe::heat_of(fs, 1), 8.0, 1e-9);
    /* The next decode pass ages what was counted by its own rows and adds its counts. */
    decode.n_tok = decode.n_tok_decode = 100;
    c[5] = 1;
    c[9] = 0;
    c[40] = 2;
    REQUIRE(EpCountsHostProbe::heat_pass(fs, decode));
    EpCountsHostProbe::heat_layer(fs, 0);
    CHECK_NEAR(EpCountsHostProbe::heat_of(fs, 0), 2.5, 1e-9);
    CHECK_NEAR(EpCountsHostProbe::heat_of(fs, 1), 4.0, 1e-9);
    CHECK_NEAR(EpCountsHostProbe::heat_of(fs, 2), 2.0, 1e-9);
    EpCountsHostProbe::heat_off(fs);
}

/* ---- the lent arena ----------------------------------------------------------------------------------------
 *
 * The slots carved from the arena's top (p1::arena_slots): whole slots from the top down, each on a 4 KiB
 * boundary, inside the region and apart; as many as fit up to the count. */
TEST(p1_lend_arena_slots_are_whole_aligned_and_inside_the_region_from_the_top) {
    const int64_t S = ds41p1::kSlotBytes;
    char* const base = (char*)(uintptr_t)0x7e0000000000ull;
    struct Case { int64_t lo_off, bytes; int max, want; } cases[] = {
        { 0, 10 * S, 64, 10 }, { 0, 10 * S, 4, 4 }, { 0, 10 * S - 1, 64, 9 }, { 100, 10 * S, 64, 9 },
        { 4096, 10 * S, 64, 10 }, { 0, S, 1, 1 }, { 0, S - 4096, 8, 0 }, { 0, 0, 8, 0 }, { 0, 5 * S, 0, 0 },
        { 0, 760ll << 20, 64, 42 }, { 0, (760ll << 20) + 5000, 64, 42 },
    };
    for (const Case& c : cases) {
        char* const lo = base + c.lo_off;
        const std::vector<char*> v = arena_slots(lo, c.bytes, c.max);
        CHECK_EQ((int)v.size(), c.want);
        for (size_t i = 0; i < v.size(); ++i) {
            CHECK_EQ(((uintptr_t)v[i] & 4095), (uintptr_t)0);
            CHECK(v[i] >= lo && v[i] + S <= lo + c.bytes);
            if (i) CHECK(v[i] + S <= v[i - 1]);   /* from the top down, apart */
        }
        if (!v.empty()) CHECK(v[0] + S + 4096 > lo + c.bytes);   /* the first at the top */
    }
    CHECK(arena_slots(nullptr, 10 * S, 4).empty());
}

/* The lent arena on a simulated card: RingLender with a host whose slots are arena memory, the compute stream
 * and the DMA lane as queues run in random order wherever their waits allow (events with record generations, as
 * HIP's). A step at the full plan writes its activations over the arena's top and reads them back later in the
 * stream; a step below it reads every published resident from its slot. FileStager's part is the driver's: a
 * full step takes the slots back (RingLender fences each copy still queued: the compute stream waits on it), and
 * at its end the DMA lane waits on an event behind it. Checks: no reload copy lands inside a full step's
 * activations (between its write and its read), and a resident's read always finds its unit. Without the fence,
 * or without the end's hand-over, the same runs find the race. */
namespace {
struct ArenaHost final : LendHost {
    enum T { Wait, Rec, CopyBeg, CopyEnd, FullWrite, FullRead, ResRead };
    struct Cmd { T t = Wait; int ev = -1; uint64_t gen = 0; int k = -1, unit = -1, step = -1; };
    std::deque<Cmd> q[2];          /* 0 the compute stream, 1 the DMA lane */
    std::vector<uint64_t> rec, done;
    std::vector<int> content, ev, src;   /* by slot: the unit its bytes are (-1 garbage), copy event, source */
    /* By slot: the unit it is for (the heat policy's retargets rename it), and the one its load read. */
    std::vector<int> cur, loaded;
    std::vector<uint8_t> reading;
    std::deque<std::pair<int, bool>> landings;
    std::mt19937 rng;
    int  full_owner = -1, bufs = 2, bufs_used = 0, lag = 4, end_ev = -1, fences = 0;
    /* The residents' last-reader event (FileStager's p1_lend_ev_: the driver records it on the compute stream
     * at the end of every pass that had residents), the retargets' fences and the next unit a retarget names. */
    int  last_ev = -1, rfences = 0, next_unit = 1000;
    bool do_fence = true, do_rfence = true;
    int  violations = 0;
    std::string first;

    ArenaHost(unsigned seed, int n) : rng(seed) {
        content.assign((size_t)n, -1);
        src.assign((size_t)n, 0);
        reading.assign((size_t)n, 0);
        loaded.assign((size_t)n, -1);
        for (int k = 0; k < n; ++k) {
            ev.push_back(event());
            cur.push_back(100 + k);
        }
        end_ev = event();
        last_ev = event();
    }
    int event() {
        rec.push_back(0);
        done.push_back(0);
        return (int)rec.size() - 1;
    }
    void bad(const std::string& w) {
        if (!violations++) first = w;
    }
    void push(int s, Cmd c) { q[s].push_back(c); }
    void wait(int s, int e) {
        Cmd c;
        c.t = Wait;
        c.ev = e;
        c.gen = rec[(size_t)e];
        push(s, c);
    }
    void record(int s, int e) {
        Cmd c;
        c.t = Rec;
        c.ev = e;
        c.gen = ++rec[(size_t)e];
        push(s, c);
    }
    bool runnable(int s) const {
        if (q[s].empty()) return false;
        const Cmd& c = q[s].front();
        return c.t != Wait || done[(size_t)c.ev] >= c.gen;
    }
    void exec(int s) {
        const Cmd c = q[s].front();
        q[s].pop_front();
        switch (c.t) {
        case Wait: break;
        case Rec: done[(size_t)c.ev] = std::max(done[(size_t)c.ev], c.gen); break;
        case CopyBeg:
        case CopyEnd:
            if (full_owner >= 0)
                bad("slot " + std::to_string(c.k) + "'s reload copy landed inside full step " + std::to_string(full_owner) +
                    "'s activations");
            content[(size_t)c.k] = c.t == CopyEnd ? c.unit : -2;
            break;
        case FullWrite:
            full_owner = c.step;
            for (int& x : content) x = -1;
            break;
        case FullRead:
            if (full_owner != c.step) bad("full step " + std::to_string(c.step) + "'s activations were overwritten");
            full_owner = -1;
            break;
        case ResRead:
            if (content[(size_t)c.k] != c.unit)
                bad("a resident read slot " + std::to_string(c.k) + " holding " + std::to_string(content[(size_t)c.k]));
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
        const int n = (int)(rng() % (unsigned)(lag + 1));
        for (int i = 0; i < n && step(); ++i) {}
    }
    void drain() {
        while (step()) {}
        if (!q[0].empty() || !q[1].empty()) bad("the streams deadlocked");
    }
    /* LendHost: a RAM-cache lease lands at once, a drive read after a few host calls, two buffers. */
    int load(int k) override {
        if (src[(size_t)k]) bad("a load of slot " + std::to_string(k) + " that holds a source");
        if (rng() % 3 == 0) {
            src[(size_t)k] = 2;
            loaded[(size_t)k] = cur[(size_t)k];
            landings.push_back({ k, true });
            return 1;
        }
        if (bufs_used >= bufs) return 0;
        ++bufs_used;
        src[(size_t)k] = 1;
        reading[(size_t)k] = 1;
        loaded[(size_t)k] = cur[(size_t)k];
        return 1;
    }
    int landed(bool* ok) override {
        progress();
        for (size_t k = 0; k < reading.size(); ++k)
            if (reading[k] && rng() % 2) {
                reading[k] = 0;
                landings.push_back({ (int)k, true });
            }
        if (landings.empty()) return -1;
        const auto d = landings.front();
        landings.pop_front();
        *ok = d.second;
        return d.first;
    }
    int copy(int k) override {
        Cmd c;
        c.t = CopyBeg;
        c.k = k;
        push(1, c);
        c.t = CopyEnd;
        c.unit = loaded[(size_t)k];
        push(1, c);
        record(1, ev[(size_t)k]);
        return 0;
    }
    int copied(int k) override {
        progress();
        return done[(size_t)ev[(size_t)k]] >= rec[(size_t)ev[(size_t)k]] ? 1 : 0;
    }
    void drop(int k) override {
        if (reading[(size_t)k]) bad("slot " + std::to_string(k) + "'s buffer let go while its read runs");
        if (src[(size_t)k] == 1) --bufs_used;
        src[(size_t)k] = 0;
    }
    void publish(int, bool) override {}
    void fence(int k) override {
        ++fences;
        if (do_fence) wait(0, ev[(size_t)k]);
    }
    /* CardExpertPath::retarget_fence: the DMA lane waits on the residents' last-reader event as recorded so
     * far, which the driver recorded at the end of the pass before this one. */
    int retarget_fence(int) override {
        ++rfences;
        if (do_rfence) wait(1, last_ev);
        return 0;
    }
    void retarget(int k) override { cur[(size_t)k] = next_unit++; }
};

/* `steps` passes, about one in `full_in` at the full plan; each step's host calls let the card run a little.
 * Then a long run of steps below it: every slot resident again. Returns the host's violations. */
int arena_run(unsigned seed, bool fence, bool hand, int steps, int full_in, std::string* why) {
    std::mt19937 rng(seed);
    const int n = 1 + (int)(rng() % 8);
    ArenaHost h(seed * 31u + 7u, n);
    h.do_fence = fence;
    h.bufs = 1 + (int)(rng() % 3);
    RingLender l;
    if (l.init(&h, n, (int64_t)(rng() % 3) * 20) < 0) return -1;
    int64_t t = 0;
    auto pass = [&](int s, bool full) {
        l.begin_pass(full, t);
        if (full) {
            ArenaHost::Cmd c;
            c.t = ArenaHost::FullWrite;
            c.step = s;
            h.push(0, c);
            h.progress();
            c.t = ArenaHost::FullRead;
            h.push(0, c);
        } else {
            for (int k = 0; k < n; ++k)
                if (l.resident(k)) {
                    ArenaHost::Cmd c;
                    c.t = ArenaHost::ResRead;
                    c.k = k;
                    c.unit = h.cur[(size_t)k];
                    h.push(0, c);
                }
        }
        h.progress();
        if (full) {
            l.end_p1(t);
            if (hand) {   /* FileStager::end_pass: the DMA lane behind the step's end */
                h.record(0, h.end_ev);
                h.wait(1, h.end_ev);
            }
        }
        t += 1 + (int64_t)(rng() % 30);
    };
    for (int s = 0; s < steps; ++s) pass(s, rng() % (unsigned)full_in == 0);
    for (int s = 0; s < 6 * n + 20; ++s) {
        pass(steps + s, false);
        h.drain();
        t += 100;
    }
    h.drain();
    if (l.residents() != n) h.bad("only " + std::to_string(l.residents()) + " of " + std::to_string(n) + " slots resident at the end");
    if (why && h.violations) *why = h.first;
    return h.violations;
}
}  // namespace

TEST(p1_lend_arena_full_steps_and_reloads_never_overlap_on_the_two_streams) {
    int runs = 0;
    for (unsigned seed = 1; seed <= 300; ++seed) {
        std::string why;
        const int v = arena_run(seed, true, true, 120, 3 + (int)(seed % 4), &why);
        if (v) {
            CHECK_EQ(v, 0);
            std::fprintf(stderr, "seed %u: %s\n", seed, why.c_str());
            return;
        }
        ++runs;
    }
    CHECK_EQ(runs, 300);
    /* The controls: without the fence, or without the hand-over at a full step's end, some run lands a reload copy
     * inside a full step's activations. */
    int no_fence = 0, no_hand = 0;
    for (unsigned seed = 1; seed <= 300; ++seed) {
        no_fence += arena_run(seed, false, true, 120, 3 + (int)(seed % 4), nullptr) > 0;
        no_hand += arena_run(seed, true, false, 120, 3 + (int)(seed % 4), nullptr) > 0;
    }
    CHECK(no_fence > 0);
    CHECK(no_hand > 0);
    std::fprintf(stderr, "  runs that race: %d of 300 without the fence, %d of 300 without the hand-over\n", no_fence, no_hand);
}

/* The heat policy's retargets on the same simulated card, decode going on: every pass below the full plan reads
 * each published resident from its slot on the compute stream, which may lag the host by many passes, and at
 * the end of every pass that had residents the driver records the residents' last-reader event there
 * (FileStager::end_pass's p1_lend_ev_). At a pass's begin, as FileStager::p1_heat_policy does it, a settled
 * lender whose event was recorded at the end of the pass right before retargets one or two slots:
 * RingLender::retarget has the DMA lane wait on that event (retarget_fence), then withdraws the resident and
 * renames the slot; the new unit's copy is queued passes later. Checks, when the commands run: a resident's
 * read always finds its unit (so no copy for a slot's next unit began under a reader of the unit before), no
 * copy lands inside a full step's activations, and at the end every slot is resident with the unit it was
 * last given. Without the retarget's fence the same runs find a copy under a lagging reader. */
namespace {
int heat_run(unsigned seed, bool rfence, int steps, int* moved, std::string* why) {
    std::mt19937 rng(seed);
    const int n = 1 + (int)(rng() % 8);
    ArenaHost h(seed * 131u + 5u, n);
    h.do_rfence = rfence;
    h.lag = 1 + (int)(rng() % 3);
    h.bufs = 1 + (int)(rng() % 3);
    const int full_in = 12 + (int)(rng() % 12);
    RingLender l;
    if (l.init(&h, n, (int64_t)(rng() % 3) * 20) < 0) return -1;
    int64_t t = 0;
    uint64_t pass_no = 0, rec_pass = 0;   /* FileStager's p1_lend_pass_ and p1_lend_rec_pass_ */
    bool rec = false;
    auto pass = [&](int s, bool full, bool move) {
        ++pass_no;
        l.begin_pass(full, t);
        if (move && !full && l.settled() && rec && rec_pass + 1 == pass_no) {
            const int ks[2] = { (int)(rng() % (unsigned)n), (int)(rng() % (unsigned)n) };
            const int done = l.retarget(ks, 1 + (int)(rng() % 2));
            if (moved) *moved += done;
        }
        if (full) {
            ArenaHost::Cmd c;
            c.t = ArenaHost::FullWrite;
            c.step = s;
            h.push(0, c);
            h.progress();
            c.t = ArenaHost::FullRead;
            h.push(0, c);
        } else {
            for (int k = 0; k < n; ++k)
                if (l.resident(k)) {
                    ArenaHost::Cmd c;
                    c.t = ArenaHost::ResRead;
                    c.k = k;
                    c.unit = h.cur[(size_t)k];
                    h.push(0, c);
                }
        }
        h.progress();
        if (l.residents() > 0) {   /* FileStager::end_pass: behind this pass's readers of the residents */
            h.record(0, h.last_ev);
            rec = true;
            rec_pass = pass_no;
        }
        if (full) {
            l.end_p1(t);
            h.record(0, h.end_ev);
            h.wait(1, h.end_ev);
        }
        t += 1 + (int64_t)(rng() % 30);
    };
    for (int s = 0; s < steps; ++s) {
        pass(s, rng() % (unsigned)full_in == 0, rng() % 3 != 0);
        if (rng() % 6 == 0) h.drain();   /* now and then the card catches up with the host */
    }
    for (int s = 0; s < 6 * n + 20; ++s) {
        pass(steps + s, false, false);
        h.drain();
        t += 100;
    }
    h.drain();
    if (l.residents() != n) h.bad("only " + std::to_string(l.residents()) + " of " + std::to_string(n) + " slots resident at the end");
    for (int k = 0; k < n; ++k)
        if (h.content[(size_t)k] != h.cur[(size_t)k])
            h.bad("slot " + std::to_string(k) + " holds unit " + std::to_string(h.content[(size_t)k]) + " at the end, not its unit " +
                  std::to_string(h.cur[(size_t)k]));
    if (why && h.violations) *why = h.first;
    return h.violations;
}
}  // namespace

TEST(p1_lend_heat_a_retargeted_slot_is_never_copied_under_a_reader_of_its_withdrawn_unit) {
    int runs = 0, moved = 0;
    for (unsigned seed = 1; seed <= 300; ++seed) {
        std::string why;
        const int v = heat_run(seed, true, 240, &moved, &why);
        if (v) {
            CHECK_EQ(v, 0);
            std::fprintf(stderr, "seed %u: %s\n", seed, why.c_str());
            return;
        }
        ++runs;
    }
    CHECK_EQ(runs, 300);
    CHECK(moved > 1000);   /* the runs do retarget, with decode going on */
    /* The control: with RingLender::retarget's order but no wait on the DMA lane, some run copies a slot's next
     * unit while a read of the unit before is still queued on the compute stream. */
    int no_fence = 0, moved_nf = 0;
    for (unsigned seed = 1; seed <= 300; ++seed) no_fence += heat_run(seed, false, 240, &moved_nf, nullptr) > 0;
    CHECK(no_fence > 0);
    std::fprintf(stderr, "  %d retargets in 300 runs behind the fence; runs that race without it: %d of 300\n", moved, no_fence);
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
    for (int64_t C : { 2048, 1024, 4096 })
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
