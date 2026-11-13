/* p1_hybrid_test.cpp -- hybrid card-path decode (RADIANCE_EXPERIMENTAL_DS41_P1_HYBRID, core/place/p1_hybrid.h), no
 * card (the host backend's fake devices):
 *  - the split (p1hyb::choose): only RAM-cache units with rows, fewest rows first (stable on ties), each taken
 *    only while (backlog + cost) / cpu rate <= share * (bytes left to the card) / link rate, the scan stopping at
 *    the first that does not fit; a larger share takes a superset; the pool's backlog (other ranks) counts;
 *    random layers hold the inequality and its maximality;
 *  - the share (p1hyb::Share): late layers shrink it to its floor, early ones grow it back to the switch's value,
 *    a hold changes nothing, ..._ADAPT=0 holds it; the tunables (..._P1_HYBRID 0-400, ..._ADAPT 0/1/unset);
 *  - the stager's hybrid layer through its test friend, with real packed experts and the process's CPU pool:
 *    p1_hyb_pick -> p1_hyb_borrow (in the layer's order, as p1_layer runs it) -> p1_hyb_submit -> hx_finish on a
 *    down-projection buffer whose card units' slots hold the card's rows (the CPU path's bits, as the P1 twin's
 *    are by the device check) and whose CPU units' slots no card kernel writes (skip-marked): at every share,
 *    with drive units that never go to the CPU and a lease the cache refuses at borrow time (the unit goes to the
 *    ring), every live sorted slot ends byte-identical to the all-CPU reference, the CPU and card sets partition
 *    the selected units, the borrows come in the layer's order with the admission's touch, the pool's
 *    reservation is given back in full, and a fixed-order weighted combine over each token's top-k rows (the
 *    gather's shape) is bit-identical to the base's whatever the split;
 *  - the H31 failure (window ab-20261113T072152Z, N128 and BIRD N256 arms with Qy60): the first armed decode pass
 *    after prefill chunks at the activation arena's full plan read the rows and the sorted order from the full
 *    plan's addresses while it ran at the lent arena's smaller level ("expert 92 has 0 rows in the sorted order
 *    and 4 in its count", then "gate_up reads rows other than the ones the pool was given"). A layer now splits
 *    only when its gate_up operands were last seen at this pass's level (p1_hyb_layer_ok): the pass sequence
 *    full-plan prefill -> 256-row-level decode declines (counted), its gate_up records that level's addresses,
 *    the next decode pass splits; at the 1024-row level and back at the full plan it declines again; and the live check that failed on the
 *    card refuses the moved operands as it did there. */
#include "rad_test.h"

#include "rad_internal.h"
#include "device/device.h"
#include "place/host_expert_pool.h"
#include "place/p1_hybrid.h"
#include "place/stager.h"
#include "../tools/ds41-host-experts/ds41_host_experts.h"
#include "../tools/ds41-host-experts/ds41_p1.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

#define REQUIRE_OK(expr) REQUIRE((expr) >= 0)

namespace {

const int g_devices = [] {
    ::setenv("RADIANCE_HOST_DEVICES", "4", 1);
    return 4;
}();

rad::HostExpertPool* pool() {
    static rad::HostExpertPool* p = []() -> rad::HostExpertPool* {
        if (!__builtin_cpu_supports("avx2")) return nullptr;
        const std::string path = radtest::tmpdir() + "/silu_hyb.bin";
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
constexpr int kPacked = 6;   /* distinct experts' bytes; unit u is lent experts[u % kPacked] */
std::vector<Packed>& experts() {
    static std::vector<Packed> v = [] {
        std::vector<Packed> e(kPacked);
        for (int k = 0; k < kPacked; ++k)
            if (!e[(size_t)k].make(7000 + (uint32_t)k)) return std::vector<Packed>{};
        return e;
    }();
    return v;
}

using rad::p1hyb::Cand;
using rad::p1hyb::Split;
using rad::HostExpertPool;

constexpr int64_t kB = HostExpertPool::kExpertBytes;

}  // namespace

/* ---- the split -------------------------------------------------------------------------------------------- */

TEST(choose_takes_cached_units_fewest_rows_first_while_they_fit) {
    std::vector<int> order;
    /* rows: 5, 1, 3, 1, 2, 9 (index 3 not cached, index 5 drive too). */
    std::vector<Cand> c = { { 5, kB, true }, { 1, kB, true }, { 3, kB, true }, { 1, kB, false }, { 2, kB, true }, { 9, kB, false } };
    std::vector<uint8_t> to(c.size());
    const double cpu = 8.0, link = 20.0;   /* bytes a ns */
    /* share 0: nothing, the card keeps every byte. */
    Split s = rad::p1hyb::choose(c.data(), (int)c.size(), 0, cpu, link, 0.0, to.data(), &order);
    CHECK_EQ(s.n_cpu, 0);
    CHECK_EQ(s.card_bytes, 6 * kB);
    for (uint8_t t : to) CHECK_EQ((int)t, 0);
    /* A huge share: every cached unit with rows, never the uncached ones. */
    s = rad::p1hyb::choose(c.data(), (int)c.size(), 0, cpu, link, 1e9, to.data(), &order);
    CHECK_EQ(s.n_cpu, 4);
    CHECK(to[0] && to[1] && to[2] && to[4] && !to[3] && !to[5]);
    CHECK_EQ(s.card_bytes, 2 * kB);
    /* The order: index 1 (1 row), then 4 (2), then 2 (3), then 0 (5). Find the share that takes exactly k. */
    const int want[4] = { 1, 4, 2, 0 };
    int64_t cost = 0;
    for (int k = 0; k < 4; ++k) {
        cost += HostExpertPool::cost_rows(c[(size_t)want[k]].rows);
        const int64_t left = (6 - (k + 1)) * kB;
        const double need = ((double)cost / cpu) / ((double)left / link);
        s = rad::p1hyb::choose(c.data(), (int)c.size(), 0, cpu, link, need * 1.000001, to.data(), &order);
        CHECK_EQ(s.n_cpu, k + 1);
        for (int j = 0; j <= k; ++j) CHECK_EQ((int)to[(size_t)want[j]], 1);
        CHECK_EQ(s.cpu_cost, cost);
        CHECK_EQ(s.card_bytes, left);
        s = rad::p1hyb::choose(c.data(), (int)c.size(), 0, cpu, link, need * 0.999999, to.data(), &order);
        CHECK_EQ(s.n_cpu, k);
    }
    /* Other ranks' backlog counts: one that fills the pool for longer than the card's whole layer takes nothing. */
    s = rad::p1hyb::choose(c.data(), (int)c.size(), (int64_t)(6 * kB / link * cpu * 2), cpu, link, 1.0, to.data(), &order);
    CHECK_EQ(s.n_cpu, 0);
    /* Units with no rows or no bytes are never the CPU's. */
    std::vector<Cand> z = { { 0, kB, true }, { 2, 0, true } };
    std::vector<uint8_t> tz(2);
    s = rad::p1hyb::choose(z.data(), 2, 0, cpu, link, 1e9, tz.data(), &order);
    CHECK_EQ(s.n_cpu, 0);
    /* Ties keep the layer's order (stable). */
    std::vector<Cand> t = { { 2, kB, true }, { 2, kB, true }, { 2, kB, true }, { 2, kB, true } };
    std::vector<uint8_t> tt(4);
    const double one = ((double)HostExpertPool::cost_rows(2) / cpu) / ((double)(3 * kB) / link);
    s = rad::p1hyb::choose(t.data(), 4, 0, cpu, link, one * 1.000001, tt.data(), &order);
    CHECK_EQ(s.n_cpu, 1);
    CHECK_EQ((int)tt[0], 1);
}

TEST(choose_random_layers_hold_the_bound_and_its_maximality) {
    std::mt19937 rng(12345);
    std::vector<int> order;
    for (int run = 0; run < 2000; ++run) {
        const int n = 1 + (int)(rng() % 60);
        std::vector<Cand> c((size_t)n);
        for (auto& x : c) {
            x.rows = (int)(rng() % 12);
            x.bytes = kB;
            x.cached = rng() % 4 != 0;
        }
        const double cpu = 2.0 + (rng() % 80), link = 10.0 + (rng() % 20);
        const double share = (rng() % 400) / 100.0;
        const int64_t backlog = (int64_t)(rng() % 4) * kB;
        std::vector<uint8_t> to((size_t)n), to2((size_t)n);
        const Split s = rad::p1hyb::choose(c.data(), n, backlog, cpu, link, share, to.data(), &order);
        int64_t cost = 0, left = 0;
        int taken = 0, max_rows_taken = -1, min_rows_left = 1 << 30;
        for (int i = 0; i < n; ++i) {
            if (to[(size_t)i]) {
                CHECK(c[(size_t)i].cached && c[(size_t)i].rows > 0);
                cost += HostExpertPool::cost_rows(c[(size_t)i].rows);
                ++taken;
                max_rows_taken = std::max(max_rows_taken, c[(size_t)i].rows);
            } else {
                left += c[(size_t)i].bytes;
                if (c[(size_t)i].cached && c[(size_t)i].rows > 0) min_rows_left = std::min(min_rows_left, c[(size_t)i].rows);
            }
        }
        CHECK_EQ(taken, s.n_cpu);
        CHECK_EQ(cost, s.cpu_cost);
        CHECK_EQ(left, s.card_bytes);
        if (taken) CHECK((backlog + (double)cost) / cpu <= share * (double)left / link * (1 + 1e-12));
        /* Fewest rows first: nothing left that is cheaper than a unit taken. */
        if (taken) CHECK(min_rows_left >= max_rows_taken);
        /* A larger share never takes fewer: a superset. */
        rad::p1hyb::choose(c.data(), n, backlog, cpu, link, share * 1.5 + 0.01, to2.data(), &order);
        for (int i = 0; i < n; ++i)
            if (to[(size_t)i]) CHECK_EQ((int)to2[(size_t)i], 1);
    }
}

TEST(share_shrinks_when_late_grows_when_early_and_holds_when_fixed) {
    rad::p1hyb::Share s;
    s.init(50, true);
    CHECK_NEAR(s.share(), 0.5, 1e-12);
    s.layer(false, false);   /* hold */
    CHECK_NEAR(s.share(), 0.5, 1e-12);
    s.layer(true, false);    /* early at the top: stays at the switch's value */
    CHECK_NEAR(s.share(), 0.5, 1e-12);
    s.layer(false, true);
    CHECK_NEAR(s.share(), 0.4, 1e-12);
    for (int i = 0; i < 200; ++i) s.layer(false, true);
    CHECK_NEAR(s.share(), rad::p1hyb::Share::kMin, 1e-12);
    CHECK_NEAR(s.min_seen(), rad::p1hyb::Share::kMin, 1e-12);
    for (int i = 0; i < 500; ++i) s.layer(true, false);
    CHECK_NEAR(s.share(), 0.5, 1e-12);
    rad::p1hyb::Share f;
    f.init(30, false);
    for (int i = 0; i < 10; ++i) f.layer(false, true);
    CHECK_NEAR(f.share(), 0.3, 1e-12);
}

TEST(tunables_take_the_hybrid_switches) {
    auto parse = [](const char* hy, const char* ad, rad::p1::Tunables* t, std::string* why) {
        return rad::p1::parse_tunables([&](const char* k) -> const char* {
            if (!std::strcmp(k, ds41p1::kHybridEnv)) return hy;
            if (!std::strcmp(k, ds41p1::kHybridAdaptEnv)) return ad;
            return nullptr;
        }, t, why);
    };
    rad::p1::Tunables t;
    std::string why;
    REQUIRE(parse(nullptr, nullptr, &t, &why));
    CHECK_EQ(t.hybrid, 0);
    CHECK(t.hybrid_adapt);
    REQUIRE(parse("25", "", &t, &why));
    CHECK_EQ(t.hybrid, 25);
    CHECK(t.hybrid_adapt);
    REQUIRE(parse("400", "0", &t, &why));
    CHECK_EQ(t.hybrid, 400);
    CHECK(!t.hybrid_adapt);
    REQUIRE(parse("1", "1", &t, &why));
    CHECK(t.hybrid_adapt);
    CHECK(!parse("401", nullptr, &t, &why));
    CHECK(why.find(ds41p1::kHybridEnv) != std::string::npos);
    CHECK(!parse("-1", nullptr, &t, &why));
    CHECK(!parse("x", nullptr, &t, &why));
    CHECK(!parse("10", "2", &t, &why));
    CHECK(why.find(ds41p1::kHybridAdaptEnv) != std::string::npos);
}

/* ---- the stager's hybrid layer ---------------------------------------------------------------------------- */

namespace rad {

struct P1HybridProbe {
    Plan plan;
    Mover mover;
    FileStager fs;
    std::vector<int32_t> counts = std::vector<int32_t>(96, 0);
    static constexpr int R = 16;   /* the pass's rows at most */
    int AR = 0;
    std::vector<float> w;          /* the gather's weights, [AR][6] */
    std::vector<int32_t> slot_of;  /* [AR][6]: the sorted slot of token t's pick k, -1 another rank's */
    std::vector<uint16_t> ref;     /* the all-CPU rows, by sorted slot */

    /* The cache: which units it holds, which it refuses at borrow time, and the borrows it saw. */
    std::vector<uint8_t> cached = std::vector<uint8_t>(96, 0), refuse = std::vector<uint8_t>(96, 0);
    std::vector<int32_t> borrowed;
    int touched = 0;
    static bool borrow(void* ctx, int32_t u, Mover::HostUnit* hu, bool touch) {
        P1HybridProbe* p = (P1HybridProbe*)ctx;
        if (!hu || hu->n || u < 0 || u >= 96 || !p->cached[(size_t)u] || p->refuse[(size_t)u]) return false;
        p->borrowed.push_back(u);
        p->touched += touch;
        const Packed& e = experts()[(size_t)(u % kPacked)];
        hu->unit = u;
        hu->n = 2;
        hu->packed = true;
        hu->host[0] = e.gu.data();
        hu->host[1] = e.dn.data();
        hu->bytes[0] = HostExpertPool::kGuBytes;
        hu->bytes[1] = HostExpertPool::kDnBytes;
        hu->slot[0] = (int64_t)e.gu.size();
        hu->slot[1] = (int64_t)e.dn.size();
        return true;
    }
    static bool is_cached(void* ctx, int32_t u) {
        P1HybridProbe* p = (P1HybridProbe*)ctx;
        return u >= 0 && u < 96 && p->cached[(size_t)u];
    }

    P1HybridProbe() {
        plan.rank = 0;
        plan.world_size = 4;
        plan.units.resize(96);
        for (int e = 0; e < 96; ++e) {
            plan.units[(size_t)e].expert = e;
            plan.units[(size_t)e].layer = 3;
            plan.units[(size_t)e].bytes = HostExpertPool::kExpertBytes;
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
        mover.borrow_hook_.context = this;
        mover.borrow_hook_.borrow = &borrow;
        mover.borrow_hook_.cached = &is_cached;
    }
    ~P1HybridProbe() {
        fs.hx_drain();
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

    /* A decode pass of `rows` tokens: each picks five distinct local experts from a skewed law over 0..23 (so a
     * few experts get many rows and most one or two) and one on another rank, written as moe_scatter writes the
     * sorted order and the counts; the token rows; the gather's weights; then the all-CPU reference rows. */
    bool pass(int rows, uint32_t seed) {
        AR = rows;
        fs.pass_tok_ = rows;
        std::mt19937 rng(seed);
        std::vector<std::vector<int32_t>> by_e(96);
        slot_of.assign((size_t)rows * 6, -1);
        for (int t = 0; t < rows; ++t) {
            std::vector<int> picked;
            while ((int)picked.size() < 5) {
                const double r = std::uniform_real_distribution<double>(0, 1)(rng);
                const int e = std::min(23, (int)(24 * r * r * r));
                if (std::find(picked.begin(), picked.end(), e) == picked.end()) picked.push_back(e);
            }
            for (int k = 0; k < 5; ++k) by_e[(size_t)picked[(size_t)k]].push_back(t * 6 + k);
        }
        int32_t* const sorted = fs.hx_idx_;
        int32_t* const off = fs.hx_idx_ + fs.hx_sorted_max();
        int at = 0;
        std::fill(counts.begin(), counts.end(), 0);
        for (int e = 0; e < 96; ++e) {
            off[e] = at;
            for (int32_t s : by_e[(size_t)e]) {
                slot_of[(size_t)s] = at;
                sorted[at++] = s;
            }
            counts[(size_t)e] = (int32_t)by_e[(size_t)e].size();
        }
        off[96] = at;
        while (at < rows * 6) sorted[at++] = -1;
        for (int i = 0; i < rows * 5120; ++i)
            fs.hx_x_[i] = ds41hx::f32_to_bf16((float)((int)(rng() % 2001) - 1000) / 400.0f);
        w.assign((size_t)rows * 6, 0.f);
        for (auto& x : w) x = (float)(rng() % 1000) / 997.0f;
        /* The reference: every live slot's row from its expert's bytes, a job a row. */
        const int T = off[96];
        ref.assign((size_t)T * 5120, 0);
        std::vector<HostExpertPool::Job> jobs;
        for (int e = 0; e < 96; ++e)
            for (int i = off[e]; i < off[e + 1]; ++i) {
                const Packed& p = experts()[(size_t)(e % kPacked)];
                HostExpertPool::Job jb;
                jb.gu = p.gu.data();
                jb.dn = p.dn.data();
                jb.packed = true;
                jb.gu_bytes = (int64_t)p.gu.size();
                jb.dn_bytes = (int64_t)p.dn.size();
                jb.x = fs.hx_x_ + (int64_t)(sorted[i] / 6) * 5120;
                jb.y = ref.data() + (int64_t)i * 5120;
                jb.cost = HostExpertPool::job_cost(1);
                jobs.push_back(jb);
            }
        int64_t cost = 0;
        for (auto& jb : jobs) cost += jb.cost;
        HostExpertPool::Ticket tk;
        fs.hx_pool_->reserve(cost);
        if (fs.hx_pool_->submit(jobs.data(), (int)jobs.size(), &tk) < 0) return false;
        fs.hx_pool_->wait(&tk);
        return true;
    }

    /* The gather's shape: each token's picks weighed in k order, float. */
    std::vector<float> combine(const uint16_t* y) const {
        std::vector<float> out((size_t)AR * 5120, 0.f);
        for (int t = 0; t < AR; ++t)
            for (int k = 0; k < 6; ++k) {
                const int32_t s = slot_of[(size_t)t * 6 + k];
                if (s < 0) continue;
                for (int c = 0; c < 5120; ++c)
                    out[(size_t)t * 5120 + c] += w[(size_t)t * 6 + k] * ds41hx::bf16_to_f32(y[(size_t)s * 5120 + c]);
            }
        return out;
    }

    struct Layer {
        std::vector<int32_t> cpu, card;
        std::vector<uint16_t> y;
        int64_t backlog_after = 0;
    };
    /* One hybrid layer as p1_layer runs it, at share `pct`: the pick, the borrows in want_'s order, the submit;
     * the card's waves write their units' slots (the card's rows: the reference's bytes) and nothing else; then
     * hx_finish at the gather's operands. */
    void layer(int pct, RadStream s, uint16_t* y, Layer* out) {
        Layer& L = *out;
        L = Layer{};
        borrowed.clear();
        touched = 0;
        fs.hx_li_ = -1;
        fs.hx_stage_ = 0;
        fs.want_.clear();
        for (int e = 0; e < 96; ++e)
            if (counts[(size_t)e] > 0) fs.want_.push_back(e);
        fs.p1_hyb_share_.init(pct, false);
        fs.p1_hyb_cpu_.clear();
        const int64_t backlog0 = fs.hx_pool_->backlog();
        const bool split = fs.p1_hyb_pick(0) > 0;
        for (size_t i = 0; i < fs.want_.size(); ++i) {
            const int32_t u = fs.want_[i];
            if (split && fs.p1_hyb_to_[i] && fs.p1_hyb_borrow(u, counts[(size_t)u], true)) continue;
            L.card.push_back(u);
        }
        L.cpu = fs.p1_hyb_cpu_;
        if (fs.hx_n_ > 0) REQUIRE_OK(fs.p1_hyb_submit(0));
        const int32_t* const off = fs.hx_idx_ + fs.hx_sorted_max();
        const int T = AR * 6;
        for (int i = 0; i < T * 5120; ++i) y[i] = 0x5555;   /* skipped slots: whatever was there */
        for (int32_t u : L.card)
            std::memcpy(y + (size_t)off[u] * 5120, ref.data() + (size_t)off[u] * 5120,
                        (size_t)(off[u + 1] - off[u]) * 5120 * 2);
        if (fs.hx_li_ == 0) {
            CHECK_EQ(fs.hx_stage_, 2);
            CHECK_EQ(fs.p1_hyb_li_, 0);
            fs.hx_confirmed_ = true;   /* hx_capture saw the gate_up read the rows */
            RadTensor Y{};
            Y.data = y;
            Y.dtype = RAD_BF16;
            Y.rank = 2;
            Y.shape[0] = T;
            Y.shape[1] = 5120;
            Y.stride[0] = 5120;
            Y.stride[1] = 1;
            REQUIRE_OK(fs.hx_finish(s, &Y, 1));
            REQUIRE_OK(rad_stream_sync(s));
            fs.hx_out_rec_ = false;
            CHECK_EQ(fs.p1_hyb_li_, -1);
        }
        L.y.assign(y, y + (size_t)T * 5120);
        L.backlog_after = fs.hx_pool_->backlog() - backlog0;
    }

    void splits_are_exact() {
        REQUIRE(experts().size() == (size_t)kPacked);
        RadStream s = nullptr;
        RadEvent ev = nullptr;
        REQUIRE_OK(rad_stream_create(&s, 0));
        REQUIRE_OK(rad_event_create(&ev));
        fs.hx_out_ev_ = ev;
        uint16_t* y = (uint16_t*)rad_dev_alloc((int64_t)R * 6 * 5120 * 2, RAD_MEM_DEVICE);
        REQUIRE(y);
        int splits_seen = 0;
        for (uint32_t seed : { 11u, 12u }) {
            REQUIRE(pass(seed == 11u ? 16 : 9, seed));
            const int32_t* const off = fs.hx_idx_ + fs.hx_sorted_max();
            const int T = off[96];
            /* Most units cached; a few drive units (not cached), and one cached unit refused at borrow time. */
            std::fill(cached.begin(), cached.end(), 1);
            std::fill(refuse.begin(), refuse.end(), 0);
            cached[2] = cached[17] = 0;
            refuse[5] = 1;
            const std::vector<float> base = combine(ref.data());
            int last_cpu = -1;
            for (int pct : { 0, 2, 10, 40, 100, 400 }) {
                Layer L;
                layer(pct, s, y, &L);
                /* A partition of the selected units; drive units on the card; the refused unit there too. */
                std::vector<int32_t> all = L.cpu;
                all.insert(all.end(), L.card.begin(), L.card.end());
                std::sort(all.begin(), all.end());
                CHECK(all == fs.want_);
                for (int32_t u : L.cpu) {
                    CHECK(cached[(size_t)u] && !refuse[(size_t)u]);
                }
                /* The borrows came in want_'s order, each with the admission's touch. */
                CHECK(std::is_sorted(borrowed.begin(), borrowed.end()));
                CHECK_EQ(touched, (int)borrowed.size());
                /* Every reservation went back with the jobs. */
                CHECK_EQ(L.backlog_after, (int64_t)0);
                CHECK_EQ(fs.hx_reserved_, (int64_t)0);
                CHECK_EQ(fs.hx_n_, 0);
                /* Every live slot byte-identical to the all-CPU rows, whatever the split. */
                CHECK(std::memcmp(L.y.data(), ref.data(), (size_t)T * 5120 * 2) == 0);
                const std::vector<float> got = combine(L.y.data());
                CHECK(std::memcmp(got.data(), base.data(), got.size() * sizeof(float)) == 0);
                CHECK((int)L.cpu.size() >= last_cpu);   /* a larger share takes no fewer */
                last_cpu = (int)L.cpu.size();
                if (!L.cpu.empty() && !L.card.empty()) ++splits_seen;
            }
            CHECK(last_cpu > 0);
        }
        CHECK(splits_seen > 0);
        CHECK(fs.p1_hyb_st_.fell_back > 0);   /* unit 5, refused at its borrow */
        CHECK(fs.p1_hyb_st_.layers > 0);
        fs.hx_out_ev_ = nullptr;
        rad_event_destroy(ev);
        rad_stream_destroy(s);
        rad_dev_free(y, RAD_MEM_DEVICE);
    }
};

}  // namespace rad

namespace rad {
/* The arena-level gate on hx_capture's records, as start_ep_routed asks it. */
struct P1HybridLevelProbe {
    FileStager fs;
    int32_t sorted_buf[2][16 * 6] = {}, off_buf[2][128] = {};
    uint16_t x_buf[2][16] = {};   /* addresses only: nothing reads them */
    void setup() {
        StagedLayer L;
        L.layer = 0;
        L.first_op = 1;
        L.last_op = 3;
        fs.layers_ = { L };
        FileStager::HxOps o;
        o.gu = 1;
        o.n_expert = 96;
        o.top_k = 6;
        fs.hx_ops_ = { o };
        fs.ep_rows_ = 2048;
        fs.routed_ = true;
        fs.hx_pool_ = pool();
        fs.p1_hyb_pass_ = true;
    }
    ~P1HybridLevelProbe() { fs.hx_pool_ = nullptr; }
    /* One pass's gate_up operands (`which` buffer set: 0 the full plan's, 1 the small level's), as hx_capture sees
     * them; `live` makes it the check on a layer whose CPU rows the pool has. */
    int capture(int which, int rows) {
        fs.pass_tok_ = rows;
        RadTensor t[5] = {};
        t[0].data = x_buf[which];
        t[0].dtype = RAD_BF16;
        t[0].rank = 2;
        t[0].shape[0] = rows;
        t[0].shape[1] = 5120;
        t[0].stride[0] = 5120;
        t[0].stride[1] = 1;
        t[2].data = sorted_buf[which];
        t[2].dtype = RAD_I32;
        t[2].rank = 1;
        t[2].shape[0] = rows * 6;
        t[3].data = off_buf[which];
        t[3].dtype = RAD_I32;
        t[3].rank = 1;
        t[3].shape[0] = 97;
        t[4].rank = 2;
        t[4].shape[0] = rows * 6;
        t[4].shape[1] = 5120;
        return fs.hx_capture(0, nullptr, t, 5);
    }
    void run() {
        setup();
        /* A prefill chunk at the full plan records the full plan's addresses. */
        fs.pass_level_ = 0;
        REQUIRE_OK(capture(0, 16));
        CHECK_EQ(fs.hx_ops_[0].x_level, 0);
        CHECK(fs.p1_hyb_layer_ok(0));            /* another full-level pass could split */
        /* The first armed decode pass runs at the small level: it must not split (its rows are elsewhere). */
        fs.pass_level_ = 2;
        CHECK(!fs.p1_hyb_layer_ok(0));
        /* What happened on the card without the gate: the pool given the full plan's rows, the gate_up reading the
         * small level's: the live check refuses it. */
        fs.hx_li_ = 0;
        fs.hx_stage_ = 2;
        CHECK(capture(1, 8) < 0);
        fs.hx_li_ = -1;
        fs.hx_stage_ = 0;
        /* With the gate: the declining pass's gate_up records the small level's addresses; the next decode pass
         * at that level splits, and the live check passes on them. */
        REQUIRE_OK(capture(1, 8));
        CHECK_EQ(fs.hx_ops_[0].x_level, 2);
        CHECK(fs.hx_ops_[0].x == (const void*)x_buf[1]);
        CHECK(fs.p1_hyb_layer_ok(0));
        fs.hx_li_ = 0;
        fs.hx_stage_ = 2;
        REQUIRE_OK(capture(1, 8));
        CHECK(fs.hx_confirmed_);
        fs.hx_li_ = -1;
        fs.hx_stage_ = 0;
        fs.hx_confirmed_ = false;
        /* Another smaller level (the 1024-row one, a short chunk): not the 256-row level's addresses either. */
        fs.pass_level_ = 1;
        CHECK(!fs.p1_hyb_layer_ok(0));
        /* Back at the full plan (a later prompt's chunk): declined until a full-level pass records its own. */
        fs.pass_level_ = 0;
        CHECK(!fs.p1_hyb_layer_ok(0));
        REQUIRE_OK(capture(0, 16));
        CHECK(fs.p1_hyb_layer_ok(0));
        /* No pass to split, no pool, or a layer never seen: never. */
        fs.p1_hyb_pass_ = false;
        CHECK(!fs.p1_hyb_layer_ok(0));
        fs.p1_hyb_pass_ = true;
        fs.hx_ops_[0].x = nullptr;
        CHECK(!fs.p1_hyb_layer_ok(0));
    }
};
}  // namespace rad

TEST(a_layer_splits_only_on_operands_seen_at_its_passes_arena_level) {
    if (!pool()) SKIP("no AVX2 or no CPU pool");
    rad::P1HybridLevelProbe p;
    p.run();
}

TEST(a_hybrid_layer_leaves_every_row_and_the_combine_as_the_base_at_every_share) {
    if (!pool()) SKIP("no AVX2 or no CPU pool");
    rad::P1HybridProbe p;
    p.splits_are_exact();
}

RAD_TEST_MAIN()
