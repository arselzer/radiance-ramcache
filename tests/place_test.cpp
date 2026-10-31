/* place_test.cpp -- the placement planner, the mover's slot bookkeeping, and the heat engine.
 *
 * The key test is `heat_does_not_thrash`. Dynamic placement pays off only if it declines to move
 * when moving would not help, since a swap costs two shards over the link where a miss costs one.
 * The test runs the same alternating access pattern with the guards at their shipped values and
 * with them removed, and compares the two.
 */
#include "rad_test.h"

#include "device/device.h"
#include "device/directio.h"
#include "mem/pools.h"
#include "mem/vram_budget.h"
#include "place/heat.h"
#include "place/mover.h"
#include "place/stager.h"
#include "place/planner.h"
#include "place/vram_packed.h"
#include "../tools/ds41-host-experts/ds41_host_experts.h"
#include "../tools/ds41-host-experts/ds41_p1.h"
#include "place/expert_profile_map.h"
#include "place/native_decode_profile.h"
#include "format/radfile.h"
#include "dspark_bank_plan_fixture.h"
#include "format/dspark_rad_banks.h"
#include "place/draft_vram.h"
#include "place/draft_vram_load.h"
#include "place/dspark_draft_control_scope.h"
#include <filesystem>
#include <functional>
#include <stdexcept>
#include <fstream>
#include <fcntl.h>
#include <sys/stat.h>

#include <dirent.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <atomic>
#include <vector>

using namespace rad;

/* ================================================================== a synthetic Program */
namespace {

constexpr int64_t MiB = 1024 * 1024;
constexpr int64_t KiB = 1024;

/* Two kernel rows, one per domain. The planner reads Band::dom[] to learn whether a host site
 * exists, so fixtures resolve real rows. */
const RadKernelInfo g_dev_info = { "fake_dev", "gemm", "gemm", "", "", "",
                                   RAD_DOMAIN_DEVICE, 0, nullptr, 0, nullptr, 0,
                                   nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
const RadKernelInfo g_host_info = { "fake_host", "gemm", "gemm", "", "", "",
                                    RAD_DOMAIN_HOST, 0, nullptr, 0, nullptr, 0,
                                    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
KernelRow g_dev_row;
KernelRow g_host_row;

struct Fixture {
    Program prog;

    Fixture() {
        g_dev_row.info = &g_dev_info;
        g_host_row.info = &g_host_info;
        prog.weights.resize(1);        /* index 0 is the sentinel */
        prog.ctx.rank = 0;
        prog.ctx.world_size = 1;
    }

    rad_weight weight(const std::string& name, int64_t bytes, int access, int32_t layer,
                      int32_t expert, int shard = RAD_SHARD_NONE) {
        WeightInfo w;
        w.name = name;
        w.decl.dtype = RAD_U8;         /* one byte per element, so bytes are exact */
        w.decl.rank = 1;
        w.decl.shape[0] = bytes;
        w.decl.access = access;
        w.decl.shard = shard;
        w.decl.group.layer = layer;
        w.decl.group.expert = expert;
        w.logical_bytes = bytes;
        w.align = RAD_ALIGN_UNIT;
        prog.weights.push_back(w);
        return (rad_weight)(prog.weights.size() - 1);
    }

    void op(const std::string& name, const std::vector<rad_weight>& ws, bool host_kernel) {
        OpInfo o;
        o.op = name;
        o.index = (int32_t)prog.ops.size();
        o.weights = ws;
        Band b;
        b.hi = 1 << 20;
        b.dom[RAD_DOMAIN_DEVICE].row = &g_dev_row;
        if (host_kernel) b.dom[RAD_DOMAIN_HOST].row = &g_host_row;
        else             b.miss[RAD_DOMAIN_HOST] = "no host kernel for this geometry";
        o.bands.push_back(b);
        for (rad_weight w : ws) {
            WeightInfo& wi = prog.weights[w];
            if (wi.first_use_op < 0) wi.first_use_op = o.index;
            wi.last_use_op = o.index;
        }
        prog.ops.push_back(o);
    }

    /* An op only a host kernel serves, its device band a miss. Unlike op(), it leaves the
     * weights' first and last use as they were. */
    void host_op(const std::string& name, const std::vector<rad_weight>& ws) {
        OpInfo o;
        o.op = name;
        o.index = (int32_t)prog.ops.size();
        o.weights = ws;
        Band b;
        b.hi = 1 << 20;
        b.dom[RAD_DOMAIN_HOST].row = &g_host_row;
        b.miss[RAD_DOMAIN_DEVICE] = "no device kernel";
        o.bands.push_back(b);
        prog.ops.push_back(o);
    }
};

Config base_config() {
    Config c;
    c.placement = "auto";
    c.vram_weights_mib = 0;
    c.host_pool_mib = 0;
    return c;
}

/* A dense model: `layers` layers of one weight each, plus one model-level weight. */
void build_dense(Fixture& f, int layers, int64_t per_layer, bool host_kernel) {
    const rad_weight m = f.weight("token_embd.weight", 1 * MiB, RAD_ACCESS_VOCAB, -1, -1);
    f.op("embed", {m}, host_kernel);
    for (int l = 0; l < layers; ++l) {
        const rad_weight w = f.weight("blk." + std::to_string(l) + ".ffn_down.weight", per_layer,
                                      RAD_ACCESS_PER_TOKEN, l, -1);
        f.op("gemm", {w}, host_kernel);
    }
}

/* One MoE layer: `n` experts of `each` bytes, plus one dense model-level weight. */
void build_moe(Fixture& f, int n, int64_t each, std::vector<rad_weight>* experts) {
    const rad_weight m = f.weight("token_embd.weight", 1 * MiB, RAD_ACCESS_VOCAB, -1, -1);
    f.op("embed", {m}, false);
    for (int e = 0; e < n; ++e) {
        const rad_weight w = f.weight("blk.0.ffn_gate_exps." + std::to_string(e) + ".weight", each,
                                      RAD_ACCESS_CONDITIONAL, 0, e);
        f.op("moe_gemm", {w}, false);
        if (experts) experts->push_back(w);
    }
}

/* `layers` routed layers of `n` experts, each a unit of two weights (gate_up of `gu` bytes, down
 * of `dn`) read by two grouped ops per layer, as a routed block declares them: a unit whose
 * weights are not contiguous in memory as they are in a container. */
void build_moe_layers(Fixture& f, int layers, int n, int64_t gu, int64_t dn) {
    const rad_weight m = f.weight("token_embd.weight", 1 * MiB, RAD_ACCESS_VOCAB, -1, -1);
    f.op("embed", {m}, false);
    for (int l = 0; l < layers; ++l) {
        std::vector<rad_weight> g, d;
        for (int e = 0; e < n; ++e) {
            const std::string b = "blk." + std::to_string(l) + ".";
            g.push_back(f.weight(b + "ffn_gate_up_exps." + std::to_string(e) + ".weight", gu,
                                 RAD_ACCESS_CONDITIONAL, l, e));
            d.push_back(f.weight(b + "ffn_down_exps." + std::to_string(e) + ".weight", dn,
                                 RAD_ACCESS_CONDITIONAL, l, e));
        }
        f.op("moe_gemm", g, false);
        f.op("moe_gemm", d, false);
    }
}

/* Every placement decision as one string, so any two differing plans differ here. */
std::string digest(const Program& prog, const Plan& p) {
    std::string s = fmt("strategy=%s dynamic=%d\n", strategy_name(p.strategy), (int)p.dynamic);
    for (rad_weight w = 1; (size_t)w < prog.weights.size(); ++w)
        s += fmt("%s tier=%s site=%s slot=%d\n", prog.weights[w].name.c_str(),
                 tier_name(p.weight[w].tier), site_name(p.weight[w].site),
                 (int)p.weight[w].slab_slot);
    return s;
}

int count_tier(const Program& prog, const Plan& p, Tier t) {
    int n = 0;
    for (rad_weight w = 1; (size_t)w < prog.weights.size(); ++w)
        if (p.weight[w].tier == t) ++n;
    return n;
}

}  /* namespace */

/* ================================================================== the planner */

TEST(planner_fits_an_all_vram_model) {
    Fixture f;
    build_dense(f, 4, 1 * MiB, false);

    Config c = base_config();
    c.placement = "all_vram";
    c.vram_weights_mib = 64;

    PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0};
    Plan p;
    CHECK_OK(Planner(in).plan(&p));

    CHECK_EQ(p.totals.bytes[(int)Tier::VRAM], 5 * MiB);
    CHECK_EQ(p.totals.bytes[(int)Tier::HostPinned], (int64_t)0);
    CHECK_EQ(count_tier(f.prog, p, Tier::VRAM), 5);
    CHECK(p.fits());
    CHECK(p.host_runs.empty());
    CHECK_EQ(p.link_crossings, 0);
    /* Everything on tier 0 and nothing for the mover to do. */
    for (const MoveUnit& mu : p.units) CHECK(!mu.movable);
}

TEST(planner_reports_what_did_not_fit) {
    Fixture f;
    build_dense(f, 4, 1 * MiB, false);

    Config c = base_config();
    c.placement = "all_vram";
    c.vram_weights_mib = 2;

    PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0};
    Plan p;
    CHECK_EQ(Planner(in).plan(&p), RAD_E_NOMEM);
    CHECK(!p.fits());
    /* The report must say what did not fit; the status code alone does not. */
    CHECK(p.report(f.prog).find("DID NOT FIT") != std::string::npos);
}

TEST(planner_host_runs_are_contiguous) {
    Fixture f;
    build_dense(f, 8, 1 * MiB, /*host_kernel=*/true);

    Config c = base_config();
    c.placement = "layer_offload";
    c.vram_weights_mib = 5;      /* 9 MiB of weights: four layers must leave */
    c.host_pool_mib = 64;

    PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0};
    Plan p;
    CHECK_OK(Planner(in).plan(&p));

    /* One contiguous run: alternating per layer would cost eight link crossings per step for the
     * same bytes, a run costs two. */
    CHECK_EQ((int)p.host_runs.size(), 1);
    CHECK_EQ(p.link_crossings, 2);
    if (!p.host_runs.empty()) {
        CHECK_EQ(p.host_runs[0].first_layer, 4);
        CHECK_EQ(p.host_runs[0].last_layer, 7);
    }
    CHECK_EQ(p.totals.bytes[(int)Tier::VRAM], 5 * MiB);
    /* Pageable, not pinned: the host site decides the tier. A host kernel reads a host-site
     * layer, so pinning buys nothing, and pinned memory is scarce (past a driver-dependent total,
     * registration succeeds but leaves the process broken). */
    CHECK_EQ(p.totals.bytes[(int)Tier::HostPageable], 4 * MiB);
    CHECK_EQ(p.totals.bytes[(int)Tier::HostPinned], (int64_t)0);

    /* The run is a suffix, so the device keeps the early layers, which a staged prefix would have
     * to land before the step could start. */
    for (const MoveUnit& mu : p.units) {
        if (mu.layer < 0) continue;
        if (mu.layer < 4) { CHECK(mu.site == Site::Device); CHECK(mu.tier == Tier::VRAM); }
        else              { CHECK(mu.site == Site::Host);   CHECK(mu.tier == Tier::HostPageable); }
    }
}

TEST(planner_stages_when_no_host_kernel_resolved) {
    /* As above but with no RAD_DOMAIN_HOST kernel resolved, so the host site is unavailable and
     * the spilled bytes get a different site: tier and site are independent. */
    Fixture f;
    build_dense(f, 8, 1 * MiB, /*host_kernel=*/false);

    Config c = base_config();
    c.placement = "layer_offload";
    c.vram_weights_mib = 5;
    c.host_pool_mib = 64;

    PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0};
    Plan p;
    CHECK_OK(Planner(in).plan(&p));

    CHECK(p.host_runs.empty());
    CHECK_EQ(p.link_crossings, 0);
    /* Two more layers spill than in the host-kernel case, for the staging ring: a staged unit
     * needs a slab slot, so the planner reserves the ring from the VRAM budget before choosing
     * the cut. Otherwise the plan would fill the budget and have nowhere to stage. */
    CHECK_EQ(p.slab_overhead, (int64_t)RAD_PLACE_STAGE_RING * MiB + RAD_ALIGN_UNIT);
    CHECK_EQ(p.totals.bytes[(int)Tier::HostPinned], 7 * MiB);
    CHECK_EQ(p.totals.bytes[(int)Tier::VRAM], 2 * MiB);
    CHECK(p.totals.bytes[(int)Tier::VRAM] + p.slab_overhead <= p.vram_budget);
    for (const MoveUnit& mu : p.units)
        if (mu.layer >= 1) CHECK(mu.site == Site::DeviceStaged);

    /* Every staged unit has an exact deadline from the declared op order, and the prefetch
     * schedule is sorted by it. */
    CHECK_EQ((int)p.prefetch.size(), 7);
    for (size_t i = 1; i < p.prefetch.size(); ++i)
        CHECK(p.prefetch[i - 1].at_op <= p.prefetch[i].at_op);
    CHECK(p.prefetch[0].at_op > 0);
}

TEST(planner_shards_under_world_size_2) {
    /* The plugin declares dimensions already divided (spec §9), so a rank's unit is its own
     * shard. Invariant: the resident set is identical on every rank, since a slab slot means the
     * same everywhere only if every rank chose the same experts for it. */
    Fixture whole, half0, half1;
    std::vector<rad_weight> e_whole, e_half;
    build_moe(whole, 8, 4 * MiB, &e_whole);
    build_moe(half0, 8, 2 * MiB, &e_half);
    build_moe(half1, 8, 2 * MiB, nullptr);

    Config c = base_config();
    c.placement = "expert_tiered";
    c.vram_weights_mib = 1024;
    c.host_pool_mib = 1024;

    Plan pw, p0, p1;
    { PlannerInput in{&whole.prog, &c, 0, 1, nullptr, 0}; CHECK_OK(Planner(in).plan(&pw)); }
    { PlannerInput in{&half0.prog, &c, 0, 2, nullptr, 0}; CHECK_OK(Planner(in).plan(&p0)); }
    { PlannerInput in{&half1.prog, &c, 1, 2, nullptr, 0}; CHECK_OK(Planner(in).plan(&p1)); }

    /* Each rank plans against its own card with half the expert bytes; the embedding is
     * replicated, so this is not simply half of everything. */
    CHECK_EQ(pw.totals.bytes[(int)Tier::VRAM], 33 * MiB);
    CHECK_EQ(p0.totals.bytes[(int)Tier::VRAM], 17 * MiB);

    /* Same decisions on both ranks, weight for weight; anything rank-local in the ranking (a
     * device query, a clock, a hash seed) fails this. */
    CHECK_EQ(digest(half0.prog, p0), digest(half1.prog, p1));
    CHECK_EQ(p0.rank, 0);
    CHECK_EQ(p1.rank, 1);
}

TEST(planner_spills_experts_before_dense_weights) {
    Fixture f;
    std::vector<rad_weight> ex;
    build_moe(f, 8, 1 * MiB, &ex);

    Config c = base_config();
    c.placement = "expert_tiered";
    c.vram_weights_mib = 5;      /* 1 MiB embedding; the rest is the expert slab */
    c.host_pool_mib = 64;

    PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0};
    Plan p;
    CHECK_OK(Planner(in).plan(&p));

    /* Three resident, not four: the fourth slot is the shadow. Spare slots come from the weights
     * pool (--gpu-headroom-mib is memory the engine never touches), so dynamic placement costs a
     * resident unit and the report says how many. */
    CHECK_EQ(p.totals.bytes[(int)Tier::VRAM], 3 * MiB);
    CHECK_EQ(p.totals.bytes[(int)Tier::HostPinned], 6 * MiB);
    CHECK_EQ(p.slab_overhead, 1 * MiB + RAD_ALIGN_UNIT);
    /* The total fits the plan's budget, so the mover cannot overrun the pool sized for it. */
    CHECK(p.totals.bytes[(int)Tier::VRAM] + p.slab_overhead <= p.vram_budget);

    /* The embedding is read every token and never leaves. */
    CHECK(p.weight[1].tier == Tier::VRAM);
    /* A pooled expert is zero-copy, not staged: the GEMM streams it over the link, so a miss
     * costs one shard and no slab slot. */
    int zero_copy = 0;
    for (const MoveUnit& mu : p.units)
        if (mu.tier == Tier::HostPinned && mu.site == Site::DeviceZeroCopy) ++zero_copy;
    CHECK_EQ(zero_copy, 6);
}

/* A re-split rank's Config reaches both readers of the pool sizes. Engine::plan gives a rank whose
 * cache costs less per token its own Config (vram_budget_rank_split); Pools::configure sizes the
 * weights pool and Budgets::from the slab from the same copy, or the pool is wasted or overrun.
 * plan_check_weights_pool checks this at startup. */
TEST(a_resplit_ranks_config_reaches_the_pools_and_the_planner) {
    Fixture f;
    build_moe(f, 8, 1 * MiB, nullptr);

    Config common = base_config();
    common.placement = "expert_tiered";
    common.vram_weights_mib = 5;      /* 1 MiB embedding, three experts and a shadow */
    common.vram_kv_mib = 8;
    common.host_pool_mib = 64;
    const RankSplit rs = vram_budget_rank_split(common.vram_weights_mib, common.vram_kv_mib, 2,
                                                64 * MiB);
    CHECK(rs.resplit);
    Config rc = common;
    rc.vram_weights_mib = rs.weights_mib;
    rc.vram_kv_mib = rs.kv_mib;

    Pools pools;
    CHECK_OK(pools.configure(rc, 0, &rad_malloc_allocator(), &rad_malloc_allocator()));
    CHECK_EQ(pools.weights().bytes(), rs.weights_mib * MiB);
    CHECK_EQ(pools.kv().bytes(), rs.kv_mib * MiB);
    CHECK_EQ(Budgets::from(rc).vram_weights, pools.weights().bytes());

    Plan grown, flat;
    { PlannerInput in{&f.prog, &rc, 0, 1, nullptr, 0}; CHECK_OK(Planner(in).plan(&grown)); }
    { PlannerInput in{&f.prog, &common, 0, 1, nullptr, 0}; CHECK_OK(Planner(in).plan(&flat)); }
    CHECK_EQ(grown.vram_budget, pools.weights().bytes());
    CHECK_EQ(plan_check_weights_pool(grown, pools.weights().bytes()), RAD_OK);
    /* The grown pool is used: more experts are on the card. */
    CHECK(grown.totals.bytes[(int)Tier::VRAM] > flat.totals.bytes[(int)Tier::VRAM]);
    CHECK(grown.totals.bytes[(int)Tier::VRAM] + grown.slab_overhead <= pools.weights().bytes());
    /* A plan from the common split against the rank's grown pool is refused. */
    CHECK_EQ(plan_check_weights_pool(flat, pools.weights().bytes()), RAD_E_STATE);
}

/* expert_vram_floor places every expert. A lending cache leaves the expert slab exactly the floor,
 * so it must cover the residents, the slab's spare slots, and the slack each first-fit pass can
 * strand, or some experts get no tier and the server refuses to start. Two strides give two slab
 * classes with their own shadow slots, as in a mixed-precision container. */
TEST(the_expert_floor_places_every_expert) {
    Fixture f;
    const rad_weight m = f.weight("token_embd.weight", 1 * MiB, RAD_ACCESS_VOCAB, -1, -1);
    f.op("embed", {m}, false);
    for (int l = 0; l < 8; ++l) {
        const int64_t gu = l < 4 ? 600 * KiB : 1 * MiB, dn = l < 4 ? 300 * KiB : 500 * KiB;
        std::vector<rad_weight> g, d;
        for (int e = 0; e < 16; ++e) {
            const std::string b = "blk." + std::to_string(l) + ".";
            g.push_back(f.weight(b + "ffn_gate_up_exps." + std::to_string(e) + ".weight", gu,
                                 RAD_ACCESS_CONDITIONAL, l, e));
            d.push_back(f.weight(b + "ffn_down_exps." + std::to_string(e) + ".weight", dn,
                                 RAD_ACCESS_CONDITIONAL, l, e));
        }
        f.op("moe_gemm", g, false);
        f.op("moe_gemm", d, false);
    }
    WeightFootprint wf;
    weight_footprint(f.prog, &wf, TablePlacement::Vram, TablePlacement::Disk, false, true);
    CHECK(wf.expert_spare > 0);

    const int64_t host = 100 * MiB;
    auto plan_with = [&](int64_t floor, Plan* p) {
        Config c = base_config();
        c.placement = "expert_tiered";
        c.vram_weights_mib = (wf.statics + floor + MiB - 1) / MiB;
        c.host_pool_mib = host / MiB;
        PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0};
        return Planner(in).plan(p);
    };

    const int64_t floor = expert_vram_floor(wf, host);
    CHECK(floor > 0 && floor < wf.experts);
    Plan p;
    CHECK_OK(plan_with(floor, &p));
    CHECK(p.unfit.empty());
    CHECK_EQ(count_tier(f.prog, p, Tier::SSD), 0);
    CHECK(p.totals.bytes[(int)Tier::VRAM] + p.slab_overhead <= p.vram_budget);

    /* A floor counting only the experts the pool cannot hold plus one unit of slack strands
     * some, because of the spare slots. */
    Plan q;
    CHECK(plan_with(wf.experts - host + wf.expert_unit_max, &q) == RAD_E_NOMEM);
    CHECK(!q.unfit.empty());
}

TEST(planner_warm_starts_from_the_container_profile) {
    Fixture f;
    std::vector<rad_weight> ex;
    build_moe(f, 8, 1 * MiB, &ex);

    /* Experts 5, 6 and 7 are popular in the calibration profile. The profile is only a warm
     * start: it decides which experts begin resident. */
    std::vector<RadFileProfile> prof;
    for (int e = 0; e < 8; ++e) prof.push_back(RadFileProfile{0, e, e >= 5 ? 0.3f : 0.01f, 0.0f});

    Config c = base_config();
    c.placement = "expert_tiered";
    c.vram_weights_mib = 6;      /* 1 MiB embedding + three experts + one shadow slot */
    c.host_pool_mib = 64;

    PlannerInput in{&f.prog, &c, 0, 1, prof.data(), (int64_t)prof.size()};
    Plan p;
    CHECK_OK(Planner(in).plan(&p));

    for (const MoveUnit& mu : p.units) {
        if (mu.expert < 0) continue;
        if (mu.expert >= 5) CHECK(mu.tier == Tier::VRAM);
        else                CHECK(mu.tier == Tier::HostPinned);
    }
}

TEST(planner_refuses_an_offload_strategy_with_no_budget) {
    /* Every pool is budgeted explicitly (spec §6). An offload strategy without a budget would
     * have to infer one, making benchmarks irreproducible. */
    Fixture f;
    build_dense(f, 4, 1 * MiB, true);
    Config c = base_config();
    c.placement = "layer_offload";
    c.vram_weights_mib = 0;

    PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0};
    Plan p;
    CHECK_EQ(Planner(in).plan(&p), RAD_E_INVAL);
}

TEST(deterministic_mode_pins_placement_to_static) {
    Fixture f;
    std::vector<rad_weight> ex;
    build_moe(f, 8, 1 * MiB, &ex);

    Config c = base_config();
    c.placement = "expert_tiered";
    c.vram_weights_mib = 5;
    c.host_pool_mib = 64;
    c.deterministic = true;

    PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0};
    Plan a, b;
    CHECK_OK(Planner(in).plan(&a));
    CHECK_OK(Planner(in).plan(&b));

    /* Identical plans across runs, and nothing movable, so the heat engine stays idle and
     * residency (hence summation order) is fixed for the run (spec §17). */
    CHECK_EQ(digest(f.prog, a), digest(f.prog, b));
    CHECK(!a.dynamic);
    for (const MoveUnit& mu : a.units) CHECK(!mu.movable);

    HeatEngine h;
    CHECK_OK(h.init(a, HeatParams{}));
    CHECK(!h.enabled());
    std::vector<Swap> sw;
    CHECK_OK(h.step(&sw));
    CHECK(sw.empty());

    /* Control: with determinism off the same fixture is movable. */
    c.deterministic = false;
    PlannerInput in2{&f.prog, &c, 0, 1, nullptr, 0};
    Plan d;
    CHECK_OK(Planner(in2).plan(&d));
    CHECK(d.dynamic);
    int movable = 0;
    for (const MoveUnit& mu : d.units) if (mu.movable) ++movable;
    CHECK_EQ(movable, 8);
}

TEST(planner_reads_routed_experts_it_cannot_hold_from_the_container) {
    /* Two routed layers of eight 1.5 MiB experts, a VRAM budget for a few and a host pool for
     * two. With the file tier on, the rest go there and the plan still fits: they are served by
     * reading the whole layer into one of two buffers charged to the same VRAM budget. The
     * budget also carries the unconditional hot cache — each resident expert may hold a shadow
     * slot (up to the cap) — so it is sized for residents plus their shadow slots. */
    Fixture f;
    build_moe_layers(f, 2, 8, 1 * MiB, MiB / 2);

    Config c = base_config();
    c.placement = "expert_tiered";
    c.vram_weights_mib = 26;
    /* The file staging reserve (four slots of the widest unit plus the read windows) and room
     * for two pooled experts. */
    const int64_t reserve = 4 * (MiB + MiB / 2) + RAD_PLACE_FILE_WINDOWS * RAD_PLACE_FILE_WINDOW_BYTES;
    c.host_pool_mib = (reserve + 3 * MiB) / MiB;
    c.weights_disk_tier = true;

    PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0};
    Plan p;
    CHECK_OK(Planner(in).plan(&p));
    CHECK(p.fits());
    CHECK_EQ(p.file_windows, (int64_t)RAD_PLACE_FILE_WINDOWS * RAD_PLACE_FILE_WINDOW_BYTES);

    std::vector<int64_t> per_layer(2, 0);
    int ssd = 0, pooled = 0, resident = 0;
    for (const MoveUnit& mu : p.units) {
        if (mu.expert < 0) continue;
        if (mu.tier == Tier::SSD) {
            ++ssd;
            /* Staged, never movable: the heat engine promotes from a pool slot, which a file-tier
             * unit lacks. */
            CHECK(mu.site == Site::DeviceStaged);
            CHECK(!mu.movable);
            per_layer[(size_t)mu.layer] += align_up(mu.bytes, RAD_ALIGN_UNIT);
        } else if (mu.tier == Tier::HostPinned) {
            ++pooled;
        } else if (mu.tier == Tier::VRAM) {
            ++resident;
        }
    }
    CHECK(ssd > 0);
    CHECK_EQ(pooled, 2);
    CHECK(resident > 0);
    CHECK_EQ(resident + pooled + ssd, 16);
    /* Two buffers, each sized for the largest layer's file-tier units at the stager's packing. */
    CHECK_EQ(p.file_layer_max, std::max(per_layer[0], per_layer[1]));
    CHECK(p.file_stage_vram >= 2 * p.file_layer_max + (int64_t)RAD_ALIGN_UNIT);
    /* Residents, shadow slots and the two buffers fit the VRAM budget, so the mover cannot
     * overrun the pool. */
    CHECK(p.totals.bytes[(int)Tier::VRAM] + p.slab_overhead + p.file_stage_vram <= p.vram_budget);
    /* The prefetch ring is unused for these; the file stager has its own buffers. */
    for (const SlabClass& sc : p.classes) CHECK_EQ(sc.n_stage, (int64_t)0);

    /* Without the tier the same budgets strand them, and the refusal names the flag. */
    c.weights_disk_tier = false;
    c.host_pool_mib = 3;
    PlannerInput in2{&f.prog, &c, 0, 1, nullptr, 0};
    Plan q;
    CHECK_EQ(Planner(in2).plan(&q), RAD_E_NOMEM);
    CHECK(!q.unfit.empty());
    bool named = false;
    for (const Plan::Unfit& u : q.unfit)
        if (u.why.find("--weights-disk-tier") != std::string::npos) named = true;
    CHECK(named);
}

TEST(weight_class_key_collapses_indices) {
    CHECK_EQ(weight_class_key("blk.12.ffn_gate_exps.3.weight"), std::string("blk.*.ffn_gate_exps.*.weight"));
    CHECK_EQ(weight_class_key("token_embd.weight"), std::string("token_embd.weight"));
}

/* ================================================================== the heat engine */
namespace {

/* An eight-expert MoE with four resident, ready for the heat engine. */
struct HeatFixture {
    Fixture f;
    Plan    plan;
    Config  cfg;

    explicit HeatFixture(int n_expert = 8, int64_t vram_mib = 7) {
        std::vector<rad_weight> ex;
        build_moe(f, n_expert, 1 * MiB, &ex);
        cfg = base_config();
        cfg.placement = "expert_tiered";
        cfg.vram_weights_mib = vram_mib;
        cfg.host_pool_mib = 64;
        PlannerInput in{&f.prog, &cfg, 0, 1, nullptr, 0};
        Planner(in).plan(&plan);
    }

    int32_t unit_of_expert(int32_t e) const {
        for (int32_t i = 0; i < (int32_t)plan.units.size(); ++i)
            if (plan.units[(size_t)i].expert == e) return i;
        return -1;
    }
};

}  /* namespace */

TEST(heat_promotes_a_hot_pooled_expert) {
    HeatFixture hf;
    HeatEngine h;
    HeatParams hp;
    CHECK_OK(h.init(hf.plan, hp));
    CHECK(h.enabled());

    /* Expert 7 is pooled (no profile, so the first four win on unit order). Ten tokens route to
     * it, and the credit is that count, not a flat one: an expert chosen by several tokens is
     * likelier to be chosen again. */
    const int32_t hot = hf.unit_of_expert(7);
    CHECK(hot >= 0);
    CHECK(h.tier(hot) != Tier::VRAM);

    const int32_t ids[1] = {7};
    const int32_t counts[1] = {10};
    h.observe_ids(0, ids, counts, 1);

    std::vector<Swap> sw;
    CHECK_OK(h.step(&sw));
    CHECK(!sw.empty());
    if (!sw.empty()) {
        CHECK_EQ(sw[0].promote, hot);
        CHECK(h.tier(sw[0].demote) == Tier::VRAM);
        /* Promotion and demotion come as a pair; promoting alone would overfill the slab. */
        CHECK(sw[0].promote != sw[0].demote);
    }
}

TEST(heat_does_not_thrash_on_an_alternating_pattern) {
    /* The four resident and four pooled experts are selected in strict alternation, so both
     * halves are of equal value and their heats converge within one decay step. A swap here moves
     * two shards over the link for no gain, on the link the zero-copy tier uses.
     *
     * The guards (hysteresis, min_gain) are the only defence; there is no admission filter. The
     * same pattern with guards removed is the control, since an engine that never proposes would
     * also score zero. */
    auto run = [](HeatParams hp) {
        HeatFixture hf;
        HeatEngine h;
        h.init(hf.plan, hp);
        CHECK(h.tier(hf.unit_of_expert(0)) == Tier::VRAM);
        CHECK(h.tier(hf.unit_of_expert(7)) != Tier::VRAM);

        const int32_t resident[4] = {0, 1, 2, 3};
        const int32_t pooled[4] = {4, 5, 6, 7};
        const int32_t ones[4] = {1, 1, 1, 1};

        uint64_t proposed = 0;
        for (int step = 0; step < 200; ++step) {
            h.observe_ids(0, (step & 1) ? pooled : resident, ones, 4);
            std::vector<Swap> sw;
            h.step(&sw);
            /* Nothing is issued, so release the units; otherwise busy flags, not policy, would
             * silence the engine. */
            for (const Swap& s : sw) h.abandon(s);
            proposed += sw.size();
        }
        return proposed;
    };

    HeatParams shipped;                    /* hysteresis 1.25, min_gain 1.5 */
    const uint64_t guarded = run(shipped);

    HeatParams none = shipped;
    none.hysteresis = 1.0f;
    none.min_gain = 0.0f;
    const uint64_t unguarded = run(none);

    CHECK_EQ(guarded, (uint64_t)0);
    /* Control: without guards the pattern churns whenever the pooled half was credited last. */
    CHECK(unguarded > 100);
}

/* A prefill chunk is credited as one capped histogram. Ten times the cap's selections credits a
 * tenth of each count, keeping the experts' order while one chunk cannot outweigh the decode steps
 * around it; a histogram under the cap is credited in full. */
TEST(heat_caps_what_one_histogram_credits) {
    HeatFixture hf;
    HeatEngine h;
    HeatParams hp;
    hp.credit_cap = 512.0f;
    CHECK_OK(h.init(hf.plan, hp));
    const int32_t a = hf.unit_of_expert(4), b = hf.unit_of_expert(5), c = hf.unit_of_expert(6);

    const int32_t big_ids[2] = {4, 5};
    const int32_t big[2] = {4000, 1120};
    h.observe_ids(0, big_ids, big, 2);
    CHECK(std::fabs(h.heat(a) - 400.0f) < 1e-3f);
    CHECK(std::fabs(h.heat(b) - 112.0f) < 1e-3f);

    const int32_t small_ids[1] = {6};
    const int32_t small[1] = {12};
    h.observe_ids(0, small_ids, small, 1);
    CHECK_EQ(h.heat(c), 12.0f);
    CHECK(std::fabs(h.heat(a) - 400.0f * hp.decay) < 1e-3f);
}

TEST(heat_respects_the_move_budget) {
    HeatFixture hf;
    HeatEngine h;
    HeatParams hp;
    hp.moves_per_step = 3;
    hp.min_gain = 0.5f;
    CHECK_OK(h.init(hf.plan, hp));

    /* Every pooled expert is hot, so the guards refuse nothing and only the move budget can bound
     * the proposals. */
    const int32_t ids[4] = {4, 5, 6, 7};
    const int32_t counts[4] = {20, 20, 20, 20};
    h.observe_ids(0, ids, counts, 4);

    std::vector<Swap> sw;
    CHECK_OK(h.step(&sw));
    CHECK_EQ((int)sw.size(), 3);

    /* No unit appears twice: a proposal marks both halves busy, or the mover would repeat a
     * transfer. */
    for (size_t i = 0; i < sw.size(); ++i)
        for (size_t j = i + 1; j < sw.size(); ++j) {
            CHECK(sw[i].promote != sw[j].promote);
            CHECK(sw[i].demote != sw[j].demote);
        }
}

namespace {

/* A xorshift32 draw below n. */
struct Rnd {
    uint32_t x;
    uint32_t operator()(uint32_t n) { x ^= x << 13; x ^= x >> 17; x ^= x << 5; return x % n; }
};

/* The heat tests' plan: an embedding and four layers of 16 experts (expert e of `bytes(e)`),
 * expert_tiered with `vram_mib` of VRAM for weights. */
template <class Bytes>
Plan expert_plan(Fixture& f, Bytes bytes, int64_t vram_mib, int64_t host_mib) {
    const rad_weight m = f.weight("token_embd.weight", 1 * MiB, RAD_ACCESS_VOCAB, -1, -1);
    f.op("embed", {m}, false);
    for (int l = 0; l < 4; ++l)
        for (int e = 0; e < 16; ++e) {
            const rad_weight w = f.weight("blk." + std::to_string(l) + ".ffn_gate_exps." +
                                              std::to_string(e) + ".weight",
                                          bytes(e), RAD_ACCESS_CONDITIONAL, l, e);
            f.op("moe_gemm", {w}, false);
        }
    Config cfg = base_config();
    cfg.placement = "expert_tiered";
    cfg.vram_weights_mib = vram_mib;
    cfg.host_pool_mib = host_mib;
    PlannerInput in{&f.prog, &cfg, 0, 1, nullptr, 0};
    Plan plan;
    CHECK_OK(Planner(in).plan(&plan));
    return plan;
}

/* What step() must propose: up to `moves` picks on `ref`, a copy of the engine, each pair marked
 * busy as step() marks it. */
std::vector<Swap> repeated_picks(HeatEngine& ref, int moves) {
    std::vector<Swap> want;
    for (int mv = 0; mv < moves; ++mv) {
        Swap sw;
        if (!ref.pick(&sw)) break;
        ref.set_busy(sw.promote, true);
        ref.set_busy(sw.demote, true);
        want.push_back(sw);
    }
    return want;
}

bool same_swaps(const std::vector<Swap>& a, const std::vector<Swap>& b) {
    bool same = a.size() == b.size();
    for (size_t i = 0; same && i < a.size(); ++i)
        same = a[i].promote == b[i].promote && a[i].demote == b[i].demote;
    return same;
}

/* The mover's side of proposed swaps: three in four land, the rest are abandoned. */
void land_or_abandon(HeatEngine& h, const std::vector<Swap>& got, Rnd& rnd) {
    for (const Swap& sw : got) {
        if (rnd(4) == 0) { h.abandon(sw); continue; }
        h.set_tier(sw.promote, Tier::VRAM);
        h.set_tier(sw.demote, Tier::HostPinned);
        h.set_busy(sw.promote, false);
        h.set_busy(sw.demote, false);
    }
}

}  /* namespace */

/* step() ranks every movable unit once and proposes from both ends; pick() states the policy one
 * pair per call. They must agree exactly: same pairs in the same order, same units left busy, same
 * stopping point. Checked over random heats with frequent ties (small integer counts), random busy
 * units, and four layers with differing resident counts, exercising every tie-break. */
TEST(heat_step_proposes_exactly_what_repeated_picks_would) {
    Fixture f;
    const Plan plan = expert_plan(f, [](int) { return 1 * MiB; }, 29, 128);

    Rnd rnd{0x9E3779B9u};
    int agreed = 0;
    for (int trial = 0; trial < 400; ++trial) {
        HeatParams hp;
        hp.moves_per_step = 1 + (int)rnd(8);
        hp.min_gain = (float)rnd(3);
        hp.hysteresis = 1.0f + 0.25f * (float)rnd(3);
        HeatEngine h;
        CHECK_OK(h.init(plan, hp));
        if (!h.enabled()) return;
        const int32_t n_unit = (int32_t)plan.units.size();
        /* Uneven residency across layers exercises the resident-count tie-break. */
        for (int k = 0; k < 6; ++k) {
            const int32_t u = (int32_t)rnd((uint32_t)n_unit);
            h.set_tier(u, h.tier(u) == Tier::VRAM ? Tier::HostPinned : Tier::VRAM);
        }
        for (int l = 0; l < 4; ++l) {
            int32_t ids[16], counts[16];
            for (int e = 0; e < 16; ++e) { ids[e] = e; counts[e] = (int32_t)rnd(4); }
            h.observe_ids(l, ids, counts, 16);
        }
        for (int k = 0; k < 5; ++k) h.set_busy((int32_t)rnd((uint32_t)n_unit), true);

        HeatEngine ref = h;
        const std::vector<Swap> want = repeated_picks(ref, hp.moves_per_step);
        std::vector<Swap> got;
        CHECK_OK(h.step(&got));
        bool same = same_swaps(got, want);
        for (int32_t u = 0; same && u < n_unit; ++u) same = h.busy(u) == ref.busy(u);
        CHECK(same);
        if (same) ++agreed;
    }
    CHECK_EQ(agreed, 400);
}

/* Two slab classes, as on a rank holding an uneven slice of every expert (three blocks of one
 * parity, two of the other), so even and odd experts' units differ in size. A swap stays inside a
 * class (the promotion takes and the demotion frees a slot of that class), each class has its own
 * move budget, and step() still matches repeated picks. */
TEST(heat_swaps_stay_inside_a_slab_class) {
    Fixture f;
    const Plan plan = expert_plan(f, [](int e) { return (e % 2 ? 2 : 3) * MiB; }, 81, 256);
    int expert_classes = 0;
    for (const SlabClass& sc : plan.classes) expert_classes += sc.name.rfind("expert", 0) == 0;
    CHECK_EQ(expert_classes, 2);

    Rnd rnd{0x2545F491u};
    int agreed = 0, crossed = 0, proposed = 0;
    for (int trial = 0; trial < 400; ++trial) {
        HeatParams hp;
        hp.moves_per_step = 1 + (int)rnd(8);
        hp.min_gain = (float)rnd(3);
        hp.hysteresis = 1.0f + 0.25f * (float)rnd(3);
        HeatEngine h;
        CHECK_OK(h.init(plan, hp));
        if (!h.enabled()) return;
        const int32_t n_unit = (int32_t)plan.units.size();
        for (int k = 0; k < 6; ++k) {
            const int32_t u = (int32_t)rnd((uint32_t)n_unit);
            h.set_tier(u, h.tier(u) == Tier::VRAM ? Tier::HostPinned : Tier::VRAM);
        }
        /* Some layers route only one parity, so one class can run out of candidates while the
         * other still has pairs. */
        for (int l = 0; l < 4; ++l) {
            int32_t ids[16], counts[16];
            const int only = (int)rnd(3);
            for (int e = 0; e < 16; ++e) {
                ids[e] = e;
                counts[e] = (only < 2 && e % 2 != only) ? 0 : (int32_t)rnd(5);
            }
            h.observe_ids(l, ids, counts, 16);
        }
        for (int k = 0; k < 5; ++k) h.set_busy((int32_t)rnd((uint32_t)n_unit), true);

        /* Reference: repeated picks, at most `moves_per_step` per class. A pick from a class at
         * its budget is set aside, busy in the copy only. */
        HeatEngine ref = h;
        std::vector<Swap> want;
        std::vector<uint8_t> aside((size_t)n_unit, 0);
        std::vector<int> per(plan.classes.size(), 0);
        for (int guard = 0; guard < n_unit; ++guard) {
            Swap sw;
            if (!ref.pick(&sw)) break;
            ref.set_busy(sw.promote, true);
            ref.set_busy(sw.demote, true);
            int& n = per[(size_t)plan.units[(size_t)sw.promote].slab_class];
            if (n >= hp.moves_per_step) {
                aside[(size_t)sw.promote] = aside[(size_t)sw.demote] = 1;
                continue;
            }
            ++n;
            want.push_back(sw);
        }
        std::vector<Swap> got;
        CHECK_OK(h.step(&got));
        bool same = same_swaps(got, want);
        for (int32_t u = 0; same && u < n_unit; ++u)
            same = aside[(size_t)u] || h.busy(u) == ref.busy(u);
        CHECK(same);
        if (same) ++agreed;
        for (const Swap& sw : got) {
            ++proposed;
            crossed += plan.units[(size_t)sw.promote].slab_class !=
                       plan.units[(size_t)sw.demote].slab_class;
        }
    }
    CHECK_EQ(agreed, 400);
    CHECK_EQ(crossed, 0);
    CHECK(proposed > 0);
}

/* The same agreement across dispatches. step() maintains each layer's two ends incrementally
 * through decays, credits, tier changes and units entering or leaving flight, so a long run mixing
 * all four is compared dispatch by dispatch against pick() on a copy. */
TEST(heat_step_agrees_with_picks_over_a_run_of_dispatches) {
    Fixture f;
    const Plan plan = expert_plan(f, [](int) { return 1 * MiB; }, 29, 128);

    Rnd rnd{0x2545F491u};
    for (int run = 0; run < 20; ++run) {
        HeatParams hp;
        hp.moves_per_step = 1 + (int)rnd(3);
        hp.min_gain = 1.0f + (float)rnd(4);
        hp.hysteresis = 1.0f + 0.25f * (float)rnd(4);
        HeatEngine h;
        CHECK_OK(h.init(plan, hp));
        if (!h.enabled()) return;
        const int32_t n_unit = (int32_t)plan.units.size();
        int agreed = 0, proposed = 0;
        for (int d = 0; d < 300; ++d) {
            /* Per-dispatch histograms: mostly quiet, a few hot experts, occasional bursts. */
            for (int l = 0; l < 4; ++l) {
                int32_t counts[16];
                for (int e = 0; e < 16; ++e)
                    counts[e] = rnd(5) == 0 ? (int32_t)rnd(d % 37 == 0 ? 40 : 6) : 0;
                h.observe(l, counts, 16, 1.0f);
            }
            HeatEngine ref = h;
            const std::vector<Swap> want = repeated_picks(ref, hp.moves_per_step);
            std::vector<Swap> got;
            CHECK_OK(h.step(&got));
            const bool same = same_swaps(got, want);
            CHECK(same);
            if (same) ++agreed;
            proposed += (int)got.size();
            /* Most swaps land, some are abandoned, and occasionally a unit changes tier
             * independently. */
            land_or_abandon(h, got, rnd);
            if (rnd(10) == 0) {
                const int32_t u = (int32_t)rnd((uint32_t)n_unit);
                h.set_tier(u, h.tier(u) == Tier::VRAM ? Tier::HostPinned : Tier::VRAM);
            }
        }
        CHECK_EQ(agreed, 300);
        CHECK(proposed > 0);
    }
}

/* Histograms are matched to units by expert id, so the layouts that could break that are checked:
 * an expert count that is not a whole number of blocks, an expert id no unit holds, an immovable
 * expert, two units naming one expert, a movable unit with a layer but no expert id, one with no
 * layer, and histograms shorter or longer than the layer. Every heat and priced read is compared
 * per dispatch with a per-unit model, and every proposal with pick() on a copy. Layers are skipped
 * at random so step(), not the next observe(), rebuilds an end a tier change invalidated. */
TEST(heat_credits_and_prices_every_layout_the_plan_can_have) {
    Plan plan;
    plan.dynamic = true;
    const int n_ex[5] = {5, 17, 33, 16, 1};
    auto add = [&](int32_t layer, int32_t expert, bool movable, int64_t bytes) {
        MoveUnit mu;
        mu.layer = layer; mu.expert = expert; mu.movable = movable; mu.bytes = bytes;
        mu.tier = plan.units.size() % 3 == 0 ? Tier::VRAM : Tier::HostPinned;
        plan.units.push_back(mu);
        return (int32_t)plan.units.size() - 1;
    };
    /* by_expert[l][e]: the unit a histogram entry credits (the last unit naming it), or -1. */
    std::vector<std::vector<int32_t>> by_expert(5);
    for (int l = 0; l < 5; ++l) {
        by_expert[(size_t)l].assign((size_t)n_ex[l], -1);
        for (int e = 0; e < n_ex[l]; ++e) {
            if (l == 2 && (e == 7 || e == 20)) continue;                 /* a hole */
            by_expert[(size_t)l][(size_t)e] = add(l, e, e % 7 != 6, (int64_t)(1 + l * 64 + e) << 10);
        }
    }
    by_expert[1][3] = add(1, 3, true, 5 << 10);                          /* a second unit for (1, 3) */
    add(3, -1, true, 7 << 10);
    add(3, -1, true, 9 << 10);
    add(-1, -1, true, 11 << 10);
    add(-1, -1, true, 13 << 10);
    add(-1, -1, false, 15 << 10);
    const int32_t n_unit = (int32_t)plan.units.size();

    HeatParams hp;
    hp.moves_per_step = 3;
    HeatEngine h;
    CHECK_OK(h.init(plan, hp));
    CHECK(h.enabled());

    std::vector<float> want((size_t)n_unit, 0.0f);
    uint64_t routed = 0, resident = 0, stream = 0;
    Rnd rnd{0x9E3779B9u};
    int agreed = 0, proposed = 0, compared = 0;
    for (int d = 0; d < 400; ++d) {
        for (int l = 0; l < 5; ++l) {
            if (rnd(6) == 0) continue;
            /* Shorter than, equal to, or longer than the layer; entries past the layer's last
             * expert must be ignored, not read as the next layer's. */
            const int n = std::max(0, n_ex[l] + (int)rnd(9) - 4);
            std::vector<int32_t> counts((size_t)n + 1);   /* never empty: a null histogram is ignored */
            for (int e = 0; e < n; ++e) counts[(size_t)e] = rnd(3) == 0 ? (int)rnd(7) - 1 : 0;
            const float w = rnd(4) == 0 ? 3.0f : 1.0f;
            const uint64_t passes = w >= 1.0f ? (uint64_t)(w + 0.5f) : 1u;
            h.observe(l, counts.data(), n, w);
            for (int32_t u = 0; u < n_unit; ++u)
                if (plan.units[(size_t)u].layer == l) want[(size_t)u] = want[(size_t)u] * hp.decay;
            for (int e = 0; e < std::min(n, n_ex[l]); ++e) {
                const int32_t u = by_expert[(size_t)l][(size_t)e];
                if (u < 0 || counts[(size_t)e] <= 0) continue;
                want[(size_t)u] = want[(size_t)u] + (float)counts[(size_t)e] * w;
                routed += passes;
                if (h.tier(u) == Tier::VRAM) resident += passes;
                else stream += (uint64_t)plan.units[(size_t)u].bytes * passes;
            }
        }
        for (int32_t u = 0; u < n_unit; ++u) {
            if (!plan.units[(size_t)u].movable) continue;
            CHECK_EQ(h.heat(u), want[(size_t)u]);
            ++compared;
        }
        CHECK_EQ(h.stats().routed, routed);
        CHECK_EQ(h.stats().routed_resident, resident);
        CHECK_EQ(h.stats().stream_bytes, stream);

        HeatEngine ref = h;
        const std::vector<Swap> expect = repeated_picks(ref, hp.moves_per_step);
        std::vector<Swap> got;
        CHECK_OK(h.step(&got));
        const bool same = same_swaps(got, expect);
        CHECK(same);
        if (same) ++agreed;
        proposed += (int)got.size();
        land_or_abandon(h, got, rnd);
        if (rnd(5) == 0) {
            const int32_t u = (int32_t)rnd((uint32_t)n_unit);
            h.set_tier(u, h.tier(u) == Tier::VRAM ? Tier::HostPinned : Tier::VRAM);
        }
    }
    CHECK_EQ(agreed, 400);
    CHECK(proposed > 0);
    CHECK(compared > 0);
}

TEST(heat_is_off_when_nothing_is_movable) {
    Fixture f;
    build_dense(f, 4, 1 * MiB, false);
    Config c = base_config();
    c.placement = "all_vram";
    c.vram_weights_mib = 64;
    PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0};
    Plan p;
    CHECK_OK(Planner(in).plan(&p));

    HeatEngine h;
    CHECK_OK(h.init(p, HeatParams{}));
    CHECK(!h.enabled());   /* all in VRAM: the mover never runs either */
}

/* ================================================================== the mover */

/* A dispatch's compute work, standing in for the layer's kernels. Every mover test that drives
 * dispatches submits this to the compute stream. With an empty compute stream:
 *
 *   The quarantine barrier is vacuous. It is an event on the compute stream meaning "every kernel
 *   that could still name the freed slot has retired"; on an empty stream it completes before the
 *   mover's transfer starts.
 *
 *   The lead gate is a no-op. Its host wait paces the dispatch thread to the card, and so supplies
 *   the mover with slots; on an empty stream it paces nothing.
 *
 *   The loop outruns the transfers. A transfer takes microseconds and an empty iteration
 *   nanoseconds, so a dispatch-count bound assumes a relation between dispatch rate and link
 *   latency that does not hold (on the host backend the worker may not even be scheduled).
 */
struct ComputeWork {
    static constexpr int64_t kBytes = 2 * 1024 * 1024;
    void* a = nullptr;
    void* b = nullptr;

    ComputeWork() {
        a = rad_dev_alloc(kBytes, RAD_MEM_DEVICE);
        b = rad_dev_alloc(kBytes, RAD_MEM_DEVICE);
    }
    ~ComputeWork() {
        if (a) rad_dev_free(a, RAD_MEM_DEVICE);
        if (b) rad_dev_free(b, RAD_MEM_DEVICE);
    }
    ComputeWork(const ComputeWork&) = delete;
    ComputeWork& operator=(const ComputeWork&) = delete;

    /* Issued right after begin_dispatch, where a layer's launches go: the mover records its
     * ordering gate after them, which is why engine_tick sits at the end of a layer dispatch. */
    void run(RadStream s) const {
        if (a && b) rad_memcpy_async(a, b, kBytes, s);
    }
};

namespace {

/* Loops waiting for the mover are bounded by wall-clock time, not dispatch count. The lead gate
 * paces this thread to the compute stream only; on a loaded machine the host backend's mover
 * worker can go unscheduled for hundreds of dispatches, so a count bound fails spuriously while a
 * deadline fails only if the mover has stopped. */
struct Deadline {
    std::chrono::steady_clock::time_point end;
    explicit Deadline(int seconds = 60)
        : end(std::chrono::steady_clock::now() + std::chrono::seconds(seconds)) {}
    bool passed() const { return std::chrono::steady_clock::now() >= end; }
};

bool any_busy(const HeatEngine& h, const Plan& p) {
    for (int32_t u = 0; u < (int32_t)p.units.size(); ++u)
        if (h.busy(u)) return true;
    return false;
}

}  /* namespace */

TEST(slot_pool_hands_out_and_takes_back) {
    SlotPool s;
    s.init(4, 64);
    CHECK(s.empty());
    CHECK_EQ(s.take(), (int64_t)-1);   /* starvation is the normal steady state, not an error */
    s.add_free(2);
    s.add_free(3);
    CHECK_EQ(s.n_free(), (int64_t)2);
    /* Lowest first, so occupied slots pack toward the bottom and the top can be handed back. */
    CHECK_EQ(s.take(), (int64_t)2);
    CHECK_EQ(s.take(), (int64_t)3);
    CHECK(s.empty());
}

/* take_where returns the lowest free slot the predicate accepts and leaves the others free, still
 * handed out lowest first (Mover::pool_slot_for uses it to keep a layer's pool slots together). */
TEST(a_slot_pool_takes_the_lowest_slot_a_predicate_wants) {
    SlotPool s;
    s.init(8, 64);
    for (int64_t k : {1, 3, 4, 6, 7}) s.add_free(k);
    CHECK_EQ(s.take_where([](int64_t k) { return k % 3 == 0; }), (int64_t)3);
    CHECK_EQ(s.take_where([](int64_t k) { return k == 5; }), (int64_t)-1);   /* not free */
    CHECK_EQ(s.n_free(), (int64_t)4);
    CHECK_EQ(s.take(), (int64_t)1);
    CHECK_EQ(s.take_where([](int64_t k) { return k > 4; }), (int64_t)6);
    CHECK_EQ(s.take(), (int64_t)4);
    CHECK_EQ(s.take(), (int64_t)7);
    CHECK(s.empty());
}

TEST(a_slot_pool_grows_only_into_the_capacity_it_was_given) {
    SlotPool s;
    s.init(2, 64, 5);
    CHECK_EQ(s.size(), (int64_t)2);
    CHECK_EQ(s.capacity(), (int64_t)5);
    CHECK_EQ(s.take(), (int64_t)-1);   /* both live slots are held by the plan */

    s.set_target(4);
    CHECK_EQ(s.size(), (int64_t)4);
    CHECK_EQ(s.n_free(), (int64_t)2);
    CHECK_EQ(s.take(), (int64_t)2);
    CHECK_EQ(s.take(), (int64_t)3);

    s.set_target(99);                  /* clamped to the capacity */
    CHECK_EQ(s.size(), (int64_t)5);
    CHECK_EQ(s.take(), (int64_t)4);
    CHECK_EQ(s.take(), (int64_t)-1);
}

TEST(a_slot_pool_shrinks_past_what_nothing_holds) {
    SlotPool s;
    s.init(2, 64, 6);
    s.set_target(6);
    CHECK_EQ(s.take(), (int64_t)2);
    CHECK_EQ(s.take(), (int64_t)3);
    /* 4 and 5 are free, so the mark falls to 4 at once and stops at held slot 3. */
    s.set_target(0);
    CHECK_EQ(s.size(), (int64_t)4);
    CHECK_EQ(s.n_free(), (int64_t)0);
    CHECK_EQ(s.take(), (int64_t)-1);   /* nothing above the target is handed out */

    s.add_free(3);
    CHECK_EQ(s.size(), (int64_t)3);
    s.add_free(2);
    CHECK_EQ(s.size(), (int64_t)2);    /* the original live set is the floor */
}

TEST(a_slot_a_shrink_stranded_is_never_handed_out_twice) {
    SlotPool s;
    s.init(0, 64, 4);
    s.set_target(4);
    CHECK_EQ(s.n_free(), (int64_t)4);
    s.set_target(2);                   /* 2 and 3 were free, so the mark falls at once */
    CHECK_EQ(s.size(), (int64_t)2);
    CHECK_EQ(s.n_free(), (int64_t)2);
    s.set_target(4);                   /* regrows without duplicating heap entries */
    CHECK_EQ(s.n_free(), (int64_t)4);
    CHECK_EQ(s.take(), (int64_t)0);
    CHECK_EQ(s.take(), (int64_t)1);
    CHECK_EQ(s.take(), (int64_t)2);
    CHECK_EQ(s.take(), (int64_t)3);
    CHECK_EQ(s.take(), (int64_t)-1);
}

TEST(a_slot_pool_regrows_every_slot_a_shrink_dropped) {
    /* A dropped slot that take() skipped on the way to a lower one. */
    {
        SlotPool s;
        s.init(0, 64, 4);
        s.set_target(4);
        s.set_target(2);
        CHECK_EQ(s.take(), (int64_t)0);
        CHECK_EQ(s.take(), (int64_t)1);
        CHECK_EQ(s.take(), (int64_t)-1);   /* 2 and 3 are above the target */
        s.set_target(4);
        CHECK_EQ(s.n_free(), (int64_t)2);
        CHECK_EQ(s.take(), (int64_t)2);
        CHECK_EQ(s.take(), (int64_t)3);
        CHECK_EQ(s.take(), (int64_t)-1);
    }
    /* A slot held above the target at shrink time, freed afterwards. */
    {
        SlotPool s;
        s.init(2, 64, 6);
        s.set_target(6);
        CHECK_EQ(s.take(), (int64_t)2);
        CHECK_EQ(s.take(), (int64_t)3);
        CHECK_EQ(s.take(), (int64_t)4);
        s.set_target(2);
        CHECK_EQ(s.size(), (int64_t)5);    /* the mark stops at the slot 4 holds */
        s.add_free(4);
        CHECK_EQ(s.size(), (int64_t)4);
        s.set_target(6);
        CHECK_EQ(s.size(), (int64_t)6);
        CHECK_EQ(s.n_free(), (int64_t)2);
        CHECK_EQ(s.take(), (int64_t)4);
        CHECK_EQ(s.take(), (int64_t)5);
        CHECK_EQ(s.take(), (int64_t)-1);
        /* The full capacity is back: every slot returned comes out once. */
        for (int64_t k = 2; k < 6; ++k) s.add_free(k);
        CHECK_EQ(s.n_free(), (int64_t)4);
        for (int64_t k = 2; k < 6; ++k) CHECK_EQ(s.take(), k);
        CHECK_EQ(s.take(), (int64_t)-1);
    }
}

/* Expert-tiered placement keeps every dense unit on the card when they fit, so the file tier's
 * staging is sized for the routed experts alone: a dense layer far wider than an expert no longer
 * sets it, and a pool too small for that dense layer's slots is accepted. When the dense units do
 * not fit the card they may spill, the full reserve applies, and the same pool is refused. */
TEST(planner_sizes_file_staging_for_routed_experts_when_dense_units_stay_on_the_card) {
    Fixture f;
    build_moe_layers(f, 2, 8, 1 * MiB, MiB / 2);
    for (int l = 0; l < 2; ++l)
        f.op("gemm", { f.weight("blk." + std::to_string(l) + ".attn.weight", 8 * MiB,
                                RAD_ACCESS_PER_TOKEN, l, -1) }, false);

    Config c = base_config();
    c.placement = "expert_tiered";
    c.vram_weights_mib = 40;   /* the embedding, both dense layers and some experts */
    const int64_t routed = 4 * (MiB + MiB / 2) + RAD_PLACE_FILE_WINDOWS * RAD_PLACE_FILE_WINDOW_BYTES;
    c.host_pool_mib = (routed + 3 * MiB) / MiB + 1;   /* below 4 x 8 MiB + the windows */
    c.weights_disk_tier = true;
    PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0};
    Plan p;
    CHECK_OK(Planner(in).plan(&p));
    CHECK(p.fits());
    CHECK_EQ(p.file_stage_reserve, routed);
    int dense = 0, ssd = 0;
    for (const MoveUnit& mu : p.units) {
        if (mu.expert < 0 && mu.layer >= 0) { ++dense; CHECK(mu.tier == Tier::VRAM); }
        if (mu.tier == Tier::SSD) { ++ssd; CHECK(mu.expert >= 0); }
    }
    CHECK_EQ(dense, 2);
    CHECK(ssd > 0);

    c.vram_weights_mib = 12;   /* the dense layers alone do not fit */
    PlannerInput in2{&f.prog, &c, 0, 1, nullptr, 0};
    Plan q;
    CHECK_EQ(Planner(in2).plan(&q), RAD_E_NOMEM);
}

TEST(file_tier_alignment_is_the_container_alignment) {
    /* O_DIRECT needs aligned offsets, lengths and buffers, and every RadFileEntry offset is
     * RAD_ALIGN_UNIT-aligned, so the fast path needs no head adjustment. */
    CHECK_EQ(FileTier::direct_align(1), (int64_t)RAD_ALIGN_UNIT);
    CHECK_EQ(FileTier::direct_align(RAD_ALIGN_UNIT), (int64_t)RAD_ALIGN_UNIT);
    CHECK_EQ(FileTier::direct_align(RAD_ALIGN_UNIT + 1), (int64_t)(2 * RAD_ALIGN_UNIT));
    FileTier t;
    CHECK(!t.ok());
    CHECK_EQ(t.read(0, 16, nullptr, 0), RAD_E_STATE);
}

TEST(mover_publishes_addresses_for_every_tier) {
    HeatFixture hf;
    Mover m;
    MoverConfig mc;
    CHECK_OK(m.init(hf.f.prog, hf.plan, mc));

    ResidencyTable* rt = m.residency();
    for (rad_weight w = 1; (size_t)w < hf.f.prog.weights.size(); ++w) {
        const WeightSlot* s = rt->lookup(w);
        CHECK(s != nullptr);
        /* Every unit here is in VRAM or the pinned pool, so every weight has an address (a
         * file-tier unit has none until staged). */
        if (s) CHECK(s->ptr != nullptr);
    }
    CHECK(rt->lookup(9999u) == nullptr);
}

namespace {
/* A position-hashed container byte, so a weight read from the wrong offset or copied to the wrong
 * place in its unit mismatches on almost every byte. */
uint8_t container_byte(int64_t i) {
    uint64_t z = (uint64_t)i + 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return (uint8_t)((z ^ (z >> 31)) >> 56);
}

/* What a published weight holds, read back off the device after the mover's copies. */
bool published_matches(Mover& m, const Program& prog, rad_weight w,
                       const std::vector<uint8_t>& img, RadStream s) {
    const WeightSlot* slot = m.residency()->lookup(w);
    if (!slot || !slot->ptr) return false;
    const WeightInfo& wi = prog.weights[w];
    std::vector<uint8_t> got((size_t)wi.stored_bytes);
    rad_stream_sync(m.transfer_stream());
    if (rad_memcpy_async(got.data(), slot->ptr, wi.stored_bytes, s) < 0) return false;
    rad_stream_sync(s);
    return std::memcmp(got.data(), img.data() + wi.file_offset, (size_t)wi.stored_bytes) == 0;
}
}  /* namespace */

TEST(the_file_stager_reads_each_routed_layer_from_the_container) {
    /* Three routed layers of four 20 MiB experts; 0 and 2 are popular and stay resident, so 1
     * and 3 of each layer are read from the container. This places resident bytes between
     * file-tier ones (skipped, not read through), makes a layer's file-tier part (40 MiB) wider
     * than a read window (32 MiB) so weights split across windows, and leaves each weight a
     * non-whole number of blocks so the memory layout differs from the container's. A hole between
     * layers is skipped too. */
    Fixture f;
    const int64_t gu = 14 * MiB + 1000, dn = 6 * MiB + 24;
    const int L = 3;
    build_moe_layers(f, L, 4, gu, dn);

    int64_t off = 4 * (int64_t)RAD_ALIGN_UNIT;
    int32_t prev = -1;
    for (rad_weight w = 1; (size_t)w < f.prog.weights.size(); ++w) {
        WeightInfo& wi = f.prog.weights[w];
        if (prev >= 0 && wi.decl.group.layer != prev) off += 2 * MiB;   /* the hole */
        prev = wi.decl.group.layer;
        wi.stored_bytes = wi.logical_bytes;
        wi.file_offset = (uint64_t)off;
        off = align_up(off + wi.stored_bytes, RAD_ALIGN_UNIT);
    }
    std::vector<uint8_t> img((size_t)off);
    for (int64_t i = 0; i < off; ++i) img[(size_t)i] = container_byte(i);
    const std::string path = "place_test_file_tier.rad";
    {
        FILE* fp = std::fopen(path.c_str(), "wb");
        CHECK(fp != nullptr);
        if (!fp) return;
        CHECK_EQ(std::fwrite(img.data(), 1, img.size(), fp), img.size());
        std::fclose(fp);
    }
    {
        DirectFile probe;
        if (probe.open(path.c_str(), RAD_DIO_READ) < 0) {
            std::printf("  SKIP the working directory has no O_DIRECT\n");
            std::remove(path.c_str());
            return;
        }
    }

    std::vector<RadFileProfile> prof;
    for (int l = 0; l < L; ++l)
        for (int e = 0; e < 4; ++e)
            prof.push_back(RadFileProfile{l, e, e == 0 ? 0.5f : e == 2 ? 0.4f : 0.01f, 0.0f});
    Config c = base_config();
    c.placement = "expert_tiered";
    c.deterministic = true;          /* no shadow slots, so residents are exactly the fill's */
    c.vram_weights_mib = 210;
    c.host_pool_mib = (4 * 21 * MiB + RAD_PLACE_FILE_WINDOWS * RAD_PLACE_FILE_WINDOW_BYTES) / MiB + 1;
    c.weights_disk_tier = true;
    PlannerInput in{&f.prog, &c, 0, 1, prof.data(), (int64_t)prof.size()};
    Plan p;
    CHECK_OK(Planner(in).plan(&p));
    std::vector<std::vector<rad_weight>> file_w((size_t)L), res_w((size_t)L);
    for (const MoveUnit& mu : p.units) {
        if (mu.expert < 0) continue;
        CHECK(mu.tier == ((mu.expert & 1) ? Tier::SSD : Tier::VRAM));
        for (rad_weight w : mu.weights)
            (mu.tier == Tier::SSD ? file_w : res_w)[(size_t)mu.layer].push_back(w);
    }

    Mover m;
    MoverConfig mc;
    mc.container_path = path;
    CHECK_OK(m.init(f.prog, p, mc));
    RadStream s = nullptr;
    CHECK_OK(rad_stream_create(&s, 0));
    CHECK_OK(m.bind_compute_stream(s));
    FileStager fs;
    CHECK_OK(fs.init(f.prog, p, &m));
    CHECK(fs.enabled());
    CHECK_EQ(fs.n_layers(), (size_t)L);

    /* Whether every file-tier weight of layer `l` is published with its container bytes (want),
     * or none is. */
    auto all = [&](int l, bool want) {
        int ok = 0;
        for (rad_weight w : file_w[(size_t)l]) {
            const WeightSlot* slot = m.residency()->lookup(w);
            const bool have = slot && slot->ptr;
            if (have != want) continue;
            if (!want || published_matches(m, f.prog, w, img, s)) ++ok;
        }
        return ok == (int)file_w[(size_t)l].size();
    };
    for (int l = 0; l < L; ++l) CHECK(all(l, false));

    /* Two passes; in the second the stager knows where a pass ends. Op 0 is the embedding and
     * layer l's expert ops are 2l+1 and 2l+2. */
    for (int pass = 0; pass < 2; ++pass) {
        RadBatch b{};
        CHECK(fs.begin_pass(&b));
        bool resync = false;
        CHECK_OK(fs.before_op(0, s, &resync));          /* the pass's first op reads layer 0 */
        CHECK(resync);
        CHECK(all(0, true) && all(1, false) && all(2, false));
        CHECK_OK(fs.before_op(1, s, &resync));          /* layer 0 starts; layer 1 read ahead */
        CHECK(all(0, true) && all(1, true) && all(2, false));
        CHECK_OK(fs.after_op(1, s));
        CHECK_OK(fs.before_op(1, s, &resync));          /* a second slice of layer 0: nothing */
        CHECK(!resync);
        CHECK_OK(fs.before_op(2, s, &resync));
        CHECK_OK(fs.before_op(3, s, &resync));          /* layer 1 starts: 0 released, 2 read */
        CHECK(all(0, false) && all(1, true) && all(2, true));
        CHECK_OK(fs.before_op(4, s, &resync));
        CHECK_OK(fs.before_op(5, s, &resync));          /* layer 2 starts: 1 released */
        CHECK(all(0, false) && all(1, false) && all(2, true));
        CHECK_OK(fs.before_op(6, s, &resync));
        CHECK_OK(fs.end_pass(s));
        for (int l = 0; l < L; ++l) CHECK(all(l, false));
    }
    /* A resident expert is never touched. */
    for (int l = 0; l < L; ++l)
        for (rad_weight w : res_w[(size_t)l]) {
            const WeightSlot* slot = m.residency()->lookup(w);
            CHECK(slot && slot->ptr);
        }
    /* Three layers per pass, every unit read once, and each contiguous run of a window one read:
     * a layer's 40 MiB takes two 32 MiB windows, the first holding expert 1 and the start of
     * expert 3 (two runs, the resident expert 2 between them skipped, not read), the second the
     * rest of expert 3. */
    CHECK_EQ(m.stats().ssd_layers, (uint64_t)(2 * L));
    CHECK_EQ(m.stats().ssd_units, (uint64_t)(2 * L * 2));
    CHECK_EQ(m.stats().ssd_preads, (uint64_t)(2 * L * 3));
    rad_stream_sync(m.transfer_stream());
    rad_stream_destroy(s);
    std::remove(path.c_str());
}

namespace {
/* A relayout a test can predict, standing in for a kernel's (RadRelayoutFn): the first plane's
 * bytes reversed and xored, then the second plane as it is. Refuses a size that is not the two
 * planes', so a weight handed the wrong geometry fails loudly. */
int reversing_relayout(const RadParam*, int, int, const RadEncoding*, const int*,
                       const RadTensor* planes, int n_planes, void* dst, int64_t dst_bytes) {
    if (n_planes != 2) return RAD_E_SHAPE;
    const int64_t n0 = planes[0].shape[0] * planes[0].shape[1];
    const int64_t n1 = planes[1].shape[0] * planes[1].shape[1];
    if (n0 + n1 != dst_bytes || !planes[0].data || !planes[1].data) return RAD_E_SHAPE;
    const uint8_t* a = (const uint8_t*)planes[0].data;
    uint8_t* d = (uint8_t*)dst;
    for (int64_t i = 0; i < n0; ++i) d[i] = (uint8_t)(a[n0 - 1 - i] ^ 0x5a);
    std::memcpy(d + n0, planes[1].data, (size_t)n1);
    return RAD_OK;
}
}  /* namespace */

/* A weight whose kernel stores a relayout of the container's planes goes to the file tier too
 * (weight_file_relaid, PlannerInput::file_relayout) and is relaid on every read, by the kernel's
 * own relayout, into the read windows the copy to the card reads. Three routed layers of four
 * experts, two planes per weight: the second starts 256 bytes past the first's end, so not on an
 * O_DIRECT block, and a layer's relaid bytes span two read windows. Experts 0 and 2 stay resident.
 * Refused without the planner's flag, and by a mover with no relayout. */
TEST(the_file_tier_relays_a_weight_its_kernel_relayouts_on_every_read) {
    Fixture f;
    const int64_t tail = 600;                         /* the second plane */
    const int64_t gu = 20 * MiB + 1000 + tail, dn = 1 * MiB + 24 + tail;
    const int L = 3;    /* with two, the two layer buffers cost what the file-tier experts would */
    build_moe_layers(f, L, 4, gu, dn);
    RadKernelInfo info = g_dev_info;
    info.relayout = reversing_relayout;
    KernelRow row;
    row.info = &info;
    for (OpInfo& o : f.prog.ops)
        if (o.op == "moe_gemm") o.bands[0].dom[RAD_DOMAIN_DEVICE].row = &row;

    int64_t off = 4 * (int64_t)RAD_ALIGN_UNIT;
    std::vector<rad_weight> relaid;
    for (rad_weight w = 1; (size_t)w < f.prog.weights.size(); ++w) {
        WeightInfo& wi = f.prog.weights[w];
        wi.stored_bytes = wi.logical_bytes;
        wi.file_offset = (uint64_t)off;
        if (wi.decl.group.expert < 0) {
            off = align_up(off + wi.stored_bytes, RAD_ALIGN_UNIT);
            continue;
        }
        const int64_t n0 = wi.logical_bytes - tail;
        wi.identity = false;
        wi.layout_tag = "test.reversed";
        wi.lay_op = wi.first_use_op;
        wi.lay_band = 0;
        wi.lay_dom = RAD_DOMAIN_DEVICE;
        wi.lay_operand = 0;
        wi.enc.n_planes = 2;
        wi.enc.plane[0].dtype = RAD_U8;
        wi.enc.plane[1].dtype = RAD_U8;
        wi.n_sel = 2;
        wi.sel[0] = 0;
        wi.sel[1] = 1;
        wi.sel_rows[0] = 1; wi.sel_cols[0] = n0;
        wi.sel_rows[1] = 1; wi.sel_cols[1] = tail;
        wi.file_plane_offset[0] = (uint64_t)off;
        wi.file_plane_offset[1] = (uint64_t)align_up(off + n0, 256);
        off = align_up((int64_t)wi.file_plane_offset[1] + tail, RAD_ALIGN_UNIT);
        relaid.push_back(w);
    }
    std::vector<uint8_t> img((size_t)off);
    for (int64_t i = 0; i < off; ++i) img[(size_t)i] = container_byte(i);
    const std::string path = "place_test_file_relay.rad";
    {
        FILE* fp = std::fopen(path.c_str(), "wb");
        CHECK(fp != nullptr);
        if (!fp) return;
        CHECK_EQ(std::fwrite(img.data(), 1, img.size(), fp), img.size());
        std::fclose(fp);
    }
    {
        DirectFile probe;
        if (probe.open(path.c_str(), RAD_DIO_READ) < 0) {
            std::printf("  SKIP the working directory has no O_DIRECT\n");
            std::remove(path.c_str());
            return;
        }
    }
    /* What the relayout makes of weight w's planes. */
    auto expect = [&](rad_weight w) {
        const WeightInfo& wi = f.prog.weights[w];
        const int64_t n0 = wi.sel_cols[0];
        std::vector<uint8_t> e((size_t)wi.stored_bytes);
        for (int64_t i = 0; i < n0; ++i)
            e[(size_t)i] = (uint8_t)(img[(size_t)(wi.file_plane_offset[0] + n0 - 1 - i)] ^ 0x5a);
        std::memcpy(e.data() + n0, img.data() + wi.file_plane_offset[1], (size_t)tail);
        return e;
    };

    std::vector<RadFileProfile> prof;
    for (int l = 0; l < L; ++l)
        for (int e = 0; e < 4; ++e)
            prof.push_back(RadFileProfile{l, e, e == 0 ? 0.5f : e == 2 ? 0.4f : 0.01f, 0.0f});
    Config c = base_config();
    c.placement = "expert_tiered";
    c.deterministic = true;
    c.vram_weights_mib = 222;
    c.host_pool_mib = (4 * 22 * MiB + RAD_PLACE_FILE_WINDOWS * RAD_PLACE_FILE_WINDOW_BYTES) / MiB + 1;
    c.weights_disk_tier = true;
    PlannerInput in{&f.prog, &c, 0, 1, prof.data(), (int64_t)prof.size()};
    {
        /* Without the flag a relaid unit fits in no tier, as before. */
        Plan refused;
        CHECK(Planner(in).plan(&refused) < 0);
    }
    in.file_relayout = true;
    Plan p;
    CHECK_OK(Planner(in).plan(&p));
    std::vector<std::vector<rad_weight>> file_w((size_t)L);
    for (const MoveUnit& mu : p.units) {
        if (mu.expert < 0) continue;
        CHECK(mu.tier == ((mu.expert & 1) ? Tier::SSD : Tier::VRAM));
        if (mu.tier == Tier::SSD)
            for (rad_weight w : mu.weights) file_w[(size_t)mu.layer].push_back(w);
    }

    MoverConfig mc;
    mc.container_path = path;
    {
        /* A mover with no relayout refuses the plan rather than copy the planes as they lie. */
        Mover bare;
        CHECK(bare.init(f.prog, p, mc) < 0);
    }
    Mover m;
    std::atomic<int> calls{ 0 };
    m.set_file_relayout([&](const WeightInfo& w, const RadTensor* planes, void* dst, int64_t n) {
        ++calls;
        const RadKernelInfo* ki = weight_layout_kernel(f.prog, w);
        if (!ki || !ki->relayout) return (int)RAD_E_STATE;
        return ki->relayout(nullptr, 0, w.lay_operand, &w.enc, w.sel, planes, w.n_sel, dst, n);
    });
    CHECK_OK(m.init(f.prog, p, mc));
    RadStream s = nullptr;
    CHECK_OK(rad_stream_create(&s, 0));
    CHECK_OK(m.bind_compute_stream(s));
    FileStager fs;
    CHECK_OK(fs.init(f.prog, p, &m));
    CHECK(fs.enabled());

    /* Every file-tier weight of layer l is published holding its relayout. */
    auto relaid_ok = [&](int l) {
        int ok = 0;
        for (rad_weight w : file_w[(size_t)l]) {
            const WeightSlot* slot = m.residency()->lookup(w);
            if (!slot || !slot->ptr) continue;
            const WeightInfo& wi = f.prog.weights[w];
            std::vector<uint8_t> got((size_t)wi.stored_bytes);
            rad_stream_sync(m.transfer_stream());
            if (rad_memcpy_async(got.data(), slot->ptr, wi.stored_bytes, s) < 0) continue;
            rad_stream_sync(s);
            if (got == expect(w)) ++ok;
        }
        return ok == (int)file_w[(size_t)l].size() && ok > 0;
    };
    for (int pass = 0; pass < 2; ++pass) {
        RadBatch b{};
        CHECK(fs.begin_pass(&b));
        bool resync = false;
        CHECK_OK(fs.before_op(0, s, &resync));          /* layer 0 */
        CHECK(relaid_ok(0));
        CHECK_OK(fs.before_op(1, s, &resync));          /* layer 1 read ahead */
        CHECK(relaid_ok(1));
        CHECK_OK(fs.before_op(2, s, &resync));
        CHECK_OK(fs.before_op(3, s, &resync));          /* layer 2 read ahead */
        CHECK(relaid_ok(1) && relaid_ok(2));
        CHECK_OK(fs.before_op(4, s, &resync));
        CHECK_OK(fs.before_op(5, s, &resync));
        CHECK(relaid_ok(2));
        CHECK_OK(fs.before_op(6, s, &resync));
        CHECK_OK(fs.end_pass(s));
    }
    /* Two passes over three layers of two file-tier experts of two weights, every one relaid. */
    CHECK_EQ(m.stats().file_relaid, (uint64_t)(2 * L * 2 * 2));
    CHECK_EQ((uint64_t)calls.load(), m.stats().file_relaid);
    CHECK_EQ(m.stats().ssd_units, (uint64_t)(2 * L * 2));
    rad_stream_sync(m.transfer_stream());
    rad_stream_destroy(s);
    std::remove(path.c_str());
}

TEST(the_file_stager_reads_only_the_experts_a_small_pass_routes_to) {
    /* The fixture above: three routed layers of four experts, 1 and 3 of each on the file tier.
     * A one-token pass at top_k 2 routes, per layer, to the counts below. Layer 0 routes to
     * expert 1 and is then sliced (its first op issued again), so the whole layer is read
     * behind it; layer 1 routes to 3; layer 2's counts do not add up to the pass, so it is read
     * whole. */
    Fixture f;
    const int64_t gu = 14 * MiB + 1000, dn = 6 * MiB + 24;
    const int L = 3;
    build_moe_layers(f, L, 4, gu, dn);

    int64_t off = 4 * (int64_t)RAD_ALIGN_UNIT;
    int32_t prev = -1;
    for (rad_weight w = 1; (size_t)w < f.prog.weights.size(); ++w) {
        WeightInfo& wi = f.prog.weights[w];
        if (prev >= 0 && wi.decl.group.layer != prev) off += 2 * MiB;
        prev = wi.decl.group.layer;
        wi.stored_bytes = wi.logical_bytes;
        wi.file_offset = (uint64_t)off;
        off = align_up(off + wi.stored_bytes, RAD_ALIGN_UNIT);
    }
    std::vector<uint8_t> img((size_t)off);
    for (int64_t i = 0; i < off; ++i) img[(size_t)i] = container_byte(i);
    const std::string path = "place_test_file_routed.rad";
    {
        FILE* fp = std::fopen(path.c_str(), "wb");
        CHECK(fp != nullptr);
        if (!fp) return;
        CHECK_EQ(std::fwrite(img.data(), 1, img.size(), fp), img.size());
        std::fclose(fp);
    }
    {
        DirectFile probe;
        if (probe.open(path.c_str(), RAD_DIO_READ) < 0) {
            std::printf("  SKIP the working directory has no O_DIRECT\n");
            std::remove(path.c_str());
            return;
        }
    }

    std::vector<RadFileProfile> prof;
    for (int l = 0; l < L; ++l)
        for (int e = 0; e < 4; ++e)
            prof.push_back(RadFileProfile{l, e, e == 0 ? 0.5f : e == 2 ? 0.4f : 0.01f, 0.0f});
    Config c = base_config();
    c.placement = "expert_tiered";
    c.deterministic = true;
    c.vram_weights_mib = 210;
    c.host_pool_mib = (4 * 21 * MiB + RAD_PLACE_FILE_WINDOWS * RAD_PLACE_FILE_WINDOW_BYTES) / MiB + 1;
    c.weights_disk_tier = true;
    PlannerInput in{&f.prog, &c, 0, 1, prof.data(), (int64_t)prof.size()};
    Plan p;
    CHECK_OK(Planner(in).plan(&p));
    /* file_w[l][e]: the file-tier weights of expert e of layer l. */
    std::vector<std::vector<std::vector<rad_weight>>> file_w(
        (size_t)L, std::vector<std::vector<rad_weight>>(4));
    for (const MoveUnit& mu : p.units)
        if (mu.expert >= 0 && mu.tier == Tier::SSD)
            for (rad_weight w : mu.weights) file_w[(size_t)mu.layer][(size_t)mu.expert].push_back(w);

    Mover m;
    MoverConfig mc;
    mc.container_path = path;
    CHECK_OK(m.init(f.prog, p, mc));
    RadStream s = nullptr;
    CHECK_OK(rad_stream_create(&s, 0));
    CHECK_OK(m.bind_compute_stream(s));

    int32_t counts[L][4] = { { 1, 1, 0, 0 }, { 0, 0, 1, 1 }, { 1, 0, 0, 0 } };
    FileStager fs;
    CHECK_OK(fs.init(f.prog, p, &m, nullptr, nullptr, [&](int32_t layer, int64_t* ne, int64_t* top_k) {
        *ne = 4;
        *top_k = 2;
        return (layer >= 0 && layer < L) ? (const int32_t*)counts[layer] : nullptr;
    }));
    CHECK(fs.enabled());
    CHECK(fs.routed_reads());

    /* 0: no address; 1: its container bytes; 2: an alias into the layer's file buffer. */
    auto state = [&](int l, int e) {
        int got = -1;
        for (rad_weight w : file_w[(size_t)l][(size_t)e]) {
            const WeightSlot* slot = m.residency()->lookup(w);
            int st = 0;
            if (slot && slot->ptr) {
                if (published_matches(m, f.prog, w, img, s)) st = 1;
                else if (slot->ptr == m.file_buffer(l & 1)) st = 2;
                else st = 3;
            }
            if (got >= 0 && st != got) return 4;          /* the unit's weights disagree */
            got = st;
        }
        return got;
    };

    RadBatch b{};
    b.n_tok = 1;
    CHECK(fs.begin_pass(&b));
    bool resync = false;
    CHECK_OK(fs.before_op(0, s, &resync));     /* no read before layer 0's router has run */
    CHECK(!resync);
    CHECK_EQ(state(0, 1), 0);
    CHECK_OK(fs.before_op(1, s, &resync));     /* layer 0: expert 1 read, 3 aliased, no read-ahead */
    CHECK(resync);
    CHECK_EQ(state(0, 1), 1);
    CHECK_EQ(state(0, 3), 2);
    CHECK_EQ(state(1, 1), 0);
    CHECK_OK(fs.after_op(1, s));
    CHECK_OK(fs.before_op(1, s, &resync));     /* a second slice: the whole layer */
    CHECK(resync);
    CHECK_EQ(state(0, 1), 1);
    CHECK_EQ(state(0, 3), 1);
    CHECK_OK(fs.before_op(2, s, &resync));
    CHECK_OK(fs.before_op(3, s, &resync));     /* layer 1: 0 released, expert 3 read, 1 aliased */
    CHECK_EQ(state(0, 1), 0);
    CHECK_EQ(state(0, 3), 0);
    CHECK_EQ(state(1, 3), 1);
    CHECK_EQ(state(1, 1), 2);
    CHECK_OK(fs.before_op(4, s, &resync));
    CHECK_OK(fs.before_op(5, s, &resync));     /* layer 2: counts sum to 1, not 2: read whole */
    CHECK_EQ(state(1, 1), 0);
    CHECK_EQ(state(2, 1), 1);
    CHECK_EQ(state(2, 3), 1);
    CHECK_OK(fs.before_op(6, s, &resync));
    CHECK_OK(fs.end_pass(s));
    for (int l = 0; l < L; ++l) {
        CHECK_EQ(state(l, 1), 0);
        CHECK_EQ(state(l, 3), 0);
    }
    /* Units read: 1 + 2 (layer 0, routed then whole) + 1 + 2; aliased: one each in layers 0, 1. */
    CHECK_EQ(m.stats().ssd_units, (uint64_t)6);
    CHECK_EQ(m.stats().file_aliased, (uint64_t)2);

    /* A prefill-sized pass keeps the whole-layer read and its read-ahead. */
    RadBatch big{};
    big.n_tok = FileStager::kRoutedRows + 1;
    CHECK(fs.begin_pass(&big));
    CHECK_OK(fs.before_op(0, s, &resync));
    CHECK(resync);
    CHECK_EQ(state(0, 1), 1);
    CHECK_EQ(state(0, 3), 1);
    CHECK_OK(fs.before_op(1, s, &resync));
    CHECK_EQ(state(1, 1), 1);
    CHECK_OK(fs.end_pass(s));

    rad_stream_sync(m.transfer_stream());
    rad_stream_destroy(s);
    std::remove(path.c_str());
}

TEST(mover_promotes_and_demotes_through_the_shadow_slots) {
    HeatFixture hf;
    HeatEngine h;
    HeatParams hp;
    CHECK_OK(h.init(hf.plan, hp));

    Mover m;
    MoverConfig mc;
    CHECK_OK(m.init(hf.f.prog, hf.plan, mc));

    RadStream compute = nullptr;
    CHECK_OK(rad_stream_create(&compute, 0));
    CHECK_OK(m.bind_compute_stream(compute));

    const int32_t hot = hf.unit_of_expert(4);
    const rad_weight hw = hf.plan.units[(size_t)hot].weights.front();
    void* before = m.residency()->lookup(hw)->ptr;
    const uint32_t gen_before = m.residency()->lookup(hw)->generation;

    /* Every pooled expert is wanted. Each iteration is a whole dispatch (retire, observe, decide,
     * issue), as a real caller runs it; slot supply only turns over if all four repeat.
     *
     * The pool free list starts empty, so the first round can only promote, into the shadow slot.
     * The pool slot that frees is available only after its copy lands and its quarantine clears,
     * many iterations later on a card. A loop that stops issuing early would never see a demotion
     * on a card, yet pass on the host backend where copies land within the dispatch; hence the
     * loop keeps issuing until both happen. */
    ComputeWork work;
    const int32_t ids[4] = {4, 5, 6, 7};
    const int32_t counts[4] = {10, 10, 10, 10};
    int d = 0;
    for (Deadline dl; !dl.passed(); ++d) {
        CHECK_OK(m.begin_dispatch(d, &h));
        work.run(compute);
        h.observe_ids(0, ids, counts, 4);
        std::vector<Swap> sw;
        CHECK_OK(h.step(&sw));
        for (const Swap& s : sw) m.issue(s, &h);
        if (m.stats().promotions >= 1 && m.stats().demotions >= 1) break;
    }
    /* Bounded, not exact: retire() publishes only transfers that have landed (it polls
     * rad_event_query), so the dispatch count depends on the link. Only convergence and both
     * halves of the swap are asserted. */

    CHECK(m.stats().promotions >= 1);
    CHECK(m.stats().demotions >= 1);
    CHECK(m.stats().h2d_bytes > 0);
    CHECK(m.stats().d2h_bytes > 0);
    /* The first round found no pool slot for its demotion, and the starvation counter records it. */
    CHECK(m.stats().starved_pool >= 1);

    const WeightSlot* after = m.residency()->lookup(hw);
    CHECK(after->ptr != before);
    CHECK(after->generation > gen_before);
    CHECK(h.tier(hot) == Tier::VRAM);

    /* Once the mover is quiescent nothing is left busy: every proposal was published or handed
     * back. A unit may still be busy when the loop above breaks (`busy` clears when the slot
     * leaves quarantine, several dispatches after the copy lands), so stop proposing and keep
     * retiring until it settles. */
    for (Deadline dl; any_busy(h, hf.plan) && !dl.passed(); ++d) {
        CHECK_OK(m.begin_dispatch(d, &h));
        work.run(compute);
    }
    for (int32_t u = 0; u < (int32_t)hf.plan.units.size(); ++u) CHECK(!h.busy(u));

    rad_stream_destroy(compute);
}

namespace {
int32_t resident_count(const HeatEngine& h, const Plan& p) {
    int32_t n = 0;
    for (int32_t u = 0; u < (int32_t)p.units.size(); ++u)
        if (h.tier(u) == Tier::VRAM) ++n;
    return n;
}
}  /* namespace */

/* ---- the loan. The KV cache lends the slab its unused VRAM, in place. A plain device allocation
 * stands in for the cache's carve: the mover is given the stretches and a loan in tokens, and
 * nothing about what the owner keeps there. */
namespace {
struct Loan {
    static constexpr int64_t kStrip = 4 * MiB;
    static constexpr int64_t kUnitBytes = 64 * 1024;   /* bytes of a stretch per owner unit */
    static constexpr int64_t kUnitTokens = 16;
    void* mem = nullptr;
    std::vector<LoanStrip> strips;
    Loan() {
        mem = rad_dev_alloc(2 * kStrip, RAD_MEM_DEVICE);
        for (int k = 0; k < 2; ++k) {
            LoanStrip st;
            st.top = (char*)mem + (k + 1) * kStrip;
            st.bytes = kStrip;
            st.unit_bytes = kUnitBytes;
            st.unit_tokens = kUnitTokens;
            strips.push_back(st);
        }
    }
    ~Loan() { if (mem) rad_dev_free(mem, RAD_MEM_DEVICE); }
    bool inside(const void* p) const {
        return (const char*)p >= (const char*)mem && (const char*)p < (const char*)mem + 2 * kStrip;
    }
    /* Everything the stretches can hold. */
    static constexpr int64_t kAll = kStrip / kUnitBytes * kUnitTokens;
};

struct LoanRig {
    HeatFixture hf;
    HeatEngine  h;
    Mover       m;
    Loan        loan;
    RadStream   compute = nullptr;
    ComputeWork work;
    int         d = 0;
    int32_t     planned = 0;
    /* None by default: on a slab this small the production count would leave most of the loan
     * empty. Tests of turnover set their own. */
    int         turnover = 0;

    explicit LoanRig(int n_expert = 8) : hf(n_expert) {}

    int build() {
        HeatParams hp;
        RAD_TRY(h.init(hf.plan, hp));
        MoverConfig mc;
        mc.turnover_slots = turnover;
        RAD_TRY(m.init(hf.f.prog, hf.plan, mc));
        RAD_TRY(rad_stream_create(&compute, 0));
        RAD_TRY(m.bind_compute_stream(compute));
        RAD_TRY(m.lend_from(Mover::kLendKV, loan.strips));
        planned = resident_count(h, hf.plan);
        return RAD_OK;
    }
    ~LoanRig() { if (compute) { rad_stream_sync(compute); rad_stream_destroy(compute); } }

    /* One dispatch as the engine runs it: retire, the step's work, then the loan's fill. */
    int dispatch() {
        RAD_TRY(m.begin_dispatch(d++, &h));
        work.run(compute);
        return m.balance(&h);
    }
    int32_t pooled() const {
        int32_t n = 0;
        for (int32_t u = 0; u < (int32_t)hf.plan.units.size(); ++u)
            if (hf.plan.units[(size_t)u].tier == Tier::HostPinned &&
                hf.plan.units[(size_t)u].movable) ++n;
        return n;
    }
    const void* addr(int32_t u) {
        return m.residency()->lookup(hf.plan.units[(size_t)u].weights.front())->ptr;
    }
};
}  /* namespace */

TEST(the_lent_slots_hold_nothing_until_a_loan_covers_them) {
    LoanRig r;
    CHECK_OK(r.build());
    CHECK(r.pooled() > 0);
    /* Capacity exists but nothing is offered: the stretches are the owner's until lent. */
    CHECK(r.m.flex_capacity() > 0);
    CHECK_EQ(r.m.flex_offered_bytes(), (int64_t)0);
    CHECK_EQ(r.m.flex_used_bytes(), (int64_t)0);
    CHECK_EQ(r.m.loan_held(Mover::kLendKV), (int64_t)0);
    /* A dispatch with nothing lent fills nothing. */
    CHECK_OK(r.dispatch());
    CHECK_EQ(r.m.stats().promotions, (uint64_t)0);
    CHECK_OK(r.m.set_loan(Mover::kLendKV, Loan::kAll));
    CHECK(r.m.flex_offered_bytes() > 0);
    CHECK_EQ(r.m.lend_from(Mover::kLendKV, r.loan.strips), RAD_E_STATE);   /* carved once */

    /* A mover with nothing to borrow has nothing to lend into; not an error. */
    HeatFixture hf;
    Mover none;
    MoverConfig nc;
    CHECK_OK(none.init(hf.f.prog, hf.plan, nc));
    CHECK_OK(none.lend_from(Mover::kLendKV, {}));
    CHECK_EQ(none.flex_capacity(), (int64_t)0);
    CHECK_OK(none.set_loan(Mover::kLendKV, Loan::kAll));
    CHECK_EQ(none.flex_offered_bytes(), (int64_t)0);
}

/* An empty lent slot wastes VRAM every step, so a loan is filled in the dispatch it arrives in:
 * every slot beyond the turnover slots at once, hottest units first. */
TEST(every_lent_slot_is_filled_in_one_dispatch) {
    LoanRig r;
    CHECK_OK(r.build());
    const int64_t slots = r.m.flex_capacity() / r.hf.plan.classes[0].stride;
    const int64_t expect = std::min<int64_t>(slots - r.m.turnover_slots(), r.pooled());
    CHECK(expect > 1);

    CHECK_OK(r.m.set_loan(Mover::kLendKV, Loan::kAll));
    CHECK_OK(r.dispatch());
    /* All issued in one call. */
    int64_t issued = 0;
    for (int32_t u = 0; u < (int32_t)r.hf.plan.units.size(); ++u) if (r.h.busy(u)) ++issued;
    CHECK_EQ(issued, expect);

    for (Deadline dl; resident_count(r.h, r.hf.plan) < r.planned + expect && !dl.passed();)
        CHECK_OK(r.dispatch());
    CHECK_EQ(resident_count(r.h, r.hf.plan), r.planned + (int32_t)expect);
    CHECK_EQ(r.m.flex_used_bytes(), expect * r.hf.plan.classes[0].stride);
    CHECK(r.m.loan_held(Mover::kLendKV) > 0);
    CHECK(r.m.loan_held(Mover::kLendKV) <= Loan::kAll);

    /* Every unit the fill promoted lives inside the stretches, at least one slot apart, never
     * overlapping and never in the alignment unit left empty at a stretch's top. */
    std::vector<const char*> at;
    for (int32_t u = 0; u < (int32_t)r.hf.plan.units.size(); ++u) {
        if (r.hf.plan.units[(size_t)u].tier != Tier::HostPinned) continue;
        const char* p = (const char*)r.addr(u);
        if (!r.loan.inside(p)) continue;
        at.push_back(p);
        for (const LoanStrip& st : r.loan.strips)
            if (p >= st.top - st.bytes && p < st.top)
                CHECK(p + r.hf.plan.classes[0].stride <= st.top - (int64_t)RAD_ALIGN_UNIT);
    }
    CHECK_EQ((int64_t)at.size(), expect);
    std::sort(at.begin(), at.end());
    for (size_t k = 1; k < at.size(); ++k)
        CHECK(at[k] - at[k - 1] >= r.hf.plan.classes[0].stride);
}

/* The turnover slots stay free, and a shrinking loan does not take them for good. The fill stops
 * that many slots short so the engine's swaps have somewhere to land. Free slots are at the loan's
 * top, which a reclaiming cache takes first; the mover then rebuilds them by dropping the coldest
 * lent units, which still have host copies, so nothing crosses the link. */
TEST(the_turnover_slots_stay_free_and_are_rebuilt_by_drops) {
    LoanRig r(24);
    r.turnover = 2;
    CHECK_OK(r.build());
    CHECK_EQ(r.m.turnover_slots(), (int64_t)2);
    const int64_t slots = r.m.flex_capacity() / r.hf.plan.classes[0].stride;
    CHECK(r.pooled() > slots);           /* more candidates than room: only the reserve stops the fill */

    CHECK_OK(r.m.set_loan(Mover::kLendKV, Loan::kAll));
    for (Deadline dl; (r.m.flex_used_bytes() == 0 || any_busy(r.h, r.hf.plan)) && !dl.passed();)
        CHECK_OK(r.dispatch());
    const int32_t full = resident_count(r.h, r.hf.plan);
    CHECK_EQ(full, r.planned + (int32_t)(slots - 2));
    CHECK_EQ(r.m.flex_used_bytes(), (slots - 2) * r.hf.plan.classes[0].stride);

    /* The fill packs from the bottom, so a loan cut to what is held takes exactly the free top
     * slots, and the next dispatch drops two lent units to free two more. */
    const int64_t held = r.m.loan_held(Mover::kLendKV);
    CHECK(held < Loan::kAll);
    const uint64_t d2h = r.m.stats().d2h_bytes;
    CHECK_OK(r.m.set_loan(Mover::kLendKV, held));
    for (Deadline dl; (r.m.stats().dropped < 2 || any_busy(r.h, r.hf.plan)) && !dl.passed();)
        CHECK_OK(r.dispatch());
    for (int k = 0; k < 4; ++k) CHECK_OK(r.dispatch());
    CHECK_EQ(r.m.stats().dropped, (uint64_t)2);
    CHECK_EQ(r.m.stats().d2h_bytes, d2h);                          /* nothing crossed the link */
    CHECK_EQ(resident_count(r.h, r.hf.plan), full - 2);
    CHECK_EQ(r.m.flex_used_bytes(), (slots - 4) * r.hf.plan.classes[0].stride);

    /* A staging step fills them (it swaps nothing, and a unit left out is one its stager must
     * copy). The next ordinary step drops the two coldest again. */
    CHECK_OK(r.m.begin_dispatch(r.d++, &r.h));
    r.work.run(r.compute);
    CHECK_OK(r.m.balance(&r.h, /*turnover=*/false));
    for (Deadline dl; any_busy(r.h, r.hf.plan) && !dl.passed();) {
        CHECK_OK(r.m.begin_dispatch(r.d++, &r.h));
        r.work.run(r.compute);
        CHECK_OK(r.m.balance(&r.h, false));
    }
    CHECK_EQ(resident_count(r.h, r.hf.plan), full);
    for (Deadline dl; (r.m.stats().dropped < 4 || any_busy(r.h, r.hf.plan)) && !dl.passed();)
        CHECK_OK(r.dispatch());
    CHECK_EQ(resident_count(r.h, r.hf.plan), full - 2);
    CHECK_EQ(r.m.stats().d2h_bytes, d2h);

    /* While held, an idle tick fills them as a staging step does; once released, the next drops
     * them. */
    r.m.hold_turnover(true);
    for (Deadline dl; (resident_count(r.h, r.hf.plan) < full || any_busy(r.h, r.hf.plan)) && !dl.passed();)
        CHECK_OK(r.m.idle_tick(&r.h));
    CHECK_EQ(resident_count(r.h, r.hf.plan), full);
    r.m.hold_turnover(false);
    for (Deadline dl; (r.m.stats().dropped < 6 || any_busy(r.h, r.hf.plan)) && !dl.passed();)
        CHECK_OK(r.m.idle_tick(&r.h));
    CHECK_EQ(resident_count(r.h, r.hf.plan), full - 2);
    CHECK_EQ(r.m.stats().d2h_bytes, d2h);
}

namespace {
/* Stamp every weight of every unit with its own bytes (check=false), or verify them where the
 * residency table now says they are (check=true), via device copies so it also runs on a card. */
uint8_t stamp_byte(rad_weight w, size_t i) {
    uint32_t v = (uint32_t)w * 0x9E3779B1u ^ (uint32_t)i * 0x85EBCA6Bu;
    v ^= v >> 15; v *= 0x2C1B3C6Du; v ^= v >> 12; v *= 0x297A2D39u; v ^= v >> 15;
    return (uint8_t)v;
}
int stamp_units(LoanRig& r, bool check, int* bad) {
    RadStream s = nullptr;
    RAD_TRY(rad_stream_create(&s, 0));
    std::vector<uint8_t> buf;
    for (const MoveUnit& mu : r.hf.plan.units)
        for (rad_weight w : mu.weights) {
            const size_t n = (size_t)r.hf.f.prog.weights[w].logical_bytes;
            void* at = r.m.residency()->lookup(w)->ptr;
            buf.resize(n);
            if (check) {
                RAD_TRY(rad_memcpy_async(buf.data(), at, (int64_t)n, s));
                RAD_TRY(rad_stream_sync(s));
                for (size_t i = 0; i < n; ++i)
                    if (buf[i] != stamp_byte(w, i)) { ++*bad; break; }
            } else {
                for (size_t i = 0; i < n; ++i) buf[i] = stamp_byte(w, i);
                RAD_TRY(rad_memcpy_async(at, buf.data(), (int64_t)n, s));
                RAD_TRY(rad_stream_sync(s));
            }
        }
    rad_stream_destroy(s);
    return RAD_OK;
}
/* A full loan, every unit stamped, then one swap whose budgeted victim finds the pool full: the
 * promotion lands in a working slot and the victim takes the host copy of a lent unit. */
int donate_once(LoanRig& r, bool held = false) {
    RAD_TRY(stamp_units(r, false, nullptr));
    RAD_TRY(r.m.set_loan(Mover::kLendKV, Loan::kAll));
    for (Deadline dl; (r.m.flex_used_bytes() == 0 || any_busy(r.h, r.hf.plan)) && !dl.passed();)
        RAD_TRY(r.dispatch());
    int32_t hot = -1, cold = -1;
    for (int32_t u = 0; u < (int32_t)r.hf.plan.units.size(); ++u) {
        if (!r.hf.plan.units[(size_t)u].movable) continue;
        if (hot < 0 && r.h.tier(u) != Tier::VRAM) hot = u;
        if (cold < 0 && r.h.tier(u) == Tier::VRAM && !r.loan.inside(r.addr(u))) cold = u;
    }
    if (hot < 0 || cold < 0) return RAD_E_STATE;
    Swap sw;
    sw.promote = hot;
    sw.demote = cold;
    r.h.set_busy(hot, true);
    r.h.set_busy(cold, true);
    r.m.hold_turnover(held);
    RAD_TRY(r.m.issue(sw, &r.h));
    for (Deadline dl; any_busy(r.h, r.hf.plan) && !dl.passed();) RAD_TRY(r.dispatch());
    if (held) return r.h.tier(hot) == Tier::VRAM && r.h.tier(cold) == Tier::VRAM ? RAD_OK : RAD_E_STATE;
    return r.h.tier(hot) == Tier::VRAM && r.h.tier(cold) != Tier::VRAM ? RAD_OK : RAD_E_STATE;
}
}  /* namespace */

/* A budgeted victim with no free pool slot takes a lent unit's host copy (Mover::donor_slot); the
 * pool is sized to its units, so otherwise the budgeted range turns over only as fast as working
 * slots free up. A recall then moves the donor (it has no host copy to drop to), and every unit's
 * bytes stay where its table entry says. */
TEST(a_budgeted_victim_takes_a_lent_units_host_copy_and_a_recall_moves_the_donor) {
    LoanRig r(24);
    r.turnover = 2;
    CHECK_OK(r.build());
    CHECK_OK(donate_once(r));
    CHECK_EQ(r.m.stats().donated, (uint64_t)1);
    CHECK_EQ(r.m.stats().starved_pool, (uint64_t)0);
    int bad = 0;
    CHECK_OK(stamp_units(r, true, &bad));
    CHECK_EQ(bad, 0);

    CHECK_OK(r.m.reclaim(Mover::kLendKV, 0, &r.h));
    CHECK_EQ(r.m.stats().relocated, (uint64_t)1);
    CHECK_EQ(r.m.loan_held(Mover::kLendKV), (int64_t)0);
    for (int32_t u = 0; u < (int32_t)r.hf.plan.units.size(); ++u)
        CHECK(!r.loan.inside(r.addr(u)));
    /* The donor stays in VRAM: a working slot was free for it. */
    CHECK_EQ(resident_count(r.h, r.hf.plan), r.planned + 1);
    CHECK_OK(stamp_units(r, true, &bad));
    CHECK_EQ(bad, 0);
}

/* Holds the compute stream behind a long copy on another stream, so anything recorded on it
 * afterwards (a quarantine barrier) stays pending until the copy finishes. */
struct ComputeHold {
    static constexpr int64_t kBytes = 64ll << 20;
    void* a = nullptr;
    void* b = nullptr;
    RadStream s = nullptr;
    RadEvent e = nullptr;

    int hold(RadStream compute, int rounds) {
        a = rad_dev_alloc(kBytes, RAD_MEM_DEVICE);
        b = rad_dev_alloc(kBytes, RAD_MEM_DEVICE);
        if (!a || !b) return RAD_E_NOMEM;
        RAD_TRY(rad_stream_create(&s, 0));
        RAD_TRY(rad_event_create(&e));
        for (int i = 0; i < rounds; ++i) RAD_TRY(rad_memcpy_async(a, b, kBytes, s));
        RAD_TRY(rad_event_record(e, s));
        return rad_event_wait(compute, e);
    }
    ~ComputeHold() {
        if (s) { rad_stream_sync(s); rad_stream_destroy(s); }
        if (e) rad_event_destroy(e);
        if (a) rad_dev_free(a, RAD_MEM_DEVICE);
        if (b) rad_dev_free(b, RAD_MEM_DEVICE);
    }
};

/* A recall right after a donation, with the card behind: the swap's copies have landed but the
 * slots they freed (the victim's budgeted slot, the promoted unit's pool slot) are still in
 * quarantine behind the compute stream. The donor is owed those slots (donor_slot), so the recall
 * drains the stream before looking; otherwise it finds none, stops, and strands the slots it
 * already reclaimed from other units. */
TEST(a_recall_moves_a_donor_whose_slots_are_still_in_quarantine) {
    LoanRig r(24);
    r.turnover = 2;
    CHECK_OK(r.build());
    CHECK_OK(stamp_units(r, false, nullptr));
    CHECK_OK(r.m.set_loan(Mover::kLendKV, Loan::kAll));
    for (Deadline dl; (r.m.flex_used_bytes() == 0 || any_busy(r.h, r.hf.plan)) && !dl.passed();)
        CHECK_OK(r.dispatch());
    int32_t hot = -1, cold = -1;
    for (int32_t u = 0; u < (int32_t)r.hf.plan.units.size(); ++u) {
        if (!r.hf.plan.units[(size_t)u].movable) continue;
        if (hot < 0 && r.h.tier(u) != Tier::VRAM) hot = u;
        if (cold < 0 && r.h.tier(u) == Tier::VRAM && !r.loan.inside(r.addr(u))) cold = u;
    }
    CHECK(hot >= 0 && cold >= 0);
    if (hot < 0 || cold < 0) return;

    CHECK_OK(r.m.begin_dispatch(r.d++, &r.h));
    r.work.run(r.compute);
    Swap sw;
    sw.promote = hot;
    sw.demote = cold;
    r.h.set_busy(hot, true);
    r.h.set_busy(cold, true);
    CHECK_OK(r.m.issue(sw, &r.h));
    CHECK_EQ(r.m.stats().donated, (uint64_t)1);
    ComputeHold hold;
    CHECK_OK(hold.hold(r.compute, 48));
    CHECK_EQ(rad_event_query(hold.e), 0);           /* the card is still behind at recall time */

    CHECK_OK(r.m.reclaim(Mover::kLendKV, 0, &r.h));
    CHECK_EQ(r.m.stats().relocated, (uint64_t)1);
    CHECK_EQ(r.m.stats().stranded, (uint64_t)0);
    CHECK_EQ(r.m.loan_held(Mover::kLendKV), (int64_t)0);
    for (int32_t u = 0; u < (int32_t)r.hf.plan.units.size(); ++u)
        CHECK(!r.loan.inside(r.addr(u)));
    int bad = 0;
    CHECK_OK(stamp_units(r, true, &bad));
    CHECK_EQ(bad, 0);
}

/* No donation while the turnover slots are held (through a prompt and the steps after it): the
 * demotion is starved and the budgeted victim stays. */
TEST(a_held_turnover_takes_no_host_copy) {
    LoanRig r(24);
    r.turnover = 2;
    CHECK_OK(r.build());
    CHECK_OK(donate_once(r, /*held=*/true));
    CHECK_EQ(r.m.stats().donated, (uint64_t)0);
    CHECK(r.m.stats().starved_pool >= 1);
    int bad = 0;
    CHECK_OK(stamp_units(r, true, &bad));
    CHECK_EQ(bad, 0);
}

/* A demotion of a unit with no slab slot is refused before it looks for a pool slot: with the
 * pool full that search takes a lent unit's host copy, which the refusal could not give back. */
TEST(a_demotion_with_no_slab_slot_takes_no_host_copy) {
    LoanRig r(24);
    r.turnover = 2;
    CHECK_OK(r.build());
    CHECK_OK(r.m.set_loan(Mover::kLendKV, Loan::kAll));
    for (Deadline dl; (r.m.flex_used_bytes() == 0 || any_busy(r.h, r.hf.plan)) && !dl.passed();)
        CHECK_OK(r.dispatch());
    int32_t pooled = -1;
    for (int32_t u = 0; u < (int32_t)r.hf.plan.units.size() && pooled < 0; ++u)
        if (r.hf.plan.units[(size_t)u].movable && r.h.tier(u) != Tier::VRAM) pooled = u;
    CHECK(pooled >= 0);
    if (pooled < 0) return;

    CHECK_OK(r.m.begin_dispatch(r.d++, &r.h));
    r.work.run(r.compute);
    Swap sw;
    sw.demote = pooled;
    r.h.set_busy(pooled, true);
    CHECK_EQ(r.m.issue(sw, &r.h), RAD_E_FULL);
    CHECK_EQ(r.m.stats().donated, (uint64_t)0);
    CHECK(!r.h.busy(pooled));
}

/* A loan shrinking under a donor moves it out before the slot is given back (balance's sweep):
 * an on-card copy into the budgeted slot its victim left, and the shrink completes once it lands.
 * The working slot it took is rebuilt by demoting the coldest budgeted unit, and every unit's
 * bytes stay where its table entry says. */
TEST(a_shrinking_loan_moves_a_donor_out_before_its_slot_is_given_back) {
    LoanRig r(24);
    r.turnover = 2;
    CHECK_OK(r.build());
    CHECK_OK(donate_once(r));
    CHECK_EQ(r.m.stats().donated, (uint64_t)1);
    const uint64_t d2h = r.m.stats().d2h_bytes;

    CHECK_OK(r.m.set_loan(Mover::kLendKV, 0));
    for (Deadline dl; (r.m.loan_held(Mover::kLendKV) > 0 || any_busy(r.h, r.hf.plan)) && !dl.passed();)
        CHECK_OK(r.dispatch());
    CHECK_EQ(r.m.loan_held(Mover::kLendKV), (int64_t)0);
    CHECK_EQ(r.m.flex_used_bytes(), (int64_t)0);
    CHECK_EQ(r.m.stats().relocated, (uint64_t)1);
    CHECK_EQ(r.m.stats().d2h_bytes - d2h, (uint64_t)r.hf.plan.classes[0].stride);
    CHECK_EQ(resident_count(r.h, r.hf.plan), r.planned);
    for (int32_t u = 0; u < (int32_t)r.hf.plan.units.size(); ++u)
        CHECK(!r.loan.inside(r.addr(u)));
    int bad = 0;
    CHECK_OK(stamp_units(r, true, &bad));
    CHECK_EQ(bad, 0);
}

/* A unit in a lent slot keeps its host copy, so giving the slot back changes its address without
 * a copy across the link. */
TEST(a_shrinking_loan_drops_units_without_copying_them) {
    LoanRig r;
    CHECK_OK(r.build());
    CHECK_OK(r.m.set_loan(Mover::kLendKV, Loan::kAll));
    for (Deadline dl; (r.m.flex_used_bytes() == 0 || any_busy(r.h, r.hf.plan)) && !dl.passed();)
        CHECK_OK(r.dispatch());
    const int32_t full = resident_count(r.h, r.hf.plan);
    CHECK(full > r.planned);
    const uint64_t d2h = r.m.stats().d2h_bytes;

    CHECK_OK(r.m.set_loan(Mover::kLendKV, 0));
    for (Deadline dl; (r.m.loan_held(Mover::kLendKV) > 0 || any_busy(r.h, r.hf.plan)) && !dl.passed();)
        CHECK_OK(r.dispatch());
    CHECK_EQ(r.m.loan_held(Mover::kLendKV), (int64_t)0);
    CHECK_EQ(r.m.flex_used_bytes(), (int64_t)0);
    CHECK_EQ(resident_count(r.h, r.hf.plan), r.planned);
    CHECK_EQ(r.m.stats().d2h_bytes, d2h);                          /* nothing crossed the link */
    CHECK_EQ(r.m.stats().dropped, (uint64_t)(full - r.planned));
    for (int32_t u = 0; u < (int32_t)r.hf.plan.units.size(); ++u)
        CHECK(!r.loan.inside(r.addr(u)));
}

/* reclaim() returns the memory before it returns: every unit past the loan is back on its host
 * copy, and nothing that could still read the slot is in flight. */
TEST(a_recall_takes_the_slots_back_before_it_returns) {
    LoanRig r;
    CHECK_OK(r.build());
    CHECK_OK(r.m.set_loan(Mover::kLendKV, Loan::kAll));
    CHECK_OK(r.dispatch());                 /* a fill in flight, the hard case */
    CHECK(any_busy(r.h, r.hf.plan));

    CHECK_OK(r.m.reclaim(Mover::kLendKV, 0, &r.h));
    CHECK_EQ(r.m.loan_held(Mover::kLendKV), (int64_t)0);
    CHECK_EQ(r.m.flex_used_bytes(), (int64_t)0);
    CHECK_EQ(resident_count(r.h, r.hf.plan), r.planned);
    CHECK(!any_busy(r.h, r.hf.plan));
    for (int32_t u = 0; u < (int32_t)r.hf.plan.units.size(); ++u)
        CHECK(!r.loan.inside(r.addr(u)));

    /* The slots are usable again once the loan returns. */
    CHECK_OK(r.m.set_loan(Mover::kLendKV, Loan::kAll));
    for (Deadline dl; (r.m.flex_used_bytes() == 0 || any_busy(r.h, r.hf.plan)) && !dl.passed();)
        CHECK_OK(r.dispatch());
    CHECK(resident_count(r.h, r.hf.plan) > r.planned);
}

/* An idle server still fills what it is lent, via its own tick, so a loan made while idle is not
 * left empty until the next request pays for filling it. */
TEST(an_idle_mover_fills_what_it_is_lent) {
    LoanRig r;
    CHECK_OK(r.build());
    CHECK_OK(r.m.set_loan(Mover::kLendKV, Loan::kAll));
    for (Deadline dl; (r.m.flex_used_bytes() == 0 || any_busy(r.h, r.hf.plan)) && !dl.passed();)
        CHECK_OK(r.m.idle_tick(&r.h));
    CHECK(resident_count(r.h, r.hf.plan) > r.planned);
    CHECK(!any_busy(r.h, r.hf.plan));
}

/* Two lenders move independently. The arena reclaims its memory for every prefill while the KV
 * cache keeps lending, so recalling one must leave the other's slots exactly as they were. */
TEST(a_second_lender_is_recalled_without_touching_the_first) {
    LoanRig r;
    CHECK_OK(r.build());
    Loan arena;
    /* The arena lends in bytes: one unit is one byte. */
    for (LoanStrip& st : arena.strips) { st.unit_bytes = 1; st.unit_tokens = 1; }
    CHECK_OK(r.m.lend_from(Mover::kLendArena, arena.strips));
    CHECK_EQ(r.m.lend_from(Mover::kLendArena, arena.strips), RAD_E_STATE);
    CHECK_EQ(r.m.loan_ceiling(Mover::kLendArena) > 0, true);

    /* Only part of the cache is lent, so the pooled units overflow its slots and the fill (lowest
     * slot first, the cache's segment before the arena's) reaches both. */
    const int64_t kv_part = Loan::kAll / 3;
    CHECK_OK(r.m.set_loan(Mover::kLendKV, kv_part));
    CHECK_OK(r.m.set_loan(Mover::kLendArena, r.m.loan_ceiling(Mover::kLendArena)));
    for (Deadline dl; (r.m.flex_used_bytes() == 0 || any_busy(r.h, r.hf.plan)) && !dl.passed();)
        CHECK_OK(r.dispatch());
    int32_t in_kv = 0, in_arena = 0;
    for (int32_t u = 0; u < (int32_t)r.hf.plan.units.size(); ++u) {
        in_kv += r.loan.inside(r.addr(u));
        in_arena += arena.inside(r.addr(u));
    }
    CHECK(in_kv > 0);
    CHECK(in_arena > 0);
    const int64_t kv_held = r.m.loan_held(Mover::kLendKV);

    CHECK_OK(r.m.reclaim(Mover::kLendArena, 0, &r.h));
    CHECK_EQ(r.m.loan_held(Mover::kLendArena), (int64_t)0);
    CHECK_EQ(r.m.loan_held(Mover::kLendKV), kv_held);
    int32_t kv_after = 0;
    for (int32_t u = 0; u < (int32_t)r.hf.plan.units.size(); ++u) {
        CHECK(!arena.inside(r.addr(u)));
        kv_after += r.loan.inside(r.addr(u));
    }
    CHECK_EQ(kv_after, in_kv);
    CHECK_EQ(r.m.loan(Mover::kLendKV), kv_part);

    /* Re-lent: the arena's slots fill again while the cache's stay put. */
    CHECK_OK(r.m.set_loan(Mover::kLendArena, r.m.loan_ceiling(Mover::kLendArena)));
    for (Deadline dl; !dl.passed();) {
        CHECK_OK(r.dispatch());
        int32_t n = 0;
        for (int32_t u = 0; u < (int32_t)r.hf.plan.units.size(); ++u) n += arena.inside(r.addr(u));
        if (n > 0 && !any_busy(r.h, r.hf.plan)) break;
    }
    int32_t again = 0;
    for (int32_t u = 0; u < (int32_t)r.hf.plan.units.size(); ++u) again += arena.inside(r.addr(u));
    CHECK(again > 0);
}

TEST(slot_pool_segments_have_their_own_targets) {
    SlotPool s;
    s.init(2, 1);
    s.add_free(0);
    s.add_free(1);
    const int a = s.add_segment(3);
    const int b = s.add_segment(3);
    CHECK_EQ(s.seg_lo(a), 2);
    CHECK_EQ(s.seg_lo(b), 5);
    CHECK_EQ(s.n_free(), 2);                       /* nothing lent yet */
    s.set_target(b, s.seg_lo(b) + 3);
    CHECK_EQ(s.n_free(), 5);
    CHECK(!s.covered(2));
    CHECK(s.covered(5));
    /* Lowest covered slot first, skipping the segment that covers nothing. */
    CHECK_EQ(s.take(), 0);
    CHECK_EQ(s.take(), 1);
    CHECK_EQ(s.take(), 5);
    s.set_target(a, s.seg_lo(a) + 2);
    CHECK_EQ(s.take(), 2);
    /* Shrinking one segment leaves the other alone; the mark stops at what is held. */
    s.set_target(b, s.seg_lo(b));
    CHECK_EQ(s.top_held(b), 5);
    CHECK(s.covered(3));
    CHECK(!s.covered(6));
    s.add_free(5);
    CHECK_EQ(s.top_held(b), -1);
    CHECK_EQ(s.take(), 3);
    CHECK_EQ(s.take(), -1);
}

TEST(mover_prefetches_staged_units_against_the_declared_order) {
    Fixture f;
    build_dense(f, 8, 1 * MiB, /*host_kernel=*/false);
    Config c = base_config();
    c.placement = "layer_offload";
    c.vram_weights_mib = 5;
    c.host_pool_mib = 64;
    PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0};
    Plan p;
    CHECK_OK(Planner(in).plan(&p));
    CHECK_EQ((int)p.prefetch.size(), 7);

    Mover m;
    MoverConfig mc;
    CHECK_OK(m.init(f.prog, p, mc));
    RadStream compute = nullptr;
    CHECK_OK(rad_stream_create(&compute, 0));
    CHECK_OK(m.bind_compute_stream(compute));

    /* Each staged unit's pool address before staging. Staging republishes it into a slab slot,
     * so a changed pointer shows a unit landed without exposing the mover's slot table. */
    std::vector<void*> pooled;
    for (const Plan::Prefetch& pf : p.prefetch)
        pooled.push_back(
            m.residency()->lookup(p.units[(size_t)pf.unit].weights.front())->ptr);

    CHECK_OK(m.begin_dispatch(0, nullptr));
    /* At op 0 nothing is due, but the mover starts transfers until it runs out of slots. The ring
     * is two deep, so it gets exactly two ahead and stops; it does not skip the unit it cannot
     * serve, since the schedule is in deadline order. This is pure slot bookkeeping, identical on
     * every backend. */
    CHECK_OK(m.prefetch_to_op(0));
    CHECK_EQ(m.stats().prefetched, (uint64_t)RAD_PLACE_STAGE_RING);
    CHECK(m.stats().starved_slab >= 1);
    for (int i = 0; i < RAD_PLACE_STAGE_RING; ++i) {
        const rad_weight w = p.units[(size_t)p.prefetch[(size_t)i].unit].weights.front();
        CHECK(m.residency()->lookup(w)->ptr != nullptr);
        /* The residency table carries the mover's event, so rad_issue waits on the compute stream,
         * not the host. */
        CHECK(m.residency()->lookup(w)->ready != nullptr);
    }

    /* Walk the ops as a step does and check exact prefetch: each op's bytes land through a
     * two-slot ring. How many dispatches a copy takes depends on the link, so no dispatch index is
     * asserted. */
    ComputeWork work;
    int dispatch = 1;
    for (size_t i = 0; i < p.prefetch.size(); ++i) {
        const rad_weight w = p.units[(size_t)p.prefetch[i].unit].weights.front();
        bool landed = false;
        for (Deadline dl; !landed && !dl.passed();) {
            CHECK_OK(m.begin_dispatch(dispatch++, nullptr));
            work.run(compute);
            CHECK_OK(m.prefetch_to_op(p.prefetch[i].at_op));
            const WeightSlot* s = m.residency()->lookup(w);
            /* In a slab slot (pointer moved off the pool) and settled (copy landed). */
            landed = s->ptr != pooled[i] && s->ready == nullptr;
        }
        CHECK(landed);
    }

    /* Every scheduled unit was staged exactly once: the cursor never rewinds within a step or
     * skips a unit, so ring turnover never re-fetches. */
    CHECK_EQ(m.stats().prefetched, (uint64_t)p.prefetch.size());
    /* Through a two-slot ring it must have starved and recycled; otherwise a mover allocating a
     * slot per unit would also pass. */
    CHECK(m.stats().starved_slab >= 1);
    CHECK((int64_t)p.prefetch.size() > RAD_PLACE_STAGE_RING);

    rad_stream_destroy(compute);
}

/* A routed layer's pooled units staged for one pass (Mover::stage_units). Each copied unit's
 * weights are published at the copy, settled (the caller orders readers with one wait), holding the
 * pool slot's bytes. A unit that does not fit keeps its pool address and is counted; adjacent pool
 * slots share a copy; resident units are never copied; unstaging republishes the pool addresses. */
TEST(a_staged_layer_is_published_at_its_copies_and_unstaged_at_its_pool_slots) {
    HeatFixture hf;
    Mover m;
    MoverConfig mc;
    CHECK_OK(m.init(hf.f.prog, hf.plan, mc));
    RadStream compute = nullptr;
    CHECK_OK(rad_stream_create(&compute, 0));
    CHECK_OK(m.bind_compute_stream(compute));

    std::vector<int32_t> units, pooled;
    for (int32_t u = 0; u < (int32_t)hf.plan.units.size(); ++u) {
        if (hf.plan.units[(size_t)u].expert < 0) continue;
        units.push_back(u);
        if (hf.plan.units[(size_t)u].tier == Tier::HostPinned) pooled.push_back(u);
    }
    REQUIRE(pooled.size() >= 2);
    REQUIRE(pooled.size() < units.size());           /* some are resident */

    std::vector<void*> before(hf.f.prog.weights.size(), nullptr);
    for (rad_weight w = 1; (size_t)w < hf.f.prog.weights.size(); ++w)
        before[w] = m.residency()->lookup(w)->ptr;

    /* Room for every pooled unit but one. */
    const int64_t each = align_up(hf.plan.units[(size_t)pooled[0]].bytes, RAD_ALIGN_UNIT);
    const int64_t cap = each * (int64_t)(pooled.size() - 1);
    std::vector<char> buf((size_t)cap);
    RadEvent done = nullptr;
    CHECK_OK(rad_event_create_local(&done));

    std::vector<int32_t> staged;
    CHECK_OK(m.stage_units(units, buf.data(), cap, nullptr, done, &staged));
    CHECK_EQ(staged.size(), pooled.size() - 1);
    CHECK_EQ(m.stats().stage_overflow, (uint64_t)1);
    CHECK_EQ(m.stats().staged_units, (uint64_t)staged.size());
    CHECK_EQ(m.stats().stage_layers, (uint64_t)1);

    /* Adjacent pool slots go as one copy, so copies equal the runs of staged pool addresses,
     * fewer than the units since a fresh pool hands out slots in order. */
    uint64_t runs = 0;
    const char* run_end = nullptr;
    for (int32_t u : staged) {
        const MoveUnit& mu = hf.plan.units[(size_t)u];
        const char* slot = (const char*)before[mu.weights[0]] - mu.offset[0];
        if (slot != run_end) ++runs;
        run_end = slot + each;
    }
    CHECK_EQ(m.stats().stage_copies, runs);
    CHECK(runs < (uint64_t)staged.size());

    /* The copies run on the mover's stream (a worker thread on the host backend). */
    CHECK_OK(rad_stream_sync(m.transfer_stream()));
    std::vector<uint8_t> is_staged(hf.plan.units.size(), 0);
    for (size_t k = 0; k < staged.size(); ++k) {
        const int32_t u = staged[k];
        is_staged[(size_t)u] = 1;
        CHECK(hf.plan.units[(size_t)u].tier == Tier::HostPinned);
        const MoveUnit& mu = hf.plan.units[(size_t)u];
        for (size_t i = 0; i < mu.weights.size(); ++i) {
            const WeightSlot* sl = m.residency()->lookup(mu.weights[i]);
            CHECK(sl->ptr == buf.data() + (int64_t)k * each + mu.offset[i]);
            CHECK(sl->ready == nullptr);
            CHECK(std::memcmp(sl->ptr, before[mu.weights[i]],
                              (size_t)hf.f.prog.weights[mu.weights[i]].logical_bytes) == 0);
        }
    }
    for (int32_t u : units) {
        if (is_staged[(size_t)u]) continue;
        for (rad_weight w : hf.plan.units[(size_t)u].weights)
            CHECK(m.residency()->lookup(w)->ptr == before[w]);
    }

    m.unstage_units(staged);
    for (int32_t u : units)
        for (rad_weight w : hf.plan.units[(size_t)u].weights)
            CHECK(m.residency()->lookup(w)->ptr == before[w]);
    CHECK_OK(rad_stream_sync(m.transfer_stream()));
    rad_event_destroy(done);
    rad_stream_destroy(compute);
}

/* The prefill stager's order over a pass of two routed layers: at the pass's first op the first
 * layer goes out, and the second too if the ring has room; each layer returns to its pool slots
 * when the next layer starts; nothing is staged in a pass that was not armed.
 *
 * A layer is released when the next starts, not at its own last op: a step longer than the MoE
 * block's row pass issues a layer's ops once per pass, so the last op runs again and would read
 * copies the mover is already overwriting. */
TEST(the_stager_stages_ahead_and_releases_each_layer_when_the_next_starts) {
    Fixture f;
    const rad_weight emb = f.weight("token_embd.weight", 1 * MiB, RAD_ACCESS_VOCAB, -1, -1);
    f.op("embed", {emb}, false);
    std::vector<rad_weight> l0, l1;
    for (int l = 0; l < 2; ++l)
        for (int e = 0; e < 8; ++e) {
            const rad_weight w = f.weight("blk." + std::to_string(l) + ".ffn_gate_exps." +
                                          std::to_string(e) + ".weight", 1 * MiB,
                                          RAD_ACCESS_CONDITIONAL, l, e);
            (l ? l1 : l0).push_back(w);
        }
    f.op("attn0", {}, false);
    f.op("moe0", l0, false);
    f.op("attn1", {}, false);
    f.op("moe1", l1, false);
    Config cfg = base_config();
    cfg.placement = "expert_tiered";
    cfg.vram_weights_mib = 12;
    cfg.host_pool_mib = 64;
    Plan plan;
    PlannerInput in{&f.prog, &cfg, 0, 1, nullptr, 0};
    CHECK_OK(Planner(in).plan(&plan));
    REQUIRE(plan.dynamic);

    Mover m;
    MoverConfig mc;
    CHECK_OK(m.init(f.prog, plan, mc));
    RadStream compute = nullptr;
    CHECK_OK(rad_stream_create(&compute, 0));
    CHECK_OK(m.bind_compute_stream(compute));

    auto pooled_of = [&](const std::vector<rad_weight>& ws) {
        std::vector<rad_weight> p;
        for (rad_weight w : ws)
            if (plan.weight[w].tier == Tier::HostPinned) p.push_back(w);
        return p;
    };
    const std::vector<rad_weight> p0 = pooled_of(l0), p1 = pooled_of(l1);
    REQUIRE(!p0.empty() && !p1.empty());
    std::vector<void*> home(f.prog.weights.size(), nullptr);
    for (rad_weight w = 1; (size_t)w < f.prog.weights.size(); ++w)
        home[w] = m.residency()->lookup(w)->ptr;

    const int64_t each = align_up(1 * MiB, RAD_ALIGN_UNIT);
    const int64_t half = each * 8;
    /* The ring's region and, above it, its other place (the span's shared level). */
    std::vector<char> region((size_t)(4 * half));
    char* const low = region.data() + 2 * half;
    auto in_ring = [&](const std::vector<rad_weight>& ws) {
        for (rad_weight w : ws) {
            const char* p = (const char*)m.residency()->lookup(w)->ptr;
            if (p < region.data() || p >= region.data() + 2 * half) return false;
        }
        return true;
    };
    auto in_low = [&](const std::vector<rad_weight>& ws) {
        for (rad_weight w : ws) {
            const char* p = (const char*)m.residency()->lookup(w)->ptr;
            if (p < low || p >= low + 2 * half) return false;
        }
        return true;
    };
    auto at_home = [&](const std::vector<rad_weight>& ws) {
        for (rad_weight w : ws) if (m.residency()->lookup(w)->ptr != home[w]) return false;
        return true;
    };

    PrefillStager st;
    CHECK_OK(st.init(f.prog, plan, &m, region.data(), low, 2 * half, 64));
    REQUIRE(st.enabled());
    CHECK_EQ(st.n_layers(), (size_t)2);
    CHECK_EQ(st.ring_bytes(), 2 * half);

    RadBatch b{};
    b.n_tok = 8;
    st.set_ring(st.ring_full());
    CHECK(!st.begin_pass(&b));                        /* too few rows */
    b.n_tok = 128;
    st.set_ring(nullptr);
    CHECK(!st.begin_pass(&b));                        /* the level lends the ring's places */
    st.set_ring(st.ring_full());
    CHECK(st.begin_pass(&b));

    bool resync = false;
    CHECK_OK(st.before_op(0, compute, &resync));      /* embed: both layers go out */
    CHECK(resync);
    CHECK(in_ring(p0));
    CHECK(in_ring(p1));
    CHECK_OK(st.after_op(0, compute));
    CHECK_OK(st.before_op(1, compute, &resync));      /* attn0: nothing */
    CHECK(!resync);
    CHECK_OK(st.before_op(2, compute, &resync));      /* moe0: nothing left to stage */
    CHECK(!resync);
    CHECK(in_ring(p0));
    CHECK(in_ring(p1));
    CHECK_OK(st.after_op(2, compute));                /* moe0 is layer 0's last op... */
    CHECK(in_ring(p0));
    CHECK_OK(st.before_op(2, compute, &resync));      /* ...and runs again for the next rows */
    CHECK(in_ring(p0));
    CHECK_OK(st.after_op(2, compute));
    CHECK(in_ring(p0));
    CHECK(in_ring(p1));
    CHECK_OK(st.before_op(3, compute, &resync));      /* attn1 */
    CHECK(in_ring(p0));
    CHECK_OK(st.before_op(4, compute, &resync));      /* moe1: layer 0 is past every pass */
    CHECK(at_home(p0));
    CHECK(in_ring(p1));
    CHECK_OK(st.after_op(4, compute));
    CHECK_OK(st.end_pass(compute));
    CHECK(at_home(p1));
    CHECK_EQ(m.stats().stage_layers, (uint64_t)2);
    CHECK_EQ(m.stats().stage_overflow, (uint64_t)0);

    /* A pass that ends after its first op leaves nothing published at a copy. */
    CHECK(st.begin_pass(&b));
    CHECK_OK(st.before_op(0, compute, &resync));
    CHECK(in_ring(p0));
    CHECK_OK(st.end_pass(compute));
    CHECK(at_home(p0));
    CHECK(at_home(p1));

    /* At the span's shared level the ring is in its other place: the pass moves it there, and the
     * next pass at the full plan moves it back. */
    st.set_ring(st.ring_low());
    CHECK(st.begin_pass(&b));
    CHECK_OK(st.before_op(0, compute, &resync));
    CHECK(resync);
    CHECK(in_low(p0));
    CHECK(in_low(p1));
    CHECK_OK(st.end_pass(compute));
    CHECK(at_home(p0));
    CHECK(at_home(p1));
    st.set_ring(st.ring_full());
    CHECK(st.begin_pass(&b));
    CHECK_OK(st.before_op(0, compute, &resync));
    CHECK(in_ring(p0));
    CHECK(in_ring(p1));
    CHECK_OK(st.end_pass(compute));
    CHECK(at_home(p0));
    CHECK(at_home(p1));
    /* The copies are the mover stream's, and the region is this test's. */
    CHECK_OK(rad_stream_sync(m.transfer_stream()));
    rad_stream_destroy(compute);
}

/* The ring over many layers and passes: room for 3.5 layers, six layers per pass, three passes,
 * so placements lap the region and wrap. At every op, layers published at copies do not overlap;
 * at a layer's first op all its pooled units are at copies holding the pool bytes and earlier
 * layers are home; copies run more than one layer ahead when room allows; nothing falls back to a
 * read in place. */
TEST(the_stager_ring_laps_without_overwriting_a_published_layer) {
    Fixture f;
    const rad_weight emb = f.weight("token_embd.weight", 1 * MiB, RAD_ACCESS_VOCAB, -1, -1);
    f.op("embed", {emb}, false);
    constexpr int kL = 6, kE = 8;
    std::vector<std::vector<rad_weight>> lw(kL);
    for (int l = 0; l < kL; ++l)
        for (int e = 0; e < kE; ++e)
            lw[(size_t)l].push_back(f.weight("blk." + std::to_string(l) + ".ffn_gate_exps." +
                                             std::to_string(e) + ".weight", 1 * MiB,
                                             RAD_ACCESS_CONDITIONAL, l, e));
    for (int l = 0; l < kL; ++l) {
        f.op("attn" + std::to_string(l), {}, false);
        f.op("moe" + std::to_string(l), lw[(size_t)l], false);
    }
    Config cfg = base_config();
    cfg.placement = "expert_tiered";
    cfg.vram_weights_mib = 16;
    cfg.host_pool_mib = 128;
    Plan plan;
    PlannerInput in{&f.prog, &cfg, 0, 1, nullptr, 0};
    CHECK_OK(Planner(in).plan(&plan));
    REQUIRE(plan.dynamic);

    Mover m;
    MoverConfig mc;
    CHECK_OK(m.init(f.prog, plan, mc));
    RadStream compute = nullptr;
    CHECK_OK(rad_stream_create(&compute, 0));
    CHECK_OK(m.bind_compute_stream(compute));

    std::vector<std::vector<rad_weight>> pw(kL);
    int64_t most = 0;
    const int64_t each = align_up(1 * MiB, RAD_ALIGN_UNIT);
    for (int l = 0; l < kL; ++l) {
        for (rad_weight w : lw[(size_t)l])
            if (plan.weight[w].tier == Tier::HostPinned) pw[(size_t)l].push_back(w);
        most = std::max(most, each * (int64_t)pw[(size_t)l].size());
    }
    REQUIRE(most > 0);
    std::vector<void*> home(f.prog.weights.size(), nullptr);
    for (rad_weight w = 1; (size_t)w < f.prog.weights.size(); ++w)
        home[w] = m.residency()->lookup(w)->ptr;

    /* 3.5 times the heaviest layer: three go out at the pass's first op and the fourth wraps past
     * the region's end. */
    const int64_t ring = most * 7 / 2 / (2 * RAD_ALIGN_UNIT) * (2 * RAD_ALIGN_UNIT);
    std::vector<char> region((size_t)ring);
    PrefillStager st;
    CHECK_OK(st.init(f.prog, plan, &m, region.data(), nullptr, ring, 64));
    REQUIRE(st.enabled());
    CHECK_EQ(st.n_layers(), (size_t)kL);
    REQUIRE(st.ring_bytes() == ring);

    auto in_ring = [&](rad_weight w) {
        const char* p = (const char*)m.residency()->lookup(w)->ptr;
        return p >= region.data() && p < region.data() + ring;
    };
    /* The layers published at copies; false if any two of their units share a byte. */
    auto published = [&](std::vector<int>* ls) {
        std::vector<std::pair<const char*, const char*>> spans;
        ls->clear();
        for (int l = 0; l < kL; ++l) {
            bool any = false;
            for (rad_weight w : pw[(size_t)l])
                if (in_ring(w)) {
                    const char* p = (const char*)m.residency()->lookup(w)->ptr;
                    spans.emplace_back(p, p + f.prog.weights[w].logical_bytes);
                    any = true;
                }
            if (any) ls->push_back(l);
        }
        std::sort(spans.begin(), spans.end());
        for (size_t i = 1; i < spans.size(); ++i)
            if (spans[i].first < spans[i - 1].second) return false;
        return true;
    };

    RadBatch b{};
    b.n_tok = 128;
    st.set_ring(st.ring_full());
    bool ran_ahead = false;
    std::vector<int> ls;
    for (int pass = 0; pass < 3; ++pass) {
        CHECK(st.begin_pass(&b));
        const int n_ops = 1 + 2 * kL;
        for (int h = 0; h < n_ops; ++h) {
            bool resync = false;
            CHECK_OK(st.before_op(h, compute, &resync));
            CHECK(published(&ls));
            if (h >= 2 && h % 2 == 0) {
                const int l = (h - 2) / 2;            /* moe_l: the layer's first and last op */
                for (int k = 0; k < l; ++k)
                    for (rad_weight w : pw[(size_t)k]) CHECK(!in_ring(w));
                /* The stager's wait orders the op; the test drains the mover's stream to read
                 * the copies. */
                CHECK_OK(rad_stream_sync(m.transfer_stream()));
                for (rad_weight w : pw[(size_t)l]) {
                    CHECK(in_ring(w));
                    CHECK(std::memcmp(m.residency()->lookup(w)->ptr, home[w],
                                      (size_t)f.prog.weights[w].logical_bytes) == 0);
                }
                if (ls.size() > 2) ran_ahead = true;
            }
            CHECK_OK(st.after_op(h, compute));
        }
        CHECK_OK(st.end_pass(compute));
        for (int l = 0; l < kL; ++l)
            for (rad_weight w : pw[(size_t)l]) CHECK(m.residency()->lookup(w)->ptr == home[w]);
    }
    CHECK(ran_ahead);
    CHECK_EQ(m.stats().stage_overflow, (uint64_t)0);
    CHECK_OK(rad_stream_sync(m.transfer_stream()));
    rad_stream_destroy(compute);
}

/* The next step's first layers and the draft head. Six trunk layers plus a declared draft head's
 * seventh, which is never staged. When the next step continues with a prompt, a pass's end stages
 * the next trunk pass's first PrefillStager::lead() layers; a draft pass in between neither arms
 * nor touches them; the next armed trunk pass reads them in place, holding the pool's bytes; and a
 * trunk pass that does not arm returns them home before resolving anything. */
TEST(the_stager_stages_the_next_steps_first_layers_and_never_the_draft_head) {
    Fixture f;
    const rad_weight emb = f.weight("token_embd.weight", 1 * MiB, RAD_ACCESS_VOCAB, -1, -1);
    f.op("embed", {emb}, false);
    constexpr int kL = 7, kE = 8;                   /* layer 6 is the draft head's */
    std::vector<std::vector<rad_weight>> lw(kL);
    for (int l = 0; l < kL; ++l)
        for (int e = 0; e < kE; ++e)
            lw[(size_t)l].push_back(f.weight("blk." + std::to_string(l) + ".ffn_gate_exps." +
                                             std::to_string(e) + ".weight", 1 * MiB,
                                             RAD_ACCESS_CONDITIONAL, l, e));
    for (int l = 0; l < kL; ++l) {
        f.op("attn" + std::to_string(l), {}, false);
        f.op("moe" + std::to_string(l), lw[(size_t)l], false);
    }
    f.prog.drafter.kind = RAD_DRAFT_SERIAL;
    f.prog.meta.n_layers = kL - 1;
    Config cfg = base_config();
    cfg.placement = "expert_tiered";
    cfg.vram_weights_mib = 16;
    cfg.host_pool_mib = 128;
    Plan plan;
    PlannerInput in{&f.prog, &cfg, 0, 1, nullptr, 0};
    CHECK_OK(Planner(in).plan(&plan));
    REQUIRE(plan.dynamic);

    Mover m;
    MoverConfig mc;
    CHECK_OK(m.init(f.prog, plan, mc));
    RadStream compute = nullptr;
    CHECK_OK(rad_stream_create(&compute, 0));
    CHECK_OK(m.bind_compute_stream(compute));

    std::vector<std::vector<rad_weight>> pw(kL);
    int64_t most = 0;
    const int64_t each = align_up(1 * MiB, RAD_ALIGN_UNIT);
    for (int l = 0; l < kL; ++l) {
        for (rad_weight w : lw[(size_t)l])
            if (plan.weight[w].tier == Tier::HostPinned) pw[(size_t)l].push_back(w);
        most = std::max(most, each * (int64_t)pw[(size_t)l].size());
    }
    REQUIRE(most > 0);
    for (int l = 0; l < kL; ++l) REQUIRE(!pw[(size_t)l].empty());
    std::vector<void*> home(f.prog.weights.size(), nullptr);
    for (rad_weight w = 1; (size_t)w < f.prog.weights.size(); ++w)
        home[w] = m.residency()->lookup(w)->ptr;

    /* Room for every trunk layer at once, so what the ring holds is set by pacing, not size. */
    const int64_t ring = most * 8 / (2 * RAD_ALIGN_UNIT) * (2 * RAD_ALIGN_UNIT);
    std::vector<char> region((size_t)ring);
    PrefillStager st;
    CHECK_OK(st.init(f.prog, plan, &m, region.data(), nullptr, ring, 64));
    REQUIRE(st.enabled());
    CHECK_EQ(st.n_layers(), (size_t)(kL - 1));
    const int lead = (int)PrefillStager::lead();
    REQUIRE(lead >= 1 && lead < kL - 1);

    auto in_ring = [&](int l) {
        for (rad_weight w : pw[(size_t)l]) {
            const char* p = (const char*)m.residency()->lookup(w)->ptr;
            if (p < region.data() || p >= region.data() + ring) return false;
        }
        return true;
    };
    auto at_home = [&](int l) {
        for (rad_weight w : pw[(size_t)l]) if (m.residency()->lookup(w)->ptr != home[w]) return false;
        return true;
    };
    auto copies_hold_the_pool = [&](int l) {
        CHECK_OK(rad_stream_sync(m.transfer_stream()));
        for (rad_weight w : pw[(size_t)l])
            if (std::memcmp(m.residency()->lookup(w)->ptr, home[w],
                            (size_t)f.prog.weights[w].logical_bytes) != 0) return false;
        return true;
    };
    /* A whole trunk pass, checking at each layer's MoE that it reads copies and that the head's
     * layer stays home. */
    const int n_ops = 1 + 2 * (kL - 1);
    auto run_pass = [&]() {
        for (int h = 0; h < n_ops; ++h) {
            bool resync = false;
            CHECK_OK(st.before_op(h, compute, &resync));
            if (h >= 2 && h % 2 == 0) {
                const int l = (h - 2) / 2;
                CHECK(in_ring(l));
                CHECK(copies_hold_the_pool(l));
            }
            CHECK(at_home(kL - 1));
            CHECK_OK(st.after_op(h, compute));
        }
    };

    RadBatch b{};
    b.n_tok = 128;
    st.set_ring(st.ring_full());

    /* No next step: the end leaves everything home. staged() is read after the pass ends, so it
     * outlives end_pass. */
    CHECK(!st.staged());
    st.set_next_prefill(false);
    CHECK(st.begin_pass(&b));
    run_pass();
    CHECK_OK(st.end_pass(compute));
    CHECK(st.staged());
    for (int l = 0; l < kL; ++l) CHECK(at_home(l));

    /* The next step continues: the first `lead` layers go out at the end. */
    st.set_next_prefill(true);
    CHECK(st.begin_pass(&b));
    run_pass();
    CHECK_OK(st.end_pass(compute));
    for (int l = 0; l < kL; ++l) CHECK(l < lead ? in_ring(l) : at_home(l));
    for (int l = 0; l < lead; ++l) CHECK(copies_hold_the_pool(l));

    /* A draft pass neither arms nor returns them, and leaves staged() as the trunk pass set it. */
    RadBatch d = b;
    d.draft_pass = -1;
    CHECK(!st.begin_pass(&d));
    CHECK(st.staged());
    for (int l = 0; l < lead; ++l) CHECK(in_ring(l));

    /* The next armed trunk pass reads them in place and does not restage them at its first op. */
    const uint64_t layers_before = m.stats().stage_layers;
    st.set_next_prefill(true);
    CHECK(st.begin_pass(&b));
    {
        bool resync = false;
        CHECK_OK(st.before_op(0, compute, &resync));
        CHECK_EQ(m.stats().stage_layers, layers_before);
        CHECK_OK(st.after_op(0, compute));
    }
    for (int h = 1; h < n_ops; ++h) {
        bool resync = false;
        CHECK_OK(st.before_op(h, compute, &resync));
        if (h >= 2 && h % 2 == 0) {
            const int l = (h - 2) / 2;
            CHECK(in_ring(l));
            CHECK(copies_hold_the_pool(l));
        }
        CHECK_OK(st.after_op(h, compute));
    }
    CHECK_OK(st.end_pass(compute));
    for (int l = 0; l < kL; ++l) CHECK(l < lead ? in_ring(l) : at_home(l));

    /* A trunk pass too small to arm returns everything home before resolving a table. */
    b.n_tok = 8;
    CHECK(!st.begin_pass(&b));
    CHECK(!st.staged());
    for (int l = 0; l < kL; ++l) CHECK(at_home(l));
    CHECK_EQ(m.stats().stage_overflow, (uint64_t)0);
    CHECK_OK(rad_stream_sync(m.transfer_stream()));
    rad_stream_destroy(compute);
}

TEST(mover_reserves_from_the_weight_pool_rather_than_beside_it) {
    /* The production path. Pools has already allocated the weights pool as one block; allocating
     * beside it would spend the budget twice and surface later as an unrelated out-of-memory.
     * Reserving from Pools::weights() also means a promotion can never reach a KV block. */
    HeatFixture hf;
    Pools pools;
    Config c = hf.cfg;
    c.vram_kv_mib = 16;
    c.gpu_headroom_mib = 1;
    CHECK_OK(pools.configure(c, /*vram_capacity=*/0, &rad_malloc_allocator(),
                             &rad_malloc_allocator()));

    const int64_t weights_before = pools.weights().used();
    const int64_t host_before = pools.host_pinned().used();

    Mover m;
    MoverConfig mc;
    CHECK_OK(m.init(hf.f.prog, hf.plan, mc, &pools));

    /* Every byte the mover took came from the pools' budgets, none from KV. */
    CHECK(pools.weights().used() > weights_before);
    CHECK(pools.host_pinned().used() > host_before);
    CHECK_EQ(pools.kv().used(), (int64_t)0);
    CHECK(pools.weights().ok());
    CHECK(pools.host_pinned().ok());
    for (rad_weight w = 1; (size_t)w < hf.f.prog.weights.size(); ++w)
        CHECK(m.residency()->lookup(w)->ptr != nullptr);
}

TEST(mover_declines_to_move_under_a_static_plan) {
    Fixture f;
    std::vector<rad_weight> ex;
    build_moe(f, 8, 1 * MiB, &ex);
    Config c = base_config();
    c.placement = "expert_tiered";
    c.vram_weights_mib = 5;
    c.host_pool_mib = 64;
    c.deterministic = true;
    PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0};
    Plan p;
    CHECK_OK(Planner(in).plan(&p));

    Mover m;
    MoverConfig mc;
    CHECK_OK(m.init(f.prog, p, mc));
    Swap s{0, 1};
    CHECK_OK(m.issue(s, nullptr));
    CHECK_EQ(m.stats().promotions, (uint64_t)0);
    CHECK_EQ(m.stats().h2d_bytes, (uint64_t)0);
}

TEST(the_plan_report_names_every_class_once) {
    Fixture f;
    std::vector<rad_weight> ex;
    build_moe(f, 8, 1 * MiB, &ex);
    Config c = base_config();
    c.placement = "expert_tiered";
    c.vram_weights_mib = 5;
    c.host_pool_mib = 64;
    PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0};
    Plan p;
    CHECK_OK(Planner(in).plan(&p));

    const std::string r = p.report(f.prog);
    /* Eight experts, one line: the class key collapses indexed weights in the report. */
    size_t n = 0, at = 0;
    const std::string key = "blk.*.ffn_gate_exps.*.weight";
    while ((at = r.find(key, at)) != std::string::npos) { ++n; at += key.size(); }
    CHECK_EQ((int)n, 1);
    CHECK(r.find("token_embd.weight") != std::string::npos);
    CHECK(r.find("zero-copy") != std::string::npos);
    CHECK(r.find("expert_tiered") != std::string::npos);
}

/* ================================================================== the planner's own helpers
 *
 * Small functions the planner is built from, where two components must agree or a budget
 * overruns a pool. The mover, the report and the budget resolver call them instead of rederiving
 * the answer, so they are tested directly.
 */

TEST(the_strategy_names_round_trip_and_an_unknown_one_is_refused) {
    for (Strategy s : { Strategy::AllVram, Strategy::LayerOffload, Strategy::ExpertTiered,
                        Strategy::Auto }) {
        Strategy back = Strategy::Auto;
        CHECK_OK(strategy_parse(strategy_name(s), &back));
        CHECK((int)back == (int)s);
    }
    /* These spellings are the --placement flag's legal set, so they are pinned literally. */
    CHECK_EQ(std::string(strategy_name(Strategy::AllVram)), std::string("all_vram"));
    CHECK_EQ(std::string(strategy_name(Strategy::LayerOffload)), std::string("layer_offload"));
    CHECK_EQ(std::string(strategy_name(Strategy::ExpertTiered)), std::string("expert_tiered"));
    CHECK_EQ(std::string(strategy_name(Strategy::Auto)), std::string("auto"));

    /* *out is untouched on a refusal, so a caller ignoring the status keeps its prior value. */
    Strategy keep = Strategy::ExpertTiered;
    CHECK_EQ(strategy_parse("layerOffload", &keep), RAD_E_INVAL);
    CHECK((int)keep == (int)Strategy::ExpertTiered);
    CHECK_EQ(strategy_parse("", &keep), RAD_E_INVAL);
    CHECK_EQ(strategy_parse(nullptr, &keep), RAD_E_INVAL);
    CHECK((int)keep == (int)Strategy::ExpertTiered);
    CHECK_EQ(strategy_parse("all_vram", nullptr), RAD_E_INVAL);
}

/* The strongest access class in a group decides the group's, so this ordering keeps the heat
 * engine from demoting what the next token needs. VOCAB ranks equal to per-token: lm_head and the
 * embedding are read every token, and are a separate class only for their size and the gather. */
TEST(access_strength_orders_the_classes_and_is_pessimistic_about_an_unknown_one) {
    CHECK_EQ(access_strength(RAD_ACCESS_PER_TOKEN), access_strength(RAD_ACCESS_VOCAB));
    CHECK(access_strength(RAD_ACCESS_PER_TOKEN) > access_strength(RAD_ACCESS_PER_REQUEST));
    CHECK(access_strength(RAD_ACCESS_PER_REQUEST) > access_strength(RAD_ACCESS_CONDITIONAL));
    CHECK(access_strength(RAD_ACCESS_CONDITIONAL) > access_strength(RAD_ACCESS_RARE));
    CHECK_EQ(access_strength(RAD_ACCESS_RARE), 0);

    /* An unknown access class (from a newer plugin) is treated as per-token, not as rare as a
       zero default would. */
    CHECK_EQ(access_strength(9999), access_strength(RAD_ACCESS_PER_TOKEN));
    CHECK_EQ(access_strength(-1), access_strength(RAD_ACCESS_PER_TOKEN));
}

/* The stored layout is what occupies a tier. A kernel that re-layouts its weights may store a
 * different byte count than the declared shape, so budgeting the logical size would short the
 * pool. */
TEST(weight_bytes_prefers_the_stored_layout_then_the_logical_then_the_shape) {
    WeightInfo w;
    w.decl.dtype = RAD_BF16;
    w.decl.rank = 2;
    w.decl.shape[0] = 4;
    w.decl.shape[1] = 8;
    /* Nothing resolved yet: the declared dtype and shape, as in planner unit-test fixtures. */
    CHECK_EQ(weight_bytes(w), (int64_t)(4 * 8 * 2));

    w.logical_bytes = 100;
    CHECK_EQ(weight_bytes(w), 100ll);
    w.stored_bytes = 160;                 /* the layout pass re-laid this one out */
    CHECK_EQ(weight_bytes(w), 160ll);

    /* A rank-0 weight has no elements, not one. */
    WeightInfo z;
    z.decl.dtype = RAD_BF16;
    z.decl.rank = 0;
    CHECK_EQ(weight_bytes(z), 0ll);
}

/* What a host reader of the container (the router predictor, through Mover::read_weight_file)
 * reads of a weight: the stored bytes of one stored as it is, or the selected plane a kernel's
 * relayout reads. DeepSeek-V4.1's router under RADIANCE_EXPERIMENTAL_DS41_ROUTER_BF16 is the second
 * kind: gemm_nt_f32bf16 keeps its bf16 halves on the card (half the bytes), and the predictor still
 * reads the container's f32 plane, [384, 5120], so read-ahead (..._PREFETCH) keeps working with the
 * switch on. Nothing for a weight the loader widens or narrows, whose stored bytes are not the
 * file's, nor for one joining several planes or of unknown encoding. */
TEST(weight_file_bytes_is_the_container_plane_not_the_stored_form) {
    WeightInfo w;
    w.stored_bytes = 384ll * 5120 * 4;
    CHECK_EQ(weight_file_bytes(w), 384ll * 5120 * 4);   /* identity, as without the switch */
    w.widen = true;
    CHECK_EQ(weight_file_bytes(w), 0ll);
    w.widen = false;
    w.narrow = true;
    CHECK_EQ(weight_file_bytes(w), 0ll);

    WeightInfo r;   /* relaid: the f32 plane held on the card as its bf16 halves */
    r.identity = false;
    r.stored_bytes = 384ll * 5120 * 2;
    r.enc = rad_enc_plain(RAD_F32);
    r.enc_known = true;
    r.n_sel = 1;
    r.sel_rows[0] = 384;
    r.sel_cols[0] = 5120;
    CHECK_EQ(weight_file_bytes(r), 384ll * 5120 * 4);
    r.enc = rad_enc_plain(RAD_BF16);   /* a bf16 plane: two bytes an element in the file */
    CHECK_EQ(weight_file_bytes(r), 384ll * 5120 * 2);
    r.enc_known = false;
    CHECK_EQ(weight_file_bytes(r), 0ll);
    r.enc_known = true;
    r.n_sel = 2;
    CHECK_EQ(weight_file_bytes(r), 0ll);
}

/* The footprint must agree with the planner: the VRAM budget resolver calls it before a Planner
 * exists and gives the planner a budget computed from it. It must reproduce the unit grouping, the
 * per-unit alignment and the mapped-tier pin. */
TEST(the_weight_footprint_splits_statics_from_the_elastic_expert_plane) {
    Fixture f;
    std::vector<rad_weight> experts;
    build_moe(f, /*n=*/4, /*each=*/2 * MiB, &experts);

    WeightFootprint fp;
    weight_footprint(f.prog, &fp, TablePlacement::Vram, TablePlacement::Disk, false, true);

    /* One model-level weight and four experts: five units. */
    CHECK_EQ(fp.n_units, 5ll);
    CHECK_EQ(fp.statics, 1 * MiB);
    CHECK_EQ(fp.experts, 4 * 2 * MiB);
    CHECK_EQ(fp.mapped, 0ll);

    /* Every weight is in exactly one bucket; one counted twice or not at all would silently skew
       the budget by its size. */
    int64_t declared = 0;
    for (rad_weight w = 1; (size_t)w < f.prog.weights.size(); ++w)
        declared += weight_bytes(f.prog.weights[w]);
    CHECK_EQ(fp.statics + fp.experts + fp.mapped + fp.hosted, declared);
}

TEST(weights_in_one_layer_are_one_unit_and_are_aligned_once_each) {
    /* Two weights in each of two layers, at a size not a multiple of the unit alignment, so the
       padding shows in the total. */
    Fixture f;
    const int64_t odd = 1000;
    for (int l = 0; l < 2; ++l) {
        const rad_weight a = f.weight("blk." + std::to_string(l) + ".attn.weight", odd,
                                      RAD_ACCESS_PER_TOKEN, l, -1);
        const rad_weight b = f.weight("blk." + std::to_string(l) + ".ffn.weight", odd,
                                      RAD_ACCESS_PER_TOKEN, l, -1);
        f.op("gemm", { a, b }, false);
    }

    WeightFootprint fp;
    weight_footprint(f.prog, &fp, TablePlacement::Vram, TablePlacement::Disk, false, true);
    CHECK_EQ(fp.n_units, 2ll);              /* the declared group is the unit, not the weight */

    /* One align_up inside the unit and one for the unit, as Mover::init spends the slab. Raw byte
       sums would fall short by an alignment per unit and overrun the weights pool on large MoEs. */
    const int64_t per_unit = align_up(align_up(odd, RAD_ALIGN_UNIT) + odd, RAD_ALIGN_UNIT);
    CHECK_EQ(fp.statics, 2 * per_unit);
    CHECK(fp.statics > 4 * odd);
}

/* A weight no device kernel can read never enters a pool, and its whole unit goes with it, since
 * a unit is what moves. Counting it against VRAM would reserve card memory for something read only
 * in the container's mapping. */
TEST(a_weight_with_no_device_band_takes_its_whole_unit_to_the_mapped_tier) {
    Fixture f;
    /* Layer 0 has two weights; the op reading one of them has only a host kernel. */
    const rad_weight a = f.weight("blk.0.a.weight", 4 * MiB, RAD_ACCESS_PER_TOKEN, 0, -1);
    const rad_weight b = f.weight("blk.0.b.weight", 2 * MiB, RAD_ACCESS_PER_TOKEN, 0, -1);
    f.op("gemm", { a }, /*host_kernel=*/false);

    f.host_op("host_op", { b });

    WeightFootprint fp;
    weight_footprint(f.prog, &fp, TablePlacement::Vram, TablePlacement::Disk, false, true);
    CHECK_EQ(fp.n_units, 1ll);
    CHECK_EQ(fp.statics, 0ll);
    CHECK_EQ(fp.mapped, 6 * MiB);          /* both weights: the unit is what moves */
}

/* On the host backend the weights pool is host memory, so a host-only weight stays in the
 * container's mapping only if it is stored as its kernel reads it. One its host kernel widens at
 * load takes a copy in the pool instead, and the footprint counts it the same way. */
TEST(on_the_host_backend_a_converted_host_only_weight_takes_a_pool_copy) {
    Fixture f;
    const rad_weight direct = f.weight("blk.0.a.weight", 4 * MiB, RAD_ACCESS_PER_TOKEN, 0, -1);
    const rad_weight widened = f.weight("output_norm.weight", 2 * MiB, RAD_ACCESS_PER_TOKEN, -1, -1);
    f.prog.weights[widened].widen = true;
    for (rad_weight w : { direct, widened }) f.host_op("host_op", { w });

    WeightFootprint card, host;
    weight_footprint(f.prog, &card, TablePlacement::Vram, TablePlacement::Disk, false, true);
    weight_footprint(f.prog, &host, TablePlacement::Vram, TablePlacement::Disk, true, true);
    CHECK_EQ(card.mapped, 6 * MiB);          /* on a card both stay in the mapping */
    CHECK_EQ(card.statics, 0ll);
    CHECK_EQ(host.mapped, 4 * MiB);
    CHECK_EQ(host.statics, 2 * MiB);         /* the widened one is the pool's */

    Config cfg = base_config();
    cfg.vram_weights_mib = 8;
    Plan plan;
    PlannerInput in{&f.prog, &cfg, 0, 1, nullptr, 0};
    in.host_backend = true;
    CHECK_OK(Planner(in).plan(&plan));
    CHECK(plan.weight[direct].tier == Tier::Mapped);
    CHECK(plan.weight[widened].tier == Tier::VRAM);   /* the weights pool, host memory here */

    /* A checkpoint served directly has no container mapping, so on the host backend both take a
     * pool copy and the footprint counts both, or the planner would overfill the pool. */
    WeightFootprint ckpt;
    weight_footprint(f.prog, &ckpt, TablePlacement::Vram, TablePlacement::Disk, true, false);
    CHECK_EQ(ckpt.mapped, 0ll);
    CHECK_EQ(ckpt.statics, 6 * MiB);
    Plan cplan;
    PlannerInput cin{&f.prog, &cfg, 0, 1, nullptr, 0};
    cin.host_backend = true;
    cin.container = false;
    CHECK_OK(Planner(cin).plan(&cplan));
    CHECK(cplan.weight[direct].tier == Tier::VRAM);
    CHECK(cplan.weight[widened].tier == Tier::VRAM);
}

/* On the host backend an op may read a weight from the container's mapping beside one its kernel
 * converted into the pool (Flash-Next's hc_read: a gain read as stored, mixing matrices joined at
 * load). Both are host memory and the op runs its host kernel, so the plan reports no site
 * disagreement, which on a card would mean the op cannot run. */
TEST(on_the_host_backend_an_op_reads_a_mapped_and_a_pooled_weight_together) {
    Fixture f;
    const rad_weight gain = f.weight("blk.0.hc_norm.weight", 1 * MiB, RAD_ACCESS_PER_TOKEN, 0, -1);
    const rad_weight mix = f.weight("blk.0.hc_down.weight", 2 * MiB, RAD_ACCESS_PER_TOKEN, -1, -1);
    f.prog.weights[mix].widen = true;
    f.host_op("hc_read", { gain, mix });

    Config cfg = base_config();
    cfg.vram_weights_mib = 8;
    Plan plan;
    PlannerInput in{&f.prog, &cfg, 0, 1, nullptr, 0};
    in.host_backend = true;
    int rc = 0;
    const std::string err = stderr_of([&] { rc = Planner(in).plan(&plan); });
    CHECK_OK(rc);
    CHECK(plan.weight[gain].tier == Tier::Mapped);
    CHECK(plan.weight[mix].tier == Tier::VRAM);   /* the weights pool, host memory here */
    for (const std::string& n : plan.notes) CHECK(!has(n, "no site serves both"));
    CHECK(!has(err, "mixes a mapped weight"));
}

/* A unit that cannot be read from the mapping is one misfit, however many weights it groups. */
TEST(a_mapped_unit_that_cannot_be_read_in_place_is_counted_once) {
    Fixture f;
    const rad_weight a = f.weight("blk.0.a.weight", 4 * MiB, RAD_ACCESS_PER_TOKEN, 0, -1);
    const rad_weight b = f.weight("blk.0.b.weight", 2 * MiB, RAD_ACCESS_PER_TOKEN, 0, -1);
    f.prog.weights[a].widen = true;
    f.prog.weights[b].widen = true;
    f.host_op("host_op", { a, b });

    Config cfg = base_config();
    cfg.vram_weights_mib = 8;
    Plan plan;
    PlannerInput in{&f.prog, &cfg, 0, 1, nullptr, 0};
    (void)Planner(in).plan(&plan);
    CHECK(plan.weight[a].tier == Tier::Mapped);
    REQUIRE_EQ(plan.unfit.size(), (size_t)1);
    CHECK_EQ(plan.unfit[0].count, 1ll);
    CHECK_EQ(plan.unfit[0].bytes, plan.units[(size_t)plan.weight[a].unit].bytes);
}

/* --embedding-placement fixture: the embedding is read by `embed_lookup`, as every arch declares
 * it, and the head by a GEMM; the flag moves only the former. */
void build_dense_lookup(Fixture& f, int layers, int64_t per_layer, int64_t emb = 1 * MiB) {
    const rad_weight m = f.weight("token_embd.weight", emb, RAD_ACCESS_VOCAB, -1, -1);
    f.op("embed_lookup", {m}, false);
    for (int l = 0; l < layers; ++l) {
        const rad_weight w = f.weight("blk." + std::to_string(l) + ".ffn_down.weight", per_layer,
                                      RAD_ACCESS_PER_TOKEN, l, -1);
        f.op("gemm", {w}, false);
    }
    const rad_weight o = f.weight("output.weight", 1 * MiB, RAD_ACCESS_VOCAB, -1, -1);
    f.op("logits_gemm", {o}, false);
}

/* An n-gram table: model-level, read only by `embed_lookup_q`, which only a host kernel serves. */
rad_weight add_ngram_table(Fixture& f, int64_t bytes) {
    const rad_weight t = f.weight("ngram.weight", bytes, RAD_ACCESS_VOCAB, -1, -1);
    f.host_op("embed_lookup_q", { t });
    return t;
}

/* Tables are identified by their readers: a model-level weight read only by embed_lookup is the
 * embedding, one read only by embed_lookup_q an n-gram table. A head read by a GEMM is neither, and
 * neither is a table also read by a GEMM (a tied head would stream over the link every step).
 * Placed in RAM, the embedding moves from statics to hosted, since the resolver sizes a derived
 * weight budget from the statics. */
TEST(the_placement_flags_find_their_tables_by_what_reads_them) {
    Fixture f;
    build_dense_lookup(f, /*layers=*/2, /*per_layer=*/4 * MiB);
    add_ngram_table(f, 8 * MiB);

    WeightFootprint fp;
    weight_footprint(f.prog, &fp, TablePlacement::Vram, TablePlacement::Disk, false, true);
    CHECK_EQ(fp.embedding, 1 * MiB);
    CHECK_EQ(fp.ngram, 8 * MiB);
    CHECK(!fp.ngram_device);
    CHECK_EQ(fp.hosted, 0ll);
    CHECK_EQ(fp.statics, 2 * 4 * MiB + 2 * MiB);
    CHECK_EQ(fp.mapped, 8 * MiB);

    weight_footprint(f.prog, &fp, TablePlacement::Ram, TablePlacement::Disk, false, true);
    CHECK_EQ(fp.embedding, 1 * MiB);
    CHECK_EQ(fp.hosted, 1 * MiB);
    CHECK_EQ(fp.statics, 2 * 4 * MiB + 1 * MiB);

    CHECK(is_embedding_table(f.prog, 1));
    CHECK(!is_embedding_table(f.prog, 2));                 /* a layer's GEMM operand */
    CHECK(!is_embedding_table(f.prog, 4));                 /* the head */
    CHECK(is_ngram_table(f.prog, 5));
    CHECK(!is_ngram_table(f.prog, 1));

    /* A head tied to the table: one weight read by both the lookup and the logits GEMM. */
    Fixture t;
    const rad_weight m = t.weight("token_embd.weight", 1 * MiB, RAD_ACCESS_VOCAB, -1, -1);
    t.op("embed_lookup", {m}, false);
    t.op("logits_gemm", {m}, false);
    CHECK(!is_embedding_table(t.prog, m));
    weight_footprint(t.prog, &fp, TablePlacement::Ram, TablePlacement::Disk, false, true);
    CHECK_EQ(fp.embedding, 0ll);
    CHECK_EQ(fp.hosted, 0ll);
}

TEST(embedding_ram_goes_to_the_pinned_pool_read_in_place_and_costs_no_vram) {
    Fixture f;
    build_dense_lookup(f, /*layers=*/2, /*per_layer=*/4 * MiB);
    Config cfg = base_config();
    cfg.placement = "all_vram";
    cfg.vram_weights_mib = 9;           /* exactly the layers and the head, not the embedding */
    cfg.host_pool_mib = 2;
    cfg.embedding_placement = TablePlacement::Ram;
    cfg.embedding_resolved = TablePlacement::Ram;
    Plan plan;
    PlannerInput in{&f.prog, &cfg, 0, 1, nullptr, 0};
    CHECK_OK(Planner(in).plan(&plan));

    const rad_weight emb = 1;           /* declared first */
    CHECK(plan.weight[emb].tier == Tier::HostPinned);
    CHECK(plan.weight[emb].site == Site::DeviceZeroCopy);
    CHECK(!plan.units[(size_t)plan.weight[emb].unit].movable);
    CHECK_EQ(plan.embedding_host_bytes, 1 * MiB);
    for (rad_weight w = 2; (size_t)w < f.prog.weights.size(); ++w)
        CHECK(plan.weight[w].tier == Tier::VRAM);

    /* Vram keeps it with the other weights. */
    cfg.embedding_resolved = TablePlacement::Vram;
    cfg.vram_weights_mib = 10;
    Plan plan2;
    CHECK_OK(Planner(in).plan(&plan2));
    CHECK(plan2.weight[emb].tier == Tier::VRAM);
    CHECK_EQ(plan2.embedding_host_bytes, 0ll);
}

/* RADIANCE_EXPERIMENTAL_DS41_DSPARK_CODEBOOK_RAM: a drafter's codebooks, model-level and read only
 * by dflash_select, are lookup tables beside the embedding, pinned with it; off, they stay
 * statics. A codebook a GEMM also reads is not one. */
TEST(dspark_codebook_ram_pins_the_codebooks_with_the_embedding) {
    for (int on = 0; on < 2; ++on) {
        if (on) setenv(kDraftCodebookRamEnv, "1", 1);
        else unsetenv(kDraftCodebookRamEnv);
        Fixture f;
        build_dense_lookup(f, /*layers=*/2, /*per_layer=*/4 * MiB);
        const rad_weight w1 = f.weight("dspark.markov_w1.weight", 2 * MiB, RAD_ACCESS_VOCAB, -1, -1);
        const rad_weight w2 = f.weight("dspark.markov_w2.weight", 2 * MiB, RAD_ACCESS_VOCAB, -1, -1);
        f.op("dflash_select", { w1, w2 }, false);
        const rad_weight w3 = f.weight("other.weight", 1 * MiB, RAD_ACCESS_VOCAB, -1, -1);
        f.op("dflash_select", { w3 }, false);
        f.op("gemm", { w3 }, false);
        CHECK_EQ(is_embedding_table(f.prog, w1), on == 1);
        CHECK_EQ(is_embedding_table(f.prog, w2), on == 1);
        CHECK(!is_embedding_table(f.prog, w3));
        WeightFootprint fp;
        weight_footprint(f.prog, &fp, TablePlacement::Ram, TablePlacement::Disk, false, true);
        CHECK_EQ(fp.embedding, on ? 5 * MiB : 1 * MiB);
        CHECK_EQ(fp.hosted, on ? 5 * MiB : 1 * MiB);
        Config cfg = base_config();
        cfg.placement = "all_vram";
        cfg.vram_weights_mib = on ? 10 : 14;   /* the layers, the head, w3 (and off, both codebooks) */
        cfg.host_pool_mib = 0;
        cfg.embedding_host_mib = on ? 5 : 1;
        cfg.embedding_resolved = TablePlacement::Ram;
        Plan plan;
        PlannerInput in{&f.prog, &cfg, 0, 1, nullptr, 0};
        CHECK_OK(Planner(in).plan(&plan));
        CHECK(plan.weight[1].tier == Tier::HostPinned);
        CHECK(plan.weight[w1].tier == (on ? Tier::HostPinned : Tier::VRAM));
        CHECK(plan.weight[w2].tier == (on ? Tier::HostPinned : Tier::VRAM));
        if (on) CHECK(plan.weight[w1].site == Site::DeviceZeroCopy);
        CHECK(plan.weight[w3].tier == Tier::VRAM);
        CHECK_EQ(plan.embedding_host_bytes, on ? 5 * MiB : 1 * MiB);
    }
    unsetenv(kDraftCodebookRamEnv);
}

/* The embedding gets its own room: the resolver sizes Config::embedding_host_mib on top of
 * --host-pool-mib, so a dense model with no pool still places the table, and the experts keep all
 * of --host-pool-mib after the table is pinned. */
TEST(embedding_ram_brings_its_own_room_beside_the_host_pool) {
    Fixture f;
    build_dense_lookup(f, /*layers=*/2, /*per_layer=*/4 * MiB);
    Config cfg = base_config();
    cfg.placement = "all_vram";
    cfg.vram_weights_mib = 9;
    cfg.host_pool_mib = 0;              /* no weights pool */
    cfg.embedding_host_mib = 1;
    cfg.embedding_resolved = TablePlacement::Ram;
    Plan plan;
    PlannerInput in{&f.prog, &cfg, 0, 1, nullptr, 0};
    CHECK_OK(Planner(in).plan(&plan));
    CHECK(plan.weight[1].tier == Tier::HostPinned);
    CHECK_EQ(plan.host_budget, 0ll);

    cfg.host_pool_mib = 3;
    Plan plan2;
    CHECK_OK(Planner(in).plan(&plan2));
    CHECK(plan2.weight[1].tier == Tier::HostPinned);
    CHECK_EQ(plan2.host_budget, 3 * MiB);  /* --host-pool-mib, whole */
}

/* An encoder tower as architectures declare one: a stem and a merger at model level, one layer
 * unit per block numbered past the text layers, every weight read once per request. */
void add_tower(Fixture& f, int first_layer, int blocks, int64_t per_block) {
    const rad_weight s = f.weight("v.patch_embd.weight", 1 * MiB, RAD_ACCESS_PER_REQUEST, -1, -1);
    f.op("vit_gemm", {s}, false);
    for (int b = 0; b < blocks; ++b) {
        const rad_weight w = f.weight("v.blk." + std::to_string(b) + ".attn_qkv.weight", per_block,
                                      RAD_ACCESS_PER_REQUEST, first_layer + b, -1);
        f.op("vit_gemm", {w}, false);
    }
    const rad_weight m = f.weight("v.merger.weight", 1 * MiB, RAD_ACCESS_PER_REQUEST, -1, -1);
    f.op("vit_gemm", {m}, false);
}

/* The tower is identified by its access class at every level: the footprint counts every unit
 * whose weights are all per-request, model-level and per-block alike, within the statics (the
 * resolver subtracts it from them). */
TEST(the_footprint_counts_an_encoder_tower_at_every_level) {
    Fixture f;
    build_dense_lookup(f, /*layers=*/2, /*per_layer=*/4 * MiB);
    add_tower(f, /*first_layer=*/2, /*blocks=*/3, /*per_block=*/2 * MiB);
    WeightFootprint fp;
    weight_footprint(f.prog, &fp, TablePlacement::Vram, TablePlacement::Disk, false, true);
    CHECK_EQ(fp.tower, 1 * MiB + 3 * 2 * MiB + 1 * MiB);
    CHECK_EQ(fp.statics, 2 * 4 * MiB + 2 * MiB + fp.tower);

    Fixture t;
    build_dense_lookup(t, /*layers=*/2, /*per_layer=*/4 * MiB);
    weight_footprint(t.prog, &fp, TablePlacement::Vram, TablePlacement::Disk, false, true);
    CHECK_EQ(fp.tower, 0ll);
}

/* A tower's own lookup table: model-level, read once a request and only by embed_lookup, like
 * Gemma 4's patch-position tables. Both the embedding's test and the tower's take it. */
rad_weight add_tower_lookup(Fixture& f, int64_t bytes) {
    const rad_weight t = f.weight("v.pos_x.weight", bytes, RAD_ACCESS_PER_REQUEST, -1, -1);
    f.op("embed_lookup", {t}, false);
    return t;
}

/* Both tables in RAM: pin_embedding pins the tower's lookup table and pin_towers skips it, so the
 * split Engine::plan makes (the statics less the embedding and less tower_pin_bytes) is what the
 * planner keeps on the card, and every text weight stays there. Taking the table out twice
 * leaves the split short by it, and auto offloads the last layers in issue order. With the
 * embedding on the card the table stays with the tower. */
TEST(a_tower_lookup_table_the_embedding_pins_leaves_the_card_once) {
    Fixture f;
    build_dense_lookup(f, /*layers=*/2, /*per_layer=*/4 * MiB);
    add_tower(f, /*first_layer=*/2, /*blocks=*/3, /*per_block=*/2 * MiB);
    const rad_weight pos = add_tower_lookup(f, 1 * MiB);
    WeightFootprint fp;
    weight_footprint(f.prog, &fp, TablePlacement::Vram, TablePlacement::Disk, false, true);
    CHECK_EQ(fp.embedding, 1 * MiB + 1 * MiB);                      /* the token table and pos */
    CHECK_EQ(fp.tower, 1 * MiB + 3 * 2 * MiB + 1 * MiB + 1 * MiB);  /* stem, blocks, merger, pos */
    CHECK_EQ(fp.tower_lookup, 1 * MiB);
    CHECK_EQ(tower_pin_bytes(fp, TablePlacement::Vram), fp.tower);

    const int64_t emb = fp.embedding, towers = tower_pin_bytes(fp, TablePlacement::Ram);
    Config cfg = base_config();
    cfg.placement = "auto";
    cfg.vram_weights_mib = (fp.statics - emb - towers) / MiB;       /* the layers and the head */
    cfg.host_pool_mib = 0;
    cfg.embedding_placement = TablePlacement::Ram;
    cfg.embedding_resolved = TablePlacement::Ram;
    cfg.embedding_host_mib = emb / MiB;
    cfg.tower_resolved = TablePlacement::Ram;
    cfg.tower_host_mib = towers / MiB;
    Plan plan;
    PlannerInput in{&f.prog, &cfg, 0, 1, nullptr, 0};
    CHECK_OK(Planner(in).plan(&plan));
    CHECK(plan.strategy == Strategy::AllVram);
    CHECK(plan.weight[pos].tier == Tier::HostPinned);
    CHECK_EQ(plan.embedding_host_bytes, emb);
    CHECK_EQ(plan.tower_host_bytes, towers);
    CHECK_EQ(plan.host_budget, 0ll);
    int64_t vram = 0;
    for (const MoveUnit& mu : plan.units)
        if (mu.tier == Tier::VRAM) vram += mu.slot_bytes;
    CHECK_EQ(vram, cfg.vram_weights_mib * MiB);

    /* The table taken out of the statics twice: a table short, and the tail leaves the card. */
    cfg.vram_weights_mib = (fp.statics - emb - fp.tower) / MiB;
    Plan plan2;
    CHECK(Planner(in).plan(&plan2) < 0);
    CHECK(plan2.strategy == Strategy::LayerOffload);

    /* The embedding on the card: pin_towers pins the table with the rest of the tower. */
    const int64_t whole = tower_pin_bytes(fp, TablePlacement::Vram);
    Config vc = base_config();
    vc.placement = "auto";
    vc.vram_weights_mib = (fp.statics - whole) / MiB;
    vc.host_pool_mib = 0;
    vc.embedding_placement = TablePlacement::Vram;
    vc.embedding_resolved = TablePlacement::Vram;
    vc.tower_resolved = TablePlacement::Ram;
    vc.tower_host_mib = whole / MiB;
    Plan plan3;
    PlannerInput vin{&f.prog, &vc, 0, 1, nullptr, 0};
    CHECK_OK(Planner(vin).plan(&plan3));
    CHECK(plan3.strategy == Strategy::AllVram);
    CHECK(plan3.weight[pos].tier == Tier::HostPinned);
    CHECK_EQ(plan3.embedding_host_bytes, 0ll);
    CHECK_EQ(plan3.tower_host_bytes, whole);
    CHECK_EQ(plan3.host_budget, 0ll);
}

/* --vision-tower-placement ram: every tower unit goes to the pinned pool, read in place, before
 * any strategy runs, so a weights budget sized for the text model alone still plans and keeps the
 * text model on the card. Vram keeps the tower with the other weights. */
TEST(tower_ram_pins_every_tower_unit_in_the_pool_and_costs_no_vram) {
    Fixture f;
    build_dense_lookup(f, /*layers=*/2, /*per_layer=*/4 * MiB);
    const size_t first_tower = f.prog.weights.size();
    add_tower(f, /*first_layer=*/2, /*blocks=*/3, /*per_block=*/2 * MiB);
    Config cfg = base_config();
    cfg.placement = "all_vram";
    cfg.vram_weights_mib = 10;          /* exactly the text model: layers, embedding and head */
    cfg.host_pool_mib = 0;
    cfg.tower_resolved = TablePlacement::Ram;
    cfg.tower_host_mib = 8;             /* the resolver's room beside the pool */
    Plan plan;
    PlannerInput in{&f.prog, &cfg, 0, 1, nullptr, 0};
    CHECK_OK(Planner(in).plan(&plan));
    for (rad_weight w = 1; (size_t)w < f.prog.weights.size(); ++w) {
        if ((size_t)w < first_tower) {
            CHECK(plan.weight[w].tier == Tier::VRAM);
            continue;
        }
        CHECK(plan.weight[w].tier == Tier::HostPinned);
        CHECK(plan.weight[w].site == Site::DeviceZeroCopy);
        const MoveUnit& mu = plan.units[(size_t)plan.weight[w].unit];
        CHECK(!mu.movable);
        CHECK(mu.preplaced());
    }
    CHECK_EQ(plan.tower_host_bytes, 8 * MiB);
    CHECK_EQ(plan.forced_host_used, 8 * MiB);
    CHECK_EQ(plan.host_budget, 0ll);
    CHECK(plan.report(f.prog).find("--vision-tower-placement ram") != std::string::npos);

    cfg.tower_resolved = TablePlacement::Vram;
    cfg.tower_host_mib = 0;
    cfg.vram_weights_mib = 18;
    Plan plan2;
    CHECK_OK(Planner(in).plan(&plan2));
    for (rad_weight w = 1; (size_t)w < f.prog.weights.size(); ++w)
        CHECK(plan2.weight[w].tier == Tier::VRAM);
    CHECK_EQ(plan2.tower_host_bytes, 0ll);
}

/* A tower too large for its room is refused, and so is ram for a program with no tower (the
 * resolver only picks Ram when the footprint found one, so that is a wiring fault). */
TEST(tower_ram_refuses_a_pool_too_small_and_a_program_with_no_tower) {
    Fixture f;
    build_dense_lookup(f, /*layers=*/2, /*per_layer=*/4 * MiB);
    add_tower(f, /*first_layer=*/2, /*blocks=*/3, /*per_block=*/2 * MiB);
    Config cfg = base_config();
    cfg.placement = "all_vram";
    cfg.vram_weights_mib = 10;
    cfg.host_pool_mib = 0;
    cfg.tower_resolved = TablePlacement::Ram;
    cfg.tower_host_mib = 7;
    Plan plan;
    PlannerInput in{&f.prog, &cfg, 0, 1, nullptr, 0};
    CHECK_EQ(Planner(in).plan(&plan), RAD_E_NOMEM);

    Fixture t;
    build_dense_lookup(t, /*layers=*/2, /*per_layer=*/4 * MiB);
    cfg.tower_host_mib = 8;
    Plan plan2;
    PlannerInput tin{&t.prog, &cfg, 0, 1, nullptr, 0};
    CHECK_EQ(Planner(tin).plan(&plan2), RAD_E_STATE);
}

/* A table that is not a whole number of pages still leaves the report's pinned total consistent:
 * the pool slot is charged rounded up while the tier total counts weight bytes, so the report
 * subtracts the same weight bytes it added. */
TEST(the_report_counts_a_hosted_embedding_once_in_one_unit) {
    Fixture f;
    build_dense_lookup(f, /*layers=*/1, /*per_layer=*/4 * MiB, /*emb=*/3 * MiB + 100);
    Config cfg = base_config();
    cfg.placement = "all_vram";
    cfg.host_pool_mib = 8;
    cfg.embedding_resolved = TablePlacement::Ram;
    Plan plan;
    PlannerInput in{&f.prog, &cfg, 0, 1, nullptr, 0};
    CHECK_OK(Planner(in).plan(&plan));
    CHECK_EQ(plan.forced_host_used, 3 * MiB + 100);
    CHECK(plan.embedding_host_bytes >= plan.forced_host_used);
    const std::string r = plan.report(f.prog);
    const size_t at = r.find("host pinned");
    CHECK(at != std::string::npos);
    if (at != std::string::npos) {
        const std::string line = r.substr(at, r.find('\n', at) - at);
        CHECK(line.find('-') == std::string::npos);           /* no negative total */
    }
    CHECK(r.find("--embedding-placement ram") != std::string::npos);
}

/* Ram resolved for a program with no table read only by embed_lookup means the budget was split
 * for a different program; the plan refuses rather than keep the table on a split without it. */
TEST(embedding_ram_with_no_table_to_place_is_refused) {
    Fixture t;
    const rad_weight m = t.weight("token_embd.weight", 1 * MiB, RAD_ACCESS_VOCAB, -1, -1);
    t.op("embed_lookup", {m}, false);
    t.op("logits_gemm", {m}, false);
    Config cfg = base_config();
    cfg.placement = "all_vram";
    cfg.host_pool_mib = 4;
    cfg.embedding_resolved = TablePlacement::Ram;
    Plan plan;
    PlannerInput in{&t.prog, &cfg, 0, 1, nullptr, 0};
    CHECK_EQ(Planner(in).plan(&plan), RAD_E_STATE);
    CHECK(!plan.notes.empty());
    CHECK(plan.weight[m].tier == Tier::VRAM);
}

TEST(embedding_ram_that_does_not_fit_the_host_pool_is_refused_with_the_plan_filled_in) {
    Fixture f;
    build_dense_lookup(f, /*layers=*/1, /*per_layer=*/4 * MiB);
    Config cfg = base_config();
    cfg.placement = "all_vram";
    cfg.host_pool_mib = 0;              /* nowhere to put it */
    cfg.embedding_resolved = TablePlacement::Ram;
    Plan plan;
    PlannerInput in{&f.prog, &cfg, 0, 1, nullptr, 0};
    CHECK_EQ(Planner(in).plan(&plan), RAD_E_NOMEM);
    CHECK_EQ(plan.weight.size(), f.prog.weights.size());   /* finish() ran, so the report prints */
    CHECK(!plan.notes.empty());
}

/* --ngram-placement ram keeps the table on the mapped tier (counted by no pool) and names the
 * load-time copy: the reason says where rows come from, the report what the copy costs in RAM. */
TEST(ngram_ram_stays_mapped_and_reports_the_copy) {
    Fixture f;
    build_dense_lookup(f, /*layers=*/1, /*per_layer=*/4 * MiB);
    const rad_weight t = add_ngram_table(f, 8 * MiB);
    Config cfg = base_config();
    cfg.placement = "all_vram";
    cfg.ngram_resolved = TablePlacement::Ram;
    Plan plan;
    PlannerInput in{&f.prog, &cfg, 0, 1, nullptr, 0};
    CHECK_OK(Planner(in).plan(&plan));
    CHECK(plan.weight[t].tier == Tier::Mapped);
    CHECK(plan.weight[t].site == Site::Host);
    CHECK_EQ(plan.ngram_copy_bytes, 8 * MiB);
    CHECK(std::string(plan.weight[t].reason).find("--ngram-placement ram") != std::string::npos);
    CHECK(plan.report(f.prog).find("n-gram copy") != std::string::npos);

    cfg.ngram_resolved = TablePlacement::Disk;
    Plan plan2;
    CHECK_OK(Planner(in).plan(&plan2));
    CHECK_EQ(plan2.ngram_copy_bytes, 0ll);
    CHECK(plan2.report(f.prog).find("n-gram copy") == std::string::npos);
}

/* An n-gram table read by both a device and a host kernel: --ngram-placement decides where it is
 * and so who gathers. vram counts and places it with the statics; ram keeps it mapped at the
 * zero-copy site (the pinned copy a card reads) and reports the copy; disk keeps it mapped at the
 * host site, where the host gather reads the container. */
TEST(an_ngram_table_a_device_kernel_reads_goes_where_the_placement_says) {
    Fixture f;
    build_dense_lookup(f, /*layers=*/1, /*per_layer=*/4 * MiB);
    const rad_weight t = f.weight("ngram.weight", 8 * MiB, RAD_ACCESS_VOCAB, -1, -1);
    f.op("embed_lookup_q", {t}, /*host_kernel=*/true);

    WeightFootprint fp;
    weight_footprint(f.prog, &fp, TablePlacement::Vram, TablePlacement::Vram, false, true);
    CHECK(fp.ngram_device);
    CHECK(fp.ngram_host);
    CHECK_EQ(fp.ngram, 8 * MiB);
    CHECK_EQ(fp.mapped, 0ll);
    CHECK_EQ(fp.statics, 4 * MiB + 2 * MiB + 8 * MiB);
    for (TablePlacement off : { TablePlacement::Ram, TablePlacement::Disk }) {
        weight_footprint(f.prog, &fp, TablePlacement::Vram, off, false, true);
        CHECK_EQ(fp.ngram, 8 * MiB);
        CHECK_EQ(fp.mapped, 8 * MiB);
        CHECK_EQ(fp.statics, 4 * MiB + 2 * MiB);
    }

    Config cfg = base_config();
    cfg.placement = "all_vram";
    PlannerInput in{&f.prog, &cfg, 0, 1, nullptr, 0};

    cfg.ngram_resolved = TablePlacement::Vram;
    Plan on_card;
    CHECK_OK(Planner(in).plan(&on_card));
    CHECK(on_card.weight[t].tier == Tier::VRAM);
    CHECK(on_card.weight[t].site == Site::Device);
    CHECK_EQ(on_card.ngram_copy_bytes, 0ll);

    cfg.ngram_resolved = TablePlacement::Ram;
    Plan in_ram;
    CHECK_OK(Planner(in).plan(&in_ram));
    CHECK(in_ram.weight[t].tier == Tier::Mapped);
    CHECK(in_ram.weight[t].site == Site::DeviceZeroCopy);
    CHECK_EQ(in_ram.ngram_copy_bytes, 8 * MiB);
    CHECK(in_ram.ngram_copy_card);
    CHECK(std::string(in_ram.weight[t].reason).find("the card reads") != std::string::npos);
    CHECK(in_ram.report(f.prog).find("pinned RAM the card reads") != std::string::npos);

    cfg.ngram_resolved = TablePlacement::Disk;
    Plan on_disk;
    CHECK_OK(Planner(in).plan(&on_disk));
    CHECK(on_disk.weight[t].tier == Tier::Mapped);
    CHECK(on_disk.weight[t].site == Site::Host);
    CHECK_EQ(on_disk.ngram_copy_bytes, 0ll);
    CHECK(std::string(on_disk.weight[t].reason).find("--ngram-placement disk") !=
          std::string::npos);
}

/* Disk needs the host gather kernel: an n-gram table only a device kernel reads has no host site,
 * and the footprint reports that. */
TEST(an_ngram_table_with_no_host_kernel_cannot_stay_on_disk) {
    Fixture f;
    const rad_weight t = f.weight("ngram.weight", 8 * MiB, RAD_ACCESS_VOCAB, -1, -1);
    f.op("embed_lookup_q", {t}, /*host_kernel=*/false);
    WeightFootprint fp;
    weight_footprint(f.prog, &fp, TablePlacement::Vram, TablePlacement::Disk, false, true);
    CHECK(fp.ngram_device);
    CHECK(!fp.ngram_host);
}

TEST(an_empty_program_has_an_empty_footprint) {
    Fixture f;
    WeightFootprint fp;
    fp.statics = 1;                         /* the out parameter is reset, not accumulated */
    weight_footprint(f.prog, &fp, TablePlacement::Vram, TablePlacement::Disk, false, true);
    CHECK_EQ(fp.statics, 0ll);
    CHECK_EQ(fp.experts, 0ll);
    CHECK_EQ(fp.mapped, 0ll);
    CHECK_EQ(fp.n_units, 0ll);
    /* a null out parameter does not crash */
    weight_footprint(f.prog, nullptr, TablePlacement::Vram, TablePlacement::Disk, false, true);
}

/* ================================================================== planner decisions */
namespace {

/* `layers` routed layers of `n` experts of `each` bytes, one weight and one op per layer, plus a
 * model-level embedding. */
void build_moe_grid(Fixture& f, int layers, int n, int64_t each) {
    const rad_weight m = f.weight("token_embd.weight", 1 * MiB, RAD_ACCESS_VOCAB, -1, -1);
    f.op("embed", {m}, false);
    for (int l = 0; l < layers; ++l) {
        std::vector<rad_weight> g;
        for (int e = 0; e < n; ++e)
            g.push_back(f.weight("blk." + std::to_string(l) + ".ffn_gate_exps." + std::to_string(e) +
                                 ".weight", each, RAD_ACCESS_CONDITIONAL, l, e));
        f.op("moe_gemm", g, false);
    }
}

/* Resident (VRAM) routed units per layer. */
std::vector<int> residents_per_layer(const Plan& p, int layers) {
    std::vector<int> n((size_t)layers, 0);
    for (const MoveUnit& mu : p.units)
        if (mu.expert >= 0 && mu.layer >= 0 && mu.layer < layers && mu.tier == Tier::VRAM)
            ++n[(size_t)mu.layer];
    return n;
}

int spread(const std::vector<int>& v, size_t from = 0, size_t to = SIZE_MAX) {
    int lo = INT32_MAX, hi = INT32_MIN;
    for (size_t i = from; i < v.size() && i < to; ++i) { lo = std::min(lo, v[i]); hi = std::max(hi, v[i]); }
    return hi - lo;
}

}  /* namespace */

/* Without a calibration profile every share is zero and the tie-break is the whole policy. It is
 * expert-major: expert 0 of every layer before expert 1 of any, so the budget cut lands at the
 * same expert everywhere and every layer starts with its share of the slab. Layer-major filling
 * leaves the last layers (and a draft head) with nothing resident, streaming every expert over the
 * link each step. With a profile, the profile decides. */
TEST(planner_spreads_residents_over_the_layers_and_follows_a_profile) {
    Fixture f;
    build_moe_grid(f, 4, 16, 1 * MiB);
    Config c = base_config();
    c.placement = "expert_tiered";
    c.vram_weights_mib = 31;
    c.host_pool_mib = 128;
    Plan p;
    { PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0}; CHECK_OK(Planner(in).plan(&p)); }
    const std::vector<int> flat = residents_per_layer(p, 4);
    const int total = flat[0] + flat[1] + flat[2] + flat[3];
    CHECK(total >= 8 && total < 64);
    CHECK(spread(flat) <= 1);

    /* Layer 3 is hot in the calibration set: it fills first, and the rest still spreads. */
    std::vector<RadFileProfile> prof;
    for (int l = 0; l < 4; ++l)
        for (int e = 0; e < 16; ++e) prof.push_back(RadFileProfile{ l, e, l == 3 ? 0.05f : 0.001f, 0.0f });
    Plan q;
    {
        PlannerInput in{&f.prog, &c, 0, 1, prof.data(), (int64_t)prof.size()};
        CHECK_OK(Planner(in).plan(&q));
    }
    const std::vector<int> hot = residents_per_layer(q, 4);
    CHECK_EQ(hot[0] + hot[1] + hot[2] + hot[3], total);
    CHECK(total > 16);
    CHECK_EQ(hot[3], 16);
    CHECK(spread(hot, 0, 3) <= 1);
}

/* `auto`, the default, is one of the three named strategies: everything on the card when it fits
 * (or no budget caps it), routed experts spilled when there are any, whole layers otherwise; the
 * plan says which and why. */
TEST(auto_placement_picks_the_strategy_the_model_and_budget_call_for) {
    struct Case { const char* what; bool moe; int64_t mib; Strategy want; };
    for (const Case& k : { Case{ "fits", true, 64, Strategy::AllVram },
                           Case{ "no budget", true, 0, Strategy::AllVram },
                           Case{ "routed experts spill", true, 4, Strategy::ExpertTiered },
                           Case{ "dense layers spill", false, 3, Strategy::LayerOffload } }) {
        Fixture f;
        if (k.moe) build_moe(f, 8, 1 * MiB, nullptr);
        else       build_dense(f, 4, 1 * MiB, /*host_kernel=*/true);
        Config c = base_config();
        c.vram_weights_mib = k.mib;
        c.host_pool_mib = 64;
        PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0};
        Plan p;
        CHECK_OK(Planner(in).plan(&p));
        if (p.strategy != k.want) fprintf(stderr, "    %s: auto chose %s\n", k.what, strategy_name(p.strategy));
        CHECK(p.strategy == k.want);
        CHECK(!p.strategy_why.empty());
    }
}

/* The spilled layers are a suffix of execution order, which a plugin's layer numbering need not
 * follow: declared last-numbered first, the layers executed last are layers 0-3. */
TEST(layer_offload_spills_the_layers_that_run_last_not_the_highest_numbered) {
    Fixture f;
    const rad_weight m = f.weight("token_embd.weight", 1 * MiB, RAD_ACCESS_VOCAB, -1, -1);
    f.op("embed", {m}, true);
    for (int l = 7; l >= 0; --l)
        f.op("gemm", { f.weight("blk." + std::to_string(l) + ".ffn_down.weight", 1 * MiB,
                                RAD_ACCESS_PER_TOKEN, l, -1) }, true);
    Config c = base_config();
    c.placement = "layer_offload";
    c.vram_weights_mib = 5;
    c.host_pool_mib = 64;
    PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0};
    Plan p;
    CHECK_OK(Planner(in).plan(&p));
    for (const MoveUnit& mu : p.units) {
        if (mu.layer < 0) continue;
        CHECK(mu.site == (mu.layer < 4 ? Site::Host : Site::Device));
    }
    REQUIRE_EQ(p.host_runs.size(), 1u);
    CHECK_EQ(p.host_runs[0].first_layer, 0);
    CHECK_EQ(p.host_runs[0].last_layer, 3);
}

/* When the dense weights alone overflow the budget, expert_tiered spills layers first and then
 * places the routed experts. The host pool is spent once across both: the experts get what the
 * spilled dense units left, the rest are listed as unfit, and nothing past the pool is placed,
 * which the mover would refuse at reserve. */
TEST(an_expert_fill_after_a_layer_spill_spends_the_host_pool_once) {
    Fixture f;
    const rad_weight m = f.weight("token_embd.weight", 1 * MiB, RAD_ACCESS_VOCAB, -1, -1);
    f.op("embed", {m}, true);
    for (int l = 0; l < 4; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        f.op("gemm", { f.weight(b + "ffn_down.weight", 2 * MiB, RAD_ACCESS_PER_TOKEN, l, -1) }, true);
        f.op("moe_gemm", { f.weight(b + "ffn_gate_exps.0.weight", 1 * MiB, RAD_ACCESS_CONDITIONAL, l, 0),
                           f.weight(b + "ffn_gate_exps.1.weight", 1 * MiB, RAD_ACCESS_CONDITIONAL, l, 1) },
             true);
    }
    Config c = base_config();
    c.placement = "expert_tiered";
    c.vram_weights_mib = 5;          /* below the 9 MiB of dense weights */
    /* What the layer spill places: three layers' dense units and experts. The expert fill
     * that follows keeps none resident, so it has the six dense MiB left over for eight experts. */
    c.host_pool_mib = 12;
    PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0};
    Plan p;
    (void)stderr_of([&] { (void)Planner(in).plan(&p); });
    int64_t host = 0, dense_host = 0, unfit = 0, lost = 0;
    for (const MoveUnit& mu : p.units) {
        const bool off = mu.tier == Tier::HostPinned || mu.tier == Tier::HostPageable;
        if (off) {
            host += mu.slot_bytes;
            if (mu.expert < 0) dense_host += mu.slot_bytes;
        }
        if (mu.reason && std::string(mu.reason) == "fits in no tier") ++unfit;
        else if (!off && mu.tier != Tier::VRAM) ++lost;
    }
    CHECK(dense_host > 0);
    CHECK(host <= p.host_budget);
    /* Every unit is placed or listed as unfit, never left in between. */
    int64_t listed = 0;
    for (const Plan::Unfit& u : p.unfit) listed += u.count;
    CHECK_EQ(listed, unfit);
    CHECK(unfit > 0);
    CHECK_EQ(lost, 0ll);
    CHECK(p.totals.bytes[(int)Tier::VRAM] + p.slab_overhead <= p.vram_budget);
}

/* An op runs in one domain, so its weights must agree on the site. An op reading a host-run
 * layer's weight beside a device one demotes the host unit to staged, which every tier supports,
 * and says so; the plan still fits. */
TEST(an_op_reading_host_and_device_weights_demotes_the_host_one_to_staged) {
    Fixture f;
    build_dense(f, 8, 1 * MiB, /*host_kernel=*/true);
    /* Layer 7 runs on the host under this budget, layer 0 on the device. */
    f.op("mix", { (rad_weight)9, (rad_weight)2 }, true);
    Config c = base_config();
    c.placement = "layer_offload";
    c.vram_weights_mib = 5;
    c.host_pool_mib = 64;
    PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0};
    Plan p;
    int rc = 0;
    const std::string log = stderr_of([&] { rc = Planner(in).plan(&p); });
    CHECK_OK(rc);
    CHECK(has(log, "mixes host-site and device-site"));
    for (const MoveUnit& mu : p.units) {
        if (mu.layer == 7) {
            CHECK(mu.site == Site::DeviceStaged);
            CHECK(mu.reason && has(mu.reason, "demoted"));
        } else if (mu.layer >= 4) {
            CHECK(mu.site == Site::Host);
        }
    }
    CHECK(p.fits());
}

/* ================================================================== heat tie-breaks
 *
 * Most residents sit at exactly zero heat, so the tie-break is effectively the policy: on equal
 * heat, promote into the layer holding the fewest residents and demote from the one holding the
 * most, then the lower unit index. step() is checked against pick() elsewhere; these pin pick()
 * itself against hand-set residency.
 */
namespace {
struct TwoLayers {
    Fixture f;
    Plan plan;
    HeatEngine h;
    int build(int res0, int res1) {
        build_moe_grid(f, 2, 8, 1 * MiB);
        Config c = base_config();
        c.placement = "expert_tiered";
        c.vram_weights_mib = 12;
        c.host_pool_mib = 64;
        PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0};
        RAD_TRY(Planner(in).plan(&plan));
        RAD_TRY(h.init(plan, HeatParams{}));
        /* Residency as asked: the first `res` experts of each layer. */
        for (int32_t u = 0; u < (int32_t)plan.units.size(); ++u) {
            const MoveUnit& mu = plan.units[(size_t)u];
            if (mu.expert < 0) continue;
            h.set_tier(u, mu.expert < (mu.layer == 0 ? res0 : res1) ? Tier::VRAM : Tier::HostPinned);
        }
        return RAD_OK;
    }
    int32_t unit(int layer, int expert) const {
        for (int32_t u = 0; u < (int32_t)plan.units.size(); ++u)
            if (plan.units[(size_t)u].layer == layer && plan.units[(size_t)u].expert == expert) return u;
        return -1;
    }
    void heat(int layer, int expert, float credit) {
        const int32_t counts[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
        int32_t c[8];
        std::memcpy(c, counts, sizeof c);
        c[expert] = 1;
        h.observe(layer, c, 8, credit);
    }
};
}  /* namespace */

TEST(on_equal_heat_the_thinner_layer_is_promoted_into_and_the_fuller_one_gives_up) {
    TwoLayers t;
    REQUIRE_EQ(t.build(5, 2), RAD_OK);
    CHECK_EQ(t.h.layer_residents(t.unit(0, 0)), 5);
    CHECK_EQ(t.h.layer_residents(t.unit(1, 0)), 2);
    t.heat(0, 6, 4.0f);
    t.heat(1, 6, 4.0f);
    CHECK_EQ(t.h.heat(t.unit(0, 6)), t.h.heat(t.unit(1, 6)));
    Swap s;
    REQUIRE(t.h.pick(&s));
    CHECK_EQ(s.promote, t.unit(1, 6));
    /* Every resident is at zero: the victim comes from the layer holding five, lowest index first. */
    CHECK_EQ(s.demote, t.unit(0, 0));
    std::vector<Swap> sw;
    CHECK_OK(t.h.step(&sw));
    REQUIRE(!sw.empty());
    CHECK_EQ(sw[0].promote, s.promote);
    CHECK_EQ(sw[0].demote, s.demote);
}

TEST(on_equal_heat_and_residents_the_lower_unit_wins) {
    TwoLayers t;
    REQUIRE_EQ(t.build(3, 3), RAD_OK);
    t.heat(1, 5, 4.0f);
    t.heat(0, 5, 4.0f);
    Swap s;
    REQUIRE(t.h.pick(&s));
    const int32_t a = t.unit(0, 5), b = t.unit(1, 5);
    CHECK_EQ(s.promote, std::min(a, b));
    CHECK_EQ(s.demote, std::min(t.unit(0, 0), t.unit(1, 0)));
}

/* The guard is `hot >= cold * hysteresis + min_gain`, inclusive: against a zero-heat victim a
 * candidate at exactly min_gain is promoted, and one a float step below it is not. */
TEST(the_promotion_guard_is_inclusive_at_its_boundary) {
    const HeatParams hp;
    for (bool at : { true, false }) {
        TwoLayers t;
        REQUIRE_EQ(t.build(4, 4), RAD_OK);
        const float credit = at ? hp.min_gain : std::nextafter(hp.min_gain, 0.0f);
        t.heat(0, 6, credit);
        CHECK_EQ(t.h.heat(t.unit(0, 6)), credit);
        Swap s;
        CHECK_EQ(t.h.pick(&s), at);
        std::vector<Swap> sw;
        CHECK_OK(t.h.step(&sw));
        CHECK_EQ(sw.empty(), !at);
    }
}

/* ================================================================== the loan fill, by layer */

/* On a cold start every heat is equal, so the fill's tie-break decides where a loan's slots go:
 * the layer with the fewest residents first, counting the promotions the batch has issued, since
 * none has retired yet. A fill in unit order gives the first layers everything and the last layers
 * nothing. Every slot is issued in the dispatch the loan arrives in. */
TEST(a_cold_fill_spreads_the_loan_over_the_layers) {
    Fixture f;
    build_moe_grid(f, 4, 16, 256 * KiB);
    Config c = base_config();
    c.placement = "expert_tiered";
    c.vram_weights_mib = 3;
    c.host_pool_mib = 64;
    Plan plan;
    { PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0}; REQUIRE_EQ(Planner(in).plan(&plan), RAD_OK); }
    HeatEngine h;
    REQUIRE_EQ(h.init(plan, HeatParams{}), RAD_OK);
    Mover m;
    MoverConfig mc;
    mc.turnover_slots = 0;
    REQUIRE_EQ(m.init(f.prog, plan, mc), RAD_OK);
    RadStream compute = nullptr;
    REQUIRE_EQ(rad_stream_create(&compute, 0), RAD_OK);
    CHECK_OK(m.bind_compute_stream(compute));
    Loan loan;
    CHECK_OK(m.lend_from(Mover::kLendKV, loan.strips));
    const std::vector<int> before = residents_per_layer(plan, 4);
    CHECK(spread(before) <= 1);
    int32_t pooled = 0, cls = -1;
    for (const MoveUnit& mu : plan.units) {
        if (mu.expert < 0) continue;
        pooled += mu.tier == Tier::HostPinned;
        cls = mu.slab_class;
    }
    REQUIRE(cls >= 0);
    const int64_t slots = m.flex_capacity() / plan.classes[(size_t)cls].stride;
    const int64_t expect = std::min<int64_t>(slots, pooled);
    CHECK(expect >= 8);

    ComputeWork work;
    int d = 0;
    CHECK_OK(m.set_loan(Mover::kLendKV, Loan::kAll));
    CHECK_OK(m.begin_dispatch(d++, &h));
    work.run(compute);
    CHECK_OK(m.balance(&h));
    int64_t issued = 0;
    for (int32_t u = 0; u < (int32_t)plan.units.size(); ++u) issued += h.busy(u);
    CHECK_EQ(issued, expect);
    auto now = [&]() {
        std::vector<int> n(4, 0);
        for (int32_t u = 0; u < (int32_t)plan.units.size(); ++u) {
            const MoveUnit& mu = plan.units[(size_t)u];
            if (mu.expert >= 0 && h.tier(u) == Tier::VRAM) ++n[(size_t)mu.layer];
        }
        return n;
    };
    const int planned = before[0] + before[1] + before[2] + before[3];
    for (Deadline dl; any_busy(h, plan) && !dl.passed();) {
        CHECK_OK(m.begin_dispatch(d++, &h));
        work.run(compute);
        CHECK_OK(m.balance(&h));
    }
    const std::vector<int> after = now();
    CHECK_EQ(after[0] + after[1] + after[2] + after[3], planned + (int)expect);
    CHECK(spread(after) <= 1);
    CHECK_OK(rad_stream_sync(compute));
    rad_stream_destroy(compute);
}

/* ================================================================== teardown */

namespace {

/* The VRAM this process holds, as KFD counts it per process (/sys/class/kfd/kfd/proc/<pid>/
 * vram_<gpu>, summed over the cards). The card-wide figure also moves with other processes and
 * with this process's page tables, which grow as address ranges are first mapped and are not
 * returned on free: a page, or a page plus a 2 MiB directory, of difference there is no leak.
 * Without KFD (the host backend) the backend's own figure. */
int process_vram(int64_t* out) {
    const std::string dir = "/sys/class/kfd/kfd/proc/" + std::to_string((long long)getpid());
    DIR* d = opendir(dir.c_str());
    if (!d) return dev_vram_used(0, out);
    int64_t sum = 0;
    bool any = false;
    while (const dirent* e = readdir(d)) {
        if (std::strncmp(e->d_name, "vram_", 5) != 0) continue;
        FILE* f = std::fopen((dir + "/" + e->d_name).c_str(), "r");
        long long v = 0;
        if (f && std::fscanf(f, "%lld", &v) == 1) { sum += v; any = true; }
        if (f) std::fclose(f);
    }
    closedir(d);
    if (!any) return dev_vram_used(0, out);
    *out = sum;
    return RAD_OK;
}

/* process_vram once it holds still. A stream is a hardware queue, and KFD gives each queue a
 * context save area (the topology's cwsr_size + ctl_stack_size: 29.3 MiB on a 64-CU RDNA4 card)
 * that can be released a few milliseconds after the queue is destroyed, so a reading straight
 * after a stream's destruction may still include it. Reads every 10 ms until 100 ms pass
 * without a change, for at most 2 s. */
int settled_vram(int64_t* out) {
    int64_t v = 0;
    RAD_TRY(process_vram(&v));
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    for (int still = 0; still < 10 && std::chrono::steady_clock::now() < end;) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        int64_t w = 0;
        RAD_TRY(process_vram(&w));
        still = w == v ? still + 1 : 0;
        v = w;
    }
    *out = v;
    return RAD_OK;
}

}  /* namespace */

/* Tools and tests build and tear down movers repeatedly, and on a card every event or stream is
 * a hardware queue and VRAM. A mover that staged, promoted and filled a loan gives back every
 * event and every byte of device memory it took. */
/* The runtime and the driver keep some memory for the life of the process once a path is first
 * used: page tables for address ranges first mapped (a page, or a page and a 2 MiB directory). So
 * the work runs once before the count is taken, and the count after a second run may differ by a
 * page and a directory, never by a buffer. Both counts are taken once the figure holds still: a
 * destroyed stream's queue save area can outlive it by a few milliseconds (settled_vram). */
TEST(a_mover_gives_back_every_event_and_device_byte) {
    auto run = [&]() {
        const int64_t events0 = rad_dev_live_events();
        {
            Fixture f;
            build_dense(f, 8, 1 * MiB, /*host_kernel=*/false);
            Config c = base_config();
            c.placement = "layer_offload";
            c.vram_weights_mib = 5;
            c.host_pool_mib = 64;
            PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0};
            Plan p;
            REQUIRE_EQ(Planner(in).plan(&p), RAD_OK);
            Mover m;
            MoverConfig mc;
            REQUIRE_EQ(m.init(f.prog, p, mc), RAD_OK);
            RadStream compute = nullptr;
            REQUIRE_EQ(rad_stream_create(&compute, 0), RAD_OK);
            CHECK_OK(m.bind_compute_stream(compute));
            CHECK(rad_dev_live_events() > events0);
            ComputeWork work;
            for (int d = 0; d < 20; ++d) {
                CHECK_OK(m.begin_dispatch(d, nullptr));
                work.run(compute);
                CHECK_OK(m.prefetch_to_op(p.prefetch[(size_t)d % p.prefetch.size()].at_op));
            }
            CHECK(m.stats().prefetched > 0);
            CHECK_OK(rad_stream_sync(compute));
            rad_stream_destroy(compute);
        }
        {
            LoanRig r;
            REQUIRE_EQ(r.build(), RAD_OK);
            CHECK_OK(r.m.set_loan(Mover::kLendKV, Loan::kAll));
            for (Deadline dl; (r.m.flex_used_bytes() == 0 || any_busy(r.h, r.hf.plan)) && !dl.passed();)
                CHECK_OK(r.dispatch());
            CHECK(r.m.stats().promotions > 0);
        }
        CHECK_EQ(rad_dev_live_events(), events0);
    };
    run();
    int64_t vram0 = 0, vram1 = 0;
    CHECK_OK(settled_vram(&vram0));
    run();
    CHECK_OK(settled_vram(&vram1));
    CHECK(vram1 - vram0 <= 2 * MiB + 4096);
}

TEST(source_localized_decode_profile_ranks_whole_GU_DN_units_without_budget_changes) {
    Fixture f;
    build_moe_layers(f, 1, 8, 2 * MiB, 1 * MiB);
    f.prog.ctx.rank = 1;
    f.prog.ctx.world_size = 4;
    std::map<std::string, RadFileEntry> directory;
    for (size_t i = 1; i < f.prog.weights.size(); ++i) {
        auto& w = f.prog.weights[i];
        if (w.decl.group.expert < 0) continue;
        const int local = w.decl.group.expert;
        const int slot = w.logical_bytes == 2 * MiB ? 0 : 1;
        w.decl.group.slot = slot;
        w.source = "source." + std::to_string(96 + local) + "." + std::to_string(slot);
        RadFileEntry entry{};
        entry.layer = 0; entry.expert = 96 + local; entry.slot = slot;
        directory[w.source] = entry;
    }
    Config c = base_config();
    c.placement = "expert_tiered"; c.deterministic = true;
    c.vram_weights_mib = 10; c.host_pool_mib = 64;
    PlannerInput input{};
    input.program = &f.prog; input.config = &c; input.rank = 1; input.world_size = 4;
    Plan baseline;
    CHECK_OK(Planner(input).plan(&baseline));
    for (const auto& u : baseline.units) if (u.expert >= 0)
        CHECK((u.expert < 3) == (u.tier == Tier::VRAM));

    std::vector<RadFileProfile> global;
    for (int e = 0; e < 384; ++e)
        global.push_back({0, e, e >= 100 && e <= 102 ? float(103-e)/10.0f : 0.0f, 0.0f});
    std::vector<RadFileProfile> local;
    const auto find = [&](const std::string& source) -> const RadFileEntry* {
        const auto at = directory.find(source); return at == directory.end() ? nullptr : &at->second;
    };
    CHECK(local_expert_profile(f.prog.weights, find, global.data(), (int64_t)global.size(),
                              40, 384, &local) == ExpertProfileMapStatus::Complete);
    CHECK_EQ(local.size(), size_t(8));
    input.profile = local.data(); input.n_profile = (int64_t)local.size();
    Plan ranked, repeated;
    CHECK_OK(Planner(input).plan(&ranked));
    CHECK_OK(Planner(input).plan(&repeated));
    CHECK(digest(f.prog, ranked) == digest(f.prog, repeated));
    for (const auto& u : ranked.units) if (u.expert >= 0) {
        CHECK_EQ(u.weights.size(), size_t(2));
        CHECK((u.expert >= 4 && u.expert <= 6) == (u.tier == Tier::VRAM));
        CHECK_EQ(u.bytes, int64_t(3 * MiB));
    }
    CHECK_EQ(ranked.totals.bytes[(int)Tier::VRAM], baseline.totals.bytes[(int)Tier::VRAM]);
    CHECK_EQ(ranked.totals.bytes[(int)Tier::HostPinned], baseline.totals.bytes[(int)Tier::HostPinned]);
    CHECK(!ranked.dynamic);
    // One conflicting GU source must refuse localization transactionally, not rank it silently.
    for (auto& [name, entry] : directory) if (entry.expert == 100 && entry.slot == 0) entry.expert = 200;
    auto preserved = local;
    CHECK(local_expert_profile(f.prog.weights, find, global.data(), (int64_t)global.size(),
                              40, 384, &local) == ExpertProfileMapStatus::ConflictingGroup);
    CHECK_EQ(local.size(), preserved.size());
    for (size_t i = 0; i < local.size(); ++i) {
        CHECK_EQ(local[i].expert, preserved[i].expert);
        CHECK_EQ(local[i].share, preserved[i].share);
    }
}

/* A real sidecar fd and RadFile directory feed the actual parser, tested key mapper,
 * and actual Planner for all four EP ranks. Only tiny synthetic 1x32 expert blocks
 * are written: planned real native GU/DN dimensions are metadata, never allocated
 * as model payload or executed. No Engine, GPU, serving process or real model. */
TEST(decode_profile_file_to_four_rank_planner_checks_local_identity_and_source_generation) {
    struct Files {
        std::string directory, model, profile, link, backup;
        Files() {
            std::string name = radtest::tmpdir() + "/decode_profile.XXXXXX";
            if (::mkdtemp(name.data())) {
                directory = name; model = name + "/metadata.rad";
                profile = name + "/profile.json"; link = name + "/profile-link";
                backup = name + "/metadata-old.rad";
            }
        }
        ~Files() {
            for (const auto* path : {&link, &profile, &model, &backup})
                if (!path->empty()) ::unlink(path->c_str());
            if (!directory.empty()) ::rmdir(directory.c_str());
        }
    } files;
    REQUIRE(!files.directory.empty());
    const auto source_name = [](int layer, int global, int slot) {
        return "blk." + std::to_string(layer) + (slot ? ".ffn_down_exps." : ".ffn_gate_up_exps.") +
               std::to_string(global) + ".weight";
    };
    RadEncoding encoding{};
    rad_enc_copy_str(encoding.scheme, "affine");
    rad_enc_add_plane(&encoding, "codes", RAD_FP4E2M1, 1, 1);
    rad_enc_add_plane(&encoding, "scale", RAD_E8M0, 1, 32);
    {
        RadWriter writer;
        REQUIRE(writer.begin(files.model.c_str()) == RAD_OK);
        writer.set_arch("deepseek41"); writer.set_model_name("profile-pipeline-fixture");
        writer.set_quant("synthetic-native-MXFP4-metadata");
        writer.meta_i("n_layers", 40); writer.meta_i("n_embd", 5120);
        writer.meta_i("n_expert", 384); writer.meta_i("n_expert_used", 6);
        const uint8_t codes[16] = {}; const uint8_t scale = 127;
        const void* planes[2] = {codes, &scale};
        for (int layer = 0; layer < 40; ++layer)
            for (int global = 0; global < 384; ++global)
                for (int slot = 0; slot < 2; ++slot) {
                    RadWriter::Weight w;
                    w.name = source_name(layer, global, slot); w.enc = encoding;
                    w.rank = 2; w.shape[0] = 1; w.shape[1] = 32;
                    w.layer = layer; w.expert = global; w.slot = slot;
                    REQUIRE(writer.add_weight(w, planes) >= 0);
                }
        REQUIRE(writer.finish() == RAD_OK);
    }
    RadFile source;
    REQUIRE(source.open(files.model.c_str()) == RAD_OK);
    REQUIRE(source.entry_count() == 40 * 384 * 2);
    NativeProfileIdentity identity;
    REQUIRE(native_profile_identity(source, &identity) == RAD_OK);
    using Json = nlohmann::json;
    Json doc = {{"format", "radiance-expert-stats/1"}, {"final", true},
        {"model", identity.model}, {"container", identity.container}, {"arch", "deepseek41"},
        {"tp", 4}, {"n_layers", 40}, {"n_trunk_layers", 40}, {"n_experts", 384},
        {"source_identity", {{"format", "rad-metadata-prefix-sha256/1"},
            {"file_bytes", identity.file_bytes}, {"metadata_bytes", identity.metadata_bytes},
            {"device", identity.device}, {"inode", identity.inode},
            {"mtime_ns", identity.mtime_ns}, {"ctime_ns", identity.ctime_ns},
            {"sha256", identity.metadata_sha256}}},
        {"tokens", {{"decode", 1}}}, {"steps", {{"decode", 1}}}};
    doc["layer_experts"] = std::vector<int>(40, 384);
    doc["top_k"] = std::vector<int>(40, 6);
    doc["map"] = std::vector<std::string>(40, "container");
    for (const char* key : {"observed_tokens", "histograms", "passes"})
        doc[key]["decode"] = std::vector<int>(40, 1);
    doc["selections"]["decode"] = std::vector<int>(40, 6);
    doc["counts"]["decode"] = std::vector<std::vector<int>>(40, std::vector<int>(384));
    // Deliberately different local hot id on every rank: 1,4,8,12. None is local0.
    for (int layer = 0; layer < 40; ++layer)
        for (int expert : {1, 2, 3, 100, 200, 300}) doc["counts"]["decode"][layer][expert] = 1;
    for (const char* key : {"credited", "unmapped", "unclassified", "dropped"})
        doc["ranks"][key] = std::vector<int>(4, std::strcmp(key, "credited") == 0 ? 40 : 0);
    {
        std::ofstream output(files.profile, std::ios::binary | std::ios::trunc);
        REQUIRE(bool(output)); output << doc.dump(); output.close(); REQUIRE(bool(output));
    }
    NativeDecodeProfile full;
    std::string reason;
    REQUIRE(read_native_decode_profile(files.profile, identity, &full, &reason) == RAD_OK);
    REQUIRE(full.rows.size() == 40 * 384);
    const std::array<int, 4> hot_local{1, 4, 8, 12};
    for (int rank = 0; rank < 4; ++rank) {
        Fixture f;
        build_moe_layers(f, 40, 96, 12533760, 6266880);
        f.prog.ctx.rank = rank; f.prog.ctx.world_size = 4;
        for (size_t i = 1; i < f.prog.weights.size(); ++i) {
            auto& w = f.prog.weights[i];
            if (w.decl.group.expert < 0) continue;
            const int layer = w.decl.group.layer, local = w.decl.group.expert;
            const int slot = w.logical_bytes == 12533760 ? 0 : 1;
            w.decl.group.slot = slot;
            w.source = source_name(layer, rank * 96 + local, slot);
            const auto* entry = source.find(w.source);
            REQUIRE(entry != nullptr);
            w.enc = source.encoding(*entry); w.enc_known = true;
            w.n_sel = 2; w.sel[0] = 0; w.sel[1] = 1;
            w.decl.dtype = RAD_FP4E2M1; w.decl.rank = 2;
            w.decl.shape[0] = slot ? 5120 : 4608; w.decl.shape[1] = slot ? 2304 : 5120;
            w.stored_bytes = slot ? 6266880 : 12533760;
            w.layout_tag = "ggml_mxfp4";
        }
        REQUIRE(native_decode_profile_geometry(f.prog.weights));
        std::vector<RadFileProfile> local;
        REQUIRE(local_expert_profile(f.prog.weights,
            [&](const std::string& name) { return source.find(name); }, full.rows.data(),
            (int64_t)full.rows.size(), 40, 384, &local) == ExpertProfileMapStatus::Complete);
        REQUIRE(local.size() == 40 * 96);
        Config c = base_config(); c.placement = "expert_tiered"; c.deterministic = true; c.tp = 4;
        c.vram_weights_mib = 32;
        // Planner-only virtual budget. No host pool, tensor payload or device is allocated.
        c.host_pool_mib = 128 * 1024;
        PlannerInput input{};
        input.program = &f.prog; input.config = &c; input.rank = rank; input.world_size = 4;
        Plan baseline, ranked;
        REQUIRE(Planner(input).plan(&baseline) == RAD_OK);
        input.profile = local.data(); input.n_profile = (int64_t)local.size();
        REQUIRE(Planner(input).plan(&ranked) == RAD_OK);
        size_t baseline_residents = 0, profile_residents = 0;
        for (const auto& u : baseline.units) if (u.expert >= 0 && u.tier == Tier::VRAM) {
            ++baseline_residents; CHECK_EQ(u.layer, 0); CHECK_EQ(u.expert, 0);
        }
        for (const auto& u : ranked.units) if (u.expert >= 0 && u.tier == Tier::VRAM) {
            ++profile_residents; CHECK_EQ(u.layer, 0); CHECK_EQ(u.expert, hot_local[size_t(rank)]);
            CHECK_EQ(u.weights.size(), size_t(2)); CHECK_EQ(u.bytes, int64_t(18800640));
        }
        CHECK_EQ(baseline_residents, size_t(1)); CHECK_EQ(profile_residents, size_t(1));
        CHECK_EQ(ranked.rank, rank); CHECK(!ranked.dynamic);
        CHECK_EQ(ranked.totals.bytes[(int)Tier::VRAM], baseline.totals.bytes[(int)Tier::VRAM]);
    }
    NativeDecodeProfile unchanged;
    unchanged.decode_tokens = 777; unchanged.rows.push_back({7, 13, 0.25f, 0.0f});
    REQUIRE(::symlink(files.profile.c_str(), files.link.c_str()) == 0);
    CHECK(read_native_decode_profile(files.link, identity, &unchanged, &reason) < 0);
    CHECK_EQ(unchanged.decode_tokens, uint64_t(777)); CHECK_EQ(unchanged.rows.size(), size_t(1));
    NativeProfileIdentity wrong = identity; ++wrong.inode;
    CHECK(read_native_decode_profile(files.profile, wrong, &unchanged, &reason) == RAD_E_FORMAT);
    CHECK_EQ(unchanged.rows[0].expert, 13);
    struct stat before{};
    REQUIRE(::fstat(source.fd_for_metadata_identity(), &before) == 0);
    const timespec times[2] = {{0, UTIME_OMIT}, {before.st_mtim.tv_sec - 1, before.st_mtim.tv_nsec}};
    REQUIRE(::utimensat(AT_FDCWD, files.model.c_str(), times, 0) == 0);
    NativeProfileIdentity changed;
    REQUIRE(native_profile_identity(source, &changed) == RAD_OK);
    CHECK(changed.mtime_ns != identity.mtime_ns);
    CHECK(read_native_decode_profile(files.profile, changed, &unchanged, &reason) == RAD_E_FORMAT);
    CHECK_EQ(unchanged.decode_tokens, uint64_t(777));
    // Same path with a different inode must not be mistaken for the held RadFile.
    REQUIRE(::rename(files.model.c_str(), files.backup.c_str()) == 0);
    const int replacement = ::open(files.model.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    REQUIRE(replacement >= 0); REQUIRE(::close(replacement) == 0);
    CHECK(native_profile_identity(source, &changed) == RAD_E_FORMAT);
}

/* ================================================================== the two placement flags
 *
 * --draft-expert-placement host pins a draft head's routed experts in the host pool before any
 * strategy runs, so they fit beside a trunk that already fills the card. --expert-pin-top keeps
 * the hottest fraction of the routed experts resident, so heat never demotes them. */

/* A trunk of `trunk` routed layers and a draft head of `head` layers past it, `n` experts each. */
void build_drafted(Fixture& f, int trunk, int head, int n, int64_t each) {
    f.prog.meta.n_layers = (uint32_t)trunk;
    f.prog.drafter.kind = RAD_DRAFT_BLOCK;
    f.prog.drafter.depth = 5;
    build_moe_layers(f, trunk + head, n, each, 0);
}

TEST(draft_experts_pin_to_the_host_pool) {
    Fixture f;
    build_drafted(f, 2, 1, 4, 1 * MiB);

    Config c = base_config();
    c.placement = "expert_tiered";
    c.vram_weights_mib = 1024;
    c.host_pool_mib = 64;
    c.draft_expert_placement = "host";
    c.draft_expert_placement_given = true;

    PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0};
    Plan p;
    CHECK_OK(Planner(in).plan(&p));

    int pinned = 0, trunk_vram = 0;
    for (const MoveUnit& mu : p.units) {
        if (mu.expert < 0) continue;
        if (mu.layer >= 2) {
            CHECK(mu.host_forced);
            CHECK(mu.tier == Tier::HostPinned);
            CHECK(mu.site == Site::DeviceZeroCopy);
            CHECK(!mu.movable);
            ++pinned;
        } else if (mu.tier == Tier::VRAM) {
            ++trunk_vram;
        }
    }
    /* The head's four experts, and only those, are forced into the pool. */
    CHECK_EQ(pinned, 4);
    CHECK(trunk_vram > 0);
    CHECK(p.draft_host_bytes >= 4 * MiB);
    CHECK(p.report(f.prog).find("drafter") != std::string::npos);

    /* The footprint agrees: the head is the part of the plane the flag takes out. */
    WeightFootprint wf;
    weight_footprint(f.prog, &wf, TablePlacement::Vram, TablePlacement::Disk, false, false);
    CHECK(wf.draft_experts >= 4 * MiB);
    CHECK(wf.draft_experts < wf.experts);
}

TEST(draft_placement_auto_leaves_the_head_to_the_strategy) {
    Fixture f;
    build_drafted(f, 2, 1, 4, 1 * MiB);

    Config c = base_config();
    c.placement = "all_vram";
    c.vram_weights_mib = 1024;

    PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0};
    Plan p;
    CHECK_OK(Planner(in).plan(&p));
    for (const MoveUnit& mu : p.units)
        if (mu.expert >= 0 && mu.layer >= 2) CHECK(!mu.host_forced);
    CHECK_EQ(p.draft_host_bytes, (int64_t)0);
}

TEST(draft_placement_host_refuses_a_model_with_no_drafter) {
    Fixture f;
    build_moe_layers(f, 2, 4, 1 * MiB, 0);

    Config c = base_config();
    c.placement = "all_vram";
    c.vram_weights_mib = 1024;
    c.host_pool_mib = 64;
    c.draft_expert_placement = "host";
    c.draft_expert_placement_given = true;

    PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0};
    Plan p;
    CHECK_EQ(Planner(in).plan(&p), RAD_E_INVAL);
}

TEST(draft_placement_host_refuses_when_the_pool_is_short) {
    Fixture f;
    build_drafted(f, 1, 1, 8, 4 * MiB);

    Config c = base_config();
    c.placement = "expert_tiered";
    c.vram_weights_mib = 1024;
    c.host_pool_mib = 1;
    c.draft_expert_placement = "host";

    PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0};
    Plan p;
    CHECK_EQ(Planner(in).plan(&p), RAD_E_NOMEM);
    CHECK(p.report(f.prog).find("DID NOT FIT") != std::string::npos);
}

TEST(expert_pin_top_keeps_the_hottest_resident) {
    Fixture f;
    build_moe_layers(f, 2, 8, 1 * MiB, 0);

    /* Expert e of each layer takes share e + 1, so the hottest are the high indices. */
    std::vector<RadFileProfile> prof;
    for (int l = 0; l < 2; ++l)
        for (int e = 0; e < 8; ++e) {
            RadFileProfile r{};
            r.layer = l;
            r.expert = e;
            r.share = (float)(e + 1);
            prof.push_back(r);
        }

    Config c = base_config();
    c.placement = "expert_tiered";
    c.vram_weights_mib = 1024;
    c.host_pool_mib = 1024;
    c.expert_pin_top = 0.25f;

    PlannerInput in{&f.prog, &c, 0, 1, prof.data(), (int64_t)prof.size()};
    Plan p;
    CHECK_OK(Planner(in).plan(&p));

    int pinned = 0, hot_pinned = 0;
    for (const MoveUnit& mu : p.units) {
        if (mu.expert < 0) continue;
        if (mu.pinned) {
            CHECK(mu.tier == Tier::VRAM);
            CHECK(!mu.movable);
            ++pinned;
            if (mu.expert >= 6) ++hot_pinned;
        }
    }
    /* The top quarter of 16 experts: the two hottest of each layer. */
    CHECK_EQ(pinned, 4);
    CHECK_EQ(hot_pinned, 4);
    CHECK(p.report(f.prog).find("pinned hot") != std::string::npos);
}

TEST(expert_pin_top_zero_pins_nothing) {
    Fixture f;
    build_moe_layers(f, 1, 4, 1 * MiB, 0);

    Config c = base_config();
    c.placement = "all_vram";
    c.vram_weights_mib = 1024;
    c.expert_pin_top = 0.0f;

    PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0};
    Plan p;
    CHECK_OK(Planner(in).plan(&p));
    for (const MoveUnit& mu : p.units) CHECK(!mu.pinned);
    CHECK_EQ(p.pinned_units, (int64_t)0);
}

TEST(expert_pin_top_normalized_shares_keep_the_same_set_after_rescaling) {
    const std::vector<float> shares{0.01f, 0.02f, 0.04f, 0.08f, 0.12f, 0.18f, 0.25f, 0.30f};
    for (float scale : {1.0f, 257.0f}) {
        Fixture f;
        build_moe_layers(f, 1, (int)shares.size(), 1 * MiB, 0);
        std::vector<RadFileProfile> prof;
        for (size_t e = 0; e < shares.size(); ++e) {
            RadFileProfile r{};
            r.layer = 0; r.expert = (int)e; r.share = shares[e] * scale;
            prof.push_back(r);
        }
        Config c = base_config();
        c.placement = "expert_tiered";
        c.vram_weights_mib = 1024;
        c.host_pool_mib = 1024;
        c.expert_pin_top = 0.25f;
        PlannerInput in{&f.prog, &c, 0, 1, prof.data(), (int64_t)prof.size()};
        Plan p;
        CHECK_OK(Planner(in).plan(&p));
        CHECK_EQ(p.pinned_units, int64_t(2));
        for (const MoveUnit& mu : p.units) {
            if (mu.expert < 0) continue;
            CHECK_EQ(mu.pinned, mu.expert >= 6);
            if (mu.pinned) CHECK(!mu.movable);
        }
    }
}

TEST(expert_pin_top_leaves_boundary_ties_and_an_unobserved_profile_unpinned) {
    for (bool zero : {false, true}) {
        Fixture f;
        build_moe_layers(f, 1, 8, 1 * MiB, 0);
        const float shares[] = {0.02f, 0.04f, 0.06f, 0.08f, 0.10f, 0.20f, 0.20f, 0.30f};
        std::vector<RadFileProfile> prof;
        for (int e = 0; e < 8; ++e) {
            RadFileProfile r{};
            r.layer = 0; r.expert = e; r.share = zero ? 0.0f : shares[e];
            prof.push_back(r);
        }
        Config c = base_config();
        c.placement = "expert_tiered";
        c.vram_weights_mib = 1024;
        c.host_pool_mib = 1024;
        c.expert_pin_top = 0.25f;
        PlannerInput in{&f.prog, &c, 0, 1, prof.data(), (int64_t)prof.size()};
        Plan p;
        CHECK_OK(Planner(in).plan(&p));
        CHECK_EQ(p.pinned_units, zero ? int64_t(0) : int64_t(1));
        for (const MoveUnit& mu : p.units)
            if (mu.expert >= 0) CHECK_EQ(mu.pinned, !zero && mu.expert == 7);
    }
}

TEST(expert_pin_top_all_requested_pins_every_resident_including_zero_share) {
    for (float fraction : {1.0f, 0.99f}) {
        Fixture f;
        build_moe_layers(f, 1, 4, 1 * MiB, 0);
        Config c = base_config();
        c.placement = "expert_tiered";
        c.vram_weights_mib = 1024;
        c.host_pool_mib = 1024;
        c.expert_pin_top = fraction;
        /* No profile: every share is zero. ceil(F * 4) still requests all four. */
        PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0};
        Plan p;
        CHECK_OK(Planner(in).plan(&p));
        CHECK_EQ(p.pinned_units, int64_t(4));
        CHECK(std::isinf(p.pin_threshold) && p.pin_threshold < 0.0f);
        for (const MoveUnit& mu : p.units)
            if (mu.expert >= 0) CHECK(mu.pinned && !mu.movable && mu.tier == Tier::VRAM);
    }
}

TEST(a_pinned_resident_is_never_demoted) {
    Fixture f;
    build_moe_layers(f, 1, 4, 1 * MiB, 0);

    std::vector<RadFileProfile> prof;
    for (int e = 0; e < 4; ++e) {
        RadFileProfile r{};
        r.layer = 0;
        r.expert = e;
        r.share = (float)(e + 1);
        prof.push_back(r);
    }
    Config c = base_config();
    c.placement = "expert_tiered";
    c.vram_weights_mib = 1024;
    c.host_pool_mib = 1024;
    c.expert_pin_top = 0.5f;

    PlannerInput in{&f.prog, &c, 0, 1, prof.data(), (int64_t)prof.size()};
    Plan p;
    CHECK_OK(Planner(in).plan(&p));

    /* Force one pinned unit and one cold unit onto the host, so the engine has a swap to make,
     * and heat the cold one far above the pinned one. */
    int32_t pinned_u = -1, cold_u = -1;
    for (size_t u = 0; u < p.units.size(); ++u) {
        if (p.units[u].pinned && pinned_u < 0) pinned_u = (int32_t)u;
        if (p.units[u].expert == 0) cold_u = (int32_t)u;
    }
    CHECK(pinned_u >= 0 && cold_u >= 0);
    p.units[(size_t)cold_u].tier = Tier::HostPinned;
    p.units[(size_t)cold_u].movable = true;

    HeatEngine h;
    HeatParams hp;
    hp.warmup_dispatches = 0;
    CHECK_OK(h.init(p, hp));
    const int32_t ids[] = {0};
    const int32_t counts[] = {1000};
    for (int i = 0; i < 8; ++i) h.observe_ids(0, ids, counts, 1);
    Swap sw{};
    if (h.pick(&sw)) CHECK(sw.demote != pinned_u);
}

TEST(the_draft_floor_charges_only_the_trunk) {
    WeightFootprint wf;
    wf.experts = 100;
    wf.draft_experts = 30;
    wf.expert_unit_max = 5;
    wf.expert_spare = 10;
    /* With the drafter hosted, the pool's room serves the trunk alone. */
    const int64_t hosted = expert_vram_floor_draft_hosted(wf, 80);
    const int64_t plain = expert_vram_floor(wf, 80);
    CHECK(hosted < plain);
    CHECK(hosted <= wf.experts - wf.draft_experts + wf.expert_spare);
    /* No drafter: the two floors agree. */
    wf.draft_experts = 0;
    CHECK_EQ(expert_vram_floor_draft_hosted(wf, 80), expert_vram_floor(wf, 80));
}

namespace {
Program bank_placement_program(int rank) {
    Program p = dspark_scope_fixture::make(rank);
    g_dev_row.info = &g_dev_info;
    for (auto& op : p.ops) {
        Band b; b.hi = 1 << 20; b.dom[RAD_DOMAIN_DEVICE].row = &g_dev_row;
        op.bands.push_back(b);
        for (auto w : op.weights) {
            p.weights[w].align = RAD_ALIGN_UNIT;
            p.weights[w].first_use_op = p.weights[w].last_use_op = op.index;
        }
    }
    WeightInfo trunk;
    trunk.name = trunk.source = "bank_test.trunk";
    trunk.decl.dtype = RAD_U8; trunk.decl.rank = 1; trunk.decl.shape[0] = 4096;
    trunk.decl.access = RAD_ACCESS_PER_TOKEN;
    trunk.decl.group.layer = 0; trunk.decl.group.expert = -1;
    trunk.stored_bytes = 4096; trunk.align = RAD_ALIGN_UNIT;
    trunk.first_use_op = trunk.last_use_op = int(p.ops.size());
    OpInfo op; op.index = int(p.ops.size()); op.op = "gemm";
    op.weights.push_back(rad_weight(p.weights.size())); p.weights.push_back(std::move(trunk));
    Band b; b.hi = 1 << 20; b.dom[RAD_DOMAIN_DEVICE].row = &g_dev_row;
    op.bands.push_back(b); p.ops.push_back(std::move(op));
    return p;
}
Config bank_placement_config() {
    Config c; c.deterministic = true; c.placement = "all_vram";
    c.vram_weights_mib = 8192; c.host_pool_mib = 4096;
    c.draft_bank = "synthetic-head.rad";
    c.tp=4;c.max_seqs=1;c.max_tok=16;c.n_spec=5;c.kv_cache_dtype="bf16";
    return c;
}
}
TEST(dspark_bank_host_placement_charges_existing_pool_and_preserves_trunk_on_all_ranks) {
    for (int rank = 0; rank < 4; ++rank) {
        Program prog = bank_placement_program(rank);
        Config cfg = bank_placement_config();
        cfg.draft_bank_expert_placement = "host";
        PlannerInput in{&prog, &cfg, rank, 4, nullptr, 0};
        Plan p; REQUIRE(Planner(in).plan(&p) == RAD_OK);
        int64_t charged = 0; int pairs = 0;
        for (const auto& mu : p.units) {
            if (mu.layer >= 40 && mu.expert >= 0) {
                CHECK(mu.tier == Tier::HostPinned); CHECK(mu.site == Site::DeviceZeroCopy);
                CHECK(mu.host_forced); CHECK(!mu.movable); CHECK_EQ(mu.slab_slot, -1);
                CHECK_EQ(mu.weights.size(), size_t(2)); charged += mu.slot_bytes; ++pairs;
                for (auto w : mu.weights) {
                    CHECK(p.weight[w].tier == Tier::HostPinned);
                    CHECK(p.weight[w].site == Site::DeviceZeroCopy);
                }
            } else {
                CHECK(mu.tier == Tier::VRAM); CHECK(!mu.host_forced);
            }
        }
        CHECK_EQ(pairs, 96); CHECK(charged >= int64_t(1804861440));
        CHECK_EQ(p.host_budget, int64_t(4096) * MiB - charged);
        CHECK_EQ(cfg.host_pool_mib, int64_t(4096));
    }
}
TEST(dspark_bank_default_does_not_force_or_charge_host_experts) {
    Program prog = bank_placement_program(0); Config cfg = bank_placement_config();
    CHECK(cfg.draft_bank_expert_placement == "auto");
    PlannerInput in{&prog, &cfg, 0, 4, nullptr, 0}; Plan p;
    REQUIRE(Planner(in).plan(&p) == RAD_OK);
    CHECK_EQ(p.host_budget, int64_t(4096) * MiB);
    for (const auto& mu : p.units) { CHECK(mu.tier == Tier::VRAM); CHECK(!mu.host_forced); }
}
TEST(dspark_bank_host_refuses_insufficient_budget_without_partial_pinning_or_growth) {
    Program prog = bank_placement_program(0); Config cfg = bank_placement_config();
    cfg.draft_bank_expert_placement = "host"; cfg.host_pool_mib = 128;
    PlannerInput in{&prog, &cfg, 0, 4, nullptr, 0}; Plan p;
    CHECK_EQ(Planner(in).plan(&p), RAD_E_NOMEM);
    CHECK_EQ(cfg.host_pool_mib, int64_t(128)); CHECK_EQ(p.host_budget, int64_t(128) * MiB);
    for (const auto& mu : p.units) CHECK(!mu.host_forced);
}
TEST(dspark_bank_host_refuses_wrong_declaration_and_missing_bank) {
    for (int bad = 0; bad < 4; ++bad) {
        Program prog = bank_placement_program(0); Config cfg = bank_placement_config();
        cfg.draft_bank_expert_placement = "host";
        if (bad == 0) cfg.draft_bank.clear();
        if (bad == 1) prog.ctx.max_spec = 4;
        if (bad == 2) prog.weights[1].layout_tag = "other";
        if (bad == 3) prog.ops.front().bands.clear();
        PlannerInput in{&prog, &cfg, 0, 4, nullptr, 0}; Plan p;
        CHECK_EQ(Planner(in).plan(&p), RAD_E_UNSUPPORTED);
        for (const auto& mu : p.units) CHECK(!mu.host_forced);
    }
}

namespace {
/* Real ordinary files + actual HOST Mover transport; synthetic group attributes deliberately
 * differ from native DSpark. This proves only source ownership/IO/cache publication, not a full
 * donor, learned MTP, native math or a TP4 model. No finish_coverage bypass exists in Engine. */
struct BankIoTemp {
    std::string dir;
    BankIoTemp() {
        std::string pattern = radtest::tmpdir() + "/bank-mover-XXXXXX";
        char* got = ::mkdtemp(pattern.data());
        if (!got) throw std::runtime_error("bank Mover mkdtemp");
        dir = got;
    }
    ~BankIoTemp() { std::error_code ec; std::filesystem::remove_all(dir, ec); }
    std::string path(const char* name) const { return dir + "/" + name; }
};
void bank_io_good(int st) { if (st != RAD_OK) throw std::runtime_error(rad_strerror(st)); }
void bank_io_metadata(RadWriter& w) {
    w.set_arch("deepseek41"); w.set_model_name("synthetic-bank-Mover"); w.set_quant("");
    w.meta_i("n_layers",40); w.meta_i("n_embd",5120); w.meta_i("n_ff",2304);
    w.meta_i("n_expert",384); w.meta_i("n_expert_used",6); w.meta_i("n_vocab",129280);
    w.meta_i("num_nextn_predict_layers",3); w.meta_i("dspark_block_size",5);
    w.meta_i("dspark_n_routed_experts",128); w.meta_i("dspark_num_experts_per_tok",3);
}
void bank_io_add(RadWriter& writer, const std::string& name, const std::vector<uint8_t>& bytes) {
    RadWriter::Weight w; w.name = name; w.enc = rad_enc_plain(RAD_U8);
    w.rank = 1; w.shape[0] = int64_t(bytes.size());
    const void* planes[] = {bytes.data()};
    if (writer.add_weight(w, planes) < 0) throw std::runtime_error("bank Mover write tiny weight");
}
struct BankIoFixture {
    BankIoTemp temp;
    Fixture f;
    RadFile main, donor;
    DsparkRadBanks banks;
    std::vector<std::vector<uint8_t>> payload;
    Plan plan;
    BankIoFixture() {
        build_moe_layers(f,3,2,768*KiB+17,256*KiB-17);
        payload.resize(f.prog.weights.size());
        for (size_t h=1;h<f.prog.weights.size();++h) {
            auto& w=f.prog.weights[h];
            const bool draft=w.decl.group.expert>=0&&(w.decl.group.expert&1);
            w.source=draft?"dspark.synthetic."+w.name:w.name;
            w.enc=rad_enc_plain(RAD_U8); w.stored_bytes=w.logical_bytes;
            w.n_sel=1; w.sel[0]=0; w.sel_rows[0]=1; w.sel_cols[0]=w.stored_bytes;
            payload[h].resize(size_t(w.stored_bytes));
            for(size_t j=0;j<payload[h].size();++j)
                payload[h][j]=uint8_t(container_byte(int64_t(j)+int64_t(h)*1048583)^(draft?0xa5:0x3c));
        }
        {
            RadWriter writer; bank_io_good(writer.begin(temp.path("main.rad").c_str()));
            bank_io_metadata(writer);
            for(size_t h=1;h<f.prog.weights.size();++h) {
                const auto& w=f.prog.weights[h];
                if(w.decl.group.expert<0||!(w.decl.group.expert&1))bank_io_add(writer,w.source,payload[h]);
            }
            bank_io_good(writer.finish()); bank_io_good(main.open(temp.path("main.rad").c_str()));
        }
        {
            RadWriter writer; bank_io_good(writer.begin(temp.path("draft.rad").c_str()));
            bank_io_metadata(writer); std::string digest;
            bank_io_good(DsparkRadBanks::metadata_digest(main,&digest));
            writer.meta_i("draft_bank.version",1); writer.meta_s("draft_bank.kind","deepseek41.dspark");
            writer.meta_s("draft_bank.parent_metadata_sha256",digest);
            writer.meta_i("draft_bank.parent_file_bytes",main.size());
            writer.meta_s("draft_bank.parent_payload_sha256",std::string(64,'a'));
            writer.meta_s("draft_bank.source_identity","synthetic-only-not-a-model-payload-proof");
            bank_io_add(writer,"dspark.synthetic.padding",payload[1]); // same plane alignment as canonical embedding
            for(size_t h=1;h<f.prog.weights.size();++h) {
                const auto& w=f.prog.weights[h];
                if(w.decl.group.expert>=0&&(w.decl.group.expert&1))bank_io_add(writer,w.source,payload[h]);
            }
            bank_io_good(writer.finish()); bank_io_good(donor.open(temp.path("draft.rad").c_str()));
        }
        bank_io_good(banks.bind(main,&donor)); bank_io_good(banks.bind_weights(f.prog.weights));
        for(size_t h=1;h<f.prog.weights.size();++h) {
            auto& w=f.prog.weights[h]; DsparkRadBanks::Entry src;
            bank_io_good(banks.bound(w,&src));
            w.file_offset=src.file->plane(*src.entry,0).offset;
            w.file_plane_offset[0]=w.file_offset;
        }
        Config cfg=base_config(); cfg.placement="expert_tiered";cfg.deterministic=true;
        cfg.vram_weights_mib=6;cfg.host_pool_mib=133;cfg.weights_disk_tier=true;
        PlannerInput in{&f.prog,&cfg,0,1,nullptr,0};
        bank_io_good(Planner(in).plan(&plan));
    }
    std::vector<int32_t> pair() const {
        for(int layer=0;layer<3;++layer) {
            std::vector<int32_t> units;
            for(size_t u=0;u<plan.units.size();++u)
                if(plan.units[u].layer==layer&&plan.units[u].expert>=0&&plan.units[u].tier==Tier::SSD)
                    units.push_back(int32_t(u));
            if(units.size()==2)return units;
        }
        throw std::runtime_error("bank Mover fixture requires actual two-owner SSD pair");
    }
};
struct BankIoTransport {
    // The owners in BankIoFixture outlive this whole stream/Mover scope.
    RadStream compute=nullptr; RadEvent done=nullptr; std::unique_ptr<Mover> mover;
    ~BankIoTransport() {
        if(mover&&mover->transfer_stream()&&rad_stream_sync(mover->transfer_stream())<0)std::abort();
        if(compute&&rad_stream_sync(compute)<0)std::abort();
        mover.reset(); if(done)rad_event_destroy(done); if(compute)rad_stream_destroy(compute);
    }
    void start(BankIoFixture& f,int64_t cache) {
        bank_io_good(rad_stream_create(&compute,0)); bank_io_good(rad_event_create(&done));
        mover=std::make_unique<Mover>(); MoverConfig c;
        c.container_path=f.main.path();c.source_banks=&f.banks;c.native_file_cache_bytes=cache;
        bank_io_good(mover->init(f.f.prog,f.plan,c));
        bank_io_good(mover->bind_compute_stream(compute));
    }
    void pass(BankIoFixture& f,const std::vector<int32_t>& units,NativeFileCacheReadPolicy policy=NativeFileCacheReadPolicy::Normal) {
        std::vector<int32_t> staged;
        bank_io_good(mover->stage_file_units(units,mover->file_buffer(0),mover->file_buffer_bytes(),nullptr,done,&staged,nullptr,nullptr,0,nullptr,policy));
        REQUIRE(staged==units); bank_io_good(rad_event_wait(compute,done));
        for(auto u:units)for(auto h:f.plan.units[size_t(u)].weights) {
            const auto* slot=mover->residency()->lookup(h); REQUIRE(slot&&slot->ptr);
            std::vector<uint8_t> got(f.payload[h].size());
            bank_io_good(rad_memcpy_async(got.data(),slot->ptr,int64_t(got.size()),compute));
            bank_io_good(rad_stream_sync(compute)); CHECK(got==f.payload[h]);
        }
        bank_io_good(rad_stream_sync(mover->transfer_stream()));
        mover->unstage_units(staged);
    }
};
struct BankIoLegacyCacheEnv {
    const char* key="RADIANCE_EXPERIMENTAL_NATIVE_CACHE_MAPPED_SOURCE";
    bool present; std::string old;
    BankIoLegacyCacheEnv():present(std::getenv(key)!=nullptr),old(present?std::getenv(key):"") {
        if(::setenv(key,"0",1))throw std::runtime_error("bank fixture cache selection");
    }
    ~BankIoLegacyCacheEnv() { if(present)::setenv(key,old.c_str(),1);else ::unsetenv(key); }
};
void bank_io_case(bool cached) {
    REQUIRE(device_is_host());
    BankIoLegacyCacheEnv legacy; // explicit legacy-cache control; mapped-native shape proof is separate

    BankIoFixture f; const auto units=f.pair();
    const auto& a=f.plan.units[size_t(units[0])];const auto& b=f.plan.units[size_t(units[1])];
    REQUIRE_EQ(f.f.prog.weights[a.weights[0]].file_id,uint32_t(0));
    REQUIRE_EQ(f.f.prog.weights[b.weights[0]].file_id,uint32_t(1));
    // Equal numeric file offsets from different immutable owners must never be coalesced or aliased.
    REQUIRE_EQ(f.f.prog.weights[a.weights[0]].file_offset,f.f.prog.weights[b.weights[0]].file_offset);
    BankIoTransport t;t.start(f,cached?4*MiB:0);
    t.pass(f,units);const auto before=t.mover->stats();
    t.pass(f,units,cached?NativeFileCacheReadPolicy::ReadOnly:NativeFileCacheReadPolicy::Normal);
    if(cached)CHECK_EQ(t.mover->stats().file_prefill_ro_hits-before.file_prefill_ro_hits,uint64_t(4));
    else CHECK(t.mover->stats().ssd_preads>before.ssd_preads);
    if(cached) {
        const auto old=t.mover->stats();
        const int fd=::open(f.donor.path(),O_WRONLY|O_CLOEXEC);REQUIRE(fd>=0);
        const int changed=::ftruncate(fd,off_t(f.donor.size()+1));const int closed=::close(fd);
        REQUIRE_EQ(changed,0);REQUIRE_EQ(closed,0);
        std::vector<int32_t> staged;
        CHECK_EQ(t.mover->stage_file_units({units[0]},t.mover->file_buffer(0),t.mover->file_buffer_bytes(),nullptr,t.done,&staged),RAD_E_STATE);
        CHECK_EQ(t.mover->stats().h2d_bytes,old.h2d_bytes); CHECK(staged.empty());
    }
}
}
TEST(dspark_two_bank_actual_HOST_mover_direct_extents_keep_equal_offsets_separate) { bank_io_case(false); }
TEST(dspark_two_bank_actual_HOST_mover_cache_hit_checks_donor_epoch_for_main_source) { bank_io_case(true); }
TEST(dspark_two_bank_actual_HOST_mover_refuses_wrong_open_owner_and_stale_generation) {
    REQUIRE(device_is_host()); BankIoFixture f;
    {
        Mover m;MoverConfig c;c.container_path=f.donor.path();c.source_banks=&f.banks;
        CHECK_EQ(m.init(f.f.prog,f.plan,c),RAD_E_STATE);
    }
    auto& w=f.f.prog.weights[1];++w.source_bank_generation;
    Mover m;MoverConfig c;c.container_path=f.main.path();c.source_banks=&f.banks;
    CHECK_EQ(m.init(f.f.prog,f.plan,c),RAD_E_STATE);
}

namespace {
//Pure actual Planner regression: native-sized metadata only, no allocations or payload IO.
//The forced heads deliberately retain identity=false/no-relayout, so losing their
//preplacement cannot silently fall back to a synthetic SSD encoding.
Program bank_expert_tiered_program(int rank) {
    Program p = bank_placement_program(rank);
    for (int layer=0; layer<40; ++layer) for (int slot=0; slot<2; ++slot) {
        OpInfo op; op.op="moe_gemm_ggml"; op.index=int32_t(p.ops.size());
        op.base.set_i("n_expert",96); op.base.set_i("top_k",6);
        op.base.set_i("N",slot?5120:4608); op.base.set_i("K",slot?2304:5120);
        op.base.set_s("type","mxfp4"); op.base.set_s("dtype","bf16");
        op.base.set_s("a_order",slot?"sorted":"token");
        Band band; band.hi=1<<20; band.dom[RAD_DOMAIN_DEVICE].row=&g_dev_row;
        op.bands.push_back(band);
        for (int expert=0; expert<96; ++expert) {
            WeightInfo w;
            w.name=w.source="blk."+std::to_string(layer)+
                (slot?".ffn_down_exps.":".ffn_gate_up_exps.")+std::to_string(rank*96+expert)+".weight";
            w.decl.dtype=RAD_U8; w.decl.rank=2;
            w.decl.shape[0]=slot?5120:4608; w.decl.shape[1]=slot?2304:5120;
            w.decl.access=RAD_ACCESS_CONDITIONAL; w.decl.shard=RAD_SHARD_NONE;
            w.decl.group.layer=layer; w.decl.group.expert=expert; w.decl.group.slot=slot;
            w.stored_bytes=slot?6266880:12533760; w.align=RAD_ALIGN_UNIT;
            w.layout_tag="ggml_mxfp4"; w.identity=true;
            w.first_use_op=w.last_use_op=op.index;
            op.weights.push_back(rad_weight(p.weights.size())); p.weights.push_back(std::move(w));
        }
        p.ops.push_back(std::move(op));
    }
    return p;
}
Config bank_expert_tiered_config() {
    Config cfg=bank_placement_config();
    cfg.placement="expert_tiered"; cfg.vram_weights_mib=4096;
    cfg.host_pool_mib=4096; cfg.weights_disk_tier=true;
    cfg.draft_bank_expert_placement="host";
    return cfg;
}
void check_bank_expert_tiered_forced_heads(const Program& prog,const Config& cfg,const Plan& p) {
    constexpr int64_t P=18800640;
    int heads=0,main_vram=0,main_pinned=0,main_ssd=0;
    int64_t head_bytes=0;
    CHECK(p.strategy==Strategy::ExpertTiered); CHECK(!p.dynamic); CHECK(p.fits());
    for (const auto&mu:p.units) {
        if (mu.expert<0) continue;
        if (mu.layer>=40) {
            ++heads; head_bytes+=mu.slot_bytes;
            CHECK(mu.layer<43); CHECK(mu.expert>=0&&mu.expert<32);
            CHECK(mu.host_forced); CHECK(mu.preplaced()); CHECK(!mu.movable);
            CHECK(mu.tier==Tier::HostPinned); CHECK(mu.site==Site::DeviceZeroCopy);
            CHECK_EQ(mu.slab_slot,-1); CHECK_EQ(mu.slot_bytes,P); CHECK_EQ(mu.weights.size(),size_t(2));
            for (auto h:mu.weights) {
                CHECK(prog.weights[h].decl.access==RAD_ACCESS_CONDITIONAL);
                CHECK(!prog.weights[h].identity); //not a file-tier escape hatch
                CHECK(p.weight[h].tier==Tier::HostPinned);
                CHECK(p.weight[h].site==Site::DeviceZeroCopy);
            }
        } else {
            CHECK(!mu.host_forced);
            if (mu.tier==Tier::VRAM) ++main_vram;
            else if (mu.tier==Tier::HostPinned) ++main_pinned;
            else if (mu.tier==Tier::SSD) ++main_ssd;
            else CHECK(false);
        }
    }
    CHECK_EQ(heads,96); CHECK_EQ(head_bytes,96*P); CHECK_EQ(p.forced_host_used,96*P);
    CHECK_EQ(main_vram+main_pinned+main_ssd,40*96);
    CHECK(main_vram>0); CHECK(main_pinned>0); CHECK(main_ssd>0);
    CHECK(p.file_layer_max>0); CHECK(p.file_stage_vram>0);
    CHECK_EQ(p.host_budget,cfg.host_pool_mib*MiB-p.file_stage_reserve-96*P);
    CHECK(p.totals.bytes[int(Tier::HostPinned)]+p.file_stage_reserve<=cfg.host_pool_mib*MiB);
    CHECK(p.totals.bytes[int(Tier::VRAM)]+p.slab_overhead+p.file_stage_vram<=cfg.vram_weights_mib*MiB);
    CHECK_EQ(cfg.host_pool_mib,int64_t(4096));
}
}
TEST(dspark_bank_expert_tiered_preserves_forced_heads_with_SSD_trunk_all_ranks) {
    for (int rank=0; rank<4; ++rank) {
        Program prog=bank_expert_tiered_program(rank); Config cfg=bank_expert_tiered_config();
        PlannerInput input{&prog,&cfg,rank,4,nullptr,0}; input.container=true;
        Plan p; REQUIRE(Planner(input).plan(&p)==RAD_OK);
        check_bank_expert_tiered_forced_heads(prog,cfg,p);
    }
}
TEST(dspark_bank_expert_tiered_hot_head_profile_cannot_override_forced_placement) {
    for (int rank=0; rank<4; ++rank) {
        Program prog=bank_expert_tiered_program(rank); Config cfg=bank_expert_tiered_config();
        PlannerInput input{&prog,&cfg,rank,4,nullptr,0}; input.container=true;
        Plan cold; REQUIRE(Planner(input).plan(&cold)==RAD_OK);
        std::vector<RadFileProfile> only_heads;
        for (int layer=40;layer<43;++layer) for (int expert=0;expert<32;++expert) {
            RadFileProfile row{}; row.layer=layer; row.expert=expert; row.share=1.0f/32.0f;
            only_heads.push_back(row);
        }
        input.profile=only_heads.data(); input.n_profile=int64_t(only_heads.size());
        Plan hot; REQUIRE(Planner(input).plan(&hot)==RAD_OK);
        check_bank_expert_tiered_forced_heads(prog,cfg,hot);
        REQUIRE(cold.units.size()==hot.units.size());
        for (size_t u=0;u<cold.units.size();++u) {
            CHECK(cold.units[u].tier==hot.units[u].tier); CHECK(cold.units[u].site==hot.units[u].site);
            CHECK_EQ(cold.units[u].movable,hot.units[u].movable);
            CHECK_EQ(cold.units[u].host_forced,hot.units[u].host_forced);
            if (hot.units[u].layer>=40) CHECK_EQ(hot.units[u].profile_share,1.0f/32.0f);
        }
        CHECK_EQ(cold.host_budget,hot.host_budget); CHECK_EQ(cold.file_layer_max,hot.file_layer_max);
        CHECK_EQ(cold.file_stage_vram,hot.file_stage_vram);
    }
}

TEST(dspark_bank_manual_collision_refuses_before_any_direct_plan_or_budget_mutation) {
    Program prog=bank_placement_program(0);
    for(bool explicit_auto : {false,true}) {
        Config cfg=bank_placement_config();cfg.draft_bank_expert_placement="host";
        cfg.draft_expert_placement=explicit_auto?"auto":"host";
        cfg.draft_expert_placement_given=explicit_auto;
        Plan out;out.host_budget=123;out.vram_budget=456;out.notes.push_back("unchanged");
        PlannerInput in{&prog,&cfg,0,4,nullptr,0};
        CHECK_EQ(Planner(in).plan(&out),RAD_E_INVAL);
        CHECK_EQ(out.host_budget,int64_t(123));CHECK_EQ(out.vram_budget,int64_t(456));
        CHECK(out.units.empty());CHECK_EQ(out.notes.size(),size_t(1));CHECK(out.notes[0]=="unchanged");
        CHECK_EQ(cfg.host_pool_mib,int64_t(4096));
    }
}

/* ================================================================== packed VRAM residents
 *
 * RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED (core/place/vram_packed.h): the expert fill charges a static
 * resident what the card holds of it -- its weights in the packed-scale form where the scan allows -- and
 * the mover places and publishes that layout. Without the callback, or in a dynamic plan, nothing
 * changes. */
namespace {

/* DeepSeek-V4.1's served unit: gate_up and down in block_mxfp4 at the ggml layout's 64-byte alignment,
 * and their packed forms with room for 512 and 320 raw rounds (ds41hx::packed_bytes). */
constexpr int64_t kGuBlock = 12533760, kDnBlock = 6266880, kGuPacked = 11999296, kDnPacked = 6001984;

void build_ds41_like(Fixture& f, int layers, int n, int64_t gu, int64_t dn) {
    build_moe_layers(f, layers, n, gu, dn);
    for (size_t w = 1; w < f.prog.weights.size(); ++w) {
        WeightInfo& wi = f.prog.weights[w];
        wi.align = 64;
        if (wi.name.find("ffn_down_exps") != std::string::npos) wi.decl.group.slot = 1;
    }
}

/* RADIANCE_EXPERIMENTAL_DS41_FILE_BUFFER_UNITS for the scope, as the recommended configuration runs. */
struct FileBufferUnitsEnv {
    std::string old;
    bool had = false;
    explicit FileBufferUnitsEnv(const char* v) {
        const char* o = std::getenv("RADIANCE_EXPERIMENTAL_DS41_FILE_BUFFER_UNITS");
        had = o != nullptr;
        if (o) old = o;
        ::setenv("RADIANCE_EXPERIMENTAL_DS41_FILE_BUFFER_UNITS", v, 1);
    }
    ~FileBufferUnitsEnv() {
        if (had) ::setenv("RADIANCE_EXPERIMENTAL_DS41_FILE_BUFFER_UNITS", old.c_str(), 1);
        else ::unsetenv("RADIANCE_EXPERIMENTAL_DS41_FILE_BUFFER_UNITS");
    }
};

/* A stand-in for the engine's callback: every routed weight eligible at `gu` / `dn` bytes packed (slot 0
 * / 1), its scan within budget unless it is in `over` (then 600 raw rounds against its room). Counts the
 * scans. */
struct FakePack {
    const Program* prog = nullptr;
    int64_t gu = kGuPacked, dn = kDnPacked;
    std::vector<rad_weight> over;
    int scans = 0;
    std::function<int64_t(rad_weight, bool, int32_t*, int32_t*)> fn() {
        return [this](rad_weight w, bool scan, int32_t* cap, int32_t* raw) -> int64_t {
            const WeightInfo& wi = prog->weights[w];
            if (wi.decl.group.expert < 0) return 0;
            const bool down = wi.decl.group.slot == 1;
            if (cap) *cap = down ? 320 : 512;
            if (!scan) return down ? dn : gu;
            ++scans;
            const bool bad = std::find(over.begin(), over.end(), w) != over.end();
            if (raw) *raw = bad ? 600 : 3;
            return bad ? 0 : (down ? dn : gu);
        };
    }
};

int vram_residents(const Plan& p) {
    int n = 0;
    for (const MoveUnit& mu : p.units)
        if (mu.expert >= 0 && mu.tier == Tier::VRAM) ++n;
    return n;
}

}  /* namespace */

TEST(packed_vram_residents_are_charged_what_the_card_holds) {
    /* 8 layers of 48 experts at the served sizes, the file tier for the rest and the file buffers capped
     * at one unit, as the recommended configuration runs. */
    FileBufferUnitsEnv cap("1");
    Fixture f;
    build_ds41_like(f, 8, 48, kGuBlock, kDnBlock);
    Config c = base_config();
    c.placement = "expert_tiered";
    c.vram_weights_mib = 4000;
    c.host_pool_mib = 210;   /* the file tier's staging and windows, no room for a pooled expert */
    c.weights_disk_tier = true;
    c.deterministic = true;

    PlannerInput in0{&f.prog, &c, 0, 1, nullptr, 0};
    Plan p0;
    CHECK_OK(Planner(in0).plan(&p0));
    const int r0 = vram_residents(p0);
    CHECK(r0 > 100);
    CHECK_EQ(p0.vram_packed.units, (int64_t)0);

    FakePack fake;
    fake.prog = &f.prog;
    PlannerInput in1{&f.prog, &c, 0, 1, nullptr, 0};
    in1.vram_pack = fake.fn();
    Plan p1;
    CHECK_OK(Planner(in1).plan(&p1));
    CHECK(p1.fits());
    const int r1 = vram_residents(p1);
    /* The same room at 18,001,920 bytes a unit instead of 18,800,640: x1.0444 the residents. */
    const int64_t room = (int64_t)r0 * 18800640;
    CHECK(r1 >= (int)(room / 18001920) - 1);
    CHECK(r1 > r0 + r0 * 4 / 100 - 1);
    CHECK_EQ(p1.vram_packed.units, (int64_t)r1);
    CHECK_EQ(p1.vram_packed.weights, (int64_t)2 * r1);
    CHECK_EQ(p1.vram_packed.gate_up, (int64_t)r1);
    CHECK_EQ(p1.vram_packed.down, (int64_t)r1);
    CHECK_EQ(p1.vram_packed.bytes, (int64_t)r1 * 18001920);
    CHECK_EQ(p1.vram_packed.block_bytes, (int64_t)r1 * 18800640);
    CHECK_EQ(p1.vram_packed.over_budget, (int64_t)0);
    /* Only the units the fill considered were read: the residents and the few the room turned down. */
    CHECK(p1.vram_packed.scanned >= 2 * r1 && p1.vram_packed.scanned <= 2 * r1 + 4);
    CHECK_EQ((int64_t)fake.scans, p1.vram_packed.scanned);
    for (const MoveUnit& mu : p1.units) {
        if (mu.expert < 0) continue;
        if (mu.tier != Tier::VRAM) {
            CHECK_EQ(mu.vram_bytes, (int64_t)0);
            for (rad_weight w : mu.weights) CHECK_EQ(p1.weight[w].vram_bytes, (int64_t)0);
            continue;
        }
        CHECK(!mu.movable);
        CHECK_EQ(mu.vram_bytes, kGuPacked + kDnPacked);
        REQUIRE(mu.vram_offset.size() == 2);
        CHECK_EQ(mu.vram_offset[0], (int64_t)0);
        CHECK_EQ(mu.vram_offset[1], kGuPacked);
        /* The stored layout is untouched: every other tier and the file tier read it. */
        CHECK_EQ(mu.bytes, align_up(kGuBlock, 64) + kDnBlock);
        CHECK_EQ(p1.weight[mu.weights[0]].vram_bytes, kGuPacked);
        CHECK_EQ(p1.weight[mu.weights[0]].vram_pk_cap, 512);
        CHECK_EQ(p1.weight[mu.weights[1]].vram_bytes, kDnPacked);
        CHECK_EQ(p1.weight[mu.weights[1]].vram_pk_cap, 320);
    }
    /* Residents, spares and the file buffers still fit the weights pool. */
    CHECK(p1.totals.bytes[(int)Tier::VRAM] + p1.slab_overhead + p1.file_stage_vram <= p1.vram_budget);
    CHECK(p1.report(f.prog).find("vram packed") != std::string::npos);
}

TEST(a_packed_weight_over_its_budget_stays_block_and_is_charged_so) {
    FileBufferUnitsEnv cap("1");
    Fixture f;
    build_ds41_like(f, 4, 32, kGuBlock, kDnBlock);
    Config c = base_config();
    c.placement = "expert_tiered";
    c.vram_weights_mib = 1500;
    c.host_pool_mib = 210;
    c.weights_disk_tier = true;
    c.deterministic = true;
    /* Every unit ranks alike, so expert 0 of every layer is resident first (rank_units): layer 0
     * expert 0's down is over budget. */
    FakePack fake;
    fake.prog = &f.prog;
    PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0};
    Plan probe;
    CHECK_OK(Planner(in).plan(&probe));
    int32_t u00 = -1;
    for (int32_t u = 0; u < (int32_t)probe.units.size(); ++u)
        if (probe.units[(size_t)u].layer == 0 && probe.units[(size_t)u].expert == 0) u00 = u;
    REQUIRE(u00 >= 0);
    REQUIRE(probe.units[(size_t)u00].tier == Tier::VRAM);
    fake.over.push_back(probe.units[(size_t)u00].weights[1]);
    in.vram_pack = fake.fn();
    Plan p;
    CHECK_OK(Planner(in).plan(&p));
    const MoveUnit& mu = p.units[(size_t)u00];
    REQUIRE(mu.tier == Tier::VRAM);
    /* gate_up packed, down kept block_mxfp4 at its stored size, laid out as the card holds them. */
    CHECK_EQ(mu.vram_bytes, kGuPacked + kDnBlock);
    REQUIRE(mu.vram_offset.size() == 2);
    CHECK_EQ(mu.vram_offset[1], kGuPacked);
    CHECK_EQ(p.weight[mu.weights[0]].vram_bytes, kGuPacked);
    CHECK_EQ(p.weight[mu.weights[1]].vram_bytes, (int64_t)0);
    CHECK_EQ(p.vram_packed.over_budget, (int64_t)1);
    CHECK_EQ(p.vram_packed.weights, 2 * p.vram_packed.units - 1);
    CHECK_EQ(p.vram_packed.raw_max, (int64_t)3);
    CHECK(p.totals.bytes[(int)Tier::VRAM] + p.slab_overhead + p.file_stage_vram <= p.vram_budget);
}

TEST(without_packing_the_plan_is_the_plain_one) {
    /* No callback, a callback that finds nothing eligible, and a dynamic plan with the callback: all
     * three give the plain plan, decision for decision. */
    FileBufferUnitsEnv cap("1");
    Fixture f;
    build_ds41_like(f, 4, 32, kGuBlock, kDnBlock);
    Config c = base_config();
    c.placement = "expert_tiered";
    c.vram_weights_mib = 1500;
    c.host_pool_mib = 210;
    c.weights_disk_tier = true;
    c.deterministic = true;
    PlannerInput in0{&f.prog, &c, 0, 1, nullptr, 0};
    Plan p0;
    CHECK_OK(Planner(in0).plan(&p0));

    int asked = 0;
    PlannerInput in1{&f.prog, &c, 0, 1, nullptr, 0};
    in1.vram_pack = [&](rad_weight, bool, int32_t*, int32_t*) -> int64_t { ++asked; return 0; };
    Plan p1;
    CHECK_OK(Planner(in1).plan(&p1));
    CHECK_EQ(digest(f.prog, p0), digest(f.prog, p1));
    CHECK(asked > 0);
    CHECK_EQ(p1.vram_packed.units, (int64_t)0);
    CHECK_EQ(p1.vram_packed.scanned, (int64_t)0);
    CHECK_EQ(p1.totals.bytes[(int)Tier::VRAM], p0.totals.bytes[(int)Tier::VRAM]);

    c.deterministic = false;
    PlannerInput in2{&f.prog, &c, 0, 1, nullptr, 0};
    Plan d0;
    CHECK_OK(Planner(in2).plan(&d0));
    FakePack fake;
    fake.prog = &f.prog;
    PlannerInput in3{&f.prog, &c, 0, 1, nullptr, 0};
    in3.vram_pack = fake.fn();
    Plan d1;
    CHECK_OK(Planner(in3).plan(&d1));
    CHECK_EQ(digest(f.prog, d0), digest(f.prog, d1));
    CHECK_EQ(fake.scans, 0);
    CHECK_EQ(d1.vram_packed.units, (int64_t)0);
    for (const MoveUnit& mu : d1.units) CHECK_EQ(mu.vram_bytes, (int64_t)0);
}

TEST(the_mover_places_and_publishes_packed_residents_as_the_card_holds_them) {
    /* Small units on the host backend: 12 KiB gate_ups and 6 KiB downs, packed to 11 KiB + 64 and
     * 5 KiB + 64 (each packed size off a 64-byte boundary, so the down's offset shows the alignment). */
    Fixture f;
    build_ds41_like(f, 2, 8, 12 * KiB, 6 * KiB);
    Config c = base_config();
    c.placement = "expert_tiered";
    c.vram_weights_mib = 2;   /* the 1 MiB embedding and every unit */
    c.host_pool_mib = 64;
    c.deterministic = true;
    FakePack fake;
    fake.prog = &f.prog;
    fake.gu = 11 * KiB + 40;
    fake.dn = 5 * KiB + 40;
    PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0};
    in.vram_pack = fake.fn();
    Plan p;
    CHECK_OK(Planner(in).plan(&p));
    CHECK(p.vram_packed.units > 0);

    Mover m;
    MoverConfig mc;
    CHECK_OK(m.init(f.prog, p, mc));
    ResidencyTable* rt = m.residency();
    const int64_t want_dn_off = align_up(fake.gu, 64);
    int packed_units = 0;
    std::vector<std::pair<uintptr_t, int64_t>> spans;
    for (const MoveUnit& mu : p.units) {
        if (mu.expert < 0 || mu.tier != Tier::VRAM) continue;
        const WeightSlot* g = rt->lookup(mu.weights[0]);
        const WeightSlot* d = rt->lookup(mu.weights[1]);
        REQUIRE(g && d && g->ptr && d->ptr);
        CHECK_EQ((uintptr_t)g->ptr % 64, (uintptr_t)0);
        if (mu.vram_bytes > 0) {
            ++packed_units;
            CHECK_EQ((int64_t)((const char*)d->ptr - (const char*)g->ptr), want_dn_off);
            spans.emplace_back((uintptr_t)g->ptr, mu.vram_bytes);
        } else {
            CHECK_EQ((int64_t)((const char*)d->ptr - (const char*)g->ptr), mu.offset[1]);
        }
    }
    CHECK_EQ((int64_t)packed_units, p.vram_packed.units);
    /* Each packed unit starts on its own allocation unit and no two overlap. */
    std::sort(spans.begin(), spans.end());
    for (size_t i = 0; i < spans.size(); ++i) {
        CHECK_EQ(spans[i].first % RAD_ALIGN_UNIT, (uintptr_t)0);
        if (i) CHECK(spans[i - 1].first + (uintptr_t)spans[i - 1].second <= spans[i].first);
    }
}

TEST(the_load_packs_a_weight_and_checks_what_the_card_will_read) {
    /* vram_pack_weight, the loader's packer: a [64, 2304] weight (down's shape: two full rounds and a partial
     * one of 8 blocks a row) with scales at 120..121 and a few raw rounds. */
    const int N = 64, K = 2304, UK = K / 32;
    std::vector<uint8_t> codes((size_t)N * K / 2), scales((size_t)N * UK);
    uint64_t z = 7;
    auto next = [&] { z = z * 6364136223846793005ull + 1442695040888963407ull; return (uint32_t)(z >> 33); };
    for (auto& c : codes) c = (uint8_t)next();
    for (auto& v : scales) v = (uint8_t)(120 + (next() & 1));
    scales[5] = 200;                          /* row 0, round 0: raw */
    scales[(size_t)3 * UK + 70] = 2;          /* row 3, the partial round: raw */
    scales[(size_t)40 * UK + 33] = 255;       /* row 40, round 1: raw */
    const int cap = 4;
    const size_t bytes = ds41hx::packed_bytes(N, K, cap);
    REQUIRE(bytes > 0);
    std::vector<uint8_t> dst(bytes, 0xA5);
    int n_exc = -1;
    CHECK_EQ(vram_pack_weight(codes.data(), scales.data(), N, K, cap, dst.data(), (int64_t)bytes, &n_exc), RAD_OK);
    CHECK_EQ(n_exc, 3);
    ds41hx::Mx v;
    REQUIRE(ds41hx::packed_view(dst.data(), bytes, N, K, &v));
    CHECK(ds41hx::packed_scales_equal(v, scales.data()));
    CHECK_EQ(std::memcmp(v.codes, codes.data(), codes.size()), 0);
    /* The same values as the block form, row for row (the host's reference decode of each layout). */
    std::vector<uint16_t> x(K), y_pk(N), y_pl(N);
    for (auto& e : x) e = (uint16_t)(0x3c00 + (next() & 0x3ff));
    ds41hx::Mx pl;
    pl.N = N; pl.K = K; pl.codes = codes.data(); pl.scales = scales.data();
    ds41hx::gemv_ref(v, x.data(), y_pk.data(), 0, N);
    ds41hx::gemv_ref(pl, x.data(), y_pl.data(), 0, N);
    CHECK(y_pk == y_pl);
    /* More raw rounds than room: refused, nothing claimed. */
    std::vector<uint8_t> small(ds41hx::packed_bytes(N, K, 2));
    CHECK_EQ(vram_pack_weight(codes.data(), scales.data(), N, K, 2, small.data(), (int64_t)small.size(), &n_exc),
             RAD_E_FULL);
    CHECK_EQ(n_exc, 3);
    /* A destination short of the form, and a shape it cannot hold. */
    CHECK_EQ(vram_pack_weight(codes.data(), scales.data(), N, K, cap, dst.data(), (int64_t)bytes - 1, &n_exc),
             RAD_E_SHAPE);
    CHECK_EQ(vram_pack_weight(codes.data(), scales.data(), N, K + 16, cap, dst.data(), (int64_t)bytes, &n_exc),
             RAD_E_SHAPE);
}

TEST(the_vram_packed_switch_and_its_budget_parse) {
    struct Env {
        const char* k;
        std::string old;
        bool had;
        explicit Env(const char* key) : k(key) { const char* o = std::getenv(k); had = o; if (o) old = o; }
        ~Env() { if (had) ::setenv(k, old.c_str(), 1); else ::unsetenv(k); }
    } on("RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED"), raw("RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED_RAW");
    VramPackConfig c;
    ::unsetenv("RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED");
    ::unsetenv("RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED_RAW");
    CHECK_OK(vram_pack_config(&c));
    CHECK(!c.on);
    ::setenv("RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED", "0", 1);
    CHECK_OK(vram_pack_config(&c));
    CHECK(!c.on);
    ::setenv("RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED", "1", 1);
    CHECK_OK(vram_pack_config(&c));
    CHECK(c.on);
    CHECK_EQ(c.raw[0], 512);
    CHECK_EQ(c.raw[1], 320);
    ::setenv("RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED_RAW", "600,3400", 1);
    CHECK_OK(vram_pack_config(&c));
    CHECK_EQ(c.raw[0], 600);
    CHECK_EQ(c.raw[1], 3400);
    ::setenv("RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED_RAW", "600", 1);
    CHECK_EQ(vram_pack_config(&c), RAD_E_INVAL);
    ::setenv("RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED_RAW", "512,320", 1);
    ::setenv("RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED", "yes", 1);
    CHECK_EQ(vram_pack_config(&c), RAD_E_INVAL);
}

/* ================================================================== which weights may be packed
 *
 * vram_pack_eligible decides from the declaration and the container's directory alone, before any scale
 * is read, and it alone keeps every other reader away from a packed weight's bytes: only a trunk routed
 * expert's ggml_mxfp4 gate_up or down, read by moe_gemm_ggml tables and nothing else, every band of each
 * such op resolved to gemv_vp on the card, whole in the container and smaller packed. vram_pack_scan reads
 * one weight's scales and vram_pack_query is the engine's answer to the planner from both. A container
 * written by RadWriter, and a Program declared as deepseek41's routed layers are once the layout pass has
 * chosen libr4d's block_mxfp4: gate_up [64, 1024] (one full round of 32 blocks a row) and down [32, 512]
 * (a partial round of 16). */
namespace {

const RadKernelInfo g_vp_info = { kVramPackRow, "moe_gemm_ggml", "moe", "", "", "",
                                  RAD_DOMAIN_DEVICE, 59, nullptr, 0, nullptr, 0,
                                  nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
const RadKernelInfo g_mx_gemv_info = { "moe_gemm_ggml_mxfp4_gemv", "moe_gemm_ggml", "moe", "", "", "",
                                       RAD_DOMAIN_DEVICE, 58, nullptr, 0, nullptr, 0,
                                       nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
const RadKernelInfo g_mx_tiled_info = { "moe_gemm_ggml_mxfp4_tiled", "moe_gemm_ggml", "moe", "", "", "",
                                        RAD_DOMAIN_DEVICE, 57, nullptr, 0, nullptr, 0,
                                        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
const RadKernelInfo g_mx_p1_info = { ds41p1::kTwinRow, "moe_gemm_ggml", "moe", "", "", "",
                                    RAD_DOMAIN_DEVICE, 57, nullptr, 0, nullptr, 0,
                                    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
const RadKernelInfo g_mx_lcp_info = { ds41p1::kLcpTwinRow, "moe_gemm_ggml", "moe", "", "", "",
                                     RAD_DOMAIN_DEVICE, 57, nullptr, 0, nullptr, 0,
                                     nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
KernelRow g_vp_row, g_mx_gemv_row, g_mx_tiled_row, g_mx_p1_row, g_mx_lcp_row;
const RadOperandSpec g_vp_table_opds[] = {
    { "a", RAD_OPD_IN, 0 }, { "w", RAD_OPD_WTAB, 0 }, { "sorted", RAD_OPD_IN, 0 },
    { "offsets", RAD_OPD_IN, 0 }, { "y", RAD_OPD_OUT, 0 },
};
const RadOperandSpec g_vp_stacked_opds[] = {
    { "a", RAD_OPD_IN, 0 }, { "w", RAD_OPD_WEIGHT, 0 }, { "sorted", RAD_OPD_IN, 0 },
    { "offsets", RAD_OPD_IN, 0 }, { "y", RAD_OPD_OUT, 0 },
};
const RadOpSchema g_vp_table_schema = { "moe_gemm_ggml", nullptr, 0, g_vp_table_opds, 5, "table fixture" };
const RadOpSchema g_vp_stacked_schema = { "moe_gemm_ggml", nullptr, 0, g_vp_stacked_opds, 5, "stacked fixture" };

constexpr int kVpN[2] = { 64, 32 }, kVpK[2] = { 1024, 512 };   /* gate_up, down */

/* `layers` routed layers of `experts` experts, a gate_up and a down each, written in movement-unit order;
 * layers at or past `trunk` (meta.n_layers) are a draft head's. Every scale is 120 or 121 but three blocks
 * of layer 0 expert 0's gate_up, rows 0, 5 and 40: three raw rounds. */
struct VpModel {
    Fixture f;
    RadFile file;
    std::vector<std::vector<uint8_t>> scales;   /* per weight handle, the e8m0 plane written */
    std::vector<std::vector<rad_weight>> w;     /* [layer][2 * expert + slot] */
    bool ok = false;

    VpModel(int layers, int experts, int trunk, const char* name) {
        g_vp_row.info = &g_vp_info;
        g_mx_gemv_row.info = &g_mx_gemv_info;
        g_mx_tiled_row.info = &g_mx_tiled_info;
        g_mx_p1_row.info = &g_mx_p1_info;
        g_mx_lcp_row.info = &g_mx_lcp_info;
        f.prog.meta.n_layers = trunk;
        RadEncoding enc{};
        rad_enc_copy_str(enc.scheme, "affine");
        rad_enc_add_plane(&enc, "codes", RAD_FP4E2M1, 1, 1);
        rad_enc_add_plane(&enc, "scale", RAD_E8M0, 1, 32);
        const std::string path = radtest::tmp_path(name);
        RadWriter writer;
        if (writer.begin(path.c_str()) != RAD_OK) return;
        writer.set_arch("deepseek41");
        writer.meta_i("n_layers", trunk);
        uint64_t z = 11;
        auto next = [&] { z = z * 6364136223846793005ull + 1442695040888963407ull; return (uint32_t)(z >> 33); };
        w.assign((size_t)layers, {});
        for (int l = 0; l < layers; ++l)
            for (int e = 0; e < experts; ++e)
                for (int s = 0; s < 2; ++s) {
                    const int N = kVpN[s], K = kVpK[s];
                    const std::string nm = fmt("blk.%d.ffn_%s_exps.%d.weight", l, s ? "down" : "gate_up", e);
                    const rad_weight h = f.weight(nm, (int64_t)N * (K / 32) * 17, RAD_ACCESS_CONDITIONAL, l, e);
                    WeightInfo& wi = f.prog.weights[h];
                    wi.decl.group.slot = s;
                    wi.source = nm;
                    wi.enc = enc;
                    wi.enc_known = true;
                    wi.n_sel = 2;
                    wi.sel[0] = 0;
                    wi.sel[1] = 1;
                    wi.sel_rows[0] = wi.sel_rows[1] = N;
                    wi.sel_cols[0] = K;
                    wi.sel_cols[1] = K / 32;
                    wi.layout_tag = "ggml_mxfp4";
                    wi.identity = false;   /* the loader relayouts it to block_mxfp4 */
                    wi.stored_bytes = (int64_t)N * (K / 32) * 17;
                    wi.align = 64;
                    std::vector<uint8_t> codes((size_t)N * K / 2), sc((size_t)N * (K / 32));
                    for (auto& c : codes) c = (uint8_t)next();
                    for (auto& v : sc) v = (uint8_t)(120 + (next() & 1));
                    if (l == 0 && e == 0 && s == 0) {
                        sc[5] = 200;
                        sc[(size_t)5 * 32 + 7] = 2;
                        sc[(size_t)40 * 32 + 31] = 255;
                    }
                    RadWriter::Weight rw;
                    rw.name = nm;
                    rw.enc = enc;
                    rw.rank = 2;
                    rw.shape[0] = N;
                    rw.shape[1] = K;
                    rw.layer = l;
                    rw.expert = e;
                    rw.slot = s;
                    const void* planes[2] = { codes.data(), sc.data() };
                    if (writer.add_weight(rw, planes) < 0) return;
                    if (scales.size() <= (size_t)h) scales.resize((size_t)h + 1);
                    scales[(size_t)h] = std::move(sc);
                    w[(size_t)l].push_back(h);
                }
        if (writer.finish() != RAD_OK) return;
        ok = file.open(path.c_str()) == RAD_OK;
    }
    rad_weight at(int l, int e, int s) const { return w[(size_t)l][(size_t)(2 * e + s)]; }
    /* A moe_gemm_ggml op whose weight table (operand 1) is layer l's gate_ups (s 0) or downs (s 1), a band
     * a row: up to 16 tokens, then past them; a null row is a band the card has no kernel for. */
    void op(int l, int s, const std::vector<const KernelRow*>& rows) {
        OpInfo o;
        o.op = "moe_gemm_ggml";
        o.index = (int32_t)f.prog.ops.size();
        for (size_t k = (size_t)s; k < w[(size_t)l].size(); k += 2) o.weights.push_back(w[(size_t)l][k]);
        o.weight_opd.assign(o.weights.size(), 1);
        int64_t hi = 16;
        for (const KernelRow* r : rows) {
            Band b;
            b.hi = hi;
            b.span = hi == 16 ? "M <= 16" : "M in (16, 1048576]";
            if (r) b.dom[RAD_DOMAIN_DEVICE].row = r;
            else b.miss[RAD_DOMAIN_DEVICE] = "no device kernel";
            o.bands.push_back(b);
            hi = 1 << 20;
        }
        f.prog.ops.push_back(o);
    }
    void p1_op(int l, int s, int64_t rows = 2048) {
        op(l, s, { &g_vp_row, &g_mx_p1_row });
        OpInfo& o = f.prog.ops.back();
        o.ranged_key = "M";
        o.range_lo = 1;
        o.range_hi = rows;
        o.schema = &g_vp_table_schema;
        o.bands.back().hi = rows;
        o.bands.back().span = fmt("M in (16, %lld]", (long long)rows);
    }
};

bool listed(const std::vector<rad_weight>& v, rad_weight w) { return std::find(v.begin(), v.end(), w) != v.end(); }

}  /* namespace */

TEST(vram_pack_eligible_packs_only_what_gemv_vp_tables_alone_read) {
    VpModel m(3, 4, 2, "vp_eligible.rad");
    REQUIRE(m.ok);
    /* Layer 0: both ops gemv_vp. Layer 1: the gate_up op has a band past 16 tokens the tiled row took, so
     * that band reads its table as block_mxfp4; the down op is gemv_vp, but expert 3's down is also a
     * dense op's operand. Layer 2: a draft head's experts, though gemv_vp tables read them. */
    m.op(0, 0, { &g_vp_row });
    m.op(0, 1, { &g_vp_row });
    m.op(1, 0, { &g_vp_row, &g_mx_tiled_row });
    m.op(1, 1, { &g_vp_row });
    m.f.op("gemm", { m.at(1, 3, 1) }, false);
    m.op(2, 0, { &g_vp_row });
    m.op(2, 1, { &g_vp_row });
    /* Two the gemv_vp tables alone read that the form cannot take: one stored as the container holds it
     * (the file tier would read it in place), and one this rank holds half of. */
    m.f.prog.weights[m.at(0, 2, 0)].identity = true;
    WeightInfo& half = m.f.prog.weights[m.at(0, 3, 1)];
    half.sel_rows[0] = half.sel_rows[1] = kVpN[1] / 2;
    half.stored_bytes /= 2;

    VramPackConfig cfg;
    cfg.on = true;
    cfg.raw[0] = 16;
    cfg.raw[1] = 4;
    std::vector<VramPackWeight> t;
    std::string why;
    vram_pack_eligible(m.f.prog, m.file, cfg, &t, &why);
    REQUIRE(t.size() == m.f.prog.weights.size());
    const std::vector<rad_weight> want = { m.at(0, 0, 0), m.at(0, 1, 0), m.at(0, 3, 0), m.at(0, 0, 1),
                                           m.at(0, 1, 1), m.at(0, 2, 1), m.at(1, 0, 1), m.at(1, 1, 1),
                                           m.at(1, 2, 1) };
    int packed = 0;
    for (rad_weight h = 1; (size_t)h < t.size(); ++h) {
        const VramPackWeight& v = t[(size_t)h];
        const WeightInfo& wi = m.f.prog.weights[h];
        if (!listed(want, h)) {
            if (v.bytes != 0) std::printf("  %s is eligible and should not be\n", wi.name.c_str());
            CHECK_EQ(v.bytes, (int64_t)0);
            continue;
        }
        ++packed;
        const int s = wi.decl.group.slot;
        CHECK_EQ(v.bytes, (int64_t)ds41hx::packed_bytes(kVpN[s], kVpK[s], cfg.raw[s]));
        CHECK(v.bytes > 0 && v.bytes < wi.stored_bytes);
        CHECK_EQ(v.cap, cfg.raw[s]);
        CHECK_EQ(v.N, kVpN[s]);
        CHECK_EQ(v.K, kVpK[s]);
        const RadFileEntry* e = m.file.find(wi.source);
        REQUIRE(e != nullptr);
        CHECK_EQ(v.scales_off, m.file.plane(*e, 1).offset);
        CHECK_EQ(v.scales_bytes, (int64_t)kVpN[s] * (kVpK[s] / 32));
    }
    CHECK_EQ(packed, 9);
    /* The startup log says why the rest stay block_mxfp4. */
    CHECK(why.find("1 routed moe_gemm_ggml op(s) have a band moe_gemm_ggml_mxfp4_gemv_vp does not serve") !=
          std::string::npos);
    CHECK(why.find("M in (16, 1048576]: moe_gemm_ggml_mxfp4_tiled") != std::string::npos);
    CHECK(why.find("10 of the 19 weights the gemv_vp tables read stay block_mxfp4: 8 not a trunk expert's "
                   "gate_up or down, 1 not ggml_mxfp4 over fp4 codes and e8m0 planes, 1 not whole in the "
                   "container, 0 no smaller packed") != std::string::npos);
    if (why.find("10 of the 19") == std::string::npos) std::printf("  why: %s\n", why.c_str());

    /* Room for so many raw rounds that the form is no smaller than block_mxfp4: nothing packs. */
    cfg.raw[0] = cfg.raw[1] = 64;
    vram_pack_eligible(m.f.prog, m.file, cfg, &t, &why);
    for (const VramPackWeight& v : t) CHECK_EQ(v.bytes, (int64_t)0);
    CHECK(why.find("19 of the 19 weights") != std::string::npos);
    CHECK(why.find("9 no smaller packed") != std::string::npos);

    /* Off: nothing, and nothing to say. */
    cfg.on = false;
    vram_pack_eligible(m.f.prog, m.file, cfg, &t, &why);
    REQUIRE(t.size() == m.f.prog.weights.size());
    for (const VramPackWeight& v : t) CHECK_EQ(v.bytes, (int64_t)0);
    CHECK(why.empty());
}

TEST(vram_pack_eligible_refuses_a_table_any_other_row_or_position_reads) {
    VpModel m(2, 2, 2, "vp_readers.rad");
    REQUIRE(m.ok);
    /* The plain gemv row (the fact unset), a band past 16 tokens the card has no kernel for (the host's),
     * and a table position other than the weight table (operand 2) beside gemv_vp tables. */
    m.op(0, 0, { &g_mx_gemv_row });
    m.op(0, 1, { &g_vp_row, nullptr });
    m.op(1, 0, { &g_vp_row });
    m.f.prog.ops.back().weight_opd[1] = 2;
    m.op(1, 1, { &g_vp_row });
    VramPackConfig cfg;
    cfg.on = true;
    cfg.raw[0] = 16;
    cfg.raw[1] = 4;
    std::vector<VramPackWeight> t;
    std::string why;
    vram_pack_eligible(m.f.prog, m.file, cfg, &t, &why);
    REQUIRE(t.size() == m.f.prog.weights.size());
    const std::vector<rad_weight> want = { m.at(1, 0, 0), m.at(1, 0, 1), m.at(1, 1, 1) };
    for (rad_weight h = 1; (size_t)h < t.size(); ++h) {
        if ((t[(size_t)h].bytes > 0) != listed(want, h))
            std::printf("  %s: %lld bytes packed\n", m.f.prog.weights[h].name.c_str(),
                        (long long)t[(size_t)h].bytes);
        CHECK_EQ(t[(size_t)h].bytes > 0, listed(want, h));
    }
    CHECK(why.find("2 routed moe_gemm_ggml op(s) have a band") != std::string::npos);
    CHECK(why.find("the first: op 0 (M <= 16: moe_gemm_ggml_mxfp4_gemv)") != std::string::npos);

    /* No op reads a weight table: nothing, and the log says so. */
    Fixture empty;
    vram_pack_eligible(empty.prog, m.file, cfg, &t, &why);
    CHECK_EQ(t.size(), (size_t)1);
    CHECK(why.find("no moe_gemm_ggml op reads a weight table") != std::string::npos);
}

TEST(vram_pack_eligible_accepts_packed_aware_p1_bands_at_the_same_sizes_as_decode) {
    VpModel m(1, 3, 1, "vp_p1_eligible.rad");
    REQUIRE(m.ok);
    m.op(0, 0, { &g_vp_row });
    m.op(0, 1, { &g_vp_row });
    VramPackConfig cfg;
    cfg.on = true;
    cfg.raw[0] = 16;
    cfg.raw[1] = 4;
    std::vector<VramPackWeight> decode, p1;
    std::string why;
    vram_pack_eligible(m.f.prog, m.file, cfg, &decode, &why);
    CHECK(why.empty());
    REQUIRE(decode.size() == 7);
    for (size_t w = 1; w < decode.size(); ++w) REQUIRE(decode[w].bytes > 0);

    /* Same weight bytes and raw-round budgets with the exact high-band P1 reader. The final
     * case also splits both proven ranges into more bands; every reader remains packed-aware. */
    for (int64_t rows : { int64_t(17), int64_t(64), int64_t(2048), int64_t(ds41p1::kTwinMaxRows) }) {
        m.f.prog.ops.clear();
        m.p1_op(0, 0, rows);
        m.p1_op(0, 1, rows);
        if (rows == ds41p1::kTwinMaxRows) {
            for (OpInfo& o : m.f.prog.ops) {
                Band low = o.bands.front(), high = o.bands.back();
                low.hi = 8;
                high.hi = 1024;
                o.bands = { low, o.bands.front(), high, o.bands.back() };
            }
        }
        vram_pack_eligible(m.f.prog, m.file, cfg, &p1, &why);
        CHECK(why.empty());
        REQUIRE(p1.size() == decode.size());
        for (size_t w = 1; w < p1.size(); ++w) {
            CHECK_EQ(p1[w].bytes, decode[w].bytes);
            CHECK_EQ(p1[w].cap, decode[w].cap);
            CHECK_EQ(p1[w].N, decode[w].N);
            CHECK_EQ(p1[w].K, decode[w].K);
            CHECK_EQ(p1[w].scales_off, decode[w].scales_off);
            CHECK_EQ(p1[w].scales_bytes, decode[w].scales_bytes);
        }
    }
}

TEST(vram_pack_p1_eligibility_refuses_unproven_bands_and_stacked_weights) {
    VpModel m(1, 2, 1, "vp_p1_refuse.rad");
    REQUIRE(m.ok);
    m.p1_op(0, 0);
    m.p1_op(0, 1);
    const std::vector<OpInfo> good = m.f.prog.ops;
    const KernelRow no_info{};
    struct BadReader {
        const char* name;
        std::function<void(OpInfo&)> change;
    };
    const std::vector<BadReader> cases = {
        { "ordinary LCP twin", [](OpInfo& o) { o.bands[1].dom[RAD_DOMAIN_DEVICE].row = &g_mx_lcp_row; } },
        { "tiled high band", [](OpInfo& o) { o.bands[1].dom[RAD_DOMAIN_DEVICE].row = &g_mx_tiled_row; } },
        { "plain low band", [](OpInfo& o) { o.bands[0].dom[RAD_DOMAIN_DEVICE].row = &g_mx_gemv_row; } },
        { "missing device reader", [](OpInfo& o) { o.bands[1].dom[RAD_DOMAIN_DEVICE].row = nullptr; } },
        { "missing reader metadata", [&](OpInfo& o) { o.bands[1].dom[RAD_DOMAIN_DEVICE].row = &no_info; } },
        { "extra block reader", [](OpInfo& o) {
            Band extra = o.bands.back();
            o.bands.back().hi = 1024;
            extra.dom[RAD_DOMAIN_DEVICE].row = &g_mx_gemv_row;
            o.bands.push_back(extra);
        } },
        { "stacked weight", [](OpInfo& o) { o.schema = &g_vp_stacked_schema; } },
        { "missing table schema", [](OpInfo& o) { o.schema = nullptr; } },
        { "different ranged axis", [](OpInfo& o) { o.ranged_key = "N"; } },
        { "range misses decode row", [](OpInfo& o) { o.range_lo = 2; } },
        { "P1 above its bound", [](OpInfo& o) {
            o.range_hi = o.bands[1].hi = ds41p1::kTwinMaxRows + 1;
        } },
        { "decode band above 16", [](OpInfo& o) { o.bands[0].hi = 17; } },
        { "P1 band starts below 17", [](OpInfo& o) { o.bands[0].hi = 15; } },
        { "no low band", [](OpInfo& o) { o.bands.erase(o.bands.begin()); } },
        { "incomplete high coverage", [](OpInfo& o) { --o.bands[1].hi; } },
        { "overlapping bounds", [](OpInfo& o) { o.bands[1].hi = 16; } },
        { "no bands", [](OpInfo& o) { o.bands.clear(); } },
    };
    VramPackConfig cfg;
    cfg.on = true;
    cfg.raw[0] = 16;
    cfg.raw[1] = 4;
    std::vector<VramPackWeight> t;
    std::string why;
    for (const BadReader& bad : cases) {
        m.f.prog.ops = good;
        bad.change(m.f.prog.ops[0]);
        vram_pack_eligible(m.f.prog, m.file, cfg, &t, &why);
        REQUIRE(t.size() == m.f.prog.weights.size());
        for (int e = 0; e < 2; ++e) {
            if (t[m.at(0, e, 0)].bytes) std::printf("  accepted %s\n", bad.name);
            CHECK_EQ(t[m.at(0, e, 0)].bytes, int64_t(0));
            CHECK(t[m.at(0, e, 1)].bytes > 0);   /* the independent down table is still eligible */
        }
        CHECK(why.find("1 routed moe_gemm_ggml op(s) have a band") != std::string::npos);
    }
}

TEST(vram_pack_p1_tables_keep_other_readers_positions_and_draft_weights_block) {
    VpModel m(2, 4, 1, "vp_p1_shared.rad");
    REQUIRE(m.ok);
    for (int l = 0; l < 2; ++l) {
        m.p1_op(l, 0);
        m.p1_op(l, 1);
    }
    /* P1 does not override an additional block-form reader, the operand's table position,
     * file_id (even for a trunk-shaped bank weight), or the trunk layer bound. */
    m.f.op("gemm", { m.at(0, 0, 0) }, false);
    m.f.prog.ops[0].weight_opd[1] = 2;
    m.f.prog.weights[m.at(0, 2, 0)].file_id = 1;
    VramPackConfig cfg;
    cfg.on = true;
    cfg.raw[0] = 16;
    cfg.raw[1] = 4;
    std::vector<VramPackWeight> t;
    std::string why;
    vram_pack_eligible(m.f.prog, m.file, cfg, &t, &why);
    REQUIRE(t.size() == m.f.prog.weights.size());
    for (int e = 0; e < 4; ++e) {
        CHECK_EQ(t[m.at(0, e, 0)].bytes > 0, e == 3);
        CHECK(t[m.at(0, e, 1)].bytes > 0);
        CHECK_EQ(t[m.at(1, e, 0)].bytes, int64_t(0));
        CHECK_EQ(t[m.at(1, e, 1)].bytes, int64_t(0));
    }
    CHECK(why.find("DSpark bank's") != std::string::npos);
}

TEST(the_scan_reads_a_weights_scales_and_the_query_keeps_one_over_its_room_block) {
    VpModel m(1, 2, 1, "vp_scan.rad");
    REQUIRE(m.ok);
    m.op(0, 0, { &g_vp_row });
    m.op(0, 1, { &g_vp_row });
    VramPackConfig cfg;
    cfg.on = true;
    cfg.raw[0] = 16;
    cfg.raw[1] = 4;
    std::vector<VramPackWeight> t;
    std::string why;
    vram_pack_eligible(m.f.prog, m.file, cfg, &t, &why);
    CHECK(why.empty());
    const rad_weight raw3 = m.at(0, 0, 0), plain = m.at(0, 1, 0), down = m.at(0, 1, 1);
    REQUIRE(t[raw3].bytes > 0 && t[plain].bytes > 0 && t[down].bytes > 0);

    /* The scan: the raw rounds ds41hx::pack_plan finds in the container's scales, left in buf. */
    std::vector<uint8_t> buf;
    CHECK_EQ(vram_pack_scan(m.file, t[raw3], &buf), 3);
    CHECK(buf == m.scales[raw3]);
    CHECK_EQ(vram_pack_scan(m.file, t[plain], &buf), 0);
    CHECK(buf == m.scales[plain]);
    CHECK_EQ(vram_pack_scan(m.file, t[down], &buf), 0);
    CHECK(buf == m.scales[down]);

    /* The planner's question: unscanned, the table's answer and nothing read; scanned, the same within
     * the room. */
    int32_t cap = -9, raw = -9;
    CHECK_EQ(vram_pack_query(m.file, t, raw3, false, &cap, &raw, &buf), t[raw3].bytes);
    CHECK_EQ(cap, 16);
    CHECK_EQ(raw, -9);
    CHECK_EQ(vram_pack_query(m.file, t, raw3, true, &cap, &raw, &buf), t[raw3].bytes);
    CHECK_EQ(raw, 3);
    CHECK_EQ(vram_pack_query(m.file, t, down, true, &cap, &raw, &buf), t[down].bytes);
    CHECK_EQ(cap, 4);
    CHECK_EQ(raw, 0);

    /* Room for two raw rounds: eligible from the declaration, but the scan finds three, so the weight
     * stays block_mxfp4 (0) and says how many it needs; its neighbour packs. */
    cfg.raw[0] = 2;
    std::vector<VramPackWeight> t2;
    vram_pack_eligible(m.f.prog, m.file, cfg, &t2, &why);
    REQUIRE(t2[raw3].bytes > 0);
    CHECK_EQ(vram_pack_query(m.file, t2, raw3, false, &cap, &raw, &buf), t2[raw3].bytes);
    CHECK_EQ(vram_pack_query(m.file, t2, raw3, true, &cap, &raw, &buf), (int64_t)0);
    CHECK_EQ(cap, 2);
    CHECK_EQ(raw, 3);
    CHECK_EQ(vram_pack_query(m.file, t2, plain, true, &cap, &raw, &buf), t2[plain].bytes);
    CHECK_EQ(raw, 0);

    /* Not eligible, or past the table: 0, nothing set. */
    cap = raw = -9;
    const std::vector<VramPackWeight> none(t.size());
    CHECK_EQ(vram_pack_query(m.file, none, raw3, true, &cap, &raw, &buf), (int64_t)0);
    CHECK_EQ(vram_pack_query(m.file, t, (rad_weight)t.size(), true, &cap, &raw, &buf), (int64_t)0);
    CHECK_EQ(cap, -9);
    CHECK_EQ(raw, -9);

    /* Unreadable scales (past the container's end) or a plane of the wrong size: -1, and the query keeps
     * the weight block_mxfp4. */
    std::vector<VramPackWeight> bad = t;
    bad[raw3].scales_off = (uint64_t)m.file.size();
    CHECK_EQ(vram_pack_scan(m.file, bad[raw3], &buf), -1);
    CHECK_EQ(vram_pack_query(m.file, bad, raw3, true, &cap, &raw, &buf), (int64_t)0);
    CHECK_EQ(raw, -1);
    bad[raw3] = t[raw3];
    bad[raw3].scales_bytes -= 1;
    CHECK_EQ(vram_pack_scan(m.file, bad[raw3], &buf), -1);
}

/* ================================================================== the draft bank in VRAM
 *
 * RADIANCE_EXPERIMENTAL_DS41_DRAFT_VRAM (core/place/draft_vram.h): Planner::place_draft_bank_vram puts
 * a rank's hot draft experts static in VRAM, charged to the weights pool before the expert fill, and
 * parks the rest with no class, slot or pool; nothing of the bank is pinned. On the native-sized bank
 * fixtures above (bank_expert_tiered_program: 40 trunk layers of 96 experts beside the three stages). */
namespace {
/* Rank r keeps, of stage s, the local experts (7 i + 3 s + r) mod 32 for i < n: a spread that differs
 * by stage and rank (7 is prime to 32, so they are distinct). */
DraftVramRank dv_spec(int rank, int n) {
    DraftVramRank d;
    d.n = n;
    d.resident.assign(3, std::vector<uint8_t>(32, 0));
    d.global.assign(3, std::vector<int32_t>(32, -1));
    for (int st = 0; st < 3; ++st) {
        for (int e = 0; e < 32; ++e) d.global[(size_t)st][(size_t)e] = 32 * rank + e;
        for (int i = 0; i < n; ++i) d.resident[(size_t)st][(size_t)((7 * i + 3 * st + rank) % 32)] = 1;
    }
    return d;
}
int trunk_vram_residents(const Plan& p) {
    int n = 0;
    for (const MoveUnit& mu : p.units) n += mu.expert >= 0 && mu.layer < 40 && mu.tier == Tier::VRAM;
    return n;
}
bool no_draft_vram_marks(const Plan& p) {
    for (const MoveUnit& mu : p.units) if (mu.draft_vram || mu.draft_parked) return false;
    return true;
}
}  /* namespace */

TEST(dspark_bank_vram_holds_the_hot_set_static_and_parks_the_rest_on_all_ranks) {
    constexpr int64_t P = 18800640;
    for (int rank = 0; rank < 4; ++rank) {
        Program prog = bank_expert_tiered_program(rank);
        Config cfg = bank_expert_tiered_config();
        const DraftVramRank dv = dv_spec(rank, 8);
        PlannerInput in{&prog, &cfg, rank, 4, nullptr, 0};
        in.container = true;
        in.draft_vram = &dv;
        Plan p;
        REQUIRE(Planner(in).plan(&p) == RAD_OK);
        CHECK(p.fits());
        CHECK(p.strategy == Strategy::ExpertTiered);
        int hot = 0, parked = 0, trunk = 0;
        for (const MoveUnit& mu : p.units) {
            if (mu.expert < 0 || mu.layer < 40) {
                CHECK(!mu.draft_vram); CHECK(!mu.draft_parked);
                if (mu.expert >= 0) { ++trunk; CHECK(mu.tier == Tier::VRAM || mu.tier == Tier::HostPinned || mu.tier == Tier::SSD); }
                continue;
            }
            const bool keep = dv.resident[(size_t)(mu.layer - 40)][(size_t)mu.expert] != 0;
            CHECK_EQ(mu.draft_vram, keep);
            CHECK_EQ(mu.draft_parked, !keep);
            CHECK(!mu.host_forced); CHECK(!mu.movable); CHECK(mu.preplaced());
            if (keep) {
                ++hot;
                CHECK(mu.tier == Tier::VRAM); CHECK(mu.site == Site::Device);
                CHECK_EQ(mu.slab_slot, -1); CHECK_EQ(mu.slot_bytes, P); CHECK(mu.slab_class >= 0);
                for (auto w : mu.weights) {
                    CHECK(p.weight[w].tier == Tier::VRAM); CHECK(p.weight[w].site == Site::Device);
                }
            } else {
                ++parked;
                CHECK(mu.tier == Tier::Mapped); CHECK(mu.site == Site::Device);
                CHECK_EQ(mu.slab_class, -1); CHECK_EQ(mu.slot_bytes, (int64_t)0);
                for (auto w : mu.weights) CHECK(p.weight[w].tier == Tier::Mapped);
            }
        }
        CHECK_EQ(hot, 24); CHECK_EQ(parked, 72); CHECK_EQ(trunk, 40 * 96);
        CHECK_EQ(p.draft_vram_units, (int64_t)24); CHECK_EQ(p.draft_vram_bytes, 24 * P);
        CHECK_EQ(p.draft_parked_units, (int64_t)72); CHECK_EQ(p.draft_parked_bytes, 72 * P);
        /* Nothing of the bank in the host pool: the budget is the flag's less the file tier's staging. */
        CHECK_EQ(p.forced_host_used, (int64_t)0); CHECK_EQ(p.draft_host_bytes, (int64_t)0);
        CHECK_EQ(p.host_budget, cfg.host_pool_mib * MiB - p.file_stage_reserve);
        /* A parked expert is held nowhere: no tier counts it (the fixture maps nothing else). */
        CHECK_EQ(p.totals.bytes[int(Tier::Mapped)], (int64_t)0);
        CHECK(p.totals.bytes[int(Tier::VRAM)] + p.slab_overhead + p.file_stage_vram <= cfg.vram_weights_mib * MiB);
        CHECK(p.report(prog).find("draft vram") != std::string::npos);
    }
}

/* The charge: with the file buffers held at one unit (RADIANCE_EXPERIMENTAL_DS41_FILE_BUFFER_UNITS=1, as
 * the recommended configuration runs), every hot pair costs the trunk exactly one VRAM resident, and the
 * VRAM total is unchanged (a pair is a trunk expert's size). */
TEST(dspark_bank_vram_charges_the_weights_pool_so_the_trunk_gives_up_as_many_residents) {
    FileBufferUnitsEnv units("1");
    for (int rank : {0, 3}) {
        Program prog = bank_expert_tiered_program(rank);
        Config cfg = bank_expert_tiered_config();
        PlannerInput host_in{&prog, &cfg, rank, 4, nullptr, 0};
        host_in.container = true;
        Plan host;
        REQUIRE(Planner(host_in).plan(&host) == RAD_OK);
        const int base = trunk_vram_residents(host);
        CHECK(base > 100);
        for (int n : {1, 8, 16, 24}) {
            const DraftVramRank dv = dv_spec(rank, n);
            PlannerInput in = host_in;
            in.draft_vram = &dv;
            Plan p;
            REQUIRE(Planner(in).plan(&p) == RAD_OK);
            CHECK_EQ(trunk_vram_residents(p), base - 3 * n);
            CHECK_EQ(p.draft_vram_units, (int64_t)(3 * n));
            CHECK_EQ(p.file_stage_vram, host.file_stage_vram);
            CHECK_EQ(p.totals.bytes[int(Tier::VRAM)], host.totals.bytes[int(Tier::VRAM)]);
            /* The host pool the bank no longer takes goes to the trunk's spill. */
            CHECK(p.totals.bytes[int(Tier::HostPinned)] > host.totals.bytes[int(Tier::HostPinned)] - host.forced_host_used);
        }
    }
}

TEST(dspark_bank_vram_refuses_without_placing_or_parking_anything) {
    for (int bad = 0; bad < 6; ++bad) {
        Program prog = bank_expert_tiered_program(0);
        Config cfg = bank_expert_tiered_config();
        DraftVramRank dv = dv_spec(0, 8);
        PlannerInput in{&prog, &cfg, 0, 4, nullptr, 0};
        in.container = true;
        int want = RAD_E_UNSUPPORTED;
        if (bad == 0) { dv.resident[1][(size_t)((3 * 1) % 32)] = 0; want = RAD_E_INVAL; }   /* stage 41 keeps 7 */
        if (bad == 1) cfg.placement = "all_vram";                    /* no expert fill to charge */
        if (bad == 2) cfg.draft_bank_expert_placement = "auto";      /* not the bank's manual contract */
        if (bad == 3) cfg.draft_bank.clear();
        if (bad == 4) in.host_backend = true;
        if (bad == 5) prog.ctx.max_spec = 4;                          /* not the native bank */
        in.draft_vram = &dv;
        Plan p;
        CHECK_EQ(Planner(in).plan(&p), want);
        CHECK(no_draft_vram_marks(p));
        for (const MoveUnit& mu : p.units) CHECK(!mu.host_forced);
    }
}

TEST(dspark_bank_vram_passes_the_draft_control_scope_only_as_placed) {
    FileBufferUnitsEnv units("1");   /* 48 hot pairs beside two whole-layer file buffers overfill 4 GiB */
    Program prog = bank_expert_tiered_program(1);
    Config cfg = bank_expert_tiered_config();
    const DraftVramRank dv = dv_spec(1, 16);
    PlannerInput in{&prog, &cfg, 1, 4, nullptr, 0};
    in.container = true;
    Plan host;
    REQUIRE(Planner(in).plan(&host) == RAD_OK);
    in.draft_vram = &dv;
    Plan vram;
    REQUIRE(Planner(in).plan(&vram) == RAD_OK);
    CHECK(dspark_draft_control_scope(prog, host, cfg, true));
    CHECK(dspark_draft_control_scope(prog, host, cfg, true, 0));
    CHECK(!dspark_draft_control_scope(prog, host, cfg, true, 16));
    CHECK(dspark_draft_control_scope(prog, vram, cfg, true, 16));
    CHECK(!dspark_draft_control_scope(prog, vram, cfg, true, 0));
    CHECK(!dspark_draft_control_scope(prog, vram, cfg, true, 8));
    CHECK(!dspark_draft_control_scope(prog, vram, cfg, false, 16));
    /* Any --max-num-seqs: a step that drafts carries one sequence (the scheduler's one-sequence
     * rule, sched/draft_conf.h). */
    Config several = cfg;
    several.max_seqs = 4;
    CHECK(dspark_draft_control_scope(prog, vram, several, true, 16));
    several.max_seqs = 0;
    CHECK(!dspark_draft_control_scope(prog, vram, several, true, 16));
}

/* The mover gives the hot set static addresses and the parked experts none (no class, no pool); the
 * engine then publishes each parked weight at its stage's first resident's weight of the same slot
 * (draft_vram_aliases), so the grouped GEMM's table names card memory for every expert. A small plan
 * (the bank alone, one expert a stage), so the host backend allocates 56 MB. */
TEST(dspark_bank_vram_mover_holds_only_the_hot_set_and_parked_weights_publish_at_residents) {
    Program prog = bank_placement_program(2);
    Config cfg = bank_placement_config();
    cfg.placement = "expert_tiered";
    cfg.vram_weights_mib = 256;
    cfg.host_pool_mib = 64;
    cfg.draft_bank_expert_placement = "host";
    const DraftVramRank dv = dv_spec(2, 1);
    PlannerInput in{&prog, &cfg, 2, 4, nullptr, 0};
    in.draft_vram = &dv;
    Plan p;
    REQUIRE(Planner(in).plan(&p) == RAD_OK);
    CHECK_EQ(p.totals.bytes[int(Tier::HostPinned)], (int64_t)0);
    Mover m;
    MoverConfig mc;
    REQUIRE(m.init(prog, p, mc) == RAD_OK);
    std::vector<void*> alias;
    std::string why;
    REQUIRE_EQ(draft_vram_aliases(prog, p, m, &alias, &why), RAD_OK);
    void* first[3][2] = {};
    int hot_w = 0, parked_w = 0;
    for (const MoveUnit& mu : p.units) {
        if (!mu.draft_vram) continue;
        for (auto w : mu.weights) {
            const WeightSlot* ws = m.residency()->lookup(w);
            REQUIRE(ws && ws->ptr);
            CHECK(alias[w] == nullptr);
            first[mu.layer - 40][prog.weights[w].decl.group.slot] = ws->ptr;
            ++hot_w;
        }
    }
    CHECK(first[0][0] != first[0][1]);
    for (const MoveUnit& mu : p.units) {
        if (!mu.draft_parked) continue;
        for (auto w : mu.weights) {
            const WeightSlot* ws = m.residency()->lookup(w);
            CHECK(!ws || !ws->ptr);
            void* want = first[mu.layer - 40][prog.weights[w].decl.group.slot];
            REQUIRE(want != nullptr);
            CHECK(alias[w] == want);
            m.publish_mapped(w, alias[w]);
            const WeightSlot* now = m.residency()->lookup(w);
            REQUIRE(now != nullptr);
            CHECK(now->ptr == want);
            ++parked_w;
        }
    }
    CHECK_EQ(hot_w, 2 * 3);
    CHECK_EQ(parked_w, 2 * 93);
    /* A stage with no resident leaves its parked weights nowhere to be published: refused. */
    Plan broken = p;
    for (MoveUnit& mu : broken.units)
        if (mu.draft_vram && mu.layer == 41) { mu.draft_vram = false; mu.draft_parked = true; }
    std::vector<void*> none;
    CHECK_EQ(draft_vram_aliases(prog, broken, m, &none, &why), RAD_E_STATE);
}

/* ================================================================== packed trunk residents beside a DSpark bank
 *
 * RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED with --draft-bank (vram_packed.h, `bank`): the trunk's routed
 * experts come from the main container and pack as without a bank; every bank weight (file_id 1, the draft
 * layers') keeps its block form whatever reads it, and the hot draft set of ..._DRAFT_VRAM is charged at its
 * slots before the fill exactly as without the switch. */
TEST(vram_pack_eligible_with_a_dspark_bank_packs_the_trunk_and_never_a_bank_weight) {
    VpModel m(3, 4, 2, "vp_bank.rad");
    REQUIRE(m.ok);
    /* Every op on gemv_vp, the draft layer's too (which the trunk-only row prevents; refused here anyway). */
    for (int l = 0; l < 3; ++l) {
        m.op(l, 0, { &g_vp_row });
        m.op(l, 1, { &g_vp_row });
    }
    VramPackConfig cfg;
    cfg.on = true;
    cfg.raw[0] = 16;
    cfg.raw[1] = 4;
    std::vector<VramPackWeight> plain, bank;
    std::string why_plain, why_bank;
    vram_pack_eligible(m.f.prog, m.file, cfg, &plain, &why_plain);
    int trunk_plain = 0;
    for (rad_weight h = 1; (size_t)h < plain.size(); ++h) {
        const bool trunk = m.f.prog.weights[h].decl.group.layer < 2;
        CHECK_EQ(plain[(size_t)h].bytes > 0, trunk);
        trunk_plain += plain[(size_t)h].bytes > 0;
    }
    CHECK_EQ(trunk_plain, 16);
    /* Bound as DsparkRadBanks::bind_weights binds them: the draft layer's weights to the bank. And one
     * trunk-shaped weight in the bank too: the bank alone keeps it block_mxfp4. */
    for (int e = 0; e < 4; ++e)
        for (int s = 0; s < 2; ++s) m.f.prog.weights[m.at(2, e, s)].file_id = 1;
    m.f.prog.weights[m.at(0, 1, 1)].file_id = 1;
    vram_pack_eligible(m.f.prog, m.file, cfg, &bank, &why_bank);
    REQUIRE(bank.size() == plain.size());
    int trunk_bank = 0;
    for (rad_weight h = 1; (size_t)h < bank.size(); ++h) {
        if (m.f.prog.weights[h].file_id != 0) {
            CHECK_EQ(bank[(size_t)h].bytes, (int64_t)0);
            continue;
        }
        /* The main container's: the same answer as without a bank, field for field. */
        CHECK_EQ(bank[(size_t)h].bytes, plain[(size_t)h].bytes);
        CHECK_EQ(bank[(size_t)h].cap, plain[(size_t)h].cap);
        CHECK_EQ(bank[(size_t)h].scales_off, plain[(size_t)h].scales_off);
        trunk_bank += bank[(size_t)h].bytes > 0;
    }
    CHECK_EQ(trunk_bank, 15);
    CHECK(why_bank.find("9 more weights the gemv_vp tables read are a DSpark bank's (the draft's) and stay "
                        "block_mxfp4") != std::string::npos);
    if (why_bank.find("9 more weights") == std::string::npos) std::printf("  why: %s\n", why_bank.c_str());
    /* No bank, no clause: the line reads as before. */
    CHECK(why_plain.find("DSpark bank") == std::string::npos);
}

TEST(vram_pack_bank_check_refuses_a_draft_op_on_gemv_vp_or_a_packed_bank_weight) {
    VpModel m(3, 2, 2, "vp_bank_check.rad");
    REQUIRE(m.ok);
    m.op(0, 0, { &g_vp_row });
    m.op(0, 1, { &g_vp_row });
    m.op(1, 0, { &g_vp_row, &g_mx_tiled_row });
    m.op(1, 1, { &g_vp_row });
    /* The draft's ops as the trunk-only row leaves them: the plain gemv row, then the tiled one. */
    m.op(2, 0, { &g_mx_gemv_row, &g_mx_tiled_row });
    m.op(2, 1, { &g_mx_gemv_row, &g_mx_tiled_row });
    for (int e = 0; e < 2; ++e)
        for (int s = 0; s < 2; ++s) m.f.prog.weights[m.at(2, e, s)].file_id = 1;
    Plan p;
    p.weight.resize(m.f.prog.weights.size());
    p.weight[m.at(0, 0, 0)].vram_bytes = 1000;
    p.weight[m.at(0, 1, 1)].vram_bytes = 1000;
    p.weight[m.at(2, 0, 0)].tier = Tier::VRAM;
    p.weight[m.at(2, 0, 1)].tier = Tier::VRAM;
    p.weight[m.at(2, 1, 0)].tier = Tier::Mapped;
    p.weight[m.at(2, 1, 1)].tier = Tier::Mapped;
    VramPackBankReport r;
    std::string why;
    CHECK_OK(vram_pack_bank_check(m.f.prog, p, &r, &why));
    CHECK(why.empty());
    CHECK_EQ(r.draft_ops, (int64_t)2);
    CHECK_EQ(r.bank_weights, (int64_t)4);
    CHECK_EQ(r.bank_vram, (int64_t)2);
    CHECK_EQ(r.trunk_packed, (int64_t)2);
    /* A bank weight planned packed. */
    Plan bad = p;
    bad.weight[m.at(2, 0, 1)].vram_bytes = 1000;
    CHECK_EQ(vram_pack_bank_check(m.f.prog, bad, &r, &why), RAD_E_STATE);
    CHECK(why.find("blk.2.ffn_down_exps.0.weight") != std::string::npos);
    /* A draft op that resolved gemv_vp: by its layer, even with the weights not bound to the bank. */
    for (int e = 0; e < 2; ++e)
        for (int s = 0; s < 2; ++s) m.f.prog.weights[m.at(2, e, s)].file_id = 0;
    m.f.prog.ops.back().bands[0].dom[RAD_DOMAIN_DEVICE].row = &g_vp_row;
    CHECK_EQ(vram_pack_bank_check(m.f.prog, p, &r, &why), RAD_E_STATE);
    CHECK(why.find(kVramPackRow) != std::string::npos);
}

/* The accounting with the draft hot set in VRAM: the hot pairs are charged at their block slots before the
 * fill (as without the switch) and are never packed, even by a callback that would pack them; the trunk's
 * residents pack, and the room that frees goes to more trunk residents. */
TEST(dspark_bank_vram_with_packed_trunk_residents_keeps_the_hot_set_block_and_fills_the_freed_room) {
    FileBufferUnitsEnv units("1");
    constexpr int64_t P = 18800640;
    for (int rank : {0, 3}) {
        Program prog = bank_expert_tiered_program(rank);
        for (size_t w = 1; w < prog.weights.size(); ++w)
            if (prog.weights[w].decl.group.layer >= 40) prog.weights[w].file_id = 1;
        Config cfg = bank_expert_tiered_config();
        REQUIRE(cfg.deterministic);
        const DraftVramRank dv = dv_spec(rank, 24);
        PlannerInput in{&prog, &cfg, rank, 4, nullptr, 0};
        in.container = true;
        in.draft_vram = &dv;
        Plan plain;
        REQUIRE(Planner(in).plan(&plain) == RAD_OK);
        /* As the engine answers (vram_pack_eligible: the bank's weights 0), and a callback that would pack
         * every expert, the draft's too. */
        FakePack trunk_only;
        trunk_only.prog = &prog;
        auto trunk_fn = trunk_only.fn();
        FakePack every;
        every.prog = &prog;
        for (int which = 0; which < 2; ++which) {
            PlannerInput pin = in;
            if (which == 0)
                pin.vram_pack = [&](rad_weight w, bool scan, int32_t* cap, int32_t* raw) -> int64_t {
                    return prog.weights[w].file_id != 0 ? 0 : trunk_fn(w, scan, cap, raw);
                };
            else
                pin.vram_pack = every.fn();
            Plan p;
            REQUIRE(Planner(pin).plan(&p) == RAD_OK);
            CHECK(p.fits());
            CHECK(dspark_draft_control_scope(prog, p, cfg, true, 24));
            /* The hot set: the same units, block form, the same charge. */
            CHECK_EQ(p.draft_vram_units, plain.draft_vram_units);
            CHECK_EQ(p.draft_vram_bytes, plain.draft_vram_bytes);
            CHECK_EQ(p.draft_vram_bytes, (int64_t)72 * P);
            CHECK_EQ(p.draft_parked_units, plain.draft_parked_units);
            for (size_t u = 0; u < p.units.size(); ++u) {
                const MoveUnit& mu = p.units[u];
                if (mu.layer < 40 || mu.expert < 0) continue;
                CHECK_EQ(mu.draft_vram, plain.units[u].draft_vram);
                CHECK(mu.tier == plain.units[u].tier);
                CHECK_EQ(mu.vram_bytes, (int64_t)0);
                if (mu.draft_vram) CHECK_EQ(mu.slot_bytes, P);
                for (rad_weight w : mu.weights) CHECK_EQ(p.weight[w].vram_bytes, (int64_t)0);
            }
            /* The trunk: every resident packed, and x1.0444 as many in the room the hot set leaves. */
            const int r0 = trunk_vram_residents(plain), r1 = trunk_vram_residents(p);
            CHECK(r1 > r0 + r0 * 4 / 100 - 1);
            CHECK_EQ(p.vram_packed.units, (int64_t)r1);
            CHECK_EQ(p.vram_packed.weights, (int64_t)2 * r1);
            CHECK_EQ(p.vram_packed.bytes, (int64_t)r1 * align_up(align_up(kGuPacked, RAD_ALIGN_UNIT) + kDnPacked, RAD_ALIGN_UNIT));
            VramPackBankReport rep;
            std::string why;
            CHECK_OK(vram_pack_bank_check(prog, p, &rep, &why));
            CHECK_EQ(rep.trunk_packed, (int64_t)2 * r1);
            CHECK_EQ(rep.bank_vram, (int64_t)2 * 72);
            CHECK(p.totals.bytes[int(Tier::VRAM)] + p.slab_overhead + p.file_stage_vram <= p.vram_budget);
        }
    }
}

RAD_TEST_MAIN()
