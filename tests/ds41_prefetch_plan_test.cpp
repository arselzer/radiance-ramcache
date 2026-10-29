/* ds41_prefetch_plan_test.cpp -- deepseek41's on-die read plan (arch/deepseek41/ds41_prefetch_plan.h):
 * RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ATTN alone keeps the four attention reads 16be66c5 measured;
 * a budget reads the use-order list's first bytes, cutting the read that crosses it; ..._SET names
 * the reads and their order; malformed values are named and read as unset. The sizes are a TP4
 * card's of DeepSeek-V4.1-Flash, bf16 and (..._DENSE_FP8: the linears' E4M3 codes) half that. */
#include "rad_test.h"

#include "../arch/deepseek41/ds41_prefetch_plan.h"

#include <cstdint>
#include <vector>

using namespace deepseek41;

namespace {

constexpr int64_t MiB = 1ll << 20;

std::vector<pf::Read> fit_kinds(const std::vector<pf::Cand>& c, int64_t budget, std::vector<int>* kinds) {
    const std::vector<pf::Read> r = pf::fit(c, budget);
    kinds->clear();
    for (const pf::Read& x : r) kinds->push_back(c[x.cand].kind);
    return r;
}

/* A layer away from the sources, bf16 at TP4: q_a 12.5 MiB, kv 5, q_b 20, wo_a 2 x 8, wo_b 20, the
 * f32 router 7.5, rank 0's shared gate|up 15 and down 7.5. `half`: the linears in FP8 (the router
 * stays f32). */
std::vector<pf::Cand> layer_cands(bool half) {
    const int64_t d = half ? 2 : 1;
    return { { pf::kQa, 25 * MiB / 2 / d }, { pf::kKv, 5 * MiB / d },   { pf::kQb, 20 * MiB / d },
             { pf::kOa, 8 * MiB / d },      { pf::kOa, 8 * MiB / d },   { pf::kOb, 20 * MiB / d },
             { pf::kRt, 15 * MiB / 2 },     { pf::kSgu, 15 * MiB / d }, { pf::kSdn, 15 * MiB / 2 / d } };
}

}  // namespace

TEST(ds41_prefetch_cfg_switch_alone_keeps_the_measured_reads) {
    const pf::Cfg off = pf::parse(nullptr, nullptr, nullptr, nullptr);
    CHECK(!off.on);
    CHECK(!pf::parse("0", nullptr, nullptr, nullptr).on);
    CHECK(!pf::parse("yes", nullptr, nullptr, nullptr).on);
    const pf::Cfg c = pf::parse("1", nullptr, nullptr, nullptr);
    CHECK(c.on);
    CHECK_EQ(c.budget, (int64_t)0);
    CHECK(!c.engram);
    CHECK(!c.listed);
    CHECK(c.note.empty());
    CHECK(c.order == (std::vector<int>{ pf::kQa, pf::kKv, pf::kQb, pf::kOa }));
    /* Unset or empty modifiers are no modifiers. */
    const pf::Cfg e = pf::parse("1", "", "", nullptr);
    CHECK(e.order == c.order);
    CHECK_EQ(e.budget, (int64_t)0);
    CHECK(e.note.empty());
}

TEST(ds41_prefetch_cfg_budget_reads_the_use_order) {
    const pf::Cfg c = pf::parse("1", "56", nullptr, "1");
    CHECK_EQ(c.budget, 56 * MiB);
    CHECK(c.engram);
    CHECK(c.order == pf::use_order());
    CHECK(c.note.empty());
    /* A fractional budget, down to whole 4 KiB. */
    CHECK_EQ(pf::parse("1", "53.5", nullptr, nullptr).budget, 107 * MiB / 2);
    CHECK_EQ(pf::parse("1", "0.001", nullptr, nullptr).budget, (int64_t)0);
    /* The kinds are declared in use order: the default list runs up the enum. */
    const std::vector<int> u = pf::use_order();
    for (size_t i = 1; i < u.size(); ++i) CHECK(u[i - 1] < u[i]);
    CHECK_EQ(u.front(), (int)pf::kEkv);
    CHECK_EQ(u.back(), (int)pf::kSdn);
}

TEST(ds41_prefetch_cfg_malformed_values_are_named_and_unset) {
    for (const char* bad : { "abc", "-3", "0", "5000", "56x", "nan" }) {
        const pf::Cfg c = pf::parse("1", bad, nullptr, nullptr);
        CHECK_EQ(c.budget, (int64_t)0);
        CHECK(c.order == pf::legacy_order());
        CHECK(c.note.find("PREFETCH_MIB") != std::string::npos);
    }
    const pf::Cfg e = pf::parse("1", nullptr, nullptr, "2");
    CHECK(!e.engram);
    CHECK(e.note.find("PREFETCH_ENGRAM") != std::string::npos);
}

TEST(ds41_prefetch_cfg_set_names_the_reads_and_their_order) {
    const pf::Cfg c = pf::parse("1", "48", "ob,qa+kv qa  sgu", nullptr);
    CHECK(c.listed);
    CHECK(c.order == (std::vector<int>{ pf::kOb, pf::kQa, pf::kKv, pf::kSgu }));
    CHECK(c.note.empty());
    CHECK_EQ(c.budget, 48 * MiB);
    /* Every name round-trips; an unknown one is named and skipped. */
    for (int k = 0; k < pf::kKinds; ++k) {
        const char* n = pf::name(k);
        CHECK_EQ(pf::kind_of(n, std::strlen(n)), k);
    }
    const pf::Cfg u = pf::parse("1", nullptr, "qa,qbb,ekv", nullptr);
    CHECK(u.order == (std::vector<int>{ pf::kQa, pf::kEkv }));
    CHECK(u.note.find("'qbb'") != std::string::npos);
    /* Nothing valid: the default list of the budget (or of none). */
    CHECK(pf::parse("1", nullptr, "x,y", nullptr).order == pf::legacy_order());
    CHECK(pf::parse("1", "56", ",,", nullptr).order == pf::use_order());
    CHECK(!pf::parse("1", "56", ",,", nullptr).listed);
}

TEST(ds41_prefetch_fit_without_budget_reads_all_whole) {
    const std::vector<pf::Cand> c = layer_cands(false);
    const std::vector<pf::Read> r = pf::fit(c, 0);
    REQUIRE_EQ(r.size(), c.size());
    for (size_t i = 0; i < r.size(); ++i) {
        CHECK_EQ(r[i].cand, i);
        CHECK_EQ(r[i].bytes, c[i].bytes);
        CHECK(!r[i].cut);
    }
}

TEST(ds41_prefetch_fit_cuts_the_read_that_crosses_the_budget) {
    std::vector<int> k;
    /* bf16 at 56 MiB: the measured four whole (53.5 MiB), then wo_b's first 2.5 MiB, and no more. */
    const std::vector<pf::Read> r = fit_kinds(layer_cands(false), 56 * MiB, &k);
    REQUIRE_EQ(r.size(), (size_t)6);
    CHECK(k == (std::vector<int>{ pf::kQa, pf::kKv, pf::kQb, pf::kOa, pf::kOa, pf::kOb }));
    for (size_t i = 0; i + 1 < r.size(); ++i) CHECK(!r[i].cut);
    CHECK(r.back().cut);
    CHECK_EQ(r.back().bytes, 5 * MiB / 2);
    int64_t s = 0;
    for (const pf::Read& x : r) s += x.bytes;
    CHECK_EQ(s, 56 * MiB);
    /* 54 MiB leaves wo_b 0.5 MiB, under kMinCut: dropped, and nothing after it. */
    const std::vector<pf::Read> d = fit_kinds(layer_cands(false), 54 * MiB, &k);
    CHECK(k == (std::vector<int>{ pf::kQa, pf::kKv, pf::kQb, pf::kOa, pf::kOa }));
    for (const pf::Read& x : d) CHECK(!x.cut);
    /* A cut is whole 4 KiB. */
    const std::vector<pf::Read> o = pf::fit({ { pf::kQa, 10 * MiB } }, 3 * MiB + 1000);
    REQUIRE_EQ(o.size(), (size_t)1);
    CHECK_EQ(o[0].bytes, 3 * MiB);
    CHECK(o[0].cut);
}

TEST(ds41_prefetch_fit_fp8_sizes_reach_the_shared_expert) {
    std::vector<int> k;
    /* Half the linears' bytes: 36.75 MiB of attention, the router, rank 0's whole shared expert
     * (55.5 MiB in all) under 56. */
    const std::vector<pf::Read> r = fit_kinds(layer_cands(true), 56 * MiB, &k);
    REQUIRE_EQ(r.size(), layer_cands(true).size());
    for (const pf::Read& x : r) CHECK(!x.cut);
    CHECK_EQ(k.back(), (int)pf::kSdn);
}

TEST(ds41_prefetch_fit_skips_weights_a_layer_lacks) {
    std::vector<int> k;
    /* A weight of no bytes (none on this rank, or llama.cpp's blocks) is no read and no stop. */
    const std::vector<pf::Read> r =
        fit_kinds({ { pf::kComp, 0 }, { pf::kQa, 2 * MiB }, { pf::kIq, 0 }, { pf::kQb, 3 * MiB } },
                  4 * MiB, &k);
    CHECK(k == (std::vector<int>{ pf::kQa, pf::kQb }));
    REQUIRE_EQ(r.size(), (size_t)2);
    CHECK_EQ(r[0].cand, (size_t)1);
    CHECK_EQ(r[1].cand, (size_t)3);
    CHECK_EQ(r[1].bytes, 2 * MiB);
    CHECK(r[1].cut);
    CHECK(pf::fit({}, 56 * MiB).empty());
}

RAD_TEST_MAIN()
