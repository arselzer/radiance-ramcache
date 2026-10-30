/* draft_vram_test.cpp -- RADIANCE_EXPERIMENTAL_DS41_DRAFT_VRAM (core/place/draft_vram.h and
 * draft_vram_load.h): the switch, the hot-set file, each stage's kept experts and each rank's part of
 * them, the masked correction plane, and the load-time mask written into a rank's three planes on the
 * host backend (cards hidden: "device" memory is host memory, the copies the engine issues are the
 * same). The placement and the planner's charge are place_test's (dspark_bank_vram_*); the router
 * under the mask is ref_test's (libref) and r4d_emu_ds41mask's (libr4d's kernel source). */
#include "rad_test.h"

#include "device/device.h"
#include "place/draft_vram.h"
#include "place/draft_vram_load.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

using namespace rad;

namespace {

/* A valid hot-set file: rank r lists `len` of its experts 32r + (5 i + s) mod 32, hottest first. */
std::string hotset_text(int len = 32, const char* extra = "") {
    std::string t = std::string("{\"format\": \"") + kDraftHotsetFormat +
                    "\", \"layers\": [40, 41, 42], \"ranks\": 4, \"experts\": 128" + extra + ", \"hot\": [";
    for (int s = 0; s < 3; ++s) {
        t += s ? ", [" : "[";
        for (int r = 0; r < 4; ++r) {
            t += r ? ", [" : "[";
            for (int i = 0; i < len; ++i) t += (i ? "," : "") + std::to_string(32 * r + (5 * i + s) % 32);
            t += "]";
        }
        t += "]";
    }
    return t + "]}";
}

/* Parsed; a refusal fails the case's check (and leaves an empty set, which the case's sizes catch). */
DraftHotset parsed(const std::string& text) {
    DraftHotset hs;
    std::string why;
    CHECK_EQ(draft_hotset_parse(text, &hs, &why), RAD_OK);
    return hs;
}

/* The identity numbering the bank's directory gives rank r: local e is 32 r + e. */
std::function<int32_t(int32_t, int32_t)> contiguous(int rank) {
    return [rank](int32_t, int32_t local) { return 32 * rank + local; };
}

}  /* namespace */

TEST(draft_vram_switch_takes_one_to_thirty_two) {
    int n = -1;
    CHECK_EQ(draft_vram_switch(nullptr, &n), RAD_OK); CHECK_EQ(n, 0);
    CHECK_EQ(draft_vram_switch("", &n), RAD_OK); CHECK_EQ(n, 0);
    CHECK_EQ(draft_vram_switch("0", &n), RAD_OK); CHECK_EQ(n, 0);
    CHECK_EQ(draft_vram_switch("1", &n), RAD_OK); CHECK_EQ(n, 1);
    CHECK_EQ(draft_vram_switch("24", &n), RAD_OK); CHECK_EQ(n, 24);
    CHECK_EQ(draft_vram_switch("32", &n), RAD_OK); CHECK_EQ(n, 32);
    CHECK_EQ(draft_vram_switch("08", &n), RAD_OK); CHECK_EQ(n, 8);
    for (const char* bad : {"33", "-1", "8x", "x", " 8", "1e1", "+8", "008", "08 "}) {
        CHECK_EQ(draft_vram_switch(bad, &n), RAD_E_INVAL);
        CHECK_EQ(n, 0);
    }
}

TEST(draft_hotset_parses_the_tools_format_and_refuses_anything_else) {
    const DraftHotset hs = parsed(hotset_text(32, ", \"top\": 32, \"counts\": [[1], [2], [3]]"));
    REQUIRE_EQ(hs.hot.size(), (size_t)3);
    for (int s = 0; s < 3; ++s) {
        REQUIRE_EQ(hs.hot[(size_t)s].size(), (size_t)4);
        for (int r = 0; r < 4; ++r) {
            REQUIRE_EQ(hs.hot[(size_t)s][(size_t)r].size(), (size_t)32);
            CHECK_EQ(hs.hot[(size_t)s][(size_t)r][1], 32 * r + (5 + s) % 32);
        }
    }
    const DraftHotset short_lists = parsed(hotset_text(3));
    CHECK_EQ(short_lists.hot[2][3].size(), (size_t)3);
    std::string why;
    DraftHotset out;
    const std::string good = hotset_text(4);
    auto refused = [&](std::string text) {
        why.clear();
        const int st = draft_hotset_parse(text, &out, &why);
        CHECK_EQ(st, RAD_E_INVAL);
        CHECK(!why.empty());
    };
    auto swap = [&](const std::string& from, const std::string& to) -> std::string {
        std::string t = good;
        const size_t at = t.find(from);
        CHECK(at != std::string::npos);
        return at == std::string::npos ? std::string("not found") : t.replace(at, from.size(), to);
    };
    refused("not json");
    refused("[]");
    refused(swap(kDraftHotsetFormat, "radiance-expert-stats/1"));
    refused(swap("\"ranks\": 4", "\"ranks\": 3"));
    refused(swap("\"experts\": 128", "\"experts\": 96"));
    refused(swap("[40, 41, 42]", "[40, 41]"));
    refused(swap("[40, 41, 42]", "[41, 42, 43]"));
    refused(swap("[0,5,10,15]", "[0,5,5,15]"));                  /* listed twice */
    refused(swap("[0,5,10,15]", "[0,5,10,128]"));                /* out of range */
    refused(swap("[0,5,10,15]", "[]"));                          /* empty */
    refused(swap("[0,5,10,15]", "[0,5,10,1.5]"));                /* not an id */
    refused(swap("\"hot\": [", "\"hot\": [[[1],[33],[65],[97]], "));   /* four stages */
    refused(std::string("{\"format\": \"") + kDraftHotsetFormat + "\", \"layers\": [40, 41, 42], \"ranks\": 4, \"experts\": 128}");
    CHECK_EQ(draft_hotset_load("/nonexistent/hotset.json", &out, &why), RAD_E_IO);
}

TEST(draft_vram_allowed_is_the_first_n_of_every_rank_and_refuses_a_stage_below_top_k) {
    const DraftHotset hs = parsed(hotset_text(32));
    std::vector<std::vector<uint8_t>> a;
    std::string why;
    for (int n : {1, 8, 24, 32}) {
        REQUIRE_EQ(draft_vram_allowed(hs, n, &a, &why), RAD_OK);
        for (int s = 0; s < 3; ++s) {
            int kept = 0;
            for (int g = 0; g < 128; ++g) kept += a[(size_t)s][(size_t)g];
            CHECK_EQ(kept, 4 * n);
            for (int r = 0; r < 4; ++r)
                for (int i = 0; i < 32; ++i)
                    CHECK_EQ(a[(size_t)s][(size_t)(32 * r + (5 * i + s) % 32)] != 0, i < n);
        }
    }
    CHECK_EQ(draft_vram_allowed(hs, 0, &a, &why), RAD_E_INVAL);
    CHECK_EQ(draft_vram_allowed(hs, 33, &a, &why), RAD_E_INVAL);
    const DraftHotset four = parsed(hotset_text(4));
    CHECK_EQ(draft_vram_allowed(four, 4, &a, &why), RAD_OK);
    CHECK_EQ(draft_vram_allowed(four, 5, &a, &why), RAD_E_INVAL);   /* lists shorter than asked */
    CHECK(why.find("fewer than the 5 asked for") != std::string::npos);
    DraftHotset twice = four;
    twice.hot[1][2][0] = twice.hot[1][1][0];                          /* two ranks list one expert */
    CHECK_EQ(draft_vram_allowed(twice, 1, &a, &why), RAD_E_INVAL);
    CHECK(why.find("listed for two ranks") != std::string::npos);
    /* Below top-3 a stage cannot be honoured (four ranks keep at least four; a smaller set is a
     * hand-built one). */
    DraftHotset two = four;
    two.ranks = 1;
    for (auto& st : two.hot) st.resize(1);
    CHECK_EQ(draft_vram_allowed(two, 2, &a, &why), RAD_E_INVAL);
    CHECK(why.find("fewer than the router's top-3") != std::string::npos);
    CHECK_EQ(draft_vram_allowed(two, 3, &a, &why), RAD_OK);
}

TEST(draft_vram_rank_maps_listed_ids_to_the_ranks_own_units) {
    const DraftHotset hs = parsed(hotset_text(32));
    std::string why;
    for (int rank = 0; rank < 4; ++rank) {
        DraftVramRank d;
        REQUIRE_EQ(draft_vram_rank(hs, 8, rank, contiguous(rank), &d, &why), RAD_OK);
        CHECK_EQ(d.n, 8);
        CHECK_EQ(d.residents(), (int64_t)24);
        for (int s = 0; s < 3; ++s)
            for (int i = 0; i < 32; ++i) {
                CHECK_EQ(d.resident[(size_t)s][(size_t)((5 * i + s) % 32)] != 0, i < 8);
                CHECK_EQ(d.global[(size_t)s][(size_t)i], 32 * rank + i);
            }
    }
    /* A numbering the container gives otherwise (reversed within the rank): the same experts, other
     * local units. */
    DraftVramRank rev;
    REQUIRE_EQ(draft_vram_rank(hs, 2, 1, [](int32_t, int32_t e) { return 32 + 31 - e; }, &rev, &why), RAD_OK);
    CHECK(rev.resident[0][31 - 0]);
    CHECK(rev.resident[0][31 - 5]);
    CHECK_EQ(rev.residents(), (int64_t)6);
    DraftVramRank d;
    /* An expert listed for a rank that does not hold it. */
    CHECK_EQ(draft_vram_rank(hs, 8, 1, contiguous(2), &d, &why), RAD_E_INVAL);
    CHECK(why.find("which does not hold it") != std::string::npos);
    /* A unit the container names no expert for, and one named twice. */
    CHECK_EQ(draft_vram_rank(hs, 8, 0, [](int32_t, int32_t e) { return e == 9 ? -1 : e; }, &d, &why), RAD_E_INVAL);
    CHECK_EQ(draft_vram_rank(hs, 8, 0, [](int32_t, int32_t e) { return e == 9 ? 8 : e; }, &d, &why), RAD_E_INVAL);
    CHECK_EQ(draft_vram_rank(hs, 8, 4, contiguous(0), &d, &why), RAD_E_INVAL);
}

TEST(draft_mask_plane_keeps_kept_values_bit_for_bit_and_drops_the_rest_to_minus_inf) {
    float plane[128], out[128];
    uint8_t allowed[128];
    for (int e = 0; e < 128; ++e) {
        plane[e] = e % 3 ? 0.0f : -0.0f;
        if (e == 7) plane[e] = 0.125f;
        allowed[e] = e % 5 == 0 || e == 7;
    }
    draft_mask_plane(plane, allowed, 128, out);
    for (int e = 0; e < 128; ++e) {
        if (allowed[e]) CHECK_EQ(std::memcmp(&out[e], &plane[e], sizeof(float)), 0);
        else { CHECK(std::isinf(out[e])); CHECK(out[e] < 0); }
    }
    CHECK(draft_mask_holds(out, plane, allowed, 128));
    float wrong[128];
    std::memcpy(wrong, out, sizeof wrong);
    wrong[1] = -1e30f;                         /* finite: a router could still take it */
    CHECK(!draft_mask_holds(wrong, plane, allowed, 128));
    std::memcpy(wrong, out, sizeof wrong);
    wrong[0] = 0.0f;                           /* -0.0 kept as +0.0: not the loaded bits */
    CHECK(!draft_mask_holds(wrong, plane, allowed, 128));
    std::memcpy(wrong, out, sizeof wrong);
    wrong[2] = std::numeric_limits<float>::infinity();
    CHECK(!draft_mask_holds(wrong, plane, allowed, 128));
}

namespace {

/* One rank's Program as far as the mask reads it: each draft stage's router selection (router_scored_topk)
 * over its correction plane (f32 [128], layer 40 + s, no expert group), resident in "VRAM" (the host
 * backend's device memory), beside a projection and a trunk router the search must pass over. */
struct MaskRank {
    Program p;
    std::vector<void*> mem;
    rad_weight plane[3] = {};
    MaskRank() {
        p.weights.resize(1);
        auto weight = [&](int layer, int64_t rows, int64_t cols, const char* name) {
            WeightInfo w;
            w.name = w.source = name;
            w.decl.dtype = RAD_F32;
            w.decl.rank = cols ? 2 : 1;
            w.decl.shape[0] = rows;
            w.decl.shape[1] = cols;
            w.decl.group.layer = layer;
            w.decl.group.expert = -1;
            w.decl.access = RAD_ACCESS_PER_TOKEN;
            w.stored_bytes = rows * (cols ? cols : 1) * 4;
            w.tier = Tier::VRAM;
            w.ptr = rad_dev_alloc(w.stored_bytes, RAD_MEM_DEVICE);
            mem.push_back(w.ptr);
            p.weights.push_back(w);
            return (rad_weight)(p.weights.size() - 1);
        };
        auto op = [&](const char* name, rad_weight w) {
            OpInfo o;
            o.op = name;
            o.index = (int)p.ops.size();
            o.weights.push_back(w);
            p.ops.push_back(o);
        };
        op("router_scored_topk", weight(10, 384, 0, "blk.10.glm_router.correction"));
        for (int s = 0; s < 3; ++s) {
            const std::string pn = "blk." + std::to_string(40 + s) + ".glm_router.weight";
            const std::string cn = "blk." + std::to_string(40 + s) + ".glm_router.correction";
            op("gemm_nt_f32", weight(40 + s, 128, 64, pn.c_str()));
            plane[s] = weight(40 + s, 128, 0, cn.c_str());
            op("router_scored_topk", plane[s]);
        }
        for (int s = 0; s < 3; ++s) {
            std::vector<float> v(128);
            for (int e = 0; e < 128; ++e) v[(size_t)e] = e % 4 == s ? 0.25f * (float)e - 3.0f : 0.0f;
            std::memcpy(p.weights[plane[s]].ptr, v.data(), sizeof(float) * 128);
        }
    }
    ~MaskRank() { for (void* m : mem) rad_dev_free(m, RAD_MEM_DEVICE); }
    std::vector<float> read(int s) const {
        std::vector<float> v(128);
        std::memcpy(v.data(), p.weights[plane[s]].ptr, sizeof(float) * 128);
        return v;
    }
};

}  /* namespace */

TEST(draft_vram_mask_planes_writes_minus_inf_where_no_rank_keeps_and_reads_back) {
    const DraftHotset hs = parsed(hotset_text(32));
    std::vector<std::vector<uint8_t>> allowed;
    std::string why;
    REQUIRE_EQ(draft_vram_allowed(hs, 8, &allowed, &why), RAD_OK);
    MaskRank m;
    std::vector<float> before[3];
    for (int s = 0; s < 3; ++s) before[s] = m.read(s);
    rad_weight found[3];
    REQUIRE_EQ(draft_vram_find_planes(m.p, found, &why), RAD_OK);
    for (int s = 0; s < 3; ++s) CHECK_EQ(found[s], m.plane[s]);
    int kept[3] = {}, nonzero = -1;
    REQUIRE_EQ(draft_vram_mask_planes(m.p, allowed, kept, &nonzero, &why), RAD_OK);
    int want_nonzero = 0;
    for (int s = 0; s < 3; ++s) {
        CHECK_EQ(kept[s], 32);
        const std::vector<float> after = m.read(s);
        for (int e = 0; e < 128; ++e) {
            if (allowed[(size_t)s][(size_t)e]) {
                CHECK_EQ(std::memcmp(&after[(size_t)e], &before[s][(size_t)e], sizeof(float)), 0);
                want_nonzero += before[s][(size_t)e] != 0.0f;
            } else {
                CHECK(std::isinf(after[(size_t)e]) && after[(size_t)e] < 0);
            }
        }
    }
    CHECK_EQ(nonzero, want_nonzero);
    /* The trunk's router and the projections are untouched. */
    const WeightInfo& trunk = m.p.weights[1];
    for (int e = 0; e < 384; ++e) CHECK_EQ(((const float*)trunk.ptr)[e], 0.0f);
}

TEST(draft_vram_mask_planes_refuses_before_writing_anything) {
    const DraftHotset hs = parsed(hotset_text(32));
    std::vector<std::vector<uint8_t>> allowed;
    std::string why;
    REQUIRE_EQ(draft_vram_allowed(hs, 8, &allowed, &why), RAD_OK);
    for (int bad = 0; bad < 5; ++bad) {
        MaskRank m;
        std::vector<float> before[3];
        for (int s = 0; s < 3; ++s) before[s] = m.read(s);
        if (bad == 0) m.p.weights[m.plane[2]].tier = Tier::HostPinned;       /* not in VRAM */
        if (bad == 1) m.p.weights[m.plane[1]].stored_bytes = 256;            /* not the declared bytes */
        if (bad == 2) m.p.ops.back().op = "gemm_nt_f32";                      /* stage 42 has no router plane */
        if (bad == 3) {                                                       /* two planes for stage 41 */
            WeightInfo w = m.p.weights[m.plane[1]];
            m.p.weights.push_back(w);
            OpInfo o;
            o.op = "router_scored_topk";
            o.weights.push_back((rad_weight)(m.p.weights.size() - 1));
            m.p.ops.push_back(o);
        }
        if (bad == 4) m.p.weights[m.plane[0]].ptr = nullptr;
        int kept[3] = {}, nonzero = 0;
        why.clear();
        CHECK_EQ(draft_vram_mask_planes(m.p, allowed, kept, &nonzero, &why), RAD_E_STATE);
        CHECK(!why.empty());
        if (bad == 4) m.p.weights[m.plane[0]].ptr = m.mem[2];
        for (int s = 0; s < 3; ++s) {
            const std::vector<float> after = m.read(s);
            CHECK_EQ(std::memcmp(after.data(), before[s].data(), sizeof(float) * 128), 0);
        }
    }
    MaskRank m;
    int kept[3] = {}, nonzero = 0;
    std::vector<std::vector<uint8_t>> narrow = allowed;
    narrow[1].resize(96);
    CHECK_EQ(draft_vram_mask_planes(m.p, narrow, kept, &nonzero, &why), RAD_E_STATE);
}

RAD_TEST_MAIN()
