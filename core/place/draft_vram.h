/* draft_vram.h -- RADIANCE_EXPERIMENTAL_DS41_DRAFT_VRAM: the DSpark draft bank's hot experts as VRAM
 * residents, nothing of the bank in the pinned host pool, and the draft router masked to the residents.
 *
 * The bank (--draft-bank: three V4 stages at layers 40..42, 128 routed experts a stage, 32 a rank at
 * EP4, top-3) is served today from the pinned host pool (--draft-bank-expert-placement host: all 96
 * pairs a rank, 1.68 GiB, read in place over the link), which costs the trunk the host pool (p2048
 * instead of p256) and 6 GiB of RAM cache (90 GiB instead of 96). With
 *
 *   RADIANCE_EXPERIMENTAL_DS41_DRAFT_VRAM=N   (1..32; 0 or unset: off)
 *   RADIANCE_EXPERIMENTAL_DS41_DRAFT_HOTSET=<file>
 *
 * and --draft-bank-expert-placement host (the bank's manual contract, whose pinned pool this replaces),
 * every rank instead
 *   - places the first N experts of each of its three stage lists in the hot-set file as static VRAM
 *     residents, charged to the weights pool before the strategy fills it, so the planner gives up as
 *     many trunk residents (Planner::place_draft_bank_vram; a pair is a trunk expert's size);
 *   - loads no other draft expert: the rest are parked (Tier::Mapped with no class, slot or pool, never
 *     read), each published at a resident draft expert's address on its own card, so every pointer the
 *     grouped GEMM is handed names card memory even where a kernel reads a zero-row expert;
 *   - pins nothing in the host pool; and
 *   - after the load writes -inf into each stage's router correction plane at every expert no rank
 *     holds (draft_mask_plane). router_scored_topk ranks the experts by score + correction and never
 *     takes a -inf key (libref's scan and libr4d's wave alike), so with at least top_k residents a
 *     stage the router takes the best three residents, weighted by their unbiased scores renormalised
 *     over the three: the static mask of the draft-cache design (DESIGN.md section 2.1). The bank's
 *     correction planes are zero, so the residents keep their scores' order.
 * The target verifies every draft, so the mask moves acceptance and speed, never a reply byte.
 *
 * ---- the hot-set file (tools/ds41-host-experts/draft_hotset.py writes it from --expert-stats files)
 *
 *   {"format": "radiance-ds41-draft-hotset/1", "layers": [40, 41, 42], "ranks": 4, "experts": 128,
 *    "hot": [ [[g, ...], [g, ...], [g, ...], [g, ...]],     <- stage 40: one list a rank
 *             [...], [...] ]}                               <- stages 41, 42
 *
 * Each list holds model-wide expert ids (the container's numbering, --expert-stats' too), hottest
 * first; the engine keeps the first N of each, so one file written with every expert ranked serves
 * every N. Each id must be one the listed rank holds (checked against the container's directory). A
 * stage left with fewer than top_k experts in all is refused: the router would have to take one the
 * card does not hold.
 *
 * Header-only, no core dependencies beyond the status codes, so the libr4d emulator's test of the
 * router kernel can build the same masked plane (tests/r4d_emu/test_ds41mask.cpp). The file is parsed
 * in draft_vram.cpp. */
#pragma once
#include "rad_types.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace rad {

inline constexpr char kDraftVramEnv[]   = "RADIANCE_EXPERIMENTAL_DS41_DRAFT_VRAM";
inline constexpr char kDraftHotsetEnv[] = "RADIANCE_EXPERIMENTAL_DS41_DRAFT_HOTSET";
inline constexpr char kDraftHotsetFormat[] = "radiance-ds41-draft-hotset/1";

/* The bank's native geometry (dspark_native_scope.h holds the Program to it). */
inline constexpr int32_t kDraftFirstLayer = 40, kDraftStages = 3, kDraftRanks = 4;
inline constexpr int32_t kDraftExperts = 128, kDraftRankExperts = 32, kDraftTopK = 3;
inline constexpr int64_t kDraftPairBytes = 18800640;   /* gate_up 12,533,760 + down 6,266,880 */

/* The switch's value: unset, empty or "0" is off (*n = 0); 1..32 is the experts of each stage a rank
 * keeps in VRAM. Anything else is RAD_E_INVAL. */
inline int draft_vram_switch(const char* v, int* n) {
    *n = 0;
    if (!v || !*v || !std::strcmp(v, "0")) return RAD_OK;
    int x = 0;
    for (const char* c = v; *c; ++c) {
        if (*c < '0' || *c > '9' || c - v >= 2) return RAD_E_INVAL;   /* one or two digits, nothing else */
        x = 10 * x + (*c - '0');
    }
    if (x < 1 || x > kDraftRankExperts) return RAD_E_INVAL;
    *n = x;
    return RAD_OK;
}

/* A hot-set file, validated: every list's ids in [0, experts), distinct within the list. */
struct DraftHotset {
    std::string path;
    int32_t     first_layer = kDraftFirstLayer;
    int32_t     ranks = kDraftRanks;
    int32_t     experts = kDraftExperts;
    /* [stage][rank]: model-wide expert ids, hottest first. */
    std::vector<std::vector<std::vector<int32_t>>> hot;
};

/* draft_vram.cpp: the file's text, or the file. RAD_E_INVAL (RAD_E_IO for an unreadable file) with
 * the reason in `why` for anything but the format above at the bank's geometry. */
int draft_hotset_parse(const std::string& text, DraftHotset* out, std::string* why);
int draft_hotset_load(const std::string& path, DraftHotset* out, std::string* why);

/* Each stage's experts some rank keeps at `n`: [stage][model-wide expert] 1 or 0, the union of the
 * first `n` ids of every rank's list. RAD_E_INVAL with `why` when a list is shorter than `n`, two
 * ranks list the same expert, or a stage keeps fewer than kDraftTopK. */
inline int draft_vram_allowed(const DraftHotset& hs, int n, std::vector<std::vector<uint8_t>>* out,
                              std::string* why) {
    auto no = [&](std::string s) { if (why) *why = std::move(s); return (int)RAD_E_INVAL; };
    if (n < 1 || n > kDraftRankExperts) return no("the experts a stage a rank must be 1..32");
    if ((int)hs.hot.size() != kDraftStages) return no("the hot set does not list three stages");
    out->assign((size_t)kDraftStages, std::vector<uint8_t>((size_t)hs.experts, 0));
    for (int s = 0; s < kDraftStages; ++s) {
        if ((int)hs.hot[(size_t)s].size() != hs.ranks) return no("a stage does not list every rank");
        int kept = 0;
        for (int r = 0; r < hs.ranks; ++r) {
            const std::vector<int32_t>& l = hs.hot[(size_t)s][(size_t)r];
            if ((int)l.size() < n)
                return no("stage " + std::to_string(hs.first_layer + s) + " lists " +
                          std::to_string(l.size()) + " experts for rank " + std::to_string(r) +
                          ", fewer than the " + std::to_string(n) + " asked for");
            for (int i = 0; i < n; ++i) {
                const int32_t g = l[(size_t)i];
                if (g < 0 || g >= hs.experts) return no("an expert id out of range");
                uint8_t& a = (*out)[(size_t)s][(size_t)g];
                if (a) return no("stage " + std::to_string(hs.first_layer + s) + ": expert " +
                                 std::to_string(g) + " is listed for two ranks");
                a = 1;
                ++kept;
            }
        }
        if (kept < kDraftTopK)
            return no("stage " + std::to_string(hs.first_layer + s) + " keeps " + std::to_string(kept) +
                      " experts, fewer than the router's top-" + std::to_string(kDraftTopK));
    }
    return RAD_OK;
}

/* One rank's part: which of its own units (local expert ids, the units' group.expert) are resident,
 * and each unit's model-wide id. */
struct DraftVramRank {
    int32_t n = 0;                                 /* experts a stage this rank keeps; 0 is off */
    std::vector<std::vector<uint8_t>> resident;    /* [stage][local expert] */
    std::vector<std::vector<int32_t>> global;      /* [stage][local expert]: model-wide id */
    int64_t residents() const {
        int64_t c = 0;
        for (const auto& s : resident) for (uint8_t v : s) c += v;
        return c;
    }
};

/* Rank `rank`'s residents at `n`: `global_of(layer, local)` is the model-wide id of the rank's unit
 * (layer, local), from the container's directory, or -1. RAD_E_INVAL with `why` when a unit has no
 * id or two share one, or a listed expert is not one of the rank's. */
inline int draft_vram_rank(const DraftHotset& hs, int n, int rank,
                           const std::function<int32_t(int32_t, int32_t)>& global_of,
                           DraftVramRank* out, std::string* why) {
    auto no = [&](std::string s) { if (why) *why = std::move(s); return (int)RAD_E_INVAL; };
    if (rank < 0 || rank >= hs.ranks || (int)hs.hot.size() != kDraftStages) return no("no such rank");
    out->n = n;
    out->resident.assign((size_t)kDraftStages, std::vector<uint8_t>((size_t)kDraftRankExperts, 0));
    out->global.assign((size_t)kDraftStages, std::vector<int32_t>((size_t)kDraftRankExperts, -1));
    for (int s = 0; s < kDraftStages; ++s) {
        const int32_t layer = hs.first_layer + s;
        std::vector<int32_t> local_of((size_t)hs.experts, -1);
        for (int e = 0; e < kDraftRankExperts; ++e) {
            const int32_t g = global_of(layer, e);
            if (g < 0 || g >= hs.experts)
                return no("rank " + std::to_string(rank) + "'s unit " + std::to_string(e) + " of layer " +
                          std::to_string(layer) + " names no model-wide expert in the container");
            if (local_of[(size_t)g] >= 0)
                return no("rank " + std::to_string(rank) + " holds expert " + std::to_string(g) +
                          " of layer " + std::to_string(layer) + " twice");
            local_of[(size_t)g] = e;
            out->global[(size_t)s][(size_t)e] = g;
        }
        const std::vector<int32_t>& l = hs.hot[(size_t)s][(size_t)rank];
        if ((int)l.size() < n) return no("a list is shorter than the experts asked for");
        for (int i = 0; i < n; ++i) {
            const int32_t g = l[(size_t)i];
            const int32_t e = g >= 0 && g < hs.experts ? local_of[(size_t)g] : -1;
            if (e < 0)
                return no("the hot set lists expert " + std::to_string(g) + " of layer " +
                          std::to_string(layer) + " for rank " + std::to_string(rank) +
                          ", which does not hold it");
            out->resident[(size_t)s][(size_t)e] = 1;
        }
    }
    return RAD_OK;
}

/* The masked correction plane: the plane's own value where `allowed`, -inf everywhere else. */
inline void draft_mask_plane(const float* plane, const uint8_t* allowed, int n, float* out) {
    for (int e = 0; e < n; ++e)
        out[e] = allowed[e] ? plane[e] : -std::numeric_limits<float>::infinity();
}

/* Whether `got` is exactly draft_mask_plane(original, allowed): the kept values bit for bit, -inf at
 * every other expert. */
inline bool draft_mask_holds(const float* got, const float* original, const uint8_t* allowed, int n) {
    for (int e = 0; e < n; ++e) {
        if (allowed[e]) {
            if (std::memcmp(&got[e], &original[e], sizeof(float))) return false;
        } else if (!(std::isinf(got[e]) && got[e] < 0)) {
            return false;
        }
    }
    return true;
}

}  /* namespace rad */
