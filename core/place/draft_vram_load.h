/* draft_vram_load.h -- RADIANCE_EXPERIMENTAL_DS41_DRAFT_VRAM's (draft_vram.h) two load-time steps, on
 * one rank's Program and Plan, apart from the Engine so the host backend can test them:
 *
 *   draft_vram_aliases     where each parked draft weight is published: the same slot (gate_up, down)
 *                          of its stage's first resident draft expert, as the mover placed it, so
 *                          every pointer a grouped GEMM is handed names card memory
 *   draft_vram_mask_planes each draft stage's router correction plane (the f32 weight of the stage's
 *                          n_expert entries a router_* op reads, found by reader and shape, not name)
 *                          read back from the card, written with -inf at every expert no rank holds
 *                          (draft_mask_plane), and read back again to check */
#pragma once
#include "draft_vram.h"
#include "mover.h"
#include "planner.h"

#include <string>
#include <vector>

namespace rad {

/* By rad_weight, the address a parked draft weight is published at (null elsewhere). Every stage keeps
 * at least one resident (draft_vram_allowed refuses less); a parked weight left without one, or a
 * resident without an address, is RAD_E_STATE with `why`. */
inline int draft_vram_aliases(const Program& prog, const Plan& plan, Mover& mover,
                              std::vector<void*>* out, std::string* why = nullptr) {
    auto no = [&](std::string s) { if (why) *why = std::move(s); return (int)RAD_E_STATE; };
    out->assign(prog.weights.size(), nullptr);
    void* first[kDraftStages][2] = {};
    for (const MoveUnit& mu : plan.units) {
        if (!mu.draft_vram) continue;
        const int s = mu.layer - kDraftFirstLayer;
        if (s < 0 || s >= kDraftStages) return no("a hot draft expert outside the stages");
        for (rad_weight w : mu.weights) {
            const int slot = prog.weights[w].decl.group.slot;
            if (slot < 0 || slot > 1) return no("a draft expert weight of an unknown slot");
            if (first[s][slot]) continue;
            const WeightSlot* ws = mover.residency()->lookup(w);
            if (!ws || !ws->ptr)
                return no("the resident '" + prog.weights[w].name + "' has no address after mover init");
            first[s][slot] = ws->ptr;
        }
    }
    for (const MoveUnit& mu : plan.units) {
        if (!mu.draft_parked) continue;
        const int s = mu.layer - kDraftFirstLayer;
        if (s < 0 || s >= kDraftStages) return no("a parked draft expert outside the stages");
        for (rad_weight w : mu.weights) {
            const int slot = prog.weights[w].decl.group.slot;
            void* a = slot >= 0 && slot <= 1 ? first[s][slot] : nullptr;
            if (!a) return no("the parked '" + prog.weights[w].name +
                              "' has no resident of its stage and slot to be published at");
            (*out)[w] = a;
        }
    }
    return RAD_OK;
}

/* The three stages' correction planes: the weight a router_* op reads at layer 40 + s with f32 rank-1
 * shape {kDraftExperts}, no expert group. RAD_E_STATE with `why` when a stage has none or two. */
inline int draft_vram_find_planes(const Program& prog, rad_weight out[kDraftStages], std::string* why) {
    for (int s = 0; s < kDraftStages; ++s) out[s] = 0;
    for (const OpInfo& o : prog.ops) {
        if (o.op.rfind("router_", 0) != 0) continue;
        for (rad_weight w : o.weights) {
            if (!w || (size_t)w >= prog.weights.size()) continue;
            const WeightInfo& wi = prog.weights[w];
            const int32_t l = wi.decl.group.layer;
            if (l < kDraftFirstLayer || l >= kDraftFirstLayer + kDraftStages || wi.decl.group.expert >= 0 ||
                wi.decl.dtype != RAD_F32 || wi.decl.rank != 1 || wi.decl.shape[0] != kDraftExperts)
                continue;
            rad_weight& at = out[l - kDraftFirstLayer];
            if (at && at != w) {
                if (why) *why = "two router correction planes for stage " + std::to_string(l);
                return RAD_E_STATE;
            }
            at = w;
        }
    }
    for (int s = 0; s < kDraftStages; ++s)
        if (!out[s]) {
            if (why) *why = "stage " + std::to_string(kDraftFirstLayer + s) +
                            " has no router correction plane (an f32 weight of 128 entries a router_* op reads)";
            return RAD_E_STATE;
        }
    return RAD_OK;
}

/* Masks the rank's three correction planes in place on the current device: `allowed` is
 * [stage][model-wide expert] (draft_vram_allowed). `kept[s]` gets the experts each keeps and
 * `nonzero` how many kept values are not 0. Each plane must be resident in VRAM at its declared
 * 512 bytes; RAD_E_STATE with `why` when not, or when a plane does not read back masked. */
inline int draft_vram_mask_planes(const Program& prog, const std::vector<std::vector<uint8_t>>& allowed,
                                  int kept[kDraftStages], int* nonzero, std::string* why) {
    auto no = [&](std::string s) { if (why) *why = std::move(s); return (int)RAD_E_STATE; };
    if (allowed.size() != (size_t)kDraftStages) return no("no mask for every stage");
    for (const auto& a : allowed) if (a.size() != (size_t)kDraftExperts) return no("a mask of the wrong width");
    rad_weight plane[kDraftStages];
    RAD_TRY(draft_vram_find_planes(prog, plane, why));
    const int64_t bytes = kDraftExperts * (int64_t)sizeof(float);
    for (int s = 0; s < kDraftStages; ++s) {
        const WeightInfo& wi = prog.weights[plane[s]];
        /* Stored as the router reads it: 128 f32 (a plane widened from bf16 at load is that too). */
        if (wi.tier != Tier::VRAM || !wi.ptr || wi.stored_bytes != bytes)
            return no("stage " + std::to_string(kDraftFirstLayer + s) + "'s correction plane '" + wi.name +
                      "' is not " + std::to_string(bytes) + " f32 bytes resident in VRAM as declared");
    }
    float* host = (float*)rad_dev_alloc(3 * bytes, RAD_MEM_HOST_PINNED);
    if (!host) return RAD_E_NOMEM;
    float* orig = host;
    float* want = host + kDraftExperts;
    float* back = host + 2 * kDraftExperts;
    RadStream st = nullptr;
    int rc = rad_stream_create(&st, 0);
    if (nonzero) *nonzero = 0;
    for (int s = 0; s < kDraftStages && rc >= 0; ++s) {
        const WeightInfo& wi = prog.weights[plane[s]];
        const uint8_t* a = allowed[(size_t)s].data();
        if ((rc = rad_memcpy_async(orig, wi.ptr, bytes, st)) < 0 || (rc = rad_stream_sync(st)) < 0) break;
        draft_mask_plane(orig, a, kDraftExperts, want);
        if ((rc = rad_memcpy_async(wi.ptr, want, bytes, st)) < 0 ||
            (rc = rad_memcpy_async(back, wi.ptr, bytes, st)) < 0 || (rc = rad_stream_sync(st)) < 0)
            break;
        if (!draft_mask_holds(back, orig, a, kDraftExperts)) {
            rc = no("stage " + std::to_string(kDraftFirstLayer + s) + "'s correction plane did not read back masked");
            break;
        }
        kept[s] = 0;
        for (int e = 0; e < kDraftExperts; ++e) {
            kept[s] += a[e];
            if (nonzero) *nonzero += a[e] && orig[e] != 0.0f;
        }
    }
    if (st) rad_stream_destroy(st);
    rad_dev_free(host, RAD_MEM_HOST_PINNED);
    return rc < 0 ? rc : RAD_OK;
}

}  /* namespace rad */
