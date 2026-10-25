/* Synchronous scalar host-reference launchers. Registration remains in the
 * explicitly marked libds41_ref plugin, never in a production serving profile.
 * Inputs must be host-accessible and ordered behind any producer before launch.
 * Payload failures return directly; no deferred GPU collector is needed. Caller
 * scratch is ignored: each call supplies a private four-byte CPU status flag so
 * a core host IssuePlan cannot accidentally expose device-arena scratch to CPU. */
#pragma once
#include "rad_plugin.h"
extern "C" {
int ds41_ref_hc_prepare(const RadArgs*,RadStream) noexcept;
int ds41_ref_hc_collapse(const RadArgs*,RadStream) noexcept;
int ds41_ref_router_topk(const RadArgs*,RadStream) noexcept;
int ds41_ref_sparse_attn(const RadArgs*,RadStream) noexcept;
int ds41_ref_compress_pool(const RadArgs*,RadStream) noexcept;
int ds41_ref_expert_act(const RadArgs*,RadStream) noexcept;
int ds41_ref_moe_combine(const RadArgs*,RadStream) noexcept;
int ds41_ref_mhc_post(const RadArgs*,RadStream) noexcept;
int ds41_ref_index_scores(const RadArgs*,RadStream) noexcept;
int ds41_ref_candidate_blocks(const RadArgs*,RadStream) noexcept;
int ds41_ref_index_topk(const RadArgs*,RadStream) noexcept;
int ds41_ref_index_scores_paged(const RadArgs*,RadStream) noexcept;
int ds41_ref_rmsnorm(const RadArgs*,RadStream) noexcept;
int ds41_ref_rope_tail(const RadArgs*,RadStream) noexcept;
int ds41_ref_sparse_attn_paged(const RadArgs*,RadStream) noexcept;
int ds41_ref_route_jobs(const RadArgs*,RadStream) noexcept;
int ds41_ref_index_stream(const RadArgs*,RadStream) noexcept;
}
