/* Reuse the validated scalar equations already checked against independent
 * component oracles. The caller/core orders host-visible memory; these launchers
 * neither queue work nor synchronize, initialize a device, allocate model state,
 * or opt in to deferred GPU validation. Small bounded CPU workspaces may allocate. */
#include "ds41_host_ops.h"
#include "ds41_math_emulate.h"
#include "ds41_compress_emulate.h"
#include "ds41_ffn_emulate.h"
#include "ds41_index_emulate.h"
#include "ds41_index_paged_emulate.h"
#include "ds41_transform_emulate.h"
#include "ds41_attention_paged_emulate.h"
#include "ds41_route_jobs_emulate.h"
#include "ds41_index_stream_emulate.h"
#include <new>
namespace {
template<class Body>int protect(const RadArgs* args,Body&& body) noexcept {
    if(!args)return RAD_E_INVAL;
    // Core's generic host IssuePlan may still carry a DEVICE-arena scratch
    // pointer. The host row owns no asynchronous validation output: use only a
    // private CPU flag and return failures directly, never dereference VRAM.
    int32_t status=RAD_OK;RadArgs host=*args;host.scratch=&status;host.scratch_bytes=sizeof status;
    try{return body(&host);}
    catch(const std::bad_alloc&){return RAD_E_NOMEM;}
    catch(...){return RAD_E_STATE;}
}
}
extern "C" {
int ds41_ref_hc_prepare(const RadArgs* a,RadStream) noexcept {
    return protect(a,[&](const RadArgs* host){return rad::ds41::math::emulate(rad::ds41::math::Op::hc_prepare,host);});
}
int ds41_ref_hc_collapse(const RadArgs* a,RadStream) noexcept {
    return protect(a,[&](const RadArgs* host){return rad::ds41::math::emulate(rad::ds41::math::Op::hc_collapse,host);});
}
int ds41_ref_router_topk(const RadArgs* a,RadStream) noexcept {
    return protect(a,[&](const RadArgs* host){return rad::ds41::math::emulate(rad::ds41::math::Op::router_topk,host);});
}
int ds41_ref_sparse_attn(const RadArgs* a,RadStream) noexcept {
    return protect(a,[&](const RadArgs* host){return rad::ds41::math::emulate(rad::ds41::math::Op::sparse_attn,host);});
}
int ds41_ref_compress_pool(const RadArgs* a,RadStream) noexcept {
    return protect(a,[&](const RadArgs* host){return rad::ds41::compress::emulate(host);});
}
int ds41_ref_expert_act(const RadArgs* a,RadStream) noexcept {
    return protect(a,[&](const RadArgs* host){return rad::ds41::ffn::emulate(rad::ds41::ffn::Op::expert_act,host);});
}
int ds41_ref_moe_combine(const RadArgs* a,RadStream) noexcept {
    return protect(a,[&](const RadArgs* host){return rad::ds41::ffn::emulate(rad::ds41::ffn::Op::moe_combine,host);});
}
int ds41_ref_mhc_post(const RadArgs* a,RadStream) noexcept {
    return protect(a,[&](const RadArgs* host){return rad::ds41::ffn::emulate(rad::ds41::ffn::Op::mhc_post,host);});
}
int ds41_ref_index_scores(const RadArgs* a,RadStream) noexcept {
    return protect(a,[&](const RadArgs* host){return rad::ds41::index::emulate(rad::ds41::index::Op::scores,host);});
}
int ds41_ref_candidate_blocks(const RadArgs* a,RadStream) noexcept {
    return protect(a,[&](const RadArgs* host){return rad::ds41::index::emulate(rad::ds41::index::Op::candidates,host);});
}
int ds41_ref_index_topk(const RadArgs* a,RadStream) noexcept {
    return protect(a,[&](const RadArgs* host){return rad::ds41::index::emulate(rad::ds41::index::Op::topk,host);});
}
int ds41_ref_index_scores_paged(const RadArgs* a,RadStream) noexcept {
    return protect(a,[&](const RadArgs* host){return rad::ds41::paged_index::emulate(host);});
}
int ds41_ref_rmsnorm(const RadArgs* a,RadStream) noexcept {
    return protect(a,[&](const RadArgs* host){return rad::ds41::transform::emulate(rad::ds41::transform::Op::rmsnorm,host);});
}
int ds41_ref_rope_tail(const RadArgs* a,RadStream) noexcept {
    return protect(a,[&](const RadArgs* host){return rad::ds41::transform::emulate(rad::ds41::transform::Op::rope_tail,host);});
}
int ds41_ref_sparse_attn_paged(const RadArgs* a,RadStream) noexcept {
    return protect(a,[&](const RadArgs* host){return rad::ds41::attention_paged::emulate(host);});
}
int ds41_ref_route_jobs(const RadArgs* a,RadStream) noexcept {
    return protect(a,[&](const RadArgs* host){return rad::ds41::route_jobs::emulate(host);});
}
int ds41_ref_index_stream(const RadArgs* a,RadStream) noexcept {
    return protect(a,[](const RadArgs* host){return rad::ds41::index_stream::emulate(host);});
}
}
