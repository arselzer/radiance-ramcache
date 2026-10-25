#ifndef RAD_EXPERT_PREPARE_H
#define RAD_EXPERT_PREPARE_H
#include "rad_runtime.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Experimental, opt-in current-route seam. Requires a matching complete Core/architecture build.
 * Slot is the scatter's regular/protected ordering, canonical is model expert numbering.
 * members: GU codes/scale, DN codes/scale; absent BF16 scales are zero. */
typedef struct RadExpertRouteEntry {
    int32_t canonical,slot,class_index,protected_expert;
    rad_weight members[4];
} RadExpertRouteEntry;
typedef struct RadExpertRouteDecl {
    int32_t layer,canonical_experts,local_experts,expert_base,regular_experts,classes;
    int64_t max_rows,top_k;
    rad_op gu;
    rad_buf counts,offsets,sorted,ids;
    const RadExpertRouteEntry* entries;
    int32_t n_entries;
} RadExpertRouteDecl;
/* Copies all entries; no borrowed caller metadata survives declaration. */
int rad_decl_expert_route_map(RadBuilder*,const RadExpertRouteDecl*);
typedef struct RadExpertPrepare {
    int32_t layer;
    int64_t total_rows,first_row,rows;
    RadOperand counts,offsets,sorted,ids;
} RadExpertPrepare;
/* A disabled owner is a zero-bind/copy/allocation no-op. A negative active result is fatal:
 * the architecture must return before any dependent GU/collective. */
int rad_expert_prepare(RadCtx*,const RadExpertPrepare*);
#ifdef __cplusplus
}
#endif
#endif
