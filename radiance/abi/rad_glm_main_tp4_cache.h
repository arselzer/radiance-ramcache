#ifndef RAD_GLM_MAIN_TP4_CACHE_H
#define RAD_GLM_MAIN_TP4_CACHE_H
// Optional additive capability. Frozen ABI27 structs/version are unchanged.
// Experimental plugins MUST weak-probe both exports and reject older cores.
// This is a distinct process-fail-stop TP4 policy, never PermanentV1 recovery.
#include "rad_builder.h"
#include <stdint.h>
#define RAD_GLM_MAIN_TP4_CACHE_PROFILE_VERSION 1u
#ifdef __cplusplus
extern "C" {
#endif
uint32_t rad_core_glm_main_tp4_cache_profile_version(void);
int rad_require_glm_main_tp4_cache_fail_stop(RadBuilder*);
#ifdef __cplusplus
}
#endif
#endif
