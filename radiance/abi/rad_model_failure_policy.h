/* Additive optional model lifecycle requirement. Existing ABI structs/version27
 * are unchanged. Plugins must weak-probe BOTH exports before declaring this
 * requirement, and refuse rad_plugin_open on older cores. This capability does
 * not promise rollback, GPU fault recovery, TP cancellation or safe failed-drain
 * arena teardown. Default Programs retain their legacy per-pass reset policy. */
#ifndef RAD_MODEL_FAILURE_POLICY_H
#define RAD_MODEL_FAILURE_POLICY_H
#include <stdint.h>
#include "rad_builder.h"
#define RAD_MODEL_FAILURE_POLICY_VERSION 1u
#ifdef __cplusplus
extern "C" {
#endif
uint32_t rad_core_model_failure_policy_version(void);
/* Monotonic/idempotent during declare; refused after Builder::finish. */
int rad_require_permanent_model_failure(RadBuilder* builder);
#ifdef __cplusplus
}
#endif
#endif
