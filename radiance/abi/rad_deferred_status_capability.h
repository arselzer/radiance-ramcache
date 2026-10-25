/* Core capability probe for plugins whose validation requires the optional
 * deferred-status collector contract. Kept separate from the kernel contract to
 * avoid changing hot include dependencies when adding this host-only probe.
 *
 * A plugin may redeclare this symbol weak and refuse rad_plugin_open when an
 * older core does not export it. Direct kernel/checker callers still own their
 * status checks explicitly. This function performs no device work. */
#ifndef RAD_DEFERRED_STATUS_CAPABILITY_H
#define RAD_DEFERRED_STATUS_CAPABILITY_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
uint32_t rad_core_deferred_status_version(void);
#ifdef __cplusplus
}
#endif
#endif
