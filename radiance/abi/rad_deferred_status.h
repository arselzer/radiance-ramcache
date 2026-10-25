/* Optional device-validation status contract, independent of RadKernelInfo's ABI.
 * Both exports are absent for a legacy plugin. An opted row's collector is called
 * on the actual issue stream immediately after successful enqueue, before scratch
 * or output-status storage can be reused. It must not allocate/synchronize.
 *
 * Collector atomically latches the first NEGATIVE source status into latch->code
 * and, only if it wins that CAS, stores source_op in latch->op. It must never clear
 * an earlier error. The core resets both words before each pass, joins its lanes,
 * copies this word to owned pinned memory, then checks all ranks before sampling
 * or publishing any state. Successful collector enqueue is not a host check.
 * Callers must keep the latch allocation stable until all queued work retires.
 *
 * Version 1 initially disables pass replay, speculation and issue-ahead for any
 * program resolving an opted row. No existing RadKernelInfo field changes. */
#ifndef RAD_DEFERRED_STATUS_H
#define RAD_DEFERRED_STATUS_H
#include "rad_abi.h"
#ifdef __cplusplus
extern "C" {
#endif
#define RAD_DEFERRED_STATUS_VERSION 1u
typedef struct RadDeviceStatus {
    int32_t code;
    uint32_t op;
} RadDeviceStatus;
typedef int (*RadDeferredStatusLaunchFn)(const RadArgs* args,RadDeviceStatus* latch,
                                         uint32_t source_op,RadStream stream);
typedef uint32_t (*RadPluginDeferredStatusVersionFn)(void);
typedef RadDeferredStatusLaunchFn (*RadKernelDeferredStatusFn)(int kernel_index);
/* Optional plugin symbols. Either both must exist or neither may exist.
 * A query returns NULL for rows that do not report asynchronous validation. */
uint32_t rad_plugin_deferred_status_version(void);
RadDeferredStatusLaunchFn rad_kernel_deferred_status(int kernel_index);
#ifdef __cplusplus
}
#endif
#endif
