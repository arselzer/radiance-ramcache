/* Private per-call native cache policy. No mutable global worker policy or public ABI. */
#pragma once
#include "rad_runtime.h"
#include <cstdint>
namespace rad {
enum class NativeFileCacheReadPolicy : uint8_t { Normal, ReadOnly };
inline constexpr const char* kNativeCachePrefillReadOnlyEnv =
    "RADIANCE_EXPERIMENTAL_NATIVE_CACHE_PREFILL_READ_ONLY";
inline bool native_file_cache_policy_valid(NativeFileCacheReadPolicy policy) {
    return policy == NativeFileCacheReadPolicy::Normal || policy == NativeFileCacheReadPolicy::ReadOnly;
}
inline NativeFileCacheReadPolicy native_file_cache_prefill_policy(
        const char* opt, const RadBatch* batch, bool exact_native_EP_layer) {
    return opt && opt[0] == '1' && opt[1] == '\0' && exact_native_EP_layer && batch &&
        batch->phase == RAD_PHASE_PREFILL && batch->n_tok_decode == 0 &&
        batch->draft_pass == 0 && batch->n_tok > 16
        ? NativeFileCacheReadPolicy::ReadOnly : NativeFileCacheReadPolicy::Normal;
}
} // namespace rad
