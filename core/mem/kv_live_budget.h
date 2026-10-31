/* Explicit physical budgets for the first live KV-growth cohort. Pure host rules: no device or
 * allocator calls, no inference from a logical pool pitch that omits the physical tail granule. */
#pragma once
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

namespace rad {
constexpr const char* kKVLiveEnv = "RADIANCE_EXPERIMENTAL_DS41_KV_GROW";
constexpr const char* kKVLiveMinEnv = "RADIANCE_EXPERIMENTAL_DS41_KV_GROW_MIN_MIB";
constexpr const char* kKVLiveMaxEnv = "RADIANCE_EXPERIMENTAL_DS41_KV_GROW_MAX_MIB";
constexpr const char* kKVLiveWarmEnv = "RADIANCE_EXPERIMENTAL_DS41_KV_GROW_WARM_MIB";
constexpr const char* kKVLiveHysteresisEnv = "RADIANCE_EXPERIMENTAL_DS41_KV_GROW_HYSTERESIS_MIB";
constexpr const char* kKVLiveHeadroomEnv = "RADIANCE_EXPERIMENTAL_DS41_KV_GROW_HEADROOM_MIB";
constexpr int64_t kKVLiveMiB = int64_t(1) << 20;

struct KVLiveOptions {
    bool enabled = false;
    int64_t minimum_bytes = 0;
    int64_t maximum_bytes = 0;
    int64_t warm_bytes = 0;
    int64_t hysteresis_bytes = 0;
    int64_t runtime_headroom_bytes = 0;
};

inline bool kv_live_mib(const char* p, int64_t* out) {
    if (!p || !*p || !out) return false;
    constexpr int64_t limit = std::numeric_limits<int64_t>::max() / kKVLiveMiB;
    int64_t n = 0;
    for (; *p; ++p) {
        if (*p < '0' || *p > '9' || n > (limit - (*p - '0')) / 10) return false;
        n = n * 10 + (*p - '0');
    }
    *out = n * kKVLiveMiB;
    return true;
}

template <class Get>
bool kv_live_options(const Get& get, KVLiveOptions* out, std::string* why) {
    const auto refuse = [&](const char* text) { if (why) *why = text; return false; };
    if (!out) return refuse("missing KV growth options output");
    const char* on = get(kKVLiveEnv);
    const bool enabled = on && !std::strcmp(on, "1");
    if (on && *on && std::strcmp(on, "0") && !enabled)
        return refuse("DS41_KV_GROW must be absent/0 or1");
    const char* names[] = { kKVLiveMinEnv, kKVLiveMaxEnv, kKVLiveWarmEnv,
                           kKVLiveHysteresisEnv, kKVLiveHeadroomEnv };
    KVLiveOptions o;
    o.enabled = enabled;
    int64_t* values[] = { &o.minimum_bytes, &o.maximum_bytes, &o.warm_bytes,
                         &o.hysteresis_bytes, &o.runtime_headroom_bytes };
    for (int i = 0; i < 5; ++i) {
        const char* value = get(names[i]);
        if (!enabled) {
            if (value && *value) return refuse("KV growth tunables require DS41_KV_GROW=1");
        } else if (!kv_live_mib(value, values[i]))
            return refuse("KV growth requires explicit nonnegative decimal MIN/MAX/WARM/HYSTERESIS/HEADROOM_MIB");
    }
    if (enabled && (o.maximum_bytes <= 0 || o.minimum_bytes > o.maximum_bytes))
        return refuse("KV growth requires 0<=MIN_MIB<=MAX_MIB and MAX_MIB>0");
    *out = o;
    if (why) why->clear();
    return true;
}

struct KVLiveBudget {
    int64_t logical_pool_bytes = 0;
    int64_t reserved_bytes = 0;
    int64_t carved_bytes = 0;
    int64_t granule_bytes = 0;
    int64_t currently_backed_bytes = 0;
    int64_t device_free_bytes = 0;
    int64_t startup_headroom_bytes = 0;
    int64_t late_bytes = 0;
};

/* Bounded first mode: maximum funds the whole stable VA reservation, so every advertised logical
 * page can eventually be backed. A smaller physical cap would need separate can_ever_fit logic.
 * The exact rounded maximum is explicit, even for187MiB logical with a2MiB granule (188MiB). */
inline const char* kv_live_budget_error(const KVLiveOptions& o, const KVLiveBudget& b) {
    if (!o.enabled || b.granule_bytes <= 0 || b.logical_pool_bytes <= 0 ||
        b.reserved_bytes <= 0 || b.reserved_bytes % b.granule_bytes ||
        b.logical_pool_bytes > b.reserved_bytes || b.carved_bytes < 0 ||
        b.carved_bytes > b.logical_pool_bytes || b.currently_backed_bytes < 0 ||
        b.currently_backed_bytes > b.reserved_bytes || b.device_free_bytes < 0 ||
        b.startup_headroom_bytes < 0 || b.late_bytes < 0 || o.minimum_bytes < 0 ||
        o.maximum_bytes <= 0 || o.minimum_bytes > o.maximum_bytes || o.warm_bytes < 0 ||
        o.hysteresis_bytes < 0 || o.runtime_headroom_bytes < 0)
        return "invalid physical KV budget/extent";
    if (o.maximum_bytes != b.reserved_bytes)
        return "MAX_MIB must equal the actual granule-rounded KV VA reservation in the first cohort";
    const int64_t need = o.maximum_bytes - b.currently_backed_bytes;
    if (need > b.device_free_bytes) return "the card cannot fund the explicit physical KV maximum";
    int64_t room = b.device_free_bytes - need;
    if (b.late_bytes > room) return "physical KV maximum leaves no room for late step/queue allocations";
    room -= b.late_bytes;
    const int64_t floor = b.startup_headroom_bytes > o.runtime_headroom_bytes
        ? b.startup_headroom_bytes : o.runtime_headroom_bytes;
    if (floor > room) return "physical KV maximum would consume explicit GPU headroom";
    return nullptr;
}

/* At a runtime resize, only positive incremental peak backing consumes free memory. A pure
 * release can heal pressure and needs no new funding. The caller keeps rollback-to-old funding
 * until commit; driver failures still poison safely if an external process races this check. */
inline bool kv_live_funding_fits(int64_t old_bytes, int64_t peak_bytes, int64_t maximum_bytes,
                                 int64_t free_bytes, int64_t reserve_bytes, bool adds) {
    if (old_bytes < 0 || peak_bytes < 0 || old_bytes > maximum_bytes ||
        peak_bytes > maximum_bytes || free_bytes < 0 || reserve_bytes < 0) return false;
    if (!adds) return true;
    const int64_t extra = peak_bytes > old_bytes ? peak_bytes - old_bytes : 0;
    return extra <= free_bytes && reserve_bytes <= free_bytes - extra;
}
}  // namespace rad
