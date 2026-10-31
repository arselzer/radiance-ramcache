/* Exact diagnostic arithmetic only. No allocation, dispatch, model or ABI changes. */
#pragma once
#include <cstdint>
#include <cstring>
#include <limits>
namespace rad {
constexpr const char* kKVLayoutEnv = "RADIANCE_EXPERIMENTAL_DS41_KV_GROW_LAYOUT";
inline int kv_layout_flag(const char* p) noexcept {
    return !p || !std::strcmp(p, "0") ? 0 : !std::strcmp(p, "1") ? 1 : -1;
}
inline bool kv_layout_add(int64_t a, int64_t b, int64_t* out) noexcept {
    if (!out || a < 0 || b < 0 || a > std::numeric_limits<int64_t>::max()-b) return false;
    *out = a+b; return true;
}
inline bool kv_layout_mul(int64_t a, int64_t b, int64_t* out) noexcept {
    if (!out || a < 0 || b < 0 || (b && a > std::numeric_limits<int64_t>::max()/b)) return false;
    *out = a*b; return true;
}
inline bool kv_layout_extent(int64_t offset, int64_t bytes, int64_t cap) noexcept {
    return cap >= 0 && offset >= 0 && bytes >= 0 && offset <= cap && bytes <= cap-offset;
}
} // namespace rad
