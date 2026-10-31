/* Only the existing index score plane's row floor changes. No cache/page/kernel arithmetic. */
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
namespace deepseek41 {
constexpr const char* kIndexCompactEnv = "RADIANCE_EXPERIMENTAL_DS41_INDEX_SCORE_COMPACT";
constexpr int64_t kScorePlane = 256ll << 20;
constexpr int64_t kScoreRunMin = 256;
constexpr int64_t kScoreCompactMin = 64;
inline bool index_compact_parse(const char* value, bool* requested) {
    *requested = value && !std::strcmp(value, "1");
    return !value || !*value || !std::strcmp(value, "0") || *requested;
}
/* Keep the old quotient, including the guard column, exactly. Above the budget's reach the
 * smaller floor is still a floor, not a strict byte ceiling (64 * (1Mi + 1) * 4 = 256MiB+256B).
 * The caller must require valid LCP and core admission of actual scalar/selection dispatches. */
inline int64_t index_plane_rows(int64_t max_tok, int64_t columns, bool compact) {
    if (max_tok <= 0 || columns <= 0) return 0;
    return std::min(max_tok, std::max(compact ? kScoreCompactMin : kScoreRunMin,
                                      (kScorePlane / 4) / columns));
}
} // namespace deepseek41
