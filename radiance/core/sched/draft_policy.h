/* Caller-supplied, TRAIN-fitted prefix thresholds. No fitted defaults live in the engine.
 * candidate_p is a share within the existing candidate set, NEVER an acceptance probability. */
#pragma once
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdint>

namespace rad {
inline bool draft_prefix_policy_profile(int max_seqs, bool tree, int64_t short_from,
                                         int64_t full_rows, int64_t depth, int64_t short_rows) noexcept {
    return max_seqs == 1 && !tree && short_from == 1 && full_rows == 5 && depth == 5 && short_rows == 2;
}
struct DraftPrefixPolicy {
    bool on = false;
    double p1_min = 0, p2_min = 0;
    char spec[160] = {};
    static bool parse(const char* text, DraftPrefixPolicy* out) noexcept {
        *out = {};
        if (!text || !std::strcmp(text, "off") || !std::strcmp(text, "0")) return true;
        if (std::strlen(text) >= sizeof(out->spec) || std::strncmp(text, "v1:", 3)) return false;
        const char* p = text + 3;
        double values[2] = {};
        for (int i = 0; i < 2; ++i) {
            const char* begin = p;
            if (*p < '0' || *p > '9') return false;
            while (*p && *p != ':') {
                if (!((*p >= '0' && *p <= '9') || *p == '.' || *p == 'e' || *p == 'E' ||
                      *p == '+' || *p == '-')) return false;
                ++p;
            }
            char* end = nullptr;
            values[i] = std::strtod(begin, &end);
            if (end != p || !std::isfinite(values[i]) || values[i] < 0 || values[i] > 1 ||
                (i == 0 ? *p != ':' : *p != '\0')) return false;
            if (i == 0) ++p;
        }
        out->on = true; out->p1_min = values[0]; out->p2_min = values[1];
        std::strcpy(out->spec, text);
        return true;
    }
    /* IDs/context are checked by the shared measurement boundary before this function. Any
     * numerically invalid observer field restores the existing full Db2 prefix, including an
     * invalid depth2 signal when depth1 alone would otherwise choose zero. */
    int select(float p1, float p2, unsigned invalid_depths, bool* fallback = nullptr) const noexcept {
        const bool invalid = invalid_depths || !std::isfinite(p1) || !std::isfinite(p2) ||
                             p1 < 0 || p1 > 1 || p2 < 0 || p2 > 1;
        if (fallback) *fallback = invalid;
        if (!on || invalid) return 2;
        return p1 < p1_min ? 0 : p2 < p2_min ? 1 : 2;
    }
};
} // namespace rad
