/* Read-only DSpark proposal observations. Scheduler owns every pending record under mu_.
 * Fixed storage: no allocation in planning/commit, no state in Request or a plugin ABI struct.
 * Anchor is the next verify's first position; depth d proposes absolute position anchor+d.
 * An epoch belongs to one actual query pass, never to the confidence controller's last score.
 */
#pragma once
#include "../../arch/common/rad_dflash_observe.h"
#include <cstdint>

namespace rad {
struct DraftProbPending {
    uint64_t epoch = 0;
    int64_t anchor = -1;
    int32_t anchor_token = -1;
    int n = 0;
    int candidates = 0;
    int32_t token[rad_dflash_observe::kDepths] = {};
    float signal[rad_dflash_observe::kWidth] = {};
};
struct DraftProbStats {
    uint64_t proposed = 0, resolved = 0, verified = 0, reached = 0, accepted = 0;
};
struct DraftProbOutcome {
    bool matched = false;
    bool verified = false;
    bool reached = false;
    bool accepted = false;
};
/* `verified` means a matching target result was observed at commit, even if a cancellation
 * suppresses emission. An evaluated second position after a rejected first is verified but
 * UNREACHED: accepted is unknown, not false. JSON writes null unless reached is true. */
inline DraftProbOutcome draft_prob_outcome(const DraftProbPending& p, int depth, int64_t anchor,
                                           const int32_t* tokens, int verified, int accepted) {
    DraftProbOutcome o;
    if (depth < 0 || depth >= p.n || p.anchor != anchor) return o;
    if (verified <= 0) { o.matched = true; return o; }
    if (!tokens) return o;
    /* Check the whole verified prefix: an identical second token behind a different predecessor
     * is not the same conditional proposal. */
    for (int j = 0; j <= depth && j < verified; ++j)
        if (tokens[j] != p.token[j]) return o;
    o.matched = true;
    o.verified = depth < verified;
    o.reached = o.verified && depth <= accepted;
    o.accepted = o.reached && depth < accepted;
    return o;
}
}
