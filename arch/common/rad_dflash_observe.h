/* Private, optional dflash_observe output contract; no plugin ABI struct changes.
 *
 * The selector is unchanged. After it finishes, the observer recomputes its scores with the
 * same lane fold, using the selected path as predecessors. candidate_p is exp(score[pick]) /
 * sum(exp(score[c])) over the EXISTING head top-K candidates, at temperature 1. It is neither a
 * full-vocabulary drafter probability nor a probability of target acceptance. Markov bias for
 * tokens outside that set is not evaluated. Calibrate this signal against reached outcomes.
 *
 * One persistent F32 [sequences, kWidth] plane, kFields per depth, only depths 1 and 2. IDs fit
 * exactly in F32 (the observer refuses vocabularies above 2^24). Score/margin/p are NaN when the
 * selected ID is not the selector's argmax, any score is nonfinite, candidates repeat, or IDs
 * are out of range. The output must not alias any input.
 * Present depths always echo the selected/predecessor IDs, even when signals are invalid;
 * numerical invalidity alone is not evidence of a stale proposal. Missing depths have NaN
 * signals and both IDs -1. margin is selected score minus the largest OTHER
 * candidate score (positive infinity for a singleton set). A separate opt-in prefix policy may
 * read this signal after ID/context binding; capture and fixed measurement alone never do.
 */
#pragma once
#include <cstring>
namespace rad_dflash_observe {
constexpr const char* kEnv = "RADIANCE_EXPERIMENTAL_DS41_DSPARK_PROB_CAPTURE";
constexpr const char* kEarlyEnv = "RADIANCE_EXPERIMENTAL_DS41_DSPARK_EARLY_VERIFY";
constexpr const char* kPolicyEnv = "RADIANCE_EXPERIMENTAL_DS41_DSPARK_PREFIX_POLICY";
inline bool policy_requested(const char* x) {
    return x && std::strcmp(x, "off") && std::strcmp(x, "0");
}
/* Absent/off leaves the old issue order;0/1/2 are FIXED measurement controls, no policy. */
inline int early_fixed(const char* value) {
    if (!value || (value[0] == 'o' && value[1] == 'f' && value[2] == 'f' && value[3] == '\0')) return -1;
    return value[0] >= '0' && value[0] <= '2' && value[1] == '\0' ? value[0] - '0' : -2;
}
constexpr const char* kBuffer = "dspark_proposal_observations";
enum { kDepths = 2, kFields = 5, kWidth = kDepths * kFields };
enum Field { kCandidateP, kScore, kMargin, kToken, kPredecessor };
}
