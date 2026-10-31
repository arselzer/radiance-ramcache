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
 * Missing depths have NaN signals and ID -1. margin is selected score minus the largest OTHER
 * candidate score (positive infinity for a singleton set). No value feeds a model or policy.
 */
#pragma once
namespace rad_dflash_observe {
constexpr const char* kEnv = "RADIANCE_EXPERIMENTAL_DS41_DSPARK_PROB_CAPTURE";
constexpr const char* kBuffer = "dspark_proposal_observations";
enum { kDepths = 2, kFields = 5, kWidth = kDepths * kFields };
enum Field { kCandidateP, kScore, kMargin, kToken, kPredecessor };
}
