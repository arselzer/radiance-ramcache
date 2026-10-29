/* consumer_skip.h -- the exact consumer-suffix skip (RADIANCE_EXPERIMENTAL_DS41_CONSUMER_SKIP): which
 * layers a pure prefill pass runs in full, which one only stores its keys, and which it skips, so
 * that every row it computes, the prompt's last row and the state the next step reads are the
 * bytes a pass that runs every layer computes. Pure host arithmetic; the plugin's half of the
 * contract is rad_declare_consumer_suffix (abi/rad_builder.h), deepseek41's in
 * arch/deepseek41/deepseek41.cpp (step(), "the consumer skip").
 *
 * ---- the dependency structure (the declared contract) -----------------------------------------
 *
 * n layers, boundary B, window W (DeepSeek-V4.1-Flash: 40, 20, 128). Layers below B run on every
 * row of every pass. For a layer L >= B, a row p's output depends on other rows only through
 *   (a) layer L's window keys at positions p - W + 1 .. p, each computed from layer L's input at
 *       that position (layer L-1's output there) by layer L's key ops, and
 *   (b) keys of layers at or below B, which layer B's key ops write for every row (deepseek41:
 *       layer 20's compressed and index keys, the only ones layers 21..39 read).
 * Everything else is row-local, and a row's arithmetic does not depend on which other rows share
 * its pass. The state a later step reads is per row too: each layer's window keys, B's keys.
 *
 * ---- what must be exact ----------------------------------------------------------------------
 *
 * Let T be the first position whose state must be exact: the prompt's end N, or with the prefix
 * cache on the deepest checkpoint below it that the prompt's idle prefill writes (its tail, else
 * the last interval boundary, else a shared head; Scheduler::idle_prefill_checkpoint), whose
 * snapshot holds every layer's keys at T - W .. T - 1. Rows at or after T - 1 run every layer. The decode after the prompt reads the
 * keys at N - W + 1 .. N - 1 and a checkpoint at T those at T - W .. T - 1, both inside T - W ..
 * N - 1.
 *
 * Unrolling (a) backwards from row T - 1 at layer n - 1: row q at layer L needs layer L's keys
 * at q - W + 1 .. q, so layer L-1's outputs there, and so on. The rows layer L must compute are
 * at least those from c(L) = T - 1 - (W - 1) (n - 1 - L) on, and layer L's keys from c(L) - (W - 1)
 * on: the 1 + 127 (39 - L) rows of the survey. That cone is only a lower bound here, for three
 * reasons the derivation below takes care of:
 *   - a pass runs a layer on all of its rows or none (passes keep their height, so every kernel
 *     keeps its band and its bits): the first pass that runs layer L in full also computes rows
 *     before c(L), and those rows read layer L's keys before c(L) - (W - 1);
 *   - such a row must still be exact ("every computed row byte-identical to the base engine"),
 *     and must never read a key this request did not write (an unwritten window slot holds a
 *     stale value, and a NaN there would route a row to an arbitrary expert);
 *   - the engine does not know the pass grid ahead of time, only its rules: a pass holds at most
 *     M rows (--max-num-batched-tokens), and starts at a multiple of A (the chunk quantum: every
 *     chunk but the prompt's last ends on one, and a prefix hit resumes on one).
 *
 * ---- the schedule ----------------------------------------------------------------------------
 *
 * Thresholds f(L) on a pass's last row e - 1, for L = n - 1 down to B:
 *
 *   f(n-1) = T - 1
 *   lo(L)  = A ceil((f(L) - M + 1) / A), at least 0:  a lower bound on where the pass holding row
 *            f(L) starts (it holds at most M rows and starts on a multiple of A)
 *   f(L-1) = lo(L) - (W - 1)                          (and -1, every pass, once it is negative)
 *
 * A pass [s, e) runs layer L in full when e - 1 >= f(L) (every layer below B always); layer B
 * stores its keys on every pass; a layer L > B stores its keys alone when e - 1 >= f(L-1) but
 * not >= f(L). f is non-decreasing in L, so a pass runs layers [0, F) in full and layer F's keys
 * alone (F >= B), F = B + #{L >= B : f(L) <= e - 1}, and skips the rest: RadBatch::consumer_full.
 *
 * Why it is exact, and why no computed row is ever anything but the base engine's row (induction
 * on L; every claim is about the rows a pass computes):
 *   - L < B: every pass runs it in full, as the base engine does.
 *   - L = B: its keys are written on every pass, from layer B-1's outputs (exact). A pass that
 *     runs it in full reads keys of B at positions <= its rows, all written by then.
 *   - L > B, keys: a pass that stores layer L's keys has e - 1 >= f(L-1), so it runs layer L-1 in
 *     full, and by induction all of layer L-1's rows in it are exact: so are the keys.
 *   - L > B, full rows: the first pass that runs layer L in full is the pass holding row f(L)
 *     (passes are consecutive), so it starts at s >= lo(L). Its rows, and every later row, read
 *     layer L's keys at positions >= s - (W - 1) >= lo(L) - (W - 1) = f(L-1). The pass holding row
 *     f(L-1) and every pass after it store layer L's keys (e - 1 >= f(L-1)): every key read is
 *     written, from an exact input, by a pass at or before the reader's own (in a pass the key
 *     ops come before the attention). The keys of B such a row reads are exact (above), and so
 *     is its input (layer L-1 runs in full in every pass layer L does, f(L-1) <= f(L)).
 *   - The outputs: row T - 1 and every later row run every layer (e - 1 >= T - 1 = f(n-1) >=
 *     f(L)). Layer L's keys are written, exactly, at every position from the pass holding f(L-1)
 *     on, and f(L-1) <= f(n-2) = lo(n-1) - (W - 1) <= T - W: the window a decode at N, or a
 *     checkpoint at T, reads is complete in every layer. A checkpoint at P < T is not (layer
 *     n-1's keys start at about T - W - A), so under a cut the scheduler writes none there.
 *
 * Positions before the pass that resumes a prefix hit hold restored state, complete for every
 * layer (a checkpoint is only ever written at or after its prompt's T), so they read as written.
 * The bounds use M and A only through lo(), so a pass grid tighter than the rules allow only
 * makes the cut conservative. At M = A = 16, W = 128, lo(L) = 16 floor(f(L) / 16): layer L runs in
 * full from the pass 8 (39 - L) passes before the last, and in the 8 passes before that stores its
 * keys alone, as PREFILL-ENGINE-SURVEY.md's idea #1 has it; a 3,086-token prompt runs 5,400 of its
 * 7,720 layer-passes in full. tests/consumer_skip_test.cpp checks all of the above against a
 * row-by-row dependency simulation over random grids. */
#pragma once

#include <cstdint>
#include <vector>

namespace rad {

/* The declared contract (RadConsumerSuffix) and the pass grid's rules. */
struct ConsumerShape {
    int32_t n_layer = 0;     /* n */
    int32_t boundary = 0;    /* B */
    int32_t window = 0;      /* W */
    int64_t max_rows = 0;    /* M: the most rows a pass holds */
    int64_t align = 1;       /* A: every pass starts on a multiple of it */

    /* M >= A: a pass that is not the prompt's last ends on a multiple of A too, so it holds at
     * least A rows (the engine refuses a budget under its quantum). It keeps lo(L) <= f(L). */
    bool valid() const {
        return n_layer >= 2 && boundary >= 0 && boundary < n_layer - 1 && window >= 1 &&
               align >= 1 && max_rows >= align;
    }
};

/* f(L) for L in [0, n) into `f` (entries below B are -1, every pass): a pass whose last row is at
 * or past f(L) runs layer L in full. False, and nothing written, for an invalid shape or `target`
 * < 1. Allocates nothing (the step path calls it). */
inline bool consumer_thresholds(const ConsumerShape& c, int64_t target, int64_t* f) {
    if (!c.valid() || target < 1 || !f) return false;
    for (int32_t L = 0; L < c.n_layer; ++L) f[L] = -1;
    int64_t t = target - 1;
    for (int32_t L = c.n_layer - 1; L >= c.boundary; --L) {
        f[L] = t;
        if (t < 0) continue;   /* every pass, from here down */
        const int64_t x = t - c.max_rows + 1;
        const int64_t lo = x <= 0 ? 0 : (x + c.align - 1) / c.align * c.align;
        t = lo - (c.window - 1);
        if (t < 0) t = -1;
    }
    return true;
}
/* The same as a vector, empty when there is no cut. */
inline std::vector<int64_t> consumer_thresholds(const ConsumerShape& c, int64_t target) {
    std::vector<int64_t> f(c.n_layer > 0 ? (size_t)c.n_layer : 0);
    if (!consumer_thresholds(c, target, f.data())) f.clear();
    return f;
}

/* The layers a pass whose last row is `last` runs in full, F in [B, n]; n runs every layer (no
 * cut), otherwise layer F stores its keys alone and the rest are skipped. `f` is
 * consumer_thresholds()'s; null (or empty) runs every layer. */
inline int32_t consumer_full_layers(const ConsumerShape& c, const int64_t* f, int64_t last) {
    if (!f || !c.valid()) return c.n_layer;
    int32_t F = c.boundary;
    while (F < c.n_layer && f[F] <= last) ++F;
    return F;
}
inline int32_t consumer_full_layers(const ConsumerShape& c, const std::vector<int64_t>& f,
                                    int64_t last) {
    return f.size() == (size_t)c.n_layer && c.n_layer > 0 ? consumer_full_layers(c, f.data(), last)
                                                          : c.n_layer;
}

/* Whether a checkpoint at `pos` of a prompt cut for `target` holds every layer's window: at or
 * past the target only (see above). */
inline bool consumer_checkpoint_complete(int64_t pos, int64_t target) { return pos >= target; }

}  /* namespace rad */
