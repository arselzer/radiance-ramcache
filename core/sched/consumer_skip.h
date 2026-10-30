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
 *     n-1's keys start at about T - W - A) unless the schedule below makes it whole.
 *
 * Positions before the pass that resumes a prefix hit hold restored state, complete for every
 * layer (a checkpoint is only ever written at or after its prompt's T), so they read as written.
 * The bounds use M and A only through lo(), so a pass grid tighter than the rules allow only
 * makes the cut conservative. At M = A = 16, W = 128, lo(L) = 16 floor(f(L) / 16): layer L runs in
 * full from the pass 8 (39 - L) passes before the last, and in the 8 passes before that stores its
 * keys alone, as PREFILL-ENGINE-SURVEY.md's idea #1 has it; a 3,086-token prompt runs 5,400 of its
 * 7,720 layer-passes in full. tests/consumer_skip_test.cpp checks all of the above against a
 * row-by-row dependency simulation over random grids.
 *
 * ---- complete checkpoints below the target ----------------------------------------------------
 *
 * The prefix cache resumes a later prompt at the deepest snapshot under what it shares, and a
 * prompt sharing only an earlier prefix needs one below T. A snapshot at P (taken after the pass
 * whose last row is P - 1; the scheduler only ever writes one at a pass end) holds every layer's
 * window keys at max(0, P - W) .. P - 1, and the state of the layers that keep one, all at or
 * below B (the contract leaves a layer past B no other way to see an earlier row). It is the base
 * engine's iff those keys are stored, exactly. Storing layer L's key at a row takes layer L's
 * input there, which is layer L-1's output: layer L-1 run in full on that row, which reads layer
 * L-1's own keys W - 1 rows further back, and so on down. So a whole snapshot at P costs its own
 * dependency cone, like a target's, except that its top layer needs only its keys: the key ops
 * alone, on the rows behind P, do not make it whole.
 *
 * C, the checkpoints to make whole: positions P, p0 < P < T, each a pass end. A checkpoint's
 * thresholds g (layer L in full on a pass ending at or before P whose last row is >= g(L)):
 *
 *   g(n-1) = never                                    (only its keys are read)
 *   g(n-2) = P - W                                    (-1 when negative)
 *   g(L-1) = lo(L) - (W - 1), lo(L) = A ceil((g(L) - M + 1) / A) at least 0, for L = n-2 .. B+1,
 *            -1 once negative; -1 below B
 *
 * A pass [s, e) runs layers [0, F) in full and layer F's keys alone, with
 *
 *   F = max(F_T(e - 1), G_P*(e - 1)),  F_x(last) = B + #{L >= B : x(L) <= last},
 *
 * P* the first checkpoint of C at or after e (none: F = F_T). Only P* counts: g is non-decreasing
 * in P (g(n-2) is, lo() is monotone), so for one last row G_P is non-increasing in P, and a
 * checkpoint before e has its snapshot already. F is still one number a pass: the union of two
 * prefixes [0, F_x) is the longer, and the shorter one's key layer lies inside it.
 *
 * Why it is exact, and why a snapshot in C is whole (the target's induction on L, per target):
 *   - A pass that runs layer L > B in full does so for a target t, T or its P*, whose threshold
 *     thr_t(L) <= e - 1 (F is one of the two, and L lies below it), and the pass lies in t's reach
 *     (T: every pass; P*: e <= P*). Passes are consecutive, so it starts at s >= lo(thr_t(L)), and
 *     its rows read layer L's keys at positions >= s - (W - 1) >= thr_t(L - 1). Each such position
 *     q < e lies in a pass at or before it (or the hit's restored state), whose last row is >= q
 *     >= thr_t(L - 1), so whose F under t alone is >= L; its own F is no less: for T directly, and
 *     for P* because its own first checkpoint ahead is no later (it ends no later) and G is
 *     non-increasing in P. So it stores layer L's keys from layer L-1's full rows, exact by
 *     induction. The target's own passes are exactly as above.
 *   - Whole: each position q in max(0, P - W) .. P - 1 lies in a pass ending at or before P whose
 *     last row is >= q >= P - W = g(n-2) >= g(L - 1) for every L <= n - 1, and whose first
 *     checkpoint ahead is P or earlier, so its F >= G_P >= n - 1: it stores every layer's key at q,
 *     layer n-1's from layer n-2's full rows. Layers at or below B run on every row, their state
 *     with them. The snapshot at P holds the base engine's bytes. The pass ending at P has
 *     last row P - 1 >= g(n-2), so F >= n - 1 there: layer n-1's keys and every layer below it in
 *     full; the batch builder checks exactly that before it lets a step that writes a snapshot be
 *     cut (consumer_ckpt_covered).
 *
 * The cost: one checkpoint alone adds what its cone covers beyond the target's. At M = A = 16,
 * W = 128, layer L in 21..38 runs in full on the 128 (39 - L) rows before P and stores its keys
 * alone on the 128 before those, layer 39 stores its keys on the last 128, and layer 20 runs in full
 * on the 2,432 rows before P where the target alone stores its keys: 1,520 16-row layer-passes in
 * full (20% of a 3,086-token prompt's 7,720) for a P more than ~2,450 rows below T and past row
 * ~2,450, about 17 a pass of distance nearer T, none within a pass of it.
 * Checkpoints every I rows overlap their cones for every layer with 128 (n-1-L) >= I, which then
 * runs in full throughout: at I = 16 or 256 every layer but the last one or two (97.5% or 96.3%
 * of the layer-passes in full, against 52-87% for the target alone), at I = 2048 layers 21..23
 * throughout and layer L on (39-L)/16 of the rows (79% on long prompts against 52-65%). Which
 * checkpoints are in C is the scheduler's policy (Scheduler::consumer_target,
 * RADIANCE_EXPERIMENTAL_DS41_CONSUMER_SKIP_CKPT); this header proves any C. */
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

/* g(L) for a checkpoint at `pos` (a pass end below the target) into `g`: a pass ending at or
 * before `pos` whose last row is at or past g(L) runs layer L in full; g(n-1) is INT64_MAX (layer
 * n-1 stores its keys alone at most) and entries below B are -1. False, and nothing written, for an
 * invalid shape or `pos` < 1. Allocates nothing (the step path calls it). */
inline bool consumer_ckpt_thresholds(const ConsumerShape& c, int64_t pos, int64_t* g) {
    if (!c.valid() || pos < 1 || !g) return false;
    for (int32_t L = 0; L < c.n_layer; ++L) g[L] = -1;
    g[c.n_layer - 1] = INT64_MAX;
    int64_t t = pos - c.window;
    if (t < 0) t = -1;
    for (int32_t L = c.n_layer - 2; L >= c.boundary; --L) {
        g[L] = t;
        if (t < 0) continue;   /* every pass, from here down */
        const int64_t x = t - c.max_rows + 1;
        const int64_t lo = x <= 0 ? 0 : (x + c.align - 1) / c.align * c.align;
        t = lo - (c.window - 1);
        if (t < 0) t = -1;
    }
    return true;
}
inline std::vector<int64_t> consumer_ckpt_thresholds(const ConsumerShape& c, int64_t pos) {
    std::vector<int64_t> g(c.n_layer > 0 ? (size_t)c.n_layer : 0);
    if (!consumer_ckpt_thresholds(c, pos, g.data())) g.clear();
    return g;
}

/* F of a pass whose last row is `last` under the target's thresholds `f` and, when `g` is not
 * null, those of the first checkpoint to make whole at or after the pass's end (see above): the
 * larger of the two. */
inline int32_t consumer_pass_layers(const ConsumerShape& c, const int64_t* f, const int64_t* g,
                                    int64_t last) {
    const int32_t F = consumer_full_layers(c, f, last);
    if (!g || !c.valid()) return F;
    const int32_t G = consumer_full_layers(c, g, last);
    return G > F ? G : F;
}
inline int32_t consumer_pass_layers(const ConsumerShape& c, const std::vector<int64_t>& f,
                                    const std::vector<int64_t>& g, int64_t last) {
    const int32_t F = consumer_full_layers(c, f, last);
    if (g.size() != (size_t)c.n_layer || c.n_layer <= 0) return F;
    const int32_t G = consumer_full_layers(c, g.data(), last);
    return G > F ? G : F;
}

/* Whether a checkpoint at `pos` of a prompt cut for `target` holds every layer's window without
 * help: at or past the target (see above). One below it is whole only when it is in C. */
inline bool consumer_checkpoint_complete(int64_t pos, int64_t target) { return pos >= target; }

/* Whether a cut step may write a snapshot at `pos`, the end of its rows (`last` + 1), with F
 * `full`: at or past the target (that step runs every layer anyway), or `pos` is the checkpoint
 * the step was planned to make whole (`next`, its first checkpoint of C ahead) and the step stores
 * every layer's keys (F >= n - 1). */
inline bool consumer_ckpt_covered(const ConsumerShape& c, int64_t pos, int64_t last, int64_t target,
                                  int64_t next, int32_t full) {
    if (pos >= target) return true;
    return next > 0 && pos == next && pos == last + 1 && full >= c.n_layer - 1;
}

}  /* namespace rad */
