/* ds41_p1.h -- prefill Phase 1: every routed expert of a large prefill chunk computed on the cards.
 *
 * The contract between the two Phase 1 work packages (PHASE1-PLAN.md in the review directory):
 *   K, the card kernel side: libr4d's twin row under the switch, its marks and ranges, and the
 *      device check that proves it equal to the 16-token gemv;
 *   S, the stager and engine side: the core's card expert stager, its ring, copy lists, drive
 *      staging, waves, and the window's arms.
 * Header-only, plain C++17, no HIP and no core types. The core and the tools include it. libr4d
 * restates the few constants it needs (as core/place/vram_packed.h restates libr4d's), and the
 * device check, which includes this header and links libr4d, checks the two agree.
 *
 * ---- what Phase 1 does, in one paragraph -------------------------------------------------------
 *
 * Under RADIANCE_EXPERIMENTAL_DS41_P1=1, beside RADIANCE_EXPERIMENTAL_DS41_LCP=<C>, a routed layer
 * of a pure prefill pass that runs on at least kMinRowsEnv's rows (320 by default, never 16 or
 * fewer) no longer gives any routed expert to the CPU. The layer's VRAM
 * residents run in the plugin's own expert ops. Every touched non-resident expert is published to
 * those ops with kMarkSkip, so they leave its rows alone. The stager then copies each such expert
 * into a VRAM ring slot: a RAM-cache unit from its pinned slab slot, a drive miss after it is read
 * and packed in pinned staging. It computes the experts in waves by launching the same three ops
 * again (gate_up, the clamped SwiGLU, down) with its own expert table, in which the wave's experts
 * name ring slots and every other entry carries kMarkSkip. The plugin's moe_gather then reads every
 * row as it would after a 16-token pass. The CPU's expert path is unchanged for decode and for
 * every layer pass below the threshold: a P1 pass streams nearly every routed expert of a layer to
 * the cards whatever its rows, which a short pass does not repay (the crossover is near 300 rows).
 */
#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace ds41p1 {

/* ---- the switch ---------------------------------------------------------------------------------
 *
 * "1": on. Unset, empty or "0": off. Anything else is invalid: the core refuses to start, and libr4d
 * treats it as off. libr4d reads the switch once, when its rows are first counted, as it reads
 * RADIANCE_EXPERIMENTAL_DS41_LCP (r4d_ds41_tall.h). The switch only acts beside LCP: the core refuses
 * P1 without a valid LCP chunk, and libr4d adds the P1 twin only when it also adds the LCP twin. */
constexpr const char* kEnv = "RADIANCE_EXPERIMENTAL_DS41_P1";

inline int env_state(const char* v) {
    if (!v || !*v || (v[0] == '0' && !v[1])) return 0;
    if (v[0] == '1' && !v[1]) return 1;
    return -1;
}
/* 1 on, 0 off, -1 invalid; read once. */
inline int state() {
    static const int s = env_state(std::getenv(kEnv));
    return s;
}

/* ---- the twin row under the switch (package K) -------------------------------------------------
 *
 * With P1 on, libr4d registers its twin of moe_gemm_ggml_mxfp4_gemv under kTwinRow in place of
 * "moe_gemm_ggml_mxfp4_gemv_tall". It has the same constraints, with M bounded by kTwinMaxRows, and
 * it sits one priority below the gemv, so it takes every band from 17 tokens up. Its launcher, for
 * a weight TABLE operand (stride[0] == 0, the only form ds41's trunk uses), promises:
 *
 *   T1 bits    Each output row it writes is the decode band's (moe_gemm_ggml_mxfp4_gemv, or
 *              moe_gemm_ggml_mxfp4_gemv_vp for a packed entry) for the same activation row, byte for
 *              byte, at every M from 1 to kTwinMaxRows. That covers any routing, any skew, dropped
 *              slots and NaN/Inf/subnormal activations.
 *   T2 forms   An entry with kMarkPacked is a weight in the packed-scale form (ds41_host_experts.h
 *              pk_layout, gemv_vp's form). Any other unmarked entry is block_mxfp4.
 *   T3 skip    An entry with kMarkSkip is not computed: no byte of its rows in the output operand is
 *              written, and no byte of the weight is read (the pointer under the mark need not be
 *              readable). Every other expert with rows has all its rows written.
 *   T4 ranges  Expert e < n_expert owns the sorted slots [off[e], off[e+1]) (the live-tail ranges;
 *              off[n_expert] is the end of the rank's local slots). The slots [off[n_expert], T),
 *              which are routed to other ranks, are written as zeros by every call, whatever the
 *              marks. No weight pass is spent on them.
 *   T5 work    Work is spread over the grid by row passes, not by experts (lcpfast's balance), so
 *              one expert with 2,000 rows does not serialise a launch.
 *   T6 stream  The call only enqueues on the given stream. It reads the table and the offsets on the
 *              card, at run time, so the stager may change the table between launches by ordered
 *              copies on that stream.
 * A stacked (non-table) weight is served as the LCP twin serves it today.
 *
 * kTwinMaxRows is the kernel's bound. The chunk stays at most RADIANCE_EXPERIMENTAL_DS41_LCP's
 * admitted maximum, 4096. Chunks above 2048 additionally need the dense ops' extended height sweep
 * (ds41_mxfp4_device_check `lcp C`) and an explicit checkpoint interval of at least C. */
constexpr const char* kTwinRow     = "moe_gemm_ggml_mxfp4_gemv_tall_p1";
constexpr const char* kLcpTwinRow  = "moe_gemm_ggml_mxfp4_gemv_tall";
constexpr const char* kGemvRow     = "moe_gemm_ggml_mxfp4_gemv";
constexpr const char* kGemvVpRow   = "moe_gemm_ggml_mxfp4_gemv_vp";
constexpr int         kTwinMaxRows = 4096;
/* The smallest pass the card path can be armed for: anything shorter runs on the gemv row, which
 * honours neither mark (T2 holds there only for gemv_vp, T3 nowhere), so the stager never marks a
 * table for it. The threshold (kMinRowsEnv) is at least this. */
constexpr int         kMinArmRows  = 17;
/* The same bound a routed layer at a time, with the threshold `min_rows` (kMinRowsEnv): under the
 * consumer skip's per-row cut (consumer_row_starts) a layer of an armed pass runs on the pass's rows
 * [row0(L), n) alone, so a deep layer of a prompt's last chunk can run on fewer rows than the
 * threshold, or on 16 or fewer, on the gemv row again. Such a layer is not the card path's: it takes
 * the CPU path for its non-resident experts, as a pass of its height does, with no mark and no
 * wave. */
constexpr bool card_layer(int64_t layer_rows, int64_t min_rows = kMinArmRows) {
    return layer_rows >= kMinArmRows && layer_rows >= min_rows;
}

/* ---- table entry marks --------------------------------------------------------------------------
 *
 * Weights, ring slots and slab slots are at least 64-byte aligned, so the low six bits of an entry
 * are free. Bit 0 is gemv_vp's packed mark, which the core sets for packed VRAM residents
 * (Ctx::table_refresh) and the stager sets for packed ring slots. Bit 1 is P1's skip. No entry ever
 * carries both, and bits 2..5 stay zero. */
constexpr uintptr_t kMarkPacked = 1;
constexpr uintptr_t kMarkSkip   = 2;
constexpr uintptr_t kMarkBits   = 63;

inline const void* mark(const void* p, uintptr_t m) { return (const void*)((uintptr_t)p | m); }
inline const void* unmark(const void* p) { return (const void*)((uintptr_t)p & ~kMarkBits); }
inline bool is_packed(const void* p) { return ((uintptr_t)p & kMarkPacked) != 0; }
inline bool is_skip(const void* p) { return ((uintptr_t)p & kMarkSkip) != 0; }

/* ---- one expert unit and its ring slot ----------------------------------------------------------
 *
 * ds41's routed expert (deepseek41.cpp dims): gate_up [4608, 5120] and down [5120, 2304], MXFP4. A
 * slot holds the two weights at fixed offsets: gate_up at 0, down at kDnOff. A block-form unit fills
 * the slot exactly. A packed unit's two weights (ds41hx::packed_bytes at the RAM cache's raw
 * budgets: 11,999,296 + 6,001,984 bytes by default) are copied to the same two offsets and leave the
 * slot's tail unused. So one table entry pair, slot + 0 and slot + kDnOff, names either form. The
 * kMarkPacked mark tells the two apart. */
constexpr int     kGuN = 4608, kGuK = 5120, kDnN = 5120, kDnK = 2304;
constexpr int64_t kBlockBytes(int64_t n, int64_t k) { return n * (k / 32) * 17; }
constexpr int64_t kGuBlockBytes = kBlockBytes(kGuN, kGuK);              /* 12,533,760 */
constexpr int64_t kDnBlockBytes = kBlockBytes(kDnN, kDnK);              /*  6,266,880 */
constexpr int64_t kDnOff        = kGuBlockBytes;                        /* 256-aligned */
constexpr int64_t kSlotBytes    = kGuBlockBytes + kDnBlockBytes;        /* 18,800,640 = 4590 x 4096 */
static_assert(kDnOff % 256 == 0 && kSlotBytes % 4096 == 0, "slot layout alignment");

/* ---- the stager's tunables (package S; the names the window's arm parts set) --------------------
 *
 * The ring is the file stager's two layer buffers seen as one region of whole slots, sized by
 * RADIANCE_EXPERIMENTAL_DS41_FILE_BUFFER_UNITS=N (2N slots; the arm part BN). P1 needs N >= 2, and
 * the window measures N = 6 (12 slots, 215 MiB a card). With kLendEnv the slots past the file
 * stager's own two (where its aliases point, B1's share) hold decode residents between the card
 * path's passes (card_expert_stager.h, the lent ring), so the ring costs decode nothing. */
constexpr const char* kWaveEnv     = "RADIANCE_EXPERIMENTAL_DS41_P1_WAVE";      /* units a wave (2) */
constexpr const char* kInflightEnv = "RADIANCE_EXPERIMENTAL_DS41_P1_INFLIGHT";  /* drive units read at once a rank (16) */
constexpr const char* kReadyEnv    = "RADIANCE_EXPERIMENTAL_DS41_P1_READY";     /* pinned packed slots a rank (6) */
constexpr const char* kAheadEnv    = "RADIANCE_EXPERIMENTAL_DS41_P1_AHEAD";     /* next layer's units read ahead a rank (8; 0 off) */
constexpr const char* kVerifyEnv   = "RADIANCE_EXPERIMENTAL_DS41_P1_VERIFY";    /* expert-rows a layer recomputed on the host (0) */
constexpr const char* kAlignEnv    = "RADIANCE_EXPERIMENTAL_DS41_P1_ALIGN";     /* "end" (default) or "start" */
/* The threshold: a routed layer's pass of at least this many rows is the card path's, any shorter
 * one the CPU path's exactly as without the switch (kMinArmRows to kTwinMaxRows; 320). The CPU path
 * then takes misses up to min(C, threshold - 1) rows, and never fewer than 16 (cpu_rows). */
constexpr const char* kMinRowsEnv  = "RADIANCE_EXPERIMENTAL_DS41_P1_MIN_ROWS";
/* The lent ring: up to this many ring slots hold the hottest file-tier units as decode residents
 * outside the card path's passes (0 off, the default; at most the ring's slots less the file
 * stager's two), given back kGraceEnv ms after the card path's last pass (1000; 0-60000). */
constexpr const char* kLendEnv     = "RADIANCE_EXPERIMENTAL_DS41_P1_LEND";
constexpr const char* kGraceEnv    = "RADIANCE_EXPERIMENTAL_DS41_P1_GRACE_MS";
/* The lent arena: up to this many slots of the activation arena's top hold the next hottest file-tier
 * units as decode residents the same way (0 off, the default; 1-64). The arena is sized for the chunk
 * (935 MiB a card at 2048 rows, 468 at 1024), and a step of at most its smallest level's rows (256)
 * leaves everything above that level's buffers and its guard untouched; a step at the full plan (any
 * step above those rows, a card-path pass among them) takes the slots back first, as a card-path pass
 * takes the ring's. The deepseek41 plugin declares at a smaller step (RadArchProbe::shape_probe_ok)
 * only beside this switch and kEnv (lend_arena_asked), so without them nothing changes. */
constexpr const char* kLendArenaEnv = "RADIANCE_EXPERIMENTAL_DS41_P1_LEND_ARENA";
constexpr int kDefWave = 2, kDefInflight = 16, kDefReady = 6, kDefAhead = 8, kDefMinRows = 320;
/* kMaxLend bounds each of kLendEnv and kLendArenaEnv; the lender holds both (2 x kMaxLend). */
constexpr int kDefGraceMs = 1000, kMaxLend = 64;
constexpr int kMinRingSlots = 4;

/* The tallest layer pass the CPU path takes under the threshold, at chunk C: every pass or layer
 * below `min_rows`, and the 16-token passes (kMinArmRows - 1) whatever the threshold. */
constexpr int64_t cpu_rows(int64_t C, int64_t min_rows) {
    const int64_t below = (min_rows > kMinArmRows ? min_rows : kMinArmRows) - 1;
    return C < below ? C : below;
}

/* End-aligned chunks (kAlignEnv "end"): of a prompt part of P tokens still to compute, in chunks of
 * at most C on a grid of g rows (16), the first chunk takes P - (n - 1) * C rounded up to the grid,
 * n = ceil(P / C), so the last chunk is as full as the grid allows and carries the consumer skip's
 * suffix. A part of at most C tokens is one chunk. Returns the first chunk's length. */
inline int64_t first_chunk_end_aligned(int64_t P, int64_t C, int64_t g) {
    if (P <= C) return P;
    const int64_t n = (P + C - 1) / C;
    int64_t f = P - (n - 1) * C;
    if (g > 1) f = (f + g - 1) / g * g;
    return f < P ? f : P;
}

/* ---- the server's log lines the window keys on (package S writes them, ab_window.sh greps) ------ */
constexpr const char* kLogOn    = "ds41 p1: card experts on";   /* rank 0, at startup, with the ring and tunables */
constexpr const char* kLogStats = "ds41 p1 stats, rank";        /* each rank, at exit */
constexpr const char* kLogChunk = "ds41 p1 chunk:";             /* rank 0, after each card-path chunk */
constexpr const char* kLogLend  = "ds41 p1: lent ring";         /* rank 0, at startup, with kLendEnv */
constexpr const char* kLogLendArena = "ds41 p1: lent arena";    /* rank 0, at startup, with kLendArenaEnv */

/* Whether the lent arena is asked for (kEnv `p1` on, kLendArenaEnv `lend` a count from 1 to kMaxLend):
 * the deepseek41 plugin then declares at a smaller step too. A value the core refuses counts as not
 * asked here; the core's tunables say why. */
inline bool lend_arena_asked(const char* p1, const char* lend) {
    if (env_state(p1) != 1 || !lend || !*lend) return false;
    long n = 0;
    for (const char* p = lend; *p; ++p) {
        if (*p < '0' || *p > '9' || n > kMaxLend) return false;
        n = n * 10 + (*p - '0');
    }
    return n >= 1 && n <= kMaxLend;
}
/* The same from the environment, kEnv read once (state()). */
inline bool lend_arena_asked() { return state() == 1 && lend_arena_asked("1", std::getenv(kLendArenaEnv)); }

/* ---- the device check (package K) --------------------------------------------------------------
 *
 *   HIP_VISIBLE_DEVICES=0 ds41_mxfp4_device_check LIBR4D.so p1 [MMAX] [ROUTES]
 *
 * It sets RADIANCE_EXPERIMENTAL_DS41_LCP=2048 and kEnv=1 before loading LIBR4D, requires kTwinRow
 * in its table, and checks T1-T5 at M in {17, 33, 48, 64, 100, 256, 1024, 2048, MMAX (4096)}, with
 * block, packed and skipped entries mixed. It also checks the wave composition the stager relies on:
 * one launch with a subset skipped, then waves over a partition of that subset (each wave's table
 * skipping the rest), with the clamped SwiGLU relaunched over all rows between gate_up and down,
 * must leave the gate_up, SwiGLU and down outputs byte-equal to one launch over every expert. ROUTES
 * is a p0_tables.py routes file (real routing). The last line is "p1: PASS ..." or "p1: FAIL ...",
 * and it exits 0 only on PASS. It also prints "p1 time:" lines (us an expert-row at M 256 / 1024 /
 * 2048, real routing, each form) for the window's log. */
constexpr const char* kCheckMode = "p1";

}  /* namespace ds41p1 */
