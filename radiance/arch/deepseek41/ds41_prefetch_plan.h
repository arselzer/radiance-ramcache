/* ds41_prefetch_plan.h -- which of a layer's weights deepseek41 reads into the card's on-die cache
 * ahead of the layer, in what order, and how many of their bytes (RADIANCE_EXPERIMENTAL_DS41_PREFETCH_*;
 * the step side is Layer::pf in deepseek41.cpp). Pure host logic with no ABI, so
 * tests/ds41_prefetch_plan_test.cpp checks it alone.
 *
 * At decode each layer's card work ends with the shared expert, and the card then idles until the
 * CPU's rows for the gather arrive (~0.3 ms a layer at 15 tok/s), while the next layer's GEMVs
 * stream their weights from VRAM at ~440-620 GB/s with no on-die cache hits. libr4d's `prefetch`
 * reads a weight's leading bytes on the second lane in that wait, so those GEMVs find them in the
 * 64 MB Infinity Cache. The reads only read: no value changes.
 *
 *   RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ATTN=1    on: by itself, the next layer's q_a, kv, q_b and
 *                                                 wo_a whole (~56 MB a card in bf16; +2.8% decode,
 *                                                 commit 16be66c5), and nothing ahead of an Engram
 *                                                 layer, whose GEMM would evict the reads
 *   ..._PREFETCH_MIB=N                            a budget of N MiB an issue point, filled from the
 *                                                 use-order list below; the read that crosses it is
 *                                                 cut to what is left (kMinCut or more), the rest
 *                                                 are dropped
 *   ..._PREFETCH_SET=k1,k2,...                    the kinds read, in this order (names below; ',',
 *                                                 '+' or ' ' between them), in place of either list
 *   ..._PREFETCH_ENGRAM=1                         an Engram layer is read too: its engram_kv by the
 *                                                 layer before (if `ekv` is listed), every other
 *                                                 read after its Engram GEMM, by the layer itself
 *
 * The use-order list is the order a decode step issues the ops that read them (deepseek41.cpp
 * step(): the compressor's GEMM goes first, then the query and key chains with the indexer
 * between them, the sparse attention, wo_a's groups, wo_b; then the MoE block's router, the shared
 * expert's gate|up behind the routed one, its down behind the routed down).
 *
 * Sizes a card at TP4, bf16: q_a 12.5 MiB, kv 5, q_b 20, wo_a 2 x 8, wo_b 20 (73.5 in all); the
 * f32 compressor 20 (ratio 2: layers 2, 8, 14) or 5 (bf16, ratio 1: layer 20); the indexer's q_b
 * 10 (index sources 2, 8, 14, 20, 24, 28, 32, 36); the f32 router 7.5; the shared expert's gate|up
 * 15 / 10 and down 7.5 / 5 (rank 0 / ranks 1-3); engram_kv 75 (split over the ranks, 300 whole).
 * With RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8=1 (+12.7%) every one of these but the compressor and the
 * router is read from its E4M3 codes, half the bytes: the attention's five 36.75, and 56 MiB then
 * covers the whole list through the shared expert (rank 0 55.5 MiB a layer) where the four reads
 * alone are 26.75. With RADIANCE_EXPERIMENTAL_DS41_ROUTER_BF16=1 the ratio-2 compressor and the
 * router are held as bf16 (gemm_nt_f32bf16), half theirs: 10 and 3.75. */
#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace deepseek41 {
namespace pf {

/* A layer's weights by the op that reads them, in use order. */
enum Kind : int {
    kEkv,   /* ekv:  the Engram GEMM's engram_kv (an Engram layer's first read) */
    kComp,  /* comp: the compressor's kv|gate (f32; bf16 under ..._ROUTER_BF16) or kv (bf16)
             *       projection, at a kv source */
    kQa,    /* qa:   attn_q_a */
    kKv,    /* kv:   attn_kv */
    kIw,    /* iw:   the indexer's weights_proj (320 KiB), at an index source */
    kIq,    /* iq:   the indexer's q_b, at an index source */
    kQb,    /* qb:   attn_q_b, this rank's heads */
    kIk,    /* ik:   the indexer's key projection (128 KiB), at a kv source */
    kOa,    /* oa:   attn_output_a, one read a group this rank holds */
    kMa,    /* ma:   the attention connection's mix (hc_attn_fn, f32, 1.9 MiB) */
    kOb,    /* ob:   attn_output_b, this rank's columns */
    kRt,    /* rt:   the router (f32; bf16 under ..._ROUTER_BF16) */
    kSgu,   /* sgu:  the shared expert's gate|up, this rank's slice */
    kMf,    /* mf:   the ffn connection's mix (hc_ffn_fn, f32, 1.9 MiB) */
    kSdn,   /* sdn:  the shared expert's down, this rank's slice */
    kKinds
};
inline const char* name(int k) {
    static const char* const n[kKinds] = { "ekv", "comp", "qa", "kv", "iw", "iq", "qb", "ik",
                                           "oa", "ma", "ob", "rt", "sgu", "mf", "sdn" };
    return k >= 0 && k < kKinds ? n[k] : "?";
}
inline int kind_of(const char* s, size_t len) {
    for (int k = 0; k < kKinds; ++k)
        if (std::strlen(name(k)) == len && !std::strncmp(name(k), s, len)) return k;
    return -1;
}

/* The list ..._PREFETCH_ATTN=1 reads by itself, whole: the four attention weights 16be66c5 read. */
inline std::vector<int> legacy_order() { return { kQa, kKv, kQb, kOa }; }
/* The list a budget reads from: every weight of a megabyte or more on the next layer's way to its
 * gather, in use order. Left out (each can be named in ..._SET): the indexer's weights_proj and key
 * projection, whose GEMVs are launch-bound (~6.5 us a call against <1 us of streaming), and the
 * two mixes, which overlap wo_a/wo_b and the routed gate_up rather than wait for them. */
inline std::vector<int> use_order() {
    return { kEkv, kComp, kQa, kKv, kIq, kQb, kOa, kOb, kRt, kSgu, kSdn };
}

/* A cut read below this is dropped: a GEMV streams a megabyte in ~1.7 us, about what a launch on
 * the second lane costs to issue. */
constexpr int64_t kMinCut = 1ll << 20;

struct Cfg {
    bool on = false;          /* ..._PREFETCH_ATTN=1 */
    int64_t budget = 0;       /* ..._PREFETCH_MIB in bytes; 0 reads every listed weight whole */
    bool engram = false;      /* ..._PREFETCH_ENGRAM=1 */
    bool listed = false;      /* the order is ..._PREFETCH_SET's */
    std::vector<int> order;   /* the kinds read, in this order */
    std::string note;         /* values refused, for the startup line; empty: none */
};

/* The configuration from the four variables' values (null: unset). Anything malformed is named in
 * `note` and read as unset, so a typo leaves the measured default rather than reading less. */
inline Cfg parse(const char* on, const char* mib, const char* set, const char* engram) {
    Cfg c;
    c.on = on && !std::strcmp(on, "1");
    c.engram = engram && !std::strcmp(engram, "1");
    if (engram && !c.engram) c.note += std::string(" PREFETCH_ENGRAM=") + engram + " is not 1;";
    if (mib && *mib) {
        char* end = nullptr;
        const double v = std::strtod(mib, &end);
        if (end && !*end && v > 0 && v <= 4096)
            c.budget = (int64_t)(v * 1048576.0) & ~(int64_t)4095;
        if (c.budget <= 0) {
            c.budget = 0;
            c.note += std::string(" PREFETCH_MIB=") + mib + " is no size in (0, 4096] MiB;";
        }
    }
    if (set && *set) {
        for (const char* p = set; *p;) {
            const size_t len = std::strcspn(p, ",+ ");
            if (len) {
                const int k = kind_of(p, len);
                bool dup = false;
                for (int d : c.order) dup = dup || d == k;
                if (k < 0) c.note += " PREFETCH_SET names no read '" + std::string(p, len) + "';";
                else if (!dup) c.order.push_back(k);
            }
            p += len + (p[len] ? 1 : 0);
        }
        c.listed = !c.order.empty();
    }
    if (!c.listed) c.order = c.budget > 0 ? use_order() : legacy_order();
    return c;
}

/* One weight a layer would read: its kind and the bytes its stored form holds. */
struct Cand {
    int kind = 0;
    int64_t bytes = 0;
};
/* One read: candidate `cand`'s leading `bytes`, `cut` when that is less than the weight. */
struct Read {
    size_t cand = 0;
    int64_t bytes = 0;
    bool cut = false;
};
/* The reads of `c` (in read order) that `budget` bytes cover: whole while they fit, then the one
 * that crosses the budget cut to what is left (whole 4 KiB, if that is kMinCut or more), and
 * nothing after it, so the reads are always the layer's first `budget` bytes of weight traffic.
 * Budget 0: every candidate whole. Candidates of no bytes are skipped. */
inline std::vector<Read> fit(const std::vector<Cand>& c, int64_t budget) {
    std::vector<Read> out;
    int64_t left = budget;
    for (size_t i = 0; i < c.size(); ++i) {
        if (c[i].bytes <= 0) continue;
        if (budget <= 0 || c[i].bytes <= left) {
            out.push_back({ i, c[i].bytes, false });
            left -= c[i].bytes;
            continue;
        }
        const int64_t b = left & ~(int64_t)4095;
        if (b >= kMinCut) out.push_back({ i, b, true });
        break;
    }
    return out;
}

}  // namespace pf
}  // namespace deepseek41
