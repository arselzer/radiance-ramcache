/* r4d_ds41_tall.h -- RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL and RADIANCE_EXPERIMENTAL_DS41_LCP on
 * libr4d's side (host code only).
 *
 * DeepSeek-V4.1's prefill runs in passes of at most 16 tokens today. A taller pass changes the
 * arithmetic of the ops whose kernel follows the pass height:
 *
 *   - the routed MXFP4 GEMM: moe_gemm_ggml_mxfp4_gemv serves up to 16 tokens, the tiled row above
 *     sums each output in another order (r4d_ds41_mxfp4.hip);
 *   - the sparse attention and the indexer score: their exact scalar forms up to 48 rows, WMMA
 *     forms above (r4d_ds41_sparse.hip, kWmRows);
 *   - every norm whose width comes from r4d_act_threads (r4d_common.h): 1024 threads up to 32
 *     rows, 512 or 256 above, which regroups the sum of squares. In ds41's graph: rmsnorm, and the
 *     fused mhc_collapse_rmsnorm, mhc_post_collapse_rmsnorm and rmsnorm_rope_roundtrip.
 *
 * With EXACT_TALL at "1" libr4d keeps the decode band's arithmetic at every height up to
 * r4d_ds41_tall_rows(): the routed GEMM's twin row moe_gemm_ggml_mxfp4_gemv_tall (r4d_rows.cpp), the
 * gemv at any number of tokens, takes the bands between 17 and that bound from the tiled row, the
 * sparse ops keep their scalar forms at every height, and r4d_act_threads gives every height the
 * 1024 threads rows up to 32 get, so every norm sums a row's squares alike. The twin's bound on M is
 * the tallest pass at which every libr4d kernel ds41 runs gives a 16-token pass's bits, so the core
 * reads the limit off the bands it resolved (FileStager::tall_init) rather than assuming it: 48, the
 * CPU path's limit (core/place/ep_route_counts.h) and the sparse ops' scalar band.
 *
 * Large-chunk prefill, RADIANCE_EXPERIMENTAL_DS41_LCP=<C> (C a multiple of 16 in 64 .. 2048): the
 * same pins at every height, with the twin's bound raised to kR4dDs41LcpMax. Every other ds41 kernel
 * that picks a form by the pass height keeps a row's bits by construction at any height (the
 * attention's head groups, the indexer's head split, mhc_mix's split, the f32 router's tile forms,
 * moe_scatter's launches, the exact all-reduce's one- and two-shot), and the gemm_nt tiled row is
 * byte-identical to the skinny one by contract; the core admits those pairs by name and refuses any
 * other op that resolves two kernels (FileStager::lcp_init), and ds41_mxfp4_device_check `lcp` holds
 * each to its rows launched alone and 16 at a time up to 2048 rows. A valid C turns EXACT_TALL's pins
 * on here whatever EXACT_TALL says (the core refuses EXACT_TALL other than 1 or unset beside it).
 *
 * Read once, the first time a launcher or the row table asks. Anything but "1" (EXACT_TALL), or a C
 * outside the range (LCP), is off here; the core refuses such values (core/place/ep_route_counts.h). */
#pragma once
#include <cstdlib>

/* The twin's bound under EXACT_TALL alone: the tallest pass that keeps a 16-token pass's bits (32
 * without the norms' pin). */
constexpr int kR4dDs41TallRows = 48;
/* Under LCP: the largest chunk the switch takes, which is the twin's bound, and the smallest. */
constexpr int kR4dDs41LcpMax = 2048;
constexpr int kR4dDs41LcpMin = 64;

/* RADIANCE_EXPERIMENTAL_DS41_LCP: C, or 0 (unset, or not a multiple of 16 in [64, 2048]). */
inline int r4d_ds41_lcp() {
    static const int c = [] {
        const char* e = std::getenv("RADIANCE_EXPERIMENTAL_DS41_LCP");
        if (!e || !*e) return 0;
        char* end = nullptr;
        const long v = std::strtol(e, &end, 10);
        return end != e && !*end && v >= kR4dDs41LcpMin && v <= kR4dDs41LcpMax && v % 16 == 0 ? (int)v : 0;
    }();
    return c;
}

inline bool r4d_ds41_exact_tall() {
    static const bool on = [] {
        const char* e = std::getenv("RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL");
        return (e && e[0] == '1' && e[1] == '\0') || r4d_ds41_lcp() > 0;
    }();
    return on;
}

/* The twin's bound on M: kR4dDs41LcpMax under LCP, kR4dDs41TallRows under EXACT_TALL alone. */
inline int r4d_ds41_tall_rows() { return r4d_ds41_lcp() > 0 ? kR4dDs41LcpMax : kR4dDs41TallRows; }
