/* r4d_ds41_tall.h -- RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL on libr4d's side (host code only).
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
 * With the variable at "1" libr4d keeps the decode band's arithmetic at every height up to
 * kR4dDs41TallRows: the routed GEMM's twin row moe_gemm_ggml_mxfp4_gemv_tall (r4d_rows.cpp), the
 * gemv at any number of tokens, takes the bands between 17 and kR4dDs41TallRows tokens from the
 * tiled row, and the sparse ops keep their scalar forms at every height. The twin's bound on M is
 * the tallest pass at which every libr4d kernel ds41 runs gives a 16-token pass's bits, so the core
 * reads the limit off the bands it resolved (FileStager::tall_init) rather than assuming it: up to
 * 32 tokens the norms keep their 1024 threads without help.
 *
 * Read once, the first time a launcher or the row table asks. Anything but "1" is off here; the core
 * refuses a value other than 0 or 1 (core/place/ep_route_counts.h). */
#pragma once
#include <cstdlib>

/* The twin's bound: the tallest pass that keeps a 16-token pass's bits under the switch. */
constexpr int kR4dDs41TallRows = 32;

inline bool r4d_ds41_exact_tall() {
    static const bool on = [] {
        const char* e = std::getenv("RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL");
        return e && e[0] == '1' && e[1] == '\0';
    }();
    return on;
}
