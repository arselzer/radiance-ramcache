/* p1_chunk.h -- prefill Phase 1's end-aligned chunks (tools/ds41-host-experts/ds41_p1.h,
 * RADIANCE_EXPERIMENTAL_DS41_P1_ALIGN=end; Scheduler::plan_chunk). */
#pragma once
#include "../../tools/ds41-host-experts/ds41_p1.h"   /* the P1 contract (also on rad_core's include path) */

#include <algorithm>
#include <cstdint>

namespace rad {
namespace p1 {

/* The next chunk of a prompt part [n_computed, n_target) under end alignment, measured on absolute
 * positions: ds41p1::first_chunk_end_aligned's length, its end rounded up to the grid (so a part that
 * starts off the grid still ends its first chunk on it), at most C (its end then rounded down to the
 * grid). A part of at most C tokens is one chunk. Called again for what is left, it gives C-token
 * chunks after the first (each ending on the grid) and a last one that ends at n_target: 3,086 tokens
 * at C 2048 and grid 16 are 1,040 + 2,046, 6,817 are 688 + 2,048 + 2,048 + 2,033. 0 when nothing is
 * left. */
inline int64_t end_aligned_chunk(int64_t n_computed, int64_t n_target, int64_t C, int64_t grid) {
    const int64_t P = n_target - n_computed;
    if (P <= 0 || C <= 0) return 0;
    if (P <= C) return P;
    const int64_t g = grid > 1 ? grid : 1;
    const int64_t f = ds41p1::first_chunk_end_aligned(P, C, 1);   /* the length, before the grid */
    int64_t end = (n_computed + f + g - 1) / g * g;               /* its end up to the grid, on positions */
    if (end - n_computed > C) end = (n_computed + C) / g * g;
    if (end <= n_computed) end = std::min(n_target, n_computed + C);   /* a grid wider than C: as is */
    return std::min(end - n_computed, P);
}

}  /* namespace p1 */
}  /* namespace rad */
