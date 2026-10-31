/* Pure byte inverse of the published ds41hx packed CPU-cache matrix.
 * No allocation, floating point, requantization, file/device access or publication.
 * This is NOT a runtime bank/RingLender hook. The caller must hold its source
 * lease through this call and validate the enclosing Program/Plan/epoch/layout.
 */
#pragma once
#include "rad_types.h"
#include <cstddef>
#include <cstdint>

namespace rad::ds41_packed_inverse {
struct Stats {
    uint64_t blocks = 0, rounds = 0, raw_rounds = 0;
    uint64_t codes_bytes = 0, destination_bytes = 0;
};
/* Complete unchanged ggml_mxfp4 matrix size,0 for invalid/overflowing shapes. */
size_t block_bytes(int N, int K) noexcept;
/* Validate the whole header, payload extents and strictly increasing/in-bounds
 * exception list BEFORE writing anything. Aliases are refused. On every refusal
 * dst and stats remain unchanged; success writes exactly block_bytes(N,K),
 * leaving any extra destination capacity untouched. Extra source slot padding
 * is ignored. Source scale bytes0/255 and every nibble are preserved verbatim. */
int to_block(const uint8_t* source, size_t source_bytes, int N, int K,
             uint8_t* dst, size_t capacity, Stats* stats = nullptr) noexcept;
} // namespace rad::ds41_packed_inverse
