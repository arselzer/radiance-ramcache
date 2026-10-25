/* Experimental original-row-indexed completed latent store. Masked rows never read source
 * data/coordinates. No cache quantization, state lifecycle, gathering or architecture. */
#pragma once
#include "ds41_math.h"
#include "ds41_latent.h"
namespace rad::ds41::latent_emit {
enum Operand {x=0,positions,slot_mapping,emit,cache,status,operand_count};
inline constexpr int64_t scratch_bytes=4,max_rows=4096;
struct Plan {
    RadTensor t[operand_count]{};int64_t spans[operand_count]{};
    latent::Launch latent;int32_t* flag=nullptr;
};
int prepare(const RadArgs*,Plan*);
} // namespace rad::ds41::latent_emit
extern "C" {
int ds41_latent_store_emitted(const RadArgs*,RadStream);
int ds41_ref_latent_store_emitted(const RadArgs*,RadStream) noexcept;
inline int64_t ds41_latent_emit_scratch(const RadArgs*) {
    return rad::ds41::gpu_scratch_bytes<rad::ds41::latent_emit::Plan>();
}
}
