/* Isolated native GGUF embedding gather; no model lifecycle/architecture registration.
 * Runtime weight metadata stays logical [N,K] with the packed GEMM private storage dtype.
 * GPU payload errors are reported through scratch[0:4], distinct from enqueue success. */
#pragma once
#include "ds41_math.h"
namespace rad::ds41::embedding {
enum Operand {token_ids=0,weight,output,operand_count};
inline constexpr int64_t scratch_bytes=4,max_rows=65536;
struct Plan {
    RadTensor t[operand_count]{};int64_t spans[operand_count]{};
    packed::Type type{};int64_t rows=0,n=0,k=0,row_bytes=0;
    uint32_t y_dtype=0;int32_t* status=nullptr;
};
int prepare(const RadArgs*,Plan*);
} // namespace rad::ds41::embedding
extern "C" {
int ds41_embedding(const RadArgs*,RadStream);
int ds41_ref_embedding(const RadArgs*,RadStream) noexcept;
inline int64_t ds41_embedding_scratch(const RadArgs*) {
    return rad::ds41::gpu_scratch_bytes<rad::ds41::embedding::Plan>();
}
}
