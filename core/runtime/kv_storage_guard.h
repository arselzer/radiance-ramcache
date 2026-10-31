/* Every actual RAD_KV read/write of a packed page must name the narrow packed contract. Declare
 * does not record KV operand handles, so checking its first block-size consumer is insufficient. */
#pragma once
#include "rad_builder.h"
#include <cstring>

namespace rad {
inline const char* kv_storage_operand_error(uint32_t format, const char* op, const char* kernel,
                                           const char* kernel_op, int operand,
                                           int64_t declared_format, int64_t resolved_format) {
    if (format == RAD_KV_STORAGE_PLAIN) return nullptr;
    if (declared_format != format || resolved_format != format)
        return "packed KV requires matching explicit declared and resolved kv_format";
    if (!op || !kernel || !kernel_op || std::strcmp(op, kernel_op))
        return "packed KV requires a dedicated matching packed kernel op";
    if (!std::strcmp(op, "entry_store_kvpack") && !std::strcmp(kernel, "ds41_entry_store_kvpack") &&
        operand == 3 && (format == RAD_KV_STORAGE_DS41_FP4_E4M3_G16 ||
                         format == RAD_KV_STORAGE_DS41_FP4_POW2_G32)) return nullptr;
    if (!std::strcmp(op, "index_score_kvpack") &&
        !std::strcmp(kernel, "ds41_index_score_d128_kvpack") && operand == 2 &&
        format == RAD_KV_STORAGE_DS41_FP4_POW2_G32) return nullptr;
    if (!std::strcmp(op, "attn_sparse_sink_kvpack") &&
        !std::strcmp(kernel, "ds41_attn_sparse_sink_d512_kvpack") && operand == 4 &&
        format == RAD_KV_STORAGE_DS41_FP4_E4M3_G16) return nullptr;
    return "packed KV operand is outside the explicit packed reader/writer contract";
}
}  // namespace rad
