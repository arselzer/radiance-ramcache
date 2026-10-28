/* Exact ISTA 1017 GGUF spelling adapter. Source maps only: the original integer
 * controls and GGML codes remain authoritative; no table is regenerated here. */
#pragma once

#include <cstring>

namespace deepseek41::ista {
struct Alias { const char* canonical; const char* source; };
inline constexpr Alias aliases[] = {
    { "attention.indexer.candidate_source_layer", "attention.candidate_source_layer_id" },
    { "attention.indexer.candidate_top_k_blocks", "attention.candidate_topk_blocks" },
    { "attention.indexer.candidate_block_size", "attention.candidate_block_size" },
    { "attention.kv_source_layers", "kv_source_layer_ids" },
    { "attention.indexer.source_layers", "index_source_layer_ids" },
    { "ple.layers", "engram.layer_ids" },
    { "ple.ngram_size", "engram_max_ngram_size" },
    { "ple.heads_per_ngram", "engram_n_heads" },
    { "embedding_length_per_layer_input", "engram_head_dim" },
    { "engram.pad_token_id", "engram_pad_token_id" },
};
inline const char* source_key(const char* canonical) {
    if (!canonical) return nullptr;
    for (const Alias& a : aliases)
        if (!std::strcmp(a.canonical, canonical)) return a.source;
    return nullptr;
}
inline constexpr const char* token_map = "engram_token_map.weight";
inline constexpr const char* primes = "engram_primes.weight";
inline constexpr const char* offsets = "engram_offsets.weight";
inline constexpr const char* multipliers = "engram_multipliers.weight";
inline constexpr long long table_rows = 768022850;
inline constexpr long long layer_rows[] = {384006168, 384016682};
inline constexpr long long layers[] = {1, 14};
}  // namespace deepseek41::ista
