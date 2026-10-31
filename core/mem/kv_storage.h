/* kv_storage.h -- physical KV pages without changing their logical BF16 geometry.
 *
 * Packed contracts: one original DS41 FP4 row per FULL page, or sixteen original FP8 rows per
 * native WINDOW page. Each physical page is
 * self-contained, so COW, prefix retention and tier transfers move exactly one byte fragment and
 * need no escape-pool lifetime or BF16 shadow. The format is deliberately independent of cache
 * capacity, host expert budget and the pool's physical backing/growth policy. */
#pragma once
#include "rad_builder.h"

namespace rad {

inline bool kv_storage_packed(const RadKVGroupDecl& d) {
    return d.storage_format != RAD_KV_STORAGE_PLAIN;
}

/* Validate at declaration and again in KVManager (host tests and sizing callers may construct
 * KVGroupInfo directly). Native512 WINDOW is a separate fixed16-slot contract. */
inline const char* kv_storage_decl_error(const RadKVGroupDecl& d) {
    if (d.storage_format == RAD_KV_STORAGE_PLAIN)
        return d.storage_block_bytes == 0 ? nullptr : "plain storage requires storage_block_bytes=0";
    if (d.storage_format == RAD_KV_STORAGE_DS41_FP8_WINDOW_G32) {
        if (d.kind != RAD_KV_WINDOW || d.dtype != RAD_BF16 || d.n_head_kv != 1 ||
            d.head_dim != 256 || d.window <= 0)
            return "packed FP8 window requires native512 logical BF16 WINDOW with one head";
        return d.storage_block_bytes == 16 * 528 ? nullptr :
            "packed FP8 window requires sixteen complete 528-byte rows per page";
    }
    if (d.storage_format != RAD_KV_STORAGE_DS41_FP4_E4M3_G16 &&
        d.storage_format != RAD_KV_STORAGE_DS41_FP4_POW2_G32)
        return "unknown physical storage format";
    if (d.kind != RAD_KV_FULL || d.dtype != RAD_BF16 || d.n_head_kv != 1)
        return "packed DS41 storage requires FULL logical BF16 with one KV head";
    const int64_t bytes = d.storage_format == RAD_KV_STORAGE_DS41_FP4_E4M3_G16 ? 288 : 68;
    if (d.storage_block_bytes != bytes)
        return "packed DS41 storage_block_bytes does not match the format's physical row";
    if (d.head_dim <= 0) return "packed DS41 logical head_dim must be positive";
    return nullptr;
}

/* Overflow-free exact geometry check. A page is one original 512-wide latent or 128-wide index
 * row, exposed logically as [1, block_size, 2*head_dim] by the old cache. DS41 currently uses
 * block_size 1/2; retaining the geometry rule avoids any cache-budget or context-size assumption. */
inline const char* kv_storage_page_error(const RadKVGroupDecl& d, int64_t block_size) {
    if (const char* why = kv_storage_decl_error(d)) return why;
    if (!kv_storage_packed(d)) return nullptr;
    if (d.storage_format == RAD_KV_STORAGE_DS41_FP8_WINDOW_G32)
        return block_size == 16 ? nullptr : "packed native FP8 WINDOW requires the original 16-slot page";
    const int64_t width = d.storage_format == RAD_KV_STORAGE_DS41_FP4_E4M3_G16 ? 512 : 128;
    if (block_size <= 0 || block_size > width / 2 || width % (2 * block_size) != 0 ||
        d.head_dim != width / (2 * block_size))
        return "packed DS41 page must hold exactly one original producer row";
    return nullptr;
}

}  // namespace rad
