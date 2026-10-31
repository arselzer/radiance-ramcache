/* Logical BF16 debug views of self-contained packed pages. Serving kernels use the same codec;
 * this checked host adapter is for hashes/oracles and never allocates a persistent BF16 shadow. */
#pragma once
#include "mem/kv_storage.h"
#include "../../arch/common/rad_ds41_kv_codec.h"
#include "../../arch/common/rad_ds41_window_codec.h"

namespace rad {

static_assert((uint32_t)rad_ds41_kv_codec::Kind::Plain == RAD_KV_STORAGE_PLAIN);
static_assert((uint32_t)rad_ds41_kv_codec::Kind::Fp4E4M3G16 == RAD_KV_STORAGE_DS41_FP4_E4M3_G16);
static_assert((uint32_t)rad_ds41_kv_codec::Kind::Fp4Pow2G32 == RAD_KV_STORAGE_DS41_FP4_POW2_G32);
static_assert(rad_ds41_kv_codec::row_bytes(RAD_KV_STORAGE_DS41_FP4_E4M3_G16, 512) == 288);
static_assert(rad_ds41_kv_codec::row_bytes(RAD_KV_STORAGE_DS41_FP4_POW2_G32, 128) == 68);
static_assert((uint32_t)rad_ds41_window_codec::Kind::Fp8Pow2G32==RAD_KV_STORAGE_DS41_FP8_WINDOW_G32);
static_assert(rad_ds41_window_codec::row_bytes(512)==528);

/* `logical_off/n` describes ONE run in the original BF16 layer pool, before packing. Keep run
 * boundaries: the prefill hash mixes each run's length, so hashing whole decoded pages instead
 * would change the receipt for a ratio-2 cache even when every BF16 bit is equal. */
inline bool kv_storage_decode_run(const RadKVGroupDecl& d, int64_t block_size,
                                  const uint8_t* physical, int64_t physical_bytes,
                                  int64_t logical_off, int64_t n, uint16_t* scratch,
                                  int64_t scratch_elements, const uint8_t** out) {
    if (!out) return false;
    *out = nullptr;
    if (!kv_storage_packed(d) || kv_storage_page_error(d, block_size) || !physical || !scratch ||
        logical_off < 0 || n <= 0 || ((logical_off | n) & 1)) return false;
    const int64_t width = 2 * block_size * d.head_dim;
    const int64_t logical_page = width * 2;
    const int64_t page = logical_off / logical_page, within = logical_off % logical_page;
    if (scratch_elements < width || n > logical_page - within ||
        physical_bytes < d.storage_block_bytes || page >= physical_bytes / d.storage_block_bytes)
        return false;
    if(d.storage_format==RAD_KV_STORAGE_DS41_FP8_WINDOW_G32) {
        for(int64_t r=0;r<block_size;++r)
            if(!rad_ds41_window_codec::decode_row(physical+page*d.storage_block_bytes+r*528,512,scratch+r*512))return false;
    } else if (!rad_ds41_kv_codec::decode_row(d.storage_format,
            physical + page * d.storage_block_bytes, width, scratch)) return false;
    *out = reinterpret_cast<const uint8_t*>(scratch) + within;
    return true;
}

}  // namespace rad
