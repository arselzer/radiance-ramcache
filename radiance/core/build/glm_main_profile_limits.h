#pragma once
// Internal closed research profiles. No public ABI/default serving change.
#include <cstdint>
#include "rad_plugin.h"
namespace rad {
inline constexpr bool glm_main_rows_supported(int64_t rows) noexcept {return rows==1||rows==4;}
inline constexpr uint64_t glm_main_record_bytes=3342336;
inline constexpr uint32_t glm_main_max_union=32;
struct GlmMainProfileCapacity {
    uint32_t rows=0,selected_union=0,regions=0;
    uint64_t cold_bytes=0,consumer_bytes=0,route_bank_bytes=0;
};
inline int glm_main_profile_capacity(int64_t rows,GlmMainProfileCapacity* out) noexcept {
    if(!out||!glm_main_rows_supported(rows))return RAD_E_UNSUPPORTED;
    GlmMainProfileCapacity q;q.rows=uint32_t(rows);q.selected_union=uint32_t(rows)*8;q.regions=q.selected_union+1;
    q.cold_bytes=uint64_t(q.selected_union)*glm_main_record_bytes;
    q.consumer_bytes=uint64_t(q.regions)*(glm_main_record_bytes+16384);
    q.route_bank_bytes=uint64_t(577+16*rows)*sizeof(int32_t);
    *out=q;return RAD_OK;
}
} // namespace rad
