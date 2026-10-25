/* Internal baseline transport for large validated launch descriptors. Core AQL has
 * 768B argument segments, including hidden ABI bytes: copy bounded chunks to owned op scratch, then pass one
 * device pointer. No core ring change, DMA, allocation or host wait is needed. */
#pragma once
#include "rad_plugin.h"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
namespace rad::ds41 {
inline constexpr size_t device_plan_offset=16,device_plan_chunk_bytes=256;
template<class Plan>constexpr int64_t gpu_scratch_bytes() noexcept {
    static_assert(std::is_trivially_copyable_v<Plan>);
    static_assert(alignof(Plan)<=device_plan_offset);
    static_assert(sizeof(Plan)<=65536);
    return int64_t(device_plan_offset+((sizeof(Plan)+device_plan_chunk_bytes-1)/device_plan_chunk_bytes)*device_plan_chunk_bytes);
}
template<class Plan>int validate_plan_storage(const RadArgs* args,const Plan& p) noexcept {
    if(!args||!args->scratch||args->scratch_bytes<gpu_scratch_bytes<Plan>())return RAD_E_INVAL;
    const auto start=reinterpret_cast<uintptr_t>(args->scratch);
    if(start%alignof(Plan))return RAD_E_ALIGN;
    if(start>UINTPTR_MAX-uint64_t(gpu_scratch_bytes<Plan>()))return RAD_E_SHAPE;
    for(size_t i=0;i<sizeof(p.t)/sizeof(p.t[0]);++i) {
        if(p.spans[i]<=0)continue;
        const auto other=reinterpret_cast<uintptr_t>(p.t[i].data);
        const bool overlaps=start<=other?uint64_t(other-start)<uint64_t(gpu_scratch_bytes<Plan>()):
                                         uint64_t(start-other)<uint64_t(p.spans[i]);
        if(overlaps)return RAD_E_INVAL;
    }
    return RAD_OK;
}
struct DevicePlanChunk {uint8_t bytes[device_plan_chunk_bytes];};
static_assert(sizeof(DevicePlanChunk)==device_plan_chunk_bytes);
template<class Plan>uint32_t make_plan_chunk(const Plan& p,size_t part,DevicePlanChunk* out) noexcept {
    static_assert(std::is_trivially_copyable_v<Plan>);
    if(!out||part>=(sizeof(Plan)+device_plan_chunk_bytes-1)/device_plan_chunk_bytes)return 0;
    const size_t offset=part*device_plan_chunk_bytes;
    const uint32_t count=uint32_t(sizeof(Plan)-offset<device_plan_chunk_bytes?sizeof(Plan)-offset:device_plan_chunk_bytes);
    std::memset(out,0,sizeof(*out));std::memcpy(out->bytes,reinterpret_cast<const uint8_t*>(&p)+offset,count);return count;
}
} // namespace rad::ds41
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
namespace rad::ds41 {
namespace device_plan_detail {
template<class Plan>__global__ void store_chunk(DevicePlanChunk chunk,uint8_t* destination,uint32_t count) {
    for(uint32_t i=threadIdx.x;i<count;i+=blockDim.x)destination[i]=chunk.bytes[i];
}
}
template<class Plan>int stage_plan(const RadArgs* args,const Plan& p,RadStream stream,const Plan** out) {
    if(!out)return RAD_E_INVAL;
    const int rc=validate_plan_storage(args,p);if(rc!=RAD_OK)return rc;
    auto* destination=static_cast<uint8_t*>(args->scratch)+device_plan_offset;
    for(size_t part=0;part<(sizeof(Plan)+device_plan_chunk_bytes-1)/device_plan_chunk_bytes;++part) {
        DevicePlanChunk chunk;const auto count=make_plan_chunk(p,part,&chunk);
        hipLaunchKernelGGL((device_plan_detail::store_chunk<Plan>),dim3(1),dim3(128),0,
                          static_cast<hipStream_t>(stream),chunk,destination+part*device_plan_chunk_bytes,count);
        if(hipGetLastError()!=hipSuccess)return RAD_E_DEVICE;
    }
    *out=reinterpret_cast<const Plan*>(destination);return RAD_OK;
}
} // namespace rad::ds41
#endif
