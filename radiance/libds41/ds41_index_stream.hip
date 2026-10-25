#include <hip/hip_runtime.h>
#include "ds41_index_stream_impl.h"
namespace {
namespace x=rad::ds41::index_stream;
__global__ void initialize(int32_t* flag){if(!threadIdx.x)*flag=RAD_OK;}
__global__ void metadata(const x::Plan* device_plan){
    const auto& p=*device_plan;
    const int64_t step=int64_t(blockDim.x)*gridDim.x;
    for(int64_t r=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;r<p.paged.rows;r+=step){const int rc=x::check_row(p,r);if(rc!=RAD_OK)atomicCAS(p.paged.status,RAD_OK,rc);}
}
template<bool Payload>__global__ __launch_bounds__(128) void preflight(const x::Plan* device_plan){
    const auto& p=*device_plan;
    if(*p.paged.status!=RAD_OK)return;const int64_t row=blockIdx.x;
    if constexpr(Payload)for(int64_t i=threadIdx.x;i<p.paged.heads*(p.paged.dim+1);i+=blockDim.x){const int rc=x::check_input(p,row,i);if(rc!=RAD_OK)atomicCAS(p.paged.status,RAD_OK,rc);}
    const int64_t count=x::items(p,row);
    for(int64_t i=threadIdx.x;i<count;i+=blockDim.x){const int rc=Payload?x::check_value(p,row,i):x::check_page(p,row,i);if(rc!=RAD_OK)atomicCAS(p.paged.status,RAD_OK,rc);}
}
__global__ __launch_bounds__(128) void selection(const x::Plan* device_plan){
    const auto& p=*device_plan;
    if(*p.paged.status!=RAD_OK)return;const int64_t row=blockIdx.x,count=x::items(p,row);
    extern __shared__ x::Entry storage[];auto* selected=storage;auto* blocks=selected+p.topk;float* scores=reinterpret_cast<float*>(blocks+p.top_blocks);
    __shared__ x::Working working;if(!threadIdx.x)working={0,0,rad::ds41::packed::from_bits(0xFF800000u)};__syncthreads();
    for(int64_t base=0;base<count;base+=x::tile_keys){
        const int64_t ordinal=base+threadIdx.x,key=ordinal<count?x::key_at(p,row,ordinal):-1;
        scores[threadIdx.x]=key>=0?x::score(p,row,key):rad::ds41::packed::from_bits(0xFF800000u);__syncthreads();
        if(!threadIdx.x)for(int64_t i=0;i<x::tile_keys&&base+i<count;++i){const int64_t at=x::key_at(p,row,base+i);if(at>=0)x::merge_key(p,row,at,scores[i],selected,blocks,&working);}
        __syncthreads();
    }
    if(!threadIdx.x)x::publish(p,row,selected,blocks,working);
}
}
extern "C" int ds41_index_stream(const RadArgs* a,RadStream stream){
    x::Plan p;const int rc=x::validate(a,&p);if(rc!=RAD_OK)return rc;auto s=static_cast<hipStream_t>(stream);
    const x::Plan* device_plan=nullptr;const int staged=rad::ds41::stage_plan(a,p,stream,&device_plan);if(staged!=RAD_OK)return staged;
    hipLaunchKernelGGL(initialize,dim3(1),dim3(32),0,s,p.paged.status);if(hipGetLastError()!=hipSuccess)return RAD_E_DEVICE;
    hipLaunchKernelGGL(metadata,dim3(unsigned(std::min<int64_t>((p.paged.rows+127)/128,4096))),dim3(128),0,s,device_plan);if(hipGetLastError()!=hipSuccess)return RAD_E_DEVICE;
    hipLaunchKernelGGL((preflight<false>),dim3(unsigned(p.paged.rows)),dim3(128),0,s,device_plan);if(hipGetLastError()!=hipSuccess)return RAD_E_DEVICE;
    hipLaunchKernelGGL((preflight<true>),dim3(unsigned(p.paged.rows)),dim3(128),0,s,device_plan);if(hipGetLastError()!=hipSuccess)return RAD_E_DEVICE;
    hipLaunchKernelGGL(selection,dim3(unsigned(p.paged.rows)),dim3(128),size_t(x::shared_bytes(p)),s,device_plan);
    return hipGetLastError()==hipSuccess?RAD_OK:RAD_E_DEVICE;
}
