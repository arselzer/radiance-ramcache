#include <hip/hip_runtime.h>
#include "ds41_device_plan.h"
#include "ds41_index_paged_impl.h"
namespace {
namespace p=rad::ds41::paged_index;
__global__ void initialize(int32_t* flag){if(!threadIdx.x)*flag=RAD_OK;}
__global__ void preflight(const p::Plan* device,int64_t count){
    const auto& plan=*device;
    const int64_t stride=int64_t(blockDim.x)*gridDim.x;
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=stride){const int rc=p::check_item(plan,i);if(rc!=RAD_OK)atomicCAS(plan.status,RAD_OK,rc);}
}
__global__ void scores(const p::Plan* device){
    const auto& plan=*device;
    if(*plan.status!=RAD_OK)return;
    const int64_t stride=int64_t(blockDim.x)*gridDim.x;
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<plan.rows*plan.keys;i+=stride)p::score_element(plan,i);
}
}
extern "C" int ds41_index_scores_paged(const RadArgs* args,RadStream stream){
    p::Plan plan;const int rc=p::validate(args,&plan);if(rc!=RAD_OK)return rc;auto s=static_cast<hipStream_t>(stream);
    const p::Plan* device=nullptr;const int staged=rad::ds41::stage_plan(args,plan,stream,&device);if(staged!=RAD_OK)return staged;
    hipLaunchKernelGGL(initialize,dim3(1),dim3(32),0,s,plan.status);if(hipGetLastError()!=hipSuccess)return RAD_E_DEVICE;
    const int64_t count=p::preflight_items(plan);const unsigned blocks=unsigned(std::min<int64_t>((count+255)/256,4096));
    hipLaunchKernelGGL(preflight,dim3(blocks),dim3(256),0,s,device,count);if(hipGetLastError()!=hipSuccess)return RAD_E_DEVICE;
    hipLaunchKernelGGL(scores,dim3(unsigned(std::min<int64_t>((plan.rows*plan.keys+127)/128,4096))),dim3(128),0,s,device);
    return hipGetLastError()==hipSuccess?RAD_OK:RAD_E_DEVICE;
}
