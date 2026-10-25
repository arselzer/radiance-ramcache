/* One block per packed sequence; each thread owns independent columns, preserving the exact
 * scalar expression order. No allocation/synchronization in launch and no speculative updates.
 * The validated descriptor is staged in op scratch, keeping AQL arguments small. */
#include <hip/hip_runtime.h>
#include <algorithm>
#include "ds41_compress_impl.h"
namespace {
namespace c=rad::ds41::compress;
__global__ void initialize(int32_t* flag) {if(threadIdx.x==0)*flag=RAD_OK;}
__global__ void preflight(const c::Plan* device,int64_t items) {
    const c::Plan& p=*device;
    const int64_t stride=int64_t(blockDim.x)*gridDim.x;
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<items;i+=stride) {
        const int rc=c::check_item(p,i);if(rc!=RAD_OK)atomicCAS(p.status,RAD_OK,rc);
    }
}
__global__ __launch_bounds__(128) void compute(const c::Plan* device) {
    const c::Plan& p=*device;
    if(*p.status!=RAD_OK)return;
    const int64_t s=blockIdx.x;
    const int64_t begin=rad::ds41::math::id(p.t[c::cu],s),end=rad::ds41::math::id(p.t[c::cu],s+1);
    if(begin==end)return;
    // No mutable header reads after this point: the prior kernel established cursor continuity.
    const int64_t slot=rad::ds41::math::id(p.t[c::state_ids],s),first=rad::ds41::math::id(p.t[c::positions],begin);
    for(int64_t d=threadIdx.x;d<p.dimension;d+=blockDim.x)c::column(p,begin,end,slot,first,d);
    __syncthreads();
    if(threadIdx.x==0)c::finish_sequence(p,begin,end,slot,first);
}
}
extern "C" int ds41_compress_pool(const RadArgs* a,RadStream stream) {
    c::Plan p;const int rc=c::validate(a,&p);if(rc!=RAD_OK)return rc;
    const c::Plan* device=nullptr;
    const int staged=rad::ds41::stage_plan(a,p,stream,&device);if(staged!=RAD_OK)return staged;
    auto s=static_cast<hipStream_t>(stream);
    hipLaunchKernelGGL(initialize,dim3(1),dim3(32),0,s,p.status);if(hipGetLastError()!=hipSuccess)return RAD_E_DEVICE;
    const int64_t items=c::preflight_items(p);const unsigned blocks=unsigned(std::min<int64_t>((items+255)/256,4096));
    hipLaunchKernelGGL(preflight,dim3(blocks),dim3(256),0,s,device,items);if(hipGetLastError()!=hipSuccess)return RAD_E_DEVICE;
    hipLaunchKernelGGL(compute,dim3(unsigned(p.sequences)),dim3(128),0,s,device);
    return hipGetLastError()==hipSuccess?RAD_OK:RAD_E_DEVICE;
}
