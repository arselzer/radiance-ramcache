#include <hip/hip_runtime.h>
#include "ds41_engram_gate_impl.h"
#include <algorithm>
namespace {
namespace eg=rad::ds41::engram_gate;
__global__ void initialize(int32_t* status) {if(!threadIdx.x)*status=RAD_OK;}
__global__ void preflight(const eg::Plan* device,int64_t items) {
    const auto& p=*device;const int64_t stride=int64_t(blockDim.x)*gridDim.x;
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<items;i+=stride) {
        const int rc=eg::check_item(p,i);if(rc!=RAD_OK)atomicCAS(p.status,RAD_OK,rc);
    }
}
__global__ void numerical_preflight(const eg::Plan* device,int64_t jobs) {
    const auto& p=*device;if(*p.status!=RAD_OK)return;
    const int64_t stride=int64_t(blockDim.x)*gridDim.x;
    for(int64_t job=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;job<jobs;job+=stride) {
        const int rc=eg::check_job(p,job);if(rc!=RAD_OK)atomicCAS(p.status,RAD_OK,rc);
    }
}
__global__ __launch_bounds__(128) void apply(const eg::Plan* device) {
    const auto& p=*device;if(*p.status!=RAD_OK)return;
    const int64_t job=blockIdx.x;__shared__ float gate;
    if(!threadIdx.x) {eg::evaluate(p,job,&gate);eg::write_gate(p,job,gate);}
    __syncthreads();
    for(int64_t d=threadIdx.x;d<p.dim;d+=blockDim.x)eg::write_element(p,job,d,gate);
}
}
extern "C" int ds41_engram_gate(const RadArgs* args,RadStream stream) {
    eg::Plan plan;const int rc=eg::validate(args,&plan);if(rc!=RAD_OK)return rc;
    const eg::Plan* device=nullptr;
    const int staged=rad::ds41::stage_plan(args,plan,stream,&device);if(staged!=RAD_OK)return staged;
    const auto hip_stream=static_cast<hipStream_t>(stream);
    hipLaunchKernelGGL(initialize,dim3(1),dim3(32),0,hip_stream,plan.status);
    if(hipGetLastError()!=hipSuccess)return RAD_E_DEVICE;
    const int64_t items=eg::preflight_items(plan),jobs=plan.tokens*plan.copies;
    hipLaunchKernelGGL(preflight,dim3(unsigned(std::min<int64_t>((items+255)/256,4096))),dim3(256),0,hip_stream,device,items);
    if(hipGetLastError()!=hipSuccess)return RAD_E_DEVICE;
    hipLaunchKernelGGL(numerical_preflight,dim3(unsigned(std::min<int64_t>((jobs+63)/64,4096))),dim3(64),0,hip_stream,device,jobs);
    if(hipGetLastError()!=hipSuccess)return RAD_E_DEVICE;
    hipLaunchKernelGGL(apply,dim3(unsigned(jobs)),dim3(128),0,hip_stream,device);
    return hipGetLastError()==hipSuccess?RAD_OK:RAD_E_DEVICE;
}
