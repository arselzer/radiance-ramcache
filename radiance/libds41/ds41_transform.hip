#include <hip/hip_runtime.h>
#include "ds41_device_plan.h"
#include <algorithm>
#include "ds41_transform_impl.h"
namespace {
namespace t=rad::ds41::transform;
__global__ void initialize(int32_t* flag) {if(!threadIdx.x)*flag=RAD_OK;}
__global__ void preflight(const t::Plan* device,int64_t items) {
    const auto& p=*device;
    const int64_t stride=int64_t(blockDim.x)*gridDim.x;
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<items;i+=stride) {
        const int rc=t::check_item(p,i);if(rc!=RAD_OK)atomicCAS(p.status,RAD_OK,rc);
    }
}
__global__ __launch_bounds__(128) void norm(const t::Plan* device) {
    const auto& p=*device;
    if(*p.status!=RAD_OK)return;
    __shared__ float inverse;const int64_t row=blockIdx.x;
    if(!threadIdx.x)inverse=t::inverse_rms(p,row);__syncthreads();
    for(int64_t d=threadIdx.x;d<p.dim;d+=blockDim.x)t::norm_element(p,row,d,inverse);
}
__global__ __launch_bounds__(128) void rope(const t::Plan* device) {
    const auto& p=*device;
    if(*p.status!=RAD_OK)return;
    const int64_t step=int64_t(blockDim.x)*gridDim.x;
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<p.rows*p.heads*p.dim;i+=step)t::rope_element(p,i);
}
int launch(t::Op op,const RadArgs* a,RadStream stream) {
    t::Plan p;const int rc=t::validate(op,a,&p);if(rc!=RAD_OK)return rc;auto s=static_cast<hipStream_t>(stream);
    const t::Plan* device=nullptr;const int staged=rad::ds41::stage_plan(a,p,stream,&device);if(staged!=RAD_OK)return staged;
    hipLaunchKernelGGL(initialize,dim3(1),dim3(32),0,s,p.status);if(hipGetLastError()!=hipSuccess)return RAD_E_DEVICE;
    const auto items=t::preflight_items(p);const auto blocks=unsigned(std::min<int64_t>((items+255)/256,4096));
    hipLaunchKernelGGL(preflight,dim3(blocks),dim3(256),0,s,device,items);if(hipGetLastError()!=hipSuccess)return RAD_E_DEVICE;
    if(op==t::Op::rmsnorm)hipLaunchKernelGGL(norm,dim3(unsigned(p.rows)),dim3(128),0,s,device);
    else hipLaunchKernelGGL(rope,dim3(unsigned(std::min<int64_t>((p.rows*p.heads*p.dim+127)/128,4096))),dim3(128),0,s,device);
    return hipGetLastError()==hipSuccess?RAD_OK:RAD_E_DEVICE;
}
}
extern "C" int ds41_rmsnorm(const RadArgs* a,RadStream s) {return launch(t::Op::rmsnorm,a,s);}
extern "C" int ds41_rope_tail(const RadArgs* a,RadStream s) {return launch(t::Op::rope_tail,a,s);}
