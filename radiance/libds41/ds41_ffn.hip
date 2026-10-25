/* Staged immutable descriptor; only its device pointer crosses the AQL launch ABI. */
#include <hip/hip_runtime.h>
#include <algorithm>
#include "ds41_ffn_impl.h"
namespace {
namespace f=rad::ds41::ffn;
__global__ void initialize(int32_t* status) {if(threadIdx.x==0)*status=RAD_OK;}
__global__ void preflight(const f::Plan* device,int64_t items) {
    const f::Plan& p=*device;
    const int64_t stride=int64_t(blockDim.x)*gridDim.x;
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<items;i+=stride) {
        const int rc=f::check_item(p,i);if(rc!=RAD_OK)atomicCAS(p.status,RAD_OK,rc);
    }
}
__global__ __launch_bounds__(128) void elementwise(const f::Plan* device) {
    const f::Plan& p=*device;
    if(*p.status!=RAD_OK)return;
    const int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(p.op==f::Op::expert_act) {if(i<p.rows*p.n)f::activate(p,i);}
    else if(i<p.rows*p.hc*p.n)f::post(p,i);
}
__global__ __launch_bounds__(128) void sum_routes(const f::Plan* device) {
    const f::Plan& p=*device;
    if(*p.status!=RAD_OK)return;
    extern __shared__ int32_t order[];__shared__ int count;
    const int64_t row=blockIdx.x;
    if(threadIdx.x==0)count=f::order_routes(p,row,order);
    __syncthreads();if(count<0)return;
    for(int64_t d=threadIdx.x;d<p.n;d+=blockDim.x)f::combine(p,row,d,order,count);
}
int launch(f::Op op,const RadArgs* a,RadStream stream) {
    f::Plan p;const int rc=f::validate(op,a,&p);if(rc!=RAD_OK)return rc;
    const f::Plan* device=nullptr;
    const int staged=rad::ds41::stage_plan(a,p,stream,&device);if(staged!=RAD_OK)return staged;
    auto s=static_cast<hipStream_t>(stream);
    hipLaunchKernelGGL(initialize,dim3(1),dim3(32),0,s,p.status);if(hipGetLastError()!=hipSuccess)return RAD_E_DEVICE;
    const int64_t items=f::preflight_items(p);const auto blocks=unsigned(std::min<int64_t>((items+255)/256,4096));
    hipLaunchKernelGGL(preflight,dim3(blocks),dim3(256),0,s,device,items);if(hipGetLastError()!=hipSuccess)return RAD_E_DEVICE;
    if(op==f::Op::moe_combine) {
        hipLaunchKernelGGL(sum_routes,dim3(unsigned(p.rows)),dim3(128),size_t(p.experts)*4,s,device);
    } else {
        const int64_t values=p.rows*p.n*(op==f::Op::mhc_post?p.hc:1);
        hipLaunchKernelGGL(elementwise,dim3(unsigned((values+127)/128)),dim3(128),0,s,device);
    }
    return hipGetLastError()==hipSuccess?RAD_OK:RAD_E_DEVICE;
}
}
extern "C" int ds41_expert_act(const RadArgs* a,RadStream s) {return launch(f::Op::expert_act,a,s);}
extern "C" int ds41_moe_combine(const RadArgs* a,RadStream s) {return launch(f::Op::moe_combine,a,s);}
extern "C" int ds41_mhc_post(const RadArgs* a,RadStream s) {return launch(f::Op::mhc_post,a,s);}
