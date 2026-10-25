/* Wave32 group reductions are exact absmax; value conversion is the independent
 * bit encoder. Whole-batch phases forbid writes until masks/payloads pass.
 * Gradual FP32 underflow MUST be proved by the index2^-127 device fixture. */
#include <hip/hip_runtime.h>
#include "ds41_qdq.h"
namespace {
namespace q=rad::ds41::qdq;namespace m=rad::ds41::qdq_math;
__global__ void initialize(int32_t* flag){if(!threadIdx.x)*flag=RAD_OK;}
__global__ void masks(const q::Plan* plan){
    const auto& p=*plan;if(!p.masked)return;
    for(int64_t row=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;row<p.rows;row+=int64_t(blockDim.x)*gridDim.x){
        const int32_t bit=rad::ds41::math::id(p.t[q::emit],row);
        if(bit!=0&&bit!=1)atomicCAS(p.status,RAD_OK,RAD_E_INVAL);
    }
}
template<bool Write>__global__ __launch_bounds__(128) void groups(const q::Plan* plan){
    const auto& p=*plan;const int lane=int(threadIdx.x)&31,warp=int(threadIdx.x)>>5;
    // A uniform wave check is required: another wave can latch an error while
    // this one starts. Divergent early exits would invalidate the shuffles.
    const int code=__shfl(lane==0?*p.status:RAD_OK,0,32);if(code!=RAD_OK)return;
    const int width=m::width(p.profile);
    for(int64_t job=int64_t(blockIdx.x)*4+warp;job<p.rows*p.groups;job+=int64_t(gridDim.x)*4){
        const int64_t row=job/p.groups,group=job%p.groups;
        const bool active=!p.masked||rad::ds41::math::id(p.t[q::emit],row)!=0;
        const int64_t flat=row*p.dim+group*width+lane;
        if(!active){if constexpr(Write)if(lane<width)static_cast<uint16_t*>(p.t[q::output].data)[rad::ds41::math::offset(p.t[q::output],flat)]=0;continue;}
        float x=0,maximum=0;
        if(lane<width){
            x=m::bf16(static_cast<const uint16_t*>(p.t[q::input].data)[rad::ds41::math::offset(p.t[q::input],flat)]);
            if(!m::finite(x)){atomicCAS(p.status,RAD_OK,RAD_E_INVAL);x=0;}
            maximum=m::magnitude(x);
        }
        for(int delta=16;delta;delta>>=1){const float other=__shfl_xor(maximum,delta,32);if(other>maximum)maximum=other;}
        float factor=0;const int scaled=m::scale(p.profile,maximum,&factor);
        if(scaled!=RAD_OK){if(!lane)atomicCAS(p.status,RAD_OK,scaled);continue;}
        if(lane<width){
            uint16_t output=0;const int converted=m::value(p.profile,x,factor,&output);
            if(converted!=RAD_OK)atomicCAS(p.status,RAD_OK,converted);
            else if constexpr(Write)static_cast<uint16_t*>(p.t[q::output].data)[rad::ds41::math::offset(p.t[q::output],flat)]=output;
        }
    }
}
}
extern "C" int ds41_qdq_candidate(const RadArgs* a,RadStream stream){
    q::Plan p;const int rc=q::validate(a,&p);if(rc!=RAD_OK)return rc;
    const q::Plan* device=nullptr;const int staged=rad::ds41::stage_plan(a,p,stream,&device);if(staged!=RAD_OK)return staged;
    const auto s=static_cast<hipStream_t>(stream);
    hipLaunchKernelGGL(initialize,dim3(1),dim3(32),0,s,p.status);if(hipGetLastError()!=hipSuccess)return RAD_E_DEVICE;
    hipLaunchKernelGGL(masks,dim3(unsigned(std::min<int64_t>((p.rows+127)/128,4096))),dim3(128),0,s,device);if(hipGetLastError()!=hipSuccess)return RAD_E_DEVICE;
    const dim3 grid(unsigned(std::min<int64_t>((p.rows*p.groups+3)/4,4096))),threads(128);
    hipLaunchKernelGGL((groups<false>),grid,threads,0,s,device);if(hipGetLastError()!=hipSuccess)return RAD_E_DEVICE;
    hipLaunchKernelGGL((groups<true>),grid,threads,0,s,device);
    return hipGetLastError()==hipSuccess?RAD_OK:RAD_E_DEVICE;
}
