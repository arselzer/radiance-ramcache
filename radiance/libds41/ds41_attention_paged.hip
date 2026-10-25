#include <hip/hip_runtime.h>
#include "ds41_device_plan.h"
#include "ds41_attention_paged_impl.h"
namespace {
namespace a=rad::ds41::attention_paged;
__global__ void initialize(int32_t* flag) {if(!threadIdx.x)*flag=RAD_OK;}
template<int Phase>__global__ void preflight(const a::Plan* device,int64_t count) {
    const auto& p=*device;
    if(*p.status!=RAD_OK)return;
    const int64_t step=int64_t(blockDim.x)*gridDim.x;
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=step) {
        const int rc=Phase==0?a::check_sequence(p,i):Phase==1?a::check_pages(p,i):a::check_payload(p,i);
        if(rc!=RAD_OK)atomicCAS(p.status,RAD_OK,rc);
    }
}
__global__ __launch_bounds__(128) void attention(const a::Plan* device) {
    const auto& p=*device;
    if(*p.status!=RAD_OK)return;
    const int64_t row=blockIdx.x/p.heads,head=blockIdx.x%p.heads,count=a::candidate_count(p,row);
    extern __shared__ float work[];float* normalization=work+p.window+p.topk;
    for(int64_t k=threadIdx.x;k<count;k+=blockDim.x)work[k]=a::score(p,row,head,k);
    __syncthreads();
    if(!threadIdx.x)*normalization=a::maximum(p,head,work,count);
    __syncthreads();const float mx=*normalization;
    for(int64_t k=threadIdx.x;k<count;k+=blockDim.x)work[k]=mx==rad::ds41::packed::from_bits(0xFF800000u)?0.f : ::expf(work[k]-mx);
    __syncthreads();
    if(!threadIdx.x)*normalization=a::denominator(p,head,work,count,mx);
    __syncthreads();
    for(int64_t d=threadIdx.x;d<p.dim;d+=blockDim.x)a::attend_column(p,row,head,d,work,count,*normalization);
}
}
extern "C" int ds41_sparse_attn_paged(const RadArgs* args,RadStream stream) {
    a::Plan p;const int rc=a::validate(args,&p);if(rc!=RAD_OK)return rc;auto s=static_cast<hipStream_t>(stream);
    const a::Plan* device=nullptr;const int staged=rad::ds41::stage_plan(args,p,stream,&device);if(staged!=RAD_OK)return staged;
    hipLaunchKernelGGL(initialize,dim3(1),dim3(32),0,s,p.status);if(hipGetLastError()!=hipSuccess)return RAD_E_DEVICE;
    const unsigned seq_blocks=unsigned(std::min<int64_t>((p.sequences+127)/128,4096)),row_blocks=unsigned(std::min<int64_t>((p.rows+127)/128,4096));
    hipLaunchKernelGGL((preflight<0>),dim3(seq_blocks),dim3(128),0,s,device,p.sequences);if(hipGetLastError()!=hipSuccess)return RAD_E_DEVICE;
    hipLaunchKernelGGL((preflight<1>),dim3(row_blocks),dim3(128),0,s,device,p.rows);if(hipGetLastError()!=hipSuccess)return RAD_E_DEVICE;
    hipLaunchKernelGGL((preflight<2>),dim3(row_blocks),dim3(128),0,s,device,p.rows);if(hipGetLastError()!=hipSuccess)return RAD_E_DEVICE;
    hipLaunchKernelGGL(attention,dim3(unsigned(p.rows*p.heads)),dim3(128),size_t((p.window+p.topk+1)*4),s,device);
    return hipGetLastError()==hipSuccess?RAD_OK:RAD_E_DEVICE;
}
