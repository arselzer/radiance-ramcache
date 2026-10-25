/* Completed original-row-indexed cache publication. Separate same-stream preflight kernels
 * establish all masks/coordinates/destination uniqueness before any cache write. */
#include <hip/hip_runtime.h>
#include "ds41_latent_emit_impl.h"
namespace {
namespace e=rad::ds41::latent_emit;
__global__ void initialize(int32_t* flag){if(!threadIdx.x)*flag=RAD_OK;}
template<int Phase>__global__ void preflight(const e::Plan* plan) {
    const auto& p=*plan;if(*p.flag!=RAD_OK)return;
    for(int64_t row=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;row<p.latent.rows;row+=int64_t(blockDim.x)*gridDim.x) {
        const int rc=Phase==0?e::check_mask(p,row):Phase==1?e::check_coordinate(p,row):e::check_duplicate(p,row);
        if(rc!=RAD_OK)atomicCAS(p.flag,RAD_OK,rc);
    }
}
__global__ void seal(const e::Plan* plan) {
    const auto& p=*plan;
    for(int64_t row=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;row<p.latent.rows;row+=int64_t(blockDim.x)*gridDim.x)e::seal_row(p,row);
}
__global__ __launch_bounds__(128) void store(const e::Plan* plan) {
    const auto& p=*plan;if(*p.flag!=RAD_OK)return;const int64_t row=blockIdx.x;
    if(!e::integer(p.t[e::emit],row))return;
    for(int64_t col=threadIdx.x;col<p.latent.layout.dimension;col+=blockDim.x)e::copy_element(p,row,col);
}
}
extern "C" int ds41_latent_store_emitted(const RadArgs* a,RadStream stream) {
    e::Plan p;const int checked=e::prepare(a,&p);if(checked!=RAD_OK)return checked;
    const e::Plan* device=nullptr;const int staged=rad::ds41::stage_plan(a,p,stream,&device);if(staged!=RAD_OK)return staged;
    const auto s=static_cast<hipStream_t>(stream);const dim3 grid(unsigned((p.latent.rows+127)/128)),threads(128);
    hipLaunchKernelGGL(initialize,dim3(1),dim3(32),0,s,p.flag);if(hipGetLastError()!=hipSuccess)return RAD_E_DEVICE;
    hipLaunchKernelGGL((preflight<0>),grid,threads,0,s,device);if(hipGetLastError()!=hipSuccess)return RAD_E_DEVICE;
    hipLaunchKernelGGL((preflight<1>),grid,threads,0,s,device);if(hipGetLastError()!=hipSuccess)return RAD_E_DEVICE;
    hipLaunchKernelGGL((preflight<2>),grid,threads,0,s,device);if(hipGetLastError()!=hipSuccess)return RAD_E_DEVICE;
    hipLaunchKernelGGL(seal,grid,threads,0,s,device);if(hipGetLastError()!=hipSuccess)return RAD_E_DEVICE;
    hipLaunchKernelGGL(store,dim3(unsigned(p.latent.rows)),threads,0,s,device);
    return hipGetLastError()==hipSuccess?RAD_OK:RAD_E_DEVICE;
}
