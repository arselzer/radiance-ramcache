/* Correctness baseline: one block/token, independent scalar decode and one output cast.
 * Complete ID preflight precedes any gather/output; a launch success is not payload success. */
#include <hip/hip_runtime.h>
#include "ds41_embedding_impl.h"
#define RAD_GGUF_IQ_TABLE_STORAGE __device__ __constant__ static const
#include "../arch/common/rad_gguf_iq_tables.h"
#undef RAD_GGUF_IQ_TABLE_STORAGE
namespace {
namespace e=rad::ds41::embedding;
__global__ void initialize(int32_t* status){if(!threadIdx.x)*status=RAD_OK;}
__global__ void preflight(const e::Plan* plan) {
    const auto& p=*plan;
    for(int64_t r=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;r<p.rows;r+=int64_t(blockDim.x)*gridDim.x) {
        const int rc=e::check_id(p,r);if(rc!=RAD_OK)atomicCAS(p.status,RAD_OK,rc);
    }
}
__global__ __launch_bounds__(128) void gather(const e::Plan* plan) {
    const auto& p=*plan;if(*p.status!=RAD_OK)return;
    const int64_t row=blockIdx.x;
    const rad::ds41::packed::Tables tables{rad::gguf::tables::iq2xxs_grid,rad::gguf::tables::iq2xs_grid,
        rad::gguf::tables::iq2s_grid,rad::gguf::tables::iq3xxs_grid,rad::gguf::tables::ksigns_iq2xs};
    for(int64_t col=threadIdx.x;col<p.k;col+=blockDim.x)
        rad::ds41::math::store(p.t[e::output],row*p.k+col,e::value(p,row,col,tables));
}
}
extern "C" int ds41_embedding(const RadArgs* args,RadStream stream) {
    e::Plan p;const int rc=e::prepare(args,&p);if(rc!=RAD_OK)return rc;
    const e::Plan* device_plan=nullptr;
    const int staged=rad::ds41::stage_plan(args,p,stream,&device_plan);if(staged!=RAD_OK)return staged;
    const auto s=static_cast<hipStream_t>(stream);
    hipLaunchKernelGGL(initialize,dim3(1),dim3(32),0,s,p.status);if(hipGetLastError()!=hipSuccess)return RAD_E_DEVICE;
    hipLaunchKernelGGL(preflight,dim3(unsigned((p.rows+127)/128)),dim3(128),0,s,device_plan);
    if(hipGetLastError()!=hipSuccess)return RAD_E_DEVICE;
    hipLaunchKernelGGL(gather,dim3(unsigned(p.rows)),dim3(128),0,s,device_plan);
    return hipGetLastError()==hipSuccess?RAD_OK:RAD_E_DEVICE;
}
