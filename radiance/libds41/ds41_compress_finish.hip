#include <hip/hip_runtime.h>
#include "ds41_compress_finish_impl.h"
namespace {
namespace f=rad::ds41::compress_finish;
__global__ void initialize(int32_t* status){if(!threadIdx.x)*status=RAD_OK;}
__global__ void preflight(const f::Plan* device,int64_t items){
    const auto& p=*device;const int64_t step=int64_t(blockDim.x)*gridDim.x;
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<items;i+=step){const int rc=f::check_item(p,i);if(rc!=RAD_OK)atomicCAS(p.status,RAD_OK,rc);}
}
__global__ __launch_bounds__(128) void finish(const f::Plan* device){
    const auto& p=*device;if(*p.status!=RAD_OK)return;
    const int64_t row=blockIdx.x;__shared__ float inverse;__shared__ int active;
    if(!threadIdx.x){active=rad::ds41::math::id(p.t[f::emit],row);inverse=active?f::inverse_rms(p,row):0.f;}__syncthreads();
    for(int64_t d=threadIdx.x;d<p.dim;d+=blockDim.x)f::output_element(p,row,d,inverse,active!=0);
}
}
extern "C" int ds41_compress_finish(const RadArgs* args,RadStream stream){
    f::Plan p;const int rc=f::validate(args,&p);if(rc!=RAD_OK)return rc;
    const f::Plan* device=nullptr;const int staged=rad::ds41::stage_plan(args,p,stream,&device);if(staged!=RAD_OK)return staged;
    auto s=static_cast<hipStream_t>(stream);hipLaunchKernelGGL(initialize,dim3(1),dim3(32),0,s,p.status);if(hipGetLastError()!=hipSuccess)return RAD_E_DEVICE;
    const int64_t items=p.dim+p.rows,blocks=(items+127)/128;
    hipLaunchKernelGGL(preflight,dim3(unsigned(blocks<4096?blocks:4096)),dim3(128),0,s,device,items);if(hipGetLastError()!=hipSuccess)return RAD_E_DEVICE;
    hipLaunchKernelGGL(finish,dim3(unsigned(p.rows)),dim3(128),0,s,device);return hipGetLastError()==hipSuccess?RAD_OK:RAD_E_DEVICE;
}
