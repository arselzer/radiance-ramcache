/* All payload checks finish before histogram clear OR any job write. No sync,
 * allocation, sort, stream creation or host read of device IDs. */
#include <hip/hip_runtime.h>
#include "ds41_device_plan.h"
#include <algorithm>
#include "ds41_route_jobs_impl.h"
namespace {
namespace r=rad::ds41::route_jobs;
__global__ void initialize(int32_t* status){if(threadIdx.x==0)*status=RAD_OK;}
__global__ void preflight(const r::Plan* device){
    const auto& plan=*device;
    const int64_t stride=int64_t(blockDim.x)*gridDim.x;
    for(int64_t job=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;job<plan.jobs;job+=stride){
        const int status=r::check_job(plan,job);if(status!=RAD_OK)atomicCAS(plan.status,RAD_OK,status);
    }
}
__global__ void clear_counts(const r::Plan* device){
    const auto& plan=*device;
    if(*plan.status!=RAD_OK)return;
    const int64_t expert=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(expert<plan.experts)r::clear_expert(plan,expert);
}
__global__ void jobs(const r::Plan* device){
    const auto& plan=*device;
    if(*plan.status!=RAD_OK)return;
    const int64_t stride=int64_t(blockDim.x)*gridDim.x;
    for(int64_t job=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;job<plan.jobs;job+=stride){
        const int32_t expert=r::write_job(plan,job);
        atomicAdd(static_cast<int32_t*>(plan.t[r::expert_counts].data)+rad::ds41::math::offset(plan.t[r::expert_counts],expert),1);
    }
}
}
extern "C" int ds41_route_jobs(const RadArgs* args,RadStream stream){
    r::Plan plan;const int status=r::validate(args,&plan);if(status!=RAD_OK)return status;
    const auto actual=static_cast<hipStream_t>(stream);
    const r::Plan* device=nullptr;const int staged=rad::ds41::stage_plan(args,plan,stream,&device);if(staged!=RAD_OK)return staged;
    hipLaunchKernelGGL(initialize,dim3(1),dim3(32),0,actual,plan.status);if(hipGetLastError()!=hipSuccess)return RAD_E_DEVICE;
    const unsigned blocks=unsigned(std::min<int64_t>((plan.jobs+255)/256,4096));
    hipLaunchKernelGGL(preflight,dim3(blocks),dim3(256),0,actual,device);if(hipGetLastError()!=hipSuccess)return RAD_E_DEVICE;
    hipLaunchKernelGGL(clear_counts,dim3(unsigned((plan.experts+127)/128)),dim3(128),0,actual,device);if(hipGetLastError()!=hipSuccess)return RAD_E_DEVICE;
    hipLaunchKernelGGL(jobs,dim3(blocks),dim3(256),0,actual,device);
    return hipGetLastError()==hipSuccess?RAD_OK:RAD_E_DEVICE;
}
