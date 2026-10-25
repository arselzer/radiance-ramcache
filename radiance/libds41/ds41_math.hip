/* Serial-per-row GPU correctness baselines for the four existing ds41_* schemas. Deliberately
 * simple, with no WMMA/fast math, allocation, host reads of device payloads or synchronization.
 * Init, whole-batch preflight and compute are ordered on the caller's stream. A payload error
 * is recorded at the documented scratch flag and every output remains untouched.
 * Descriptor staging precedes all work; kernels receive only a Plan pointer to
 * stay below the core AQL argument limit. */
#include <hip/hip_runtime.h>
#include <algorithm>
#include "ds41_math_impl.h"

namespace {
namespace m=rad::ds41::math;
__global__ void clear_status(int32_t* flag) {if(threadIdx.x==0)*flag=RAD_OK;}
__global__ void preflight(const m::Plan* device,int64_t items) {
    const m::Plan& p=*device;
    const int64_t step=int64_t(blockDim.x)*gridDim.x;
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<items;i+=step) {
        const int rc=m::check_item(p,i);
        if(rc!=RAD_OK)atomicCAS(p.status,RAD_OK,rc);
    }
}
// Separate ordered kernel: no public output is written until every derived job passes.
__global__ __launch_bounds__(128) void derived_preflight(const m::Plan* device,int64_t jobs) {
    const m::Plan& p=*device;if(*p.status!=RAD_OK)return;
    extern __shared__ float workspace[];
    if(p.op==m::Op::hc_collapse) {
        const int64_t job=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
        if(job<jobs) {const int rc=m::checked_job<false>(p,job,nullptr);if(rc!=RAD_OK)atomicCAS(p.status,RAD_OK,rc);}
    } else if(threadIdx.x==0) {
        const int rc=m::checked_job<false>(p,blockIdx.x,workspace);
        if(rc!=RAD_OK)atomicCAS(p.status,RAD_OK,rc);
    }
}
__global__ __launch_bounds__(32) void row_compute(const m::Plan* device) {
    const m::Plan& p=*device;
    if(*p.status!=RAD_OK||threadIdx.x!=0)return;
    extern __shared__ float workspace[];
    const int64_t job=blockIdx.x;
    const int rc=m::checked_job<true>(p,job,workspace);
    if(rc!=RAD_OK)atomicCAS(p.status,RAD_OK,rc);
}
__global__ __launch_bounds__(128) void collapse(const m::Plan* device) {
    const m::Plan& p=*device;
    if(*p.status!=RAD_OK)return;
    const int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<p.rows*p.n){const int rc=m::checked_job<true>(p,i,nullptr);if(rc!=RAD_OK)atomicCAS(p.status,RAD_OK,rc);}
}
int launch(m::Op op,const RadArgs* args,RadStream stream) {
    m::Plan p;const int rc=m::validate(op,args,&p);if(rc!=RAD_OK)return rc;
    const m::Plan* device=nullptr;
    const int staged=rad::ds41::stage_plan(args,p,stream,&device);if(staged!=RAD_OK)return staged;
    auto s=static_cast<hipStream_t>(stream);
    hipLaunchKernelGGL(clear_status,dim3(1),dim3(32),0,s,p.status);
    if(hipGetLastError()!=hipSuccess)return RAD_E_DEVICE;
    const int64_t items=m::preflight_items(p);
    const unsigned blocks=unsigned(std::min<int64_t>((items+255)/256,4096));
    hipLaunchKernelGGL(preflight,dim3(blocks),dim3(256),0,s,device,items);
    if(hipGetLastError()!=hipSuccess)return RAD_E_DEVICE;
    const int64_t jobs=m::derived_jobs(p);
    const bool collapsed=op==m::Op::hc_collapse;
    hipLaunchKernelGGL(derived_preflight,dim3(unsigned(collapsed?(jobs+127)/128:jobs)),
                      dim3(collapsed?128:32),size_t(m::workspace_floats(p))*sizeof(float),s,device,jobs);
    if(hipGetLastError()!=hipSuccess)return RAD_E_DEVICE;
    if(op==m::Op::hc_collapse) {
        hipLaunchKernelGGL(collapse,dim3(unsigned((p.rows*p.n+127)/128)),dim3(128),0,s,device);
    } else {
        hipLaunchKernelGGL(row_compute,dim3(unsigned(jobs)),dim3(32),size_t(m::workspace_floats(p))*sizeof(float),s,device);
    }
    return hipGetLastError()==hipSuccess?RAD_OK:RAD_E_DEVICE;
}
}
extern "C" int ds41_hc_prepare(const RadArgs* a,RadStream s) {return launch(m::Op::hc_prepare,a,s);}
extern "C" int ds41_hc_collapse(const RadArgs* a,RadStream s) {return launch(m::Op::hc_collapse,a,s);}
extern "C" int ds41_router_topk(const RadArgs* a,RadStream s) {return launch(m::Op::router_topk,a,s);}
extern "C" int ds41_sparse_attn(const RadArgs* a,RadStream s) {return launch(m::Op::sparse_attn,a,s);}
