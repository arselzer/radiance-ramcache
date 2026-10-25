/* Correctness-first floating indexer. One serial bounded heap per query for selection;
 * scoring is parallel over (query,key). No activation/cache quantization or allocation. */
#include <hip/hip_runtime.h>
#include "ds41_device_plan.h"
#include "ds41_index_impl.h"
namespace {
namespace x=rad::ds41::index;
__global__ void initialize(int32_t* flag){if(!threadIdx.x)*flag=RAD_OK;}
__global__ void preflight(const x::Plan* device,int64_t items){
    const auto& p=*device;
    const int64_t step=int64_t(blockDim.x)*gridDim.x;
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<items;i+=step){
        const int rc=x::check_item(p,i);if(rc!=RAD_OK)atomicCAS(p.status,RAD_OK,rc);
    }
}
__global__ void scores(const x::Plan* device){
    const auto& p=*device;
    if(*p.status!=RAD_OK)return;
    const int64_t step=int64_t(blockDim.x)*gridDim.x;
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<p.rows*p.keys;i+=step)x::score_element(p,i);
}
__global__ __launch_bounds__(32) void selection(const x::Plan* device){
    const auto& p=*device;
    if(*p.status!=RAD_OK||threadIdx.x)return;
    extern __shared__ x::Entry heap[];
    if(p.op==x::Op::candidates)x::candidate_row(p,blockIdx.x,heap);else x::topk_row(p,blockIdx.x,heap);
}
int launch(x::Op op,const RadArgs* args,RadStream stream){
    x::Plan p;const int rc=x::validate(op,args,&p);if(rc!=RAD_OK)return rc;auto s=static_cast<hipStream_t>(stream);
    const x::Plan* device=nullptr;const int staged=rad::ds41::stage_plan(args,p,stream,&device);if(staged!=RAD_OK)return staged;
    hipLaunchKernelGGL(initialize,dim3(1),dim3(32),0,s,p.status);if(hipGetLastError()!=hipSuccess)return RAD_E_DEVICE;
    const int64_t items=x::preflight_items(p);const unsigned blocks=unsigned(std::min<int64_t>((items+255)/256,4096));
    hipLaunchKernelGGL(preflight,dim3(blocks),dim3(256),0,s,device,items);if(hipGetLastError()!=hipSuccess)return RAD_E_DEVICE;
    if(op==x::Op::scores)hipLaunchKernelGGL(scores,dim3(unsigned(std::min<int64_t>((p.rows*p.keys+127)/128,4096))),dim3(128),0,s,device);
    else hipLaunchKernelGGL(selection,dim3(unsigned(p.rows)),dim3(32),size_t(p.topk)*sizeof(x::Entry),s,device);
    return hipGetLastError()==hipSuccess?RAD_OK:RAD_E_DEVICE;
}
}
extern "C" int ds41_index_scores(const RadArgs* a,RadStream s){return launch(x::Op::scores,a,s);}
extern "C" int ds41_candidate_blocks(const RadArgs* a,RadStream s){return launch(x::Op::candidates,a,s);}
extern "C" int ds41_index_topk(const RadArgs* a,RadStream s){return launch(x::Op::topk,a,s);}
