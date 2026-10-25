/* Optional core status collector: no host read/sync, reset, allocation or new stream.
 * The core calls it before source scratch/status reuse, then joins/checks before sampling. */
#include <hip/hip_runtime.h>
#include "ds41_status.h"
#include "ds41_math.h"
#include <algorithm>
namespace {
__global__ void collect(const int32_t* source,int64_t count,int64_t stride,RadDeviceStatus* latch,uint32_t op) {
    const int64_t step=int64_t(blockDim.x)*gridDim.x;
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=step){
        const int32_t code=source[i*stride];
        if(code<0&&atomicCAS(&latch->code,0,code)==0)latch->op=op;
    }
}
int issue(const RadArgs* a,RadDeviceStatus* latch,uint32_t op,RadStream stream,int operand) {
    if(!a||!latch||uintptr_t(latch)%alignof(RadDeviceStatus)||uintptr_t(latch)>UINTPTR_MAX-sizeof(*latch))return RAD_E_INVAL;
    const int32_t* source=nullptr;int64_t count=1,stride=1,bytes=4;
    if(operand<0){
        if(!a->scratch||a->scratch_bytes<4)return RAD_E_INVAL;
        if(uintptr_t(a->scratch)%alignof(int32_t))return RAD_E_ALIGN;
        if(uintptr_t(a->scratch)>UINTPTR_MAX-4)return RAD_E_SHAPE;
        source=static_cast<const int32_t*>(a->scratch);
    }else{
        if(!a->t||operand>=a->n_t||a->t[operand].rank!=1)return RAD_E_INVAL;
        const auto& t=a->t[operand];count=t.shape[0];if(count<=0||count>INT_MAX)return RAD_E_SHAPE;
        const int rc=rad::ds41::math::tensor(t,RAD_I32,{count},&bytes);if(rc!=RAD_OK)return rc;
        source=static_cast<const int32_t*>(t.data);stride=t.stride[0];
    }
    if(rad::ds41::math::overlaps(latch,sizeof(*latch),source,bytes))return RAD_E_INVAL;
    const unsigned blocks=unsigned(std::min<int64_t>((count+255)/256,4096));
    hipLaunchKernelGGL(collect,dim3(blocks),dim3(256),0,static_cast<hipStream_t>(stream),source,count,stride,latch,op);
    return hipGetLastError()==hipSuccess?RAD_OK:RAD_E_DEVICE;
}
}
extern "C" int ds41_collect_scratch_status(const RadArgs* a,RadDeviceStatus* l,uint32_t op,RadStream s){return issue(a,l,op,s,-1);}
extern "C" int ds41_collect_routed_status(const RadArgs* a,RadDeviceStatus* l,uint32_t op,RadStream s){return issue(a,l,op,s,5);}
extern "C" int ds41_collect_store_status(const RadArgs* a,RadDeviceStatus* l,uint32_t op,RadStream s){return issue(a,l,op,s,4);}
extern "C" int ds41_collect_gather_status(const RadArgs* a,RadDeviceStatus* l,uint32_t op,RadStream s){return issue(a,l,op,s,3);}
