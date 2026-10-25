/* Bit-exact opaque latent copies; no cache arithmetic/quantization. A separate same-stream
 * preflight validates all store mappings before any cache write. Device errors are status
 * operands; a future graph must check them before committing a request. */
#include <hip/hip_runtime.h>
#include "ds41_latent.h"
namespace {
namespace l=rad::ds41::latent;
__global__ void validate_slots(const int64_t* positions,const int64_t* input_slots,int32_t* status,
                               int64_t m,int64_t slots,int64_t ratio,int64_t b,bool store) {
    const int64_t row=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(row>=m)return;
    const int64_t s=input_slots[row];int result=RAD_OK;
    if(store) {
        const int64_t p=positions[row];
        if(p<0||p%ratio!=ratio-1||s<ratio-1||s>=slots||s%b!=p%b)result=RAD_E_SHAPE;
        if(result==RAD_OK)for(int64_t other=0;other<m;++other)if(other!=row&&input_slots[other]==s) {result=RAD_E_INVAL;break;}
    } else if(s<0||s>slots-ratio||s%b%ratio)result=RAD_E_SHAPE;
    status[row]=result;
}
__global__ void seal_status(int32_t* status,int64_t m) {
    // A separate same-stream stage avoids races with validation or status readers.
    int error=RAD_OK;for(int64_t row=0;row<m;++row)if(status[row]!=RAD_OK) {error=status[row];break;}
    if(error!=RAD_OK)for(int64_t row=0;row<m;++row)status[row]=error;
}
template<class T> __global__ void store(const T* x,const int64_t* final_slots,T* cache,
    const int32_t* status,int64_t m,int64_t d,int64_t ratio,int64_t pitch) {
    const int64_t row=blockIdx.x;if(row>=m||status[row]!=RAD_OK)return;
    const int64_t first=final_slots[row]-ratio+1,offset=first*(d/ratio);
    for(int64_t col=threadIdx.x;col<d;col+=blockDim.x)cache[offset+col]=x[row*pitch+col];
}
template<class T> __global__ void gather(const T* cache,const int64_t* first_slots,T* x,const int32_t* status,
    int64_t m,int64_t d,int64_t ratio,int64_t pitch) {
    const int64_t row=blockIdx.x;if(row>=m)return;
    if(status[row]!=RAD_OK)return;
    const int64_t first=first_slots[row];
    const int64_t offset=first*(d/ratio);
    for(int64_t col=threadIdx.x;col<d;col+=blockDim.x)x[row*pitch+col]=cache[offset+col];
}
template<class T> int issue(const RadArgs* a,RadStream stream,const l::Launch& v,bool write) {
    const auto* t=a->t;const auto& layout=v.layout;const auto s=static_cast<hipStream_t>(stream);
    const auto* slots=static_cast<const int64_t*>(t[write?2:1].data);
    auto* status=static_cast<int32_t*>(t[write?4:3].data);
    hipLaunchKernelGGL(validate_slots,dim3(unsigned((v.rows+127)/128)),dim3(128),0,s,
        write?static_cast<const int64_t*>(t[1].data):nullptr,slots,status,
        v.rows,v.slots,layout.ratio,layout.block_size,write);
    if(hipGetLastError()!=hipSuccess)return RAD_E_DEVICE;
    hipLaunchKernelGGL(seal_status,dim3(1),dim3(1),0,s,status,v.rows);
    if(hipGetLastError()!=hipSuccess)return RAD_E_DEVICE;
    if(write) {
        hipLaunchKernelGGL((store<T>),dim3(unsigned(v.rows)),dim3(128),0,s,
            static_cast<const T*>(t[0].data),static_cast<const int64_t*>(t[2].data),static_cast<T*>(t[3].data),
            static_cast<const int32_t*>(t[4].data),v.rows,layout.dimension,layout.ratio,v.pitch);
    } else hipLaunchKernelGGL((gather<T>),dim3(unsigned(v.rows)),dim3(128),0,s,
            static_cast<const T*>(t[0].data),static_cast<const int64_t*>(t[1].data),static_cast<T*>(t[2].data),
            static_cast<const int32_t*>(t[3].data),v.rows,layout.dimension,layout.ratio,v.pitch);
    return hipGetLastError()==hipSuccess?RAD_OK:RAD_E_DEVICE;
}
int launch(const RadArgs* a,RadStream stream,bool store) {
    l::Launch v;const int status=l::prepare(a,store,&v);if(status!=RAD_OK)return status;
    // Copy F32 as integer bits too: signed zero/NaN payloads are data, not an arithmetic step.
    return v.layout.element_bytes==4?issue<uint32_t>(a,stream,v,store):issue<uint16_t>(a,stream,v,store);
}
}
extern "C" int ds41_latent_store(const RadArgs* a,RadStream s) {return launch(a,s,true);}
extern "C" int ds41_latent_gather(const RadArgs* a,RadStream s) {return launch(a,s,false);}
