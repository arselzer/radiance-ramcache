/* Run ONLY in the stopped/restored-QFN two-lock GPU window. This does not benchmark or fit.
 * Every normal scale/code byte, every literal uint16 pair with random selectors/padding, mixed
 * unaligned rows, scalar/eight-value crossings and invalid metadata compare against the
 * UNCHANGED reference codec in the SAME device process; no float/NaN-equivalence comparison. */
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "ds41_kv_fast_oracle.h"
#define HC(x) do { const hipError_t e=(x); if(e!=hipSuccess){std::fprintf(stderr,"FAIL %s: %s\n",#x,hipGetErrorString(e));std::exit(2);} } while(0)
namespace o=ds41_kv_fast_oracle;
namespace kc=rad_ds41_kv_codec;
struct Report { unsigned normal=0,pair=0,row=0,metadata=0,bad=0; };
__global__ void normals(Report* r) {
    const unsigned at=blockIdx.x*blockDim.x+threadIdx.x;
    if(at>=2u*65536u)return;
    atomicAdd(&r->normal,1);atomicAdd(&r->bad,o::normal_case(at));
}
__global__ void pairs(Report* r) {
    const unsigned at=blockIdx.x*blockDim.x+threadIdx.x;
    if(at>=2u*65536u)return;
    atomicAdd(&r->pair,1);atomicAdd(&r->bad,o::pair_case(1+(at>>16),at&65535));
}
__global__ void rows(Report* r) {
    const unsigned at=blockIdx.x*blockDim.x+threadIdx.x;
    if(at>=256)return;
    const unsigned bad=at<128?o::row_case<kc::Kind::Fp4E4M3G16,512>(at)
                             :o::row_case<kc::Kind::Fp4Pow2G32,128>(at-128);
    atomicAdd(&r->row,1);atomicAdd(&r->bad,bad);
}
__global__ void metadata(Report* r) {
    if(threadIdx.x||blockIdx.x)return;
    atomicAdd(&r->metadata,1);atomicAdd(&r->bad,o::metadata_cases());
}
int main() {
    Report *d=nullptr,r;HC(hipMalloc((void**)&d,sizeof(Report)));HC(hipMemset(d,0,sizeof(Report)));
    normals<<<1024,128>>>(d);HC(hipGetLastError());pairs<<<1024,128>>>(d);HC(hipGetLastError());
    rows<<<8,32>>>(d);HC(hipGetLastError());metadata<<<1,32>>>(d);HC(hipGetLastError());
    HC(hipDeviceSynchronize());HC(hipMemcpy(&r,d,sizeof(r),hipMemcpyDeviceToHost));HC(hipFree(d));
    const bool counts=r.normal==131072&&r.pair==131072&&r.row==256&&r.metadata==1;
    std::printf("%s ds41 fast codec vs reference: normal=%u pair=%u rows=%u metadata=%u bad=%u\n",
                counts&&!r.bad?"PASS":"FAIL",r.normal,r.pair,r.row,r.metadata,r.bad);
    return counts&&!r.bad?0:1;
}
