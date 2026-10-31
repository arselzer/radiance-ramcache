/* GPU window ONLY. Exact decoder oracle; no timing/performance claim. */
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "ds41_kv_pair_oracle.h"
#define HC(x) do{const hipError_t e=(x);if(e!=hipSuccess){std::fprintf(stderr,"FAIL %s: %s\n",#x,hipGetErrorString(e));std::exit(2);}}while(0)
namespace o=ds41_kv_pair_oracle;
namespace kc=rad_ds41_kv_codec;
struct Report {unsigned normal=0,literal=0,rows=0,metadata=0,bad=0,even=0,odd=0,rejected=0;};
__device__ void add(Report* r,const o::Counts& c) {
    atomicAdd(&r->bad,c.bad);atomicAdd(&r->even,c.even);atomicAdd(&r->odd,c.odd);atomicAdd(&r->rejected,c.rejected);
}
__global__ void cases(Report* r) {
    const unsigned at=blockIdx.x*blockDim.x+threadIdx.x;
    if(at>=131072)return;
    const unsigned i=at&65535;
    add(r,at<65536?o::normal<kc::Kind::Fp4E4M3G16>(i):o::normal<kc::Kind::Fp4Pow2G32>(i));
    add(r,at<65536?o::literal<kc::Kind::Fp4E4M3G16>(i):o::literal<kc::Kind::Fp4Pow2G32>(i));
    atomicAdd(&r->normal,1);atomicAdd(&r->literal,1);
    if(at<256) {
        add(r,at<128?o::rows<kc::Kind::Fp4E4M3G16,512>(at):o::rows<kc::Kind::Fp4Pow2G32,128>(at-128));
        atomicAdd(&r->rows,1);
    }
    if(!at) {
        add(r,o::invalid<kc::Kind::Fp4E4M3G16,512>());add(r,o::invalid<kc::Kind::Fp4Pow2G32,128>());
        atomicAdd(&r->metadata,1);
    }
}
int main() {
    Report *d=nullptr,r;HC(hipMalloc((void**)&d,sizeof(Report)));HC(hipMemset(d,0,sizeof(Report)));
    cases<<<1024,128>>>(d);HC(hipGetLastError());HC(hipDeviceSynchronize());
    HC(hipMemcpy(&r,d,sizeof(r),hipMemcpyDeviceToHost));HC(hipFree(d));
    const bool ok=r.normal==131072&&r.literal==131072&&r.rows==256&&r.metadata==1&&
                  r.even>0&&r.odd>0&&r.rejected>0&&!r.bad;
    std::printf("%s ds41 pair codec vs reference: normal=%u literal=%u rows=%u metadata=%u even=%u odd=%u rejected=%u bad=%u\n",
        ok?"PASS":"FAIL",r.normal,r.literal,r.rows,r.metadata,r.even,r.odd,r.rejected,r.bad);
    return ok?0:1;
}
