/* Correctness baseline for gfx1201. One wave32 owns one output; four waves/workgroup.
 * Inputs and weights are read directly; f32 products/reduction, one output rounding.
 * No allocation, synchronization, atomics or persistent expanded weights in a launch.
 * This is NOT a WMMA prefill implementation or a measured performance claim. */
#include <hip/hip_runtime.h>
#include "ds41_packed.h"
#include "../arch/deepseek41/state_layout.h" // also type-check the future cache mapping on device
#define RAD_GGUF_IQ_TABLE_STORAGE __device__ __constant__ static const
#include "../arch/common/rad_gguf_iq_tables.h"
#undef RAD_GGUF_IQ_TABLE_STORAGE

namespace {
namespace p=rad::ds41::packed;
__device__ __forceinline__ p::Tables tables() {
    return {rad::gguf::tables::iq2xxs_grid,rad::gguf::tables::iq2xs_grid,
            rad::gguf::tables::iq2s_grid,rad::gguf::tables::iq3xxs_grid,rad::gguf::tables::ksigns_iq2xs};
}
__device__ __forceinline__ float load(const void* a,int64_t offset,uint32_t dtype) {
    if(dtype==RAD_F32)return static_cast<const float*>(a)[offset];
    const auto h=static_cast<const uint16_t*>(a)[offset];
    return dtype==RAD_BF16?p::from_bits(uint32_t(h)<<16):p::half(h);
}
__device__ __forceinline__ void store(void* y,int64_t offset,uint32_t dtype,float value) {
    if(dtype==RAD_F32)static_cast<float*>(y)[offset]=value;
    else static_cast<uint16_t*>(y)[offset]=dtype==RAD_BF16?p::to_bf16(value):p::to_half(value);
}
template<p::Type type,bool routed>
__global__ __launch_bounds__(128) void gemm(const void* a,const void* weights,void* y,
    const int32_t* expert_ids,const int32_t* row_ids,int32_t* status,
    int64_t input_rows,int64_t jobs,int64_t n,int64_t k,int64_t experts,int64_t row_bytes,
    int64_t a_pitch,int64_t y_pitch,uint32_t a_dtype,uint32_t y_dtype) {
    const int lane=int(threadIdx.x)&31,wave=int(threadIdx.x)>>5;
    const int64_t output=int64_t(blockIdx.x)*4+wave;
    if(output>=jobs*n)return;
    const int64_t job=output/n,column=output%n;
    int64_t token=job;const uint8_t* w=static_cast<const uint8_t*>(weights);int error=RAD_OK;
    if constexpr(routed) {
        const int32_t e=expert_ids[job],r=row_ids[job];token=r;
        if(e<0||e>=experts||r<0||r>=input_rows)error=RAD_E_SHAPE;
        else {w=static_cast<const uint8_t* const*>(weights)[e];if(!w)error=RAD_E_STATE;}
    }
    float sum=0;
    if(error==RAD_OK) {
        const auto* row=w+column*row_bytes;const auto grid=tables();
        for(int64_t c=lane;c<k;c+=32) {
            const auto* block=row+(c/p::block_values(type))*p::block_bytes(type);
            const float product=load(a,token*a_pitch+c,a_dtype)*p::decode(type,block,int(c%p::block_values(type)),grid);
            sum=sum+product;
        }
        // All lanes of the owning wave participate, including K < lane (K is at least 32).
#pragma unroll
        for(int delta=16;delta;delta/=2)sum=sum+__shfl_down(sum,delta,32);
    } else sum=p::from_bits(0x7fc00000u);
    if(lane==0) {
        store(y,job*y_pitch+column,y_dtype,sum);
        if constexpr(routed)if(column==0)status[job]=error;
    }
}
template<p::Type type,bool routed>
int emit(const RadArgs* a,RadStream stream,const p::Launch& l) {
    const auto* t=a->t;const auto& g=l.g;
    const dim3 block(128),grid(unsigned((l.jobs*g.n+3)/4));
    hipLaunchKernelGGL((gemm<type,routed>),grid,block,0,static_cast<hipStream_t>(stream),
        t[0].data,t[1].data,t[routed?4:2].data,
        routed?static_cast<const int32_t*>(t[2].data):nullptr,
        routed?static_cast<const int32_t*>(t[3].data):nullptr,
        routed?static_cast<int32_t*>(t[5].data):nullptr,
        l.input_rows,l.jobs,g.n,g.k,g.experts,g.row_bytes,l.a_pitch,l.y_pitch,g.a_dtype,g.y_dtype);
    return hipGetLastError()==hipSuccess?RAD_OK:RAD_E_DEVICE;
}
template<bool routed>
int launch(const RadArgs* a,RadStream stream) {
    p::Launch l;const int rc=p::prepare(a,routed,&l);if(rc!=RAD_OK)return rc;
#define DS41_CASE(f) case p::Type::f:return emit<p::Type::f,routed>(a,stream,l)
    switch(l.g.type) {
        DS41_CASE(iq2_xxs);DS41_CASE(iq2_xs);DS41_CASE(iq2_s);DS41_CASE(iq3_xxs);DS41_CASE(q5_k);DS41_CASE(q8_0);
    }
#undef DS41_CASE
    return RAD_E_DTYPE;
}
}
extern "C" int ds41_packed_gemm(const RadArgs* a,RadStream stream) {return launch<false>(a,stream);}
extern "C" int ds41_packed_routed(const RadArgs* a,RadStream stream) {return launch<true>(a,stream);}
