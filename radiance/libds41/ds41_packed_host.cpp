/* Host reference over the identical native blocks as the experimental device kernels.
 * Scalar decode is independently compared with GGML/canonical oracles by CPU tests.
 * Accumulation here is ascending K, separate f32 multiply/add; GPU wave reduction differs. */
#include "ds41_packed_args.h"
#include "../arch/common/rad_gguf_iq_tables.h"

namespace rad::ds41::packed {
namespace {
const Tables tables{gguf::tables::iq2xxs_grid,gguf::tables::iq2xs_grid,gguf::tables::iq2s_grid,
                    gguf::tables::iq3xxs_grid,gguf::tables::ksigns_iq2xs};
float load(const RadTensor& t,int64_t index) {
    if(t.dtype==RAD_F32)return static_cast<const float*>(t.data)[index];
    const auto h=static_cast<const uint16_t*>(t.data)[index];
    return t.dtype==RAD_BF16?from_bits(uint32_t(h)<<16):half(h);
}
void store(const RadTensor& t,int64_t index,float value) {
    if(t.dtype==RAD_F32)static_cast<float*>(t.data)[index]=value;
    else static_cast<uint16_t*>(t.data)[index]=t.dtype==RAD_BF16?to_bf16(value):to_half(value);
}
int host(const RadArgs* args,bool routed) {
    Launch launch;int status=validate(args,routed,&launch);if(status!=RAD_OK)return status;
    const auto& g=launch.g;const auto* t=args->t;const int yi=routed?4:2;
    const auto* ids=routed?static_cast<const int32_t*>(t[2].data):nullptr;
    const auto* rows=routed?static_cast<const int32_t*>(t[3].data):nullptr;
    const auto* weights=routed?static_cast<const uint8_t* const*>(t[1].data):nullptr;
    // No output changes on a malformed final route or a missing/aliased selected expert.
    if(routed)for(int64_t j=0;j<launch.jobs;++j) {
        if(ids[j]<0||ids[j]>=g.experts||rows[j]<0||rows[j]>=launch.input_rows)return RAD_E_SHAPE;
        const auto* w=weights[ids[j]];if(!w)return RAD_E_STATE;
        const int64_t ybytes=((launch.jobs-1)*launch.y_pitch+g.n)*(rad_dtype_bits(g.y_dtype)/8);
        if(alias(w,g.n*g.row_bytes,t[yi].data,ybytes)||
           alias(w,g.n*g.row_bytes,t[5].data,launch.jobs*4))return RAD_E_INVAL;
    }
    for(int64_t j=0;j<launch.jobs;++j) {
        const int64_t token=routed?rows[j]:j;
        const auto* w=routed?weights[ids[j]]:static_cast<const uint8_t*>(t[1].data);
        for(int64_t n=0;n<g.n;++n) {
            float sum=0;
            const auto* row=w+n*g.row_bytes;
            for(int64_t k=0;k<g.k;++k) {
                const auto* block=row+(k/block_values(g.type))*block_bytes(g.type);
                const float product=load(t[0],token*launch.a_pitch+k)*decode(g.type,block,int(k%block_values(g.type)),tables);
                sum=sum+product;
            }
            store(t[yi],j*launch.y_pitch+n,sum);
        }
        if(routed)static_cast<int32_t*>(t[5].data)[j]=RAD_OK;
    }
    return RAD_OK;
}
}
} // namespace rad::ds41::packed
extern "C" int ds41_ref_packed_gemm(const RadArgs* a,RadStream) {return rad::ds41::packed::host(a,false);}
extern "C" int ds41_ref_packed_routed(const RadArgs* a,RadStream) {return rad::ds41::packed::host(a,true);}
