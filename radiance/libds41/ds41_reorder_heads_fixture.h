/* Tiny raw activation words, intentionally including nonfinite/signed-zero/subnormal bits.
 * Independent nested rank/token/head oracle; no new reorder routine creates expectations. */
#pragma once
#include <cstring>
#include "ds41_reorder_heads_impl.h"
#include <array>
#include <initializer_list>
#include <random>
#include <vector>
namespace rad::ds41::reorder_heads::testing {
struct Buffer {
    RadTensor t{};std::vector<uint32_t> words;int64_t bytes=0;
    Buffer()=default;
    Buffer(uint32_t dtype,std::initializer_list<int64_t> shape) {
        t.dtype=dtype;t.rank=shape.size();unsigned axis=0;for(auto n:shape)t.shape[axis++]=n;
        int64_t count=1;for(int i=int(t.rank)-1;i>=0;--i){t.stride[i]=count;count*=t.shape[i];}
        bytes=count*(rad_dtype_bits(dtype)/8);words.resize(size_t((bytes+3)/4)+8,0xA5A5A5A5u);refresh();
    }
    void refresh(){t.data=words.data()+4;}
    void* data(){return words.data()+4;}const void* data()const{return words.data()+4;}
};
struct Fixture {
    uint32_t dtype;int64_t tp,rows,heads,dim;bool flat;std::array<Buffer,operand_count> buffers;
    std::array<RadTensor,operand_count> tensors{};std::vector<RadParam> params;
    Fixture(uint32_t d=RAD_F32,int64_t ranks=4,int64_t tokens=3,int64_t h=2,int64_t width=7,bool flattened=false)
        :dtype(d),tp(ranks),rows(tokens),heads(h),dim(width),flat(flattened),
         buffers{flat?Buffer(d,{tp*rows,heads,dim}):Buffer(d,{tp,rows,heads,dim}),
                 flat?Buffer(RAD_F32,{tp*rows,heads}):Buffer(RAD_F32,{tp,rows,heads}),
                 Buffer(d,{rows,tp*heads,dim}),Buffer(RAD_F32,{rows,tp*heads})} {
        params={RAD_INT("M",rows+2),RAD_INT("tp",tp),RAD_INT("heads",heads),RAD_INT("head_dim",dim),RAD_STR("dtype",rad_dtype_name(dtype))};
        constexpr uint32_t special32[]{0,0x80000000u,0x7F800000u,0xFF800000u,0x7F800001u,0x7FC12345u,1,0x7FFFFFu,0x800000u,0xFFFFFFFFu};
        constexpr uint16_t special16[]{0,0x8000,0x7C00,0xFC00,0x7C01,0x7E55,1,0x3FF,0x400,0xFFFF};
        constexpr uint16_t specialbf[]{0,0x8000,0x7F80,0xFF80,0x7F81,0x7FC5,1,0x7F,0x80,0xFFFF};
        std::mt19937 rng(411017+unsigned(dtype)+unsigned(tp*rows*heads*dim));
        for(int input:{query,weights}) {
            auto& b=buffers[size_t(input)];const int width_bytes=rad_dtype_bits(b.t.dtype)/8;
            for(int64_t i=0;i<b.bytes/width_bytes;++i) {
                if(width_bytes==4){const uint32_t bits=i%3?uint32_t(rng()):special32[(size_t(i)/3)%std::size(special32)];
                    std::memcpy(static_cast<uint8_t*>(b.data())+i*4,&bits,4);}
                else {const size_t special=(size_t(i)/3)%std::size(special16);
                    const uint16_t bits=i%3?uint16_t(rng()):dtype==RAD_F16?special16[special]:specialbf[special];
                    std::memcpy(static_cast<uint8_t*>(b.data())+i*2,&bits,2);}
            }
        }
        refresh();
    }
    void refresh(){for(int i=0;i<operand_count;++i){buffers[size_t(i)].refresh();tensors[size_t(i)]=buffers[size_t(i)].t;}}
    RadArgs args(){refresh();RadArgs a{};a.t=tensors.data();a.n_t=operand_count;a.p=params.data();a.n_p=params.size();return a;}
    void integer(const char* key,int64_t value){for(auto& p:params)if(!std::strcmp(p.key,key))p.ival=value;}
    void text(const char* key,const char* value){for(auto& p:params)if(!std::strcmp(p.key,key))p.sval=value;}
    int run(){auto a=args();return emulate(&a);}
    std::array<Buffer,operand_count> expected()const {
        auto result=buffers;
        for(auto& b:result)b.refresh();
        const size_t bytes=size_t(rad_dtype_bits(dtype)/8);
        // Independent traversal starts with the all-gather's rank-major axes and scatters a
        // complete head byte row into each token's canonical global-head axis.
        for(int64_t rank=0;rank<tp;++rank)for(int64_t token=0;token<rows;++token)for(int64_t local=0;local<heads;++local) {
            const int64_t input_head=(rank*rows+token)*heads+local,output_head=token*(tp*heads)+rank*heads+local;
            const auto* source=static_cast<const uint8_t*>(buffers[query].data())+size_t(input_head*dim)*bytes;
            auto* output=static_cast<uint8_t*>(result[query_out].data())+size_t(output_head*dim)*bytes;
            std::memcpy(output,source,size_t(dim)*bytes);
            std::memcpy(static_cast<uint8_t*>(result[weights_out].data())+output_head*4,
                        static_cast<const uint8_t*>(buffers[weights].data())+input_head*4,4);
        }
        return result;
    }
};
} // namespace rad::ds41::reorder_heads::testing
