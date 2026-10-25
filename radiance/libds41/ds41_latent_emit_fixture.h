/* Synthetic cache pages in a permuted physical order. Raw bits, no weights or GPU state. */
#pragma once
#include "ds41_latent_emit_emulate.h"
#include <algorithm>
#include <array>
#include <random>
#include <stdexcept>
#include <vector>
namespace rad::ds41::latent_emit::testing {
struct Buffer {
    RadTensor t{};std::vector<uint64_t> words;int64_t allocation_bytes=0;
    Buffer()=default;
    Buffer(uint32_t dtype,std::initializer_list<int64_t> shape,int pitch) {
        t.dtype=dtype;t.rank=shape.size();unsigned axis=0;for(auto n:shape)t.shape[axis++]=n;
        int64_t stride=pitch==2?2:1;
        for(int i=int(t.rank)-1;i>=0;--i){t.stride[i]=stride;stride=stride*t.shape[i]+(pitch?3:0);}
        allocation_bytes=stride*(rad_dtype_bits(dtype)/8);words.resize(size_t((allocation_bytes+7)/8)+4,0xA5A5A5A5A5A5A5A5ull);refresh();
    }
    void refresh(){t.data=words.data()+2;}
    void* data(){return words.data()+2;}const void* data()const{return words.data()+2;}
    int64_t offset(int64_t flat)const{return math::offset(t,flat);}
    int64_t integer(int64_t flat)const {
        const auto at=offset(flat);return t.dtype==RAD_I32?int64_t(static_cast<const int32_t*>(data())[at]):static_cast<const int64_t*>(data())[at];
    }
    void integer(int64_t flat,int64_t value) {
        const auto at=offset(flat);if(t.dtype==RAD_I32)static_cast<int32_t*>(data())[at]=int32_t(value);else static_cast<int64_t*>(data())[at]=value;
    }
};
struct Fixture {
    uint32_t dtype,position_dtype,slot_dtype;int pitch;int64_t dimension,ratio,rows,start,blocks=3,block=4;
    std::array<Buffer,operand_count> buffers;std::array<RadTensor,operand_count> tensors{};
    std::vector<RadParam> params;std::array<int32_t,4> flag{911,912,913,914};
    Fixture(uint32_t dt=RAD_F32,int64_t d=128,int64_t r=2,int spacing=0,uint32_t pd=RAD_I32,uint32_t sd=RAD_I32,int64_t n=7,int64_t first=0)
        :dtype(dt),position_dtype(pd),slot_dtype(sd),pitch(spacing),dimension(d),ratio(r),rows(n),start(first),
         buffers{Buffer(dt,{rows,dimension},pitch),Buffer(pd,{rows},pitch),Buffer(sd,{rows},pitch),Buffer(RAD_I32,{rows},pitch),
                 Buffer(dt,{blocks,1,block,dimension/ratio},0),Buffer(RAD_I32,{rows},pitch)} {
        params={RAD_INT("M",rows+2),RAD_INT("dimension",dimension),RAD_INT("ratio",ratio),RAD_INT("block_size",block),RAD_INT("n_blocks",blocks),RAD_STR("dtype",rad_dtype_name(dtype))};
        std::mt19937 random(411017+unsigned(dtype+dimension+ratio));auto* raw=static_cast<uint8_t*>(buffers[x].data());
        for(int64_t i=0;i<buffers[x].allocation_bytes;++i)raw[i]=uint8_t(random());
        for(int64_t row=0;row<rows;++row) {
            const int64_t position=start+row;const bool active=position%ratio==ratio-1;
            buffers[emit].integer(row,active?1:0);buffers[positions].integer(row,position);
            const int64_t physical=(2-position/block+blocks)%blocks;
            buffers[slot_mapping].integer(row,physical*block+position%block);
            buffers[status].integer(row,0x12345678);
            if(!active){buffers[positions].integer(row,-123);buffers[slot_mapping].integer(row,INT32_MIN);}
        }
        constexpr uint32_t special32[]{0,0x80000000u,0x7F800001u,0x7FC12345u,1,0x7FFFFFu,0x7F800000u,0xFF800000u};
        constexpr uint16_t special16[]{0,0x8000,0x7C01,0x7E55,1,0x3FF,0x7C00,0xFC00};
        constexpr uint16_t specialbf[]{0,0x8000,0x7F81,0x7FC5,1,0x7F,0x7F80,0xFF80};
        const int width=rad_dtype_bits(dtype)/8;
        for(int64_t row=0;row<rows;++row)for(int64_t col=0;col<std::min<int64_t>(dimension,8);++col) {
            auto* out=raw+buffers[x].offset(row*dimension+col)*width;
            if(width==4){const uint32_t bits=special32[size_t(col)];std::memcpy(out,&bits,4);}
            else{const uint16_t bits=(dtype==RAD_F16?special16:specialbf)[size_t(col)];std::memcpy(out,&bits,2);}
        }
        refresh();
    }
    void refresh(){for(int i=0;i<operand_count;++i){buffers[size_t(i)].refresh();tensors[size_t(i)]=buffers[size_t(i)].t;}}
    RadArgs args(){refresh();RadArgs a{};a.t=tensors.data();a.n_t=operand_count;a.p=params.data();a.n_p=params.size();a.scratch=flag.data();a.scratch_bytes=4;return a;}
    void integer(const char* key,int64_t value){for(auto& p:params)if(!std::strcmp(p.key,key))p.ival=value;}
    int run(){auto a=args();return emulate(&a);}
    std::array<Buffer,operand_count> expected(int code=RAD_OK)const {
        auto result=buffers;for(auto& b:result)b.refresh();
        for(int64_t row=0;row<rows;++row)result[status].integer(row,code);
        if(code!=RAD_OK)return result;
        deepseek41::state::Layout layout;
        if(deepseek41::state::make_layout(dimension,ratio,block,rad_dtype_bits(dtype)/8,&layout)!=RAD_OK)throw std::runtime_error("bad emit fixture layout");
        for(int64_t row=0;row<rows;++row)if(buffers[emit].integer(row)==1) {
            const int64_t slot=buffers[slot_mapping].integer(row),page=slot/block,entry=(slot%block)/ratio;
            for(int64_t d=0;d<dimension;++d) {
                int64_t byte=0;if(deepseek41::state::physical_byte(layout,page,entry,d,&byte)!=RAD_OK)throw std::runtime_error("bad independent physical address");
                const auto* source=static_cast<const uint8_t*>(buffers[x].data())+buffers[x].offset(row*dimension+d)*layout.element_bytes;
                std::memcpy(static_cast<uint8_t*>(result[cache].data())+byte,source,size_t(layout.element_bytes));
            }
        }
        return result;
    }
};
} // namespace rad::ds41::latent_emit::testing
