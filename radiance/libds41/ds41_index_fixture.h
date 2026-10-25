/* Tiny host fixtures for CPU checks and the explicitly gated GPU checker only. */
#pragma once
#include "ds41_index_emulate.h"
#include <array>
#include <optional>
#include <random>
namespace rad::ds41::index::testing {
struct Buffer {
    RadTensor t{};std::vector<uint32_t> words;
    Buffer(uint32_t dtype,std::initializer_list<int64_t> dims,int pitch) {
        t.dtype=dtype;t.rank=dims.size();unsigned axis=0;for(auto d:dims)t.shape[axis++]=d;
        int64_t stride=pitch==2?2:1;
        for(int i=int(t.rank)-1;i>=0;--i){t.stride[i]=stride;stride=stride*t.shape[i]+(pitch?3:0);}
        words.resize(size_t((stride*4+31)/4),0xA5A5A5A5u);t.data=words.data();
    }
    int64_t count() const {return math::count(t);}
    int64_t at(int64_t i) const {return math::offset(t,i);}
    float get(int64_t i) const {return rad_load_f32(words.data(),t.dtype,at(i));}
    void set(int64_t i,float x){rad_store_f32(words.data(),t.dtype,at(i),x);}
    int32_t id(int64_t i)const{return reinterpret_cast<const int32_t*>(words.data())[at(i)];}
    void id(int64_t i,int32_t x){reinterpret_cast<int32_t*>(words.data())[at(i)]=x;}
    uint8_t byte(int64_t i)const{return reinterpret_cast<const uint8_t*>(words.data())[at(i)];}
    void byte(int64_t i,uint8_t x){reinterpret_cast<uint8_t*>(words.data())[at(i)]=x;}
    std::vector<float> values()const {std::vector<float> out;for(int64_t i=0;i<count();++i)out.push_back(get(i));return out;}
};
struct Fixture {
    Op op;int64_t rows,keys,heads=3,dim=6,sequences=3,topk,block=4;int pitch;
    std::array<std::optional<Buffer>,6> buffers;std::array<RadTensor,6> tensors{};
    std::vector<RadParam> params;std::array<int32_t,4> status{911,912,913,914};
    Fixture(Op kind=Op::scores,uint32_t dtype=RAD_F32,int spacing=0,int64_t row_count=3,int64_t key_count=11,int64_t desired=4,bool mask=true,
            int64_t dimension=6,int64_t local_heads=3)
        :op(kind),rows(row_count),keys(key_count),heads(local_heads),dim(dimension),topk(desired),pitch(spacing) {
        params={RAD_INT("M",rows+2),RAD_INT("key_rows",keys+3)};
        if(op==Op::scores) {
            params.insert(params.end(),{RAD_INT("heads",heads),RAD_INT("global_heads",heads*4),RAD_INT("head_dim",dim),
                RAD_INT("sequences",sequences+1),RAD_STR("dtype",rad_dtype_name(dtype))});
            add(0,dtype,{rows,heads,dim});add(1,dtype,{sequences,keys,dim});add(2,RAD_F32,{rows,heads});
            add(3,RAD_I32,{rows});add(4,RAD_I32,{rows});add(5,RAD_F32,{rows,keys});
        } else {
            add(0,RAD_F32,{rows,keys});add(1,RAD_I32,{rows});
            if(op==Op::candidates){params.push_back(RAD_INT("block_size",block));params.push_back(RAD_INT("top_blocks",topk));add(2,RAD_U8,{rows,keys});}
            else {params.push_back(RAD_INT("topk",topk));if(mask)add(2,RAD_U8,{rows,keys});add(3,RAD_I32,{rows,std::min(topk,keys)});}
        }
        std::mt19937 rng(531017+unsigned(op));
        for(int i=0;i<output();++i)if(buffers[i]) {
            auto& b=buffer(i);
            for(int64_t j=0;j<b.count();++j) {
                if(b.t.dtype==RAD_I32)b.id(j,0);
                else if(b.t.dtype==RAD_U8)b.byte(j,j%3?255:0);
                else b.set(j,float(int(rng()%101)-50)/32.f);
            }
        }
        for(int64_t row=0;row<rows;++row) {
            visible().id(row,row%3==0?0:row%3==1?int32_t(keys/2):int32_t(keys));
            if(op==Op::scores)buffer(3).id(row,row%3==0?2:row%3==1?0:2);
        }
        refresh();
    }
    int nt()const{return op==Op::scores?6:op==Op::candidates?3:4;}
    int output()const{return nt()-1;}
    void add(int i,uint32_t dtype,std::initializer_list<int64_t> shape){buffers[i].emplace(dtype,shape,pitch);}
    Buffer& buffer(int i){return *buffers[i];}
    const Buffer& buffer(int i)const{return *buffers[i];}
    Buffer& visible(){return buffer(op==Op::scores?4:1);}
    void refresh(){for(int i=0;i<nt();++i)if(buffers[i]){buffer(i).t.data=buffer(i).words.data();tensors[i]=buffer(i).t;}else tensors[i]={};}
    RadArgs args(){refresh();RadArgs a{};a.t=tensors.data();a.n_t=nt();a.p=params.data();a.n_p=params.size();a.scratch=status.data();a.scratch_bytes=4;return a;}
    int run(){auto a=args();return emulate(op,&a);}
};
} // namespace rad::ds41::index::testing
