/* Small synthetic native blocks only. No model payloads; reused by CPU tests and gated checker. */
#pragma once
#include "ds41_embedding_emulate.h"
#include <algorithm>
#include <array>
#include <random>
#include <vector>
namespace rad::ds41::embedding::testing {
struct Buffer {
    RadTensor t{};std::vector<uint32_t> words;
    Buffer()=default;
    Buffer(uint32_t dtype,std::initializer_list<int64_t> shape,int pitch) {
        t.dtype=dtype;t.rank=shape.size();unsigned axis=0;for(auto n:shape)t.shape[axis++]=n;
        int64_t stride=pitch==2?2:1;
        for(int i=int(t.rank)-1;i>=0;--i){t.stride[i]=stride;stride=stride*t.shape[i]+(pitch?3:0);}
        // Interior and exterior canaries are compared byte-for-byte after every test.
        words.resize(size_t((stride*4+3)/4)+8,0xA5A5A5A5u);refresh();
    }
    void refresh(){t.data=words.data()+4;}
    int64_t at(int64_t flat)const{return math::offset(t,flat);}
    float get(int64_t flat)const{return rad_load_f32(words.data()+4,t.dtype,at(flat));}
    void set(int64_t flat,float x){rad_store_f32(words.data()+4,t.dtype,at(flat),x);}
    int32_t id(int64_t flat)const{return reinterpret_cast<const int32_t*>(words.data()+4)[at(flat)];}
    void id(int64_t flat,int32_t x){reinterpret_cast<int32_t*>(words.data()+4)[at(flat)]=x;}
};
struct Fixture {
    packed::Type type;uint32_t dtype;int pitch;int64_t rows,n,k,row_bytes;
    Buffer ids,y;std::vector<uint32_t> native;std::array<RadTensor,operand_count> tensors{};
    std::vector<RadParam> params;std::array<int32_t,4> flag{911,912,913,914};
    Fixture(packed::Type f=packed::Type::q8_0,uint32_t d=RAD_F32,int spacing=0,int64_t width=256,
            int64_t vocabulary=5,int64_t tokens=5)
        :type(f),dtype(d),pitch(spacing),rows(tokens),n(vocabulary),k(width),
         row_bytes(width/packed::block_values(f)*packed::block_bytes(f)),ids(RAD_I32,{rows},spacing),y(d,{rows,k},spacing) {
        native.resize(size_t((n*row_bytes+3)/4)+8,0xA6A6A6A6u);
        std::mt19937 random(51017+unsigned(type)+unsigned(k));auto* bytes=raw();
        for(int64_t i=0;i<n*row_bytes;++i)bytes[i]=uint8_t(random());
        for(int64_t i=0;i<n*row_bytes;i+=packed::block_bytes(type)) {
            const uint16_t scale=uint16_t(0x2000+(random()%32)*0x20);bytes[i]=uint8_t(scale);bytes[i+1]=uint8_t(scale>>8);
            if(type==packed::Type::q5_k){bytes[i+2]=0;bytes[i+3]=0x18;}
        }
        for(int64_t row=0;row<rows;++row)ids.id(row,int32_t(row%3==0?n-1:row%3==1?0:std::min<int64_t>(1,n-1)));
        params={RAD_INT("M",rows+2),RAD_INT("N",n),RAD_INT("K",k),RAD_STR("format",packed::type_name(type)),RAD_STR("y_dtype",rad_dtype_name(dtype))};
        refresh();
    }
    uint8_t* raw(){return reinterpret_cast<uint8_t*>(native.data()+4);}
    const uint8_t* raw()const{return reinterpret_cast<const uint8_t*>(native.data()+4);}
    void refresh() {
        ids.refresh();y.refresh();tensors[token_ids]=ids.t;tensors[output]=y.t;
        RadTensor w{};w.data=raw();w.dtype=packed::storage_dtype(type);w.rank=2;
        w.shape[0]=n;w.shape[1]=k;w.stride[0]=k;w.stride[1]=1;tensors[weight]=w;
    }
    RadArgs args(){refresh();RadArgs a{};a.t=tensors.data();a.n_t=operand_count;a.p=params.data();a.n_p=params.size();a.scratch=flag.data();a.scratch_bytes=4;return a;}
    void integer(const char* key,int64_t value){for(auto& p:params)if(!std::strcmp(p.key,key))p.ival=value;}
    void text(const char* key,const char* value){for(auto& p:params)if(!std::strcmp(p.key,key))p.sval=value;}
    int run(){auto a=args();return emulate(&a);}
};
} // namespace rad::ds41::embedding::testing
