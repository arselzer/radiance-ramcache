/* Synthetic test fixtures, not architecture state or a serving executor. */
#pragma once
#include <cstring>
#include "ds41_ffn_emulate.h"
#include <array>
#include <random>
namespace rad::ds41::ffn::testing {
struct Buffer {
    RadTensor t{};std::vector<uint32_t> words;
    Buffer(uint32_t dtype,std::initializer_list<int64_t> dims,int pitch) {
        t.dtype=dtype;t.rank=dims.size();unsigned i=0;for(auto d:dims)t.shape[i++]=d;
        int64_t stride=pitch==2?2:1;
        for(int d=int(t.rank)-1;d>=0;--d) {t.stride[d]=stride;stride=stride*t.shape[d]+(pitch?3:0);}
        words.resize(size_t(stride+8),0xD5D5D5D5u);t.data=words.data();
    }
    int64_t count() const {int64_t n=1;for(unsigned i=0;i<t.rank;++i)n*=t.shape[i];return n;}
    int64_t offset(int64_t flat) const {
        int64_t at=0;for(int d=int(t.rank)-1;d>=0;--d) {at+=(flat%t.shape[d])*t.stride[d];flat/=t.shape[d];}return at;
    }
    float getf(int64_t i) const {return rad_load_f32(words.data(),t.dtype,offset(i));}
    void setf(int64_t i,float x) {rad_store_f32(words.data(),t.dtype,offset(i),x);}
    int32_t geti(int64_t i) const {return reinterpret_cast<const int32_t*>(words.data())[offset(i)];}
    void seti(int64_t i,int32_t x) {reinterpret_cast<int32_t*>(words.data())[offset(i)]=x;}
    std::vector<float> values() const {std::vector<float> v;for(int64_t i=0;i<count();++i)v.push_back(getf(i));return v;}
};
struct Fixture {
    Op op;uint32_t dtype;int pitch;std::vector<Buffer> buffers;std::vector<RadTensor> tensors;std::vector<RadParam> params;
    std::array<int32_t,4> flag{911,912,913,914};
    Fixture(Op kind,uint32_t d=RAD_F32,int p=0):op(kind),dtype(d),pitch(p) {
        const auto* name=rad_dtype_name(dtype);
        if(op==Op::expert_act) {
            params={RAD_INT("M",6),RAD_INT("n",7),RAD_F64("limit",1.25),RAD_STR("dtype",name)};
            add(dtype,{3,7});add(dtype,{3,7});add(RAD_F32,{3});add(dtype,{3,7});
        } else if(op==Op::moe_combine) {
            params={RAD_INT("M",5),RAD_INT("R",9),RAD_INT("n",7),RAD_INT("n_expert",4),RAD_STR("dtype",name),RAD_INT("tp_world",1)};
            add(dtype,{6,7});add(RAD_I32,{6});add(RAD_I32,{6});add(dtype,{3,7});add(dtype,{3,7});
        } else {
            params={RAD_INT("M",4),RAD_INT("n",7),RAD_INT("hc",4),RAD_STR("dtype",name)};
            add(dtype,{2,7});add(dtype,{2,28});add(RAD_F32,{2,4});add(RAD_F32,{2,16});add(dtype,{2,28});
        }
        std::mt19937 rng(411017+unsigned(op));
        for(size_t b=0;b+1<buffers.size();++b)if(buffers[b].t.dtype!=RAD_I32)
            for(int64_t i=0;i<buffers[b].count();++i)buffers[b].setf(i,float(int(rng()%2001)-1000)/256.f);
        if(op==Op::expert_act) {buffers[2].setf(0,.314159f);buffers[2].setf(1,0.f);buffers[2].setf(2,1.125f);}
        if(op==Op::moe_combine) {
            constexpr int32_t experts[]{2,0,3,1,0,2},rows[]{0,0,1,0,2,2};
            for(int i=0;i<6;++i) {buffers[1].seti(i,experts[i]);buffers[2].seti(i,rows[i]);}
        }
        refresh();
    }
    void add(uint32_t d,std::initializer_list<int64_t> shape) {buffers.emplace_back(d,shape,pitch);}
    void refresh() {tensors.clear();for(auto& b:buffers) {b.t.data=b.words.data();tensors.push_back(b.t);}}
    RadArgs args() {
        refresh();RadArgs a{};a.t=tensors.data();a.n_t=tensors.size();a.p=params.data();a.n_p=params.size();
        a.scratch=flag.data();a.scratch_bytes=scratch_bytes;return a;
    }
    int run() {auto a=args();return emulate(op,&a);}
    void integer(const char* name,int64_t x) {for(auto& p:params)if(!std::strcmp(p.key,name))p.ival=x;}
    void real(const char* name,double x) {for(auto& p:params)if(!std::strcmp(p.key,name))p.dval=x;}
};
} // namespace rad::ds41::ffn::testing
