/* Tiny synthetic fixtures shared by CPU tests and the explicitly gated checker. */
#pragma once
#include "ds41_engram_gate_emulate.h"
#include "../arch/deepseek41/engram_ref.h"
#include <array>
#include <vector>
namespace rad::ds41::engram_gate::testing {
struct Buffer {
    RadTensor t{};std::vector<uint32_t> words;
    Buffer()=default;
    Buffer(uint32_t dtype,std::initializer_list<int64_t> shape,int pitch) {
        t.dtype=dtype;t.rank=shape.size();unsigned axis=0;for(auto d:shape)t.shape[axis++]=d;
        int64_t stride=pitch==2?2:1;
        for(int d=int(t.rank)-1;d>=0;--d) {t.stride[d]=stride;stride=stride*t.shape[d]+(pitch?3:0);}
        words.resize(size_t(stride+8),0xD5D5D5D5u);t.data=words.data();
    }
    int64_t count()const{return t.rank?math::count(t):0;}
    int64_t offset(int64_t i)const{return math::offset(t,i);}
    float get(int64_t i)const{return rad_load_f32(words.data(),t.dtype,offset(i));}
    void set(int64_t i,float value){rad_store_f32(words.data(),t.dtype,offset(i),value);}
    int32_t id(int64_t i)const{return reinterpret_cast<const int32_t*>(words.data())[offset(i)];}
    void id(int64_t i,int32_t value){reinterpret_cast<int32_t*>(words.data())[offset(i)]=value;}
    std::vector<float> values()const {
        std::vector<float> out;
        for(int64_t i=0;i<count();++i)out.push_back(get(i));
        return out;
    }
};
struct Fixture {
    uint32_t dtype;int pitch;int64_t tokens,copies,dim;bool mask,gates;
    std::array<Buffer,operand_count> buffers;std::array<RadTensor,operand_count> tensors{};
    std::array<int32_t,4> flag{911,912,913,914};std::vector<RadParam> params;
    Fixture(uint32_t type=RAD_F32,int spacing=0,bool with_mask=true,bool with_gates=true,
            int64_t n=7,int64_t hc=2,int64_t rows=3)
        :dtype(type),pitch(spacing),tokens(rows),copies(hc),dim(n),mask(with_mask),gates(with_gates) {
        params={RAD_INT("M",tokens+1),RAD_INT("copies",copies),RAD_INT("n",dim),RAD_STR("dtype",rad_dtype_name(dtype)),
                RAD_F64("eps",1e-6),RAD_F64("clamp",1e-6),RAD_INT("token_mask",mask?1:0),RAD_INT("gates",gates?1:0)};
        buffers[input]=Buffer(dtype,{tokens,copies,dim},pitch);
        buffers[projected_kv]=Buffer(dtype,{tokens,copies+1,dim},pitch);
        buffers[q_weight]=Buffer(RAD_F32,{copies,dim},pitch);
        buffers[k_weight]=Buffer(RAD_F32,{copies,dim},pitch);
        if(mask)buffers[mask_operand]=Buffer(RAD_I32,{tokens},pitch);
        buffers[output]=Buffer(dtype,{tokens,copies,dim},pitch);
        if(gates)buffers[gate_output]=Buffer(RAD_F32,{tokens,copies},pitch);
        for(int64_t i=0;i<buffers[input].count();++i)buffers[input].set(i,float(int(i*17%101)-50)/32.f);
        for(int64_t i=0;i<buffers[projected_kv].count();++i)buffers[projected_kv].set(i,float(int(i*13%113)-56)/32.f);
        for(int64_t i=0;i<copies*dim;++i) {
            buffers[q_weight].set(i,1.f+float(int(i%7)-3)/16.f);
            buffers[k_weight].set(i,(i%3==0?-1.f:1.f)*(1.f+float(int(i%5)-2)/32.f));
        }
        if(mask)for(int64_t row=0;row<tokens;++row)buffers[mask_operand].id(row,row%3==1?0:1);
        refresh();
    }
    void refresh() {
        for(int i=0;i<operand_count;++i) {
            buffers[i].t.data=buffers[i].t.rank?buffers[i].words.data():nullptr;tensors[i]=buffers[i].t;
        }
    }
    RadArgs args() {
        refresh();RadArgs a{};a.t=tensors.data();a.n_t=operand_count;a.p=params.data();a.n_p=params.size();
        a.scratch=flag.data();a.scratch_bytes=scratch_bytes;return a;
    }
    int run(){auto a=args();return emulate(&a);}
};
/* Independent earlier CPU reference, already compared to official-forward NumPy
 * goldens. Output storage cast is applied once by Buffer::set. Clamp1e-6 only. */
inline int reference(Fixture& fixture) {
    namespace ref=rad::deepseek41::engram;
    const auto x=fixture.buffers[input].values(),kv=fixture.buffers[projected_kv].values();
    const auto q=fixture.buffers[q_weight].values(),k=fixture.buffers[k_weight].values();
    std::vector<uint8_t> mask;
    if(fixture.mask)for(int64_t row=0;row<fixture.tokens;++row)mask.push_back(uint8_t(fixture.buffers[mask_operand].id(row)));
    std::vector<float> out(x.size()),gates(size_t(fixture.tokens*fixture.copies));
    const int rc=ref::gate_residual(fixture.tokens,fixture.copies,fixture.dim,x,kv,q,k,mask,1e-6f,false,out,gates);
    if(rc!=RAD_OK)return rc;
    for(size_t i=0;i<out.size();++i)fixture.buffers[output].set(int64_t(i),out[i]);
    if(fixture.gates)for(size_t i=0;i<gates.size();++i)fixture.buffers[gate_output].set(int64_t(i),gates[i]);
    return RAD_OK;
}
} // namespace rad::ds41::engram_gate::testing
