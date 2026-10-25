/* Synthetic fixtures shared only by compressor CPU tests and the explicitly gated checker.
 * No model payloads, device calls, architecture registration or serving state allocation. */
#pragma once
#include "ds41_compress_emulate.h"
#include <array>
#include <optional>
#include <stdexcept>
#include <vector>
namespace rad::ds41::compress::testing {
struct Buffer {
    RadTensor t{};std::vector<uint32_t> words;
    Buffer(uint32_t dtype,std::initializer_list<int64_t> shape,int pitch) {
        t.dtype=dtype;t.rank=shape.size();unsigned axis=0;for(auto d:shape)t.shape[axis++]=d;
        int64_t stride=pitch==2?2:1;
        for(int i=int(t.rank)-1;i>=0;--i) {t.stride[i]=stride;stride=stride*t.shape[i]+(pitch?3:0);}
        words.resize(size_t(stride+8),0xA5A5A5A5u);t.data=words.data();
    }
    int64_t count() const {int64_t n=1;for(unsigned i=0;i<t.rank;++i)n*=t.shape[i];return n;}
    int64_t offset(int64_t flat) const {
        int64_t n=0;for(int i=int(t.rank)-1;i>=0;--i) {n+=(flat%t.shape[i])*t.stride[i];flat/=t.shape[i];}return n;
    }
    float getf(int64_t i) const {return rad_load_f32(words.data(),t.dtype,offset(i));}
    void setf(int64_t i,float x) {rad_store_f32(words.data(),t.dtype,offset(i),x);}
    int32_t geti(int64_t i) const {return reinterpret_cast<const int32_t*>(words.data())[offset(i)];}
    void seti(int64_t i,int32_t x) {reinterpret_cast<int32_t*>(words.data())[offset(i)]=x;}
    uint32_t getu(int64_t i) const {return words[size_t(offset(i))];}
    void setu(int64_t i,uint32_t x) {words[size_t(offset(i))]=x;}
};
struct Fixture {
    int64_t dimension,ratio,tokens=0,sequences,states=4;uint32_t dtype;int pitch;
    std::vector<int32_t> lengths,starts,slots;
    std::array<std::optional<Buffer>,tensors_count> buffers;
    std::array<RadTensor,tensors_count> tensors{};std::vector<RadParam> params;
    std::array<int32_t,4> flag{911,912,913,914};
    Fixture(int64_t d=5,int64_t r=2,uint32_t dt=RAD_F32,int p=0,
            std::vector<int32_t> sizes={1,3,0},std::vector<int32_t> firsts={0,0,0},
            std::vector<int32_t> ids={2,0,-1},bool accepted=true)
        :dimension(d),ratio(r),sequences(sizes.size()),dtype(dt),pitch(p),
         lengths(std::move(sizes)),starts(std::move(firsts)),slots(std::move(ids)) {
        if(sequences<=0||starts.size()!=size_t(sequences)||slots.size()!=size_t(sequences))throw std::runtime_error("bad fixture sequences");
        for(auto n:lengths)tokens+=n;
        if(tokens<=0)throw std::runtime_error("empty fixture");
        params={RAD_INT("M",tokens+3),RAD_INT("dim",dimension),RAD_INT("ratio",ratio),
                RAD_STR("dtype",rad_dtype_name(dtype)),RAD_INT("n_spec",0)};
        add(projected_kv,dtype,{tokens,dimension});if(ratio==2)add(gate_scores,RAD_F32,{tokens,dimension});
        add(pending_state,RAD_U32,{states,1,2,dimension+1});add(state_ids,RAD_I32,{sequences,1});
        add(has_init,RAD_I32,{sequences});add(cu,RAD_I32,{sequences+1});add(positions,RAD_I32,{tokens});
        if(accepted)add(num_accepted,RAD_I32,{sequences});
        add(pooled,RAD_F32,{tokens,dimension});add(emit,RAD_I32,{tokens});add(first_position,RAD_I32,{tokens});
        int64_t row=0;buffer(cu).seti(0,0);
        for(int64_t s=0;s<sequences;++s) {
            buffer(state_ids).seti(s,slots[size_t(s)]);buffer(has_init).seti(s,starts[size_t(s)]);
            if(accepted)buffer(num_accepted).seti(s,lengths[size_t(s)]?1:0);
            for(int32_t j=0;j<lengths[size_t(s)];++j,++row) {
                const int64_t pos=int64_t(starts[size_t(s)])+j;buffer(positions).seti(row,int32_t(pos));
                for(int64_t col=0;col<dimension;++col) {
                    buffer(projected_kv).setf(row*dimension+col,kv_value(slots[size_t(s)],pos,col));
                    if(ratio==2)buffer(gate_scores).setf(row*dimension+col,score_value(slots[size_t(s)],pos,col));
                }
            }
            buffer(cu).seti(s+1,int32_t(row));
        }
        refresh();
    }
    static float kv_value(int64_t slot,int64_t pos,int64_t d) {return float(((slot+3)*37+(pos+1)*13+d*17)%101-50)/32.f;}
    static float score_value(int64_t slot,int64_t pos,int64_t d) {return float(((slot+7)*19+(pos+1)*29+d*7)%151-75)/16.f;}
    void add(int operand,uint32_t dt,std::initializer_list<int64_t> shape) {buffers[size_t(operand)].emplace(dt,shape,pitch);}
    Buffer& buffer(int operand) {return *buffers[size_t(operand)];}
    const Buffer& buffer(int operand) const {return *buffers[size_t(operand)];}
    int64_t state_flat(int64_t slot,int64_t plane,int64_t column) const {return (slot*2+plane)*(dimension+1)+column;}
    void state_set(int64_t slot,int64_t plane,int64_t column,uint32_t value) {buffer(pending_state).setu(state_flat(slot,plane,column),value);}
    uint32_t state_get(int64_t slot,int64_t plane,int64_t column) const {return buffer(pending_state).getu(state_flat(slot,plane,column));}
    void seed(int64_t slot,int32_t next,bool old_pending_finite=true) {
        state_set(slot,0,dimension,uint32_t(next));state_set(slot,1,dimension,magic(ratio));
        for(int64_t d=0;d<dimension;++d) {
            state_set(slot,0,d,old_pending_finite?packed::float_bits(kv_value(slot,int64_t(next)-1,d)):0x7FC01234u);
            state_set(slot,1,d,old_pending_finite?packed::float_bits(score_value(slot,int64_t(next)-1,d)):0xFF800000u);
        }
    }
    void refresh() {
        for(int i=0;i<tensors_count;++i) {
            if(buffers[size_t(i)]) {auto& b=buffer(i);b.t.data=b.words.data();tensors[size_t(i)]=b.t;}
            else tensors[size_t(i)]=RadTensor{};
        }
    }
    RadArgs args() {
        refresh();RadArgs a{};a.t=tensors.data();a.n_t=tensors_count;a.p=params.data();a.n_p=params.size();
        a.scratch=flag.data();a.scratch_bytes=scratch_bytes;return a;
    }
    int run() {auto a=args();return emulate(&a);}
};
} // namespace rad::ds41::compress::testing
