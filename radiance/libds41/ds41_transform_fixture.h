/* Small synthetic test/checker fixtures, no production buffers or automatic context tables. */
#pragma once
#include <cstring>
#include "ds41_transform_emulate.h"
#include "ds41_transform_table.h"
#include <stdexcept>
namespace rad::ds41::transform::testing {
struct Buffer {
    RadTensor t{};std::vector<uint32_t> words;
    Buffer(uint32_t dtype,std::initializer_list<int64_t> shape,int pitch) {
        t.dtype=dtype;t.rank=shape.size();unsigned i=0;for(auto d:shape)t.shape[i++]=d;
        int64_t stride=pitch==2?2:1;
        for(int d=int(t.rank)-1;d>=0;--d) {t.stride[d]=stride;stride=stride*t.shape[d]+(pitch?3:0);}
        words.resize(size_t(stride+8),0xD5D5D5D5u);t.data=words.data();
    }
    int64_t count() const {int64_t n=1;for(unsigned i=0;i<t.rank;++i)n*=t.shape[i];return n;}
    int64_t offset(int64_t flat) const {
        int64_t at=0;for(int d=int(t.rank)-1;d>=0;--d) {at+=(flat%t.shape[d])*t.stride[d];flat/=t.shape[d];}return at;
    }
    float get(int64_t i) const {return rad_load_f32(words.data(),t.dtype,offset(i));}
    void set(int64_t i,float x) {rad_store_f32(words.data(),t.dtype,offset(i),x);}
    int32_t id(int64_t i) const {return reinterpret_cast<const int32_t*>(words.data())[offset(i)];}
    void id(int64_t i,int32_t x) {reinterpret_cast<int32_t*>(words.data())[offset(i)]=x;}
    std::vector<float> values() const {std::vector<float> out;for(int64_t i=0;i<count();++i)out.push_back(get(i));return out;}
};
struct Fixture {
    Op op;uint32_t input_dtype,output_dtype;int pitch;int64_t rows=3,dim,heads,rotary,table_positions=64;
    RopeSpec spec;TableReceipt receipt{};std::vector<Buffer> buffers;std::vector<RadTensor> tensors;std::vector<RadParam> params;
    std::array<int32_t,4> flag{911,912,913,914};
    Fixture(Op kind=Op::rmsnorm,uint32_t ad=RAD_F32,uint32_t yd=RAD_F32,int p=0,
            bool inverse=false,int64_t width=0,int64_t rot=4,int64_t head_count=2)
        :op(kind),input_dtype(ad),output_dtype(kind==Op::rmsnorm?yd:ad),pitch(p),dim(width?width:kind==Op::rmsnorm?7:9),heads(head_count),rotary(rot),
         spec{rot,4096,40000.f,40.f,32.f,1.f} {
        if(op==Op::rmsnorm) {
            params={RAD_INT("M",5),RAD_INT("n",dim),RAD_F64("eps",1e-6),RAD_STR("a_dtype",rad_dtype_name(ad)),RAD_STR("y_dtype",rad_dtype_name(yd))};
            add(ad,{rows,dim});add(RAD_F32,{dim});add(yd,{rows,dim});
            for(int64_t i=0;i<buffers[0].count();++i)buffers[0].set(i,float(int(i*17%211)-105)/64.f);
            for(int64_t i=0;i<dim;++i)buffers[1].set(i,1.f+float(int(i%7)-3)/32.f);
        } else {
            params={RAD_INT("M",5),RAD_INT("heads",heads),RAD_INT("head_dim",dim),RAD_INT("rotary_dim",rotary),
                    RAD_STR("dtype",rad_dtype_name(ad)),RAD_INT("inverse",inverse?1:0)};
            add(ad,{rows,heads,dim});add(RAD_I32,{rows});add(RAD_F32,{table_positions,rotary/2,2});add(ad,{rows,heads,dim});
            for(int64_t i=0;i<buffers[0].count();++i)buffers[0].set(i,float(int(i*17%211)-105)/32.f);
            buffers[1].id(0,0);buffers[1].id(1,5);buffers[1].id(2,37);
            if(dim>rotary)for(int64_t r=0;r<rows;++r)for(int64_t h=0;h<heads;++h)buffers[0].set((r*heads+h)*dim,-0.f);
            rebuild();
        }
        refresh();
    }
    void add(uint32_t d,std::initializer_list<int64_t> shape) {buffers.emplace_back(d,shape,pitch);}
    void rebuild() {
        std::vector<float> pairs;const auto cap=uint64_t(table_positions*rotary*4);
        if(build_pairs(spec,table_positions,cap,&pairs,&receipt)!=RAD_OK)throw std::runtime_error("synthetic pair table construction failed");
        for(size_t i=0;i<pairs.size();++i)buffers[2].set(i,pairs[i]);
    }
    void refresh() {tensors.clear();for(auto& b:buffers) {b.t.data=b.words.data();tensors.push_back(b.t);}}
    RadArgs args() {
        refresh();RadArgs a{};a.t=tensors.data();a.n_t=tensors.size();a.p=params.data();a.n_p=params.size();
        a.scratch=flag.data();a.scratch_bytes=scratch_bytes;return a;
    }
    int run() {auto a=args();return emulate(op,&a);}
};
} // namespace rad::ds41::transform::testing
