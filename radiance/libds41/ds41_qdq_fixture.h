/* Tiny pitched BF16 inputs; reference outputs use the independent interpreted
 * CPU row candidate backed by actual Torch cast/codebook fixtures. */
#pragma once
#include "ds41_qdq_emulate.h"
#include <array>
#include <stdexcept>
#include <vector>
namespace rad::ds41::qdq::testing {
struct Buffer {
    RadTensor t{};std::vector<uint64_t> words;int64_t elements=0;
    Buffer()=default;
    Buffer(uint32_t type,std::initializer_list<int64_t> shape,int pitch){
        t.dtype=type;t.rank=unsigned(shape.size());unsigned axis=0;for(auto n:shape)t.shape[axis++]=n;
        int64_t stride=pitch==2?2:1;elements=1;
        for(int d=int(t.rank)-1;d>=0;--d){t.stride[d]=stride;stride=stride*t.shape[d]+(pitch?3:0);elements*=t.shape[d];}
        const int64_t bytes=stride*(rad_dtype_bits(type)/8);words.assign(size_t((bytes+7)/8)+4,0xBDBDBDBDBDBDBDBDull);refresh();
    }
    void refresh(){t.data=t.rank?static_cast<void*>(words.data()+2):nullptr;}
    void* data(){return words.data()+2;}const void* data()const{return words.data()+2;}
    int64_t at(int64_t i)const{return math::offset(t,i);}
    uint16_t raw(int64_t i)const{return static_cast<const uint16_t*>(data())[at(i)];}
    void raw(int64_t i,uint16_t value){static_cast<uint16_t*>(data())[at(i)]=value;}
    int32_t id(int64_t i)const{return static_cast<const int32_t*>(data())[at(i)];}
    void id(int64_t i,int32_t value){static_cast<int32_t*>(data())[at(i)]=value;}
};
inline const char* name(qdq_math::Profile profile){
    return profile==qdq_math::Profile::window_fp8?"window_fp8":profile==qdq_math::Profile::compressed_fp4?"compressed_fp4":"index_fp4";
}
struct Fixture {
    qdq_math::Profile profile;int64_t rows,dim;bool masked;
    std::array<Buffer,operand_count> buffers;std::array<RadTensor,operand_count> tensors{};
    std::vector<RadParam> params;std::array<int32_t,4> flag{911,912,913,914};
    Fixture(qdq_math::Profile kind=qdq_math::Profile::index_fp4,int64_t width=128,int pitch=0,bool mask=true,int64_t count=7)
        :profile(kind),rows(count),dim(width),masked(mask){
        buffers[input]=Buffer(RAD_BF16,{rows,dim},pitch);if(masked)buffers[emit]=Buffer(RAD_I32,{rows},pitch);
        buffers[output]=Buffer(RAD_BF16,{rows,dim},pitch);
        params={RAD_INT("M",rows+1),RAD_INT("n",dim),RAD_STR("profile",name(profile)),RAD_INT("masked",masked?1:0)};
        for(int64_t row=0;row<rows;++row){
            if(masked)buffers[emit].id(row,row%3==1?0:1);
            for(int64_t d=0;d<dim;++d){
                const float x=float(int((row*17+d*11)%83)-41)/64.f;uint16_t raw=0;
                if(rad::deepseek41::qdq::bf16_rne(x,&raw)!=RAD_OK)throw std::runtime_error("tiny BF16 QDQ fixture conversion");
                buffers[input].raw(row*dim+d,raw);
            }
        }
        // Isolate the index gradual-underflow case in its own all-tiny group.
        // The other profiles quantize these magnitudes to signed zero instead.
        for(int i=0;i<qdq_math::width(profile);++i)buffers[input].raw(i,i%2?0x8000:0);
        buffers[input].raw(0,0x0040);buffers[input].raw(1,0x8040);refresh();
    }
    void refresh(){for(int i=0;i<operand_count;++i){buffers[size_t(i)].refresh();tensors[size_t(i)]=buffers[size_t(i)].t;}}
    RadArgs args(){refresh();RadArgs a{};a.t=tensors.data();a.n_t=operand_count;a.p=params.data();a.n_p=int(params.size());a.scratch=flag.data();a.scratch_bytes=4;return a;}
    int run(){auto a=args();return emulate(&a);}
    void integer(const char* key,int64_t value){for(auto& p:params)if(!std::strcmp(p.key,key))p.ival=value;}
    int reference(std::array<Buffer,operand_count>* out)const{
        if(!out)return RAD_E_INVAL;
        auto result=buffers;for(auto& b:result)b.refresh();
        if(masked)for(int64_t row=0;row<rows;++row){const int bit=buffers[emit].id(row);if(bit!=0&&bit!=1)return RAD_E_INVAL;}
        for(int64_t row=0;row<rows;++row){
            if(masked&&!buffers[emit].id(row)){for(int64_t d=0;d<dim;++d)result[output].raw(row*dim+d,0);continue;}
            std::vector<uint16_t> x(size_t(dim),0),y(size_t(dim),0);
            for(int64_t d=0;d<dim;++d)x[size_t(d)]=buffers[input].raw(row*dim+d);
            const int rc=rad::deepseek41::qdq::row(rad::deepseek41::qdq::Profile(uint32_t(profile)),x,{y});if(rc!=RAD_OK)return rc;
            for(int64_t d=0;d<dim;++d)result[output].raw(row*dim+d,y[size_t(d)]);
        }
        *out=std::move(result);return RAD_OK;
    }
};
} // namespace rad::ds41::qdq::testing
