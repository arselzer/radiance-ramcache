/* Bounded HOST/checker fixture. Numeric oracle composes independent reference cast/RMS/RoPE;
 * it does not call compress_finish's input, normalization or output routines. */
#pragma once
#include "ds41_compress_finish_emulate.h"
#include "ds41_transform_fixture.h"
#include "../arch/common/rad_ds41_ref.h"
#include <limits>
namespace rad::ds41::compress_finish::testing {
using Buffer=transform::testing::Buffer;
namespace ref=rad::deepseek41::ref;
inline float cast(float x,uint32_t dtype){uint64_t storage=0;rad_store_f32(&storage,dtype,0,x);return rad_load_f32(&storage,dtype,0);}
struct Fixture {
    uint32_t input_dtype,dtype;int64_t rows,dim,rotary,ratio,positions=64;
    transform::RopeSpec spec;std::vector<Buffer> buffers;std::vector<RadTensor> tensors;std::vector<RadParam> params;
    std::array<int32_t,4> flag{911,912,913,914};
    Fixture(uint32_t ad=RAD_F32,uint32_t yd=RAD_BF16,int64_t width=128,int64_t grouping=2,int pitch=0,int64_t count=7)
        :input_dtype(ad),dtype(yd),rows(count),dim(width),rotary(width>=64?64:width>=4?4:2),ratio(grouping),spec{rotary,4096,40000.f,40.f,32.f,1.f}{
        params={RAD_INT("M",rows+2),RAD_INT("n",dim),RAD_INT("rotary_dim",rotary),RAD_INT("ratio",ratio),RAD_STR("a_dtype",rad_dtype_name(ad)),RAD_STR("dtype",rad_dtype_name(yd)),RAD_F64("eps",1e-6)};
        buffers.emplace_back(ad,std::initializer_list<int64_t>{rows,dim},pitch);buffers.emplace_back(RAD_I32,std::initializer_list<int64_t>{rows},pitch);
        buffers.emplace_back(RAD_I32,std::initializer_list<int64_t>{rows},pitch);buffers.emplace_back(RAD_F32,std::initializer_list<int64_t>{dim},pitch);
        buffers.emplace_back(RAD_F32,std::initializer_list<int64_t>{positions,rotary/2,2},pitch);
        buffers.emplace_back(yd,std::initializer_list<int64_t>{rows,dim},pitch);buffers.emplace_back(yd,std::initializer_list<int64_t>{rows,dim},pitch);
        for(int64_t row=0;row<rows;++row){const bool active=ratio==1||row%2==1;buffers[emit].id(row,active?1:0);buffers[first_positions].id(row,active?int32_t(ratio==1?row:row-1):INT32_MIN);
            for(int64_t d=0;d<dim;++d)buffers[pooled].set(row*dim+d,active?float(int((row*dim+d)*17%211)-105)/64.f+0.0001f:std::numeric_limits<float>::quiet_NaN());}
        for(int64_t d=0;d<dim;++d)buffers[gain].set(d,1.f+float(int(d%7)-3)/32.f);
        std::vector<float> table;if(transform::build_pairs(spec,positions,uint64_t(positions*rotary*4),&table)!=RAD_OK)throw std::runtime_error("finish fixture pair table refused");
        for(size_t i=0;i<table.size();++i)buffers[pairs].set(int64_t(i),table[i]);
        refresh();
    }
    void refresh(){tensors.clear();for(auto& b:buffers){b.t.data=b.words.data();tensors.push_back(b.t);}}
    RadArgs args(){refresh();RadArgs a{};a.t=tensors.data();a.n_t=operand_count;a.p=params.data();a.n_p=int(params.size());a.scratch=flag.data();a.scratch_bytes=scratch_bytes;return a;}
    int run(){auto a=args();return emulate(&a);}
    void integer(const char* key,int64_t value){for(auto& p:params)if(!std::strcmp(p.key,key))p.ival=value;}
    void floating(const char* key,double value){for(auto& p:params)if(!std::strcmp(p.key,key))p.dval=value;}
    void text(const char* key,const char* value){for(auto& p:params)if(!std::strcmp(p.key,key))p.sval=value;}
    std::vector<Buffer> expected()const {
        auto output=buffers;const auto weights=buffers[gain].values();const ref::Rope rope{size_t(rotary),spec.theta,spec.factor,spec.beta_fast,spec.beta_slow,spec.original_context};
        for(int64_t row=0;row<rows;++row){
            if(!buffers[emit].id(row)){for(int64_t d=0;d<dim;++d){output[unrotated].set(row*dim+d,0.f);output[rotated].set(row*dim+d,0.f);}continue;}
            std::vector<float> input_row(size_t(dim),0),normalized_row(size_t(dim),0);
            for(int64_t d=0;d<dim;++d)input_row[size_t(d)]=cast(buffers[pooled].get(row*dim+d),dtype);
            const float eps=float(rad_param_getf(params.data(),int(params.size()),"eps",0));
            if(ref::rms_norm(input_row,weights,eps,normalized_row)!=RAD_OK)throw std::runtime_error("independent finish RMS refused");
            for(int64_t d=0;d<dim;++d){normalized_row[size_t(d)]=cast(normalized_row[size_t(d)],dtype);output[unrotated].set(row*dim+d,normalized_row[size_t(d)]);}
            if(ref::rotary(normalized_row,buffers[first_positions].id(row),rope)!=RAD_OK)throw std::runtime_error("independent finish RoPE refused");
            for(int64_t d=0;d<dim;++d)output[rotated].set(row*dim+d,normalized_row[size_t(d)]);
        }
        for(auto& b:output)b.t.data=b.words.data();
        return output;
    }
};
} // namespace rad::ds41::compress_finish::testing
