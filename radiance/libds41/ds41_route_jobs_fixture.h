/* Small synthetic inputs shared only by the CPU tests and explicit GPU checker. */
#pragma once
#include "ds41_route_jobs_emulate.h"
#include <array>
#include <cstring>
#include <vector>
namespace rad::ds41::route_jobs::testing {
struct Buffer {
    RadTensor t{};std::vector<uint32_t> words;
    Buffer(uint32_t dtype,std::initializer_list<int64_t> shape,int pitch){
        t.dtype=dtype;t.rank=shape.size();unsigned at=0;for(auto d:shape)t.shape[at++]=d;
        int64_t stride=pitch==2?2:1;
        for(int d=int(t.rank)-1;d>=0;--d){t.stride[d]=stride;stride=stride*t.shape[d]+(pitch?3:0);}
        words.resize(size_t(stride+8),0xD5D5D5D5u);t.data=words.data();
    }
    int64_t count()const{return math::count(t);}
    int64_t offset(int64_t i)const{return math::offset(t,i);}
    int32_t id(int64_t i)const{return static_cast<int32_t>(words[size_t(offset(i))]);}
    void id(int64_t i,int32_t value){words[size_t(offset(i))]=uint32_t(value);}
    float weight(int64_t i)const{return packed::from_bits(words[size_t(offset(i))]);}
    void weight(int64_t i,float value){words[size_t(offset(i))]=packed::float_bits(value);}
};
struct Fixture {
    int64_t rows,topk,experts,jobs;int pitch;
    std::vector<Buffer> buffers;std::array<RadTensor,operand_count> tensors{};std::array<int32_t,4> flag{911,912,913,914};
    std::vector<RadParam> params;
    explicit Fixture(int spacing=0,int64_t m=3,int64_t k=2,int64_t e=5):rows(m),topk(k),experts(e),jobs(m*k),pitch(spacing){
        params={RAD_INT("M",rows+1),RAD_INT("topk",topk),RAD_INT("n_expert",experts)};
        buffers.emplace_back(RAD_I32,std::initializer_list<int64_t>{rows,topk},pitch);
        buffers.emplace_back(RAD_F32,std::initializer_list<int64_t>{rows,topk},pitch);
        for(int i=expert_ids;i<operand_count;++i)buffers.emplace_back(i==job_weights?RAD_F32:RAD_I32,std::initializer_list<int64_t>{i==expert_counts?experts:jobs},pitch);
        for(int64_t row=0;row<rows;++row)for(int64_t choice=0;choice<topk;++choice){const int64_t job=row*topk+choice;
            buffers[ids].id(job,int32_t((row*3+choice)%experts));buffers[weights].weight(job,float(job+1)/8.f);}
        refresh();
    }
    void refresh(){for(int i=0;i<operand_count;++i){buffers[i].t.data=buffers[i].words.data();tensors[i]=buffers[i].t;}}
    RadArgs args(){refresh();RadArgs a{};a.t=tensors.data();a.n_t=operand_count;a.p=params.data();a.n_p=params.size();a.scratch=flag.data();a.scratch_bytes=scratch_bytes;return a;}
    int run(){auto a=args();return emulate(&a);}
};
} // namespace rad::ds41::route_jobs::testing
