/* Tiny canonical keys and separate physical pages for CPU tests/explicit device checker only. */
#pragma once
#include "ds41_index_stream_emulate.h"
#include "ds41_index_fixture.h"
#include "../arch/deepseek41/state_layout.h"
#include <stdexcept>
namespace rad::ds41::index_stream::testing {
using Buffer=index::testing::Buffer;
inline const char* name(Mode mode){return mode==Mode::source?"source":mode==Mode::consumer?"consumer":"plain";}
struct Fixture {
    Mode mode;uint32_t dtype;int pitch;int64_t rows,keys,heads,dim,ratio,block=4,candidate_width,topk,top_blocks,sequences=3,columns;
    std::array<std::optional<Buffer>,operand_count> buffers;std::array<RadTensor,operand_count> tensors{};
    std::vector<RadParam> params;std::array<int32_t,4> flag{911,912,913,914};std::vector<float> canonical_keys;
    Fixture(Mode m=Mode::source,uint32_t dt=RAD_F32,int p=0,int64_t compression=2,int64_t key_count=11,
            int64_t dimension=8,int64_t local_heads=2,int64_t selected=4,int64_t blocks=3,int64_t width=8,int64_t row_count=5)
        :mode(m),dtype(dt),pitch(p),rows(row_count),keys(key_count),heads(local_heads),dim(dimension),ratio(compression),
         candidate_width(width),topk(selected),top_blocks(m==Mode::plain?0:blocks),columns((key_count*compression+block-1)/block){
        params={RAD_INT("M",rows+2),RAD_INT("key_rows",keys),RAD_INT("heads",heads),RAD_INT("global_heads",heads),
            RAD_INT("head_dim",dim),RAD_INT("sequences",sequences),RAD_STR("dtype",rad_dtype_name(dtype)),RAD_INT("ratio",ratio),
            RAD_INT("block_size",block),RAD_INT("candidate_block_size",candidate_width),RAD_INT("topk",topk),RAD_INT("top_blocks",top_blocks),RAD_STR("mode",name(mode))};
        add(query,dtype,{rows,heads,dim});add(cache,dtype,{sequences*columns+1,1,block,dim/ratio});add(head_weights,RAD_F32,{rows,heads});
        add(sequence_ids,RAD_I32,{rows});add(visible,RAD_I32,{rows});add(block_table,RAD_I32,{sequences,columns});add(origins,RAD_I32,{sequences});
        if(mode==Mode::consumer)add(candidates_in,RAD_I32,{rows,top_blocks});
        add(selected_out,RAD_I32,{rows,topk});
        if(mode==Mode::source)add(candidates_out,RAD_I32,{rows,top_blocks});
        for(int64_t i=0;i<buffer(cache).count();++i)buffer(cache).set(i,NAN);
        for(int64_t s=0;s<sequences;++s){buffer(origins).id(s,0);for(int64_t c=0;c<columns;++c)buffer(block_table).id(s*columns+c,int32_t(sequences*columns-(s*columns+c)));}
        canonical_keys.resize(size_t(sequences*keys*dim));
        for(int64_t s=0;s<sequences;++s)for(int64_t k=0;k<keys;++k)for(int64_t d=0;d<dim;++d)
            canonical_keys[size_t((s*keys+k)*dim+d)]=round(float(int((s+1)*19+k*17+d*7)%97-48)/32.f);
        rewrite_keys();
        for(int64_t r=0;r<rows;++r){const int pattern=int(r%5);buffer(sequence_ids).id(r,pattern==0||pattern==2?2:pattern==3?1:0);
            buffer(visible).id(r,pattern==0?0:pattern==1?int32_t(keys/3):pattern==2?int32_t(keys):pattern==3?int32_t(keys/2):1);
            for(int64_t h=0;h<heads;++h){buffer(head_weights).set(r*heads+h,.5f+float(h)/8.f);
                for(int64_t d=0;d<dim;++d)buffer(query).set((r*heads+h)*dim+d,float(int((r+1)*11+h*13+d*5)%89-44)/32.f);}}
        if(mode==Mode::consumer)default_candidates();
        refresh();
    }
    float round(float value)const{return dtype==RAD_F32?value:dtype==RAD_F16?rad_f16_to_f32(rad_f32_to_f16(value)):rad_bf16_to_f32(rad_f32_to_bf16(value));}
    void add(int i,uint32_t d,std::initializer_list<int64_t> shape){buffers[size_t(i)].emplace(d,shape,pitch);}
    Buffer& buffer(int i){return *buffers[size_t(i)];}const Buffer& buffer(int i)const{return *buffers[size_t(i)];}
    void integer(const char* key,int64_t value){for(auto& param:params)if(!std::strcmp(param.key,key))param.ival=value;}
    int64_t scalar_index(int64_t s,int64_t key,int64_t d)const{
        namespace state=rad::deepseek41::state;state::Layout l;state::Address a;
        if(state::make_layout(dim,ratio,block,rad_dtype_bits(dtype)/8,&l)!=RAD_OK||state::map(l,key,d,&a)!=RAD_OK)throw std::runtime_error("stream fixture logical map");
        const int64_t physical=buffer(block_table).id(s*columns+a.logical_block);int64_t offset=0;
        if(state::physical_byte(l,physical,key,d,&offset)!=RAD_OK)throw std::runtime_error("stream fixture physical map");
        return offset/l.element_bytes;
    }
    void rewrite_keys(){for(int64_t s=0;s<sequences;++s)for(int64_t k=0;k<keys;++k)for(int64_t d=0;d<dim;++d)buffer(cache).set(scalar_index(s,k,d),canonical_keys[size_t((s*keys+k)*dim+d)]);}
    void default_candidates(){for(int64_t r=0;r<rows;++r){const int64_t n=buffer(visible).id(r),available=(n+candidate_width-1)/candidate_width,count=std::min(top_blocks,available);
        for(int64_t i=0;i<top_blocks;++i)buffer(candidates_in).id(r*top_blocks+i,i<count?int32_t(i*available/count):-1);}}
    void refresh(){for(int i=0;i<operand_count;++i)if(buffers[size_t(i)]){buffer(i).t.data=buffer(i).words.data();tensors[size_t(i)]=buffer(i).t;}else tensors[size_t(i)]={};}
    RadArgs args(){refresh();RadArgs a{};a.t=tensors.data();a.n_t=operand_count;a.p=params.data();a.n_p=params.size();a.scratch=flag.data();a.scratch_bytes=4;return a;}
    int run(){auto a=args();return emulate(&a);}
};
} // namespace rad::ds41::index_stream::testing
