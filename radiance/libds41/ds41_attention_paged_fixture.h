/* Synthetic canonical sequences are separate from the page layout. The pre-existing checked
 * state mapper, rather than the new attention reader, places them into permuted physical pages. */
#pragma once
#include <cstring>
#include "ds41_attention_paged_emulate.h"
#include "ds41_index_fixture.h"
#include "../arch/deepseek41/state_layout.h"
#include <stdexcept>
namespace rad::ds41::attention_paged::testing {
using Buffer=index::testing::Buffer;
struct Fixture {
    uint32_t dtype;int pitch;int64_t rows=0,sequences,heads,dim,window,topk,ratio,block;
    std::vector<int32_t> lengths,starts,firsts;
    std::vector<std::vector<float>> canonical_window,canonical_compressed;
    std::array<std::optional<Buffer>,tensors_count> buffers;std::array<RadTensor,tensors_count> tensors{};
    std::vector<RadParam> params;std::array<int32_t,4> flag{911,912,913,914};
    Fixture(uint32_t dt=RAD_F32,int p=0,int64_t compression=2,int64_t dimension=8,int64_t h=2,
            int64_t w=5,int64_t selected_count=4,int64_t bs=4,
            std::vector<int32_t> sizes={7,0,10},std::vector<int32_t> contexts={13,0,4})
        :dtype(dt),pitch(p),sequences(sizes.size()),heads(h),dim(dimension),window(w),topk(compression?selected_count:0),ratio(compression),block(bs),
         lengths(std::move(sizes)),starts(std::move(contexts)),firsts(size_t(sequences),0),
         canonical_window(size_t(sequences)),canonical_compressed(size_t(sequences)) {
        if(sequences<=0||starts.size()!=size_t(sequences))throw std::runtime_error("bad attention fixture sequences");
        int64_t win_columns=1,comp_columns=1;
        for(int64_t seq=0;seq<sequences;++seq) {
            rows+=lengths[size_t(seq)];const int64_t end=int64_t(starts[size_t(seq)])+lengths[size_t(seq)];
            const int64_t oldest=starts[size_t(seq)]>=window?starts[size_t(seq)]-window+1:0;
            firsts[size_t(seq)]=int32_t(oldest/block*block);
            win_columns=std::max(win_columns,(end-firsts[size_t(seq)]+block-1)/block);comp_columns=std::max(comp_columns,(end+block-1)/block);
        }
        if(rows<=0)throw std::runtime_error("empty attention fixture");
        params={RAD_INT("M",rows+3),RAD_INT("sequences",sequences+2),RAD_INT("heads",heads),RAD_INT("head_dim",dim),
                RAD_INT("window",window),RAD_INT("topk",topk),RAD_INT("ratio",ratio),RAD_INT("block_size",block),
                RAD_STR("dtype",rad_dtype_name(dtype)),RAD_F64("scale",1./std::sqrt(double(dim))),RAD_INT("n_spec",0)};
        add(query,dtype,{rows,heads,dim});add(window_cache,dtype,{sequences*win_columns+1,1,block,dim});
        add(window_table,RAD_I32,{sequences,win_columns});add(window_used,RAD_I32,{sequences});
        if(ratio) {
            add(compressed_cache,dtype,{sequences*comp_columns+1,1,block,dim/ratio});
            add(compressed_table,RAD_I32,{sequences,comp_columns});add(compressed_used,RAD_I32,{sequences});add(selected,RAD_I32,{rows,topk});
        }
        add(positions,RAD_I32,{rows});add(cu,RAD_I32,{sequences+1});add(ctx_lens,RAD_I32,{sequences});
        add(sinks,RAD_F32,{heads});add(output,dtype,{rows,heads,dim});
        for(int cache:{window_cache,compressed_cache})if(buffers[size_t(cache)])for(int64_t i=0;i<buffer(cache).count();++i)buffer(cache).set(i,NAN);
        for(int64_t seq=0;seq<sequences;++seq) {
            for(int64_t page=0;page<win_columns;++page)buffer(window_table).id(seq*win_columns+page,int32_t(sequences*win_columns-(seq*win_columns+page)));
            if(ratio)for(int64_t page=0;page<comp_columns;++page)buffer(compressed_table).id(seq*comp_columns+page,int32_t(sequences*comp_columns-(seq*comp_columns+page)));
            const int64_t end=int64_t(starts[size_t(seq)])+lengths[size_t(seq)];buffer(ctx_lens).id(seq,starts[size_t(seq)]);
            buffer(window_used).id(seq,int32_t(end-firsts[size_t(seq)]));if(ratio)buffer(compressed_used).id(seq,int32_t(end));
            auto& win=canonical_window[size_t(seq)];win.resize(size_t(end*dim));
            for(int64_t token=0;token<end;++token)for(int64_t d=0;d<dim;++d)win[size_t(token*dim+d)]=rounded(float(int((seq+1)*19+token*7+d*13)%97-48)/32.f);
            if(ratio) {
                auto& comp=canonical_compressed[size_t(seq)];comp.resize(size_t((end/ratio)*dim));
                for(int64_t token=0;token<end/ratio;++token)for(int64_t d=0;d<dim;++d)comp[size_t(token*dim+d)]=rounded(float(int((seq+1)*11+token*17+d*5)%109-54)/32.f);
            }
        }
        rewrite_cache();int64_t row=0;buffer(cu).id(0,0);
        for(int64_t seq=0;seq<sequences;++seq) {
            for(int64_t j=0;j<lengths[size_t(seq)];++j,++row) {
                const int64_t pos=int64_t(starts[size_t(seq)])+j;buffer(positions).id(row,int32_t(pos));
                for(int64_t h0=0;h0<heads;++h0)for(int64_t d=0;d<dim;++d)buffer(query).set((row*heads+h0)*dim+d,float(int((row+3)*17+h0*7+d*11)%101-50)/64.f);
                if(ratio)for(int64_t k=0;k<topk;++k) {
                    const int64_t visible=(pos+1)/ratio;
                    buffer(selected).id(row*topk+k,!visible||(k+1==topk&&!(row&1))?-1:int32_t((k*3+row)%visible));
                }
            }
            buffer(cu).id(seq+1,int32_t(row));
        }
        for(int64_t h0=0;h0<heads;++h0)buffer(sinks).set(h0,float(h0)-.5f);
        refresh();
    }
    float rounded(float value)const {return dtype==RAD_F32?value:dtype==RAD_F16?rad_f16_to_f32(rad_f32_to_f16(value)):rad_bf16_to_f32(rad_f32_to_bf16(value));}
    void add(int i,uint32_t d,std::initializer_list<int64_t> shape) {buffers[size_t(i)].emplace(d,shape,pitch);}
    Buffer& buffer(int i){return *buffers[size_t(i)];}const Buffer& buffer(int i)const{return *buffers[size_t(i)];}
    void integer(const char* name,int64_t value){for(auto& param:params)if(!std::strcmp(param.key,name))param.ival=value;}
    int64_t scalar_index(bool compressed,int64_t seq,int64_t entry,int64_t d)const {
        namespace state=rad::deepseek41::state;state::Layout layout;state::Address address;
        const int64_t r=compressed?ratio:1;const int table=compressed?compressed_table:window_table;
        if(state::make_layout(dim,r,block,rad_dtype_bits(dtype)/8,&layout)!=RAD_OK||state::map(layout,entry,d,&address)!=RAD_OK)throw std::runtime_error("fixture state map failed");
        const int64_t first=compressed?0:firsts[size_t(seq)],column=address.logical_block-first/block;
        int64_t byte=0;const int64_t page=buffer(table).id(seq*buffer(table).t.shape[1]+column);
        if(state::physical_byte(layout,page,entry,d,&byte)!=RAD_OK)throw std::runtime_error("fixture physical byte failed");
        return byte/layout.element_bytes;
    }
    void rewrite_cache() {
        for(int64_t seq=0;seq<sequences;++seq) {
            const int64_t end=int64_t(starts[size_t(seq)])+lengths[size_t(seq)];
            for(int64_t token=firsts[size_t(seq)];token<end;++token)for(int64_t d=0;d<dim;++d)
                buffer(window_cache).set(scalar_index(false,seq,token,d),canonical_window[size_t(seq)][size_t(token*dim+d)]);
            if(ratio)for(int64_t token=0;token<end/ratio;++token)for(int64_t d=0;d<dim;++d)
                buffer(compressed_cache).set(scalar_index(true,seq,token,d),canonical_compressed[size_t(seq)][size_t(token*dim+d)]);
        }
    }
    void refresh(){for(int i=0;i<tensors_count;++i)if(buffers[size_t(i)]){buffer(i).t.data=buffer(i).words.data();tensors[size_t(i)]=buffer(i).t;}else tensors[size_t(i)]={};}
    RadArgs args(){refresh();RadArgs a{};a.t=tensors.data();a.n_t=tensors_count;a.p=params.data();a.n_p=params.size();a.scratch=flag.data();a.scratch_bytes=scratch_bytes;return a;}
    int run(){auto a=args();return emulate(&a);}
};
} // namespace rad::ds41::attention_paged::testing
