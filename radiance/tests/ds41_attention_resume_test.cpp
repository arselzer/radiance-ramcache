/* Headers/emulators only. Real U32 pending state and prior FULL cache contents survive an odd
 * ragged chunk boundary. Separate index dimensions/projection/norm precede its own cache RoPE. */
#include "rad_test.h"
#include "../libds41/ds41_compress_fixture.h"
#include "../libds41/ds41_transform_emulate.h"
#include "../libds41/ds41_transform_table.h"
#include "../libds41/ds41_index_fixture.h"
#include "../libds41/ds41_index_paged_emulate.h"
#include "../libds41/ds41_attention_paged_emulate.h"
#include "../libds41/ds41_latent_args.h"
#include "../arch/common/rad_ds41_ref.h"
#include <map>
#include <memory>
namespace comp=rad::ds41::compress;
namespace tr=rad::ds41::transform;
namespace indexer=rad::ds41::index;
namespace paged=rad::ds41::paged_index;
namespace attn=rad::ds41::attention_paged;
namespace state=rad::deepseek41::state;
namespace ref=rad::deepseek41::ref;
namespace {
// The harness REQUIRE macros return from void cases. Value-returning fixture/oracle helpers
// need an exception instead, with each operand evaluated once and its original source context.
void need(bool condition,const char* expression,int line){
    ::radtest::checks().fetch_add(1);
    if(!condition)throw std::runtime_error(std::string(__FILE__)+":"+std::to_string(line)+": "+expression);
}
template<class A,class B>void need_equal(const A& actual,const B& expected,const char* expression,int line){
    ::radtest::checks().fetch_add(1);
    if(!(actual==expected))throw std::runtime_error(std::string(__FILE__)+":"+std::to_string(line)+": "+expression+" ("+::radtest::show(actual,expected)+")");
}
#define DS41_NEED(expression) need(bool(expression),#expression,__LINE__)
#define DS41_NEED_EQ(actual,expected) need_equal((actual),(expected),#actual " == " #expected,__LINE__)
using Buffer=indexer::testing::Buffer;
constexpr int64_t latent_dim=8,index_dim=4,heads=2,block=4,ratio=2,seqs=3,keys=8,topk=2;
const tr::RopeSpec rotation{4,16,40000.f,4.f,32.f,1.f};
const std::array<int32_t,3> slots{2,0,-1};
float bf16(float x){return ref::rounded(x,ref::Precision::bf16);}
std::vector<float> gain(int64_t dim){std::vector<float> out;for(int64_t d=0;d<dim;++d)out.push_back(.75f+float(d)/32.f);return out;}
std::vector<float> projection(){std::vector<float> out;for(int64_t d=0;d<index_dim;++d)for(int64_t k=0;k<latent_dim;++k)out.push_back(float(int((d+1)*11+k*7)%17-8)/32.f);return out;}
float supplied(int kind,int seq,int64_t pos,int64_t head,int64_t d){return bf16(float(int((seq+1)*13+pos*5+head*17+d*9+kind*7)%73-36)/32.f);}
std::vector<uint8_t> bytes(const Buffer& b){std::vector<uint8_t> out(b.words.size()*4);std::memcpy(out.data(),b.words.data(),out.size());return out;}
RadTensor tensor(Buffer& b){b.t.data=b.words.data();return b.t;}
Buffer normalize(const Buffer& input,const std::vector<float>& weights,int pitch){
    Buffer x=input,w(RAD_F32,{int64_t(weights.size())},pitch),y(RAD_BF16,{input.t.shape[0],input.t.shape[1]},pitch);
    for(size_t i=0;i<weights.size();++i)w.set(int64_t(i),weights[i]);
    const auto original=x.words;std::array<RadTensor,3> t{tensor(x),tensor(w),tensor(y)};
    const RadParam p[]{RAD_INT("M",input.t.shape[0]),RAD_INT("n",input.t.shape[1]),RAD_F64("eps",1e-6),RAD_STR("a_dtype","bf16"),RAD_STR("y_dtype","bf16")};
    std::array<int32_t,4> status{911,912,913,914};RadArgs a{};a.t=t.data();a.n_t=3;a.p=p;a.n_p=5;a.scratch=status.data();a.scratch_bytes=4;
    DS41_NEED_EQ(tr::emulate(tr::Op::rmsnorm,&a),RAD_OK);CHECK(x.words==original);CHECK_EQ(status[1],912);return y;
}
Buffer rotate(const Buffer& input,const Buffer& position,int pitch){
    Buffer x=input,pos=position,y(RAD_BF16,{input.t.shape[0],input.t.shape[1],input.t.shape[2]},pitch),table(RAD_F32,{32,2,2},pitch);
    std::vector<float> pairs;tr::TableReceipt receipt;DS41_NEED_EQ(tr::build_pairs(rotation,32,512,&pairs,&receipt),RAD_OK);CHECK_EQ(receipt.bytes,512);
    for(size_t i=0;i<pairs.size();++i)table.set(int64_t(i),pairs[i]);
    const auto original=x.words;
    std::array<RadTensor,4> t{tensor(x),tensor(pos),tensor(table),tensor(y)};
    const RadParam p[]{RAD_INT("M",input.t.shape[0]),RAD_INT("heads",input.t.shape[1]),RAD_INT("head_dim",input.t.shape[2]),RAD_INT("rotary_dim",4),RAD_STR("dtype","bf16"),RAD_INT("inverse",0)};
    std::array<int32_t,4> status{911,912,913,914};RadArgs a{};a.t=t.data();a.n_t=4;a.p=p;a.n_p=6;a.scratch=status.data();a.scratch_bytes=4;
    DS41_NEED_EQ(tr::emulate(tr::Op::rope_tail,&a),RAD_OK);CHECK(x.words==original);CHECK_EQ(status[1],912);return y;
}
Buffer heads_view(const Buffer& rows,int64_t h,int64_t d,int pitch){Buffer out(RAD_BF16,{rows.t.shape[0],h,d},pitch);for(int64_t i=0;i<rows.count();++i)out.set(i,rows.get(i));return out;}
/* An independent fixture projection with an explicit output storage boundary. Its expected
 * values use ref::linear below; this is no assertion about quantized checkpoint GEMMs. */
Buffer index_projection(const Buffer& latent,int pitch){
    Buffer output(RAD_BF16,{latent.t.shape[0],index_dim},pitch);const auto w=projection();
    for(int64_t r=0;r<latent.t.shape[0];++r)for(int64_t d=0;d<index_dim;++d){float sum=0;
        for(int64_t k=0;k<latent_dim;++k)sum=sum+latent.get(r*latent_dim+k)*w[size_t(d*latent_dim+k)];
        output.set(r*index_dim+d,sum);}
    return output;
}
struct Cache {
    int64_t dim,compression;Buffer payload,table;
    Cache(int64_t d,int64_t r):dim(d),compression(r),payload(RAD_BF16,{13,1,block,d/r},0),table(RAD_I32,{seqs,4},1){
        for(int64_t i=0;i<payload.count();++i)payload.set(i,NAN);
        for(int64_t s=0;s<seqs;++s)for(int64_t p=0;p<4;++p)table.id(s*4+p,int32_t(12-(s*4+p)));
        for(int64_t entry=0;entry<2;++entry){std::vector<float> v(size_t(d),0.f);for(int64_t k=0;k<d;++k)v[size_t(k)]=float(entry+k+1)/8.f;write(2,entry,v);}
    }
    int64_t start_slot(int64_t s,int64_t entry)const {
        const int64_t final=(entry+1)*compression-1,page=table.id(s*4+final/block),slot=page*block+final%block;
        rad::ds41::latent::Launch store;DS41_NEED_EQ(state::make_layout(dim,compression,block,2,&store.layout),RAD_OK);store.slots=13*block;
        int64_t first=-1;DS41_NEED_EQ(rad::ds41::latent::store_row(store,final,slot,&first),RAD_OK);return first;
    }
    void write(int64_t s,int64_t entry,const std::vector<float>& values){
        DS41_NEED_EQ(values.size(),size_t(dim));const int64_t first=start_slot(s,entry);const auto before=bytes(payload);std::vector<bool> touched(before.size(),false);
        for(int64_t d=0;d<dim;++d){const int64_t flat=first*(dim/compression)+d;payload.set(flat,values[size_t(d)]);
            const size_t at=size_t(payload.at(flat))*2;touched[at]=touched[at+1]=true;}
        const auto after=bytes(payload);for(size_t i=0;i<after.size();++i)if(!touched[i])CHECK_EQ(after[i],before[i]);
    }
    std::vector<uint16_t> sequence_bits(int64_t s)const {
        const auto all=bytes(payload);std::vector<uint16_t> out;
        for(int64_t p=0;p<4;++p){const int64_t physical=table.id(s*4+p);for(int64_t k=0;k<block*(dim/compression);++k){
            const int64_t at=physical*block*(dim/compression)+k;uint16_t h;std::memcpy(&h,all.data()+size_t(at)*2,2);out.push_back(h);}}
        return out;
    }
    bool missing(int64_t s,int64_t entry)const {const int64_t first=start_slot(s,entry);for(int64_t d=0;d<dim;++d)if(!std::isnan(payload.get(first*(dim/compression)+d)))return false;return true;}
};
struct Record {std::vector<float> scores,attention;std::vector<int32_t> ids;};
using Records=std::map<std::pair<int,int64_t>,Record>;
std::vector<float> reference_rotate(std::vector<float> values,int64_t position){
    const ref::Rope spec{4,rotation.theta,rotation.factor,rotation.beta_fast,rotation.beta_slow,rotation.original_context};
    DS41_NEED_EQ(ref::rotary(values,position,spec),RAD_OK);for(auto& value:values)value=bf16(value);return values;
}
Records canonical(){
    Records expected;const int64_t totals[]{7,5};const auto latent_gain=gain(latent_dim),index_gain=gain(index_dim),wk=projection();
    for(int s=0;s<2;++s){std::vector<std::vector<float>> window,latents,index_keys;
        ref::Compressor compressor(2,latent_gain,1e-6f,"independent",ref::Precision::bf16);
        for(int64_t p=0;p<totals[s];++p){std::vector<float> kv,gate,latent,win(size_t(latent_dim),0.f),source;
            for(int64_t d=0;d<latent_dim;++d){kv.push_back(comp::testing::Fixture::kv_value(slots[size_t(s)],p,d));gate.push_back(comp::testing::Fixture::score_value(slots[size_t(s)],p,d));source.push_back(supplied(0,s,p,0,d));}
            DS41_NEED_EQ(ref::rms_norm(source,latent_gain,1e-6f,win,ref::Precision::bf16),RAD_OK);window.push_back(reference_rotate(win,p));
            int64_t first=-1;DS41_NEED_EQ(compressor.push(kv,gate,&latent,&first),RAD_OK);
            if(!latent.empty()){std::vector<float> projected(size_t(index_dim),0.f),normalized(size_t(index_dim),0.f);
                DS41_NEED_EQ(ref::linear(latent,wk,index_dim,projected,ref::Precision::bf16),RAD_OK);
                DS41_NEED_EQ(ref::rms_norm(projected,index_gain,1e-6f,normalized,ref::Precision::bf16),RAD_OK);
                latents.push_back(reference_rotate(latent,first));index_keys.push_back(reference_rotate(normalized,first));}
        }
        for(int64_t p=0;p<totals[s];++p){std::vector<float> q,iq,head_weights;
            for(int64_t h=0;h<heads;++h){std::vector<float> qh,ih;for(int64_t d=0;d<latent_dim;++d)qh.push_back(supplied(1,s,p,h,d));for(int64_t d=0;d<index_dim;++d)ih.push_back(supplied(2,s,p,h,d));
                qh=reference_rotate(qh,p);ih=reference_rotate(ih,p);q.insert(q.end(),qh.begin(),qh.end());iq.insert(iq.end(),ih.begin(),ih.end());head_weights.push_back(.5f+float(h)/4.f+float(p%3)/16.f);}
            const int64_t visible=(p+1)/2;std::vector<float> key_prefix,visible_scores(size_t(visible),0.f);
            for(int64_t k=0;k<visible;++k)key_prefix.insert(key_prefix.end(),index_keys[size_t(k)].begin(),index_keys[size_t(k)].end());
            if(visible)DS41_NEED_EQ(ref::index_scores(iq,key_prefix,head_weights,index_dim,heads,visible_scores),RAD_OK);
            Record record;record.scores.assign(size_t(keys),-INFINITY);std::copy(visible_scores.begin(),visible_scores.end(),record.scores.begin());
            DS41_NEED_EQ(ref::index_topk(record.scores,size_t(visible),topk,{},&record.ids),RAD_OK);
            std::vector<float> joint;for(const auto& row:window)joint.insert(joint.end(),row.begin(),row.end());for(const auto& row:latents)joint.insert(joint.end(),row.begin(),row.end());
            std::vector<int32_t> selection;for(int64_t k=std::max<int64_t>(0,p-4);k<=p;++k)selection.push_back(int32_t(k));
            for(auto id:record.ids)selection.push_back(id<0?-1:int32_t(totals[s]+id));
            record.attention.resize(size_t(heads*latent_dim));
            const std::array<float,2> sinks{.5f,-.75f};DS41_NEED_EQ(ref::sparse_attention(q,joint,heads,latent_dim,sinks,selection,float(1./std::sqrt(double(latent_dim))),record.attention),RAD_OK);
            for(auto& value:record.attention)value=bf16(value);
            expected[{s,p}]=std::move(record);
        }
    }
    return expected;
}
struct Pipeline {
    int pitch;Buffer pending;const void* pending_address=nullptr;Cache window{latent_dim,1},latent{latent_dim,2},index_keys{index_dim,2};Records records;
    explicit Pipeline(int p):pitch(p),pending(RAD_U32,{4,1,2,latent_dim+1},p){
        comp::testing::Fixture seed(latent_dim,2,RAD_F32,p,{1},{0},{2});seed.seed(1,9);seed.seed(3,4);
        pending.words=seed.buffer(comp::pending_state).words;pending.t=seed.buffer(comp::pending_state).t;pending_address=pending.words.data();
    }
    std::vector<uint32_t> inactive_state()const {
        std::vector<uint32_t> out;
        for(int slot:{1,3})for(int plane=0;plane<2;++plane)for(int64_t d=0;d<=latent_dim;++d){
            out.push_back(pending.words[size_t(rad::ds41::math::offset(pending.t,(slot*2+plane)*(latent_dim+1)+d))]);
        }
        return out;
    }
    int step(std::vector<int32_t> lengths,std::vector<int32_t> starts,bool corrupt=false){
        DS41_NEED_EQ(lengths.size(),size_t(seqs));DS41_NEED_EQ(starts.size(),size_t(seqs));
        CHECK(pending.words.data()==pending_address);const auto old_state=pending.words,old_window=window.payload.words,old_latent=latent.payload.words,old_index=index_keys.payload.words;
        const auto inactive_pending=inactive_state();
        const auto inactive_window=window.sequence_bits(2),inactive_latent=latent.sequence_bits(2),inactive_index=index_keys.sequence_bits(2);
        comp::testing::Fixture pool(latent_dim,2,RAD_F32,pitch,lengths,starts,std::vector<int32_t>(slots.begin(),slots.end()));
        pool.buffer(comp::pending_state).words=pending.words;if(corrupt)pool.state_set(2,0,latent_dim,uint32_t(starts[0]-1));
        const auto projected=pool.buffer(comp::projected_kv).words,gate=pool.buffer(comp::gate_scores).words;
        auto pool_args=pool.args();if(!corrupt)pool.tensors[comp::pending_state].data=pending.words.data();
        const int status=comp::emulate(&pool_args);CHECK(pool.buffer(comp::projected_kv).words==projected);CHECK(pool.buffer(comp::gate_scores).words==gate);
        if(status!=RAD_OK){CHECK(pending.words==old_state);CHECK(window.payload.words==old_window);CHECK(latent.payload.words==old_latent);CHECK(index_keys.payload.words==old_index);return status;}
        CHECK(pending.words.data()==pending_address);CHECK(inactive_state()==inactive_pending);int64_t count=0;for(int64_t r=0;r<pool.tokens;++r)count+=pool.buffer(comp::emit).geti(r)!=0;
        DS41_NEED(count>0);Buffer cast(RAD_BF16,{count,latent_dim},pitch),first_position(RAD_I32,{count},pitch);
        std::vector<std::pair<int,int64_t>> emitted;int64_t compact=0;
        for(int64_t s=0;s<seqs;++s)for(int64_t r=pool.buffer(comp::cu).geti(s);r<pool.buffer(comp::cu).geti(s+1);++r)if(pool.buffer(comp::emit).geti(r)){
            const int64_t first=pool.buffer(comp::first_position).geti(r);CHECK_EQ(first%2,0);first_position.id(compact,int32_t(first));emitted.emplace_back(int(s),first/2);
            for(int64_t d=0;d<latent_dim;++d)cast.set(compact*latent_dim+d,pool.buffer(comp::pooled).getf(r*latent_dim+d));
            ++compact;}
        auto normalized=normalize(cast,gain(latent_dim),pitch);const auto unrotated=normalized.words;
        auto projected_index=index_projection(normalized,pitch),normalized_index=normalize(projected_index,gain(index_dim),pitch);
        auto latent_rows=rotate(heads_view(normalized,1,latent_dim,pitch),first_position,pitch),key_rows=rotate(heads_view(normalized_index,1,index_dim,pitch),first_position,pitch);
        CHECK(normalized.words==unrotated);bool distinct=false;
        for(int64_t r=0;r<count;++r){std::vector<float> lv,iv;for(int64_t d=0;d<latent_dim;++d)lv.push_back(latent_rows.get(r*latent_dim+d));for(int64_t d=0;d<index_dim;++d){iv.push_back(key_rows.get(r*index_dim+d));distinct=distinct||iv.back()!=lv[size_t(d)];}
            latent.write(emitted[size_t(r)].first,emitted[size_t(r)].second,lv);index_keys.write(emitted[size_t(r)].first,emitted[size_t(r)].second,iv);}
        CHECK(distinct);const int64_t rows=pool.tokens;
        Buffer q_source(RAD_BF16,{rows,heads,latent_dim},pitch),iq_source(RAD_BF16,{rows,heads,index_dim},pitch),win_source(RAD_BF16,{rows,latent_dim},pitch),position(RAD_I32,{rows},pitch),sequence(RAD_I32,{rows},pitch),visible(RAD_I32,{rows},pitch),head_weights(RAD_F32,{rows,heads},pitch),ctx(RAD_I32,{seqs},pitch),cu(RAD_I32,{seqs+1},pitch),used(RAD_I32,{seqs},pitch);
        int64_t row=0;cu.id(0,0);
        for(int s=0;s<seqs;++s){ctx.id(s,starts[size_t(s)]);used.id(s,starts[size_t(s)]+lengths[size_t(s)]);
            for(int64_t j=0;j<lengths[size_t(s)];++j,++row){const int64_t pos=starts[size_t(s)]+j;position.id(row,int32_t(pos));sequence.id(row,s);visible.id(row,int32_t((pos+1)/2));
                for(int64_t d=0;d<latent_dim;++d)win_source.set(row*latent_dim+d,supplied(0,s,pos,0,d));
                for(int64_t h=0;h<heads;++h){head_weights.set(row*heads+h,.5f+float(h)/4.f+float(pos%3)/16.f);for(int64_t d=0;d<latent_dim;++d)q_source.set((row*heads+h)*latent_dim+d,supplied(1,s,pos,h,d));for(int64_t d=0;d<index_dim;++d)iq_source.set((row*heads+h)*index_dim+d,supplied(2,s,pos,h,d));}}
            cu.id(s+1,int32_t(row));}
        auto query=rotate(q_source,position,pitch),index_query=rotate(iq_source,position,pitch),win_norm=normalize(win_source,gain(latent_dim),pitch),win_rows=rotate(heads_view(win_norm,1,latent_dim,pitch),position,pitch);
        for(int64_t r=0;r<rows;++r){std::vector<float> values;for(int64_t d=0;d<latent_dim;++d)values.push_back(win_rows.get(r*latent_dim+d));window.write(sequence.id(r),position.id(r),values);}
        Buffer scores(RAD_F32,{rows,keys},pitch);std::array<int32_t,4> flag{911,912,913,914};
        std::array<RadTensor,8> score_tensors{tensor(index_query),tensor(index_keys.payload),tensor(head_weights),tensor(sequence),tensor(visible),tensor(index_keys.table),{},tensor(scores)};
        const RadParam score_params[]{RAD_INT("M",rows),RAD_INT("key_rows",keys),RAD_INT("heads",heads),RAD_INT("global_heads",heads),RAD_INT("head_dim",index_dim),RAD_INT("sequences",seqs),RAD_STR("dtype","bf16"),RAD_INT("ratio",2),RAD_INT("block_size",block)};
        RadArgs args{};args.t=score_tensors.data();args.n_t=8;args.p=score_params;args.n_p=9;args.scratch=flag.data();args.scratch_bytes=4;
        const auto index_before=index_keys.payload.words,query_before=index_query.words;DS41_NEED_EQ(paged::emulate(&args),RAD_OK);CHECK(index_keys.payload.words==index_before);CHECK(index_query.words==query_before);
        Buffer ids(RAD_I32,{rows,topk},pitch);std::array<RadTensor,4> top_tensors{tensor(scores),tensor(visible),{},tensor(ids)};
        const RadParam top_params[]{RAD_INT("M",rows),RAD_INT("key_rows",keys),RAD_INT("topk",topk)};args.t=top_tensors.data();args.n_t=4;args.p=top_params;args.n_p=3;
        DS41_NEED_EQ(indexer::emulate(indexer::Op::topk,&args),RAD_OK);
        Buffer sink(RAD_F32,{heads},pitch),output(RAD_BF16,{rows,heads,latent_dim},pitch);sink.set(0,.5f);sink.set(1,-.75f);
        std::array<RadTensor,attn::tensors_count> tensors{tensor(query),tensor(window.payload),tensor(window.table),tensor(used),tensor(latent.payload),tensor(latent.table),tensor(used),tensor(ids),tensor(position),tensor(cu),tensor(ctx),tensor(sink),tensor(output)};
        const RadParam parameters[]{RAD_INT("M",rows),RAD_INT("sequences",seqs),RAD_INT("heads",heads),RAD_INT("head_dim",latent_dim),RAD_INT("window",5),RAD_INT("topk",topk),RAD_INT("ratio",2),RAD_INT("block_size",block),RAD_STR("dtype","bf16"),RAD_F64("scale",1./std::sqrt(double(latent_dim))),RAD_INT("n_spec",0)};
        args.t=tensors.data();args.n_t=attn::tensors_count;args.p=parameters;args.n_p=11;
        const auto window_before=window.payload.words,latent_before=latent.payload.words;DS41_NEED_EQ(attn::emulate(&args),RAD_OK);
        CHECK(window.payload.words==window_before);CHECK(latent.payload.words==latent_before);CHECK(index_keys.payload.words==index_before);
        for(int64_t r=0;r<rows;++r){Record record;for(int64_t k=0;k<keys;++k){const float value=scores.get(r*keys+k);record.scores.push_back(value);if(k>=visible.id(r))CHECK_EQ(value,-INFINITY);}
            for(int64_t k=0;k<topk;++k){const auto id=ids.id(r*topk+k);CHECK(id<visible.id(r));record.ids.push_back(id);}for(int64_t d=0;d<heads*latent_dim;++d)record.attention.push_back(output.get(r*heads*latent_dim+d));records[{sequence.id(r),position.id(r)}]=std::move(record);}
        const auto output_before=output.words;const int64_t last=lengths[0]-1;const int32_t original=ids.id(last*topk);ids.id(last*topk,visible.id(last));
        CHECK_EQ(attn::emulate(&args),RAD_E_SHAPE);CHECK(output.words==output_before);ids.id(last*topk,original);
        tensors[attn::output].data=tensors[attn::window_cache].data;attn::Plan plan;const auto prior_flag=flag;CHECK(attn::validate(&args,&plan)<0);CHECK(flag==prior_flag);CHECK(window.payload.words==window_before);CHECK(latent.payload.words==latent_before);
        CHECK(window.sequence_bits(2)==inactive_window);CHECK(latent.sequence_bits(2)==inactive_latent);CHECK(index_keys.sequence_bits(2)==inactive_index);
        for(int64_t s=0;s<2;++s)if((starts[size_t(s)]+lengths[size_t(s)])&1){const int64_t incomplete=(starts[size_t(s)]+lengths[size_t(s)])/2;CHECK(latent.missing(s,incomplete));CHECK(index_keys.missing(s,incomplete));}
        return RAD_OK;
    }
};
void compare(const Records& actual,const Records& expected){DS41_NEED_EQ(actual.size(),expected.size());for(const auto& [key,row]:actual){const auto& wanted=expected.at(key);CHECK(row.ids==wanted.ids);CHECK(row.attention==wanted.attention);DS41_NEED_EQ(row.scores.size(),wanted.scores.size());for(size_t k=0;k<row.scores.size();++k)CHECK_EQ(row.scores[k],wanted.scores[k]);}}
}
TEST(ds41_attention_resume_ragged_odd_c2_chunks_match_cold_and_independent_separate_index_oracle){
    static_assert(latent_dim!=index_dim);const auto expected=canonical();
    for(int pitch=0;pitch<3;++pitch){Pipeline cold(pitch),split(pitch);REQUIRE_EQ(cold.step({7,5,0},{0,0,19}),RAD_OK);
        REQUIRE_EQ(split.step({3,2,0},{0,0,19}),RAD_OK);const auto prior_latent=split.latent.sequence_bits(0),prior_keys=split.index_keys.sequence_bits(0);
        REQUIRE_EQ(split.step({4,3,0},{3,2,19}),RAD_OK);compare(cold.records,expected);compare(split.records,expected);
        CHECK(cold.pending.words==split.pending.words);CHECK(cold.window.payload.words==split.window.payload.words);CHECK(cold.latent.payload.words==split.latent.payload.words);CHECK(cold.index_keys.payload.words==split.index_keys.payload.words);
        CHECK(prior_latent!=split.latent.sequence_bits(0));CHECK(prior_keys!=split.index_keys.sequence_bits(0));
    }
}
TEST(ds41_attention_resume_bad_pending_cursor_does_not_publish_any_new_cache_or_query_result){
    Pipeline pipeline(2);REQUIRE_EQ(pipeline.step({3,2,0},{0,0,19}),RAD_OK);const auto state=pipeline.pending.words,window=pipeline.window.payload.words,latent=pipeline.latent.payload.words,index=pipeline.index_keys.payload.words;const auto records=pipeline.records;
    CHECK_EQ(pipeline.step({4,3,0},{3,2,19},true),RAD_E_STATE);CHECK(pipeline.pending.words==state);CHECK(pipeline.window.payload.words==window);CHECK(pipeline.latent.payload.words==latent);CHECK(pipeline.index_keys.payload.words==index);compare(pipeline.records,records);
}
RAD_TEST_MAIN()
