// Explicit parent-run HOST proof and CPU-only real Checkpoint/Tokenizer preflight.
#include "engine_glm_main.h"
#include "build/rad_build.h"
#include "device/device.h"
#include "sample/sampler.h"
#include "sample/vocab_view.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace rad;
namespace {
size_t checks=0,refusals=0,groups=0;
bool observe_backend=false;uint64_t backend_calls=0;
void need(bool ok,const char* why){++checks;if(!ok)throw std::runtime_error(why);}
void okay(int s,const char* why){need(s==RAD_OK,why);}
GlmMainVocabularyFacts facts(){GlmMainVocabularyFacts f;f.model_rows=154880;f.model_width=4096;
    f.tokenizer_tokens=f.view_tokens=154856;f.embedding_rows=f.head_rows=154880;
    f.embedding_cols=f.head_cols=4096;f.embedding_bytes=f.head_bytes=154880ll*4096*2;
    f.embedding_dtype=f.head_dtype=RAD_BF16;f.embedding_physical_plain=f.head_physical_plain=true;return f;}
void pure_guards(){++groups;const std::array<int32_t,5> specials{-1,154820,-1,-1,-1};
    const std::array<int32_t,3> eog{154820,154827,154829};GlmMainVocabularyAdmission out;
    okay(glm_main_vocabulary_guard(facts(),specials,eog,&out),"actual padded-vocabulary metadata admitted");
    need(out.model_rows==154880&&out.valid_tokens==154856&&out.padded_rows==24&&out.local_rows==38720,"model rows remain padded and distinct from tokenizer IDs");
    for(int mutation=0;mutation<16;++mutation){auto f=facts();switch(mutation){
        case 0:f.model_rows=154856;break;case 1:f.model_width=4095;break;case 2:f.tokenizer_tokens=154880;break;
        case 3:f.tokenizer_tokens=154855;break;case 4:f.view_tokens=154855;break;case 5:f.embedding_rows=154856;break;
        case 6:f.embedding_cols=4095;break;case 7:--f.embedding_bytes;break;case 8:f.embedding_dtype=RAD_F32;break;
        case 9:f.embedding_physical_plain=false;break;case 10:f.head_rows=154881;break;case 11:f.head_cols=4095;break;
        case 12:--f.head_bytes;break;case 13:f.head_dtype=RAD_F32;break;case 14:f.head_physical_plain=false;break;
        case 15:f.tokenizer_tokens=-1;break;}
        GlmMainVocabularyAdmission x{17,19,23,29,3,5};std::array<unsigned char,sizeof x> before{};std::memcpy(before.data(),&x,sizeof x);
        need(glm_main_vocabulary_guard(f,specials,eog,&x)<0&&!std::memcmp(&x,before.data(),sizeof x),"bad source/count metadata preserves caller output");++refusals;}
    auto bad_special=specials;bad_special[0]=154856;
    need(glm_main_vocabulary_guard(facts(),bad_special,eog,&out)==RAD_E_FORMAT,"padded BOS is refused");++refusals;
    bad_special=specials;bad_special[4]=-2;
    need(glm_main_vocabulary_guard(facts(),bad_special,eog,&out)==RAD_E_FORMAT,"invalid negative special ID refused");++refusals;
    const std::array<int32_t,1> padded{154879},negative{-1};const std::array<int32_t,2> duplicate{154820,154820};
    for(const auto ids:{std::span<const int32_t>(padded),std::span<const int32_t>(negative),std::span<const int32_t>(duplicate),std::span<const int32_t>()}){
        need(glm_main_vocabulary_guard(facts(),specials,ids,&out)==RAD_E_FORMAT,"invalid EOG set refused");++refusals;}
    std::array<int32_t,65> too_many{};for(size_t i=0;i<too_many.size();++i)too_many[i]=int32_t(i);
    need(glm_main_vocabulary_guard(facts(),specials,too_many,&out)==RAD_E_FORMAT,"bounded EOG list retains64-ID cap");++refusals;
    need(glm_main_vocabulary_guard(facts(),specials,eog,nullptr)==RAD_E_INVAL,"null admission output refused");++refusals;
    const std::array<int32_t,1> last_real{154855},first_pad{154856};
    okay(glm_main_one_prompt_admission(last_real,154856,512),"last real token remains valid prompt");
    need(glm_main_one_prompt_admission(first_pad,154856,512)==RAD_E_SHAPE,"padded prompt ID refused");++refusals;
}
struct TinyView final:VocabView {
    int32_t n_tokens()const override{return 154856;}
    const std::string& token_piece(int32_t id)const override{static const std::string a="a",z="z",empty;
        return id<0||id>=154856?empty:id==154855?z:a;}
    bool is_eog(int32_t)const override{return false;}
};
struct RankSampler {
    Registry registry;TinyView view;std::unique_ptr<Builder> builder;Sampler sampler;
    RankSampler(const std::string& home,int rank,bool enabled){okay(registry.load_plugin(home+"/kernels/libref.so",0),"actual unchanged HOST sampler plugin");
        RadModelMeta m{};m.arch_id="sampler-only-HOST-fixture";m.n_vocab=154880;RadBuildCtx bc{};
        bc.rank=rank;bc.world_size=4;bc.max_tok=bc.max_seqs=1;bc.max_ctx=512;
        builder=std::make_unique<Builder>(registry,m,bc);builder->set_instances(false);
        SamplerConfig cfg;cfg.n_vocab=154880;cfg.n_vocab_local=38720;cfg.vocab_off=rank*38720;
        cfg.rank=rank;cfg.world_size=4;cfg.max_seqs=1;cfg.valid_token_count=enabled?154856:0;
        okay(sampler.declare(builder.get(),cfg),"actual Sampler declaration with real padded row geometry");sampler.attach_vocab(&view);}
};
RadTensor tensor(void* p,uint32_t dt,std::initializer_list<int64_t> shape){RadTensor t{};t.data=p;t.dtype=dt;t.rank=uint32_t(shape.size());size_t i=0;for(auto d:shape)t.shape[i++]=d;
    int64_t stride=1;for(int axis=int(t.rank)-1;axis>=0;--axis){t.stride[axis]=stride;stride*=t.shape[axis];}return t;}
const RadKernelInfo* row(const Registry& r,const char* op){const RadKernelInfo* found=nullptr;
    for(const auto& k:r.rows())if(k.info&&k.info->domain==RAD_DOMAIN_HOST&&k.info->op&&!std::strcmp(k.info->op,op)){
        need(!found,"one actual reference row per sampler op");found=k.info;}
    need(found&&found->launch&&!found->init,"actual unmodified synchronous HOST row");return found;}
void actual_mask_topk(RankSampler& r,bool masked){
    constexpr int64_t n=38720,legal=38696;std::vector<float> logits(size_t(n),-3.f);
    logits[size_t(legal-1)]=7.f;for(int64_t i=legal;i<n;++i)logits[size_t(i)]=1000.f+float(i-legal);
    const auto original=logits;int64_t count=0;const auto* params=r.sampler.staged_params(&count);need(count==1,"one actually staged row");
    std::array<RadTensor,3> mt{tensor(logits.data(),RAD_F32,{1,n}),tensor(const_cast<DeviceSampleParams*>(params),RAD_U32,{1,sizeof(DeviceSampleParams)/4}),
        tensor(const_cast<uint32_t*>(r.sampler.staged_mask(0)),RAD_U32,{1,r.sampler.mask_words()})};
    const RadParam mp[]{RAD_INT("M",1),RAD_INT("n_vocab",n)};RadArgs ma{};ma.t=mt.data();ma.n_t=3;ma.p=mp;ma.n_p=2;ma.world_size=4;ma.rank=3;
    okay(row(r.registry,"sample_mask")->launch(&ma,nullptr),"actual existing sample_mask launch");
    for(int64_t i=0;i<n;++i)need(i<legal||!masked?logits[size_t(i)]==original[size_t(i)]:std::isinf(logits[size_t(i)])&&logits[size_t(i)]<0,"actual row changes only padded logits to negative infinity");
    int32_t index=-2;float value=0;std::array<uint32_t,2> pair{};
    std::array<RadTensor,5> tt{tensor(logits.data(),RAD_F32,{1,n}),mt[1],tensor(&index,RAD_I32,{1,1}),tensor(&value,RAD_F32,{1,1}),tensor(pair.data(),RAD_U32,{1,2})};
    const RadParam tp[]{RAD_INT("M",1),RAD_INT("n_vocab",n),RAD_INT("vocab_off",116160)};RadArgs ta{};ta.t=tt.data();ta.n_t=5;ta.p=tp;ta.n_p=3;ta.rank=3;ta.world_size=4;
    okay(row(r.registry,"sample_topk")->launch(&ta,nullptr),"actual unchanged local topk launch, no collective substitution");
    need(index==(masked?legal-1:n-1)&&pair[0]==uint32_t(116160+index),"higher padded logits cannot win; local/global ID convention exact");
    need(masked?pair[0]==154855:pair[0]==154879,"real last token wins only with mask; legacy control exposes padded winner");
}
void staged_masks(const std::string& home){++groups;
    for(int rank=0;rank<4;++rank){RankSampler r(home,rank,true);Request q;q.id=1;q.prompt={0};q.sp.temp=0;q.sp.top_k=1;
        TinySamplerQuote current,legacy_quote;okay(Sampler::glm_main_startup_quote(r.sampler.config(),&current),"actual opt-in sampler quote");
        auto old=r.sampler.config();old.valid_token_count=0;okay(Sampler::glm_main_startup_quote(old,&legacy_quote),"unchanged default sampler quote");
        need(current.storage.embedded_object_bytes==sizeof(Sampler)&&current.storage.page_and_heap_total==legacy_quote.storage.page_and_heap_total&&
             current.storage.pinned_page_bytes==legacy_quote.storage.pinned_page_bytes,"mask reuses real quote/staging; no invented sizeof or extra plane");
        okay(r.sampler.build_step({&q},0),"actual unconstrained build_step");int64_t rows=0;const auto* p=r.sampler.staged_params(&rows);
        need(rows==1&&(p->flags&RAD_SP_MASK)&&!(p->flags&RAD_SP_GREEDY)&&p->mask_row==0,"TP4 top1 chain carries real mask dependency");
        const auto* bits=r.sampler.staged_mask(0);for(int64_t j=0;j<38720;++j)
            need(bool((bits[j/32]>>(j%32))&1u)==(int64_t(rank)*38720+j<154856),"exact per-rank prefix bit and all24 padded exclusions");
        if(rank==3){need(bits[r.sampler.mask_words()-1]==0xffu,"last rank last word exactly eight real IDs");actual_mask_topk(r,true);
            okay(r.sampler.build_greedy(1),"actual greedy builder uses mask");actual_mask_topk(r,true);
            Request grammar;grammar.id=2;grammar.prompt={0};grammar.sp.temp=0;grammar.sp.top_k=1;grammar.sp.grammar="root ::= \"z\"";
            okay(r.sampler.begin_request(2,grammar.sp,nullptr),"actual synthetic grammar state");okay(r.sampler.build_step({&grammar},1),"actual grammar mask AND prefix");
            bits=r.sampler.staged_mask(0);for(int64_t j=0;j<38720;++j)need(bool((bits[j/32]>>(j%32))&1u)==(j==38695),"grammar cannot be widened by tail mask");
            okay(r.sampler.build_greedy(1),"cached greedy builder refreshes staging after grammar");actual_mask_topk(r,true);
            r.sampler.end_request(2);okay(r.sampler.build_step({&q},2),"subsequent ordinary row refreshes full real prefix");actual_mask_topk(r,true);
        }
    }
    ++groups;
    {RankSampler invalid(home,3,false);const auto original=invalid.sampler.config();
        for(int mutation=0;mutation<6;++mutation){auto cfg=original;cfg.valid_token_count=154856;
            switch(mutation){case 0:cfg.valid_token_count=154880;break;case 1:cfg.valid_token_count=154855;break;
                case 2:cfg.valid_token_count=-1;break;case 3:cfg.world_size=1;break;case 4:cfg.vocab_off=116159;break;case 5:cfg.max_spec=1;break;}
            const auto buffers=invalid.builder->program().buffers.size();Sampler rejected;
            need(rejected.declare(invalid.builder.get(),cfg)==RAD_E_UNSUPPORTED&&invalid.builder->program().buffers.size()==buffers,"bad opt-in refuses before declaration mutation");++refusals;}}
    RankSampler legacy(home,3,false);Request q;q.id=3;q.prompt={0};q.sp.temp=0;q.sp.top_k=1;
    okay(legacy.sampler.build_step({&q},0),"legacy default zero opt-in remains unchanged");int64_t rows=0;
    need(!(legacy.sampler.staged_params(&rows)->flags&RAD_SP_MASK),"legacy unconstrained flags unchanged");actual_mask_topk(legacy,false);
    okay(legacy.sampler.build_greedy(1),"legacy greedy path remains unmasked");need(!(legacy.sampler.staged_params(&rows)->flags&RAD_SP_MASK),"legacy greedy flags unchanged");
}
}
extern "C" int __real_rad_dev_count();extern "C" int __wrap_rad_dev_count(){if(observe_backend)++backend_calls;return __real_rad_dev_count();}
extern "C" int __real_rad_dev_set(int d);extern "C" int __wrap_rad_dev_set(int d){if(observe_backend)++backend_calls;return __real_rad_dev_set(d);}
extern "C" int __real_rad_dev_props(int d,RadDeviceProps* p);extern "C" int __wrap_rad_dev_props(int d,RadDeviceProps* p){if(observe_backend)++backend_calls;return __real_rad_dev_props(d,p);}
extern "C" void* __real_rad_dev_alloc(int64_t n,int k);extern "C" void* __wrap_rad_dev_alloc(int64_t n,int k){if(observe_backend)++backend_calls;return __real_rad_dev_alloc(n,k);}
extern "C" int __real_rad_stream_create(RadStream* p,int h);extern "C" int __wrap_rad_stream_create(RadStream* p,int h){if(observe_backend)++backend_calls;return __real_rad_stream_create(p,h);}
int main(int argc,char** argv){try{
    if(argc==3&&!std::strcmp(argv[1],"--model-vocab-check")){
        GlmMainVocabularyAdmission out;observe_backend=true;const int rc=GlmMainEngineHarness::vocabulary_preflight(argv[2],&out);observe_backend=false;
        need(!backend_calls,"actual CPU preflight creates no device/query/stream/model owner");
        nlohmann::json result{{"schema","glm53-real-vocabulary-cpu-preflight-v1"},{"passed",rc==RAD_OK},{"status",rc},{"model_rows",out.model_rows},{"tokenizer_tokens",out.valid_tokens},{"padded_rows",out.padded_rows},{"local_rows",out.local_rows},
            {"eog_ids",out.eog_ids},{"backend_calls",backend_calls},{"GPU_execution",false},{"source_admitted",false},{"scope","actual Engine metadata/tokenizer and real raw embedding/head headers; no payload scan/model start"}};
        std::cout<<result.dump(2)<<'\n';return rc==RAD_OK?0:1;
    }
    need(argc==3&&!std::strcmp(argv[1],"--run-host-vocab"),"explicit --run-host-vocab MATCHING_HOME or --model-vocab-check ACTUAL_CHECKPOINT");
    need(device_is_host(),"HOST-only checker, no GPU fixture");pure_guards();staged_masks(argv[2]);
    std::cout<<"PASS actual GLM padded vocabulary HOST: "<<groups<<" groups "<<checks<<" checks "<<refusals<<" refusals; unchanged HOST mask/topk, no TP collective/model execution\n";return 0;
}catch(const std::exception& e){std::cerr<<"FAIL vocabulary: "<<e.what()<<'\n';return 1;}}
