#include "build/glm_main_ordinary_loader.h"
#include "build/tiny_component_quotes.h"
#include "runtime/ctx.h"
#include "sample/sampler.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
namespace {
std::atomic<uint64_t> allocations{0},reads{0};
size_t checks=0,refusals=0;
void check(bool v,const char* why){++checks;if(!v)throw std::runtime_error(why);}
void untouched(uint64_t a,uint64_t r){check(allocations.load()==a&&reads.load()==r,"pure quote used backend allocation/payload I/O");}
template<class F>void refused(F f,int wanted){
    rad::TinySamplerQuote output;std::memset(&output,0x6d,sizeof output);const auto before=output;
    const auto a=allocations.load(),r=reads.load();check(f(&output)==wanted,"exact sampler refusal status");
    check(!std::memcmp(&output,&before,sizeof output),"sampler refusal changed caller quote");untouched(a,r);++refusals;
}
rad::SamplerConfig actual(int rank,int64_t candidates=1024){
    rad::SamplerConfig c;c.n_vocab=154880;c.n_vocab_local=38720;c.max_seqs=1;c.max_spec=0;c.rank=rank;c.world_size=4;
    c.vocab_off=int64_t(rank)*38720;c.max_candidates=candidates;return c;
}
void actual_positive(){
    for(int rank=0;rank<4;++rank)for(const auto requested:{1,512,1024,2048,4096}){
        auto c=actual(rank,requested);rad::TinySamplerQuote q;const auto a=allocations.load(),r=reads.load();
        check(rad::Sampler::glm_main_startup_quote(c,&q)==RAD_OK,"actual config quote");untouched(a,r);
        const uint64_t candidates=std::min<int64_t>(requested,rad::kSampleMergeCandidates/4);
        const uint64_t words=(38720+31)/32;
        const uint64_t width=4*(2+2*(rad::RAD_MAX_LOGPROBS+1));
        const uint64_t bytes[]{sizeof(RadSampleParams),sizeof(RadSampleParams),sizeof(RadSampleParams),2048*4,2048,8,
            words*4,words*4,rank==0?width*4:0,rank==0?candidates*4:0,rank==0?candidates*4:0};
        uint64_t payload=0,pages=0;
        for(size_t i=0;i<11;++i){check(q.pinned_request_bytes[i]==bytes[i],"actual individual staging request bytes");payload+=bytes[i];if(bytes[i])pages+=(bytes[i]+4095)/4096*4096;}
        check(q.rows==1&&q.history==2048&&q.candidates==candidates&&q.mask_words==words&&q.logprob_width==width,"actual candidate/local-mask geometry");
        check(q.storage.pinned_requests==uint64_t(rank?8:11)&&q.storage.pinned_bytes==payload&&q.storage.pinned_page_bytes==pages,"actual request count/pinned page arithmetic");
        check(!q.storage.device_requests&&!q.storage.mapped_requests&&!q.storage.event_count&&!q.storage.stream_count,"Sampler quote invented runtime resources");
        check(q.storage.heap_payload_bytes>uint64_t((154880+31)/32)*4,"whole-vocab mask and actual private request storage reserved");
    }
    auto c=actual(2);c.n_vocab_local=0;rad::TinySamplerQuote q;
    check(rad::Sampler::glm_main_startup_quote(c,&q)==RAD_OK&&q.mask_words==1210,"same actual declare default local split");
}
void actual_negative(){
    const auto c=actual(0);
    auto test=[&](auto mutate){auto bad=c;mutate(bad);refused([&](auto* out){return rad::Sampler::glm_main_startup_quote(bad,out);},RAD_E_UNSUPPORTED);};
    test([](auto& x){x.rank=-1;});test([](auto& x){x.rank=4;});test([](auto& x){x.world_size=1;});
    test([](auto& x){x.n_vocab=154881;});test([](auto& x){x.n_vocab_local=154880;});test([](auto& x){x.vocab_off=1;});
    test([](auto& x){x.max_seqs=2;});test([](auto& x){x.max_spec=1;});test([](auto& x){x.draft_coupling=true;});
    test([](auto& x){x.max_hist=INT64_MAX;});test([](auto& x){x.max_candidates=0;});test([](auto& x){x.max_candidates=4097;});
    test([](auto& x){x.max_candidates=INT64_MAX;});
    check(rad::Sampler::glm_main_startup_quote(c,nullptr)==RAD_E_INVAL,"null actual quote refuses");++refusals;
    rad::Sampler never_declared;
    refused([&](auto* out){return never_declared.glm_main_startup_quote(out);},RAD_E_STATE);
}
void old_tiny(){
    rad::SamplerConfig c;c.n_vocab=320;c.n_vocab_local=320;c.max_seqs=1;
    rad::TinySamplerQuote q;const auto a=allocations.load(),r=reads.load();
    check(rad::Sampler::tiny_startup_quote(c,&q)==RAD_OK,"unchanged old Tiny Sampler positive");untouched(a,r);
    const uint64_t width=2+2*(rad::RAD_MAX_LOGPROBS+1);
    const uint64_t bytes[]{sizeof(RadSampleParams),sizeof(RadSampleParams),sizeof(RadSampleParams),8192,2048,8,40,40,width*4,4096,4096};
    for(size_t i=0;i<11;++i)check(q.pinned_request_bytes[i]==bytes[i],"old Tiny staging arithmetic unchanged");
    check(q.storage.pinned_requests==11&&q.candidates==1024&&q.mask_words==10,"old Tiny profile unchanged");
    c.world_size=4;refused([&](auto* out){return rad::Sampler::tiny_startup_quote(c,out);},RAD_E_UNSUPPORTED);
    c.world_size=1;c.n_vocab=154880;refused([&](auto* out){return rad::Sampler::tiny_startup_quote(c,out);},RAD_E_UNSUPPORTED);
}
void early_loader(){
    std::array<rad::GlmMainOrdinaryRankSpec,4> ranks{};std::unique_ptr<rad::GlmMainOrdinaryLoader> out;
    rad::GlmMainOrdinaryLoaderLimits limits;
    auto run=[&](std::span<const rad::GlmMainOrdinaryRankSpec> input,const auto& cap){
        rad::GlmMainOrdinaryLoaderQuote q;std::memset(&q,0x39,sizeof q);const auto before=q;
        const auto a=allocations.load(),r=reads.load();check(rad::GlmMainOrdinaryLoader::prepare(input,cap,&out,&q)<0,"unready loader prewalk refuses");
        check(!out&&!std::memcmp(&q,&before,sizeof q),"loader early refusal changed outputs");untouched(a,r);++refusals;
    };
    run({},limits);run(ranks,limits);
    auto bad=limits;bad.max_jobs=0;run(ranks,bad);
    bad=limits;bad.max_input_bytes=UINT64_MAX;run(ranks,bad);
    bad=limits;bad.max_output_bytes=UINT64_MAX;run(ranks,bad);
    bad=limits;bad.max_buffer_bytes=UINT64_MAX;run(ranks,bad);
    bad=limits;bad.max_metadata_bytes=UINT64_MAX;run(ranks,bad);
    bad=limits;bad.max_column_bytes=(64u<<20)+1;run(ranks,bad);
}
void no_program(){
    rad::CtxDesc d;rad::TinyCtxQuote q;std::memset(&q,0x71,sizeof q);const auto before=q;
    const auto a=allocations.load(),r=reads.load();check(rad::Ctx::glm_main_startup_quote(d,&q)==RAD_E_INVAL,"null real Ctx quote refuses");
    check(!std::memcmp(&q,&before,sizeof q),"null Ctx refusal changed output");untouched(a,r);++refusals;
}
}
extern "C" void* __real_rad_dev_alloc(int64_t,int);
extern "C" void* __wrap_rad_dev_alloc(int64_t n,int k){++allocations;return __real_rad_dev_alloc(n,k);}
extern "C" ssize_t __real_pread(int,void*,size_t,off_t);
extern "C" ssize_t __wrap_pread(int fd,void* p,size_t n,off_t off){++reads;return __real_pread(fd,p,n,off);}
int main(int argc,char** argv){try{
    check(argc==2&&!std::strcmp(argv[1],"--run-host-quotes"),"explicit --run-host-quotes required");
    actual_positive();actual_negative();old_tiny();early_loader();no_program();
    std::cout<<"PASS 5 HOST pure-quote groups "<<checks<<" checks "<<refusals<<" refusals; no source/Engine readiness\n";return 0;
}catch(const std::exception& e){std::cerr<<"FAIL HOST quotes checks="<<checks<<" refusals="<<refusals<<": "<<e.what()<<'\n';return 1;}}
