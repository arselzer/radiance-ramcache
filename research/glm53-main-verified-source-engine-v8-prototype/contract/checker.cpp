#include "fixture_writer.h"
#include "graph_quote.h"
#include "build/rad_build.h"
#include "build/tiny_component_quotes.h"
#include "runtime/ctx.h"
#include "place/native_backing_owner.h"
#include "build/startup.h"
#include "device/device.h"
#include "format/checkpoint.h"
#include "place/planner.h"
#include "sample/sampler.h"
#include "text/tokenizer.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fcntl.h>
#include <iostream>
#include <stdexcept>
#include <functional>
#include <sys/stat.h>
#include <unistd.h>
using namespace glm53::tiny_startup;
namespace {
size_t checks=0,groups=0,shape_refusals=0,graph_refusals=0,sampler_oom=0,source_refusals=0,stale_witnesses=0,vocab_refusals=0,vocab_witnesses=0;
void need(bool b,const std::string& why){++checks;if(!b)throw std::runtime_error(why);}
void good(int status,const std::string& why){need(status==RAD_OK,why+" status="+std::to_string(status));}
struct Scratch {
    std::string root,checkpoint;std::array<std::string,6> known;
    Scratch(){char p[]="/tmp/glm53-startup-metadata-XXXXXX";const char* v=::mkdtemp(p);if(!v)throw std::runtime_error("scratch directory");try{root=v;checkpoint=root+"/checkpoint";size_t i=0;for(const char* n:{"model.safetensors","config.json","tokenizer.json","tokenizer_config.json","unexpected.json","chat_template.jinja"})known[i++]=checkpoint+"/"+n;}catch(...){::rmdir(v);throw;}}
    ~Scratch()noexcept{for(const auto& p:known)if(!p.empty())::unlink(p.c_str());if(!checkpoint.empty())::rmdir(checkpoint.c_str());if(!root.empty())::rmdir(root.c_str());}
};
struct Allocation {void* ptr=nullptr;int64_t bytes=0;};
struct AllocationTrace {
    bool enabled=false,unadmitted=false;
    int fail_at=-1;uint64_t attempts=0,successes=0,requested=0,page_sum=0,live=0;
    std::array<Allocation,16> items{};
} trace;
void begin_trace(int fail=-1){need(trace.live==0,"previous sampler backing still live");trace={};trace.enabled=true;trace.fail_at=fail;}
void end_trace(){need(trace.live==0&&!trace.unadmitted,"sampler backing cleanup/admission");trace.enabled=false;}
RadBuildCtx context(){RadBuildCtx c{};c.world_size=1;c.max_tok=T;c.max_seqs=1;c.max_ctx=T;return c;}
rad::SamplerConfig sampler_config(){rad::SamplerConfig s;s.n_vocab=s.n_vocab_local=V;s.max_seqs=1;s.max_spec=0;s.max_hist=2048;s.max_candidates=1024;return s;}
void cold_tests(rad::Checkpoint& ck,const InspectedSource& inspected,const ColdSourceSeal& seal){++groups;
    auto m=ck.meta().m;auto c=context();need(seal.permits_zero_head_metadata(m,c)&&!seal.engine_eligible(),"actual source seal zero-head declaration only");
    for(int which=0;which<24;++which){m=ck.meta().m;c=context();switch(which){
      case 0:m.n_head=1;break;case 1:m.n_head_kv=1;break;case 2:m.head_dim=1;break;case 3:m.n_layers=9;break;
      case 4:m.n_embd=129;break;case 5:m.n_ff=256;break;case 6:m.n_expert=5;break;case 7:m.n_expert_used=3;break;
      case 8:m.n_expert_shared=1;break;case 9:m.n_vocab=319;break;case 10:m.n_ctx_train=6;break;case 11:m.arch_id="glm53";break;
      case 12:c.world_size=2;break;case 13:c.rank=1;break;case 14:c.max_tok=6;break;case 15:c.max_seqs=2;break;
      case 16:c.max_ctx=6;break;case 17:c.max_spec=1;break;case 18:c.is_draft=1;break;case 19:c.scope="draft";break;
      case 20:c.shape_probe=1;break;case 21:c.max_enc_patches=1;break;case 22:c.tp_attn_rows=1;break;case 23:c.draft_coupling=1;break;}
      need(!seal.permits_zero_head_metadata(m,c)&&admit_declaration_shape(&m,&c)<0,"closed profile admitted mutated field "+std::to_string(which));++shape_refusals;}
    std::unique_ptr<const ColdSourceSeal> occupied;good(seal_source(inspected,&occupied),"second source seal");const auto* original=occupied.get();
    need(seal_source(inspected,&occupied)==RAD_E_INVAL&&occupied.get()==original,"occupied seal output changed");
    AdmissionQuote q;q.cap=17;BackendQuote b;b.stack_bytes=8*MiB;b.guard_bytes=4096;
    need(quote_admission(seal,b,&q)==RAD_E_UNSUPPORTED&&q.cap==17,"missing actual private HOST quote changed caller output");
    // Supplied arithmetic test input, explicitly NOT a private Core sizeof query.
    // Neither this nor future-hook booleans can create an Engine capability.
    b.matching_private_size_quote=true;b.host_stream_control_bytes=128u<<10;b.tiny_loader_retirement_implemented=b.host_stream_raii_implemented=true;
    good(quote_admission(seal,b,&q),"conditional HOST upper-bound arithmetic only");
    need(q.backend_stream_bytes==25571328&&q.owned_threads==5&&q.categories==135244800&&q.cap==192*MiB&&!q.engine_eligible,"HOST conditional five-stack ledger arithmetic");
    b.backend=Backend::Aql;good(quote_admission(seal,b,&q),"AQL ledger arithmetic only");need(q.backend_stream_bytes==38538240&&q.owned_threads==2&&!q.engine_eligible,"AQL known stream reservation arithmetic");
    b.streams=4;q.cap=19;need(quote_admission(seal,b,&q)<0&&q.cap==19,"extra stream admitted or output changed");
}
void writer_tests(const std::string& directory,const FixtureFiles& made){++groups;
    need(made.payload_bytes==1020544&&made.file_bytes<=2*MiB&&made.artifact_files==4&&made.sidecar_bytes<=131072,"writer actual requested four-file bytes");std::string why;good(preflight_source_files(directory,&why),"cold bounded source: "+why);
    FixtureFiles collision;need(write_fixture(directory,&collision)<0&&collision.directory.empty(),"NOREPLACE overwrote destination or caller");good(preflight_source_files(directory),"NOREPLACE preserved exact fixture");
    const auto extra=directory+"/unexpected.json";int fd=::open(extra.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600);need(fd>=0,"extra-file negative open");need(::write(fd,"{}",2)==2,"extra-file negative write");::close(fd);
    need(preflight_source_files(directory)<0,"extra checkpoint namespace admitted");need(::unlink(extra.c_str())==0,"extra negative cleanup");
    fd=::open((directory+"/model.safetensors").c_str(),O_RDWR|O_CLOEXEC|O_NOFOLLOW);need(fd>=0,"header negative open");uint64_t original=0,bad=65537;
    need(::pread(fd,&original,8,0)==8&&::pwrite(fd,&bad,8,0)==8,"header negative mutate");need(preflight_source_files(directory)<0,"oversized header admitted before Core open");
    need(::pwrite(fd,&original,8,0)==8,"header negative restore");::close(fd);good(preflight_source_files(directory),"restored exact source");
}
void replace_sidecar(const std::string& path,const std::string& text){struct Fd{int fd=-1;~Fd(){if(fd>=0)::close(fd);}} f{::open(path.c_str(),O_WRONLY|O_CREAT|O_TRUNC|O_CLOEXEC|O_NOFOLLOW,0600)};
    need(f.fd>=0,"bounded tokenizer sidecar replacement open");size_t at=0;while(at<text.size()){const auto n=::write(f.fd,text.data()+at,text.size()-at);need(n>0,"bounded tokenizer sidecar replacement write");at+=size_t(n);}need(::fsync(f.fd)==0,"bounded tokenizer sidecar replacement fsync");}
std::string tokenizer_metadata_for_test(const std::string& directory){const auto path=directory+"/tokenizer.json";struct stat s{};need(::lstat(path.c_str(),&s)==0&&S_ISREG(s.st_mode)&&s.st_size>=0&&s.st_size<=131072,"bounded test tokenizer metadata extent");
    struct Fd{int fd=-1;~Fd(){if(fd>=0)::close(fd);}} f{::open(path.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW)};need(f.fd>=0,"bounded test tokenizer metadata open");std::string text(size_t(s.st_size),' ');
    need(::pread(f.fd,text.data(),text.size(),0)==ssize_t(text.size()),"bounded test tokenizer metadata read");return text;}
void vocabulary_contract_tests(){++groups;using Json=nlohmann::ordered_json;Scratch fixture,replacement;FixtureFiles made,other;good(write_fixture(fixture.checkpoint,&made),"four-file vocabulary fixture");
    const auto path=made.directory+"/tokenizer_config.json",canonical=tokenizer_configuration_json();VocabQuote quote;good(inspect_vocab_contract(made.directory,&quote),"actual explicit Core BOS/EOS contract");
    need(quote.tokens==320&&quote.bos==257&&quote.eos==256&&quote.eot==256&&quote.eog_count==2&&!quote.add_bos&&!quote.add_eos&&!quote.bos_terminal&&quote.eos_terminal&&quote.im_end_terminal,"actual distinct special-token IDs/termination");
    for(int mutation=0;mutation<5;++mutation){auto config=Json::parse(canonical);switch(mutation){case 0:config["bos_token"]="<|im_end|>";break;case 1:config["eos_token"]="<|im_start|>";break;
        case 2:config["bos_token"]="<|missing_synthetic_bos|>";break;case 3:config["add_bos_token"]=true;break;case 4:config["add_eos_token"]=true;break;}
        replace_sidecar(path,config.dump());good(preflight_source_files(made.directory),"bounded semantic tokenizer mutation reaches actual Core parser");
        rad::VocabBuild build;good(rad::parse_tokenizer_json((made.directory+"/tokenizer.json").c_str(),build),"independent actual tokenizer mutation parser witness");rad::Vocab actual;good(actual.load(std::move(build)),"independent actual tokenizer mutation Vocab witness");
        need((mutation==0&&actual.bos()==258)||(mutation==1&&actual.eos()==257)||(mutation==2&&actual.bos()==-1)||(mutation==3&&actual.add_bos())||(mutation==4&&actual.add_eos()),"intended actual special-token semantic mutation did not occur");++vocab_witnesses;
        quote.tokens=999;std::string why;need(inspect_vocab_contract(made.directory,&quote,&why)==RAD_E_SHAPE&&quote.tokens==999&&why=="actual Core BOS/EOS/EOT/token spelling/automatic-add contract","semantic tokenizer mutation admitted or caller changed");++vocab_refusals;
    }
    need(::unlink(path.c_str())==0,"missing tokenizer sibling mutation witness");quote.tokens=999;need(inspect_vocab_contract(made.directory,&quote)<0&&quote.tokens==999,"missing Loader-visible tokenizer sidecar accepted");++vocab_refusals;
    replace_sidecar(path,std::string(4097,' '));need(inspect_vocab_contract(made.directory,&quote)<0&&quote.tokens==999,"over-budget tokenizer sidecar accepted");++vocab_refusals;replace_sidecar(path,canonical);
    const auto original_tokenizer=tokenizer_metadata_for_test(made.directory);auto tokenizer=Json::parse(original_tokenizer);tokenizer["model"]["vocab"]["<|im_start|>"]=320;tokenizer["added_tokens"][1]["id"]=320;
    replace_sidecar(made.directory+"/tokenizer.json",tokenizer.dump());std::string envelope_why;need(inspect_vocab_contract(made.directory,&quote,&envelope_why)==RAD_E_FORMAT&&quote.tokens==999&&envelope_why=="exact bounded synthetic tokenizer envelope","out-of-profile token ID reached Core allocation or caller changed");++vocab_refusals;replace_sidecar(made.directory+"/tokenizer.json",original_tokenizer);
    std::shared_ptr<const SourceCheckpoint> owner;good(open_source_checkpoint(made.directory,&owner),"four-artifact owned source");std::unique_ptr<const InspectedSource> inspected;good(inspect_checkpoint_source(owner,made.directory,&inspected),"four-artifact actual source inspection");std::unique_ptr<const ColdSourceSeal> seal;good(seal_source(*inspected,&seal),"four-artifact source seal");
    replace_sidecar(made.directory+"/chat_template.jinja","x");AdmissionQuote extra_refused;extra_refused.cap=999;BackendQuote extra_backend;std::string extra_why;
    need(quote_admission(*seal,extra_backend,&extra_refused,&extra_why)==RAD_E_FORMAT&&extra_refused.cap==999&&extra_why=="extra source file outside closed fixture profile","new Loader-visible sidecar after sealing escaped closed namespace");++vocab_refusals;
    need(::unlink((made.directory+"/chat_template.jinja").c_str())==0,"extra future-loader sidecar cleanup");good(owner->audit(),"original four-artifact namespace restored");
    good(write_fixture(replacement.checkpoint,&other),"tokenizer artifact replacement fixture");auto changed=Json::parse(canonical);changed["bos_token"]="<|im_end|>";replace_sidecar(other.directory+"/tokenizer_config.json",changed.dump());
    need(::rename((other.directory+"/tokenizer_config.json").c_str(),path.c_str())==0,"actual tokenizer_config inode replacement witness");AdmissionQuote refused;refused.cap=999;BackendQuote b;std::string why;
    need(quote_admission(*seal,b,&refused,&why)==RAD_E_STATE&&refused.cap==999&&why=="opened source artifact changed or replaced","fourth Loader-visible artifact not sealed or caller changed");++vocab_refusals;
}
void mutate_physical_payload(const std::string& directory){const auto path=directory+"/model.safetensors";const int fd=::open(path.c_str(),O_RDWR|O_CLOEXEC|O_NOFOLLOW);need(fd>=0,"source binding payload mutation open");
    uint64_t header=0;uint8_t byte=0;need(::pread(fd,&header,8,0)==8&&header<=65536&&::pread(fd,&byte,1,off_t(8+header))==1,"source binding payload mutation read");
    byte^=1;need(::pwrite(fd,&byte,1,off_t(8+header))==1&&::fsync(fd)==0,"source binding payload mutation write witness");::close(fd);}
void source_file_binding_tests(const std::shared_ptr<const SourceCheckpoint>& a){++groups;
    Scratch b,replacement;FixtureFiles bf,rf;good(write_fixture(b.checkpoint,&bf),"separate B fixture");good(write_fixture(replacement.checkpoint,&rf),"separate replacement fixture");
    std::shared_ptr<const SourceCheckpoint> b_open;good(open_source_checkpoint(bf.directory,&b_open),"actual B before replacement");
    mutate_physical_payload(rf.directory);
    need(::rename((rf.directory+"/model.safetensors").c_str(),(bf.directory+"/model.safetensors").c_str())==0,"actual post-open replacement witness");
    std::unique_ptr<const InspectedSource> refused;std::string why;
    need(inspect_checkpoint_source(a,bf.directory,&refused,&why)==RAD_E_INVAL&&!refused&&why=="mixed Checkpoint and inspected directory identity","mixed mapped A / altered directory B authority admitted");++source_refusals;
    why.clear();need(inspect_checkpoint_source(b_open,bf.directory,&refused,&why)==RAD_E_STATE&&!refused&&why=="opened source artifact changed or replaced","post-open same-header altered-payload replacement admitted");++source_refusals;
    std::shared_ptr<const SourceCheckpoint> bad_current;good(open_source_checkpoint(bf.directory,&bad_current),"actual bounded altered-current source metadata open");
    why.clear();need(inspect_checkpoint_source(bad_current,bf.directory,&refused,&why)==RAD_E_FORMAT&&!refused&&why=="current physical file payload differs from inspected mapping contract","actual current altered payload audit not witnessed");++source_refusals;
    const auto* occupied=a.get();auto nonempty=a;need(open_source_checkpoint(bf.directory,&nonempty)==RAD_E_INVAL&&nonempty.get()==occupied,"occupied opened-source owner overwritten");
}
rad::Plan metadata_plan(const rad::Program& p){rad::Config config;config.tp=1;config.max_tok=T;config.max_seqs=1;config.max_ctx=T;config.placement="all_vram";config.deterministic=true;
    config.vram_weights_mib=4;config.vram_kv_mib=1;config.embedding_resolved=rad::TablePlacement::Vram;config.embedding_placement=rad::TablePlacement::Vram;
    rad::PlannerInput input{&p,&config,0,1,nullptr,0,false,true};rad::Plan plan;good(rad::Planner(input).plan(&plan),"binding case actual pure metadata plan");return plan;}
void with_binding_graph(rad::Registry& registry,const rad::Plugin& architecture,rad::Checkpoint& ck,BoundSources& sources,int stale,
                        const std::function<void(const rad::Program&,const rad::Plan&)>& body){
    rad::Builder builder(registry,ck.meta().m,context());const auto actual=sources.fn();bool swapped=false;
    builder.set_sources([&](const std::string& name,const rad::Program& prog,rad::WeightSource* out){
        if(stale<0||swapped||name!="blk.0.ffn_gate_up_exps.0.weight")return actual(name,prog,out);
        auto& maps=builder.program().name_map;const auto found=std::find_if(maps.begin(),maps.end(),[&](const auto& m){return m.declared&&name==m.declared;});
        need(found!=maps.end()&&found->n_src==2,"stale-cache physical name-map seam");const auto old=*found;
        struct Restore {RadNameMap* target;RadNameMap saved;~Restore(){*target=saved;}} restore{&*found,old};
        if(stale==0)std::swap(found->src[0],found->src[1]);else{found->src[0]="tiny.layers.0.experts.1.gate_proj.weight";found->src[1]="tiny.layers.0.experts.1.up_proj.weight";}
        swapped=true;++stale_witnesses;return actual(name,prog,out);
    });builder.set_instances(false);good(architecture.arch_declare(&builder,&ck.meta().m,&builder.program().ctx),"binding actual tiny declaration");
    begin_trace();{rad::Sampler sampler;good(sampler.declare(&builder,sampler_config()),"binding actual same-Builder sampler");good(builder.finish(),"binding finished Program");
        if(stale>=0){need(swapped,"intended stale source cache seam not reached");const auto* cached=sources.find("blk.0.ffn_gate_up_exps.0.weight");
            need(cached&&cached->pieces.size()==2&&cached->pieces[0]->name==(stale==0?"tiny.layers.0.experts.0.up_proj.weight":"tiny.layers.0.experts.1.gate_proj.weight"),"actual wrong ordered cache witness absent");
            const auto m=std::find_if(builder.program().name_map.begin(),builder.program().name_map.end(),[](const auto& x){return x.declared&&!std::strcmp(x.declared,"blk.0.ffn_gate_up_exps.0.weight");});
            need(m!=builder.program().name_map.end()&&!std::strcmp(m->src[0],"tiny.layers.0.experts.0.gate_proj.weight")&&!std::strcmp(m->src[1],"tiny.layers.0.experts.0.up_proj.weight"),"stale cache current map was not restored exactly");}
        auto plan=metadata_plan(builder.program());body(builder.program(),plan);
    }end_trace();
}
void cached_source_binding_tests(rad::Registry& registry,const rad::Plugin& architecture,const ColdSourceSeal& a){++groups;
    Scratch b;FixtureFiles bf;good(write_fixture(b.checkpoint,&bf),"valid foreign binding fixture");std::shared_ptr<const SourceCheckpoint> owner_b;good(open_source_checkpoint(bf.directory,&owner_b),"valid foreign owned Checkpoint");
    std::unique_ptr<const InspectedSource> inspected_b;good(inspect_checkpoint_source(owner_b,bf.directory,&inspected_b),"valid foreign actual inspection");std::unique_ptr<const ColdSourceSeal> seal_b;good(seal_source(*inspected_b,&seal_b),"valid foreign seal");
    BoundSources a_sources(a),b_sources(*seal_b);
    with_binding_graph(registry,architecture,a.owner().checkpoint(),a_sources,-1,[&](const auto& p,const auto& plan){GraphQuote q;good(quote_declared_graph(a,p,plan,a_sources,&q),"correct A cache and A seal");q.weights=999;std::string why;
        need(quote_declared_graph(*seal_b,p,plan,a_sources,&q,&why)<0&&q.weights==999&&why=="bound resolver owns another Checkpoint","foreign seal accepted same-shape foreign source graph");++source_refusals;});
    with_binding_graph(registry,architecture,owner_b->checkpoint(),b_sources,-1,[&](const auto& p,const auto& plan){GraphQuote q;good(quote_declared_graph(*seal_b,p,plan,b_sources,&q),"correct B cache and B seal");q.weights=999;std::string why;
        need(quote_declared_graph(a,p,plan,b_sources,&q,&why)<0&&q.weights==999&&why=="bound resolver owns another Checkpoint","foreign source resolver accepted A seal");++source_refusals;});
    for(int stale=0;stale<2;++stale){BoundSources wrong(a);with_binding_graph(registry,architecture,a.owner().checkpoint(),wrong,stale,[&](const auto& p,const auto& plan){GraphQuote q;q.weights=999;std::string why;
        need(quote_declared_graph(a,p,plan,wrong,&q,&why)<0&&q.weights==999&&why=="bound cached code/scale NAME ORDER object and physical byte span","restored current maps hid stale swapped source cache");++source_refusals;});}
}
GraphQuote graph_tests(rad::Registry& registry,const rad::Plugin& architecture,rad::Checkpoint& ck,BoundSources& sources,const ColdSourceSeal& seal){++groups;
    rad::Builder builder(registry,ck.meta().m,context());builder.set_sources(sources.fn());builder.set_instances(false);
    good(architecture.arch_declare(&builder,&ck.meta().m,&builder.program().ctx),"actual tiny architecture declaration");
    begin_trace();GraphQuote quote;
    {rad::Sampler sampler;good(sampler.declare(&builder,sampler_config()),"actual admitted sampler declaration");good(builder.finish(),"actual finished Program");
      need(trace.attempts==11&&trace.successes==11&&trace.requested==18984&&trace.page_sum==49152&&trace.live==11,"actual sampler pre-plan requested backing differs from reviewed ledger");
      rad::Config config;config.tp=1;config.max_tok=T;config.max_seqs=1;config.max_ctx=T;config.placement="all_vram";config.deterministic=true;
      config.vram_weights_mib=4;config.vram_kv_mib=1;config.embedding_resolved=rad::TablePlacement::Vram;config.embedding_placement=rad::TablePlacement::Vram;
      rad::PlannerInput input{&builder.program(),&config,0,1,nullptr,0,false,true};rad::Plan plan;good(rad::Planner(input).plan(&plan),"actual pure metadata planner");
      std::string why;const auto status=quote_declared_graph(seal,builder.program(),plan,sources,&quote,&why);good(status,"actual graph/source/layout/authority quote: "+why);
      // BEGIN additive foundation private quotes; inherited case bodies below unchanged.
      rad::TinySamplerQuote sq{};good(rad::Sampler::tiny_startup_quote(sampler_config(),&sq),"actual eleven-request private Sampler quote");
      need(sq.storage.pinned_requests==trace.attempts&&sq.storage.pinned_bytes==trace.requested&&sq.storage.pinned_page_bytes==trace.page_sum,"private Sampler quote matches real staged allocation trace");
      rad::CtxDesc cd;cd.program=&builder.program();cd.record_passes=false;
      cd.expert_prepare_limits.max_rows=T;cd.expert_prepare_limits.max_experts=4;cd.expert_prepare_limits.max_top_k=2;
      cd.expert_prepare_limits.max_maps=8;cd.expert_prepare_limits.max_entries=32;
      cd.expert_prepare_limits.max_readback_bytes=4096;cd.expert_prepare_limits.max_metadata_bytes=65536;
      rad::TinyCtxQuote cq{};good(rad::Ctx::tiny_startup_quote(cd,true,&cq),"actual declared Program private Ctx quote");
      need(cq.weights==156&&cq.bands==quote.bands&&cq.prepare_maps==8&&cq.prepare_entries==32&&cq.storage.stream_count==1&&cq.sizeof_ctx&&cq.sizeof_op_plan&&cq.sizeof_issue_plan,"actual Ctx private geometry and requested storage");
      // The REAL preserved declaration is classes2. The one-class variation
      // below is deliberately metadata-only API coverage, not a re-declaration,
      // Native cache-layout class claim or numerical/Engine capability.
      struct RestoreRouteClasses {
          rad::Program& p;std::array<int,8> classes{};std::array<std::array<int,4>,8> index{};
          explicit RestoreRouteClasses(rad::Program& program):p(program){
              for(size_t r=0;r<8;++r){classes[r]=p.expert_route_maps[r].decl.classes;
                  for(size_t e=0;e<4;++e)index[r][e]=p.expert_route_maps[r].entries[e].class_index;}
          }
          ~RestoreRouteClasses(){for(size_t r=0;r<8;++r){p.expert_route_maps[r].decl.classes=classes[r];
              for(size_t e=0;e<4;++e)p.expert_route_maps[r].entries[e].class_index=index[r][e];}}
          void reset(){for(size_t r=0;r<8;++r){p.expert_route_maps[r].decl.classes=classes[r];
              for(size_t e=0;e<4;++e)p.expert_route_maps[r].entries[e].class_index=index[r][e];}}
      } route_restore(builder.program());
      for(size_t r=0;r<8;++r){need(route_restore.classes[r]==2,"preserved actual two-route-class declaration");
          for(size_t e=0;e<4;++e)need(route_restore.index[r][e]==int(e%2),"preserved actual two-class modulo map");}
      const auto unchanged_storage=[&](const rad::TinyCtxQuote& value){return value.prepare_bank==cq.prepare_bank&&
          value.prepare_maps==cq.prepare_maps&&value.prepare_entries==cq.prepare_entries&&
          value.storage.requested_total==cq.storage.requested_total&&value.storage.page_and_heap_total==cq.storage.page_and_heap_total;};
      for(int count:{1,2}){
          for(auto& route:builder.program().expert_route_maps){route.decl.classes=count;
              for(size_t e=0;e<route.entries.size();++e)route.entries[e].class_index=int(e%size_t(count));}
          rad::TinyCtxQuote variant{};good(rad::Ctx::tiny_startup_quote(cd,true,&variant),"bounded metadata-only one/two-class quote API coverage");
          need(unchanged_storage(variant),"route class hint does not change actual storage requests");route_restore.reset();
      }
      for(int fault=0;fault<7;++fault){
          auto& route=builder.program().expert_route_maps[0];
          if(fault<3)route.decl.classes=std::array<int,3>{0,3,4}[size_t(fault)];
          else if(fault==3)route.entries[0].class_index=1; // IN-RANGE wrong modulo
          else if(fault==4)route.entries[1].class_index=0; // IN-RANGE wrong modulo
          else if(fault==5)route.entries[0].class_index=-2;
          else route.entries[0].class_index=2;
          rad::TinyCtxQuote variant{};variant.weights=777;
          need(rad::Ctx::tiny_startup_quote(cd,true,&variant)<0&&variant.weights==777,"unsupported route class/modulo preserves quote output");
          route_restore.reset();
      }
      rad::TinyCtxQuote restored{};good(rad::Ctx::tiny_startup_quote(cd,true,&restored),"actual two-class source declaration restored after API-only mutations");
      need(unchanged_storage(restored),"restored actual route storage quote");
      cq.weights=777;cd.record_passes=true;need(rad::Ctx::tiny_startup_quote(cd,true,&cq)<0&&cq.weights==777,"recorded Ctx quote refuses and preserves output");
      rad::TinyOwnerQuote oq{};oq.weights=777;need(rad::NativeBackingOwner::tiny_startup_quote(builder.program(),plan,&oq)<0&&oq.weights==777,"metadata-only Plan cannot fake Native owner quote");
      // END additive foundation private quotes.
      auto& p=builder.program();const auto h=rad_weight(1);const auto w=p.weights[h]; // One bounded record; NEVER a W-sized WeightInfo snapshot.
      GraphQuote refused;refused.weights=999;const auto bad=[&](const char* label){std::string reason;need(quote_declared_graph(seal,p,plan,sources,&refused,&reason)<0&&refused.weights==999,label);++graph_refusals;};
      p.weights[h].source="wrong.source";bad("wrong selected source admitted");p.weights[h]=w;
      ++p.weights[h].decl.group.layer;bad("wrong declared movement group admitted");p.weights[h]=w;
      ++p.weights[h].stored_bytes;bad("changed actual stored extent admitted");p.weights[h]=w;
      const auto map=plan.unit_of_weight[h];plan.unit_of_weight[h]=-1;bad("foreign unit map admitted");plan.unit_of_weight[h]=map;
      const auto name_map=p.name_map.front();p.name_map.front().src[0]="wrong.physical.weight";bad("cached source hid changed physical name map");p.name_map.front()=name_map;
      const auto old_heads=p.meta.n_head;p.meta.n_head=1;bad("post-declaration fake attention admitted");p.meta.n_head=old_heads;
      const auto old_op=p.ops.back().op;p.ops.back().op="all_reduce";bad("unexpected collective admitted");p.ops.back().op=old_op;
      good(quote_declared_graph(seal,p,plan,sources,&quote),"restored actual graph quote");need(!quote.engine_eligible,"graph quote fabricated Engine eligibility");
    }end_trace();return quote;
}
void sampler_faults(rad::Registry& registry,const rad::Plugin& architecture,rad::Checkpoint& ck,BoundSources& sources){++groups;
    for(int fault=0;fault<11;++fault){rad::Builder builder(registry,ck.meta().m,context());builder.set_sources(sources.fn());builder.set_instances(false);
        good(architecture.arch_declare(&builder,&ck.meta().m,&builder.program().ctx),"sampler fault architecture declaration");begin_trace(fault);
        {rad::Sampler sampler;need(sampler.declare(&builder,sampler_config())==RAD_E_NOMEM,"sampler injected allocation did not refuse");need(trace.attempts==11&&trace.successes==10&&trace.live==0,"sampler fault not witnessed or partial backing retained");++sampler_oom;}
        end_trace();}
}
} // namespace
extern "C" void* __real_rad_dev_alloc(int64_t,int);
extern "C" void __real_rad_dev_free(void*,int);
extern "C" void* __wrap_rad_dev_alloc(int64_t bytes,int kind){
    if(!trace.enabled||kind!=RAD_MEM_HOST_PINNED||bytes<=0||bytes>8192||trace.attempts>=11){trace.unadmitted=true;return nullptr;}
    const auto attempt=trace.attempts++;if(int(attempt)==trace.fail_at)return nullptr;
    void* p=__real_rad_dev_alloc(bytes,kind);if(!p)return nullptr;
    trace.items[size_t(trace.successes)]={p,bytes};++trace.successes;++trace.live;trace.requested+=uint64_t(bytes);trace.page_sum+=(uint64_t(bytes)+4095)/4096*4096;return p;
}
extern "C" void __wrap_rad_dev_free(void* p,int kind){
    if(!p)return;bool found=false;for(auto& item:trace.items)if(item.ptr==p){item.ptr=nullptr;--trace.live;found=true;break;}
    if(!found||kind!=RAD_MEM_HOST_PINNED)trace.unadmitted=true;__real_rad_dev_free(p,kind);
}
int main(int argc,char** argv){try{
    need(argc==6&&!std::strcmp(argv[1],"--run-host")&&!std::strcmp(argv[2],"--radiance-home")&&!std::strcmp(argv[4],"--architecture"),
         "usage: tiny_startup_metadata_check --run-host --radiance-home MATCHING_COMBINED_HOST_HOME --architecture ABS_NEW_MODULE");
    need(rad::device_is_host(),"metadata checker requires matching whole HOST Core");
    const std::string home=argv[3],arch=argv[5];need(std::filesystem::path(home).is_absolute()&&std::filesystem::path(arch).is_absolute(),"absolute matching module/home paths");
    // Fixed request pre-admission precedes writer bytes, parser metadata, module
    // declaration and the real sampler's early pinning. No Engine is eligible.
    need(category_sum==135244800&&category_sum<admitted&&admitted==192*MiB,"reviewed conditional whole-Engine reservation");
    Scratch scratch;FixtureFiles made;std::string why;good(write_fixture(scratch.checkpoint,&made,&why),"synthetic fixture writer: "+why);writer_tests(made.directory,made);vocabulary_contract_tests();
    std::shared_ptr<const SourceCheckpoint> owner;good(open_source_checkpoint(made.directory,&owner),"actual bounded owned Core checkpoint open");auto& ck=owner->checkpoint();std::weak_ptr<const SourceCheckpoint> lifetime=owner;
    std::unique_ptr<const InspectedSource> inspected;good(inspect_checkpoint_source(owner,made.directory,&inspected,&why),"actual source inspection: "+why);
    std::unique_ptr<const ColdSourceSeal> seal;good(seal_source(*inspected,&seal),"cold source-only seal");cold_tests(ck,*inspected,*seal);source_file_binding_tests(owner);
    owner.reset();inspected.reset();need(!lifetime.expired()&& &seal->owner().checkpoint()==&ck,"seal did not retain exact inspected Checkpoint after caller release");
    GraphQuote q;
    {rad::Registry registry;registry.set_no_device_code(true);good(registry.load_plugin(home+"/kernels/libref.so",0),"matching actual reference schemas/kernels");good(registry.load_plugin(arch,1),"actual tiny architecture module");good(registry.check_complete(),"actual loaded schema completeness");
      const auto* plugin=registry.plugin("glm53_tiny_native_arch");need(plugin&&plugin->arch_id==arch_id&&plugin->arch_declare&&plugin->arch_step,"actual tiny architecture identity and real step symbol");
      BoundSources sources(*seal);q=graph_tests(registry,*plugin,ck,sources,*seal);sampler_faults(registry,*plugin,ck,sources);cached_source_binding_tests(registry,*plugin,*seal);}
    need(groups==7&&shape_refusals==24&&graph_refusals==7&&sampler_oom==11&&source_refusals==7&&stale_witnesses==2&&vocab_refusals==10&&vocab_witnesses==5&&!trace.enabled&&!trace.unadmitted,"complete bounded metadata/source/tokenizer case matrix");
    const auto& a=q.authority;
    std::cout<<"ACTUAL_SOURCE physical="<<seal->source().physical_tensors<<" payload="<<seal->source().payload_bytes<<" header="<<seal->source().header_bytes<<" file="<<seal->source().file_bytes<<" artifacts="<<seal->source().artifact_files<<" sidecarBytes="<<seal->source().sidecar_bytes<<" vocab="<<seal->source().vocabulary<<" BOS="<<seal->source().vocab.bos<<" EOS="<<seal->source().vocab.eos<<" EOT="<<seal->source().vocab.eot<<" eogCount="<<seal->source().vocab.eog_count<<" addBOS="<<seal->source().vocab.add_bos<<" addEOS="<<seal->source().vocab.add_eos<<" sizeofRetainedSourceOwner="<<sizeof(SourceCheckpoint)<<" retainedOpenedOwner=true\n";
    std::cout<<"ACTUAL_GRAPH W="<<q.weights<<" expertViews="<<q.expert_views<<" ordinaryViews="<<q.ordinary_views<<" sources="<<q.logical_sources<<" units="<<q.units<<" modelUnits="<<q.model_units<<" layerUnits="<<q.layer_units<<" expertUnits="<<q.expert_units<<" explicitSpans="<<q.explicit_full_spans<<" rowpartViews="<<q.gate_up_rowparts<<" rectangles="<<q.selected_plane_rects<<" bands="<<q.bands<<" arenaScratch="<<q.actual_arena_scratch_bytes<<" ordinaryStored="<<q.ordinary_stored_bytes<<" ordinaryPageUpper="<<q.ordinary_page_upper<<"\n";
    std::cout<<"ACTUAL_COMPACT W="<<a.weights<<" G="<<a.geometries<<" S="<<a.strings<<" sizeofWeightInfo="<<a.geometry_size<<" sizeofRef="<<a.reference_size<<" sizeofSlice="<<a.slice_size<<" sizeofBucket="<<a.bucket_size<<" namesSources="<<a.name_source_bytes<<" tags="<<a.layout_tag_bytes<<" immutable="<<a.immutable_bytes<<" scratch="<<a.scratch_bytes<<" peak="<<a.construction_peak_bytes<<" reservation="<<a.metadata_budget<<"\n";
    std::cout<<"PASS source/header/importer/declaration/pure-plan/compact-quote only groups="<<groups<<" checks="<<checks<<" shapeRefusals="<<shape_refusals<<" graphRefusals="<<graph_refusals<<" samplerAllocationRefusals="<<sampler_oom<<" sourceBindingRefusals="<<source_refusals<<" witnessedStaleCaches="<<stale_witnesses<<" tokenizerRefusals="<<vocab_refusals<<" witnessedTokenizerSemantics="<<vocab_witnesses<<" actualSamplerRequested=18984 actualSamplerPageUpper=49152 EngineEligible=false; no Pools/NativeSource/Owner/Ctx/streams/Engine/model or sampler step; not inference/full-model provenance\n";return 0;
}catch(const std::exception& e){std::cerr<<"FAIL startup metadata checker: "<<e.what()<<"\n";return 1;}}
