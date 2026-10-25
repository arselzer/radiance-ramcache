// Actual new Engine/Registry/geometry APIs on HOST. The large geometry object is
// explicitly synthetic METADATA coverage, never a source/DEVICE/Ready proof.
#include "engine.h"
#include "engine_glm_main.h"
#include "build/rad_build.h"
#include "build/glm_main_ordinary_loader.h"
#include "format/checkpoint.h"
#include "runtime/cache_prepare4_session.h"
#include "mem/pools.h"
#include "mem/kv.h"
#include "mem/prefix.h"
#include "sched/mem_iface.h"
#include "sched/scheduler.h"
#include "regressions/compact/allocation_hooks.h"
#include <array>
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <sys/wait.h>
#include <signal.h>
#include <unistd.h>
namespace {
using namespace rad;
size_t checks=0,groups=0,refusals=0;
thread_local bool witness=false;uint64_t opens=0,reads=0,allocations=0;
void need(bool condition,const char* why){++checks;if(!condition)throw std::runtime_error(why);}
void good(int status,const char* why){need(status==RAD_OK,why);}
GlmMainEngineOptions options(){GlmMainEngineOptions o;o.checkpoint="/not-opened/model";o.home="/not-opened/home";o.bundle_root="/not-opened/bundle";o.source_proof="/not-opened/proof";o.expected_bundle_id=17;return o;}
void cold_gates(){++groups;need(device_is_host(),"HOST checker only");
    auto o=options();
    for(int mutation=0;mutation<14;++mutation){auto bad=o;
        switch(mutation){case 0:bad.expected_bundle_id=0;break;case 1:bad.checkpoint.clear();break;case 2:bad.home.clear();break;case 3:bad.bundle_root.clear();break;case 4:bad.source_proof.clear();break;case 5:bad.context=0;break;case 6:bad.context=513;break;case 7:bad.hot_bytes_per_rank=2ull<<30;break;case 8:bad.metadata_limit=1;break;case 9:bad.host_payload_limit=1;break;case 10:bad.loader_input_limit=3ull<<30;break;case 11:bad.loader_column_limit=65u<<20;break;case 12:bad.gpu_headroom=1;break;case 13:bad.checkpoint=std::string("a\0b",3);break;}
        std::unique_ptr<Engine> e;opens=reads=allocations=0;allocation_fault::begin();witness=true;
        const int st=GlmMainEngineHarness::create(bad,&e);witness=false;const auto calls=allocation_fault::calls;allocation_fault::end();
        need(st<0&&!e&&!opens&&!reads&&!allocations&&!calls,"invalid request before any source/model allocation");++refusals;
    }
    std::unique_ptr<Engine> e;good(GlmMainEngineHarness::create(o,&e),"valid closed request shell");
    // Warm backend detection above; bracket actual start, not fixture setup.
    opens=reads=allocations=0;allocation_fault::begin();witness=true;const int first=e->start();witness=false;const auto calls=allocation_fault::calls;allocation_fault::end();
    need(first==RAD_E_UNSUPPORTED&&!opens&&!reads&&!allocations&&!calls,"HOST actual start refuses before source/Plugins/Sampler");++refusals;
    GlmMainEngineSnapshot snap;good(GlmMainEngineHarness::snapshot(*e,&snap),"sticky snapshot");need(snap.attempts==1&&snap.first_status==first&&snap.retired&&!snap.source_admitted,"failed start terminal and retired");
    good(GlmMainEngineHarness::retire(*e),"retire idempotent");
    opens=reads=allocations=0;allocation_fault::begin();witness=true;const int again=e->start();witness=false;const auto retry_calls=allocation_fault::calls;allocation_fault::end();
    need(again==first&&!opens&&!reads&&!allocations&&!retry_calls,"same failed Engine never retries");++refusals;
    e.reset();good(GlmMainEngineHarness::create(o,&e),"fresh request");good(GlmMainEngineHarness::retire(*e),"retire before start");
    opens=reads=allocations=0;allocation_fault::begin();witness=true;const int retired=e->start();witness=false;const auto retired_calls=allocation_fault::calls;allocation_fault::end();
    need(retired==RAD_E_STATE&&!opens&&!reads&&!allocations&&!retired_calls,"retired Engine cannot spend released admission");++refusals;
}
std::atomic<int> init_calls{0},fini_calls{0};int init_mode=0;
int initialize(const RadParam*,int,int rank,int,void** out){++init_calls;if(init_mode==1)return RAD_E_IO;if(init_mode==2)throw 7;*out=reinterpret_cast<void*>(uintptr_t(rank+1));return RAD_OK;}
void finalize(void*){++fini_calls;}
void returning_hook(void* p,int status)noexcept{if(status==RAD_E_STATE)++*static_cast<int*>(p);}
struct Hook {int fd;};
void terminating_hook(void* p,int status)noexcept{const char value=status==RAD_E_IO?'I':'E';(void)::write(static_cast<Hook*>(p)->fd,&value,1);::_exit(88);}
void registry_guards(){++groups;Registry registry;RadKernelInfo info{};info.name="explicit HOST init seam";info.init=initialize;info.fini=finalize;KernelRow row;row.info=&info;row.plugin="checker";Resolved resolved;resolved.row=&row;
    int hits=0;const auto before=init_calls.load();need(registry.init_instance(resolved,0,4,returning_hook,&hits)==RAD_E_STATE&&hits==1&&init_calls==before,"returning zero-credit hook cannot underflow or initialize");++refusals;
    allocation_fault::remaining=0;const int oom=registry.reserve_instance_tracking(4);allocation_fault::remaining=-1;need(oom==RAD_E_NOMEM&&registry.n_instances()==0,"tracking allocation refused before init");++refusals;
    good(registry.reserve_instance_tracking(4),"precharge actual private tracking");init_mode=0;
    std::array<Resolved,4> instances;std::array<std::thread,4> threads;
    for(int rank=0;rank<4;++rank){instances[size_t(rank)].row=&row;threads[size_t(rank)]=std::thread([&,rank]{const int st=registry.init_instance(instances[size_t(rank)],rank,4,returning_hook,&hits);if(st<0)std::abort();});}
    for(auto& t:threads)t.join();need(registry.n_instances()==4&&init_calls==before+4,"four claimed tracking slots publish without growth");registry.fini_instances();need(fini_calls==4&&!registry.n_instances(),"actual successful init/fini pairs");
    // Actual process-global Registry owner must survive a refused manual
    // Engine shell; this would fail the inherited unconditional close path.
    auto& global=rad_tools_registry();need(global.n_instances()==0,"test starts with no global owner");
    good(global.reserve_instance_tracking(1),"global explicit test instance credit");Resolved retained;retained.row=&row;
    good(global.init_instance(retained,0,4,returning_hook,&hits),"actual global init owner");
    const auto global_fini=fini_calls.load();
    {std::unique_ptr<Engine> e;good(GlmMainEngineHarness::create(options(),&e),"refused Engine beside actual global owner");
        need(e->start()==RAD_E_UNSUPPORTED,"HOST Engine refuses without adopting global owner");++refusals;}
    need(global.n_instances()==1&&fini_calls==global_fini,"refusal/destruction preserves another Registry owner");
    global.fini_instances();need(!global.n_instances()&&fini_calls==global_fini+1,"actual test owner retires explicitly");
    // Lowest actual Registry seam: callback terminates before logging/unwind.
    for(int mode=1;mode<=2;++mode){int pipefd[2];need(::pipe2(pipefd,O_CLOEXEC|O_NONBLOCK)==0,"owned death witness");const pid_t child=::fork();need(child>=0,"owned death child");
        if(!child){::close(pipefd[0]);init_mode=mode;Hook hook{pipefd[1]};Registry r;Resolved x;x.row=&row;if(r.reserve_instance_tracking(1)<0)::_exit(91);(void)r.init_instance(x,0,4,terminating_hook,&hook);::_exit(92);}
        ::close(pipefd[1]);int status=0;bool reaped=false;const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
        while(std::chrono::steady_clock::now()<deadline){const auto w=::waitpid(child,&status,WNOHANG);if(w==child){reaped=true;break;}if(w<0)break;std::this_thread::sleep_for(std::chrono::milliseconds(1));}
        if(!reaped){::kill(child,SIGKILL);while(::waitpid(child,&status,0)<0&&errno==EINTR){}}
        char bytes[2]{};const auto n=::read(pipefd[0],bytes,sizeof bytes);::close(pipefd[0]);
        need(reaped&&WIFEXITED(status)&&WEXITSTATUS(status)==88&&n==1&&bytes[0]==(mode==1?'I':'E'),"negative/exception init hits exact lowest hook without watchdog rescue");++refusals;
    }
}
void ordinary_source_guards(){++groups;
    std::array<uint8_t,128> code_bytes{},second_bytes{},scale_bytes{};
    SrcTensor code;code.name="HOST plain codes";code.rad_dtype=RAD_BF16;code.shape={2,4};code.data=code_bytes.data();code.bytes=16;
    SrcTensor second=code;second.name="HOST second codes";second.data=second_bytes.data();
    SrcTensor scale;scale.name="HOST affine scales";scale.rad_dtype=RAD_F32;scale.shape={2,1};scale.data=scale_bytes.data();scale.bytes=8;
    CkptWeight source;source.name="HOST metadata-only ordinary source";source.pieces={&code};source.scales={nullptr};
    source.rank=2;source.shape[0]=2;source.shape[1]=4;source.rows=2;source.cols=4;source.trivial=true;source.enc=rad_enc_plain(RAD_BF16);
    auto positive=[&](const CkptWeight& value,bool plain){GlmMainOrdinarySourceEligibility out;
        opens=reads=allocations=0;witness=true;const int status=GlmMainOrdinaryLoader::validate_source(value,&out);witness=false;
        need(status==RAD_OK&&out.pieces==value.pieces.size()&&out.scales==value.scales.size()&&
             out.encoding_planes==uint32_t(value.enc.n_planes)&&out.canonical_plain==plain,"same production source eligibility accepts exact native storage");
        need(!opens&&!reads&&!allocations,"pure source eligibility makes no I/O/backend request");};
    auto negative=[&](const CkptWeight& value){GlmMainOrdinarySourceEligibility out;std::memset(&out,0x51,sizeof out);const auto prior=out;
        opens=reads=allocations=0;witness=true;const int status=GlmMainOrdinaryLoader::validate_source(value,&out);witness=false;
        need(status<0&&!std::memcmp(&out,&prior,sizeof out),"source eligibility refusal preserves caller bytes");
        need(!opens&&!reads&&!allocations,"source eligibility refusal precedes I/O/backend");++refusals;};
    positive(source,true); // normal resolver representation: one NULL scale per plain piece
    code.rad_dtype=RAD_F32;code.bytes=32;source.enc=rad_enc_plain(RAD_F32);positive(source,true);
    code.rad_dtype=RAD_BF16;code.shape={1,4};code.bytes=8;second.shape={1,4};second.bytes=8;source.enc=rad_enc_plain(RAD_BF16);
    source.pieces={&code,&second};source.scales={nullptr,nullptr};positive(source,true);
    auto plain=source;plain.scales[1]=&scale;negative(plain); // mixed NULL/physical is not unscaled plain
    plain=source;plain.scales.pop_back();negative(plain); // exact parallel cardinality required
    plain=source;plain.scales.push_back(nullptr);negative(plain);
    plain=source;plain.pieces[0]=nullptr;negative(plain);
    source.pieces={&code};source.scales={&scale};code.rad_dtype=RAD_F8E4M3;code.shape={2,4};code.bytes=8;
    rad_enc_clear(&source.enc);rad_enc_copy_str(source.enc.scheme,"affine");
    rad_enc_add_plane(&source.enc,"codes",RAD_F8E4M3,1,1);rad_enc_add_plane(&source.enc,"scale",RAD_F32,1,4);
    positive(source,false); // affine FP8 requires its actual non-null physical scale
    auto bad=source;bad.scales[0]=nullptr;negative(bad);
    bad=source;bad.pieces.push_back(&code);bad.scales.push_back(nullptr);bad.shape[0]=4;bad.rows=4;negative(bad);
    bad=source;bad.scales.clear();negative(bad);
    bad=source;bad.trivial=false;bad.imported=false;negative(bad);
    bad=source;bad.paro=true;negative(bad);
    bad=source;bad.wnorm=true;negative(bad);
    bad=source;bad.derived.push_back(1);negative(bad);
    auto* saved=scale.data;scale.data=nullptr;negative(source);scale.data=saved;
    saved=code.data;code.data=nullptr;negative(source);code.data=saved;
    code.gptq_group=32;negative(source);code.gptq_group=0;
    code.ct_group=32;negative(source);code.ct_group=0;
    code.pbits=4;negative(source);code.pbits=0;
    const auto bytes=code.bytes;code.bytes=0;negative(source);code.bytes=bytes;
    bad=source;bad.shape[0]=3;negative(bad); // existing source logical-view bounds retained
    opens=reads=allocations=0;witness=true;const int absent=GlmMainOrdinaryLoader::validate_source(source,nullptr);witness=false;
    need(absent==RAD_E_INVAL&&!opens&&!reads&&!allocations,"pure source check requires caller output without side effects");++refusals;
}
void geometry_guards(){++groups;Registry registry;RadModelMeta meta{};meta.arch_id="glm5next";meta.n_layers=45;meta.n_embd=4096;meta.n_expert=288;meta.n_expert_used=8;meta.n_vocab=154880;
    RadBuildCtx bc{};bc.rank=0;bc.world_size=4;bc.max_tok=bc.max_seqs=1;bc.max_ctx=512;
    Builder builder(registry,meta,bc);builder.set_instances(false);good(builder.require_glm_main_tp4_cache_fail_stop(),"metadata-only policy bit");auto& p=builder.program();
    p.weights.resize(1+42*288*4);p.buffers.resize(5);p.ops.resize(2);p.logits_buf=1;
    for(size_t b=1;b<5;++b)p.buffers[b].decl.dtype=RAD_I32;
    for(int layer=3;layer<=44;++layer){ExpertRouteMap map;auto& d=map.decl;d.layer=layer;d.canonical_experts=d.local_experts=d.regular_experts=d.n_entries=288;d.classes=2;d.top_k=8;d.max_rows=1;d.gu=1;d.counts=1;d.offsets=2;d.sorted=3;d.ids=4;map.entries.resize(288);
        for(int expert=0;expert<288;++expert){auto& e=map.entries[size_t(expert)];e.slot=e.canonical=expert;e.class_index=expert%2;for(int m=0;m<4;++m){const auto h=rad_weight(1+((layer-3)*288+expert)*4+m);e.members[m]=h;p.weights[h].decl.group={layer,expert,m/2};}}
        p.expert_route_maps.push_back(std::move(map));}
    p.kv_groups.resize(2);auto& group=p.kv_groups[1];group.name="metadata-only paged quantum";group.decl.kind=RAD_KV_FULL;group.decl.dtype=RAD_BF16;group.decl.n_head_kv=1;group.decl.head_dim=1;group.block_size=128;group.bytes_per_block=512;group.layers={0};
    good(glm_main_tp4_cache_program_profile(p),"closed shape metadata only; no source/device proof");
    Config cfg;cfg.tp=4;cfg.max_tok=cfg.max_seqs=1;cfg.max_ctx=512;cfg.n_spec=0;cfg.prefix_cache=false;cfg.prefix_cache_mib=0;cfg.checkpoint_slots=0;cfg.checkpoint_interval=0;cfg.prefill_span=1;cfg.deterministic=true;cfg.tp_wire_exact=1;
    ChunkGeometry old;need(chunk_geometry_resolve(p,cfg,&old)==RAD_E_INVAL&&old.quantum==128&&old.min_max_tok==128,"ordinary resolver retains original Q minimum");++refusals;
    ChunkGeometry actual;good(glm_main_final_token_geometry_resolve(p,cfg,&actual),"closed final-token metadata admission");need(actual.quantum==128&&actual.attn_block==128&&actual.min_max_tok==128&&!actual.checkpoint_interval,"new admission preserves real Q and minimum");
    for(int mutation=0;mutation<9;++mutation){Config bad=cfg;switch(mutation){case 0:bad.tp=1;break;case 1:bad.max_tok=2;break;case 2:bad.max_seqs=2;break;case 3:bad.n_spec=1;break;case 4:bad.prefix_cache=true;break;case 5:bad.checkpoint_slots=1;break;case 6:bad.checkpoint_interval=128;break;case 7:bad.prefill_span=2;break;case 8:bad.tp_wire_exact=0;break;}ChunkGeometry untouched;untouched.quantum=777;need(glm_main_final_token_geometry_resolve(p,bad,&untouched)<0&&untouched.quantum==777,"closed-geometry refusal preserves caller");++refusals;}
    const std::array<int32_t,1> one{4};const std::array<int32_t,2> two{4,5};good(glm_main_one_prompt_admission(one,154880,512),"one actual token admitted");need(glm_main_one_prompt_admission(two,154880,512)==RAD_E_SHAPE,"actual Engine prompt guard rejects more than one token");++refusals;
    Pools pools;Config pcfg=cfg;pcfg.vram_weights_mib=1;pcfg.vram_kv_mib=1;pcfg.host_pool_mib=0;pcfg.embedding_host_mib=pcfg.tower_host_mib=0;
    good(pools.configure(pcfg,16u<<20,&rad_device_allocator()),"real HOST pool for pure chunker initialization");KVManager kv;good(kv.configure(p.kv_groups,cfg,pools.kv()),"real tiny KV owner for chunker");PrefixCache prefix;good(prefix.configure(cfg,&kv,0),"prefix explicitly off");KVBinding<KVManager> binding(&kv);PCBinding<PrefixCache> cache(&prefix);Scheduler scheduler;RadStream stream=nullptr;good(rad_stream_create(&stream,0),"real HOST stream");
    const std::vector<BatchBuilder::RankIO> io{{0,stream,&p}};good(scheduler.init(p,cfg,&binding,&cache,io,&actual),"actual Scheduler consumes preserved geometry");int64_t checkpoint=-1;
    need(scheduler.plan_chunk(0,1,1,&checkpoint)==1,"existing final partial row allowed");need(scheduler.plan_chunk(0,2,1,&checkpoint)==0,"existing nonfinal sub-quantum remains refused");
    scheduler.fini();good(rad_stream_sync(stream),"real HOST scheduler staging retired");rad_stream_destroy(stream);
}
}
extern "C" int __real_open(const char*,int,...);
extern "C" int __wrap_open(const char* name,int flags,...){if(witness)++opens;if(flags&O_CREAT){va_list a;va_start(a,flags);const int mode=va_arg(a,int);va_end(a);return __real_open(name,flags,mode);}return __real_open(name,flags);}
extern "C" ssize_t __real_pread(int,void*,size_t,off_t);
extern "C" ssize_t __wrap_pread(int f,void* p,size_t n,off_t o){if(witness)++reads;return __real_pread(f,p,n,o);}
extern "C" void* __real_rad_dev_alloc(int64_t,int);
extern "C" void* __wrap_rad_dev_alloc(int64_t n,int k){if(witness)++allocations;return __real_rad_dev_alloc(n,k);}
int main(int argc,char** argv){try{need(argc==2&&!std::strcmp(argv[1],"--run-source-gates"),"explicit --run-source-gates required");cold_gates();registry_guards();geometry_guards();ordinary_source_guards();std::cout<<"PASS actual Engine cold/lifecycle/Registry/geometry/source HOST gates: "<<groups<<" groups "<<checks<<" checks "<<refusals<<" refusals; no complete source or inference claim\n";return 0;}catch(const std::exception& e){std::cerr<<"FAIL actual Engine HOST gates: "<<e.what()<<'\n';return 1;}}
