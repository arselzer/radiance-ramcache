// Explicit CPU-only new bounded-decode checks. Inherited gate source stays exact.
#define main inherited_source_gate_main
#include "gate_checker.cpp"
#undef main
#include "server/sink.h"
namespace {
void output_cache_admission(){++groups;
    for(const int count:{1,2,16})good(glm_main_output_admission(count,512),"bounded output count accepts");
    for(const int count:{-1,0,17,INT32_MAX}){need(glm_main_output_admission(count,512)<0,"output bound rejects");++refusals;}
    need(glm_main_output_admission(16,16)<0,"context must hold prompt plus output bound");++refusals;
    auto base=options();need(base.hot_bytes_per_rank==(64u<<20)&&base.host_payload_limit==(8ull<<30)&&base.read_chunk_bytes==65536,"old cache/envelope/read defaults preserved");
    for(int mutation=0;mutation<6;++mutation){auto bad=base;switch(mutation){
        case 0:bad.hot_bytes_per_rank=(2ull<<30)+1;bad.host_payload_limit=16ull<<30;break;
        case 1:bad.hot_bytes_per_rank=2ull<<30;break;
        case 2:bad.hot_bytes_per_rank=2ull<<30;bad.host_payload_limit=15ull<<30;break;
        case 3:bad.host_payload_limit=(16ull<<30)+1;break;
        case 4:bad.read_chunk_bytes=4096;break;
        case 5:bad.read_chunk_bytes=(1u<<20)+1;break;}
        std::unique_ptr<Engine> e;opens=reads=allocations=0;allocation_fault::begin();witness=true;
        const int st=GlmMainEngineHarness::create(bad,&e);witness=false;const auto calls=allocation_fault::calls;allocation_fault::end();
        need(st<0&&!e&&!opens&&!reads&&!allocations&&!calls,"invalid new experiment before allocation/I/O/backend");++refusals;}
    for(const uint32_t chunk:{65536u,1048576u}){auto admitted=base;admitted.hot_bytes_per_rank=2ull<<30;admitted.host_payload_limit=16ull<<30;admitted.read_chunk_bytes=chunk;
        std::unique_ptr<Engine> e;good(GlmMainEngineHarness::create(admitted,&e),"explicit 2GiB HOT/16GiB envelope shell");
        GlmMainEngineSnapshot s;good(GlmMainEngineHarness::snapshot(*e,&s),"actual envelope quote");
        const uint64_t exact=(4ull<<30)+(64u<<20)+(8u<<20)+4*((2ull<<30)+8*3342336ull+30228480ull);
        need(s.host_payload_reserved==exact&&exact<=admitted.host_payload_limit,"all four HOT/COLD/landing and loader named budgets included");
        std::array<int32_t,16> out;out.fill(777);opens=reads=allocations=0;allocation_fault::begin();witness=true;
        const int st=GlmMainEngineHarness::generate(*e,"x",out);witness=false;const auto calls=allocation_fault::calls;allocation_fault::end();
        need(st<0&&std::all_of(out.begin(),out.end(),[](int32_t t){return t==777;})&&!opens&&!reads&&!allocations&&!calls,"unstarted generate preserves every caller token before work");++refusals;
        good(GlmMainEngineHarness::retire(*e),"shell retirement");}
    server::Sink sink(32);for(int32_t i=0;i<16;++i)need(sink.push(i),"whole output bound fits without a consumer");sink.finish(server::Finish::Length);
    std::array<server::SinkToken,16> got{};need(sink.read(got.data(),got.size())==16&&!sink.overflowed()&&sink.n_produced()==16,"bounded sink never needs ring-full recovery");
    for(size_t i=0;i<got.size();++i)need(got[i].token==int32_t(i),"all sink tokens exact");
}
void scheduler_decode(){++groups;
Registry registry;RadModelMeta meta{};meta.arch_id="glm5next";meta.n_layers=45;meta.n_embd=4096;meta.n_expert=288;meta.n_expert_used=8;meta.n_vocab=154880;
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

    ChunkGeometry actual;good(glm_main_final_token_geometry_resolve(p,cfg,&actual),"actual final-token geometry");
    Pools pools;Config pcfg=cfg;pcfg.vram_weights_mib=1;pcfg.vram_kv_mib=1;pcfg.host_pool_mib=0;pcfg.embedding_host_mib=pcfg.tower_host_mib=0;
    good(pools.configure(pcfg,16u<<20,&rad_device_allocator()),"real HOST pools");KVManager kv;good(kv.configure(p.kv_groups,cfg,pools.kv()),"real small KV");
    PrefixCache prefix;good(prefix.configure(cfg,&kv,0),"prefix off");KVBinding<KVManager> binding(&kv);PCBinding<PrefixCache> cache(&prefix);
    RadStream stream=nullptr;good(rad_stream_create(&stream,0),"real HOST scheduler stream");
    {Scheduler scheduler;const std::vector<BatchBuilder::RankIO> io{{0,stream,&p}};good(scheduler.init(p,cfg,&binding,&cache,io,&actual),"actual Scheduler maxTok1");
        for(const int outputs:{2,16}){server::Sink sink(32);auto request=std::make_unique<Request>();request->id=uint64_t(outputs);request->prompt={4};request->max_tokens=outputs;request->sink=&sink;request->ignore_eos=true;request->may_speculate=false;
            good(scheduler.add(std::move(request)),"single prompt/request admission");
            for(int i=0;i<outputs;++i){const auto* batch=scheduler.step();need(batch&&batch->n_tok==1&&batch->n_seq==1&&batch->phase==(i?RAD_PHASE_DECODE:RAD_PHASE_PREFILL),"actual PREFILL1 then DECODE1 metadata");
                const auto& plan=scheduler.plan();need(plan.e.size()==1&&plan.e[0].ctx_len==i&&!plan.e[0].dev&&plan.e[0].n_tokens==1&&!plan.e[0].n_spec,"real scheduler position/capacity");
                const auto& e=plan.e[0];const int32_t input=i?e.req->output.back():e.req->prompt[0];need(input==(i?99+i:4),"decode consumes prior committed sample");
                int32_t token=100+i,accepted=0;uint8_t eos=0;Scheduler::StepResult result;result.n_seq=1;result.n_accepted=&accepted;result.token=&token;result.eos=&eos;
                good(rad_stream_sync(stream),"actual HOST batch upload before commit/reuse");good(scheduler.commit(result),"actual scheduler commit");scheduler.step_finished();}
            need(sink.done()&&sink.n_produced()==outputs&&!sink.overflowed(),"length completion exact");std::array<server::SinkToken,16> values{};
            need(sink.read(values.data(),values.size())==size_t(outputs),"all committed samples returned");for(int i=0;i<outputs;++i)need(values[size_t(i)].token==100+i,"committed samples exact");
            good(scheduler.reap(uint64_t(outputs)),"reap completed request");}
        scheduler.fini();}
    good(rad_stream_sync(stream),"actual scheduler staging retired");rad_stream_destroy(stream);
}
void completed_frames(){++groups;
    std::unique_ptr<CachePrepare4Session> session;good(CachePrepare4ProtocolHarness::create(17,19,&session),"protocol-only Session no runtime capability");
    for(int frame=0;frame<2;++frame){RadBatch batch{};batch.step=frame;batch.n_tok=batch.n_seq=batch.n_out=1;batch.phase=frame?RAD_PHASE_DECODE:RAD_PHASE_PREFILL;
        good(session->begin(batch),"next actual frame admission");std::atomic<bool> done=false,timed_out=false;std::array<int,4> status{};
        std::thread watchdog([&]{const auto until=std::chrono::steady_clock::now()+std::chrono::seconds(5);while(!done.load()&&std::chrono::steady_clock::now()<until)std::this_thread::sleep_for(std::chrono::milliseconds(1));if(!done.load()){timed_out=true;(void)session->binding(0)->finish(batch,RAD_E_IO);}});
        std::array<std::thread,4> workers;for(int rank=0;rank<4;++rank)workers[size_t(rank)]=std::thread([&,rank]{auto* b=session->binding(rank);const std::array<int32_t,8> ids{0,1,2,3,4,5,6,7};
            for(int layer=3;layer<=44;++layer){int st=b->enter(layer,0,1,ids);if(st<0){status[size_t(rank)]=st;return;}st=b->ready(RAD_OK);if(st<0){status[size_t(rank)]=st;return;}}
            status[size_t(rank)]=b->finish(batch,RAD_OK);});
        for(auto& worker:workers)worker.join();done=true;watchdog.join();need(!timed_out.load()&&std::all_of(status.begin(),status.end(),[](int s){return s==RAD_OK;}),"42 gates/all ranks without watchdog rescue");
        const auto before=session->observe();need(session->active()&&!before.active&&before.ordinal==42&&before.finished==15&&before.epoch==uint64_t(frame+1),"Session remains active through sampler/collect boundary");
        good(session->complete_frame(batch),"explicit final collect completion");need(!session->active(),"frame closes once");}
    session->stop_admission();
    std::unique_ptr<CachePrepare4Session> premature;good(CachePrepare4ProtocolHarness::create(31,37,&premature),"separate negative protocol state");RadBatch b{};b.step=0;b.n_tok=b.n_seq=b.n_out=1;b.phase=RAD_PHASE_PREFILL;
    good(premature->begin(b),"begin negative frame");need(premature->begin(b)<0,"no next begin before sampler/collect completion");++refusals;
}
}
int main(int argc,char** argv){try{need(argc==2&&!std::strcmp(argv[1],"--run-host-decode"),"explicit --run-host-decode required");need(device_is_host(),"HOST backend required");output_cache_admission();scheduler_decode();completed_frames();
    std::cout<<"PASS bounded one-input decode HOST checks: "<<groups<<" groups "<<checks<<" checks "<<refusals<<" refusals; no source/model/GPU correctness claim\n";return 0;
}catch(const std::exception& e){std::cerr<<"FAIL bounded decode HOST checks: "<<e.what()<<'\n';return 1;}}
