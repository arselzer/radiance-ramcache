// Explicit HOST metadata/protocol/Scheduler proof. No checkpoint/model/SourceReady.
#define main inherited_gate_main
#include "gate_checker.cpp"
#undef main
#include "build/glm_main_profile_limits.h"
#include "server/sink.h"
namespace {
void prefill_bounds(){++groups;
    GlmMainProfileCapacity cap;good(glm_main_profile_capacity(4,&cap),"closed R4 capacity");
    need(cap.rows==4&&cap.selected_union==32&&cap.regions==33&&cap.cold_bytes==106954752&&
        cap.consumer_bytes==110837760&&cap.route_bank_bytes==2564,"independent exact R4 budgets");
    GlmMainProfileCapacity legacy;good(glm_main_profile_capacity(1,&legacy),"old R1 capacity");
    need(legacy.selected_union==8&&legacy.cold_bytes==26738688&&legacy.consumer_bytes==30228480,"old exact budgets preserved");
    for(int rows:{-1,0,2,3,5,8,128}){auto out=cap;std::array<unsigned char,sizeof out> before{};std::memcpy(before.data(),&out,sizeof out);
        need(glm_main_profile_capacity(rows,&out)<0&&!std::memcmp(before.data(),&out,sizeof out),"unsupported capacity output atomic");++refusals;}
    std::array<int32_t,65> tokens{};for(size_t i=0;i<tokens.size();++i)tokens[i]=int32_t(4+i);
    for(size_t n:{size_t(1),size_t(8),size_t(9),size_t(64)})for(int outputs:{1,16})good(glm_main_prefill_prompt_admission({tokens.data(),n},154856,512,outputs),"bounded real-ID metadata admission");
    for(int which=0;which<7;++which){auto t=tokens;size_t n=9;int64_t ctx=512;int outputs=2;
        switch(which){case 0:n=0;break;case 1:n=65;break;case 2:t[8]=-1;break;case 3:t[8]=154856;break;case 4:ctx=511;break;case 5:outputs=0;break;case 6:outputs=17;break;}
        need(glm_main_prefill_prompt_admission({t.data(),n},154856,ctx,outputs)<0,"late token/context/output bound refused");++refusals;}
    auto o=options();need(o.prefill_max_tok==1,"legacy default remains one");o.prefill_max_tok=4;
    const uint64_t exact=(4ull<<30)+(72u<<20)+4*((64u<<20)+106954752ull+110837760ull);
    std::unique_ptr<Engine> e;good(GlmMainEngineHarness::create(o,&e),"normal-prefill shell only");GlmMainEngineSnapshot q;good(GlmMainEngineHarness::snapshot(*e,&q),"shell budget snapshot");
    need(q.max_tok==4&&q.host_payload_reserved==exact&&exact==5510070272ull,"all rank COLD and landing charged before sources");good(GlmMainEngineHarness::retire(*e),"unused shell retire");
    for(int which=0;which<5;++which){auto bad=o;switch(which){case 0:bad.prefill_max_tok=8;break;case 1:bad.prefill_max_tok=0;break;case 2:bad.context=256;break;case 3:bad.host_payload_limit=exact-1;break;case 4:bad.prefill_max_tok=1;bad.formatted_prompt=true;break;}
        std::unique_ptr<Engine> out;opens=reads=allocations=0;allocation_fault::begin();witness=true;
        const int status=GlmMainEngineHarness::create(bad,&out);witness=false;const auto calls=allocation_fault::calls;allocation_fault::end();
        need(status<0&&!out&&!opens&&!reads&&!allocations&&!calls,"profile/budget-minus-one refuses before allocation/I/O/backend");++refusals;}
    need(glm_main_max_steps==80&&16+15<glm_main_max_steps&&sizeof(EngineGlmMainState)+sizeof(server::Sink)+48*sizeof(server::SinkToken)<(8u<<20),"fixed witness storage actual sizeof and31-pass closed maximum");
}
void metadata_program(Registry& registry,Program* out){
    RadModelMeta meta{};meta.arch_id="glm5next";meta.n_layers=45;meta.n_embd=4096;meta.n_expert=288;meta.n_expert_used=8;meta.n_vocab=154880;
    RadBuildCtx bc{};bc.rank=0;bc.world_size=4;bc.max_tok=4;bc.max_seqs=1;bc.max_ctx=512;
    Builder b(registry,meta,bc);b.set_instances(false);good(b.require_glm_main_tp4_cache_fail_stop(),"metadata-only profile bit");auto& p=b.program();
    p.weights.resize(1+42*288*4);p.buffers.resize(5);p.ops.resize(2);p.logits_buf=1;for(size_t i=1;i<5;++i)p.buffers[i].decl.dtype=RAD_I32;
    for(int layer=3;layer<=44;++layer){ExpertRouteMap map;auto& d=map.decl;d.layer=layer;d.canonical_experts=d.local_experts=d.regular_experts=d.n_entries=288;d.classes=2;d.top_k=8;d.max_rows=4;d.gu=1;d.counts=1;d.offsets=2;d.sorted=3;d.ids=4;map.entries.resize(288);
        for(int expert=0;expert<288;++expert){auto& e=map.entries[size_t(expert)];e.slot=e.canonical=expert;e.class_index=expert%2;for(int m=0;m<4;++m){const auto h=rad_weight(1+((layer-3)*288+expert)*4+m);e.members[m]=h;p.weights[h].decl.group={layer,expert,m/2};}}
        p.expert_route_maps.push_back(std::move(map));}
    p.kv_groups.resize(2);auto& g=p.kv_groups[1];g.name="synthetic metadata4-page";g.decl.kind=RAD_KV_FULL;g.decl.dtype=RAD_BF16;g.decl.n_head_kv=1;g.decl.head_dim=1;g.block_size=4;g.bytes_per_block=16;g.layers={0};
    *out=std::move(p); // This Program is only a HOST Scheduler fixture, never source authority.
}
Config prefill_config(){Config c;c.tp=4;c.max_tok=4;c.max_seqs=1;c.max_ctx=512;c.n_spec=0;c.prefix_cache=false;c.prefix_cache_mib=0;c.checkpoint_slots=0;c.checkpoint_interval=0;c.prefill_span=1;c.deterministic=true;c.tp_wire_exact=1;return c;}
void normal_scheduler(){++groups;Registry registry;Program p;metadata_program(registry,&p);const auto cfg=prefill_config();ChunkGeometry grid;
    good(glm_main_prefill4_geometry_resolve(p,cfg,&grid),"ordinary geometry minimum with metadata4-page");need(grid.quantum==4&&grid.attn_block==4&&grid.state_chunk==1,"actual resolver4/4/1 from fixture metadata");
    for(int which=0;which<4;++which){auto c=cfg;auto old=p.kv_groups[1].block_size;switch(which){case 0:c.max_tok=3;break;case 1:c.max_ctx=256;break;case 2:p.kv_groups[1].block_size=8;break;case 3:c.prefix_cache=true;break;}
        auto unchanged=grid;need(glm_main_prefill4_geometry_resolve(p,c,&unchanged)<0,"normal geometry/profile mismatch refuses");need(unchanged.quantum==grid.quantum&&unchanged.why==grid.why,"geometry caller untouched");p.kv_groups[1].block_size=old;++refusals;}
    Pools pools;auto pcfg=cfg;pcfg.vram_weights_mib=1;pcfg.vram_kv_mib=1;pcfg.host_pool_mib=0;pcfg.embedding_host_mib=pcfg.tower_host_mib=0;good(pools.configure(pcfg,16u<<20,&rad_device_allocator()),"real HOST Pools");
    KVManager kv;good(kv.configure(p.kv_groups,cfg,pools.kv()),"real tiny HOST KV");PrefixCache prefix;good(prefix.configure(cfg,&kv,0),"prefix off");KVBinding<KVManager> mem(&kv);PCBinding<PrefixCache> cache(&prefix);
    RadStream stream=nullptr;good(rad_stream_create(&stream,0),"real HOST staging stream");
    {Scheduler scheduler;const std::vector<BatchBuilder::RankIO> io{{0,stream,&p}};good(scheduler.init(p,cfg,&mem,&cache,io,&grid),"real Scheduler normal4");
        uint64_t id=10;for(const int prompt:{8,9,10,11,64})for(const int outputs:{1,2,16}){server::Sink sink(32);auto request=std::make_unique<Request>();request->id=++id;request->prompt.resize(size_t(prompt));for(int i=0;i<prompt;++i)request->prompt[size_t(i)]=4+i;
            request->max_tokens=outputs;request->sink=&sink;request->ignore_eos=true;request->may_speculate=false;good(scheduler.add(std::move(request)),"normal8+ request");
            int position=0,produced=0,passes=0;while(!sink.done()){need(passes<31,"bounded normal pass count");const auto* batch=scheduler.step();need(batch&&batch->n_seq==1,"actual normal batch");
                const auto& e=scheduler.plan().e[0];const bool pf=position<prompt;const int rows=pf?std::min(4,prompt-position):1;const bool sample=!pf||position+rows==prompt;
                need(batch->n_tok==rows&&batch->n_out==int(sample)&&batch->phase==(pf?RAD_PHASE_PREFILL:RAD_PHASE_DECODE)&&
                    e.ctx_len==position&&e.n_tokens==rows&&e.is_prefill==pf&&e.produces_token==sample&&!e.dev,"truthful normal/nonfinal/tail metadata");
                good(rad_stream_sync(stream),"real HOST staging completed before readback/reuse");
                for(int row=0;row<rows;++row){need(batch->positions[row]==position+row,"actual staged positions");const int input=pf?4+position+row:100+produced-1;need(batch->token_ids[row]==input,"actual staged prompt/decode input");}
                if(!sample)need((position+rows)%4==0&&sink.n_produced()==produced,"nonfinal4 alignment and no fake sample");
                int32_t token=sample?100+produced:777,accepted=0;uint8_t eos=0;Scheduler::StepResult result;result.n_seq=1;result.n_accepted=&accepted;result.token=&token;result.eos=&eos;
                good(scheduler.commit(result),"actual normal commit");scheduler.step_finished();position+=rows;++passes;if(sample)++produced;need(sink.n_produced()==produced,"only truthful sampled passes publish");}
            need(produced==outputs&&passes==(prompt+3)/4+outputs-1&&position==prompt+outputs-1&&!sink.overflowed(),"complete prompt and decode progress");
            std::array<server::SinkToken,16> got{};need(sink.read(got.data(),got.size())==size_t(outputs),"exact output count");for(int i=0;i<outputs;++i)need(got[size_t(i)].token==100+i,"no nonfinal poison token leaked");good(scheduler.reap(id),"reap normal request");}
        scheduler.fini();}
    good(rad_stream_sync(stream),"real HOST staging retirement");rad_stream_destroy(stream);
}
void normal_protocol(){++groups;std::unique_ptr<CachePrepare4Session> session;good(CachePrepare4ProtocolHarness::create(17,19,&session),"HOST protocol only, no runtime authority");
    const int rows[]{4,4,1,1},out[]{0,0,1,1};for(int pass=0;pass<4;++pass){RadBatch b{};b.step=pass;b.n_tok=rows[pass];b.n_seq=1;b.n_out=out[pass];b.phase=pass<3?RAD_PHASE_PREFILL:RAD_PHASE_DECODE;good(session->begin(b),"nonfinal/final/decode session");
        std::atomic<bool> done=false,rescue=false;std::thread watchdog([&]{const auto end=std::chrono::steady_clock::now()+std::chrono::seconds(5);while(!done.load()&&std::chrono::steady_clock::now()<end)std::this_thread::sleep_for(std::chrono::milliseconds(1));if(!done.load()){rescue=true;(void)session->binding(0)->finish(b,RAD_E_IO);}});
        std::array<int,4> status{};std::array<std::thread,4> workers;for(int rank=0;rank<4;++rank)workers[size_t(rank)]=std::thread([&,rank]{std::array<int32_t,32> ids{};for(int i=0;i<rows[pass]*8;++i)ids[size_t(i)]=i;auto* binding=session->binding(rank);
            for(int layer=3;layer<=44;++layer){int s=binding->enter(layer,0,rows[pass],{ids.data(),size_t(rows[pass]*8)});if(s<0){status[size_t(rank)]=s;return;}s=binding->ready(RAD_OK);if(s<0){status[size_t(rank)]=s;return;}}status[size_t(rank)]=binding->finish(b,RAD_OK);});
        for(auto& worker:workers)worker.join();done=true;watchdog.join();need(!rescue.load()&&std::all_of(status.begin(),status.end(),[](int s){return s==RAD_OK;}),"all42 gates all4 ranks, no rescue");good(session->complete_frame(b),"truthful n_out0 frame closes without sample");const auto state=session->observe();need(!session->active()&&state.ordinal==42&&state.finished==15&&state.epoch==uint64_t(pass+1),"complete no-output model frame");}
    session->stop_admission();
    std::unique_ptr<CachePrepare4Session> failed;good(CachePrepare4ProtocolHarness::create(29,31,&failed),"separate failedR4 protocol");
    RadBatch bad{};bad.step=0;bad.n_tok=4;bad.n_seq=1;bad.n_out=0;bad.phase=RAD_PHASE_PREFILL;good(failed->begin(bad),"begin late-rank failure frame");
    std::atomic<bool> done=false,rescue=false;std::thread watchdog([&]{const auto end=std::chrono::steady_clock::now()+std::chrono::seconds(5);while(!done.load()&&std::chrono::steady_clock::now()<end)std::this_thread::sleep_for(std::chrono::milliseconds(1));if(!done.load()){rescue=true;(void)failed->binding(0)->finish(bad,RAD_E_IO);}});
    std::array<int,4> status{};std::array<std::thread,4> workers;for(int rank=0;rank<4;++rank)workers[size_t(rank)]=std::thread([&,rank]{std::array<int32_t,32> ids{};for(size_t i=0;i<ids.size();++i)ids[i]=int32_t(i);auto* b=failed->binding(rank);
        for(int layer=3;layer<=44;++layer){int st=b->enter(layer,0,4,ids);if(st<0){status[size_t(rank)]=st;return;}st=b->ready(rank==3&&layer==44?RAD_E_IO:RAD_OK);if(st<0){status[size_t(rank)]=st;return;}}status[size_t(rank)]=RAD_E_STATE;});
    for(auto& worker:workers)worker.join();done=true;watchdog.join();need(!rescue.load()&&std::all_of(status.begin(),status.end(),[](int s){return s==RAD_E_IO;}),"late rank3/source error propagated exactly before last GU");
    const auto first=failed->observe();need(first.first.status==RAD_E_IO&&first.first.rank==3&&first.ordinal==41&&failed->begin(bad)==RAD_E_IO,"first error sticky; no next frame after failed no-output prefill");++refusals;
}
}
int main(int argc,char** argv){try{need(argc==2&&!std::strcmp(argv[1],"--run-host-prefill"),"explicit --run-host-prefill required");need(device_is_host(),"HOST only");prefill_bounds();normal_scheduler();normal_protocol();std::cout<<"PASS normal4 HOST metadata/Scheduler/protocol: "<<groups<<" groups "<<checks<<" checks "<<refusals<<" refusals; no actual source/model/GPU proof\n";return 0;}catch(const std::exception& e){std::cerr<<"FAIL normal4 HOST: "<<e.what()<<'\n';return 1;}}
