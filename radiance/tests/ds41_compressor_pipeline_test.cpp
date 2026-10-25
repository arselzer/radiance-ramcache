/* Actual HOST Builder/Ctx compressor -> finish -> emitted store, with real KVManager,
 * BatchBuilder I32 coordinates and persistent opaque U32 state. No payload readback between
 * operators. Supplied projections/gains/pairs only; no model/QDQ/distributed graph claim.
 * Real Pools has a fixed minimum2MiB CPU backing (1MiB unused weights +1MiB KV).
 * Capped used KV/cache/state, guarded tensors and batch slab are separately bounded<1MiB;
 * no large model payload/weights are introduced. Parent owns compilation/execution. */
#define SCHED_FAKES_KV_ONLY
#include "sched_fakes.h"
#include "build/rad_build.h"
#include "runtime/ctx.h"
#include "runtime/kvbind.h"
#include "mem/pools.h"
#include "../libds41/ds41_compress.h"
#include "../libds41/ds41_compress_finish.h"
#include "../arch/deepseek41/state_layout.h"
#include "../arch/common/rad_ds41_ref.h"
#include <array>
#include <bit>
#include <memory>
namespace pipe_test {
using namespace rad;
namespace state=rad::deepseek41::state;
namespace ref=rad::deepseek41::ref;
constexpr int64_t D=8,R=2,B=4,P=16,rotary=4,maxT=8,maxS=3;
constexpr uint8_t poison=0xA5;
void need(bool value,const std::string& message){if(!value)throw std::runtime_error(message);}
void good(int rc,const char* message){need(rc==RAD_OK,std::string(message)+": "+rad_strerror(rc));}
float projection(int seq,int64_t pos,int64_t d){return float(((seq+1)*31+(pos+1)*13+d*7)%71-35)/32.f;}
float score(int seq,int64_t pos,int64_t d){return float(((seq+2)*17+(pos+1)*11+d*5)%59-29)/8.f;}
float cast(float x,uint32_t dtype){uint64_t word=0;rad_store_f32(&word,dtype,0,x);return rad_load_f32(&word,dtype,0);}
struct Buffer {rad_buf handle=0;uint32_t dtype=0;int64_t bytes=0,offset=0,off_elements=0;void* base=nullptr;};
enum Tensor {projected=0,scores,pooled,emits,firsts,gain,pairs,unrotated,rotated,status,n_buffers};
struct Pipeline;
thread_local Pipeline* active=nullptr;
void step(RadCtx*,const RadBatch*);
struct Pipeline {
    Registry registry;std::unique_ptr<Builder> builder;Config cfg;Pools pools;KVManager kv;BatchBuilder batch;ChunkGeometry chunk;
    std::vector<KVPoolBinding> bindings;RadStream stream=nullptr;Ctx ctx;bool batch_ready=false,fatal=false;
    uint32_t dtype;rad_kvgroup full=0,pending=0;std::array<Buffer,n_buffers> buffers;std::array<rad_op,3> ops{};
    std::array<int,3> calls{};std::vector<RadLaunchFn> launches;std::vector<float> norm_weights;
    std::array<std::unique_ptr<ref::Compressor>,2> oracle;
    std::vector<uint8_t> expected_kv;int64_t logical_receipt=0;state::Layout layout;int ordinal=0;
    explicit Pipeline(uint32_t activation=RAD_F32):dtype(activation) {
        need(device_is_host(),"pipeline must use real HOST backend");good(registry.load_plugin(home()+"/experimental/kernels/libds41_ref.so",0),"load real reference plugin");
        RadModelMeta meta{};meta.arch_id="ds41-compressor-pipeline-fixture";meta.name="synthetic";meta.quant="none";meta.n_layers=1;meta.n_embd=D;meta.n_ctx_train=P;meta.n_vocab=128;
        RadBuildCtx bc{};bc.rank=0;bc.world_size=1;bc.max_tok=maxT;bc.max_seqs=maxS;bc.max_ctx=P;bc.scope="";
        builder=std::make_unique<Builder>(registry,meta,bc);builder->set_instances(false);
        good(state::make_layout(D,R,B,rad_dtype_bits(dtype)/8,&layout),"make latent geometry");RadKVGroupDecl fd{},sd{};
        good(state::paged_decl(layout,dtype,RAD_KV_FULL,0,&fd),"make FULL group declaration");good(rad::ds41::compress::state_decl(D,RAD_U32,0,&sd),"make opaque U32 declaration");
        full=rad_decl_kv_group(builder.get(),"compressed",&fd);pending=rad_decl_kv_group(builder.get(),"pending",&sd);need(full&&pending,"Builder refused KV groups");
        good(rad_bind_layer_kv(builder.get(),0,full),"bind FULL layer");good(rad_bind_layer_kv(builder.get(),0,pending),"bind LINEAR layer");
        declare_buffer(projected,RAD_F32,{maxT,D});declare_buffer(scores,RAD_F32,{maxT,D});declare_buffer(pooled,RAD_F32,{maxT,D});
        declare_buffer(emits,RAD_I32,{maxT});declare_buffer(firsts,RAD_I32,{maxT});declare_buffer(gain,RAD_F32,{D});declare_buffer(pairs,RAD_F32,{P,rotary/2,2});
        declare_buffer(unrotated,dtype,{maxT,D});declare_buffer(rotated,dtype,{maxT,D});declare_buffer(status,RAD_I32,{maxT});
        const RadParam cp[]{RAD_RANGE("M",1,maxT),RAD_INT("dim",D),RAD_INT("ratio",R),RAD_STR("dtype","f32"),RAD_INT("n_spec",0)};
        const RadParam fp[]{RAD_RANGE("M",1,maxT),RAD_INT("n",D),RAD_INT("rotary_dim",rotary),RAD_INT("ratio",R),RAD_STR("a_dtype","f32"),RAD_STR("dtype",rad_dtype_name(dtype)),RAD_F64("eps",1e-6)};
        const RadParam sp[]{RAD_RANGE("M",1,maxT),RAD_INT("dimension",D),RAD_INT("ratio",R),RAD_INT("block_size",B),RAD_STR("dtype",rad_dtype_name(dtype))};
        ops[0]=rad_decl_op(builder.get(),"ds41_compress_pool",cp,5,nullptr,0);ops[1]=rad_decl_op(builder.get(),"ds41_compress_finish",fp,7,nullptr,0);
        ops[2]=rad_decl_op(builder.get(),"ds41_latent_store_emitted",sp,5,nullptr,0);for(auto op:ops)need(op!=0,"real registered pipeline row did not resolve");
        hazards(ops[0],{projected,scores},{pooled,emits,firsts});hazards(ops[1],{pooled,emits,firsts,gain,pairs},{unrotated,rotated});hazards(ops[2],{rotated,emits},{status});
        good(builder->finish(),"finish real pipeline Builder");auto& program=builder->program();need(program.kv_groups[full].block_size==B,"resolved cache block size differs");
        for(auto op:ops)for(const auto& band:program.ops[op].bands){const auto& row=band.dom[RAD_DOMAIN_HOST];need(row.row&&row.row->reference&&row.row->plugin=="libds41_ref"&&!row.row->deferred_status,"real HOST pipeline row/marker missing");launches.push_back(row.row->info->launch);}
        cfg.max_tok=maxT;cfg.max_seqs=maxS;cfg.max_ctx=P;cfg.n_spec=0;cfg.vram_kv_mib=1;cfg.vram_weights_mib=1;cfg.host_pool_mib=0;cfg.checkpoint_slots=0;cfg.prefix_cache=false;
        good(pools.configure(cfg,0,&rad_malloc_allocator(),&rad_malloc_allocator()),"configure minimum2MiB HOST pools");
        need(pools.weights().bytes()+pools.kv().bytes()==2*(1<<20),"unexpected fixed CPU pool backing");
        good(kv.plan_groups(program.kv_groups,cfg),"plan real small KV groups");kv.cap_paged_tokens(64);good(kv.carve(pools.kv()),"carve capped real KV groups");
        good(rad_stream_create(&stream,0),"create real HOST stream");for(uint64_t id:{10,20,30}){good(kv.add_sequence(id),"admit sequence");good(kv.ensure(id,1),"allocate initial slots");}
        logical_receipt=pools.kv().used()+256;need(logical_receipt>0&&logical_receipt<65536&&logical_receipt<=pools.kv().bytes(),"logical KV/cache/state payload exceeds tiny receipt");
        std::memset(pools.kv().base(),0xC7,size_t(logical_receipt));
        for(const auto& clear:kv.state_clears()){const auto* p=kv.plan(clear.group);for(int layer=0;layer<p->n_layers;++layer)good(rad_memset_async(kv.state_ptr(clear.group,layer,clear.slot),0,p->layer_bytes_per_state(),stream),"zero fresh opaque state");}kv.clear_state_clears();good(rad_stream_sync(stream),"finish fresh state zeroing");
        // A third sequence's state is a raw-word guard, never used by this pipeline.
        auto* unused=static_cast<uint32_t*>(kv.state_ptr(pending,0,kv.state_slot(30,pending)));for(int i=0;i<2*(D+1);++i)unused[i]=0xDEAD0000u+uint32_t(i);
        bind_kv_pools(program,[&](int32_t g){return kv.plan(g);},&bindings);const auto geometry=KVGeom::of(program.kv_groups);
        need(BatchBuilder::device_bytes(program,cfg,geometry)>0&&BatchBuilder::device_bytes(program,cfg,geometry)<65536,"batch slab is not tiny");
        good(batch.init(program,cfg,chunk,geometry,{{0,stream,&program}}),"init actual BatchBuilder");batch_ready=true;
        CtxDesc cd{};cd.program=&program;cd.rank=0;cd.world_size=1;cd.stream=stream;cd.arch_step=step;cd.record_passes=false;cd.kv=bindings;
        good(ctx.init(cd),"init actual HOST Ctx");good(ctx.bind_thread(),"bind HOST thread");need(!ctx.checks_device_status(),"HOST graph incorrectly needs GPU collector");
        int64_t tensors_bytes=0;for(auto& b:buffers){b.base=rad_buf_ptr(&ctx,b.handle);need(b.base!=nullptr,"HOST tensor arena absent");tensors_bytes+=b.bytes;}
        const auto activation_receipt=arena_plan_bytes(program,nullptr,nullptr)+align_up(arena_host_bytes(program),RAD_ALIGN_UNIT);
        need(activation_receipt<65536&&tensors_bytes+activation_receipt+logical_receipt+BatchBuilder::device_bytes(program,cfg,geometry)<(1<<20),"logical synthetic payload exceeds1MiB");
        norm_weights.resize(D);for(int d=0;d<D;++d)norm_weights[size_t(d)]=1.f+float(d-3)/32.f;
        for(int seq=0;seq<2;++seq)oracle[size_t(seq)]=std::make_unique<ref::Compressor>(R,norm_weights,1e-6f,"seq"+std::to_string(seq),dtype==RAD_F32?ref::Precision::f32:ref::Precision::bf16);
        const auto* begin=static_cast<const uint8_t*>(pools.kv().base());expected_kv.assign(begin,begin+logical_receipt);
    }
    ~Pipeline(){if(stream&&rad_stream_sync(stream)<0)std::abort();ctx.shutdown();if(batch_ready)batch.fini();if(stream)rad_stream_destroy(stream);}
    void declare_buffer(int index,uint32_t dtype_value,std::initializer_list<int64_t> shape){
        auto& b=buffers[size_t(index)];b.dtype=dtype_value;RadBufDecl d{};d.dtype=dtype_value;d.rank=shape.size();d.kind=RAD_BUF_PERSIST;d.domain=RAD_DOMAIN_HOST;
        unsigned axis=0;for(auto n:shape)d.shape[axis++]=n;int64_t trailing=1;for(unsigned i=1;i<d.rank;++i)trailing*=d.shape[i];
        const int64_t width=rad_dtype_bits(dtype_value)/8,row_bytes=trailing*width,guards=(256+row_bytes-1)/row_bytes;
        b.off_elements=guards*trailing;b.offset=b.off_elements*width;d.shape[0]+=2*guards;b.bytes=d.shape[0]*row_bytes;
        b.handle=rad_decl_buffer(builder.get(),("tensor"+std::to_string(index)).c_str(),&d);need(b.handle!=0,"Builder refused guarded HOST buffer");
    }
    void hazards(rad_op op,std::initializer_list<int> reads,std::initializer_list<int> writes){std::vector<rad_buf> r,w;for(int i:reads)r.push_back(buffers[size_t(i)].handle);for(int i:writes)w.push_back(buffers[size_t(i)].handle);good(rad_op_reads(builder.get(),op,r.data(),r.size()),"declare reads");good(rad_op_writes(builder.get(),op,w.data(),w.size()),"declare writes");}
    void* data(int i){return static_cast<uint8_t*>(buffers[size_t(i)].base)+buffers[size_t(i)].offset;}
    RadOperand operand(int i,int64_t rows){return RAD_B_N(buffers[size_t(i)].handle,buffers[size_t(i)].off_elements,rows);}
    static RadOperand vector(const int32_t* p,int64_t rows){RadOperand o=RAD_P_T(p,RAD_I32);o.rows=rows;return o;}
    void execute(RadCtx* context,const RadBatch* b){
        need(b->kv[pending].state_index_pitch==1,"unexpected speculative state-index columns");
        const RadOperand pool[]{operand(projected,b->n_tok),operand(scores,b->n_tok),RAD_KV(pending,0),RAD_P_T2(b->kv[pending].state_index,RAD_I32,b->n_seq,1),vector(b->ctx_lens,b->n_seq),vector(b->cu_seqlens,b->n_seq+1),vector(b->positions,b->n_tok),RAD_NONE,operand(pooled,b->n_tok),operand(emits,b->n_tok),operand(firsts,b->n_tok)};
        ++calls[0];if(rad_issue(context,ops[0],pool,11,b->n_tok)<0)return;
        const RadOperand finish[]{operand(pooled,b->n_tok),operand(emits,b->n_tok),operand(firsts,b->n_tok),operand(gain,D),operand(pairs,P),operand(unrotated,b->n_tok),operand(rotated,b->n_tok)};
        ++calls[1];if(rad_issue(context,ops[1],finish,7,b->n_tok)<0)return;
        const RadOperand store[]{operand(rotated,b->n_tok),vector(b->positions,b->n_tok),vector(b->kv[full].slot_mapping,b->n_tok),operand(emits,b->n_tok),RAD_KV(full,0),operand(status,b->n_tok)};
        ++calls[2];(void)rad_issue(context,ops[2],store,6,b->n_tok);
    }
    enum class Fault {none,late_projection,finish_gain};
    static void expected_scalar(std::vector<uint8_t>& out,const Buffer& b,int64_t flat,float value){rad_store_f32(out.data()+b.offset,b.dtype,flat,value);}
    static void expected_int(std::vector<uint8_t>& out,const Buffer& b,int64_t flat,int32_t value){std::memcpy(out.data()+b.offset+flat*4,&value,4);}
    int run(const std::array<int,2>& order,const std::array<int,2>& starts,const std::array<int,2>& lengths,Fault fault=Fault::none){
        need(!fatal,"failed pipeline profile is fatal and cannot be reused");good(rad_stream_sync(stream),"drain before preparing original-token chunk");
        StepPlan plan;plan.phase=RAD_PHASE_PREFILL;plan.n_tok=lengths[0]+lengths[1];need(plan.n_tok>0&&plan.n_tok<=maxT,"bad tiny ragged token count");
        for(int s=0;s<2;++s){const uint64_t id=uint64_t(10+order[size_t(s)]*10);good(kv.ensure(id,starts[size_t(s)]+lengths[size_t(s)],starts[size_t(s)]),"ensure actual cache/state slots");StepEntry e;e.seq=id;e.card=s;e.ctx_len=starts[size_t(s)];e.n_tokens=lengths[size_t(s)];e.is_prefill=true;plan.e.push_back(e);}
        KVBinding<KVManager> memory(&kv);const RadBatch* staged=batch.build(plan,memory,++ordinal);need(staged!=nullptr,"actual BatchBuilder failed");good(rad_stream_sync(stream),"finish original-token batch upload");
        need(staged->n_tok==plan.n_tok&&staged->n_seq==2&&staged->kv[pending].state_index_pitch==1,"actual ragged batch geometry mismatch");
        for(auto& b:buffers)std::memset(b.base,poison,size_t(b.bytes));
        int64_t row=0;const auto geometry=KVGeom::of(builder->program().kv_groups);
        for(int s=0;s<2;++s)for(int j=0;j<lengths[size_t(s)];++j,++row){const int seq=order[size_t(s)];const int64_t pos=starts[size_t(s)]+j;
            need(staged->positions[row]==pos,"BatchBuilder changed original token position");need(staged->kv[full].slot_mapping[row]==geometry.slot(full,kv.block_table(uint64_t(10+seq*10),full),0,pos),"actual I32 slot map differs");
            for(int d=0;d<D;++d){rad_store_f32(data(projected),RAD_F32,row*D+d,projection(seq,pos,d));rad_store_f32(data(scores),RAD_F32,row*D+d,score(seq,pos,d));}}
        for(int d=0;d<D;++d)rad_store_f32(data(gain),RAD_F32,d,norm_weights[size_t(d)]);
        for(int pos=0;pos<P;++pos)for(int pair=0;pair<rotary/2;++pair){const float frequency=1.f/std::pow(10000.f,float(2*pair)/float(rotary)),angle=float(pos)*frequency;rad_store_f32(data(pairs),RAD_F32,pos*rotary+pair*2,std::cos(angle));rad_store_f32(data(pairs),RAD_F32,pos*rotary+pair*2+1,std::sin(angle));}
        if(fault==Fault::late_projection)rad_store_f32(data(projected),RAD_F32,(plan.n_tok-1)*D+D-1,NAN);
        if(fault==Fault::finish_gain)rad_store_f32(data(gain),RAD_F32,D-1,NAN);
        std::array<std::vector<uint8_t>,n_buffers> expected;
        for(int i=0;i<n_buffers;++i){const auto& b=buffers[size_t(i)];const auto* begin=static_cast<const uint8_t*>(b.base);expected[size_t(i)].assign(begin,begin+b.bytes);}
        const auto kv_before=expected_kv;
        if(fault!=Fault::late_projection){
            row=0;
            for(int s=0;s<2;++s){const int seq=order[size_t(s)];const uint64_t id=uint64_t(10+seq*10);
                for(int j=0;j<lengths[size_t(s)];++j,++row){const int64_t pos=starts[size_t(s)]+j;std::vector<float> kvrow(D),scoresrow(D),normalized;for(int d=0;d<D;++d){kvrow[size_t(d)]=projection(seq,pos,d);scoresrow[size_t(d)]=score(seq,pos,d);}
                    int64_t first=-1;good(oracle[size_t(seq)]->push(kvrow,scoresrow,&normalized,&first),"independent Compressor oracle");const bool emitted=!normalized.empty();
                    expected_int(expected[emits],buffers[emits],row,emitted?1:0);expected_int(expected[firsts],buffers[firsts],row,int32_t(first));
                    for(int d=0;d<D;++d){float pooled_value=0;
                        if(emitted){const float previous=projection(seq,pos-1,d),a_score=score(seq,pos-1,d),b_score=score(seq,pos,d),maximum=std::max(a_score,b_score),a=std::exp(a_score-maximum),b=std::exp(b_score-maximum);pooled_value=previous*(a/(a+b))+kvrow[size_t(d)]*(b/(a+b));}
                        expected_scalar(expected[pooled],buffers[pooled],row*D+d,pooled_value);}
                    if(fault==Fault::finish_gain)continue;
                    std::vector<float> rotated_row=normalized;if(emitted)good(ref::rotary(rotated_row,first,{size_t(rotary),10000.f,1.f,32.f,1.f,0}),"independent rotary oracle");
                    for(int d=0;d<D;++d){expected_scalar(expected[unrotated],buffers[unrotated],row*D+d,emitted?normalized[size_t(d)]:0.f);expected_scalar(expected[rotated],buffers[rotated],row*D+d,emitted?rotated_row[size_t(d)]:0.f);}
                    expected_int(expected[status],buffers[status],row,RAD_OK);
                    if(emitted){const int64_t slot=geometry.slot(full,kv.block_table(id,full),0,pos),base=static_cast<const uint8_t*>(bindings[full].base)-static_cast<const uint8_t*>(pools.kv().base());
                        for(int d=0;d<D;++d){int64_t offset=0;good(state::physical_byte(layout,slot/B,(slot%B)/R,d,&offset),"independent physical cache address");rad_store_f32(expected_kv.data()+base+offset,dtype,0,cast(rotated_row[size_t(d)],dtype));}}
                }
                const int64_t end=starts[size_t(s)]+lengths[size_t(s)],last_even=((end-1)/R)*R;
                const int32_t slot=kv.state_slot(id,pending);const int64_t base=static_cast<const uint8_t*>(kv.state_ptr(pending,0,slot))-static_cast<const uint8_t*>(pools.kv().base());
                for(int d=0;d<D;++d){const uint32_t a=std::bit_cast<uint32_t>(projection(seq,last_even,d)),b=std::bit_cast<uint32_t>(score(seq,last_even,d));std::memcpy(expected_kv.data()+base+d*4,&a,4);std::memcpy(expected_kv.data()+base+(D+1+d)*4,&b,4);}
                const uint32_t cursor=uint32_t(end),magic=0x44534302u;std::memcpy(expected_kv.data()+base+D*4,&cursor,4);std::memcpy(expected_kv.data()+base+(2*D+1)*4,&magic,4);
                need(oracle[size_t(seq)]->snapshot().next_position==end,"oracle cursor differs from original token end");
            }
        }
        rad_dev_host_wrote();const auto before_calls=calls;active=this;const int submitted=ctx.run_step(staged);active=nullptr;good(ctx.await_device_status(),"HOST completion without deferred device status");good(rad_stream_sync(stream),"finish complete HOST pipeline");
        need(submitted==(fault==Fault::none?RAD_OK:RAD_E_INVAL),"actual registered pipeline first-error result differs");fatal=submitted<0;
        need(calls[0]-before_calls[0]==1&&calls[1]-before_calls[1]==(fault==Fault::late_projection?0:1)&&calls[2]-before_calls[2]==(fault==Fault::none?1:0),"architecture issued operators after synchronous first error");
        for(int i=0;i<n_buffers;++i){const auto& b=buffers[size_t(i)];need(std::memcmp(b.base,expected[size_t(i)].data(),size_t(b.bytes))==0,"pipeline source/output/mask/status/guard bits differ from independent oracle");}
        need(std::memcmp(pools.kv().base(),expected_kv.data(),size_t(logical_receipt))==0,"persistent U32/cache/unrelated-sequence/guard bytes differ");
        if(fault==Fault::late_projection)need(expected_kv==kv_before,"invalid compressor input published state/cache");
        if(fault==Fault::finish_gain)need(expected_kv!=kv_before,"fatal finish fixture did not demonstrate already-mutated compressor state");
        size_t ix=0;for(auto op:ops)for(const auto& band:builder->program().ops[op].bands)need(band.dom[RAD_DOMAIN_HOST].row->info->launch==launches[ix++],"test replaced actual registered launch");
        return submitted;
    }
};
void step(RadCtx* context,const RadBatch* batch){if(active)active->execute(context,batch);else rad_step_fail(context,"missing compressor pipeline owner");}
} // namespace pipe_test
TEST(ds41_actual_host_compressor_finish_emit_pipeline_interleaves_two_sequences_with_odd_resume) {
    if(!rad::device_is_host())SKIP("actual HOST kernels/KV only");
    for(uint32_t dtype:{RAD_F32,RAD_BF16}){pipe_test::Pipeline p(dtype);
        REQUIRE_EQ(p.run({0,1},{0,0},{3,3}),RAD_OK);
        REQUIRE_EQ(p.run({1,0},{3,3},{4,4}),RAD_OK);
        // Both sequences have a pending original row6. An invalid LAST projected row
        // rejects the complete batch before ANY state write; finish/store are never issued.
        REQUIRE_EQ(p.run({0,1},{7,7},{1,1},pipe_test::Pipeline::Fault::late_projection),RAD_E_INVAL);
        CHECK(p.fatal);CHECK_EQ(p.calls[0],3);CHECK_EQ(p.calls[1],2);CHECK_EQ(p.calls[2],2);
    }
}
TEST(ds41_actual_host_pipeline_later_finish_error_is_fatal_and_does_not_claim_state_rollback) {
    if(!rad::device_is_host())SKIP("actual HOST kernels/KV only");
    pipe_test::Pipeline p(RAD_BF16);
    REQUIRE_EQ(p.run({0,1},{0,0},{3,3},pipe_test::Pipeline::Fault::finish_gain),RAD_E_INVAL);
    // A valid compressor has ALREADY mutated LINEAR state before finish fails. The cache
    // remains unchanged and store was not issued. This profile is fatal, not a reusable
    // failed pass; current ABI does not offer cross-operator transactional rollback.
    // The same fatal boundary applies to a later store refusal after valid prior ops.
    CHECK(p.fatal);CHECK_EQ(p.calls[0],1);CHECK_EQ(p.calls[1],1);CHECK_EQ(p.calls[2],0);
    bool refused=false;try{(void)p.run({0,1},{0,0},{3,3});}catch(const std::runtime_error&){refused=true;}CHECK(refused);
}
RAD_TEST_MAIN()
