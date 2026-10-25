/* Explicit actual Core/AQL compressor -> finish -> emitted store integration gate.
 * Real Pools/KVManager/BatchBuilder; no models/QDQ/TP/launch replacement/rollback claim. */
#include "build/rad_build.h"
#include "runtime/ctx.h"
#include "runtime/kvbind.h"
#include "mem/pools.h"
#include "mem/kv.h"
#include "sched/batch.h"
#include "../libds41/ds41_compress.h"
#include "../libds41/ds41_compress_finish.h"
#include "../arch/deepseek41/state_layout.h"
#include "../arch/common/rad_ds41_ref.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace pipe_aql {
using namespace rad;
namespace state=rad::deepseek41::state;
namespace ref=rad::deepseek41::ref;
constexpr int64_t D=8,R=2,B=4,P=16,rotary=4,maxT=8,maxS=3;
constexpr int64_t MiB=1<<20,total_cap=8*MiB,logical_cap=MiB,arena_cap=128<<10,host_stage_cap=512<<10;
constexpr uint8_t poison=0xA5;
void need(bool value,const std::string& text){if(!value)throw std::runtime_error(text);}
void good(int status,const char* text){need(status==RAD_OK,std::string(text)+": "+rad_strerror(status));}
float projection(int seq,int64_t pos,int64_t d){return float(((seq+1)*31+(pos+1)*13+d*7)%71-35)/32.f;}
float score(int seq,int64_t pos,int64_t d){return float(((seq+2)*17+(pos+1)*11+d*5)%59-29)/8.f;}
float cast(float x,uint32_t dtype){uint64_t word=0;rad_store_f32(&word,dtype,0,x);return rad_load_f32(&word,dtype,0);}
int bf_order(uint16_t value){return value&0x8000u?32768-int(value&0x7FFFu):32768+int(value);}
void numeric(const uint8_t* got,const uint8_t* expected,uint32_t dtype){
    const float a=rad_load_f32(got,dtype,0),b=rad_load_f32(expected,dtype,0);need(std::isfinite(a)&&std::isfinite(b),"nonfinite actual/oracle pipeline output");
    if(dtype==RAD_F32)need(std::fabs(a-b)<=5e-6f+5e-5f*std::fabs(b),"F32 pipeline output differs from independent composition");
    else {uint16_t x=0,y=0;std::memcpy(&x,got,2);std::memcpy(&y,expected,2);need(std::abs(bf_order(x)-bf_order(y))<=1,"BF16 pipeline output differs by more than one stored ULP");}
}
struct Buffer {rad_buf handle=0;uint32_t dtype=0;int64_t bytes=0,offset=0,off_elements=0;void* base=nullptr;};
enum Tensor {projected=0,scores,pooled,emits,firsts,gain,pairs,unrotated,rotated,status,n_buffers};
struct Pipeline;
thread_local Pipeline* active=nullptr;
void step(RadCtx*,const RadBatch*);
struct Pipeline {
    Registry registry;std::unique_ptr<Builder> builder;Config cfg;Pools pools;KVManager kv;BatchBuilder batch;ChunkGeometry chunk;
    std::vector<KVPoolBinding> bindings;RadStream stream=nullptr;Ctx ctx;
    bool batch_started=false,closed=false;uint32_t dtype=0;rad_kvgroup full=0,pending=0;
    std::array<Buffer,n_buffers> buffers;std::array<rad_op,3> ops{};std::array<int,3> calls{};
    std::vector<RadLaunchFn> launches;std::vector<float> norm_weights;std::array<std::unique_ptr<ref::Compressor>,2> oracle;
    std::vector<uint8_t> expected_kv;state::Layout layout;int ordinal=0,published=0;
    int64_t logical_kv=0,tensor_bytes=0,arena_bytes=0,batch_bytes=0,batch_ring_bound=0,pool_rounded_bound=0;
    std::vector<uint8_t> read(const void* source,int64_t bytes){need(source&&bytes>0&&bytes<logical_cap,"readback is not tiny");std::vector<uint8_t> out(size_t(bytes),0);good(rad_memcpy_async(out.data(),source,bytes,stream),"read completed pipeline buffer");good(rad_stream_sync(stream),"complete pipeline readback");return out;}
    std::vector<int32_t> read_i32(const int32_t* source,int64_t count){const auto bytes=read(source,count*4);std::vector<int32_t> out(size_t(count),0);std::memcpy(out.data(),bytes.data(),bytes.size());return out;}
    int64_t kv_offset(const void* pointer,int64_t bytes)const{const uintptr_t base=uintptr_t(pools.kv().base()),at=uintptr_t(pointer);need(at>=base&&at-base<=uint64_t(logical_kv)&&uint64_t(bytes)<=uint64_t(logical_kv)-(at-base),"KV pointer is outside guarded synthetic receipt");return int64_t(at-base);}
    void init(const std::string& home,uint32_t activation){
        dtype=activation;need(dtype==RAD_F32||dtype==RAD_BF16,"unsupported explicit activation profile");
        good(registry.load_plugin(home+"/experimental/kernels/libds41.so",0),"load actual isolated26-row GPU plugin");
        RadModelMeta meta{};meta.arch_id="ds41-aql-compressor-pipeline-fixture";meta.name="synthetic";meta.quant="none";meta.n_layers=1;meta.n_embd=D;meta.n_ctx_train=P;meta.n_vocab=128;
        RadBuildCtx bc{};bc.rank=0;bc.world_size=1;bc.max_tok=maxT;bc.max_seqs=maxS;bc.max_ctx=P;bc.scope="";
        builder=std::make_unique<Builder>(registry,meta,bc);builder->set_instances(false);
        good(state::make_layout(D,R,B,rad_dtype_bits(dtype)/8,&layout),"make independent latent physical geometry");RadKVGroupDecl fd{},sd{};
        good(state::paged_decl(layout,dtype,RAD_KV_FULL,0,&fd),"FULL declaration");good(rad::ds41::compress::state_decl(D,RAD_U32,0,&sd),"opaque U32 LINEAR declaration");
        full=rad_decl_kv_group(builder.get(),"compressed",&fd);pending=rad_decl_kv_group(builder.get(),"pending",&sd);need(full&&pending,"Builder refused real KV groups");
        good(rad_bind_layer_kv(builder.get(),0,full),"bind FULL layer");good(rad_bind_layer_kv(builder.get(),0,pending),"bind pending LINEAR layer");
        declare_buffer(projected,RAD_F32,{maxT,D});declare_buffer(scores,RAD_F32,{maxT,D});declare_buffer(pooled,RAD_F32,{maxT,D});
        declare_buffer(emits,RAD_I32,{maxT});declare_buffer(firsts,RAD_I32,{maxT});declare_buffer(gain,RAD_F32,{D});declare_buffer(pairs,RAD_F32,{P,rotary/2,2});
        declare_buffer(unrotated,dtype,{maxT,D});declare_buffer(rotated,dtype,{maxT,D});declare_buffer(status,RAD_I32,{maxT});
        const RadParam cp[]{RAD_RANGE("M",1,maxT),RAD_INT("dim",D),RAD_INT("ratio",R),RAD_STR("dtype","f32"),RAD_INT("n_spec",0)};
        const RadParam fp[]{RAD_RANGE("M",1,maxT),RAD_INT("n",D),RAD_INT("rotary_dim",rotary),RAD_INT("ratio",R),RAD_STR("a_dtype","f32"),RAD_STR("dtype",rad_dtype_name(dtype)),RAD_F64("eps",1e-6)};
        const RadParam sp[]{RAD_RANGE("M",1,maxT),RAD_INT("dimension",D),RAD_INT("ratio",R),RAD_INT("block_size",B),RAD_STR("dtype",rad_dtype_name(dtype))};
        ops[0]=rad_decl_op(builder.get(),"ds41_compress_pool",cp,5,nullptr,0);ops[1]=rad_decl_op(builder.get(),"ds41_compress_finish",fp,7,nullptr,0);ops[2]=rad_decl_op(builder.get(),"ds41_latent_store_emitted",sp,5,nullptr,0);
        for(auto op:ops)need(op!=0,"actual registered pipeline op did not resolve");
        hazards(ops[0],{projected,scores},{pooled,emits,firsts});hazards(ops[1],{pooled,emits,firsts,gain,pairs},{unrotated,rotated});hazards(ops[2],{rotated,emits},{status});
        good(builder->finish(),"finish actual pipeline Builder");auto& program=builder->program();need(program.kv_groups.size()==3&&program.kv_groups[full].block_size==B,"unexpected tiny KV group geometry");
        for(auto op:ops)for(const auto& band:program.ops[op].bands){const auto& row=band.dom[RAD_DOMAIN_DEVICE];need(row.row&&row.row->plugin=="libds41"&&row.row->info&&row.row->info->launch&&row.row->deferred_status&&!row.row->collective,"actual GPU launch/collector contract missing");launches.push_back(row.row->info->launch);}
        cfg.max_tok=maxT;cfg.max_seqs=maxS;cfg.max_ctx=P;cfg.n_spec=0;cfg.vram_kv_mib=1;cfg.vram_weights_mib=1;cfg.host_pool_mib=0;cfg.checkpoint_slots=0;cfg.prefix_cache=false;
        arena_bytes=arena_plan_bytes(program,nullptr,nullptr)+align_up(arena_host_bytes(program),RAD_ALIGN_UNIT);need(arena_bytes>0&&arena_bytes<=arena_cap,"Program-sized pipeline arena cap");
        const auto geometry=KVGeom::of(program.kv_groups);batch_bytes=BatchBuilder::device_bytes(program,cfg,geometry);need(batch_bytes>0&&batch_bytes<65536,"BatchBuilder device slab is not tiny");
        // batch.cpp mirrors the fixed host/device carve, and the no-media host ring is
        // three worst-case slabs plus table-range descriptors and256B alignment.
        int64_t worst_pay=0;for(size_t group=0;group<program.kv_groups.size();++group)worst_pay+=maxS*(group==full?(P+B-1)/B+1:1)*4;
        const int64_t worst_ranges=worst_pay/16384+maxS*int64_t(program.kv_groups.size())+1;
        batch_ring_bound=3*(batch_bytes+worst_ranges*int64_t(sizeof(RadCopyRange))+256);
        const int64_t granule=VMem::supported()?VMem::granularity():0;need(granule>=0&&granule<=total_cap,"unsupported oversized VMem granularity");
        const int64_t quantum=std::max<int64_t>(RAD_ALIGN_POOL,granule);pool_rounded_bound=2*align_up(MiB,quantum);
        need(pool_rounded_bound+arena_bytes+batch_bytes+batch_ring_bound+host_stage_cap<=total_cap,"refuse before allocation: rounded pool/data request exceeds8MiB");
        good(rad_stream_create(&stream,0),"create actual Core compute stream");need(stream!=nullptr,"null Core stream");
        good(pools.configure(cfg,0,&rad_device_allocator(),&rad_device_allocator()),"configure real minimum2MiB DEVICE Pools");need(pools.weights().bytes()+pools.kv().bytes()==2*MiB&&pools.host_pinned().bytes()==0,"Pools ignored tiny explicit budgets");
        good(kv.plan_groups(program.kv_groups,cfg),"plan real capped KV groups");kv.cap_paged_tokens(64);good(kv.carve(pools.kv()),"carve real small FULL/LINEAR regions");
        for(uint64_t id:{10,20,30}){good(kv.add_sequence(id),"admit sequence");good(kv.ensure(id,1),"allocate initial cache/state slots");}
        logical_kv=pools.kv().used()+256;need(logical_kv>0&&logical_kv<65536&&logical_kv<=pools.kv().bytes(),"logical KV/state receipt is not tiny");good(pools.kv().commit(0,logical_kv),"commit logical KV plus tail guard");
        need(pools.weights().committed()+pools.kv().committed()<=pool_rounded_bound,"actual pool commitment exceeds conservative rounding receipt");
        expected_kv.assign(size_t(logical_kv),0xC7);good(rad_memset_async(pools.kv().base(),0xC7,logical_kv,stream),"poison cache/state/unused bytes");
        for(const auto& clear:kv.state_clears()){const auto* plan=kv.plan(clear.group);for(int l=0;l<plan->n_layers;++l){void* ptr=kv.state_ptr(clear.group,l,clear.slot);const int64_t bytes=plan->layer_bytes_per_state(),off=kv_offset(ptr,bytes);std::fill_n(expected_kv.data()+off,size_t(bytes),uint8_t(0));good(rad_memset_async(ptr,0,bytes,stream),"clear fresh real LINEAR state");}}kv.clear_state_clears();
        std::array<uint32_t,2*(D+1)> untouched{};for(size_t i=0;i<untouched.size();++i)untouched[i]=0xDEAD0000u+uint32_t(i);
        void* third=kv.state_ptr(pending,0,kv.state_slot(30,pending));const auto third_off=kv_offset(third,int64_t(sizeof untouched));std::memcpy(expected_kv.data()+third_off,untouched.data(),sizeof untouched);good(rad_memcpy_async(third,untouched.data(),int64_t(sizeof untouched),stream),"protect unused third sequence state");good(rad_stream_sync(stream),"finish initial immutable guard/state setup");
        bind_kv_pools(program,[&](int32_t g){return kv.plan(g);},&bindings);batch_started=true;good(batch.init(program,cfg,chunk,geometry,{{0,stream,&program}}),"init actual GPU BatchBuilder");
        CtxDesc cd{};cd.program=&program;cd.rank=0;cd.world_size=1;cd.device=0;cd.stream=stream;cd.arch_step=step;cd.record_passes=false;cd.kv=bindings;
        good(ctx.init(cd),"init actual AQL Ctx");good(ctx.bind_thread(),"bind actual AQL thread");need(ctx.checks_device_status()&&!ctx.records_passes(),"pipeline lost deferred/eager contract");
        for(auto& b:buffers){b.base=rad_buf_ptr(&ctx,b.handle);need(b.base!=nullptr,"actual DEVICE public tensor absent");tensor_bytes+=b.bytes;}
        need(logical_kv+tensor_bytes+arena_bytes+batch_bytes+batch_ring_bound+host_stage_cap<logical_cap,"logical synthetic data exceeds1MiB");
        norm_weights.resize(D);for(int d=0;d<D;++d)norm_weights[size_t(d)]=1.f+float(d-3)/32.f;
        for(int seq=0;seq<2;++seq)oracle[size_t(seq)]=std::make_unique<ref::Compressor>(R,norm_weights,1e-6f,"seq"+std::to_string(seq),dtype==RAD_F32?ref::Precision::f32:ref::Precision::bf16);
        std::printf("  %s receipt: Pools requested=%lldB, Pool-reported committed=%lldB, conservative rounded pool bound=%lldB; Ctx arena=%lldB (tensor subset=%lldB), batch DEVICE=%lldB, pinned ring bound=%lldB, explicit HOST staging cap=%lldB; total bound=%lldB; logical KV/state/guard=%lldB.\n",rad_dtype_name(dtype),(long long)(pools.weights().bytes()+pools.kv().bytes()),(long long)(pools.weights().committed()+pools.kv().committed()),(long long)pool_rounded_bound,(long long)arena_bytes,(long long)tensor_bytes,(long long)batch_bytes,(long long)batch_ring_bound,(long long)host_stage_cap,(long long)(pool_rounded_bound+arena_bytes+batch_bytes+batch_ring_bound+host_stage_cap),(long long)logical_kv);
        std::printf("  KV address reservation requested=%lldB, conservative rounded VA bound=%lldB, elastic=%d, reported VMem granularity=%lldB; VA is separate from committed physical bytes.\n",(long long)pools.kv().bytes(),(long long)align_up(pools.kv().bytes(),quantum),int(pools.kv().elastic()),(long long)granule);
    }
    bool shutdown()noexcept{
        if(closed)return true;
        if(stream&&rad_stream_sync(stream)!=RAD_OK)return false;
        ctx.shutdown();if(batch_started){batch.fini();batch_started=false;}if(stream){rad_stream_destroy(stream);stream=nullptr;}closed=true;return true;
    }
    // Main retains this WHOLE owner on an unproven drain; no destructor frees live Pools.
    ~Pipeline(){if(!closed&&stream)std::abort();}
    void declare_buffer(int index,uint32_t dtype_value,std::initializer_list<int64_t> shape){
        auto& b=buffers[size_t(index)];b.dtype=dtype_value;RadBufDecl d{};d.dtype=dtype_value;d.rank=shape.size();d.kind=RAD_BUF_PERSIST;d.domain=RAD_DOMAIN_DEVICE;
        unsigned axis=0;for(auto n:shape)d.shape[axis++]=n;int64_t trailing=1;for(unsigned i=1;i<d.rank;++i)trailing*=d.shape[i];
        const int64_t width=rad_dtype_bits(dtype_value)/8,row_bytes=trailing*width,guards=(256+row_bytes-1)/row_bytes;
        b.off_elements=guards*trailing;b.offset=b.off_elements*width;d.shape[0]+=2*guards;b.bytes=d.shape[0]*row_bytes;need(b.bytes>0&&b.bytes<16384,"guarded DEVICE tensor cap");
        b.handle=rad_decl_buffer(builder.get(),("tensor"+std::to_string(index)).c_str(),&d);need(b.handle!=0,"Builder refused guarded public DEVICE buffer");
    }
    void hazards(rad_op op,std::initializer_list<int> in,std::initializer_list<int> out){std::vector<rad_buf> r,w;for(int i:in)r.push_back(buffers[size_t(i)].handle);for(int i:out)w.push_back(buffers[size_t(i)].handle);good(rad_op_reads(builder.get(),op,r.data(),int(r.size())),"declare reads");good(rad_op_writes(builder.get(),op,w.data(),int(w.size())),"declare writes");}
    RadOperand operand(int i,int64_t rows){return RAD_B_N(buffers[size_t(i)].handle,buffers[size_t(i)].off_elements,rows);}
    static RadOperand vector(const int32_t* ptr,int64_t rows){auto o=RAD_P_T(ptr,RAD_I32);o.rows=rows;return o;}
    void execute(RadCtx* context,const RadBatch* b){
        if(b->kv[pending].state_index_pitch!=1){rad_step_fail(context,"unexpected speculative state columns");return;}
        const RadOperand cp[]{operand(projected,b->n_tok),operand(scores,b->n_tok),RAD_KV(pending,0),RAD_P_T2(b->kv[pending].state_index,RAD_I32,b->n_seq,1),vector(b->ctx_lens,b->n_seq),vector(b->cu_seqlens,b->n_seq+1),vector(b->positions,b->n_tok),RAD_NONE,operand(pooled,b->n_tok),operand(emits,b->n_tok),operand(firsts,b->n_tok)};
        ++calls[0];if(rad_issue(context,ops[0],cp,11,b->n_tok)<0)return;
        const RadOperand fp[]{operand(pooled,b->n_tok),operand(emits,b->n_tok),operand(firsts,b->n_tok),operand(gain,D),operand(pairs,P),operand(unrotated,b->n_tok),operand(rotated,b->n_tok)};
        ++calls[1];if(rad_issue(context,ops[1],fp,7,b->n_tok)<0)return;
        const RadOperand sp[]{operand(rotated,b->n_tok),vector(b->positions,b->n_tok),vector(b->kv[full].slot_mapping,b->n_tok),operand(emits,b->n_tok),RAD_KV(full,0),operand(status,b->n_tok)};
        ++calls[2];(void)rad_issue(context,ops[2],sp,6,b->n_tok);
    }
    static void scalar(std::vector<uint8_t>& out,const Buffer& b,int64_t flat,float x){rad_store_f32(out.data()+b.offset,b.dtype,flat,x);}
    static void integer(std::vector<uint8_t>& out,const Buffer& b,int64_t flat,int32_t x){std::memcpy(out.data()+b.offset+flat*4,&x,4);}
    void run(const std::array<int,2>& order,const std::array<int,2>& starts,const std::array<int,2>& lengths){
        good(rad_stream_sync(stream),"drain previous successful pass before slot changes");StepPlan plan;plan.phase=RAD_PHASE_PREFILL;plan.n_tok=lengths[0]+lengths[1];need(plan.n_tok>0&&plan.n_tok<=maxT,"invalid tiny ragged chunk");
        for(int s=0;s<2;++s){const uint64_t id=uint64_t(10+order[size_t(s)]*10);good(kv.ensure(id,starts[size_t(s)]+lengths[size_t(s)],starts[size_t(s)]),"ensure real original-token KV slots");StepEntry e;e.seq=id;e.card=s;e.ctx_len=starts[size_t(s)];e.n_tokens=lengths[size_t(s)];e.is_prefill=true;plan.e.push_back(e);}
        KVBinding<KVManager> memory(&kv);const RadBatch* staged=batch.build(plan,memory,++ordinal);need(staged&&staged->n_tok==plan.n_tok&&staged->n_seq==2&&staged->kv[pending].state_index_pitch==1,"actual GPU batch descriptor differs");
        std::array<std::vector<uint8_t>,n_buffers> before,expected,got;std::array<std::vector<int64_t>,n_buffers> numeric_cells;
        for(int i=0;i<n_buffers;++i)before[size_t(i)].assign(size_t(buffers[size_t(i)].bytes),poison);
        std::vector<int32_t> expected_pos,expected_slot,expected_state,expected_cu{0},expected_ctx;
        const auto geometry=KVGeom::of(builder->program().kv_groups);int64_t row=0;
        for(int s=0;s<2;++s){const int seq=order[size_t(s)];const uint64_t id=uint64_t(10+seq*10);expected_state.push_back(kv.state_slot(id,pending));expected_ctx.push_back(starts[size_t(s)]);
            for(int j=0;j<lengths[size_t(s)];++j,++row){const int64_t pos=starts[size_t(s)]+j;expected_pos.push_back(int32_t(pos));expected_slot.push_back(geometry.slot(full,kv.block_table(id,full),0,pos));
                for(int d=0;d<D;++d){scalar(before[projected],buffers[projected],row*D+d,projection(seq,pos,d));scalar(before[scores],buffers[scores],row*D+d,score(seq,pos,d));}}
            expected_cu.push_back(int32_t(row));}
        for(int d=0;d<D;++d)scalar(before[gain],buffers[gain],d,norm_weights[size_t(d)]);
        for(int pos=0;pos<P;++pos)for(int pair=0;pair<rotary/2;++pair){const float frequency=1.f/std::pow(10000.f,float(2*pair)/float(rotary)),angle=float(pos)*frequency;scalar(before[pairs],buffers[pairs],pos*rotary+pair*2,std::cos(angle));scalar(before[pairs],buffers[pairs],pos*rotary+pair*2+1,std::sin(angle));}
        expected=before;struct CacheWrite{int64_t off,flat;};std::vector<CacheWrite> cache_writes;std::set<int64_t> unique_cache;
        row=0;
        for(int s=0;s<2;++s){const int seq=order[size_t(s)];const uint64_t id=uint64_t(10+seq*10);
            for(int j=0;j<lengths[size_t(s)];++j,++row){const int64_t pos=starts[size_t(s)]+j;std::vector<float> kvrow(D),scoresrow(D),normalized;for(int d=0;d<D;++d){kvrow[size_t(d)]=projection(seq,pos,d);scoresrow[size_t(d)]=score(seq,pos,d);}
                int64_t first=-1;good(oracle[size_t(seq)]->push(kvrow,scoresrow,&normalized,&first),"independent CPU compressor/cast/RMS");const bool emitted=!normalized.empty();integer(expected[emits],buffers[emits],row,emitted?1:0);integer(expected[firsts],buffers[firsts],row,int32_t(first));
                std::vector<float> rotated_row=normalized;if(emitted)good(ref::rotary(rotated_row,first,{size_t(rotary),10000.f,1.f,32.f,1.f,0}),"independent original-position RoPE");
                for(int d=0;d<D;++d){float pooled_value=0;if(emitted){const float a_score=score(seq,pos-1,d),b_score=score(seq,pos,d),maximum=std::max(a_score,b_score),a=std::exp(a_score-maximum),b=std::exp(b_score-maximum);pooled_value=projection(seq,pos-1,d)*(a/(a+b))+kvrow[size_t(d)]*(b/(a+b));}
                    scalar(expected[pooled],buffers[pooled],row*D+d,pooled_value);scalar(expected[unrotated],buffers[unrotated],row*D+d,emitted?normalized[size_t(d)]:0.f);scalar(expected[rotated],buffers[rotated],row*D+d,emitted?rotated_row[size_t(d)]:0.f);
                    if(emitted)for(int output:{pooled,unrotated,rotated})numeric_cells[size_t(output)].push_back(row*D+d);}
                integer(expected[status],buffers[status],row,RAD_OK);
                if(emitted){const int64_t slot=geometry.slot(full,kv.block_table(id,full),0,pos),base=kv_offset(bindings[full].base,1);for(int d=0;d<D;++d){int64_t local=0;good(state::physical_byte(layout,slot/B,(slot%B)/R,d,&local),"independent cache physical map");const int64_t off=base+local;need(off>=0&&off+rad_dtype_bits(dtype)/8<=logical_kv&&unique_cache.insert(off).second,"cache write map aliases/out-of-bounds");rad_store_f32(expected_kv.data()+off,dtype,0,cast(rotated_row[size_t(d)],dtype));cache_writes.push_back({off,row*D+d});}}
            }
            const int64_t end=starts[size_t(s)]+lengths[size_t(s)],last_even=((end-1)/R)*R,base=kv_offset(kv.state_ptr(pending,0,kv.state_slot(id,pending)),2*(D+1)*4);
            for(int d=0;d<D;++d){const uint32_t a=std::bit_cast<uint32_t>(projection(seq,last_even,d)),b=std::bit_cast<uint32_t>(score(seq,last_even,d));std::memcpy(expected_kv.data()+base+d*4,&a,4);std::memcpy(expected_kv.data()+base+(D+1+d)*4,&b,4);}
            const uint32_t cursor=uint32_t(end),magic=0x44534302u;std::memcpy(expected_kv.data()+base+D*4,&cursor,4);std::memcpy(expected_kv.data()+base+(2*D+1)*4,&magic,4);need(oracle[size_t(seq)]->snapshot().next_position==end,"oracle did not retain original-token cursor");
        }
        int64_t host_staging=3*tensor_bytes+2*logical_kv+int64_t((expected_pos.size()+expected_slot.size()+expected_state.size()+expected_cu.size()+expected_ctx.size())*sizeof(int32_t))+4096;
        need(host_staging<=host_stage_cap,"explicit checker HOST vector staging cap");
        // BatchBuilder's I32 uploads, tensor uploads and ALL three operators share this
        // actual Core stream. No payload readback/sync is inserted between operators.
        for(int i=0;i<n_buffers;++i)good(rad_memcpy_async(buffers[size_t(i)].base,before[size_t(i)].data(),buffers[size_t(i)].bytes,stream),"upload guarded pipeline tensor");
        const auto before_calls=calls;active=this;const int submitted=ctx.run_step(staged);active=nullptr;need(submitted==RAD_OK,std::string("pipeline enqueue failed: ")+ctx.step_error());
        const int awaited=ctx.await_device_status();need(awaited==RAD_OK,std::string("pipeline deferred status failed: ")+ctx.step_error());good(rad_stream_sync(stream),"complete successful three-op AQL pipeline");
        need(calls[0]-before_calls[0]==1&&calls[1]-before_calls[1]==1&&calls[2]-before_calls[2]==1,"real architecture omitted a pipeline issue");
        // First HOST inspection of any runtime payload is after the complete pass.
        need(read_i32(staged->positions,plan.n_tok)==expected_pos,"actual DEVICE original positions differ");need(read_i32(staged->ctx_lens,2)==expected_ctx,"actual DEVICE init/cursor positions differ");need(read_i32(staged->cu_seqlens,3)==expected_cu,"actual DEVICE ragged cu differs");
        need(read_i32(staged->kv[pending].state_index,2)==expected_state,"actual DEVICE state slots did not follow reordered sequence identities");need(read_i32(staged->kv[full].slot_mapping,plan.n_tok)==expected_slot,"actual DEVICE physical token slot mapping differs");
        for(int i=0;i<n_buffers;++i){const auto& buffer=buffers[size_t(i)];got[size_t(i)]=read(buffer.base,buffer.bytes);std::vector<bool> numeric_bytes(size_t(buffer.bytes),false);const int width=rad_dtype_bits(buffer.dtype)/8;
            for(int64_t flat:numeric_cells[size_t(i)]){const int64_t off=buffer.offset+flat*width;numeric(got[size_t(i)].data()+off,expected[size_t(i)].data()+off,buffer.dtype);for(int b=0;b<width;++b)numeric_bytes[size_t(off+b)]=true;}
            for(size_t off=0;off<got[size_t(i)].size();++off)if(!numeric_bytes[off])need(got[size_t(i)][off]==expected[size_t(i)][off],"pipeline changed source/status/inactive-zero/unused-capacity/guard bits");}
        const auto kv_got=read(pools.kv().base(),logical_kv);std::vector<bool> new_cache_bytes(size_t(logical_kv),false);const int width=rad_dtype_bits(dtype)/8;
        for(const auto& write:cache_writes){numeric(kv_got.data()+write.off,expected_kv.data()+write.off,dtype);const int64_t tensor_off=buffers[rotated].offset+write.flat*width;need(std::memcmp(kv_got.data()+write.off,got[rotated].data()+tensor_off,size_t(width))==0,"emitted store did not preserve actual rotated bits at independent physical slot");for(int b=0;b<width;++b)new_cache_bytes[size_t(write.off+b)]=true;}
        for(size_t off=0;off<kv_got.size();++off)if(!new_cache_bytes[off])need(kv_got[off]==expected_kv[off],"literal pending U32/cursor/magic, old cache, third sequence or pool guard changed");
        expected_kv=kv_got; // Keep verified prior GPU cache bytes bit-exact across the next pass.
        size_t ix=0;for(auto op:ops)for(const auto& band:builder->program().ops[op].bands)need(band.dom[RAD_DOMAIN_DEVICE].row->info->launch==launches[ix++],"manual checker replaced actual registered launch");
        need(ctx.pass_counts().played==0,"deferred-status pipeline unexpectedly replayed");++published;
        std::printf("  %s successful pass%d: T=%lld, order=%c%lld+%c%lld, emitted=%zu rows, literal state/cursor+third guard and exact physical store verified.\n",rad_dtype_name(dtype),ordinal,(long long)plan.n_tok,char('A'+order[0]),(long long)lengths[0],char('A'+order[1]),(long long)lengths[1],cache_writes.size()/size_t(D));
    }
};
void step(RadCtx* c,const RadBatch* b){if(active)active->execute(c,b);else rad_step_fail(c,"missing manual compressor pipeline owner");}
} // namespace pipe_aql

int main(int argc,char** argv){
    bool run=false,host_kargs=false;std::string home;for(int i=1;i<argc;++i){const std::string option=argv[i];if(option=="--run-gpu")run=true;else if(option=="--host-kargs")host_kargs=true;else if(option=="--home"&&i+1<argc)home=argv[++i];else{std::fprintf(stderr,"usage: ds41-aql-compressor-pipeline-check --run-gpu --home ISOLATED_HOME [--host-kargs]\n");return 2;}}
    if(!run||home.empty()){std::fprintf(stderr,"usage: ds41-aql-compressor-pipeline-check --run-gpu --home ISOLATED_HOME [--host-kargs]\n");return 2;}
    // No explicit device/plugin query/allocation before this gate. Linked HIP registration
    // constructors may run before main; parent owns build and the authorized idle window.
    using namespace pipe_aql;std::unique_ptr<Pipeline> pipeline;
    try {
#if defined(RAD_HAVE_HIP) && RAD_HAVE_HIP
        if(host_kargs)good(rad::aql_force_host_kargs(),"force HOST argument storage");
#else
        (void)host_kargs;throw std::runtime_error("checker requires HIP-enabled Core");
#endif
        need(rad_dev_count()==1,"expose exactly one GPU");need(!rad::device_is_host()&&std::strcmp(rad::device_backend_name(),"aql")==0,"actual AQL backend required");RadDeviceProps props{};good(rad_dev_props(0,&props),"query selected GPU");need(!props.is_host_backend&&std::strcmp(props.arch,"gfx1201")==0,"exactly one gfx1201 required");good(rad_dev_set(0),"select only visible GPU");
        int passes=0;for(uint32_t dtype:{RAD_F32,RAD_BF16}){pipeline=std::make_unique<Pipeline>();pipeline->init(home,dtype);pipeline->run({0,1},{0,0},{3,3});pipeline->run({1,0},{3,3},{4,4});need(pipeline->published==2&&pipeline->calls==std::array<int,3>{2,2,2},"successful pipeline coverage drifted");passes+=pipeline->published;need(pipeline->shutdown(),"cannot prove final stream drain; retain whole pipeline until exit");pipeline.reset();}
        need(passes==4,"expected four complete good passes");std::puts("PASS: four actual Builder/Ctx/AQL compressor->finish->emitted-store passes; F32/BF16, A3+B3 then B4+A4, persistent opaque state, real Pools/KVManager/BatchBuilder device I32 metadata, deferred gate and physical cache ownership. No launch replacement/model/QDQ/TP/failure rollback proof.");
        std::puts("Receipt excludes driver/module metadata, allocator rounding beyond the explicit conservative pool bound, Core queue/kernel-argument/bounce rings and device-status runtime metadata. Virtual address reservation is not physical commitment.");return 0;
    }catch(const std::exception& error){active=nullptr;if(pipeline&&!pipeline->shutdown()){std::fputs("unproven drain: retain WHOLE pipeline/Pools/Batch/Ctx/plugin until process exit; no cancellation claim\n",stderr);static_cast<void>(pipeline.release());}std::fprintf(stderr,"FAIL: %s\n",error.what());return 1;}
}
