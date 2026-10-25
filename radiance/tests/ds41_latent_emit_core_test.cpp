/* Actual KVManager/BatchBuilder I32 slots -> completed original-row publication. HOST only,
 * tiny pool, no new architecture/core implementation or GPU. */
#define SCHED_FAKES_KV_ONLY
#include "sched_fakes.h"
#include "device/device.h"
#include "mem/pools.h"
#include "runtime/kvbind.h"
#include "../libds41/ds41_latent_emit_fixture.h"
namespace e=rad::ds41::latent_emit;
namespace f=e::testing;
namespace state=rad::deepseek41::state;
namespace {
void need(bool value,const char* message){if(!value)throw std::runtime_error(message);}
void good(int status,const char* message){if(status!=RAD_OK)throw std::runtime_error(std::string(message)+": "+rad_strerror(status));}
struct Core {
    static constexpr int group=1,dim=128,ratio=2;
    rad::Program program;rad::Config cfg;rad::Pools pools;rad::KVManager kv;
    std::vector<rad::KVPoolBinding> bindings;rad::ChunkGeometry chunk;rad::BatchBuilder batch;RadStream stream=nullptr;
    state::Layout layout;bool batch_ready=false;
    Core() {
        cfg.max_tok=8;cfg.max_seqs=2;cfg.max_ctx=32;cfg.n_spec=0;cfg.vram_kv_mib=1;cfg.vram_weights_mib=1;
        cfg.gpu_headroom_mib=1;cfg.host_pool_mib=0;cfg.checkpoint_slots=0;cfg.prefix_cache=false;
        program.buffers.emplace_back();program.ops.emplace_back();program.meta.n_ctx_train=32;program.meta.n_vocab=128;program.meta.n_layers=1;
        program.kv_groups.resize(2);auto& g=program.kv_groups[group];g.name="emitted-original-coordinate";g.layers={0};g.block_size=4;
        good(state::make_layout(dim,ratio,4,4,&layout),"create opaque layout");good(state::paged_decl(layout,RAD_F32,RAD_KV_FULL,0,&g.decl),"declare FULL opaque group");
        good(pools.configure(cfg,0,&rad::rad_malloc_allocator(),&rad::rad_malloc_allocator()),"configure tiny HOST pools");
        good(kv.configure(program.kv_groups,cfg,pools.kv()),"configure real KVManager");
        good(kv.add_sequence(10),"add target sequence");good(kv.ensure(10,3),"allocate prefix slots");
        good(kv.add_sequence(20),"add independent sequence");good(kv.ensure(20,5),"allocate unrelated slots");
        rad::bind_kv_pools(program,[&](int32_t i){return kv.plan(i);},&bindings);
        const auto* plan=kv.plan(group);need(plan&&plan->pool_bytes>0&&plan->pool_bytes<=2*(1<<20),"actual cache pool is not tiny");
        std::memset(plan->base,0xC7,size_t(plan->pool_bytes));good(rad_stream_create(&stream,0),"create real HOST stream");
        const auto geometry=rad::KVGeom::of(program.kv_groups);std::vector<rad::BatchBuilder::RankIO> ranks{{0,stream,&program}};
        good(batch.init(program,cfg,chunk,geometry,ranks),"initialize real BatchBuilder");batch_ready=true;
    }
    ~Core(){
        // BatchBuilder owns staging that queued copies may still reference on
        // an exception path. Drain BEFORE fini(), then retire the stream.
        if(stream&&rad_stream_sync(stream)<0)std::abort();
        if(batch_ready)batch.fini();
        if(stream)rad_stream_destroy(stream);
    }
    std::vector<uint8_t> payload()const {
        const auto* p=kv.plan(group);const auto* begin=static_cast<const uint8_t*>(p->base);return {begin,begin+p->pool_bytes};
    }
    void publish(int start,int count,bool invalid=false) {
        good(kv.ensure(10,start+count,start),"ensure chunk slots");
        rad::StepPlan plan;plan.phase=RAD_PHASE_PREFILL;plan.n_tok=count;rad::StepEntry entry;
        entry.seq=10;entry.card=0;entry.ctx_len=start;entry.n_tokens=count;entry.is_prefill=true;plan.e.push_back(entry);
        rad::KVBinding<rad::KVManager> memory(&kv);const auto* staged=batch.build(plan,memory,1);need(staged!=nullptr,"actual batch staging failed");
        good(rad_stream_sync(stream),"finish real HOST batch upload");need(staged->n_tok==count&&staged->positions&&staged->kv[group].slot_mapping,"actual batch fields missing");
        f::Fixture x(RAD_F32,dim,ratio,0,RAD_I32,RAD_I32,count,start);auto args=x.args();
        const auto& bound=bindings[group];RadTensor cache{};cache.data=bound.base;cache.dtype=bound.dtype;cache.rank=bound.rank;
        int64_t stride=1;for(int axis=int(cache.rank)-1;axis>=0;--axis){cache.shape[axis]=bound.shape[axis];cache.stride[axis]=stride;stride*=cache.shape[axis];}
        x.tensors[e::cache]=cache;x.integer("n_blocks",cache.shape[0]);
        auto vector=[&](const int32_t* data){RadTensor t{};t.data=const_cast<int32_t*>(data);t.dtype=RAD_I32;t.rank=1;t.shape[0]=count;t.stride[0]=1;return t;};
        x.tensors[e::positions]=vector(staged->positions);x.tensors[e::slot_mapping]=vector(staged->kv[group].slot_mapping);
        const auto before=payload();auto expected=before;const auto geometry=rad::KVGeom::of(program.kv_groups);
        for(int row=0;row<count;++row) {
            const int64_t position=start+row;need(staged->positions[row]==position,"core changed original token position");
            const int64_t slot=geometry.slot(group,kv.block_table(10,group),0,position);need(slot>=0&&staged->kv[group].slot_mapping[row]==slot,"core I32 slot differs from actual block-table geometry");
            if(position%ratio!=ratio-1)continue;
            for(int d=0;d<dim;++d) {
                int64_t offset=0;good(state::physical_byte(layout,slot/4,(slot%4)/ratio,d,&offset),"independent cache physical map");
                const auto* source=static_cast<const uint8_t*>(x.buffers[e::x].data())+x.buffers[e::x].offset(int64_t(row)*dim+d)*4;
                std::memcpy(expected.data()+offset,source,4);
            }
        }
        if(invalid)x.buffers[e::emit].integer(count-1,2);
        const int result=e::emulate(&args);need(result==(invalid?RAD_E_INVAL:RAD_OK),"core emitted store status differs");
        need(payload()==(invalid?before:expected),"core emitted store changed unrelated sequence/cache bytes or wrong target entry");
        for(int row=0;row<count;++row)need(x.buffers[e::status].integer(row)==(invalid?RAD_E_INVAL:RAD_OK),"core chunk status did not seal every original row");
    }
};
}
TEST(ds41_latent_emit_consumes_actual_core_I32_original_positions_and_slots_across_odd_chunks) {
    if(!rad::device_is_host())SKIP("HOST KV/BatchBuilder bytes only");
    Core core;core.publish(0,3);core.publish(3,4);
    // A failed later row must not touch either earlier target entries or the other sequence.
    core.publish(7,2,true);
    CHECK_EQ(core.kv.first_block_pos(10,Core::group),0);
}
RAD_TEST_MAIN()
