/* Actual KVManager + BatchBuilder staging on the real host backend. No GPU,
 * serving architecture, model payload or production core change is involved. */
#define SCHED_FAKES_KV_ONLY
#include "sched_fakes.h"
#include "device/device.h"
#include "../libds41/ds41_attention_paged_fixture.h"
#include "../libds41/ds41_latent_args.h"
#include "../arch/common/rad_ds41_ref.h"
#include "mem/pools.h"
#include "runtime/kvbind.h"
#include <map>
namespace a=rad::ds41::attention_paged;
namespace fix=a::testing;
namespace state=rad::deepseek41::state;
namespace ref=rad::deepseek41::ref;
namespace {
rad::Config config() {
    rad::Config c;c.max_tok=32;c.max_seqs=2;c.max_ctx=128;c.n_spec=0;c.vram_kv_mib=4;c.vram_weights_mib=1;
    c.gpu_headroom_mib=1;c.host_pool_mib=0;c.checkpoint_slots=0;c.prefix_cache=false;return c;
}
rad::KVGroupInfo group(const char* name,int kind,int64_t ratio,int64_t window) {
    rad::KVGroupInfo g;g.name=name;g.layers={0};g.block_size=4;state::Layout l;
    if(state::make_layout(8,ratio,4,4,&l)!=RAD_OK||state::paged_decl(l,RAD_F32,kind,window,&g.decl)!=RAD_OK)throw std::runtime_error("bad core fixture group");
    return g;
}
void copy_i32(fix::Buffer& destination,const int32_t* source) {for(int64_t i=0;i<destination.count();++i)destination.id(i,source[i]);}
void check_output(fix::Fixture& c) {
    const auto before=c.buffers;REQUIRE_EQ(c.run(),RAD_OK);int64_t row=0;
    for(int64_t s=0;s<c.sequences;++s)for(int64_t j=0;j<c.lengths[size_t(s)];++j,++row) {
        const int64_t pos=c.starts[size_t(s)]+j;std::vector<float> kv=c.canonical_window[size_t(s)],q,sinks,result(size_t(c.heads*c.dim),0.f);
        const int64_t comp_offset=kv.size()/size_t(c.dim);kv.insert(kv.end(),c.canonical_compressed[size_t(s)].begin(),c.canonical_compressed[size_t(s)].end());
        std::vector<int32_t> selected;for(int64_t k=std::max<int64_t>(0,pos-c.window+1);k<=pos;++k)selected.push_back(int32_t(k));
        for(int64_t k=0;k<c.topk;++k) {const auto id=c.buffer(a::selected).id(row*c.topk+k);selected.push_back(id<0?-1:int32_t(comp_offset+id));}
        for(int64_t h=0;h<c.heads;++h) {sinks.push_back(c.buffer(a::sinks).get(h));for(int64_t d=0;d<c.dim;++d)q.push_back(c.buffer(a::query).get((row*c.heads+h)*c.dim+d));}
        REQUIRE_EQ(ref::sparse_attention(q,kv,size_t(c.heads),size_t(c.dim),sinks,selected,float(1./std::sqrt(double(c.dim))),result),RAD_OK);
        for(int64_t k=0;k<c.heads*c.dim;++k)CHECK_EQ(c.buffer(a::output).get(row*c.heads*c.dim+k),result[size_t(k)]);
    }
    for(int i=0;i<a::output;++i)if(c.buffers[size_t(i)])CHECK(c.buffer(i).words==before[size_t(i)]->words);
}
}
TEST(ds41_paged_attention_consumes_actual_core_staged_window_tables_and_original_coordinate_slots) {
    if(!rad::device_is_host())SKIP("host backend staging only");
    rad::Program program;program.buffers.emplace_back();program.ops.emplace_back();program.meta.n_ctx_train=128;program.meta.n_vocab=128;program.meta.n_layers=1;
    program.kv_groups={rad::KVGroupInfo{},group("window",RAD_KV_WINDOW,1,5),group("compressed",RAD_KV_FULL,2,0)};
    auto cfg=config();rad::Pools pools;REQUIRE_EQ(pools.configure(cfg,0,&rad::rad_malloc_allocator(),&rad::rad_malloc_allocator()),RAD_OK);
    rad::KVManager kv;REQUIRE_EQ(kv.configure(program.kv_groups,cfg,pools.kv()),RAD_OK);const auto geometry=rad::KVGeom::of(program.kv_groups);
    REQUIRE_EQ(kv.add_sequence(10),RAD_OK);REQUIRE_EQ(kv.ensure(10,12),RAD_OK);REQUIRE_EQ(kv.ensure(10,29,12),RAD_OK);
    REQUIRE_EQ(kv.add_sequence(20),RAD_OK);REQUIRE_EQ(kv.ensure(20,5),RAD_OK);
    // The core deliberately keeps an extra older page: do not infer floor from window length alone.
    CHECK_EQ(kv.first_block_pos(10,1),4);CHECK(kv.first_block_pos(10,1)<8);
    RadStream stream=nullptr;REQUIRE_EQ(rad_stream_create(&stream,0),RAD_OK);
    struct StreamOwner {RadStream stream;~StreamOwner(){static_cast<void>(rad_stream_sync(stream));rad_stream_destroy(stream);}} stream_owner{stream};
    rad::ChunkGeometry chunk;rad::BatchBuilder batch;std::vector<rad::BatchBuilder::RankIO> ranks{{0,stream,&program}};
    REQUIRE_EQ(batch.init(program,cfg,chunk,geometry,ranks),RAD_OK);
    struct Finish {rad::BatchBuilder& b;~Finish(){b.fini();}} finish{batch};
    rad::StepPlan plan;plan.phase=RAD_PHASE_PREFILL;plan.n_tok=22;
    rad::StepEntry first;first.seq=10;first.card=0;first.ctx_len=12;first.n_tokens=17;first.is_prefill=true;plan.e.push_back(first);
    rad::StepEntry second;second.seq=20;second.card=1;second.ctx_len=0;second.n_tokens=5;second.is_prefill=true;plan.e.push_back(second);
    rad::KVBinding<rad::KVManager> memory(&kv);const RadBatch* staged=batch.build(plan,memory,1);REQUIRE(staged);REQUIRE_EQ(rad_stream_sync(stream),RAD_OK);
    fix::Fixture c(RAD_F32,0,2,8,2,5,4,4,{17,5},{12,0});
    std::vector<rad::KVPoolBinding> bindings;rad::bind_kv_pools(program,[&](int32_t id){return kv.plan(id);},&bindings);
    for(int g:{1,2}) {
        const int cache=g==1?a::window_cache:a::compressed_cache,table=g==1?a::window_table:a::compressed_table,used=g==1?a::window_used:a::compressed_used;
        const auto& view=bindings[size_t(g)];const auto& group_batch=staged->kv[g];
        REQUIRE_EQ(view.rank,4u);c.buffers[size_t(cache)].emplace(RAD_F32,std::initializer_list<int64_t>{view.shape[0],view.shape[1],view.shape[2],view.shape[3]},0);
        c.buffers[size_t(table)].emplace(RAD_I32,std::initializer_list<int64_t>{2,group_batch.block_table_pitch},0);
        copy_i32(c.buffer(table),group_batch.block_table);copy_i32(c.buffer(used),group_batch.seqused);
        for(int64_t i=0;i<c.buffer(cache).count();++i)c.buffer(cache).set(i,NAN);
    }
    copy_i32(c.buffer(a::positions),staged->positions);copy_i32(c.buffer(a::cu),staged->cu_seqlens);copy_i32(c.buffer(a::ctx_lens),staged->ctx_lens);
    const uint64_t ids[]{10,20};int64_t row=0;
    for(int64_t s=0;s<2;++s) {
        const int64_t end=c.starts[size_t(s)]+c.lengths[size_t(s)];c.firsts[size_t(s)]=int32_t(kv.first_block_pos(ids[s],1));
        for(int64_t pos=c.firsts[size_t(s)];pos<end;++pos) {
            const int64_t slot=geometry.slot(1,kv.block_table(ids[s],1),c.firsts[size_t(s)],pos);REQUIRE(slot>=0);
            for(int64_t d=0;d<c.dim;++d)c.buffer(a::window_cache).set(slot*c.dim+d,c.canonical_window[size_t(s)][size_t(pos*c.dim+d)]);
            if(pos>=c.starts[size_t(s)]) {CHECK_EQ(staged->kv[1].slot_mapping[row],slot);++row;}
        }
        rad::ds41::latent::Launch store;REQUIRE_EQ(state::make_layout(c.dim,2,4,4,&store.layout),RAD_OK);
        store.slots=kv.plan(2)->n_blocks*4;
        for(int64_t key=0;key<end/2;++key) {
            const int64_t final_position=key*2+1,final_slot=geometry.slot(2,kv.block_table(ids[s],2),0,final_position);int64_t first_slot=-1;
            REQUIRE_EQ(rad::ds41::latent::store_row(store,final_position,final_slot,&first_slot),RAD_OK);
            for(int64_t d=0;d<c.dim;++d)c.buffer(a::compressed_cache).set(first_slot*(c.dim/2)+d,c.canonical_compressed[size_t(s)][size_t(key*c.dim+d)]);
        }
    }
    auto args=c.args();a::Plan checked;REQUIRE_EQ(a::validate(&args,&checked),RAD_OK);
    for(int64_t s=0;s<2;++s)CHECK_EQ(a::first_window_position(checked,s),kv.first_block_pos(ids[s],1));
    check_output(c);
    // A premature trim to the chunk's END drops early-query keys. The reader must reject it.
    kv.trim_window(10,29);CHECK(kv.first_block_pos(10,1)>8);
    c.buffer(a::window_used).id(0,int32_t(29-kv.first_block_pos(10,1)));const auto output=c.buffer(a::output).words;
    CHECK_EQ(c.run(),RAD_E_STATE);CHECK(c.buffer(a::output).words==output);
}
TEST(ds41_paged_attention_ring_negative_control_exposes_future_overwrite_in_a_multiquery_chunk) {
    auto cfg=config();auto g=group("runtime-ring",RAD_KV_WINDOW,1,4);std::vector<rad::KVGroupInfo> groups{rad::KVGroupInfo{},g};
    rad::Pools pools;REQUIRE_EQ(pools.configure(cfg,0,&rad::rad_malloc_allocator(),&rad::rad_malloc_allocator()),RAD_OK);
    rad::KVManager kv;REQUIRE_EQ(kv.declare_runtime_window(1,4),RAD_OK);REQUIRE_EQ(kv.configure(groups,cfg,pools.kv()),RAD_OK);
    REQUIRE_EQ(kv.add_sequence(1),RAD_OK);REQUIRE_EQ(kv.set_window(1,0,4),RAD_OK);REQUIRE_EQ(kv.ensure(1,12),RAD_OK);
    rad::SeqWindow ring;REQUIRE(kv.runtime_window(1,1,&ring));const auto geometry=rad::KVGeom::of(groups);
    std::map<int64_t,int64_t> contents;
    for(int64_t p=0;p<12;++p)contents[geometry.slot_ring(1,kv.block_table(1,1),0,p,ring)]=p;
    const int64_t first_slot=geometry.slot_ring(1,kv.block_table(1,1),0,0,ring);
    CHECK_EQ(contents[first_slot],8);CHECK(contents[first_slot]>0); // a query at position 0 would read its future
    CHECK_EQ(first_slot,geometry.slot_ring(1,kv.block_table(1,1),0,8,ring));
}
RAD_TEST_MAIN()
