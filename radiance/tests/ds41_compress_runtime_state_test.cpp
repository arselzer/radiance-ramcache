/* Exercise opaque compressor state through the actual KV pool/checkpoint implementation.
 * Host backend only. Tiny synthetic projections; no architecture registration or serving I/O. */
#include "rad_test.h"
#include "device/device.h"
#include "mem/kv.h"
#include "mem/pools.h"
#include "runtime/kvbind.h"
#include "../libds41/ds41_compress_fixture.h"
#include <cstring>

namespace c=rad::ds41::compress;
namespace f=c::testing;
using namespace rad;
namespace {
void good(int rc) {if(rc!=RAD_OK)throw std::runtime_error("runtime state fixture status "+std::to_string(rc));}
struct StatePool {
    static constexpr int group=1,dim=8;
    Config cfg;Pools pools;HostArena checkpoints;KVManager kv;Program program;
    std::vector<KVPoolBinding> bindings;RadStream stream=nullptr;
    StatePool() {
        cfg.max_seqs=4;cfg.max_ctx=64;cfg.max_tok=16;cfg.n_spec=0;
        cfg.vram_weights_mib=1;cfg.vram_kv_mib=1;cfg.gpu_headroom_mib=1;cfg.host_pool_mib=0;cfg.checkpoint_slots=2;
        program.kv_groups.resize(2);auto& g=program.kv_groups[group];g.name="DS41-pending-U32";g.layers={0,2};
        good(c::state_decl(dim,RAD_U32,0,&g.decl));
        good(pools.configure(cfg,0,&rad_malloc_allocator(),&rad_malloc_allocator()));
        good(kv.configure(program.kv_groups,cfg,pools.kv()));
        good(checkpoints.open(kv.checkpoint_bytes(),kv.checkpoint_bytes()*cfg.checkpoint_slots));
        kv.set_checkpoint_store(&checkpoints,0);
        bind_kv_pools(program,[&](int32_t i){return kv.plan(i);},&bindings);
        good(rad_stream_create(&stream,0));
        std::memset(kv.plan(group)->base,0xA5,size_t(kv.plan(group)->pool_bytes));
        good(kv.add_sequence(101));good(kv.add_sequence(202));good(kv.add_sequence(303));
        good(kv.ensure(101,1));good(kv.ensure(202,1));good(kv.ensure(303,1));
        // Same raw-byte fresh-state protocol as Engine::clear_fresh_states().
        for(const auto& clear:kv.state_clears()) {
            const auto* p=kv.plan(clear.group);
            for(int layer=0;layer<p->n_layers;++layer)
                good(rad_memset_async(kv.state_ptr(clear.group,layer,clear.slot),0,p->layer_bytes_per_state(),stream));
        }
        good(rad_stream_sync(stream));kv.clear_state_clears();
    }
    ~StatePool(){if(stream){static_cast<void>(rad_stream_sync(stream));rad_stream_destroy(stream);}}
    int slot(uint64_t seq) const {return kv.state_slot(seq,group);}
    std::vector<int32_t> slots(uint64_t seq) const {return {-1,slot(seq)};}
    std::span<uint32_t> state(int layer,int slot) {
        return {static_cast<uint32_t*>(kv.state_ptr(group,layer,slot)),size_t(2*(dim+1))};
    }
    RadTensor tensor(int layer=0) {
        RadTensor t{};t.dtype=RAD_U32;t.rank=4;t.data=kv.state_ptr(group,layer,0);
        int64_t stride=1;
        for(int axis=3;axis>=0;--axis){t.shape[axis]=bindings[group].shape[axis];t.stride[axis]=stride;stride*=t.shape[axis];}
        return t;
    }
    std::vector<uint32_t> all() const {
        const auto* p=kv.plan(group);const auto* ptr=static_cast<const uint32_t*>(p->base);
        return {ptr,ptr+p->pool_bytes/4};
    }
    int run(f::Fixture& x,int layer=0) {
        auto args=x.args();x.tensors[c::pending_state]=tensor(layer);return c::emulate(&args);
    }
};
void logical_inputs(f::Fixture& x) {
    int64_t row=0;
    for(size_t seq=0;seq<x.lengths.size();++seq)for(int j=0;j<x.lengths[seq];++j,++row)
        for(int64_t d=0;d<x.dimension;++d){
            x.buffer(c::projected_kv).setf(row*x.dimension+d,f::Fixture::kv_value(0,x.starts[seq]+j,d));
            x.buffer(c::gate_scores).setf(row*x.dimension+d,f::Fixture::score_value(0,x.starts[seq]+j,d));
        }
}
}
TEST(ds41_u32_pending_state_checkpoint_preserves_every_bit_and_other_sequence_slots) {
    if(!device_is_host())SKIP("reads tiny host pool bytes only");
    StatePool p;const auto* plan=p.kv.plan(StatePool::group);REQUIRE(plan);
    CHECK_EQ(plan->state_copies,1);CHECK_EQ(plan->n_states,4);
    CHECK_EQ(plan->layer_bytes_per_state(),2*(StatePool::dim+1)*4);
    CHECK_EQ(p.bindings[StatePool::group].dtype,uint32_t(RAD_U32));
    CHECK_EQ(p.bindings[StatePool::group].layer_slot[2],1);
    const uint32_t patterns[]{0x7F800001,0xFF800000,0x7FC0DEAD,0x80000000,0xFFFFFFFF,0x01000001};
    for(int layer=0;layer<2;++layer)for(int slot=0;slot<plan->n_states;++slot) {
        auto words=p.state(layer,slot);
        for(size_t i=0;i<words.size();++i)words[i]=patterns[i%6]^uint32_t((layer*8+slot)<<8);
    }
    const auto before=p.all();int32_t checkpoint=-1;good(p.kv.alloc_checkpoint(&checkpoint));
    const auto source=p.slots(101),target=p.slots(202);
    good(p.kv.checkpoint_save(source.data(),source.size(),checkpoint,p.stream));good(rad_stream_sync(p.stream));
    CHECK(p.all()==before);
    good(p.kv.checkpoint_restore(target.data(),target.size(),checkpoint,p.stream));good(rad_stream_sync(p.stream));
    for(int layer=0;layer<2;++layer) {
        CHECK(std::equal(p.state(layer,p.slot(202)).begin(),p.state(layer,p.slot(202)).end(),p.state(layer,p.slot(101)).begin()));
        for(int slot=0;slot<plan->n_states;++slot)if(slot!=p.slot(202)) {
            const size_t at=size_t(layer*plan->layer_stride()/4+slot*plan->layer_bytes_per_state()/4);
            CHECK(std::equal(p.state(layer,slot).begin(),p.state(layer,slot).end(),before.begin()+at));
        }
    }
    p.kv.free_checkpoint(checkpoint);
}
TEST(ds41_c2_pending_resume_from_another_core_slot_matches_continuous_pooling) {
    if(!device_is_host())SKIP("CPU emulator over real tiny host pool");
    StatePool p;
    f::Fixture prefix(StatePool::dim,2,RAD_F32,0,{3},{0},{p.slot(101)});logical_inputs(prefix);
    REQUIRE_EQ(p.run(prefix),RAD_OK);CHECK_EQ(prefix.flag[0],RAD_OK);
    CHECK_EQ(p.state(0,p.slot(101))[StatePool::dim],3u);
    int32_t checkpoint=-1;good(p.kv.alloc_checkpoint(&checkpoint));const auto source=p.slots(101),target=p.slots(202);
    good(p.kv.checkpoint_save(source.data(),source.size(),checkpoint,p.stream));good(rad_stream_sync(p.stream));
    const auto prefix_state=std::vector<uint32_t>(p.state(0,p.slot(101)).begin(),p.state(0,p.slot(101)).end());
    std::fill(p.state(0,p.slot(101)).begin(),p.state(0,p.slot(101)).end(),0xDEADC0DEu);
    good(p.kv.checkpoint_restore(target.data(),target.size(),checkpoint,p.stream));good(rad_stream_sync(p.stream));
    CHECK(std::equal(p.state(0,p.slot(202)).begin(),p.state(0,p.slot(202)).end(),prefix_state.begin()));
    f::Fixture tail(StatePool::dim,2,RAD_F32,0,{5},{3},{p.slot(202)});logical_inputs(tail);
    REQUIRE_EQ(p.run(tail),RAD_OK);
    f::Fixture continuous(StatePool::dim,2,RAD_F32,0,{8},{0},{0});logical_inputs(continuous);REQUIRE_EQ(continuous.run(),RAD_OK);
    for(int64_t row=0;row<5;++row) {
        CHECK_EQ(tail.buffer(c::emit).geti(row),continuous.buffer(c::emit).geti(row+3));
        CHECK_EQ(tail.buffer(c::first_position).geti(row),continuous.buffer(c::first_position).geti(row+3));
        for(int64_t d=0;d<StatePool::dim;++d)
            CHECK_EQ(tail.buffer(c::pooled).getf(row*StatePool::dim+d),continuous.buffer(c::pooled).getf((row+3)*StatePool::dim+d));
    }
    for(int plane=0;plane<2;++plane)for(int d=0;d<=StatePool::dim;++d)
        CHECK_EQ(p.state(0,p.slot(202))[plane*(StatePool::dim+1)+d],continuous.state_get(0,plane,d));
    // An unrelated sequence cannot borrow the pending state by claiming a warm cursor.
    f::Fixture alien(StatePool::dim,2,RAD_F32,0,{1},{3},{p.slot(303)});logical_inputs(alien);
    std::fill(p.state(0,p.slot(303)).begin(),p.state(0,p.slot(303)).end(),0u);const auto before=p.all();
    CHECK_EQ(p.run(alien),RAD_E_STATE);CHECK(p.all()==before);
    p.kv.free_checkpoint(checkpoint);
}
RAD_TEST_MAIN()
