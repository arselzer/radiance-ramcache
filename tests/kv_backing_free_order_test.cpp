/* Compare real legacy/backing KVManager ID traces. Metadata maps only: no GPU, no cache payload
 * reads/writes, and no large allocation is touched. The host VMem reference maps anonymous VA. */
#include "rad_test.h"
#include "mem/kv.h"
#include "mem/kv_backing_allocator.h"
#include "device/device.h"
#include "rad_internal.h"
#include <algorithm>
#include <vector>
using namespace rad;
namespace {
constexpr int64_t MiB = int64_t(1) << 20;
constexpr int G_WIN=1, G_CONV=2, G_TAIL=3, G_I2=4, G_C2=5, G_I1=6, G_C1=7;
struct Fixture {
    Pools pools;
    KVManager kv;
    KVManagerBackingPublisher pub{&kv};
    KVBackingSnapshot map;
    bool quiet = true;
    std::vector<std::vector<KVBackingBlock>> calls;
    int build(bool backing, bool native512=false) {
        Config c; c.vram_weights_mib=c.gpu_headroom_mib=1; c.host_pool_mib=0;
        c.vram_kv_mib=native512 ? 1607 : 64;
        c.max_ctx=native512 ? 524288 : 4096; c.max_seqs=native512 ? 1 : 4;
        c.max_tok=16; c.n_spec=0; c.prefix_cache=!native512; c.checkpoint_slots=0;
        std::vector<KVGroupInfo> groups(1);
        KVGroupInfo w; w.name="kv_window"; w.decl.kind=RAD_KV_WINDOW; w.decl.dtype=RAD_BF16;
        w.decl.n_head_kv=1; w.decl.head_dim=256; w.decl.window=128; w.block_size=16;
        for (int l=0;l<40;++l) w.layers.push_back(l); groups.push_back(w);
        KVGroupInfo conv; conv.name="kv_engram_tokens"; conv.decl.kind=RAD_KV_CONV;
        conv.decl.dtype=RAD_I32; conv.decl.n_head_kv=1; conv.decl.head_dim=1;
        conv.decl.conv_width=4; conv.layers={1,14}; groups.push_back(conv);
        KVGroupInfo tail; tail.name="kv_comp_tail_r2"; tail.decl.kind=RAD_KV_LINEAR;
        tail.decl.dtype=RAD_F32; tail.decl.n_head_kv=1; tail.decl.state_dim[0]=2;
        tail.decl.state_dim[1]=1024; tail.layers={2,8,14}; groups.push_back(tail);
        const auto full=[&](const char* name,int hd,int bs,std::vector<int> layers) {
            KVGroupInfo g; g.name=name; g.decl.kind=RAD_KV_FULL; g.decl.dtype=RAD_BF16;
            g.decl.n_head_kv=1; g.decl.head_dim=hd; g.block_size=bs; g.layers=std::move(layers);
            groups.push_back(std::move(g));
        };
        full("kv_index_r2",32,2,{2,8,14}); full("kv_compressed_r2",128,2,{2,8,14});
        full("kv_index_r1",64,1,{20}); full("kv_compressed_r1",256,1,{20});
        int rc=pools.configure(c,0,&rad_malloc_allocator(),&rad_malloc_allocator()); if(rc<0)return rc;
        rc=kv.plan_groups(groups,c); if(rc<0)return rc;
        if(backing){rc=kv.enable_backing_mode();if(rc<0)return rc;}
        rc=kv.carve(pools.kv());if(rc<0)return rc;
        map.reserved_bytes=pools.kv().backing_reserved_bytes();map.granule_bytes=pools.kv().backing_granule_bytes();
        if(map.granule_bytes!=2*MiB)return RAD_E_UNSUPPORTED;
        map.backed.assign((size_t)(map.reserved_bytes/map.granule_bytes),0);
        if(backing){rc=publish();if(rc<0)return rc;grow();}
        return RAD_OK;
    }
    KVBackingSnapshot target(const std::vector<KVBackingBlock>& ids) {
        std::vector<KVBackingRange> required,extra;
        CHECK_EQ(kv.backing_required_ranges(kv,&required),RAD_OK);
        CHECK_EQ(kv.backing_block_ranges(ids,&extra),RAD_OK);
        required.insert(required.end(),extra.begin(),extra.end());
        KVBackingLimits limits{map.reserved_bytes,map.granule_bytes,0,map.reserved_bytes,0,0};
        KVBackingPlan p; CHECK(kv_backing_plan(limits,map.backed,required,&p)==KVBackingResult::Ok);
        auto out=map;out.backed=p.target;return out;
    }
    int publish(const std::vector<KVBackingBlock>& ids={}) {
        if(!quiet)return RAD_E_STATE;
        const auto next=target(ids);int rc=pub.stage(map,next);if(rc<0)return rc;
        pub.withdraw();for(int g=1;g<kv.n_groups();++g)CHECK_EQ(kv.free_blocks(g),int64_t(0));
        map=next;pub.publish_target();return kv.backing_error();
    }
    void grow() {
        kv.set_backing_grow([this](const std::vector<KVBackingBlock>& ids){
            if(!quiet)return KVBackingGrowResult{RAD_E_STATE,true};
            CHECK(quiet);calls.push_back(ids);
            for(size_t i=1;i<ids.size();++i)CHECK(ids[i-1].group==ids[i].group&&ids[i-1].block<ids[i].block);
            const int rc=publish(ids);return KVBackingGrowResult{rc,rc<0};
        });
    }
};
bool host(){return device_is_host()&&VMem::supported()&&VMem::granularity()==2*MiB;}
void compare(const Fixture& old,const Fixture& now,const std::vector<uint64_t>& seqs) {
    CHECK_EQ(old.kv.n_groups(),now.kv.n_groups());
    for(int g=1;g<old.kv.n_groups();++g){
        CHECK_EQ(old.kv.total_blocks(g),now.kv.total_blocks(g));
        CHECK_EQ(old.kv.held_blocks(g),now.kv.held_blocks(g));
        CHECK_EQ(old.kv.available_blocks(g),now.kv.available_blocks(g));
        CHECK(now.kv.free_blocks(g)<=old.kv.free_blocks(g));
        for(uint64_t id:seqs){
            CHECK(old.kv.block_table(id,g)==now.kv.block_table(id,g));
            CHECK_EQ(old.kv.first_block_pos(id,g),now.kv.first_block_pos(id,g));
            CHECK_EQ(old.kv.state_slot(id,g),now.kv.state_slot(id,g));
            for(int32_t b:now.kv.block_table(id,g)){
                CHECK(now.kv.block_backed(g,b));
                CHECK_EQ(old.kv.block_refcount(g,b),now.kv.block_refcount(g,b));
            }
        }
    }
}
void copies(Fixture& old,Fixture& now){
    const auto a=old.kv.take_pending_copies(),b=now.kv.take_pending_copies();CHECK_EQ(a.size(),b.size());
    if(a.size()!=b.size())return;
    for(size_t i=0;i<a.size();++i){CHECK_EQ(a[i].group,b[i].group);CHECK_EQ(a[i].src_block,b[i].src_block);CHECK_EQ(a[i].dst_block,b[i].dst_block);}
}
} // namespace

TEST(kv_backing_free_order_native512_g483_g484_and_neighbor_g547_never_skip_low_id) {
    if(!host())SKIP("host VMem metadata oracle requires 2MiB granules");
    Fixture old,now;REQUIRE_EQ(old.build(false,true),RAD_OK);REQUIRE_EQ(now.build(true,true),RAD_OK);
    const auto* i=now.kv.plan(G_I1);const auto* v=now.kv.plan(G_C1);
    REQUIRE(i&&v);CHECK_EQ(now.map.reserved_bytes,int64_t(1686110208));
    CHECK_EQ(i->pool_offset,int64_t(1013919744));CHECK_EQ(i->n_blocks,int64_t(524313));
    CHECK_EQ(v->pool_offset,int64_t(1148145664));CHECK_EQ(i->layer_bytes_per_block(),int64_t(256));
    CHECK_EQ(i->pool_offset+4304*256,int64_t(484)*now.map.granule_bytes);
    REQUIRE_EQ(old.kv.reserve_specific_block(G_C1,0),RAD_OK);
    REQUIRE_EQ(now.kv.reserve_specific_block(G_C1,0),RAD_OK); // G547 also backs the high index tail
    REQUIRE_EQ(now.publish({{G_I1,0}}),RAD_OK);
    CHECK(now.kv.block_backed(G_I1,520400));CHECK(!now.kv.block_backed(G_I1,4304));
    // A repaired/rolled-back publication must retain the same global candidate order.
    const auto next=now.target({{G_I1,4304}});
    REQUIRE_EQ(now.pub.stage(now.map,next),RAD_OK);now.pub.withdraw();now.pub.publish_old();
    CHECK(!now.kv.block_backed(G_I1,4304));
    std::vector<int32_t> a,b;
    REQUIRE_EQ(old.kv.reserve_blocks(G_I1,4304,&a),int64_t(4304));
    REQUIRE_EQ(now.kv.reserve_blocks(G_I1,4304,&b),int64_t(4304));REQUIRE(a==b);
    for(int k=0;k<4304;++k)CHECK_EQ(b[(size_t)k],k);
    // Backed high-tail IDs must not make ordinary lower-ID budget refusal fall through.
    now.kv.set_backing_grow([](const std::vector<KVBackingBlock>& ids){
        CHECK_EQ(ids.size(),size_t(1));
        if(ids.size()!=1)return KVBackingGrowResult{RAD_E_STATE,true};
        CHECK_EQ(ids[0].group,G_I1);CHECK_EQ(ids[0].block,4304);
        return KVBackingGrowResult{RAD_E_FULL,false};
    });
    int32_t denied=99;CHECK_EQ(now.kv.reserve_block(G_I1,&denied),RAD_E_FULL);CHECK_EQ(denied,-1);
    CHECK_EQ(now.kv.backing_error(),RAD_OK);CHECK_EQ(now.kv.block_refcount(G_I1,4304),0);
    now.grow();int32_t x=-1,y=-1;
    REQUIRE_EQ(old.kv.reserve_block(G_I1,&x),RAD_OK);REQUIRE_EQ(now.kv.reserve_block(G_I1,&y),RAD_OK);
    CHECK_EQ(x,4304);CHECK_EQ(y,x);CHECK(now.kv.block_backed(G_I1,y));
    CHECK(now.map.backed[484]);CHECK(now.map.backed[547]);
    REQUIRE(!now.calls.empty());CHECK_EQ(now.calls.back()[0].block,4304);
    for(int32_t q:a)old.kv.release_block(G_I1,q);
    for(int32_t q:b)now.kv.release_block(G_I1,q);
    old.kv.release_block(G_I1,x);now.kv.release_block(G_I1,y);
    old.kv.release_block(G_C1,0);now.kv.release_block(G_C1,0);REQUIRE_EQ(now.publish(),RAD_OK);
    compare(old,now,{});
}

TEST(kv_backing_free_order_native_paired_page_traces_match_legacy_across_boundary_chunks) {
    if(!host())SKIP("host VMem metadata oracle requires 2MiB granules");
    Fixture old,now;REQUIRE_EQ(old.build(false,true),RAD_OK);REQUIRE_EQ(now.build(true,true),RAD_OK);
    REQUIRE_EQ(old.kv.add_sequence(1),RAD_OK);REQUIRE_EQ(now.kv.add_sequence(1),RAD_OK);
    int64_t pos=0;
    for(int64_t n:{1,16,31,2144,2151,2152,2153,4303,4304,4305}){
        // The native window is sized for M16: retain its real per-chunk lifetime instead of
        // asking ensure to keep the keys of an impossible multi-thousand-row launch alive.
        while(pos<n){
            const int64_t next=std::min(n,pos+16);
            REQUIRE_EQ(old.kv.ensure(1,next),RAD_OK);REQUIRE_EQ(now.kv.ensure(1,next),RAD_OK);
            pos=next;
        }
        compare(old,now,{1});
        CHECK(now.kv.block_table(1,G_I2)==now.kv.block_table(1,G_C2));
        CHECK(now.kv.block_table(1,G_I1)==now.kv.block_table(1,G_C1));
    }
    CHECK(now.map.backed[484]); // actual index physical page4304 is now covered
    old.kv.free_sequence(1);now.kv.free_sequence(1);REQUIRE_EQ(now.publish(),RAD_OK);
    compare(old,now,{1});
}

TEST(kv_backing_free_order_prefix_cow_adopt_cancel_cleanup_and_window_restore_match_legacy) {
    if(!host())SKIP("host VMem metadata oracle requires 2MiB granules");
    Fixture old,now;REQUIRE_EQ(old.build(false),RAD_OK);REQUIRE_EQ(now.build(true),RAD_OK);
    REQUIRE_EQ(old.kv.add_sequence(1),RAD_OK);REQUIRE_EQ(now.kv.add_sequence(1),RAD_OK);
    REQUIRE_EQ(old.kv.ensure(1,3),RAD_OK);REQUIRE_EQ(now.kv.ensure(1,3),RAD_OK);compare(old,now,{1});
    std::vector<std::vector<int32_t>> prefix((size_t)old.kv.n_groups());
    for(int g:{G_I2,G_C2,G_I1,G_C1}){
        const size_t n=(size_t)(2/old.kv.plan(g)->block_size);
        const auto& table=old.kv.block_table(1,g);REQUIRE(table.size()>=n);
        prefix[(size_t)g].assign(table.begin(),table.begin()+n);
        for(int32_t p:prefix[(size_t)g]){REQUIRE_EQ(old.kv.retain_block(g,p),RAD_OK);REQUIRE_EQ(now.kv.retain_block(g,p),RAD_OK);}
    }
    REQUIRE_EQ(old.kv.fork(1,2),RAD_OK);REQUIRE_EQ(now.kv.fork(1,2),RAD_OK);
    REQUIRE_EQ(old.kv.ensure(2,4),RAD_OK);REQUIRE_EQ(now.kv.ensure(2,4),RAD_OK);compare(old,now,{1,2});
    old.kv.free_sequence(1);now.kv.free_sequence(1);old.kv.free_sequence(2);now.kv.free_sequence(2);
    REQUIRE_EQ(now.publish(),RAD_OK); // pending COW copies still pin source and destination
    copies(old,now);REQUIRE_EQ(now.publish(),RAD_OK);
    REQUIRE_EQ(old.kv.add_sequence(3),RAD_OK);REQUIRE_EQ(now.kv.add_sequence(3),RAD_OK);
    for(int g:{G_I2,G_C2,G_I1,G_C1}){
        REQUIRE_EQ(old.kv.adopt(3,g,prefix[(size_t)g],2),RAD_OK);
        REQUIRE_EQ(now.kv.adopt(3,g,prefix[(size_t)g],2),RAD_OK);
    }
    REQUIRE_EQ(old.kv.ensure(3,7),RAD_OK);REQUIRE_EQ(now.kv.ensure(3,7),RAD_OK);compare(old,now,{3});
    old.kv.free_sequence(3);now.kv.free_sequence(3);
    for(int g:{G_I2,G_C2,G_I1,G_C1})for(int32_t p:prefix[(size_t)g]){old.kv.release_block(g,p);now.kv.release_block(g,p);}
    REQUIRE_EQ(now.publish(),RAD_OK);
    REQUIRE_EQ(old.kv.add_sequence(4),RAD_OK);REQUIRE_EQ(now.kv.add_sequence(4),RAD_OK);
    REQUIRE_EQ(old.kv.ensure(4,5),RAD_OK);REQUIRE_EQ(now.kv.ensure(4,5),RAD_OK);
    old.kv.free_sequence(4);now.kv.free_sequence(4);copies(old,now);REQUIRE_EQ(now.publish(),RAD_OK); // cancel-like retirement
    REQUIRE(old.kv.has_window_snapshot());REQUIRE(now.kv.has_window_snapshot());
    REQUIRE_EQ(old.kv.add_sequence(5),RAD_OK);REQUIRE_EQ(now.kv.add_sequence(5),RAD_OK);
    REQUIRE_EQ(old.kv.seed_window(5,512),RAD_OK);REQUIRE_EQ(now.kv.seed_window(5,512),RAD_OK);
    CHECK(!now.kv.block_table(5,G_WIN).empty());compare(old,now,{5});
    REQUIRE_EQ(old.kv.ensure(5,513),RAD_OK);REQUIRE_EQ(now.kv.ensure(5,513),RAD_OK);compare(old,now,{5});
    old.kv.free_sequence(5);now.kv.free_sequence(5);copies(old,now);REQUIRE_EQ(now.publish(),RAD_OK);
    compare(old,now,{1,2,3,4,5});
}

TEST(kv_backing_free_order_bulk_specific_restore_duplicate_entries_and_fatal_callback_fail_closed) {
    if(!host())SKIP("host VMem metadata oracle requires 2MiB granules");
    Fixture old,now;REQUIRE_EQ(old.build(false),RAD_OK);REQUIRE_EQ(now.build(true),RAD_OK);
    for(int g:{G_I2,G_C2,G_I1,G_C1}){
        // Specific restore leaves a stale heap entry. Release makes a duplicate; N-peek must
        // collect distinct candidates without changing the legacy returned IDs.
        REQUIRE_EQ(old.kv.reserve_specific_block(g,0),RAD_OK);REQUIRE_EQ(now.kv.reserve_specific_block(g,0),RAD_OK);
        old.kv.release_block(g,0);now.kv.release_block(g,0);
        std::vector<int32_t> a,b;REQUIRE_EQ(old.kv.reserve_blocks(g,5,&a),int64_t(5));
        REQUIRE_EQ(now.kv.reserve_blocks(g,5,&b),int64_t(5));CHECK(a==b);
        for(int k=0;k<5;++k)CHECK_EQ(b[(size_t)k],k);
        for(int32_t q:a)old.kv.release_block(g,q);for(int32_t q:b)now.kv.release_block(g,q);
    }
    REQUIRE_EQ(now.publish(),RAD_OK);compare(old,now,{});
    int32_t missing=-1;for(int32_t b=0;b<now.kv.total_blocks(G_C1);++b)if(!now.kv.block_backed(G_C1,b)){missing=b;break;}
    REQUIRE(missing>=0);
    now.quiet=false; // same real callback interface, outside its required pre-plan lease
    CHECK_EQ(now.kv.reserve_specific_block(G_C1,missing),RAD_E_STATE);
    CHECK(now.kv.backing_error()<0);CHECK_EQ(now.kv.free_blocks(G_C1),int64_t(0));
    int32_t out=99;CHECK(now.kv.reserve_block(G_C1,&out)<0);CHECK_EQ(out,-1);
    std::vector<int32_t> denied;CHECK(now.kv.reserve_blocks(G_I1,1,&denied)<0);CHECK(denied.empty());
    CHECK(now.kv.reserve_specific_block(G_I1,0)<0);now.pub.poison();now.pub.poison();
    CHECK_EQ(now.kv.free_blocks(G_I1),int64_t(0));
}
RAD_TEST_MAIN()
