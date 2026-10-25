/* Prove DS41 opaque latent packing against real current KV binding/planning metadata.
 * No pool is carved, device is touched, model payload is read or architecture is registered. */
#include "rad_test.h"
#include "../arch/deepseek41/state_layout.h"
#include "mem/kv.h"
#include "runtime/kvbind.h"
#include <array>

namespace state=rad::deepseek41::state;
namespace {
state::Layout layout(int64_t dim=512,int64_t ratio=2,int64_t block=16,int64_t bytes=2) {
    state::Layout l;
    if(state::make_layout(dim,ratio,block,bytes,&l)!=RAD_OK)throw std::runtime_error("invalid test layout");
    return l;
}
std::vector<rad::KVGroupInfo> window_groups() {
    std::vector<rad::KVGroupInfo> result(2);auto& g=result[1];g.name="DS41-window";g.layers={0,3};
    const auto l=layout(512,1);if(state::paged_decl(l,RAD_BF16,RAD_KV_WINDOW,128,&g.decl)!=RAD_OK)
        throw std::runtime_error("window metadata failed");
    g.block_size=l.block_size;return result;
}
rad::Config config() {rad::Config c;c.max_seqs=2;c.max_ctx=256;c.max_tok=16;c.n_spec=0;return c;}
}

TEST(deepseek41_state_layout_has_exact_contiguous_latent_and_index_byte_budgets) {
    for(int64_t dim:{512,128})for(int64_t ratio:{1,2})for(int64_t bytes:{1,2,4,8}) {
        const auto l=layout(dim,ratio,16,bytes);CHECK(state::valid(l));
        CHECK_EQ(l.head_dim,dim/(2*ratio));CHECK_EQ(l.entry_bytes,dim*bytes);
        CHECK_EQ(l.slot_bytes,dim*bytes/ratio);CHECK_EQ(l.page_bytes,16*dim*bytes/ratio);
        for(int64_t tokens:{0,1,2,15,16,17,31,32,33}) {
            state::Budget b;REQUIRE_EQ(state::budget(l,tokens,3,&b),RAD_OK);
            CHECK_EQ(b.completed_entries,tokens/ratio);CHECK_EQ(b.pending_tokens,tokens%ratio);
            CHECK_EQ(b.pages,tokens/16+(tokens%16!=0));
            CHECK_EQ(b.payload_bytes,(tokens/ratio)*dim*bytes*3);
            CHECK_EQ(b.reserved_bytes,b.pages*16*dim*bytes/ratio*3);
            CHECK(b.reserved_bytes>=b.payload_bytes);
        }
    }
    CHECK_EQ(layout(512,2).head_dim,128);CHECK_EQ(layout(128,2).head_dim,32);
    CHECK_EQ(layout(512,1).head_dim,256);
}
TEST(deepseek41_state_layout_bijection_covers_each_physical_scalar_once_without_overlap) {
    constexpr int64_t physical_pages[]{2,0,1};
    for(int64_t dim:{512,128})for(int64_t ratio:{1,2})for(int64_t block:{2,4,16}) {
        const auto l=layout(dim,ratio,block);const int64_t per_page=block/ratio;
        std::vector<bool> seen(size_t(3*l.page_bytes/l.element_bytes),false);
        for(int64_t entry=0;entry<3*per_page;++entry)for(int64_t e=0;e<dim;++e) {
            state::Address a;REQUIRE_EQ(state::map(l,entry,e,&a),RAD_OK);
            CHECK_EQ(a.original_token,entry*ratio+e/(dim/ratio));
            CHECK_EQ(a.logical_block,entry/per_page);CHECK_EQ(a.slot,a.original_token%block);
            CHECK(a.plane==0||a.plane==1);CHECK(a.column>=0&&a.column<l.head_dim);
            CHECK_EQ(a.page_element,(entry%per_page)*dim+e);
            CHECK_EQ(a.page_byte,a.page_element*l.element_bytes);
            int64_t offset=0;REQUIRE_EQ(state::physical_byte(l,physical_pages[a.logical_block],entry,e,&offset),RAD_OK);
            const size_t scalar=size_t(offset/l.element_bytes);REQUIRE(scalar<seen.size());CHECK(!seen[scalar]);seen[scalar]=true;
        }
        CHECK(std::all_of(seen.begin(),seen.end(),[](bool b){return b;}));
    }
}
TEST(deepseek41_state_layout_pages_and_window_table_floors_remain_original_token_coordinates) {
    for(int64_t ratio:{1,2}) {
        const auto l=layout(512,ratio);const int64_t page_entries=16/ratio;
        for(int64_t entry:{page_entries-1,page_entries,page_entries+1,3*page_entries-1}) {
            state::Address first,last;
            REQUIRE_EQ(state::map(l,entry,0,&first),RAD_OK);REQUIRE_EQ(state::map(l,entry,511,&last),RAD_OK);
            CHECK_EQ(first.logical_block,last.logical_block);CHECK_EQ(last.page_byte-first.page_byte,511*2);
        }
        int64_t index=-1;
        REQUIRE_EQ(state::table_index(l,2*page_entries,511,32,2,&index),RAD_OK);CHECK_EQ(index,0);
        REQUIRE_EQ(state::table_index(l,3*page_entries,0,32,2,&index),RAD_OK);CHECK_EQ(index,1);
        CHECK(state::table_index(l,page_entries,0,32,2,&index)<0);
        CHECK(state::table_index(l,4*page_entries,0,32,2,&index)<0);
        CHECK(state::table_index(l,2*page_entries,0,31,2,&index)<0);
    }
}
TEST(deepseek41_state_layout_partial_c2_prefix_exposes_only_completed_entries) {
    for(int64_t ratio:{1,2}) {
        const auto l=layout(512,ratio);
        for(int64_t committed=0;committed<51;++committed) {
            state::Prefix p;REQUIRE_EQ(state::prefix(l,committed,&p),RAD_OK);
            CHECK_EQ(p.completed_entries,committed/ratio);CHECK_EQ(p.pending_tokens,committed%ratio);
            CHECK_EQ(p.pending_first_token,committed-committed%ratio);
            state::Address a{91,92,93,94,95,96,97},before=a;
            CHECK_EQ(state::map_completed(l,committed,p.completed_entries,0,&a),RAD_E_STATE);
            CHECK(std::memcmp(&a,&before,sizeof a)==0);
            if(p.completed_entries) {
                REQUIRE_EQ(state::map_completed(l,committed,p.completed_entries-1,511,&a),RAD_OK);
                CHECK(a.original_token<committed);CHECK_EQ(a.original_token,(committed/ratio)*ratio-1);
            }
        }
    }
}
TEST(deepseek41_state_layout_refuses_invalid_geometry_and_checks_all_arithmetic_before_output) {
    state::Layout l=layout(),before=l;
    for(const auto args:{std::array<int64_t,4>{0,2,16,2},{511,2,16,2},{512,3,16,2},
                         {512,2,15,2},{512,2,0,2},{512,2,16,3},{INT64_MAX-3,1,16,8},
                         {512,1,INT64_MAX,2}}) {
        CHECK(state::make_layout(args[0],args[1],args[2],args[3],&l)<0);CHECK(std::memcmp(&l,&before,sizeof l)==0);
    }
    CHECK(state::make_layout(512,2,16,2,nullptr)<0);
    state::Address a{1,2,3,4,5,6,7},original=a;
    for(const auto pair:{std::array<int64_t,2>{-1,0},{0,-1},{0,512},{INT64_MAX,0},{INT64_MAX/2+1,511}}) {
        CHECK(state::map(l,pair[0],pair[1],&a)<0);CHECK(std::memcmp(&a,&original,sizeof a)==0);
    }
    auto bad=l;++bad.head_dim;CHECK(!state::valid(bad));CHECK(state::map(bad,0,0,&a)<0);
    int64_t offset=77;CHECK(state::physical_byte(l,INT64_MAX,0,0,&offset)<0);CHECK_EQ(offset,77);
    CHECK(state::physical_byte(l,-1,0,0,&offset)<0);CHECK_EQ(offset,77);
    state::Budget budget{1,2,3,4,5},old_budget=budget;
    CHECK(state::budget(l,INT64_MAX,1,&budget)<0);CHECK(std::memcmp(&budget,&old_budget,sizeof budget)==0);
    CHECK(state::budget(l,16,INT64_MAX,&budget)<0);CHECK(state::budget(l,16,0,&budget)<0);
    state::Prefix p;CHECK(state::prefix(l,-1,&p)<0);REQUIRE_EQ(state::prefix(l,INT64_MAX,&p),RAD_OK);
    CHECK_EQ(p.completed_entries,INT64_MAX/2);CHECK_EQ(p.pending_tokens,1);
    CHECK(state::map(l,0,0,nullptr)<0);CHECK(state::prefix(l,0,nullptr)<0);CHECK(state::budget(l,0,1,nullptr)<0);
}
TEST(deepseek41_state_metadata_declares_existing_paged_and_fixed_abi_shapes_only) {
    const auto l=layout();RadKVGroupDecl d{};
    REQUIRE_EQ(state::paged_decl(l,RAD_BF16,RAD_KV_FULL,0,&d),RAD_OK);
    CHECK_EQ(d.n_head_kv,1);CHECK_EQ(d.head_dim,128);CHECK_EQ(d.kind,RAD_KV_FULL);
    REQUIRE_EQ(state::paged_decl(layout(512,1),RAD_BF16,RAD_KV_WINDOW,128,&d),RAD_OK);
    CHECK_EQ(d.head_dim,256);CHECK_EQ(d.window,128);
    CHECK(state::paged_decl(l,RAD_F32,RAD_KV_FULL,0,&d)<0);
    CHECK(state::paged_decl(l,RAD_BF16,RAD_KV_WINDOW,0,&d)<0);
    CHECK(state::paged_decl(l,RAD_BF16,RAD_KV_FULL,1,&d)<0);
    CHECK(state::paged_decl(l,RAD_BF16,RAD_KV_LINEAR,0,&d)<0);
    REQUIRE_EQ(state::pending_c2_decl(512,RAD_F32,&d),RAD_OK);
    CHECK_EQ(d.kind,RAD_KV_LINEAR);CHECK_EQ(d.state_dim[0],2);CHECK_EQ(d.state_dim[1],512);
    CHECK_EQ(d.state_dim[0]*d.state_dim[1]*rad_dtype_bytes(d.dtype,1),4096);
    REQUIRE_EQ(state::engram_history_decl(4,&d),RAD_OK);CHECK_EQ(d.dtype,uint32_t(RAD_I32));CHECK_EQ(d.state_dim[1],3);
    CHECK(state::fixed_decl(RAD_U4,2,512,&d)<0);CHECK(state::fixed_decl(RAD_F32,INT64_MAX,2,&d)<0);
    CHECK(state::engram_history_decl(1,&d)<0);CHECK(state::pending_c2_decl(0,RAD_F32,&d)<0);
}
TEST(deepseek41_state_actual_core_kv_binding_is_token_interleaved_and_layer_major) {
    for(int64_t dim:{512,128})for(int64_t ratio:{1,2}) {
        const auto l=layout(dim,ratio);auto groups=window_groups();
        REQUIRE_EQ(state::paged_decl(l,RAD_BF16,RAD_KV_WINDOW,128,&groups[1].decl),RAD_OK);
        rad::Program program;program.kv_groups=groups;
        rad::KVGroupPlan plan;plan.kind=RAD_KV_WINDOW;plan.n_layers=2;plan.n_blocks=3;plan.block_size=16;
        plan.bytes_per_block=l.page_bytes*plan.n_layers;plan.pool_bytes=plan.bytes_per_block*plan.n_blocks;
        std::vector<rad::KVPoolBinding> bound;
        rad::bind_kv_pools(program,[&](int32_t g)->const rad::KVGroupPlan* {return g==1?&plan:nullptr;},&bound);
        REQUIRE_EQ(bound.size(),2u);const auto& b=bound[1];CHECK_EQ(b.rank,4u);
        CHECK_EQ(b.shape[0],3);CHECK_EQ(b.shape[1],1);CHECK_EQ(b.shape[2],16);CHECK_EQ(b.shape[3],dim/ratio);
        CHECK_EQ(b.layer_stride,3*l.page_bytes);CHECK_EQ(b.layer_slot[0],0);CHECK_EQ(b.layer_slot[3],1);
        for(int64_t entry=0;entry<48/ratio;++entry)for(int64_t e=0;e<dim;++e) {
            state::Address a;REQUIRE_EQ(state::map(l,entry,e,&a),RAD_OK);int64_t byte=0;
            REQUIRE_EQ(state::physical_byte(l,a.logical_block,entry,e,&byte),RAD_OK);
            const int64_t actual=(((a.logical_block*b.shape[1])*b.shape[2]+a.slot)*b.shape[3]+a.plane*l.head_dim+a.column)*2;
            CHECK_EQ(byte,actual);
        }
    }
}
TEST(deepseek41_state_actual_core_planning_rejects_window_snapshot_reuse_with_linear_state) {
    auto groups=window_groups();auto cfg=config();rad::KVManager window;
    REQUIRE_EQ(window.plan_groups(groups,cfg),RAD_OK);const auto* wp=window.plan(1);REQUIRE(wp);
    CHECK_EQ(wp->bytes_per_block,16*512*2*2);CHECK(wp->win_snap);CHECK(window.has_window_snapshot());
    rad::KVGroupInfo pending;pending.name="pending-C2";pending.layers={2};
    REQUIRE_EQ(state::pending_c2_decl(512,RAD_F32,&pending.decl),RAD_OK);groups.push_back(pending);
    cfg.n_spec=3;rad::KVManager hybrid;REQUIRE_EQ(hybrid.plan_groups(groups,cfg),RAD_OK);
    const auto* hp=hybrid.plan(1);const auto* state=hybrid.plan(2);REQUIRE(hp);REQUIRE(state);
    CHECK(!hp->win_snap);CHECK(!hybrid.has_window_snapshot());CHECK_EQ(state->bytes_per_state,4096);
    CHECK_EQ(state->state_copies,2);CHECK_EQ(state->n_states,4);
}
RAD_TEST_MAIN()
