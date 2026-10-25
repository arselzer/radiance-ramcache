/* Direct paged-index reads versus independent dense equations and actual core KV binding. */
#include "rad_test.h"
#include "../libds41/ds41_index_paged_fixture.h"
#include "../arch/common/rad_ds41_ref.h"
#include "../arch/deepseek41/state_layout.h"
#include "runtime/kvbind.h"
#include "mem/kv.h"
namespace p=rad::ds41::paged_index;
namespace f=p::testing;
namespace ref=rad::deepseek41::ref;
namespace state=rad::deepseek41::state;
namespace {
void compare(f::Fixture& test) {
    const auto before=test.buffers;
    std::vector<float> wanted(size_t(test.rows*test.keys),-INFINITY);
    for(int64_t row=0;row<test.rows;++row){
        const int64_t visible=test.buffer(4).id(row),seq=test.buffer(3).id(row);
        std::vector<float> q,k,w,s(size_t(visible),0.f);
        for(int64_t h=0;h<test.heads;++h){w.push_back(test.buffer(2).get(row*test.heads+h));for(int64_t d=0;d<test.dim;++d)q.push_back(test.buffer(0).get((row*test.heads+h)*test.dim+d));}
        for(int64_t key=0;key<visible;++key)for(int64_t d=0;d<test.dim;++d)k.push_back(test.buffer(1).get(test.physical_scalar(seq,key,d)));
        REQUIRE_EQ(ref::index_scores(q,k,w,test.dim,test.heads*4,s),RAD_OK);
        std::copy(s.begin(),s.end(),wanted.begin()+row*test.keys);
    }
    REQUIRE_EQ(test.run(),RAD_OK);CHECK_EQ(test.status[0],RAD_OK);
    for(int i=0;i<7;++i)if(before[i])CHECK(test.buffer(i).words==before[i]->words);
    const auto& output=test.buffer(7);std::vector<bool> touched(output.words.size(),false);
    for(int64_t i=0;i<output.count();++i){
        if(std::isinf(wanted[size_t(i)]))CHECK_EQ(output.get(i),wanted[size_t(i)]);
        else CHECK_NEAR(output.get(i),wanted[size_t(i)],2e-6+2e-5*std::fabs(wanted[size_t(i)]));
        touched[size_t(output.at(i))]=true;
    }
    for(size_t i=0;i<touched.size();++i)if(!touched[i])CHECK_EQ(output.words[i],before[7]->words[i]);
    CHECK_EQ(test.status[1],912);CHECK_EQ(test.status[2],913);CHECK_EQ(test.status[3],914);
}
void rejects(f::Fixture& test,int expected){
    const auto before=test.buffers;CHECK_EQ(test.run(),expected);CHECK_EQ(test.status[0],expected);
    for(int i=0;i<8;++i)if(before[i])CHECK(test.buffer(i).words==before[i]->words);
    CHECK_EQ(test.status[1],912);CHECK_EQ(test.status[2],913);CHECK_EQ(test.status[3],914);
}
}
TEST(ds41_paged_index_scores_match_dense_reference_across_ratio_page_crossings_and_pitches){
    for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16})for(int pitch=0;pitch<3;++pitch)
        for(int ratio:{1,2})for(int block:{ratio,4,8,16})for(int dim:{8,128}){
            f::Fixture test(dtype,pitch,ratio,block,dim,3,(pitch&1)==0);compare(test);
        }
}
TEST(ds41_paged_index_views_match_actual_core_binding_and_checked_original_slot_mapper){
    for(int ratio:{1,2}){
        f::Fixture test(RAD_BF16,0,ratio,4,128);
        state::Layout layout;REQUIRE_EQ(state::make_layout(test.dim,ratio,test.block,2,&layout),RAD_OK);
        rad::Program program;program.kv_groups.resize(2);auto& group=program.kv_groups[1];group.name="indexkeys";group.layers={2,8};
        REQUIRE_EQ(state::paged_decl(layout,RAD_BF16,RAD_KV_FULL,0,&group.decl),RAD_OK);
        rad::KVGroupPlan plan;plan.kind=RAD_KV_FULL;plan.n_layers=2;plan.n_blocks=test.physical_blocks;plan.block_size=test.block;
        plan.bytes_per_block=layout.page_bytes*2;plan.pool_bytes=plan.bytes_per_block*plan.n_blocks;
        std::vector<rad::KVPoolBinding> bindings;rad::bind_kv_pools(program,[&](int32_t g){return g==1?&plan:nullptr;},&bindings);
        REQUIRE_EQ(bindings.size(),2u);const auto& bound=bindings[1];CHECK_EQ(bound.dtype,uint32_t(RAD_BF16));CHECK_EQ(bound.rank,4u);
        for(int axis=0;axis<4;++axis)CHECK_EQ(bound.shape[axis],test.buffer(1).t.shape[axis]);
        for(int64_t seq=0;seq<test.sequences;++seq)for(int64_t key=0;key<test.keys;++key)for(int64_t d=0;d<test.dim;++d){
            const int64_t logical=key*ratio/test.block,physical=test.buffer(5).id(seq*test.blocks_per_sequence+logical);int64_t bytes=0;
            REQUIRE_EQ(state::physical_byte(layout,physical,key,d,&bytes),RAD_OK);CHECK_EQ(bytes,test.physical_scalar(seq,key,d)*2);
        }
    }
}
TEST(ds41_paged_index_invalid_selected_pages_origins_and_cache_payload_are_batch_atomic){
    f::Fixture missing;missing.buffer(5).id(2*missing.blocks_per_sequence+missing.blocks_per_sequence-1,-1);rejects(missing,RAD_E_STATE);
    f::Fixture beyond;beyond.buffer(5).id(2*beyond.blocks_per_sequence, int32_t(beyond.physical_blocks));rejects(beyond,RAD_E_STATE);
    f::Fixture origin;origin.buffer(6).id(2,4);rejects(origin,RAD_E_UNSUPPORTED);
    f::Fixture ids;ids.buffer(3).id(2,3);rejects(ids,RAD_E_STATE);
    f::Fixture high;high.buffer(4).id(2,12);rejects(high,RAD_E_SHAPE);
    f::Fixture negative;negative.buffer(5).id(2*negative.blocks_per_sequence,-2);rejects(negative,RAD_E_STATE);
    f::Fixture short_table;short_table.buffer(5).t.shape[1]--;rejects(short_table,RAD_E_SHAPE);
    f::Fixture selected;selected.buffer(1).set(selected.physical_scalar(2,10,7),NAN);rejects(selected,RAD_E_INVAL);
    f::Fixture unused;unused.buffer(6).id(1,123);
    for(int64_t page=0;page<unused.blocks_per_sequence;++page)unused.buffer(5).id(unused.blocks_per_sequence+page,-1);
    compare(unused);
}
TEST(ds41_paged_index_bad_descriptors_leave_scratch_and_output_untouched){
    f::Fixture test;const auto before=test.buffers;p::Plan plan;auto args=test.args();
    args.scratch=reinterpret_cast<void*>(UINTPTR_MAX-3);CHECK_EQ(p::validate(&args,&plan),RAD_E_SHAPE);
    args=test.args();test.tensors[1].shape[3]++;CHECK_EQ(p::validate(&args,&plan),RAD_E_SHAPE);
    args=test.args();test.tensors[7].data=test.tensors[1].data;CHECK_EQ(p::validate(&args,&plan),RAD_E_INVAL);
    args=test.args();test.tensors[5].stride[0]=INT64_MAX;CHECK_EQ(p::validate(&args,&plan),RAD_E_SHAPE);
    CHECK_EQ(test.status[0],911);for(int i=0;i<8;++i)if(before[i])CHECK(test.buffer(i).words==before[i]->words);
}
RAD_TEST_MAIN()
