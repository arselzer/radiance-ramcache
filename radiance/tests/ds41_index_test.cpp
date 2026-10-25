/* Independent serial index equations/partial_sort versus the GPU's shared scalar/heap code. */
#include "rad_test.h"
#include "../libds41/ds41_index_fixture.h"
#include "../arch/common/rad_ds41_ref.h"
namespace x=rad::ds41::index;
namespace f=x::testing;
namespace ref=rad::deepseek41::ref;
namespace {
void expected(f::Fixture& test) {
    for(int64_t row=0;row<test.rows;++row) {
        const int64_t n=test.visible().id(row);
        if(test.op==x::Op::scores) {
            std::vector<float> q,k,w,s(size_t(n),0.f);
            for(int64_t h=0;h<test.heads;++h){w.push_back(test.buffer(2).get(row*test.heads+h));for(int64_t d=0;d<test.dim;++d)q.push_back(test.buffer(0).get((row*test.heads+h)*test.dim+d));}
            for(int64_t key=0;key<n;++key)for(int64_t d=0;d<test.dim;++d)k.push_back(test.buffer(1).get((test.buffer(3).id(row)*test.keys+key)*test.dim+d));
            REQUIRE_EQ(ref::index_scores(q,k,w,test.dim,test.heads*4,s),RAD_OK);
            for(int64_t key=0;key<test.keys;++key)test.buffer(5).set(row*test.keys+key,key<n?s[size_t(key)]:-INFINITY);
        } else {
            std::vector<float> scores;for(int64_t key=0;key<test.keys;++key)scores.push_back(test.buffer(0).get(row*test.keys+key));
            if(test.op==x::Op::candidates){
                std::vector<uint8_t> mask(size_t(test.keys),0);REQUIRE_EQ(ref::candidate_blocks(scores,n,test.block,test.topk,mask),RAD_OK);
                for(int64_t key=0;key<test.keys;++key)test.buffer(2).byte(row*test.keys+key,mask[size_t(key)]);
            } else {
                std::vector<uint8_t> mask;if(test.buffers[2])for(int64_t key=0;key<test.keys;++key)mask.push_back(test.buffer(2).byte(row*test.keys+key));
                std::vector<int32_t> ids;REQUIRE_EQ(ref::index_topk(scores,n,test.topk,mask,&ids),RAD_OK);
                for(size_t i=0;i<ids.size();++i)test.buffer(3).id(row*int64_t(ids.size())+i,ids[i]);
            }
        }
    }
}
void compare(f::Fixture& test) {
    auto oracle=test;expected(oracle);const auto before=test.buffers;REQUIRE_EQ(test.run(),RAD_OK);CHECK_EQ(test.status[0],RAD_OK);
    for(int i=0;i<test.output();++i)if(before[i])CHECK(test.buffer(i).words==before[i]->words);
    const auto& a=test.buffer(test.output());const auto& b=oracle.buffer(test.output());std::vector<bool> touched(a.words.size()*4,false);
    const int bytes=rad_dtype_bits(a.t.dtype)/8;
    for(int64_t i=0;i<a.count();++i){
        if(a.t.dtype==RAD_F32){if(std::isinf(b.get(i)))CHECK_EQ(a.get(i),b.get(i));else CHECK_NEAR(a.get(i),b.get(i),2e-6+2e-5*std::fabs(b.get(i)));}
        else if(a.t.dtype==RAD_U8)CHECK_EQ(a.byte(i),b.byte(i));else CHECK_EQ(a.id(i),b.id(i));
        for(int j=0;j<bytes;++j)touched[size_t(a.at(i)*bytes+j)]=true;
    }
    const auto* old=reinterpret_cast<const uint8_t*>(before[test.output()]->words.data());const auto* now=reinterpret_cast<const uint8_t*>(a.words.data());
    for(size_t i=0;i<touched.size();++i)if(!touched[i])CHECK_EQ(now[i],old[i]);
    CHECK_EQ(test.status[1],912);CHECK_EQ(test.status[2],913);CHECK_EQ(test.status[3],914);
}
void rejects(f::Fixture& test,int rc) {
    const auto before=test.buffers;CHECK_EQ(test.run(),rc);CHECK_EQ(test.status[0],rc);
    for(int i=0;i<test.nt();++i)if(before[i])CHECK(test.buffer(i).words==before[i]->words);
    CHECK_EQ(test.status[1],912);CHECK_EQ(test.status[2],913);CHECK_EQ(test.status[3],914);
}
}
TEST(ds41_index_scores_match_independent_equations_with_sequence_reorder_visibility_and_pitches) {
    for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16})for(int pitch=0;pitch<3;++pitch){f::Fixture test(x::Op::scores,dtype,pitch);compare(test);}
}
TEST(ds41_index_candidates_force_newest_block_and_keep_full_invisible_tail) {
    for(int pitch=0;pitch<3;++pitch)for(int top:{1,2,4}) {
        f::Fixture test(x::Op::candidates,RAD_F32,pitch,3,11,top);
        for(int64_t key=0;key<11;++key)test.buffer(0).set(2*11+key,key<8?100.f:float(-key));
        test.visible().id(2,9);test.buffer(0).set(2*11+10,NAN);compare(test);
        for(int key=8;key<11;++key)CHECK_EQ(test.buffer(2).byte(2*11+key),1);
        if(top==2){for(int key=0;key<4;++key)CHECK_EQ(test.buffer(2).byte(2*11+key),1);for(int key=4;key<8;++key)CHECK_EQ(test.buffer(2).byte(2*11+key),0);}
    }
}
TEST(ds41_index_topk_ties_position_sort_optional_mask_and_padding_match_reference) {
    for(int pitch=0;pitch<3;++pitch)for(int top:{1,4,16})for(bool mask:{false,true}){
        f::Fixture test(x::Op::topk,RAD_F32,pitch,3,11,top,mask);
        for(int64_t i=0;i<test.buffer(0).count();++i)test.buffer(0).set(i,float(i%4));
        compare(test);
    }
}
TEST(ds41_index_heaps_match_independent_partial_sort_across_random_capacities) {
    std::mt19937 rng(101753);
    for(int round=0;round<100;++round){
        const int keys=1+int(rng()%257),top=1+int(rng()%64),pitch=round%3;
        f::Fixture test(round&1?x::Op::topk:x::Op::candidates,RAD_F32,pitch,3,keys,top,round%3!=0);
        for(int row=0;row<3;++row)test.visible().id(row,int(rng()%unsigned(keys+1)));
        for(int64_t i=0;i<test.buffer(0).count();++i)test.buffer(0).set(i,float(int(rng()%17)-8));
        compare(test);
    }
}
TEST(ds41_index_full_capacity_heap_matches_independent_partial_sort) {
    for(auto op:{x::Op::candidates,x::Op::topk}) {
        f::Fixture test(op,RAD_F32,0,1,4097,4096,false);
        test.visible().id(0,4097);compare(test);
    }
}
TEST(ds41_index_c2_chunk_visibility_is_per_original_position_before_candidate_and_topk) {
    f::Fixture score(x::Op::scores,RAD_BF16,1,6,11,4);
    const int32_t positions[]{0,1,2,3,15,16};
    for(int row=0;row<6;++row){score.buffer(3).id(row,row<4?0:2);score.visible().id(row,(positions[row]+1)/2);}
    compare(score);
    f::Fixture candidates(x::Op::candidates,RAD_F32,2,6,11,1);
    f::Fixture selected(x::Op::topk,RAD_F32,1,6,11,4,true);
    for(int row=0;row<6;++row){
        const int32_t visible=(positions[row]+1)/2;candidates.visible().id(row,visible);selected.visible().id(row,visible);
        for(int key=0;key<11;++key){const float v=score.buffer(5).get(row*11+key);
            candidates.buffer(0).set(row*11+key,v);selected.buffer(0).set(row*11+key,v);}
    }
    compare(candidates);
    for(int row=0;row<6;++row)for(int key=0;key<11;++key)selected.buffer(2).byte(row*11+key,candidates.buffer(2).byte(row*11+key));
    compare(selected);
    for(int row=0;row<6;++row)for(int i=0;i<4;++i){
        const int id=selected.buffer(3).id(row*4+i);CHECK(id==-1||(id>=0&&id<(positions[row]+1)/2));
    }
}
TEST(ds41_index_payload_failures_are_global_atomic_and_unselected_cache_bytes_are_ignored) {
    for(auto op:{x::Op::scores,x::Op::candidates,x::Op::topk}) {
        f::Fixture low(op);low.visible().id(2,-1);rejects(low,RAD_E_SHAPE);
        f::Fixture high(op);high.visible().id(2,12);rejects(high,RAD_E_SHAPE);
        f::Fixture bad(op);bad.buffer(0).set(bad.buffer(0).count()-1,NAN);rejects(bad,RAD_E_INVAL);
    }
    f::Fixture badseq(x::Op::scores);badseq.buffer(3).id(2,3);rejects(badseq,RAD_E_STATE);
    f::Fixture selected(x::Op::scores);selected.buffer(1).set((2*11+10)*6+5,INFINITY);rejects(selected,RAD_E_INVAL);
    f::Fixture unselected(x::Op::scores);for(int i=0;i<11*6;++i)unselected.buffer(1).set(11*6+i,NAN);compare(unselected);
    f::Fixture tail(x::Op::topk);tail.visible().id(2,10);tail.buffer(0).set(2*11+10,NAN);compare(tail);
}
TEST(ds41_index_descriptor_errors_leave_scratch_and_outputs_untouched) {
    for(auto op:{x::Op::scores,x::Op::candidates,x::Op::topk}) {
        f::Fixture test(op);x::Plan plan;auto a=test.args();
        a.scratch=reinterpret_cast<void*>(UINTPTR_MAX-3);CHECK_EQ(x::validate(op,&a,&plan),RAD_E_SHAPE);
        a=test.args();a.scratch_bytes=3;CHECK_EQ(x::validate(op,&a,&plan),RAD_E_INVAL);
        a=test.args();test.tensors[test.output()].data=test.tensors[0].data;CHECK_EQ(x::validate(op,&a,&plan),RAD_E_INVAL);
        a=test.args();test.tensors[0].stride[0]=INT64_MAX;CHECK_EQ(x::validate(op,&a,&plan),RAD_E_SHAPE);
        CHECK_EQ(test.status[0],911);
    }
}
RAD_TEST_MAIN()
