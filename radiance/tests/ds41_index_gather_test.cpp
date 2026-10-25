/* CPU-only semantic proposal: gather whole completed local query heads and F32 unscaled
 * weights, then score all heads against replicated keys. No collectives or graph are added. */
#include "rad_test.h"
#include "../arch/common/rad_ds41_ref.h"
#include <bit>
#include <cmath>
#include <cstring>

namespace ref=rad::deepseek41::ref;
namespace {
constexpr size_t heads=32,dim=128,world=4,keys=19;
struct Fixture {
    std::vector<float> query,weights,key;
    explicit Fixture(bool bf16) :query(heads*dim),weights(heads),key(keys*dim) {
        for(size_t i=0;i<query.size();++i)query[i]=std::sin(float(i+3)*.071f)*.5f;
        for(size_t i=0;i<weights.size();++i)weights[i]=std::cos(float(i+2)*.19f);
        for(size_t i=0;i<key.size();++i)key[i]=std::cos(float(i+1)*.037f)*.7f;
        if(bf16) {
            for(auto& x:query)x=ref::rounded(x,ref::Precision::bf16);
            for(auto& x:key)x=ref::rounded(x,ref::Precision::bf16);
        }
    }
};
}
TEST(ds41_index_whole_head_gather_preserves_oracle_inputs_scores_and_selected_ids) {
    for(bool bf16:{false,true}) {
        Fixture f(bf16);std::vector<float> q(heads*dim),w(heads),baseline(keys),gathered(keys);
        // Canonical rank order with complete contiguous head partitions. The actual graph
        // must establish these ownership/ordering and transport contracts before use.
        for(size_t rank=0;rank<world;++rank) {
            const size_t first=rank*heads/world,count=heads/world;
            std::memcpy(q.data()+first*dim,f.query.data()+first*dim,count*dim*sizeof(float));
            std::memcpy(w.data()+first,f.weights.data()+first,count*sizeof(float));
        }
        CHECK(std::memcmp(q.data(),f.query.data(),q.size()*sizeof(float))==0);
        CHECK(std::memcmp(w.data(),f.weights.data(),w.size()*sizeof(float))==0);
        REQUIRE_EQ(ref::index_scores(f.query,f.key,f.weights,dim,heads,baseline),RAD_OK);
        REQUIRE_EQ(ref::index_scores(q,f.key,w,dim,heads,gathered),RAD_OK);
        CHECK(std::memcmp(baseline.data(),gathered.data(),keys*sizeof(float))==0);
        for(size_t visible:{size_t(0),size_t(1),size_t(8),keys}) {
            std::vector<int32_t> a,b;std::vector<uint8_t> mask(keys);
            REQUIRE_EQ(ref::candidate_blocks(baseline,visible,8,2,mask),RAD_OK);
            REQUIRE_EQ(ref::index_topk(baseline,visible,6,mask,&a),RAD_OK);
            REQUIRE_EQ(ref::index_topk(gathered,visible,6,mask,&b),RAD_OK);
            CHECK(a==b);
        }
    }
}
TEST(ds41_index_partial_head_score_sum_has_same_equation_with_reduction_tolerance) {
    for(bool bf16:{false,true}) {
        Fixture f(bf16);std::vector<float> full(keys),sum(keys,0.f),part(keys);
        REQUIRE_EQ(ref::index_scores(f.query,f.key,f.weights,dim,heads,full),RAD_OK);
        for(size_t rank=0;rank<world;++rank) {
            const size_t first=rank*heads/world,count=heads/world;
            REQUIRE_EQ(ref::index_scores(ref::Floats(f.query).subspan(first*dim,count*dim),f.key,
                ref::Floats(f.weights).subspan(first,count),dim,heads,part),RAD_OK);
            for(size_t k=0;k<keys;++k)sum[k]+=part[k];
        }
        // Grouping head sums changes F32 rounding; this is not a bitwise equivalence or
        // any promise about near-tie top-k identities in a different reduction tree.
        for(size_t k=0;k<keys;++k)CHECK_NEAR(sum[k],full[k],2e-6f+2e-5f*std::fabs(full[k]));
    }
}
TEST(ds41_index_gather_requires_complete_head_dots_before_relu_and_global_scale_once) {
    std::vector<float> query{1.f,1.f},key{1.f,-2.f},weight{1.f},score(1);
    REQUIRE_EQ(ref::index_scores(query,key,weight,2,1,score),RAD_OK);
    CHECK_EQ(score[0],0.f);
    const float wrong_relu_before_dim_reduce=(std::max(1.f,0.f)+std::max(-2.f,0.f))/std::sqrt(2.f);
    CHECK(wrong_relu_before_dim_reduce>score[0]);
    Fixture f(false);std::vector<float> correct(keys),wrong(keys);
    const size_t count=heads/world;
    const auto q=ref::Floats(f.query).first(count*dim),w=ref::Floats(f.weights).first(count);
    REQUIRE_EQ(ref::index_scores(q,f.key,w,dim,heads,correct),RAD_OK);
    REQUIRE_EQ(ref::index_scores(q,f.key,w,dim,count,wrong),RAD_OK);
    for(size_t k=0;k<keys;++k)CHECK_NEAR(wrong[k],correct[k]*2.f,2e-6f+2e-5f*std::fabs(wrong[k]));
}
TEST(ds41_index_compact_candidate_block_membership_reconstructs_current_oracle_mask_exactly) {
    // Representation proposal only, not a new operator. Source selection and tie handling
    // remain actual ref::candidate_blocks. Include partial/invisible selected block tails.
    constexpr size_t nkeys=41,block=8;
    std::vector<float> scores(nkeys);
    for(size_t k=0;k<nkeys;++k)scores[k]=float((k*7)%13);
    for(size_t visible:{size_t(0),size_t(1),size_t(17),size_t(37),nkeys}) {
        std::vector<uint8_t> mask(nkeys),restored(nkeys);
        REQUIRE_EQ(ref::candidate_blocks(scores,visible,block,2,mask),RAD_OK);
        std::vector<int32_t> block_ids;
        for(size_t first=0;first<nkeys;first+=block)if(mask[first])block_ids.push_back(int32_t(first/block));
        CHECK(block_ids.size()<=2);
        for(size_t k=0;k<nkeys;++k)
            restored[k]=std::find(block_ids.begin(),block_ids.end(),int32_t(k/block))!=block_ids.end();
        CHECK(mask==restored);
        std::vector<int32_t> old_ids,new_ids;
        REQUIRE_EQ(ref::index_topk(scores,visible,6,mask,&old_ids),RAD_OK);
        REQUIRE_EQ(ref::index_topk(scores,visible,6,restored,&new_ids),RAD_OK);
        CHECK(old_ids==new_ids);
    }
}
RAD_TEST_MAIN()
