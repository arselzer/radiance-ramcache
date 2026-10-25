#include "rad_test.h"
#include "../libds41/ds41_attention_paged_fixture.h"
#include "../libds41/ds41_index_paged_emulate.h"
#include "../libds41/ds41_compress_fixture.h"
#include "../libds41/ds41_transform_fixture.h"
#include "../arch/common/rad_ds41_ref.h"
#include <sys/mman.h>
namespace a=rad::ds41::attention_paged;
namespace f=a::testing;
namespace ref=rad::deepseek41::ref;
namespace {
void expected(f::Fixture& c) {
    int64_t row=0;
    for(int64_t seq=0;seq<c.sequences;++seq)for(int64_t i=0;i<c.lengths[size_t(seq)];++i,++row) {
        const int64_t pos=int64_t(c.starts[size_t(seq)])+i;
        auto joint=c.canonical_window[size_t(seq)];const int64_t offset=joint.size()/size_t(c.dim);
        joint.insert(joint.end(),c.canonical_compressed[size_t(seq)].begin(),c.canonical_compressed[size_t(seq)].end());
        std::vector<int32_t> ids;
        for(int64_t key=std::max<int64_t>(0,pos-c.window+1);key<=pos;++key)ids.push_back(int32_t(key));
        if(c.ratio)for(int64_t k=0;k<c.topk;++k) {const int32_t id=c.buffer(a::selected).id(row*c.topk+k);ids.push_back(id<0?-1:int32_t(offset+id));}
        std::vector<float> q,result(size_t(c.heads*c.dim),0.f),sinks;
        for(int64_t h=0;h<c.heads;++h) {sinks.push_back(c.buffer(a::sinks).get(h));for(int64_t d=0;d<c.dim;++d)q.push_back(c.buffer(a::query).get((row*c.heads+h)*c.dim+d));}
        REQUIRE_EQ(ref::sparse_attention(q,joint,size_t(c.heads),size_t(c.dim),sinks,ids,float(1./std::sqrt(double(c.dim))),result),RAD_OK);
        for(int64_t j=0;j<c.heads*c.dim;++j)c.buffer(a::output).set(row*c.heads*c.dim+j,result[size_t(j)]);
    }
}
void compare(f::Fixture& c) {
    auto oracle=c;expected(oracle);const auto before=c.buffers;REQUIRE_EQ(c.run(),RAD_OK);CHECK_EQ(c.flag[0],RAD_OK);
    for(int i=0;i<a::output;++i)if(c.buffers[size_t(i)])CHECK(c.buffer(i).words==before[size_t(i)]->words);
    CHECK(c.buffer(a::output).words==oracle.buffer(a::output).words);CHECK_EQ(c.flag[1],912);CHECK_EQ(c.flag[2],913);CHECK_EQ(c.flag[3],914);
}
void reject(f::Fixture& c,int status) {
    const auto before=c.buffers;CHECK_EQ(c.run(),status);CHECK_EQ(c.flag[0],status);
    for(int i=0;i<a::tensors_count;++i)if(c.buffers[size_t(i)])CHECK(c.buffer(i).words==before[size_t(i)]->words);
    CHECK_EQ(c.flag[1],912);CHECK_EQ(c.flag[2],913);CHECK_EQ(c.flag[3],914);
}
struct Inaccessible {
    void* data;size_t bytes;
    explicit Inaccessible(size_t capacity):bytes((capacity+4095)/4096*4096) {
        data=mmap(nullptr,bytes,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);if(data==MAP_FAILED)throw std::runtime_error("mmap failed");
    }
    ~Inaccessible(){munmap(data,bytes);}
};
}
TEST(ds41_attention_paged_matches_independent_dense_equations_for_two_sources_and_typed_pitches) {
    for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16})for(int pitch=0;pitch<3;++pitch)for(int ratio:{0,1,2})
        for(int block:{2,4,8})for(int dim:{8,128}) {f::Fixture c(dtype,pitch,ratio,dim,2,5,4,block);compare(c);}
}
TEST(ds41_attention_paged_c2_visibility_is_per_query_even_when_later_groups_already_exist) {
    f::Fixture c(RAD_F32,2,2,8,2,5,4,4,{7},{0});compare(c);
    for(int64_t row=0;row<c.rows;++row)for(int64_t k=0;k<c.topk;++k)CHECK(c.buffer(a::selected).id(row*c.topk+k)<(row+1)/2);
    auto future=c;future.buffer(a::selected).id(0,0);reject(future,RAD_E_SHAPE);
    auto final=c;final.buffer(a::selected).id(6*c.topk,3);reject(final,RAD_E_SHAPE);
    auto negative=c;negative.buffer(a::selected).id(6*c.topk,-2);reject(negative,RAD_E_SHAPE);
    auto incomplete=c;incomplete.buffer(a::compressed_used).id(0,6);reject(incomplete,RAD_E_STATE);
}
TEST(ds41_attention_paged_joint_sink_and_duplicate_multiplicity_match_one_attention_call) {
    for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16}) {
        f::Fixture repeated(dtype,1);for(int64_t row=0;row<repeated.rows;++row)for(int64_t k=0;k<repeated.topk;++k)repeated.buffer(a::selected).id(row*repeated.topk+k,0);compare(repeated);
        f::Fixture disabled(dtype,2);for(int64_t h=0;h<disabled.heads;++h)disabled.buffer(a::sinks).set(h,-INFINITY);compare(disabled);
        f::Fixture dominant(dtype);for(int64_t h=0;h<dominant.heads;++h)dominant.buffer(a::sinks).set(h,10000.f);compare(dominant);
        for(int64_t i=0;i<dominant.buffer(a::output).count();++i)CHECK_EQ(dominant.buffer(a::output).get(i),0.f);
    }
}
TEST(ds41_attention_paged_selected_page_and_payload_failures_are_globally_atomic) {
    f::Fixture page;page.buffer(a::window_table).id(page.buffer(a::window_table).count()-1,-1);reject(page,RAD_E_STATE);
    f::Fixture high;high.buffer(a::compressed_table).id(2*high.buffer(a::compressed_table).t.shape[1],INT32_MAX);reject(high,RAD_E_STATE);
    f::Fixture value;value.buffer(a::window_cache).set(value.scalar_index(false,2,13,7),NAN);reject(value,RAD_E_INVAL);
    f::Fixture comp;comp.buffer(a::selected).id((comp.rows-1)*comp.topk,0);comp.buffer(a::compressed_cache).set(comp.scalar_index(true,2,0,7),INFINITY);reject(comp,RAD_E_INVAL);
    f::Fixture query;query.buffer(a::query).set(query.buffer(a::query).count()-1,NAN);reject(query,RAD_E_INVAL);
    f::Fixture sink;sink.buffer(a::sinks).set(sink.heads-1,INFINITY);reject(sink,RAD_E_INVAL);
    f::Fixture unused;unused.buffer(a::ctx_lens).id(1,-123);unused.buffer(a::window_used).id(1,-456);unused.buffer(a::compressed_used).id(1,-789);
    for(int table:{a::window_table,a::compressed_table})for(int64_t col=0;col<unused.buffer(table).t.shape[1];++col)unused.buffer(table).id(unused.buffer(table).t.shape[1]+col,INT32_MAX);
    compare(unused);
}
TEST(ds41_attention_paged_stage_boundaries_prevent_page_or_cache_reads_after_earlier_invalid_metadata) {
    f::Fixture c;auto args=c.args();Inaccessible pages(c.buffer(a::window_table).words.size()*4),cache(c.buffer(a::window_cache).words.size()*4);
    const auto before=c.buffer(a::output).words;
    c.tensors[a::window_table].data=pages.data;c.tensors[a::compressed_table].data=pages.data;c.buffer(a::positions).id(c.rows-1,-1);
    CHECK_EQ(a::emulate(&args),RAD_E_SHAPE);CHECK(c.buffer(a::output).words==before);
    args=c.args();c.buffer(a::positions).id(c.rows-1,13);c.buffer(a::window_table).id(c.buffer(a::window_table).count()-1,INT32_MAX);
    c.tensors[a::window_cache].data=cache.data;c.tensors[a::compressed_cache].data=cache.data;
    CHECK_EQ(a::emulate(&args),RAD_E_STATE);CHECK(c.buffer(a::output).words==before);
}
TEST(ds41_attention_paged_rejects_geometry_aliases_context_gaps_and_speculation_without_writes) {
    for(auto name:{"n_spec","window","ratio","block_size"}) {
        f::Fixture c;c.integer(name,!std::strcmp(name,"n_spec")?1:!std::strcmp(name,"ratio")?3:0);auto args=c.args();a::Plan plan;const auto before=c.buffer(a::output).words;
        CHECK(a::validate(&args,&plan)<0);CHECK_EQ(c.flag[0],911);CHECK(c.buffer(a::output).words==before);
    }
    f::Fixture gap;gap.buffer(a::positions).id(gap.rows-1,12);reject(gap,RAD_E_SHAPE);
    f::Fixture first;first.buffer(a::window_used).id(2,13);reject(first,RAD_E_STATE);
    f::Fixture over;over.buffer(a::ctx_lens).id(2,INT32_MAX);reject(over,RAD_E_SHAPE);
    f::Fixture c;auto args=c.args();a::Plan plan;
    c.tensors[a::output].data=c.tensors[a::window_cache].data;CHECK(a::validate(&args,&plan)<0);args=c.args();
    c.tensors[a::window_table].stride[0]=INT64_MAX;CHECK(a::validate(&args,&plan)<0);args=c.args();
    c.tensors[a::compressed_cache].shape[3]++;CHECK(a::validate(&args,&plan)<0);args=c.args();args.scratch_bytes=3;CHECK(a::validate(&args,&plan)<0);
    args=c.args();args.scratch=reinterpret_cast<void*>(UINTPTR_MAX-3);CHECK(a::validate(&args,&plan)<0);CHECK_EQ(c.flag[0],911);
}
TEST(ds41_attention_paged_handles_the_largest_probability_workspace_without_changing_order) {
    f::Fixture c(RAD_F32,1,1,4,1,4096,4096,4,{1},{4099});compare(c);
}
TEST(ds41_attention_paged_composes_c2_pool_cast_norm_rope_and_paged_index_selection) {
    namespace comp=rad::ds41::compress;namespace tr=rad::ds41::transform;namespace ix=rad::ds41::index;
    comp::testing::Fixture pooled(8,2,RAD_F32,1,{7},{0},{2});REQUIRE_EQ(pooled.run(),RAD_OK);
    tr::testing::Fixture norm(tr::Op::rmsnorm,RAD_BF16,RAD_BF16,1,false,8),rope(tr::Op::rope_tail,RAD_BF16,RAD_BF16,1,false,8,4,1);
    for(int64_t d=0;d<8;++d)norm.buffers[1].set(d,1.f);
    int64_t emitted=0;
    for(int64_t r=0;r<7;++r)if(pooled.buffer(comp::emit).geti(r)) {
        for(int64_t d=0;d<8;++d)norm.buffers[0].set(emitted*8+d,pooled.buffer(comp::pooled).getf(r*8+d));
        rope.buffers[1].id(emitted,pooled.buffer(comp::first_position).geti(r));++emitted;
    }
    REQUIRE_EQ(emitted,3);REQUIRE_EQ(norm.run(),RAD_OK);
    for(int64_t i=0;i<24;++i)rope.buffers[0].set(i,norm.buffers[2].get(i));
    REQUIRE_EQ(rope.run(),RAD_OK);
    f::Fixture attention(RAD_BF16,1,2,8,2,5,3,4,{7},{0});attention.canonical_compressed[0]=rope.buffers[3].values();attention.rewrite_cache();
    ix::testing::Fixture score(ix::Op::scores,RAD_BF16,1,7,3,3,true,8,2);
    for(int64_t r=0;r<7;++r) {score.buffer(3).id(r,0);score.buffer(4).id(r,int32_t((r+1)/2));}
    score.refresh();std::array<RadTensor,8> tensors{score.tensors[0],attention.buffer(a::compressed_cache).t,score.tensors[2],score.tensors[3],score.tensors[4],attention.buffer(a::compressed_table).t,{},score.tensors[5]};
    // Fixture copies carry their own vectors: explicitly refresh the two externally borrowed cache descriptors.
    tensors[1].data=attention.buffer(a::compressed_cache).words.data();tensors[5].data=attention.buffer(a::compressed_table).words.data();
    auto parameters=score.params;parameters.push_back(RAD_INT("ratio",2));parameters.push_back(RAD_INT("block_size",4));
    RadArgs args{};args.t=tensors.data();args.n_t=8;args.p=parameters.data();args.n_p=parameters.size();args.scratch=score.status.data();args.scratch_bytes=4;
    REQUIRE_EQ(rad::ds41::paged_index::emulate(&args),RAD_OK);
    ix::testing::Fixture top(ix::Op::topk,RAD_F32,1,7,3,3,false);
    for(int64_t r=0;r<7;++r) {top.buffer(1).id(r,int32_t((r+1)/2));for(int64_t j=0;j<3;++j)top.buffer(0).set(r*3+j,score.buffer(5).get(r*3+j));}
    REQUIRE_EQ(top.run(),RAD_OK);
    for(int64_t r=0;r<7;++r)for(int64_t j=0;j<3;++j)attention.buffer(a::selected).id(r*3+j,top.buffer(3).id(r*3+j));
    compare(attention);
    for(int64_t j=0;j<3;++j)CHECK_EQ(attention.buffer(a::selected).id(j),-1);
}
RAD_TEST_MAIN()
