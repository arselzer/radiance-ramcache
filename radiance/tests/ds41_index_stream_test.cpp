/* Streaming selections versus both existing dense operators and independent canonical equations.
 * Only the test oracle owns dense score/mask planes; the implementation does not. */
#include "rad_test.h"
#include "../libds41/ds41_index_stream_fixture.h"
#include "../arch/common/rad_ds41_ref.h"
#include <sys/mman.h>
namespace s=rad::ds41::index_stream;
namespace f=s::testing;
namespace ix=rad::ds41::index;
namespace ref=rad::deepseek41::ref;
namespace {
void expected(f::Fixture& c){
    ix::testing::Fixture dense(ix::Op::scores,c.dtype,c.pitch,c.rows,c.keys,c.topk,true,c.dim,c.heads);
    for(auto& p:dense.params)if(!std::strcmp(p.key,"global_heads"))p.ival=c.heads;
    for(int64_t i=0;i<c.buffer(s::query).count();++i)dense.buffer(0).set(i,c.buffer(s::query).get(i));
    for(int64_t i=0;i<int64_t(c.canonical_keys.size());++i)dense.buffer(1).set(i,c.canonical_keys[size_t(i)]);
    for(int64_t i=0;i<c.buffer(s::head_weights).count();++i)dense.buffer(2).set(i,c.buffer(s::head_weights).get(i));
    for(int64_t r=0;r<c.rows;++r){dense.buffer(3).id(r,c.buffer(s::sequence_ids).id(r));dense.buffer(4).id(r,c.buffer(s::visible).id(r));}
    REQUIRE_EQ(dense.run(),RAD_OK);
    for(int64_t r=0;r<c.rows;++r){const int64_t n=c.buffer(s::visible).id(r),seq=c.buffer(s::sequence_ids).id(r);std::vector<float> q,w,key,values(size_t(n),0.f);
        for(int64_t h=0;h<c.heads;++h){w.push_back(c.buffer(s::head_weights).get(r*c.heads+h));for(int64_t d=0;d<c.dim;++d)q.push_back(c.buffer(s::query).get((r*c.heads+h)*c.dim+d));}
        key.insert(key.end(),c.canonical_keys.begin()+seq*c.keys*c.dim,c.canonical_keys.begin()+(seq*c.keys+n)*c.dim);
        if(n)REQUIRE_EQ(ref::index_scores(q,key,w,size_t(c.dim),size_t(c.heads),values),RAD_OK);
        for(int64_t k=0;k<c.keys;++k)CHECK_EQ(dense.buffer(5).get(r*c.keys+k),k<n?values[size_t(k)]:-INFINITY);
    }
    std::vector<uint8_t> mask(size_t(c.rows*c.keys),0);
    if(c.mode==s::Mode::source){
        ix::testing::Fixture blocks(ix::Op::candidates,RAD_F32,c.pitch,c.rows,c.keys,c.top_blocks);
        for(auto& p:blocks.params)if(!std::strcmp(p.key,"block_size"))p.ival=c.candidate_width;
        for(int64_t i=0;i<c.rows*c.keys;++i)blocks.buffer(0).set(i,dense.buffer(5).get(i));
        for(int64_t r=0;r<c.rows;++r)blocks.buffer(1).id(r,c.buffer(s::visible).id(r));
        REQUIRE_EQ(blocks.run(),RAD_OK);
        for(int64_t r=0;r<c.rows;++r){std::vector<float> scores;std::vector<uint8_t> independent(size_t(c.keys),0);std::vector<int32_t> ids;
            for(int64_t k=0;k<c.keys;++k)scores.push_back(dense.buffer(5).get(r*c.keys+k));
            REQUIRE_EQ(ref::candidate_blocks(scores,size_t(c.buffer(s::visible).id(r)),size_t(c.candidate_width),size_t(c.top_blocks),independent),RAD_OK);
            for(int64_t k=0;k<c.keys;++k)CHECK_EQ(blocks.buffer(2).byte(r*c.keys+k),independent[size_t(k)]);
            for(int64_t b=0;b*c.candidate_width<c.keys;++b)if(independent[size_t(b*c.candidate_width)])ids.push_back(int32_t(b));
            for(int64_t i=0;i<c.top_blocks;++i)c.buffer(s::candidates_out).id(r*c.top_blocks+i,i<int64_t(ids.size())?ids[size_t(i)]:-1);
        }
    } else if(c.mode==s::Mode::consumer)for(int64_t r=0;r<c.rows;++r)for(int64_t b=0;b<c.top_blocks;++b){const auto id=c.buffer(s::candidates_in).id(r*c.top_blocks+b);if(id<0)continue;
        for(int64_t k=int64_t(id)*c.candidate_width;k<std::min(c.keys,int64_t(id+1)*c.candidate_width);++k)mask[size_t(r*c.keys+k)]=1;}
    ix::testing::Fixture top(ix::Op::topk,RAD_F32,c.pitch,c.rows,c.keys,c.topk,c.mode==s::Mode::consumer);
    for(int64_t i=0;i<c.rows*c.keys;++i){top.buffer(0).set(i,dense.buffer(5).get(i));if(c.mode==s::Mode::consumer)top.buffer(2).byte(i,mask[size_t(i)]);}
    for(int64_t r=0;r<c.rows;++r)top.buffer(1).id(r,c.buffer(s::visible).id(r));
    REQUIRE_EQ(top.run(),RAD_OK);
    for(int64_t r=0;r<c.rows;++r){std::vector<float> scores;for(int64_t k=0;k<c.keys;++k)scores.push_back(dense.buffer(5).get(r*c.keys+k));std::vector<int32_t> wanted;
        const std::span<const uint8_t> eligible=c.mode==s::Mode::consumer?std::span<const uint8_t>(mask).subspan(size_t(r*c.keys),size_t(c.keys)):std::span<const uint8_t>{};
        REQUIRE_EQ(ref::index_topk(scores,size_t(c.buffer(s::visible).id(r)),size_t(c.topk),eligible,&wanted),RAD_OK);
        for(int64_t k=0;k<c.topk;++k){CHECK_EQ(top.buffer(3).id(r*c.topk+k),wanted[size_t(k)]);c.buffer(s::selected_out).id(r*c.topk+k,wanted[size_t(k)]);}
    }
}
void compare(f::Fixture& c){auto oracle=c;expected(oracle);const auto before=c.buffers;REQUIRE_EQ(c.run(),RAD_OK);CHECK_EQ(c.flag[0],RAD_OK);
    for(int i=0;i<s::selected_out;++i)if(c.buffers[size_t(i)])CHECK(c.buffer(i).words==before[size_t(i)]->words);
    for(int i=s::selected_out;i<s::operand_count;++i)if(c.buffers[size_t(i)])CHECK(c.buffer(i).words==oracle.buffer(i).words);
    CHECK_EQ(c.flag[1],912);CHECK_EQ(c.flag[2],913);CHECK_EQ(c.flag[3],914);
}
void reject(f::Fixture& c,int wanted){const auto before=c.buffers;CHECK_EQ(c.run(),wanted);CHECK_EQ(c.flag[0],wanted);
    for(int i=0;i<s::operand_count;++i)if(c.buffers[size_t(i)])CHECK(c.buffer(i).words==before[size_t(i)]->words);
    CHECK_EQ(c.flag[1],912);}
struct Inaccessible{void* data;size_t count;explicit Inaccessible(size_t bytes):count((bytes+4095)/4096*4096){data=mmap(nullptr,count,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);if(data==MAP_FAILED)throw std::runtime_error("mmap");}~Inaccessible(){munmap(data,count);}};
}
TEST(ds41_index_stream_all_modes_match_dense_and_canonical_scores_masks_selection){
    for(auto mode:{s::Mode::source,s::Mode::consumer,s::Mode::plain})for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16})for(int pitch=0;pitch<3;++pitch)
        for(int ratio:{1,2})for(int keys:{11,129})for(int dim:{4,128}){f::Fixture c(mode,dtype,pitch,ratio,keys,dim);compare(c);}
}
TEST(ds41_index_stream_source_topk_is_not_restricted_to_its_own_forced_newest_candidate){
    f::Fixture c(s::Mode::source,RAD_F32,1,2,17,4,1,1,1,8);
    for(int64_t i=0;i<c.buffer(s::query).count();++i)c.buffer(s::query).set(i,0);
    for(int64_t r=0;r<c.rows;++r)c.buffer(s::query).set(r*c.dim,1);
    std::fill(c.canonical_keys.begin(),c.canonical_keys.end(),0.f);
    for(int64_t seq=0;seq<c.sequences;++seq){c.canonical_keys[size_t(seq*c.keys*c.dim)]=10;c.canonical_keys[size_t((seq*c.keys+16)*c.dim)]=1;}c.rewrite_keys();
    compare(c);CHECK_EQ(c.buffer(s::selected_out).id(2),0);CHECK_EQ(c.buffer(s::candidates_out).id(2),2);
}
TEST(ds41_index_stream_compact_source_can_feed_different_consumer_scores_and_c2_ragged_rows){
    for(int pitch=0;pitch<3;++pitch){f::Fixture source(s::Mode::source,RAD_BF16,pitch,2,19,8,2,4,3,3,7);
        const int32_t seq[]{0,0,0,0,2,2,2},pos[]{0,1,3,6,2,5,7};for(int r=0;r<7;++r){source.buffer(s::sequence_ids).id(r,seq[r]);source.buffer(s::visible).id(r,(pos[r]+1)/2);}compare(source);
        f::Fixture consumer(s::Mode::consumer,RAD_BF16,pitch,2,19,8,2,4,3,3,7);
        for(int r=0;r<7;++r){consumer.buffer(s::sequence_ids).id(r,seq[r]);consumer.buffer(s::visible).id(r,(pos[r]+1)/2);
            for(int k=0;k<3;++k)consumer.buffer(s::candidates_in).id(r*3+k,source.buffer(s::candidates_out).id(r*3+k));}
        for(int64_t i=0;i<consumer.buffer(s::query).count();++i)consumer.buffer(s::query).set(i,-consumer.buffer(s::query).get(i));
        compare(consumer);
        for(int r=0;r<7;++r)for(int k=0;k<4;++k)CHECK(consumer.buffer(s::selected_out).id(r*4+k)<(pos[r]+1)/2);
    }
}
TEST(ds41_index_stream_ties_newest_partial_block_zero_visibility_and_heap_limits){
    for(auto mode:{s::Mode::source,s::Mode::consumer,s::Mode::plain}){f::Fixture zero(mode,RAD_F32,2,2,131,4,2,4,3,3);
        for(int64_t i=0;i<zero.buffer(s::query).count();++i)zero.buffer(s::query).set(i,0);
        compare(zero);
        for(int64_t r=0;r<zero.rows;++r)zero.buffer(s::visible).id(r,0);
        if(mode==s::Mode::consumer)zero.default_candidates();
        compare(zero);
        for(int64_t i=0;i<zero.buffer(s::selected_out).count();++i)CHECK_EQ(zero.buffer(s::selected_out).id(i),-1);
    }
    f::Fixture maximum(s::Mode::source,RAD_F32,0,2,16389,4,1,1024,2048,8,1);maximum.buffer(s::visible).id(0,int32_t(maximum.keys));compare(maximum);
    auto args=maximum.args();s::Plan p;REQUIRE_EQ(s::validate(&args,&p),RAD_OK);CHECK(s::shared_bytes(p)+32<32768);
}
TEST(ds41_index_stream_consumer_ignores_unselected_pages_and_invisible_whole_block_tail){
    for(bool tail:{false,true}){f::Fixture c(s::Mode::consumer,RAD_F32,2);
        for(int64_t i=0;i<c.buffer(s::candidates_in).count();++i)c.buffer(s::candidates_in).id(i,-1);
        c.buffer(s::visible).id(2,9);c.buffer(s::candidates_in).id(2*c.top_blocks,tail?1:0);
        c.buffer(s::cache).set(c.scalar_index(2,tail?9:8,c.dim-1),NAN);c.buffer(s::block_table).id(2*c.columns+5,INT32_MAX);compare(c);
    }
}
TEST(ds41_index_stream_invalid_lists_pages_payloads_and_derived_scores_are_batch_atomic){
    for(int fault=0;fault<6;++fault){f::Fixture c(s::Mode::consumer,RAD_F32,1,2,25,8,2,4,3,8);
        const int64_t first=2*c.top_blocks;c.buffer(s::candidates_in).id(first,0);c.buffer(s::candidates_in).id(first+1,1);c.buffer(s::candidates_in).id(first+2,2);
        if(fault==0)c.buffer(s::candidates_in).id(first+1,0);
        if(fault==1){c.buffer(s::candidates_in).id(first,1);c.buffer(s::candidates_in).id(first+1,0);}
        if(fault==2)c.buffer(s::candidates_in).id(first+1,-1);
        if(fault==3)c.buffer(s::candidates_in).id(first+1,-2);
        if(fault==4)c.buffer(s::candidates_in).id(first+2,4);
        if(fault==5)c.buffer(s::visible).id(2,0);
        reject(c,RAD_E_SHAPE);
    }
    for(auto mode:{s::Mode::source,s::Mode::consumer,s::Mode::plain})for(int fault=0;fault<8;++fault){f::Fixture c(mode);
        if(fault==0)c.buffer(s::query).set(c.buffer(s::query).count()-1,NAN);
        if(fault==1)c.buffer(s::head_weights).set(c.buffer(s::head_weights).count()-1,INFINITY);
        if(fault==2)c.buffer(s::cache).set(c.scalar_index(2,0,c.dim-1),NAN);
        if(fault==3)c.buffer(s::block_table).id(2*c.columns,INT32_MAX);
        if(fault==4)c.buffer(s::sequence_ids).id(2,3);
        if(fault==5)c.buffer(s::visible).id(2,int32_t(c.keys+1));
        if(fault==6)c.buffer(s::origins).id(2,4);
        if(fault==7){for(int64_t i=0;i<c.buffer(s::query).count();++i)c.buffer(s::query).set(i,1e30f);for(int64_t i=0;i<c.buffer(s::cache).count();++i)c.buffer(s::cache).set(i,1e30f);}
        reject(c,fault==3||fault==4?RAD_E_STATE:fault==5?RAD_E_SHAPE:fault==6?RAD_E_UNSUPPORTED:RAD_E_INVAL);
    }
}
TEST(ds41_index_stream_validates_metadata_then_all_pages_before_reading_cache_or_publishing){
    f::Fixture c(s::Mode::consumer);auto args=c.args();Inaccessible pages(c.buffer(s::block_table).words.size()*4);const auto before=c.buffers;
    c.tensors[s::block_table].data=pages.data;c.buffer(s::candidates_in).id(2*c.top_blocks,-2);CHECK_EQ(s::emulate(&args),RAD_E_SHAPE);
    for(int out:{s::selected_out,s::candidates_out})if(before[size_t(out)])CHECK(c.buffer(out).words==before[size_t(out)]->words);
    f::Fixture source;args=source.args();Inaccessible cache(source.buffer(s::cache).words.size()*4);source.tensors[s::cache].data=cache.data;source.buffer(s::block_table).id(2*source.columns,INT32_MAX);
    CHECK_EQ(s::emulate(&args),RAD_E_STATE);CHECK_EQ(source.flag[0],RAD_E_STATE);
}
TEST(ds41_index_stream_refuses_partial_heads_bad_geometry_output_aliases_and_scratch_without_writes){
    for(int out:{s::selected_out,s::candidates_out})for(int other=0;other<s::operand_count;++other)if(out!=other){f::Fixture c;auto args=c.args();const auto before=c.buffers;const auto flag=c.flag;c.tensors[size_t(out)].data=c.tensors[size_t(other)].data;
        s::Plan p;CHECK(s::validate(&args,&p)<0);CHECK(c.flag==flag);for(int i=0;i<s::operand_count;++i)if(before[size_t(i)])CHECK(c.buffer(i).words==before[size_t(i)]->words);}
    for(int fault=0;fault<10;++fault){f::Fixture c;auto args=c.args();s::Plan p;
        if(fault==0)c.integer("global_heads",c.heads*2);
        if(fault==1)c.integer("topk",1025);
        if(fault==2)c.integer("top_blocks",2049);
        if(fault==3)c.integer("candidate_block_size",0);
        if(fault==4)c.integer("ratio",3);
        if(fault==5)c.tensors[s::cache].stride[0]=INT64_MAX;
        if(fault==6)args.scratch_bytes=3;
        if(fault==7)args.scratch=c.tensors[s::cache].data;
        if(fault==8)args.scratch=reinterpret_cast<void*>(UINTPTR_MAX-3);
        if(fault==9)c.tensors[s::selected_out].dtype=RAD_F32;
        CHECK(s::validate(&args,&p)<0);CHECK_EQ(c.flag[0],911);
    }
}
RAD_TEST_MAIN()
