/* Bit-for-bit local head reorder; no floating-point oracle, device or communication. */
#include "rad_test.h"
#include "../libds41/ds41_reorder_heads_fixture.h"
#include <random>
#include <sys/mman.h>
namespace r=rad::ds41::reorder_heads;
namespace f=r::testing;
namespace {
void compare(f::Fixture& c) {
    const auto expected=c.expected();REQUIRE_EQ(c.run(),RAD_OK);
    for(int i=0;i<r::operand_count;++i)CHECK(c.buffers[size_t(i)].words==expected[size_t(i)].words);
}
}
TEST(ds41_reorder_heads_tp_1_2_4_8_all_scalar_widths_and_raw_bits) {
    for(bool flat:{false,true})for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16})for(int tp:{1,2,4,8})for(int rows:{1,3,17})for(int heads:{1,2,7})for(int dim:{1,7,128}) {
        f::Fixture c(dtype,tp,rows,heads,dim,flat);compare(c);
    }
}
TEST(ds41_reorder_heads_random_token_head_dimensions_match_independent_scatter) {
    std::mt19937 random(411017);
    for(int i=0;i<100;++i) {
        const uint32_t dtypes[]{RAD_F32,RAD_F16,RAD_BF16};
        f::Fixture c(dtypes[random()%3],int64_t(1)<<(random()%4),1+random()%19,1+random()%11,1+random()%257,bool(random()%2));compare(c);
    }
}
TEST(ds41_reorder_heads_rejects_every_pairwise_alias_and_preserves_all_bytes) {
    for(int a=0;a<r::operand_count;++a)for(int b=0;b<a;++b) {
        f::Fixture c;auto args=c.args();const auto before=c.buffers;c.tensors[size_t(a)].data=c.tensors[size_t(b)].data;
        r::Launch p;CHECK_EQ(r::validate(&args,&p),RAD_E_INVAL);
        CHECK_EQ(r::emulate(&args),RAD_E_INVAL);
        for(int i=0;i<r::operand_count;++i)CHECK(c.buffers[size_t(i)].words==before[size_t(i)].words);
    }
}
TEST(ds41_reorder_heads_refuses_malformed_geometry_dtypes_pitches_alignment_and_wrap) {
    for(int fault=0;fault<18;++fault) {
        f::Fixture c;auto args=c.args();const auto before=c.buffers;
        if(fault==0)c.integer("tp",0);
        if(fault==1)c.integer("tp",65537);
        if(fault==2)c.integer("heads",INT64_MAX);
        if(fault==3)c.integer("head_dim",0);
        if(fault==4)c.integer("M",c.rows-1);
        if(fault==5)c.text("dtype","i32");
        if(fault==6)c.tensors[r::query].rank=3;
        if(fault==7)++c.tensors[r::query].shape[0];
        if(fault==8)c.tensors[r::weights].dtype=RAD_F16;
        if(fault==9)c.tensors[r::query_out].dtype=RAD_BF16;
        if(fault==10)++c.tensors[r::query_out].shape[1];
        if(fault==11)++c.tensors[r::query].stride[0];
        if(fault==12)c.tensors[r::query].stride[3]=0;
        if(fault==13)c.tensors[r::weights_out].stride[1]=-1;
        if(fault==14)c.tensors[r::query_out].data=nullptr;
        if(fault==15)c.tensors[r::query].data=static_cast<uint8_t*>(c.tensors[r::query].data)+1;
        if(fault==16)c.tensors[r::query].data=reinterpret_cast<void*>(UINTPTR_MAX-3);
        if(fault==17){c.integer("M",INT_MAX);c.integer("tp",65536);c.integer("heads",INT_MAX);c.integer("head_dim",INT_MAX);}
        r::Launch p;CHECK(r::validate(&args,&p)<0);CHECK(r::emulate(&args)<0);
        for(int i=0;i<r::operand_count;++i)CHECK(c.buffers[size_t(i)].words==before[size_t(i)].words);
    }
}
TEST(ds41_reorder_heads_flattened_profiles_refuse_nondivisible_rows_and_mixed_input_ranks) {
    for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16})for(int tp:{2,4,8})for(int fault=0;fault<4;++fault) {
        f::Fixture c(dtype,tp,3,2,7,true);auto args=c.args();const auto before=c.buffers;
        if(fault==0)++c.tensors[r::query].shape[0];
        if(fault==1)c.tensors[r::weights].rank=3;
        if(fault==2)++c.tensors[r::weights].shape[0];
        if(fault==3)++c.tensors[r::query].stride[0];
        r::Launch p;CHECK(r::validate(&args,&p)<0);CHECK(r::emulate(&args)<0);
        for(int i=0;i<r::operand_count;++i)CHECK(c.buffers[size_t(i)].words==before[size_t(i)].words);
    }
}
TEST(ds41_reorder_heads_metadata_shape_hooks_use_max_M_and_ignore_caller_scratch) {
    f::Fixture c(RAD_BF16,8,3,7,128);auto args=c.args();r::Launch launch;REQUIRE_EQ(r::validate(&args,&launch),RAD_OK);CHECK(sizeof launch<=128);
    c.params[0]=RAD_RANGE("M",1,37);
    for(int i=0;i<r::operand_count;++i){RadOpdDesc d{};REQUIRE_EQ(r::shape(c.params.data(),c.params.size(),i,&d),RAD_OK);
        CHECK_EQ(d.dtype,i==r::weights||i==r::weights_out?uint32_t(RAD_F32):uint32_t(RAD_BF16));
        CHECK_EQ(d.shape[i<2?1:0],37);CHECK_EQ(d.shape[0],i<2?8:37);}
    c.params[0]=RAD_INT("M",c.rows+2);const auto expected=c.expected();
    void* inaccessible=mmap(nullptr,4096,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);REQUIRE(inaccessible!=MAP_FAILED);
    args.scratch=inaccessible;args.scratch_bytes=4096;CHECK_EQ(ds41_ref_reorder_heads(&args,nullptr),RAD_OK);munmap(inaccessible,4096);
    for(int i=0;i<r::operand_count;++i)CHECK(c.buffers[size_t(i)].words==expected[size_t(i)].words);
}
RAD_TEST_MAIN()
