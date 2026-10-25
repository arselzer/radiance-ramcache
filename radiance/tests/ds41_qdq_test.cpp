#include "rad_test.h"
#include "../libds41/ds41_qdq_fixture.h"
#include <sys/mman.h>
#include <unistd.h>
namespace q=rad::ds41::qdq;namespace m=rad::ds41::qdq_math;namespace f=q::testing;
namespace {
void check(f::Fixture fixture,int expected=RAD_OK){
    const auto before=fixture.buffers;const auto flag=fixture.flag;std::array<f::Buffer,q::operand_count> reference;
    REQUIRE_EQ(fixture.reference(&reference),expected);REQUIRE_EQ(fixture.run(),expected);CHECK_EQ(fixture.flag[0],expected);
    for(size_t i=1;i<flag.size();++i)CHECK_EQ(fixture.flag[i],flag[i]);
    for(int i=0;i<q::operand_count;++i)CHECK(fixture.buffers[size_t(i)].words==(expected==RAD_OK&&i==q::output?reference[size_t(i)].words:before[size_t(i)].words));
}
}
TEST(ds41_qdq_all_profiles_pitches_masks_and_primary_dimensions_match_independent_row_reference){
    for(auto profile:{m::Profile::window_fp8,m::Profile::compressed_fp4,m::Profile::index_fp4})
        for(int width:{32,128,512,1024})for(int pitch:{0,1,2})for(bool masked:{false,true})check(f::Fixture(profile,width,pitch,masked));
    check(f::Fixture(m::Profile::compressed_fp4,16));
}
TEST(ds41_qdq_inactive_payloads_are_unread_and_inactive_outputs_are_positive_zero){
    for(auto profile:{m::Profile::window_fp8,m::Profile::compressed_fp4,m::Profile::index_fp4}){
        f::Fixture fixture(profile);for(int64_t row=0;row<fixture.rows;++row)if(!fixture.buffers[q::emit].id(row))for(int64_t d=0;d<fixture.dim;++d)fixture.buffers[q::input].raw(row*fixture.dim+d,0x7FC1);
        check(fixture);
        const auto page=size_t(sysconf(_SC_PAGESIZE)),bytes=size_t(fixture.rows*fixture.dim*2);const auto mapped=(bytes+page-1)/page*page;
        void* inaccessible=mmap(nullptr,mapped,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);REQUIRE(inaccessible!=MAP_FAILED);
        for(int64_t row=0;row<fixture.rows;++row)fixture.buffers[q::emit].id(row,0);
        auto a=fixture.args();fixture.tensors[q::input].data=inaccessible;CHECK_EQ(q::emulate(&a),RAD_OK);
        for(int64_t i=0;i<fixture.rows*fixture.dim;++i)CHECK_EQ(fixture.buffers[q::output].raw(i),0);
        fixture.buffers[q::emit].id(0,1);fixture.buffers[q::emit].id(fixture.rows-1,2);const auto output=fixture.buffers[q::output].words;
        CHECK_EQ(q::emulate(&a),RAD_E_INVAL);CHECK(fixture.buffers[q::output].words==output);
        CHECK_EQ(munmap(inaccessible,mapped),0);
    }
}
TEST(ds41_qdq_late_mask_nonfinite_and_overflow_refusals_are_whole_batch_atomic){
    for(auto profile:{m::Profile::window_fp8,m::Profile::compressed_fp4,m::Profile::index_fp4}){
        for(uint16_t bad:{uint16_t(0x7F80),uint16_t(0xFF80),uint16_t(0x7FC1),uint16_t(0x7F7F)}){
            f::Fixture fixture(profile);fixture.buffers[q::input].raw(5*fixture.dim+fixture.dim-1,bad);check(fixture,RAD_E_INVAL);
        }
        f::Fixture fixture(profile);fixture.buffers[q::emit].id(fixture.rows-1,2);check(fixture,RAD_E_INVAL);
    }
}
TEST(ds41_qdq_metadata_and_full_plan_reservation_refuse_before_output_or_flag_writes){
    for(int fault=0;fault<13;++fault){
        f::Fixture fixture;auto a=fixture.args();const auto before=fixture.buffers;const auto flag=fixture.flag;
        if(fault==0)fixture.tensors[q::input].dtype=RAD_F32;
        if(fault==1)fixture.tensors[q::output].dtype=RAD_F16;
        if(fault==2)fixture.tensors[q::emit].dtype=RAD_I64;
        if(fault==3)fixture.tensors[q::output].stride[0]=0;
        if(fault==4)fixture.integer("n",15);
        if(fault==5)fixture.integer("masked",2);
        if(fault==6)fixture.tensors[q::output].data=fixture.tensors[q::input].data;
        if(fault==7)a.scratch=fixture.tensors[q::output].data;
        if(fault==8)a.scratch_bytes=3;
        if(fault==9)fixture.tensors[q::input].data=reinterpret_cast<void*>(UINTPTR_MAX-1);
        if(fault==10)a.n_t=2;
        if(fault==11)fixture.integer("M",q::max_rows+1);
        if(fault==12)fixture.tensors[q::input].shape[0]=fixture.rows+2;
        q::Plan p;CHECK(q::validate(&a,&p)<0);CHECK(fixture.flag==flag);
        for(int i=0;i<q::operand_count;++i)CHECK(fixture.buffers[size_t(i)].words==before[size_t(i)].words);
    }
    f::Fixture fixture;auto a=fixture.args();q::Plan p;REQUIRE_EQ(q::validate(&a,&p),RAD_OK);
    std::vector<uint64_t> scratch(size_t((rad::ds41::gpu_scratch_bytes<q::Plan>()+7)/8));a.scratch=scratch.data();a.scratch_bytes=rad::ds41::gpu_scratch_bytes<q::Plan>();
    REQUIRE_EQ(rad::ds41::validate_plan_storage(&a,p),RAD_OK);p.t[q::input].data=static_cast<uint8_t*>(a.scratch)+rad::ds41::device_plan_offset;
    CHECK_EQ(rad::ds41::validate_plan_storage(&a,p),RAD_E_INVAL);
}
TEST(ds41_qdq_host_reference_uses_a_private_flag_and_explicit_profile_scope){
    f::Fixture fixture;auto a=fixture.args();const auto flag=fixture.flag;
    a.scratch=reinterpret_cast<void*>(uintptr_t(1));a.scratch_bytes=INT64_MIN;
    REQUIRE_EQ(ds41_ref_qdq_candidate(&a,nullptr),RAD_OK);CHECK(fixture.flag==flag);
    CHECK_EQ(ds41_ref_qdq_candidate(nullptr,nullptr),RAD_E_INVAL);
}
RAD_TEST_MAIN()
