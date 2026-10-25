#include "rad_test.h"
#include "../libds41/ds41_latent_emit_fixture.h"
#include "../libds41/ds41_compress_fixture.h"
#include <memory>
#include <sys/mman.h>
namespace e=rad::ds41::latent_emit;
namespace f=e::testing;
namespace c=rad::ds41::compress;
namespace cf=c::testing;
namespace {
void compare(f::Fixture& x,int expected=RAD_OK) {
    const auto oracle=x.expected(expected);REQUIRE_EQ(x.run(),expected);CHECK_EQ(x.flag[0],expected);
    for(int i=0;i<e::operand_count;++i)CHECK(x.buffers[size_t(i)].words==oracle[size_t(i)].words);
    CHECK_EQ(x.flag[1],912);CHECK_EQ(x.flag[2],913);CHECK_EQ(x.flag[3],914);
}
struct Protected {
    void* data=nullptr;size_t bytes;
    explicit Protected(size_t count):bytes((count+4095)/4096*4096){data=mmap(nullptr,bytes,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);if(data==MAP_FAILED)throw std::runtime_error("mmap");}
    ~Protected(){munmap(data,bytes);}
};
void projected_store(cf::Fixture& pool,f::Fixture& target) {
    REQUIRE_EQ(pool.run(),RAD_OK);
    for(int64_t row=0;row<target.rows;++row){const bool active=pool.buffer(c::emit).geti(row)==1;target.buffers[e::emit].integer(row,active?1:0);
        if(active)for(int64_t d=0;d<target.dimension;++d)
            rad_store_f32(target.buffers[e::x].data(),target.dtype,target.buffers[e::x].offset(row*target.dimension+d),pool.buffer(c::pooled).getf(row*target.dimension+d));}
    compare(target);
}
}
TEST(ds41_latent_emit_all_dtypes_coordinate_widths_pitches_and_ratio_bits_match_independent_pages) {
    for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16})for(uint32_t pd:{RAD_I32,RAD_I64})for(uint32_t sd:{RAD_I32,RAD_I64})
        for(int ratio:{1,2})for(int d:{128,512})for(int pitch=0;pitch<3;++pitch){f::Fixture x(dtype,d,ratio,pitch,pd,sd);compare(x);}
}
TEST(ds41_latent_emit_c2_compressor_odd_chunk_resume_publishes_completed_original_rows_only) {
    for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16})for(int pitch=0;pitch<3;++pitch) {
        cf::Fixture cold(128,2,RAD_F32,pitch,{7},{0},{0});f::Fixture reference(dtype,128,2,pitch);projected_store(cold,reference);
        cf::Fixture first(128,2,RAD_F32,pitch,{3},{0},{0});f::Fixture prefix(dtype,128,2,pitch,RAD_I32,RAD_I32,3,0);projected_store(first,prefix);
        cf::Fixture last(128,2,RAD_F32,pitch,{4},{3},{0});
        for(int plane=0;plane<2;++plane)for(int d=0;d<=128;++d)last.state_set(0,plane,d,first.state_get(0,plane,d));
        f::Fixture suffix(dtype,128,2,pitch,RAD_I32,RAD_I32,4,3);suffix.buffers[e::cache].words=prefix.buffers[e::cache].words;
        projected_store(last,suffix);CHECK(suffix.buffers[e::cache].words==reference.buffers[e::cache].words);
        CHECK_EQ(last.state_get(0,0,128),7u);CHECK_EQ(last.buffer(c::emit).geti(3),0);
    }
}
TEST(ds41_latent_emit_all_zero_masks_never_touch_protected_payloads_coordinates_or_cache) {
    f::Fixture x;
    for(int64_t row=0;row<x.rows;++row)x.buffers[e::emit].integer(row,0);
    auto a=x.args();
    std::array<std::unique_ptr<Protected>,4> inaccessible;
    for(int i=0;i<4;++i){const int operand=i==3?e::cache:i;inaccessible[size_t(i)]=std::make_unique<Protected>(size_t(x.buffers[size_t(operand)].allocation_bytes));x.tensors[size_t(operand)].data=inaccessible[size_t(i)]->data;}
    REQUIRE_EQ(e::emulate(&a),RAD_OK);CHECK_EQ(x.flag[0],RAD_OK);for(int64_t row=0;row<x.rows;++row)CHECK_EQ(x.buffers[e::status].integer(row),0);
}
TEST(ds41_latent_emit_mixed_masks_do_not_read_inactive_protected_pages) {
    f::Fixture x(RAD_F32,128,2);auto a=x.args();const auto expected=x.expected();
    constexpr size_t page=4096;const size_t total=size_t(x.rows)*page;
    std::array<void*,3> mappings{};
    struct Cleanup{std::array<void*,3>& values;size_t bytes;~Cleanup(){for(auto* p:values)if(p&&p!=MAP_FAILED)munmap(p,bytes);}} cleanup{mappings,total};
    for(int operand=0;operand<3;++operand) {
        auto* memory=mmap(nullptr,total,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);REQUIRE(memory!=MAP_FAILED);mappings[size_t(operand)]=memory;
        const int width=rad_dtype_bits(x.tensors[size_t(operand)].dtype)/8;
        for(int64_t row=0;row<x.rows;++row) {
            auto* at=static_cast<uint8_t*>(memory)+size_t(row)*page;
            if(x.buffers[e::emit].integer(row)==0){REQUIRE_EQ(mprotect(at,page,PROT_NONE),0);continue;}
            if(operand==e::x)std::memcpy(at,static_cast<const uint8_t*>(x.buffers[e::x].data())+row*x.dimension*width,size_t(x.dimension*width));
            else{const auto value=int32_t(x.buffers[size_t(operand)].integer(row));std::memcpy(at,&value,4);}
        }
        x.tensors[size_t(operand)].data=memory;x.tensors[size_t(operand)].stride[0]=int64_t(page)/width;
    }
    REQUIRE_EQ(e::emulate(&a),RAD_OK);CHECK(x.buffers[e::cache].words==expected[e::cache].words);CHECK(x.buffers[e::status].words==expected[e::status].words);
}
TEST(ds41_latent_emit_validates_entire_mask_before_any_active_coordinate_read) {
    f::Fixture x;x.buffers[e::emit].integer(x.rows-1,2);auto a=x.args();Protected unavailable(65536);
    x.tensors[e::positions].data=unavailable.data;x.tensors[e::slot_mapping].data=unavailable.data;
    const auto before=x.buffers[e::cache].words;CHECK_EQ(e::emulate(&a),RAD_E_INVAL);CHECK(x.buffers[e::cache].words==before);
    for(int64_t row=0;row<x.rows;++row)CHECK_EQ(x.buffers[e::status].integer(row),RAD_E_INVAL);
}
TEST(ds41_latent_emit_invalid_late_coordinates_and_duplicate_destinations_seal_all_status_atomically) {
    for(int ratio:{1,2})for(uint32_t coordinate:{RAD_I32,RAD_I64})for(int fault=0;fault<6;++fault) {
        f::Fixture x(RAD_BF16,128,ratio,2,coordinate,coordinate);const int64_t row=ratio==2?5:6;
        if(fault==0)x.buffers[e::positions].integer(row,-1);
        if(fault==1)x.buffers[e::slot_mapping].integer(row,x.blocks*x.block);
        if(fault==2)x.buffers[e::slot_mapping].integer(row,0);
        if(fault==3){x.buffers[e::positions].integer(row,x.buffers[e::positions].integer(ratio==2?1:0));x.buffers[e::slot_mapping].integer(row,x.buffers[e::slot_mapping].integer(ratio==2?1:0));}
        if(fault==4)x.buffers[e::emit].integer(row,-1);
        if(fault==5)x.buffers[e::slot_mapping].integer(row,-1);
        compare(x,fault==3||fault==4?RAD_E_INVAL:RAD_E_SHAPE);
    }
}
TEST(ds41_latent_emit_metadata_aliases_and_bad_geometry_preserve_status_cache_and_inputs) {
    for(int destination:{e::cache,e::status})for(int source=0;source<e::operand_count;++source)if(source!=destination) {
        f::Fixture x;auto a=x.args();const auto before=x.buffers;const auto flag=x.flag;x.tensors[size_t(destination)].data=x.tensors[size_t(source)].data;
        e::Plan p;CHECK(e::validate(&a,&p)<0);CHECK(x.flag==flag);for(int i=0;i<e::operand_count;++i)CHECK(x.buffers[size_t(i)].words==before[size_t(i)].words);
    }
    for(int fault=0;fault<12;++fault) {
        f::Fixture x;auto a=x.args();const auto before=x.buffers;const auto flag=x.flag;
        if(fault==0)x.tensors[e::positions].dtype=RAD_F32;
        if(fault==1)x.tensors[e::slot_mapping].dtype=RAD_U32;
        if(fault==2)x.tensors[e::cache].dtype=RAD_I32;
        if(fault==3)++x.tensors[e::cache].stride[0];
        if(fault==4)x.tensors[e::status].stride[0]=0;
        if(fault==5)x.integer("block_size",8);
        if(fault==6)x.integer("ratio",3);
        if(fault==7)x.integer("n_blocks",2);
        if(fault==8)a.scratch_bytes=3;
        if(fault==9)a.scratch=x.tensors[e::cache].data;
        if(fault==10)a.scratch=reinterpret_cast<void*>(UINTPTR_MAX-3);
        if(fault==11)x.tensors[e::cache].data=reinterpret_cast<void*>(UINTPTR_MAX-3);
        e::Plan p;CHECK(e::validate(&a,&p)<0);CHECK(x.flag==flag);for(int i=0;i<e::operand_count;++i)CHECK(x.buffers[size_t(i)].words==before[size_t(i)].words);
    }
}
TEST(ds41_latent_emit_host_wrapper_ignores_caller_device_scratch_and_returns_payload_failure) {
    f::Fixture x;auto a=x.args();const auto oracle=x.expected();Protected scratch(4096);a.scratch=scratch.data;a.scratch_bytes=4096;
    CHECK_EQ(ds41_ref_latent_store_emitted(&a,nullptr),RAD_OK);
    for(int i=0;i<e::operand_count;++i)CHECK(x.buffers[size_t(i)].words==oracle[size_t(i)].words);
    CHECK_EQ(x.flag[0],911);
    x.buffers[e::emit].integer(6,2);const auto before=x.buffers[e::cache].words;CHECK_EQ(ds41_ref_latent_store_emitted(&a,nullptr),RAD_E_INVAL);CHECK(x.buffers[e::cache].words==before);CHECK_EQ(x.flag[0],911);
}
RAD_TEST_MAIN()
