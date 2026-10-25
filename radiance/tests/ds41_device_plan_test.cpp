/* Host-only transport geometry/chunk proof. Actual chunk execution is checked by
 * the separately approved manual AQL checker, never inferred from this test. */
#include "rad_test.h"
#include "../libds41/ds41_device_plan.h"
#include "../libds41/ds41_math.h"
#include "../libds41/ds41_compress.h"
#include "../libds41/ds41_ffn.h"
#include "../libds41/ds41_index_paged.h"
#include "../libds41/ds41_transform.h"
#include "../libds41/ds41_attention_paged.h"
#include "../libds41/ds41_route_jobs.h"
#include "../libds41/ds41_index_stream.h"
#include "../libds41/ds41_embedding.h"
#include "../libds41/ds41_compress_finish.h"
#include "../libds41/ds41_latent_emit.h"
#include "../libds41/ds41_engram_gate.h"
#include "../libds41/ds41_qdq.h"
#include <array>
#include <vector>
namespace d=rad::ds41;
namespace {
struct Transport {RadTensor t[2]{};int64_t spans[2]{};std::array<uint8_t,765> payload{};};
template<class Plan>void geometry() {
    CHECK_EQ(d::device_plan_offset%alignof(Plan),0u);
    CHECK(d::gpu_scratch_bytes<Plan>()>=int64_t(d::device_plan_offset+sizeof(Plan)));
    CHECK_EQ((d::gpu_scratch_bytes<Plan>()-d::device_plan_offset)%d::device_plan_chunk_bytes,0u);
    CHECK(d::gpu_scratch_bytes<Plan>()<8192);
}
}
TEST(ds41_device_plan_chunks_preserve_every_byte_and_leave_rounded_tail_untouched) {
    Transport p;for(size_t i=0;i<p.payload.size();++i)p.payload[i]=uint8_t(i*37+13);
    std::vector<uint8_t> actual(size_t(d::gpu_scratch_bytes<Transport>())+16,0xCD);
    const auto before=actual;size_t copied=0;
    for(size_t part=0;part<(sizeof(p)+d::device_plan_chunk_bytes-1)/d::device_plan_chunk_bytes;++part) {
        d::DevicePlanChunk chunk;const auto count=d::make_plan_chunk(p,part,&chunk);
        REQUIRE(count>0&&count<=d::device_plan_chunk_bytes);CHECK_EQ(count,std::min<size_t>(d::device_plan_chunk_bytes,sizeof(p)-copied));
        std::memcpy(actual.data()+d::device_plan_offset+part*d::device_plan_chunk_bytes,chunk.bytes,count);copied+=count;
        for(size_t i=count;i<d::device_plan_chunk_bytes;++i)CHECK_EQ(chunk.bytes[i],0u);
    }
    CHECK_EQ(copied,sizeof(p));CHECK_EQ(std::memcmp(actual.data()+d::device_plan_offset,&p,sizeof(p)),0);
    for(size_t i=0;i<d::device_plan_offset;++i)CHECK_EQ(actual[i],before[i]);
    for(size_t i=d::device_plan_offset+sizeof(p);i<actual.size();++i)CHECK_EQ(actual[i],before[i]);
    d::DevicePlanChunk untouched{};untouched.bytes[0]=91;
    CHECK_EQ(d::make_plan_chunk(p,(sizeof(p)+d::device_plan_chunk_bytes-1)/d::device_plan_chunk_bytes,&untouched),0u);CHECK_EQ(untouched.bytes[0],91u);
    CHECK_EQ(d::make_plan_chunk(p,0,nullptr),0u);
}
TEST(ds41_device_plan_storage_rejects_aliases_only_inside_the_staged_descriptor_before_writes) {
    alignas(16) std::array<uint8_t,4096> storage{};std::array<uint8_t,32> operand{};Transport p;
    p.t[0].data=operand.data();p.spans[0]=operand.size();
    RadArgs a{};a.scratch=storage.data();a.scratch_bytes=d::gpu_scratch_bytes<Transport>();
    REQUIRE_EQ(d::validate_plan_storage(&a,p),RAD_OK);const auto before=storage;
    a.scratch_bytes=4;CHECK(d::validate_plan_storage(&a,p)<0);a.scratch_bytes=d::gpu_scratch_bytes<Transport>();
    p.t[1].data=storage.data()+d::device_plan_offset+64;p.spans[1]=4;
    CHECK(d::validate_plan_storage(&a,p)<0);p.t[1]={};p.spans[1]=0;
    a.scratch=storage.data()+4;CHECK_EQ(d::validate_plan_storage(&a,p),RAD_E_ALIGN);
    a.scratch=reinterpret_cast<void*>(UINTPTR_MAX&~uintptr_t(15));CHECK_EQ(d::validate_plan_storage(&a,p),RAD_E_SHAPE);
    a.scratch=nullptr;CHECK(d::validate_plan_storage(&a,p)<0);CHECK(d::validate_plan_storage(nullptr,p)<0);
    CHECK(storage==before);
}
TEST(ds41_device_plan_reservations_cover_every_current_large_operator_descriptor) {
    geometry<d::math::Plan>();geometry<d::compress::Plan>();geometry<d::ffn::Plan>();geometry<d::index::Plan>();
    geometry<d::paged_index::Plan>();geometry<d::transform::Plan>();geometry<d::attention_paged::Plan>();geometry<d::route_jobs::Plan>();
    geometry<d::index_stream::Plan>();geometry<d::embedding::Plan>();geometry<d::compress_finish::Plan>();
    geometry<d::latent_emit::Plan>();geometry<d::engram_gate::Plan>();geometry<d::qdq::Plan>();
    CHECK_EQ(d::math::scratch_bytes,4);CHECK_EQ(d::compress::scratch_bytes,4);CHECK_EQ(d::transform::scratch_bytes,4);
    CHECK_EQ(ds41_math_scratch(nullptr),d::gpu_scratch_bytes<d::math::Plan>());
    CHECK_EQ(ds41_compress_scratch(nullptr),d::gpu_scratch_bytes<d::compress::Plan>());
    CHECK_EQ(ds41_index_paged_scratch(nullptr),d::gpu_scratch_bytes<d::paged_index::Plan>());
}
RAD_TEST_MAIN()
