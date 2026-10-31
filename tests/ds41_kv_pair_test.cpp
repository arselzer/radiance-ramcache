#include "rad_test.h"
#include "../tools/ds41-host-experts/ds41_kv_pair_oracle.h"
namespace o=ds41_kv_pair_oracle;
namespace kc=rad_ds41_kv_codec;
TEST(ds41_kv_pair_all_scale_and_nibble_pairs_match_reference_without_partial_refusal_writes) {
    unsigned even=0,rejected=0;
    for(unsigned at=0;at<65536;++at) {
        auto a=o::normal<kc::Kind::Fp4E4M3G16>(at),b=o::normal<kc::Kind::Fp4Pow2G32>(at);
        REQUIRE_EQ(a.bad+b.bad,0u);even+=a.even+b.even;rejected+=a.rejected+b.rejected;
    }
    CHECK_EQ(even,97792u);CHECK_EQ(rejected,33280u);
}
TEST(ds41_kv_pair_every_uint16_literal_payload_and_odd_selector_boundary_preserves_raw_bits) {
    unsigned even=0,odd=0;
    for(unsigned at=0;at<65536;++at) {
        auto a=o::literal<kc::Kind::Fp4E4M3G16>(at),b=o::literal<kc::Kind::Fp4Pow2G32>(at);
        REQUIRE_EQ(a.bad+b.bad,0u);even+=a.even+b.even;odd+=a.odd+b.odd;
    }
    CHECK_EQ(even,131072u);CHECK_EQ(odd,131072u);
}
TEST(ds41_kv_pair_unaligned_zero_mixed_ftz_rows_crossings_and_last_columns_match_reference) {
    unsigned even=0,odd=0,rejected=0;
    for(unsigned at=0;at<128;++at) {
        auto a=o::rows<kc::Kind::Fp4E4M3G16,512>(at),b=o::rows<kc::Kind::Fp4Pow2G32,128>(at);
        REQUIRE_EQ(a.bad+b.bad,0u);even+=a.even+b.even;odd+=a.odd+b.odd;rejected+=a.rejected+b.rejected;
    }
    CHECK(even>0);CHECK(odd>0);CHECK(rejected>0);
}
TEST(ds41_kv_pair_invalid_row_metadata_geometry_and_null_output_leave_output_untouched) {
    auto a=o::invalid<kc::Kind::Fp4E4M3G16,512>(),b=o::invalid<kc::Kind::Fp4Pow2G32,128>();
    CHECK_EQ(a.bad+b.bad,0u);CHECK_EQ(a.rejected+b.rejected,16u);
}
TEST(ds41_kv_pair_mode_is_strict_default_off_and_native_m1_h16_only) {
    namespace p=rad_ds41_kv_pair;
    CHECK_EQ(p::parse_mode(nullptr),0);CHECK_EQ(p::parse_mode("0"),0);CHECK_EQ(p::parse_mode("1"),1);
    for(const char* s:{"","yes","2","01","1 ","-1"})CHECK_EQ(p::parse_mode(s),-1);
    CHECK(p::dependencies(0,false,false,true));CHECK(p::dependencies(1,true,true,false));
    CHECK(!p::dependencies(-1,true,true,false));CHECK(!p::dependencies(1,false,true,false));
    CHECK(!p::dependencies(1,true,false,false));CHECK(!p::dependencies(1,true,true,true));
    CHECK(p::normal_candidate(1,1,true,1,16));
    for(int M:{0,2,4,16,49,2048})CHECK(!p::normal_candidate(1,1,true,M,16));
    for(int H:{0,1,2,4,8,32,64})CHECK(!p::normal_candidate(1,1,true,1,H));
    CHECK(!p::normal_candidate(0,1,true,1,16));CHECK(!p::normal_candidate(1,0,true,1,16));
    CHECK(!p::normal_candidate(1,2,true,1,16));CHECK(!p::normal_candidate(1,1,false,1,16));
}
RAD_TEST_MAIN()
