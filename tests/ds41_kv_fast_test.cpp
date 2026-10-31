/* Reference-vs-optional-reader integer BYTE oracles. No cards/model and no float equality. */
#include "rad_test.h"
#include "../tools/ds41-host-experts/ds41_kv_fast_oracle.h"
namespace o = ds41_kv_fast_oracle;
namespace kc = rad_ds41_kv_codec;

TEST(ds41_kv_fast_all_scales_and_all_uint8_codes_match_reference_validity_and_bits) {
    for (unsigned at = 0; at < 2u * 256u * 256u; ++at) REQUIRE_EQ(o::normal_case(at), 0u);
}
TEST(ds41_kv_fast_every_literal_uint16_with_random_selectors_and_padding_matches_reference) {
    for (uint32_t k : {1u, 2u}) for (unsigned at = 0; at < 65536; ++at) REQUIRE_EQ(o::pair_case(k, at), 0u);
}
TEST(ds41_kv_fast_scalar_and_eight_value_reads_match_unaligned_mixed_rows_and_crossings) {
    for (unsigned at = 0; at < 128; ++at) {
        REQUIRE_EQ((o::row_case<kc::Kind::Fp4E4M3G16, 512>(at)), 0u);
        REQUIRE_EQ((o::row_case<kc::Kind::Fp4Pow2G32, 128>(at)), 0u);
    }
}
TEST(ds41_kv_fast_invalid_metadata_rejects_without_partial_output_like_reference) {
    REQUIRE_EQ(o::metadata_cases(), 0u);
}
RAD_TEST_MAIN()
