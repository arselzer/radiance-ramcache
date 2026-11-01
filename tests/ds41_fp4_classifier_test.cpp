#include "rad_test.h"
#include "../arch/common/rad_ds41_fp4_classifier.h"
#include "../tools/ds41-host-experts/ds41_fp4_classifier_oracle.h"
#include <array>
#include <cfenv>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
namespace fp = rad_ds41_fp4_classifier;
namespace ref = ds41_fp4_oracle;
TEST(ds41_fp4_classifier_magnitudes_signed_zero_and_closed_refusal) {
    constexpr std::array<uint32_t,8> mag{0,0x3F000000,0x3F800000,0x3FC00000,0x40000000,0x40400000,0x40800000,0x40C00000};
    for (uint32_t sign : {0u,0x80000000u}) for (uint8_t c = 0; c < 8; ++c) {
        fp::Result out; REQUIRE(fp::classify_bits(mag[c] | sign,&out));
        CHECK_EQ(out.value_bits,mag[c] | sign);CHECK_EQ(out.code,uint8_t(c | (sign ? 8 : 0)));
    }
    for (uint32_t u : {0x40C00001u,0x7F7FFFFFu,0x7F800000u,0xFF800000u,0x7F800001u,0xFFC12345u}) {
        fp::Result out{0x12345678u,0xA5};CHECK(!fp::classify_bits(u,&out));
        CHECK_EQ(out.value_bits,uint32_t{0x12345678});CHECK_EQ(out.code,uint8_t{0xA5});
    }
    CHECK(!fp::classify_bits(0,nullptr));
    // A blanket threshold saturation would be wrong outside the proved domain:
    // at huge magnitudes all rounded distances tie, leaving the final EVEN code4.
    CHECK_EQ(ref::bits(ref::old_nearest(ref::value(0x7F7FFFFFu))),uint32_t{0x40800000});
    CHECK_EQ(ref::bits(ref::prototype(ref::value(0xFF7FFFFFu))),uint32_t{0xC0800000});
}
TEST(ds41_fp4_classifier_every_tie_and_adjacent_float_selects_old_even_code) {
    REQUIRE(std::fegetround()==FE_TONEAREST);
    const uint32_t cuts[]={0x3E800000,0x3F400000,0x3FA00000,0x3FE00000,0x40200000,0x40600000,0x40A00000};
    const uint8_t ties[]={0,2,2,4,4,6,6};
    for (unsigned i=0;i<7;++i) for (int delta : {-1,0,1}) for (uint32_t sign : {0u,0x80000000u}) {
        const uint32_t u=uint32_t(int64_t(cuts[i])+delta)|sign;fp::Result out;REQUIRE(fp::classify_bits(u,&out));
        CHECK_EQ(out.code,uint8_t((delta<0?i:delta>0?i+1:ties[i]) | (sign?8:0)));
        CHECK_EQ(out.value_bits,ref::bits(ref::old_nearest(ref::value(u))));
    }
}
TEST(ds41_fp4_classifier_bf16_images_and_float_boundary_intervals_are_literal_equal) {
    REQUIRE(std::fegetround()==FE_TONEAREST);
    const auto inputs=ref::corpus();uint64_t fast=0,fallback=0;
    for (uint32_t u : inputs) {
        fp::Result out;const bool exact=fp::classify_bits(u,&out);fast+=exact;fallback+=!exact;
        CHECK_EQ(ref::bits(ref::prototype(ref::value(u))),ref::bits(ref::old_nearest(ref::value(u))));
        if(exact) CHECK_EQ(out.value_bits,ref::bits(ref::old_nearest(ref::value(u))));
    }
    CHECK(fast>65536);CHECK(fallback>1000);CHECK_EQ(fast+fallback,uint64_t(inputs.size()));
    uint32_t n=0;for(uint32_t h=0;h<65536;++h)n+=fp::admitted(h<<16);CHECK_EQ(n,uint32_t{33154});
}
TEST(ds41_fp4_classifier_flush_and_daz_keep_tiny_signed_zero_and_boundary_bits) {
    REQUIRE(std::fegetround()==FE_TONEAREST);
#if defined(__x86_64__) || defined(__i386__)
    struct Restore { unsigned csr=_mm_getcsr();~Restore(){_mm_setcsr(csr);} } restore;
    const unsigned original=_mm_getcsr();
    for(unsigned mode : {0u,0x8000u,0x40u,0x8040u}) {
        _mm_setcsr((original & ~0x8040u)|mode);
#endif
        for(uint32_t a : {1u,0x7Fu,0x7FFFu,0x7FFFFFu,0x800000u,0x3E7FFFFFu,0x3E800000u,0x3E800001u})
            for(uint32_t sign : {0u,0x80000000u}) {
                volatile uint32_t source=a|sign;const float x=ref::value(source);
                CHECK_EQ(ref::bits(ref::prototype(x)),ref::bits(ref::old_nearest(x)));
            }
#if defined(__x86_64__) || defined(__i386__)
    }
#endif
}
TEST(ds41_fp4_classifier_unchanged_scale_product_and_bf16_store_keep_all_bits) {
    const uint32_t scale_bits[]={0u,0x00800000u,0x3B000000u,0x3F800000u,0x43E00000u,0x7F000000u,0x7F800000u};
    for(uint32_t s : scale_bits) for(uint32_t u : {0u,0x80000000u,1u,0x80000001u,0x3E800000u,0xBF400000u,0x40A00000u,0xC0C00000u,0x7FC12345u,0xFFC12345u}) {
        const float x=ref::value(u),scale=ref::value(s);
        volatile float old=ref::old_nearest(x)*scale;
        volatile float now=ref::prototype(x)*scale;
        CHECK_EQ(ref::bits(old),ref::bits(now));CHECK_EQ(ref::bf16(old),ref::bf16(now));
    }
}
RAD_TEST_MAIN()
