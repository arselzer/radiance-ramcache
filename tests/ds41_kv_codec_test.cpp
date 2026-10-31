/* Standalone codec proofs: compare uint16 bits, never float equality or NaN equivalence.
 * The finite-product reference uses independent ABI converters and ordinary f32 multiplication.
 * GPU NaN payloads are checked by the separate device oracle, not predicted from this host. */
#include "rad_test.h"
#include "rad_plugin.h"
#include "../arch/common/rad_ds41_kv_codec.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

namespace kc = rad_ds41_kv_codec;
namespace {
float ref_q(uint8_t c) {
    const float mag[] = { 0, .5f, 1, 1.5f, 2, 3, 4, 6 };
    return c & 8 ? -mag[c & 7] : mag[c & 7];
}
float ref_scale(kc::Kind k, uint8_t s) {
    if (k == kc::Kind::Fp4E4M3G16) return rad_fp8e4m3_to_f32(s);
    const uint32_t bits = (uint32_t)s << 23;
    float f; std::memcpy(&f, &bits, 4); return f;
}
uint16_t ref_bits(kc::Kind k, uint8_t s, uint8_t c) {
    volatile float product = ref_q(c) * ref_scale(k, s);
    return rad_f32_to_bf16(product);
}
void fill_codes(std::array<uint8_t, 32>& codes) {
    for (int i = 0; i < 32; ++i) codes[(size_t)i] = (uint8_t)(i & 15);
}
}

TEST(ds41_kv_codec_planar_layout_is_self_contained_and_has_no_row_padding) {
    CHECK_EQ((uint32_t)kc::Kind::Plain, uint32_t(0));
    CHECK_EQ((uint32_t)kc::Kind::Fp4E4M3G16, uint32_t(1));
    CHECK_EQ((uint32_t)kc::Kind::Fp4Pow2G32, uint32_t(2));
    CHECK_EQ(kc::row_bytes(1u, 512), int64_t(288));
    CHECK_EQ(kc::row_bytes(2u, 128), int64_t(68));
    CHECK_EQ(kc::row_used_bytes(2u, 128), kc::row_bytes(2u, 128));
    CHECK_EQ(kc::payload_offset(1u, 31), int64_t(248));
    CHECK_EQ(kc::scale_offset(1u, 512, 31), int64_t(287));
    CHECK_EQ(kc::scale_offset(2u, 128, 3), int64_t(67));
    for (uint32_t k : { 0u, 3u, 255u }) CHECK_EQ(kc::row_bytes(k, 512), int64_t(0));
    CHECK(!kc::valid_width(1u, 0)); CHECK(!kc::valid_width(1u, 17)); CHECK(!kc::valid_width(2u, 16));
    for (uint32_t k : { 1u, 2u }) {
        const int n = k == 1 ? 512 : 128;
        std::vector<uint8_t> rows((size_t)kc::row_bytes(k, n) * 2 + 1, 0);
        std::vector<uint16_t> out((size_t)n, 0xBEEF);
        // Deliberately unaligned byte base and the second 68-byte row.
        for (int r = 0; r < 2; ++r) {
            REQUIRE(kc::decode_row(k, rows.data() + 1 + r * kc::row_bytes(k, n), n, out.data()));
            for (uint16_t v : out) CHECK_EQ(v, uint16_t(0));
        }
    }
}

TEST(ds41_kv_codec_every_finite_scale_code_product_matches_bf16_bits_and_zero_signs) {
    for (kc::Kind k : { kc::Kind::Fp4E4M3G16, kc::Kind::Fp4Pow2G32 }) {
        const int top = k == kc::Kind::Fp4E4M3G16 ? 126 : 253;
        for (int s = 1; s <= top; ++s) for (int c = 0; c < 16; ++c) {
            uint16_t got = 0xBEEF;
            REQUIRE(kc::decode_normal_bits(k, (uint8_t)s, (uint8_t)c, &got));
            CHECK_EQ(got, ref_bits(k, (uint8_t)s, (uint8_t)c));
        }
    }
    uint16_t got = 0;
    REQUIRE(kc::decode_normal_bits(kc::Kind::Fp4E4M3G16, 1, 8, &got)); CHECK_EQ(got, uint16_t(0x8000));
    REQUIRE(kc::decode_normal_bits(kc::Kind::Fp4Pow2G32, 1, 1, &got)); CHECK_EQ(got, uint16_t(0x0040));
    REQUIRE(kc::decode_normal_bits(kc::Kind::Fp4Pow2G32, 1, 9, &got)); CHECK_EQ(got, uint16_t(0x8040));
    REQUIRE(kc::decode_normal_bits(kc::Kind::Fp4Pow2G32, 253, 6, &got)); CHECK_EQ(got, uint16_t(0x7F80));
    REQUIRE(kc::decode_normal_bits(kc::Kind::Fp4Pow2G32, 253, 14, &got)); CHECK_EQ(got, uint16_t(0xFF80));
    for (int c = 0; c < 16; ++c) {
        REQUIRE(kc::decode_normal_bits(kc::Kind::Fp4Pow2G32, 254, (uint8_t)c, &got));
        const uint16_t want = (c & 7) == 1 ? (uint16_t)((c & 8) << 12)
                                           : ref_bits(kc::Kind::Fp4Pow2G32, 1, (uint8_t)c);
        CHECK_EQ(got, want);
    }
    for (int c = 0; c < 16; ++c) {
        uint8_t code = 255;
        REQUIRE(kc::code_from_fp32_bits(kc::fp4_fp32_bits((uint8_t)c), &code));
        CHECK_EQ(code, (uint8_t)c);
    }
    uint8_t untouched = 77;
    CHECK(!kc::code_from_fp32_bits(0x7FC00000u, &untouched)); CHECK_EQ(untouched, uint8_t(77));
    uint32_t bits = 0;
    REQUIRE(kc::source_scale_fp32_bits(2u, 255, &bits)); CHECK_EQ(bits, uint32_t(0x7F800000));
    REQUIRE(kc::source_scale_fp32_bits(2u, 254, &bits)); CHECK_EQ(bits, uint32_t(0x7F000000));
    CHECK(!kc::decode_normal_bits(kc::Kind::Fp4Pow2G32, 255, 0, &got));
}

TEST(ds41_kv_codec_mixed_normal_pair_and_ftz_groups_decode_full_unaligned_rows) {
    std::array<uint8_t, 32> codes{}; fill_codes(codes);
    std::array<uint16_t, 32> actual{};
    for (uint32_t k : { 1u, 2u }) {
        const int n = k == 1 ? 512 : 128, G = kc::group_size(k);
        const int64_t rb = kc::row_bytes(k, n);
        std::vector<uint8_t> packed((size_t)(2 * rb + 2), 0xA5);
        std::vector<uint16_t> expected((size_t)(2 * n)), got((size_t)n);
        for (int r = 0; r < 2; ++r) for (int g = 0; g < n / G; ++g) {
            uint8_t source = (uint8_t)((g + r) % 125 + 1);
            const uint8_t* captured_codes = codes.data();
            if (g % 3 == 1) {
                source = 255; captured_codes = nullptr;
                for (int i = 0; i < G; ++i) actual[(size_t)i] = i & 1 ? (uint16_t)(0xFFC1 + r) : (uint16_t)(0x7FC1 + g);
            } else if (k == 2 && g % 3 == 2) {
                source = 1;
                for (int i = 0; i < G; ++i) actual[(size_t)i] = (codes[(size_t)i] & 7) == 1
                    ? (uint16_t)((codes[(size_t)i] & 8) << 12) : ref_bits(kc::Kind::Fp4Pow2G32, 1, codes[(size_t)i]);
            } else for (int i = 0; i < G; ++i) actual[(size_t)i] = ref_bits((kc::Kind)k, source, codes[(size_t)i]);
            uint8_t* row = packed.data() + 1 + r * rb;
            REQUIRE(kc::pack_captured_group(k, source, captured_codes, actual.data(),
                    row + kc::payload_offset(k, g), row + kc::scale_offset(k, n, g)) == kc::PackResult::Exact);
            for (int i = 0; i < G; ++i) expected[(size_t)(r * n + g * G + i)] = actual[(size_t)i];
        }
        for (int r = 0; r < 2; ++r) {
            const uint8_t* row = packed.data() + 1 + r * rb;
            REQUIRE(kc::decode_row(k, row, n, got.data()));
            for (int i = 0; i < n; ++i) {
                CHECK_EQ(got[(size_t)i], expected[(size_t)(r * n + i)]);
                CHECK_EQ(kc::decode_row_bits(k, row, n, i), expected[(size_t)(r * n + i)]);
            }
        }
        CHECK_EQ(packed.front(), uint8_t(0xA5)); CHECK_EQ(packed.back(), uint8_t(0xA5));
    }
}

TEST(ds41_kv_codec_original_capture_preserves_all_finite_products_and_records_half_ftz) {
    std::array<uint8_t, 32> codes{}; fill_codes(codes);
    std::array<uint16_t, 32> actual{};
    std::array<uint8_t, 16> packed{};
    for (kc::Kind k : { kc::Kind::Fp4E4M3G16, kc::Kind::Fp4Pow2G32 }) {
        const int G = kc::group_size(k), top = k == kc::Kind::Fp4E4M3G16 ? 126 : 253;
        for (int s = 1; s <= top; ++s) {
            for (int i = 0; i < G; ++i) actual[(size_t)i] = ref_bits(k, (uint8_t)s, codes[(size_t)i]);
            uint8_t stored = 255;
            REQUIRE(kc::pack_captured_group(k, (uint8_t)s, codes.data(), actual.data(), packed.data(), &stored) == kc::PackResult::Exact);
            CHECK_EQ(stored, (uint8_t)s);
            for (int i = 0; i < G; ++i) {
                uint16_t got = 0;
                REQUIRE(kc::decode_group_value(k, stored, packed.data(), i, &got));
                CHECK_EQ(got, actual[(size_t)i]);
            }
        }
    }
    for (int i = 0; i < 32; ++i) actual[(size_t)i] = (codes[(size_t)i] & 7) == 1
        ? (uint16_t)((codes[(size_t)i] & 8) << 12) : ref_bits(kc::Kind::Fp4Pow2G32, 1, codes[(size_t)i]);
    uint8_t stored = 255;
    REQUIRE(kc::pack_captured_group(2u, 1, codes.data(), actual.data(), packed.data(), &stored) == kc::PackResult::Exact);
    CHECK_EQ(stored, uint8_t(254));
    for (int i = 0; i < 32; ++i) {
        uint16_t got = 0; REQUIRE(kc::decode_group_value(2u, stored, packed.data(), i, &got));
        CHECK_EQ(got, actual[(size_t)i]);
    }
    // Raw ORIGINAL source254 must never be confused with the stored FTZ descriptor254.
    for (int i = 0; i < 32; ++i) actual[(size_t)i] = ref_bits(kc::Kind::Fp4Pow2G32, 254, codes[(size_t)i]);
    packed.fill(0xA5); stored = 77;
    CHECK(kc::pack_captured_group(2u, 254, codes.data(), actual.data(), packed.data(), &stored) == kc::PackResult::Raw);
    for (uint8_t v : packed) CHECK_EQ(v, uint8_t(0xA5)); CHECK_EQ(stored, uint8_t(77));
}

TEST(ds41_kv_codec_pair_escape_preserves_every_uint16_pattern_and_explicit_byte_order) {
    std::array<uint16_t, 32> actual{};
    std::array<uint8_t, 16> packed{};
    for (uint32_t k : { 1u, 2u }) for (unsigned a = 0; a < 65536; ++a) {
        const int G = kc::group_size(k);
        for (int i = 0; i < G; ++i) actual[(size_t)i] = (uint16_t)((i & 1) ? (a ^ 0xA55A) : a);
        uint8_t stored = 255;
        REQUIRE(kc::pack_captured_group(k, 255, nullptr, actual.data(), packed.data(), &stored) == kc::PackResult::Exact);
        CHECK_EQ(stored, uint8_t(0));
        for (int i = 0; i < G; ++i) {
            uint16_t got = 0;
            REQUIRE(kc::decode_group_value(k, stored, packed.data(), i, &got));
            CHECK_EQ(got, actual[(size_t)i]); // includes signaling NaNs and signed zeros as RAW BITS
        }
    }
    for (int i = 0; i < 32; ++i) actual[(size_t)i] = i & 1 ? 0xFFC3 : 0x7FC1;
    uint8_t stored = 255;
    REQUIRE(kc::pack_captured_group(2u, 255, nullptr, actual.data(), packed.data(), &stored) == kc::PackResult::Exact);
    CHECK_EQ(packed[0], uint8_t(0xC1)); CHECK_EQ(packed[1], uint8_t(0x7F));
    CHECK_EQ(packed[2], uint8_t(0xC3)); CHECK_EQ(packed[3], uint8_t(0xFF));
    for (int i = 4; i < 8; ++i) CHECK_EQ(packed[(size_t)i], uint8_t(0xAA));
    for (int i = 8; i < 16; ++i) CHECK_EQ(packed[(size_t)i], uint8_t(0));
}

TEST(ds41_kv_codec_unrepresentable_capture_returns_raw_without_touching_outputs) {
    std::array<uint8_t, 32> codes{};
    std::array<uint16_t, 32> actual{};
    std::array<uint8_t, 16> packed{};
    for (uint32_t k : { 1u, 2u }) {
        actual.fill(0); actual[1] = 0x8000; actual[2] = 0x7FC1; // three distinct uint16 patterns
        packed.fill(0xA5); uint8_t stored = 77;
        CHECK(kc::pack_captured_group(k, 1, codes.data(), actual.data(), packed.data(), &stored) == kc::PackResult::Raw);
        for (uint8_t v : packed) CHECK_EQ(v, uint8_t(0xA5)); CHECK_EQ(stored, uint8_t(77));
    }
    std::vector<uint8_t> row(68, 0); row[67] = 255; // invalid LAST descriptor: no partial decode
    std::vector<uint16_t> out(128, 0xBEEF);
    CHECK(!kc::decode_row(2u, row.data(), 128, out.data()));
    for (uint16_t v : out) CHECK_EQ(v, uint16_t(0xBEEF));
    CHECK(kc::pack_captured_group(0u, 1, codes.data(), actual.data(), packed.data(), &row[0]) == kc::PackResult::Raw);
}

RAD_TEST_MAIN()
