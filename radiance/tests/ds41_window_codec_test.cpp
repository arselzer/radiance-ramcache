#include "rad_test.h"
#include "rad_plugin.h"
#include "../arch/common/rad_ds41_window_codec.h"
#include <array>
#include <cstring>
#include <vector>
namespace wc = rad_ds41_window_codec;
namespace {
uint16_t ref(uint8_t s, uint8_t c) {
    const uint32_t u = uint32_t(s) << 23;
    float scale; std::memcpy(&scale, &u, 4);
    volatile float product = rad_fp8e4m3_to_f32(c) * scale;
    return rad_f32_to_bf16(product);
}
}
TEST(ds41_window_codec_native_layout_zero_pages_and_unaligned_rows) {
    CHECK_EQ(uint32_t(wc::Kind::Fp8Pow2G32), uint32_t{3});
    CHECK_EQ(wc::row_bytes(512), int64_t{528});
    CHECK_EQ(wc::row_bytes(32), int64_t{33});
    for (int64_t n : {int64_t{0}, int64_t{-32}, int64_t{31}, INT64_MAX - 31}) CHECK_EQ(wc::row_bytes(n), int64_t{0});
    std::array<uint8_t, 1057> storage{};
    std::array<uint16_t, 512> out{};
    for (int row = 0; row < 2; ++row) {
        out.fill(0xFFFF);
        REQUIRE(wc::decode_row(storage.data() + 1 + row * 528, 512, out.data()));
        for (uint16_t v : out) CHECK_EQ(v, uint16_t{0});
    }
    CHECK_EQ(wc::payload_offset(15), int64_t{480});
    CHECK_EQ(wc::scale_offset(512, 15), int64_t{527});
}
TEST(ds41_window_codec_every_native_finite_product_is_byte_exact) {
    for (int s = 105; s <= 247; ++s) for (int c = 0; c < 256; ++c) {
        if ((c & 127) == 127) continue; // GPU NaN payload proof is separate
        uint16_t out = 0xDEAD;
        REQUIRE(wc::decode_normal_bits(uint8_t(s), uint8_t(c), &out));
        CHECK_EQ(out, ref(uint8_t(s), uint8_t(c)));
        if ((c & 127) == 0) CHECK_EQ(out, uint16_t(c & 128 ? 0x8000 : 0));
    }
    uint16_t out = 0xDEAD;
    for (uint8_t s : {uint8_t{0}, uint8_t{1}, uint8_t{104}, uint8_t{248}, uint8_t{254}, uint8_t{255}}) {
        CHECK(!wc::decode_normal_bits(s, 1, &out)); CHECK_EQ(out, uint16_t{0xDEAD});
    }
}
TEST(ds41_window_codec_original_bf16_amax_only_produces_native_scale_domain) {
    int low = 256, high = 0;
    for (uint32_t bits = 0; bits < 65536; ++bits) {
        float a = std::fabs(rad_bf16_to_f32(uint16_t(bits)));
        if (std::isnan(a)) a = 0;
        a = std::max(a, 1e-4f);
        volatile float ratio = a * (1.0f / 448.0f);
        const float value = ratio; uint32_t u = 0; std::memcpy(&u, &value, 4);
        const int scale = int((u >> 23) & 255) + ((u & 0x7FFFFF) ? 1 : 0);
        if (std::isinf(a)) CHECK_EQ(scale, 255);
        else { CHECK(scale >= 105 && scale <= 247); low = std::min(low, scale); high = std::max(high, scale); }
    }
    CHECK_EQ(low, 105); CHECK_EQ(high, 247);
}
TEST(ds41_window_codec_captured_codes_survive_and_every_actual_bit_is_checked) {
    std::array<uint8_t, 32> codes{}, payload{};
    std::array<uint16_t, 32> actual{};
    for (int s : {105, 127, 247}) for (int start = 0; start < 256; start += 32) {
        for (int i = 0; i < 32; ++i) {
            codes[i] = uint8_t(start + i);
            REQUIRE(wc::decode_normal_bits(uint8_t(s), codes[i], &actual[i]));
        }
        uint8_t scale = 0xA5;
        REQUIRE(wc::pack_captured_group(uint8_t(s), codes.data(), actual.data(), payload.data(), &scale) == wc::PackResult::Exact);
        CHECK_EQ(scale, uint8_t(s)); CHECK(payload == codes);
        for (int i = 0; i < 32; ++i) {
            uint16_t got = 0; REQUIRE(wc::decode_group_bits(payload.data(), scale, i, &got)); CHECK_EQ(got, actual[i]);
        }
    }
    // This source case verifies a proposed encoding, NOT CPU/GPU NaN equivalence.
    for (int i = 0; i < 32; ++i) { codes[i] = uint8_t(i); actual[i] = uint16_t(0x3F00 + i); }
    payload.fill(0xA5); uint8_t scale = 0xA5;
    CHECK(wc::pack_captured_group(127, codes.data(), actual.data(), payload.data(), &scale) == wc::PackResult::Raw);
    for (uint8_t v : payload) CHECK_EQ(v, uint8_t{0xA5}); CHECK_EQ(scale, uint8_t{0xA5});
}
TEST(ds41_window_codec_inline_pair_keeps_payloads_signedzeros_and_selector_tail) {
    std::array<uint8_t, 32> codes{}, payload{};
    std::array<uint16_t, 32> actual{};
    for (auto pair : {std::array<uint16_t,2>{0,0x8000}, {0x7FC1,0xFFDE}, {0x7F80,0xFF80}}) {
        for (int i = 0; i < 32; ++i) actual[i] = pair[size_t((i / 3) & 1)];
        actual[31] = pair[1]; uint8_t scale = 0xA5;
        REQUIRE(wc::pack_captured_group(255, codes.data(), actual.data(), payload.data(), &scale) == wc::PackResult::Exact);
        CHECK_EQ(scale, uint8_t{0});
        for (int i = 8; i < 32; ++i) CHECK_EQ(payload[i], uint8_t{0});
        for (int i = 0; i < 32; ++i) { uint16_t got = 0;
            REQUIRE(wc::decode_group_bits(payload.data(), scale, i, &got)); CHECK_EQ(got, actual[i]); }
    }
    actual[0] = 0; actual[1] = 0x8000; actual[2] = 1;
    payload.fill(0xA5); uint8_t scale = 0xA5;
    CHECK(wc::pack_captured_group(255, codes.data(), actual.data(), payload.data(), &scale) == wc::PackResult::Raw);
    for (uint8_t v : payload) CHECK_EQ(v, uint8_t{0xA5}); CHECK_EQ(scale, uint8_t{0xA5});
}
TEST(ds41_window_codec_invalid_rows_fail_before_writing_output) {
    std::array<uint8_t, 528> row{}; std::array<uint16_t, 512> actual{};
    actual.fill(0xA55A); row[527] = 254;
    CHECK(!wc::decode_row(row.data(), 512, actual.data()));
    for (uint16_t v : actual) CHECK_EQ(v, uint16_t{0xA55A});
    CHECK(!wc::decode_row(nullptr, 512, actual.data()));
    CHECK(!wc::decode_row(row.data(), 512, nullptr));
    CHECK(!wc::decode_row(row.data(), 511, actual.data()));
}
RAD_TEST_MAIN()
