#include "rad_test.h"
#include "place/ds41_packed_inverse.h"
#include "../tools/ds41-host-experts/ds41_host_experts.h"
#include "../libquant/lq_ggml_planes.h"
#include <algorithm>
#include <cstring>
#include <vector>
namespace inv = rad::ds41_packed_inverse;
namespace {
struct Fixture {
    int N, K, blocks, rounds;
    std::vector<uint8_t> codes, scales, packed, original;
    Fixture(int n, int k) : N(n), K(k), blocks(K / 32), rounds((blocks + 31) / 32),
        codes(size_t(N) * size_t(K / 2)), scales(size_t(N) * size_t(blocks)) {
        for (size_t i = 0; i < codes.size(); ++i) codes[i] = uint8_t(i * 131 + i / 17);
        for (size_t i = 0; i < scales.size(); ++i) scales[i] = uint8_t(120 + i % 4);
    }
    void build(int cap = -1) {
        ds41hx::PackInfo plan;
        REQUIRE(ds41hx::pack_plan(scales.data(), N, K, &plan));
        if (cap < 0) cap = N * rounds;
        packed.assign(ds41hx::packed_bytes(N, K, cap), 0xA5);
        REQUIRE_EQ(ds41hx::pack_planes(codes.data(), scales.data(), N, K, cap, plan,
                                       packed.data(), packed.size()), ds41hx::kPackOk);
        ds41hx::Mx view;
        REQUIRE(ds41hx::packed_view(packed.data(), packed.size(), N, K, &view));
        REQUIRE(ds41hx::packed_scales_equal(view, scales.data()));
        const auto pack = lq::ggml::pack_of(lq::ggml::GGML_TYPE_MXFP4);
        REQUIRE(pack);
        original.assign(inv::block_bytes(N, K), 0);
        for (int n = 0; n < N; ++n) {
            const uint8_t* p[2] = {codes.data() + size_t(n) * size_t(K / 2), scales.data() + size_t(n) * size_t(blocks)};
            pack(p, blocks, original.data() + size_t(n) * size_t(blocks) * 17);
        }
    }
    void check(bool unaligned = false) {
        const auto before = packed;
        const int guard = unaligned ? 1 : 32;
        std::vector<uint8_t> source(packed.size() + 2 * size_t(guard), 0x6A);
        std::memcpy(source.data() + guard, packed.data(), packed.size());
        const auto source_before = source;
        std::vector<uint8_t> dst(original.size() + 2 * size_t(guard) + 19, 0xA5);
        inv::Stats stats;
        REQUIRE_EQ(inv::to_block(source.data() + guard, packed.size(), N, K,
                                 dst.data() + guard, original.size() + 19, &stats), RAD_OK);
        CHECK_EQ(std::memcmp(dst.data() + guard, original.data(), original.size()), 0);
        for (int i = 0; i < guard; ++i) CHECK_EQ(dst[size_t(i)], uint8_t{0xA5});
        for (size_t i = size_t(guard) + original.size(); i < dst.size(); ++i) CHECK_EQ(dst[i], uint8_t{0xA5});
        CHECK(source == source_before); CHECK(packed == before);
        CHECK_EQ(stats.blocks, uint64_t(N) * uint64_t(blocks));
        CHECK_EQ(stats.rounds, uint64_t(N) * uint64_t(rounds));
        CHECK_EQ(stats.codes_bytes, uint64_t(codes.size()));
        CHECK_EQ(stats.destination_bytes, uint64_t(original.size()));
        const auto* type = lq::ggml::plane_kind(lq::ggml::GGML_TYPE_MXFP4);
        REQUIRE(type);
        std::vector<uint8_t> back_codes(codes.size()), back_scales(scales.size());
        for (int n = 0; n < N; ++n) {
            uint8_t* p[2] = {back_codes.data() + size_t(n) * size_t(K / 2), back_scales.data() + size_t(n) * size_t(blocks)};
            type->unpack(dst.data() + guard + size_t(n) * size_t(blocks) * 17, blocks, p);
        }
        CHECK(back_codes == codes); CHECK(back_scales == scales);
    }
};
void put_index(Fixture& f, int i, uint32_t value) {
    ds41hx::Mx v; REQUIRE(ds41hx::packed_view(f.packed.data(), f.packed.size(), f.N, f.K, &v));
    uint8_t* p = f.packed.data() + (reinterpret_cast<const uint8_t*>(v.exc_idx) - f.packed.data());
    std::memcpy(p + size_t(i) * 4, &value, 4);
}
void refused(const Fixture& f, size_t source_bytes, int N, int K, size_t cap) {
    std::vector<uint8_t> dst(std::max(f.original.size(), cap) + 32, 0xA5);
    inv::Stats stats{11, 22, 33, 44, 55};
    CHECK(inv::to_block(f.packed.data(), source_bytes, N, K, dst.data(), cap, &stats) < 0);
    for (uint8_t x : dst) CHECK_EQ(x, uint8_t{0xA5});
    CHECK_EQ(stats.blocks, uint64_t{11}); CHECK_EQ(stats.destination_bytes, uint64_t{55});
}
}

TEST(ds41_packed_inverse_all_code_and_scale_bytes_match_existing_relayout) {
    Fixture f(1, 256 * 32);
    for (size_t i = 0; i < f.codes.size(); ++i) f.codes[i] = uint8_t(i);
    for (int i = 0; i < 256; ++i) f.scales[size_t(i)] = uint8_t(i);
    f.build(); f.check(); f.check(true);
    inv::Stats stats;
    std::vector<uint8_t> out(f.original.size());
    REQUIRE_EQ(inv::to_block(f.packed.data(), f.packed.size(), f.N, f.K, out.data(), out.size(), &stats), RAD_OK);
    CHECK_EQ(stats.raw_rounds, uint64_t{8});
}
TEST(ds41_packed_inverse_every_partial_round_and_row_boundary_is_exact) {
    for (int tail = 0; tail < 32; ++tail) for (int raw = 0; raw < 3; ++raw) {
        Fixture f(3, (32 + tail) * 32);
        if (raw) f.scales[size_t((raw == 1 ? 0 : 2) * f.blocks + f.blocks - 1)] = 255;
        f.build(); f.check(tail & 1);
    }
}
TEST(ds41_packed_inverse_native_gu_down_full_bytes_and_configured_budgets) {
    for (bool down : {false, true}) {
        Fixture f(down ? 5120 : 4608, down ? 2304 : 5120);
        f.scales.front() = 0; f.scales.back() = 255;
        f.build(down ? 320 : 512); f.check(true);
        CHECK_EQ(f.original.size(), down ? size_t{6266880} : size_t{12533760});
    }
}
TEST(ds41_packed_inverse_refuses_bad_headers_shape_truncation_and_capacity_without_writes) {
    Fixture f(2, 2048); f.scales.front() = 0; f.scales.back() = 255; f.build();
    for (size_t n : {size_t{0}, size_t{63}, f.packed.size() - 1}) refused(f, n, f.N, f.K, f.original.size());
    refused(f, f.packed.size(), f.N, f.K + 32, f.original.size() + 32);
    refused(f, f.packed.size(), f.N + 1, f.K, f.original.size() + 32);
    refused(f, f.packed.size(), f.N, f.K, f.original.size() - 1);
    for (size_t byte : {size_t{0}, size_t{4}, size_t{24}, size_t{32}}) {
        const auto saved = f.packed[byte];
        f.packed[byte] = byte == 24 ? uint8_t{255} : uint8_t(f.packed[byte] ^ 0x80);
        refused(f, f.packed.size(), f.N, f.K, f.original.size()); f.packed[byte] = saved;
    }
    CHECK_EQ(inv::block_bytes(0, 32), size_t{0}); CHECK_EQ(inv::block_bytes(1, 31), size_t{0});
    CHECK_EQ(inv::block_bytes(INT32_MAX, 2048), size_t{0});
}
TEST(ds41_packed_inverse_refuses_duplicate_unsorted_or_out_of_range_raw_indices) {
    Fixture f(2, 2048);
    for (size_t i = 0; i < f.scales.size(); ++i) f.scales[i] = uint8_t(i);
    f.build(); const auto original = f.packed;
    ds41hx::Mx v; REQUIRE(ds41hx::packed_view(f.packed.data(), f.packed.size(), f.N, f.K, &v));
    REQUIRE(v.n_exc >= 2);
    put_index(f, 0, uint32_t(f.N * f.rounds)); refused(f, f.packed.size(), f.N, f.K, f.original.size());
    f.packed = original; put_index(f, 1, 0); refused(f, f.packed.size(), f.N, f.K, f.original.size());
    f.packed = original; put_index(f, 0, 1); put_index(f, 1, 0);
    refused(f, f.packed.size(), f.N, f.K, f.original.size());
}
TEST(ds41_packed_inverse_refuses_source_destination_and_stats_aliases) {
    Fixture f(1, 2048); f.build();
    std::vector<uint8_t> arena(f.packed.size() + f.original.size() + 64, 0xA5);
    std::memcpy(arena.data(), f.packed.data(), f.packed.size()); const auto original = arena;
    CHECK_EQ(inv::to_block(arena.data(), f.packed.size(), f.N, f.K, arena.data() + 64, f.original.size()), RAD_E_INVAL);
    CHECK(arena == original);
    std::vector<uint8_t> out(f.original.size(), 0xA5);
    CHECK_EQ(inv::to_block(f.packed.data(), f.packed.size(), f.N, f.K, out.data(), out.size(),
                           reinterpret_cast<inv::Stats*>(out.data())), RAD_E_INVAL);
    CHECK_EQ(inv::to_block(f.packed.data(), f.packed.size(), f.N, f.K, out.data(), out.size(),
                           reinterpret_cast<inv::Stats*>(f.packed.data())), RAD_E_INVAL);
    for (uint8_t v : out) CHECK_EQ(v, uint8_t{0xA5});
    CHECK_EQ(inv::to_block(nullptr, f.packed.size(), f.N, f.K, out.data(), out.size()), RAD_E_INVAL);
    CHECK_EQ(inv::to_block(f.packed.data(), f.packed.size(), f.N, f.K, nullptr, out.size()), RAD_E_INVAL);
}
RAD_TEST_MAIN()
