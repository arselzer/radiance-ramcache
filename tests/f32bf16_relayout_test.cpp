/* f32bf16_relayout_test.cpp -- gemm_nt_f32bf16's stored form (libr4d/r4d_layout.cpp): an f32
 * weight kept as its bf16 halves, the conversion RADIANCE_EXPERIMENTAL_DS41_ROUTER_BF16 puts on
 * DeepSeek-V4.1's routers and compressor projections at load.
 *
 *   Layout     an f32 plane is stored bf16 [rows, cols] under one tag whatever column slice of it
 *              an op reads (the split router's part reads N/world of the rows); a bf16 or f16
 *              plane is left to the loader as it is; anything but a plain float plane is refused
 *   Relayout   the stored halves are the f32 bits' high halves, row-major, for every bf16 pattern
 *              (+-0, +-inf, NaNs with payload, subnormals included); one nonzero low half anywhere
 *              refuses the whole weight rather than rounding it
 *   Inverse    the stored halves widen back to the f32 plane bit for bit
 *
 * The kernel side, that the widened halves give every output bit of gemm_nt_f32, is
 * tests/r4d_emu/test_f32.cpp's (all tile forms on the host emulator) and r4d_selftest's on a card.
 */
#include "rad_test.h"

#include "r4d_plugin.h"

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace {

uint32_t widen(uint16_t h) { return (uint32_t)h << 16; }

RadParam pint(const char* k, long long v) {
    RadParam p{};
    p.key = k;
    p.kind = RAD_P_INT;
    p.ival = v;
    return p;
}

/* gemm_nt_f32bf16's parameters at one column slice N of a [rows, K] weight. */
struct Geo {
    RadParam p[4];
    Geo(long long n, long long k, long long m = 1) {
        p[0] = pint("M", m);
        p[1] = pint("N", n);
        p[2] = pint("K", k);
        p[3] = RadParam{};
        p[3].key = "dtype";
        p[3].kind = RAD_P_STR;
        p[3].sval = "bf16";
    }
};

RadTensor plane(uint32_t dtype, int64_t rows, int64_t cols, void* data = nullptr) {
    RadTensor t{};
    t.dtype = dtype;
    t.rank = 2;
    t.shape[0] = rows;
    t.shape[1] = cols;
    t.stride[0] = cols;
    t.stride[1] = 1;
    t.data = data;
    return t;
}

constexpr int kW = 1;   /* the weight operand, oGemmNt's `b` */

/* A [rows, cols] f32 plane of bf16 values: random patterns and, at the front, the specials. */
std::vector<uint32_t> bf16_valued(int64_t rows, int64_t cols, uint32_t seed) {
    std::vector<uint32_t> w((size_t)(rows * cols));
    std::mt19937 rng(seed);
    for (auto& v : w) v = widen((uint16_t)(rng() & 0xFFFFu));
    const uint16_t special[] = { 0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0x7FC1, 0xFFFF,
                                 0x0001, 0x807F, 0x0080, 0x7F7F, 0xFF7F, 0x3F80, 0xBF80 };
    for (size_t i = 0; i < sizeof special / sizeof special[0] && i < w.size(); ++i)
        w[i] = widen(special[i]);
    return w;
}

}  // namespace

TEST(f32bf16_layout_stores_an_f32_plane_as_bf16_whatever_slice_an_op_reads) {
    const RadEncoding enc = rad_enc_plain(RAD_F32);
    const int sel = 0;
    const RadTensor pl = plane(RAD_F32, 384, 5120);
    /* The router's own projection, and the split part's column slice at TP4: one stored form. */
    for (long long n : { 384LL, 96LL, 1LL }) {
        Geo g(n, 5120);
        RadLayout L{};
        REQUIRE_EQ(r4d_rad_layout_f32bf16(g.p, 4, kW, &enc, &sel, &pl, 1, &L), RAD_OK);
        CHECK_EQ(std::string(L.tag), std::string("r4d.f32bf16.rowmajor"));
        CHECK_EQ(L.dtype, (uint32_t)RAD_BF16);
        CHECK_EQ(L.rank, 2u);
        CHECK_EQ(L.shape[0], 384);
        CHECK_EQ(L.shape[1], 5120);
        CHECK_EQ(L.bytes, (int64_t)384 * 5120 * 2);
        CHECK_EQ(L.align, 256);
    }
    /* The compressor's kv|gate projection. */
    {
        Geo g(1024, 5120, 16);
        const RadTensor pc = plane(RAD_F32, 1024, 5120);
        RadLayout L{};
        REQUIRE_EQ(r4d_rad_layout_f32bf16(g.p, 4, kW, &enc, &sel, &pc, 1, &L), RAD_OK);
        CHECK_EQ(L.bytes, (int64_t)1024 * 5120 * 2);
    }
}

TEST(f32bf16_layout_leaves_bf16_and_f16_planes_and_other_operands_as_they_are) {
    Geo g(384, 5120);
    const int sel = 0;
    RadLayout L{};
    for (uint32_t dt : { (uint32_t)RAD_BF16, (uint32_t)RAD_F16 }) {
        const RadEncoding enc = rad_enc_plain(dt);
        const RadTensor pl = plane(dt, 384, 5120);
        CHECK_EQ(r4d_rad_layout_f32bf16(g.p, 4, kW, &enc, &sel, &pl, 1, &L), RAD_E_UNSUPPORTED);
    }
    const RadEncoding enc = rad_enc_plain(RAD_F32);
    const RadTensor pl = plane(RAD_F32, 384, 5120);
    for (int operand : { 0, 2 })
        CHECK_EQ(r4d_rad_layout_f32bf16(g.p, 4, operand, &enc, &sel, &pl, 1, &L),
                 RAD_E_UNSUPPORTED);
}

TEST(f32bf16_layout_refuses_what_is_not_a_plain_float_plane_of_the_op) {
    const int sel[2] = { 0, 1 };
    RadLayout L{};
    {
        Geo g(384, 5120);
        const RadEncoding enc = rad_enc_plain(RAD_I32);
        const RadTensor pl = plane(RAD_I32, 384, 5120);
        CHECK_EQ(r4d_rad_layout_f32bf16(g.p, 4, kW, &enc, sel, &pl, 1, &L), RAD_E_DTYPE);
    }
    {
        /* A quantised weight: codes and a scale plane. */
        Geo g(384, 5120);
        const RadEncoding enc = rad_enc_affine(RAD_I8, RAD_F32, 1, 128);
        const RadTensor pl[2] = { plane(RAD_I8, 384, 5120), plane(RAD_F32, 384, 40) };
        CHECK_EQ(r4d_rad_layout_f32bf16(g.p, 4, kW, &enc, sel, pl, 2, &L), RAD_E_DTYPE);
    }
    {
        /* K is not the plane's width; N is wider than the plane. */
        const RadEncoding enc = rad_enc_plain(RAD_F32);
        const RadTensor pl = plane(RAD_F32, 384, 5120);
        Geo k(384, 4096), n(512, 5120);
        CHECK_EQ(r4d_rad_layout_f32bf16(k.p, 4, kW, &enc, sel, &pl, 1, &L), RAD_E_SHAPE);
        CHECK_EQ(r4d_rad_layout_f32bf16(n.p, 4, kW, &enc, sel, &pl, 1, &L), RAD_E_SHAPE);
    }
}

TEST(f32bf16_relayout_keeps_the_high_halves_and_the_inverse_widens_them_back) {
    const RadEncoding enc = rad_enc_plain(RAD_F32);
    const int sel = 0;
    for (const auto& shape : { std::pair<int64_t, int64_t>{ 384, 5120 }, { 7, 129 }, { 1, 1 } }) {
        const int64_t rows = shape.first, cols = shape.second;
        std::vector<uint32_t> w = bf16_valued(rows, cols, (uint32_t)(rows * 31 + cols));
        const RadTensor pl = plane(RAD_F32, rows, cols, w.data());
        Geo g(rows, cols);
        std::vector<uint16_t> stored((size_t)(rows * cols), 0xA5A5);
        REQUIRE_EQ(r4d_rad_relayout_f32bf16(g.p, 4, kW, &enc, &sel, &pl, 1, stored.data(),
                                            (int64_t)stored.size() * 2),
                   RAD_OK);
        int64_t bad = 0;
        for (size_t i = 0; i < w.size(); ++i) bad += stored[i] != (uint16_t)(w[i] >> 16);
        CHECK_EQ(bad, 0);
        std::vector<uint32_t> back(w.size(), 0xDEADBEEFu);
        const RadTensor out = plane(RAD_F32, rows, cols, back.data());
        REQUIRE_EQ(r4d_rad_unrelayout_f32bf16(g.p, 4, kW, &enc, &sel, stored.data(),
                                              (int64_t)stored.size() * 2, &out, 1),
                   RAD_OK);
        CHECK(std::memcmp(back.data(), w.data(), w.size() * 4) == 0);
    }
}

/* Every bf16 bit pattern survives the round trip: the f32 the kernel multiplies is the stored one. */
TEST(f32bf16_round_trip_holds_for_every_bf16_pattern) {
    const RadEncoding enc = rad_enc_plain(RAD_F32);
    const int sel = 0;
    std::vector<uint32_t> w(65536);
    for (uint32_t h = 0; h < 65536; ++h) w[h] = widen((uint16_t)h);
    const RadTensor pl = plane(RAD_F32, 256, 256, w.data());
    Geo g(256, 256);
    std::vector<uint16_t> stored(65536);
    REQUIRE_EQ(r4d_rad_relayout_f32bf16(g.p, 4, kW, &enc, &sel, &pl, 1, stored.data(), 65536 * 2),
               RAD_OK);
    int64_t bad = 0;
    for (uint32_t h = 0; h < 65536; ++h) {
        const uint32_t again = widen(stored[h]);
        bad += stored[h] != (uint16_t)h || std::memcmp(&again, &w[h], 4) != 0;
    }
    CHECK_EQ(bad, 0);
}

/* A value bf16 cannot hold refuses the weight, wherever it is: first element, last, middle; the
 * smallest and the largest nonzero low half. Nothing rounds. */
TEST(f32bf16_relayout_refuses_a_weight_with_any_nonzero_low_half) {
    const RadEncoding enc = rad_enc_plain(RAD_F32);
    const int sel = 0;
    const int64_t rows = 96, cols = 640;
    Geo g(rows, cols);
    const std::vector<uint32_t> clean = bf16_valued(rows, cols, 7);
    for (int64_t at : { (int64_t)0, rows * cols - 1, 37 * cols + 211 })
        for (uint32_t low : { 0x0001u, 0x8000u, 0xFFFFu }) {
            std::vector<uint32_t> w = clean;
            w[(size_t)at] |= low;
            const RadTensor pl = plane(RAD_F32, rows, cols, w.data());
            std::vector<uint16_t> stored((size_t)(rows * cols));
            CHECK_EQ(r4d_rad_relayout_f32bf16(g.p, 4, kW, &enc, &sel, &pl, 1, stored.data(),
                                              (int64_t)stored.size() * 2),
                     RAD_E_DTYPE);
        }
    /* The genuine f32 router a GLM checkpoint ships: refused at its first element. */
    std::vector<uint32_t> f32(clean.size());
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 0.02f);
    for (auto& v : f32) {
        const float x = nd(rng);
        std::memcpy(&v, &x, 4);
    }
    const RadTensor pl = plane(RAD_F32, rows, cols, f32.data());
    std::vector<uint16_t> stored((size_t)(rows * cols));
    CHECK_EQ(r4d_rad_relayout_f32bf16(g.p, 4, kW, &enc, &sel, &pl, 1, stored.data(),
                                      (int64_t)stored.size() * 2),
             RAD_E_DTYPE);
}

/* Large M does not weaken the load-time contract. In particular the split router's N=96
 * reader must still reject a bad low half in any of the whole 384-row source plane, including
 * the rows outside that reader's slice. The compressor can arrive directly as bf16, in which
 * case this conversion remains unsupported and the loader keeps its existing bytes. */
TEST(f32bf16_tall_served_geometries_check_the_whole_source_plane_before_slicing) {
    const RadEncoding enc = rad_enc_plain(RAD_F32);
    const int sel = 0;
    constexpr int64_t K = 5120;
    struct Shape { int64_t m, n, whole_n; };
    const Shape shapes[] = { { 256, 96, 384 }, { 1024, 384, 384 }, { 2048, 1024, 1024 } };
    for (const Shape& s : shapes) {
        Geo g(s.n, K, s.m);
        std::vector<uint32_t> w((size_t)(s.whole_n * K), widen(0x3f80));
        const RadTensor source = plane(RAD_F32, s.whole_n, K, w.data());
        RadLayout L{};
        REQUIRE_EQ(r4d_rad_layout_f32bf16(g.p, 4, kW, &enc, &sel, &source, 1, &L), RAD_OK);
        CHECK_EQ(L.shape[0], s.whole_n);
        CHECK_EQ(L.shape[1], K);
        CHECK_EQ(L.bytes, s.whole_n * K * 2);
        std::vector<uint16_t> stored((size_t)(s.whole_n * K), 0xa5a5);
        REQUIRE_EQ(r4d_rad_relayout_f32bf16(g.p, 4, kW, &enc, &sel, &source, 1,
                                            stored.data(), L.bytes), RAD_OK);
        int64_t bad = 0;
        for (uint16_t h : stored) bad += h != 0x3f80;
        CHECK_EQ(bad, 0);
        const int64_t where[] = { 0, s.n * K / 2 + 211, s.whole_n * K - 1 };
        const uint32_t low[] = { 0x0001u, 0x8000u, 0xffffu };
        for (int i = 0; i < 3; ++i) {
            w[(size_t)where[i]] |= low[i];
            CHECK_EQ(r4d_rad_relayout_f32bf16(g.p, 4, kW, &enc, &sel, &source, 1,
                                              stored.data(), L.bytes), RAD_E_DTYPE);
            w[(size_t)where[i]] = widen(0x3f80);
        }
    }
    Geo compressor(1024, K, 2048);
    const RadEncoding direct = rad_enc_plain(RAD_BF16);
    const RadTensor source = plane(RAD_BF16, 1024, K);
    RadLayout L{};
    CHECK_EQ(r4d_rad_layout_f32bf16(compressor.p, 4, kW, &direct, &sel, &source, 1, &L),
             RAD_E_UNSUPPORTED);
}

TEST(f32bf16_relayout_and_inverse_check_the_stored_size) {
    const RadEncoding enc = rad_enc_plain(RAD_F32);
    const int sel = 0;
    std::vector<uint32_t> w = bf16_valued(8, 64, 3);
    const RadTensor pl = plane(RAD_F32, 8, 64, w.data());
    Geo g(8, 64);
    std::vector<uint16_t> stored(8 * 64 + 1);
    CHECK_EQ(r4d_rad_relayout_f32bf16(g.p, 4, kW, &enc, &sel, &pl, 1, stored.data(), 8 * 64 * 4),
             RAD_E_SHAPE);
    CHECK_EQ(r4d_rad_relayout_f32bf16(g.p, 4, kW, &enc, &sel, &pl, 1, stored.data(),
                                      8 * 64 * 2 + 2),
             RAD_E_SHAPE);
    std::vector<uint32_t> back(8 * 64);
    const RadTensor out = plane(RAD_F32, 8, 64, back.data());
    CHECK_EQ(r4d_rad_unrelayout_f32bf16(g.p, 4, kW, &enc, &sel, stored.data(), 8 * 64 * 2 - 2,
                                        &out, 1),
             RAD_E_SHAPE);
}

RAD_TEST_MAIN()
