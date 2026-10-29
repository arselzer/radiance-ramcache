/* bf16f8_layout_test.cpp -- gemm_nt_bf16f8's stored form (libr4d/r4d_bf16f8_layout.h) and the
 * hooks that write it (libr4d/r4d_layout.cpp), on the host.
 *
 * The row's claim is that the stored form decodes to the bf16 plane's own bits, so the kernel
 * hands WMMA the operand gemm_bf16_nt_m64 reads. Here:
 *   - the encoder accepts, at every exponent, exactly the 254 non-NaN codes' images, and each with
 *     its own code (all 65,536 bf16 patterns at all 237 exponents);
 *   - the kernel's decode (exact E4M3 to f32, times 2^(s-127), the high half) is the expansion;
 *   - the derived block exponent is the smallest that holds the block, and none larger holds a
 *     block it refuses;
 *   - the kernel's addressing (r4d_gemm_bf16f8_nt_m64.hip: the slice's packet and scale pointers,
 *     the peel of a half block, the KU unroll with its whole-dword or byte-wise scales, the tail)
 *     reads, for every wave, k step, lane and slot, the element fragU reads, at every legal split;
 *   - refusals, a round trip through the inverse, a pinned hash and the ds41 byte counts.
 * The emulated kernels themselves are compared in tests/r4d_emu/test_bf16f8.cpp, and on a card by
 * tools/ds41-host-experts/ds41_bf16f8_device_check.hip. */
#include "rad_test.h"

#include "rad_plugin.h"
#include "r4d_plugin.h"
#include "r4d_bf16f8_layout.h"

#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

/* gemm_nt's parameters as the core hands them to the hooks. */
struct Geo {
    RadParam p[4];
    Geo(int64_t N, int64_t K) {
        p[0] = RAD_INT("M", 1);
        p[1] = RAD_INT("N", N);
        p[2] = RAD_INT("K", K);
        p[3] = RAD_STR("dtype", "bf16");
    }
};

/* One plain bf16 plane [N][K] and its selection. */
struct Plane {
    RadEncoding enc = rad_enc_plain(RAD_BF16);
    int sel[1] = { 0 };
    RadTensor t{};
    Plane(void* data, int64_t N, int64_t K) {
        t.data = data;
        t.dtype = RAD_BF16;
        t.rank = 2;
        t.shape[0] = N;
        t.shape[1] = K;
        t.stride[0] = K;
        t.stride[1] = 1;
    }
};

int layout(int64_t N, int64_t K, void* w, RadLayout* L, int operand = 1) {
    Geo g(N, K);
    Plane pl(w, N, K);
    return r4d_rad_layout_bf16f8(g.p, 4, operand, &pl.enc, pl.sel, &pl.t, 1, L);
}
int relayout(int64_t N, int64_t K, const std::vector<uint16_t>& w, std::vector<uint8_t>* out) {
    Geo g(N, K);
    Plane pl(const_cast<uint16_t*>(w.data()), N, K);
    out->assign((size_t)r4d_bf16f8_bytes(N, K), 0xEE);
    return r4d_rad_relayout_bf16f8(g.p, 4, 1, &pl.enc, pl.sel, &pl.t, 1, out->data(),
                                   (int64_t)out->size());
}

uint16_t f2b(float f) { return rad_f32_to_bf16(f); }

/* A weight whose every 32x32 block is codes times its own 2^(s-127), s in [lo, lo + span): random
 * codes, with signed zeros, subnormals and the largest normals mixed in. */
std::vector<uint16_t> exact_weight(int64_t N, int64_t K, uint64_t seed, int lo = 100, int span = 30,
                                   std::vector<int>* scales = nullptr) {
    std::mt19937_64 rng(seed);
    const int64_t nb = (N + 31) / 32, kb = K / 32;
    std::vector<int> s((size_t)(nb * kb));
    for (auto& v : s) v = lo + (int)(rng() % (uint64_t)span);
    std::vector<uint16_t> w((size_t)(N * K));
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k) {
            const uint64_t r = rng() % 100;
            uint32_t c;
            if (r < 3) c = 0x00;
            else if (r < 6) c = 0x80;
            else if (r < 10) c = (uint32_t)((rng() % 7 + 1) | (rng() & 0x80));    /* subnormals */
            else if (r < 11) c = (uint32_t)(0x7E | (rng() & 0x80));               /* +-448 */
            else do c = (uint32_t)(rng() & 0xFF); while ((c & 0x7F) == 0x7F);
            w[(size_t)(n * K + k)] = r4d_bf16f8_expand(c, (uint32_t)s[(size_t)((n / 32) * kb + k / 32)]);
        }
    if (scales) *scales = s;
    return w;
}

uint64_t fnv1a(const std::vector<uint8_t>& b) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (uint8_t v : b) h = (h ^ v) * 0x100000001b3ull;
    return h;
}

}  /* namespace */

/* ------------------------------------------------------------------ the codec */

TEST(the_encoder_accepts_exactly_the_images_of_the_codes_at_every_exponent) {
    long long accepted = 0, wrong = 0;
    for (uint32_t s = R4D_BF16F8_SMIN; s <= R4D_BF16F8_SMAX; ++s) {
        std::vector<int> image(65536, -1);
        int dup = 0;
        for (uint32_t b = 0; b < 256; ++b) {
            if ((b & 0x7F) == 0x7F) continue;
            const uint16_t x = r4d_bf16f8_expand(b, s);
            if (image[x] >= 0) ++dup;
            image[x] = (int)b;
        }
        CHECK_EQ(dup, 0);   /* 254 distinct images, +0 and -0 among them */
        for (uint32_t x = 0; x < 65536; ++x) {
            const int c = r4d_bf16f8_encode(x, s);
            wrong += c != image[x];
            accepted += c >= 0;
        }
    }
    CHECK_EQ(wrong, 0LL);
    CHECK_EQ(accepted, 254LL * (R4D_BF16F8_SMAX - R4D_BF16F8_SMIN + 1));   /* 60,198 */
}

/* The kernel decodes a code exactly to f32 (v_cvt_pk_f32_fp8; here rad_fp8e4m3_to_f32, an
 * independent decoder), multiplies by 2^(s-127) and keeps the high half. That product is exact, a
 * normal number or a signed zero, and its low half is zero, for every pair. */
TEST(the_kernels_decode_is_the_expansion_for_every_code_and_exponent) {
    int bad = 0, pairs = 0;
    for (uint32_t s = R4D_BF16F8_SMIN; s <= R4D_BF16F8_SMAX; ++s) {
        const uint32_t su = s << 23;
        float f;
        std::memcpy(&f, &su, 4);
        for (uint32_t b = 0; b < 256; ++b) {
            if ((b & 0x7F) == 0x7F) continue;
            const float p = rad_fp8e4m3_to_f32((uint8_t)b) * f;
            uint32_t u;
            std::memcpy(&u, &p, 4);
            const bool normal_or_zero = (u & 0x7FFFFFFFu) == 0 || ((u >> 23) & 0xFF) != 0;
            bad += (u & 0xFFFFu) != 0 || (uint16_t)(u >> 16) != r4d_bf16f8_expand(b, s) ||
                   !normal_or_zero;
            ++pairs;
        }
    }
    CHECK_EQ(pairs, 60198);
    CHECK_EQ(bad, 0);
}

/* The derived exponent is never above the one a block was built from, and a block it refuses no
 * larger exponent holds either. */
TEST(the_block_exponent_is_the_smallest_that_holds_the_block) {
    std::mt19937_64 rng(7);
    int above = 0, below = 0, blocks = 0, refused = 0, held_later = 0;
    for (int trial = 0; trial < 600; ++trial) {
        const int strue = R4D_BF16F8_SMIN + (int)(rng() % (R4D_BF16F8_SMAX - R4D_BF16F8_SMIN + 1));
        uint16_t blk[32][32];
        for (auto& row : blk)
            for (auto& x : row) {
                uint32_t c;
                do c = (uint32_t)(rng() & 0xFF); while ((c & 0x7F) == 0x7F);
                x = r4d_bf16f8_expand(c, (uint32_t)strue);
            }
        /* A third of the blocks get one value that is not a code at any exponent the block can
         * take: an extra mantissa bit on a large element. */
        const bool spoil = trial % 3 == 0;
        if (spoil) blk[rng() % 32][rng() % 32] |= 0x0001;
        const int s = r4d_bf16f8_block_exp(&blk[0][0], 32, 32);
        ++blocks;
        if (!spoil) {
            above += s > strue;
            below += s < strue;
        }
        bool ok = true;
        for (auto& row : blk)
            for (uint16_t x : row) ok = ok && r4d_bf16f8_encode(x, (uint32_t)s) >= 0;
        if (!spoil) CHECK(ok);
        if (ok) continue;
        ++refused;
        for (int s2 = s + 1; s2 <= R4D_BF16F8_SMAX; ++s2) {
            bool all = true;
            for (auto& row : blk)
                for (uint16_t x : row) all = all && r4d_bf16f8_encode(x, (uint32_t)s2) >= 0;
            held_later += all;
        }
    }
    CHECK_EQ(above, 0);
    CHECK(refused > 0);
    CHECK_EQ(held_later, 0);
    (void)below;   /* a block whose largest code is small gets a smaller exponent: allowed */
    CHECK_EQ(blocks, 600);
}

TEST(a_block_of_zeros_takes_127_and_tiny_blocks_the_floor) {
    uint16_t z[16][32] = {};
    z[3][4] = 0x8000;
    CHECK_EQ(r4d_bf16f8_block_exp(&z[0][0], 32, 16), 127);
    z[5][6] = 0x0080;   /* 2^-126, the least normal bf16: code 1 (a subnormal) at s = 10 */
    CHECK_EQ(r4d_bf16f8_block_exp(&z[0][0], 32, 16), R4D_BF16F8_SMIN);
    CHECK_EQ(r4d_bf16f8_encode(0x0080, R4D_BF16F8_SMIN), 1);
    CHECK_EQ(r4d_bf16f8_encode(0x8080, R4D_BF16F8_SMIN), 0x81);
    /* Every exponent field from 1 up to the floor's: the block takes the floor and holds them. */
    for (int E = 1; E <= 17; ++E) {
        uint16_t t[16][32] = {};
        t[0][0] = (uint16_t)(E << 7);
        CHECK_EQ(r4d_bf16f8_block_exp(&t[0][0], 32, 16), R4D_BF16F8_SMIN);
        CHECK(r4d_bf16f8_encode(t[0][0], R4D_BF16F8_SMIN) >= 0);
    }
}

/* ------------------------------------------------------------------ the kernel's addressing */

namespace {

/* pick_sk at wv 1 (r4d_gemm_args.h): the split the row uses. */
int law_sk(int64_t n_tiles, int64_t K) {
    static const int ladder[] = { 1, 2, 4, 5, 8, 10, 16, 20 };
    int best = 1;
    for (int s : ladder) {
        if (K % (s * 16)) continue;
        best = s;
        if (n_tiles * s >= 128) return s;
    }
    return best;
}

/* r4d_gemm_bf16f8_nt_m64_body's weight addressing, wave by wave, as written there: `pw` in 16-byte
 * units plus the lane, `ps` a byte offset, the scale of a round from whole dwords when every slice
 * starts on 128 K (SALIGN) and from bytes otherwise. Each k step's 32 lanes x 8 slots, decoded,
 * must be the plane's element at fragU's slot: row n0 + (lane & 15), k = kbase + 16u + 8(e >> 2) +
 * 4(lane >> 4) + (e & 3). Returns mismatches; *steps counts wave k steps checked. */
long long emulate(const std::vector<uint16_t>& W, const std::vector<uint8_t>& blob, int64_t N,
                  int64_t K, int SK, int KU, long long* steps) {
    const int64_t soff = r4d_bf16f8_scale_off(N, K), ldws = r4d_bf16f8_ldws(K);
    const int Ks = (int)(K / SK), ksteps = Ks / 16;
    const bool salign = (K / SK) % 128 == 0;
    long long bad = 0;
    for (int64_t nt = 0; nt < N / 16; ++nt)
        for (int ksplit = 0; ksplit < SK; ++ksplit) {
            const int kbase = ksplit * Ks;
            const int64_t kb0 = kbase >> 5;
            int64_t pw = (nt * (K >> 5) + kb0) * 32;           /* + lane, 16-byte units */
            int64_t ps = soff + (nt >> 1) * ldws + kb0;
            int u_all = 0;
            auto step = [&](int64_t pwq, int half, uint32_t s) {
                for (int lane = 0; lane < 32; ++lane)
                    for (int e = 0; e < 8; ++e) {
                        const uint8_t c = blob[(size_t)((pwq + lane) * 16 + 8 * half + e)];
                        const int64_t n = nt * 16 + (lane & 15);
                        const int64_t k = kbase + 16 * u_all + 8 * (e >> 2) + 4 * (lane >> 4) + (e & 3);
                        bad += r4d_bf16f8_expand(c, s) != W[(size_t)(n * K + k)];
                    }
                ++u_all;
                ++*steps;
            };
            int s = 0;
            if (!salign && (kbase & 16)) {                      /* the peel: a block's second step */
                step(pw, 1, blob[(size_t)ps]);
                pw += 32;
                ps += 1;
                s = 1;
            }
            for (; s + KU <= ksteps; s += KU) {
                uint32_t sw[2] = { 0, 0 };
                for (int j = 0; j < KU / 8; ++j) {
                    if (salign && (ps % 4)) ++bad;              /* a whole-dword read off its dword */
                    for (int b = 0; b < 4; ++b) sw[j] |= (uint32_t)blob[(size_t)(ps + 4 * j + b)] << (8 * b);
                }
                for (int u = 0; u < KU; ++u)
                    step(pw + 32 * (u >> 1), u & 1, (sw[u >> 3] >> (8 * ((u >> 1) & 3))) & 0xFFu);
                pw += (KU / 2) * 32;
                ps += KU / 2;
            }
            for (; s < ksteps; s += 2) {                        /* whole blocks, then a half */
                step(pw, 0, blob[(size_t)ps]);
                if (s + 1 < ksteps) step(pw, 1, blob[(size_t)ps]);
                pw += 32;
                ps += 1;
            }
            bad += u_all != ksteps;
        }
    return bad;
}

}  /* namespace */

TEST(the_kernels_addressing_reads_fragUs_element_at_every_legal_split) {
    struct Shape { int64_t N, K; };
    /* The peel (K 96 at SK 2, K 160 at SK 10), the byte-wise scales, N % 32 == 16, the half-block
     * tail, and K 5120 / 1280 / 768 at the ds41 shapes' splits. */
    const Shape shapes[] = { { 16, 96 },   { 48, 96 },   { 64, 160 },  { 32, 512 },
                             { 48, 1280 }, { 128, 5120 }, { 80, 768 }, { 1024, 576 } };
    long long steps = 0, want = 0;
    for (const Shape& sh : shapes) {
        const auto W = exact_weight(sh.N, sh.K, 1000 + (uint64_t)sh.N * 7 + (uint64_t)sh.K);
        std::vector<uint8_t> blob;
        REQUIRE_EQ(relayout(sh.N, sh.K, W, &blob), RAD_OK);
        int legal = 0;
        for (int SK : { 1, 2, 4, 5, 8, 10, 16, 20 }) {
            if (sh.K % (SK * 16)) continue;
            ++legal;
            want += 2 * (sh.N / 16) * (sh.K / 16);   /* every wave's every k step, at both KU */
            for (int KU : { 16, 8 }) {
                const long long bad = emulate(W, blob, sh.N, sh.K, SK, KU, &steps);
                if (bad) fprintf(stderr, "    N=%lld K=%lld SK=%d KU=%d: %lld mismatches\n",
                                 (long long)sh.N, (long long)sh.K, SK, KU, bad);
                CHECK_EQ(bad, 0LL);
            }
        }
        CHECK(legal > 0);
        CHECK(law_sk(sh.N / 16, sh.K) >= 1);
    }
    CHECK_EQ(steps, want);
    CHECK(want > 50000);
}

/* ------------------------------------------------------------------ the hooks */

TEST(the_layout_hook_sizes_the_stored_form_and_refuses_what_the_kernel_cannot_read) {
    std::vector<uint16_t> w(64 * 128, 0);
    RadLayout L{};
    REQUIRE_EQ(layout(64, 128, w.data(), &L), RAD_OK);
    CHECK_EQ(L.dtype, (uint32_t)R4D_DT_BF16F8);
    CHECK_EQ(L.align, (int64_t)256);
    CHECK_EQ(L.bytes, r4d_bf16f8_bytes(64, 128));
    CHECK_EQ(std::string(L.tag), std::string("r4d.bf16f8.e4m3.e8m0b32.frag32"));
    CHECK_EQ(L.shape[0], (int64_t)64);
    CHECK_EQ(L.shape[1], (int64_t)128);

    CHECK_EQ(layout(64, 128, w.data(), &L, 0), RAD_E_UNSUPPORTED);   /* gemm_nt's `a` */
    CHECK_EQ(layout(64, 128, w.data(), &L, 2), RAD_E_UNSUPPORTED);
    CHECK_EQ(layout(56, 128, w.data(), &L), RAD_E_SHAPE);            /* N % 16 */
    CHECK_EQ(layout(64, 112, w.data(), &L), RAD_E_SHAPE);            /* K % 32 */
    {
        Geo g(64, 128);
        Plane pl(w.data(), 64, 96);                                      /* plane is not [N][K] */
        CHECK_EQ(r4d_rad_layout_bf16f8(g.p, 4, 1, &pl.enc, pl.sel, &pl.t, 1, &L), RAD_E_SHAPE);
    }
    {
        Geo g(64, 128);
        Plane pl(w.data(), 64, 128);
        pl.enc = rad_enc_plain(RAD_F16);                                 /* not bf16 */
        pl.t.dtype = RAD_F16;
        CHECK_EQ(r4d_rad_layout_bf16f8(g.p, 4, 1, &pl.enc, pl.sel, &pl.t, 1, &L), RAD_E_DTYPE);
        pl.enc = rad_enc_affine(RAD_F8E4M3, RAD_E8M0, 32, 32);           /* not plain */
        CHECK_EQ(r4d_rad_layout_bf16f8(g.p, 4, 1, &pl.enc, pl.sel, &pl.t, 1, &L), RAD_E_DTYPE);
        pl.enc = rad_enc_plain(RAD_BF16);
        CHECK_EQ(r4d_rad_layout_bf16f8(g.p, 4, 1, nullptr, pl.sel, &pl.t, 1, &L), RAD_E_DTYPE);
        CHECK_EQ(r4d_rad_layout_bf16f8(g.p, 4, 1, &pl.enc, pl.sel, &pl.t, 2, &L), RAD_E_SHAPE);
    }
}

TEST(the_relayout_refuses_any_weight_that_is_not_exactly_codes_times_a_block_power) {
    const int64_t N = 32, K = 64;
    const auto good = exact_weight(N, K, 3);
    std::vector<uint8_t> blob;
    REQUIRE_EQ(relayout(N, K, good, &blob), RAD_OK);
    /* One element each: a mantissa bit E4M3 does not have, a bf16 subnormal, a NaN, an infinity. */
    for (uint16_t bad : { (uint16_t)0x3F81, (uint16_t)0x0001, (uint16_t)0x7FC0, (uint16_t)0x7F80,
                          (uint16_t)0xFF80 }) {
        auto w = good;
        w[(size_t)(5 * K + 40)] = bad;
        int rc = 0;
        const std::string err = stderr_of([&] { rc = relayout(N, K, w, &blob); });
        CHECK_EQ(rc, RAD_E_FORMAT);
        CHECK(has(err, "bf16f8"));
    }
    /* A block that needs 2^(247-127): its largest magnitude is past 448 * 2^119. */
    {
        auto w = good;
        w[0] = 0x7F7F;   /* the largest finite bf16, 1.99 * 2^127 */
        int rc = 0;
        const std::string err = stderr_of([&] { rc = relayout(N, K, w, &blob); });
        CHECK_EQ(rc, RAD_E_FORMAT);
        CHECK(has(err, "past E8M0"));
    }
    /* A value two orders of magnitude below its block's maximum keeps fewer mantissa bits: 1 + 1/8
     * is a code, but not when the block's exponent puts it among the subnormals. */
    {
        std::vector<uint16_t> w((size_t)(N * K), 0);
        w[0] = f2b(448.0f);
        w[1] = f2b(1.125f / 128.0f);  /* at s = 127 a subnormal, 4.5 * 2^-9: not a code */
        int rc = 0;
        (void)stderr_of([&] { rc = relayout(N, K, w, &blob); });
        CHECK_EQ(rc, RAD_E_FORMAT);
        w[1] = f2b(1.0f / 128.0f);    /* 4 * 2^-9: code 0x04 */
        CHECK_EQ(relayout(N, K, w, &blob), RAD_OK);
    }
    /* The destination must be exactly the stored size. */
    {
        Geo g(N, K);
        Plane pl(const_cast<uint16_t*>(good.data()), N, K);
        std::vector<uint8_t> d((size_t)r4d_bf16f8_bytes(N, K) + 256);
        CHECK_EQ(r4d_rad_relayout_bf16f8(g.p, 4, 1, &pl.enc, pl.sel, &pl.t, 1, d.data(),
                                         (int64_t)d.size()), RAD_E_SHAPE);
        CHECK_EQ(r4d_rad_relayout_bf16f8(g.p, 4, 1, &pl.enc, pl.sel, &pl.t, 1, d.data(),
                                         r4d_bf16f8_bytes(N, K) - 1), RAD_E_SHAPE);
    }
}

TEST(the_inverse_gives_back_every_bit_signed_zeros_and_subnormals_included) {
    for (int64_t N : { 16, 48, 96 })
        for (int64_t K : { 32, 96, 160 }) {
            std::vector<int> sc;
            const auto W = exact_weight(N, K, 11 + (uint64_t)(N * K), R4D_BF16F8_SMIN, 237, &sc);
            std::vector<uint8_t> blob;
            REQUIRE_EQ(relayout(N, K, W, &blob), RAD_OK);
            std::vector<uint16_t> back((size_t)(N * K), 0x5555);
            Geo g(N, K);
            Plane pl(back.data(), N, K);
            REQUIRE_EQ(r4d_rad_unrelayout_bf16f8(g.p, 4, 1, &pl.enc, pl.sel, blob.data(),
                                                 (int64_t)blob.size(), &pl.t, 1), RAD_OK);
            CHECK(back == W);
            /* The pads are zeros (the loader hands the relayout an unwritten buffer). */
            const int64_t soff = r4d_bf16f8_scale_off(N, K), ldws = r4d_bf16f8_ldws(K);
            bool pads = true;
            for (int64_t i = N * K; i < soff; ++i) pads = pads && blob[(size_t)i] == 0;
            for (int64_t nb = 0; nb < (N + 31) / 32; ++nb)
                for (int64_t j = K / 32; j < ldws; ++j) pads = pads && blob[(size_t)(soff + nb * ldws + j)] == 0;
            for (int64_t i = soff + ((N + 31) / 32) * ldws; i < (int64_t)blob.size(); ++i)
                pads = pads && blob[(size_t)i] == 0;
            CHECK(pads);
        }
    /* A NaN code or an out-of-range exponent in the stored bytes is refused by the inverse. */
    const auto W = exact_weight(16, 32, 5);
    std::vector<uint8_t> blob;
    REQUIRE_EQ(relayout(16, 32, W, &blob), RAD_OK);
    std::vector<uint16_t> back(16 * 32);
    Geo g(16, 32);
    Plane pl(back.data(), 16, 32);
    auto nan = blob;
    nan[7] = 0x7F;
    CHECK_EQ(r4d_rad_unrelayout_bf16f8(g.p, 4, 1, &pl.enc, pl.sel, nan.data(), (int64_t)nan.size(),
                                       &pl.t, 1), RAD_E_FORMAT);
    auto big = blob;
    big[(size_t)r4d_bf16f8_scale_off(16, 32)] = 250;
    CHECK_EQ(r4d_rad_unrelayout_bf16f8(g.p, 4, 1, &pl.enc, pl.sel, big.data(), (int64_t)big.size(),
                                       &pl.t, 1), RAD_E_FORMAT);
}

/* The hook's bytes against the closed form (r4d_bf16f8_slot, the derived exponents), then pinned:
 * a deliberate layout change updates the hash. */
TEST(the_stored_bytes_are_the_closed_form_and_pinned) {
    const int64_t N = 48, K = 192;
    std::vector<int> sc;
    const auto W = exact_weight(N, K, 2026, 110, 20, &sc);
    std::vector<uint8_t> blob;
    REQUIRE_EQ(relayout(N, K, W, &blob), RAD_OK);
    std::vector<uint8_t> ref((size_t)r4d_bf16f8_bytes(N, K), 0);
    const int64_t soff = r4d_bf16f8_scale_off(N, K), ldws = r4d_bf16f8_ldws(K);
    for (int64_t nb = 0; nb < (N + 31) / 32; ++nb)
        for (int64_t kb = 0; kb < K / 32; ++kb) {
            const int rows = (int)std::min<int64_t>(32, N - nb * 32);
            ref[(size_t)(soff + nb * ldws + kb)] =
                (uint8_t)r4d_bf16f8_block_exp(&W[(size_t)(nb * 32 * K + kb * 32)], K, rows);
        }
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k) {
            const uint32_t s = ref[(size_t)(soff + (n / 32) * ldws + k / 32)];
            const int c = r4d_bf16f8_encode(W[(size_t)(n * K + k)], s);
            REQUIRE(c >= 0);
            ref[(size_t)r4d_bf16f8_slot(K, n, k)] = (uint8_t)c;
        }
    CHECK(blob == ref);
    const uint64_t h = fnv1a(blob);
    if (h != 0x919f3c855a8d0714ull)
        fprintf(stderr, "    bf16f8 stored-form hash 0x%016llxull\n", (unsigned long long)h);
    CHECK_EQ(h, 0x919f3c855a8d0714ull);
}

/* The per-rank sizes of ds41's FP8-sourced dense weights at TP4: what the planner sees. */
TEST(the_stored_size_of_the_ds41_dense_weights) {
    struct Want { int64_t N, K, bytes; };
    const Want want[] = {
        { 1280, 5120,     6560000 },   /* attn_q_a */
        { 8192, 1280,    10496000 },   /* attn_q_b, a rank's rows */
        { 4096, 1280,     5248000 },   /* indexer.attn_q_b */
        {  512, 5120,     2624000 },   /* attn_kv */
        { 1024, 4096,     4198400 },   /* attn_output_a.gK */
        { 5120, 2048,    10496000 },   /* attn_output_b, a rank's columns */
        { 1536, 5120,     7872000 },   /* ffn_gate_up_shexp, rank 0's rows */
        { 1024, 5120,     5248000 },   /* ffn_gate_up_shexp, ranks 1-3 */
        { 5120,  768,     3936000 },   /* ffn_down_shexp, rank 0's columns */
        { 5120,  512,     2624000 },   /* ffn_down_shexp, ranks 1-3 */
        { 25600, 6144,  157440000 },   /* engram_kv, replicated */
        { 6400, 6144,    39360000 },   /* engram_kv, a rank's rows (ENGRAM_SPLIT) */
    };
    for (const Want& w : want) {
        CHECK_EQ(r4d_bf16f8_bytes(w.N, w.K), w.bytes);
        CHECK(r4d_bf16f8_bytes(w.N, w.K) % 256 == 0);
        /* A byte a value, plus a byte a 32x32 block (+1/1024), plus at most two pads. */
        CHECK(r4d_bf16f8_bytes(w.N, w.K) - w.N * w.K <= ((w.N + 31) / 32) * r4d_bf16f8_ldws(w.K) + 255);
    }
}

RAD_TEST_MAIN()
