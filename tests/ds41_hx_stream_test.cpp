/* Source-prepared byte oracle for the default-off native one-row stream decoder.
 * The unchanged AVX2 entry is the oracle, including its FP16 fast path, general path,
 * half-input treatment of exceptional values, fixed wave fold and BF16 NaN bits.
 * This does NOT assert that arbitrary exceptional inputs equal a different CPU/GPU algorithm. */
#include "rad_test.h"
#include "../tools/ds41-host-experts/ds41_host_experts.h"
#include <immintrin.h>

namespace {

bool supported() { return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("f16c"); }

struct Matrix {
    int N, K;
    std::vector<uint8_t> codes, scales, blocks, packed;
    ds41hx::Mx planar, ggml, pk;

    Matrix(int n, int k, int profile, uint32_t seed = 7) : N(n), K(k),
        codes((size_t)n * k / 2), scales((size_t)n * k / 32), blocks((size_t)n * k / 32 * 17) {
        uint32_t state = seed;
        for (size_t i = 0; i < codes.size(); ++i) {
            state ^= state << 13; state ^= state >> 17; state ^= state << 5;
            codes[i] = (uint8_t)(state >> 8);
        }
        const int uk = K / 32;
        for (int row = 0; row < N; ++row)
            for (int u = 0; u < uk; ++u) {
                uint8_t e = (uint8_t)(120 + ((row + u) & 1));
                if (profile == 1) e = (uint8_t)(113 + (row + u) % 15);
                if (profile == 2) {
                    const uint8_t exceptional[] = { 0, 1, 112, 128, 254, 255 };
                    e = exceptional[(row + u) % 6];
                }
                // Deliberate raw first/middle/last rounds, a raw tail and a fast raw round.
                if (profile == 3 && (u == 0 || u == uk / 2 || u == uk - 1))
                    e = (uint8_t)((row & 1) ? 112 : 127);
                scales[(size_t)row * uk + u] = e;
            }
        planar.N = ggml.N = pk.N = N;
        planar.K = ggml.K = pk.K = K;
        planar.codes = codes.data(); planar.scales = scales.data();
        rebuild();
    }

    void rebuild() {
        const int uk = K / 32;
        for (int row = 0; row < N; ++row)
            for (int u = 0; u < uk; ++u) {
                const uint8_t* c = codes.data() + (size_t)row * K / 2 + (size_t)u * 16;
                uint8_t* b = blocks.data() + ((size_t)row * uk + u) * 17;
                b[0] = scales[(size_t)row * uk + u];
                std::memset(b + 1, 0, 16);
                for (int v = 0; v < 32; ++v) {
                    const uint8_t nibble = (uint8_t)((c[v / 2] >> ((v & 1) * 4)) & 15);
                    b[1 + v % 16] |= (uint8_t)(nibble << ((v / 16) * 4));
                }
            }
        ggml.blocks = blocks.data();
        ds41hx::PackInfo plan;
        if (!ds41hx::pack_plan(scales.data(), N, K, &plan)) throw std::runtime_error("pack_plan");
        const size_t bytes = ds41hx::packed_bytes(N, K, plan.n_exc);
        packed.assign(bytes + 64, 0xA5);
        if (ds41hx::pack_planes(codes.data(), scales.data(), N, K, plan.n_exc, plan,
                               packed.data() + 32, bytes) != ds41hx::kPackOk ||
            !ds41hx::packed_view(packed.data() + 32, bytes, N, K, &pk))
            throw std::runtime_error("pack roundtrip");
    }

    bool fast_between(int n0, int n1) const {
        const int uk = K / 32;
        for (int n = n0; n < n1; ++n)
            for (int r = 0; r < uk / 32; ++r) {
                bool fast = true;
                for (int u = 0; u < 32; ++u) {
                    const int e = scales[(size_t)n * uk + 32 * r + u];
                    fast = fast && e >= 113 && e <= 127;
                }
                if (fast) return true;
            }
        return false;
    }
};

std::vector<uint16_t> input(int K, int mode, unsigned offset = 0) {
    static const uint16_t exceptional[] = {
        0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x807F, 0x0080, 0x8080,
        0x3F00, 0xBF00, 0x3F80, 0xBF80, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80,
        0x7F81, 0xFF81, 0x7FC0, 0xFFC0, 0x7FFF, 0xFFFF, 0x7FA5, 0xFFA5
    };
    std::vector<uint16_t> x((size_t)K + 16, 0xBEEF);
    uint32_t s = 37 + offset;
    for (int k = 0; k < K; ++k) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        if (mode == 0) x[k] = (uint16_t)((s & 0x807F) | ((117 + (s >> 9) % 21) << 7));
        if (mode == 1) x[k] = exceptional[(k + offset) % (sizeof(exceptional) / sizeof(*exceptional))];
        if (mode == 2) x[k] = (uint16_t)(offset + (unsigned)k);
    }
    return x;
}

struct alignas(32) Eight { float f[8]; };

void compare(const Matrix& m, const ds41hx::Mx& w, const std::vector<uint16_t>& x, int n0, int n1,
             bool expected_used) {
    constexpr uint16_t guard = 0xA17E;
    std::vector<uint16_t> old((size_t)m.N + 16, guard), now = old, lanes = old, old_lanes = old;
    const auto before_x = x;
    ds41hx::gemv_avx2(w, x.data(), old.data() + 8, n0, n1);
    CHECK_EQ(ds41hx::gemv_avx2_stream(w, x.data(), now.data() + 8, n0, n1), expected_used);
    CHECK(now == old);
    std::vector<Eight> xt(ds41hx::few_lanes_floats(m.K) / 8);
    const uint16_t* xs[1] = { x.data() };
    ds41hx::few_lanes(xs, 1, m.K, xt.data()->f);
    uint16_t* ys[1] = { old_lanes.data() + 8 };
    ds41hx::gemv_avx2_few_lanes(w, xt.data()->f, ys, 1, n0, n1);
    CHECK_EQ(ds41hx::gemv_avx2_stream_lanes(w, xt.data()->f, lanes.data() + 8, n0, n1), expected_used);
    CHECK(lanes == old);
    CHECK(old_lanes == old);
    CHECK(x == before_x);
    // Check BOTH implementations' writes against explicit row and surrounding redzones.
    for (int i = 0; i < m.N + 16; ++i)
        if (i < n0 + 8 || i >= n1 + 8) CHECK_EQ(old[(size_t)i], guard);
    for (int i = 0; i < 32; ++i) {
        CHECK_EQ(m.packed[(size_t)i], 0xA5);
        CHECK_EQ(m.packed[m.packed.size() - 32 + (size_t)i], 0xA5);
    }
}

void native_pair(const Matrix& m, const std::vector<uint16_t>& x, int n0, int n1) {
    const bool used = m.fast_between(n0, n1);
    compare(m, m.planar, x, n0, n1, used);
    compare(m, m.pk, x, n0, n1, used);
}

struct CsrRestore {
    unsigned saved = _mm_getcsr();
    ~CsrRestore() { _mm_setcsr(saved); }
    void set(int ftz, int daz) const {
        // Mask exceptions, nearest rounding; clear flags, select both denormal controls independently.
        _mm_setcsr(((saved | 0x1F80u) & ~(0xE07Fu)) | (ftz ? 0x8000u : 0u) | (daz ? 0x40u : 0u));
    }
};

}  // namespace

TEST(ds41_hx_stream_shapes_nonsquare_ranges_tails_and_raw_cursors) {
    if (!supported()) { SKIP("AVX2/F16C required"); return; }
    const int ks[] = { 32, 224, 256, 992, 1024, 1056, 1280, 2016, 2048, 2304, 5120 };
    for (int K : ks) for (int profile = 0; profile < 4; ++profile) {
        Matrix m(24, K, profile, (uint32_t)K + profile);
        const auto x = input(K, 0);
        native_pair(m, x, 0, m.N);
        native_pair(m, x, 8, 16);  // packed exception cursor must start at a nonzero row
        native_pair(m, x, 16, 16); // no actual round, and no output writes
    }
}

TEST(ds41_hx_stream_all_scale_bytes_and_fast_boundaries) {
    if (!supported()) { SKIP("AVX2/F16C required"); return; }
    Matrix m(8, 1056, 0);
    for (int e = 0; e < 256; ++e) {
        std::fill(m.scales.begin(), m.scales.end(), (uint8_t)e);
        m.rebuild();
        native_pair(m, input(m.K, 0, (unsigned)e), 0, m.N);
        native_pair(m, input(m.K, 1, (unsigned)e), 0, m.N);
    }
}

TEST(ds41_hx_stream_all_bf16_patterns_and_all_code_nibbles) {
    if (!supported()) { SKIP("AVX2/F16C required"); return; }
    Matrix m(8, 5120, 1);
    for (size_t i = 0; i < m.codes.size(); ++i)
        m.codes[i] = (uint8_t)((i & 15) | (((i / 16) & 15) << 4));
    m.rebuild();
    // Every uint16 activation bit pattern occurs, including every signed NaN payload.
    for (unsigned offset = 0; offset < 65536; offset += (unsigned)m.K)
        native_pair(m, input(m.K, 2, offset), 0, m.N);
}

TEST(ds41_hx_stream_exceptional_bytes_under_each_ftz_daz_mode) {
    if (!supported()) { SKIP("AVX2/F16C required"); return; }
    CsrRestore csr;
    for (int ftz = 0; ftz < 2; ++ftz) for (int daz = 0; daz < 2; ++daz) {
        csr.set(ftz, daz);
        for (int K : { 992, 1024, 1056, 2304, 5120 }) for (int profile = 0; profile < 4; ++profile) {
            Matrix m(8, K, profile);
            native_pair(m, input(K, 1), 0, m.N);
        }
    }
}

TEST(ds41_hx_stream_ggml_uses_original_fallback) {
    if (!supported()) { SKIP("AVX2/F16C required"); return; }
    for (int K : { 32, 1024, 1056, 2304, 5120 }) for (int profile = 0; profile < 4; ++profile) {
        Matrix m(16, K, profile);
        compare(m, m.ggml, input(K, 1), 0, m.N, false);
        compare(m, m.ggml, input(K, 0), 8, 16, false);
    }
}

TEST(ds41_hx_stream_served_gate_swiglu_down_bytes) {
    if (!supported()) { SKIP("AVX2/F16C required"); return; }
    Matrix gu(4608, 5120, 3, 101), dn(5120, 2304, 3, 202);
    const auto x = input(5120, 0);
    std::vector<float> silu(65536);
    ds41hx::silu_table_host(silu.data(), 10.f); // host-only identity; model gate uses card-captured table
    std::vector<uint16_t> g0(4608), a0(2304), d0(5120), g(4608), a(2304), d(5120);
    ds41hx::gemv_avx2(gu.planar, x.data(), g0.data(), 0, 4608);
    ds41hx::swiglu_avx2(silu.data(), g0.data(), a0.data(), 2304, 10.f);
    ds41hx::gemv_avx2(dn.planar, a0.data(), d0.data(), 0, 5120);
    for (int packed = 0; packed < 2; ++packed) {
        CHECK(ds41hx::gemv_avx2_stream(packed ? gu.pk : gu.planar, x.data(), g.data(), 0, 4608));
        CHECK(g == g0);
        ds41hx::swiglu_avx2(silu.data(), g.data(), a.data(), 2304, 10.f);
        CHECK(a == a0);
        CHECK(ds41hx::gemv_avx2_stream(packed ? dn.pk : dn.planar, a.data(), d.data(), 0, 5120));
        CHECK(d == d0);
    }
}

RAD_TEST_MAIN()
