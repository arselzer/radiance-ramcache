/* avx_ggml.cpp -- gemm_nt_ggml over llama.cpp's own blocks (libref/ref_ggml.cpp), at this
 * level.
 *
 * The weight reaches the kernel as the native blocks this library's layout hook packed from the
 * type's planes (avx_registry.cpp, the same bytes as libref's hook), [N, K / block * block_bytes]
 * bytes, or as the float [N, K] logical form a tool passes. A native row is dequantised with
 * llama.cpp's own dequantize_row_<type> (libquant's lq_ggml_dequant, compiled once at the baseline
 * as libref compiles it), so every weight value is libref's bit for bit; the products then run on
 * this level's tile driver (avx_gemm_tile.cpp), which asks a row decoder for whole blocks of a row
 * at a time. The sums differ from libref's index-order sum only in their order.
 *
 * Built only when libquant's sources are beside libavx's (AVX_HAVE_GGML), as libref's rows are;
 * without them the ops have no row. */
#include "avx_gemm.h"

#ifdef AVX_HAVE_GGML
#include "lq_ggml_dequant.h"
#include "lq_ggml_planes.h"

#include <algorithm>
#include <vector>

using namespace avx;

namespace {

const lq::ggml::PlaneKind* kind_of(const RadArgs* a) {
    const char* s = p_str(a, "type", nullptr);
    if (!s) return nullptr;
    for (uint32_t t = 0; t < 64; ++t) {
        const lq::ggml::PlaneKind* k = lq::ggml::plane_kind(t);
        if (k && !std::strcmp(k->name, s) && lq::ggml::pack_of(t) && lq::ggml::dequant_of(t)) return k;
    }
    return nullptr;
}

/* Rows of native blocks, `rb` bytes apart from `base`: whole blocks of a row dequantised to f32. */
struct GgmlDec final : RowDecoder {
    const uint8_t* base = nullptr;
    int64_t rb = 0, block = 0, bbytes = 0, K = 0;
    lq::ggml::DequantFn dq = nullptr;
    void row(int64_t r, float* dst) const override { dq(base + r * rb, dst, K); }
    void cols(int64_t r, int64_t k0, int64_t k1, float* dst) const override {
        dq(base + r * rb + k0 / block * bbytes, dst, k1 - k0);
    }
    int64_t align() const override { return block; }
    void codes(int64_t r, int64_t k0, int64_t k1, const void** p, int64_t* bytes) const override {
        *p = base + r * rb + k0 / block * bbytes;
        *bytes = (k1 - k0) / block * bbytes;
    }
};

GgmlDec make_dec(const lq::ggml::PlaneKind* k, const void* base, int64_t K) {
    GgmlDec d;
    d.base = (const uint8_t*)base;
    d.block = k->block;
    d.bbytes = (int64_t)k->block_bytes;
    d.rb = K / k->block * d.bbytes;
    d.K = K;
    d.dq = lq::ggml::dequant_of(k->type);
    return d;
}

/* One row of the weight as f32: dequantised, or read from the logical form. */
void weight_row(const RadTensor& w, const GgmlDec* dec, int64_t n, int64_t K, float* dst) {
    if (dec) { dec->row(n, dst); return; }
    const float* r = in_row(&w, rowoff(&w, n, K), K, dst);
    if (r != dst) std::memcpy(dst, r, (size_t)K * sizeof(float));
}

/* Y[m, n] = A[m] . W[n] for every (m, n): the tile driver, or, if it declines, a dot per output
 * (each weight row decoded once). */
void product(const Mat& ma, const Mat& mb, const RadTensor& w, const GgmlDec* dec,
             const RadTensor* Y, int64_t M, int64_t N, int64_t K) {
    if (gemm_tile_f32(ma, mb, nullptr, nullptr, Y, nullptr, M, N, K)) return;
    float* apack = (float*)scratch_raw((size_t)(M * K) * sizeof(float) + 64);
    for (int64_t m = 0; m < M; ++m) widen_cols(ma, m, 0, K, apack + m * K);
    const int64_t ys_r = row_stride(Y, N), ys_c = laststride(Y);
    AVX_PARALLEL_FOR_IF(M * N * K >= AVX_PAR_MIN)
    for (int64_t n = 0; n < N; ++n) {
        float* wr = scratch_f32((size_t)K);
        weight_row(w, dec, n, K, wr);
        for (int64_t m = 0; m < M; ++m) stt(Y, m * ys_r + n * ys_c, dotf(apack + m * K, wr, K));
    }
}

/* gemm_nt_ggml with N and K under the names the caller gives. */
int gemm_ggml_impl(const RadArgs* a, const char* nN, const char* nK) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor &x = a->t[0], &w = a->t[1], &y = a->t[2];
    const lq::ggml::PlaneKind* k = kind_of(a);
    const int64_t M = band_rows(x.rank >= 1 ? x.shape[0] : 0, p_int(a, "M", 0));
    const int64_t N = p_int(a, nN, 0), K = p_int(a, nK, 0);
    if (!k || M <= 0 || N <= 0 || K <= 0 || K % k->block || x.rank != 2 || x.shape[1] != K ||
        y.rank != 2 || y.shape[1] != N || y.shape[0] < M || w.rank != 2 || w.shape[0] != N)
        return RAD_E_SHAPE;
    const int64_t rb = K / k->block * (int64_t)k->block_bytes;
    const bool native = w.dtype == RAD_U8;
    if (native ? w.shape[1] != rb : w.shape[1] != K) return RAD_E_SHAPE;
    const GgmlDec dec = make_dec(k, w.data, K);
    const Mat ma = as_mat(&x, K);
    const Mat mb = native ? Mat{ &w, K, 1, &dec } : as_mat(&w, K);
    product(ma, mb, w, native ? &dec : nullptr, &y, M, N, K);
    return RAD_OK;
}

}  // namespace

AVX_KERNEL(gemm_nt_ggml) { return gemm_ggml_impl(a, "N", "K"); }

#else  /* !AVX_HAVE_GGML: the ops have no row (avx_registry.cpp), so these never run. */

AVX_KERNEL(gemm_nt_ggml)      { (void)a; return RAD_E_UNSUPPORTED; }

#endif /* AVX_HAVE_GGML */
