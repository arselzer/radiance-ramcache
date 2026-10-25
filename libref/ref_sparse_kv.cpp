/* Compressed-KV building blocks: the block-scaled quantise-dequantise round trip a model applies to
 * its cached keys and activations (the values the cache holds are exactly representable in the
 * activation dtype, so the codes need not be stored), and the channel-wise softmax pooling that
 * folds `ratio` consecutive tokens into one compressed entry. DeepSeek-V4 uses both; neither names
 * a model. */
#include "ref_common.h"
#include "ref_ops.h"

#include <algorithm>
#include <cmath>
#include <cstring>

using namespace ref;

namespace {
/* ceil(log2(v)) from v's IEEE bits: exponent - 127, plus one when any mantissa bit is set. */
int ceil_log2_bits(float v) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    return (int)((u >> 23) & 0xffu) - 127 + ((u & 0x7fffffu) ? 1 : 0);
}
float pow2_bits(int e) {
    const uint32_t u = (uint32_t)(e + 127) << 23;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

enum class QFmt { Fp8E4m3, Fp4E2m1 };
enum class QScale { Ue8m0, E4m3, F32 };

/* One block's scale from its absolute maximum, as the reference kernels compute it: fp8 floors the
 * maximum at 1e-4; fp4 at 6 * 2^-126 (ue8m0) or 6 * 2^-9 (e4m3). A ue8m0 scale is the power of two
 * at or above amax * f32(1/max) -- the product, not the quotient, which differs at a power of two;
 * an e4m3 scale is fp8(amax / 6); an f32 scale is amax * f32(1/max). */
float block_scale(float amax, QFmt f, QScale s) {
    if (f == QFmt::Fp8E4m3) {
        amax = std::max(amax, 1e-4f);
        const float inv = 1.0f / 448.0f;
        return s == QScale::Ue8m0 ? pow2_bits(ceil_log2_bits(amax * inv)) : amax * inv;
    }
    if (s == QScale::E4m3) {
        amax = std::max(amax, 6.0f * 0x1p-9f);
        return fp8e4m3_lut(f32_to_fp8e4m3(amax / 6.0f));
    }
    amax = std::max(amax, 6.0f * 0x1p-126f);
    const float inv = 1.0f / 6.0f;
    return s == QScale::Ue8m0 ? pow2_bits(ceil_log2_bits(amax * inv)) : amax * inv;
}

float roundtrip(float v, float scale, QFmt f) {
    if (f == QFmt::Fp8E4m3)
        return fp8e4m3_lut(f32_to_fp8e4m3(std::clamp(v / scale, -448.0f, 448.0f))) * scale;
    return mxfp4_elem_to_f32((uint8_t)e2m1_code(std::clamp(v / scale, -6.0f, 6.0f))) * scale;
}
}  // namespace

/* x[m, b*block .. (b+1)*block) quantised to `format` with one `scale`-format scale a block, then
 * dequantised in f32 and rounded once to x's dtype, in place. */
int ref_quant_roundtrip(const RadArgs* a, RadStream) {
    if (!have(a, 1)) return RAD_E_INVAL;
    const RadTensor& x = a->t[0];
    const int64_t n = p_int(a, "n", 0), block = p_int(a, "block", 0);
    const char* fs = p_str(a, "format", "");
    const char* ss = p_str(a, "scale", "");
    QFmt f;
    QScale s;
    if (!std::strcmp(fs, "fp8_e4m3")) f = QFmt::Fp8E4m3;
    else if (!std::strcmp(fs, "fp4_e2m1")) f = QFmt::Fp4E2m1;
    else return RAD_E_UNSUPPORTED;
    if (!std::strcmp(ss, "ue8m0")) s = QScale::Ue8m0;
    else if (!std::strcmp(ss, "e4m3") && f == QFmt::Fp4E2m1) s = QScale::E4m3;
    else if (!std::strcmp(ss, "f32") && f == QFmt::Fp8E4m3) s = QScale::F32;
    else return RAD_E_UNSUPPORTED;
    if (n < 1 || block < 1 || n % block || !x.data || x.rank != 2 || x.shape[0] < 1 ||
        x.shape[1] != n)
        return RAD_E_SHAPE;
#pragma omp parallel for schedule(static)
    for (int64_t m = 0; m < x.shape[0]; ++m) {
        const int64_t xb = rowoff(&x, m, n), xs = laststride(&x);
        for (int64_t b0 = 0; b0 < n; b0 += block) {
            float amax = 0;
            for (int64_t i = 0; i < block; ++i)
                amax = std::max(amax, std::fabs(ldt(&x, xb + (b0 + i) * xs)));
            const float sc = block_scale(amax, f, s);
            for (int64_t i = 0; i < block; ++i) {
                const int64_t o = xb + (b0 + i) * xs;
                stt(&x, o, roundtrip(ldt(&x, o), sc, f));
            }
        }
    }
    return RAD_OK;
}

/* Channel-wise softmax pooling of `ratio` consecutive rows: stage [W*ratio, 2n] holds each
 * pooled entry's rows, values in columns [0, n) and scores in [n, 2n). out[w, c] = sum over the
 * entry's rows i of softmax_i(score[i, c]) * value[i, c], in f32 (max subtracted, then exp, sum,
 * divide), rounded once to out's dtype. An entry whose `page` is negative is not written (page
 * is optional; without it every entry is). */
int ref_softmax_pool(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor &st = a->t[0], &out = a->t[1];
    const RadTensor* page = a->n_t > 2 && a->t[2].data ? &a->t[2] : nullptr;
    const int64_t n = p_int(a, "n", 0), ratio = p_int(a, "ratio", 0);
    if (n < 1 || ratio < 1 || ratio > 64 || !st.data || !out.data || st.rank != 2 ||
        out.rank != 2 || out.shape[0] < 1 || out.shape[1] != n ||
        st.shape[0] != out.shape[0] * ratio || st.shape[1] != 2 * n ||
        (page && numel(page) < out.shape[0]))
        return RAD_E_SHAPE;
#pragma omp parallel for schedule(static)
    for (int64_t w = 0; w < out.shape[0]; ++w) {
        if (page && ld_int(page->dtype, page->data, offlin(page, w)) < 0) continue;
        const int64_t ob = rowoff(&out, w, n), os = laststride(&out), ss = laststride(&st);
        for (int64_t c = 0; c < n; ++c) {
            float sc[64], mx = -INFINITY, den = 0, acc = 0;
            for (int64_t i = 0; i < ratio; ++i) {
                sc[i] = ldt(&st, rowoff(&st, w * ratio + i, 2 * n) + (n + c) * ss);
                mx = std::max(mx, sc[i]);
            }
            for (int64_t i = 0; i < ratio; ++i) {
                sc[i] = std::exp(sc[i] - mx);
                den += sc[i];
            }
            for (int64_t i = 0; i < ratio; ++i)
                acc += sc[i] / den * ldt(&st, rowoff(&st, w * ratio + i, 2 * n) + c * ss);
            stt(&out, ob + c * os, acc);
        }
    }
    return RAD_OK;
}
