/* avx_sparse_kv.cpp -- DeepSeek-V4.1's compressed-KV ops, libref/ref_sparse_kv.cpp at this level:
 *
 *   quant_roundtrip    the block-scaled quantise-dequantise round trip of a cached key
 *   softmax_pool       `ratio` rows folded into one compressed entry by a channel-wise softmax
 *
 * Each kernel computes libref's expression. Where libref's result is exact (the round trip's codes
 * and scales) this one is the same bits; elsewhere it differs from libref only in summation
 * order: rows are widened to f32 once and reduced with this level's vector dot.
 * Parallel regions are over rows and every output has one owner, so the bytes do not depend on the
 * team size. */
#include "avx_vec.h"
#include "avx_common.h"

#include <algorithm>
#include <vector>

using namespace avx;

namespace {

/* ================================================================== the round trip
 * libref's block_scale and roundtrip, value for value (ref_sparse_kv.cpp). */
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

float block_scale(float amax, QFmt f, QScale s) {
    if (f == QFmt::Fp8E4m3) {
        amax = std::max(amax, 1e-4f);
        const float inv = 1.0f / 448.0f;
        return s == QScale::Ue8m0 ? pow2_bits(ceil_log2_bits(amax * inv)) : amax * inv;
    }
    if (s == QScale::E4m3) {
        amax = std::max(amax, 6.0f * 0x1p-9f);
        return avx_fp8e4m3_tab[rad_f32_to_fp8e4m3(amax / 6.0f)];
    }
    amax = std::max(amax, 6.0f * 0x1p-126f);
    const float inv = 1.0f / 6.0f;
    return s == QScale::Ue8m0 ? pow2_bits(ceil_log2_bits(amax * inv)) : amax * inv;
}

/* max over |v[i]| as libref's std::max(amax, fabs(v)): a NaN never replaces the running maximum.
 * vf_max(x, acc) is (x > acc) ? x : acc lane by lane, which keeps acc for a NaN x, and the
 * maximum of the lanes is the same whatever the order. */
float block_amax(const float* v, int64_t n) {
    vf acc = vf_zero();
    int64_t i = 0;
    for (; i + VF_N <= n; i += VF_N) acc = vf_max(vf_abs(vf_loadu(v + i)), acc);
    float m = vf_hmax(acc);
    for (; i < n; ++i) m = std::max(m, std::fabs(v[i]));
    return m;
}

/* v / scale clamped to [-lim, lim] with a NaN kept (std::clamp's), into q. */
void scaled_clamp(const float* v, float scale, float lim, float* q, int64_t n) {
    const vf vs = vf_set1(scale), lo = vf_set1(-lim), hi = vf_set1(lim);
    int64_t i = 0;
    /* vf_max(lo, x) is x for a NaN x, and vf_min(hi, y) is y for a NaN y. */
    for (; i + VF_N <= n; i += VF_N)
        vf_storeu(q + i, vf_min(hi, vf_max(lo, vf_div(vf_loadu(v + i), vs))));
    for (; i < n; ++i) q[i] = std::clamp(v[i] / scale, -lim, lim);
}

const float kE2m1[16] = { 0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                          -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f };

/* One block of `n` values in place: the scale from its maximum, each value to a code and back. */
void roundtrip_block(float* v, int64_t n, QFmt f, QScale s, float* q, uint8_t* codes) {
    const float sc = block_scale(block_amax(v, n), f, s);
    if (f == QFmt::Fp8E4m3) {
        scaled_clamp(v, sc, 448.0f, q, n);
        cvt_f32_to_fp8e4m3(q, codes, n);
        cvt_fp8e4m3_to_f32(codes, q, n);
    } else {
        scaled_clamp(v, sc, 6.0f, q, n);
        for (int64_t i = 0; i < n; ++i) q[i] = kE2m1[e2m1_code(q[i])];
    }
    const vf vs = vf_set1(sc);
    int64_t i = 0;
    for (; i + VF_N <= n; i += VF_N) vf_storeu(v + i, vf_mul(vf_loadu(q + i), vs));
    for (; i < n; ++i) v[i] = q[i] * sc;
}

}  // namespace

AVX_KERNEL(quant_roundtrip) {
    if (!rad_args_have(a, 1)) return RAD_E_INVAL;
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
    AVX_PARALLEL_FOR_IF(x.shape[0] * n >= AVX_PAR_MIN)
    for (int64_t m = 0; m < x.shape[0]; ++m) {
        float* buf = scratch_f32((size_t)(2 * n + (n + 3) / 4 + 16));
        float* row = buf;
        float* q = buf + n;
        uint8_t* codes = (uint8_t*)(q + n);
        const int64_t xb = rowoff(&x, m, n);
        const float* in = in_row(&x, xb, n, row);
        if (in != row) std::memcpy(row, in, (size_t)n * sizeof(float));
        for (int64_t b0 = 0; b0 < n; b0 += block) roundtrip_block(row + b0, block, f, s, q, codes);
        out_row(&x, xb, n, row);
    }
    return RAD_OK;
}

/* out[w, c] = sum_i softmax_i(score[i, c]) value[i, c] over the entry's `ratio` rows, in f32:
 * the max subtracted, exp, the sum, then each weight p / den times the value, as libref. Vectors
 * run along the columns, so every column keeps libref's order over the rows; only exp differs
 * (this level's vf_exp against libref's expf). */
AVX_KERNEL(softmax_pool) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor &st = a->t[0], &out = a->t[1];
    const RadTensor* page = a->n_t > 2 && a->t[2].data ? &a->t[2] : nullptr;
    const int64_t n = p_int(a, "n", 0), ratio = p_int(a, "ratio", 0);
    if (n < 1 || ratio < 1 || ratio > 64 || !st.data || !out.data || st.rank != 2 ||
        out.rank != 2 || out.shape[0] < 1 || out.shape[1] != n ||
        st.shape[0] != out.shape[0] * ratio || st.shape[1] != 2 * n ||
        (page && numel(page) < out.shape[0]))
        return RAD_E_SHAPE;
    const int64_t W = out.shape[0];
    AVX_PARALLEL_FOR_IF(W * ratio * n >= AVX_PAR_MIN)
    for (int64_t w = 0; w < W; ++w) {
        if (page && ldi(page, offlin(page, w)) < 0) continue;
        /* The entry's rows widened: values then scores, `ratio` rows of 2n. */
        float* buf = scratch_f32((size_t)(ratio * 2 * n + n));
        float* acc = buf + ratio * 2 * n;
        for (int64_t i = 0; i < ratio; ++i) {
            float* r = buf + i * 2 * n;
            const float* in = in_row(&st, rowoff(&st, w * ratio + i, 2 * n), 2 * n, r);
            if (in != r) std::memcpy(r, in, (size_t)(2 * n) * sizeof(float));
        }
        int64_t c = 0;
        for (; c + VF_N <= n; c += VF_N) {
            vf mx = vf_set1(-INFINITY);
            for (int64_t i = 0; i < ratio; ++i) mx = vf_max(vf_loadu(buf + i * 2 * n + n + c), mx);
            vf d = vf_zero();
            for (int64_t i = 0; i < ratio; ++i) {
                float* sp = buf + i * 2 * n + n + c;
                const vf e = vf_exp(vf_sub(vf_loadu(sp), mx));
                vf_storeu(sp, e);
                d = vf_add(d, e);
            }
            vf s = vf_zero();
            for (int64_t i = 0; i < ratio; ++i)
                s = vf_add(s, vf_mul(vf_div(vf_loadu(buf + i * 2 * n + n + c), d),
                                     vf_loadu(buf + i * 2 * n + c)));
            vf_storeu(acc + c, s);
        }
        for (; c < n; ++c) {
            float mx = -INFINITY, d = 0, s = 0;
            for (int64_t i = 0; i < ratio; ++i) mx = std::max(mx, buf[i * 2 * n + n + c]);
            for (int64_t i = 0; i < ratio; ++i) {
                float& sp = buf[i * 2 * n + n + c];
                sp = std::exp(sp - mx);
                d += sp;
            }
            for (int64_t i = 0; i < ratio; ++i) s += buf[i * 2 * n + n + c] / d * buf[i * 2 * n + c];
            acc[c] = s;
        }
        out_row(&out, rowoff(&out, w, n), n, acc);
    }
    return RAD_OK;
}
