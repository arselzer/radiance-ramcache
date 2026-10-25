/* avx_sparse_kv.cpp -- DeepSeek-V4.1's compressed-KV ops, libref/ref_sparse_kv.cpp at this level:
 *
 *   quant_roundtrip    the block-scaled quantise-dequantise round trip of a cached key
 *   softmax_pool       `ratio` rows folded into one compressed entry by a channel-wise softmax
 *   attn_sparse_sink   latent attention over a window and the selected compressed entries
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

namespace {
/* The s with cu[s] <= row < cu[s+1] (libref's seq_of_row). */
int64_t seq_of_row(const RadTensor& cu, int64_t n_seq, int64_t row) {
    int64_t lo = 0, hi = n_seq;
    while (hi - lo > 1) {
        const int64_t mid = (lo + hi) / 2;
        if (ldi(&cu, offlin(&cu, mid)) <= row) lo = mid;
        else hi = mid;
    }
    return lo;
}

/* `n` elements of t from flat offset `off`, as f32 (a cache entry: dense, libref's flat read). */
void widen_flat(const RadTensor& t, int64_t off, int64_t n, float* dst) {
    row_to_f32(t.dtype, byte_at(t.data, t.dtype, off), dst, n);
}
}  // namespace

/* Latent attention with a sink over a window and the selected compressed entries (libref's
 * attn_sparse_sink, the same keys): each row's keys are found and widened once, then per head the
 * scores are vector dots, p = exp(s - max) as libref (expf), and the output a vector sum of p k. */
AVX_KERNEL(attn_sparse_sink) {
    if (!a || a->n_t < 10 || !a->t) return RAD_E_INVAL;
    const RadTensor &q = a->t[0], &wc = a->t[1], &wbt = a->t[2];
    const RadTensor *cc = a->t[3].data ? &a->t[3] : nullptr, *cbt = a->t[4].data ? &a->t[4] : nullptr;
    const RadTensor* sel = a->t[5].data ? &a->t[5] : nullptr;
    const RadTensor &pos = a->t[6], &cu = a->t[7], &sink = a->t[8], &out = a->t[9];
    const int64_t H = p_int(a, "heads", 0), D = p_int(a, "dim", 0), win = p_int(a, "window", 0);
    const int64_t ratio = p_int(a, "ratio", 0), topk = p_int(a, "topk", 0);
    const float scale = p_f32(a, "scale", 0.0f);
    const bool comp = ratio > 0;
    if (H < 1 || D < 2 || D > 4096 || win < 1 || ratio < 0 || topk < 0 || !std::isfinite(scale) ||
        scale <= 0 || (comp && (!cc || !cbt || !sel || topk < 1 || D % ratio)))
        return RAD_E_SHAPE;
    if (!q.data || !wc.data || !wbt.data || !pos.data || !cu.data || !sink.data || !out.data ||
        q.rank != 2 || q.shape[0] < 1 || q.shape[1] != H * D || out.rank != 2 ||
        out.shape[0] != q.shape[0] || out.shape[1] != H * D || wc.rank != 4 || wc.shape[1] != 1 ||
        wc.shape[3] != D || wbt.rank != 2 || numel(&pos) != q.shape[0] || numel(&sink) != H ||
        sink.dtype != RAD_F32 || numel(&cu) < 2)
        return RAD_E_SHAPE;
    const int64_t cbs = comp ? cc->shape[2] : 0;
    if (comp && (cc->rank != 4 || cc->shape[1] != 1 || cc->shape[3] * ratio != D || cbs % ratio ||
                 cbt->rank != 2 || sel->rank != 2 || sel->shape[0] != q.shape[0] ||
                 sel->shape[1] != topk || cbt->shape[0] != wbt.shape[0]))
        return RAD_E_SHAPE;
    const int64_t M = q.shape[0], S = numel(&cu) - 1, bs = wc.shape[2];
    const int64_t nkeys = win + (comp ? topk : 0);
    const int64_t qs = laststride(&q), os = laststride(&out);
    AVX_PARALLEL_FOR_IF(M * H * nkeys * D >= AVX_PAR_MIN)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t s = seq_of_row(cu, S, m);
        const int64_t p = ldi(&pos, offlin(&pos, m));
        /* The row's keys widened, in libref's key order (the window, then the selection); a slot
         * with no key is left out, as every one of libref's sums skips it. */
        float* buf = scratch_f32((size_t)(nkeys * D + 2 * D + nkeys + 16));
        float* keys = buf;
        float* qh = keys + nkeys * D;
        float* acc = qh + D;
        float* sc = acc + D;
        int64_t nk = 0;
        for (int64_t i = 0; i < win; ++i) {
            const int64_t j = p - win + 1 + i;
            if (j < 0 || j / bs >= wbt.shape[1]) continue;
            const int64_t blk = ldi(&wbt, rowoff(&wbt, s, wbt.shape[1]) + (j / bs) * laststride(&wbt));
            if (blk < 0 || blk >= wc.shape[0]) continue;
            widen_flat(wc, ((blk * bs) + j % bs) * D, D, keys + nk * D);
            ++nk;
        }
        for (int64_t t = 0; comp && t < topk; ++t) {
            const int64_t c = ldi(sel, rowoff(sel, m, topk) + t * laststride(sel));
            if (c < 0 || (c * ratio) / cbs >= cbt->shape[1]) continue;
            const int64_t blk =
                ldi(cbt, rowoff(cbt, s, cbt->shape[1]) + ((c * ratio) / cbs) * laststride(cbt));
            if (blk < 0 || blk >= cc->shape[0]) continue;
            widen_flat(*cc, (blk * cbs + (c * ratio) % cbs) * (D / ratio), D, keys + nk * D);
            ++nk;
        }
        for (int64_t h = 0; h < H; ++h) {
            const int64_t qb = rowoff(&q, m, H * D) + h * D * qs;
            const float* qv = in_row(&q, qb, D, qh);
            float mx = -1e30f;
            for (int64_t k = 0; k < nk; ++k) {
                sc[k] = dotf(qv, keys + k * D, D) * scale;
                mx = std::max(mx, sc[k]);
            }
            float den = 0;
            for (int64_t d = 0; d < D; ++d) acc[d] = 0.0f;
            for (int64_t k = 0; k < nk; ++k) {
                const float pk = std::exp(sc[k] - mx);
                den += pk;
                const float* kv = keys + k * D;
                const vf vp = vf_set1(pk);
                int64_t d = 0;
                for (; d + VF_N <= D; d += VF_N)
                    vf_storeu(acc + d, vf_fma(vp, vf_loadu(kv + d), vf_loadu(acc + d)));
                for (; d < D; ++d) acc[d] += pk * kv[d];
            }
            den += std::exp(ldt(&sink, h * laststride(&sink)) - mx);
            const vf vd = vf_set1(den);
            int64_t d = 0;
            for (; d + VF_N <= D; d += VF_N) vf_storeu(acc + d, vf_div(vf_loadu(acc + d), vd));
            for (; d < D; ++d) acc[d] = acc[d] / den;
            out_row(&out, rowoff(&out, m, H * D) + h * D * os, D, acc);
        }
    }
    return RAD_OK;
}
