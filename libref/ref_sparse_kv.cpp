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
#include <vector>

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

namespace {
int64_t seq_of_row(const RadTensor& cu, int64_t n_seq, int64_t row) {
    int64_t lo = 0, hi = n_seq;   /* the s with cu[s] <= row < cu[s+1] */
    while (hi - lo > 1) {
        const int64_t mid = (lo + hi) / 2;
        if (ld_int(cu.dtype, cu.data, offlin(&cu, mid)) <= row) lo = mid;
        else hi = mid;
    }
    return lo;
}
}  // namespace

/* Latent attention with a sink over two paged sources: the window cache's keys at positions
 * [pos - window + 1, pos], and the compressed cache's entries the row selected. A key is its own
 * value (`dim` wide). Window slot of position j: block wbt[s][j / bs] (its cache is
 * [blocks, 1, bs, dim]); compressed entry c: tokens [c ratio, (c+1) ratio), slots of block
 * cbt[s][c ratio / cbs] (cache [blocks, 1, cbs, dim / ratio], cbs a multiple of ratio). A negative
 * block or selection is no key. Per head: f32 scores q.k * scale, max floored at -1e30,
 * p = exp(s - max), out = sum p k / (sum p + exp(sink - max)), rounded once to out's dtype. A row
 * with no key gives zeros. */
int ref_attn_sparse_sink(const RadArgs* a, RadStream) {
    /* Operands 3-5 are optional (null without a compressed source), so count rather than have(). */
    if (!a || a->n_t < 10 || !a->t) return RAD_E_INVAL;
    const RadTensor &q = a->t[0], &wc = a->t[1], &wbt = a->t[2];
    const RadTensor *cc = a->t[3].data ? &a->t[3] : nullptr, *cbt = a->t[4].data ? &a->t[4] : nullptr;
    const RadTensor *sel = a->t[5].data ? &a->t[5] : nullptr;
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
#pragma omp parallel for schedule(dynamic)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t s = seq_of_row(cu, S, m);
        const int64_t p = ld_int(pos.dtype, pos.data, offlin(&pos, m));
        /* Each key's first element as a (tensor, flat offset) pair; -1 when there is no key. */
        std::vector<const RadTensor*> kt((size_t)nkeys, nullptr);
        std::vector<int64_t> ko((size_t)nkeys, -1);
        for (int64_t i = 0; i < win; ++i) {
            const int64_t j = p - win + 1 + i;
            if (j < 0 || j / bs >= wbt.shape[1]) continue;
            const int64_t blk = ld_int(wbt.dtype, wbt.data, rowoff(&wbt, s, wbt.shape[1]) +
                                                               (j / bs) * laststride(&wbt));
            if (blk < 0 || blk >= wc.shape[0]) continue;
            kt[(size_t)i] = &wc;
            ko[(size_t)i] = ((blk * bs) + j % bs) * D;
        }
        for (int64_t t = 0; comp && t < topk; ++t) {
            const int64_t c = ld_int(sel->dtype, sel->data, rowoff(sel, m, topk) + t * laststride(sel));
            if (c < 0 || (c * ratio) / cbs >= cbt->shape[1]) continue;
            const int64_t blk = ld_int(cbt->dtype, cbt->data, rowoff(cbt, s, cbt->shape[1]) +
                                                                  ((c * ratio) / cbs) * laststride(cbt));
            if (blk < 0 || blk >= cc->shape[0]) continue;
            kt[(size_t)(win + t)] = cc;
            ko[(size_t)(win + t)] = (blk * cbs + (c * ratio) % cbs) * (D / ratio);
        }
        std::vector<float> sc((size_t)nkeys), acc((size_t)D);
        for (int64_t h = 0; h < H; ++h) {
            const int64_t qb = rowoff(&q, m, H * D) + h * D * laststride(&q);
            float mx = -1e30f;
            for (int64_t k = 0; k < nkeys; ++k) {
                if (!kt[(size_t)k]) continue;
                float dot = 0;
                for (int64_t d = 0; d < D; ++d)
                    dot += ldt(&q, qb + d * laststride(&q)) * ldt(kt[(size_t)k], ko[(size_t)k] + d);
                sc[(size_t)k] = dot * scale;
                mx = std::max(mx, sc[(size_t)k]);
            }
            float den = 0;
            std::fill(acc.begin(), acc.end(), 0.0f);
            for (int64_t k = 0; k < nkeys; ++k) {
                if (!kt[(size_t)k]) continue;
                const float pk = std::exp(sc[(size_t)k] - mx);
                den += pk;
                for (int64_t d = 0; d < D; ++d) acc[(size_t)d] += pk * ldt(kt[(size_t)k], ko[(size_t)k] + d);
            }
            den += std::exp(ldt(&sink, h * laststride(&sink)) - mx);
            const int64_t ob = rowoff(&out, m, H * D) + h * D * laststride(&out);
            for (int64_t d = 0; d < D; ++d) stt(&out, ob + d * laststride(&out), acc[(size_t)d] / den);
        }
    }
    return RAD_OK;
}

/* The indexer's score of each compressed position for each row: sum over heads h of
 * relu(q_h . k_j) w_h, where k_j is entry j of the paged index cache (addressed as
 * attn_sparse_sink's compressed entries). Rounded as the reference computes it in q's dtype: the
 * dot, then relu(dot) w_h, then the head sum accumulated in f32, each rounded once to the dtype
 * (no rounding for f32). A position the row has not passed, j >= (pos + 1) / ratio, or whose
 * block is unmapped, scores -inf. score is f32 [M, width]. */
int ref_index_score(const RadArgs* a, RadStream) {
    if (!have(a, 7)) return RAD_E_INVAL;
    const RadTensor &q = a->t[0], &w = a->t[1], &kc = a->t[2], &bt = a->t[3];
    const RadTensor &pos = a->t[4], &cu = a->t[5], &sc = a->t[6];
    const int64_t H = p_int(a, "heads", 0), Dm = p_int(a, "dim", 0), ratio = p_int(a, "ratio", 0);
    if (H < 1 || Dm < 1 || ratio < 1 || Dm % ratio || q.rank != 2 || q.shape[0] < 1 ||
        q.shape[1] != H * Dm || w.rank != 2 || w.shape[0] != q.shape[0] || w.shape[1] != H ||
        w.dtype != RAD_F32 || kc.rank != 4 || kc.shape[1] != 1 || kc.shape[3] * ratio != Dm ||
        kc.shape[2] % ratio || bt.rank != 2 || numel(&pos) != q.shape[0] || numel(&cu) < 2 ||
        sc.rank != 2 || sc.shape[0] != q.shape[0] || sc.shape[1] < 1 || sc.dtype != RAD_F32)
        return RAD_E_SHAPE;
    const int64_t M = q.shape[0], C = sc.shape[1], S = numel(&cu) - 1, cbs = kc.shape[2];
    const uint32_t dt = q.dtype;
#pragma omp parallel for schedule(dynamic)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t s = seq_of_row(cu, S, m);
        const int64_t reach = (ld_int(pos.dtype, pos.data, offlin(&pos, m)) + 1) / ratio;
        const int64_t qb = rowoff(&q, m, H * Dm), qs = laststride(&q);
        for (int64_t j = 0; j < C; ++j) {
            float v = -INFINITY;
            const int64_t col = (j * ratio) / cbs;
            if (j < reach && col < bt.shape[1]) {
                const int64_t blk = ld_int(bt.dtype, bt.data, rowoff(&bt, s, bt.shape[1]) + col * laststride(&bt));
                if (blk >= 0 && blk < kc.shape[0]) {
                    const int64_t ko = (blk * cbs + (j * ratio) % cbs) * (Dm / ratio);
                    float acc = 0;
                    for (int64_t h = 0; h < H; ++h) {
                        float dot = 0;
                        for (int64_t d = 0; d < Dm; ++d)
                            dot += ldt(&q, qb + (h * Dm + d) * qs) * ldt(&kc, ko + d);
                        dot = round_dt(dt, dot);
                        acc += round_dt(dt, std::max(dot, 0.0f) *
                                                ldt(&w, rowoff(&w, m, H) + h * laststride(&w)));
                    }
                    v = round_dt(dt, acc);
                }
            }
            stt(&sc, rowoff(&sc, m, C) + j * laststride(&sc), v);
        }
    }
    return RAD_OK;
}

namespace {
/* The k largest of v[0..n) by value, ties to the lower index, as indices (fewer when n < k). */
std::vector<int64_t> top_indices(const std::vector<float>& v, int64_t k) {
    std::vector<int64_t> idx(v.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int64_t)i;
    k = std::min<int64_t>(k, (int64_t)idx.size());
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int64_t x, int64_t y) {
        return v[(size_t)x] > v[(size_t)y] || (v[(size_t)x] == v[(size_t)y] && x < y);
    });
    idx.resize((size_t)k);
    return idx;
}
}  // namespace

/* Level one of a two-level top-k: each `block` positions of a score row scores as its maximum
 * (a short last block padded with -inf), the block holding the row's newest reachable position
 * ((pos + 1) / ratio - 1) is pinned to +inf, and the topk_blocks highest blocks (ties to the lower
 * block) are kept unless they score -inf. keep[m, j] is 1 inside a kept block, else 0. */
int ref_index_candidates(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor &sc = a->t[0], &pos = a->t[1], &keep = a->t[2];
    const int64_t block = p_int(a, "block", 0), kb = p_int(a, "topk_blocks", 0);
    const int64_t ratio = p_int(a, "ratio", 0);
    if (block < 1 || kb < 1 || ratio < 1 || sc.rank != 2 || sc.shape[0] < 1 || sc.shape[1] < 1 ||
        sc.dtype != RAD_F32 || numel(&pos) != sc.shape[0] || keep.rank != 2 ||
        keep.shape[0] != sc.shape[0] || keep.shape[1] != sc.shape[1])
        return RAD_E_SHAPE;
    const int64_t M = sc.shape[0], C = sc.shape[1], nb = (C + block - 1) / block;
#pragma omp parallel for schedule(dynamic)
    for (int64_t m = 0; m < M; ++m) {
        std::vector<float> bs((size_t)nb, -INFINITY);
        for (int64_t j = 0; j < C; ++j)
            bs[(size_t)(j / block)] = std::max(bs[(size_t)(j / block)],
                                               ldt(&sc, rowoff(&sc, m, C) + j * laststride(&sc)));
        const int64_t reach = (ld_int(pos.dtype, pos.data, offlin(&pos, m)) + 1) / ratio;
        if (reach >= 1 && (reach - 1) / block < nb) bs[(size_t)((reach - 1) / block)] = INFINITY;
        std::vector<char> kept((size_t)nb, 0);
        for (int64_t b : top_indices(bs, kb)) kept[(size_t)b] = bs[(size_t)b] > -INFINITY;
        for (int64_t j = 0; j < C; ++j)
            st_int(keep.dtype, keep.data, rowoff(&keep, m, C) + j * laststride(&keep),
                   kept[(size_t)(j / block)] ? 1 : 0);
    }
    return RAD_OK;
}

/* The topk highest-scoring positions a row has passed (j < (pos + 1) / ratio), inside the kept
 * positions when `keep` is given, never a -inf score; ties to the lower position; written in
 * ascending position order, then -1 to topk. */
int ref_index_topk(const RadArgs* a, RadStream) {
    if (!a || a->n_t < 4 || !a->t) return RAD_E_INVAL;
    const RadTensor &sc = a->t[0], &pos = a->t[2], &sel = a->t[3];
    const RadTensor* keep = a->t[1].data ? &a->t[1] : nullptr;
    const int64_t topk = p_int(a, "topk", 0), ratio = p_int(a, "ratio", 0);
    if (topk < 1 || ratio < 1 || !sc.data || !pos.data || !sel.data || sc.rank != 2 ||
        sc.shape[0] < 1 || sc.dtype != RAD_F32 || numel(&pos) != sc.shape[0] || sel.rank != 2 ||
        sel.shape[0] != sc.shape[0] || sel.shape[1] != topk ||
        (keep && (keep->rank != 2 || keep->shape[0] != sc.shape[0] || keep->shape[1] != sc.shape[1])))
        return RAD_E_SHAPE;
    const int64_t M = sc.shape[0], C = sc.shape[1];
#pragma omp parallel for schedule(dynamic)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t reach = std::min<int64_t>(C, (ld_int(pos.dtype, pos.data, offlin(&pos, m)) + 1) / ratio);
        std::vector<float> v((size_t)std::max<int64_t>(reach, 0), -INFINITY);
        for (int64_t j = 0; j < reach; ++j)
            if (!keep || ld_int(keep->dtype, keep->data, rowoff(keep, m, C) + j * laststride(keep)))
                v[(size_t)j] = ldt(&sc, rowoff(&sc, m, C) + j * laststride(&sc));
        std::vector<int64_t> pick;
        for (int64_t j : top_indices(v, topk))
            if (v[(size_t)j] > -INFINITY) pick.push_back(j);
        std::sort(pick.begin(), pick.end());
        for (int64_t t = 0; t < topk; ++t)
            st_int(sel.dtype, sel.data, rowoff(&sel, m, topk) + t * laststride(&sel),
                   t < (int64_t)pick.size() ? pick[(size_t)t] : -1);
    }
    return RAD_OK;
}

/* One entry a block: x[w] (n values) into the whole of block page[w] of a paged cache
 * [blocks, kv_heads, block_size, slot] with kv_heads * block_size * slot == n, rounded once to the
 * cache's dtype. A negative page writes nothing. The pooled compressed entries qsa_work pages
 * (one block of `ratio` slots each) are stored this way; `block_size` is a parameter so that the
 * op, declared right after its group, sets the group's block size (spec section 7.2). */
int ref_entry_store(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor &x = a->t[0], &page = a->t[1], &cache = a->t[2];
    const int64_t n = p_int(a, "n", 0), bs = p_int(a, "block_size", 0);
    if (n < 1 || bs < 1 || x.rank != 2 || x.shape[0] < 1 || x.shape[1] != n ||
        numel(&page) < x.shape[0] || cache.rank != 4 || cache.shape[2] != bs ||
        cache.shape[1] * cache.shape[2] * cache.shape[3] != n)
        return RAD_E_SHAPE;
    for (int64_t w = 0; w < x.shape[0]; ++w) {
        const int64_t pg = ld_int(page.dtype, page.data, offlin(&page, w));
        if (pg < 0) continue;
        if (pg >= cache.shape[0]) return RAD_E_INVAL;
        const int64_t xb = rowoff(&x, w, n), xs = laststride(&x);
        for (int64_t i = 0; i < n; ++i) stt(&cache, pg * n + i, ldt(&x, xb + i * xs));
    }
    return RAD_OK;
}
