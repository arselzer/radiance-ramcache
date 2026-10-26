// R4D -- DeepSeek-V4.1's sparse ops. Each kernel rounds where its libref function does
// (libref/ref_sparse_kv.cpp) and sums in libref's order, so the outputs are libref's bits:
//
//   index_score       one thread an entry (or HS waves an entry group, each a slice of the heads,
//                     the head sum then closed in head order from LDS): the dot in d order,
//                     rounded to bf16, relu x the rounded head weight, rounded, the head sum in head
//                     order, rounded.
//   index_candidates  one 1024-thread block a row: block maxima, the newest block pinned, an
//   index_topk        exact radix select of the k largest (value desc, position asc: libref's
//                     partial_sort order), then an ordered compaction. index_topk at decode
//                     heights over long rows splits a row into chunks (each keeps its exact top-k,
//                     which holds every global winner) and merges them in a second launch.
//   attn_sparse_sink  one workgroup a (d half, head group, row): every score q.k in d order (keys
//                     transposed through a wave-private LDS tile so the loads stay coalesced), the
//                     max, p = exp(s - max) by glibc's expf algorithm (the same double table and
//                     polynomial, so p is std::exp's float), sum p and sum p k in key order, then
//                     exp(sink - max), the division and one bf16 rounding.
//
// q.k and q_h.k_j multiply two bf16 values, exact in f32, so a fused multiply-add rounds exactly
// where libref's multiply-then-add does; p k is not exact and stays an unfused multiply and add.
//
// Prefill heights (more than kWmRows = 48 rows a pass) take WMMA forms instead, which are NOT
// libref's bits (kbench tolerance): attn_sparse_sink with an online softmax, v_exp_f32 and P as
// bf16 hi + lo; index_score with only the dot's summation order changed (every rounding after it
// is libref's). The decode band keeps the exact kernels, so speculative verify reproduces decode;
// R4D_ATTN_WM_OFF=1 / R4D_IDX_WM_OFF=1 restore the exact kernels at every height.
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_common.h"
#include "r4d_emit.h"
#include "r4d_plugin.h"

#include <climits>
#include <cstdlib>
#include <cmath>
#include <cstdint>

extern __shared__ __align__(16) unsigned char r4d_ds41_sparse_dyn[];

namespace {

constexpr float kDsInf = __builtin_huge_valf();

/* ================================================================== scalar helpers */

/* rad_f32_to_bf16 (abi/rad_plugin.h): NaN quieted, everything else to nearest even. */
__device__ __forceinline__ uint32_t ds_bf16_bits(float f) {
    const uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return ((u >> 16) | 0x0040u) & 0xffffu;
    return ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16) & 0xffffu;
}
__device__ __forceinline__ float ds_round_bf16(float f) { return __uint_as_float(ds_bf16_bits(f) << 16); }
__device__ __forceinline__ float ds_bf(uint32_t h) { return __uint_as_float((h & 0xffffu) << 16); }

/* libref's seq_of_row: the s with cu[s] <= row < cu[s+1], by bisection. */
__device__ __forceinline__ int ds_seq_of_row(const int* cu, long cs, int S, long row) {
    int lo = 0, hi = S;
    while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        if ((long)cu[(long)mid * cs] <= row) lo = mid;
        else hi = mid;
    }
    return lo;
}

/* glibc's expf (sysdeps/ieee754/flt-32/e_expf.c, EXP2F_TABLE_BITS 5): 2^(k/32) from a table and a
 * cubic for 2^(r/32), in double, rounded once to float. A host replica of exactly these steps
 * matched libm's expf on 685k arguments (subnormal and overflow edges included). */
__constant__ unsigned long long kDsExp2fTab[32] = {
    0x3ff0000000000000ull, 0x3fefd9b0d3158574ull, 0x3fefb5586cf9890full, 0x3fef9301d0125b51ull,
    0x3fef72b83c7d517bull, 0x3fef54873168b9aaull, 0x3fef387a6e756238ull, 0x3fef1e9df51fdee1ull,
    0x3fef06fe0a31b715ull, 0x3feef1a7373aa9cbull, 0x3feedea64c123422ull, 0x3feece086061892dull,
    0x3feebfdad5362a27ull, 0x3feeb42b569d4f82ull, 0x3feeab07dd485429ull, 0x3feea47eb03a5585ull,
    0x3feea09e667f3bcdull, 0x3fee9f75e8ec5f74ull, 0x3feea11473eb0187ull, 0x3feea589994cce13ull,
    0x3feeace5422aa0dbull, 0x3feeb737b0cdc5e5ull, 0x3feec49182a3f090ull, 0x3feed503b23e255dull,
    0x3feee89f995ad3adull, 0x3feeff76f2fb5e47ull, 0x3fef199bdd85529cull, 0x3fef3720dcef9069ull,
    0x3fef5818dcfba487ull, 0x3fef7c97337b9b5full, 0x3fefa4afa2a490daull, 0x3fefd0765b6e4540ull,
};
__device__ __forceinline__ float ds_expf(float x) {
    const uint32_t ux = __float_as_uint(x);
    const uint32_t abstop = (ux >> 20) & 0x7ffu;
    if (abstop >= 0x42bu) {                          /* |x| >= 88, or inf/nan */
        if (ux == 0xff800000u) return 0.0f;
        if (abstop >= 0x7f8u) return x + x;
        if (x > 0x1.62e42ep6f) return kDsInf;
        if (x < -0x1.9fe368p6f) return 0.0f;
    }
    const double xd = (double)x;
    double z = 0x1.71547652b82fep+5 * xd;            /* x * 32 / ln 2 */
    double kd = z + 0x1.8p+52;
    const unsigned long long ki = (unsigned long long)__double_as_longlong(kd);
    kd -= 0x1.8p+52;
    const double r = z - kd;
    unsigned long long t = kDsExp2fTab[ki % 32];
    t += ki << 47;
    const double s = __longlong_as_double((long long)t);
    z = 0x1.c6af84b912394p-20 * r + 0x1.ebfce50fac4f3p-13;
    const double r2 = r * r;
    double y = 0x1.62e42ff0c52d6p-6 * r + 1.0;
    y = z * r2 + y;
    y = y * s;
    return (float)y;
}

/* An order-preserving image of a float with -0 folded onto +0 (libref's comparator calls them
 * equal). A candidate is > -inf, so its key is > 0x007fffff; 0 marks "not a candidate". */
__device__ __forceinline__ uint32_t ds_key(float v) {
    const uint32_t u = __float_as_uint(v + 0.0f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ void ds_wave_sync() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

/* ================================================================== index_score */
struct DsScoreArgs {
    const uint16_t* q;
    const uint16_t* w;
    const uint16_t* kc;
    const int* bt;
    const int* pos;
    const int* cu;
    float* sc;
    long qr, wr, wc, btr, btc, nblk, ps, cs, sr, scs;
    int nbt, S, M, C, H, ratio, cbs, slot;
    float wscale;
};

/* One thread an entry; HS waves of a block share 32 entries, each wave a slice of the heads (its
 * q then uniform, read through scalar loads). A thread holds its key as 128 floats. */
template <int HS>
__global__ __launch_bounds__(256) void ds41_index_score_kernel(DsScoreArgs a) {
    constexpr int DM = 128, EPB = 256 / HS;
    float* part = (float*)r4d_ds41_sparse_dyn;       /* [H][EPB], HS > 1 */
    /* The wave index through readfirstlane, so a wave's head slice (and its q) is provably uniform
     * and q stays in scalar loads. */
    const int lane = threadIdx.x & 31, wave = __builtin_amdgcn_readfirstlane((int)(threadIdx.x >> 5));
    const int grp = wave / HS, sub = wave % HS;
    const int e = grp * 32 + lane;
    const long jb = (long)blockIdx.x * EPB, j = jb + e;
    const int hp = (a.H + HS - 1) / HS, h0 = sub * hp, h1 = min(a.H, h0 + hp);
    for (long m = blockIdx.y; m < a.M; m += gridDim.y) {
        float* srow = a.sc + m * a.sr;
        const long reach = ((long)a.pos[m * a.ps] + 1) / a.ratio;
        if (jb >= reach) {                           /* block-uniform: no key in this block */
            if (sub == 0 && j < a.C) srow[j * a.scs] = -kDsInf;
            continue;
        }
        const int s = ds_seq_of_row(a.cu, a.cs, a.S, m);
        const uint16_t* kp = nullptr;
        if (j < a.C && j < reach) {
            const long col = (j * a.ratio) / a.cbs;
            if (col < a.nbt) {
                const long blk = a.bt[(long)s * a.btr + col * a.btc];
                if (blk >= 0 && blk < a.nblk)
                    kp = a.kc + (blk * a.cbs + (j * a.ratio) % a.cbs) * (long)a.slot;
            }
        }
        float acc = 0.0f;
        if (kp) {
            float k[DM];
#pragma unroll
            for (int c = 0; c < DM / 8; ++c) {
                const uint4 v = *(const uint4*)(kp + c * 8);
                k[c * 8 + 0] = bf16w_lo(v.x); k[c * 8 + 1] = bf16w_hi(v.x);
                k[c * 8 + 2] = bf16w_lo(v.y); k[c * 8 + 3] = bf16w_hi(v.y);
                k[c * 8 + 4] = bf16w_lo(v.z); k[c * 8 + 5] = bf16w_hi(v.z);
                k[c * 8 + 6] = bf16w_lo(v.w); k[c * 8 + 7] = bf16w_hi(v.w);
            }
            const uint32_t* qrow = (const uint32_t*)(a.q + m * a.qr);
            const uint16_t* wrow = a.w + m * a.wr;
            for (int h = h0; h < h1; h += 4) {
                const int nh = min(4, h1 - h);
                const uint32_t* q0 = qrow + (long)h * (DM / 2);
                const uint32_t* q1 = qrow + (long)(nh > 1 ? h + 1 : h) * (DM / 2);
                const uint32_t* q2 = qrow + (long)(nh > 2 ? h + 2 : h) * (DM / 2);
                const uint32_t* q3 = qrow + (long)(nh > 3 ? h + 3 : h) * (DM / 2);
                float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
#pragma unroll
                for (int i = 0; i < DM / 2; ++i) {
                    const uint32_t w0 = q0[i], w1 = q1[i], w2 = q2[i], w3 = q3[i];
                    d0 = __builtin_fmaf(bf16w_lo(w0), k[2 * i], d0);
                    d1 = __builtin_fmaf(bf16w_lo(w1), k[2 * i], d1);
                    d2 = __builtin_fmaf(bf16w_lo(w2), k[2 * i], d2);
                    d3 = __builtin_fmaf(bf16w_lo(w3), k[2 * i], d3);
                    d0 = __builtin_fmaf(bf16w_hi(w0), k[2 * i + 1], d0);
                    d1 = __builtin_fmaf(bf16w_hi(w1), k[2 * i + 1], d1);
                    d2 = __builtin_fmaf(bf16w_hi(w2), k[2 * i + 1], d2);
                    d3 = __builtin_fmaf(bf16w_hi(w3), k[2 * i + 1], d3);
                }
                const float dd[4] = { d0, d1, d2, d3 };
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    if (u >= nh) break;
                    const int hh = h + u;
                    const float dot = ds_round_bf16(dd[u]);
                    const float wh = ds_round_bf16(ds_bf(wrow[(long)hh * a.wc]) * a.wscale);
                    const float c = ds_round_bf16((dot < 0.0f ? 0.0f : dot) * wh);
                    if (HS == 1) acc = acc + c;
                    else part[hh * EPB + e] = c;
                }
            }
        }
        if (HS > 1) {
            __syncthreads();
            if (sub == 0 && kp) {
                acc = 0.0f;
                for (int hh = 0; hh < a.H; ++hh) acc = acc + part[hh * EPB + e];
            }
            __syncthreads();
        }
        if (sub == 0 && j < a.C) srow[j * a.scs] = kp ? ds_round_bf16(acc) : -kDsInf;
    }
}

/* ================================================================== exact top-k by radix select
 *
 * One block of 1024 threads selects, among n keys (0 = not a candidate), the K largest under (key
 * desc, index asc). Three histogram passes (11, 11, 10 bits) find the K-th key T and how many keys
 * equal to T are taken (`need`, the lowest-indexed ones); an emit pass then walks the keys in index
 * order with a block scan, so each taken key knows its rank. A thread folds runs of one digit into
 * a single LDS add, which keeps the first pass (most keys share their top bits) off the atomic
 * unit. Up to kDsCache keys stay in LDS after the first pass. */
constexpr int kDsSelThreads = 1024;
constexpr int kDsCache = 8192;
constexpr int kDsCacheUse = kDsCache;               /* rows up to this many keys run from LDS */

struct DsSelShared {
    uint32_t hist[2048];
    uint32_t ws[33];
    uint32_t prefix, need, total, all, digit;
};

__device__ __forceinline__ uint32_t ds_block_excl_scan(uint32_t v, uint32_t* ws, uint32_t* total) {
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const uint32_t inc = r4d_wave_prefix_u32(v, lane);
    if (lane == 31) ws[wave] = inc;
    __syncthreads();
    if (wave == 0) {
        const uint32_t w = ws[lane];
        const uint32_t wi = r4d_wave_prefix_u32(w, lane);
        ws[lane] = wi - w;
        if (lane == 31) ws[32] = wi;
    }
    __syncthreads();
    const uint32_t r = ws[wave] + inc - v;
    *total = ws[32];
    __syncthreads();
    return r;
}

/* The digit (from the top) whose cumulative count reaches `need`; updates need and total. */
__device__ __forceinline__ void ds_find_digit(DsSelShared* s, int nbins) {
    const int t = threadIdx.x;
    const uint32_t need = s->need;
    const int r0 = 2 * t, r1 = 2 * t + 1;
    const uint32_t c0 = r0 < nbins ? s->hist[nbins - 1 - r0] : 0u;
    const uint32_t c1 = r1 < nbins ? s->hist[nbins - 1 - r1] : 0u;
    uint32_t total;
    const uint32_t above = ds_block_excl_scan(c0 + c1, s->ws, &total);
    if (c0 && above < need && need <= above + c0) {
        s->digit = (uint32_t)(nbins - 1 - r0);
        s->need = need - above;
    } else if (c1 && above + c0 < need && need <= above + c0 + c1) {
        s->digit = (uint32_t)(nbins - 1 - r1);
        s->need = need - above - c0;
    }
    if (t == 0) s->total = total;
    __syncthreads();
}

template <class Get>
__device__ void ds_select(const Get& get, uint32_t n, uint32_t K, uint32_t* cache, DsSelShared* s) {
    const int t = threadIdx.x;
    for (int i = t; i < 2048; i += kDsSelThreads) s->hist[i] = 0u;
    if (t == 0) { s->need = K; s->prefix = 0u; s->all = 0u; }
    __syncthreads();
    {
        uint32_t cur = ~0u, cnt = 0u;
        for (uint32_t i = t; i < n; i += kDsSelThreads) {
            const uint32_t key = get(i);
            if (cache) cache[i] = key;
            if (key) {
                const uint32_t d = key >> 21;
                if (d == cur) ++cnt;
                else {
                    if (cnt) atomicAdd(&s->hist[cur], cnt);
                    cur = d;
                    cnt = 1u;
                }
            }
        }
        if (cnt) atomicAdd(&s->hist[cur], cnt);
    }
    __syncthreads();
    ds_find_digit(s, 2048);
    if (s->total <= K) {                             /* every candidate is taken */
        __syncthreads();
        if (t == 0) s->all = 1u;
        __syncthreads();
        return;
    }
    uint32_t prefix = s->digit;
    for (int pass = 1; pass < 3; ++pass) {
        const int shift = pass == 1 ? 10 : 0, nbits = pass == 1 ? 11 : 10, hi = shift + nbits;
        const uint32_t mask = (1u << nbits) - 1u;
        __syncthreads();
        for (int i = t; i < 2048; i += kDsSelThreads) s->hist[i] = 0u;
        __syncthreads();
        uint32_t cur = ~0u, cnt = 0u;
        for (uint32_t i = t; i < n; i += kDsSelThreads) {
            const uint32_t key = cache ? cache[i] : get(i);
            if (key && (key >> hi) == prefix) {
                const uint32_t d = (key >> shift) & mask;
                if (d == cur) ++cnt;
                else {
                    if (cnt) atomicAdd(&s->hist[cur], cnt);
                    cur = d;
                    cnt = 1u;
                }
            }
        }
        if (cnt) atomicAdd(&s->hist[cur], cnt);
        __syncthreads();
        ds_find_digit(s, 1 << nbits);
        prefix = (prefix << nbits) | s->digit;
    }
    __syncthreads();
    if (t == 0) s->prefix = prefix;
    __syncthreads();
}

/* Walks the keys in index order: emit(i, take, out) for every i < n, `out` the taken key's rank in
 * index order; then finish(count taken). */
template <class Get, class Emit, class Finish>
__device__ void ds_emit(const Get& get, uint32_t n, const uint32_t* cache, DsSelShared* s,
                        const Emit& emit, const Finish& finish) {
    const bool all = s->all != 0u;
    const uint32_t T = all ? 0u : s->prefix, need = all ? 0u : s->need;
    uint32_t base_gt = 0u, base_eq = 0u;
    for (uint32_t i0 = 0; i0 < n; i0 += kDsSelThreads) {
        const uint32_t i = i0 + threadIdx.x;
        const uint32_t key = i < n ? (cache ? cache[i] : get(i)) : 0u;
        const uint32_t gt = key > T ? 1u : 0u;
        const uint32_t eq = (T != 0u && key == T) ? 1u : 0u;
        uint32_t tot;
        const uint32_t ex = ds_block_excl_scan(gt | (eq << 16), s->ws, &tot);
        const uint32_t eq_rank = base_eq + (ex >> 16);
        const bool take = gt || (eq && eq_rank < need);
        const uint32_t out = base_gt + (ex & 0xffffu) + min(eq_rank, need);
        if (i < n) emit(i, take, out);
        base_gt += tot & 0xffffu;
        base_eq += tot >> 16;
    }
    finish(base_gt + min(base_eq, need));
}

/* ------------------------------------------------------------------ index_topk */
struct DsTopkArgs {
    const float* sc;
    const int* keep;
    const int* pos;
    int* sel;
    long sr, scs, kr, kcs, ps, selr, selc;
    int M, C, K, ratio;
};

__device__ __forceinline__ long ds_topk_reach(const DsTopkArgs& a, long m) {
    long reach = ((long)a.pos[m * a.ps] + 1) / a.ratio;
    if (reach > a.C) reach = a.C;
    return reach < 0 ? 0 : reach;
}

/* The key of position j0 + i of a row: not a candidate outside keep or at -inf (or NaN). */
struct DsTopkKey {
    const float* row;
    const int* krow;
    long scs, kcs, j0;
    __device__ __forceinline__ uint32_t operator()(uint32_t i) const {
        const long j = j0 + (long)i;
        if (krow && !krow[j * kcs]) return 0u;
        const float v = row[j * scs];
        return v > -kDsInf ? ds_key(v) : 0u;
    }
};

__global__ __launch_bounds__(kDsSelThreads) void ds41_index_topk_kernel(DsTopkArgs a) {
    __shared__ DsSelShared s;
    __shared__ uint32_t cache[kDsCache];
    for (long m = blockIdx.x; m < a.M; m += gridDim.x) {
        const uint32_t n = (uint32_t)ds_topk_reach(a, m);
        const DsTopkKey get{ a.sc + m * a.sr, a.keep ? a.keep + m * a.kr : nullptr, a.scs, a.kcs, 0 };
        uint32_t* c = n <= (uint32_t)kDsCacheUse ? cache : nullptr;
        ds_select(get, n, (uint32_t)a.K, c, &s);
        int* srow = a.sel + m * a.selr;
        const long selc = a.selc;
        const int K = a.K;
        ds_emit(get, n, c, &s,
                [&](uint32_t i, bool take, uint32_t out) { if (take) srow[(long)out * selc] = (int)i; },
                [&](uint32_t tot) {
                    for (long t = (long)tot + threadIdx.x; t < K; t += kDsSelThreads) srow[t * selc] = -1;
                });
        __syncthreads();
    }
}

/* The decode form over long rows. Stage 1: each (chunk, row) keeps its chunk's exact top-K as
 * (key << 32 | position), in position order, and its count. A row's winner is among its chunk's
 * K best, so stage 2's top-K over the chunks in order (position order again) is the row's. */
struct DsSplitArgs {
    DsTopkArgs t;
    unsigned long long* cand;
    int* cnt;
    int nch, chunk;
};

__global__ __launch_bounds__(kDsSelThreads) void ds41_index_topk_split1_kernel(DsSplitArgs a) {
    __shared__ DsSelShared s;
    __shared__ uint32_t cache[kDsCache];
    const int c = blockIdx.x;
    const long m = blockIdx.y;
    const long j0 = (long)c * a.chunk;
    long n = ds_topk_reach(a.t, m) - j0;
    n = n < 0 ? 0 : (n > a.chunk ? a.chunk : n);
    const DsTopkKey get{ a.t.sc + m * a.t.sr, a.t.keep ? a.t.keep + m * a.t.kr : nullptr, a.t.scs,
                         a.t.kcs, j0 };
    ds_select(get, (uint32_t)n, (uint32_t)a.t.K, cache, &s);
    unsigned long long* out = a.cand + (m * a.nch + c) * (long)a.t.K;
    int* cnt = a.cnt + m * a.nch + c;
    ds_emit(get, (uint32_t)n, cache, &s,
            [&](uint32_t i, bool take, uint32_t o) {
                if (take)
                    out[o] = ((unsigned long long)cache[i] << 32) |
                             (unsigned long long)(uint32_t)(j0 + (long)i);
            },
            [&](uint32_t tot) { if (threadIdx.x == 0) *cnt = (int)tot; });
}

struct DsCandKey {
    const unsigned long long* cr;
    const int* cn;
    uint32_t K;
    __device__ __forceinline__ uint32_t operator()(uint32_t i) const {
        return (int)(i % K) < cn[i / K] ? (uint32_t)(cr[i] >> 32) : 0u;
    }
};

__global__ __launch_bounds__(kDsSelThreads) void ds41_index_topk_split2_kernel(DsSplitArgs a) {
    __shared__ DsSelShared s;
    __shared__ uint32_t cache[kDsCache];
    for (long m = blockIdx.x; m < a.t.M; m += gridDim.x) {
        const unsigned long long* cr = a.cand + m * a.nch * (long)a.t.K;
        const uint32_t n = (uint32_t)a.nch * (uint32_t)a.t.K;
        const DsCandKey get{ cr, a.cnt + m * a.nch, (uint32_t)a.t.K };
        uint32_t* c = n <= (uint32_t)kDsCacheUse ? cache : nullptr;
        ds_select(get, n, (uint32_t)a.t.K, c, &s);
        int* srow = a.t.sel + m * a.t.selr;
        const long selc = a.t.selc;
        const int K = a.t.K;
        ds_emit(get, n, c, &s,
                [&](uint32_t i, bool take, uint32_t out) {
                    if (take) srow[(long)out * selc] = (int)(uint32_t)(cr[i] & 0xffffffffull);
                },
                [&](uint32_t tot) {
                    for (long t = (long)tot + threadIdx.x; t < K; t += kDsSelThreads) srow[t * selc] = -1;
                });
        __syncthreads();
    }
}

/* ------------------------------------------------------------------ index_candidates */
struct DsCandArgs {
    const float* sc;
    const int* pos;
    int* keep;
    long sr, scs, ps, kr, kcs;
    int M, C, block, KB, ratio;
};

/* Block b's score: the max of its positions (std::max(cur, v): a NaN never replaces), +inf when
 * pinned. A -inf block is never kept, so it is not a candidate. */
struct DsBlockKey {
    const float* row;
    long scs, C, block, pin;
    __device__ __forceinline__ uint32_t operator()(uint32_t b) const {
        float bs = -kDsInf;
        const long j0 = (long)b * block;
        const long j1 = j0 + block < C ? j0 + block : C;
        for (long j = j0; j < j1; ++j) {
            const float v = row[j * scs];
            bs = bs < v ? v : bs;
        }
        if ((long)b == pin) bs = kDsInf;
        return bs > -kDsInf ? ds_key(bs) : 0u;
    }
};

__global__ __launch_bounds__(kDsSelThreads) void ds41_index_candidates_kernel(DsCandArgs a) {
    __shared__ DsSelShared s;
    __shared__ uint32_t cache[kDsCache];
    const long nb = ((long)a.C + a.block - 1) / a.block;
    for (long m = blockIdx.x; m < a.M; m += gridDim.x) {
        const long reach = ((long)a.pos[m * a.ps] + 1) / a.ratio;
        const long pin = (reach >= 1 && (reach - 1) / a.block < nb) ? (reach - 1) / a.block : -1;
        const DsBlockKey get{ a.sc + m * a.sr, a.scs, a.C, a.block, pin };
        uint32_t* c = nb <= kDsCacheUse ? cache : nullptr;
        ds_select(get, (uint32_t)nb, (uint32_t)a.KB, c, &s);
        int* krow = a.keep + m * a.kr;
        const long kcs = a.kcs, C = a.C, block = a.block;
        ds_emit(get, (uint32_t)nb, c, &s,
                [&](uint32_t b, bool take, uint32_t) {
                    const long j0 = (long)b * block, j1 = j0 + block < C ? j0 + block : C;
                    for (long j = j0; j < j1; ++j) krow[j * kcs] = take ? 1 : 0;
                },
                [&](uint32_t) {});
        __syncthreads();
    }
}

/* ================================================================== attn_sparse_sink */
struct DsAttnArgs {
    const uint16_t* q;
    const uint16_t* wc;
    const int* wbt;
    const int* wused;
    const uint16_t* cc;
    const int* cbt;
    const int* sel;
    const int* pos;
    const int* cu;
    const float* sink;
    uint16_t* out;
    long qr, outr, wbtr, wbtc, nwblk, cbtr, cbtc, ncblk, selr, selc, wus, ps, cs, sks;
    int nwbt, bs, ncbt, cbs, ratio, S, M, H, win, nk, nkp;
    float scale;
};

constexpr int kDsAttnD = 512;
constexpr int kDsTileRow = 80;                       /* 32 bf16 of a key + 16 bytes of padding */
constexpr int kDsTileBytes = 8 * 32 * kDsTileRow;    /* one 32-key tile a wave */

__host__ __device__ constexpr long ds_attn_tile_off(int nkp, int hg) {
    return ((long)nkp * 8 + (long)hg * nkp * 4 + (long)hg * 8 + 15) / 16 * 16;
}

/* Workgroup (d half, head group, row). LDS: the row's key pointers (null = no key), scores then p
 * [HG][nkp], max and den [HG], and the waves' transpose tiles. */
template <int HG, int DS>
__global__ __launch_bounds__(256) void ds41_attn_sparse_sink_kernel(DsAttnArgs a) {
    constexpr int D = kDsAttnD, NW = 8, DPT = D / DS / 256;
    static_assert(DPT == 1 || DPT == 2, "one or two columns a thread");
    const uint16_t** kp = (const uint16_t**)r4d_ds41_sparse_dyn;
    float* sp = (float*)(r4d_ds41_sparse_dyn + (long)a.nkp * 8);
    float* smx = sp + (long)HG * a.nkp;
    float* sden = smx + HG;
    unsigned char* tile = r4d_ds41_sparse_dyn + ds_attn_tile_off(a.nkp, HG);
    const int tid = threadIdx.x, lane = tid & 31;
    const int wave = __builtin_amdgcn_readfirstlane((int)(tid >> 5));
    const int hg0 = blockIdx.y * HG;
    const int hmax = min(HG, a.H - hg0);
    const int dcol = blockIdx.x * (D / DS) + tid * DPT;
    for (long m = blockIdx.z; m < a.M; m += gridDim.z) {
        /* ---- the row's keys, in libref's order: the window, then the selections */
        const int s = ds_seq_of_row(a.cu, a.cs, a.S, m);
        const long p = a.pos[m * a.ps];
        long lr = (long)a.cu[(long)(s + 1) * a.cs] - 1;
        lr = lr < 0 ? 0 : (lr >= a.M ? a.M - 1 : lr);
        const long last = a.pos[lr * a.ps];
        const long first = last + 1 - (long)a.wused[(long)s * a.wus];
        for (int k = tid; k < a.nkp; k += 256) {
            const uint16_t* ptr = nullptr;
            if (k < a.win) {
                const long j = p - a.win + 1 + k, r = j - first;
                if (j >= 0 && r >= 0 && r / a.bs < a.nwbt) {
                    const long blk = a.wbt[(long)s * a.wbtr + (r / a.bs) * a.wbtc];
                    if (blk >= 0 && blk < a.nwblk) ptr = a.wc + (blk * a.bs + r % a.bs) * (long)D;
                }
            } else if (k < a.nk) {
                const long c = a.sel[m * a.selr + (long)(k - a.win) * a.selc];
                if (c >= 0 && (c * a.ratio) / a.cbs < a.ncbt) {
                    const long blk = a.cbt[(long)s * a.cbtr + ((c * a.ratio) / a.cbs) * a.cbtc];
                    if (blk >= 0 && blk < a.ncblk)
                        ptr = a.cc + (blk * a.cbs + (c * a.ratio) % a.cbs) * (long)(D / a.ratio);
                }
            }
            kp[k] = ptr;
        }
        __syncthreads();

        /* ---- scores: a lane a key, each dot in d order. A wave loads its 32 keys 32 columns at a
         * time (four lanes a key, 16 bytes each), transposes them through its tile and reads its own
         * key's row back; q is uniform, so it comes through scalar loads. */
        {
            const uint32_t* qrow = (const uint32_t*)(a.q + m * a.qr) + (long)hg0 * (D / 2);
            unsigned char* tb = tile + wave * (32 * kDsTileRow);
            for (int kb = wave * 32; kb < a.nk; kb += NW * 32) {
                const uint16_t* lp[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int kk = kb + i * 8 + (lane >> 2);
                    lp[i] = kk < a.nk ? kp[kk] : nullptr;
                }
                float acc[HG];
#pragma unroll
                for (int h = 0; h < HG; ++h) acc[h] = 0.0f;
                for (int c = 0; c < D / 32; ++c) {
                    uint4 v[4];
#pragma unroll
                    for (int i = 0; i < 4; ++i)
                        v[i] = lp[i] ? *(const uint4*)(lp[i] + c * 32 + (lane & 3) * 8) : make_uint4(0, 0, 0, 0);
#pragma unroll
                    for (int i = 0; i < 4; ++i)
                        *(uint4*)(tb + (i * 8 + (lane >> 2)) * kDsTileRow + (lane & 3) * 16) = v[i];
                    ds_wave_sync();
                    /* Eight columns at a time keeps HG x 4 q words live, within the scalar file. */
#pragma unroll 1
                    for (int i = 0; i < 4; ++i) {
                        const uint4 ri = *(const uint4*)(tb + lane * kDsTileRow + i * 16);
                        const uint32_t wk[4] = { ri.x, ri.y, ri.z, ri.w };
#pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            const float k0 = bf16w_lo(wk[e]), k1 = bf16w_hi(wk[e]);
                            const int qi = c * 16 + i * 4 + e;
#pragma unroll
                            for (int h = 0; h < HG; ++h) {
                                const uint32_t qw = qrow[(long)(h < hmax ? h : 0) * (D / 2) + qi];
                                acc[h] = __builtin_fmaf(bf16w_lo(qw), k0, acc[h]);
                                acc[h] = __builtin_fmaf(bf16w_hi(qw), k1, acc[h]);
                            }
                        }
                    }
                    ds_wave_sync();                  /* the tile's reads before the next chunk's writes */
                }
                const int k = kb + lane;
                if (k < a.nk) {
                    const bool have = kp[k] != nullptr;
#pragma unroll
                    for (int h = 0; h < HG; ++h) sp[(long)h * a.nkp + k] = have ? acc[h] * a.scale : -kDsInf;
                }
            }
        }
        __syncthreads();

        /* ---- per head (one wave each): max floored at -1e30, then p in place (0 for no key) */
        if (wave < HG) {
            float* sh = sp + (long)wave * a.nkp;
            float mx = -1e30f;
            for (int k = lane; k < a.nk; k += 32) {
                const float v = sh[k];
                mx = mx < v ? v : mx;
            }
            mx = r4d_wave_max(mx);
            for (int k = lane; k < a.nkp; k += 32)
                sh[k] = (k < a.nk && kp[k]) ? ds_expf(sh[k] - mx) : 0.0f;
            if (lane == 0) smx[wave] = mx;
        }
        __syncthreads();

        /* ---- sum p k and sum p, key by key in libref's order. A key with no entry adds p = 0 times
         * k = 0, which leaves both sums as they were (they start at +0 and never reach -0). */
        float acc[HG][DPT], den[HG];
#pragma unroll
        for (int h = 0; h < HG; ++h) {
            den[h] = 0.0f;
#pragma unroll
            for (int d = 0; d < DPT; ++d) acc[h][d] = 0.0f;
        }
        for (int k = 0; k < a.nkp; k += 4) {
            float kv[4][DPT];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const uint16_t* ptr = kp[k + u];
                if constexpr (DPT == 2) {
                    const uint32_t w = ptr ? *(const uint32_t*)(ptr + dcol) : 0u;
                    kv[u][0] = bf16w_lo(w);
                    kv[u][DPT - 1] = bf16w_hi(w);
                } else {
                    kv[u][0] = ptr ? bf16_to_f32((uint16_t)ldg_u16(ptr + dcol)) : 0.0f;
                }
            }
#pragma unroll
            for (int h = 0; h < HG; ++h) {
                const float4 pv = *(const float4*)(sp + (long)h * a.nkp + k);
                const float pk[4] = { pv.x, pv.y, pv.z, pv.w };
#pragma unroll
                for (int u = 0; u < 4; ++u) {
#pragma unroll
                    for (int d = 0; d < DPT; ++d) {
                        const float prod = pk[u] * kv[u][d];
                        acc[h][d] = acc[h][d] + prod;
                    }
                }
                if (wave == 0) {
#pragma unroll
                    for (int u = 0; u < 4; ++u) den[h] = den[h] + pk[u];
                }
            }
        }
        if (wave == 0 && lane < hmax) {
            float dn = den[0];
#pragma unroll
            for (int h = 1; h < HG; ++h)
                if (lane == h) dn = den[h];
            sden[lane] = dn + ds_expf(a.sink[(long)(hg0 + lane) * a.sks] - smx[lane]);
        }
        __syncthreads();
        uint16_t* orow = a.out + m * a.outr + (long)hg0 * D + dcol;
#pragma unroll
        for (int h = 0; h < HG; ++h) {
            if (h >= hmax) break;
            const float dn = sden[h];
            if constexpr (DPT == 2) {
                const uint32_t lo = ds_bf16_bits(acc[h][0] / dn), hi = ds_bf16_bits(acc[h][DPT - 1] / dn);
                *(uint32_t*)(orow + (long)h * D) = lo | (hi << 16);
            } else {
                orow[(long)h * D] = (uint16_t)ds_bf16_bits(acc[h][0] / dn);
            }
        }
        __syncthreads();
    }
}

/* ================================================================== attn_sparse_sink on WMMA
 *
 * Passes of more than kWmRows rows (prefill) on v_wmma_f32_16x16x16_bf16. A workgroup is a
 * (16-head tile, row); its four waves own 128 latent columns each and walk the row's keys 16 at
 * a time:
 *   S^T = K Q^T     over the wave's 128 columns: A = 16 keys as stored (8 columns a lane), B = Q^T
 *                   held in registers. The four partials meet in LDS and every wave sums them in
 *                   wave order, so all four hold the same scores: lane L head L%16, keys 8(L/16)+e.
 *   softmax         online: a running max per head (floored at -1e30), p = exp2(s log2e - max) by
 *                   v_exp_f32, the accumulators rescaled when the max moves, den summing f32 p.
 *   O^T += K^T P^T  A = K^T by global_load_tr_b128 (lane L column L%16, keys 8(L/16)+e), B = P^T
 *                   straight from the score layout, P = hi + lo in bf16 (two WMMAs; hi truncated,
 *                   lo = p - hi rounded, so |p - hi - lo| <= 2^-17 p).
 * Not libref's bits: q.k products are exact but sum in WMMA order, the exponential is v_exp_f32
 * and the softmax online; the old kernel stays for the decode band (and R4D_ATTN_WM_OFF=1). A key
 * with no entry reads a zero row and scores -inf (0 * NaN from a recycled slot would be NaN). */
typedef __bf16 ds_v8bf __attribute__((ext_vector_type(8)));
typedef float ds_v8f __attribute__((ext_vector_type(8)));
typedef short ds_v8s __attribute__((ext_vector_type(8)));
typedef __attribute__((address_space(1))) ds_v8s ds_gv8s;

constexpr int kWmRows = 48;                          /* passes up to this many rows stay scalar */
constexpr int kWmWaves = 4;
constexpr int kWmCols = kDsAttnD / kWmWaves;         /* latent columns a wave */
constexpr int kWmSbufBytes = 2 * kWmWaves * 32 * 8 * 4;

__device__ __align__(16) uint16_t kDsZeroKey[kDsAttnD];

__host__ __device__ constexpr long ds_wm_mask_off(int nkt) { return (long)nkt * 16 * 8; }
__host__ __device__ constexpr long ds_wm_sbuf_off(int nkt) {
    return (ds_wm_mask_off(nkt) + (long)nkt * 4 + 15) / 16 * 16;
}

__device__ __forceinline__ ds_v8f ds_wmma(uint4 a, uint4 b, ds_v8f c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(__builtin_bit_cast(ds_v8bf, a),
                                                             __builtin_bit_cast(ds_v8bf, b), c);
}
/* global_load_tr_b128: lane 8g+e gives the address of 8 consecutive bf16 and lane 8g+i receives
 * value i of lanes 8g..8g+7, in lane order (an 8x8 transpose within each group of eight lanes). */
__device__ __forceinline__ uint4 ds_load_tr(const uint16_t* p) {
    return __builtin_bit_cast(uint4, __builtin_amdgcn_global_load_tr_b128_v8i16((ds_gv8s*)p));
}
__device__ __forceinline__ float ds_xor16(float v) {
    const int x = __builtin_bit_cast(int, v);
    return __builtin_bit_cast(float, __builtin_amdgcn_permlanex16(x, x, 0x76543210, 0xfedcba98, false, false));
}
__device__ __forceinline__ float ds_ex2(float x) { return __builtin_amdgcn_exp2f(x); }
__device__ __forceinline__ uint32_t ds_rne_hi(float f) {        /* bf16 of a finite f, high half */
    const uint32_t u = __float_as_uint(f);
    return (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u;
}

template <bool LO>
__global__ __launch_bounds__(128) void ds41_attn_sparse_sink_wm_kernel(DsAttnArgs a, float sl2) {
    constexpr int D = kDsAttnD;
    constexpr float kL2e = 1.4426950408889634f;
    const int nkt = (a.nk + 15) >> 4, nkk = nkt * 16;
    const uint16_t** kp = (const uint16_t**)r4d_ds41_sparse_dyn;
    uint32_t* vmask = (uint32_t*)(r4d_ds41_sparse_dyn + ds_wm_mask_off(nkt));
    float4* sbuf = (float4*)(r4d_ds41_sparse_dyn + ds_wm_sbuf_off(nkt));
    const int tid = threadIdx.x, lane = tid & 31, hf = lane >> 4;
    const int wave = __builtin_amdgcn_readfirstlane((int)(tid >> 5));
    const int hl = blockIdx.x * 16 + (lane & 15);
    const bool hok = hl < a.H;
    const int c0 = wave * kWmCols;
    const int kq = lane & 15;                        /* S^T's A row: key kq of the tile */
    const int ktr = 8 * hf + (lane & 7);             /* K^T's transposing row: key ktr */
    const int dtr = c0 + 8 * ((lane >> 3) & 1);
    for (long m = blockIdx.y; m < a.M; m += gridDim.y) {
        /* ---- the row's keys in libref's order (the window, then the selections), a null one
         * pointing at a zero row; a 16-bit mask a tile of which are keys */
        const int s = ds_seq_of_row(a.cu, a.cs, a.S, m);
        const long p = a.pos[m * a.ps];
        long lr = (long)a.cu[(long)(s + 1) * a.cs] - 1;
        lr = lr < 0 ? 0 : (lr >= a.M ? a.M - 1 : lr);
        const long last = a.pos[lr * a.ps];
        const long first = last + 1 - (long)a.wused[(long)s * a.wus];
        for (int k0 = 0; k0 < nkk; k0 += 128) {
            const int k = k0 + tid;
            const uint16_t* ptr = nullptr;
            if (k < a.win) {
                const long j = p - a.win + 1 + k, r = j - first;
                if (j >= 0 && r >= 0 && r / a.bs < a.nwbt) {
                    const long blk = a.wbt[(long)s * a.wbtr + (r / a.bs) * a.wbtc];
                    if (blk >= 0 && blk < a.nwblk) ptr = a.wc + (blk * a.bs + r % a.bs) * (long)D;
                }
            } else if (k < a.nk) {
                const long c = a.sel[m * a.selr + (long)(k - a.win) * a.selc];
                if (c >= 0 && (c * a.ratio) / a.cbs < a.ncbt) {
                    const long blk = a.cbt[(long)s * a.cbtr + ((c * a.ratio) / a.cbs) * a.cbtc];
                    if (blk >= 0 && blk < a.ncblk)
                        ptr = a.cc + (blk * a.cbs + (c * a.ratio) % a.cbs) * (long)(D / a.ratio);
                }
            }
            if (k < nkk) kp[k] = ptr ? ptr : kDsZeroKey;
            const uint32_t b = (uint32_t)__builtin_amdgcn_ballot_w32(ptr != nullptr);
            const int t0 = (k0 + wave * 32) >> 4;
            if (lane == 0) {
                if (t0 < nkt) vmask[t0] = b & 0xffffu;
                if (t0 + 1 < nkt) vmask[t0 + 1] = b >> 16;
            }
        }
        __syncthreads();

        /* ---- Q^T for this lane's head and the wave's columns: step i, half hf = columns
         * c0 + 64 hf + 8 i .. + 7 (the same split as the key rows below) */
        uint4 qf[8];
        {
            const uint16_t* qp = a.q + m * a.qr + (long)(hok ? hl : 0) * D + c0 + 64 * hf;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                qf[i] = *(const uint4*)(qp + 8 * i);
                if (!hok) qf[i] = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        ds_v8f o[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) o[j] = (ds_v8f)(0.0f);
        float mrun = -1e30f, den = 0.0f;
        uint4 kn[8];
        {
            const uint16_t* pk = kp[kq] + c0 + 64 * hf;
#pragma unroll
            for (int i = 0; i < 8; ++i) kn[i] = *(const uint4*)(pk + 8 * i);
        }
        for (int t = 0; t < nkt; ++t) {
            uint4 kt[8];
            {
                const uint16_t* pt = kp[t * 16 + ktr] + dtr;
#pragma unroll
                for (int j = 0; j < 8; ++j) kt[j] = ds_load_tr(pt + 16 * j);
            }
            ds_v8f sp = (ds_v8f)(0.0f);
#pragma unroll
            for (int i = 0; i < 8; ++i) sp = ds_wmma(kn[i], qf[i], sp);
            if (t + 1 < nkt) {
                const uint16_t* pk = kp[(t + 1) * 16 + kq] + c0 + 64 * hf;
#pragma unroll
                for (int i = 0; i < 8; ++i) kn[i] = *(const uint4*)(pk + 8 * i);
            }
            float4* sb = sbuf + (t & 1) * (kWmWaves * 64);
            sb[wave * 64 + lane] = make_float4(sp[0], sp[1], sp[2], sp[3]);
            sb[wave * 64 + 32 + lane] = make_float4(sp[4], sp[5], sp[6], sp[7]);
            lds_barrier();
            float x[8];
            {
                const float4 l0 = sb[lane], h0 = sb[32 + lane];
                x[0] = l0.x; x[1] = l0.y; x[2] = l0.z; x[3] = l0.w;
                x[4] = h0.x; x[5] = h0.y; x[6] = h0.z; x[7] = h0.w;
#pragma unroll
                for (int w = 1; w < kWmWaves; ++w) {
                    const float4 l = sb[w * 64 + lane], h = sb[w * 64 + 32 + lane];
                    x[0] += l.x; x[1] += l.y; x[2] += l.z; x[3] += l.w;
                    x[4] += h.x; x[5] += h.y; x[6] += h.z; x[7] += h.w;
                }
            }
            const uint32_t vm = vmask[t] >> (8 * hf);
            float tmax = -kDsInf;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                x[e] = ((vm >> e) & 1u) ? x[e] * sl2 : -kDsInf;
                tmax = fmaxf(tmax, x[e]);
            }
            tmax = fmaxf(tmax, ds_xor16(tmax));
            const float mnew = fmaxf(mrun, tmax);
            const float alpha = ds_ex2(mrun - mnew);
            float pe[8], ps = 0.0f;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                pe[e] = ds_ex2(x[e] - mnew);
                ps += pe[e];
            }
            den = den * alpha + ps;
            if (__builtin_amdgcn_ballot_w32(alpha != 1.0f)) {
#pragma unroll
                for (int j = 0; j < 8; ++j) o[j] *= alpha;
            }
            mrun = mnew;
            uint32_t hw[4], lw[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const float p0 = pe[2 * q], p1 = pe[2 * q + 1];
                if constexpr (LO) {
                    const uint32_t u0 = __float_as_uint(p0) & 0xffff0000u, u1 = __float_as_uint(p1) & 0xffff0000u;
                    hw[q] = __builtin_amdgcn_perm(u1, u0, 0x07060302u);
                    lw[q] = __builtin_amdgcn_perm(ds_rne_hi(p1 - __uint_as_float(u1)),
                                                  ds_rne_hi(p0 - __uint_as_float(u0)), 0x07060302u);
                } else {
                    hw[q] = __builtin_amdgcn_perm(ds_rne_hi(p1), ds_rne_hi(p0), 0x07060302u);
                }
            }
            const uint4 ph = make_uint4(hw[0], hw[1], hw[2], hw[3]);
#pragma unroll
            for (int j = 0; j < 8; ++j) o[j] = ds_wmma(kt[j], ph, o[j]);
            if constexpr (LO) {
                const uint4 pl = make_uint4(lw[0], lw[1], lw[2], lw[3]);
#pragma unroll
                for (int j = 0; j < 8; ++j) o[j] = ds_wmma(kt[j], pl, o[j]);
            }
        }
        /* ---- den over both key halves, the sink, one rounding: lane L holds head L%16 at
         * columns c0 + 16 j + 8 hf + e */
        float dt = den + ds_xor16(den);
        if (hok) {
            dt += ds_ex2(a.sink[(long)hl * a.sks] * kL2e - mrun);
            uint16_t* op = a.out + m * a.outr + (long)hl * D + c0 + 8 * hf;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                uint32_t w[4];
#pragma unroll
                for (int q = 0; q < 4; ++q)
                    w[q] = ds_bf16_bits(o[j][2 * q] / dt) | (ds_bf16_bits(o[j][2 * q + 1] / dt) << 16);
                *(uint4*)(op + 16 * j) = make_uint4(w[0], w[1], w[2], w[3]);
            }
        }
        __syncthreads();
    }
}

/* ================================================================== index_score on WMMA
 *
 * Passes of more than kWmRows rows. A wave is a row and walks 16-entry tiles of its workgroup's
 * kIwEB entries: S = Q K^T (A = the row's 16-head tiles of q, B = 16 entries as stored; 8 steps of
 * 16 dims, q held in registers), D lane L = entry L%16, heads 8(L/16)+e. Then libref's roundings:
 * the dot to bf16, relu x the rounded head weight, rounded, summed in head order (the running sum
 * hops lanes L <-> L^16 between 8-head runs), rounded. Only the dot's summation order differs
 * from libref, so a score moves only where the f32 dot sat at a bf16 rounding boundary. */
constexpr int kIwEB = 128;                           /* entries a workgroup */

template <int HT>
__global__ __launch_bounds__(256) void ds41_index_score_wm_kernel(DsScoreArgs a) {
    constexpr int DM = 128;
    const int lane = threadIdx.x & 31, hf = lane >> 4;
    const int wave = __builtin_amdgcn_readfirstlane((int)(threadIdx.x >> 5));
    const long m = (long)blockIdx.y * 8 + wave;
    if (m >= a.M) return;
    const long jb = (long)blockIdx.x * kIwEB;
    const long jend = jb + kIwEB < (long)a.C ? jb + kIwEB : (long)a.C;
    float* srow = a.sc + m * a.sr;
    const long reach = ((long)a.pos[m * a.ps] + 1) / a.ratio;
    if (jb >= reach) {
        for (long j = jb + lane; j < jend; j += 32) srow[j * a.scs] = -kDsInf;
        return;
    }
    const int s = ds_seq_of_row(a.cu, a.cs, a.S, m);
    uint4 qf[HT][8];
    float wh[HT][8];
#pragma unroll
    for (int t = 0; t < HT; ++t) {
        const int h = t * 16 + (lane & 15);
        const uint16_t* qp = a.q + m * a.qr + (long)(h < a.H ? h : 0) * DM + 64 * hf;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            qf[t][i] = *(const uint4*)(qp + 8 * i);
            if (h >= a.H) qf[t][i] = make_uint4(0u, 0u, 0u, 0u);
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int hh = t * 16 + 8 * hf + e;
            wh[t][e] = hh < a.H ? ds_round_bf16(ds_bf(a.w[m * a.wr + (long)hh * a.wc]) * a.wscale) : 0.0f;
        }
    }
    for (long j0 = jb; j0 < jend; j0 += 16) {
        if (j0 >= reach) {                           /* wave-uniform: the rest is past the row */
            for (long j = j0 + lane; j < jend; j += 32) srow[j * a.scs] = -kDsInf;
            break;
        }
        const long j = j0 + (lane & 15);
        const uint16_t* kp = nullptr;
        if (j < jend && j < reach) {
            const long col = (j * a.ratio) / a.cbs;
            if (col < a.nbt) {
                const long blk = a.bt[(long)s * a.btr + col * a.btc];
                if (blk >= 0 && blk < a.nblk) kp = a.kc + (blk * a.cbs + (j * a.ratio) % a.cbs) * (long)a.slot;
            }
        }
        const uint16_t* pk = (kp ? kp : kDsZeroKey) + 64 * hf;
        uint4 kb[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) kb[i] = *(const uint4*)(pk + 8 * i);
        ds_v8f d[HT];
#pragma unroll
        for (int t = 0; t < HT; ++t) {
            d[t] = (ds_v8f)(0.0f);
#pragma unroll
            for (int i = 0; i < 8; ++i) d[t] = ds_wmma(qf[t][i], kb[i], d[t]);
        }
        /* heads 0-7 in lane L, 8-15 in L^16, then 16-23 back in L and 24-31 in L^16 */
        float acc = 0.0f;
#pragma unroll
        for (int t = 0; t < HT; ++t) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                if (t + half > 0) acc = ds_xor16(acc);
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const float dot = ds_round_bf16(d[t][e]);
                    acc = acc + ds_round_bf16((dot < 0.0f ? 0.0f : dot) * wh[t][e]);
                }
            }
        }
        if (hf == 1 && j < jend) srow[j * a.scs] = kp ? ds_round_bf16(acc) : -kDsInf;
    }
}

/* ================================================================== launch shims */
/* Linear addressing of an index operand: rank 1 by its stride, anything else only if dense. */
long lin_stride(const RadTensor& t) {
    if (t.rank == 1) return (long)t.stride[0];
    return r4d_contig(&t) ? 1 : -1;
}

bool aligned(const void* p, unsigned bytes) { return ((uintptr_t)p & (uintptr_t)(bytes - 1)) == 0; }

int ds_env_int(const char* k, int dflt) {
    const char* v = std::getenv(k);
    return v && *v ? std::atoi(v) : dflt;
}

constexpr int kDsChunk = 8192;                       /* index_topk's split: positions a chunk */
constexpr int kDsSplitRows = 16;                     /* ... at up to this many rows */
constexpr int kDsSplitMinChunks = 4;                 /* ... over at least this many chunks */
constexpr int kDsSplitMaxK = 2048;

int64_t topk_split_bytes(long rows, long C, long K) {
    const long nch = (C + kDsChunk - 1) / kDsChunk;
    if (rows < 1 || nch < kDsSplitMinChunks || K < 1 || K > kDsSplitMaxK) return 0;
    return (int64_t)rows * nch * K * 8 + (int64_t)rows * nch * 4 + 16;
}

int index_score_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 7 || !a->t) return RAD_E_INVAL;
    const RadTensor &q = a->t[0], &w = a->t[1], &kc = a->t[2], &bt = a->t[3];
    const RadTensor &pos = a->t[4], &cu = a->t[5], &sc = a->t[6];
    const long long H = r4d_geti_or(a, "heads", 0), Dm = r4d_geti_or(a, "dim", 0);
    const long long ratio = r4d_geti_or(a, "ratio", 0);
    const float wscale = (float)r4d_getf(a, "wscale", 0.0);
    for (int i = 0; i < 7; ++i)
        if (!a->t[i].data) return R4D_NO(RAD_E_INVAL);
    if (H < 1 || Dm < 1 || ratio < 1 || Dm % ratio || !std::isfinite(wscale) || q.rank != 2 ||
        q.shape[0] < 1 || q.shape[1] != H * Dm || w.rank != 2 || w.shape[0] != q.shape[0] ||
        w.shape[1] != H || kc.rank != 4 || kc.shape[1] != 1 || kc.shape[3] * ratio != Dm ||
        kc.shape[2] % ratio || bt.rank != 2 || r4d_numel(&pos) != q.shape[0] || r4d_numel(&cu) < 2 ||
        sc.rank != 2 || sc.shape[0] != q.shape[0] || sc.shape[1] < 1)
        return R4D_NO(RAD_E_SHAPE);
    if (q.dtype != RAD_BF16 || w.dtype != RAD_BF16 || kc.dtype != RAD_BF16 || bt.dtype != RAD_I32 ||
        pos.dtype != RAD_I32 || cu.dtype != RAD_I32 || sc.dtype != RAD_F32)
        return R4D_NO(RAD_E_DTYPE);
    /* This row: 128-wide keys read 16 bytes at a time, q read as bf16 pairs. */
    if (Dm != 128 || ratio > 16) return R4D_NO(RAD_E_UNSUPPORTED);
    const long ps = lin_stride(pos), cs = lin_stride(cu);
    if (ps < 0 || cs < 0 || q.stride[1] != 1 || (q.stride[0] & 1) || !aligned(q.data, 4) ||
        !aligned(kc.data, 16) || q.shape[0] > INT_MAX || sc.shape[1] > INT_MAX / 2 ||
        r4d_numel(&cu) - 1 > INT_MAX || bt.shape[1] > INT_MAX || H > INT_MAX / 128)
        return R4D_NO(RAD_E_SHAPE);
    DsScoreArgs s{};
    s.q = (const uint16_t*)q.data;
    s.w = (const uint16_t*)w.data;
    s.kc = (const uint16_t*)kc.data;
    s.bt = (const int*)bt.data;
    s.pos = (const int*)pos.data;
    s.cu = (const int*)cu.data;
    s.sc = (float*)sc.data;
    s.qr = (long)q.stride[0];
    s.wr = (long)w.stride[0];
    s.wc = (long)w.stride[1];
    s.btr = (long)bt.stride[0];
    s.btc = (long)bt.stride[1];
    s.nblk = (long)kc.shape[0];
    s.ps = ps;
    s.cs = cs;
    s.sr = (long)sc.stride[0];
    s.scs = (long)sc.stride[1];
    s.nbt = (int)bt.shape[1];
    s.S = (int)(r4d_numel(&cu) - 1);
    s.M = (int)q.shape[0];
    s.C = (int)sc.shape[1];
    s.H = (int)H;
    s.ratio = (int)ratio;
    s.cbs = (int)kc.shape[2];
    s.slot = (int)(Dm / ratio);
    s.wscale = wscale;
    /* Prefill heights on WMMA (up to 32 heads, 16-byte q rows); the decode band, and every height
     * with R4D_IDX_WM_OFF=1, on the scalar kernel below (libref's bits). */
    static const int iw_off = ds_env_int("R4D_IDX_WM_OFF", 0);
    if (!iw_off && s.M > kWmRows && H <= 32 && s.qr % 8 == 0 && aligned(q.data, 16) &&
        (s.M + 7) / 8 <= 65535) {
        const dim3 grid((unsigned)((s.C + kIwEB - 1) / kIwEB), (unsigned)((s.M + 7) / 8));
        if (H > 16) return R4D_EMIT(stream, ds41_index_score_wm_kernel<2>, grid, dim3(256), 0u, s);
        return R4D_EMIT(stream, ds41_index_score_wm_kernel<1>, grid, dim3(256), 0u, s);
    }
    /* Head split: more workgroups while the grid is short of the card. The head sum is closed in
     * head order either way, so every split gives the same bits. */
    int HS = 1;
    while (HS < 8 && HS * 2 <= H && ((long)s.C * HS * 2 + 255) / 256 * s.M <= 1024 &&
           (long)H * (256 / (HS * 2)) * 4 <= 32768)
        HS *= 2;
    const unsigned gy = (unsigned)(s.M < 65535 ? s.M : 65535);
    const int epb = 256 / HS;
    const dim3 grid((unsigned)((s.C + epb - 1) / epb), gy);
    const uint32_t lds = HS > 1 ? (uint32_t)(H * epb * 4) : 0u;
    switch (HS) {
        case 1: return R4D_EMIT(stream, ds41_index_score_kernel<1>, grid, dim3(256), lds, s);
        case 2: return R4D_EMIT(stream, ds41_index_score_kernel<2>, grid, dim3(256), lds, s);
        case 4: return R4D_EMIT(stream, ds41_index_score_kernel<4>, grid, dim3(256), lds, s);
        default: return R4D_EMIT(stream, ds41_index_score_kernel<8>, grid, dim3(256), lds, s);
    }
}

int index_candidates_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 3 || !a->t) return RAD_E_INVAL;
    const RadTensor &sc = a->t[0], &pos = a->t[1], &keep = a->t[2];
    const long long block = r4d_geti_or(a, "block", 0), kb = r4d_geti_or(a, "topk_blocks", 0);
    const long long ratio = r4d_geti_or(a, "ratio", 0);
    if (!sc.data || !pos.data || !keep.data) return R4D_NO(RAD_E_INVAL);
    if (block < 1 || kb < 1 || ratio < 1 || sc.rank != 2 || sc.shape[0] < 1 || sc.shape[1] < 1 ||
        r4d_numel(&pos) != sc.shape[0] || keep.rank != 2 || keep.shape[0] != sc.shape[0] ||
        keep.shape[1] != sc.shape[1])
        return R4D_NO(RAD_E_SHAPE);
    if (sc.dtype != RAD_F32 || pos.dtype != RAD_I32 || keep.dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);
    const long ps = lin_stride(pos);
    if (ps < 0 || sc.shape[0] > INT_MAX || sc.shape[1] > INT_MAX / 2 || block > INT_MAX)
        return R4D_NO(RAD_E_SHAPE);
    DsCandArgs c{};
    c.sc = (const float*)sc.data;
    c.pos = (const int*)pos.data;
    c.keep = (int*)keep.data;
    c.sr = (long)sc.stride[0];
    c.scs = (long)sc.stride[1];
    c.ps = ps;
    c.kr = (long)keep.stride[0];
    c.kcs = (long)keep.stride[1];
    c.M = (int)sc.shape[0];
    c.C = (int)sc.shape[1];
    c.block = (int)block;
    c.KB = (int)(kb < INT_MAX ? kb : INT_MAX);
    c.ratio = (int)(ratio < INT_MAX ? ratio : INT_MAX);
    const unsigned g = (unsigned)(c.M < 65535 ? c.M : 65535);
    return R4D_EMIT(stream, ds41_index_candidates_kernel, dim3(g), dim3(kDsSelThreads), 0, c);
}

int index_topk_params(const RadArgs* a, long long* topk, long long* ratio) {
    *topk = r4d_geti_or(a, "topk", 0);
    *ratio = r4d_geti_or(a, "ratio", 0);
    return (*topk < 1 || *ratio < 1) ? RAD_E_SHAPE : RAD_OK;
}

int index_topk_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 4 || !a->t) return RAD_E_INVAL;
    const RadTensor &sc = a->t[0], &pos = a->t[2], &sel = a->t[3];
    const RadTensor* keep = a->t[1].data ? &a->t[1] : nullptr;
    long long topk, ratio;
    if (index_topk_params(a, &topk, &ratio) != RAD_OK) return R4D_NO(RAD_E_SHAPE);
    if (!sc.data || !pos.data || !sel.data || sc.rank != 2 || sc.shape[0] < 1 ||
        r4d_numel(&pos) != sc.shape[0] || sel.rank != 2 || sel.shape[0] != sc.shape[0] ||
        sel.shape[1] != topk ||
        (keep && (keep->rank != 2 || keep->shape[0] != sc.shape[0] || keep->shape[1] != sc.shape[1])))
        return R4D_NO(RAD_E_SHAPE);
    if (sc.dtype != RAD_F32 || pos.dtype != RAD_I32 || sel.dtype != RAD_I32 ||
        (keep && keep->dtype != RAD_I32))
        return R4D_NO(RAD_E_DTYPE);
    const long ps = lin_stride(pos);
    if (ps < 0 || sc.shape[0] > INT_MAX || sc.shape[1] < 0 || sc.shape[1] > INT_MAX / 2 ||
        topk > INT_MAX / 2)
        return R4D_NO(RAD_E_SHAPE);
    DsTopkArgs t{};
    t.sc = (const float*)sc.data;
    t.keep = keep ? (const int*)keep->data : nullptr;
    t.pos = (const int*)pos.data;
    t.sel = (int*)sel.data;
    t.sr = (long)sc.stride[0];
    t.scs = (long)sc.stride[1];
    t.kr = keep ? (long)keep->stride[0] : 0;
    t.kcs = keep ? (long)keep->stride[1] : 0;
    t.ps = ps;
    t.selr = (long)sel.stride[0];
    t.selc = (long)sel.stride[1];
    t.M = (int)sc.shape[0];
    t.C = (int)sc.shape[1];
    t.K = (int)topk;
    t.ratio = (int)(ratio < INT_MAX ? ratio : INT_MAX);
    /* Few rows over a long row: chunks in parallel, then one merge a row. The same selection, so
     * the same bits, as the one-block-a-row kernel. */
    const int64_t need = t.M <= kDsSplitRows ? topk_split_bytes(t.M, t.C, t.K) : 0;
    if (need > 0 && a->scratch && a->scratch_bytes >= need && aligned(a->scratch, 8)) {
        DsSplitArgs sp{};
        sp.t = t;
        sp.chunk = kDsChunk;
        sp.nch = (t.C + kDsChunk - 1) / kDsChunk;
        sp.cand = (unsigned long long*)a->scratch;
        sp.cnt = (int*)((unsigned char*)a->scratch + (int64_t)t.M * sp.nch * t.K * 8);
        const int rc = R4D_EMIT(stream, ds41_index_topk_split1_kernel,
                                dim3((unsigned)sp.nch, (unsigned)t.M), dim3(kDsSelThreads), 0, sp);
        if (rc != RAD_OK) return rc;
        return R4D_EMIT(stream, ds41_index_topk_split2_kernel, dim3((unsigned)t.M),
                        dim3(kDsSelThreads), 0, sp);
    }
    const unsigned g = (unsigned)(t.M < 65535 ? t.M : 65535);
    return R4D_EMIT(stream, ds41_index_topk_kernel, dim3(g), dim3(kDsSelThreads), 0, t);
}

template <int HG, int DS>
int attn_emit(RadStream stream, const DsAttnArgs& s, unsigned gz) {
    const uint32_t lds = (uint32_t)(ds_attn_tile_off(s.nkp, HG) + kDsTileBytes);
    const dim3 grid((unsigned)DS, (unsigned)((s.H + HG - 1) / HG), gz);
    return R4D_EMIT(stream, (ds41_attn_sparse_sink_kernel<HG, DS>), grid, dim3(256), lds, s);
}

int attn_body(const RadArgs* a, RadStream stream) {
    /* Operands 4-6 are optional (null without a compressed source), as in libref. */
    if (!a || a->n_t < 11 || !a->t) return RAD_E_INVAL;
    const RadTensor &q = a->t[0], &wc = a->t[1], &wbt = a->t[2], &wused = a->t[3];
    const RadTensor* cc = a->t[4].data ? &a->t[4] : nullptr;
    const RadTensor* cbt = a->t[5].data ? &a->t[5] : nullptr;
    const RadTensor* sel = a->t[6].data ? &a->t[6] : nullptr;
    const RadTensor &pos = a->t[7], &cu = a->t[8], &sink = a->t[9], &out = a->t[10];
    const long long H = r4d_geti_or(a, "heads", 0), D = r4d_geti_or(a, "dim", 0);
    const long long win = r4d_geti_or(a, "window", 0), ratio = r4d_geti_or(a, "ratio", 0);
    const long long topk = r4d_geti_or(a, "topk", 0);
    const float scale = (float)r4d_getf(a, "scale", 0.0);
    const bool comp = ratio > 0;
    if (H < 1 || D < 2 || D > 4096 || win < 1 || ratio < 0 || topk < 0 || !std::isfinite(scale) ||
        scale <= 0 || (comp && (!cc || !cbt || !sel || topk < 1 || D % ratio)))
        return R4D_NO(RAD_E_SHAPE);
    if (!q.data || !wc.data || !wbt.data || !wused.data || !pos.data || !cu.data || !sink.data ||
        !out.data || q.rank != 2 || q.shape[0] < 1 || q.shape[1] != H * D || out.rank != 2 ||
        out.shape[0] != q.shape[0] || out.shape[1] != H * D || wc.rank != 4 || wc.shape[1] != 1 ||
        wc.shape[3] != D || wbt.rank != 2 || r4d_numel(&pos) != q.shape[0] || r4d_numel(&sink) != H ||
        sink.dtype != RAD_F32 || r4d_numel(&cu) < 2 || r4d_numel(&wused) < r4d_numel(&cu) - 1)
        return R4D_NO(RAD_E_SHAPE);
    const long long cbs = comp ? cc->shape[2] : 0;
    if (comp && (cc->rank != 4 || cc->shape[1] != 1 || cc->shape[3] * ratio != D || cbs % ratio ||
                 cbt->rank != 2 || sel->rank != 2 || sel->shape[0] != q.shape[0] ||
                 sel->shape[1] != topk || cbt->shape[0] != wbt.shape[0]))
        return R4D_NO(RAD_E_SHAPE);
    if (q.dtype != RAD_BF16 || wc.dtype != RAD_BF16 || out.dtype != RAD_BF16 || wbt.dtype != RAD_I32 ||
        wused.dtype != RAD_I32 || pos.dtype != RAD_I32 || cu.dtype != RAD_I32 ||
        (comp && (cc->dtype != RAD_BF16 || cbt->dtype != RAD_I32 || sel->dtype != RAD_I32)))
        return R4D_NO(RAD_E_DTYPE);
    /* This row: the 512-wide latent, read 16 bytes at a time; at most 1024 + 2048 keys a row. */
    if (D != kDsAttnD || win > 1024 || topk > 2048 || ratio > 64) return R4D_NO(RAD_E_UNSUPPORTED);
    const long S = (long)(r4d_numel(&cu) - 1);
    const long ps = lin_stride(pos), cs = lin_stride(cu), wus = lin_stride(wused), sks = lin_stride(sink);
    if (ps < 0 || cs < 0 || wus < 0 || sks < 0 || q.stride[1] != 1 || out.stride[1] != 1 ||
        (q.stride[0] & 1) || (out.stride[0] & 1) || !aligned(q.data, 4) || !aligned(out.data, 4) ||
        !aligned(wc.data, 16) || (comp && !aligned(cc->data, 16)) || wbt.shape[0] < S ||
        q.shape[0] > INT_MAX || S > INT_MAX || wbt.shape[1] > INT_MAX ||
        (comp && cbt->shape[1] > INT_MAX) || wc.shape[2] < 1 || wc.shape[2] > INT_MAX || cbs > INT_MAX)
        return R4D_NO(RAD_E_SHAPE);
    DsAttnArgs s{};
    s.q = (const uint16_t*)q.data;
    s.wc = (const uint16_t*)wc.data;
    s.wbt = (const int*)wbt.data;
    s.wused = (const int*)wused.data;
    s.cc = comp ? (const uint16_t*)cc->data : nullptr;
    s.cbt = comp ? (const int*)cbt->data : nullptr;
    s.sel = comp ? (const int*)sel->data : nullptr;
    s.pos = (const int*)pos.data;
    s.cu = (const int*)cu.data;
    s.sink = (const float*)sink.data;
    s.out = (uint16_t*)out.data;
    s.qr = (long)q.stride[0];
    s.outr = (long)out.stride[0];
    s.wbtr = (long)wbt.stride[0];
    s.wbtc = (long)wbt.stride[1];
    s.nwblk = (long)wc.shape[0];
    s.cbtr = comp ? (long)cbt->stride[0] : 0;
    s.cbtc = comp ? (long)cbt->stride[1] : 0;
    s.ncblk = comp ? (long)cc->shape[0] : 0;
    s.selr = comp ? (long)sel->stride[0] : 0;
    s.selc = comp ? (long)sel->stride[1] : 0;
    s.wus = wus;
    s.ps = ps;
    s.cs = cs;
    s.sks = sks;
    s.nwbt = (int)wbt.shape[1];
    s.bs = (int)wc.shape[2];
    s.ncbt = comp ? (int)cbt->shape[1] : 0;
    s.cbs = comp ? (int)cbs : 1;
    s.ratio = comp ? (int)ratio : 1;
    s.S = (int)S;
    s.M = (int)q.shape[0];
    s.H = (int)H;
    s.win = (int)win;
    s.nk = (int)(win + (comp ? topk : 0));
    s.nkp = (s.nk + 3) / 4 * 4;
    s.scale = scale;
    /* Prefill heights on WMMA (16-byte q and out rows); the decode band, and every height with
     * R4D_ATTN_WM_OFF=1, on the scalar kernel below (libref's bits). */
    static const int wm_off = ds_env_int("R4D_ATTN_WM_OFF", 0);
    {
        const int nkt = (s.nk + 15) / 16;
        const long lds = ds_wm_sbuf_off(nkt) + kWmSbufBytes;
        if (!wm_off && s.M > kWmRows && s.qr % 8 == 0 && s.outr % 8 == 0 && aligned(q.data, 16) &&
            aligned(out.data, 16) && lds <= 65536 && (H + 15) / 16 <= 65535) {
            const unsigned gy = (unsigned)(s.M < 65535 ? s.M : 65535);
            const dim3 grid((unsigned)((H + 15) / 16), gy);
            const float sl2 = (float)((double)scale * 1.4426950408889634);
            return R4D_EMIT(stream, ds41_attn_sparse_sink_wm_kernel<true>, grid, dim3(128), (uint32_t)lds, s, sl2);
        }
    }
    /* Head group and d split by how many workgroups the rows give. Every choice sums each output
     * in the same order, so the bits do not depend on it. */
    const long rows = s.M;
    int HG = 8, DS = 1;
    if (rows * H < 2048) HG = rows * H >= 512 ? 4 : 2;
    if (HG == 2) DS = 2;
    while (HG > 1 && ds_attn_tile_off(s.nkp, HG) + kDsTileBytes > 65536) HG /= 2;
    if (ds_attn_tile_off(s.nkp, HG) + kDsTileBytes > 65536) return R4D_NO(RAD_E_UNSUPPORTED);
    const unsigned gz = (unsigned)(s.M < 65535 ? s.M : 65535);
    if (DS == 2) {
        if (HG == 2) return attn_emit<2, 2>(stream, s, gz);
        return attn_emit<1, 2>(stream, s, gz);
    }
    switch (HG) {
        case 8: return attn_emit<8, 1>(stream, s, gz);
        case 4: return attn_emit<4, 1>(stream, s, gz);
        case 2: return attn_emit<2, 1>(stream, s, gz);
        default: return attn_emit<1, 1>(stream, s, gz);
    }
}

}  // namespace

extern "C" int r4d_rad_ds41_index_score(const RadArgs* a, RadStream stream) {
    return index_score_body(a, stream);
}
extern "C" int r4d_rad_ds41_index_candidates(const RadArgs* a, RadStream stream) {
    return index_candidates_body(a, stream);
}
extern "C" int r4d_rad_ds41_index_topk(const RadArgs* a, RadStream stream) {
    return index_topk_body(a, stream);
}
/* Called at declare with the band's bound and no tensors: the split's partials for up to
 * kDsSplitRows rows of `width` positions; 0 (no split) when a row is short. */
extern "C" int64_t r4d_rad_ds41_index_topk_scratch(const RadArgs* a) {
    long long topk, ratio;
    if (index_topk_params(a, &topk, &ratio) != RAD_OK) return 0;
    long long M = r4d_geti_or(a, "M", kDsSplitRows);
    if (M < 1 || M > kDsSplitRows) M = kDsSplitRows;
    return topk_split_bytes((long)M, (long)r4d_geti_or(a, "width", 0), (long)topk);
}
extern "C" int r4d_rad_ds41_attn_sparse_sink(const RadArgs* a, RadStream stream) {
    return attn_body(a, stream);
}
