/* Lossless16-bit score-plane variants. Included inside sparse.hip anonymous namespace AFTER
 * all original kernels/helpers: originals above remain the independent f32 reference.
 * Copies differ ONLY at score stores, score loads and dtype admission. All FMA, rounding,
 * head folds, grids, radix selection/merge and ordered compaction remain the originals. */
struct DsScoreB16Args {
    const uint16_t* q;
    const uint16_t* w;
    const uint16_t* kc;
    const int* bt;
    const int* pos;
    const int* cu;
    uint16_t* sc;
    long qr, wr, wc, btr, btc, nblk, ps, cs, sr, scs, row0;
    int nbt, S, M, Mv, C, H, ratio, cbs, slot;
    float wscale;
};
struct DsTopkB16Args {
    const uint16_t* sc;
    const int* keep;
    const int* pos;
    int* sel;
    long sr, scs, kr, kcs, ps, selr, selc;
    int M, C, K, ratio, block, bshift;
};
/* The causal reach uses only pos/ps/ratio/C, whose types and meaning are the
 * original DsTopkArgs fields. Keep its arithmetic/clamps literal, without a
 * float score pointer projection or widening the original f32 helper. */
__device__ __forceinline__ long ds_topk_reach(const DsTopkB16Args& a, long m) {
    long reach = ((long)a.pos[m * a.ps] + 1) / a.ratio;
    if (reach > a.C) reach = a.C;
    return reach < 0 ? 0 : reach;
}
struct DsSplitB16Args {
    DsTopkB16Args t;
    unsigned long long* cand;
    int* cnt;
    int nch, chunk;
};
struct DsCandB16Args {
    const uint16_t* sc;
    const int* pos;
    int* keep;
    long sr, scs, ps, kr, kcs;
    int M, C, block, KB, ratio;
};

std::atomic<uint32_t> ds_b16_paths{0};
int ds_b16_path(int rc,uint32_t bit){if(rc==RAD_OK)ds_b16_paths.fetch_or(bit,std::memory_order_relaxed);return rc;}

template <int HS>
__global__ __launch_bounds__(256) void ds41_index_score_b16_kernel(DsScoreB16Args a) {
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
        uint16_t* srow = (uint16_t*)a.sc + m * a.sr;
        const long reach = m < a.Mv ? ((long)a.pos[m * a.ps] + 1) / a.ratio : 0;
        if (jb >= reach) {                           /* block-uniform: no key in this block */
            if (sub == 0 && j < a.C) srow[j * a.scs] = 0xff80u;
            continue;
        }
        const int s = ds_seq_of_row(a.cu, a.cs, a.S, a.row0 + m);
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
        if (sub == 0 && j < a.C) srow[j * a.scs] = kp ? (uint16_t)(__float_as_uint(ds_round_bf16(acc)) >> 16) : 0xff80u;
    }
}


template <int HT>
__global__ __launch_bounds__(256) void ds41_index_score_b16_wm_kernel(DsScoreB16Args a) {
    constexpr int DM = 128;
    const int lane = threadIdx.x & 31, hf = lane >> 4;
    const int wave = __builtin_amdgcn_readfirstlane((int)(threadIdx.x >> 5));
    const long m = (long)blockIdx.y * 8 + wave;
    if (m >= a.M) return;
    const long jb = (long)blockIdx.x * kIwEB;
    const long jend = jb + kIwEB < (long)a.C ? jb + kIwEB : (long)a.C;
    uint16_t* srow = (uint16_t*)a.sc + m * a.sr;
    const long reach = m < a.Mv ? ((long)a.pos[m * a.ps] + 1) / a.ratio : 0;
    if (jb >= reach) {
        for (long j = jb + lane; j < jend; j += 32) srow[j * a.scs] = 0xff80u;
        return;
    }
    const int s = ds_seq_of_row(a.cu, a.cs, a.S, a.row0 + m);
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
            for (long j = j0 + lane; j < jend; j += 32) srow[j * a.scs] = 0xff80u;
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
        if (hf == 1 && j < jend) srow[j * a.scs] = kp ? (uint16_t)(__float_as_uint(ds_round_bf16(acc)) >> 16) : 0xff80u;
    }
}


struct DsTopkB16Key {
    const uint16_t* row;
    const int* krow;
    long scs, kcs, j0;
    int block, bshift;
    __device__ __forceinline__ uint32_t operator()(uint32_t i) const {
        const long j = j0 + (long)i;
        if (krow) {
            const uint32_t b = bshift >= 0 ? (uint32_t)j >> bshift : (uint32_t)j / (uint32_t)block;
            if (!(((uint32_t)krow[(long)(b >> 5) * kcs] >> (b & 31u)) & 1u)) return 0u;
        }
        const float v = __uint_as_float((uint32_t)row[j * scs] << 16);
        return v > -kDsInf ? ds_key(v) : 0u;
    }
};


__global__ __launch_bounds__(kDsSelThreads) void ds41_index_topk_b16_kernel(DsTopkB16Args a) {
    __shared__ DsSelShared s;
    __shared__ uint32_t cache[kDsCache];
    for (long m = blockIdx.x; m < a.M; m += gridDim.x) {
        const uint32_t n = (uint32_t)ds_topk_reach(a, m);
        const DsTopkB16Key get{ (const uint16_t*)a.sc + m * a.sr, a.keep ? a.keep + m * a.kr : nullptr, a.scs, a.kcs, 0,
                             a.block, a.bshift };
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


__global__ __launch_bounds__(kDsSelThreads) void ds41_index_topk_b16_split1_kernel(DsSplitB16Args a) {
    __shared__ DsSelShared s;
    __shared__ uint32_t cache[kDsCache];
    const int c = blockIdx.x;
    const long m = blockIdx.y;
    const long j0 = (long)c * a.chunk;
    long n = ds_topk_reach(a.t, m) - j0;
    n = n < 0 ? 0 : (n > a.chunk ? a.chunk : n);
    const DsTopkB16Key get{ (const uint16_t*)a.t.sc + m * a.t.sr, a.t.keep ? a.t.keep + m * a.t.kr : nullptr, a.t.scs,
                         a.t.kcs, j0, a.t.block, a.t.bshift };
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


struct DsBlockB16Key {
    const uint16_t* row;
    long scs, C, block, pin;
    __device__ __forceinline__ uint32_t operator()(uint32_t b) const {
        float bs = -kDsInf;
        const long j0 = (long)b * block;
        const long j1 = j0 + block < C ? j0 + block : C;
        for (long j = j0; j < j1; ++j) {
            const float v = __uint_as_float((uint32_t)row[j * scs] << 16);
            bs = bs < v ? v : bs;
        }
        if ((long)b == pin) bs = kDsInf;
        return bs > -kDsInf ? ds_key(bs) : 0u;
    }
};

static_assert(kDsSelThreads % 32 == 0, "index_candidates writes a wave's ballot as a keep word");

__global__ __launch_bounds__(kDsSelThreads) void ds41_index_candidates_b16_kernel(DsCandB16Args a) {
    __shared__ DsSelShared s;
    __shared__ uint32_t cache[kDsCache];
    const long nb = ((long)a.C + a.block - 1) / a.block;
    for (long m = blockIdx.x; m < a.M; m += gridDim.x) {
        const long reach = ((long)a.pos[m * a.ps] + 1) / a.ratio;
        const long pin = (reach >= 1 && (reach - 1) / a.block < nb) ? (reach - 1) / a.block : -1;
        const DsBlockB16Key get{ (const uint16_t*)a.sc + m * a.sr, a.scs, a.C, a.block, pin };
        uint32_t* c = nb <= kDsCacheUse ? cache : nullptr;
        ds_select(get, (uint32_t)nb, (uint32_t)a.KB, c, &s);
        int* krow = a.keep + m * a.kr;
        const long kcs = a.kcs;
        /* ds_emit walks the blocks kDsSelThreads at a time, thread t on block i0 + t, so a wave's
         * 32 lanes hold one word's blocks and its lane 0 the word's first: the wave's ballot of
         * `take` is the word. Past the last block the lanes do not run and their bits are 0; a
         * wave whose lane 0 does not run holds no block. */
        ds_emit(get, (uint32_t)nb, c, &s,
                [&](uint32_t b, bool take, uint32_t) {
                    const uint32_t word = __builtin_amdgcn_ballot_w32(take);
                    if ((threadIdx.x & 31u) == 0u) krow[(long)(b >> 5) * kcs] = (int)word;
                },
                [&](uint32_t) {});
        __syncthreads();
    }
}


int index_score_b16_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 7 || !a->t) return RAD_E_INVAL;
    const RadTensor &q = a->t[0], &w = a->t[1], &kc = a->t[2], &bt = a->t[3];
    const RadTensor &pos = a->t[4], &cu = a->t[5], &sc = a->t[6];
    const long long H = r4d_geti_or(a, "heads", 0), Dm = r4d_geti_or(a, "dim", 0);
    const long long ratio = r4d_geti_or(a, "ratio", 0);
    const long long part = r4d_geti_or(a, "row_part", -1);
    const float wscale = (float)r4d_getf(a, "wscale", 0.0);
    for (int i = 0; i < 7; ++i)
        if (!a->t[i].data) return R4D_NO(RAD_E_INVAL);
    if (H < 1 || Dm < 1 || ratio < 1 || Dm % ratio || !std::isfinite(wscale) || q.rank != 2 ||
        q.shape[0] < 1 || q.shape[1] != H * Dm || w.rank != 2 || w.shape[0] != q.shape[0] ||
        w.shape[1] != H || kc.rank != 4 || kc.shape[1] != 1 || kc.shape[3] * ratio != Dm ||
        kc.shape[2] % ratio || bt.rank != 2 || r4d_numel(&pos) != q.shape[0] || r4d_numel(&cu) < 2 ||
        sc.rank != 2 || sc.shape[0] < 1 || (part < 0 && sc.shape[0] != q.shape[0]) || sc.shape[1] < 1)
        return R4D_NO(RAD_E_SHAPE);
    if (q.dtype != RAD_BF16 || w.dtype != RAD_BF16 || kc.dtype != RAD_BF16 || bt.dtype != RAD_I32 ||
        pos.dtype != RAD_I32 || cu.dtype != RAD_I32 || sc.dtype != RAD_BF16)
        return R4D_NO(RAD_E_DTYPE);
    /* This row: 128-wide keys read 16 bytes at a time, q read as bf16 pairs. */
    if (Dm != 128 || ratio > 16) return R4D_NO(RAD_E_UNSUPPORTED);
    const long ps = lin_stride(pos), cs = lin_stride(cu);
    if (ps < 0 || cs < 0 || q.stride[1] != 1 || (q.stride[0] & 1) || !aligned(q.data, 4) ||
        !aligned(kc.data, 16) || q.shape[0] > INT_MAX || sc.shape[1] > INT_MAX / 2 ||
        r4d_numel(&cu) - 1 > INT_MAX || bt.shape[1] > INT_MAX || H > INT_MAX / 128)
        return R4D_NO(RAD_E_SHAPE);
    if (sc.shape[0] > INT_MAX || (part > 0 && part > (long long)INT_MAX)) return R4D_NO(RAD_E_SHAPE);
    /* A run (row_part p) scores step rows [p R, p R + R), R the score's rows: q, w and pos are read
     * from row p R on; the rows the step does not hold score -inf. */
    const long row0 = part > 0 ? (long)part * (long)sc.shape[0] : 0;
    const long Mv = row0 < (long)q.shape[0]
                        ? ((long)q.shape[0] - row0 < (long)sc.shape[0] ? (long)q.shape[0] - row0
                                                                       : (long)sc.shape[0])
                        : 0;
    const long r0 = Mv > 0 ? row0 : 0;               /* no pointer past the operands */
    DsScoreB16Args s{};
    s.q = (const uint16_t*)q.data + r0 * (long)q.stride[0];
    s.w = (const uint16_t*)w.data + r0 * (long)w.stride[0];
    s.kc = (const uint16_t*)kc.data;
    s.bt = (const int*)bt.data;
    s.pos = (const int*)pos.data + r0 * ps;
    s.cu = (const int*)cu.data;
    s.sc = (uint16_t*)sc.data;
    s.row0 = row0;
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
    s.M = (int)sc.shape[0];
    s.Mv = (int)Mv;
    s.C = (int)sc.shape[1];
    s.H = (int)H;
    s.ratio = (int)ratio;
    s.cbs = (int)kc.shape[2];
    s.slot = (int)(Dm / ratio);
    s.wscale = wscale;
    /* Prefill heights on WMMA (up to 32 heads, 16-byte q rows); the decode band on the scalar
     * kernel below (libref's bits), and every height under ..._EXACT_TALL. */
    if (s.M > kWmRows && !ds_exact_tall() && H <= 32 && s.qr % 8 == 0 && aligned(s.q, 16) &&
        (s.M + 7) / 8 <= 65535) {
        const dim3 grid((unsigned)((s.C + kIwEB - 1) / kIwEB), (unsigned)((s.M + 7) / 8));
        if (H > 16) return ds_b16_path(R4D_EMIT(stream, ds41_index_score_b16_wm_kernel<2>, grid, dim3(256), 0u, s), 32u);
        return ds_b16_path(R4D_EMIT(stream, ds41_index_score_b16_wm_kernel<1>, grid, dim3(256), 0u, s), 16u);
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
        case 1: return ds_b16_path(R4D_EMIT(stream, ds41_index_score_b16_kernel<1>, grid, dim3(256), lds, s), 1u);
        case 2: return ds_b16_path(R4D_EMIT(stream, ds41_index_score_b16_kernel<2>, grid, dim3(256), lds, s), 2u);
        case 4: return ds_b16_path(R4D_EMIT(stream, ds41_index_score_b16_kernel<4>, grid, dim3(256), lds, s), 4u);
        default: return ds_b16_path(R4D_EMIT(stream, ds41_index_score_b16_kernel<8>, grid, dim3(256), lds, s), 8u);
    }
}


int index_candidates_b16_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 3 || !a->t) return RAD_E_INVAL;
    const RadTensor &sc = a->t[0], &pos = a->t[1], &keep = a->t[2];
    const long long block = r4d_geti_or(a, "block", 0), kb = r4d_geti_or(a, "topk_blocks", 0);
    const long long ratio = r4d_geti_or(a, "ratio", 0);
    if (!sc.data || !pos.data || !keep.data) return R4D_NO(RAD_E_INVAL);
    if (block < 1 || kb < 1 || ratio < 1 || sc.rank != 2 || sc.shape[0] < 1 || sc.shape[1] < 1 ||
        r4d_numel(&pos) != sc.shape[0] || keep.rank != 2 || keep.shape[0] != sc.shape[0] ||
        keep.shape[1] != keep_words(sc.shape[1], block))
        return R4D_NO(RAD_E_SHAPE);
    if (sc.dtype != RAD_BF16 || pos.dtype != RAD_I32 || keep.dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);
    const long ps = lin_stride(pos);
    if (ps < 0 || sc.shape[0] > INT_MAX || sc.shape[1] > INT_MAX / 2 || block > INT_MAX)
        return R4D_NO(RAD_E_SHAPE);
    DsCandB16Args c{};
    c.sc = (const uint16_t*)sc.data;
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
    return ds_b16_path(R4D_EMIT(stream, ds41_index_candidates_b16_kernel, dim3(g), dim3(kDsSelThreads), 0, c), 512u);
}


int index_topk_b16_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 4 || !a->t) return RAD_E_INVAL;
    const RadTensor &sc = a->t[0], &pos = a->t[2], &sel = a->t[3];
    const RadTensor* keep = a->t[1].data ? &a->t[1] : nullptr;
    long long topk, ratio;
    if (index_topk_params(a, &topk, &ratio) != RAD_OK) return R4D_NO(RAD_E_SHAPE);
    const long long block = r4d_geti_or(a, "block", 0);
    if (!sc.data || !pos.data || !sel.data || sc.rank != 2 || sc.shape[0] < 1 ||
        r4d_numel(&pos) != sc.shape[0] || sel.rank != 2 || sel.shape[0] != sc.shape[0] ||
        sel.shape[1] != topk ||
        (keep && (block < 1 || block > INT_MAX || keep->rank != 2 || keep->shape[0] != sc.shape[0] ||
                  keep->shape[1] != keep_words(sc.shape[1], block))))
        return R4D_NO(RAD_E_SHAPE);
    if (sc.dtype != RAD_BF16 || pos.dtype != RAD_I32 || sel.dtype != RAD_I32 ||
        (keep && keep->dtype != RAD_I32))
        return R4D_NO(RAD_E_DTYPE);
    const long ps = lin_stride(pos);
    if (ps < 0 || sc.shape[0] > INT_MAX || sc.shape[1] < 0 || sc.shape[1] > INT_MAX / 2 ||
        topk > INT_MAX / 2)
        return R4D_NO(RAD_E_SHAPE);
    DsTopkB16Args t{};
    t.sc = (const uint16_t*)sc.data;
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
    t.block = keep ? (int)block : 1;
    t.bshift = -1;
    for (int sh = 0; sh < 31; ++sh)
        if ((1LL << sh) == (long long)t.block) t.bshift = sh;
    /* Few rows over a long row: chunks in parallel, then one merge a row. The same selection, so
     * the same bits, as the one-block-a-row kernel. */
    const int64_t need = t.M <= kDsSplitRows ? topk_split_bytes(t.M, t.C, t.K) : 0;
    if (need > 0 && a->scratch && a->scratch_bytes >= need && aligned(a->scratch, 8)) {
        DsSplitB16Args sp{};
        sp.t = t;
        sp.chunk = kDsChunk;
        sp.nch = (t.C + kDsChunk - 1) / kDsChunk;
        sp.cand = (unsigned long long*)a->scratch;
        sp.cnt = (int*)((unsigned char*)a->scratch + (int64_t)t.M * sp.nch * t.K * 8);
        const int rc = R4D_EMIT(stream, ds41_index_topk_b16_split1_kernel,
                                dim3((unsigned)sp.nch, (unsigned)t.M), dim3(kDsSelThreads), 0, sp);
        if (rc != RAD_OK) return rc;
        ds_b16_path(rc,128u);
        /* Original merger reads candidate keys/positions, never the score plane.
         * Carry only the fields it consumes; no mis-typed float pointer to BF16 storage. */
        DsSplitArgs merged{};
        merged.cand=sp.cand;merged.cnt=sp.cnt;merged.nch=sp.nch;merged.chunk=sp.chunk;
        merged.t.M=t.M;merged.t.K=t.K;merged.t.sel=t.sel;merged.t.selr=t.selr;merged.t.selc=t.selc;
        return ds_b16_path(R4D_EMIT(stream, ds41_index_topk_split2_kernel, dim3((unsigned)t.M),
                        dim3(kDsSelThreads), 0, merged),256u);
    }
    const unsigned g = (unsigned)(t.M < 65535 ? t.M : 65535);
    return ds_b16_path(R4D_EMIT(stream, ds41_index_topk_b16_kernel, dim3(g), dim3(kDsSelThreads), 0, t), 64u);
}
