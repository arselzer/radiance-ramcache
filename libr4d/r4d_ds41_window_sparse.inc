/* Exact native FP8 WINDOW attention variant. SOURCE-COPIED from the13bfee8b
 * attention reader; the original r4d_ds41_sparse.hip bodies/layouts remain unchanged.
 * Only physical pointer/loading/validation seams differ; every scalar/WMMA arithmetic
 * expression and dispatch choice is retained. Independent all-form byte oracle required.
 * Included INSIDE the original anonymous namespace, after its original launch shims. */
namespace windowpack {
extern __device__ __align__(16) uint16_t kDsZeroKey[512];
__device__ __forceinline__ uint16_t ds_kv_half(const uint16_t* p, int col, uint32_t kind, int width, bool fast = false) {
    if (!p || p == kDsZeroKey) return 0;
    if (kind == 3) {
        if (width != 512 || col < 0 || col >= width) __builtin_trap();
        const auto* row = (const uint8_t*)p; uint16_t out = 0;
        if (!rad_ds41_window_codec::decode_group_bits(row + (col / 32) * 32, row[512 + col / 32], col % 32, &out))
            __builtin_trap();
        return out;
    }
    if (fast && kind) {
        if (kind == 1 && width == 512)
            return rad_ds41_kv_fast::decode_row_bits<rad_ds41_kv_codec::Kind::Fp4E4M3G16, 512>((const uint8_t*)p, col);
        if (kind == 2 && width == 128)
            return rad_ds41_kv_fast::decode_row_bits<rad_ds41_kv_codec::Kind::Fp4Pow2G32, 128>((const uint8_t*)p, col);
        __builtin_trap();
    }
    return kind ? rad_ds41_kv_codec::decode_row_bits(kind, (const uint8_t*)p, width, col) : p[col];
}
__device__ __forceinline__ uint32_t ds_kv_word(const uint16_t* p, int col, uint32_t kind, int width, bool fast = false) {
    if (!p || p == kDsZeroKey) return 0;
    if (!kind) return *(const uint32_t*)(p + col);
    return (uint32_t)ds_kv_half(p, col, kind, width, fast) | ((uint32_t)ds_kv_half(p, col + 1, kind, width, fast) << 16);
}
__device__ __forceinline__ uint4 ds_kv_vec8(const uint16_t* p, int col, uint32_t kind, int width, bool fast = false) {
    if (!p || p == kDsZeroKey) return make_uint4(0, 0, 0, 0);
    if (!kind) return *(const uint4*)(p + col);
    if (fast && kind != 3) {
        uint16_t v[8];
        if (kind == 1 && width == 512)
            rad_ds41_kv_fast::decode_row8_bits<rad_ds41_kv_codec::Kind::Fp4E4M3G16, 512>((const uint8_t*)p, col, v);
        else if (kind == 2 && width == 128)
            rad_ds41_kv_fast::decode_row8_bits<rad_ds41_kv_codec::Kind::Fp4Pow2G32, 128>((const uint8_t*)p, col, v);
        else __builtin_trap();
        return make_uint4((uint32_t)v[0] | ((uint32_t)v[1] << 16), (uint32_t)v[2] | ((uint32_t)v[3] << 16),
                          (uint32_t)v[4] | ((uint32_t)v[5] << 16), (uint32_t)v[6] | ((uint32_t)v[7] << 16));
    }
    return make_uint4(ds_kv_word(p, col, kind, width), ds_kv_word(p, col + 2, kind, width),
                     ds_kv_word(p, col + 4, kind, width), ds_kv_word(p, col + 6, kind, width));
}

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
    uint32_t kv_format, window_format;
    bool kv_fast;
    long packed_stride, window_stride;
};
__device__ __forceinline__ const uint16_t* ds_comp_ptr(const DsAttnArgs& a, long blk, long c) {
    return a.kv_format ? (const uint16_t*)((const uint8_t*)a.cc + blk * a.packed_stride)
                      : a.cc + (blk * a.cbs + (c * a.ratio) % a.cbs) * (long)(512 / a.ratio);
}

__device__ __forceinline__ const uint16_t* ds_window_ptr(const DsAttnArgs& a, long blk, long row) {
    return a.window_format ? (const uint16_t*)((const uint8_t*)a.wc + blk * a.window_stride + row * 528)
                           : a.wc + (blk * a.bs + row) * 512L;
}

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
                    if (blk >= 0 && blk < a.nwblk) ptr = ds_window_ptr(a, blk, r % a.bs);
                }
            } else if (k < a.nk) {
                const long c = a.sel[m * a.selr + (long)(k - a.win) * a.selc];
                if (c >= 0 && (c * a.ratio) / a.cbs < a.ncbt) {
                    const long blk = a.cbt[(long)s * a.cbtr + ((c * a.ratio) / a.cbs) * a.cbtc];
                    if (blk >= 0 && blk < a.ncblk)
                        ptr = ds_comp_ptr(a, blk, c);
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
                    for (int i = 0; i < 4; ++i) {
                        const int key = kb + i * 8 + (lane >> 2);
                        v[i] = ds_kv_vec8(lp[i], c * 32 + (lane & 3) * 8,
                                         key >= a.win ? a.kv_format : a.window_format, D, a.kv_fast);
                    }
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
                    const uint32_t w = ds_kv_word(ptr, dcol, k + u >= a.win ? a.kv_format : a.window_format, D, a.kv_fast);
                    kv[u][0] = bf16w_lo(w);
                    kv[u][DPT - 1] = bf16w_hi(w);
                } else {
                    kv[u][0] = ptr ? bf16_to_f32(ds_kv_half(ptr, dcol,
                                                        k + u >= a.win ? a.kv_format : a.window_format, D, a.kv_fast)) : 0.0f;
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

/* ================================================================== attn_sparse_sink, decode band
 *
 * Passes of at most kWmRows rows (decode, speculative verify) run the chains of the kernel above,
 * each in its order and with its roundings, so its bits; only which thread runs a chain changes.
 * (Where two NaNs of different sign meet in one add, which one propagates follows the operand order
 * the compiler picked, in the kernel above as here: such outputs are NaN in both, their sign bit
 * may differ.) A workgroup is a (head group of HG, row) of NT threads:
 *   keys     the row's key pointers in LDS (a zero row for no key), q as f32 in LDS, the expf
 *            table, the sinks.
 *   scores   a lane a key, HS heads at a time: each dot in d order, the key read straight from
 *            global 16 bytes at a time, two 128-byte lines in flight, q broadcast from LDS (four
 *            heads: from its bf16 row by scalar loads). Each
 *            wave keeps its 32 keys' maximum per head (floored at -1e30; a NaN score never wins,
 *            as in `mx < v ? v : mx`); the max is order-free, a -0/+0 choice included (s - max
 *            and sink - max give the same exp either way).
 *   softmax  p = glibc expf(s - max), a thread a key.
 *   PV       a thread owns DPT latent columns x HPT heads and sums p k over the keys in key order;
 *            sum p, key by key, in the first wave of each head part. The keys come either straight
 *            from global (PVD: a wave walks 32-key blocks, one block's loads in flight, no barrier)
 *            or through LDS tiles of NT/64 keys (a thread one 16-byte piece, P tiles in flight in
 *            registers, a barrier a tile). A no-key slot adds p = 0 times a zero row: +0, which
 *            leaves a sum that starts at +0 unchanged.
 * No global load may sit under a branch: the wait-count pass then waits for every load in flight
 * (s_wait_loadcnt 0) and the pipelining is gone; loads past the end read the zero row instead. */
__device__ __align__(16) uint16_t kDsZeroKey[kDsAttnD];

/* 16 bytes at p[i] by a global (not flat) load. */
typedef unsigned int ds_v4u __attribute__((ext_vector_type(4)));
__device__ __forceinline__ uint4 ds_ldg16(const uint16_t* p, int i) {
    const ds_v4u v = ((__attribute__((address_space(1))) const ds_v4u*)p)[i];
    return make_uint4(v.x, v.y, v.z, v.w);
}
__device__ __forceinline__ uint4 ds_ldg16_key(const uint16_t* p, int i, uint32_t kind, bool fast) {
    return kind ? ds_kv_vec8(p, i * 8, kind, kDsAttnD, fast) : ds_ldg16(p, i);
}

struct DsDecLds {
    long kp, sp, smx, sden, tab, u, sq, bmax, total;
};
/* nkq: keys rounded up to 32. kp [nkq] pointers, sp [HG][nkq] scores then p, smx / sden [HG] (sden
 * holds the sinks until the den), tab the 32 doubles of the expf table; then, in one region,
 * sq [HG][512] q and bmax [HG][nkq/32] block maxima (scores) or the two key tiles of a tiled PV
 * (2 x nt x 16 bytes; nt = 0 for the direct PV). */
__host__ __device__ constexpr DsDecLds ds_dec_lds(int nkq, int hg, int nt) {
    DsDecLds l{};
    l.kp = 0;
    l.sp = (long)nkq * 8;
    l.smx = l.sp + (long)hg * nkq * 4;
    l.sden = l.smx + (long)hg * 4;
    l.tab = (l.sden + (long)hg * 4 + 15) / 16 * 16;
    l.u = l.tab + 32 * 8;
    l.sq = l.u;
    l.bmax = l.sq + (long)hg * kDsAttnD * 4;
    const long sc = l.bmax + (long)hg * (nkq / 32) * 4, pv = l.u + 2L * nt * 16;
    l.total = sc > pv ? sc : pv;
    return l;
}

/* ds_expf with the table read from `tab` (an LDS copy of kDsExp2fTab): the same operations. */
__device__ __forceinline__ float ds_expf_tab(float x, const unsigned long long* tab) {
    const uint32_t ux = __float_as_uint(x);
    const uint32_t abstop = (ux >> 20) & 0x7ffu;
    if (abstop >= 0x42bu) {
        if (ux == 0xff800000u) return 0.0f;
        if (abstop >= 0x7f8u) return x + x;
        if (x > 0x1.62e42ep6f) return kDsInf;
        if (x < -0x1.9fe368p6f) return 0.0f;
    }
    const double xd = (double)x;
    double z = 0x1.71547652b82fep+5 * xd;
    double kd = z + 0x1.8p+52;
    const unsigned long long ki = (unsigned long long)__double_as_longlong(kd);
    kd -= 0x1.8p+52;
    const double r = z - kd;
    unsigned long long t = tab[ki % 32];
    t += ki << 47;
    const double s = __longlong_as_double((long long)t);
    z = 0x1.c6af84b912394p-20 * r + 0x1.ebfce50fac4f3p-13;
    const double r2 = r * r;
    double y = 0x1.62e42ff0c52d6p-6 * r + 1.0;
    y = z * r2 + y;
    y = y * s;
    return (float)y;
}

/* An offset of 0 the compiler cannot see through, made after the running sums and before the
 * next 16 key bytes are unpacked: the q reads and key conversions of 8 columns cannot be hoisted
 * above the previous columns' fmas (HS x 64 live floats spilled otherwise). */
template <int HS>
__device__ __forceinline__ int ds_dec_step(const float (&acc)[HS], uint32_t (&w)[4]) {
    int z;
    if constexpr (HS == 1)
        asm volatile("v_mov_b32 %0, 0" : "=v"(z), "+v"(w[0]), "+v"(w[1]), "+v"(w[2]), "+v"(w[3]) : "v"(acc[0]));
    else if constexpr (HS == 2)
        asm volatile("v_mov_b32 %0, 0" : "=v"(z), "+v"(w[0]), "+v"(w[1]), "+v"(w[2]), "+v"(w[3])
                     : "v"(acc[0]), "v"(acc[1]));
    else
        asm volatile("v_mov_b32 %0, 0" : "=v"(z), "+v"(w[0]), "+v"(w[1]), "+v"(w[2]), "+v"(w[3])
                     : "v"(acc[0]), "v"(acc[1]), "v"(acc[2]), "v"(acc[3]));
    return z;
}

/* 64 latent columns of one key (8 x 16 bytes) into the HS dots, in d order. With more than one
 * head, step i+1's q reads go out (behind the fence of step i-1's sums) before step i's fmas. */
template <int HS>
__device__ __forceinline__ void ds_dec_dot(float (&acc)[HS], const uint4 (&r)[8], const float* qp) {
    static_assert(HS == 1 || HS == 2 || HS == 4, "head subgroup");
    if constexpr (HS == 1) {
        /* one head: 64 hoisted q floats fit (no spill), and the hoist hides the LDS latency */
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t wk[4] = { r[i].x, r[i].y, r[i].z, r[i].w };
            const float4 qa = *(const float4*)(qp + i * 8);
            const float4 qb = *(const float4*)(qp + i * 8 + 4);
            const float qv[8] = { qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w };
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                acc[0] = __builtin_fmaf(qv[2 * e], bf16w_lo(wk[e]), acc[0]);
                acc[0] = __builtin_fmaf(qv[2 * e + 1], bf16w_hi(wk[e]), acc[0]);
            }
        }
    } else {
        float4 qc[HS][2], qn[HS][2];
        uint32_t wc[4] = { r[0].x, r[0].y, r[0].z, r[0].w }, wn[4];
        {
            const float* qi = qp + ds_dec_step<HS>(acc, wc);
#pragma unroll
            for (int h = 0; h < HS; ++h) {
                qc[h][0] = *(const float4*)(qi + h * kDsAttnD);
                qc[h][1] = *(const float4*)(qi + h * kDsAttnD + 4);
            }
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (i + 1 < 8) {
                wn[0] = r[i + 1].x; wn[1] = r[i + 1].y; wn[2] = r[i + 1].z; wn[3] = r[i + 1].w;
                const float* qi = qp + ds_dec_step<HS>(acc, wn) + (i + 1) * 8;
#pragma unroll
                for (int h = 0; h < HS; ++h) {
                    qn[h][0] = *(const float4*)(qi + h * kDsAttnD);
                    qn[h][1] = *(const float4*)(qi + h * kDsAttnD + 4);
                }
            }
#pragma unroll
            for (int h = 0; h < HS; ++h) {
                const float qv[8] = { qc[h][0].x, qc[h][0].y, qc[h][0].z, qc[h][0].w,
                                      qc[h][1].x, qc[h][1].y, qc[h][1].z, qc[h][1].w };
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    acc[h] = __builtin_fmaf(qv[2 * e], bf16w_lo(wc[e]), acc[h]);
                    acc[h] = __builtin_fmaf(qv[2 * e + 1], bf16w_hi(wc[e]), acc[h]);
                }
            }
            if (i + 1 < 8) {
#pragma unroll
                for (int h = 0; h < HS; ++h) { qc[h][0] = qn[h][0]; qc[h][1] = qn[h][1]; }
#pragma unroll
                for (int e = 0; e < 4; ++e) wc[e] = wn[e];
            }
        }
    }
}

/* The HS-head dot of ds_dec_dot with q read straight from its bf16 row by scalar loads (wave-uniform
 * addresses: qw the row's words at this chunk, ho[h] head h's word offset), so the q values take
 * neither LDS bandwidth nor VGPRs. The same fmas in the same order. */
template <int HS>
__device__ __forceinline__ void ds_dec_dot_s(float (&acc)[HS], const uint4 (&r)[8], const uint32_t* qw,
                                             const int (&ho)[HS]) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint32_t wk[4] = { r[i].x, r[i].y, r[i].z, r[i].w };
        const int z = __builtin_amdgcn_readfirstlane(ds_dec_step<HS>(acc, wk));
        const uint32_t* qi = qw + z + i * 4;
#pragma unroll
        for (int h = 0; h < HS; ++h) {
            const uint32_t* qh = qi + ho[h];
            const uint32_t qq[4] = { qh[0], qh[1], qh[2], qh[3] };
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                acc[h] = __builtin_fmaf(bf16w_lo(qq[e]), bf16w_lo(wk[e]), acc[h]);
                acc[h] = __builtin_fmaf(bf16w_hi(qq[e]), bf16w_hi(wk[e]), acc[h]);
            }
        }
    }
}

/* Four keys (their DPT columns in kw) into the HPT heads' sums p k, and sum p in the den wave,
 * each chain in key order (libref's mul, then add). */
template <int HPT, int DPT>
__device__ __forceinline__ void ds_dec_pv4(float (&acc)[HPT][DPT], float (&den)[HPT], const uint32_t (&kw)[4],
                                           const float* ph, int nkq, int k0, bool dw) {
    float kf[4][DPT];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
        kf[u][0] = bf16w_lo(kw[u]);
        if constexpr (DPT == 2) kf[u][DPT - 1] = bf16w_hi(kw[u]);
    }
#pragma unroll
    for (int j = 0; j < HPT; ++j) {
        const float4 pv = *(const float4*)(ph + (long)j * nkq + k0);
        const float pk[4] = { pv.x, pv.y, pv.z, pv.w };
#pragma unroll
        for (int u = 0; u < 4; ++u) {
#pragma unroll
            for (int d = 0; d < DPT; ++d) {
                const float prod = pk[u] * kf[u][d];
                acc[j][d] = acc[j][d] + prod;
            }
        }
        if (dw) {
#pragma unroll
            for (int u = 0; u < 4; ++u) den[j] = den[j] + pk[u];
        }
    }
}

/* The DPT columns at dcol of the 32 keys whose pointers lanes 0..31 hold (a wave-uniform pointer a
 * key, through readlane), straight from global. */
template <int DPT>
__device__ __forceinline__ void ds_dec_pvld(uint32_t (&kv)[32], const uint16_t* lp, int dcol,
                                           uint32_t kind, uint32_t window_kind, int win, int key0, bool fast) {
    const unsigned long long v = (unsigned long long)lp;
    const int lo = (int)(uint32_t)v, hi = (int)(uint32_t)(v >> 32);
#pragma unroll
    for (int u = 0; u < 32; ++u) {
        const unsigned long long pu = ((unsigned long long)(uint32_t)__builtin_amdgcn_readlane(hi, u) << 32) |
                                      (uint32_t)__builtin_amdgcn_readlane(lo, u);
        const uint16_t* p = (const uint16_t*)pu;
        const uint32_t source_kind = key0 + u >= win ? kind : window_kind;
        if (source_kind) {
            if constexpr (DPT == 2) kv[u] = ds_kv_word(p, dcol, source_kind, kDsAttnD, fast);
            else kv[u] = ds_kv_half(p, dcol, source_kind, kDsAttnD, fast);
        } else {
            if constexpr (DPT == 2) kv[u] = *(__attribute__((address_space(1))) const uint32_t*)(p + dcol);
            else kv[u] = ldg_u16(p + dcol);
        }
    }
}

template <int HPT, int DPT>
__device__ __forceinline__ void ds_dec_pvsum(float (&acc)[HPT][DPT], float (&den)[HPT], const uint32_t (&kv)[32],
                                             const float* ph, int nkq, int k0, bool dw) {
#pragma unroll
    for (int u0 = 0; u0 < 32; u0 += 4) {
        const uint32_t kw[4] = { kv[u0], kv[u0 + 1], kv[u0 + 2], kv[u0 + 3] };
        ds_dec_pv4<HPT, DPT>(acc, den, kw, ph, nkq, k0 + u0, dw);
    }
}


template <int HG, int NT, int DPT, int P, bool PVD>
__global__ __launch_bounds__(NT) void ds41_attn_dec_kernel(DsAttnArgs a) {
    constexpr int D = kDsAttnD, NW = NT / 32, NCG = D / DPT, TK = NT / 64;
    constexpr int NHP = (NT / NCG) < HG ? (NT / NCG) : HG;   /* head parts in PV */
    constexpr int HPT = HG / NHP;                           /* heads a PV thread */
    constexpr int HS = HG < 4 ? HG : 4, NSG = HG / HS;      /* heads a score task, tasks a key block */
    constexpr bool QS = HS == 4;                            /* q by scalar loads in the scores */
    static_assert(HG % HS == 0, "head subgroups");
    static_assert(NT % 64 == 0 && 32 % TK == 0 && TK % 4 == 0 && NT >= NCG && HG % NHP == 0 &&
                  (DPT == 1 || DPT == 2) && P >= 1, "layout");
    const int nkq = (a.nk + 31) / 32 * 32, nkb = nkq / 32, ntl = nkq / TK;
    const DsDecLds L = ds_dec_lds(nkq, HG, PVD ? 0 : NT);
    const uint16_t** kp = (const uint16_t**)(r4d_ds41_sparse_dyn + L.kp);
    float* sp = (float*)(r4d_ds41_sparse_dyn + L.sp);
    float* sq = (float*)(r4d_ds41_sparse_dyn + L.sq);
    float* bmax = (float*)(r4d_ds41_sparse_dyn + L.bmax);
    float* smx = (float*)(r4d_ds41_sparse_dyn + L.smx);
    float* sden = (float*)(r4d_ds41_sparse_dyn + L.sden);
    unsigned long long* tab = (unsigned long long*)(r4d_ds41_sparse_dyn + L.tab);
    unsigned char* tiles = r4d_ds41_sparse_dyn + L.u;
    const int tid = threadIdx.x, lane = tid & 31;
    const int wave = __builtin_amdgcn_readfirstlane((int)(tid >> 5));
    const int hg0 = blockIdx.x * HG;
    const int hmax = min(HG, a.H - hg0);
    const long m = blockIdx.y;
    const uint16_t* zero = kDsZeroKey;

    /* ---- the row's keys, in libref's order: the window, then the selections */
    {
        const int s = ds_seq_of_row(a.cu, a.cs, a.S, m);
        const long p = a.pos[m * a.ps];
        long lr = (long)a.cu[(long)(s + 1) * a.cs] - 1;
        lr = lr < 0 ? 0 : (lr >= a.M ? a.M - 1 : lr);
        const long last = a.pos[lr * a.ps];
        const long first = last + 1 - (long)a.wused[(long)s * a.wus];
        for (int k = tid; k < nkq; k += NT) {
            const uint16_t* ptr = zero;
            if (k < a.win) {
                const long j = p - a.win + 1 + k, r = j - first;
                if (j >= 0 && r >= 0 && r / a.bs < a.nwbt) {
                    const long blk = a.wbt[(long)s * a.wbtr + (r / a.bs) * a.wbtc];
                    if (blk >= 0 && blk < a.nwblk) ptr = ds_window_ptr(a, blk, r % a.bs);
                }
            } else if (k < a.nk) {
                const long c = a.sel[m * a.selr + (long)(k - a.win) * a.selc];
                if (c >= 0 && (c * a.ratio) / a.cbs < a.ncbt) {
                    const long blk = a.cbt[(long)s * a.cbtr + ((c * a.ratio) / a.cbs) * a.cbtc];
                    if (blk >= 0 && blk < a.ncblk)
                        ptr = ds_comp_ptr(a, blk, c);
                }
            }
            kp[k] = ptr;
        }
        const uint32_t* qrow = (const uint32_t*)(a.q + m * a.qr) + (long)hg0 * (D / 2);
        for (int i = tid; i < (QS ? 0 : HG * (D / 2)); i += NT) {
            const int h = i / (D / 2), c = i % (D / 2);
            const uint32_t w = qrow[(long)(h < hmax ? h : 0) * (D / 2) + c];
            *(float2*)(sq + h * D + 2 * c) = make_float2(bf16w_lo(w), bf16w_hi(w));
        }
        if (tid < 32) tab[tid] = kDsExp2fTab[tid];
        if (tid < hmax) sden[tid] = a.sink[(long)(hg0 + tid) * a.sks];   /* the sinks, until the den */
    }
    lds_barrier();

    /* ---- scores: a lane a key, every dot in d order. A wave task is (32 keys, HS heads); the
     * NSG tasks of a key block go to neighbouring waves, so the block's second read hits L0.
     * Four heads read q by scalar loads (QS; LDS-bound otherwise), fewer from LDS. */
    for (int t = wave; t < nkb * NSG; t += NW) {
        const int kb = t / NSG, g = t % NSG;
        const int k = kb * 32 + lane;
        const uint16_t* kk = kp[k];
        const bool have = kk != zero;
        const float* qg = sq + g * HS * D;
        float acc[HS];
#pragma unroll
        for (int h = 0; h < HS; ++h) acc[h] = 0.0f;
        const uint32_t* qw = (const uint32_t*)(a.q + m * a.qr) + (long)hg0 * (D / 2);
        int ho[HS];
#pragma unroll
        for (int h = 0; h < HS; ++h) ho[h] = (g * HS + h < hmax ? g * HS + h : 0) * (D / 2);
        uint4 ra[8], rb[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) ra[i] = ds_ldg16_key(kk, i, k >= a.win ? a.kv_format : a.window_format, a.kv_fast);
        /* No branch around a load (the wait counts would fall back to 0): the last pass reloads
         * its own first half, unused. */
#pragma unroll 1
        for (int c = 0; c < D / 8; c += 16) {
#pragma unroll
            for (int i = 0; i < 8; ++i) rb[i] = ds_ldg16_key(kk, c + 8 + i, k >= a.win ? a.kv_format : a.window_format, a.kv_fast);
            if constexpr (QS) ds_dec_dot_s<HS>(acc, ra, qw + c * 4, ho);
            else ds_dec_dot<HS>(acc, ra, qg + c * 8);
            const int cn = c + 16 < D / 8 ? c + 16 : c;
#pragma unroll
            for (int i = 0; i < 8; ++i) ra[i] = ds_ldg16_key(kk, cn + i, k >= a.win ? a.kv_format : a.window_format, a.kv_fast);
            if constexpr (QS) ds_dec_dot_s<HS>(acc, rb, qw + (c + 8) * 4, ho);
            else ds_dec_dot<HS>(acc, rb, qg + (c + 8) * 8);
        }
#pragma unroll
        for (int h = 0; h < HS; ++h) {
            const int hh = g * HS + h;
            const float v = have ? acc[h] * a.scale : -kDsInf;
            sp[(long)hh * nkq + k] = v;
            const float t2 = r4d_wave_max(-1e30f < v ? v : -1e30f);
            if (lane == 0) bmax[hh * nkb + kb] = t2;
        }
    }

    /* PV through LDS tiles: the first tiles' loads go out now, under the softmax. A thread carries
     * one 16-byte piece (key lk of the tile, piece lc) of each of P tiles. No load sits under a
     * branch (the wait counts would fall back to 0): past the last tile they read the zero row. */
    const int lk = tid >> 6, lc = tid & 63;
    uint4 rg[PVD ? 1 : P];
    if constexpr (!PVD) {
#pragma unroll
        for (int s = 0; s < P; ++s) {
            const uint16_t* pp = kp[min(s, ntl - 1) * TK + lk];
            rg[s] = ds_ldg16_key(s < ntl ? pp : zero, lc,
                                  min(s, ntl - 1) * TK + lk >= a.win ? a.kv_format : a.window_format, a.kv_fast);
        }
    }
    lds_barrier();

    /* ---- per head: the max, then p in place (0 for no key) */
#pragma unroll 1
    for (int h = 0; h < HG; ++h) {
        float t = -1e30f;
        for (int j = lane; j < nkb; j += 32) t = fmaxf(t, bmax[h * nkb + j]);
        const float mx = r4d_wave_max(t);
        if (tid == 0) smx[h] = mx;
        float* sh = sp + (long)h * nkq;
        for (int k = tid; k < nkq; k += NT) sh[k] = kp[k] != zero ? ds_expf_tab(sh[k] - mx, tab) : 0.0f;
    }
    lds_barrier();                                       /* sq and bmax are dead: tiles may land */

    /* ---- sum p k and sum p, key by key in libref's order */
    const int cg = tid % NCG, hp = tid / NCG;
    const int dcol = cg * DPT;
    float acc[HPT][DPT], den[HPT];
#pragma unroll
    for (int j = 0; j < HPT; ++j) {
        den[j] = 0.0f;
#pragma unroll
        for (int d = 0; d < DPT; ++d) acc[j][d] = 0.0f;
    }
    const bool pvt = hp < NHP;                           /* wave-uniform: NCG is a multiple of 32 */
    const bool dw = cg < 32;
    const float* ph = sp + (long)(pvt ? hp * HPT : 0) * nkq;
    if constexpr (PVD) {
        /* Straight from global: a wave walks 32-key blocks, lane L holding block key L's pointer
         * (handed to the whole wave by readlane), one block's loads in flight while the previous
         * block is summed. Past the last block the loads read the zero row. */
        if (pvt) {
            uint32_t kv[2][32];
            const uint16_t* lp = kp[lane];
            ds_dec_pvld<DPT>(kv[0], lp, dcol, a.kv_format, a.window_format, a.win, 0, a.kv_fast);
#pragma unroll 1
            for (int kb = 0; kb < nkb; kb += 2) {
                lp = kb + 1 < nkb ? kp[(kb + 1) * 32 + lane] : zero;
                ds_dec_pvld<DPT>(kv[1], lp, dcol, a.kv_format, a.window_format, a.win, (kb + 1) * 32, a.kv_fast);
                ds_dec_pvsum<HPT, DPT>(acc, den, kv[0], ph, nkq, kb * 32, dw);
                lp = kb + 2 < nkb ? kp[(kb + 2) * 32 + lane] : zero;
                ds_dec_pvld<DPT>(kv[0], lp, dcol, a.kv_format, a.window_format, a.win, (kb + 2) * 32, a.kv_fast);
                if (kb + 1 < nkb) ds_dec_pvsum<HPT, DPT>(acc, den, kv[1], ph, nkq, (kb + 1) * 32, dw);
            }
        }
    } else {
        *(uint4*)(tiles + lk * (D * 2) + lc * 16) = rg[0];
        {
            const uint16_t* pp = kp[min(P, ntl - 1) * TK + lk];
            rg[0] = ds_ldg16_key(P < ntl ? pp : zero, lc,
                                  min(P, ntl - 1) * TK + lk >= a.win ? a.kv_format : a.window_format, a.kv_fast);
        }
        lds_barrier();
#pragma unroll 1
        for (int t0 = 0; t0 < ntl; t0 += P) {
#pragma unroll
            for (int s = 0; s < P; ++s) {
                const int t = t0 + s;
                /* the pointer of the ring's next load first, off the tail of the step */
                const uint16_t* pp = kp[min(t + 1 + P, ntl - 1) * TK + lk];
                if (pvt && t < ntl) {
                    const unsigned char* tb = tiles + (t & 1) * (TK * D * 2) + dcol * 2;
#pragma unroll
                    for (int u0 = 0; u0 < TK; u0 += 4) {
                        uint32_t kw[4];
#pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            if constexpr (DPT == 2) kw[u] = *(const uint32_t*)(tb + (u0 + u) * (D * 2));
                            else kw[u] = *(const uint16_t*)(tb + (u0 + u) * (D * 2));
                        }
                        ds_dec_pv4<HPT, DPT>(acc, den, kw, ph, nkq, t * TK + u0, dw);
                    }
                }
                /* tile t+1 into the other buffer (read last before the previous barrier) */
                const int sn = (s + 1) % P;
                *(uint4*)(tiles + ((t + 1) & 1) * (TK * D * 2) + lk * (D * 2) + lc * 16) = rg[sn];
                rg[sn] = ds_ldg16_key(t + 1 + P < ntl ? pp : zero, lc,
                                       min(t + 1 + P, ntl - 1) * TK + lk >= a.win ? a.kv_format : a.window_format, a.kv_fast);
                lds_barrier();
            }
        }
    }
    if (pvt && cg == 0) {
#pragma unroll
        for (int j = 0; j < HPT; ++j) {
            const int h = hp * HPT + j;
            if (h < hmax) sden[h] = den[j] + ds_expf_tab(sden[h] - smx[h], tab);
        }
    }
    lds_barrier();
    if (pvt) {
        uint16_t* orow = a.out + m * a.outr + (long)hg0 * D + dcol;
#pragma unroll
        for (int j = 0; j < HPT; ++j) {
            const int h = hp * HPT + j;
            if (h >= hmax) break;
            const float dn = sden[h];
            if constexpr (DPT == 2) {
                const uint32_t lo = ds_bf16_bits(acc[j][0] / dn), hi = ds_bf16_bits(acc[j][DPT - 1] / dn);
                *(uint32_t*)(orow + (long)h * D) = lo | (hi << 16);
            } else {
                orow[(long)h * D] = (uint16_t)ds_bf16_bits(acc[j][0] / dn);
            }
        }
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
 * and the softmax online; the old kernel stays for the decode band. A key
 * with no entry reads a zero row and scores -inf (0 * NaN from a recycled slot would be NaN). */
typedef __bf16 ds_v8bf __attribute__((ext_vector_type(8)));
typedef float ds_v8f __attribute__((ext_vector_type(8)));
typedef short ds_v8s __attribute__((ext_vector_type(8)));
typedef __attribute__((address_space(1))) ds_v8s ds_gv8s;

constexpr int kWmRows = 48;                          /* passes up to this many rows stay scalar */
constexpr int kWmWaves = 4;
constexpr int kWmCols = kDsAttnD / kWmWaves;         /* latent columns a wave */
constexpr int kWmSbufBytes = 2 * kWmWaves * 32 * 8 * 4;

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
/* Packed bytes cannot use global_load_tr. Reconstruct the exact eight BF16s a lane would
 * load, then perform that instruction's documented 8x8 transpose as integer shuffles. */
__device__ __forceinline__ uint4 ds_load_tr_key(const uint16_t* p, int col, uint32_t kind, bool packed_pass, bool fast) {
    if (!packed_pass) return ds_load_tr(p + col);
    const int lane = threadIdx.x & 31, element = lane & 7, base = lane & ~7;
    const unsigned long long address = (unsigned long long)p;
    uint32_t out[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint32_t lo = __shfl((uint32_t)address, base + j, 32);
        const uint32_t hi = __shfl((uint32_t)(address >> 32), base + j, 32);
        const uint16_t* source = (const uint16_t*)(((unsigned long long)hi << 32) | lo);
        const int source_col = __shfl(col, base + j, 32);
        const uint32_t source_kind = __shfl(kind, base + j, 32);
        out[j] = ds_kv_half(source, source_col + element, source_kind, kDsAttnD, fast);
    }
    return make_uint4(out[0] | (out[1] << 16), out[2] | (out[3] << 16),
                     out[4] | (out[5] << 16), out[6] | (out[7] << 16));
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
                    if (blk >= 0 && blk < a.nwblk) ptr = ds_window_ptr(a, blk, r % a.bs);
                }
            } else if (k < a.nk) {
                const long c = a.sel[m * a.selr + (long)(k - a.win) * a.selc];
                if (c >= 0 && (c * a.ratio) / a.cbs < a.ncbt) {
                    const long blk = a.cbt[(long)s * a.cbtr + ((c * a.ratio) / a.cbs) * a.cbtc];
                    if (blk >= 0 && blk < a.ncblk)
                        ptr = ds_comp_ptr(a, blk, c);
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
            const uint16_t* pk = kp[kq];
#pragma unroll
            for (int i = 0; i < 8; ++i) kn[i] = ds_kv_vec8(pk, c0 + 64 * hf + 8 * i,
                                                          kq >= a.win ? a.kv_format : a.window_format, D, a.kv_fast);
        }
        for (int t = 0; t < nkt; ++t) {
            uint4 kt[8];
            {
                const uint16_t* pt = kp[t * 16 + ktr];
#pragma unroll
                for (int j = 0; j < 8; ++j) kt[j] = ds_load_tr_key(pt, dtr + 16 * j,
                                                               t * 16 + ktr >= a.win ? a.kv_format : a.window_format,
                                                               a.kv_format != 0 || a.window_format != 0, a.kv_fast);
            }
            ds_v8f sp = (ds_v8f)(0.0f);
#pragma unroll
            for (int i = 0; i < 8; ++i) sp = ds_wmma(kn[i], qf[i], sp);
            if (t + 1 < nkt) {
                const uint16_t* pk = kp[(t + 1) * 16 + kq];
#pragma unroll
                for (int i = 0; i < 8; ++i) kn[i] = ds_kv_vec8(pk, c0 + 64 * hf + 8 * i,
                                                              (t + 1) * 16 + kq >= a.win ? a.kv_format : a.window_format, D, a.kv_fast);
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


template <int HG, int DS>
int attn_emit(RadStream stream, const DsAttnArgs& s, unsigned gz) {
    const uint32_t lds = (uint32_t)(ds_attn_tile_off(s.nkp, HG) + kDsTileBytes);
    const dim3 grid((unsigned)DS, (unsigned)((s.H + HG - 1) / HG), gz);
    return R4D_EMIT(stream, (ds41_attn_sparse_sink_kernel<HG, DS>), grid, dim3(256), lds, s);
}

template <int HG, int NT, int DPT, bool PVD>
int attn_dec_go(RadStream stream, const DsAttnArgs& s) {
    constexpr int P = 4;                                /* key tiles in flight in PV */
    const int nkq = (s.nk + 31) / 32 * 32;
    const long lds = ds_dec_lds(nkq, HG, PVD ? 0 : NT).total;
    if (lds > 65536) return RAD_E_UNSUPPORTED;
    const dim3 grid((unsigned)((s.H + HG - 1) / HG), (unsigned)s.M);
    return R4D_EMIT(stream, (ds41_attn_dec_kernel<HG, NT, DPT, P, PVD>), grid, dim3(NT), (uint32_t)lds, s);
}

/* The decode band's kernel. The form by how many (row, head) pairs the pass has, any choice giving
 * the same bits (measured on R9700, 32 heads x 640 keys: one row 17 us, 2 rows 27, 3-4 rows 44,
 * 5-8 rows 73, 48 rows 367; the scalar kernel 129-582). On an LDS refusal (long rows), the next
 * smaller form; the one-head form takes every row this op accepts. */
int attn_dec_emit(RadStream stream, const DsAttnArgs& s) {
    if (s.M > 65535 || (s.H + 0L) > 65535L * 8) return RAD_E_UNSUPPORTED;
    const long hr = (long)s.M * s.H;
    int c = hr <= 32 ? 0 : (hr <= 64 ? 1 : (hr <= 128 ? 2 : (hr <= 256 ? 3 : 4)));
    static const int next[7] = { -1, 0, 1, 2, 5, 6, 1 };
    while (c >= 0) {
        int rc = RAD_E_UNSUPPORTED;
        switch (c) {
            case 0: rc = attn_dec_go<1, 1024, 2, true>(stream, s); break;
            case 1: rc = attn_dec_go<2, 1024, 2, true>(stream, s); break;
            case 2: rc = attn_dec_go<4, 1024, 2, true>(stream, s); break;
            case 3: rc = attn_dec_go<8, 1024, 2, true>(stream, s); break;
            case 4: rc = attn_dec_go<8, 512, 2, false>(stream, s); break;
            case 5: rc = attn_dec_go<4, 512, 2, false>(stream, s); break;
            case 6: rc = attn_dec_go<2, 512, 2, false>(stream, s); break;
            default: break;
        }
        if (rc != RAD_E_UNSUPPORTED) return rc;
        c = next[c];
    }
    return RAD_E_UNSUPPORTED;
}

int attn_body(const RadArgs* a, RadStream stream, int oracle_form = -1) {
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
    const uint32_t kind = (uint32_t)r4d_geti_or(a, "kv_format", 0);
    const bool packed = kind != 0;
    const uint32_t window_kind = (uint32_t)r4d_geti_or(a, "window_format", 0);
    const bool packed_window = window_kind != 0;
    if (packed_window && (window_kind != 3 || D != 512)) return R4D_NO(RAD_E_UNSUPPORTED);
    if (packed && (kind != 1 || ratio < 1 || ratio > 2)) return R4D_NO(RAD_E_UNSUPPORTED);
    if (packed && ds_kv_fast_mode() < 0) return R4D_NO(RAD_E_INVAL);
    const float scale = (float)r4d_getf(a, "scale", 0.0);
    const bool comp = ratio > 0;
    if (H < 1 || D < 2 || D > 4096 || win < 1 || ratio < 0 || topk < 0 || !std::isfinite(scale) ||
        scale <= 0 || (comp && (!cc || !cbt || !sel || topk < 1 || D % ratio)))
        return R4D_NO(RAD_E_SHAPE);
    if (!q.data || !wc.data || !wbt.data || !wused.data || !pos.data || !cu.data || !sink.data ||
        !out.data || q.rank != 2 || q.shape[0] < 1 || q.shape[1] != H * D || out.rank != 2 ||
        out.shape[0] != q.shape[0] || out.shape[1] != H * D ||
        (packed_window ? (wc.rank != 2 || wc.shape[1] != 8448 || wc.stride[1] != 1 || wc.stride[0] < 8448)
                       : (wc.rank != 4 || wc.shape[1] != 1 || wc.shape[3] != D)) || wbt.rank != 2 || r4d_numel(&pos) != q.shape[0] || r4d_numel(&sink) != H ||
        sink.dtype != RAD_F32 || r4d_numel(&cu) < 2 || r4d_numel(&wused) < r4d_numel(&cu) - 1)
        return R4D_NO(RAD_E_SHAPE);
    const long long cbs = comp ? (packed ? ratio : cc->shape[2]) : 0;
    if (comp && ((packed ? (cc->rank != 2 || cc->shape[1] != rad_ds41_kv_codec::row_bytes(kind, D) ||
                            cc->stride[1] != 1 || cc->stride[0] < cc->shape[1])
                         : (cc->rank != 4 || cc->shape[1] != 1 || cc->shape[3] * ratio != D || cbs % ratio)) ||
                 cbt->rank != 2 || sel->rank != 2 || sel->shape[0] != q.shape[0] ||
                 sel->shape[1] != topk || cbt->shape[0] != wbt.shape[0]))
        return R4D_NO(RAD_E_SHAPE);
    if (q.dtype != RAD_BF16 || wc.dtype != (packed_window ? RAD_U8 : RAD_BF16) || out.dtype != RAD_BF16 || wbt.dtype != RAD_I32 ||
        wused.dtype != RAD_I32 || pos.dtype != RAD_I32 || cu.dtype != RAD_I32 ||
        (comp && (cc->dtype != (packed ? RAD_U8 : RAD_BF16) || cbt->dtype != RAD_I32 || sel->dtype != RAD_I32)))
        return R4D_NO(RAD_E_DTYPE);
    /* This row: the 512-wide latent, read 16 bytes at a time; at most 1024 + 2048 keys a row. */
    if (D != kDsAttnD || win > 1024 || topk > 2048 || ratio > 64) return R4D_NO(RAD_E_UNSUPPORTED);
    const long S = (long)(r4d_numel(&cu) - 1);
    const long ps = lin_stride(pos), cs = lin_stride(cu), wus = lin_stride(wused), sks = lin_stride(sink);
    if (ps < 0 || cs < 0 || wus < 0 || sks < 0 || q.stride[1] != 1 || out.stride[1] != 1 ||
        (q.stride[0] & 1) || (out.stride[0] & 1) || !aligned(q.data, 4) || !aligned(out.data, 4) ||
        (!packed_window && !aligned(wc.data, 16)) || (comp && !packed && !aligned(cc->data, 16)) || wbt.shape[0] < S ||
        q.shape[0] > INT_MAX || S > INT_MAX || wbt.shape[1] > INT_MAX ||
        (comp && cbt->shape[1] > INT_MAX) || wc.shape[0] < 1 ||
        (!packed_window && (wc.shape[2] < 1 || wc.shape[2] > INT_MAX)) || cbs > INT_MAX)
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
    s.bs = packed_window ? 16 : (int)wc.shape[2];
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
    s.kv_format = kind; s.packed_stride = comp && packed ? (long)cc->stride[0] : 0;
    s.window_format = window_kind; s.window_stride = packed_window ? (long)wc.stride[0] : 0;
    s.kv_fast = packed && ds_kv_fast_mode() == 1;
    /* Unregistered oracle exports force an existing reader without changing normal dispatch. */
    if (oracle_form >= 1 && oracle_form <= 7) {
        switch (oracle_form) {
            case 1: return attn_dec_go<1, 1024, 2, true>(stream, s);
            case 2: return attn_dec_go<2, 1024, 2, true>(stream, s);
            case 3: return attn_dec_go<4, 1024, 2, true>(stream, s);
            case 4: return attn_dec_go<8, 1024, 2, true>(stream, s);
            case 5: return attn_dec_go<8, 512, 2, false>(stream, s);
            case 6: return attn_dec_go<4, 512, 2, false>(stream, s);
            case 7: return attn_dec_go<2, 512, 2, false>(stream, s);
        }
    }
    if (oracle_form >= 9) {
        const unsigned gz = (unsigned)(s.M < 65535 ? s.M : 65535);
        switch (oracle_form) {
            case 9: return attn_emit<8, 1>(stream, s, gz);
            case 10: return attn_emit<4, 1>(stream, s, gz);
            case 11: return attn_emit<2, 1>(stream, s, gz);
            case 12: return attn_emit<1, 1>(stream, s, gz);
            case 13: return attn_emit<2, 2>(stream, s, gz);
            case 14: return attn_emit<1, 2>(stream, s, gz);
        }
    }
    /* Prefill heights on WMMA (16-byte q and out rows); the decode band on the scalar kernels
     * below (libref's bits), and every height under ..._EXACT_TALL. */
    {
        const int nkt = (s.nk + 15) / 16;
        const long lds = ds_wm_sbuf_off(nkt) + kWmSbufBytes;
        const bool wm_ok = s.qr % 8 == 0 && s.outr % 8 == 0 && aligned(q.data, 16) &&
                           aligned(out.data, 16) && lds <= 65536 && (H + 15) / 16 <= 65535;
        if (oracle_form == 8 && !wm_ok) return RAD_E_UNSUPPORTED;
        if (oracle_form == 8 || (oracle_form < 0 && s.M > kWmRows && !ds_exact_tall() && wm_ok)) {
            const unsigned gy = (unsigned)(s.M < 65535 ? s.M : 65535);
            const dim3 grid((unsigned)((H + 15) / 16), gy);
            const float sl2 = (float)((double)scale * 1.4426950408889634);
            return R4D_EMIT(stream, ds41_attn_sparse_sink_wm_kernel<true>, grid, dim3(128), (uint32_t)lds, s, sl2);
        }
    }
    /* Head group and d split by how many workgroups the rows give. Every choice sums each output
     * in the same order, so the bits do not depend on it. A tall pass under ..._EXACT_TALL takes
     * the decode band's own kernel, as a 16-token pass does. */
    const long rows = s.M;
    if (oracle_form < 0 && (rows <= kWmRows || ds_exact_tall())) {
        const int rc = attn_dec_emit(stream, s);
        if (rc != RAD_E_UNSUPPORTED) return rc;
    }
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

} // namespace windowpack
