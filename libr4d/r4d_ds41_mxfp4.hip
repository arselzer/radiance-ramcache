/* r4d_ds41_mxfp4.hip -- gemm_nt_ggml and moe_gemm_ggml over GGUF MXFP4 blocks (GGML_TYPE_MXFP4, ggml type
 * 39: block_mxfp4 { uint8_t e; uint8_t qs[16]; } over 32 values), libref/ref_ggml.cpp the oracle.
 *
 * The weight is a GGUF import's planes (fp4_e2m1 codes, e8m0 per 32: OCP MXFP4's own spelling) relayouted at load
 * into llama.cpp's native blocks by r4d_ds41_ggml_layout.cpp (libref's bytes, tag "ggml_mxfp4"): rows back to back,
 * K / 32 * 17 bytes a row, handed as u8 at the DECLARED shape [N, K], so the row stride comes from the type. A
 * float [N, K] weight is the logical form (rad-kbench's random operands) and is read as it is. The DeepSeek-V4.1
 * DSpark drafter's experts are these in a GGUF that carries them as MXFP4; so are gpt-oss's in a GGUF.
 *
 * Arithmetic, as libref: each weight value is dequantised to f32 as llama.cpp's dequantize_row_mxfp4 does,
 * kvalues_mxfp4[nibble] * 2^(e - 128) (the doubled e2m1 at half the scale; an integer times a power of two, so
 * exact), multiplied by the bf16 activation widened to f32 (a rounded product, no FMA: -ffp-contract=off) and
 * summed in f32, then rounded once to bf16. Only the order of the sum differs from libref's index order.
 *
 * A block is 17 bytes, so blocks sit at any byte offset. A unit (one block, 32 values) is read as the five
 * aligned dwords that cover it -- every one holds a byte of the block, so a read never leaves the block's own
 * dword granules (no K or pointer alignment needed) -- and funnel-shifted into the exponent and four code dwords.
 *
 * Two paths an op, picked by the row band (the same shapes as r4d_ds41_ggml's qk32 rows):
 *   gemv  (M <= 16 rows, or tokens for the grouped GEMM): the weight streamed once. A wave owns 1 or 2 weight rows,
 *         its lanes walk the row's blocks (lane = block mod 32, four blocks a lane in flight), each decoded into
 *         registers and dotted with every activation row, the activation chunk staged once a block in LDS as f32.
 *         A wave butterfly closes each sum. Up to 8 activation rows a pass, 16 in two.
 *   tiled (any M): 128 x 128 output tiles, the weight tile dequantised into LDS once a block-wide slab and reused
 *         by the tile's activation rows, the next slab's raw words prefetched; f32 8 x 8 (or 4 x 4) register tiles.
 *         For the grouped GEMM a tile is 32 or 128 rows of one expert's run.
 * A unit's four groups of 8 consecutive values: groups 0, 1 the low nibbles of qs[0..15], 2, 3 the high ones.
 *
 * Design after r4d_ds41_ggml.hip's qk32 rows (this unit is MXFP4's own so the two can change apart).
 */
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_common.h"
#include "r4d_emit.h"
#include "r4d_launch.h"
#include "r4d_plugin.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <type_traits>

namespace {

/* ggml-common.h's kvalues_mxfp4 (kvalues_fp4): e2m1 doubled, as floats. */
__constant__ __attribute__((aligned(16))) float kMxValues[16] = { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 12.0f,
                                                                   0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -6.0f, -8.0f, -12.0f };

/* ================================================================== loads and conversions */
enum : int { MX_FP4 = 0, MX_F32 = 1 };

typedef __attribute__((address_space(1))) const uint32_t* mx_gu32;
typedef __attribute__((address_space(1))) const r4d_u32x4* mx_gu128;

__device__ __forceinline__ uint32_t mx_d32(const uint8_t* p) { return *(mx_gu32)(const void*)p; }
__device__ __forceinline__ uint32_t mx_byte(uint32_t w, int i) { return (w >> (8 * i)) & 0xffu; }
/* The dword at byte offset `s` (0..3) of the pair lo:hi. */
__device__ __forceinline__ uint32_t mx_funnel(uint32_t lo, uint32_t hi, int s) {
    return (uint32_t)((((uint64_t)hi << 32) | lo) >> (8 * s));
}
/* ggml-impl.h's ggml_e8m0_to_fp32_half: 2^(e - 128), the two subnormals (e = 0, 1) as their bit patterns. */
__device__ __forceinline__ float mx_e8m0_half(uint32_t e) {
    return __uint_as_float(e < 2u ? (0x00200000u << e) : ((e - 1u) << 23));
}

/* Rounded once to the output dtype as libref's stt does: bf16 RNE with a NaN kept quiet. */
__device__ __forceinline__ void mx_put(uint16_t* p, float v) { *p = (uint16_t)(bf16_hi_rne_nan(v) >> 16); }

template <int N, int I = 0, class F>
__device__ __forceinline__ void mx_for(F&& f) {
    if constexpr (I < N) {
        f(std::integral_constant<int, I>{});
        mx_for<N, I + 1>(f);
    }
}

/* ================================================================== the types
 *
 * QK values and BB bytes a block, NR raw dwords a unit (a unit is the whole block), TB bytes of LDS table.
 * load() reads the block at `b` into raw dwords; goff(G) is where group G (8 consecutive values) starts in the
 * block; dec<G>() turns the raw dwords into the group's eight f32 values. */
template <int T> struct Mx;

/* MXFP4: e, qs[16]. raw[0] = e, raw[1..4] = qs as four dwords; y[j] = kv[qs[j] & 15] * d, y[j + 16] =
 * kv[qs[j] >> 4] * d, d = 2^(e - 128). */
template <> struct Mx<MX_FP4> {
    static constexpr int QK = 32, BB = 17, NR = 5, TB = 64;
    __device__ static __forceinline__ void load(const uint8_t* b, uint32_t* r) {
        const uintptr_t p = (uintptr_t)b;
        const uint8_t* a = (const uint8_t*)(p & ~(uintptr_t)3);
        const int s = (int)(p & 3u);
        uint32_t w[5];
#pragma unroll
        for (int i = 0; i < 5; ++i) w[i] = mx_d32(a + 4 * i);
        r[0] = mx_byte(w[0], s);
        /* qs starts at byte s + 1 of the window: dword (s + 1) / 4, shift (s + 1) % 4. */
        const bool next = s == 3;
        const int sh = (s + 1) & 3;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t lo = next ? w[i + 1] : w[i];
            const uint32_t hi = next ? w[i + 1 < 4 ? i + 2 : 4] : w[i + 1];
            r[1 + i] = mx_funnel(lo, hi, sh);
        }
    }
    __device__ static __forceinline__ int goff(int g) { return 8 * g; }
    template <int G>
    __device__ static __forceinline__ void dec(const uint32_t* r, const void* tab, float* w) {
        const float* kv = (const float*)tab;
        const float d = mx_e8m0_half(r[0]);
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            const uint32_t by = mx_byte(r[1 + 2 * (G & 1) + t / 4], t % 4);
            w[t] = kv[G >= 2 ? (by >> 4) : (by & 0xFu)] * d;
        }
    }
};

/* The float logical form: 32 values a unit, read as they are. */
template <> struct Mx<MX_F32> {
    static constexpr int QK = 32, BB = 128, NR = 32, TB = 0;
    __device__ static __forceinline__ void load(const uint8_t* b, uint32_t* r) {
#pragma unroll
        for (int i = 0; i < 32; ++i) r[i] = mx_d32(b + 4 * i);
    }
    __device__ static __forceinline__ int goff(int g) { return 8 * g; }
    template <int G>
    __device__ static __forceinline__ void dec(const uint32_t* r, const void*, float* w) {
#pragma unroll
        for (int t = 0; t < 8; ++t) w[t] = __uint_as_float(r[8 * G + t]);
    }
};

/* The type's table into LDS at `lds`, by the whole block; the caller barriers. */
template <int T>
__device__ __forceinline__ void mx_load_table(uint8_t* lds) {
    if constexpr (Mx<T>::TB > 0) {
        const r4d_u32x4* src = (const r4d_u32x4*)(const void*)kMxValues;
        r4d_u32x4* dst = (r4d_u32x4*)lds;
        for (int i = (int)threadIdx.x; i < Mx<T>::TB / 16; i += (int)blockDim.x) dst[i] = src[i];
    }
}

/* Eight bf16 activations widened to f32 (exact). `al`: the address is 16-byte aligned. */
__device__ __forceinline__ void mx_x8(const uint16_t* p, bool al, float* v) {
    uint32_t q[4];
    if (al) {
        const r4d_u32x4 t = *(mx_gu128)(const void*)p;
        q[0] = t[0]; q[1] = t[1]; q[2] = t[2]; q[3] = t[3];
    } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) q[i] = ldg_u16(p + 2 * i) | (ldg_u16(p + 2 * i + 1) << 16);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) { v[2 * i] = bf16w_lo(q[i]); v[2 * i + 1] = bf16w_hi(q[i]); }
}

/* The same eight bf16 as four bf16-pair dwords, unconverted. */
__device__ __forceinline__ void mx_x8raw(const uint16_t* p, bool al, uint32_t* q) {
    if (al) {
        const r4d_u32x4 t = *(mx_gu128)(const void*)p;
        q[0] = t[0]; q[1] = t[1]; q[2] = t[2]; q[3] = t[3];
    } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) q[i] = ldg_u16(p + 2 * i) | (ldg_u16(p + 2 * i + 1) << 16);
    }
}

/* Expert e's rows of the sorted order as libref assigns them: its binary search gives row i to the last expert
 * whose offset is <= i (expert 0 below off[0], the last one up to T), which for moe_scatter's non-decreasing
 * offsets is [off[e], off[e + 1]) with those two ends opened. */
__device__ __forceinline__ void mx_range(const int32_t* off, int e, int ne, int T, int* s, int* t) {
    int a = e == 0 ? 0 : off[e];
    int b = e == ne - 1 ? T : off[e + 1];
    a = a < 0 ? 0 : (a > T ? T : a);
    b = b < a ? a : (b > T ? T : b);
    *s = a;
    *t = b;
}

/* The activation row a sorted slot reads (a_order "token": sorted / top_k, "sorted": the slot), or -1 for a
 * dropped slot or a token past `a`: a row of zeros, as libref writes it. */
__device__ __forceinline__ int mx_moe_tok(const int32_t* sorted, int i, int top_k, int a_sorted, int AR) {
    const int flat = sorted[i];
    if (flat < 0) return -1;
    const int tok = a_sorted ? i : flat / top_k;
    return tok < AR ? tok : -1;
}

/* ================================================================== gemv (decode) */
constexpr int kMxWaves = 8;
constexpr int kMxThreads = 256;
constexpr int kMxXM = 8;                 /* activation rows a pass */
constexpr int kMxMaxExpert = 1024;

struct MxGemvArgs {
    const uint16_t* x;
    long long x_ld;                      /* elements */
    const uint8_t* w;                    /* plain weight, or stacked experts */
    const uint8_t* const* wtab;          /* expert table (device pointers), or null */
    long long w_ld;                      /* bytes a weight row */
    long long w_es;                      /* bytes an expert (stacked) */
    uint16_t* y;
    long long y_ld;
    const int32_t* sorted;
    const int32_t* off;
    int M;                               /* GEMM: activation rows; MoE: output rows T */
    int N, K;
    int XM;                              /* activation rows a pass, <= kMxXM */
    int XLM;                             /* rows whose whole K fits the LDS buffer (0..XM) */
    int XR;                              /* floats a whole staged row (K plus its pad) */
    int KC;                              /* values a chunk when the pass's rows do not fit whole */
    int rgpi;                            /* row groups a work item */
    int ipe;                             /* work items an expert (GEMM: in all) */
    int n_expert, top_k, a_sorted, AR, x_al;
    int live_off, x_off;                 /* LDS byte offsets of the live list and the rows */
};

/* A staged row: its values with 4 floats of pad after every 32, so the lanes' blocks read different banks. */
__device__ __forceinline__ int mx_xphys(int kk) { return kk + ((kk >> 5) << 2); }

/* The activation row a pass row reads (null: a dropped slot, a row of zeros). */
template <bool MOE>
__device__ __forceinline__ const uint16_t* mx_xrow(const MxGemvArgs& A, int i) {
    int row;
    if constexpr (MOE) row = mx_moe_tok(A.sorted, i, A.top_k, A.a_sorted, A.AR);
    else row = i;
    return row >= 0 ? A.x + (long long)row * A.x_ld : nullptr;
}

/* Values [k0, k1) of rows p0 .. p0 + mc - 1 widened into LDS by the block, `xr` floats a row. */
template <bool MOE>
__device__ __forceinline__ void mx_stage(const MxGemvArgs& A, float* xs, int p0, int mc, int k0, int k1, int xr) {
    const int c8n = (k1 - k0) >> 3;
    for (int idx = (int)threadIdx.x; idx < mc * c8n; idx += kMxThreads) {
        const int m = idx / c8n, kk = (idx - m * c8n) << 3;
        const uint16_t* xrow = mx_xrow<MOE>(A, p0 + m);
        float v[8];
        if (xrow) {
            mx_x8(xrow + k0 + kk, A.x_al != 0, v);
        } else {
#pragma unroll
            for (int t = 0; t < 8; ++t) v[t] = 0.0f;
        }
        float* d = xs + m * xr + mx_xphys(kk);
        *(float4*)d = make_float4(v[0], v[1], v[2], v[3]);
        *(float4*)(d + 4) = make_float4(v[4], v[5], v[6], v[7]);
    }
}

/* One block of RPW rows into the sums; the staged rows hold values from kb on, xr floats a row. */
template <int T, int RPW>
__device__ __forceinline__ void mx_unit(const uint32_t (&raw)[RPW][Mx<T>::NR], int U, const float* xs, int xr,
                                        int kb, int mc, const void* tab, const bool (&ok)[RPW],
                                        float (&acc)[RPW][kMxXM]) {
    using Q = Mx<T>;
    mx_for<4>([&](auto gi) {
        constexpr int G = decltype(gi)::value;
        const int kk = U * Q::QK + Q::goff(G) - kb;
        float w[RPW][8];
#pragma unroll
        for (int r = 0; r < RPW; ++r)
            if (ok[r]) Q::template dec<G>(raw[r], tab, w[r]);
#pragma unroll
        for (int m = 0; m < kMxXM; ++m) {
            if (m < mc) {
                const float* xp = xs + m * xr + mx_xphys(kk);
                const float4 a = *(const float4*)xp;
                const float4 c = *(const float4*)(xp + 4);
                const float xv[8] = { a.x, a.y, a.z, a.w, c.x, c.y, c.z, c.w };
#pragma unroll
                for (int r = 0; r < RPW; ++r)
                    if (ok[r]) {
#pragma unroll
                        for (int t = 0; t < 8; ++t) acc[r][m] += xv[t] * w[r][t];
                    }
            }
        }
    });
}

/* Blocks [U0, U1) of rows n0.. into the sums: lane = block mod 32 (U0 a multiple of 32), UF blocks a lane in
 * flight. A lane's blocks and their order are the same whether K is staged whole or in chunks of a multiple of
 * 32 blocks, so the bits do not depend on M or on the path. */
template <int T, int RPW>
__device__ __forceinline__ void mx_row(const MxGemvArgs& A, const uint8_t* wb, const float* xs, int xr, int kb,
                                       int U0, int U1, int n0, int mc, int lane, const void* tab,
                                       float (&acc)[RPW][kMxXM]) {
    using Q = Mx<T>;
    constexpr int UF = T == MX_F32 ? 1 : 4;
    bool ok[RPW];
    const uint8_t* wr[RPW];
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        ok[r] = n0 + r < A.N;
        wr[r] = wb + (long long)(ok[r] ? n0 + r : 0) * A.w_ld;
    }
    for (int U = U0 + lane; U < U1; U += 32 * UF) {
        uint32_t raw[UF][RPW][Q::NR];
        mx_for<UF>([&](auto fi) {
            constexpr int F = decltype(fi)::value;
            const int V = U + 32 * F;
            if (V < U1) {
#pragma unroll
                for (int r = 0; r < RPW; ++r)
                    if (ok[r]) Q::load(wr[r] + (long long)V * Q::BB, raw[F][r]);
            }
        });
        mx_for<UF>([&](auto fi) {
            constexpr int F = decltype(fi)::value;
            const int V = U + 32 * F;
            if (V < U1) mx_unit<T, RPW>(raw[F], V, xs, xr, kb, mc, tab, ok, acc);
        });
    }
}

template <int T, int RPW, bool MOE>
__global__ __launch_bounds__(kMxThreads) void mx_gemv_kernel(MxGemvArgs A) {
    using Q = Mx<T>;
    R4D_DYN_LDS(r4d_u32x4, mx_lds);
    uint8_t* const mx_smem = (uint8_t*)mx_lds;
    __shared__ int s_nlive;
    const int tid = (int)threadIdx.x, lane = tid & 31, wave = tid >> 5;
    mx_load_table<T>(mx_smem);
    const void* tab = mx_smem;
    int* live = (int*)(mx_smem + A.live_off);
    float* xs = (float*)(mx_smem + A.x_off);
    constexpr int RG = kMxWaves * RPW;
    const int nrg = (A.N + RG - 1) / RG;
    const int UK = A.K / Q::QK;                       /* blocks a row */
    const int XRC = A.KC + A.KC / 8;                  /* floats a chunked row */
    int n_items = A.ipe;
    if constexpr (MOE) {
        /* The experts with rows, in ascending order. */
        if (wave == 0) {
            int cnt = 0;
            for (int e0 = 0; e0 < A.n_expert; e0 += 32) {
                const int e = e0 + lane;
                int s = 0, t = 0;
                if (e < A.n_expert) mx_range(A.off, e, A.n_expert, A.M, &s, &t);
                const bool nz = e < A.n_expert && t > s;
                const uint32_t bal = __builtin_amdgcn_ballot_w32(nz);
                if (nz) live[cnt + __builtin_popcount(bal & ((1u << lane) - 1u))] = e;
                cnt += __builtin_popcount(bal);
            }
            if (lane == 0) s_nlive = cnt;
        }
        __syncthreads();
        n_items = s_nlive * A.ipe;
    } else {
        __syncthreads();
    }
    int cur_e = -1, cur_p = -1;                       /* the rows the LDS buffer holds whole */
    for (int it = (int)blockIdx.x; it < n_items; it += (int)gridDim.x) {
        int e = 0, ri = it, rs = 0, re = A.M;
        const uint8_t* wb = A.w;
        if constexpr (MOE) {
            e = live[it / A.ipe];
            ri = it - (it / A.ipe) * A.ipe;
            mx_range(A.off, e, A.n_expert, A.M, &rs, &re);
            wb = A.wtab ? A.wtab[e] : A.w + (long long)e * A.w_es;
        }
        const int rg0 = ri * A.rgpi, rg1 = min(rg0 + A.rgpi, nrg);
        for (int p0 = rs; p0 < re; p0 += A.XM) {
            const int mc = min(A.XM, re - p0);
            const bool whole = mc <= A.XLM;            /* block-uniform */
            if (whole && (cur_e != e || cur_p != p0)) {
                lds_barrier();
                mx_stage<MOE>(A, xs, p0, mc, 0, A.K, A.XR);
                lds_barrier();
                cur_e = e;
                cur_p = p0;
            }
            /* The pass's live rows (a dropped slot writes zeros), read once, not a row group. */
            uint32_t livem = (1u << mc) - 1u;
            if constexpr (MOE) {
                livem = 0;
                for (int m = 0; m < mc; ++m)
                    if (mx_moe_tok(A.sorted, p0 + m, A.top_k, A.a_sorted, A.AR) >= 0) livem |= 1u << m;
            }
            for (int rg = rg0; rg < rg1; ++rg) {
                const int n0 = rg * RG + wave * RPW;
                const bool active = n0 < A.N;          /* wave-uniform */
                float acc[RPW][kMxXM];
#pragma unroll
                for (int r = 0; r < RPW; ++r)
#pragma unroll
                    for (int m = 0; m < kMxXM; ++m) acc[r][m] = 0.0f;
                if (whole) {
                    if (active) mx_row<T, RPW>(A, wb, xs, A.XR, 0, 0, UK, n0, mc, lane, tab, acc);
                } else {
                    /* The pass's rows do not fit whole: K in chunks, restaged a row group (every wave takes
                     * part in the barriers). */
                    for (int k0 = 0; k0 < A.K; k0 += A.KC) {
                        const int k1 = min(k0 + A.KC, A.K);
                        lds_barrier();
                        mx_stage<MOE>(A, xs, p0, mc, k0, k1, XRC);
                        lds_barrier();
                        if (active) mx_row<T, RPW>(A, wb, xs, XRC, k0, k0 / Q::QK, k1 / Q::QK, n0, mc, lane, tab, acc);
                    }
                    cur_e = -1;
                }
                if (!active) continue;
#pragma unroll
                for (int r = 0; r < RPW; ++r) {
                    const int n = n0 + r;
                    if (n >= A.N) continue;
#pragma unroll
                    for (int m = 0; m < kMxXM; ++m) {
                        if (m >= mc) continue;
                        const float v = r4d_wave_sum(acc[r][m]);
                        if (lane == 0) mx_put(A.y + (long long)(p0 + m) * A.y_ld + n, ((livem >> m) & 1u) ? v : 0.0f);
                    }
                }
            }
        }
    }
}

/* ================================================================== tiled (prefill) */
constexpr int kTN = 128, kTP = 132;   /* tile columns (weight rows), LDS row floats of Ws */

struct MxTiledArgs {
    const uint16_t* x;
    long long x_ld;
    const uint8_t* w;
    const uint8_t* const* wtab;
    long long w_ld, w_es;
    uint16_t* y;
    long long y_ld;
    const int32_t* sorted;
    const int32_t* off;
    int M, N, K;                         /* MoE: M is T, the output rows */
    int n_expert, top_k, a_sorted, AR, x_al, tab_bytes;
};

/* BM activation rows x 128 weight rows a block. BM 128: 8 x 8 outputs a thread; BM 32: 4 x 4. Each output is the
 * same chain at either height (slabs in order, the slab's 32 values in order), so the bits do not depend on BM. */
template <int T, int BM, bool MOE>
__global__ __launch_bounds__(kMxThreads) void mx_tiled_kernel(MxTiledArgs A) {
    using Q = Mx<T>;
    static_assert(BM == 32 || BM == 128, "a tile is 32 or 128 rows");
    constexpr int TM = BM == 128 ? 8 : 4, TN = BM == 128 ? 8 : 4;
    constexpr int TXN = kTN / TN;                   /* threads across the columns */
    constexpr int XP = BM + 4;                      /* LDS row floats of Xs */
    R4D_DYN_LDS(r4d_u32x4, mx_lds);
    uint8_t* const mx_smem = (uint8_t*)mx_lds;
    __shared__ int s_e, s_r0, s_r1;
    __shared__ int s_tok[BM];
    const int tid = (int)threadIdx.x, lane = tid & 31, wave = tid >> 5;
    mx_load_table<T>(mx_smem);
    const void* tab = mx_smem;
    float* Ws = (float*)(mx_smem + A.tab_bytes);   /* [32][kTP]: slab value, weight row */
    float* Xs = Ws + 32 * kTP;                      /* [32][XP]: slab value, activation row */
    int e = 0, r0, r1;
    if constexpr (MOE) {
        /* Slot blockIdx.x -> (expert, its tile): the experts' tile counts scanned a wave of 32 at a time. */
        if (wave == 0) {
            if (lane == 0) s_e = -1;
            const uint32_t slot = blockIdx.x;
            uint32_t base = 0;
            for (int e0 = 0; e0 < A.n_expert; e0 += 32) {
                const int ee = e0 + lane;
                int s = 0, t = 0;
                if (ee < A.n_expert) mx_range(A.off, ee, A.n_expert, A.M, &s, &t);
                const uint32_t tiles = (uint32_t)((t - s + BM - 1) / BM);
                const uint32_t inc = r4d_wave_prefix_u32(tiles, lane);
                const uint32_t tot = (uint32_t)__builtin_amdgcn_readlane((int)inc, 31);
                if (slot < base + tot) {
                    if (slot >= base + inc - tiles && slot < base + inc) {
                        s_e = ee;
                        s_r0 = s + (int)(slot - (base + inc - tiles)) * BM;
                        s_r1 = min(s_r0 + BM, t);
                    }
                    break;
                }
                base += tot;
            }
        }
        __syncthreads();
        e = s_e;
        if (e < 0) return;
        r0 = s_r0;
        r1 = s_r1;
    } else {
        r0 = (int)blockIdx.x * BM;
        r1 = min(r0 + BM, A.M);
    }
    if (tid < BM) {
        const int i = r0 + tid;
        int tk = -1;
        if (i < r1) {
            if constexpr (MOE) tk = mx_moe_tok(A.sorted, i, A.top_k, A.a_sorted, A.AR);
            else tk = i;
        }
        s_tok[tid] = tk;
    }
    __syncthreads();

    const int n0 = (int)blockIdx.y * kTN;
    const uint8_t* wb = A.w;
    if constexpr (MOE) wb = A.wtab ? A.wtab[e] : A.w + (long long)e * A.w_es;
    const int S = A.K / Q::QK;
    /* Waves 0..3 decode a weight row each a slab, the next BM threads widen an activation row each. */
    const bool wrole = tid < kTN;
    const bool xrole = tid >= kTN && tid < kTN + BM;
    const int ri = wrole ? tid : (xrole ? tid - kTN : 0);
    const bool nok = n0 + ri < A.N;
    const uint8_t* wrow = wb + (long long)(nok ? n0 + ri : 0) * A.w_ld;
    const int tokr = xrole ? s_tok[ri] : -1;
    const uint16_t* xrow = A.x + (long long)(tokr >= 0 ? tokr : 0) * A.x_ld;
    /* One prefetch register set for both roles: a block's raw words, or four groups of eight bf16 (four dwords
     * each). */
    constexpr int PF = Q::NR > 16 ? Q::NR : 16;
    uint32_t pf[PF];
    auto prefetch = [&](int s) {
        if (wrole) {
            if (nok) Q::load(wrow + (long long)s * Q::BB, pf);
        } else if (tokr >= 0) {
            mx_for<4>([&](auto gi) {
                constexpr int G = decltype(gi)::value;
                mx_x8raw(xrow + s * Q::QK + Q::goff(G), A.x_al != 0, pf + 4 * G);
            });
        }
    };
    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;
    const int ty = tid / TXN, tx = tid % TXN;
    prefetch(0);
    for (int s = 0; s < S; ++s) {
        if (wrole) {
            mx_for<4>([&](auto gi) {
                constexpr int G = decltype(gi)::value;
                float w[8];
                if (nok) {
                    Q::template dec<G>(pf, tab, w);
                } else {
#pragma unroll
                    for (int t = 0; t < 8; ++t) w[t] = 0.0f;
                }
#pragma unroll
                for (int t = 0; t < 8; ++t) Ws[(8 * G + t) * kTP + ri] = w[t];
            });
        } else if (xrole) {
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const uint32_t q = tokr >= 0 ? pf[i] : 0u;
                Xs[(2 * i) * XP + ri] = bf16w_lo(q);
                Xs[(2 * i + 1) * XP + ri] = bf16w_hi(q);
            }
        }
        lds_barrier();
        if (s + 1 < S) prefetch(s + 1);
#pragma unroll 2
        for (int kk = 0; kk < 32; ++kk) {
            float av[TM], bv[TN];
#pragma unroll
            for (int i = 0; i < TM; i += 4) {
                const float4 v = *(const float4*)(Xs + kk * XP + ty * TM + i);
                av[i] = v.x; av[i + 1] = v.y; av[i + 2] = v.z; av[i + 3] = v.w;
            }
#pragma unroll
            for (int j = 0; j < TN; j += 4) {
                const float4 v = *(const float4*)(Ws + kk * kTP + tx * TN + j);
                bv[j] = v.x; bv[j + 1] = v.y; bv[j + 2] = v.z; bv[j + 3] = v.w;
            }
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += av[i] * bv[j];
        }
        lds_barrier();
    }
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int lr = ty * TM + i, row = r0 + lr;
        if (row >= r1) continue;
        const bool live_row = s_tok[lr] >= 0;
        uint16_t* yr = A.y + (long long)row * A.y_ld;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int n = n0 + tx * TN + j;
            if (n < A.N) mx_put(yr + n, live_row ? acc[i][j] : 0.0f);
        }
    }
}

/* ================================================================== host side */
constexpr long long kQK = 32, kBB = 17;
long long mx_r16(long long v) { return (v + 15) / 16 * 16; }
int mx_tab_bytes(int ty) { return ty == MX_FP4 ? Mx<MX_FP4>::TB : 0; }

/* The activation rows staged whole in LDS: a row is K floats plus a pad of 4 every 32 (XR), and as many rows as
 * fit 36 KiB, at most `xm` (0 when one row does not fit). Past that a pass's rows are staged in chunks of KC
 * values, a multiple of 1024 (32 blocks of 32), as many as fit the same 36 KiB for `xm` rows. Returns the buffer's
 * floats. */
long long mx_xbuf(int xm, long long K, int* xlm, int* xr, int* kc) {
    *xr = (int)(K + K / 8);
    const long long f = 9216 / (K + K / 8);
    *xlm = (int)(f < xm ? f : xm);
    long long c = 9216 / ((long long)xm * 1152);
    if (c < 1) c = 1;
    const long long kr = (K + 1023) / 1024 * 1024;
    *kc = (int)(1024LL * c < kr ? 1024LL * c : kr);
    const long long whole = (long long)*xlm * *xr, chunk = (long long)xm * (*kc + *kc / 8);
    return *xlm >= xm ? whole : (whole > chunk ? whole : chunk);
}

bool mx_inner1(const RadTensor* t) { return t->rank < 2 || t->shape[t->rank - 1] == 1 || t->stride[t->rank - 1] == 1; }
bool mx_al16(const void* p, long long ld) { return ((uintptr_t)p & 15u) == 0 && ld % 8 == 0; }

template <int T>
int mx_launch_gemv_t(RadStream s, int rpw, bool moe, const MxGemvArgs& A, unsigned grid, unsigned lds) {
    constexpr int R2 = T == MX_F32 ? 1 : 2;
    if (moe) {
        if (rpw == 2) return R4D_EMIT(s, (mx_gemv_kernel<T, R2, true>), dim3(grid), dim3(kMxThreads), lds, A);
        return R4D_EMIT(s, (mx_gemv_kernel<T, 1, true>), dim3(grid), dim3(kMxThreads), lds, A);
    }
    if (rpw == 2) return R4D_EMIT(s, (mx_gemv_kernel<T, R2, false>), dim3(grid), dim3(kMxThreads), lds, A);
    return R4D_EMIT(s, (mx_gemv_kernel<T, 1, false>), dim3(grid), dim3(kMxThreads), lds, A);
}

template <int T>
int mx_launch_tiled_t(RadStream s, int bm, bool moe, const MxTiledArgs& A, dim3 grid, unsigned lds) {
    if (bm == 32) {
        if (moe) return R4D_EMIT(s, (mx_tiled_kernel<T, 32, true>), grid, dim3(kMxThreads), lds, A);
        return R4D_EMIT(s, (mx_tiled_kernel<T, 32, false>), grid, dim3(kMxThreads), lds, A);
    }
    if (moe) return R4D_EMIT(s, (mx_tiled_kernel<T, 128, true>), grid, dim3(kMxThreads), lds, A);
    return R4D_EMIT(s, (mx_tiled_kernel<T, 128, false>), grid, dim3(kMxThreads), lds, A);
}

int mx_launch_gemv(RadStream s, int ty, int rpw, bool moe, const MxGemvArgs& A, unsigned grid, unsigned lds) {
    return ty == MX_FP4 ? mx_launch_gemv_t<MX_FP4>(s, rpw, moe, A, grid, lds)
                        : mx_launch_gemv_t<MX_F32>(s, rpw, moe, A, grid, lds);
}

int mx_launch_tiled(RadStream s, int ty, int bm, bool moe, const MxTiledArgs& A, dim3 grid, unsigned lds) {
    return ty == MX_FP4 ? mx_launch_tiled_t<MX_FP4>(s, bm, moe, A, grid, lds)
                        : mx_launch_tiled_t<MX_F32>(s, bm, moe, A, grid, lds);
}

/* The `type` parameter must name MXFP4. */
bool mx_type_ok(const RadArgs* a) {
    const char* t = r4d_gets(a, "type");
    return t && !std::strcmp(t, "mxfp4");
}

/* gemm_nt_ggml type=mxfp4: y[M, N] = x[M, K] . w[N, K]^T, w native blocks (u8, [N, K] declared or [N, row bytes])
 * or the float logical form [N, K]. */
int mx_gemm(const RadArgs* a, RadStream s, bool gemv) {
    if (!a || a->n_t < 3) return R4D_NO(RAD_E_INVAL);
    const RadTensor *x = r4d_opt(a, 0), *w = r4d_opt(a, 1), *y = r4d_opt(a, 2);
    if (!x || !w || !y || !x->data || !w->data || !y->data) return R4D_NO(RAD_E_INVAL);
    if (!mx_type_ok(a)) return R4D_NO(RAD_E_DTYPE);
    const long long N = r4d_geti_or(a, "N", 0), K = r4d_geti_or(a, "K", 0);
    if (N <= 0 || K <= 0 || K % kQK || N > 0x7fffffffLL || K > 0x7fffffffLL) return R4D_NO(RAD_E_SHAPE);
    if (x->rank != 2 || x->dtype != RAD_BF16 || x->shape[1] != K || !mx_inner1(x)) return R4D_NO(RAD_E_SHAPE);
    const long long M = x->shape[0], Mb = r4d_geti_or(a, "M", 0);
    if (M <= 0 || M > 0x7fffffffLL || (Mb > 0 && M > Mb)) return R4D_NO(RAD_E_SHAPE);
    if (y->rank != 2 || y->dtype != RAD_BF16 || y->shape[1] != N || y->shape[0] < M || !mx_inner1(y))
        return R4D_NO(RAD_E_SHAPE);
    if (w->rank != 2 || w->shape[0] != N) return R4D_NO(RAD_E_SHAPE);
    const long long rb = K / kQK * kBB;
    int ty = MX_FP4;
    long long w_ld = rb;
    if (w->dtype == RAD_U8) {
        if (w->shape[1] != rb && w->shape[1] != K) return R4D_NO(RAD_E_SHAPE);
    } else if (w->dtype == RAD_F32) {
        if (w->shape[1] != K || !mx_inner1(w) || ((uintptr_t)w->data & 3u)) return R4D_NO(RAD_E_SHAPE);
        ty = MX_F32;
        w_ld = (long long)w->stride[0] * 4;
    } else {
        return R4D_NO(RAD_E_DTYPE);
    }
    const long long x_ld = M > 1 ? x->stride[0] : K, y_ld = M > 1 ? y->stride[0] : N;
    const int x_al = mx_al16(x->data, x_ld) ? 1 : 0;
    const long long tb = mx_r16(mx_tab_bytes(ty));
    if (gemv && M <= 16) {
        MxGemvArgs A{};
        A.x = (const uint16_t*)x->data; A.x_ld = x_ld;
        A.w = (const uint8_t*)w->data; A.wtab = nullptr; A.w_ld = w_ld; A.w_es = 0;
        A.y = (uint16_t*)y->data; A.y_ld = y_ld;
        A.M = (int)M; A.N = (int)N; A.K = (int)K;
        A.XM = (int)(M < kMxXM ? M : kMxXM);
        const long long xbuf = mx_xbuf(A.XM, K, &A.XLM, &A.XR, &A.KC);
        const int rpw = (M >= 3 && ty != MX_F32 && (N + 15) / 16 >= 128) ? 2 : 1;
        const long long nrg = (N + 8 * rpw - 1) / (8 * rpw);
        A.rgpi = 1; A.ipe = (int)nrg;
        A.x_al = x_al;
        A.live_off = (int)tb; A.x_off = (int)tb;
        const long long lds = tb + xbuf * 4;
        const unsigned grid = (unsigned)(nrg < 256 ? nrg : 256);
        return mx_launch_gemv(s, ty, rpw, false, A, grid, (unsigned)lds);
    }
    MxTiledArgs A{};
    A.x = (const uint16_t*)x->data; A.x_ld = x_ld;
    A.w = (const uint8_t*)w->data; A.wtab = nullptr; A.w_ld = w_ld; A.w_es = 0;
    A.y = (uint16_t*)y->data; A.y_ld = y_ld;
    A.M = (int)M; A.N = (int)N; A.K = (int)K;
    A.x_al = x_al; A.tab_bytes = (int)tb;
    const long long gy = (N + kTN - 1) / kTN;
    if (gy > 65535) return R4D_NO(RAD_E_SHAPE);
    const int bm = M <= 64 ? 32 : 128;
    const dim3 grid((unsigned)((M + bm - 1) / bm), (unsigned)gy);
    return mx_launch_tiled(s, ty, bm, false, A, grid, (unsigned)(tb + 32 * (kTP + bm + 4) * 4));
}

/* moe_gemm_ggml type=mxfp4: y[i] = a[row(i)] . w[e(i)]^T over the expert-sorted rows. */
int mx_moe(const RadArgs* a, RadStream s, bool gemv) {
    if (!a || a->n_t < 5) return R4D_NO(RAD_E_INVAL);
    const RadTensor *A_ = r4d_opt(a, 0), *W = r4d_opt(a, 1), *so = r4d_opt(a, 2), *of = r4d_opt(a, 3),
                    *Y = r4d_opt(a, 4);
    if (!A_ || !W || !so || !of || !Y || !A_->data || !W->data || !so->data || !of->data || !Y->data)
        return R4D_NO(RAD_E_INVAL);
    if (!mx_type_ok(a)) return R4D_NO(RAD_E_DTYPE);
    const long long N = r4d_geti_or(a, "N", 0), K = r4d_geti_or(a, "K", 0);
    const long long ne = r4d_geti_or(a, "n_expert", 0), top_k = r4d_geti_or(a, "top_k", 0);
    const char* ord = r4d_gets(a, "a_order");
    const bool a_sorted = ord && !std::strcmp(ord, "sorted");
    if (ord && !a_sorted && std::strcmp(ord, "token")) return R4D_NO(RAD_E_UNSUPPORTED);
    if (N <= 0 || K <= 0 || K % kQK || ne <= 0 || ne > kMxMaxExpert || top_k <= 0 || N > 0x7fffffffLL ||
        K > 0x7fffffffLL)
        return R4D_NO(RAD_E_SHAPE);
    if (A_->rank != 2 || A_->dtype != RAD_BF16 || A_->shape[1] != K || !mx_inner1(A_)) return R4D_NO(RAD_E_SHAPE);
    if (Y->rank != 2 || Y->dtype != RAD_BF16 || Y->shape[1] != N || !mx_inner1(Y)) return R4D_NO(RAD_E_SHAPE);
    const long long T = Y->shape[0], AR = A_->shape[0];
    if (T <= 0 || AR <= 0 || T > 0x7fffffffLL || AR > 0x7fffffffLL) return R4D_NO(RAD_E_SHAPE);
    if ((so->dtype != RAD_I32 && so->dtype != RAD_U32) || (of->dtype != RAD_I32 && of->dtype != RAD_U32) ||
        !r4d_contig(so) || !r4d_contig(of))
        return R4D_NO(RAD_E_DTYPE);
    if (r4d_numel(so) < T || r4d_numel(of) < ne + 1) return R4D_NO(RAD_E_SHAPE);
    if (W->rank != 3 || W->shape[0] < ne || W->shape[1] != N) return R4D_NO(RAD_E_SHAPE);
    const long long rb = K / kQK * kBB;
    int ty = MX_FP4;
    long long w_ld = 0, w_es = 0;
    const bool table = W->stride[0] == 0;
    if (W->dtype == RAD_U8) {
        if (W->shape[2] != rb && W->shape[2] != K) return R4D_NO(RAD_E_SHAPE);
        w_ld = rb;
        w_es = N * rb;
    } else if (W->dtype == RAD_F32) {
        if (W->shape[2] != K || W->stride[2] != 1) return R4D_NO(RAD_E_SHAPE);
        ty = MX_F32;
        w_ld = (long long)W->stride[1] * 4;
        w_es = (long long)W->stride[0] * 4;
    } else {
        return R4D_NO(RAD_E_DTYPE);
    }
    const long long x_ld = AR > 1 ? A_->stride[0] : K, y_ld = T > 1 ? Y->stride[0] : N;
    const long long tb = mx_r16(mx_tab_bytes(ty));
    const long long tokens = a_sorted ? (T + top_k - 1) / top_k : AR;
    const long long L = ne < T ? ne : T;       /* experts that can have rows */
    if (gemv && tokens <= 16) {
        MxGemvArgs A{};
        A.x = (const uint16_t*)A_->data; A.x_ld = x_ld;
        A.w = table ? nullptr : (const uint8_t*)W->data;
        A.wtab = table ? (const uint8_t* const*)W->data : nullptr;
        A.w_ld = w_ld; A.w_es = w_es;
        A.y = (uint16_t*)Y->data; A.y_ld = y_ld;
        A.sorted = (const int32_t*)so->data; A.off = (const int32_t*)of->data;
        A.M = (int)T; A.N = (int)N; A.K = (int)K;
        /* An expert's run is at most a row a token under a valid routing (top-k picks distinct experts); a longer
         * one takes more passes. */
        A.XM = (int)(tokens < kMxXM ? (tokens < 1 ? 1 : tokens) : kMxXM);
        const long long xbuf = mx_xbuf(A.XM, K, &A.XLM, &A.XR, &A.KC);
        /* Two weight rows a wave on wide experts (twice the loads in flight). */
        const int rpw = (ty != MX_F32 && N >= 1024) ? 2 : 1;
        const long long nrg = (N + 8 * rpw - 1) / (8 * rpw);
        long long rgpi = (L * nrg + 255) / 256;
        if (rgpi < 1) rgpi = 1;
        const long long ipe = (nrg + rgpi - 1) / rgpi;
        A.rgpi = (int)rgpi; A.ipe = (int)ipe;
        A.n_expert = (int)ne; A.top_k = (int)top_k; A.a_sorted = a_sorted ? 1 : 0; A.AR = (int)AR;
        A.x_al = mx_al16(A_->data, x_ld) ? 1 : 0;
        A.live_off = (int)tb;
        A.x_off = (int)(tb + mx_r16(ne * 4));
        const long long lds = A.x_off + xbuf * 4;
        const long long items = L * ipe;
        const unsigned grid = (unsigned)(items < 256 ? items : 256);
        return mx_launch_gemv(s, ty, rpw, true, A, grid, (unsigned)lds);
    }
    MxTiledArgs A{};
    A.x = (const uint16_t*)A_->data; A.x_ld = x_ld;
    A.w = table ? nullptr : (const uint8_t*)W->data;
    A.wtab = table ? (const uint8_t* const*)W->data : nullptr;
    A.w_ld = w_ld; A.w_es = w_es;
    A.y = (uint16_t*)Y->data; A.y_ld = y_ld;
    A.sorted = (const int32_t*)so->data; A.off = (const int32_t*)of->data;
    A.M = (int)T; A.N = (int)N; A.K = (int)K;
    A.n_expert = (int)ne; A.top_k = (int)top_k; A.a_sorted = a_sorted ? 1 : 0; A.AR = (int)AR;
    A.x_al = mx_al16(A_->data, x_ld) ? 1 : 0;
    A.tab_bytes = (int)tb;
    /* 32-row tiles while the experts' runs average under 64 rows (prefill over many experts). */
    const int bm = T < 64 * L ? 32 : 128;
    const long long gx = (T + bm - 1) / bm + L, gy = (N + kTN - 1) / kTN;
    if (gy > 65535 || gx > 0x7fffffffLL) return R4D_NO(RAD_E_SHAPE);
    return mx_launch_tiled(s, ty, bm, true, A, dim3((unsigned)gx, (unsigned)gy),
                           (unsigned)(tb + 32 * (kTP + bm + 4) * 4));
}

}  // namespace

extern "C" int r4d_rad_gemm_nt_mxfp4_gemv(const RadArgs* a, RadStream s) { return mx_gemm(a, s, true); }
extern "C" int r4d_rad_gemm_nt_mxfp4_tiled(const RadArgs* a, RadStream s) { return mx_gemm(a, s, false); }
extern "C" int r4d_rad_moe_gemm_mxfp4_gemv(const RadArgs* a, RadStream s) { return mx_moe(a, s, true); }
extern "C" int r4d_rad_moe_gemm_mxfp4_tiled(const RadArgs* a, RadStream s) { return mx_moe(a, s, false); }
