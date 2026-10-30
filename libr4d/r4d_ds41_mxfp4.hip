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
 *         A wave butterfly closes each sum. Up to 8 activation rows a pass, 16 in two. Under
 *         RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL=1 the grouped GEMM's twin row moe_gemm_ggml_mxfp4_gemv_tall runs this
 *         path at any number of tokens (r4d_rows.cpp), so a tall pass keeps a 16-token pass's bits.
 *   tiled (any M): 128 x 128 output tiles, the weight tile dequantised into LDS once a block-wide slab and reused
 *         by the tile's activation rows, the next slab's raw words prefetched; f32 8 x 8 (or 4 x 4) register tiles.
 *         For the grouped GEMM a tile is 32 or 128 rows of one expert's run.
 * A unit's four groups of 8 consecutive values: groups 0, 1 the low nibbles of qs[0..15], 2, 3 the high ones.
 *
 * moe_gemm_ggml_mxfp4_gemv_vp (RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED, core/place/vram_packed.h) is the gemv
 * path over a table some of whose entries name the packed-scale form instead of block_mxfp4 (see "the
 * packed-scale form" below): the engine holds those VRAM residents 4.4% smaller and marks their entries with
 * bit 0. Each packed block is widened back to its e8m0 byte and decoded by the block path's arithmetic, value
 * for value in the same order, so its outputs are the block row's bit for bit; an unmarked entry is read as
 * block_mxfp4 exactly as the plain row reads it. Its kernel is mx_vp2_kernel ("gemv_vp's kernel (vp2)" below:
 * a pass's loads issued together, a row's blocks in one pass decoded by one rolled loop, the scales' raw rounds
 * found and read on the scalar unit, a pass of dropped slots storing its zeros alone), faster a call than the
 * plain row's own over packed and over block_mxfp4 entries alike; RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED_KERNEL
 * keeps the earlier kernels selectable (old: the first form; r2, r2cf: earlier vp2 forms).
 *
 * Design after r4d_ds41_ggml.hip's qk32 rows (this unit is MXFP4's own so the two can change apart).
 */
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_common.h"
#include "r4d_emit.h"
#include "r4d_launch.h"
#include "r4d_plugin.h"
#include "r4d_ds41_live_tail_policy.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

namespace {

/* ggml-common.h's kvalues_mxfp4 (kvalues_fp4): e2m1 doubled, as floats. */
__constant__ __attribute__((aligned(16))) float kMxValues[16] = { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 12.0f,
                                                                   0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -6.0f, -8.0f, -12.0f };

/* ================================================================== loads and conversions */
/* MX_PK: a block of the packed-scale form (the gemv_vp row's marked entries); it has no load() of its own, the
 * packed row walker (mx_row_pk) fills its raw words. */
enum : int { MX_FP4 = 0, MX_F32 = 1, MX_PK = 2 };

typedef __attribute__((address_space(1))) const uint8_t* mx_gu8;
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

/* A block of the packed-scale form: raw[0] = its e8m0 (widened from its 2-bit code, or its raw round's byte),
 * raw[1..4] = its 16 bytes of the codes plane, where value 2i is the low nibble of byte i. So group G (values
 * 8G..8G+7) is code dword G, value 8G + t in its bits 4t..4t+3, and y = kv[nibble] * 2^(e - 128) as in
 * Mx<MX_FP4>::dec: the same eight f32 values of the same block, from the same table and the same scale. */
template <> struct Mx<MX_PK> {
    static constexpr int QK = 32, NR = 5;   /* the table is MX_FP4's, loaded by the kernel's T */
    __device__ static __forceinline__ int goff(int g) { return 8 * g; }
    template <int G>
    __device__ static __forceinline__ void dec(const uint32_t* r, const void* tab, float* w) {
        const float* kv = (const float*)tab;
        const float d = mx_e8m0_half(r[0]);
#pragma unroll
        for (int t = 0; t < 8; ++t) w[t] = kv[(r[1 + G] >> (4 * t)) & 0xFu] * d;
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
    int live_end;
    int one_pass;                        /* an expert's rows (up to XM) in one pass over its weights */
    int probe_passes;                    /* the probe also counts weight passes (probe[5]) */
    unsigned long long* probe;
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

/* ================================================================== the packed-scale form
 *
 * tools/ds41-host-experts/ds41_host_experts.h's packed form (pk_layout there; the core packs with that code and
 * checks every weight's round trip before it marks the entry), restated because a plugin links nothing of
 * another. A weight [N, K] of UK = K / 32 blocks a row, R = ceil(UK / 32) rounds of 32 blocks a row (the last
 * holding the t = UK % 32 left over), its sections at 64-byte boundaries:
 *
 *   header   u32 magic "HXP2", u32 version 1, i32 N, K, cap, n_exc, u32 e0, u32 0, u64 total, 24 bytes 0
 *   codes    [N, K / 2]: the release's codes plane, value 2i in the low nibble of byte i
 *   pk       [N, PB], PB = 8 * (UK / 32) + (t + 3) / 4: a round of b blocks takes tb = (b + 3) / 4 bytes, its
 *            block j's 2-bit code in bits 2 (j / tb) .. +1 of byte j % tb (a full round: byte j % 8, bits
 *            2 (j / 8)); the block's e8m0 is e0 + code
 *   exc_idx  cap u32: the n_exc raw rounds' indices n * R + r, ascending (a raw round has a block outside
 *            e0..e0+3; its codes in pk are 0)
 *   exc_raw  cap x 32 bytes: the raw rounds' e8m0 bytes, in block order
 *
 * The decode reads a block's 16 code bytes as one aligned 128-bit load and its scale from its round (two bits,
 * or the raw byte), then runs the block path's arithmetic (Mx<MX_PK>::dec, mx_unit), so a packed weight's
 * outputs are its block_mxfp4 weight's bit for bit. Lane L's blocks are L, L + 32, ...: every lane of a wave
 * is in the same round at the same time, and a row's rounds come in increasing order, so the raw rounds are a
 * cursor that walks the ascending index, not a search per block. */
constexpr uint32_t kPkMagic = 0x32505848u;   /* "HXP2", ds41_host_experts.cpp kPkMagic */
constexpr uint32_t kPkVersion = 1;

__device__ __forceinline__ long long mx_up64(long long v) { return (v + 63) & ~63LL; }

/* One packed weight's sections. `ok` false: the header does not describe exactly a [N, K] weight laid out as
 * pk_layout lays it, and nothing past the header is read (its rows come out NaN, loudly). */
struct MxPk {
    const uint8_t*  codes;
    const uint8_t*  pk;
    const uint8_t*  idx;     /* u32 entries */
    const uint8_t*  raw;
    int      n_exc, R, RF, PB, tbl;
    uint32_t e0;
    bool     ok;
};
__device__ __forceinline__ MxPk mx_pk_open(const uint8_t* p, int N, int K) {
    uint32_t h[12];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const r4d_u32x4 q = *(mx_gu128)(const void*)(p + 16 * i);
        h[4 * i] = q[0]; h[4 * i + 1] = q[1]; h[4 * i + 2] = q[2]; h[4 * i + 3] = q[3];
    }
    MxPk v;
    const int UK = K / 32;
    v.R = (UK + 31) / 32;
    v.RF = UK / 32;
    v.tbl = (UK % 32 + 3) / 4;
    v.PB = 8 * v.RF + v.tbl;
    const int cap = (int)h[4], ne = (int)h[5];
    const long long o_pk = mx_up64(64 + (long long)N * (K / 2)), o_idx = mx_up64(o_pk + (long long)N * v.PB);
    const long long o_raw = mx_up64(o_idx + 4LL * cap), total = mx_up64(o_raw + 32LL * cap);
    v.ok = h[0] == kPkMagic && h[1] == kPkVersion && (int)h[2] == N && (int)h[3] == K && cap >= 0 &&
           (long long)cap <= (long long)N * v.R && ne >= 0 && ne <= cap && h[6] <= 252u &&
           (((unsigned long long)h[9] << 32) | h[8]) == (unsigned long long)total;
    v.codes = p + 64;
    v.pk = p + o_pk;
    v.idx = p + o_idx;
    v.raw = p + o_raw;
    v.n_exc = v.ok ? ne : 0;
    v.e0 = h[6];
    return v;
}
/* The first raw round at or after round index g, found by the whole wave (every lane active, as at the start
 * of a row): each step the 32 lanes test the last entry of 32 even buckets of [lo, hi) at once and the ballot
 * of "below g" (a prefix: the index ascends) names the bucket it is in, so up to 1024 raw rounds take two
 * loads a lane and a last one, not a binary search's ten dependent loads a row. */
__device__ __forceinline__ int mx_pk_lower(const MxPk& v, uint32_t g, int lane) {
    int lo = 0, hi = v.n_exc;
    while (hi - lo > 32) {
        const long long span = hi - lo;
        const int last = lo + (int)(((long long)(lane + 1) * span) / 32) - 1;
        const int nb = __builtin_popcount(__builtin_amdgcn_ballot_w32(mx_d32(v.idx + 4 * last) < g));
        const int nlo = lo + (int)(((long long)nb * span) / 32);
        hi = nb < 32 ? lo + (int)(((long long)(nb + 1) * span) / 32) : hi;
        lo = nlo;
    }
    const int i = lo + lane;
    return lo + __builtin_popcount(__builtin_amdgcn_ballot_w32(i < hi && mx_d32(v.idx + 4 * i) < g));
}

/* mx_row over a packed weight: blocks [U0, U1) of rows n0.. into the sums, lane = block mod 32 (U0 a multiple
 * of 32), UF blocks a lane in flight. The same blocks, groups and order as mx_row, so the same bits. */
template <int RPW>
__device__ __forceinline__ void mx_row_pk(const MxGemvArgs& A, const MxPk& v, const float* xs, int xr, int kb,
                                          int U0, int U1, int n0, int mc, int lane, const void* tab,
                                          float (&acc)[RPW][kMxXM]) {
    constexpr int UF = 4;
    const long long CB = A.K / 2;                       /* code bytes a row */
    /* This lane's block j = lane of every round: its code's byte and shift in a full round and in the last. */
    const int fb = lane & 7, fs = 2 * (lane >> 3);
    const int pb = v.tbl ? lane % v.tbl : 0, ps = v.tbl ? 2 * (lane / v.tbl) : 0;
    bool ok[RPW];
    const uint8_t* cr[RPW];
    const uint8_t* pr[RPW];
    uint32_t g[RPW];                                    /* the row's first round index, n * R */
    int cur[RPW];                                       /* its next raw round */
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        ok[r] = n0 + r < A.N;
        const long long n = ok[r] ? n0 + r : 0;
        cr[r] = v.codes + n * CB;
        pr[r] = v.pk + n * v.PB;
        g[r] = (uint32_t)n * (uint32_t)v.R;
        cur[r] = v.n_exc ? mx_pk_lower(v, g[r] + (uint32_t)(U0 >> 5), lane) : 0;
    }
    for (int U = U0 + lane; U < U1; U += 32 * UF) {
        uint32_t raw[UF][RPW][Mx<MX_PK>::NR];
        mx_for<UF>([&](auto fi) {
            constexpr int F = decltype(fi)::value;
            const int V = U + 32 * F;
            if (V < U1) {
                const int rr = V >> 5;
#pragma unroll
                for (int r = 0; r < RPW; ++r)
                    if (ok[r]) {
                        const r4d_u32x4 q = *(mx_gu128)(const void*)(cr[r] + (long long)V * 16);
                        raw[F][r][1] = q[0]; raw[F][r][2] = q[1]; raw[F][r][3] = q[2]; raw[F][r][4] = q[3];
                        uint32_t e;
                        if (cur[r] < v.n_exc && mx_d32(v.idx + 4 * cur[r]) == g[r] + (uint32_t)rr) {
                            e = *(mx_gu8)(const void*)(v.raw + 32LL * cur[r] + lane);
                            ++cur[r];
                        } else {
                            const bool full = rr < v.RF;
                            const uint32_t by = *(mx_gu8)(const void*)(pr[r] + 8 * rr + (full ? fb : pb));
                            e = v.e0 + ((by >> (full ? fs : ps)) & 3u);
                        }
                        raw[F][r][0] = e;
                    }
            }
        });
        mx_for<UF>([&](auto fi) {
            constexpr int F = decltype(fi)::value;
            const int V = U + 32 * F;
            if (V < U1) mx_unit<MX_PK, RPW>(raw[F], V, xs, xr, kb, mc, tab, ok, acc);
        });
    }
}
/* A marked entry: its packed rows, or NaN sums when its header is not a packed [N, K] weight's -- an engine
 * that marked a weight it did not pack gets NaN rows and a reply that differs, never quiet wrong bits. */
template <int RPW>
__device__ __forceinline__ void mx_row_vp(const MxGemvArgs& A, const MxPk& v, const float* xs, int xr, int kb,
                                          int U0, int U1, int n0, int mc, int lane, const void* tab,
                                          float (&acc)[RPW][kMxXM]) {
    if (v.ok) {
        mx_row_pk<RPW>(A, v, xs, xr, kb, U0, U1, n0, mc, lane, tab, acc);
        return;
    }
#pragma unroll
    for (int r = 0; r < RPW; ++r)
#pragma unroll
        for (int m = 0; m < kMxXM; ++m) acc[r][m] = __uint_as_float(0x7fc00000u);
}

/* VP: the gemv_vp row's kernel. A table entry with bit 0 set is a packed weight (mx_row_pk), any other is
 * block_mxfp4 (mx_row), so the same kernel serves the engine's packed VRAM residents and every other expert
 * (pinned-pool, staged, aliased) beside them. Only MX_FP4 over a table is instantiated with it. */
template <int T, int RPW, bool MOE, bool PROBE=false, bool VP=false>
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
    int n_items = A.ipe, rgpi = A.rgpi, ipe = A.ipe;
    if constexpr (MOE) {
        /* The experts with rows, in ascending order. */
        if (wave == 0) {
            int cnt = 0;
            for (int e0 = 0; e0 < A.n_expert; e0 += 32) {
                const int e = e0 + lane;
                int s = 0, t = 0;
                if (e < A.n_expert) { if(A.live_end)ds41_live_tail::range(A.off,e,A.M,&s,&t);else mx_range(A.off, e, A.n_expert, A.M, &s, &t); }
                const bool nz = e < A.n_expert && t > s;
                const uint32_t bal = __builtin_amdgcn_ballot_w32(nz);
                if (nz) live[cnt + __builtin_popcount(bal & ((1u << lane) - 1u))] = e;
                cnt += __builtin_popcount(bal);
            }
            if (lane == 0) s_nlive = cnt;
        }
        __syncthreads();
        /* The host sized rgpi and ipe for min(n_expert, T) experts, but a decode pass on one EP
         * rank usually has one or two live: spread the live experts' row groups over every
         * launched block instead. A row is still one wave's, in the same order, so the bits are
         * those of any other split. */
        const int total = s_nlive * nrg;
        rgpi = total > (int)gridDim.x ? (total + (int)gridDim.x - 1) / (int)gridDim.x : 1;
        ipe = (nrg + rgpi - 1) / rgpi;
        n_items = s_nlive * ipe;
    } else {
        __syncthreads();
    }
    bool did_work=false;unsigned long long owned_rows=0,passes=0;
    int cur_e = -1, cur_p = -1;                       /* the rows the LDS buffer holds whole */
    for (int it = (int)blockIdx.x; it < n_items; it += (int)gridDim.x) {
        int e = 0, ri = it, rs = 0, re = A.M;
        const uint8_t* wb = A.w;
        if constexpr (MOE) {
            e = live[it / ipe];
            ri = it - (it / ipe) * ipe;
            if(A.live_end)ds41_live_tail::range(A.off,e,A.M,&rs,&re);else mx_range(A.off, e, A.n_expert, A.M, &rs, &re);
            wb = A.wtab ? A.wtab[e] : A.w + (long long)e * A.w_es;
        }
        /* gemv_vp: bit 0 of a table entry marks a packed weight (block-uniform: one expert an item). Weights are
         * 64-byte aligned, so an unmarked entry is the address as it is. */
        [[maybe_unused]] bool pk = false;
        [[maybe_unused]] MxPk pv{};
        if constexpr (VP) {
            if (A.wtab) {
                pk = ((uintptr_t)wb & 1u) != 0;
                wb = (const uint8_t*)((uintptr_t)wb & ~(uintptr_t)1);
            }
            if (pk) pv = mx_pk_open(wb, A.N, A.K);
        }
        did_work=true;if constexpr(PROBE)if(ri==0)owned_rows+=(unsigned long long)(re-rs);
        const int rg0 = ri * rgpi, rg1 = min(rg0 + rgpi, nrg);
        /* An expert with more rows than the LDS holds whole (one gate_up row at K 5120) takes them in
         * whole passes of XLM rows, re-reading its weights, rather than restaging K chunks for every
         * row group. Each output is still its weight row dotted with its activation row in the same
         * lane and block order, so the bits are unchanged; only rows that cannot fit even one whole
         * (XLM 0) use the chunked path.
         *
         * one_pass (RADIANCE_EXPERIMENTAL_DS41_DRAFT_GEMV): the other way round, for weights the card
         * reads in place from pinned host memory (the DSpark draft bank). There a re-read is another
         * trip over the link -- 12.5 MB of gate_up at ~18 GB/s, 0.7 ms -- while restaging a row
         * group's K chunks reads activations the caches already hold, so all of the expert's rows (up
         * to XM) take one pass over its weights, in K chunks when they do not fit whole. The chunked
         * path keeps each lane's blocks and their order (chunks are whole multiples of 32 blocks), so
         * the bits are again unchanged. In whole passes of one row the draft's gate_up took 6.1-6.7 ms
         * a call on ranks 0 and 1 and 11.2-11.8 ms on ranks 2 and 3 (whose cards share a root
         * complex), its down (three rows a pass) 0.5-1.2 ms (ab-20261029T154954Z, Ds1-o). */
        const int step = !A.one_pass && A.XLM >= 1 && A.XLM < A.XM ? A.XLM : A.XM;
        if constexpr(PROBE)if(ri==0)passes+=(unsigned long long)((re-rs+step-1)/step);
        for (int p0 = rs; p0 < re; p0 += step) {
            const int mc = min(step, re - p0);
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
                    if (active) {
                        if constexpr (VP) {
                            if (pk) mx_row_vp<RPW>(A, pv, xs, A.XR, 0, 0, UK, n0, mc, lane, tab, acc);
                            else mx_row<T, RPW>(A, wb, xs, A.XR, 0, 0, UK, n0, mc, lane, tab, acc);
                        } else {
                            mx_row<T, RPW>(A, wb, xs, A.XR, 0, 0, UK, n0, mc, lane, tab, acc);
                        }
                    }
                } else {
                    /* The pass's rows do not fit whole: K in chunks, restaged a row group (every wave takes
                     * part in the barriers). */
                    for (int k0 = 0; k0 < A.K; k0 += A.KC) {
                        const int k1 = min(k0 + A.KC, A.K);
                        lds_barrier();
                        mx_stage<MOE>(A, xs, p0, mc, k0, k1, XRC);
                        lds_barrier();
                        if (active) {
                            if constexpr (VP) {
                                if (pk) mx_row_vp<RPW>(A, pv, xs, XRC, k0, k0 / Q::QK, k1 / Q::QK, n0, mc, lane, tab, acc);
                                else mx_row<T, RPW>(A, wb, xs, XRC, k0, k0 / Q::QK, k1 / Q::QK, n0, mc, lane, tab, acc);
                            } else {
                                mx_row<T, RPW>(A, wb, xs, XRC, k0, k0 / Q::QK, k1 / Q::QK, n0, mc, lane, tab, acc);
                            }
                        }
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
    if constexpr(PROBE){__syncthreads();if(tid==0){atomicAdd(A.probe+(did_work?1:4),1ull);atomicAdd(A.probe,owned_rows);
                        if(A.probe_passes)atomicAdd(A.probe+5,passes);}}
}

/* ================================================================== gemv_vp's kernel (vp2)
 *
 * mx_vp2_kernel serves the gemv_vp row: a table whose entries name the packed-scale form (bit 0) or
 * block_mxfp4, as mx_gemv_kernel<.., VP> does, and the same bits as the plain row -- each output is lane L's
 * blocks L, L + 32, ... in increasing order, a block's four groups of eight values in order, one rounded
 * product and one rounded sum a value (w = kv[nibble] * 2^(e - 128) from the same table), then r4d_wave_sum.
 * What changes is how the work reaches the card:
 *   - a pass's weight loads are all issued before any of them is used. The plain loop puts each of a lane's
 *     four blocks behind its own `V < U1` branch, and the compiler waits for one block's bytes (to funnel and
 *     decode them) before it issues the next block's loads: four round trips an iteration. Here a lane's
 *     blocks past U1 read its last valid block again (in bounds, never summed), so the loads are one run;
 *   - UF blocks a lane a pass, so a whole row is one pass: 5 for gate_up's 160 blocks, 3 for down's 72;
 *   - MC activation rows as a template (1 at decode) instead of eight guarded ones;
 *   - the packed scales: a block's 2-bit code is a byte load beside its codes; a row's raw rounds come from
 *     a binary search on the scalar unit (its loads keep their own counter, so the vector loads in flight
 *     are not waited for) instead of a vector cursor walk inside the loop;
 *   - a value's table offset is one byte of a shifted and masked code dword (kept opaque, or the compiler
 *     folds it back into a shift and a mask a value), and the table is a static LDS array at a constant
 *     address, so the table load takes that byte as its whole address (a table in the dynamic LDS costs an
 *     add a value: its base is a link-time constant the load's offset field cannot hold);
 *   - the wave index, the expert, its range and its weight's address sit in scalar registers; the live
 *     scan issues its offset loads together and keeps each live expert's range in LDS;
 *   - the activation row's token through the scalar unit and its chunks loaded before any is stored.
 * Two weight rows a wave (four measured slower: 236-255 VGPRs), the compiler's own register budget (a target of
 * 10 waves a SIMD measured slower). Its forms (kVpSraw, kVpRoll, kVpZero, below) change how a pass is
 * decoded, never what; RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED_KERNEL keeps the first form and the first vp2 form
 * selectable for an A/B (mx_vp_pick). */
typedef __attribute__((address_space(4))) const uint32_t* mx_cu32;
/* A dword every lane reads at the same 4-aligned address, through the scalar unit. */
__device__ __forceinline__ uint32_t mx_s32(const void* p) { return *(mx_cu32)p; }
__device__ __forceinline__ int mx_uni(int v) { return __builtin_amdgcn_readfirstlane(v); }
__device__ __forceinline__ const uint8_t* mx_uni_ptr(const uint8_t* p) {
    const unsigned long long u = (unsigned long long)(uintptr_t)p;
    const unsigned lo = (unsigned)__builtin_amdgcn_readfirstlane((int)(unsigned)u);
    const unsigned hi = (unsigned)__builtin_amdgcn_readfirstlane((int)(unsigned)(u >> 32));
    return (const uint8_t*)(uintptr_t)(((unsigned long long)hi << 32) | lo);
}

/* The dynamic LDS: the live experts' records (three words each), then the staged rows. */
constexpr int kMxVp2Live = 0;

/* A register the compiler cannot see into (device code; the host emulator has no use for it), so what reads it
 * is not folded back into how it was made. */
__device__ __forceinline__ uint32_t mx_opaque(uint32_t v) {
#if defined(__HIP_DEVICE_COMPILE__)
    asm("" : "+v"(v));
#endif
    return v;
}

/* mx_pk_open with the header read through the scalar unit: the same sections and the same judgement. */
struct MxPk2 {
    const uint8_t* codes;
    const uint8_t* pk;
    const uint8_t* idx;
    const uint8_t* raw;
    int      n_exc, R, RF, PB, tbl;
    uint32_t e0;
    bool     ok;
};
__device__ __forceinline__ MxPk2 mx_pk2_open(const uint8_t* p, int N, int K) {
    uint32_t h[12];
#pragma unroll
    for (int i = 0; i < 12; ++i) h[i] = mx_s32(p + 4 * i);
    MxPk2 v;
    const int UK = K / 32;
    v.R = (UK + 31) / 32;
    v.RF = UK / 32;
    v.tbl = (UK % 32 + 3) / 4;
    v.PB = 8 * v.RF + v.tbl;
    const int cap = (int)h[4], ne = (int)h[5];
    const long long o_pk = mx_up64(64 + (long long)N * (K / 2)), o_idx = mx_up64(o_pk + (long long)N * v.PB);
    const long long o_raw = mx_up64(o_idx + 4LL * cap), total = mx_up64(o_raw + 32LL * cap);
    v.ok = h[0] == kPkMagic && h[1] == kPkVersion && (int)h[2] == N && (int)h[3] == K && cap >= 0 &&
           (long long)cap <= (long long)N * v.R && ne >= 0 && ne <= cap && h[6] <= 252u &&
           (((unsigned long long)h[9] << 32) | h[8]) == (unsigned long long)total;
    v.codes = p + 64;
    v.pk = p + o_pk;
    v.idx = p + o_idx;
    v.raw = p + o_raw;
    v.n_exc = v.ok ? ne : 0;
    v.e0 = h[6];
    return v;
}

/* The raw rounds among row rounds g .. g + UF - 1 (bit j: g + j) and the index of the first of them, the
 * others following it in order: a lower bound over the ascending index, then the run from there, all on the
 * scalar unit (g and the weight are the wave's). */
template <int UF>
__device__ __forceinline__ uint32_t mx_vp2_raw(const MxPk2& v, uint32_t g, int* first) {
    int lo = 0, hi = v.n_exc;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (mx_s32(v.idx + 4LL * mid) < g) lo = mid + 1;
        else hi = mid;
    }
    *first = lo;
    uint32_t m = 0;
    for (int p = lo; p < v.n_exc; ++p) {
        const uint32_t x = mx_s32(v.idx + 4LL * p);   /* >= g: past the lower bound */
        if (x - g >= (uint32_t)UF) break;
        m |= 1u << (x - g);
    }
    return m;
}

/* mx_vp2_kernel's forms (FL, a set of bits; 0 the form above):
 *   kVpSraw  a packed pass's raw rounds found once for the wave's rows (one lower bound for its first row, then
 *            the run from there) and their e8m0 bytes read through the scalar unit when a block needs one, so
 *            the vector loads in flight count only the weight's own words and a block's decode waits for its
 *            own loads alone (a raw byte loaded into a block's register makes every block wait for every load);
 *   kVpRoll  the pass's blocks decoded by one rolled loop, each from the front of the loaded words (which then
 *            move up one place), so the code a pass runs is one block's decode instead of UF copies of it;
 *   kVpZero  a pass whose rows are all dropped slots (rows routed to other ranks, which mx_range gives the
 *            rank's last expert: five of a decode call's six passes) stores its zeros and nothing else -- every
 *            form stores 0.0f for a dropped slot whatever its sums are, so its loads, staging and arithmetic
 *            change no bit.
 * Every form gives the same bits; kVpSraw and kVpRoll read the same words and do the same arithmetic in the same
 * order as the first form. (Measured and dropped: each block's words read in place through M0-relative moves
 * instead of moved up, and packed values two at a time from a 256-entry pair table -- fewer VALU operations,
 * but slower a call: more registers and a full wait, and LDS bank conflicts on the table.) */
enum : int { kVpSraw = 1, kVpRoll = 2, kVpZero = 4 };

/* Lane `lane`'s byte of the 32 at p (4-aligned, the same for the whole wave), through the scalar unit. */
__device__ __forceinline__ uint32_t mx_s_lane_byte(const uint8_t* p, int lane) {
    uint32_t w[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) w[i] = mx_s32(p + 4 * i);
    const int q = lane >> 2;
    uint32_t d = w[0];
#pragma unroll
    for (int i = 1; i < 8; ++i) d = q == i ? w[i] : d;
    return (d >> (8 * (lane & 3))) & 0xFFu;
}

/* kVpSraw's raw rounds of a pass, for the wave's rows nr[0] <= nr[1] <= ... (rows past N repeat the last): bit j
 * of m[r] when round rr0 + j of row nr[r] is raw (j below min(UF, R - rr0), the pass's rounds of that row), and
 * first[r] the index of the first of row r's (the others follow it in order). One lower bound over the
 * ascending index for the first row's first round, then the run up to the last row's window, on the scalar
 * unit. */
template <int RPW, int UF>
__device__ __forceinline__ void mx_vp2_raw_rows(const MxPk2& v, const int (&nr)[RPW], int rr0, uint32_t (&m)[RPW],
                                                int (&first)[RPW]) {
    const uint32_t R = (uint32_t)v.R;
    const uint32_t wl = (uint32_t)(UF < v.R - rr0 ? UF : v.R - rr0);
    uint32_t g[RPW];
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        g[r] = (uint32_t)nr[r] * R + (uint32_t)rr0;
        m[r] = 0;
        first[r] = 0;
    }
    int lo = 0, hi = v.n_exc;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (mx_s32(v.idx + 4LL * mid) < g[0]) lo = mid + 1;
        else hi = mid;
    }
    int r = 0;
    bool seen = false;   /* row r has a raw round in its window */
    for (int p = lo; p < v.n_exc; ++p) {
        const uint32_t x = mx_s32(v.idx + 4LL * p);
        /* Past row r's window: on to the next distinct row (a repeated row is filled in below). */
        while (r < RPW && x >= g[r] + wl) {
            ++r;
            seen = false;
            while (r < RPW && g[r] == g[r - 1]) ++r;
        }
        if (r >= RPW) break;
        if (x < g[r]) continue;   /* between two rows' windows: rounds of neither in this pass */
        if (!seen) {
            first[r] = p;
            seen = true;
        }
        m[r] |= 1u << (x - g[r]);
    }
#pragma unroll
    for (int q = 1; q < RPW; ++q)
        if (g[q] == g[q - 1]) {
            m[q] = m[q - 1];
            first[q] = first[q - 1];
        }
}

/* The table value at byte offset `off` (4 * nibble). */
__device__ __forceinline__ float mx_tab_at(const float* tab, uint32_t off) {
    return *(const float*)((const uint8_t*)tab + off);
}

/* One block of RPW rows into the sums: qs[r] its 16 code bytes as four dwords, PKL the packed form's order
 * (value 2i the low nibble of byte i) else block_mxfp4's (value j < 16 the low nibble of byte j, j + 16 its
 * high one), d[r] its scale; xb the staged activations at the block. mx_unit's groups, values and sums. */
template <bool PKL, int RPW, int MC>
__device__ __forceinline__ void mx_vp2_unit(const uint32_t (&qs)[RPW][4], const float (&d)[RPW], const float* xb,
                                            int xr, int mc, const float* tab, float (&acc)[RPW][MC]) {
    mx_for<4>([&](auto gi) {
        constexpr int G = decltype(gi)::value;
        float w[RPW][8];
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            /* Two dwords of four table offsets each: the packed form's group G is code dword G (a0 its even
             * values, a1 its odd ones); block_mxfp4's is the low (G < 2) or high nibbles of qs dwords
             * 2 (G & 1) and 2 (G & 1) + 1 (a0 values 0..3, a1 4..7). */
            uint32_t a0, a1;
            if constexpr (PKL) {
                a0 = (qs[r][G] << 2) & 0x3C3C3C3Cu;
                a1 = (qs[r][G] >> 2) & 0x3C3C3C3Cu;
            } else if constexpr (G < 2) {
                a0 = (qs[r][2 * (G & 1)] << 2) & 0x3C3C3C3Cu;
                a1 = (qs[r][2 * (G & 1) + 1] << 2) & 0x3C3C3C3Cu;
            } else {
                a0 = (qs[r][2 * (G & 1)] >> 2) & 0x3C3C3C3Cu;
                a1 = (qs[r][2 * (G & 1) + 1] >> 2) & 0x3C3C3C3Cu;
            }
            a0 = mx_opaque(a0);
            a1 = mx_opaque(a1);
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                const uint32_t a = PKL ? ((t & 1) ? a1 : a0) : (t < 4 ? a0 : a1);
                const int b = PKL ? (t >> 1) : (t & 3);
                w[r][t] = mx_tab_at(tab, (a >> (8 * b)) & 0xFFu) * d[r];
            }
        }
#pragma unroll
        for (int m = 0; m < MC; ++m) {
            if (MC == 1 || m < mc) {
                const float* xp = xb + m * xr + 8 * G;
                const float4 xa = *(const float4*)xp;
                const float4 xc = *(const float4*)(xp + 4);
                const float xv[8] = { xa.x, xa.y, xa.z, xa.w, xc.x, xc.y, xc.z, xc.w };
#pragma unroll
                for (int r = 0; r < RPW; ++r)
#pragma unroll
                    for (int t = 0; t < 8; ++t) acc[r][m] += xv[t] * w[r][t];
            }
        }
    });
}

/* One block of the pass (F its place, V = U + 32 F its index) of RPW rows into the sums, from its loaded words rw
 * (mx_vp2_row's raw[F]): its codes and its scale, then mx_vp2_unit. rawm and first are the pass's raw rounds
 * (mx_vp2_raw_rows under kVpSraw, whose bytes are read here; otherwise already in rw[r][4]). */
template <bool PK, bool SRAW, int RPW, int MC>
__device__ __forceinline__ void mx_vp2_block(const MxGemvArgs& A, const uint8_t* wb, const MxPk2& v,
                                             const uint32_t (&rw)[RPW][5], const int (&nr)[RPW], int F, int U,
                                             int U1, int rr0, int xb0, const uint32_t (&rawm)[RPW],
                                             const int (&first)[RPW], int fs, int ps, const float* xs, int xr,
                                             int mc, int lane, const float* tab, float (&acc)[RPW][MC]) {
    const int V = U + 32 * F;
    if (V >= U1) return;
    uint32_t qs[RPW][4];
    float d[RPW];
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        uint32_t e;
        if constexpr (PK) {
#pragma unroll
            for (int i = 0; i < 4; ++i) qs[r][i] = rw[r][i];
            const bool full = rr0 + F < v.RF;
            const bool isr = ((rawm[r] >> F) & 1u) != 0;   /* wave-uniform */
            if constexpr (SRAW) {
                if (isr) {
                    int slot = first[r] + __builtin_popcount(rawm[r] & ((1u << F) - 1u));
                    slot = slot < v.n_exc ? slot : v.n_exc - 1;
                    e = mx_s_lane_byte(v.raw + 32LL * slot, lane);
                } else {
                    e = v.e0 + ((rw[r][4] >> (full ? fs : ps)) & 3u);
                }
            } else {
                e = isr ? rw[r][4] : v.e0 + ((rw[r][4] >> (full ? fs : ps)) & 3u);
            }
        } else {
            /* Mx<MX_FP4>::load's funnel: e at byte s of the window, qs from byte s + 1. */
            const int s = (int)((uintptr_t)(wb + (long long)nr[r] * A.w_ld + (long long)V * 17) & 3u);
            e = (rw[r][0] >> (8 * s)) & 0xFFu;
            const bool next = s == 3;
            const int sh = (s + 1) & 3;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t lo = next ? rw[r][i + 1] : rw[r][i];
                const uint32_t hi = next ? rw[r][i + 1 < 4 ? i + 2 : 4] : rw[r][i + 1];
                qs[r][i] = mx_funnel(lo, hi, sh);
            }
        }
        d[r] = mx_e8m0_half(e);
    }
    mx_vp2_unit<PK, RPW, MC>(qs, d, xs + 36 * (V - xb0), xr, mc, tab, acc);
}

/* A pass's loads: blocks U + 32 F (F < UF; a lane past U1 reads the last valid block again, never summed) of
 * rows nr[r] into raw[F][r]: block_mxfp4's five aligned dwords over the block, or the packed form's four code
 * dwords and the byte that holds the block's 2-bit code (fb, pb: this lane's byte in a full round and in the
 * last). */
template <bool PK, int RPW, int UF>
__device__ __forceinline__ void mx_vp2_load(const MxGemvArgs& A, const uint8_t* wb, const MxPk2& v,
                                            const int (&nr)[RPW], int U, int U1, int rr0, int fb, int pb,
                                            uint32_t (&raw)[UF][RPW][5]) {
    mx_for<UF>([&](auto fi) {
        constexpr int F = decltype(fi)::value;
        const int V = U + 32 * F;
        const long long Vc = V < U1 ? V : U1 - 1;
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            if constexpr (PK) {
                const r4d_u32x4 c = *(mx_gu128)(const void*)(v.codes + (long long)nr[r] * (A.K / 2) + Vc * 16);
                raw[F][r][0] = c[0];
                raw[F][r][1] = c[1];
                raw[F][r][2] = c[2];
                raw[F][r][3] = c[3];
                const int rr = rr0 + F < v.R ? rr0 + F : v.R - 1;
                raw[F][r][4] =
                    *(mx_gu8)(const void*)(v.pk + (long long)nr[r] * v.PB + 8 * rr + (rr < v.RF ? fb : pb));
            } else {
                const uint8_t* a =
                    (const uint8_t*)((uintptr_t)(wb + (long long)nr[r] * A.w_ld + Vc * 17) & ~(uintptr_t)3);
#pragma unroll
                for (int i = 0; i < 5; ++i) raw[F][r][i] = mx_d32(a + 4 * i);
            }
        }
    });
}

/* Blocks [U0, U1) of rows n0 .. n0 + RPW - 1 into the sums, lane = block mod 32 (U0 a multiple of 32), UF
 * blocks a lane a pass; the staged rows hold values from kb on, xr floats a row. PK: the packed form `v`, else
 * block_mxfp4 at `wb`. Rows past N read row N - 1 and are never stored. Each pass issues all of its loads
 * first; the compute keeps mx_row's blocks and order. FL: the form (kVpSraw, kVpRoll). */
template <bool PK, int RPW, int MC, int UF, int FL>
__device__ __forceinline__ void mx_vp2_row(const MxGemvArgs& A, const uint8_t* wb, const MxPk2& v, const float* xs,
                                           int xr, int kb, int U0, int U1, int n0, int mc, int lane,
                                           const float* tab, float (&acc)[RPW][MC]) {
    constexpr bool SRAW = (FL & kVpSraw) != 0, ROLL = (FL & kVpRoll) != 0;
    int nr[RPW];
#pragma unroll
    for (int r = 0; r < RPW; ++r) nr[r] = n0 + r < A.N ? n0 + r : A.N - 1;
    const int xb0 = kb >> 5;
    /* This lane's block j = lane of every round: its code's byte and shift in a full round and in the last. */
    [[maybe_unused]] const int fb = lane & 7, fs = 2 * (lane >> 3);
    [[maybe_unused]] int pb = 0, ps = 0;
    if constexpr (PK) {
        pb = v.tbl ? lane % v.tbl : 0;
        ps = v.tbl ? 2 * (lane / v.tbl) : 0;
    }
    for (int u0 = U0; u0 < U1; u0 += 32 * UF) {   /* the wave's pass: rounds u0 / 32 .. + UF - 1 */
        const int U = u0 + lane;
        const int rr0 = u0 >> 5;
        /* block_mxfp4: the five aligned dwords over the block; packed: its four code dwords and the byte that
         * holds its 2-bit code (raw rounds: its e8m0 byte, below or, under kVpSraw, at its decode). */
        uint32_t raw[UF][RPW][5];
        mx_vp2_load<PK, RPW, UF>(A, wb, v, nr, U, U1, rr0, fb, pb, raw);
        /* Nothing below moves above the loads: the scheduler would interleave them with their uses to save
         * registers, a round trip each again. */
        sched_barrier();
        /* The raw rounds of the pass's rounds on the scalar unit (rare: the engine's residents have a few in a
         * weight). The first form reads their e8m0 bytes over the code bytes here; kVpSraw at their blocks. */
        uint32_t rawm[RPW];
        int first[RPW];
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            rawm[r] = 0;
            first[r] = 0;
        }
        if constexpr (PK) {
            if (v.n_exc > 0) {
                if constexpr (SRAW) {
                    mx_vp2_raw_rows<RPW, UF>(v, nr, rr0, rawm, first);
                } else {
#pragma unroll
                    for (int r = 0; r < RPW; ++r)
                        rawm[r] = mx_vp2_raw<UF>(v, (uint32_t)nr[r] * (uint32_t)v.R + (uint32_t)rr0, &first[r]);
#pragma unroll
                    for (int r = 0; r < RPW; ++r) {
                        if (rawm[r]) {
                            mx_for<UF>([&](auto fi) {
                                constexpr int F = decltype(fi)::value;
                                if ((rawm[r] >> F) & 1u) {
                                    const int slot = first[r] + __builtin_popcount(rawm[r] & ((1u << F) - 1u));
                                    raw[F][r][4] = *(mx_gu8)(const void*)(v.raw + 32LL * slot + lane);
                                }
                            });
                        }
                    }
                }
            }
        }
        if constexpr (!ROLL) {
            /* A block a branch (the lanes past U1 skip theirs): each block's decode is its own region, so the
             * compiler keeps one block's table values live at a time. One region for all of them (a
             * branch-free form) hoisted every block's decode and spilled. */
            mx_for<UF>([&](auto fi) {
                constexpr int F = decltype(fi)::value;
                mx_vp2_block<PK, SRAW, RPW, MC>(A, wb, v, raw[F], nr, F, U, U1, rr0, xb0, rawm, first, fs, ps, xs,
                                                xr, mc, lane, tab, acc);
            });
        } else {
            /* kVpRoll: block F from raw[0], then every later block's words one place up. */
#pragma unroll 1
            for (int F = 0; F < UF; ++F) {
                mx_vp2_block<PK, SRAW, RPW, MC>(A, wb, v, raw[0], nr, F, U, U1, rr0, xb0, rawm, first, fs, ps, xs,
                                                xr, mc, lane, tab, acc);
#pragma unroll
                for (int f = 0; f + 1 < UF; ++f)
#pragma unroll
                    for (int r = 0; r < RPW; ++r)
#pragma unroll
                        for (int i = 0; i < 5; ++i) raw[f][r][i] = raw[f + 1][r][i];
            }
        }
    }
}

/* One weight's rows: a marked entry its packed rows, or NaN sums when its header is not a packed [N, K]
 * weight's (as mx_row_vp); any other entry block_mxfp4. */
template <int RPW, int MC, int UF, int FL>
__device__ __forceinline__ void mx_vp2_rows(const MxGemvArgs& A, const uint8_t* wb, bool pk, const MxPk2& v,
                                            const float* xs, int xr, int kb, int U0, int U1, int n0, int mc,
                                            int lane, const float* tab, float (&acc)[RPW][MC]) {
    if (!pk) {
        mx_vp2_row<false, RPW, MC, UF, FL>(A, wb, v, xs, xr, kb, U0, U1, n0, mc, lane, tab, acc);
    } else if (v.ok) {
        mx_vp2_row<true, RPW, MC, UF, FL>(A, wb, v, xs, xr, kb, U0, U1, n0, mc, lane, tab, acc);
    } else {
#pragma unroll
        for (int r = 0; r < RPW; ++r)
#pragma unroll
            for (int m = 0; m < MC; ++m) acc[r][m] = __uint_as_float(0x7fc00000u);
    }
}

/* mx_moe_tok with sorted[i] through the scalar unit (i is the block's). */
__device__ __forceinline__ int mx_vp2_tok(const MxGemvArgs& A, int i) {
    const int flat = (int)mx_s32(A.sorted + i);
    if (flat < 0) return -1;
    const int tok = A.a_sorted ? i : flat / A.top_k;
    return tok < A.AR ? tok : -1;
}

/* mx_stage for the pass's rows p0 .. p0 + mc - 1 (MC at most): values [k0, k1) widened into LDS by the block,
 * `xr` floats a row, a row's token read once and each thread's four chunks loaded before any is stored. */
template <int MC>
__device__ __forceinline__ void mx_vp2_stage(const MxGemvArgs& A, float* xs, int p0, int mc, int k0, int k1, int xr,
                                             int tid) {
    const int c8n = (k1 - k0) >> 3;
    for (int m = 0; m < (MC == 1 ? 1 : mc); ++m) {
        const int tok = mx_vp2_tok(A, p0 + m);
        const uint16_t* xrow = tok >= 0 ? A.x + (long long)tok * A.x_ld + k0 : nullptr;
        float* xd = xs + m * xr;
        for (int c0 = 0; c0 < c8n; c0 += 4 * kMxThreads) {
            uint32_t q[4][4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = c0 + j * kMxThreads + tid;
                if (xrow) {
                    mx_x8raw(xrow + 8 * (c < c8n ? c : c8n - 1), A.x_al != 0, q[j]);
                } else {
#pragma unroll
                    for (int i = 0; i < 4; ++i) q[j][i] = 0u;
                }
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = c0 + j * kMxThreads + tid;
                if (c < c8n) {
                    float* p = xd + mx_xphys(8 * c);
                    *(float4*)p = make_float4(bf16w_lo(q[j][0]), bf16w_hi(q[j][0]), bf16w_lo(q[j][1]),
                                              bf16w_hi(q[j][1]));
                    *(float4*)(p + 4) = make_float4(bf16w_lo(q[j][2]), bf16w_hi(q[j][2]), bf16w_lo(q[j][3]),
                                                    bf16w_hi(q[j][3]));
                }
            }
        }
    }
}

/* Expert e's rows (mx_range, or the live-tail ranges). */
__device__ __forceinline__ void mx_vp2_range(const MxGemvArgs& A, int e, int* s, int* t) {
    if (A.live_end) ds41_live_tail::range(A.off, e, A.M, s, t);
    else mx_range(A.off, e, A.n_expert, A.M, s, t);
}

/* The gemv_vp row's kernel body: MX_FP4 experts over a table (A.wtab), the grouped GEMM's decode band. The work
 * split and the passes are mx_gemv_kernel's (the live experts' row groups spread over the launched blocks, an
 * expert's rows in whole passes of XLM rows, K in chunks when not even one row fits); MC >= the pass height. */
template <int RPW, int MC, int UF, int FL>
__device__ __forceinline__ void mx_vp2_body(const MxGemvArgs& A) {
    R4D_DYN_LDS(r4d_u32x4, mx_lds);
    __shared__ __attribute__((aligned(16))) float s_kv[16];   /* kMxValues: static, at a constant address */
    __shared__ int s_nlive_v;
    uint8_t* const mx_smem = (uint8_t*)mx_lds;
    const int tid = (int)threadIdx.x, lane = tid & 31, wave = mx_uni(tid >> 5);
    if (tid < 16) s_kv[tid] = kMxValues[tid];
    const float* tab = s_kv;
    int* const s_nlive = &s_nlive_v;
    int* const live_e = (int*)(mx_smem + A.live_off);
    int* const live_s = live_e + A.n_expert;
    int* const live_t = live_s + A.n_expert;
    float* const xs = (float*)(mx_smem + A.x_off);
    /* The experts with rows, ascending, with their ranges: four groups of 32 experts' offsets loaded at once. */
    if (wave == 0) {
        int cnt = 0;
        for (int e0 = 0; e0 < A.n_expert; e0 += 128) {
            int s[4], t[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int e = e0 + 32 * j + lane;
                mx_vp2_range(A, e < A.n_expert ? e : A.n_expert - 1, &s[j], &t[j]);
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int e = e0 + 32 * j + lane;
                const bool nz = e < A.n_expert && t[j] > s[j];
                const uint32_t bal = __builtin_amdgcn_ballot_w32(nz);
                if (nz) {
                    const int at = cnt + __builtin_popcount(bal & ((1u << lane) - 1u));
                    live_e[at] = e;
                    live_s[at] = s[j];
                    live_t[at] = t[j];
                }
                cnt += __builtin_popcount(bal);
            }
        }
        if (lane == 0) *s_nlive = cnt;
    }
    __syncthreads();
    constexpr int RG = kMxWaves * RPW;
    const int nrg = (A.N + RG - 1) / RG;
    const int UK = A.K / 32;
    const int XRC = A.KC + A.KC / 8;
    /* mx_gemv_kernel's split: the live experts' row groups over every launched block, a row one wave's. */
    const int total = *s_nlive * nrg;
    const int rgpi = total > (int)gridDim.x ? (total + (int)gridDim.x - 1) / (int)gridDim.x : 1;
    const int ipe = (nrg + rgpi - 1) / rgpi;
    const int n_items = *s_nlive * ipe;
    const int step = !A.one_pass && A.XLM >= 1 && A.XLM < A.XM ? A.XLM : A.XM;
    int cur_e = -1, cur_p = -1;
    for (int it = (int)blockIdx.x; it < n_items; it += (int)gridDim.x) {
        const int li = it / ipe;
        const int ri = it - li * ipe;
        const int e = mx_uni(live_e[li]), rs = mx_uni(live_s[li]), re = mx_uni(live_t[li]);
        /* Bit 0 of the entry marks a packed weight (block-uniform: one expert an item). */
        const uint8_t* wb = mx_uni_ptr(A.wtab[e]);
        const bool pk = ((uintptr_t)wb & 1u) != 0;
        wb = (const uint8_t*)((uintptr_t)wb & ~(uintptr_t)1);
        MxPk2 pv{};
        if (pk) pv = mx_pk2_open(wb, A.N, A.K);
        const int rg0 = ri * rgpi, rg1 = min(rg0 + rgpi, nrg);
        for (int p0 = rs; p0 < re; p0 += step) {
            const int mc = min(step, re - p0);
            const bool whole = mc <= A.XLM;            /* block-uniform */
            /* The pass's live rows (a dropped slot writes zeros). */
            uint32_t livem = 0;
            if constexpr ((FL & kVpZero) != 0) {
                for (int m = 0; m < mc; ++m)
                    if (mx_vp2_tok(A, p0 + m) >= 0) livem |= 1u << m;
                if (livem == 0) {   /* block-uniform: only dropped slots, whose rows are zeros in every form */
                    for (int rg = rg0; rg < rg1; ++rg) {
                        const int n0 = rg * RG + wave * RPW;
                        if (lane == 0)
                            for (int r = 0; r < RPW; ++r)
                                if (n0 + r < A.N)
                                    for (int m = 0; m < mc; ++m)
                                        mx_put(A.y + (long long)(p0 + m) * A.y_ld + n0 + r, 0.0f);
                    }
                    continue;
                }
            }
            if (whole && (cur_e != e || cur_p != p0)) {
                lds_barrier();
                mx_vp2_stage<MC>(A, xs, p0, mc, 0, A.K, A.XR, tid);
                lds_barrier();
                cur_e = e;
                cur_p = p0;
            }
            if constexpr ((FL & kVpZero) == 0) {
                for (int m = 0; m < mc; ++m)
                    if (mx_vp2_tok(A, p0 + m) >= 0) livem |= 1u << m;
            }
            for (int rg = rg0; rg < rg1; ++rg) {
                const int n0 = rg * RG + wave * RPW;
                const bool active = n0 < A.N;          /* wave-uniform */
                float acc[RPW][MC];
#pragma unroll
                for (int r = 0; r < RPW; ++r)
#pragma unroll
                    for (int m = 0; m < MC; ++m) acc[r][m] = 0.0f;
                if (whole) {
                    if (active)
                        mx_vp2_rows<RPW, MC, UF, FL>(A, wb, pk, pv, xs, A.XR, 0, 0, UK, n0, mc, lane, tab, acc);
                } else {
                    /* K in chunks, restaged a row group (every wave takes part in the barriers). */
                    for (int k0 = 0; k0 < A.K; k0 += A.KC) {
                        const int k1 = min(k0 + A.KC, A.K);
                        lds_barrier();
                        mx_vp2_stage<MC>(A, xs, p0, mc, k0, k1, XRC, tid);
                        lds_barrier();
                        if (active)
                            mx_vp2_rows<RPW, MC, UF, FL>(A, wb, pk, pv, xs, XRC, k0, k0 / 32, k1 / 32, n0, mc, lane,
                                                         tab, acc);
                    }
                    cur_e = -1;
                }
                if (!active) continue;
#pragma unroll
                for (int r = 0; r < RPW; ++r) {
                    const int n = n0 + r;
                    if (n >= A.N) continue;
#pragma unroll
                    for (int m = 0; m < MC; ++m) {
                        if (m >= mc) continue;
                        const float v = r4d_wave_sum(acc[r][m]);
                        if (lane == 0) mx_put(A.y + (long long)(p0 + m) * A.y_ld + n, ((livem >> m) & 1u) ? v : 0.0f);
                    }
                }
            }
        }
    }
}

/* gemv_vp's kernel: the body in form FL, at the compiler's own register budget. */
template <int RPW, int MC, int UF, int FL>
__global__ __launch_bounds__(kMxThreads) void mx_vp2_kernel(MxGemvArgs A) {
    mx_vp2_body<RPW, MC, UF, FL>(A);
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
    int live_end;
    unsigned long long* probe;
};

/* BM activation rows x 128 weight rows a block. BM 128: 8 x 8 outputs a thread; BM 32: 4 x 4. Each output is the
 * same chain at either height (slabs in order, the slab's 32 values in order), so the bits do not depend on BM. */
template <int T, int BM, bool MOE, bool PROBE=false>
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
                if (ee < A.n_expert) { if(A.live_end)ds41_live_tail::range(A.off,ee,A.M,&s,&t);else mx_range(A.off, ee, A.n_expert, A.M, &s, &t); }
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
        if (e < 0) {if constexpr(PROBE)if(tid==0)atomicAdd(A.probe+4,1ull);return;}
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
    if constexpr(PROBE){__syncthreads();if(tid==0){atomicAdd(A.probe+1,1ull);if(blockIdx.y==0)atomicAdd(A.probe,(unsigned long long)(r1-r0));}}
}

struct MxTailArgs{uint16_t*y;long long y_ld;const int32_t*off;int ne,total,N;unsigned long long*probe;};
template<bool PROBE> __global__ void mx_zero_tail_kernel(MxTailArgs a){
 const int row=(int)blockIdx.x,first=ds41_live_tail::end(a.off,a.ne,a.total);
 if(row<first||row>=a.total)return; // Block-uniform; no weight-bearing argument exists.
 for(int n=(int)threadIdx.x;n<a.N;n+=(int)blockDim.x)a.y[(long long)row*a.y_ld+n]=0;
 if constexpr(PROBE){__syncthreads();if(threadIdx.x==0){atomicAdd(a.probe+2,1ull);atomicAdd(a.probe+3,(unsigned long long)a.N);}}
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

/* gemv_vp's first kernel: MX_FP4 experts over a table whose marked entries are packed weights. */
int mx_launch_gemv_vp(RadStream s, int rpw, const MxGemvArgs& A, unsigned grid, unsigned lds) {
    if (rpw == 2)
        return R4D_EMIT(s, (mx_gemv_kernel<MX_FP4, 2, true, false, true>), dim3(grid), dim3(kMxThreads), lds, A);
    return R4D_EMIT(s, (mx_gemv_kernel<MX_FP4, 1, true, false, true>), dim3(grid), dim3(kMxThreads), lds, A);
}

/* gemv_vp's kernels: kVpOld the first form (mx_gemv_kernel's VP path); the others mx_vp2_kernel in a form, on a grid
 * of up to 256 blocks (mx_moe's):
 *   r2    the first vp2 form (each block's decode unrolled)       r2cf  kVpSraw | kVpRoll
 *   r2cz  kVpSraw | kVpRoll | kVpZero, the default
 * RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED_KERNEL=<name> picks one for every call of the process (read once, said once
 * on stderr); the checker export r4d_rad_moe_gemm_mxfp4_gemv_vp_variant picks one a call. Every one gives the plain
 * row's bits. kVpSraw and kVpRoll exist at one token row a pass (the decode, and gate_up's passes at any height:
 * one row of K 5120 fits the LDS); a pass of more rows takes the first form, with the variant's kVpZero. */
enum : int { kVpDefault = 0, kVpOld = 1, kVpTwo = 2, kVpTwoCF = 3, kVpTwoCZ = 4, kVpN = 5 };
constexpr int kVpDefaultIs = kVpTwoCZ;
const char* const kVpName[kVpN] = { "default", "old", "r2", "r2cf", "r2cz" };
/* Each variant's form (kVpForm[kVpDefault] is unused: the default runs kVpDefaultIs). The kernel spreads the live
 * experts' row groups over whatever grid it gets, so the grid changes the speed, never the bits. */
constexpr int kVpForm[kVpN] = { kVpSraw | kVpRoll | kVpZero, 0, 0, kVpSraw | kVpRoll, kVpSraw | kVpRoll | kVpZero };
const char* const kVpWhat[kVpN] = {
    "its default kernel", "its first kernel (mx_gemv_kernel's packed path)",
    "mx_vp2_kernel, each block's decode unrolled",
    "mx_vp2_kernel, the raw rounds' bytes through the scalar unit and one rolled block loop",
    "mx_vp2_kernel, the same, with a pass of dropped slots alone storing its zeros and nothing else" };
int mx_vp_env() {
    static const int k = [] {
        const char* p = std::getenv("RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED_KERNEL");
        if (!p || !*p) return (int)kVpDefault;
        int v = -1;
        for (int i = 0; i < kVpN; ++i)
            if (!std::strcmp(p, kVpName[i])) v = i;
        if (v < 0) {
            std::fprintf(stderr, "libr4d: RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED_KERNEL=%s is not one of", p);
            for (int i = 1; i < kVpN; ++i) std::fprintf(stderr, " %s", kVpName[i]);
            std::fprintf(stderr, "; gemv_vp keeps its default kernel\n");
        } else {
            std::fprintf(stderr, "libr4d: RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED_KERNEL=%s: gemv_vp runs %s\n", p,
                         kVpWhat[v]);
        }
        return v < 0 ? (int)kVpDefault : v;
    }();
    return k;
}
int mx_vp_pick(int var) {
    const int v = var >= 0 ? var : mx_vp_env();
    return v == kVpDefault ? kVpDefaultIs : v;
}

/* mx_vp2_kernel in form FL at MC >= the pass height (mcx) and UF blocks a lane a pass (uf: 5 when a row pass is
 * more than three rounds, else 3), two weight rows a wave; passes of more than one row take the first form, with
 * FL's kVpZero (FM). */
template <int FL>
int mx_launch_vp2_t(RadStream s, int mcx, int uf, const MxGemvArgs& A, unsigned grid, unsigned lds) {
    constexpr int FM = FL & kVpZero;
    if (mcx <= 1) {
        if (uf >= 5) return R4D_EMIT(s, (mx_vp2_kernel<2, 1, 5, FL>), dim3(grid), dim3(kMxThreads), lds, A);
        return R4D_EMIT(s, (mx_vp2_kernel<2, 1, 3, FL>), dim3(grid), dim3(kMxThreads), lds, A);
    }
    if (mcx <= 4) return R4D_EMIT(s, (mx_vp2_kernel<2, 4, 3, FM>), dim3(grid), dim3(kMxThreads), lds, A);
    return R4D_EMIT(s, (mx_vp2_kernel<2, 8, 3, FM>), dim3(grid), dim3(kMxThreads), lds, A);
}
/* gemv_vp's launch for the kernel `var` (mx_vp_pick's): `A`, `rpw`, `grid` and `lds` as mx_moe sized them for
 * the plain kernel; mx_vp2_kernel takes its own row groups and dynamic LDS (three words a live expert, then the
 * staged rows: `xbuf` floats; its tables and live count are static LDS). */
int mx_launch_vp(RadStream s, int var, int rpw, const MxGemvArgs& A, unsigned grid, unsigned lds, long long L,
                 long long xbuf) {
    if (var == kVpOld) return mx_launch_gemv_vp(s, rpw, A, grid, lds);
    MxGemvArgs B = A;
    B.live_off = kMxVp2Live;
    B.x_off = (int)(kMxVp2Live + mx_r16(12LL * A.n_expert));
    const long long lds2 = B.x_off + xbuf * 4;
    const long long nrg = ((long long)A.N + 15) / 16;
    const long long items = L * nrg;
    const unsigned grid2 = (unsigned)(items < 256 ? items : 256);
    const int mcx = !A.one_pass && A.XLM >= 1 && A.XLM < A.XM ? A.XLM : A.XM;
    const long long ub = A.XLM >= 1 ? A.K / 32 : A.KC / 32;   /* blocks of a row pass (whole, or a chunk) */
    const int uf = ub > 96 ? 5 : 3;
    const unsigned l2 = (unsigned)lds2;
    switch (kVpForm[var]) {
    case kVpSraw | kVpRoll: return mx_launch_vp2_t<kVpSraw | kVpRoll>(s, mcx, uf, B, grid2, l2);
    case kVpSraw | kVpRoll | kVpZero: return mx_launch_vp2_t<kVpSraw | kVpRoll | kVpZero>(s, mcx, uf, B, grid2, l2);
    default: return mx_launch_vp2_t<0>(s, mcx, uf, B, grid2, l2);
    }
}

int mx_launch_tiled(RadStream s, int ty, int bm, bool moe, const MxTiledArgs& A, dim3 grid, unsigned lds) {
    return ty == MX_FP4 ? mx_launch_tiled_t<MX_FP4>(s, bm, moe, A, grid, lds)
                        : mx_launch_tiled_t<MX_F32>(s, bm, moe, A, grid, lds);
}

bool mx_tail_opt_in(){static const bool on=ds41_live_tail::opt_in(std::getenv("RADIANCE_EXPERIMENTAL_DS41_LIVE_TAIL"));return on;}

/* RADIANCE_EXPERIMENTAL_DS41_DRAFT_GEMV (ds41_live_tail::draft_mode): the DSpark draft bank's
 * grouped GEMMs (32 experts a rank, top-3) take the live tail -- the slots routed to other ranks no
 * longer run against the rank's last expert (mx_range gives it [off[31], T)), one more pass over
 * its weights for every three of them in down and for each in gate_up -- and with "1" one pass over
 * each live expert's weights. The calls are counted over the process (every rank), with a line at
 * the first and the counts at exit, so a server log shows the switch reached the draft's GEMMs:
 * gate_up calls (token order), down calls (sorted order), their query rows, calls on the tiled path
 * (more than 16 rows), and draft-shaped calls left on the old path (rows not one whole routing). */
int mx_draft_mode() {
    static const int m = ds41_live_tail::draft_mode(std::getenv("RADIANCE_EXPERIMENTAL_DS41_DRAFT_GEMV"));
    return m;
}
const char* mx_draft_what(int m) {
    return (m & ds41_live_tail::kOnePass) ? "the live tail and one pass over each expert's weights" : "the live tail";
}
struct MxDraftCounts {
    enum { kGu, kDn, kRows, kTiled, kKept, kN };
    std::atomic<unsigned long long> v[kN]{};
    std::atomic<bool> said{false};
    void note(int m) {
        if (said.exchange(true, std::memory_order_relaxed)) return;
        std::fprintf(stderr, "libr4d: RADIANCE_EXPERIMENTAL_DS41_DRAFT_GEMV: the DSpark draft bank's moe_gemm_ggml "
                             "(32 experts a rank, top-3) take %s\n", mx_draft_what(m));
    }
    ~MxDraftCounts() {
        if (!said.load(std::memory_order_relaxed)) return;
        std::fprintf(stderr, "libr4d: RADIANCE_EXPERIMENTAL_DS41_DRAFT_GEMV (%s): %llu draft moe_gemm_ggml calls "
                             "(%llu gate_up, %llu down; %llu query rows; %llu tiled), %llu draft-shaped calls left "
                             "on the old path\n", mx_draft_what(mx_draft_mode()), v[kGu].load() + v[kDn].load(),
                     v[kGu].load(), v[kDn].load(), v[kRows].load(), v[kTiled].load(), v[kKept].load());
    }
};
MxDraftCounts& mx_draft_counts() { static MxDraftCounts c; return c; }
int mx_tail_zero(RadStream s,uint16_t*y,long long ld,const int32_t*off,int ne,int total,int N,unsigned long long*p){
 MxTailArgs a{y,ld,off,ne,total,N,p};
 return p?R4D_EMIT(s,(mx_zero_tail_kernel<true>),dim3((unsigned)total),dim3(kMxThreads),0,a):R4D_EMIT(s,(mx_zero_tail_kernel<false>),dim3((unsigned)total),dim3(kMxThreads),0,a);
}
int mx_tail_gemv(RadStream s,int rpw,const MxGemvArgs&a,unsigned grid,unsigned lds){
 if(!a.probe)return mx_launch_gemv(s,MX_FP4,rpw,true,a,grid,lds);
 if(rpw==2)return R4D_EMIT(s,(mx_gemv_kernel<MX_FP4,2,true,true>),dim3(grid),dim3(kMxThreads),lds,a);
 return R4D_EMIT(s,(mx_gemv_kernel<MX_FP4,1,true,true>),dim3(grid),dim3(kMxThreads),lds,a);
}
int mx_tail_tiled(RadStream s,int bm,const MxTiledArgs&a,dim3 grid,unsigned lds){
 if(!a.probe)return mx_launch_tiled(s,MX_FP4,bm,true,a,grid,lds);
 if(bm==32)return R4D_EMIT(s,(mx_tiled_kernel<MX_FP4,32,true,true>),grid,dim3(kMxThreads),lds,a);
 return R4D_EMIT(s,(mx_tiled_kernel<MX_FP4,128,true,true>),grid,dim3(kMxThreads),lds,a);
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

/* moe_gemm_ggml type=mxfp4: y[i] = a[row(i)] . w[e(i)]^T over the expert-sorted rows. `baseline`
 * keeps every switch off; `force` (0..3, ds41_live_tail::kTail | kOnePass) is a checker's mode at
 * any shape, in place of the switches; `probe` counts the work (r4d_rad_moe_gemm_mxfp4_draft_probe).
 * `vp`: the gemv_vp row, whose table entries may name packed weights (bit 0); it serves the decode
 * band only. Its kernel has no one-pass form, so on a table it serves the draft bank's
 * ..._DRAFT_GEMV mode does not apply and the call is counted as left on the old path. `tall`: the
 * gemv at any number of tokens (moe_gemm_ggml_mxfp4_gemv_tall, RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL),
 * not only up to 16. */
int mx_moe(const RadArgs* a, RadStream s, bool gemv,bool baseline=false,unsigned long long* probe=nullptr,int force=-1,
           bool vp=false, bool tall=false, int vpvar=-1) {
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
    const bool vpk = vp && ty == MX_FP4 && table;   /* the gemv_vp kernel serves this call */
    /* The live tail (kTail) and the one weight pass (kOnePass): the trunk's shape takes the tail under
     * ..._LIVE_TAIL, the draft bank's its mode under ..._DRAFT_GEMV, a checker's call its `force`. The
     * tail needs the rows to be one whole routing, which a forced call is refused without. */
    int mode = 0;
    const bool whole = ds41_live_tail::whole(tokens, top_k, T, AR, a_sorted);
    if (force >= 0) {
        if (force > 3 || (force && ty != MX_FP4) || ((force & ds41_live_tail::kTail) && !whole))
            return R4D_NO(RAD_E_UNSUPPORTED);
        mode = force;
    } else if (!baseline && ty == MX_FP4) {
        if (ds41_live_tail::geometry(tokens, N, K, ne, top_k, T, AR, a_sorted) && mx_tail_opt_in()) {
            mode = ds41_live_tail::kTail;
        } else if (const int dm = ds41_live_tail::draft_eligible(tokens, N, K, ne, top_k) ? mx_draft_mode() : 0) {
            MxDraftCounts& c = mx_draft_counts();
            c.note(dm);
            if (whole && !vpk) {
                mode = dm;
                c.v[a_sorted ? MxDraftCounts::kDn : MxDraftCounts::kGu].fetch_add(1, std::memory_order_relaxed);
                c.v[MxDraftCounts::kRows].fetch_add((unsigned long long)tokens, std::memory_order_relaxed);
                if (!(gemv && tokens <= 16)) c.v[MxDraftCounts::kTiled].fetch_add(1, std::memory_order_relaxed);
            } else {
                c.v[MxDraftCounts::kKept].fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    const bool tail = (mode & ds41_live_tail::kTail) != 0;
    if (probe && (ty != MX_FP4 || (!tail && force < 0))) return R4D_NO(RAD_E_UNSUPPORTED); // Refuse before any enqueue.
    const long long L = ne < T ? ne : T;       /* experts that can have rows */
    /* Above 16 tokens the gemv is still exact: each output is its weight row dotted with its activation row in the
     * same lane and block order at any height (mx_row; an expert's rows go in whole passes of XLM), so a tall pass
     * gives a 16-token pass's bits. The tiled path below sums in another order. */
    if (gemv && (tokens <= 16 || tall)) {
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
        A.live_end=tail?1:0;A.probe=probe;
        A.one_pass = (mode & ds41_live_tail::kOnePass) ? 1 : 0;
        A.probe_passes = probe && force >= 0 ? 1 : 0;
        A.x_al = mx_al16(A_->data, x_ld) ? 1 : 0;
        A.live_off = (int)tb;
        A.x_off = (int)(tb + mx_r16(ne * 4));
        const long long lds = A.x_off + xbuf * 4;
        /* Enough blocks for every row group of L experts: the kernel spreads whichever experts
         * turn out live over all of them. */
        const long long items = L * nrg;
        const unsigned grid = (unsigned)(items < 256 ? items : 256);
        /* gemv_vp over a table of MX_FP4 experts takes the kernel that tells the forms apart (vpk, above); the
         * float logical form and stacked experts carry no marks, so the plain kernel serves them for it too. */
        if (vpk && probe) return R4D_NO(RAD_E_UNSUPPORTED);   /* the probes are the plain row's */
        if(!tail&&!probe)return vpk ? mx_launch_vp(s, mx_vp_pick(vpvar), rpw, A, grid, (unsigned)lds, L, xbuf) : mx_launch_gemv(s,ty,rpw,true,A,grid,(unsigned)lds);
        const int rc=vpk ? mx_launch_vp(s, mx_vp_pick(vpvar), rpw, A, grid, (unsigned)lds, L, xbuf) : mx_tail_gemv(s,rpw,A,grid,(unsigned)lds);if(rc!=RAD_OK||!tail)return rc;
        // No fallback or replay after a submitted prefix; existing stream owner retires both.
        return mx_tail_zero(s,A.y,A.y_ld,A.off,A.n_expert,A.M,A.N,probe);
    }
    /* The tiled kernel reads every entry as block_mxfp4: a marked table is the decode band's alone. */
    if (vp) return R4D_NO(RAD_E_UNSUPPORTED);
    MxTiledArgs A{};
    A.x = (const uint16_t*)A_->data; A.x_ld = x_ld;
    A.w = table ? nullptr : (const uint8_t*)W->data;
    A.wtab = table ? (const uint8_t* const*)W->data : nullptr;
    A.w_ld = w_ld; A.w_es = w_es;
    A.y = (uint16_t*)Y->data; A.y_ld = y_ld;
    A.sorted = (const int32_t*)so->data; A.off = (const int32_t*)of->data;
    A.M = (int)T; A.N = (int)N; A.K = (int)K;
    A.n_expert = (int)ne; A.top_k = (int)top_k; A.a_sorted = a_sorted ? 1 : 0; A.AR = (int)AR;
    A.live_end=tail?1:0;A.probe=probe;
    A.x_al = mx_al16(A_->data, x_ld) ? 1 : 0;
    A.tab_bytes = (int)tb;
    /* 32-row tiles while the experts' runs average under 64 rows (prefill over many experts). */
    const int bm = T < 64 * L ? 32 : 128;
    const long long gx = (T + bm - 1) / bm + L, gy = (N + kTN - 1) / kTN;
    if (gy > 65535 || gx > 0x7fffffffLL) return R4D_NO(RAD_E_SHAPE);
    if(!tail&&!probe)return mx_launch_tiled(s,ty,bm,true,A,dim3((unsigned)gx,(unsigned)gy),(unsigned)(tb+32*(kTP+bm+4)*4));
    const int rc=mx_tail_tiled(s,bm,A,dim3((unsigned)gx,(unsigned)gy),(unsigned)(tb+32*(kTP+bm+4)*4));if(rc!=RAD_OK||!tail)return rc;
    return mx_tail_zero(s,A.y,A.y_ld,A.off,A.n_expert,A.M,A.N,probe);
}

}  // namespace

extern "C" int r4d_rad_gemm_nt_mxfp4_gemv(const RadArgs* a, RadStream s) { return mx_gemm(a, s, true); }
extern "C" int r4d_rad_gemm_nt_mxfp4_tiled(const RadArgs* a, RadStream s) { return mx_gemm(a, s, false); }
extern "C" int r4d_rad_moe_gemm_mxfp4_gemv(const RadArgs* a, RadStream s) { return mx_moe(a, s, true); }
extern "C" int r4d_rad_moe_gemm_mxfp4_tiled(const RadArgs* a, RadStream s) { return mx_moe(a, s, false); }
/* gemv_vp: the decode band over a table whose marked entries are packed weights (the top of this file). */
extern "C" int r4d_rad_moe_gemm_mxfp4_gemv_vp(const RadArgs* a, RadStream s) {
    return mx_moe(a, s, true, false, nullptr, -1, true);
}
/* Checker-only: gemv_vp with its kernel `variant` (0 the default, then kVpName's: 1 the first form, 2.. the vp2
 * kernels, mx_launch_vp's table) for this call, whatever ..._VRAM_PACKED_KERNEL says, so a check holds every one
 * to the plain row and times them side by side. */
extern "C" int r4d_rad_moe_gemm_mxfp4_gemv_vp_variant(const RadArgs* a, RadStream s, int variant) {
    if (variant < 0 || variant >= kVpN) return RAD_E_INVAL;
    return mx_moe(a, s, true, false, nullptr, -1, true, false, variant);
}
/* Checker-only: how many kernel variants the export above takes, their names in `names` (up to `cap`; [0] is the
 * default, which is kVpDefaultIs's kernel) and in `deflt` the variant the default runs. */
extern "C" int r4d_rad_moe_gemm_mxfp4_gemv_vp_variants(const char** names, int cap, int* deflt) {
    for (int i = 0; i < kVpN && i < cap; ++i) names[i] = kVpName[i];
    if (deflt) *deflt = kVpDefaultIs;
    return kVpN;
}
/* The row moe_gemm_ggml_mxfp4_gemv_tall (r4d_rows.cpp, exported only under RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL=1):
 * the decode band's kernel at every pass height. */
extern "C" int r4d_rad_moe_gemm_mxfp4_gemv_tall(const RadArgs* a, RadStream s) {
    return mx_moe(a, s, true, false, nullptr, -1, false, true);
}

// Checker-only exports do not alter registered rows/public ABI.
extern "C" int r4d_rad_moe_gemm_mxfp4_live_tail_baseline(const RadArgs*a,RadStream s,int gemv){return mx_moe(a,s,gemv!=0,true);}
extern "C" int r4d_rad_moe_gemm_mxfp4_live_tail_probe(const RadArgs*a,RadStream s,int gemv,unsigned long long*p){
 if(!p||((uintptr_t)p&7u))return RAD_E_INVAL;return mx_moe(a,s,gemv!=0,false,p);
}
/* Checker-only: the draft bank's modes forced (`mode`: ds41_live_tail::kTail | kOnePass, 0 the old
 * path) at any shape, bypassing ..._DRAFT_GEMV and the shape list; the tail still needs one whole
 * routing. With `p`, six counts added up: [0] the rows the live experts own, [1] blocks that
 * worked, [2] rows the zero-tail kernel wrote, [3] values it wrote, [4] idle blocks, [5] passes over
 * an expert's weights (gemv only, each live expert counted once). */
extern "C" int r4d_rad_moe_gemm_mxfp4_draft_probe(const RadArgs*a,RadStream s,int gemv,int mode,unsigned long long*p){
 if(mode<0||mode>3||(p&&((uintptr_t)p&7u)))return RAD_E_INVAL;return mx_moe(a,s,gemv!=0,false,p,mode);
}
/* Checker-only: ..._DRAFT_GEMV's counts so far (gate_up calls, down calls, query rows, tiled calls,
 * draft-shaped calls left on the old path), so a test sees which path a served call took. */
extern "C" void r4d_ds41_draft_gemv_counts(unsigned long long* out){
 MxDraftCounts& c=mx_draft_counts();for(int i=0;i<MxDraftCounts::kN;++i)out[i]=c.v[i].load();
}
