// r4d_gemm_bf16f8_nt_m64.hip - r4d_gemm_bf16f8_nt_m64: gemm_bf16_nt_m64's GEMM over a bf16 weight
// stored as E4M3 codes and an E8M0 exponent per 32x32 block (r4d_bf16f8_layout.h), gfx1201.
//
// C[M,N] = A[M,K] @ W[N,K]^T, bf16 in and out, fp32 accumulate: gemm_nt's bytes. The weight is a
// bf16 weight every 32x32 block of which is E4M3 codes times one power of two (an FP8-sourced
// checkpoint's dense linears, cast to bf16), which the relayout re-encodes and checks element by
// element at load. A k step expands a lane's eight codes in registers to the eight bf16 bits
// fragU would have loaded from the plane, so WMMA sees the same operands and only the weight bytes
// read per k step halve.
//
// Byte-identical to gemm_bf16_nt_m64 by construction:
//   - the same wave geometry, K slicing, activation fragments, row clamp, LDS split-K reduction
//     and RNE store: the source is r4d_gemm_m64_shared.h, which gemm_bf16_nt_m64 expands too;
//   - the same `mma(fragU(A), B, acc)` per accumulator, one chain from zero per (tile, split, row
//     tile) in ascending k steps (the unroll KU changes how a slice is walked, never the order),
//     every accumulator in registers as that kernel has it (see the peel);
//   - the same B bits (r4d_bf16f8_frag.h): v_cvt_pk_f32_fp8 decodes every non-NaN E4M3 byte
//     exactly to f32 (a normal number or +-0), the product with 2^(s-127) is exact and, by the
//     stored exponents' bounds, a normal number or +-0 whose low 16 bits are zero, so its high
//     half is the plane's bf16;
//   - the same split: SK from gemm_bf16_nt_m64's own cfg_for call, so pick_sk with its arguments
//     when untuned. The row is NOTUNE; MB and WV cannot move bits under BR_FLAT.
//
// A wave reads 512 contiguous bytes a 32-K block (16 bytes a lane: both k steps' slots), and one
// wave-uniform exponent byte a block: a 16-row x 16-K step never crosses a 32x32 block.
#include "r4d.h"
#include "r4d_gemm_args.h"
#include "r4d_gdn_wmma.h"
#include "r4d_chain.h"
#include "r4d_gemm_m64_shared.h"
#include "r4d_bf16f8_layout.h"
#include "r4d_bf16f8_frag.h"
#include <hip/hip_bf16.h>
#include <cstdint>
#include <stdexcept>

#define R4D_GEMM_BF16F8_MAX_M 65536
#define WAVE32 32

extern "C" int r4d_rad_gemm_bf16_m64(const RadArgs*, RadStream);

namespace {

typedef r4d_bf8_u4 r4d_u4v;
// Global pointers, as r4d_gemm_m64_shared.h's r4d_gu2: held across iterations a plain pointer
// becomes generic and its loads flat (which on gfx12 also count in dscnt).
typedef __attribute__((address_space(1))) const r4d_u4v* r4d_gq4;
typedef __attribute__((address_space(1))) const unsigned* r4d_gu1;
typedef __attribute__((address_space(1))) const unsigned char* r4d_gb1;

// The weight fragments: r4d_bf16f8_frag.h, shared with the card check that dumps them.
__device__ __forceinline__ v8bf f8frag(unsigned d0, unsigned d1, float f) { return r4d_bf16f8_frag(d0, d1, f); }
__device__ __forceinline__ float e8m0(unsigned s) { return r4d_bf16f8_pow2(s); }

}  // namespace

// KU k steps a round: KU/2 b128 weight loads a lane (16 at MT = 1: 4 KiB a wave in flight, what
// gemm_bf16_nt_m64 keeps at KU 8; 8 at MT 2..4, for the accumulators' registers). SALIGN: every
// slice starts on a 128-K boundary ((K/SK) % 128 == 0), so a round's KU/2 exponent bytes are whole
// dwords at a dword-aligned address and no slice starts mid-block. Otherwise a slice may start on a
// block's second k step (the peel) and the bytes are read one by one.
template <int MT, bool CHAIN, int KU, bool SALIGN>
__device__ __forceinline__ void r4d_gemm_bf16f8_nt_m64_body(
    const unsigned short* __restrict__ A, const unsigned char* __restrict__ Wb,
    __hip_bfloat16* __restrict__ C, int M, int K, int N, int WV, int SK, int LDA, int LDC,
    long long soff, int ldws) {
  static_assert(KU % 8 == 0 && KU <= 16, "a round is whole dwords of exponents, at most two");
  if (CHAIN) r4d_chain_wait();
  R4D_M64_WAVE_SETUP();   // r4d_gemm_m64_shared.h: tid, lane, n0, mt0, Ks, ksteps, kbase, acc, red
  R4D_M64_A_OFFSETS();    // aoff[MT]
  if (n0 < N) {
    r4d_gu2 pa[MT];
#pragma unroll
    for (int i = 0; i < MT; i++) pa[i] = (r4d_gu2)(const unsigned long long*)(A + aoff[i]);
    // The wave's packets and exponents: tile nt's row of packets from block kbase/32, and the
    // exponent row of its 32-row block. Both wave-uniform, so the loads take a scalar base.
    const int ntu = __builtin_amdgcn_readfirstlane(nt);
    const int kb0 = __builtin_amdgcn_readfirstlane(kbase) >> 5;
    r4d_gq4 pw = (r4d_gq4)(const r4d_u4v*)Wb + ((size_t)ntu * (size_t)(K >> 5) + (size_t)kb0) * WAVE32 + lane;
    r4d_gb1 ps = (r4d_gb1)(const unsigned char*)Wb + soff + (size_t)(ntu >> 1) * (size_t)ldws + kb0;
    int s = 0;
    if (!SALIGN && (kbase & 16)) {          // the slice starts on its first block's second step
      // Here the compiler knows the accumulators are zero and would give this WMMA an inline
      // constant 0 as C, an encoding gemm_bf16_nt_m64 never issues (its first WMMA is inside its
      // loop). An empty asm per value keeps them in registers, as that kernel has them.
#if defined(__HIP_DEVICE_COMPILE__)
#pragma unroll
      for (int i = 0; i < MT; i++) {
        float a0 = acc[i][0], a1 = acc[i][1], a2 = acc[i][2], a3 = acc[i][3];
        float a4 = acc[i][4], a5 = acc[i][5], a6 = acc[i][6], a7 = acc[i][7];
        asm volatile("" : "+v"(a0), "+v"(a1), "+v"(a2), "+v"(a3), "+v"(a4), "+v"(a5), "+v"(a6), "+v"(a7));
        acc[i][0] = a0; acc[i][1] = a1; acc[i][2] = a2; acc[i][3] = a3;
        acc[i][4] = a4; acc[i][5] = a5; acc[i][6] = a6; acc[i][7] = a7;
      }
#endif
      const r4d_u4v q = pw[0];
      const float f = e8m0((unsigned)__builtin_amdgcn_readfirstlane((unsigned)ps[0]));
      const v8bf b = f8frag(q.z, q.w, f);
#pragma unroll
      for (int i = 0; i < MT; i++) {
        acc[i] = mma(fragU(pa[i], 0), b, acc[i]);
        pa[i] += 4;
      }
      pw += WAVE32;
      ps += 1;
      s = 1;
    }
    // Unrolled so the packet offsets are load immediates and the bases advance once a round.
    for (; s + KU <= ksteps; s += KU) {
      r4d_u4v q[KU / 2];
#pragma unroll
      for (int j = 0; j < KU / 2; j++) q[j] = pw[j * WAVE32];
      unsigned sw[KU / 8];
#pragma unroll
      for (int j = 0; j < KU / 8; j++) {
        unsigned v;
        if (SALIGN) v = ((r4d_gu1)(const unsigned*)ps)[j];
        else v = (unsigned)ps[4 * j] | ((unsigned)ps[4 * j + 1] << 8) | ((unsigned)ps[4 * j + 2] << 16) |
                 ((unsigned)ps[4 * j + 3] << 24);
        sw[j] = (unsigned)__builtin_amdgcn_readfirstlane(v);
      }
#pragma unroll
      for (int u = 0; u < KU; u++) {
        const float f = e8m0((sw[u >> 3] >> (8 * ((u >> 1) & 3))) & 0xFFu);
        const r4d_u4v w = q[u >> 1];
        const v8bf b = (u & 1) ? f8frag(w.z, w.w, f) : f8frag(w.x, w.y, f);
#pragma unroll
        for (int i = 0; i < MT; i++) acc[i] = mma(fragU(pa[i], u), b, acc[i]);
      }
      pw += (KU / 2) * WAVE32;
      ps += KU / 2;
#pragma unroll
      for (int i = 0; i < MT; i++) pa[i] += KU * 4;
    }
    for (; s < ksteps; s += 2) {            // whole blocks, then the slice's last half block
      const r4d_u4v q = pw[0];
      const float f = e8m0((unsigned)__builtin_amdgcn_readfirstlane((unsigned)ps[0]));
      const v8bf b0 = f8frag(q.x, q.y, f);
#pragma unroll
      for (int i = 0; i < MT; i++) acc[i] = mma(fragU(pa[i], 0), b0, acc[i]);
      if (s + 1 < ksteps) {
        const v8bf b1 = f8frag(q.z, q.w, f);
#pragma unroll
        for (int i = 0; i < MT; i++) acc[i] = mma(fragU(pa[i], 1), b1, acc[i]);
      }
#pragma unroll
      for (int i = 0; i < MT; i++) pa[i] += 8;
      pw += WAVE32;
      ps += 1;
    }
  }
  R4D_M64_REDUCE_STORE();
  if (CHAIN) r4d_chain_done();
}

template <int MT, int KU, bool SALIGN>
__global__ __launch_bounds__(1024) void r4d_gemm_bf16f8_nt_m64_kernel(
    const unsigned short* __restrict__ A, const unsigned char* __restrict__ Wb,
    __hip_bfloat16* __restrict__ C, int M, int K, int N, int WV, int SK, int LDA, int LDC,
    long long soff, int ldws) {
  r4d_gemm_bf16f8_nt_m64_body<MT, false, KU, SALIGN>(A, Wb, C, M, K, N, WV, SK, LDA, LDC, soff, ldws);
}
template <int MT, int KU, bool SALIGN>
__global__ __launch_bounds__(1024) void r4dchain_gemm_bf16f8_nt_m64(
    const unsigned short* __restrict__ A, const unsigned char* __restrict__ Wb,
    __hip_bfloat16* __restrict__ C, int M, int K, int N, int WV, int SK, int LDA, int LDC,
    long long soff, int ldws) {
  r4d_gemm_bf16f8_nt_m64_body<MT, true, KU, SALIGN>(A, Wb, C, M, K, N, WV, SK, LDA, LDC, soff, ldws);
}

// gemm_bf16_nt_m64's launch (r4d_gemm_bf16_nt_m64.hip), with the stored form's exponent offset
// and row pitch: grid = ceil(ceil(N/16)/WV) x ceil(ceil(M/16)/MB) blocks, block = WV*SK waves,
// WV*SK*256 floats of LDS, MT = MB. KU and SALIGN pick the instantiation; neither moves a bit.
// The exponents' offset and pitch come from the weight's own [wn, wk] (the declared per-rank
// shape), which must be the GEMM's [N, K]: the bf16 kernel strides its rows by K and would read a
// narrower output's rows right, this one would read another row's exponents.
static void gemm_bf16f8_m64_launch(long a, long w, long c, int M, int K, int N, int WV, int SK,
                                   int MB, long stream, bool chain, long long lda, long long ldc,
                                   long long wn, long long wk) {
  if (M < 1 || M > R4D_GEMM_BF16F8_MAX_M) throw std::runtime_error("r4d gemm_bf16f8_nt_m64: M out of range");
  if (lda < K || ldc < N || lda % 4 || (long long)(M - 1) * lda + K >= (1ll << 32) || ldc > INT32_MAX)
    throw std::runtime_error("r4d gemm_bf16f8_nt_m64: row pitch unsupported");
  if (N % 16 || K % 32) throw std::runtime_error("r4d gemm_bf16f8_nt_m64: N must be a multiple of 16, K of 32");
  if (wn != N || wk != K) throw std::runtime_error("r4d gemm_bf16f8_nt_m64: the weight is not [N, K]");
  if (SK < 1 || K % (SK * 16)) throw std::runtime_error("r4d gemm_bf16f8_nt_m64: K must be divisible by SK*16");
  if (WV < 1 || WV * SK * WAVE32 > 1024) throw std::runtime_error("r4d gemm_bf16f8_nt_m64: threads/block > 1024");
  const int ntiles = (N + 15) / 16, mtiles = (M + 15) / 16;
  if (MB < 1 || MB > 4) throw std::runtime_error("r4d gemm_bf16f8_nt_m64: MB must be 1..4");
  dim3 grid((ntiles + WV - 1) / WV, (mtiles + MB - 1) / MB), block(WV * SK * WAVE32);
  size_t smem = (size_t)WV * SK * 256 * sizeof(float);
  const long long soff = r4d_bf16f8_scale_off(wn, wk);
  const int ldws = (int)r4d_bf16f8_ldws(wk);
  const bool salign = ((K / SK) % 128) == 0;
  auto Ap = (const unsigned short*)a;
  auto Wp = (const unsigned char*)w;
  auto Cp = (__hip_bfloat16*)c;
  hipStream_t st = (hipStream_t)stream;
#define LAUNCH(MT, KU, SA) hipLaunchKernelGGL((chain ? r4dchain_gemm_bf16f8_nt_m64<MT, KU, SA>       \
                                                     : r4d_gemm_bf16f8_nt_m64_kernel<MT, KU, SA>),   \
                                              grid, block, smem, st, Ap, Wp, Cp, M, K, N, WV, SK,      \
                                              (int)lda, (int)ldc, soff, ldws)
#define ARM(MT, KU) do { if (salign) LAUNCH(MT, KU, true); else LAUNCH(MT, KU, false); } while (0)
  switch (MB) {
    case 1: ARM(1, 16); break;
    case 2: ARM(2, 8); break;
    case 3: ARM(3, 8); break;
    case 4: ARM(4, 8); break;
    default: throw std::runtime_error("r4d gemm_bf16f8_nt_m64: MB unsupported");
  }
#undef ARM
#undef LAUNCH
}

// ============================================================== the plugin ABI entry point
//
// gemm_bf16_nt_m64's own checks and its own split call, so SK is the bf16 row's for every shape.
extern "C" int r4d_rad_gemm_bf16f8_m64(const RadArgs* a, RadStream s) {
    GemmOps g;
    int rc = gemm_ops(a, &g, 0, 0, /*pitched=*/1);
    if (rc != RAD_OK) return rc;
    /* A tool holding the weight as its plane (no relayout): gemm_bf16_nt_m64 itself. */
    if (g.b->dtype == RAD_BF16) return r4d_rad_gemm_bf16_m64(a, s);
    if (g.b->dtype != R4D_DT_BF16F8) return R4D_NO(RAD_E_DTYPE);
    if (g.M > R4D_GEMM_BF16F8_MAX_M) return R4D_NO(RAD_E_SHAPE);
    if (g.N % 16 || g.K % 32) return R4D_NO(RAD_E_SHAPE);
    /* The stored form's exponents sit at an offset and pitch that follow from the weight's [N, K]
     * (the core passes the declared per-rank shape): an issue whose output or input is not the
     * weight's width would read them at the wrong place, so it is refused. */
    if (g.b->rank != 2 || g.b->shape[0] != g.N || g.b->shape[1] != g.K) return R4D_NO(RAD_E_SHAPE);
    if (g.lda % 4 || (g.M - 1) * g.lda + g.K >= (1ll << 32) || g.ldc > INT32_MAX) return R4D_NO(RAD_E_STRIDE);
    Cfg c;
    rc = cfg_for(a, g.K, g.M, 16, BR_FLAT, &c, (g.N + 15) / 16);   /* gemm_bf16_nt_m64's call */
    if (rc != RAD_OK) return rc;
    R4D_TRY("gemm_bf16f8_nt_m64",
        gemm_bf16f8_m64_launch(r4d_addr(g.a), r4d_addr(g.b), r4d_addr(g.y), (int)g.M, (int)g.K,
                               (int)g.N, c.wv, c.sk, c.mb, (long)(uintptr_t)s,
                               r4d_geti_or(a, "chain", 0) != 0, g.lda, g.ldc,
                               (long long)g.b->shape[0], (long long)g.b->shape[1]));
    return RAD_OK;
}
