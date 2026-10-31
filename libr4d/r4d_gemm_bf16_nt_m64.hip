// r4d_gemm_bf16_nt_m64.hip - r4d_gemm_bf16_nt_m64: skinny bf16 GEMM at any M, gfx1201.
//
// C[M,N] = A[M,K] @ W[N,K]^T, bf16 in and out, fp32 accumulate.
//
// General in M: MT row tiles per block, bounded grids with rebased row launches. The 64 is
// the registry band, not a kernel limit; the M <= 16 row forwards here too.
//
// A scalar kernel (one wave per output column, M accumulators) costs one weight load and M
// activation loads per 16 K, and cannot run above M=16. One wmma_f32_16x16x16 produces a whole
// 16x16 tile, so a 16-K step costs ceil(M/16) activation fragments and one weight fragment, with
// 8 VGPRs per lane of accumulator per tile.
//
// A block is WV column tiles x SK k-splits waves; each wave owns one 16-wide N tile over one
// contiguous K slice. Splits are reduced through LDS in wave order, so the result does not depend
// on scheduling. SK fills the machine when N is small (N=48 is three tiles).
//
// Fragment rows are clamped, not masked: rows past M or N re-read the last valid row and their
// results are never stored.
#include "r4d.h"
#include "r4d_gemm_args.h"
#include "r4d_bf16_gemm_rows.h"
#include "r4d_gdn_wmma.h"
#include "r4d_chain.h"
#include "r4d_gemm_m64_shared.h"
#include <hip/hip_bf16.h>
#include <cstdint>
#include <stdexcept>

#define WAVE32 32

// fragU, the fragment from a base pointer plus a compile-time k step, is in
// r4d_gemm_m64_shared.h.

// MT = row tiles per block, held as MT v8f accumulators (8 VGPRs each); grid.y covers the rest.
// More amortises each weight fragment, fewer puts more blocks on the machine, so MT is tunable.
// KU is the k-step unroll; see the loop.
constexpr int KU = 8;
// CHAIN brackets the body with the chained launch's wait and add (r4d_chain.h); no block returns
// early. r4d_gemm_bf16_nt_m64_kernel and r4dchain_gemm_bf16_nt_m64 below wrap it.
template <int MT, bool CHAIN>
__device__ __forceinline__ void r4d_gemm_bf16_nt_m64_body(
    const unsigned short* __restrict__ A, const unsigned short* __restrict__ W,
    __hip_bfloat16* __restrict__ C, int M, int K, int N, int WV, int SK, int LDA, int LDC) {
  if (CHAIN) r4d_chain_wait();
  R4D_M64_WAVE_SETUP();   // r4d_gemm_m64_shared.h: tid, lane, n0, mt0, Ks, ksteps, kbase, acc, red

  // Row offsets as 32-bit element offsets, turned into pointers once below; as pointers, LLVM
  // would not keep the 64-bit stride math and the row clamp out of the loop.
  unsigned woff = rowGo((unsigned)K, n0, N - 1, lane) + (unsigned)kbase;
  R4D_M64_A_OFFSETS();    // aoff[MT]

  // Unrolled so the k step is a load immediate (the 13-bit field of global_load_b64) and the base
  // pointers advance once per KU steps; rolled, LLVM rebuilds a 64-bit address every step.
  if (n0 < N) {
    r4d_gu2 pw = (r4d_gu2)(const unsigned long long*)(W + woff);
    r4d_gu2 pa[MT];
#pragma unroll
    for (int i = 0; i < MT; i++) pa[i] = (r4d_gu2)(const unsigned long long*)(A + aoff[i]);
    int s = 0;
    for (; s + KU <= ksteps; s += KU) {
#pragma unroll
      for (int u = 0; u < KU; u++) {
        const v8bf b = fragU(pw, u);
#pragma unroll
        for (int i = 0; i < MT; i++) acc[i] = mma(fragU(pa[i], u), b, acc[i]);
      }
      pw += KU * 4;
#pragma unroll
      for (int i = 0; i < MT; i++) pa[i] += KU * 4;
    }
    for (; s < ksteps; s++) {
      const v8bf b = fragU(pw, 0);
      pw += 4;
#pragma unroll
      for (int i = 0; i < MT; i++) {
        acc[i] = mma(fragU(pa[i], 0), b, acc[i]);
        pa[i] += 4;
      }
    }
  }

  // Reduce one row tile at a time through LDS, splits summed in split order from 0.f, and store
  // with the language's float-to-bf16 conversion (r4d_gemm_m64_shared.h).
  R4D_M64_REDUCE_STORE();
  if (CHAIN) r4d_chain_done();
}

template <int MT>
__global__ __launch_bounds__(1024) void r4d_gemm_bf16_nt_m64_kernel(
    const unsigned short* __restrict__ A, const unsigned short* __restrict__ W,
    __hip_bfloat16* __restrict__ C, int M, int K, int N, int WV, int SK, int LDA, int LDC) {
  r4d_gemm_bf16_nt_m64_body<MT, false>(A, W, C, M, K, N, WV, SK, LDA, LDC);
}
template <int MT>
__global__ __launch_bounds__(1024) void r4dchain_gemm_bf16_nt_m64(
    const unsigned short* __restrict__ A, const unsigned short* __restrict__ W,
    __hip_bfloat16* __restrict__ C, int M, int K, int N, int WV, int SK, int LDA, int LDC) {
  r4d_gemm_bf16_nt_m64_body<MT, true>(A, W, C, M, K, N, WV, SK, LDA, LDC);
}

// grid = ceil(ceil(N/16)/WV) x ceil(ceil(M/16)/MB) blocks; block = WV*SK waves. MT = MB is
// compile-time so the accumulators are sized exactly.
static void gemm_bf16_m64_launch(long a, long w, long c, int M, int K, int N, int WV, int SK,
                                 int MB, long stream, bool chain, long long lda, long long ldc);
void r4d_gemm_bf16_nt_m64(long a, long w, long c, int M, int K, int N, int WV, int SK, int MB,
                          long stream) {
  gemm_bf16_m64_launch(a, w, c, M, K, N, WV, SK, MB, stream, false, K, N);
}
/* `chain`: the chained form (r4d_rad_gemm_bf16_m64 passes the row's parameter; M <= 64 always). */
/* `lda` / `ldc`: the rows' pitches in elements (K and N for contiguous planes). A's fragments are
 * 8-byte loads, so its pitch is a multiple of 4; the offsets are 32-bit, so its last row is
 * addressable in 2^32 elements. */
static void gemm_bf16_m64_launch(long a, long w, long c, int M, int K, int N, int WV, int SK,
                                 int MB, long stream, bool chain, long long lda, long long ldc) {
  if (M < 1) throw std::runtime_error("r4d gemm_bf16_nt_m64: M out of range");
  if (lda < K || ldc < N || lda % 4 || (long long)(M - 1) * lda + K >= (1ll << 32) || ldc > INT32_MAX)
    throw std::runtime_error("r4d gemm_bf16_nt_m64: row pitch unsupported");
  if (WV < 1 || SK < 1 || (int64_t)WV * SK > 1024 / WAVE32)
    throw std::runtime_error("r4d gemm_bf16_nt_m64: invalid threads/block");
  if (K % ((int64_t)SK * 16))
    throw std::runtime_error("r4d gemm_bf16_nt_m64: K must be divisible by SK*16");
  if (MB < 1 || MB > 4) throw std::runtime_error("r4d gemm_bf16_nt_m64: MB must be 1..4");
  const int chunk = r4d_bf16_gemm_chunk_rows(M, N, K, MB * 16, WV * 16);
  if (!chunk) throw std::runtime_error("r4d gemm_bf16_nt_m64: invalid dimensions or weight span");
  // A chained op must publish one completion, not one per rebased launch. Existing chained
  // decode shapes (M <= 64) fit one launch; do not silently change their synchronization law.
  if (chain && chunk < M)
    throw std::runtime_error("r4d gemm_bf16_nt_m64: chained rows need one launch");
  const int ntiles = (N + 15) / 16;
  dim3 block(WV * SK * WAVE32);
  size_t smem = (size_t)WV * SK * 256 * sizeof(float);
  auto Wp = (const unsigned short*)w;
  hipStream_t st = (hipStream_t)stream;
#define LAUNCH(MT)                                                                                 \
  hipLaunchKernelGGL((chain ? r4dchain_gemm_bf16_nt_m64<MT> : r4d_gemm_bf16_nt_m64_kernel<MT>),    \
                     grid, block, smem, st, Ap, Wp, Cp, mm, K, N, WV, SK, (int)lda, (int)ldc)
  for (int64_t m0 = 0; m0 < M; m0 += chunk) {
    const int mm = (int)std::min<int64_t>(chunk, M - m0), mtiles = (mm + 15) / 16;
    dim3 grid((ntiles + WV - 1) / WV, (mtiles + MB - 1) / MB);
    auto Ap = (const unsigned short*)a + m0 * lda;
    auto Cp = (__hip_bfloat16*)c + m0 * ldc;
    switch (MB) {
    case 1:
      LAUNCH(1);
      break;
    case 2:
      LAUNCH(2);
      break;
    case 3:
      LAUNCH(3);
      break;
    case 4:
      LAUNCH(4);
      break;
    default:
      throw std::runtime_error("r4d gemm_bf16_nt_m64: MB unsupported");
    }
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e));
  }
#undef LAUNCH
}

int r4d_gemm_bf16_nt_m64_max_m(void) {
  return INT_MAX;
}

// ============================================================== the plugin ABI entry points
//
// The operand reader and the split law are in r4d_gemm_args.h; see libr4d/r4d_args.h.
extern "C" int r4d_rad_gemm_bf16_m64(const RadArgs* a, RadStream s) {
    GemmOps g;
    int rc = gemm_ops(a, &g, 0, 0, /*pitched=*/1);
    if (rc != RAD_OK) return rc;
    if (!r4d_bf16_gemm_chunk_rows(g.M, g.N, g.K, 16, 16)) return R4D_NO(RAD_E_SHAPE);
    if (g.lda % 4 || (g.M - 1) * g.lda + g.K >= (1ll << 32) || g.ldc > INT32_MAX) return R4D_NO(RAD_E_STRIDE);
    Cfg c;
    rc = cfg_for(a, g.K, g.M, 16, BR_FLAT, &c, (g.N + 15) / 16);  /* "K divisible by SK*16" */
    if (rc != RAD_OK) return rc;
    const bool chain = r4d_geti_or(a, "chain", 0) != 0;
    if (chain && r4d_bf16_gemm_chunk_rows(g.M, g.N, g.K, c.mb * 16, c.wv * 16) < g.M)
      return R4D_NO(RAD_E_UNSUPPORTED);
    R4D_TRY("gemm_bf16_nt_m64",
        gemm_bf16_m64_launch(r4d_addr(g.a), r4d_addr(g.b), r4d_addr(g.y),
                             (int)g.M, (int)g.K, (int)g.N, c.wv, c.sk, c.mb,
                             (long)(uintptr_t)s, chain, g.lda, g.ldc));
    return RAD_OK;
}

