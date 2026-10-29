// r4d_gemm_m64_shared.h - the parts of r4d_gemm_bf16_nt_m64's body a kernel over another stored
// form of the same weight repeats verbatim: the wave geometry and K slicing, the activation's row
// offsets, and the split-K reduction and store. A kernel that expands its weight to the bf16 bits
// fragU would have loaded (r4d_gemm_bf16f8_nt_m64.hip) and keeps these computes the same bytes.
//
// Macros rather than functions on purpose: written as inline functions, the same statements
// changed the bf16 kernel's address arithmetic, while these expand to its own source, so its code
// is unchanged instruction for instruction.
//
// They read the kernel's own names: the template parameter MT, the arguments A, C, M, K, N, WV, SK,
// LDA and LDC, WAVE32, and between them the accumulators `acc` and the A offsets `aoff`.
#pragma once
#include "r4d_launch.h"

// The block's waves: WV column tiles x SK k-splits, each wave one 16-wide N tile over one
// contiguous K slice of Ks = K / SK; MT row tiles of accumulators from zero. `red` is the dynamic
// LDS, WV*SK*256 floats: one 16x16 tile per (column, split).
#define R4D_M64_WAVE_SETUP()                                                                       \
  R4D_DYN_LDS(float, red);                                                                         \
  const int tid = threadIdx.x, wave = tid / WAVE32, lane = tid % WAVE32;                           \
  const int col_local = wave / SK, ksplit = wave % SK;                                             \
  const int nt = blockIdx.x * WV + col_local;                                                      \
  const int n0 = nt * 16;                                                                          \
  const int mt0 = blockIdx.y * MT;                                                                 \
  const int Ks = K / SK, ksteps = Ks / 16, kbase = ksplit * Ks;                                    \
                                                                                                   \
  v8f acc[MT];                                                                                     \
  _Pragma("unroll")                                                                                \
  for (int i = 0; i < MT; i++) acc[i] = (v8f)(0.f)

// The activation's row offsets as 32-bit element offsets (rowGo's clamp keeps rows past M on the
// last one), each at this wave's K slice.
#define R4D_M64_A_OFFSETS()                                                                        \
  unsigned aoff[MT];                                                                               \
  _Pragma("unroll")                                                                                \
  for (int i = 0; i < MT; i++)                                                                     \
    aoff[i] = rowGo((unsigned)LDA, (mt0 + i) * 16, M - 1, lane) + (unsigned)kbase

// Reduce one row tile at a time, so LDS is WV*SK KiB regardless of MT. Every thread in the block
// shares the sum (one element per thread at SK=8); the order over splits is fixed: from 0.f, in
// split order. The store is the language's float-to-bf16 conversion on purpose; the header's f2bf
// rounds half away from zero.
#define R4D_M64_REDUCE_STORE()                                                                     \
  const int slot = (col_local * SK + ksplit) * 256;                                                \
  const int nthreads = WV * SK * WAVE32;                                                           \
  _Pragma("unroll")                                                                                \
  for (int i = 0; i < MT; i++) {                                                                   \
    __syncthreads();                                                                               \
    _Pragma("unroll")                                                                              \
    for (int e = 0; e < 8; e++) red[slot + e * WAVE32 + lane] = acc[i][e];   /* conflict-free */   \
    __syncthreads();                                                                               \
    for (int idx = tid; idx < WV * 256; idx += nthreads) {                                         \
      const int col = idx >> 8, off = idx & 255;                                                   \
      float v = 0.f;                                                                               \
      for (int s = 0; s < SK; s++) v += red[(col * SK + s) * 256 + off];                           \
      const int l = off & 31;                                                                      \
      const int n = (blockIdx.x * WV + col) * 16 + (l & 15);                                       \
      const int m = (mt0 + i) * 16 + 8 * (l >> 4) + (off >> 5);                                   \
      if (n < N && m < M) C[(size_t)m * LDC + n] = (__hip_bfloat16)v;                              \
    }                                                                                              \
  }
