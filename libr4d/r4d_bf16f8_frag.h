// r4d_bf16f8_frag.h - gemm_nt_bf16f8's register expansion (gfx1201): a lane's eight E4M3 codes of
// one k step and their block's E8M0 exponent, as the bf16 WMMA fragment gemm_bf16_nt_m64 loads
// from the plane. Device code, shared by r4d_gemm_bf16f8_nt_m64.hip and the card check that dumps
// these registers for every code and exponent (tools/ds41-host-experts/ds41_bf16f8_device_check.hip),
// so the check runs the instructions the kernel runs.
//
// Why the bits are the plane's (r4d_bf16f8_layout.h): v_cvt_pk_f32_fp8 decodes every non-NaN E4M3
// byte exactly to f32, a normal number or a signed zero (subnormal codes included); 2^(s-127) is a
// normal f32 for every stored exponent, so the product is exact, keeps the sign of a zero, and with
// at most four significant bits its low 16 bits are zero. Its high half is then the plane's bf16,
// which the relayout required to be E4M3 code times 2^(s-127) exactly. The plugin build has
// -ffp-contract=off and no fast math, so the product is a plain v_mul_f32 (or its VOPD pair).
#pragma once
#include "r4d_gdn_wmma.h"   // v8bf

typedef unsigned r4d_bf8_u4 __attribute__((ext_vector_type(4)));
typedef float r4d_bf8_f2 __attribute__((ext_vector_type(2)));

// An E8M0 exponent byte as the f32 power of two it names, 2^(s-127); normal for every stored one.
__device__ __forceinline__ float r4d_bf16f8_pow2(unsigned s) { return __uint_as_float(s << 23); }

// Two E4M3 bytes of q (bytes 0 and 1, or with HI 2 and 3), decoded and each times f, as f32: the
// lower byte's in .x.
template <bool HI>
__device__ __forceinline__ r4d_bf8_f2 r4d_bf16f8_products(unsigned q, float f) {
  const r4d_bf8_f2 v = __builtin_amdgcn_cvt_pk_f32_fp8((int)q, HI);
  r4d_bf8_f2 p;
  p.x = v.x * f;
  p.y = v.y * f;
  return p;
}
// The same two values as one dword of bf16, their products' high halves: the lower byte's in the
// low half, as fragU's little-endian load has it (r4d_gdn_wmma.h's f2bf2 pairing, without its
// rounding: the products are exact and their low halves zero).
template <bool HI>
__device__ __forceinline__ unsigned r4d_bf16f8_pair(unsigned q, float f) {
  const r4d_bf8_f2 p = r4d_bf16f8_products<HI>(q, f);
  return __builtin_amdgcn_perm(__float_as_uint(p.y), __float_as_uint(p.x), 0x07060302u);
}

// A k step's eight slots, 0..3 in d0 and 4..7 in d1 (r4d_bf16f8_slot's order), as the four dwords
// of fragU's fragment.
__device__ __forceinline__ v8bf r4d_bf16f8_frag(unsigned d0, unsigned d1, float f) {
  r4d_bf8_u4 r;
  r.x = r4d_bf16f8_pair<false>(d0, f);
  r.y = r4d_bf16f8_pair<true>(d0, f);
  r.z = r4d_bf16f8_pair<false>(d1, f);
  r.w = r4d_bf16f8_pair<true>(d1, f);
  return __builtin_bit_cast(v8bf, r);
}
