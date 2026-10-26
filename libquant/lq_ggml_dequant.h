/* lq_ggml_dequant.h -- llama.cpp's dequantize_row_<type> for the types a kernel reads natively,
 * over the native blocks lq_ggml_planes' pack_of writes back.
 *
 * The reference a kernel's dot product is checked against: these are the values llama.cpp itself
 * dequantises to, and the core's enc_decode_rows of the same blocks' planes gives the same f32
 * values bit for bit (lq_ggml_planes.cpp says why; checkpoint_test pins it). Shared like
 * lq_ggml_planes: the core and libref compile it from libquant's sources.
 */
#pragma once
#include <cstdint>

namespace lq::ggml {

/* `k` values (a whole number of blocks) of one row's native blocks -> f32 at y. */
typedef void (*DequantFn)(const void* blocks, float* y, int64_t k);

/* Null for a type with no native reader yet (so far: q8_0, q5_K, q6_K, iq4_nl, iq2_xxs, iq1_s,
 * mxfp4). */
DequantFn dequant_of(uint32_t type);

/* IEEE half -> float, exactly (subnormals, infinities and NaN included). */
float f16_to_f32(uint16_t h);

/* ggml-impl.h's ggml_e8m0_to_fp32_half: MXFP4's block scale 2^(e - 128), exactly (the two
 * subnormals, e = 0 and 1, included). e = 255 is 2^127: ggml reads no NaN there, where OCP E8M0
 * (and the core's decode of the e8m0 plane) reads 0xFF as NaN. ggml's quantiser never writes 255
 * for a finite weight. */
float e8m0_to_fp32_half(uint8_t e);

}  /* namespace lq::ggml */
