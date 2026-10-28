/* r4d_ds41_mxfp4.h -- launches and shape hooks of gemm_nt_ggml and moe_gemm_ggml over GGUF MXFP4 blocks
 * (type=mxfp4, ggml type 39); included by r4d_plugin.h inside its extern "C".
 *
 * Kernels in r4d_ds41_mxfp4.hip; the weight's layout hooks are r4d_ds41_ggml_layout.cpp's (libref's bytes and
 * tag, "ggml_mxfp4"), built only with libquant's sources (R4D_HAVE_GGML). */
#ifndef R4D_DS41_MXFP4_H
#define R4D_DS41_MXFP4_H

/* -- launches: `_gemv` the weight-streaming decode path (M <= 16 rows, or tokens for the grouped GEMM),
 * `_tiled` the 128 x 128 tile path at any M. */
int r4d_rad_gemm_nt_mxfp4_gemv(const RadArgs*, RadStream);
int r4d_rad_gemm_nt_mxfp4_tiled(const RadArgs*, RadStream);
int r4d_rad_moe_gemm_mxfp4_gemv(const RadArgs*, RadStream);
int r4d_rad_moe_gemm_mxfp4_tiled(const RadArgs*, RadStream);

/* -- operand descriptions (r4d_ds41_mxfp4_shapes.inc), libref's for type=mxfp4: the weight in its float
 * logical form. */
int r4d_shape_gemm_nt_mxfp4(const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r4d_shape_moe_gemm_mxfp4(const RadParam* p, int n_p, int operand, RadOpdDesc* out);

#endif
