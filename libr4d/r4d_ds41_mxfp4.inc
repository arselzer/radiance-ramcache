/* r4d_ds41_mxfp4.inc -- gemm_nt_ggml and moe_gemm_ggml rows over GGUF MXFP4 blocks (type=mxfp4) in r4d_rows.cpp,
 * included three times: R4D_DS41_SPECS (constraints, file scope), R4D_DS41_SCHEMAS (none: the ops' schemas are
 * r4d_ds41_ggml.inc's, libref's field for field) and R4D_DS41_ROWS (entries of kKernels). No include guard, on
 * purpose.
 *
 * Kernels in r4d_ds41_mxfp4.hip. Each op has a weight-streaming `_gemv` row for the decode band (M <= 16; for the
 * grouped GEMM M counts tokens) above a `_tiled` row that serves any M, both only at type=mxfp4 (r4d_ds41_ggml's
 * rows take the other types), K whole blocks of 32. Every row relayouts a GGUF import's planes (fp4_e2m1 codes,
 * e8m0 per 32) into llama.cpp's native MXFP4 blocks with r4d_ds41_ggml_layout.cpp's hooks (libref's bytes and tag)
 * and also reads the float logical form, which is what rad-kbench's random operands pass. Built only with
 * libquant's sources (R4D_HAVE_GGML), which the layout needs. */
#if defined(R4D_DS41_SPECS)

static const RadConstraint cDs41MxGemv[] = {
    C_GE("M", 1), C_LE("M", 16), C_GE("N", 1), C_GE("K", 32), C_DIV("K", 32),
    C_IN("type", "mxfp4"), C_IN("dtype", "bf16"),
};
static const RadConstraint cDs41MxTiled[] = {
    C_GE("M", 1), C_GE("N", 1), C_GE("K", 32), C_DIV("K", 32),
    C_IN("type", "mxfp4"), C_IN("dtype", "bf16"),
};
static const RadConstraint cDs41MxMoeGemv[] = {
    C_GE("M", 1), C_LE("M", 16), C_GE("N", 1), C_GE("K", 32), C_DIV("K", 32),
    C_GE("n_expert", 1), C_LE("n_expert", 1024), C_GE("top_k", 1), C_IN("a_order", "token sorted"),
    C_IN("type", "mxfp4"), C_IN("dtype", "bf16"),
};
static const RadConstraint cDs41MxMoeTiled[] = {
    C_GE("M", 1), C_GE("N", 1), C_GE("K", 32), C_DIV("K", 32),
    C_GE("n_expert", 1), C_LE("n_expert", 1024), C_GE("top_k", 1), C_IN("a_order", "token sorted"),
    C_IN("type", "mxfp4"), C_IN("dtype", "bf16"),
};

#elif defined(R4D_DS41_SCHEMAS)

/* gemm_nt_ggml's and moe_gemm_ggml's schemas are r4d_ds41_ggml.inc's. */

#elif defined(R4D_DS41_ROWS)
#if defined(R4D_HAVE_GGML)

#define R4D_DS41_MX_HOOKS(shape)                                                                   \
    r4d_ds41_ggml_layout, r4d_ds41_ggml_relayout, shape, r4d_ds41_ggml_unrelayout

{ "gemm_nt_ggml_mxfp4_gemv", "gemm_nt_ggml", "gemm",
  "y = a . w^T over GGUF MXFP4 blocks, the weight streamed once: a wave per weight row (two from 3 rows on wide "
  "weights), lanes over its 17-byte blocks (four in flight, each read as the five aligned dwords covering it), "
  "the activation chunk staged in LDS as f32, a wave butterfly per sum",
  "M 1..16 (eight rows a pass), any N, K a multiple of 32", "bf16 a and y; mxfp4 blocks (or the f32 logical "
  "form), kvalues_mxfp4 x 2^(e - 128) as llama.cpp dequantises, f32 products and sum",
  RAD_DOMAIN_DEVICE, 58, ROW(cDs41MxGemv), NOTUNE,
  nullptr, nullptr, r4d_rad_gemm_nt_mxfp4_gemv, nullptr, R4D_DS41_MX_HOOKS(r4d_shape_gemm_nt_mxfp4) },
{ "gemm_nt_ggml_mxfp4_tiled", "gemm_nt_ggml", "gemm",
  "y = a . w^T over GGUF MXFP4 blocks in 128 x 128 tiles: the weight tile dequantised into LDS once a block-wide "
  "slab, the next slab's blocks prefetched, f32 8 x 8 register tiles",
  "any M, any N, K a multiple of 32", "bf16 a and y; mxfp4 blocks (or the f32 logical form), f32 products and sum",
  RAD_DOMAIN_DEVICE, 30, ROW(cDs41MxTiled), NOTUNE,
  nullptr, nullptr, r4d_rad_gemm_nt_mxfp4_tiled, nullptr, R4D_DS41_MX_HOOKS(r4d_shape_gemm_nt_mxfp4) },
{ "moe_gemm_ggml_mxfp4_gemv", "moe_gemm_ggml", "moe",
  "the grouped GEMM over GGUF MXFP4 experts, each live expert's weight streamed once: a block scans the offsets "
  "for the experts with rows, then a wave per weight row (two on experts of 1024 rows or more) dots it with that "
  "expert's gathered rows (eight a pass)",
  "M 1..16 tokens, any N, K a multiple of 32, up to 1024 experts", "bf16 a and y; mxfp4 experts (table or "
  "stacked, or the f32 logical form), i32 routing",
  RAD_DOMAIN_DEVICE, 58, ROW(cDs41MxMoeGemv), NOTUNE,
  nullptr, nullptr, r4d_rad_moe_gemm_mxfp4_gemv, nullptr, R4D_DS41_MX_HOOKS(r4d_shape_moe_gemm_mxfp4) },
{ "moe_gemm_ggml_mxfp4_tiled", "moe_gemm_ggml", "moe",
  "the grouped GEMM over GGUF MXFP4 experts in 128-row (or 32-row) x 128-column tiles of one expert's run (a "
  "block finds its expert by scanning the offsets), gemm_nt_ggml_mxfp4_tiled's slab loop",
  "any M, any N, K a multiple of 32, up to 1024 experts", "bf16 a and y; mxfp4 experts (or the f32 logical form), "
  "i32 routing",
  RAD_DOMAIN_DEVICE, 30, ROW(cDs41MxMoeTiled), NOTUNE,
  nullptr, nullptr, r4d_rad_moe_gemm_mxfp4_tiled, nullptr, R4D_DS41_MX_HOOKS(r4d_shape_moe_gemm_mxfp4) },

#undef R4D_DS41_MX_HOOKS
#endif  /* R4D_HAVE_GGML */
#endif
