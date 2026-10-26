/* r4d_ds41_mxfp4_shapes.inc -- operand descriptions of gemm_nt_ggml and moe_gemm_ggml at type=mxfp4; included at
 * the end of r4d_shapes.cpp for its helpers (opd, opd_idx, pick, R4D_PICK, pi, dt_act).
 *
 * libref's (ref_shapes.cpp, ref_gemm_nt_ggml_shape / ref_moe_gemm_ggml_shape with ref_ggml_block("mxfp4") = 32)
 * operand for operand: the weight in its float logical form ([N, K] f32, [n_expert, N, K] f32), which the rows read
 * as it is; a real container's weights reach them through the layout hooks instead. Any other type, or K not whole
 * blocks of 32, is a decline. */

namespace {
bool ds41_mx_type(const RadParam* p, int n_p) {
    const char* s = r4d_pgets(p, n_p, "type");
    return s && !std::strcmp(s, "mxfp4");
}
}  // namespace

extern "C" int r4d_shape_gemm_nt_mxfp4(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), N = pi(p, n_p, "N"), K = pi(p, n_p, "K");
    if (!ds41_mx_type(p, n_p) || M <= 0 || N <= 0 || K <= 0 || K % 32) return RAD_E_UNSUPPORTED;
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = {
        opd(D, M, K),                       /* a */
        opd((uint32_t)RAD_F32, N, K),       /* w, the logical form */
        opd(D, M, N),                       /* y */
    };
    R4D_PICK(l);
}

extern "C" int r4d_shape_moe_gemm_mxfp4(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), N = pi(p, n_p, "N"), K = pi(p, n_p, "K");
    const int64_t ne = pi(p, n_p, "n_expert"), k = pi(p, n_p, "top_k");
    if (!ds41_mx_type(p, n_p) || M <= 0 || N <= 0 || K <= 0 || K % 32 || ne <= 0 || k <= 0)
        return RAD_E_UNSUPPORTED;
    const char* ord = r4d_pgets(p, n_p, "a_order");
    const int64_t AR = (ord && !std::strcmp(ord, "sorted")) ? M * k : M;
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = {
        opd(D, AR, K),                                                   /* a -- see a_order */
        opd((uint32_t)RAD_F32, ne, N, K),                                /* w, the logical form */
        /* A valid routing, as moe_scatter produces it (libref's moe_gemm description). */
        opd_idx(M * k, 0, M * k, -1, (uint32_t)RAD_OPD_F_IDX_UNIQUE),   /* sorted_tok */
        opd_idx(ne + 1, 0, M * k + 1, -1, (uint32_t)RAD_OPD_F_IDX_PREFIX), /* expert_offset */
        opd(D, M * k, N),                                                /* y, sorted order */
    };
    R4D_PICK(l);
}
