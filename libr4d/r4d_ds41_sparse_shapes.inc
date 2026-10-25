/* r4d_ds41_sparse_shapes.inc -- operand descriptions of DeepSeek-V4.1's sparse ops; included at the end
 * of r4d_shapes.cpp for its helpers (opd, opd_idx, opd_none, pick, R4D_PICK, pi, dt_act).
 *
 * Each is libref's description (libref/ref_shapes.cpp) operand for operand: one sequence, window
 * blocks of 4 slots covering every position a row may hold, one compressed or index entry a block. */

extern "C" int r4d_shape_attn_sparse_sink(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), H = pi(p, n_p, "heads"), Dm = pi(p, n_p, "dim");
    const int64_t win = pi(p, n_p, "window"), ratio = pi(p, n_p, "ratio"), topk = pi(p, n_p, "topk");
    if (M <= 0 || H < 1 || Dm < 2 || win < 1 || ratio < 0 || (ratio > 0 && (topk < 1 || Dm % ratio)))
        return RAD_E_UNSUPPORTED;
    const uint32_t D = dt_act(p, n_p, "dtype");
    const int64_t bs = 4, nb = (M + bs - 1) / bs + 1, ncb = M + 1;
    const bool comp = ratio > 0;
    const RadOpdDesc l[] = {
        opd(D, M, H * Dm),                                  /* q */
        opd(D, nb, 1, bs, Dm),                              /* window cache */
        opd_idx(1, nb, nb),                                 /* window block table */
        opd_idx(1, 0, M + 1),                               /* window tokens held from the table's start */
        comp ? opd(D, ncb, 1, ratio, Dm / ratio) : opd_none(),  /* compressed cache: an entry a block */
        comp ? opd_idx(1, ncb, ncb) : opd_none(),           /* compressed block table */
        comp ? opd_idx(M, topk, ncb) : opd_none(),          /* selected compressed positions */
        opd_idx(M, 0, M),                                   /* positions */
        opd_idx(2, 0, M, -1, (uint32_t)RAD_OPD_F_IDX_CU),   /* cu_seqlens */
        opd((uint32_t)RAD_F32, H),                          /* sinks */
        opd(D, M, H * Dm),                                  /* out */
    };
    R4D_PICK(l);
}

extern "C" int r4d_shape_index_score(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), H = pi(p, n_p, "heads"), Dm = pi(p, n_p, "dim");
    const int64_t ratio = pi(p, n_p, "ratio"), C = pi(p, n_p, "width");
    if (M <= 0 || H < 1 || Dm < 1 || ratio < 1 || Dm % ratio || C < 1) return RAD_E_UNSUPPORTED;
    const uint32_t D = dt_act(p, n_p, "dtype");
    const int64_t ncb = C + 1;
    const RadOpdDesc l[] = {
        opd(D, M, H * Dm),                                  /* q */
        opd(D, M, H),                                       /* head weights */
        opd(D, ncb, 1, ratio, Dm / ratio),                  /* index cache: an entry a block */
        opd_idx(1, ncb, ncb),                               /* block table */
        opd_idx(M, 0, M),                                   /* positions */
        opd_idx(2, 0, M, -1, (uint32_t)RAD_OPD_F_IDX_CU),   /* cu_seqlens */
        opd((uint32_t)RAD_F32, M, C),                       /* score */
    };
    R4D_PICK(l);
}

extern "C" int r4d_shape_index_candidates(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), C = pi(p, n_p, "width");
    if (M <= 0 || C < 1 || pi(p, n_p, "block") < 1 || pi(p, n_p, "topk_blocks") < 1 ||
        pi(p, n_p, "ratio") < 1)
        return RAD_E_UNSUPPORTED;
    const RadOpdDesc l[] = {
        opd((uint32_t)RAD_F32, M, C),                       /* score */
        opd_idx(M, 0, M),                                   /* positions */
        opd((uint32_t)RAD_I32, M, C),                       /* keep */
    };
    R4D_PICK(l);
}

extern "C" int r4d_shape_index_topk(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), C = pi(p, n_p, "width"), k = pi(p, n_p, "topk");
    if (M <= 0 || C < 1 || k < 1 || pi(p, n_p, "ratio") < 1) return RAD_E_UNSUPPORTED;
    const RadOpdDesc l[] = {
        opd((uint32_t)RAD_F32, M, C),                       /* score */
        opd_none(),                                         /* keep, optional */
        opd_idx(M, 0, M),                                   /* positions */
        opd((uint32_t)RAD_I32, M, k),                       /* sel */
    };
    R4D_PICK(l);
}
