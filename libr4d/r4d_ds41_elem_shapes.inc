/* r4d_ds41_elem_shapes.inc -- operand descriptions of DeepSeek-V4.1's elem ops; included at the end
 * of r4d_shapes.cpp for its helpers (opd, opd_idx, opd_none, pick, R4D_PICK, pi, dt_act).
 *
 * Each is libref's ref_<op>_shape (libref/ref_shapes.cpp) operand for operand, extents, dtypes and
 * index ranges, with the same declines, since rad-kbench fills both kernels from one description. */

extern "C" int r4d_shape_token_map(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), nv = pi(p, n_p, "n_vocab");
    if (M <= 0 || nv < 1) return RAD_E_UNSUPPORTED;
    const RadOpdDesc l[] = {
        opd_idx(M, 0, nv),                 /* ids */
        opd_idx(nv, 0, nv),                /* table */
        opd((uint32_t)RAD_I32, M),         /* mapped */
    };
    R4D_PICK(l);
}

extern "C" int r4d_shape_stream_gate_add(const RadParam* p, int n_p, int operand,
                                         RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), n = pi(p, n_p, "n"), hc = pi(p, n_p, "hc");
    if (M <= 0 || n < 1 || hc < 1 || hc > 16) return RAD_E_UNSUPPORTED;
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = {
        opd(D, M, hc * n),                 /* h */
        opd(D, M, hc * n),                 /* k */
        opd(D, M, n),                      /* v */
        opd((uint32_t)RAD_F32, hc * n),    /* w */
        opd(D, M, hc * n),                 /* out */
    };
    R4D_PICK(l);
}

extern "C" int r4d_shape_rope_rows(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), nh = pi(p, n_p, "n_head"), hd = pi(p, n_p, "head_dim");
    const int64_t rot = pi(p, n_p, "rot"), off = pi(p, n_p, "off", -1);
    if (M <= 0 || nh < 1 || rot < 2 || (rot & 1) || off < 0 || off + rot > hd)
        return RAD_E_UNSUPPORTED;
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = {
        opd(D, M, nh * hd),                /* x, rotated in place */
        opd((uint32_t)RAD_F32, M, rot),    /* cos_sin: a rope_table plane of M rows */
        opd_idx(M, 0, M),                  /* positions */
    };
    R4D_PICK(l);
}

extern "C" int r4d_shape_quant_roundtrip(const RadParam* p, int n_p, int operand,
                                         RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), n = pi(p, n_p, "n"), block = pi(p, n_p, "block");
    if (M <= 0 || n < 1 || block < 1 || n % block) return RAD_E_UNSUPPORTED;
    const RadOpdDesc l[] = { opd(dt_act(p, n_p, "dtype"), M, n) };
    R4D_PICK(l);
}

extern "C" int r4d_shape_mhc_mix(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), n = pi(p, n_p, "n"), hc = pi(p, n_p, "hc");
    if (M <= 0 || n <= 0 || hc < 2 || hc > 8) return RAD_E_UNSUPPORTED;
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = {
        opd(D, M, hc * n),                              /* h */
        opd((uint32_t)RAD_F32, hc * (hc + 2), hc * n),  /* fn */
        opd((uint32_t)RAD_F32, 3),                      /* scale */
        opd((uint32_t)RAD_F32, hc * (hc + 2)),          /* base */
        opd((uint32_t)RAD_F32, M, hc),                  /* pre */
        opd((uint32_t)RAD_F32, M, hc),                  /* post */
        opd((uint32_t)RAD_F32, M, hc * hc),             /* comb */
    };
    R4D_PICK(l);
}

extern "C" int r4d_shape_mhc_collapse(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), n = pi(p, n_p, "n"), hc = pi(p, n_p, "hc");
    if (M <= 0 || n <= 0 || hc < 2 || hc > 8) return RAD_E_UNSUPPORTED;
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = { opd(D, M, hc * n), opd((uint32_t)RAD_F32, M, hc), opd(D, M, n) };
    R4D_PICK(l);
}

extern "C" int r4d_shape_softmax_pool(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), n = pi(p, n_p, "n"), ratio = pi(p, n_p, "ratio");
    if (M <= 0 || n < 1 || ratio < 1 || ratio > 64) return RAD_E_UNSUPPORTED;
    const RadOpdDesc l[] = {
        opd((uint32_t)RAD_F32, M * ratio, 2 * n),  /* stage: values then scores */
        opd(dt_act(p, n_p, "dtype"), M, n),        /* out */
        opd_idx(M, 0, M),                          /* page, optional: negative skips the entry */
    };
    R4D_PICK(l);
}

extern "C" int r4d_shape_entry_store(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), n = pi(p, n_p, "n"), bs = pi(p, n_p, "block_size");
    if (M <= 0 || n < 1 || bs < 1 || n % bs) return RAD_E_UNSUPPORTED;
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = {
        opd(D, M, n),                                       /* x */
        opd_idx(M, 0, M, -1, RAD_OPD_F_IDX_UNIQUE),         /* page: distinct blocks */
        opd(D, M, 1, bs, n / bs),                           /* cache */
    };
    R4D_PICK(l);
}

extern "C" int r4d_shape_router_scored_topk(const RadParam* p, int n_p, int operand,
                                            RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), ne = pi(p, n_p, "n_expert"), k = pi(p, n_p, "top_k");
    if (M < 1 || ne < 1 || ne > 512 || k < 1 || k > ne) return RAD_E_UNSUPPORTED;
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = { opd(D, M, ne), opd((uint32_t)RAD_F32, ne), opd_idx(M, k, ne),
                             opd(D, M, k) };
    R4D_PICK(l);
}

/* The planes as a host tool draws them: i8 codes (no table), a scale per `group` of a row. */
extern "C" int r4d_shape_embed_lookup_planes(const RadParam* p, int n_p, int operand,
                                             RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M");
    const int64_t ne = pi(p, n_p, "n_embd"), nv = pi(p, n_p, "n_vocab");
    const int64_t G = pi(p, n_p, "group"), vo = pi(p, n_p, "vocab_offset", 0);
    if (M <= 0 || ne <= 0 || nv <= 0 || G <= 0 || ne % G || vo < 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = {
        opd_idx(M, 0, vo + nv),            /* tokens */
        opd((uint32_t)RAD_I8, nv, ne),     /* codes */
        opd(D, nv, ne / G),                /* scale */
        opd_none(),                        /* table: optional */
        opd(D, M, ne),                     /* x */
    };
    R4D_PICK(l);
}

/* The fused chains: the operands of their steps' rows, as those rows describe them. */
extern "C" int r4d_shape_rmsnorm_rope_roundtrip(const RadParam* p, int n_p, int operand,
                                                RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), n = pi(p, n_p, "n"), nh = pi(p, n_p, "n_head");
    const int64_t hd = pi(p, n_p, "head_dim"), rot = pi(p, n_p, "rot"), off = pi(p, n_p, "off", -1);
    const int64_t block = pi(p, n_p, "block");
    if (M <= 0 || nh < 1 || rot < 2 || (rot & 1) || off < 0 || off + rot > hd || n != nh * hd ||
        block < 1 || n % block)
        return RAD_E_UNSUPPORTED;
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = {
        opd(D, M, n),                      /* x */
        opd((uint32_t)RAD_F32, n),         /* w, the gain */
        opd(D, M, n),                      /* y */
        opd((uint32_t)RAD_F32, M, rot),    /* cos_sin: a rope_table plane of M rows */
        opd_idx(M, 0, M),                  /* positions */
    };
    R4D_PICK(l);
}

extern "C" int r4d_shape_rope_rows_roundtrip(const RadParam* p, int n_p, int operand,
                                             RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), nh = pi(p, n_p, "n_head"), hd = pi(p, n_p, "head_dim");
    const int64_t rot = pi(p, n_p, "rot"), off = pi(p, n_p, "off", -1), block = pi(p, n_p, "block");
    if (M <= 0 || nh < 1 || rot < 2 || (rot & 1) || off < 0 || off + rot > hd || block < 1 ||
        (nh * hd) % block)
        return RAD_E_UNSUPPORTED;
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = {
        opd(D, M, nh * hd),                /* x, in place */
        opd((uint32_t)RAD_F32, M, rot),    /* cos_sin */
        opd_idx(M, 0, M),                  /* positions */
    };
    R4D_PICK(l);
}

extern "C" int r4d_shape_mhc_collapse_rmsnorm(const RadParam* p, int n_p, int operand,
                                              RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), n = pi(p, n_p, "n"), hc = pi(p, n_p, "hc");
    if (M <= 0 || n <= 0 || hc < 2 || hc > 8) return RAD_E_UNSUPPORTED;
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = {
        opd(D, M, hc * n),                 /* h */
        opd((uint32_t)RAD_F32, M, hc),     /* pre */
        opd(D, M, n),                      /* x */
        opd((uint32_t)RAD_F32, n),         /* w */
        opd(D, M, n),                      /* y */
    };
    R4D_PICK(l);
}

extern "C" int r4d_shape_mhc_post_collapse_rmsnorm(const RadParam* p, int n_p, int operand,
                                                   RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), n = pi(p, n_p, "n"), hc = pi(p, n_p, "hc");
    if (M <= 0 || n <= 0 || hc < 2 || hc > 8) return RAD_E_UNSUPPORTED;
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = {
        opd(D, M, n),                      /* x */
        opd(D, M, hc * n),                 /* h */
        opd((uint32_t)RAD_F32, M, hc),     /* post */
        opd((uint32_t)RAD_F32, M, hc * hc), /* comb */
        opd(D, M, hc * n),                 /* out */
        opd((uint32_t)RAD_F32, M, hc),     /* pre */
        opd(D, M, n),                      /* xc */
        opd((uint32_t)RAD_F32, n),         /* w */
        opd(D, M, n),                      /* y */
    };
    R4D_PICK(l);
}
