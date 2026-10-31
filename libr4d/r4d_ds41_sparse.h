/* r4d_ds41_sparse.h -- launches and shape hooks of DeepSeek-V4.1's sparse ops; included by r4d_plugin.h
 * inside its extern "C". */
#ifndef R4D_DS41_SPARSE_H
#define R4D_DS41_SPARSE_H

/* r4d_ds41_sparse.hip */
int     r4d_rad_ds41_index_score(const RadArgs*, RadStream);
int     r4d_rad_ds41_index_score_kvpack(const RadArgs*, RadStream);
int     r4d_rad_ds41_attn_sparse_sink_windowpack(const RadArgs*, RadStream);
int     r4d_rad_ds41_attn_sparse_sink_windowpack_check(const RadArgs*, RadStream, int form);
int     r4d_rad_ds41_attn_sparse_sink_kvpack(const RadArgs*, RadStream);
/* Unregistered byte-oracle entry points; no architecture selects them. */
int     r4d_rad_ds41_index_score_kvpack_check(const RadArgs*, RadStream, int form);
int     r4d_rad_ds41_attn_sparse_sink_kvpack_check(const RadArgs*, RadStream, int form);
int     r4d_rad_ds41_index_candidates(const RadArgs*, RadStream);
int     r4d_rad_ds41_index_topk(const RadArgs*, RadStream);
int     r4d_rad_ds41_index_score_b16(const RadArgs*, RadStream);
int     r4d_rad_ds41_index_candidates_b16(const RadArgs*, RadStream);
int     r4d_rad_ds41_index_topk_b16(const RadArgs*, RadStream);
uint32_t r4d_rad_ds41_index_b16_paths(); // new-alias diagnostic actual scalar/WMMA/select coverage
int64_t r4d_rad_ds41_index_topk_scratch(const RadArgs*);
int     r4d_rad_ds41_attn_sparse_sink(const RadArgs*, RadStream);

/* r4d_ds41_sparse_shapes.inc */
int r4d_shape_index_score(const RadParam*, int, int, RadOpdDesc*);
int r4d_shape_index_candidates(const RadParam*, int, int, RadOpdDesc*);
int r4d_shape_index_topk(const RadParam*, int, int, RadOpdDesc*);
int r4d_shape_index_score_b16(const RadParam*, int, int, RadOpdDesc*);
int r4d_shape_index_candidates_b16(const RadParam*, int, int, RadOpdDesc*);
int r4d_shape_index_topk_b16(const RadParam*, int, int, RadOpdDesc*);
int r4d_shape_attn_sparse_sink(const RadParam*, int, int, RadOpdDesc*);

#endif
