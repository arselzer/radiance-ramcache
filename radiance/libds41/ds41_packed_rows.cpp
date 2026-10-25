/* Experimental schemas/host layouts: no runtime initialization, device query or allocation.
 * Built only into an explicit experimental output directory, never installed automatically. */
#include "ds41_packed_args.h"
#include "ds41_latent_args.h"
#include "ds41_math.h"
#include "ds41_compress.h"
#include "ds41_ffn.h"
#include "ds41_index.h"
#include "ds41_index_paged.h"
#include "ds41_transform.h"
#include "ds41_attention_paged.h"
#include "ds41_route_jobs.h"
#include "ds41_index_stream.h"
#include "ds41_embedding_args.h"
#include "ds41_reorder_heads.h"
#include "ds41_compress_finish.h"
#include "ds41_latent_emit.h"
#include "ds41_engram_gate.h"
#include "ds41_qdq.h"
#ifdef DS41_HOST_REFERENCE
#include "ds41_host_ops.h"
#endif
#include <cstring>
#include <cstdio>
#if !defined(DS41_HOST_REFERENCE) && !defined(DS41_METADATA_ONLY)
#include "ds41_status.h"
#endif
namespace packed=rad::ds41::packed;
int packed::prepare(const RadArgs* a,bool routed,Launch* out) {return validate(a,routed,out);}
namespace {
#define I(k) {k,RAD_P_INT,RAD_REQUIRED}
#define S(k) {k,RAD_P_STR,RAD_REQUIRED}
const RadParamSpec dense_params[]{I("M"),I("N"),I("K"),S("format"),S("a_dtype"),S("y_dtype")};
const RadParamSpec routed_params[]{I("M"),I("N"),I("K"),I("R"),I("n_expert"),S("format"),S("a_dtype"),S("y_dtype")};
#undef I
#undef S
const RadOperandSpec dense_operands[]{{"a",RAD_OPD_IN,0},{"w",RAD_OPD_WEIGHT,0},{"y",RAD_OPD_OUT,0}};
const RadOperandSpec routed_operands[]{{"a",RAD_OPD_IN,0},{"w",RAD_OPD_WTAB,0},
    {"expert_ids",RAD_OPD_IN,0},{"row_ids",RAD_OPD_IN,0},{"y",RAD_OPD_OUT,0},{"status",RAD_OPD_OUT,0}};
const RadParamSpec latent_params[]{{"M",RAD_P_INT,RAD_REQUIRED,RAD_PROLE_CAPACITY},
    {"dimension",RAD_P_INT,RAD_REQUIRED},{"ratio",RAD_P_INT,RAD_REQUIRED},
    {"block_size",RAD_P_INT,RAD_DERIVED},{"n_blocks",RAD_P_INT,RAD_OPTIONAL},{"dtype",RAD_P_STR,RAD_REQUIRED}};
const RadOperandSpec store_operands[]{{"x",RAD_OPD_IN,0},{"positions",RAD_OPD_IN,0},
    {"slot_mapping",RAD_OPD_IN,0},{"cache",RAD_OPD_INOUT,0},{"status",RAD_OPD_OUT,0}};
const RadOperandSpec gather_operands[]{{"cache",RAD_OPD_IN,0},{"first_slots",RAD_OPD_IN,0},
    {"x",RAD_OPD_OUT,0},{"status",RAD_OPD_OUT,0}};
// These four descriptors match libref/ref_registry.cpp exactly.
#define I(k) {k,RAD_P_INT,RAD_REQUIRED}
#define S(k) {k,RAD_P_STR,RAD_REQUIRED}
#define F(k) {k,RAD_P_F64,RAD_REQUIRED}
const RadParamSpec hc_prepare_params[]{I("M"),I("n"),I("hc"),I("iterations"),F("eps"),F("hc_eps"),S("dtype")};
const RadParamSpec hc_collapse_params[]{I("M"),I("n"),I("hc"),S("dtype")};
const RadParamSpec router_params[]{I("M"),I("experts"),I("topk"),F("temperature"),F("route_scale"),I("normalize")};
const RadParamSpec sparse_params[]{I("M"),I("heads"),I("head_dim"),I("kv_rows"),I("topk"),F("scale"),S("dtype")};
#undef I
#undef S
#undef F
const RadOperandSpec hc_prepare_operands[]{{"h",RAD_OPD_IN,0},{"fn",RAD_OPD_WEIGHT,0},
    {"scale",RAD_OPD_WEIGHT,0},{"base",RAD_OPD_WEIGHT,0},{"pre",RAD_OPD_OUT,0},{"post",RAD_OPD_OUT,0},{"comb",RAD_OPD_OUT,0}};
const RadOperandSpec hc_collapse_operands[]{{"h",RAD_OPD_IN,0},{"previous_pre",RAD_OPD_IN,0},{"x",RAD_OPD_OUT,0}};
const RadOperandSpec router_operands[]{{"logits",RAD_OPD_IN,0},{"correction",RAD_OPD_WEIGHT,0},{"ids",RAD_OPD_OUT,0},{"weights",RAD_OPD_OUT,0}};
const RadOperandSpec sparse_operands[]{{"q",RAD_OPD_IN,0},{"kv",RAD_OPD_IN,0},{"sinks",RAD_OPD_WEIGHT,0},{"indices",RAD_OPD_IN,0},{"out",RAD_OPD_OUT,0}};
const RadParamSpec expert_act_params[]{{"M",RAD_P_INT,RAD_REQUIRED,RAD_PROLE_CAPACITY},
    {"n",RAD_P_INT,RAD_REQUIRED},{"limit",RAD_P_F64,RAD_REQUIRED},{"dtype",RAD_P_STR,RAD_REQUIRED}};
const RadParamSpec combine_params[]{{"M",RAD_P_INT,RAD_REQUIRED,RAD_PROLE_CAPACITY},
    {"R",RAD_P_INT,RAD_REQUIRED},{"n",RAD_P_INT,RAD_REQUIRED},{"n_expert",RAD_P_INT,RAD_REQUIRED},
    {"dtype",RAD_P_STR,RAD_REQUIRED},{"tp_world",RAD_P_INT,RAD_REQUIRED}};
const RadOperandSpec expert_act_operands[]{{"gate",RAD_OPD_IN,0},{"up",RAD_OPD_IN,0},
    {"route_weight",RAD_OPD_IN,0},{"down_input",RAD_OPD_OUT,0}};
const RadOperandSpec combine_operands[]{{"routed",RAD_OPD_IN,0},{"expert_ids",RAD_OPD_IN,0},
    {"token_ids",RAD_OPD_IN,0},{"shared",RAD_OPD_IN,0},{"out",RAD_OPD_OUT,0}};
const RadOperandSpec mhc_post_operands[]{{"x",RAD_OPD_IN,0},{"h",RAD_OPD_IN,0},
    {"post",RAD_OPD_IN,0},{"comb",RAD_OPD_IN,0},{"out",RAD_OPD_OUT,0}};
const RadParamSpec ds41_norm_params[]{{"M",RAD_P_INT,RAD_REQUIRED,RAD_PROLE_CAPACITY},
    {"n",RAD_P_INT,RAD_REQUIRED},{"eps",RAD_P_F64,RAD_REQUIRED},{"a_dtype",RAD_P_STR,RAD_REQUIRED},{"y_dtype",RAD_P_STR,RAD_REQUIRED}};
const RadParamSpec ds41_rope_params[]{{"M",RAD_P_INT,RAD_REQUIRED,RAD_PROLE_CAPACITY},
    {"heads",RAD_P_INT,RAD_REQUIRED},{"head_dim",RAD_P_INT,RAD_REQUIRED},{"rotary_dim",RAD_P_INT,RAD_REQUIRED},
    {"dtype",RAD_P_STR,RAD_REQUIRED},{"inverse",RAD_P_INT,RAD_REQUIRED}};
const RadOperandSpec ds41_norm_operands[]{{"x",RAD_OPD_IN,0},{"gain",RAD_OPD_WEIGHT,0},{"out",RAD_OPD_OUT,0}};
const RadOperandSpec ds41_rope_operands[]{{"x",RAD_OPD_IN,0},{"positions",RAD_OPD_IN,0},{"pairs",RAD_OPD_IN,0},{"out",RAD_OPD_OUT,0}};
const RadParamSpec paged_attention_params[]{
    {"M",RAD_P_INT,RAD_REQUIRED,RAD_PROLE_CAPACITY},{"sequences",RAD_P_INT,RAD_REQUIRED,RAD_PROLE_CAPACITY},
    {"heads",RAD_P_INT,RAD_REQUIRED},{"head_dim",RAD_P_INT,RAD_REQUIRED},{"window",RAD_P_INT,RAD_REQUIRED},
    {"topk",RAD_P_INT,RAD_REQUIRED},{"ratio",RAD_P_INT,RAD_REQUIRED},{"block_size",RAD_P_INT,RAD_DERIVED},
    {"dtype",RAD_P_STR,RAD_REQUIRED},{"scale",RAD_P_F64,RAD_REQUIRED},{"n_spec",RAD_P_INT,RAD_REQUIRED}};
const RadOperandSpec paged_attention_operands[]{
    {"query",RAD_OPD_IN,0},{"window_cache",RAD_OPD_IN,0},{"window_table",RAD_OPD_IN,0},{"window_used",RAD_OPD_IN,0},
    {"compressed_cache",RAD_OPD_IN,1},{"compressed_table",RAD_OPD_IN,1},{"compressed_used",RAD_OPD_IN,1},
    {"selected",RAD_OPD_IN,1},{"positions",RAD_OPD_IN,0},{"cu",RAD_OPD_IN,0},{"ctx_lens",RAD_OPD_IN,0},
    {"sinks",RAD_OPD_WEIGHT,0},{"output",RAD_OPD_OUT,0}};
const RadParamSpec route_jobs_params[]{{"M",RAD_P_INT,RAD_REQUIRED,RAD_PROLE_CAPACITY},
    {"topk",RAD_P_INT,RAD_REQUIRED},{"n_expert",RAD_P_INT,RAD_REQUIRED}};
const RadOperandSpec route_jobs_operands[]{{"ids",RAD_OPD_IN,0},{"weights",RAD_OPD_IN,0},
    {"expert_ids",RAD_OPD_OUT,0},{"token_ids",RAD_OPD_OUT,0},{"job_weights",RAD_OPD_OUT,0},
    {"expert_counts",RAD_OPD_OUT,0},{"identity_rows",RAD_OPD_OUT,0}};
const RadParamSpec stream_params[]{
    {"M",RAD_P_INT,RAD_REQUIRED,RAD_PROLE_CAPACITY},{"key_rows",RAD_P_INT,RAD_REQUIRED},
    {"heads",RAD_P_INT,RAD_REQUIRED},{"global_heads",RAD_P_INT,RAD_REQUIRED},{"head_dim",RAD_P_INT,RAD_REQUIRED},
    {"sequences",RAD_P_INT,RAD_REQUIRED},{"dtype",RAD_P_STR,RAD_REQUIRED},{"ratio",RAD_P_INT,RAD_REQUIRED},
    {"block_size",RAD_P_INT,RAD_DERIVED},{"candidate_block_size",RAD_P_INT,RAD_REQUIRED},
    {"topk",RAD_P_INT,RAD_REQUIRED},{"top_blocks",RAD_P_INT,RAD_REQUIRED},{"mode",RAD_P_STR,RAD_REQUIRED}};
const RadOperandSpec stream_operands[]{
    {"query",RAD_OPD_IN,0},{"cache",RAD_OPD_IN,0},{"head_weights",RAD_OPD_IN,0},
    {"sequence_ids",RAD_OPD_IN,0},{"visible",RAD_OPD_IN,0},{"block_table",RAD_OPD_IN,0},
    {"origins",RAD_OPD_IN,1},{"candidates_in",RAD_OPD_IN,1},{"selected_out",RAD_OPD_OUT,0},{"candidates_out",RAD_OPD_OUT,1}};
const RadParamSpec embedding_params[]{{"M",RAD_P_INT,RAD_REQUIRED,RAD_PROLE_CAPACITY},
    {"N",RAD_P_INT,RAD_REQUIRED},{"K",RAD_P_INT,RAD_REQUIRED},{"format",RAD_P_STR,RAD_REQUIRED},{"y_dtype",RAD_P_STR,RAD_REQUIRED}};
const RadOperandSpec embedding_operands[]{{"token_ids",RAD_OPD_IN,0},{"weight",RAD_OPD_WEIGHT,0},{"output",RAD_OPD_OUT,0}};
const RadParamSpec reorder_params[]{{"M",RAD_P_INT,RAD_REQUIRED,RAD_PROLE_CAPACITY},
    {"tp",RAD_P_INT,RAD_REQUIRED},{"heads",RAD_P_INT,RAD_REQUIRED},{"head_dim",RAD_P_INT,RAD_REQUIRED},{"dtype",RAD_P_STR,RAD_REQUIRED}};
const RadOperandSpec reorder_operands[]{{"query",RAD_OPD_IN,0},{"head_weights",RAD_OPD_IN,0},
    {"query_out",RAD_OPD_OUT,0},{"weights_out",RAD_OPD_OUT,0}};
const RadOperandSpec emit_store_operands[]{{"x",RAD_OPD_IN,0},{"positions",RAD_OPD_IN,0},
    {"slot_mapping",RAD_OPD_IN,0},{"emit",RAD_OPD_IN,0},{"cache",RAD_OPD_INOUT,0},{"status",RAD_OPD_OUT,0}};
const RadParamSpec engram_gate_params[]{{"M",RAD_P_INT,RAD_REQUIRED,RAD_PROLE_CAPACITY},
    {"copies",RAD_P_INT,RAD_REQUIRED},{"n",RAD_P_INT,RAD_REQUIRED},{"dtype",RAD_P_STR,RAD_REQUIRED},
    {"eps",RAD_P_F64,RAD_REQUIRED},{"clamp",RAD_P_F64,RAD_REQUIRED},
    {"token_mask",RAD_P_INT,RAD_REQUIRED},{"gates",RAD_P_INT,RAD_REQUIRED}};
const RadOperandSpec engram_gate_operands[]{{"x",RAD_OPD_IN,0},{"projected_kv",RAD_OPD_IN,0},
    {"q_weight",RAD_OPD_WEIGHT,0},{"k_weight",RAD_OPD_WEIGHT,0},{"token_mask",RAD_OPD_IN,1},
    {"residual",RAD_OPD_OUT,0},{"gates",RAD_OPD_OUT,1}};
const RadOpSchema schemas[]{
    {"ds41_gemm_packed",dense_params,6,dense_operands,3,"Native GGUF weights; f32 dot accumulation; one final output rounding."},
    {"ds41_routed_packed",routed_params,8,routed_operands,6,
     "One output row per routing job, supplied expert/input-row IDs. No mixture weighting or atomic sum. Status must be checked."},
    {"ds41_latent_store",latent_params,6,store_operands,5,"Copy completed shared latents to original-token slots; batch status must be checked before commit."},
    {"ds41_latent_gather",latent_params,6,gather_operands,4,"Copy contiguous shared latent entries from physical first slots; no causal selection is inferred."},
    {"ds41_hc_prepare",hc_prepare_params,7,hc_prepare_operands,7,"Shifted-mHC gates; device payload validation is in scratch status."},
    {"ds41_hc_collapse",hc_collapse_params,4,hc_collapse_operands,3,"Collapse with the supplied carried previous pre-mix."},
    {"ds41_router_topk",router_params,6,router_operands,4,"sqrt-softplus expert routing; correction bias affects IDs only."},
    {"ds41_sparse_attn",sparse_params,7,sparse_operands,5,"Selected shared-latent attention with sink/no value and -1 padding."},
    rad::ds41::compress::schema,
    {"ds41_expert_act",expert_act_params,4,expert_act_operands,4,"F32 asymmetric clamp/SiLU/product/route weight before one down-input cast."},
    {"ds41_moe_combine",combine_params,6,combine_operands,5,"Sum complete rounded expert outputs in expert-ID order, then shared, then one cast; tp_world=1 baseline."},
    {"mhc_post",hc_collapse_params,4,mhc_post_operands,5,"Manifold residual source-to-destination update."},
    rad::ds41::index::schemas[0],rad::ds41::index::schemas[1],rad::ds41::index::schemas[2],rad::ds41::paged_index::schema,
    {"ds41_rmsnorm",ds41_norm_params,5,ds41_norm_operands,3,"Explicit input/output storage boundaries; F32 RMS/gain; no hidden input cast."},
    {"ds41_rope_tail",ds41_rope_params,6,ds41_rope_operands,4,"Supplied F32 position pairs; tail-adjacent rotation/inverse; original positions."},
    {"ds41_sparse_attn_paged",paged_attention_params,11,paged_attention_operands,13,
     "Joint WINDOW/FULL shared-latent attention in original-token coordinates; fixed window, no speculation."},
    {"ds41_route_jobs",route_jobs_params,3,route_jobs_operands,7,
     "Token-major exact route jobs, gate/up token rows and down identity rows; transactional histogram and scalar status."},
    {"ds41_index_stream",stream_params,13,stream_operands,10,
     "Full-global-head scores streamed from FULL paged keys into independent source/consumer/plain top-k heaps; compact logical candidates."},
    {"ds41_embedding",embedding_params,5,embedding_operands,3,
     "Exact native GGUF row lookup, logical [N,K] private storage and one output cast; full-batch token validation."},
    {"ds41_reorder_heads",reorder_params,5,reorder_operands,4,
     "Local exact bit copy from already-gathered rank-major heads to token-major full heads; no communication or scratch."},
    rad::ds41::compress_finish::schema,
    {"ds41_latent_store_emitted",latent_params,6,emit_store_operands,6,
     "Original-row emitted shared-latent publication; masked rows read no coordinates/payload; direct I32 or I64 core slots."},
    {"ds41_engram_gate",engram_gate_params,8,engram_gate_operands,7,
     "F32 signed-root/sigmoid gating and shared-value residual over supplied projections; no table IO or official BF16 intermediate claim."},
    rad::ds41::qdq::schema};
const RadConstraint constraints[]{
    {"M",RAD_C_GE,1,nullptr},{"N",RAD_C_GE,1,nullptr},{"K",RAD_C_GE,32,nullptr},
    {"format",RAD_C_IN,0,"iq2_xxs iq2_xs iq2_s iq3_xxs q5_K q8_0"},
    {"a_dtype",RAD_C_IN,0,"f32 f16 bf16"},{"y_dtype",RAD_C_IN,0,"f32 f16 bf16"}};
int dense_shape(const RadParam* p,int n,int operand,RadOpdDesc* out) {return packed::shape(p,n,operand,out,false);}
int routed_shape(const RadParam* p,int n,int operand,RadOpdDesc* out) {return packed::shape(p,n,operand,out,true);}
#ifdef DS41_HOST_REFERENCE
constexpr int domain=RAD_DOMAIN_HOST;
constexpr const char* plugin_name="libds41_ref",*target="host";
constexpr RadLaunchFn dense_launch=ds41_ref_packed_gemm,routed_launch=ds41_ref_packed_routed;
constexpr RadLaunchFn store_launch=ds41_ref_latent_store,gather_launch=ds41_ref_latent_gather;
#else
constexpr int domain=RAD_DOMAIN_DEVICE;
#ifdef DS41_METADATA_ONLY
constexpr const char* plugin_name="radtest_ds41meta",*target="host";
#else
constexpr const char* plugin_name="libds41",*target="gfx1201";
#endif
constexpr RadLaunchFn dense_launch=ds41_packed_gemm,routed_launch=ds41_packed_routed;
constexpr RadLaunchFn store_launch=ds41_latent_store,gather_launch=ds41_latent_gather;
#endif
RadKernelInfo row(bool routed) {
    RadKernelInfo r{};
    r.name=routed?"ds41_native_routed_baseline":"ds41_native_dense_baseline";
    r.op=routed?"ds41_routed_packed":"ds41_gemm_packed";r.family="gemm";
    r.computes="f32 dot products over exact native GGUF blocks";
    r.shape="K source-block aligned; row-major packed weights; pitched dense activations";
    r.dtypes="IQ2_XXS/XS/S, IQ3_XXS, Q5_K, Q8_0; f32/f16/bf16 activations/output";
    r.domain=domain;r.priority=0;r.constraints=constraints;r.n_constraints=6;
    r.launch=routed?routed_launch:dense_launch;r.layout=packed::layout;r.relayout=packed::relayout;
    r.unrelayout=packed::unrelayout;r.opd_shape=routed?routed_shape:dense_shape;
    return r;
}
RadKernelInfo cache_row(bool store) {
    static const RadConstraint c[]{{"block_size",RAD_C_EQ,4,nullptr}};
    RadKernelInfo r{};r.name=store?"ds41_latent_store_baseline":"ds41_latent_gather_baseline";
    r.op=store?"ds41_latent_store":"ds41_latent_gather";r.family="attn";
    r.computes="bit-exact opaque latent copies in original-token KV coordinates";r.shape="ratio1/2; dimension<=1024; block4";
    r.dtypes="f32 f16 bf16";r.domain=domain;r.constraints=c;r.n_constraints=1;r.launch=store?store_launch:gather_launch;return r;
}
RadKernelInfo math_row(rad::ds41::math::Op op) {
    static const char* names[]{"ds41_hc_prepare","ds41_hc_collapse","ds41_router_topk","ds41_sparse_attn"};
#ifdef DS41_HOST_REFERENCE
    static const RadLaunchFn launch[]{ds41_ref_hc_prepare,ds41_ref_hc_collapse,ds41_ref_router_topk,ds41_ref_sparse_attn};
#else
    static const RadLaunchFn launch[]{ds41_hc_prepare,ds41_hc_collapse,ds41_router_topk,ds41_sparse_attn};
#endif
    RadKernelInfo r{};r.name=names[int(op)];r.op=names[int(op)];r.family="misc";
    r.computes="f32 correctness baseline; scratch validation status must be checked";
    r.shape="hc2..8, router<=4096, sparse topk<=4096/dim<=1024";r.dtypes="f32 f16 bf16";
    r.domain=domain;r.launch=launch[int(op)];r.scratch=domain==RAD_DOMAIN_HOST?nullptr:ds41_math_scratch;return r;
}
RadKernelInfo compress_row() {
    static const RadConstraint c[]{{"n_spec",RAD_C_EQ,0,nullptr},{"dim",RAD_C_GE,1,nullptr},
        {"dim",RAD_C_LE,rad::ds41::compress::max_dimension,nullptr},{"ratio",RAD_C_IN,0,"1 2"}};
    RadKernelInfo r{};r.name="ds41_compress_pool_baseline";r.op="ds41_compress_pool";r.family="attn";
    r.computes="ratio1/2 pool over supplied projections; exact U32 cursor; scratch status must be checked";
    r.shape="non-speculative; original-token row outputs; ratio2 f32 projections";r.dtypes="f32 f16 bf16; U32 state";
    r.domain=domain;r.constraints=c;r.n_constraints=4;r.scratch=domain==RAD_DOMAIN_HOST?nullptr:ds41_compress_scratch;
#ifdef DS41_HOST_REFERENCE
    r.launch=ds41_ref_compress_pool;
#else
    r.launch=ds41_compress_pool;
#endif
    return r;
}
RadKernelInfo ffn_row(rad::ds41::ffn::Op op) {
    static const char* names[]{"ds41_expert_act","ds41_moe_combine","mhc_post"};
    static const RadConstraint complete[]{{"tp_world",RAD_C_EQ,1,nullptr}};
    RadKernelInfo r{};r.name=names[int(op)];r.op=names[int(op)];r.family="misc";
    r.computes="DS41 clamp/weight/rounding and deterministic expert/residual sum baseline; scratch flag must be checked";
    r.shape="complete local expert outputs only; combine tp_world=1";r.dtypes="f32 f16 bf16";r.domain=domain;r.scratch=domain==RAD_DOMAIN_HOST?nullptr:ds41_ffn_scratch;
    if(op==rad::ds41::ffn::Op::moe_combine){r.constraints=complete;r.n_constraints=1;}
#ifdef DS41_HOST_REFERENCE
    static const RadLaunchFn launch[]{ds41_ref_expert_act,ds41_ref_moe_combine,ds41_ref_mhc_post};r.launch=launch[int(op)];
#else
    static const RadLaunchFn launch[]{ds41_expert_act,ds41_moe_combine,ds41_mhc_post};r.launch=launch[int(op)];
#endif
    return r;
}
RadKernelInfo index_row(rad::ds41::index::Op op) {
    static const char* names[]{"ds41_index_scores","ds41_candidate_blocks","ds41_index_topk"};
    static const RadConstraint bounds[]{{"M",RAD_C_GE,1,nullptr},{"M",RAD_C_LE,rad::ds41::index::max_rows,nullptr},
        {"key_rows",RAD_C_GE,1,nullptr},{"key_rows",RAD_C_LE,rad::ds41::index::max_keys,nullptr}};
    RadKernelInfo r{};r.name=names[int(op)];r.op=names[int(op)];r.family="attn";
    r.computes="floating index scores or deterministic visible selection; scratch flag must be checked";
    r.shape="bounded float index baseline; external TP score reduction";r.dtypes="f32 f16 bf16; F32 scores";
    r.domain=domain;r.constraints=bounds;r.n_constraints=4;r.scratch=domain==RAD_DOMAIN_HOST?nullptr:ds41_index_scratch;
#ifdef DS41_HOST_REFERENCE
    static const RadLaunchFn launch[]{ds41_ref_index_scores,ds41_ref_candidate_blocks,ds41_ref_index_topk};r.launch=launch[int(op)];
#else
    static const RadLaunchFn launch[]{ds41_index_scores,ds41_candidate_blocks,ds41_index_topk};r.launch=launch[int(op)];
#endif
    return r;
}
RadKernelInfo paged_index_row() {
    static const RadConstraint c[]{{"block_size",RAD_C_EQ,4,nullptr}};
    RadKernelInfo r{};r.name="ds41_index_scores_paged";r.op=r.name;r.family="attn";
    r.computes="floating shared-key index scores directly from FULL paged cache in original-token coordinates";
    r.shape="ratio1/2, core block4, checked visible pages; external TP score reduction";r.dtypes="f32 f16 bf16";
    r.domain=domain;r.constraints=c;r.n_constraints=1;r.scratch=domain==RAD_DOMAIN_HOST?nullptr:ds41_index_paged_scratch;
#ifdef DS41_HOST_REFERENCE
    r.launch=ds41_ref_index_scores_paged;
#else
    r.launch=ds41_index_scores_paged;
#endif
    return r;
}
RadKernelInfo transform_row(rad::ds41::transform::Op op) {
    static const char* names[]{"ds41_rmsnorm","ds41_rope_tail"};
    RadKernelInfo r{};r.name=names[int(op)];r.op=r.name;r.family="norm";
    r.computes="explicit scalar storage boundaries; F32 norm or supplied-pair tail rotation; scratch flag required";
    r.shape="positive pitched views, bounded rows and positions";r.dtypes="f32 f16 bf16";r.domain=domain;r.scratch=domain==RAD_DOMAIN_HOST?nullptr:ds41_transform_scratch;
#ifdef DS41_HOST_REFERENCE
    static const RadLaunchFn fn[]{ds41_ref_rmsnorm,ds41_ref_rope_tail};r.launch=fn[int(op)];
#else
    static const RadLaunchFn fn[]{ds41_rmsnorm,ds41_rope_tail};r.launch=fn[int(op)];
#endif
    return r;
}
RadKernelInfo paged_attention_row() {
    static const RadConstraint c[]{{"block_size",RAD_C_EQ,4,nullptr},{"n_spec",RAD_C_EQ,0,nullptr},
        {"ratio",RAD_C_IN,0,"0 1 2"}};
    RadKernelInfo r{};r.name="ds41_sparse_attn_paged";r.op=r.name;r.family="attn";
    r.computes="F32 joint shared-latent window/compressed normalization; deferred scalar status required";
    r.shape="ordinary fixed WINDOW plus FULL compressed cache; block4; non-speculative";
    r.dtypes="f32 f16 bf16";r.domain=domain;r.constraints=c;r.n_constraints=3;r.scratch=domain==RAD_DOMAIN_HOST?nullptr:ds41_attention_paged_scratch;
#ifdef DS41_HOST_REFERENCE
    r.launch=ds41_ref_sparse_attn_paged;
#else
    r.launch=ds41_sparse_attn_paged;
#endif
    return r;
}
RadKernelInfo route_jobs_row() {
    RadKernelInfo r{};r.name="ds41_route_jobs";r.op=r.name;r.family="moe";
    r.computes="exact token-major IDs/weights, token and identity row maps, integer expert histogram";
    r.shape="unique IDs within each token; whole-batch preflight; scratch flag must be checked";
    r.dtypes="I32 IDs/counts, F32 weights";r.domain=domain;r.scratch=domain==RAD_DOMAIN_HOST?nullptr:ds41_route_jobs_scratch;
#ifdef DS41_HOST_REFERENCE
    r.launch=ds41_ref_route_jobs;
#else
    r.launch=ds41_route_jobs;
#endif
    return r;
}
RadKernelInfo stream_row() {
    static const RadConstraint c[]{{"block_size",RAD_C_EQ,4,nullptr},{"topk",RAD_C_GE,1,nullptr},
        {"topk",RAD_C_LE,rad::ds41::index_stream::max_topk,nullptr},{"top_blocks",RAD_C_GE,0,nullptr},
        {"top_blocks",RAD_C_LE,rad::ds41::index_stream::max_top_blocks,nullptr},{"mode",RAD_C_IN,0,"source consumer plain"}};
    RadKernelInfo r{};r.name="ds41_index_stream";r.op=r.name;r.family="attn";
    r.computes="F32 full-head scores; bounded exact selection heaps, no dense score/mask plane";
    r.shape="FULL paged cache; heads==global_heads; compact candidates; source own top-k reads all visible keys";
    r.dtypes="f32 f16 bf16; I32 selections";r.domain=domain;r.constraints=c;r.n_constraints=6;
    r.scratch=domain==RAD_DOMAIN_HOST?nullptr:ds41_index_stream_scratch;
#ifdef DS41_HOST_REFERENCE
    r.launch=ds41_ref_index_stream;
#else
    r.launch=ds41_index_stream;
#endif
    return r;
}
RadKernelInfo embedding_row() {
    static const RadConstraint c[]{{"N",RAD_C_GE,1,nullptr},{"K",RAD_C_GE,32,nullptr},
        {"format",RAD_C_IN,0,"iq2_xxs iq2_xs iq2_s iq3_xxs q5_K q8_0"},{"y_dtype",RAD_C_IN,0,"f32 f16 bf16"}};
    RadKernelInfo r{};r.name="ds41_embedding_native";r.op="ds41_embedding";r.family="embed";
    r.computes="Exact native GGUF value decode, one final output cast; whole-batch IDs preflight";
    r.shape="Full local vocabulary [N,K], logical private DT, pitched IDs/output";r.dtypes="native six GGUF types -> f32 f16 bf16";
    r.domain=domain;r.constraints=c;r.n_constraints=4;r.layout=rad::ds41::embedding::layout;
    r.relayout=rad::ds41::embedding::relayout;r.unrelayout=rad::ds41::embedding::unrelayout;r.opd_shape=rad::ds41::embedding::shape;
    r.scratch=domain==RAD_DOMAIN_HOST?nullptr:ds41_embedding_scratch;
#ifdef DS41_HOST_REFERENCE
    r.launch=ds41_ref_embedding;
#else
    r.launch=ds41_embedding;
#endif
    return r;
}
RadKernelInfo reorder_row() {
    RadKernelInfo r{};r.name="ds41_reorder_heads";r.op=r.name;r.family="copy";
    r.computes="Bit-preserving local transpose of already-gathered query heads and F32 head weights";
    r.shape="packed rank4/rank3 or flattened rank3/rank2 inputs; no collective/status/scratch";
    r.dtypes="f32 f16 bf16 query; F32 head weights";r.domain=domain;r.opd_shape=rad::ds41::reorder_heads::shape;
#ifdef DS41_HOST_REFERENCE
    r.launch=ds41_ref_reorder_heads;
#else
    r.launch=ds41_reorder_heads;
#endif
    return r;
}
RadKernelInfo compress_finish_row() {
    static const RadConstraint c[]{{"n",RAD_C_GE,1,nullptr},{"n",RAD_C_LE,rad::ds41::compress_finish::max_dimension,nullptr},
        {"ratio",RAD_C_IN,0,"1 2"}};
    RadKernelInfo r{};r.name="ds41_compress_finish";r.op=r.name;r.family="attn";
    r.computes="Mask -> activation cast -> RMS/gain -> activation cast -> original-position tail RoPE -> cast";
    r.shape="original token rows, inactive +0 outputs; supplied F32 pairs; floating topology only";
    r.dtypes="f32 f16 bf16";r.domain=domain;r.constraints=c;r.n_constraints=3;
    r.opd_shape=rad::ds41::compress_finish::shape;r.scratch=domain==RAD_DOMAIN_HOST?nullptr:ds41_compress_finish_scratch;
#ifdef DS41_HOST_REFERENCE
    r.launch=ds41_ref_compress_finish;
#else
    r.launch=ds41_compress_finish;
#endif
    return r;
}
RadKernelInfo emit_store_row() {
    static const RadConstraint c[]{{"M",RAD_C_GE,1,nullptr},{"M",RAD_C_LE,rad::ds41::latent_emit::max_rows,nullptr},
        {"dimension",RAD_C_GE,1,nullptr},{"dimension",RAD_C_LE,1024,nullptr},
        {"ratio",RAD_C_IN,0,"1 2"},{"block_size",RAD_C_EQ,4,nullptr}};
    RadKernelInfo r{};r.name="ds41_latent_store_emitted";r.op=r.name;r.family="copy";
    r.computes="Bit-preserving emitted latent copy; all masks/coordinates/destination uniqueness checked before cache writes";
    r.shape="opaque block4 original-token cache, I32/I64 positions and final slots";
    r.dtypes="f32 f16 bf16";r.domain=domain;r.constraints=c;r.n_constraints=6;
    r.scratch=domain==RAD_DOMAIN_HOST?nullptr:ds41_latent_emit_scratch;
#ifdef DS41_HOST_REFERENCE
    r.launch=ds41_ref_latent_store_emitted;
#else
    r.launch=ds41_latent_store_emitted;
#endif
    return r;
}
RadKernelInfo engram_gate_row() {
    static const RadConstraint c[]{{"copies",RAD_C_GE,1,nullptr},{"copies",RAD_C_LE,rad::ds41::engram_gate::max_copies,nullptr},
        {"n",RAD_C_GE,1,nullptr},{"n",RAD_C_LE,rad::ds41::engram_gate::max_dimension,nullptr}};
    RadKernelInfo r{};r.name="ds41_engram_gate";r.op=r.name;r.family="misc";
    r.computes="F32 RMS/dot, signed clamped root/sigmoid and shared-value residual; whole-batch finite checks";
    r.shape="x[T,copies,D], supplied KV[T,copies+1,D], F32 norm weights, fixed optional mask/gate slots";
    r.dtypes="f32 f16 bf16";r.domain=domain;r.constraints=c;r.n_constraints=4;
    r.scratch=domain==RAD_DOMAIN_HOST?nullptr:ds41_engram_gate_scratch;
#ifdef DS41_HOST_REFERENCE
    r.launch=ds41_ref_engram_gate;
#else
    r.launch=ds41_engram_gate;
#endif
    return r;
}
RadKernelInfo qdq_row() {
    static const RadConstraint c[]{{"M",RAD_C_GE,1,nullptr},{"M",RAD_C_LE,rad::ds41::qdq::max_rows,nullptr},
        {"n",RAD_C_GE,16,nullptr},{"n",RAD_C_LE,rad::ds41::qdq::max_dimension,nullptr},
        {"profile",RAD_C_IN,0,"window_fp8 compressed_fp4 index_fp4"},{"masked",RAD_C_IN,0,"0 1"}};
    RadKernelInfo r{};r.name="ds41_qdq_candidate";r.op=r.name;r.family="quant";
    r.computes="Explicit BF16 mathematical cache QDQ candidate; independent bit RNE, exact absmax and two-phase payload preflight";
    r.shape="out-of-place, fixed group16/32, optional original-row emit mask; no packed cache or official TileLang parity claim";
    r.dtypes="bf16";r.domain=domain;r.constraints=c;r.n_constraints=6;
    r.scratch=domain==RAD_DOMAIN_HOST?nullptr:ds41_qdq_scratch;
#ifdef DS41_HOST_REFERENCE
    r.launch=ds41_ref_qdq_candidate;
#else
    r.launch=ds41_qdq_candidate;
#endif
    return r;
}
const RadKernelInfo rows[]{row(false),row(true),cache_row(true),cache_row(false),
    math_row(rad::ds41::math::Op::hc_prepare),math_row(rad::ds41::math::Op::hc_collapse),
    math_row(rad::ds41::math::Op::router_topk),math_row(rad::ds41::math::Op::sparse_attn),compress_row(),
    ffn_row(rad::ds41::ffn::Op::expert_act),ffn_row(rad::ds41::ffn::Op::moe_combine),ffn_row(rad::ds41::ffn::Op::mhc_post),
    index_row(rad::ds41::index::Op::scores),index_row(rad::ds41::index::Op::candidates),index_row(rad::ds41::index::Op::topk),paged_index_row(),
    transform_row(rad::ds41::transform::Op::rmsnorm),transform_row(rad::ds41::transform::Op::rope_tail),paged_attention_row(),route_jobs_row(),stream_row(),embedding_row(),reorder_row(),
    compress_finish_row(),emit_store_row(),engram_gate_row(),qdq_row()};
constexpr int row_count=27;
static_assert(sizeof(rows)/sizeof(rows[0])==row_count&&sizeof(schemas)/sizeof(schemas[0])==row_count);
const RadPluginInfo info{RAD_PLUGIN_KERNEL,plugin_name,"0.2","Experimental DS41 baseline; full-model serving unvalidated",target};
}
extern "C" {
uint32_t rad_plugin_abi_version(void) {return RAD_ABI_VERSION;}
const RadPluginInfo* rad_plugin_info(void) {return &info;}
int rad_kernel_schema_count(void) {return row_count;}
const RadOpSchema* rad_kernel_schema_at(int i) {return i>=0&&i<row_count?&schemas[i]:nullptr;}
int rad_kernel_count(void) {return row_count;}
const RadKernelInfo* rad_kernel_at(int i) {return i>=0&&i<row_count?&rows[i]:nullptr;}
#if !defined(DS41_HOST_REFERENCE) && !defined(DS41_METADATA_ONLY)
int rad_plugin_open(void) {
    if(!rad_core_deferred_status_version||rad_core_deferred_status_version()!=RAD_DEFERRED_STATUS_VERSION){
        std::fprintf(stderr,"libds41 requires a core with deferred device-status collection version %u\n",RAD_DEFERRED_STATUS_VERSION);
        return RAD_E_ABI;
    }
    return RAD_OK;
}
uint32_t rad_plugin_deferred_status_version(void) {return RAD_DEFERRED_STATUS_VERSION;}
RadDeferredStatusLaunchFn rad_kernel_deferred_status(int i) {
    if(i<0||i>=row_count)return nullptr;
    const char* op=rows[i].op;
    if(!std::strcmp(op,"ds41_gemm_packed"))return nullptr;
    if(!std::strcmp(op,"ds41_routed_packed"))return ds41_collect_routed_status;
    if(!std::strcmp(op,"ds41_latent_store"))return ds41_collect_store_status;
    if(!std::strcmp(op,"ds41_latent_gather"))return ds41_collect_gather_status;
    if(!std::strcmp(op,"ds41_latent_store_emitted"))return ds41_collect_routed_status;
    static const char* scalar_status_ops[]{"ds41_hc_prepare","ds41_hc_collapse","ds41_router_topk","ds41_sparse_attn",
        "ds41_compress_pool","ds41_expert_act","ds41_moe_combine","mhc_post","ds41_index_scores",
        "ds41_candidate_blocks","ds41_index_topk","ds41_index_scores_paged","ds41_rmsnorm","ds41_rope_tail","ds41_sparse_attn_paged","ds41_route_jobs","ds41_index_stream","ds41_embedding","ds41_compress_finish","ds41_engram_gate","ds41_qdq_candidate"};
    for(const char* name:scalar_status_ops)if(!std::strcmp(op,name))return ds41_collect_scratch_status;
    return nullptr;
}
#endif
#ifdef DS41_HOST_REFERENCE
int rad_plugin_reference(void) {return 1;}
#endif
}
