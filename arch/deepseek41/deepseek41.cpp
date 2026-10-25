/* deepseek41 -- DeepSeek-V4.1 (GGUF `general.architecture: deepseek41`, the layout llama.cpp's
 * converter writes): `hc` hyper-connection residual streams over latent attention layers that
 * attend a sliding window of raw keys plus, past the first layers, a two-level sparse top-k of
 * compressed keys; sqrt(softplus)-routed MoE with one shared expert; Engram n-gram lookups written
 * into the streams at a few layers. Every dimension comes from the checkpoint's metadata, so the
 * same graph serves the released model and any reduced one (tiny/ds41-tiny.gguf in the port).
 *
 * The numerics follow the official inference/model.py (DeepSeek's release): the window keys, the
 * compressed keys and the index keys are quantised and dequantised exactly where the reference
 * does it (quant_roundtrip), so the bf16 caches hold the reference's values.
 *
 * Per layer, in issue order (docs: the port's PLAN.md, "Phase 2 plumbing design"):
 *   engram (at ple layers)   token_map -> ngram_ids -> embed_lookup -> gemm -> stream_gate_add
 *   attention connection     mhc_mix (this layer's gates) ; mhc_collapse (the previous gates)
 *   attention                q path, window key into the window cache, the compressor at a
 *                            kv-source layer (qsa_work / softmax_pool, or one entry a token at
 *                            ratio 1), the indexer at an index-source layer, attn_sparse_sink,
 *                            the inverse rotation, the grouped output projection ; mhc_post
 *   ffn connection           mhc_mix ; mhc_collapse ; MoeFP8 (norm, router, experts) ; mhc_post
 * then mhc_collapse with the last gates, the final norm and the head.
 *
 * This first form is one rank and bf16 linears, without the MTP head; the rest is refused by name. */
#include <arch/rad_block_moe_fp8.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace deepseek41 {
using namespace rad::arch;

/* RAD_ARCH_TRY that names the call that failed: a refused declaration otherwise surfaces only as
 * the plugin's return code. */
#define DS_TRY(e)                                                                                \
    do {                                                                                         \
        const int rc_ = (e);                                                                     \
        if (rc_ < 0) {                                                                           \
            fprintf(stderr, "radiance: deepseek41: %s refused (%d)\n", #e, rc_);                  \
            return rc_;                                                                          \
        }                                                                                        \
    } while (0)

/* ------------------------------------------------------------------ the configuration */
struct Cfg {
    int64_t n_layer = 0, n_embd = 0, n_head = 0, head_dim = 0, rope = 0, q_lora = 0;
    int64_t o_groups = 0, o_rank = 0, window = 0;
    int64_t n_expert = 0, top_k = 0, n_ff = 0, n_shared = 0;
    double  route_scale = 1.0, swiglu_limit = 0.0;
    int     norm_topk = 1;
    int64_t ih = 0, idim = 0, itopk = 0, cand = -1, cand_blocks = 0, cand_block = 0;
    int64_t hc = 0, hc_iters = 0;
    double  hc_eps = 0, eps = 0;
    double  theta = 0, comp_theta = 0, yarn_factor = 0, yarn_low = 0, yarn_high = 0;
    std::vector<int64_t> ratio, kv_src, idx_src, ple;
    int64_t ngram = 0, ple_heads = 0, ple_dim = 0, ple_rows = 0, pad = 0;
    int64_t n_vocab = 0;
    /* derived, per layer */
    std::vector<int> src_of, idx_of;
    bool is_src(int l) const { for (int64_t s : kv_src) if (s == l) return true; return false; }
    bool is_idx(int l) const { for (int64_t s : idx_src) if (s == l) return true; return false; }
    int  ple_index(int l) const {
        for (size_t i = 0; i < ple.size(); ++i) if (ple[i] == l) return (int)i;
        return -1;
    }
};

static std::string key(const char* k) { return std::string("deepseek41.") + k; }
static int64_t gi(const RadModelMeta* m, const char* k, int64_t d) {
    return rad_meta_geti(m, key(k).c_str(), d);
}
static double gf(const RadModelMeta* m, const char* k, double d) {
    return rad_meta_getf(m, key(k).c_str(), d);
}
/* A spelled-out integer array (core/format/checkpoint.cpp, meta_from_gguf). */
static std::vector<int64_t> gl(const RadModelMeta* m, const char* k) {
    std::vector<int64_t> v;
    const char* s = rad_meta_gets(m, key(k).c_str(), "");
    if (!s || !*s || *s == '[') return v;
    while (*s) {
        char* e = nullptr;
        v.push_back(std::strtoll(s, &e, 10));
        if (e == s) return {};
        s = *e == ',' ? e + 1 : e;
    }
    return v;
}

static int read_cfg(Cfg& c, const RadModelMeta* m) {
    c.n_layer = m->n_layers;
    c.n_embd = m->n_embd;
    c.n_head = m->n_head;
    c.head_dim = m->head_dim;
    c.n_vocab = m->n_vocab;
    c.rope = gi(m, "rope.dimension_count", 0);
    c.q_lora = gi(m, "attention.q_lora_rank", 0);
    c.o_groups = gi(m, "attention.output_group_count", 0);
    c.o_rank = gi(m, "attention.output_lora_rank", 0);
    c.window = gi(m, "attention.sliding_window", 0);
    c.n_expert = m->n_expert;
    c.top_k = m->n_expert_used;
    c.n_ff = gi(m, "expert_feed_forward_length", 0);
    c.n_shared = m->n_expert_shared;
    c.route_scale = gf(m, "expert_weights_scale", 1.0);
    c.norm_topk = (int)gi(m, "expert_weights_norm", 1);
    {
        /* One clamp for every layer and arm; a per-layer schedule is refused. */
        const std::vector<int64_t> dummy;
        const char* s = rad_meta_gets(m, key("swiglu_clamp_exp").c_str(), "");
        c.swiglu_limit = s && *s && *s != '[' ? std::atof(s) : 10.0;
    }
    c.ih = gi(m, "attention.indexer.head_count", 0);
    c.idim = gi(m, "attention.indexer.key_length", 0);
    c.itopk = gi(m, "attention.indexer.top_k", 0);
    c.cand = gi(m, "attention.indexer.candidate_source_layer", -1);
    c.cand_blocks = gi(m, "attention.indexer.candidate_top_k_blocks", 0);
    c.cand_block = gi(m, "attention.indexer.candidate_block_size", 0);
    c.hc = gi(m, "hyper_connection.count", 0);
    c.hc_iters = gi(m, "hyper_connection.sinkhorn_iterations", 0);
    c.hc_eps = gf(m, "hyper_connection.epsilon", 0.0);
    c.eps = m->rms_eps;
    c.theta = m->rope_theta;
    c.comp_theta = gf(m, "attention.compress_rope_freq_base", 0.0);
    c.ratio = gl(m, "attention.compress_ratios");
    c.kv_src = gl(m, "attention.kv_source_layers");
    c.idx_src = gl(m, "attention.indexer.source_layers");
    c.ple = gl(m, "ple.layers");
    c.ngram = gi(m, "ple.ngram_size", 0);
    c.ple_heads = (c.ngram - 1) * gi(m, "ple.heads_per_ngram", 0);
    c.ple_dim = gi(m, "embedding_length_per_layer_input", 0);
    c.pad = gi(m, "engram.pad_token_id", -1);
    for (int64_t v : gl(m, "ple.head_vocab_sizes")) c.ple_rows += v;
    /* The compressed layers' rotary: YaRN with the bounds the reference derives from the
     * training length (model.py, precompute_freqs_cis). */
    c.yarn_factor = rad_meta_getf(m, "rope_scaling.factor", 0.0);
    const double orig = rad_meta_getf(m, "rope_scaling.original_max_position_embeddings", 0.0);
    const double bf = rad_meta_getf(m, "rope_scaling.beta_fast", 32.0);
    const double bs = rad_meta_getf(m, "rope_scaling.beta_slow", 1.0);
    if (c.yarn_factor > 1.0 && orig > 0 && c.comp_theta > 0 && c.rope > 0) {
        auto corr = [&](double r) {
            return (double)c.rope * std::log(orig / (r * 2.0 * M_PI)) / (2.0 * std::log(c.comp_theta));
        };
        c.yarn_low = std::max(std::floor(corr(bf)), 0.0);
        c.yarn_high = std::min(std::ceil(corr(bs)), (double)c.rope - 1);
    }

    const struct { const char* what; int64_t v; } need[] = {
        { "block_count", c.n_layer }, { "embedding_length", c.n_embd },
        { "attention.head_count", c.n_head }, { "attention.key_length", c.head_dim },
        { "rope.dimension_count", c.rope }, { "attention.q_lora_rank", c.q_lora },
        { "attention.output_group_count", c.o_groups }, { "attention.output_lora_rank", c.o_rank },
        { "attention.sliding_window", c.window }, { "expert_count", c.n_expert },
        { "expert_used_count", c.top_k }, { "expert_feed_forward_length", c.n_ff },
        { "hyper_connection.count", c.hc }, { "hyper_connection.sinkhorn_iterations", c.hc_iters },
        { "attention.indexer.head_count", c.ih }, { "attention.indexer.key_length", c.idim },
        { "attention.indexer.top_k", c.itopk } };
    for (const auto& n : need)
        if (n.v <= 0) {
            fprintf(stderr, "radiance: deepseek41: the checkpoint has no deepseek41.%s\n", n.what);
            return RAD_E_UNSUPPORTED;
        }
    if ((int64_t)c.ratio.size() < c.n_layer || c.kv_src.empty() || c.idx_src.empty()) {
        fprintf(stderr, "radiance: deepseek41: compress_ratios (%zu values for %lld layers), "
                        "kv_source_layers or indexer.source_layers is missing\n",
                c.ratio.size(), (long long)c.n_layer);
        return RAD_E_UNSUPPORTED;
    }
    if (c.n_shared != 1 || c.head_dim % 2 || c.rope > c.head_dim || c.rope > c.idim ||
        (c.n_head * c.head_dim) % c.o_groups || c.n_expert % 2 || c.hc > 8) {
        fprintf(stderr, "radiance: deepseek41: unsupported geometry (shared experts %lld, head_dim "
                        "%lld, rope %lld, groups %lld, experts %lld, hc %lld)\n",
                (long long)c.n_shared, (long long)c.head_dim, (long long)c.rope,
                (long long)c.o_groups, (long long)c.n_expert, (long long)c.hc);
        return RAD_E_UNSUPPORTED;
    }
    /* Each compressing layer reads the latest kv source and index source at or below it; their
     * ratios must agree. */
    c.src_of.assign((size_t)c.n_layer, -1);
    c.idx_of.assign((size_t)c.n_layer, -1);
    int s = -1, x = -1;
    for (int l = 0; l < c.n_layer; ++l) {
        if (c.is_src(l)) s = l;
        if (c.is_idx(l)) x = l;
        const int64_t r = c.ratio[(size_t)l];
        if (r <= 0) continue;
        if (s < 0 || x < 0 || c.ratio[(size_t)s] != r || c.ratio[(size_t)x] != r ||
            c.head_dim % (2 * r) || c.idim % (2 * r)) {
            fprintf(stderr, "radiance: deepseek41: layer %d (ratio %lld) has no kv or index source "
                            "of its ratio\n", l, (long long)r);
            return RAD_E_UNSUPPORTED;
        }
        c.src_of[(size_t)l] = s;
        c.idx_of[(size_t)l] = x;
    }
    if (!c.ple.empty() && (c.ngram < 2 || c.ple_heads <= 0 || c.ple_dim <= 0 || c.pad < 0 ||
                           c.ple_rows <= 0)) {
        fprintf(stderr, "radiance: deepseek41: the Engram layout is incomplete\n");
        return RAD_E_UNSUPPORTED;
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ the model */
struct Layer {
    int l = 0;
    int64_t r = 0;            /* compress ratio, 0 for window only */
    bool src = false, idx = false;
    /* weights */
    rad_weight hc_a_fn = 0, hc_a_scale = 0, hc_a_base = 0, hc_f_fn = 0, hc_f_scale = 0,
               hc_f_base = 0;
    rad_weight attn_norm = 0, wq_a = 0, q_norm = 0, wq_b = 0, wkv = 0, kv_norm = 0, sinks = 0,
               wo_b = 0;
    std::vector<rad_weight> wo_a;
    rad_weight comp_w = 0, comp_norm = 0, ik_w = 0, ik_norm = 0, iq_w = 0, iproj = 0;
    rad_weight e_mult = 0, e_vsz = 0, e_off = 0, e_kv = 0, e_gate = 0;
    /* ops, in issue order */
    rad_op e_ids = 0, e_emb = 0, e_gemm = 0, e_gate_op = 0, e_back = 0;
    rad_op mix_a = 0, col_a = 0, norm_a = 0;
    rad_op q_a = 0, q_an = 0, q_b = 0, q_rope = 0;
    rad_op kv_g = 0, kv_n = 0, kv_rope = 0, kv_q = 0, kv_st = 0;
    rad_op c_gemm = 0, c_work = 0, c_pool = 0, c_norm = 0, c_tab = 0;
    rad_op k_gemm = 0, k_norm = 0, k_rope = 0, k_q = 0, k_st = 0;
    rad_op l_rope = 0, l_q = 0, l_st = 0, c_tail = 0;
    rad_op i_q = 0, i_rope = 0, i_qq = 0, i_w = 0, i_score = 0, i_cand = 0, i_topk = 0;
    rad_op attn = 0, o_rope = 0, o_q = 0, o_b = 0, post_a = 0, attn_fill = 0, ar_attn = 0;
    std::vector<rad_op> o_a;
    rad_weight ffn_norm = 0;
    rad_op mix_f = 0, col_f = 0, norm_f = 0, post_f = 0;
    MoeFP8 moe{};
};

struct Model {
    Names nm{};
    Geom g{};
    Cfg c{};
    std::vector<Layer> layers;
    int64_t work = 0, cmax = 0;
    /* The attention world (geometry()): attention ranks [0, aw), this rank's index `ar` in it and
     * its `gr` output groups; `zero` for an attention-zero rank. */
    int aw = 1, ar = 0;
    int64_t gr = 0;
    bool zero = false;
    /* buffers */
    rad_buf h0 = 0, h1 = 0, xcol = 0, xn = 0, pre_a = 0, pre_f = 0, post = 0, comb = 0;
    rad_buf qr = 0, qrn = 0, q = 0, kv = 0, kvn = 0, o = 0, oa = 0, attn_out = 0;
    rad_buf kvs = 0, stage = 0, page = 0, bpos = 0, nc = 0, pooled = 0, lat = 0, latn = 0,
            ik = 0, ikn = 0;
    rad_buf iq = 0, iw = 0, score = 0, keep = 0, sel = 0, cs_a = 0, cs_b = 0;
    rad_buf ctok = 0, ids = 0, emb = 0, ekv = 0;
    rad_buf final_hidden = 0, logit_input = 0, logits = 0;
    ActFP8 act{};
    MoeFP8::Wire mw{};
    /* KV groups */
    rad_kvgroup win = 0, tok = 0;
    std::map<int64_t, rad_kvgroup> comp, index, tail;
    /* The GGUF type of each linear weight the model ships in llama.cpp's blocks (lw()). */
    std::map<rad_weight, const char*> wty;
    int64_t win_bs = 0;
    /* model-wide weights and ops */
    rad_weight token = 0, head = 0, final_gain = 0, tmap = 0, ple_table = 0;
    rad_op embed = 0, expand = 0, pre0 = 0, pre1 = 0, rope_a = 0, rope_b_step = 0, tok_map = 0;
    rad_op contract = 0, fnorm = 0, gather = 0, head_op = 0, embed_ar = 0;
};
static std::unique_ptr<Model> models[MAX_RANKS];

static rad_op op(RadBuilder* b, const char* name, std::vector<RadParam> ps,
                 std::vector<rad_weight> ws, std::initializer_list<rad_buf> rd,
                 std::initializer_list<rad_buf> wr) {
    const rad_op h = rad_decl_op(b, name, ps.data(), (int)ps.size(),
                                 ws.empty() ? nullptr : ws.data(), (int)ws.size());
    return rw(b, h, rd, wr);
}
static rad_weight wt(RadBuilder* b, const char* name, uint32_t dt,
                     std::initializer_list<int64_t> shape, RadWeightGroup grp,
                     int access = RAD_ACCESS_PER_TOKEN) {
    if (map_copy(b, name, name) != RAD_OK) return 0;
    return decl_w(b, name, dt, shape, access, RAD_SHARD_NONE, grp);
}
/* A weight split over the ranks: this rank's [lo, hi) of its split dimension (rows for
 * RAD_SHARD_ROW, columns for RAD_SHARD_COL), declared at `shape`, which must already be that
 * extent. An explicit span because the attention world can be smaller than the world. */
static rad_weight wts(RadBuilder* b, const Geom& g, const char* name, uint32_t dt,
                      std::initializer_list<int64_t> shape, RadWeightGroup grp, int shard, int64_t lo,
                      int64_t hi, int access = RAD_ACCESS_PER_TOKEN) {
    if (map_copy(b, name, name) != RAD_OK) return 0;
    const rad_weight w = decl_w(b, name, dt, shape, access, g.world > 1 ? shard : RAD_SHARD_NONE, grp);
    if (!w) {
        fprintf(stderr, "radiance: deepseek41: weight %s refused\n", name);
        return 0;
    }
    if (g.world > 1 && rad_weight_shard_span(b, w, lo, hi) != RAD_OK) {
        fprintf(stderr, "radiance: deepseek41: %s's span [%lld, %lld) refused\n", name, (long long)lo,
                (long long)hi);
        return 0;
    }
    return w;
}
/* A linear's weight, [n, k] at this rank's extents: llama.cpp's blocks when the model ships it
 * so (a GGUF weight, imported as its type's planes; rad_ggml.h), declared at the codes' dtype and
 * read by gemm_nt_ggml, else bf16. A block's values run along k, so k and a column span must cut
 * whole blocks. With `shard` and a world above one, this rank holds [lo, hi) of the split
 * dimension. */
static rad_weight lw(RadBuilder* b, Model& m, const char* name, int64_t n, int64_t k,
                     RadWeightGroup grp, int shard = RAD_SHARD_NONE, int64_t lo = 0, int64_t hi = 0,
                     bool mapped = false) {
    if (!mapped && map_copy(b, name, name) != RAD_OK) return 0;
    RadEncoding e{};
    int64_t blk = 0;
    const char* ty = weight_enc(b, name, &e) ? ggml_type_of(e, &blk) : nullptr;
    const bool split = m.g.world > 1 && shard != RAD_SHARD_NONE;
    if (ty && (k % blk || (split && shard == RAD_SHARD_COL && (lo % blk || hi % blk)))) {
        fprintf(stderr, "radiance: deepseek41: %s is GGUF %s, whose blocks cover %lld of K; this rank "
                        "reads %lld columns%s\n", name, ty, (long long)blk, (long long)k,
                split && shard == RAD_SHARD_COL ? " of a slice" : "");
        return 0;
    }
    const rad_weight w = decl_w(b, name, ty ? ggml_codes_dtype(e) : (uint32_t)RAD_BF16, { n, k },
                                RAD_ACCESS_PER_TOKEN, split ? shard : RAD_SHARD_NONE, grp);
    if (!w) {
        fprintf(stderr, "radiance: deepseek41: weight %s refused\n", name);
        return 0;
    }
    if (split && rad_weight_shard_span(b, w, lo, hi) != RAD_OK) {
        fprintf(stderr, "radiance: deepseek41: %s's span [%lld, %lld) refused\n", name, (long long)lo,
                (long long)hi);
        return 0;
    }
    if (ty) m.wty[w] = ty;
    return w;
}
/* y = x . w^T over lw()'s weight: gemm_nt_ggml for a GGUF block weight, gemm_nt for bf16. */
static rad_op linear(RadBuilder* b, const Model& m, int64_t rows, rad_weight w, int64_t n,
                     int64_t k, rad_buf x, rad_buf y) {
    const auto t = m.wty.find(w);
    if (t != m.wty.end())
        return op(b, "gemm_nt_ggml",
                  { RAD_RANGE("M", 1, rows), RAD_INT("N", n), RAD_INT("K", k), RAD_STR("type", t->second),
                    RAD_STR("dtype", m.g.dtype) },
                  { w }, { x }, { y });
    return op(b, "gemm_nt",
              { RAD_RANGE("M", 1, rows), RAD_INT("N", n), RAD_INT("K", k), RAD_STR("dtype", m.g.dtype) },
              { w }, { x }, { y });
}
static rad_op rmsnorm(RadBuilder* b, const Model& m, int64_t rows, rad_weight w, int64_t n,
                      rad_buf x, rad_buf y) {
    return op(b, "rmsnorm",
              { RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_F64("eps", m.c.eps),
                RAD_STR("dtype", "bf16") },
              { w }, { x }, { y });
}
static rad_op rope_rows(RadBuilder* b, const Model& m, int64_t rows, int64_t heads, int64_t hd,
                        int inverse, rad_buf x, rad_buf cs) {
    return op(b, "rope_rows",
              { RAD_RANGE("M", 1, rows), RAD_INT("n_head", heads), RAD_INT("head_dim", hd),
                RAD_INT("rot", m.c.rope), RAD_INT("off", hd - m.c.rope), RAD_STR("mode", "gptj"),
                RAD_INT("inverse", inverse), RAD_STR("dtype", "bf16") },
              {}, { x, cs }, { x });
}
/* RADIANCE_DEBUG_NOQ: the model's own quantise-dequantise points (window, compressed and index
 * keys, the indexer query, the attention output) become identity. Diagnosis only: it changes what
 * the model computes, and exists so a parity run against the reference's REF_NOQ=1 form is exact
 * rather than chaotic on random weights (the port's NOTES, GATE0). */
static bool debug_noq() {
    static const bool on = std::getenv("RADIANCE_DEBUG_NOQ") != nullptr;
    return on;
}
static rad_op roundtrip(RadBuilder* b, int64_t rows, int64_t n, int64_t block, const char* fmt,
                        const char* scale, rad_buf x) {
    if (debug_noq()) return 0;
    return op(b, "quant_roundtrip",
              { RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_INT("block", block),
                RAD_STR("format", fmt), RAD_STR("scale", scale), RAD_STR("dtype", "bf16") },
              {}, { x }, { x });
}
static rad_op entry_store(RadBuilder* b, int64_t rows, int64_t n, int64_t bs, rad_buf x,
                          rad_buf page) {
    return op(b, "entry_store",
              { RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_INT("block_size", bs),
                RAD_STR("dtype", "bf16") },
              {}, { x, page }, {});
}

static int geometry(Model& m, const RadBuildCtx* ctx) {
    auto& g = m.g;
    const auto& c = m.c;
    g.n_layer = c.n_layer;
    g.n_embd = c.n_embd;
    g.head_dim = c.head_dim;
    const int world = ctx->world_size ? ctx->world_size : 1;
    g.world = world;
    g.rank = ctx->rank;
    /* The attention world: the largest d <= world that divides both the output groups and the
     * heads; the ranks past it are attention-zero (Geom::attn_world). There is one latent KV head,
     * so geom_from's KV-head rule does not apply: three ranks attend on two, four on four. */
    int aw = 1;
    for (int d = world; d >= 1; --d)
        if (c.o_groups % d == 0 && c.n_head % d == 0) {
            aw = d;
            break;
        }
    g.attn_world = aw;
    g.attn_zero = ctx->rank >= aw;
    g.attn_rank = g.attn_zero ? aw - 1 : ctx->rank;
    m.aw = aw;
    m.ar = g.attn_rank;
    m.zero = g.attn_zero;
    m.gr = c.o_groups / aw;
    g.n_head_all = c.n_head;
    g.n_head = c.n_head / aw;
    g.n_head_kv = g.n_head_kv_all = 1;
    /* Routed experts are expert-parallel: each rank holds n_expert / world whole experts, an even
     * count (the grouped GEMM takes two parity tables), and the router stays replicated. Whole
     * experts are what the bf16 and quantised expert paths both serve, at any world (an expert's
     * width split into whole quant blocks holds at three ranks but not at four). The shared expert
     * is split by width. */
    if (c.n_expert % world || (c.n_expert / world) % 2 || c.n_ff % world) {
        fprintf(stderr, "radiance: deepseek41: %lld experts of width %lld do not split over %d ranks "
                        "(an even count of whole experts each, and the shared expert's width)\n",
                (long long)c.n_expert, (long long)c.n_ff, world);
        return RAD_E_UNSUPPORTED;
    }
    g.n_ff_all = c.n_ff;
    g.n_ff = c.n_ff / world;
    g.n_vocab_all = c.n_vocab;
    g.n_vocab = shard_extent(c.n_vocab, world, ctx->rank);
    g.vocab_off = shard_begin(c.n_vocab, world, ctx->rank);
    g.max_tok = ctx->max_tok;
    g.max_ctx = ctx->max_ctx;
    g.max_seqs = ctx->max_seqs;
    g.max_spec = ctx->max_spec;
    g.max_logit_rows = std::min<int64_t>(g.max_tok, g.max_seqs * (g.max_spec + 1));
    if (ctx->max_out_rows > g.max_logit_rows) g.max_logit_rows = ctx->max_out_rows;
    g.eps = (float)c.eps;
    g.theta = (float)c.theta;
    g.dtype = "bf16";
    g.act_dtype = RAD_BF16;
    g.w_bf16 = true;
    g.wire_exact = 1;
    return g.max_tok < 1 || g.max_ctx < 1 || g.max_seqs < 1 ? RAD_E_INVAL : RAD_OK;
}

static int buffers(RadBuilder* b, Model& m) {
    auto& g = m.g;
    auto& nm = m.nm;
    const auto& c = m.c;
    const int64_t T = g.max_tok, D = c.n_embd, W = m.work;
    const int64_t HN = c.hc * D, HD = g.n_head * c.head_dim;
    auto bf = [&](const char* n, std::initializer_list<int64_t> s) {
        return decl_b(b, nm.f("%s", n), RAD_BF16, s);
    };
    m.h0 = bf("streams", { T, HN });
    m.h1 = bf("streams_next", { T, HN });
    m.xcol = bf("collapsed", { T, D });
    m.xn = bf("normed", { T, D });
    m.pre_a = decl_b(b, nm.f("pre_attn"), RAD_F32, { T, c.hc });
    m.pre_f = decl_b(b, nm.f("pre_ffn"), RAD_F32, { T, c.hc });
    m.post = decl_b(b, nm.f("post"), RAD_F32, { T, c.hc });
    m.comb = decl_b(b, nm.f("comb"), RAD_F32, { T, c.hc * c.hc });
    m.qr = bf("q_lora", { T, c.q_lora });
    m.qrn = bf("q_lora_normed", { T, c.q_lora });
    m.q = bf("q", { T, HD });
    m.kv = bf("window_kv", { T, c.head_dim });
    m.kvn = bf("window_kv_normed", { T, c.head_dim });
    m.o = bf("attn_o", { T, HD });
    m.oa = bf("attn_o_lora", { T, m.gr * c.o_rank });
    m.attn_out = bf("attn_out", { T, D });
    m.kvs = decl_b(b, nm.f("comp_kv_score"), RAD_F32, { T, 2 * c.head_dim });
    m.stage = decl_b(b, nm.f("comp_stage"), RAD_F32, { 2 * W, 2 * c.head_dim });
    m.page = decl_b(b, nm.f("comp_page"), RAD_I32, { W });
    m.bpos = decl_b(b, nm.f("comp_bpos"), RAD_I32, { W });
    m.nc = decl_b(b, nm.f("comp_nc"), RAD_I32, { g.max_seqs });
    const int64_t LR = std::max(T, W);
    m.pooled = bf("comp_pooled", { LR, c.head_dim });
    m.lat = bf("comp_latent", { LR, c.head_dim });
    m.latn = bf("comp_latent_normed", { LR, c.head_dim });
    m.ik = bf("index_k", { LR, c.idim });
    m.ikn = bf("index_k_normed", { LR, c.idim });
    m.iq = bf("index_q", { T, c.ih * c.idim });
    m.iw = bf("index_w", { T, c.ih });
    m.score = decl_b(b, nm.f("index_score"), RAD_F32, { T, m.cmax });
    m.keep = decl_b(b, nm.f("index_keep"), RAD_I32, { T, m.cmax });
    m.sel = decl_b(b, nm.f("index_sel"), RAD_I32, { T, c.itopk });
    m.cs_a = decl_b(b, nm.f("rope_window"), RAD_F32, { g.max_ctx, c.rope }, RAD_BUF_PERSIST);
    m.cs_b = decl_b(b, nm.f("rope_compressed"), RAD_F32, { g.max_ctx, c.rope }, RAD_BUF_PERSIST);
    if (!c.ple.empty()) {
        m.ctok = decl_b(b, nm.f("engram_tokens"), RAD_I32, { T });
        m.ids = decl_b(b, nm.f("engram_ids"), RAD_I32, { T, c.ple_heads });
        m.emb = bf("engram_rows", { T, c.ple_heads * c.ple_dim });
        m.ekv = bf("engram_kv", { T, (c.hc + 1) * D });
    }
    /* The routed FFN's wires (rad_block_moe_fp8.h), bf16 throughout. */
    /* bf16 linears: each ActFP8 carries only its activation (`x`); the plugin norms the FFN input
     * itself (MoeFP8's external-input form) and the block leaves its delta in `act.x`. */
    m.act.x = bf("ffn_normed", { T, D });
    auto& mw = m.mw;
    mw.x = m.xcol;
    mw.h = m.act;
    mw.logits = decl_b(b, nm.f("router_logits"), RAD_F32, { T, c.n_expert });
    mw.ids = decl_b(b, nm.f("expert_ids"), RAD_I32, { T, c.top_k });
    mw.ew = decl_b(b, nm.f("expert_weights"), RAD_F32, { T, c.top_k });
    mw.eoff = decl_b(b, nm.f("expert_offsets"), RAD_I32, { c.n_expert + 1 });
    mw.ecnt = decl_b(b, nm.f("expert_counts"), RAD_I32, { c.n_expert });
    mw.sorted = decl_b(b, nm.f("expert_sorted"), RAD_I32, { T * c.top_k });
    mw.egu = bf("expert_gate_up", { T * c.top_k, 2 * c.n_ff });
    mw.eff.x = bf("expert_ff", { T * c.top_k, c.n_ff });
    mw.edn = bf("expert_down", { T * c.top_k, D });
    mw.sgu = bf("shared_gate_up", { T, 2 * g.n_ff });   /* the shared expert: this rank's width */
    mw.sff.x = bf("shared_ff", { T, g.n_ff });
    mw.sout = bf("shared_out", { T, D });
    mw.sgate = bf("shared_one", { T, 1 });
    m.final_hidden = decl_b(b, nm.f("final_hidden"), RAD_BF16, { T, D }, RAD_BUF_PERSIST);
    m.logit_input = bf("logit_input", { g.max_logit_rows, D });
    m.logits = decl_b(b, nm.f("logits"), RAD_F32, { g.max_logit_rows, g.n_vocab });
    return rad_declare_logits(b, m.logits);
}

/* The caches. The window holds the whole rotated latent per slot (K then V halves); a compressed or
 * index group holds one entry a block of `ratio` slots (entry_store fixes the block size); the
 * compressor's tail is a per-sequence ring of r + max_spec f32 rows. Declared identically on every
 * rank (an attention-zero rank zeroes them). */
static rad_kvgroup win_group(RadBuilder* b, Model& m) {
    RadKVGroupDecl d{};
    d.kind = RAD_KV_WINDOW;
    d.dtype = RAD_BF16;
    d.window = m.c.window + m.g.max_spec;
    d.n_head_kv = 1;
    d.head_dim = m.c.head_dim / 2;
    return rad_decl_kv_group(b, m.nm.f("kv_window"), &d);
}
static rad_kvgroup tail_group(RadBuilder* b, Model& m, int64_t r) {
    RadKVGroupDecl d{};
    d.kind = RAD_KV_LINEAR;
    d.dtype = RAD_F32;
    d.n_head_kv = 1;
    d.state_dim[0] = r + m.g.max_spec;
    d.state_dim[1] = 2 * m.c.head_dim;
    return rad_decl_kv_group(b, m.nm.f("kv_comp_tail_r%lld", (long long)r), &d);
}
static rad_kvgroup full_group(RadBuilder* b, Model& m, const char* fmt, int64_t n, int64_t r) {
    RadKVGroupDecl d{};
    d.kind = RAD_KV_FULL;
    d.dtype = RAD_BF16;
    d.n_head_kv = 1;
    d.head_dim = n / r / 2;
    return rad_decl_kv_group(b, m.nm.f(fmt, (long long)r), &d);
}

/* ------------------------------------------------------------------ one layer */
static int layer(RadBuilder* b, Model& m, Layer& L, int l) {
    auto& g = m.g;
    auto& nm = m.nm;
    const auto& c = m.c;
    const int64_t T = g.max_tok, D = c.n_embd, HD = g.n_head * c.head_dim, W = m.work;
    const int64_t hd = c.head_dim, mix = c.hc * (c.hc + 2);
    const auto grp = grp_layer(l);
    L.l = l;
    L.r = c.ratio[(size_t)l];
    L.src = c.is_src(l);
    L.idx = c.is_idx(l);
    auto w = [&](const char* fmt, uint32_t dt, std::initializer_list<int64_t> shape) {
        return wt(b, nm.f(fmt, l), dt, shape, grp);
    };

    /* ---- Engram, on the streams in front of the block */
    const int pi = c.ple_index(l);
    if (pi >= 0) {
        const int64_t H = c.ple_heads;
        DS_TRY(map_rows(b, nm.f("blk.%d.engram_mult", l),
                              { { "deepseek41.ple.layer_multipliers", (int)(pi * c.ngram), c.ngram } }));
        DS_TRY(map_rows(b, nm.f("blk.%d.engram_vsz", l),
                              { { "deepseek41.ple.head_vocab_sizes", (int)(pi * H), H } }));
        DS_TRY(map_rows(b, nm.f("blk.%d.engram_off", l),
                              { { "deepseek41.ple.head_offsets", (int)(pi * H), H } }));
        L.e_mult = decl_w(b, nm.f("blk.%d.engram_mult", l), RAD_I64, { c.ngram },
                          RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp);
        L.e_vsz = decl_w(b, nm.f("blk.%d.engram_vsz", l), RAD_I64, { H }, RAD_ACCESS_PER_TOKEN,
                         RAD_SHARD_NONE, grp);
        L.e_off = decl_w(b, nm.f("blk.%d.engram_off", l), RAD_I64, { H }, RAD_ACCESS_PER_TOKEN,
                         RAD_SHARD_NONE, grp);
        L.e_kv = lw(b, m, nm.f("blk.%d.engram_kv.weight", l), (c.hc + 1) * D, H * c.ple_dim, grp);
        L.e_gate = w("blk.%d.engram_gate.weight", RAD_F32, { c.hc * D });
        if (!m.tok) {
            RadKVGroupDecl d{};
            d.kind = RAD_KV_CONV;
            d.dtype = RAD_I32;
            d.conv_width = c.ngram;
            d.n_head_kv = 1;
            d.head_dim = 1;
            m.tok = rad_decl_kv_group(b, nm.f("kv_engram_tokens"), &d);
            if (!m.tok) return RAD_E_INVAL;
        }
        DS_TRY(rad_bind_layer_kv(b, l, m.tok));
        L.e_ids = op(b, "ngram_ids",
                     { RAD_RANGE("M", 1, T), RAD_INT("heads", H), RAD_INT("ngram", c.ngram),
                       RAD_INT("eos", c.pad) },
                     { L.e_mult, L.e_vsz, L.e_off }, { m.ctok }, { m.ids });
        L.e_emb = op(b, "embed_lookup",
                     { RAD_RANGE("M", 1, T * H), RAD_INT("n_embd", c.ple_dim),
                       RAD_INT("n_vocab", c.ple_rows), RAD_STR("dtype", "bf16") },
                     { m.ple_table }, { m.ids }, { m.emb });
        L.e_gemm = linear(b, m, T, L.e_kv, (c.hc + 1) * D, H * c.ple_dim, m.emb, m.ekv);
        L.e_gate_op = op(b, "stream_gate_add",
                         { RAD_RANGE("M", 1, T), RAD_INT("n", D), RAD_INT("hc", c.hc),
                           RAD_F64("eps", c.eps), RAD_F64("gate_eps", 1e-6),
                           RAD_STR("dtype", "bf16") },
                         { L.e_gate }, { m.h0, m.ekv }, { m.h1 });
        /* The block reads the streams from h0: the gated streams go back there. */
        L.e_back = op(b, "cast",
                      { RAD_RANGE("M", 1, T), RAD_INT("n", c.hc * D), RAD_STR("from", "bf16"),
                        RAD_STR("to", "bf16") },
                      {}, { m.h1 }, { m.h0 });
    }

    /* ---- the attention connection */
    L.hc_a_fn = w("blk.%d.hc_attn_fn.weight", RAD_F32, { mix, c.hc * D });
    L.hc_a_scale = w("blk.%d.hc_attn_scale.weight", RAD_F32, { 3 });
    L.hc_a_base = w("blk.%d.hc_attn_base.weight", RAD_F32, { mix });
    const std::vector<RadParam> mixp = { RAD_RANGE("M", 1, T), RAD_INT("n", D), RAD_INT("hc", c.hc),
                                         RAD_INT("iterations", c.hc_iters), RAD_F64("eps", c.eps),
                                         RAD_F64("hc_eps", c.hc_eps), RAD_STR("dtype", "bf16") };
    const std::vector<RadParam> hcp = { RAD_RANGE("M", 1, T), RAD_INT("n", D), RAD_INT("hc", c.hc),
                                        RAD_STR("dtype", "bf16") };
    L.mix_a = op(b, "mhc_mix", mixp, { L.hc_a_fn, L.hc_a_scale, L.hc_a_base }, { m.h0 },
                 { m.pre_a, m.post, m.comb });
    L.col_a = op(b, "mhc_collapse", hcp, {}, { m.h0, m.pre_f }, { m.xcol });
    if (m.zero) {
        /* Attention-zero (Geom::attn_world): no heads on this rank. Its caches are declared and
         * bound as on an attention rank, the weightless stores that fix their block sizes declared
         * but never issued, then zeroed (rad_kv_group_zero); the block's output is a fill of zeros
         * the all-reduce adds. */
        if (!m.win && !(m.win = win_group(b, m))) return RAD_E_INVAL;
        L.kv_st = op(b, "kv_store",
                     { RAD_RANGE("M", 1, T), RAD_INT("head_dim", hd / 2), RAD_INT("n_head_kv", 1),
                       RAD_INT("block_size", 16), RAD_STR("kv_dtype", "bf16") },
                     {}, { m.kvn }, {});
        if (!m.win_bs && (m.win_bs = rad_kv_block_size(b, m.win)) < 1) return RAD_E_INVAL;
        DS_TRY(rad_bind_layer_kv(b, l, m.win));
        DS_TRY(rad_kv_group_zero(b, m.win));
        const int64_t r = L.r;
        if (L.src) {
            const int64_t rows = r > 1 ? W : T;
            if (r > 1) {
                if (!m.tail.count(r) && !(m.tail[r] = tail_group(b, m, r))) return RAD_E_INVAL;
                /* Per-sequence state is kept on every rank (only paged groups zero): this rank's
                 * copy of the ring is small and never written. */
                DS_TRY(rad_bind_layer_kv(b, l, m.tail[r]));
            }
            if (!m.index.count(r) && !(m.index[r] = full_group(b, m, "kv_index_r%lld", c.idim, r))) return RAD_E_INVAL;
            L.k_st = entry_store(b, rows, c.idim, r, m.ikn, m.page);
            if (rad_kv_block_size(b, m.index[r]) != r) return RAD_E_INVAL;
            DS_TRY(rad_bind_layer_kv(b, l, m.index[r]));
            DS_TRY(rad_kv_group_zero(b, m.index[r]));
            if (!m.comp.count(r) && !(m.comp[r] = full_group(b, m, "kv_compressed_r%lld", hd, r))) return RAD_E_INVAL;
            L.l_st = entry_store(b, rows, hd, r, m.lat, m.page);
            if (rad_kv_block_size(b, m.comp[r]) != r) return RAD_E_INVAL;
            DS_TRY(rad_bind_layer_kv(b, l, m.comp[r]));
            DS_TRY(rad_kv_group_zero(b, m.comp[r]));
        }
        L.attn_fill = op(b, "fill", { RAD_RANGE("M", 1, T), RAD_INT("n", D), RAD_STR("dtype", "bf16"),
                                      RAD_F64("value", 0.0) },
                         {}, {}, { m.attn_out });
    } else {
        L.attn_norm = w("blk.%d.attn_norm.weight", RAD_F32, { D });
        L.norm_a = rmsnorm(b, m, T, L.attn_norm, D, m.xcol, m.xn);

        /* ---- attention: the query, the window key */
        L.wq_a = lw(b, m, nm.f("blk.%d.attn_q_a.weight", l), c.q_lora, D, grp);
        L.q_norm = w("blk.%d.attn_q_a_norm.weight", RAD_F32, { c.q_lora });
        L.wq_b = lw(b, m, nm.f("blk.%d.attn_q_b.weight", l), HD, c.q_lora, grp, RAD_SHARD_ROW,
                    m.ar * HD, (m.ar + 1) * HD);
        L.q_a = linear(b, m, T, L.wq_a, c.q_lora, D, m.xn, m.qr);
        L.q_an = rmsnorm(b, m, T, L.q_norm, c.q_lora, m.qr, m.qrn);
        L.q_b = linear(b, m, T, L.wq_b, HD, c.q_lora, m.qrn, m.q);
        const rad_buf cs = L.r > 0 ? m.cs_b : m.cs_a;
        L.q_rope = rope_rows(b, m, T, g.n_head, hd, 0, m.q, cs);
        L.wkv = lw(b, m, nm.f("blk.%d.attn_kv.weight", l), hd, D, grp);
        L.kv_norm = w("blk.%d.attn_kv_a_norm.weight", RAD_F32, { hd });
        L.kv_g = linear(b, m, T, L.wkv, hd, D, m.xn, m.kv);
        L.kv_n = rmsnorm(b, m, T, L.kv_norm, hd, m.kv, m.kvn);
        L.kv_rope = rope_rows(b, m, T, 1, hd, 0, m.kvn, cs);
        L.kv_q = roundtrip(b, T, hd, 32, "fp8_e4m3", "ue8m0", m.kvn);
        if (!m.win && !(m.win = win_group(b, m))) return RAD_E_INVAL;
        L.kv_st = op(b, "kv_store",
                     { RAD_RANGE("M", 1, T), RAD_INT("head_dim", hd / 2), RAD_INT("n_head_kv", 1),
                       RAD_INT("block_size", 16), RAD_STR("kv_dtype", "bf16") },
                     {}, { m.kvn }, {});
        if (!m.win_bs) {
            m.win_bs = rad_kv_block_size(b, m.win);
            if (m.win_bs < 1) return RAD_E_INVAL;
        }
        DS_TRY(rad_bind_layer_kv(b, l, m.win));

        /* ---- the compressor and the index keys, at a kv source */
        const int64_t r = L.r;
        if (L.src) {
            const int64_t rows = r > 1 ? W : T;
            L.comp_norm = w("blk.%d.attn_compressor_norm.weight", RAD_F32, { hd });
            if (r > 1) {
                const char* cn = nm.f("blk.%d.attn_compressor_kvg.weight", l);
                DS_TRY(map_concat(b, cn, { nm.f("blk.%d.attn_compressor_kv.weight", l),
                                                 nm.f("blk.%d.attn_compressor_gate.weight", l) }));
                L.comp_w = decl_w(b, cn, RAD_F32, { 2 * hd, D }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                                  grp);
                L.c_gemm = op(b, "gemm_nt_f32",
                              { RAD_RANGE("M", 1, T), RAD_INT("N", 2 * hd), RAD_INT("K", D),
                                RAD_STR("dtype", "bf16") },
                              { L.comp_w }, { m.xn }, { m.kvs });
                if (!m.tail.count(r) && !(m.tail[r] = tail_group(b, m, r))) return RAD_E_INVAL;
                DS_TRY(rad_bind_layer_kv(b, l, m.tail[r]));
                L.c_work = op(b, "qsa_work",
                              { RAD_RANGE("M", 1, T), RAD_INT("n", 2 * hd), RAD_INT("ratio", r),
                                RAD_INT("ring", r + g.max_spec), RAD_INT("seqs", g.max_seqs),
                                RAD_INT("work", W), RAD_STR("dtype", "f32") },
                              {}, { m.kvs }, { m.stage, m.page, m.bpos, m.nc });
                L.c_pool = op(b, "softmax_pool",
                              { RAD_RANGE("M", 1, W), RAD_INT("n", hd), RAD_INT("ratio", r),
                                RAD_STR("dtype", "bf16") },
                              {}, { m.stage, m.page }, { m.pooled });
                L.c_norm = rmsnorm(b, m, W, L.comp_norm, hd, m.pooled, m.lat);
            } else {
                L.comp_w = lw(b, m, nm.f("blk.%d.attn_compressor_kv.weight", l), hd, D, grp);
                L.c_gemm = linear(b, m, T, L.comp_w, hd, D, m.xn, m.pooled);
                L.c_norm = rmsnorm(b, m, T, L.comp_norm, hd, m.pooled, m.lat);
            }
            /* The compressed rows' angles, at each entry's first-token position. */
            L.c_tab = op(b, "rope_table",
                         { RAD_RANGE("M", 1, rows), RAD_INT("rot", c.rope), RAD_F64("theta", c.comp_theta),
                           RAD_F64("scale", 1.0), RAD_STR("dtype", "f32"),
                           RAD_F64("yarn_factor", c.yarn_factor > 1.0 ? c.yarn_factor : 0.0),
                           RAD_F64("yarn_low", c.yarn_low), RAD_F64("yarn_high", c.yarn_high),
                           RAD_F64("yarn_magnitude", 1.0) },
                         {}, { m.bpos }, { m.cs_b });
            /* Index keys from the unrotated latent, then the latent itself rotated and quantised. */
            L.ik_w = lw(b, m, nm.f("blk.%d.indexer.attn_k.weight", l), c.idim, hd, grp);
            L.ik_norm = w("blk.%d.indexer.k_norm.weight", RAD_F32, { c.idim });
            L.k_gemm = linear(b, m, rows, L.ik_w, c.idim, hd, m.lat, m.ik);
            L.k_norm = rmsnorm(b, m, rows, L.ik_norm, c.idim, m.ik, m.ikn);
            L.k_rope = rope_rows(b, m, rows, 1, c.idim, 0, m.ikn, m.cs_b);
            L.k_q = roundtrip(b, rows, c.idim, 32, "fp4_e2m1", "ue8m0", m.ikn);
            if (!m.index.count(r) && !(m.index[r] = full_group(b, m, "kv_index_r%lld", c.idim, r))) return RAD_E_INVAL;
            L.k_st = entry_store(b, rows, c.idim, r, m.ikn, m.page);
            if (rad_kv_block_size(b, m.index[r]) != r) return RAD_E_INVAL;
            DS_TRY(rad_bind_layer_kv(b, l, m.index[r]));
            L.l_rope = rope_rows(b, m, rows, 1, hd, 0, m.lat, m.cs_b);
            L.l_q = roundtrip(b, rows, hd, 16, "fp4_e2m1", "e4m3", m.lat);
            if (!m.comp.count(r) && !(m.comp[r] = full_group(b, m, "kv_compressed_r%lld", hd, r))) return RAD_E_INVAL;
            L.l_st = entry_store(b, rows, hd, r, m.lat, m.page);
            if (rad_kv_block_size(b, m.comp[r]) != r) return RAD_E_INVAL;
            DS_TRY(rad_bind_layer_kv(b, l, m.comp[r]));
            if (r > 1)
                L.c_tail = op(b, "qsa_tail_store",
                              { RAD_RANGE("M", 1, T), RAD_INT("n", 2 * hd), RAD_INT("ratio", r),
                                RAD_INT("ring", r + g.max_spec), RAD_INT("seqs", g.max_seqs),
                                RAD_STR("dtype", "f32") },
                              {}, { m.kvs }, {});
        }

        /* ---- the indexer, at an index source */
        if (L.idx && r > 0) {
            L.iq_w = lw(b, m, nm.f("blk.%d.indexer.attn_q_b.weight", l), c.ih * c.idim, c.q_lora, grp);
            L.iproj = lw(b, m, nm.f("blk.%d.indexer.proj.weight", l), c.ih, D, grp);
            L.i_q = linear(b, m, T, L.iq_w, c.ih * c.idim, c.q_lora, m.qrn, m.iq);
            L.i_rope = rope_rows(b, m, T, c.ih, c.idim, 0, m.iq, m.cs_b);
            L.i_qq = roundtrip(b, T, c.ih * c.idim, 32, "fp4_e2m1", "ue8m0", m.iq);
            L.i_w = linear(b, m, T, L.iproj, c.ih, D, m.xn, m.iw);
            const int64_t width = g.max_ctx / r + 1;
            L.i_score = op(b, "index_score",
                           { RAD_RANGE("M", 1, T), RAD_INT("heads", c.ih), RAD_INT("dim", c.idim),
                             RAD_INT("ratio", r), RAD_INT("width", width),
                             RAD_F64("wscale", 1.0 / std::sqrt((double)c.idim * (double)c.ih)),
                             RAD_STR("dtype", "bf16") },
                           {}, { m.iq, m.iw }, { m.score });
            if (l == c.cand)
                L.i_cand = op(b, "index_candidates",
                              { RAD_RANGE("M", 1, T), RAD_INT("width", width),
                                RAD_INT("block", c.cand_block), RAD_INT("topk_blocks", c.cand_blocks),
                                RAD_INT("ratio", r) },
                              {}, { m.score }, { m.keep });
            const bool uses = c.cand >= 0 && c.cand < l;
            L.i_topk = op(b, "index_topk",
                          { RAD_RANGE("M", 1, T), RAD_INT("width", width), RAD_INT("topk", c.itopk),
                            RAD_INT("ratio", r) },
                          {}, uses ? std::initializer_list<rad_buf>{ m.score, m.keep }
                                   : std::initializer_list<rad_buf>{ m.score },
                          { m.sel });
        }

        /* ---- the sparse attention and the output projection */
        L.sinks = wts(b, g, nm.f("blk.%d.attn_sinks.weight", l), RAD_F32, { g.n_head }, grp, RAD_SHARD_ROW,
                      m.ar * g.n_head, (m.ar + 1) * g.n_head);
        L.attn = op(b, "attn_sparse_sink",
                    { RAD_RANGE("M", 1, T), RAD_INT("heads", g.n_head), RAD_INT("dim", hd),
                      RAD_INT("window", c.window), RAD_INT("ratio", r),
                      RAD_INT("topk", r > 0 ? c.itopk : 0),
                      RAD_F64("scale", 1.0 / std::sqrt((double)hd)), RAD_STR("dtype", "bf16") },
                    { L.sinks },
                    r > 0 ? std::initializer_list<rad_buf>{ m.q, m.sel }
                          : std::initializer_list<rad_buf>{ m.q },
                    { m.o });
        L.o_rope = rope_rows(b, m, T, g.n_head, hd, 1, m.o, cs);
        L.o_q = roundtrip(b, T, HD, 32, "fp8_e4m3", "ue8m0", m.o);
        /* This rank's output groups, each a block of the block-diagonal wo_a. */
        const int64_t gd = HD / m.gr;
        for (int j = 0; j < m.gr; ++j) {
            const int gi_ = (int)(m.ar * m.gr + j);
            const char* n = nm.f("blk.%d.attn_output_a.g%d.weight", l, gi_);
            DS_TRY(map_rows(b, n, { { nm.f("blk.%d.attn_output_a.weight", l), (int)(gi_ * c.o_rank),
                                            c.o_rank } }));
            L.wo_a.push_back(lw(b, m, n, c.o_rank, gd, grp, RAD_SHARD_NONE, 0, 0, true));
            if (!L.wo_a.back()) return RAD_E_INVAL;
            L.o_a.push_back(linear(b, m, T, L.wo_a.back(), c.o_rank, gd, m.o, m.oa));
        }
        const int64_t ok = m.gr * c.o_rank;
        L.wo_b = lw(b, m, nm.f("blk.%d.attn_output_b.weight", l), D, ok, grp, RAD_SHARD_COL,
                    m.ar * ok, (m.ar + 1) * ok);
        L.o_b = linear(b, m, T, L.wo_b, D, ok, m.oa, m.attn_out);
    }
    /* The heads' partial outputs (zeros on an attention-zero rank) summed over every rank. */
    if (g.world > 1)
        L.ar_attn = op(b, "all_reduce",
                       { RAD_INT("world_size", g.world), RAD_RANGE("numel", D, T * D),
                         RAD_STR("dtype", "bf16"), RAD_INT("exact", g.wire_exact),
                         RAD_INT("min_bytes", g.wire_min_bytes) },
                       {}, { m.attn_out }, { m.attn_out });
    L.post_a = op(b, "mhc_post", hcp, {}, { m.attn_out, m.h0, m.post, m.comb }, { m.h1 });

    /* ---- the ffn connection and the MoE */
    L.hc_f_fn = w("blk.%d.hc_ffn_fn.weight", RAD_F32, { mix, c.hc * D });
    L.hc_f_scale = w("blk.%d.hc_ffn_scale.weight", RAD_F32, { 3 });
    L.hc_f_base = w("blk.%d.hc_ffn_base.weight", RAD_F32, { mix });
    L.mix_f = op(b, "mhc_mix", mixp, { L.hc_f_fn, L.hc_f_scale, L.hc_f_base }, { m.h1 },
                 { m.pre_f, m.post, m.comb });
    L.col_f = op(b, "mhc_collapse", hcp, {}, { m.h1, m.pre_a }, { m.xcol });
    L.ffn_norm = w("blk.%d.ffn_norm.weight", RAD_F32, { D });
    L.norm_f = rmsnorm(b, m, T, L.ffn_norm, D, m.xcol, m.act.x);
    MoeFP8::Src s;
    s.norm = nullptr;
    s.router = nm.f("blk.%d.ffn_gate_inp.weight", l);
    s.router_bias = nm.f("blk.%d.exp_probs_b.bias", l);
    s.experts_gate = nm.f("blk.%d.ffn_gate_exps.weight", l);
    s.experts_up = nm.f("blk.%d.ffn_up_exps.weight", l);
    s.experts_down = nm.f("blk.%d.ffn_down_exps.weight", l);
    s.shared_mlp_gate = nm.f("blk.%d.ffn_gate_shexp.weight", l);
    s.shared_mlp_up = nm.f("blk.%d.ffn_up_shexp.weight", l);
    s.shared_mlp_down = nm.f("blk.%d.ffn_down_shexp.weight", l);
    s.cfg.n_expert_all = c.n_expert;
    s.cfg.n_expert = c.n_expert / g.world;
    s.cfg.expert_base = g.rank * s.cfg.n_expert;
    s.cfg.probed_expert = s.cfg.expert_base;
    s.cfg.top_k = c.top_k;
    s.cfg.n_ff_exp = c.n_ff;
    s.cfg.n_ff_shared = g.n_ff;
    s.cfg.rows = T;
    s.cfg.norm_topk = c.norm_topk;
    s.cfg.sigmoid_router = 1;
    s.cfg.router_score = "sqrtsoftplus";
    s.cfg.route_scale = c.route_scale;
    s.cfg.swiglu_limit = c.swiglu_limit;
    s.cfg.shared_ungated = 1;
    DS_TRY(MoeFP8::probe(b, nm, l, s, &s.cfg));
    DS_TRY(L.moe.declare(b, nm, g, l, m.mw, s, false, false));
    L.post_f = op(b, "mhc_post", hcp, {}, { m.act.x, m.h1, m.post, m.comb }, { m.h0 });
    return RAD_OK;
}

/* ------------------------------------------------------------------ declare */
static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx || ctx->is_draft || ctx->rank < 0 || ctx->rank >= MAX_RANKS)
        return RAD_E_INVAL;
    if (ctx->kv_dtype && ctx->kv_dtype != RAD_BF16) {
        fprintf(stderr, "radiance: deepseek41: its caches hold bf16 (lossless for the fp8 and fp4 "
                        "values the model writes); run with --kv-cache-dtype bf16\n");
        return RAD_E_UNSUPPORTED;
    }
    auto& slot = models[ctx->rank];
    slot = std::make_unique<Model>();
    auto& m = *slot;
    m.nm = Names(ctx->scope);
    auto& nm = m.nm;
    DS_TRY(read_cfg(m.c, meta));
    DS_TRY(geometry(m, ctx));
    const auto& c = m.c;
    auto& g = m.g;
    int64_t rmin = 0;
    for (int l = 0; l < c.n_layer; ++l)
        if (c.ratio[(size_t)l] > 0 && (rmin == 0 || c.ratio[(size_t)l] < rmin)) rmin = c.ratio[(size_t)l];
    m.work = g.max_tok + g.max_seqs;
    m.cmax = g.max_ctx / std::max<int64_t>(rmin, 1) + 1;
    DS_TRY(buffers(b, m));

    const int64_t T = g.max_tok, D = c.n_embd;
    /* The vocabulary is split over the ranks (shard_extent): the lookup gives zeros outside this
     * rank's rows and an all-reduce completes it; the head's logits stay sharded for the sampler. */
    const int64_t v0 = g.vocab_off, v1 = g.vocab_off + g.n_vocab;
    m.token = wts(b, g, nm.f("token_embd.weight"), RAD_BF16, { g.n_vocab, D }, grp_model(), RAD_SHARD_ROW,
                  v0, v1, RAD_ACCESS_VOCAB);
    m.head = wts(b, g, nm.f("output.weight"), RAD_BF16, { g.n_vocab, D }, grp_model(), RAD_SHARD_ROW, v0,
                 v1, RAD_ACCESS_VOCAB);
    if (!m.token || !m.head) return RAD_E_INVAL;
    m.final_gain = wt(b, nm.f("output_norm.weight"), RAD_F32, { D }, grp_model());
    m.embed = op(b, "embed_lookup",
                 { RAD_RANGE("M", 1, T), RAD_INT("n_embd", D), RAD_INT("n_vocab", g.n_vocab),
                   RAD_STR("dtype", "bf16"), RAD_INT("vocab_offset", g.vocab_off) },
                 { m.token }, {}, { m.xcol });
    if (g.world > 1)
        m.embed_ar = op(b, "all_reduce",
                        { RAD_INT("world_size", g.world), RAD_RANGE("numel", D, T * D),
                          RAD_STR("dtype", "bf16"), RAD_INT("exact", 1) },
                        {}, { m.xcol }, { m.xcol });
    m.expand = op(b, "cast",
                  { RAD_RANGE("M", 1, T), RAD_INT("n", D), RAD_STR("from", "bf16"),
                    RAD_STR("to", "bf16") },
                  {}, { m.xcol }, { m.h0 });
    /* Layer 0's attention collapses with one-hot gates: stream 0. */
    m.pre0 = op(b, "fill", { RAD_RANGE("M", 1, T), RAD_INT("n", c.hc), RAD_STR("dtype", "f32"),
                             RAD_F64("value", 0.0) },
                {}, {}, { m.pre_f });
    m.pre1 = op(b, "fill", { RAD_RANGE("M", 1, T), RAD_INT("n", 1), RAD_STR("dtype", "f32"),
                             RAD_F64("value", 1.0) },
                {}, {}, { m.pre_f });
    m.rope_a = op(b, "rope_table",
                  { RAD_RANGE("M", 1, T), RAD_INT("rot", c.rope), RAD_F64("theta", c.theta),
                    RAD_F64("scale", 1.0), RAD_STR("dtype", "f32") },
                  {}, {}, { m.cs_a });
    m.rope_b_step = op(b, "rope_table",
                       { RAD_RANGE("M", 1, T), RAD_INT("rot", c.rope), RAD_F64("theta", c.comp_theta),
                         RAD_F64("scale", 1.0), RAD_STR("dtype", "f32"),
                         RAD_F64("yarn_factor", c.yarn_factor > 1.0 ? c.yarn_factor : 0.0),
                         RAD_F64("yarn_low", c.yarn_low), RAD_F64("yarn_high", c.yarn_high),
                         RAD_F64("yarn_magnitude", 1.0) },
                       {}, {}, { m.cs_b });
    if (!c.ple.empty()) {
        m.tmap = wt(b, "deepseek41.engram.token_map", RAD_I32, { c.n_vocab }, grp_model());
        m.ple_table = wt(b, nm.f("per_layer_token_embd.weight"), RAD_BF16, { c.ple_rows, c.ple_dim },
                         grp_model(), RAD_ACCESS_VOCAB);
        m.tok_map = op(b, "token_map", { RAD_RANGE("M", 1, T), RAD_INT("n_vocab", c.n_vocab) },
                       { m.tmap }, {}, { m.ctok });
    }
    m.layers.resize((size_t)c.n_layer);
    for (int l = 0; l < c.n_layer; ++l) DS_TRY(layer(b, m, m.layers[(size_t)l], l));
    m.contract = op(b, "mhc_collapse",
                    { RAD_RANGE("M", 1, T), RAD_INT("n", D), RAD_INT("hc", c.hc), RAD_STR("dtype", "bf16") },
                    {}, { m.h0, m.pre_f }, { m.xcol });
    m.fnorm = rmsnorm(b, m, T, m.final_gain, D, m.xcol, m.final_hidden);
    m.gather = op(b, "gather_rows",
                  { RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n", D), RAD_STR("dtype", "bf16") },
                  {}, { m.final_hidden }, { m.logit_input });
    m.head_op = op(b, "logits_gemm",
                   { RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n_vocab", g.n_vocab),
                     RAD_INT("n_embd", D), RAD_STR("dtype", "bf16") },
                   { m.head }, { m.logit_input }, { m.logits });
    if (debug_noq())
        fprintf(stderr, "radiance: deepseek41: RADIANCE_DEBUG_NOQ: the model's quantise-dequantise "
                        "points are off -- a diagnosis form, not the model\n");
    rad_note(b, "DeepSeek-V4.1: %lld layers, hc %lld, window %lld, %zu kv sources, %zu index "
                "sources, %lld experts top-%lld, Engram at %zu layers",
             (long long)c.n_layer, (long long)c.hc, (long long)c.window, c.kv_src.size(),
             c.idx_src.size(), (long long)c.n_expert, (long long)c.top_k, c.ple.size());
    return RAD_OK;
}

/* ------------------------------------------------------------------ step */
static void step(RadCtx* x, const RadBatch* batch) {
    auto& m = *models[rad_rank(x)];
    const auto& c = m.c;
    const auto& g = m.g;
    const int64_t T = batch->n_tok, S = batch->n_seq, D = c.n_embd;
    const int64_t HD = g.n_head * c.head_dim, hd = c.head_dim;
    const RadOperand pos = praw(batch->positions, RAD_I32, T);
    const RadOperand cu = praw(batch->cu_seqlens, RAD_I32, S + 1);
    const auto* wb = kv_batch(batch, m.win);
    if (!wb || !wb->block_table || !wb->slot_mapping) {
        rad_step_fail(x, "deepseek41: the window group has no block table");
        return;
    }
    const RadOperand wbt = praw2(wb->block_table, RAD_I32, S, wb->block_table_pitch);
    /* The window group's table is rebased at the sequence's first held block; its seqused counts
     * from there, and attn_sparse_sink recovers the base from it (docs/OPS.md). */
    const RadOperand wused = praw(wb->seqused, RAD_I32, S);

    RAD_ISSUE_N(x, m.embed, T, praw(batch->token_ids, RAD_I32, T), RAD_W(m.token), brows(m.xcol, T));
    if (m.embed_ar) RAD_ISSUE_N(x, m.embed_ar, T * D, brows(m.xcol, T), RAD_NONE);
    for (int64_t i = 0; i < c.hc; ++i)
        RAD_ISSUE_N(x, m.expand, T, brows(m.xcol, T), bcol(m.h0, i * D, D, T));
    RAD_ISSUE_N(x, m.pre0, T, brows(m.pre_f, T));
    RAD_ISSUE_N(x, m.pre1, T, bcol(m.pre_f, 0, 1, T));
    RAD_ISSUE_N(x, m.rope_a, T, pos, brows(m.cs_a, g.max_ctx));
    RAD_ISSUE_N(x, m.rope_b_step, T, pos, brows(m.cs_b, g.max_ctx));
    if (m.tok_map)
        RAD_ISSUE_N(x, m.tok_map, T, praw(batch->token_ids, RAD_I32, T), RAD_W(m.tmap),
                    brows(m.ctok, T));

    for (auto& L : m.layers) {
        const int l = L.l;
        const int64_t r = L.r;
        if (L.e_ids) {
            const auto* tk = kv_batch(batch, m.tok);
            const int32_t* tk_idx = tk ? tk->state_index : nullptr;
            int64_t Dn = 0, DT = 0;
            batch_split(batch, &Dn, &DT);
            const int64_t P = S - Dn;
            const int64_t H = c.ple_heads;
            if (Dn > 0)
                RAD_ISSUE_N(x, L.e_ids, DT, brows(m.ctok, DT), kv_cache(m.tok, l),
                            praw(batch->cu_seqlens, RAD_I32, Dn + 1), RAD_W(L.e_mult),
                            RAD_W(L.e_vsz), RAD_W(L.e_off), praw(tk_idx, RAD_I32, Dn),
                            praw(batch->ctx_lens, RAD_I32, Dn),
                            praw(batch->num_accepted, RAD_I32, Dn), brows(m.ids, DT), RAD_NONE);
            if (P > 0)
                RAD_ISSUE_N(x, L.e_ids, T, brows(m.ctok, T), kv_cache(m.tok, l),
                            praw(batch->cu_seqlens + Dn, RAD_I32, P + 1), RAD_W(L.e_mult),
                            RAD_W(L.e_vsz), RAD_W(L.e_off),
                            praw(tk_idx ? tk_idx + Dn : nullptr, RAD_I32, P),
                            praw(batch->ctx_lens + Dn, RAD_I32, P), RAD_NONE, brows(m.ids, T),
                            RAD_NONE);
            /* T rows of operands; the gather's band is T * H, one table row per head per token,
             * read through the ids and rows as flat (rad_block_ple.h does the same). */
            RAD_ISSUE_N(x, L.e_emb, T * H, brows(m.ids, T), RAD_W(m.ple_table), brows(m.emb, T));
            RAD_ISSUE_N(x, L.e_gemm, T, brows(m.emb, T), RAD_W(L.e_kv), brows(m.ekv, T));
            RAD_ISSUE_N(x, L.e_gate_op, T, brows(m.h0, T), bcol(m.ekv, 0, c.hc * D, T),
                        bcol(m.ekv, c.hc * D, D, T), RAD_W(L.e_gate), brows(m.h1, T));
            RAD_ISSUE_N(x, L.e_back, T, brows(m.h1, T), brows(m.h0, T));
        }
        RAD_ISSUE_N(x, L.mix_a, T, brows(m.h0, T), RAD_W(L.hc_a_fn), RAD_W(L.hc_a_scale),
                    RAD_W(L.hc_a_base), brows(m.pre_a, T), brows(m.post, T), brows(m.comb, T));
        RAD_ISSUE_N(x, L.col_a, T, brows(m.h0, T), brows(m.pre_f, T), brows(m.xcol, T));
        if (m.zero) {
            RAD_ISSUE_N(x, L.attn_fill, T, brows(m.attn_out, T));
        } else {
            RAD_ISSUE_N(x, L.norm_a, T, brows(m.xcol, T), RAD_W(L.attn_norm), brows(m.xn, T));
            RAD_ISSUE_N(x, L.q_a, T, brows(m.xn, T), RAD_W(L.wq_a), brows(m.qr, T));
            RAD_ISSUE_N(x, L.q_an, T, brows(m.qr, T), RAD_W(L.q_norm), brows(m.qrn, T));
            RAD_ISSUE_N(x, L.q_b, T, brows(m.qrn, T), RAD_W(L.wq_b), brows(m.q, T));
            const rad_buf cs = r > 0 ? m.cs_b : m.cs_a;
            RAD_ISSUE_N(x, L.q_rope, T, brows(m.q, T), brows(cs, g.max_ctx), pos);
            RAD_ISSUE_N(x, L.kv_g, T, brows(m.xn, T), RAD_W(L.wkv), brows(m.kv, T));
            RAD_ISSUE_N(x, L.kv_n, T, brows(m.kv, T), RAD_W(L.kv_norm), brows(m.kvn, T));
            RAD_ISSUE_N(x, L.kv_rope, T, brows(m.kvn, T), brows(cs, g.max_ctx), pos);
            if (L.kv_q) RAD_ISSUE_N(x, L.kv_q, T, brows(m.kvn, T));
            RAD_ISSUE_N(x, L.kv_st, T, bcol(m.kvn, 0, hd / 2, T), bcol(m.kvn, hd / 2, hd / 2, T),
                        praw(wb->slot_mapping, RAD_I32, T), kv_cache(m.win, l));

            if (L.src) {
                const auto* cb = kv_batch(batch, m.comp[r]);
                const auto* ib = kv_batch(batch, m.index[r]);
                if (!cb || !ib || !cb->block_table || !ib->block_table) {
                    rad_step_fail(x, "deepseek41: a compressed group has no block table");
                    return;
                }
                int64_t rows = T;
                RadOperand page = praw(cb->slot_mapping, RAD_I32, T), bpos = pos;
                if (r > 1) {
                    const auto* tb = kv_batch(batch, m.tail[r]);
                    if (!tb || !tb->state_index) {
                        rad_step_fail(x, "deepseek41: the compressor tail has no state slots");
                        return;
                    }
                    const int64_t pitch = tb->state_index_pitch > 0 ? tb->state_index_pitch : 1;
                    rows = m.work;
                    RAD_ISSUE_N(x, L.c_gemm, T, brows(m.xn, T), RAD_W(L.comp_w), brows(m.kvs, T));
                    RAD_ISSUE_N(x, L.c_work, T, brows(m.kvs, T), pos, cu, kv_cache(m.tail[r], l),
                                praw2(tb->state_index, RAD_I32, S, pitch),
                                praw2(cb->block_table, RAD_I32, S, cb->block_table_pitch), RAD_NONE,
                                brows(m.stage, rows * r), brows(m.page, rows), brows(m.bpos, rows),
                                brows(m.nc, S));
                    RAD_ISSUE_N(x, L.c_pool, rows, brows(m.stage, rows * r), brows(m.pooled, rows),
                                brows(m.page, rows));
                    RAD_ISSUE_N(x, L.c_norm, rows, brows(m.pooled, rows), RAD_W(L.comp_norm),
                                brows(m.lat, rows));
                    page = brows(m.page, rows);
                    bpos = brows(m.bpos, rows);
                } else {
                    RAD_ISSUE_N(x, L.c_gemm, T, brows(m.xn, T), RAD_W(L.comp_w), brows(m.pooled, T));
                    RAD_ISSUE_N(x, L.c_norm, T, brows(m.pooled, T), RAD_W(L.comp_norm), brows(m.lat, T));
                }
                RAD_ISSUE_N(x, L.c_tab, rows, bpos, brows(m.cs_b, g.max_ctx));
                RAD_ISSUE_N(x, L.k_gemm, rows, brows(m.lat, rows), RAD_W(L.ik_w), brows(m.ik, rows));
                RAD_ISSUE_N(x, L.k_norm, rows, brows(m.ik, rows), RAD_W(L.ik_norm), brows(m.ikn, rows));
                RAD_ISSUE_N(x, L.k_rope, rows, brows(m.ikn, rows), brows(m.cs_b, g.max_ctx), bpos);
                if (L.k_q) RAD_ISSUE_N(x, L.k_q, rows, brows(m.ikn, rows));
                RAD_ISSUE_N(x, L.k_st, rows, brows(m.ikn, rows), page, kv_cache(m.index[r], l));
                RAD_ISSUE_N(x, L.l_rope, rows, brows(m.lat, rows), brows(m.cs_b, g.max_ctx), bpos);
                if (L.l_q) RAD_ISSUE_N(x, L.l_q, rows, brows(m.lat, rows));
                RAD_ISSUE_N(x, L.l_st, rows, brows(m.lat, rows), page, kv_cache(m.comp[r], l));
                if (L.c_tail) {
                    const auto* tb = kv_batch(batch, m.tail[r]);
                    const int64_t pitch = tb->state_index_pitch > 0 ? tb->state_index_pitch : 1;
                    RAD_ISSUE_N(x, L.c_tail, T, brows(m.kvs, T), pos, cu, kv_cache(m.tail[r], l),
                                praw2(tb->state_index, RAD_I32, S, pitch));
                }
            }

            if (L.i_score) {
                const int src = c.src_of[(size_t)l];
                const auto* ib = kv_batch(batch, m.index[r]);
                const int64_t live = std::min<int64_t>(
                    g.max_ctx / r + 1, ((int64_t)batch->max_ctx_len + batch->max_q_len) / r + 1);
                RAD_ISSUE_N(x, L.i_q, T, brows(m.qrn, T), RAD_W(L.iq_w), brows(m.iq, T));
                RAD_ISSUE_N(x, L.i_rope, T, brows(m.iq, T), brows(m.cs_b, g.max_ctx), pos);
                if (L.i_qq) RAD_ISSUE_N(x, L.i_qq, T, brows(m.iq, T));
                RAD_ISSUE_N(x, L.i_w, T, brows(m.xn, T), RAD_W(L.iproj), brows(m.iw, T));
                RAD_ISSUE_N(x, L.i_score, T, brows(m.iq, T), brows(m.iw, T), kv_cache(m.index[r], src),
                            praw2(ib->block_table, RAD_I32, S, ib->block_table_pitch), pos, cu,
                            bcol(m.score, 0, live, T));
                if (L.i_cand)
                    RAD_ISSUE_N(x, L.i_cand, T, bcol(m.score, 0, live, T), pos, bcol(m.keep, 0, live, T));
                const bool uses = c.cand >= 0 && c.cand < l;
                RAD_ISSUE_N(x, L.i_topk, T, bcol(m.score, 0, live, T),
                            uses ? bcol(m.keep, 0, live, T) : RAD_NONE, pos, brows(m.sel, T));
            }

            if (r > 0) {
                const int src = c.src_of[(size_t)l];
                const auto* cb = kv_batch(batch, m.comp[r]);
                RAD_ISSUE_N(x, L.attn, T, brows(m.q, T), kv_cache(m.win, l), wbt, wused,
                            kv_cache(m.comp[r], src),
                            praw2(cb->block_table, RAD_I32, S, cb->block_table_pitch), brows(m.sel, T),
                            pos, cu, RAD_W(L.sinks), brows(m.o, T));
            } else {
                RAD_ISSUE_N(x, L.attn, T, brows(m.q, T), kv_cache(m.win, l), wbt, wused, RAD_NONE,
                            RAD_NONE, RAD_NONE, pos, cu, RAD_W(L.sinks), brows(m.o, T));
            }
            RAD_ISSUE_N(x, L.o_rope, T, brows(m.o, T), brows(cs, g.max_ctx), pos);
            if (L.o_q) RAD_ISSUE_N(x, L.o_q, T, brows(m.o, T));
            const int64_t gd = HD / m.gr;
            for (int j = 0; j < m.gr; ++j)
                RAD_ISSUE_N(x, L.o_a[(size_t)j], T, bcol(m.o, j * gd, gd, T), RAD_W(L.wo_a[(size_t)j]),
                            bcol(m.oa, j * c.o_rank, c.o_rank, T));
            RAD_ISSUE_N(x, L.o_b, T, brows(m.oa, T), RAD_W(L.wo_b), brows(m.attn_out, T));
        }
        if (L.ar_attn) RAD_ISSUE_N(x, L.ar_attn, T * D, brows(m.attn_out, T), RAD_NONE);
        RAD_ISSUE_N(x, L.post_a, T, brows(m.attn_out, T), brows(m.h0, T), brows(m.post, T),
                    brows(m.comb, T), brows(m.h1, T));

        RAD_ISSUE_N(x, L.mix_f, T, brows(m.h1, T), RAD_W(L.hc_f_fn), RAD_W(L.hc_f_scale),
                    RAD_W(L.hc_f_base), brows(m.pre_f, T), brows(m.post, T), brows(m.comb, T));
        RAD_ISSUE_N(x, L.col_f, T, brows(m.h1, T), brows(m.pre_a, T), brows(m.xcol, T));
        RAD_ISSUE_N(x, L.norm_f, T, brows(m.xcol, T), RAD_W(L.ffn_norm), brows(m.act.x, T));
        L.moe.step(x, batch);
        RAD_ISSUE_N(x, L.post_f, T, brows(m.act.x, T), brows(m.h1, T), brows(m.post, T),
                    brows(m.comb, T), brows(m.h0, T));
    }
    RAD_ISSUE_N(x, m.contract, T, brows(m.h0, T), brows(m.pre_f, T), brows(m.xcol, T));
    RAD_ISSUE_N(x, m.fnorm, T, brows(m.xcol, T), RAD_W(m.final_gain), brows(m.final_hidden, T));
    if (batch->n_out > 0) {
        RAD_ISSUE_N(x, m.gather, batch->n_out, brows(m.final_hidden, T),
                    praw(batch->out_ids, RAD_I32, batch->n_out), brows(m.logit_input, batch->n_out));
        RAD_ISSUE_N(x, m.head_op, batch->n_out, brows(m.logit_input, batch->n_out), RAD_W(m.head),
                    brows(m.logits, batch->n_out));
    }
}

static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    return RAD_OK;
}
}  // namespace deepseek41

#ifndef RAD_ARCH_NO_EXPORTS
RAD_ARCH_PROBE(deepseek41)
RAD_ARCH_PLUGIN(deepseek41, "deepseek41", "", "0.1.0",
                "DeepSeek-V4.1: hyper-connection streams, window + sparse compressed latent "
                "attention, sqrt(softplus) MoE, Engram")
#endif
