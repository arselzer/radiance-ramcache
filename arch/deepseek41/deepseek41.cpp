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
 *   engram (at ple layers)   token_map -> ngram_ids -> embed_lookup -> cast to the card -> gemm ->
 *                            stream_gate_add (every layer's ngram_ids issued before the first
 *                            block, see step())
 *   attention connection     mhc_mix (this layer's gates) ; mhc_collapse (the previous gates)
 *   attention                q path, window key into the window cache, the compressor at a
 *                            kv-source layer (qsa_work / softmax_pool, or one entry a token at
 *                            ratio 1), the indexer at an index-source layer, attn_sparse_sink,
 *                            the inverse rotation, the grouped output projection ; mhc_post
 *   ffn connection           mhc_mix ; mhc_collapse ; MoeFP8 (norm, router, experts) ; mhc_post
 * then mhc_collapse with the last gates, the final norm and the head.
 *
 * The checkpoint's DSpark drafter (mtp.0 .. mtp.{n-1}, when the container carries its stages as
 * blk.{n_layer}.. with `nextn_predict_layers`) runs on the block-drafter protocol: see struct Dsp. */
#include <arch/rad_block_moe_fp8.h>
#include "ds41_prefetch_plan.h"
#include "ista_dialect.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

/* rad_declare_consumer_suffix (abi/rad_builder.h) is an additive export: referenced weakly, it is
 * null on an engine older than the call, and the consumer skip is then refused (cut_declare). */
#pragma weak rad_declare_consumer_suffix
/* rad_consumer_row_grain likewise: null on an engine older than the per-row cut, which then never
 * hands a step RadBatch::consumer_row0 (cut_declare leaves Model::cut_rows false). */
#pragma weak rad_consumer_row_grain
/* The same for rad_kv_group_replicated (RADIANCE_EXPERIMENTAL_DS41_KV_REPLICATED, kv_replicate). */
#pragma weak rad_kv_group_replicated

namespace deepseek41 {
using namespace rad::arch;

/* RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ATTN=1 and its modifiers ..._PREFETCH_MIB, ..._SET and
 * ..._ENGRAM, as declare() reads them into Model::pfc (ds41_prefetch_plan.h; see Layer::pf). */
static pf::Cfg pf_env() {
    return pf::parse(std::getenv("RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ATTN"),
                     std::getenv("RADIANCE_EXPERIMENTAL_DS41_PREFETCH_MIB"),
                     std::getenv("RADIANCE_EXPERIMENTAL_DS41_PREFETCH_SET"),
                     std::getenv("RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ENGRAM"));
}

/* RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8=1: the op that reads the FP8-sourced dense linears (see
 * Model::f8). */
static constexpr const char* kDenseF8 = "gemm_nt_bf16f8";

/* RADIANCE_EXPERIMENTAL_DS41_ROUTER_BF16=1: the trunk's f32 projections that ship bf16 -- every
 * layer's router ([384, 5120], 7.86 MB a layer replicated on every card, 300 MiB) and the kv|gate
 * projection of the three compressors with a ratio above 1 (layers 2, 8 and 14 of the release,
 * [1024, 5120], 21 MB each as f32) -- are read through gemm_nt_f32bf16, which keeps them as bf16
 * and widens each element back by a shift, multiplying the same operand bits in the same order.
 * The container holds the routers widened to f32 (all 40 checked: 0 of 78.6M low halves nonzero),
 * which the kernel's layout stores as their bf16 halves, refusing the load at any nonzero low half
 * (r4d_layout.cpp) rather than rounding it; it holds the compressors as the release's bf16, which
 * the loader widened for gemm_nt_f32 and which are now declared bf16 and read as they are. Every
 * logit and compressor score is bit-identical, from half the bytes: 180 MiB a card more for the
 * experts (150 of it the routers'), and a decode router GEMV reading 3.9 MB instead of 7.9 (19.7
 * us as f32, latency-bound at 399 GB/s). The drafter's routers stay f32. Read once a declare
 * (Model::router_bf16). */
static bool router_bf16_on() {
    const char* s = std::getenv("RADIANCE_EXPERIMENTAL_DS41_ROUTER_BF16");
    return s && !std::strcmp(s, "1");
}

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
/* The ISTA export deliberately keeps Engram's four integer controls as tensors,
 * and its spelling must not silently turn the two Engram layers off. Presence
 * of any dialect marker selects strict ISTA admission below, even if malformed. */
static bool is_ista(const RadModelMeta* m) {
    return m && (rad_meta_gets(m, "deepseek41.engram.present", nullptr) ||
                 rad_meta_gets(m, "deepseek41.engram.layer_ids", nullptr) ||
                 rad_meta_gets(m, "deepseek41.engram.source", nullptr));
}
/* DeepSeek's release states the same values in its config.json (text_config's keys, which the
 * core also reads bare). Each deepseek41.* key read here and the release's name for it; a key the
 * release does not state (the Engram's bucket sizes, derived from its primes) has none. */
static const char* release_key(const char* k) {
    static const struct { const char* gguf; const char* hf; } t[] = {
        { "rope.dimension_count", "qk_rope_head_dim" },
        { "attention.q_lora_rank", "q_lora_rank" },
        { "attention.output_group_count", "o_groups" },
        { "attention.output_lora_rank", "o_lora_rank" },
        { "attention.sliding_window", "sliding_window" },
        { "expert_feed_forward_length", "moe_intermediate_size" },
        { "expert_weights_scale", "routed_scaling_factor" },
        { "expert_weights_norm", "norm_topk_prob" },
        { "swiglu_clamp_exp", "swiglu_limit" },
        { "attention.indexer.head_count", "index_n_heads" },
        { "attention.indexer.key_length", "index_head_dim" },
        { "attention.indexer.top_k", "index_topk" },
        { "attention.indexer.candidate_source_layer", "candidate_source_layer_id" },
        { "attention.indexer.candidate_top_k_blocks", "candidate_topk_blocks" },
        { "attention.indexer.candidate_block_size", "candidate_block_size" },
        { "hyper_connection.count", "hc_mult" },
        { "hyper_connection.sinkhorn_iterations", "hc_sinkhorn_iters" },
        { "hyper_connection.epsilon", "hc_eps" },
        { "attention.compress_rope_freq_base", "compress_rope_theta" },
        { "attention.compress_ratios", "compress_ratios" },
        { "attention.kv_source_layers", "kv_source_layer_ids" },
        { "attention.indexer.source_layers", "index_source_layer_ids" },
        { "ple.layers", "engram_layer_ids" },
        { "ple.ngram_size", "engram_max_ngram_size" },
        { "ple.heads_per_ngram", "engram_n_heads" },
        { "embedding_length_per_layer_input", "engram_head_dim" },
        { "engram.pad_token_id", "engram_pad_token_id" },
        { "nextn_predict_layers", "num_nextn_predict_layers" },
        { "dspark.block_size", "dspark_block_size" },
        { "dspark.noise_token_id", "dspark_noise_token_id" },
        { "dspark.target_layers", "dspark_target_layer_ids" },
        { "dspark.markov_rank", "dspark_markov_rank" },
        { "dspark.expert_count", "dspark_n_routed_experts" },
        { "dspark.expert_used_count", "dspark_num_experts_per_tok" } };
    for (const auto& e : t)
        if (std::strcmp(e.gguf, k) == 0) return e.hf;
    return nullptr;
}
/* deepseek41.<k>, else the release's spelling of it; null when neither is there. */
static const char* gs(const RadModelMeta* m, const char* k) {
    const char* s = rad_meta_gets(m, key(k).c_str(), nullptr);
    if (!s && is_ista(m))
        if (const char* a = ista::source_key(k)) s = rad_meta_gets(m, key(a).c_str(), nullptr);
    if (!s)
        if (const char* r = release_key(k)) s = rad_meta_gets(m, r, nullptr);
    return s;
}
static int64_t gi(const RadModelMeta* m, const char* k, int64_t d) {
    const char* s = gs(m, k);
    return s && *s ? std::strtoll(s, nullptr, 10) : d;
}
static double gf(const RadModelMeta* m, const char* k, double d) {
    const char* s = gs(m, k);
    return s && *s ? std::strtod(s, nullptr) : d;
}
/* Whether the checkpoint is DeepSeek's release (config.json, inference/model.py's tensor names)
 * rather than a GGUF in llama.cpp's names: it states no deepseek41.block_count. */
static bool is_release(const RadModelMeta* m) {
    return rad_meta_gets(m, key("block_count").c_str(), nullptr) == nullptr &&
           rad_meta_gets(m, "num_hidden_layers", nullptr) != nullptr;
}
/* The native candidate is explicitly hybrid: official backbone metadata and
 * tensors, exact ISTA integer controls and two original IQ4NL Engram tables. */
static bool local_ista_engram(const RadModelMeta* m) {
    const char* p=rad_meta_gets(m,"deepseek41.engram_source",nullptr);
    return is_ista(m) || (p && !std::strcmp(p,"ista_iq4_nl_v1"));
}
/* An integer array spelled out, comma-separated (core/format/checkpoint.cpp, meta_from_gguf) or
 * space-separated (a config.json array). */
static std::vector<int64_t> int_list(const char* s) {
    std::vector<int64_t> v;
    if (!s || !*s || *s == '[') return v;
    while (*s) {
        char* e = nullptr;
        v.push_back(std::strtoll(s, &e, 10));
        if (e == s) return {};
        s = *e == ',' ? e + 1 : e;
    }
    return v;
}
static std::vector<int64_t> gl(const RadModelMeta* m, const char* k) { return int_list(gs(m, k)); }

static int read_cfg(Cfg& c, const RadModelMeta* m) {
    if (const char* p=rad_meta_gets(m,"deepseek41.engram_source",nullptr)) {
        if (std::strcmp(p,"ista_iq4_nl_v1") || (!is_ista(m) && !is_release(m)))
            return RAD_E_UNSUPPORTED;
    }
    if (is_ista(m)) {
        const char* present = rad_meta_gets(m, "deepseek41.engram.present", nullptr);
        if (!present || std::strcmp(present, "1")) return RAD_E_UNSUPPORTED;
        /* Reject conflicting dual spellings rather than selecting one silently. */
        for (const auto& a : ista::aliases) {
            const char* x = rad_meta_gets(m, key(a.canonical).c_str(), nullptr);
            const char* y = rad_meta_gets(m, key(a.source).c_str(), nullptr);
            if (x && y && std::strcmp(x, y)) return RAD_E_UNSUPPORTED;
        }
    }
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
        const char* s = gs(m, "swiglu_clamp_exp");
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
    const int64_t heads_per_ngram = gi(m, "ple.heads_per_ngram", 0);
    if (local_ista_engram(m) && (c.ngram != 4 || heads_per_ngram != 8)) return RAD_E_UNSUPPORTED;
    c.ple_heads = (c.ngram - 1) * heads_per_ngram;
    c.ple_dim = gi(m, "embedding_length_per_layer_input", 0);
    c.pad = gi(m, "engram.pad_token_id", -1);
    if (local_ista_engram(m)) {
        const auto rows = is_ista(m) ? gl(m, "engram.num_embeddings")
                                    : int_list(rad_meta_gets(m,"engram_num_embeddings",nullptr));
        if (rows != std::vector<int64_t>{ista::layer_rows[0], ista::layer_rows[1]})
            return RAD_E_UNSUPPORTED;
        c.ple_rows = ista::table_rows;
    } else {
        for (int64_t v : gl(m, "ple.head_vocab_sizes")) c.ple_rows += v;
        /* The release states each Engram table's rows; their sum is the joined table's. */
        if (!c.ple_rows)
            for (int64_t v : int_list(rad_meta_gets(m, "engram_num_embeddings", nullptr))) c.ple_rows += v;
    }
    /* Strict dialect bounds precede source-vector allocation and derived products. */
    if (local_ista_engram(m)) {
        const std::vector<int64_t> rows = is_ista(m) ? gl(m, "engram.num_embeddings")
            : int_list(rad_meta_gets(m,"engram_num_embeddings",nullptr));
        const std::vector<int64_t> kv{2, 8, 14, 20};
        const std::vector<int64_t> ix{2, 8, 14, 20, 24, 28, 32, 36};
        if (c.n_layer != 40 || c.n_embd != 5120 || c.n_head != 64 || c.head_dim != 512 ||
            c.rope != 64 || c.q_lora != 1280 || c.o_groups != 8 || c.o_rank != 1024 ||
            c.window != 128 || c.n_vocab != 129280 || c.n_expert != 384 || c.top_k != 6 ||
            c.n_ff != 2304 || c.n_shared != 1 || c.hc != 4 || c.hc_iters != 20 ||
            c.ngram != 4 || c.ple_heads != 24 || c.ple_dim != 256 || c.pad != 2 ||
            c.ple != std::vector<int64_t>{1, 14} ||
            rows != std::vector<int64_t>{ista::layer_rows[0], ista::layer_rows[1]} ||
            c.ple_rows != ista::table_rows || c.ratio.size() < 40 || c.ratio.size() > 43 ||
            c.kv_src != kv || c.idx_src != ix ||
            c.cand != 20 || c.cand_blocks != 2048 || c.cand_block != 8)
            return RAD_E_UNSUPPORTED;
        for (int l = 0; l < 40; ++l)
            if (c.ratio[(size_t)l] != (l < 2 ? 0 : l < 20 ? 2 : 1)) return RAD_E_UNSUPPORTED;
    }

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
    /* The kv and index sources may be absent where no layer compresses (a model cut to its first
     * layers); a compressing layer without one is refused below. */
    if ((int64_t)c.ratio.size() < c.n_layer) {
        fprintf(stderr, "radiance: deepseek41: compress_ratios has %zu values for %lld layers\n",
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
/* One read into the card's on-die cache: `op`, libr4d's `prefetch` of weight `w`'s leading
 * `bytes` (rad_arch.h decl_prefetch); `kind` a pf::Kind, `cut` when `bytes` is less than `w`
 * holds (the read that crossed the budget). */
struct PfRead {
    rad_op op = 0;
    rad_weight w = 0;
    int64_t bytes = 0;
    int kind = 0;
    bool cut = false;
};

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
    /* RADIANCE_EXPERIMENTAL_DS41_ROUTER_BF16: the compressor's kv|gate (comp_w at a ratio above 1)
     * and the router held at two bytes an element (bf16_held()), not four. */
    bool comp_bf16 = false, rt_bf16 = false;
    /* Reads of this layer's weights into the card's on-die cache (pf_layer(), ds41_prefetch_plan.h;
     * empty unless RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ATTN=1), in read order: `pf` issued by the
     * layer before while it waits for its CPU rows (pf_tail), `pf_post` by an Engram layer itself
     * behind its Engram GEMM, which would evict anything read before it but its own weight. By
     * default q_a, kv, q_b and wo_a whole, and nothing for an Engram layer. */
    std::vector<PfRead> pf, pf_post;
    /* ops, in issue order */
    rad_op e_ids = 0, e_emb = 0, e_copy = 0, e_gemm = 0, e_gather = 0, e_gate_op = 0, e_back = 0;
    /* This Engram layer's row ids [T, heads]: one buffer a layer, since step() writes every
     * layer's ids before the first block runs (see there). */
    rad_buf ids = 0;
    rad_op mix_a = 0, col_a = 0, norm_a = 0;
    rad_op q_a = 0, q_an = 0, q_b = 0, q_rope = 0;
    rad_op kv_g = 0, kv_n = 0, kv_rope = 0, kv_q = 0, kv_st = 0;
    rad_op c_gemm = 0, c_work = 0, c_pool = 0, c_norm = 0, c_tab = 0;
    rad_op k_gemm = 0, k_norm = 0, k_rope = 0, k_q = 0, k_st = 0;
    rad_op l_rope = 0, l_q = 0, l_st = 0, c_tail = 0;
    rad_op i_q = 0, i_rope = 0, i_qq = 0, i_w = 0, i_score = 0, i_cand = 0, i_topk = 0;
    /* index_score over run p of a step's rows (row_part p); i_runs[0] is i_score. */
    std::vector<rad_op> i_runs;
    rad_op attn = 0, o_rope = 0, o_q = 0, o_b = 0, post_a = 0, attn_fill = 0, ar_attn = 0;
    std::vector<rad_op> o_a;
    /* The fused chains (fused()), each 0 where its chain above is declared instead: kv_n + kv_rope
     * + kv_q, k_norm + k_rope + k_q, o_rope + o_q, l_rope + l_q, i_rope + i_qq, col_a + norm_a,
     * col_f + norm_f. */
    rad_op kv_f = 0, k_f = 0, o_f = 0, l_f = 0, i_f = 0, cn_a = 0, cn_f = 0;
    /* mhc_post into the next sublayer's collapse + norm (post_collapse_norm()): pcn_f = post_a +
     * col_f + norm_f, issued for post_a; pcn_a = the previous layer's post_f + this layer's col_a +
     * norm_a, issued for that post_f (so before this layer's Engram-free tap and mix_a). */
    rad_op pcn_f = 0, pcn_a = 0;
    rad_weight ffn_norm = 0;
    rad_op mix_f = 0, col_f = 0, norm_f = 0, post_f = 0;
    MoeFP8 moe{};
    /* A DSpark target layer: the hc mean of its input streams (after its Engram) into column slice
     * `tap_col` of the drafter's source rows. */
    rad_op tap = 0;
    int tap_col = -1;
    /* A DSpark stage's context pass: its window key from the projected trunk rows. */
    rad_op d_kv = 0, d_kvn = 0, d_kvrope = 0, d_kvq = 0, d_kvst = 0;
    /* A DSpark stage's attention over the drafter's short block (Dsp::block_lo): `attn` at window
     * `window + block_lo`. 0 without one. */
    rad_op attn_lo = 0;
};

/* The DSpark block drafter (the checkpoint's mtp.0 .. mtp.{n-1}; official inference/model.py DSparkBlock,
 * forward_spec), on the engine's block-drafter protocol (docs/PLUGIN.md, RAD_DRAFT_BLOCK):
 *
 *   trunk step    each target layer's input streams (after its Engram) collapse to their hc mean
 *                 (mhc_contract) into one column slice of `src` [T, n_taps * D]: model.py's main_hiddens.
 *   context pass  one row per trunk row: main_x = main_norm(main_proj(src)); per stage the window key
 *                 kv_norm(wkv(main_x)), rotated at the row's position, fp8 round trip, into the stage's
 *                 layer of the drafter's window cache. The stages' blocks never run over the context
 *                 (DSparkBlock.forward at start_pos 0 only seeds the cache, and each decode step writes the
 *                 committed token's key from its main_x).
 *   query pass    `block` rows a sequence, [anchor, noise x (block - 1)] at the anchor's position on: the
 *                 trunk's embedding into hc copies, identity pre gates, then each stage a full V4 block
 *                 (hc mix, window-only latent attention, MoE) whose attention reads the last `window`
 *                 context keys before the anchor plus all `block` keys of its own rows, non-causal
 *                 (get_dspark_topk_idxs): attn_sparse_sink over the drafter's cache with window
 *                 `window + block` and every row placed at its block's last position (iota + add over
 *                 the positions), while rope uses each row's own. The head: hc collapse with the last
 *                 stage's ffn gates, the stage's norm, the trunk's lm_head; the Markov bias
 *                 W2 . W1[prev] walked greedily over the head's top `cand` (dflash_select, as
 *                 arch/common/rad_block_dspark.h) into the proposal buffer. Every row predicts the token
 *                 after it (forward_head: logits[i] + bias(output_ids[i]) -> output_ids[i + 1]), so the
 *                 block is the depth.
 *   not applied   the confidence head (adaptive block length; it changes no proposal), and the bias
 *                 from outside the head's top `cand` (the walk's only approximation, acceptance only). */
struct Dsp {
    bool on = false;
    int base = 0, n_stage = 0;            /* stages at absolute layers base .. base + n_stage - 1 */
    int64_t block = 0, noise = -1, rank = 0, n_expert = 0, top_k = 0, cand = 0, Q = 0;
    int64_t ne_rank = 0, ne_base = 0;     /* this rank's routed experts [ne_base, ne_base + ne_rank) */
    std::vector<int64_t> taps;            /* target layers, ascending (main_hiddens' order) */
    rad_kvgroup kv = 0;
    int64_t kv_bs = 0;
    /* the trunk's taps and the context pass, max_tok rows */
    rad_buf src = 0, c = 0, cn = 0, ckv = 0, ckvn = 0;
    /* the query pass, Q rows */
    rad_buf x = 0, xn = 0, h0 = 0, h1 = 0, pre_a = 0, pre_f = 0, post = 0, comb = 0;
    rad_buf qr = 0, qrn = 0, q = 0, kv_ = 0, kvn = 0, o = 0, oa = 0, attn_out = 0, poff = 0, apos = 0;
    rad_buf fin = 0, rows = 0, logits = 0, cand_ = 0, unary = 0, pairs = 0, gath = 0, tokens = 0;
    ActFP8 act{};
    MoeFP8::Wire mw{};
    rad_weight fc = 0, main_norm = 0, norm = 0, mk1 = 0, mk2 = 0;
    rad_op c_tab = 0, c_fc = 0, c_norm = 0, sizing = 0;
    rad_op embed = 0, embed_ar = 0, expand = 0, pre0 = 0, pre1 = 0, q_tab = 0, poff_op = 0, apos_op = 0;
    rad_op contract = 0, fnorm = 0, gather = 0, head = 0, topk = 0, pgather = 0, merge = 0, select = 0;
    /* RADIANCE_EXPERIMENTAL_DS41_DSPARK_SHORT_BLOCK=N (2 .. block - 1): the query pass may also run
     * a block of N rows, [anchor, noise x (N - 1)], proposing N tokens (rad_drafter_short_block; the
     * core runs it from --draft-short-from decoding sequences on). A verify cap of k reads only the
     * first k proposals, so at k + 1 rows the drafter's three MoE stages route 3 (k + 1) slots instead
     * of 15. The block is non-causal, so its rows' proposals are not the full block's first ones:
     * the replies stay the target's, acceptance may move (the drafter was trained at 5). Its own ops
     * where the block length is a parameter: the rows' offsets to the block's last position, each
     * stage's attention window, and the walk. 0: none, and nothing of it declared. */
    int64_t block_lo = 0;
    rad_op poff_lo = 0, select_lo = 0;
    std::vector<Layer> st;
    /* RADIANCE_EXPERIMENTAL_DS41_DSPARK_CTX_DEFER=1 (the core's Scheduler::block_begin decides): the
     * context pass may be skipped while nothing drafts and caught up later in one multi-row pass. The
     * trunk's taps then append to `src` (declared persistent, `src_rows` = max_tok + the drafter's window
     * + 1 rows) at row `tap_rows`, the rows written since the last context pass, in issue order; a
     * context pass of T rows reads the last T of them and starts the count over. Unset: the taps go to
     * rows [0, T) and the pass reads them, as before. Only the drafter's proposals can change (a draft is
     * always verified), and only through a multi-row pass's rounding. */
    bool ctx_defer = false;
    int64_t src_rows = 0, tap_rows = 0;
    bool defer_warned = false;
    /* RADIANCE_EXPERIMENTAL_DS41_DSPARK_CONF set: the last stage's collapsed rows (`x`, the confidence
     * head's input) are declared persistent, so the engine can read them back after the query pass. */
    bool conf_x = false;
};

struct Model {
    Names nm{};
    Geom g{};
    Cfg c{};
    std::vector<Layer> layers;
    int64_t work = 0, cmax = 0;
    /* The indexer scores a step's rows in runs of at most `irows` rows into one [irows, cmax] score
     * plane (index_runs()), at most `iparts` runs a step; `kwords` is index_candidates' keep, a bit
     * a block of the candidate layer's width, i32 words a row (0: the model has no candidates). */
    int64_t irows = 0, iparts = 0, kwords = 0;
    /* The attention world (geometry()): attention ranks [0, aw), this rank's index `ar` in it and
     * its `gr` output groups; `zero` for an attention-zero rank. */
    int aw = 1, ar = 0;
    int64_t gr = 0;
    bool zero = false;
    /* This rank's columns [sh_lo, sh_hi) of the shared expert when its width does not split
     * evenly into whole blocks (shared_slice); 0, 0 is the even split. */
    int64_t sh_lo = 0, sh_hi = 0;
    /* buffers */
    rad_buf h0 = 0, h1 = 0, xcol = 0, xn = 0, pre_a = 0, pre_f = 0, post = 0, comb = 0;
    rad_buf qr = 0, qrn = 0, q = 0, kv = 0, kvn = 0, o = 0, oa = 0, attn_out = 0;
    rad_buf kvs = 0, stage = 0, page = 0, bpos = 0, nc = 0, pooled = 0, lat = 0, latn = 0,
            ik = 0, ikn = 0;
    rad_buf iq = 0, iw = 0, score = 0, keep = 0, sel = 0, cs_a = 0, cs_b = 0;
    rad_buf ctok = 0, emb = 0, embd = 0, ekv = 0, ekv_part = 0;
    /* engram_kv's rows a rank computes when they are split over the ranks (0: replicated). */
    int64_t esplit = 0;
    /* RADIANCE_EXPERIMENTAL_DS41_ROUTER_BF16 (router_bf16_on): the routers and compressor
     * projections declared over gemm_nt_f32bf16; those held at two bytes an element (bf16_held())
     * counted, with the bytes that saves. */
    bool router_bf16 = false;
    int n_router_bf16 = 0, n_comp_bf16 = 0;
    int64_t bf16_saved = 0;
    rad_buf final_hidden = 0, logit_input = 0, logits = 0;
    ActFP8 act{};
    MoeFP8::Wire mw{};
    /* KV groups */
    rad_kvgroup win = 0, tok = 0;
    std::map<int64_t, rad_kvgroup> comp, index, tail;
    /* The GGUF type of each linear weight the model ships in llama.cpp's blocks (lw()). */
    std::map<rad_weight, const char*> wty;
    /* RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8=1 (dense_f8_gate): DeepSeek's release ships the
     * trunk's dense linears as E4M3 codes with an E8M0 exponent a 32x32 block, which the container
     * keeps cast to bf16. Those linear() reads by `gemm_nt_bf16f8`, gemm_nt's bytes from a stored
     * form of the codes and exponents (libr4d: half the weight bytes; its relayout refuses at load
     * any weight that is not exactly so). `f8` the weights lw8() marked, `f8_read` those whose op
     * resolved, `f8_skip` the names (or `*suffix` patterns) ..._DENSE_FP8_EXCLUDE keeps on gemm_nt.
     * `f8_n` and `f8_codes`: the weights the op reads on this rank and their code bytes, the
     * shared expert's (MoeFP8::Config::shared_bf16_op) among them; `f8_kept`: those it could have
     * read that stay on gemm_nt (excluded, ineligible, or declined as a runtime-LoRA target). */
    bool f8_on = false;
    std::set<rad_weight> f8, f8_read;
    std::vector<std::string> f8_skip;
    int64_t f8_codes = 0;
    int f8_n = 0, f8_kept = 0;
    /* RADIANCE_EXPERIMENTAL_DS41_DSPARK_FP8=1 (dsp_f8_gate): the DSpark drafter's dense linears the
     * same way. The release ships its mtp.* stages' (wq_a, wq_b, wkv, wo_a, wo_b, the shared
     * expert, main_proj) as E4M3 with an E8M0 exponent a 32x32 block too, and the draft bank keeps
     * them cast to bf16, as the container keeps the trunk's. Needs ..._DENSE_FP8 on. Counted apart
     * from the trunk's, on a startup line of their own: `dsp_f8_n` weights the op reads,
     * `dsp_f8_codes` their code bytes, `dsp_f8_kept` those left on gemm_nt. */
    bool dsp_f8 = false;
    int64_t dsp_f8_codes = 0;
    int dsp_f8_n = 0, dsp_f8_kept = 0;
    int64_t win_bs = 0;
    /* model-wide weights and ops */
    rad_weight token = 0, head = 0, final_gain = 0, tmap = 0, ple_table = 0;
    /* The Engram table stored as affine planes (a GGUF's IQ4_NL): views gathered in place by
     * embed_lookup_planes, `ple_group` values a scale; 0 for a plain bf16 table. */
    rad_weight ple_codes = 0, ple_scale = 0, ple_tab = 0;
    int64_t ple_group = 0;
    /* ISTA controls use offsets local to each separately stored table. Keep
     * those integers unchanged and select that layer's own physical planes. */
    struct IstaTable {
        rad_weight plain = 0, codes = 0, scale = 0, tab = 0;
        int64_t group = 0;
    } ista_tables[2];
    rad_op embed = 0, expand = 0, pre0 = 0, pre1 = 0, rope_a = 0, rope_b_step = 0, tok_map = 0;
    rad_op contract = 0, fnorm = 0, gather = 0, head_op = 0, embed_ar = 0;
    rad_op cn_fin = 0;   /* contract + fnorm fused (fused()), or 0 */
    rad_op pcn_fin = 0;  /* the last post_f + contract + fnorm fused, or 0 */
    Dsp dsp{};
    /* The checkpoint is DeepSeek's release (is_release): the name map reads its tensors by
     * inference/model.py's names (ckpt_name), `n_stage` DSpark stages (mtp.*) after the trunk. */
    bool release = false;
    bool ista = false;
    bool local_engram = false;
    int n_stage = 0;
    /* The on-die reads' configuration (pf_env(), read at declare like ..._ENGRAM_SPLIT). */
    pf::Cfg pfc{};
    /* The on-die reads (Layer::pf) step() issued live, for the shutdown report (rad_plugin_close):
     * decode steps that issued any, issue points, reads and their bytes, and of those the reads
     * behind an Engram GEMM. Only this rank's thread writes them. A pass played back from a
     * recording (core/runtime/ctx.h) is not counted; a pass the host-expert stager arms never is
     * one (core/runtime/stager.h). */
    struct PfStats {
        uint64_t steps = 0, points = 0, reads = 0, bytes = 0, epoints = 0, ereads = 0, ebytes = 0;
    } pfs;
    /* The consumer skip (cut_declare): declared, at boundary `cut_b`; and what this rank's cut
     * passes ran (rad_plugin_close's report): passes cut, layer-passes in full, key ops alone,
     * skipped. Only this rank's thread writes them. */
    bool cut_on = false;
    int  cut_b = -1;
    /* The per-row cut (RadBatch::consumer_row0): the engine took the 16-row grain this plugin
     * declares (cut_declare), so a step may run each layer on its own suffix of the rows. */
    bool cut_rows = false;
    struct CutStats {
        uint64_t passes = 0, full = 0, key = 0, skipped = 0;
        /* Steps cut by row, and their layer-rows in full, key ops alone and skipped. */
        uint64_t row_passes = 0, rows_full = 0, rows_key = 0, rows_skipped = 0;
    } cuts;
};
static std::unique_ptr<Model> models[MAX_RANKS];
/* The model whose name map is being declared (declare() on this thread). */
static thread_local Model* naming = nullptr;

/* The checkpoint tensor a declared weight is read from. The declared names are llama.cpp's GGUF
 * names (`blk.N.attn_q_a.weight`, the drafter's stages as blk.{n_layer + s}), which a GGUF ships as
 * they are. DeepSeek's release ships the same tensors under inference/model.py's names, and
 * llama.cpp's converter (conversion/deepseek41.py, DeepseekV4Model._map_dsv4_tensor_name) only
 * renames them: no permutation, split or transpose. So this is a table of renames; the joins and
 * products the converter makes (the experts' stacks, the Engram tables, the Engram gate) are maps
 * of their own at their call sites. A name the table does not know is returned unchanged. */
static const char* ckpt_name(const char* declared) {
    Model* m = naming;
    if (!m) return declared;
    if (m->local_engram && !std::strcmp(declared,"deepseek41.engram.token_map"))
        return ista::token_map;
    if (m->ista) {
        if (!std::strcmp(declared, "deepseek41.engram.token_map")) return ista::token_map;
        int l = -1, at = 0;
        if (std::sscanf(declared, "blk.%d.%n", &l, &at) == 1 && at > 0 &&
            !std::strcmp(declared + at, "engram_kv.weight"))
            return m->nm.ckpt("blk.%d.engram_wkv.weight", l);
    }
    if (!m->release) return declared;
    static const struct { const char* gguf; const char* rel; } layer_t[] = {
        { "attn_norm.weight", "attn_norm.weight" },
        { "ffn_norm.weight", "ffn_norm.weight" },
        { "hc_attn_fn.weight", "hc_attn_fn" }, { "hc_attn_base.weight", "hc_attn_base" },
        { "hc_attn_scale.weight", "hc_attn_scale" }, { "hc_ffn_fn.weight", "hc_ffn_fn" },
        { "hc_ffn_base.weight", "hc_ffn_base" }, { "hc_ffn_scale.weight", "hc_ffn_scale" },
        { "attn_sinks.weight", "attn.attn_sink" },
        { "attn_q_a.weight", "attn.wq_a.weight" },
        { "attn_q_a_norm.weight", "attn.q_norm.weight" },
        { "attn_q_b.weight", "attn.wq_b.weight" },
        { "attn_kv.weight", "attn.wkv.weight" },
        { "attn_kv_a_norm.weight", "attn.kv_norm.weight" },
        { "attn_output_a.weight", "attn.wo_a.weight" },
        { "attn_output_b.weight", "attn.wo_b.weight" },
        { "attn_compressor_kv.weight", "attn.compressor.wkv.weight" },
        { "attn_compressor_gate.weight", "attn.compressor.wgate.weight" },
        { "attn_compressor_norm.weight", "attn.compressor.norm.weight" },
        { "indexer.attn_k.weight", "attn.indexer.wk.weight" },
        { "indexer.k_norm.weight", "attn.indexer.k_norm.weight" },
        { "indexer.attn_q_b.weight", "attn.indexer.wq_b.weight" },
        { "indexer.proj.weight", "attn.indexer.weights_proj.weight" },
        { "ffn_gate_inp.weight", "ffn.gate.weight" },
        { "exp_probs_b.bias", "ffn.gate.bias" },
        { "ffn_gate_shexp.weight", "ffn.shared_experts.w1.weight" },
        { "ffn_up_shexp.weight", "ffn.shared_experts.w3.weight" },
        { "ffn_down_shexp.weight", "ffn.shared_experts.w2.weight" },
        { "engram_kv.weight", "engram.wkv.weight" } };
    const int last = m->n_stage - 1;
    const std::string d = declared;
    if (d == "token_embd.weight") return "embed.weight";
    if (d == "output.weight") return "head.weight";
    if (d == "output_norm.weight") return "norm.weight";
    if (d == "deepseek41.engram.token_map") return "engram.token_map";
    if (d == "dspark.fc.weight") return "mtp.0.main_proj.weight";
    if (d == "dspark.enc.output_norm.weight") return "mtp.0.main_norm.weight";
    if (d == "dspark.output_norm.weight") return m->nm.ckpt("mtp.%d.norm.weight", last);
    if (d == "dspark.markov_w1.weight") return m->nm.ckpt("mtp.%d.markov_head.embed.weight", last);
    if (d == "dspark.markov_w2.weight") return m->nm.ckpt("mtp.%d.markov_head.head.weight", last);
    int l = -1, at = 0;
    if (std::sscanf(declared, "blk.%d.%n", &l, &at) != 1 || at <= 0) return declared;
    const char* suffix = declared + at;
    for (const auto& e : layer_t)
        if (std::strcmp(e.gguf, suffix) == 0)
            return l < m->c.n_layer ? m->nm.ckpt("layers.%d.%s", l, e.rel)
                                    : m->nm.ckpt("mtp.%d.%s", l - (int)m->c.n_layer, e.rel);
    return declared;
}

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
    if (map_copy(b, name, ckpt_name(name)) != RAD_OK) return 0;
    return decl_w(b, name, dt, shape, access, RAD_SHARD_NONE, grp);
}
/* A weight split over the ranks: this rank's [lo, hi) of its split dimension (rows for
 * RAD_SHARD_ROW, columns for RAD_SHARD_COL), declared at `shape`, which must already be that
 * extent. An explicit span because the attention world can be smaller than the world. */
static rad_weight wts(RadBuilder* b, const Geom& g, const char* name, uint32_t dt,
                      std::initializer_list<int64_t> shape, RadWeightGroup grp, int shard, int64_t lo,
                      int64_t hi, int access = RAD_ACCESS_PER_TOKEN) {
    if (map_copy(b, name, ckpt_name(name)) != RAD_OK) return 0;
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
                     bool mapped = false, int access = RAD_ACCESS_PER_TOKEN) {
    if (!mapped && map_copy(b, name, ckpt_name(name)) != RAD_OK) return 0;
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
                                access, split ? shard : RAD_SHARD_NONE, grp);
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
static rad_op fused(RadBuilder* b, const char* name, std::vector<RadParam> ps,
                    std::vector<rad_weight> ws, std::initializer_list<rad_buf> rd,
                    std::initializer_list<rad_buf> wr);
/* y = x . w^T over lw()'s weight: gemm_nt_ggml for a GGUF block weight, gemm_nt for bf16, and
 * for a weight lw8() marked gemm_nt_bf16f8, the same bytes from the weight's E4M3 + E8M0 stored
 * form (libr4d). That op resolves wherever a library declares it, but for a runtime-LoRA target,
 * whose fused GEMM the core declines; gemm_nt then reads the bf16 store, as without the knob. */
static rad_op linear(RadBuilder* b, Model& m, int64_t rows, rad_weight w, int64_t n,
                     int64_t k, rad_buf x, rad_buf y) {
    const auto t = m.wty.find(w);
    if (t != m.wty.end())
        return op(b, "gemm_nt_ggml",
                  { RAD_RANGE("M", 1, rows), RAD_INT("N", n), RAD_INT("K", k), RAD_STR("type", t->second),
                    RAD_STR("dtype", m.g.dtype) },
                  { w }, { x }, { y });
    if (m.f8.count(w)) {
        if (const rad_op h = fused(b, kDenseF8,
                                   { RAD_RANGE("M", 1, rows), RAD_INT("N", n), RAD_INT("K", k),
                                     RAD_STR("dtype", m.g.dtype) },
                                   { w }, { x }, { y })) {
            if (m.f8_read.insert(w).second) {
                ++m.f8_n;
                m.f8_codes += n * k;
            }
            return h;
        }
        ++m.f8_kept;
    }
    return op(b, "gemm_nt",
              { RAD_RANGE("M", 1, rows), RAD_INT("N", n), RAD_INT("K", k), RAD_STR("dtype", m.g.dtype) },
              { w }, { x }, { y });
}
/* Whether RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8_EXCLUDE names this weight: its declared name, or a
 * `*suffix` of it (`*_shexp.weight`: every shared expert). */
static bool f8_skipped(const Model& m, const char* name) {
    const size_t n = std::strlen(name);
    for (const std::string& x : m.f8_skip) {
        if (x.empty()) continue;
        if (x[0] == '*' ? (n >= x.size() - 1 && !std::strcmp(name + n - (x.size() - 1), x.c_str() + 1))
                        : x == name)
            return true;
    }
    return false;
}
/* lw() for a trunk linear the release ships as E4M3 codes with an E8M0 exponent a 32x32 block
 * (Model::f8). Marks the weight for gemm_nt_bf16f8 where everything the op's row and layout hook
 * check holds -- plain bf16 (LinearFP8's test), rows in 16s and K in 32s, a slice on whole 32x32
 * blocks -- so the op resolves wherever a library declares it, and the hook can refuse only a
 * weight whose values are not exactly so, at load. The scan ahead of time
 * (tools/ds41-host-experts/ds41_dense_fp8_scan) names those for ..._EXCLUDE; the release's 570
 * entries are all exact. */
static rad_weight lw8(RadBuilder* b, Model& m, const char* name, int64_t n, int64_t k,
                      RadWeightGroup grp, int shard = RAD_SHARD_NONE, int64_t lo = 0,
                      int64_t hi = 0, bool mapped = false) {
    const rad_weight w = lw(b, m, name, n, k, grp, shard, lo, hi, mapped);
    if (!w || !m.f8_on || m.wty.count(w)) return w;
    RadEncoding e{};
    const bool plain = weight_enc(b, name, &e) && rad_enc_is(&e, "plain") && e.n_planes == 1 &&
                       e.plane[0].dtype == RAD_BF16;
    const bool split = m.g.world > 1 && shard != RAD_SHARD_NONE;
    if (f8_skipped(m, name) || !plain || n % 16 || k % 32 || (split && (lo % 32 || hi % 32))) {
        ++m.f8_kept;
        return w;
    }
    m.f8.insert(w);
    return w;
}
/* The bytes a prefetch of linear() weight `w` [n, k] reads (pf_layer()): its E4M3 codes, a byte
 * an element, where gemm_nt_bf16f8 reads it (the exponents, one byte a 32x32 block, are left to
 * the GEMM; libr4d clamps a prefetch to the stored form either way), else the bf16 plane's two; 0
 * for a weight in llama.cpp's blocks (a GGUF's), which no prefetch reads. */
static int64_t wbytes(const Model& m, rad_weight w, int64_t n, int64_t k) {
    if (!w || m.wty.count(w)) return 0;
    return m.f8_read.count(w) ? n * k : n * k * 2;
}
/* With ROUTER_BF16 (Model::router_bf16): whether gemm_nt_f32bf16 holds the weight declared as
 * `name` at two bytes an element -- an f32 plane its layout keeps as the bf16 halves, or a bf16
 * plane read as it is. An f16 plane is widened to f32 by the loader and read at four, and a plane
 * whose encoding is unknown is taken to be so too (r4d_layout.cpp, f32bf16). It decides what the
 * startup line counts and the bytes the on-die reads of the router and the compressor's kv|gate
 * take (pf_layer). */
static bool bf16_held(RadBuilder* b, const Model& m, const char* name) {
    RadEncoding e{};
    return m.router_bf16 && weight_enc(b, name, &e) && rad_enc_is(&e, "plain") && e.n_planes == 1 &&
           (e.plane[0].dtype == RAD_F32 || e.plane[0].dtype == RAD_BF16);
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
static rad_op roundtrip(RadBuilder* b, int64_t rows, int64_t n, int64_t block, const char* fmt,
                        const char* scale, rad_buf x) {
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

/* ---- fused chains (docs/OPS.md rmsnorm_rope_roundtrip, rope_rows_roundtrip,
 * mhc_collapse_rmsnorm): one launch for a chain of two or three dependent ops on the same rows,
 * the same bytes as the chain (a kernel library's `replaces`). A decode step pays a fixed cost per
 * dependent launch, so this is most of what the small ops cost there. They are device-only (libr4d
 * declares them; libref has no row): each declares the fused op only where a loaded library
 * declares and serves it, and returns 0 otherwise, and the caller then declares the chain. */
static rad_op fused(RadBuilder* b, const char* name, std::vector<RadParam> ps,
                    std::vector<rad_weight> ws, std::initializer_list<rad_buf> rd,
                    std::initializer_list<rad_buf> wr) {
    if (!rad_op_declared(b, name)) return 0;
    const rad_op h = rad_decl_op(b, name, ps.data(), (int)ps.size(),
                                 ws.empty() ? nullptr : ws.data(), (int)ws.size());
    if (!h || !rad_op_resolved(b, h)) return 0;
    return rw(b, h, rd, wr);
}
/* rmsnorm (x, w -> y) then rope_rows and roundtrip over y, as one op: one head of n columns. */
static rad_op norm_rope_rt(RadBuilder* b, const Model& m, int64_t rows, rad_weight w, int64_t n,
                           int64_t block, const char* fmt, const char* scale, rad_buf x, rad_buf y,
                           rad_buf cs) {
    return fused(b, "rmsnorm_rope_roundtrip",
                 { RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_F64("eps", m.c.eps),
                   RAD_INT("n_head", 1), RAD_INT("head_dim", n), RAD_INT("rot", m.c.rope),
                   RAD_INT("off", n - m.c.rope), RAD_STR("mode", "gptj"), RAD_INT("inverse", 0),
                   RAD_INT("block", block), RAD_STR("format", fmt), RAD_STR("scale", scale),
                   RAD_STR("dtype", "bf16") },
                 { w }, { x, cs }, { y });
}
/* rope_rows then roundtrip over all heads * hd columns of x, in place, as one op. */
static rad_op rope_rt(RadBuilder* b, const Model& m, int64_t rows, int64_t heads, int64_t hd,
                      int inverse, int64_t block, const char* fmt, const char* scale, rad_buf x,
                      rad_buf cs) {
    return fused(b, "rope_rows_roundtrip",
                 { RAD_RANGE("M", 1, rows), RAD_INT("n_head", heads), RAD_INT("head_dim", hd),
                   RAD_INT("rot", m.c.rope), RAD_INT("off", hd - m.c.rope), RAD_STR("mode", "gptj"),
                   RAD_INT("inverse", inverse), RAD_INT("block", block), RAD_STR("format", fmt),
                   RAD_STR("scale", scale), RAD_STR("dtype", "bf16") },
                 {}, { x, cs }, { x });
}
/* mhc_collapse (h, pre -> x) then rmsnorm (x, w -> y), as one op. */
static rad_op collapse_norm(RadBuilder* b, const Model& m, int64_t rows, rad_weight w, rad_buf h,
                            rad_buf pre, rad_buf x, rad_buf y) {
    return fused(b, "mhc_collapse_rmsnorm",
                 { RAD_RANGE("M", 1, rows), RAD_INT("n", m.c.n_embd), RAD_INT("hc", m.c.hc),
                   RAD_F64("eps", m.c.eps), RAD_STR("dtype", "bf16") },
                 { w }, { h, pre }, { x, y });
}
/* mhc_post (in, h, post, comb -> out), mhc_collapse (out, pre -> x), rmsnorm (x, w -> y), as one
 * op. */
static rad_op post_collapse_norm(RadBuilder* b, const Model& m, int64_t rows, rad_weight w,
                                 rad_buf in, rad_buf h, rad_buf out, rad_buf pre, rad_buf x,
                                 rad_buf y) {
    return fused(b, "mhc_post_collapse_rmsnorm",
                 { RAD_RANGE("M", 1, rows), RAD_INT("n", m.c.n_embd), RAD_INT("hc", m.c.hc),
                   RAD_F64("eps", m.c.eps), RAD_STR("dtype", "bf16") },
                 { w }, { in, h, m.post, m.comb, pre }, { out, x, y });
}

/* The shared expert is split by width, and a rank's K slice of its down projection must fall on
 * the encoding's block boundary along K: 256 columns for the GGUF k- and i-quant superblocks, 128 for
 * block fp8, 32 for legacy GGUF and MXFP4, 1 for bf16. A slice on a 256-column boundary is on every
 * one's. The even split is kept wherever each rank's share is whole 256-column blocks (and wherever
 * the width is no whole number of them, where the encoding decides at declare, as before); otherwise
 * the width splits by whole blocks, the first nb % world ranks one block more. DeepSeek-V4.1-Flash's
 * 2304 columns are nine blocks: three ranks take three each, and four ranks 768/512/512/512 where
 * the even 576 would cut a superblock. Each rank computes its gate/up columns and K slice of down,
 * and the all-reduce sums the partials (MoeFP8::Config::shared_lo). */
static constexpr int64_t kSharedBlock = 256;

static void shared_slice(int64_t n_ff, int world, int rank, int64_t* lo, int64_t* hi) {
    *lo = *hi = 0;
    if (world <= 1 || n_ff % kSharedBlock) return;
    if (n_ff % world == 0 && (n_ff / world) % kSharedBlock == 0) return;
    const int64_t nb = n_ff / kSharedBlock, base = nb / world, extra = nb % world;
    const int64_t b0 = rank * base + std::min<int64_t>(rank, extra);
    *lo = b0 * kSharedBlock;
    *hi = (b0 + base + (rank < extra ? 1 : 0)) * kSharedBlock;
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
    int64_t sh_lo = 0, sh_hi = 0;
    shared_slice(c.n_ff, world, ctx->rank, &sh_lo, &sh_hi);
    if (c.n_expert % world || (c.n_expert / world) % 2 || (sh_hi == 0 && c.n_ff % world)) {
        fprintf(stderr, "radiance: deepseek41: %lld experts of width %lld do not split over %d ranks "
                        "(an even count of whole experts each, and the shared expert's width)\n",
                (long long)c.n_expert, (long long)c.n_ff, world);
        return RAD_E_UNSUPPORTED;
    }
    g.n_ff_all = c.n_ff;
    /* This rank's width of the shared expert (its buffers and the MoE block's n_ff_shared). */
    g.n_ff = sh_hi ? sh_hi - sh_lo : c.n_ff / world;
    m.sh_lo = sh_lo;
    m.sh_hi = sh_hi;
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

/* index_candidates' keep words a row (docs/OPS.md): a bit a block of `block` positions. */
static int64_t keep_words(int64_t width, int64_t block) {
    return block < 1 || width < 1 ? 0 : ((width + block - 1) / block + 31) / 32;
}

/* The indexer's score plane is a row's score of every compressed position, [rows, cmax] f32 with
 * cmax = max_ctx / (the smallest ratio) + 1: at ratio 1, 8192-row chunks and 256k of context a
 * whole step's plane would be 8 GiB, and the candidates' mask (live from the candidate layer to the
 * last that reads it) as much again. So a step is scored in runs of rows into one plane of at most
 * kScorePlane bytes (but never fewer than kScoreRunMin rows), each run selected from before the next
 * overwrites it, and the mask is a bit a block (index_candidates).
 *
 * A step of T rows takes n = ceil(T / irows) runs of R = ceil(T / n) rows. Rows are independent,
 * so every run is the unsplit op's rows bit for bit, provided each run is scored by the kernel the
 * whole step would have taken: kernels pick their form by the pass height (libr4d's index_score
 * runs WMMA above 48 rows, a different dot order from the decode form's), and a run of a split
 * step is over irows / 2 >= 128 rows, as tall as any form's threshold. A step of at most irows
 * rows is one run, the op as before. */
static constexpr int64_t kScorePlane = 256ll << 20;
static constexpr int64_t kScoreRunMin = 256;
static void index_runs(Model& m) {
    const auto& g = m.g;
    const auto& c = m.c;
    m.irows = std::min(g.max_tok, std::max(kScoreRunMin, kScorePlane / (4 * m.cmax)));
    m.iparts = (g.max_tok + m.irows - 1) / m.irows;
    m.kwords = 0;
    if (c.cand >= 0 && c.cand < c.n_layer && c.cand_block > 0 && c.ratio[(size_t)c.cand] > 0)
        m.kwords = keep_words(g.max_ctx / c.ratio[(size_t)c.cand] + 1, c.cand_block);
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
    m.score = decl_b(b, nm.f("index_score"), RAD_F32, { m.irows, m.cmax });
    if (m.kwords > 0) m.keep = decl_b(b, nm.f("index_keep"), RAD_I32, { T, m.kwords });
    m.sel = decl_b(b, nm.f("index_sel"), RAD_I32, { T, c.itopk });
    m.cs_a = decl_b(b, nm.f("rope_window"), RAD_F32, { g.max_ctx, c.rope }, RAD_BUF_PERSIST);
    m.cs_b = decl_b(b, nm.f("rope_compressed"), RAD_F32, { g.max_ctx, c.rope }, RAD_BUF_PERSIST);
    if (!c.ple.empty()) {
        m.ctok = decl_b(b, nm.f("engram_tokens"), RAD_I32, { T });
        /* The gathered rows are host domain, as rad_block_ple.h's: a host gather (the table on the
         * disk or RAM tier) whose operands are all host-arena buffers joins on the last device op
         * that touched the host arena, where a device-domain operand drains the stream
         * (core/runtime/issue.cpp). A cast copies them onto the card for the GEMM, which reads
         * each row band once per column block. */
        m.emb = decl_b(b, nm.f("engram_rows"), RAD_BF16, { T, c.ple_heads * c.ple_dim },
                       RAD_BUF_TRANSIENT, RAD_DOMAIN_HOST);
        m.embd = bf("engram_rows_dev", { T, c.ple_heads * c.ple_dim });
        m.ekv = bf("engram_kv", { T, (c.hc + 1) * D });
        /* RADIANCE_EXPERIMENTAL_DS41_ENGRAM_SPLIT=1: each rank holds and computes 1/world of
         * engram_kv's rows (a 315 MB GEMV every rank otherwise repeats) and a row-interleaved
         * all_gather lays the slices out as the full row. Each output column is computed exactly
         * as before (one wave a 16-column tile walks K in the same order, split-K 1 at either
         * width) and the gather only moves bytes, so the values are bit-identical. */
        const int64_t en = (c.hc + 1) * D;
        if (const char* es = std::getenv("RADIANCE_EXPERIMENTAL_DS41_ENGRAM_SPLIT");
            es && !std::strcmp(es, "1") && g.world > 1 && m.aw == g.world && !m.zero &&
            en % g.world == 0 && (en / g.world) % 32 == 0) {
            m.esplit = en / g.world;
            m.ekv_part = bf("engram_kv_part", { T, m.esplit });
        }
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
    /* the shared expert: this rank's width; none where its slice is empty (shared_slice) */
    if (g.n_ff > 0) {
        mw.sgu = bf("shared_gate_up", { T, 2 * g.n_ff });
        mw.sff.x = bf("shared_ff", { T, g.n_ff });
    }
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

/* The routed FFN's checkpoint tensors at (absolute) layer `l`: a GGUF's stacks (ffn_gate_exps,
 * ffn_up_exps, ffn_down_exps, each [n_expert, ...]), or the release's experts one by one
 * (`<layer>.ffn.experts.<e>.w1/w3/w2`, gate/up/down as model.py's Expert names them). */
static void moe_names(Model& m, int l, MoeFP8::Src* s) {
    auto& nm = m.nm;
    s->router = ckpt_name(nm.f("blk.%d.ffn_gate_inp.weight", l));
    s->router_bias = ckpt_name(nm.f("blk.%d.exp_probs_b.bias", l));
    s->shared_mlp_gate = ckpt_name(nm.f("blk.%d.ffn_gate_shexp.weight", l));
    s->shared_mlp_up = ckpt_name(nm.f("blk.%d.ffn_up_shexp.weight", l));
    s->shared_mlp_down = ckpt_name(nm.f("blk.%d.ffn_down_shexp.weight", l));
    if (m.release) {
        s->experts = l < m.c.n_layer ? nm.ckpt("layers.%d.ffn.experts", l)
                                     : nm.ckpt("mtp.%d.ffn.experts", l - (int)m.c.n_layer);
        s->expert_gate = "w1";
        s->expert_up = "w3";
        s->expert_down = "w2";
        return;
    }
    s->experts_gate = nm.f("blk.%d.ffn_gate_exps.weight", l);
    s->experts_up = nm.f("blk.%d.ffn_up_exps.weight", l);
    s->experts_down = nm.f("blk.%d.ffn_down_exps.weight", l);
}

/* Layer `L`'s reads into the card's on-die cache (Layer::pf, ds41_prefetch_plan.h), declared once
 * its MoE block is: each listed kind's weights on this rank in the configured order, at the bytes
 * their stored form holds, the reads of each issue point cut to the budget. A weight the layer
 * does not have (no compressor away from a kv source, none of the attention on an attention-zero
 * rank) is no read, and the next kind goes in its place. An Engram layer is read only with
 * ..._PREFETCH_ENGRAM=1: its engram_kv by the layer before (pf: its GEMM is the layer's first
 * read), everything else behind that GEMM (pf_post), which streams 79 MB a card (split over the
 * ranks; 315 MB whole) and would evict what was read before it. Layer 0's `pf` would have no
 * layer before it to issue it, so it gets none. */
static void pf_layer(RadBuilder* b, Model& m, Layer& L) {
    const pf::Cfg& pc = m.pfc;
    if (!pc.on) return;
    /* A GGUF trunk (llama.cpp's blocks), as before: no reads. */
    if (L.wq_a && m.wty.count(L.wq_a)) return;
    const bool eng = L.e_gemm != 0;
    if (eng && !pc.engram) return;
    const auto& c = m.c;
    const auto& g = m.g;
    const int64_t D = c.n_embd, hd = c.head_dim, HD = g.n_head * c.head_dim;
    const int64_t mix = c.hc * (c.hc + 2);
    /* [0] issued by the layer before at its MoE tail, [1] behind this layer's Engram GEMM. */
    std::vector<pf::Cand> ca[2];
    std::vector<rad_weight> cw[2];
    const auto put = [&](int k, rad_weight w, int64_t bytes) {
        if (!w || bytes <= 0) return;
        const int at = eng && k != pf::kEkv ? 1 : 0;
        ca[at].push_back({ k, bytes });
        cw[at].push_back(w);
    };
    const LinearFP8& sgu = L.moe.sh_gate_up;
    const LinearFP8& sdn = L.moe.sh_down;
    for (const int k : pc.order) {
        switch (k) {
        case pf::kEkv:
            if (eng)
                put(k, L.e_kv,
                    wbytes(m, L.e_kv, m.esplit ? m.esplit : (c.hc + 1) * D, c.ple_heads * c.ple_dim));
            break;
        case pf::kComp:   /* f32 [2 hd, D] (kv|gate) at a ratio above 1, half that where ROUTER_BF16
                           * holds it as bf16 (Layer::comp_bf16), else lw()'s [hd, D] */
            put(k, L.comp_w,
                L.r > 1 ? 2 * hd * D * (L.comp_bf16 ? 2 : 4) : wbytes(m, L.comp_w, hd, D));
            break;
        case pf::kQa: put(k, L.wq_a, wbytes(m, L.wq_a, c.q_lora, D)); break;
        case pf::kKv: put(k, L.wkv, wbytes(m, L.wkv, hd, D)); break;
        case pf::kIw: put(k, L.iproj, wbytes(m, L.iproj, c.ih, D)); break;
        case pf::kIq: put(k, L.iq_w, wbytes(m, L.iq_w, c.ih * c.idim, c.q_lora)); break;
        case pf::kQb: put(k, L.wq_b, wbytes(m, L.wq_b, HD, c.q_lora)); break;
        case pf::kIk: put(k, L.ik_w, wbytes(m, L.ik_w, c.idim, hd)); break;
        case pf::kOa:
            for (const rad_weight wa : L.wo_a) put(k, wa, wbytes(m, wa, c.o_rank, HD / m.gr));
            break;
        case pf::kMa: put(k, L.hc_a_fn, mix * c.hc * D * 4); break;   /* f32 [mix, hc D] */
        case pf::kOb: put(k, L.wo_b, wbytes(m, L.wo_b, D, m.gr * c.o_rank)); break;
        case pf::kRt:   /* GlmRouter's f32 projection under the sigmoid router (bf16 where
                         * ROUTER_BF16 holds it so, Layer::rt_bf16), else bf16 */
            put(k, L.moe.w_router,
                L.moe.c.n_expert_all * D * (L.moe.c.sigmoid_router && !L.rt_bf16 ? 4 : 2));
            break;
        /* This rank's slice of the shared expert: bf16, or its E4M3 codes where gemm_nt_bf16f8
         * reads it (LinearFP8::bf16_alt, ..._DENSE_FP8), as wbytes() counts linear()'s. */
        case pf::kSgu: if (sgu.bf16) put(k, sgu.w, sgu.N * sgu.K * (sgu.bf16_alt ? 1 : 2)); break;
        case pf::kMf: put(k, L.hc_f_fn, mix * c.hc * D * 4); break;
        case pf::kSdn: if (sdn.bf16) put(k, sdn.w, sdn.N * sdn.K * (sdn.bf16_alt ? 1 : 2)); break;
        default: break;
        }
    }
    for (int at = (L.l == 0 ? 1 : 0); at < 2; ++at)
        for (const pf::Read& r : pf::fit(ca[at], pc.budget)) {
            const rad_weight w = cw[at][r.cand];
            if (const rad_op h = decl_prefetch(b, w, r.bytes))
                (at ? L.pf_post : L.pf).push_back({ h, w, r.bytes, ca[at][r.cand].kind, r.cut });
        }
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
        const auto* et = m.local_engram ? &m.ista_tables[pi] : nullptr;
        const int64_t erows = m.local_engram ? ista::layer_rows[pi] : c.ple_rows;
        const int64_t egroup = et ? et->group : m.ple_group;
        const rad_weight eplain = et ? et->plain : m.ple_table;
        const rad_weight ecodes = et ? et->codes : m.ple_codes;
        const rad_weight escale = et ? et->scale : m.ple_scale;
        const rad_weight etab = et ? et->tab : m.ple_tab;
        /* The hash constants: GGUF metadata arrays (llama.cpp's converter computes them), or the
         * release's, which the core derives from its config.json and tokenizer as
         * inference/engram.py does at load (core/format/engram.h). One layout either way. */
        if (m.local_engram) {
            DS_TRY(map_slice(b, nm.f("blk.%d.engram_mult", l), ista::multipliers, pi));
            DS_TRY(map_slice(b, nm.f("blk.%d.engram_vsz", l), ista::primes, pi));
            DS_TRY(map_slice(b, nm.f("blk.%d.engram_off", l), ista::offsets, pi));
        } else {
            const char* tab = m.release ? "engram." : "deepseek41.ple.";
            DS_TRY(map_rows(b, nm.f("blk.%d.engram_mult", l),
                            { { nm.ckpt("%slayer_multipliers", tab), (int)(pi * c.ngram), c.ngram } }));
            DS_TRY(map_rows(b, nm.f("blk.%d.engram_vsz", l),
                            { { nm.ckpt("%shead_vocab_sizes", tab), (int)(pi * H), H } }));
            DS_TRY(map_rows(b, nm.f("blk.%d.engram_off", l),
                            { { nm.ckpt("%shead_offsets", tab), (int)(pi * H), H } }));
        }
        L.e_mult = decl_w(b, nm.f("blk.%d.engram_mult", l), RAD_I64, { c.ngram },
                          RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp);
        L.e_vsz = decl_w(b, nm.f("blk.%d.engram_vsz", l), RAD_I64, { H }, RAD_ACCESS_PER_TOKEN,
                         RAD_SHARD_NONE, grp);
        L.e_off = decl_w(b, nm.f("blk.%d.engram_off", l), RAD_I64, { H }, RAD_ACCESS_PER_TOKEN,
                         RAD_SHARD_NONE, grp);
        L.e_kv = m.esplit ? lw8(b, m, nm.f("blk.%d.engram_kv.weight", l), m.esplit, H * c.ple_dim, grp,
                                RAD_SHARD_ROW, m.ar * m.esplit, (m.ar + 1) * m.esplit)
                          : lw8(b, m, nm.f("blk.%d.engram_kv.weight", l), (c.hc + 1) * D, H * c.ple_dim, grp);
        if (m.release || m.ista) {
            /* The release ships the gate as two factors, k_weight and q_weight, that the model only
             * multiplies (model.py Engram): one gain, as llama.cpp's converter folds it. */
            const char* gn = nm.f("blk.%d.engram_gate.weight", l);
            DS_TRY(map_mul(b, gn,
                           m.ista ? nm.ckpt("blk.%d.engram_q.weight", l)
                                  : nm.ckpt("layers.%d.engram.q_weight", l),
                           m.ista ? nm.ckpt("blk.%d.engram_k.weight", l)
                                  : nm.ckpt("layers.%d.engram.k_weight", l)));
            L.e_gate = decl_w(b, gn, RAD_F32, { c.hc * D }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp);
        } else {
            L.e_gate = w("blk.%d.engram_gate.weight", RAD_F32, { c.hc * D });
        }
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
        /* Host domain, like the rows (see m.emb), and persistent rather than liveness-shared:
         * liveness runs over the declared op order, and step() writes these ids before the first
         * block, far ahead of this layer's gather. */
        L.ids = decl_b(b, nm.f("blk.%d.engram_ids", l), RAD_I32, { T, H }, RAD_BUF_PERSIST,
                       RAD_DOMAIN_HOST);
        L.e_ids = op(b, "ngram_ids",
                     { RAD_RANGE("M", 1, T), RAD_INT("heads", H), RAD_INT("ngram", c.ngram),
                       RAD_INT("eos", c.pad) },
                     { L.e_mult, L.e_vsz, L.e_off }, { m.ctok }, { L.ids });
        if (egroup > 0 && etab)
            L.e_emb = op(b, "embed_lookup_planes",
                         { RAD_RANGE("M", 1, T * H), RAD_INT("n_embd", c.ple_dim),
                           RAD_INT("n_vocab", erows), RAD_INT("group", egroup),
                           RAD_STR("dtype", "bf16") },
                         { ecodes, escale, etab }, { L.ids }, { m.emb });
        else if (egroup > 0)
            L.e_emb = op(b, "embed_lookup_planes",
                         { RAD_RANGE("M", 1, T * H), RAD_INT("n_embd", c.ple_dim),
                           RAD_INT("n_vocab", erows), RAD_INT("group", egroup),
                           RAD_STR("dtype", "bf16") },
                         { ecodes, escale }, { L.ids }, { m.emb });
        else
            L.e_emb = op(b, "embed_lookup",
                         { RAD_RANGE("M", 1, T * H), RAD_INT("n_embd", c.ple_dim),
                           RAD_INT("n_vocab", erows), RAD_STR("dtype", "bf16") },
                         { eplain }, { L.ids }, { m.emb });
        L.e_copy = op(b, "cast",
                      { RAD_RANGE("M", 1, T), RAD_INT("n", H * c.ple_dim), RAD_STR("from", "bf16"),
                        RAD_STR("to", "bf16") },
                      {}, { m.emb }, { m.embd });
        if (m.esplit) {
            L.e_gemm = linear(b, m, T, L.e_kv, m.esplit, H * c.ple_dim, m.embd, m.ekv_part);
            L.e_gather = op(b, "all_gather",
                            { RAD_INT("world_size", g.world), RAD_RANGE("numel", m.esplit, T * m.esplit),
                              RAD_STR("dtype", "bf16"), RAD_INT("row", m.esplit) },
                            {}, { m.ekv_part }, { m.ekv });
            if (!L.e_gather) return RAD_E_INVAL;
        } else {
            L.e_gemm = linear(b, m, T, L.e_kv, (c.hc + 1) * D, H * c.ple_dim, m.embd, m.ekv);
        }
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

    /* ---- a DSpark target: the attention input's hc mean (model.py main_hiddens), after the Engram */
    if (m.dsp.on && !m.zero)   /* the context pass is an attention rank's */
        for (size_t j = 0; j < m.dsp.taps.size(); ++j)
            if (m.dsp.taps[j] == l) {
                L.tap_col = (int)j;
                L.tap = op(b, "mhc_contract",
                           { RAD_RANGE("M", 1, T), RAD_INT("n", D), RAD_INT("hc", c.hc), RAD_STR("dtype", "bf16") },
                           {}, { m.h0 }, { m.dsp.src });
                if (!L.tap) return RAD_E_INVAL;
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
    if (!m.zero) {
        L.attn_norm = w("blk.%d.attn_norm.weight", RAD_F32, { D });
        /* The previous layer's post_f writes h0 straight into this collapse unless an Engram
         * block (above) rewrites h0 between them. */
        if (l > 0 && !L.e_ids)
            L.pcn_a = post_collapse_norm(b, m, T, L.attn_norm, m.act.x, m.h1, m.h0, m.pre_f,
                                         m.xcol, m.xn);
        if (!L.pcn_a) L.cn_a = collapse_norm(b, m, T, L.attn_norm, m.h0, m.pre_f, m.xcol, m.xn);
    }
    if (!L.cn_a && !L.pcn_a) L.col_a = op(b, "mhc_collapse", hcp, {}, { m.h0, m.pre_f }, { m.xcol });
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
        if (!L.cn_a && !L.pcn_a) L.norm_a = rmsnorm(b, m, T, L.attn_norm, D, m.xcol, m.xn);

        /* ---- attention: the query, the window key */
        L.wq_a = lw8(b, m, nm.f("blk.%d.attn_q_a.weight", l), c.q_lora, D, grp);
        L.q_norm = w("blk.%d.attn_q_a_norm.weight", RAD_F32, { c.q_lora });
        L.wq_b = lw8(b, m, nm.f("blk.%d.attn_q_b.weight", l), HD, c.q_lora, grp, RAD_SHARD_ROW,
                     m.ar * HD, (m.ar + 1) * HD);
        L.q_a = linear(b, m, T, L.wq_a, c.q_lora, D, m.xn, m.qr);
        L.q_an = rmsnorm(b, m, T, L.q_norm, c.q_lora, m.qr, m.qrn);
        L.q_b = linear(b, m, T, L.wq_b, HD, c.q_lora, m.qrn, m.q);
        const rad_buf cs = L.r > 0 ? m.cs_b : m.cs_a;
        L.q_rope = rope_rows(b, m, T, g.n_head, hd, 0, m.q, cs);
        L.wkv = lw8(b, m, nm.f("blk.%d.attn_kv.weight", l), hd, D, grp);
        L.kv_norm = w("blk.%d.attn_kv_a_norm.weight", RAD_F32, { hd });
        L.kv_g = linear(b, m, T, L.wkv, hd, D, m.xn, m.kv);
        L.kv_f = norm_rope_rt(b, m, T, L.kv_norm, hd, 32, "fp8_e4m3", "ue8m0", m.kv, m.kvn, cs);
        if (!L.kv_f) {
            L.kv_n = rmsnorm(b, m, T, L.kv_norm, hd, m.kv, m.kvn);
            L.kv_rope = rope_rows(b, m, T, 1, hd, 0, m.kvn, cs);
            L.kv_q = roundtrip(b, T, hd, 32, "fp8_e4m3", "ue8m0", m.kvn);
        }
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
                DS_TRY(map_concat(b, cn, { ckpt_name(nm.f("blk.%d.attn_compressor_kv.weight", l)),
                                           ckpt_name(nm.f("blk.%d.attn_compressor_gate.weight", l)) }));
                /* f32 in gemm_nt_f32 (the container's bf16, widened at load); with ROUTER_BF16
                 * (router_bf16_on) declared at the source's dtype, as GlmRouter declares a
                 * router's: bf16 here, read as it is. */
                const char* cg = m.router_bf16 ? "gemm_nt_f32bf16" : "gemm_nt_f32";
                uint32_t cdt = RAD_F32;
                RadEncoding ce{};
                if (m.router_bf16 && weight_enc(b, cn, &ce) && ce.n_planes == 1 &&
                    ce.plane[0].dtype == RAD_BF16)
                    cdt = RAD_BF16;
                L.comp_w = decl_w(b, cn, cdt, { 2 * hd, D }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                                  grp);
                L.c_gemm = op(b, cg,
                              { RAD_RANGE("M", 1, T), RAD_INT("N", 2 * hd), RAD_INT("K", D),
                                RAD_STR("dtype", "bf16") },
                              { L.comp_w }, { m.xn }, { m.kvs });
                L.comp_bf16 = bf16_held(b, m, cn);
                if (L.comp_bf16) {
                    ++m.n_comp_bf16;
                    m.bf16_saved += 2 * hd * D * 2;
                }
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
            L.k_f = norm_rope_rt(b, m, rows, L.ik_norm, c.idim, 32, "fp4_e2m1", "ue8m0", m.ik, m.ikn,
                                 m.cs_b);
            if (!L.k_f) {
                L.k_norm = rmsnorm(b, m, rows, L.ik_norm, c.idim, m.ik, m.ikn);
                L.k_rope = rope_rows(b, m, rows, 1, c.idim, 0, m.ikn, m.cs_b);
                L.k_q = roundtrip(b, rows, c.idim, 32, "fp4_e2m1", "ue8m0", m.ikn);
            }
            if (!m.index.count(r) && !(m.index[r] = full_group(b, m, "kv_index_r%lld", c.idim, r))) return RAD_E_INVAL;
            L.k_st = entry_store(b, rows, c.idim, r, m.ikn, m.page);
            if (rad_kv_block_size(b, m.index[r]) != r) return RAD_E_INVAL;
            DS_TRY(rad_bind_layer_kv(b, l, m.index[r]));
            L.l_f = rope_rt(b, m, rows, 1, hd, 0, 16, "fp4_e2m1", "e4m3", m.lat, m.cs_b);
            if (!L.l_f) {
                L.l_rope = rope_rows(b, m, rows, 1, hd, 0, m.lat, m.cs_b);
                L.l_q = roundtrip(b, rows, hd, 16, "fp4_e2m1", "e4m3", m.lat);
            }
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
            L.iq_w = lw8(b, m, nm.f("blk.%d.indexer.attn_q_b.weight", l), c.ih * c.idim, c.q_lora, grp);
            L.iproj = lw(b, m, nm.f("blk.%d.indexer.proj.weight", l), c.ih, D, grp);
            L.i_q = linear(b, m, T, L.iq_w, c.ih * c.idim, c.q_lora, m.qrn, m.iq);
            L.i_f = rope_rt(b, m, T, c.ih, c.idim, 0, 32, "fp4_e2m1", "ue8m0", m.iq, m.cs_b);
            if (!L.i_f) {
                L.i_rope = rope_rows(b, m, T, c.ih, c.idim, 0, m.iq, m.cs_b);
                L.i_qq = roundtrip(b, T, c.ih * c.idim, 32, "fp4_e2m1", "ue8m0", m.iq);
            }
            L.i_w = linear(b, m, T, L.iproj, c.ih, D, m.xn, m.iw);
            const int64_t width = g.max_ctx / r + 1, IR = m.irows;
            /* One index_score a run of rows (index_runs()): run p scores step rows [p R, p R + R)
             * into the score plane's first R rows. */
            for (int64_t p = 0; p < m.iparts; ++p) {
                const rad_op h = op(b, "index_score",
                                    { RAD_RANGE("M", 1, IR), RAD_INT("heads", c.ih), RAD_INT("dim", c.idim),
                                      RAD_INT("ratio", r), RAD_INT("width", width),
                                      RAD_F64("wscale", 1.0 / std::sqrt((double)c.idim * (double)c.ih)),
                                      RAD_STR("dtype", "bf16"), RAD_INT("row_part", p) },
                                    {}, { m.iq, m.iw }, { m.score });
                if (!h) return RAD_E_INVAL;
                L.i_runs.push_back(h);
            }
            L.i_score = L.i_runs[0];
            if (l == c.cand) {
                if (!m.keep) return RAD_E_INVAL;
                L.i_cand = op(b, "index_candidates",
                              { RAD_RANGE("M", 1, IR), RAD_INT("width", width),
                                RAD_INT("block", c.cand_block), RAD_INT("topk_blocks", c.cand_blocks),
                                RAD_INT("ratio", r) },
                              {}, { m.score }, { m.keep });
                if (!L.i_cand) return RAD_E_INVAL;
            }
            const bool uses = c.cand >= 0 && c.cand < l;
            if (uses && (!m.keep || c.ratio[(size_t)c.cand] != r)) {
                fprintf(stderr, "radiance: deepseek41: layer %d reads the candidate layer's blocks at "
                                "another ratio (%lld) or there are none\n", l, (long long)r);
                return RAD_E_UNSUPPORTED;
            }
            std::vector<RadParam> tp = { RAD_RANGE("M", 1, IR), RAD_INT("width", width),
                                         RAD_INT("topk", c.itopk), RAD_INT("ratio", r) };
            if (uses) tp.push_back(RAD_INT("block", c.cand_block));
            L.i_topk = op(b, "index_topk", tp, {},
                          uses ? std::initializer_list<rad_buf>{ m.score, m.keep }
                               : std::initializer_list<rad_buf>{ m.score },
                          { m.sel });
            if (!L.i_topk) return RAD_E_INVAL;
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
        L.o_f = rope_rt(b, m, T, g.n_head, hd, 1, 32, "fp8_e4m3", "ue8m0", m.o, cs);
        if (!L.o_f) {
            L.o_rope = rope_rows(b, m, T, g.n_head, hd, 1, m.o, cs);
            L.o_q = roundtrip(b, T, HD, 32, "fp8_e4m3", "ue8m0", m.o);
        }
        /* This rank's output groups, each a block of the block-diagonal wo_a. */
        const int64_t gd = HD / m.gr;
        for (int j = 0; j < m.gr; ++j) {
            const int gi_ = (int)(m.ar * m.gr + j);
            const char* n = nm.f("blk.%d.attn_output_a.g%d.weight", l, gi_);
            if (m.ista)
                DS_TRY(map_slice(b, n, nm.f("blk.%d.attn_output_a.weight", l), gi_));
            else
                DS_TRY(map_rows(b, n, { { ckpt_name(nm.f("blk.%d.attn_output_a.weight", l)), (int)(gi_ * c.o_rank),
                                                c.o_rank } }));
            L.wo_a.push_back(lw8(b, m, n, c.o_rank, gd, grp, RAD_SHARD_NONE, 0, 0, true));
            if (!L.wo_a.back()) return RAD_E_INVAL;
            L.o_a.push_back(linear(b, m, T, L.wo_a.back(), c.o_rank, gd, m.o, m.oa));
        }
        const int64_t ok = m.gr * c.o_rank;
        L.wo_b = lw8(b, m, nm.f("blk.%d.attn_output_b.weight", l), D, ok, grp, RAD_SHARD_COL,
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
    L.ffn_norm = w("blk.%d.ffn_norm.weight", RAD_F32, { D });
    L.pcn_f = post_collapse_norm(b, m, T, L.ffn_norm, m.attn_out, m.h0, m.h1, m.pre_a, m.xcol,
                                 m.act.x);
    if (!L.pcn_f) L.cn_f = collapse_norm(b, m, T, L.ffn_norm, m.h1, m.pre_a, m.xcol, m.act.x);
    if (!L.pcn_f && !L.cn_f) {
        L.col_f = op(b, "mhc_collapse", hcp, {}, { m.h1, m.pre_a }, { m.xcol });
        L.norm_f = rmsnorm(b, m, T, L.ffn_norm, D, m.xcol, m.act.x);
    }
    MoeFP8::Src s;
    s.norm = nullptr;
    moe_names(m, l, &s);
    s.cfg.n_expert_all = c.n_expert;
    s.cfg.n_expert = c.n_expert / g.world;
    s.cfg.expert_base = g.rank * s.cfg.n_expert;
    s.cfg.probed_expert = s.cfg.expert_base;
    s.cfg.top_k = c.top_k;
    s.cfg.n_ff_exp = c.n_ff;
    s.cfg.n_ff_shared = g.n_ff;
    s.cfg.shared_lo = m.sh_lo;
    s.cfg.shared_hi = m.sh_hi;
    s.cfg.rows = T;
    s.cfg.norm_topk = c.norm_topk;
    s.cfg.sigmoid_router = 1;
    s.cfg.router_score = "sqrtsoftplus";
    s.cfg.router_bf16 = m.router_bf16 ? 1 : 0;
    s.cfg.route_scale = c.route_scale;
    s.cfg.swiglu_limit = c.swiglu_limit;
    s.cfg.shared_ungated = 1;
    /* The shared expert's two linears (LinearFP8, row parts and an uneven slice of 256-column
     * blocks) read as the attention's are, unless excluded by name. */
    const bool sh_f8 = m.f8_on && !f8_skipped(m, nm.f("blk.%d.ffn_gate_up_shexp.weight", l)) &&
                       !f8_skipped(m, nm.f("blk.%d.ffn_down_shexp.weight", l));
    s.cfg.shared_bf16_op = sh_f8 ? kDenseF8 : nullptr;
    DS_TRY(MoeFP8::probe(b, nm, l, s, &s.cfg));
    DS_TRY(L.moe.declare(b, nm, g, l, m.mw, s, false, false));
    if (m.f8_on && g.n_ff > 0)
        for (const LinearFP8* sl : { &L.moe.sh_gate_up, &L.moe.sh_down }) {
            if (sl->bf16_alt) {
                ++m.f8_n;
                m.f8_codes += sl->N * sl->K;
            } else {
                ++m.f8_kept;
            }
        }
    /* GlmRouter declared it as blk.L.glm_router.weight. */
    L.rt_bf16 = bf16_held(b, m, nm.f("blk.%d.glm_router.weight", l));
    if (L.rt_bf16) {
        ++m.n_router_bf16;
        m.bf16_saved += c.n_expert * D * 2;
    }
    L.post_f = op(b, "mhc_post", hcp, {}, { m.act.x, m.h1, m.post, m.comb }, { m.h0 });
    /* After the MoE block, whose router and shared expert the reads can reach. The reads name no
     * buffer, so where they are declared changes no liveness. */
    pf_layer(b, m, L);
    return RAD_OK;
}

/* ------------------------------------------------------------------ the DSpark drafter (Dsp) */
/* Whether this checkpoint and deployment get the drafter (m.dsp.on); an error only for a serve the
 * drafter cannot run as asked. The checkpoint states it by `nextn_predict_layers` (the stages ship as
 * blk.{n_layer}.. ) beside the 1017's `dspark.*` keys; the published UD GGUFs carry the keys but no
 * stages, and get no drafter. The official checkpoint's own stages (mtp.*, fp8 / MXFP4) reach a GGUF
 * trunk as one more split shard in these names, re-spelled value for value (BF16 linears, Q8_0 experts)
 * by a graft outside the tree, so they arrive here as any GGUF weight does. */
static int dsp_config(RadBuilder* b, Model& m, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    auto& d = m.dsp;
    const auto& c = m.c;
    const auto& g = m.g;
    const int64_t ns = gi(meta, "nextn_predict_layers", 0), blk = gi(meta, "dspark.block_size", 0);
    if (blk <= 0) return RAD_OK;
    if (ns <= 0) {
        rad_note(b, "the checkpoint names a DSpark drafter (dspark.block_size %lld) but carries none of its "
                    "stages (no nextn_predict_layers): no drafter", (long long)blk);
        return RAD_OK;
    }
    if (!rad_ctx_draft_head(ctx)) return RAD_OK;   /* no speculation, or prompt lookup alone */
    if (ctx->max_spec != blk) {
        fprintf(stderr, "radiance: deepseek41: the DSpark drafter proposes a whole block of %lld tokens "
                        "(dspark.block_size, the length it was trained at) and this serve asks for %d; run "
                        "--num-speculative-tokens %lld (or auto)\n", (long long)blk, ctx->max_spec, (long long)blk);
        return RAD_E_UNSUPPORTED;
    }
    d.n_expert = gi(meta, "dspark.expert_count", c.n_expert);
    d.top_k = gi(meta, "dspark.expert_used_count", c.top_k);
    d.rank = gi(meta, "dspark.markov_rank", 0);
    d.noise = gi(meta, "dspark.noise_token_id", -1);
    d.cand = gi(meta, "dspark.markov_top_k", 16);
    d.taps = gl(meta, "dspark.target_layers");
    std::sort(d.taps.begin(), d.taps.end());
    bool ok = !d.taps.empty() && d.noise >= 0 && d.noise < c.n_vocab && d.rank > 0 && d.rank % 32 == 0 &&
              d.rank <= 512 && d.cand >= 1 && d.cand <= 32 && d.n_expert > 0 && d.top_k > 0 &&
              d.top_k <= d.n_expert;
    for (int64_t t : d.taps) ok = ok && t >= 0 && t < c.n_layer;
    if (!ok) {
        fprintf(stderr, "radiance: deepseek41: the dspark.* configuration does not describe a drafter this "
                        "plugin runs (target layers %zu, noise id %lld, markov rank %lld (a multiple of 32 up "
                        "to 512), candidates %lld (1..32), experts %lld top-%lld)\n",
                d.taps.size(), (long long)d.noise, (long long)d.rank, (long long)d.cand,
                (long long)d.n_expert, (long long)d.top_k);
        return RAD_E_UNSUPPORTED;
    }
    /* Experts are expert-parallel as in the trunk, an even count of whole experts a rank (the grouped GEMM
     * takes two parity tables), but not necessarily the same count: the 1017's 128 over three ranks are
     * 44 / 42 / 42 (the even floor of the share each, then two more to the first ranks until all are
     * placed). The router is replicated, so routing needs no agreement on the split. */
    {
        const int64_t per = (d.n_expert / g.world) & ~(int64_t)1;
        int64_t rem = d.n_expert - per * g.world, base = 0;
        for (int r = 0; r <= g.rank; ++r) {
            const int64_t n = per + (rem >= 2 ? 2 : 0);
            rem -= rem >= 2 ? 2 : 0;
            if (r == g.rank) {
                d.ne_rank = n;
                d.ne_base = base;
            }
            base += n;
        }
    }
    if (d.n_expert % 2 || d.ne_rank < 2) {
        fprintf(stderr, "radiance: deepseek41: the DSpark drafter's %lld experts do not split over %d ranks "
                        "(an even count of whole experts each); serving without it\n",
                (long long)d.n_expert, g.world);
        rad_note(b, "DSpark drafter NOT declared: %lld experts over %d ranks", (long long)d.n_expert, g.world);
        d.taps.clear();
        return RAD_OK;
    }
    if (d.ne_rank * g.world != d.n_expert)
        rad_note(b, "DSpark drafter experts split unevenly: rank %d holds %lld of %lld from %lld", g.rank,
                 (long long)d.ne_rank, (long long)d.n_expert, (long long)d.ne_base);
    d.block = blk;
    d.base = (int)c.n_layer;
    d.n_stage = (int)ns;
    d.Q = std::min<int64_t>(g.max_seqs * blk, g.max_tok);
    d.on = d.Q >= blk;
    /* The short block (Dsp::block_lo): at least two rows (Scheduler::block_begin stages no fewer),
     * shorter than the drafter's own. */
    if (const char* e = std::getenv("RADIANCE_EXPERIMENTAL_DS41_DSPARK_SHORT_BLOCK"); e && *e && std::strcmp(e, "0")) {
        char* end = nullptr;
        const long long v = std::strtoll(e, &end, 10);
        if (*end || v < 2 || v >= blk) {
            fprintf(stderr, "radiance: deepseek41: RADIANCE_EXPERIMENTAL_DS41_DSPARK_SHORT_BLOCK=%s: the DSpark "
                            "drafter's short block is 2 .. %lld rows (its own is %lld)\n", e, (long long)blk - 1,
                    (long long)blk);
            return RAD_E_INVAL;
        }
        d.block_lo = v;
    }
    const auto env_on = [](const char* n) {
        const char* e = std::getenv(n);
        return e && *e && std::strcmp(e, "0");
    };
    d.ctx_defer = env_on("RADIANCE_EXPERIMENTAL_DS41_DSPARK_CTX_DEFER");
    d.conf_x = env_on("RADIANCE_EXPERIMENTAL_DS41_DSPARK_CONF");
    if (d.ctx_defer && c.window <= 0) {
        fprintf(stderr, "radiance: deepseek41: RADIANCE_EXPERIMENTAL_DS41_DSPARK_CTX_DEFER needs the "
                        "drafter's sliding window (this checkpoint states none)\n");
        return RAD_E_INVAL;
    }
    return RAD_OK;
}

/* One stage's weights (blk.{base + s}.*, the trunk's suffixes) and its context-pass key ops. */
static int dsp_stage_weights(RadBuilder* b, Model& m, Layer& L, int s) {
    auto& g = m.g;
    auto& nm = m.nm;
    const auto& c = m.c;
    const int l = m.dsp.base + s;
    const int64_t D = c.n_embd, HD = g.n_head * c.head_dim, hd = c.head_dim, mix = c.hc * (c.hc + 2);
    const auto grp = grp_layer(l);
    L.l = l;
    L.r = 0;
    auto w = [&](const char* fmt, uint32_t dt, std::initializer_list<int64_t> shape) {
        return wt(b, nm.f(fmt, l), dt, shape, grp);
    };
    L.hc_a_fn = w("blk.%d.hc_attn_fn.weight", RAD_F32, { mix, c.hc * D });
    L.hc_a_scale = w("blk.%d.hc_attn_scale.weight", RAD_F32, { 3 });
    L.hc_a_base = w("blk.%d.hc_attn_base.weight", RAD_F32, { mix });
    L.hc_f_fn = w("blk.%d.hc_ffn_fn.weight", RAD_F32, { mix, c.hc * D });
    L.hc_f_scale = w("blk.%d.hc_ffn_scale.weight", RAD_F32, { 3 });
    L.hc_f_base = w("blk.%d.hc_ffn_base.weight", RAD_F32, { mix });
    L.ffn_norm = w("blk.%d.ffn_norm.weight", RAD_F32, { D });
    if (!L.hc_a_fn || !L.hc_a_scale || !L.hc_a_base || !L.hc_f_fn || !L.hc_f_scale || !L.hc_f_base || !L.ffn_norm)
        return RAD_E_INVAL;
    if (m.zero) return RAD_OK;
    /* With ..._DSPARK_FP8 the trunk's lw8() (dsp_f8_gate): the same eligibility and exclusions. */
    const auto dlw = [&](const char* name, int64_t n, int64_t k, int shard = RAD_SHARD_NONE,
                         int64_t lo = 0, int64_t hi = 0, bool mapped = false) {
        return m.dsp_f8 ? lw8(b, m, name, n, k, grp, shard, lo, hi, mapped)
                        : lw(b, m, name, n, k, grp, shard, lo, hi, mapped);
    };
    L.attn_norm = w("blk.%d.attn_norm.weight", RAD_F32, { D });
    L.wq_a = dlw(nm.f("blk.%d.attn_q_a.weight", l), c.q_lora, D);
    L.q_norm = w("blk.%d.attn_q_a_norm.weight", RAD_F32, { c.q_lora });
    L.wq_b = dlw(nm.f("blk.%d.attn_q_b.weight", l), HD, c.q_lora, RAD_SHARD_ROW, m.ar * HD,
                 (m.ar + 1) * HD);
    L.wkv = dlw(nm.f("blk.%d.attn_kv.weight", l), hd, D);
    L.kv_norm = w("blk.%d.attn_kv_a_norm.weight", RAD_F32, { hd });
    L.sinks = wts(b, g, nm.f("blk.%d.attn_sinks.weight", l), RAD_F32, { g.n_head }, grp, RAD_SHARD_ROW,
                  m.ar * g.n_head, (m.ar + 1) * g.n_head);
    const int64_t gd = HD / m.gr;
    for (int j = 0; j < m.gr; ++j) {
        const int gi_ = (int)(m.ar * m.gr + j);
        const char* n = nm.f("blk.%d.attn_output_a.g%d.weight", l, gi_);
        DS_TRY(map_rows(b, n, { { ckpt_name(nm.f("blk.%d.attn_output_a.weight", l)), (int)(gi_ * c.o_rank), c.o_rank } }));
        L.wo_a.push_back(dlw(n, c.o_rank, gd, RAD_SHARD_NONE, 0, 0, true));
        if (!L.wo_a.back()) return RAD_E_INVAL;
    }
    const int64_t ok = m.gr * c.o_rank;
    L.wo_b = dlw(nm.f("blk.%d.attn_output_b.weight", l), D, ok, RAD_SHARD_COL, m.ar * ok,
                 (m.ar + 1) * ok);
    return L.attn_norm && L.wq_a && L.q_norm && L.wq_b && L.wkv && L.kv_norm && L.sinks && L.wo_b ? RAD_OK
                                                                                                  : RAD_E_INVAL;
}

/* One stage's query-pass block: Layer::layer()'s window-only path over the drafter's rows, its
 * attention over the drafter's cache. */
static int dsp_stage_ops(RadBuilder* b, Model& m, Layer& L) {
    auto& g = m.g;
    auto& nm = m.nm;
    const auto& c = m.c;
    auto& d = m.dsp;
    const int l = L.l;
    const int64_t Q = d.Q, D = c.n_embd, HD = g.n_head * c.head_dim, hd = c.head_dim;
    const std::vector<RadParam> mixp = { RAD_RANGE("M", 1, Q), RAD_INT("n", D), RAD_INT("hc", c.hc),
                                         RAD_INT("iterations", c.hc_iters), RAD_F64("eps", c.eps),
                                         RAD_F64("hc_eps", c.hc_eps), RAD_STR("dtype", "bf16") };
    const std::vector<RadParam> hcp = { RAD_RANGE("M", 1, Q), RAD_INT("n", D), RAD_INT("hc", c.hc),
                                        RAD_STR("dtype", "bf16") };
    L.mix_a = op(b, "mhc_mix", mixp, { L.hc_a_fn, L.hc_a_scale, L.hc_a_base }, { d.h0 }, { d.pre_a, d.post, d.comb });
    L.col_a = op(b, "mhc_collapse", hcp, {}, { d.h0, d.pre_f }, { d.x });
    if (m.zero) {
        L.attn_fill = op(b, "fill", { RAD_RANGE("M", 1, Q), RAD_INT("n", D), RAD_STR("dtype", "bf16"),
                                      RAD_F64("value", 0.0) },
                         {}, {}, { d.attn_out });
    } else {
        L.norm_a = rmsnorm(b, m, Q, L.attn_norm, D, d.x, d.xn);
        L.q_a = linear(b, m, Q, L.wq_a, c.q_lora, D, d.xn, d.qr);
        L.q_an = rmsnorm(b, m, Q, L.q_norm, c.q_lora, d.qr, d.qrn);
        L.q_b = linear(b, m, Q, L.wq_b, HD, c.q_lora, d.qrn, d.q);
        L.q_rope = rope_rows(b, m, Q, g.n_head, hd, 0, d.q, m.cs_a);
        L.kv_g = linear(b, m, Q, L.wkv, hd, D, d.xn, d.kv_);
        L.kv_n = rmsnorm(b, m, Q, L.kv_norm, hd, d.kv_, d.kvn);
        L.kv_rope = rope_rows(b, m, Q, 1, hd, 0, d.kvn, m.cs_a);
        L.kv_q = roundtrip(b, Q, hd, 32, "fp8_e4m3", "ue8m0", d.kvn);
        L.kv_st = op(b, "kv_store",
                     { RAD_RANGE("M", 1, Q), RAD_INT("head_dim", hd / 2), RAD_INT("n_head_kv", 1),
                       RAD_INT("block_size", d.kv_bs), RAD_STR("kv_dtype", "bf16") },
                     {}, { d.kvn }, {});
        /* The window of `c.window` context keys before the anchor plus the block's own `block` keys,
         * for every row: each row is placed at its block's last position (apos). */
        L.attn = op(b, "attn_sparse_sink",
                    { RAD_RANGE("M", 1, Q), RAD_INT("heads", g.n_head), RAD_INT("dim", hd),
                      RAD_INT("window", c.window + d.block), RAD_INT("ratio", 0), RAD_INT("topk", 0),
                      RAD_F64("scale", 1.0 / std::sqrt((double)hd)), RAD_STR("dtype", "bf16") },
                    { L.sinks }, { d.q, d.apos }, { d.o });
        /* The short block's: the same `window` context keys before the anchor plus its own rows. */
        if (d.block_lo > 0) {
            L.attn_lo = op(b, "attn_sparse_sink",
                           { RAD_RANGE("M", 1, Q), RAD_INT("heads", g.n_head), RAD_INT("dim", hd),
                             RAD_INT("window", c.window + d.block_lo), RAD_INT("ratio", 0), RAD_INT("topk", 0),
                             RAD_F64("scale", 1.0 / std::sqrt((double)hd)), RAD_STR("dtype", "bf16") },
                           { L.sinks }, { d.q, d.apos }, { d.o });
            if (!L.attn_lo) return RAD_E_INVAL;
        }
        L.o_rope = rope_rows(b, m, Q, g.n_head, hd, 1, d.o, m.cs_a);
        L.o_q = roundtrip(b, Q, HD, 32, "fp8_e4m3", "ue8m0", d.o);
        const int64_t gd = HD / m.gr;
        for (int j = 0; j < m.gr; ++j) L.o_a.push_back(linear(b, m, Q, L.wo_a[(size_t)j], c.o_rank, gd, d.o, d.oa));
        L.o_b = linear(b, m, Q, L.wo_b, D, m.gr * c.o_rank, d.oa, d.attn_out);
        if (!L.norm_a || !L.q_a || !L.q_an || !L.q_b || !L.q_rope || !L.kv_g || !L.kv_n || !L.kv_rope ||
            !L.kv_st || !L.attn || !L.o_rope || !L.o_b)
            return RAD_E_INVAL;
    }
    if (g.world > 1)
        L.ar_attn = op(b, "all_reduce",
                       { RAD_INT("world_size", g.world), RAD_RANGE("numel", D, Q * D), RAD_STR("dtype", "bf16"),
                         RAD_INT("exact", g.wire_exact), RAD_INT("min_bytes", g.wire_min_bytes) },
                       {}, { d.attn_out }, { d.attn_out });
    L.post_a = op(b, "mhc_post", hcp, {}, { d.attn_out, d.h0, d.post, d.comb }, { d.h1 });
    L.mix_f = op(b, "mhc_mix", mixp, { L.hc_f_fn, L.hc_f_scale, L.hc_f_base }, { d.h1 }, { d.pre_f, d.post, d.comb });
    L.col_f = op(b, "mhc_collapse", hcp, {}, { d.h1, d.pre_a }, { d.x });
    L.norm_f = rmsnorm(b, m, Q, L.ffn_norm, D, d.x, d.act.x);
    MoeFP8::Src s;
    s.norm = nullptr;
    moe_names(m, l, &s);
    s.cfg.n_expert_all = d.n_expert;
    s.cfg.n_expert = d.ne_rank;
    s.cfg.expert_base = d.ne_base;
    s.cfg.probed_expert = s.cfg.expert_base;
    s.cfg.top_k = d.top_k;
    s.cfg.n_ff_exp = c.n_ff;
    s.cfg.n_ff_shared = g.n_ff;
    s.cfg.shared_lo = m.sh_lo;
    s.cfg.shared_hi = m.sh_hi;
    s.cfg.rows = Q;
    s.cfg.norm_topk = c.norm_topk;
    s.cfg.sigmoid_router = 1;
    s.cfg.router_score = "sqrtsoftplus";
    s.cfg.route_scale = c.route_scale;
    s.cfg.swiglu_limit = c.swiglu_limit;
    s.cfg.shared_ungated = 1;
    /* With ..._DSPARK_FP8 the shared expert's two linears as the trunk's (layer()). */
    const bool sh_f8 = m.dsp_f8 && !f8_skipped(m, nm.f("blk.%d.ffn_gate_up_shexp.weight", l)) &&
                       !f8_skipped(m, nm.f("blk.%d.ffn_down_shexp.weight", l));
    s.cfg.shared_bf16_op = sh_f8 ? kDenseF8 : nullptr;
    DS_TRY(MoeFP8::probe(b, nm, l, s, &s.cfg));
    DS_TRY(L.moe.declare(b, nm, g, l, d.mw, s, false, false));
    if (m.dsp_f8 && g.n_ff > 0)
        for (const LinearFP8* sl : { &L.moe.sh_gate_up, &L.moe.sh_down }) {
            if (sl->bf16_alt) {
                ++m.f8_n;
                m.f8_codes += sl->N * sl->K;
            } else {
                ++m.f8_kept;
            }
        }
    L.post_f = op(b, "mhc_post", hcp, {}, { d.act.x, d.h1, d.post, d.comb }, { d.h0 });
    return L.mix_a && L.col_a && L.post_a && L.mix_f && L.col_f && L.norm_f && L.post_f ? RAD_OK : RAD_E_INVAL;
}

/* Everything of the drafter past the trunk's head (declaration order is the buffer planner's
 * liveness: the drafter's passes run after the whole trunk step, as arch/llama_fp8 declares its own). */
static int dsp_declare(RadBuilder* b, Model& m, const RadBuildCtx* ctx) {
    auto& g = m.g;
    auto& nm = m.nm;
    const auto& c = m.c;
    auto& d = m.dsp;
    const int64_t T = g.max_tok, Q = d.Q, D = c.n_embd, HD = g.n_head * c.head_dim, hd = c.head_dim;
    const int64_t R = d.cand;
    auto bf = [&](const char* n, std::initializer_list<int64_t> s) { return decl_b(b, nm.f("%s", n), RAD_BF16, s); };
    /* ..._DSPARK_FP8: what the drafter adds to the FP8 counts is its own (Model::dsp_f8_n). */
    const int f8_n0 = m.f8_n, f8_kept0 = m.f8_kept;
    const int64_t f8_codes0 = m.f8_codes;

    /* ---- buffers: the context pass (a trunk step's rows), the query pass (Q rows), the head */
    d.c = bf("dspark_main", { T, D });
    d.cn = bf("dspark_main_normed", { T, D });
    d.ckv = bf("dspark_ctx_kv", { T, hd });
    d.ckvn = bf("dspark_ctx_kv_normed", { T, hd });
    d.x = d.conf_x ? decl_b(b, nm.f("dspark_collapsed"), RAD_BF16, { Q, D }, RAD_BUF_PERSIST)
                   : bf("dspark_collapsed", { Q, D });
    d.xn = bf("dspark_normed", { Q, D });
    d.h0 = bf("dspark_streams", { Q, c.hc * D });
    d.h1 = bf("dspark_streams_next", { Q, c.hc * D });
    d.pre_a = decl_b(b, nm.f("dspark_pre_attn"), RAD_F32, { Q, c.hc });
    d.pre_f = decl_b(b, nm.f("dspark_pre_ffn"), RAD_F32, { Q, c.hc });
    d.post = decl_b(b, nm.f("dspark_post"), RAD_F32, { Q, c.hc });
    d.comb = decl_b(b, nm.f("dspark_comb"), RAD_F32, { Q, c.hc * c.hc });
    d.qr = bf("dspark_q_lora", { Q, c.q_lora });
    d.qrn = bf("dspark_q_lora_normed", { Q, c.q_lora });
    d.q = bf("dspark_q", { Q, HD });
    d.kv_ = bf("dspark_kv", { Q, hd });
    d.kvn = bf("dspark_kv_normed", { Q, hd });
    d.o = bf("dspark_attn_o", { Q, HD });
    d.oa = bf("dspark_attn_o_lora", { Q, m.gr * c.o_rank });
    d.attn_out = bf("dspark_attn_out", { Q, D });
    d.poff = decl_b(b, nm.f("dspark_pos_offset"), RAD_I32, { Q, 1 });
    d.apos = decl_b(b, nm.f("dspark_attn_pos"), RAD_I32, { Q, 1 });
    d.act.x = bf("dspark_ffn_normed", { Q, D });
    auto& mw = d.mw;
    mw.x = d.x;
    mw.h = d.act;
    mw.logits = decl_b(b, nm.f("dspark_router_logits"), RAD_F32, { Q, d.n_expert });
    mw.ids = decl_b(b, nm.f("dspark_expert_ids"), RAD_I32, { Q, d.top_k });
    mw.ew = decl_b(b, nm.f("dspark_expert_weights"), RAD_F32, { Q, d.top_k });
    mw.eoff = decl_b(b, nm.f("dspark_expert_offsets"), RAD_I32, { d.n_expert + 1 });
    mw.ecnt = decl_b(b, nm.f("dspark_expert_counts"), RAD_I32, { d.n_expert });
    mw.sorted = decl_b(b, nm.f("dspark_expert_sorted"), RAD_I32, { Q * d.top_k });
    mw.egu = bf("dspark_expert_gate_up", { Q * d.top_k, 2 * c.n_ff });
    mw.eff.x = bf("dspark_expert_ff", { Q * d.top_k, c.n_ff });
    mw.edn = bf("dspark_expert_down", { Q * d.top_k, D });
    if (g.n_ff > 0) {
        mw.sgu = bf("dspark_shared_gate_up", { Q, 2 * g.n_ff });
        mw.sff.x = bf("dspark_shared_ff", { Q, g.n_ff });
    }
    mw.sout = bf("dspark_shared_out", { Q, D });
    mw.sgate = bf("dspark_shared_one", { Q, 1 });
    d.fin = bf("dspark_final", { Q, D });
    d.rows = bf("dspark_head_in", { Q, D });
    d.logits = decl_b(b, nm.f("dspark_logits"), RAD_F32, { Q, g.n_vocab });
    d.cand_ = decl_b(b, nm.f("dspark_cand"), RAD_I32, { Q, R });
    d.unary = decl_b(b, nm.f("dspark_unary"), RAD_F32, { Q, R });
    if (g.world > 1) {
        d.pairs = decl_b(b, nm.f("dspark_pairs"), RAD_F32, { Q, R, 2 });
        d.gath = decl_b(b, nm.f("dspark_pairs_all"), RAD_F32, { Q, (int64_t)g.world * R, 2 });
    }
    /* The proposal buffer (RadDrafterDecl::proposal), `block` tokens a sequence. */
    d.tokens = decl_b(b, nm.f("dspark_tokens"), RAD_I32, { g.max_seqs, d.block });
    if (!d.c || !d.cn || !d.ckv || !d.ckvn || !d.x || !d.h0 || !d.logits || !d.tokens) return RAD_E_INVAL;

    /* ---- weights */
    d.st.resize((size_t)d.n_stage);
    for (int s = 0; s < d.n_stage; ++s) DS_TRY(dsp_stage_weights(b, m, d.st[(size_t)s], s));
    const int64_t nt = (int64_t)d.taps.size();
    /* The context pass runs on the attention ranks only (an attention-zero rank keeps no keys). */
    if (!m.zero) {
        d.fc = m.dsp_f8 ? lw8(b, m, nm.f("dspark.fc.weight"), D, nt * D, grp_model())
                        : lw(b, m, nm.f("dspark.fc.weight"), D, nt * D, grp_model());
        d.main_norm = wt(b, nm.f("dspark.enc.output_norm.weight"), RAD_F32, { D }, grp_model());
        if (!d.fc || !d.main_norm) return RAD_E_INVAL;
    }
    d.norm = wt(b, nm.f("dspark.output_norm.weight"), RAD_F32, { D }, grp_model());
    /* The Markov head's tables, [vocab, rank] each and replicated: the walk gathers rows by global id. */
    d.mk1 = wt(b, nm.f("dspark.markov_w1.weight"), RAD_BF16, { c.n_vocab, d.rank }, grp_model(), RAD_ACCESS_VOCAB);
    d.mk2 = wt(b, nm.f("dspark.markov_w2.weight"), RAD_BF16, { c.n_vocab, d.rank }, grp_model(), RAD_ACCESS_VOCAB);
    if (!d.norm || !d.mk1 || !d.mk2) return RAD_E_INVAL;

    /* ---- the context pass, and the drafter's window cache: the context keys of every committed
     * position plus, past them, the block keys a query pass writes (the next context pass overwrites
     * them). Declared right before the first store, whose block_size fixes the group's. */
    RadKVGroupDecl kd{};
    kd.kind = RAD_KV_WINDOW;
    kd.dtype = RAD_BF16;
    kd.window = c.window + d.block + g.max_spec;
    kd.n_head_kv = 1;
    kd.head_dim = hd / 2;
    const std::vector<RadParam> stp = { RAD_RANGE("M", 1, T), RAD_INT("head_dim", hd / 2), RAD_INT("n_head_kv", 1),
                                        RAD_INT("block_size", 16), RAD_STR("kv_dtype", "bf16") };
    if (m.zero) {
        if (!(d.kv = rad_decl_kv_group(b, nm.f("kv_dspark"), &kd))) return RAD_E_INVAL;
        d.sizing = op(b, "kv_store", stp, {}, { d.ckvn }, {});   /* fixes the block size; never issued */
    } else {
        d.c_tab = op(b, "rope_table",
                     { RAD_RANGE("M", 1, T), RAD_INT("rot", c.rope), RAD_F64("theta", c.theta),
                       RAD_F64("scale", 1.0), RAD_STR("dtype", "f32") },
                     {}, {}, { m.cs_a });
        d.c_fc = linear(b, m, T, d.fc, D, nt * D, d.src, d.c);
        d.c_norm = rmsnorm(b, m, T, d.main_norm, D, d.c, d.cn);
        for (auto& L : d.st) {
            L.d_kv = linear(b, m, T, L.wkv, hd, D, d.cn, d.ckv);
            L.d_kvn = rmsnorm(b, m, T, L.kv_norm, hd, d.ckv, d.ckvn);
            L.d_kvrope = rope_rows(b, m, T, 1, hd, 0, d.ckvn, m.cs_a);
            L.d_kvq = roundtrip(b, T, hd, 32, "fp8_e4m3", "ue8m0", d.ckvn);
            if (!d.kv && !(d.kv = rad_decl_kv_group(b, nm.f("kv_dspark"), &kd))) return RAD_E_INVAL;
            L.d_kvst = op(b, "kv_store", stp, {}, { d.ckvn }, {});
            if (!L.d_kv || !L.d_kvn || !L.d_kvrope || !L.d_kvst) return RAD_E_INVAL;
        }
        if (!d.c_tab || !d.c_fc || !d.c_norm) return RAD_E_INVAL;
    }
    d.kv_bs = rad_kv_block_size(b, d.kv);
    if (d.kv_bs < 1) return RAD_E_INVAL;
    for (const auto& L : d.st) DS_TRY(rad_bind_layer_kv(b, L.l, d.kv));
    if (m.zero) DS_TRY(rad_kv_group_zero(b, d.kv));

    /* ---- the query pass: [anchor, noise ...] embedded by the trunk's table into hc copies */
    const auto tt = m.wty.find(m.token);
    if (tt != m.wty.end())
        d.embed = op(b, "embed_lookup_ggml",
                     { RAD_RANGE("M", 1, Q), RAD_INT("n_embd", D), RAD_INT("n_vocab", g.n_vocab),
                       RAD_STR("type", tt->second), RAD_STR("dtype", "bf16"), RAD_INT("vocab_offset", g.vocab_off) },
                     { m.token }, {}, { d.x });
    else
        d.embed = op(b, "embed_lookup",
                     { RAD_RANGE("M", 1, Q), RAD_INT("n_embd", D), RAD_INT("n_vocab", g.n_vocab),
                       RAD_STR("dtype", "bf16"), RAD_INT("vocab_offset", g.vocab_off) },
                     { m.token }, {}, { d.x });
    if (g.world > 1)
        d.embed_ar = op(b, "all_reduce",
                        { RAD_INT("world_size", g.world), RAD_RANGE("numel", D, Q * D), RAD_STR("dtype", "bf16"),
                          RAD_INT("exact", 1) },
                        {}, { d.x }, { d.x });
    d.expand = op(b, "cast", { RAD_RANGE("M", 1, Q), RAD_INT("n", D), RAD_STR("from", "bf16"), RAD_STR("to", "bf16") },
                  {}, { d.x }, { d.h0 });
    d.pre0 = op(b, "fill", { RAD_RANGE("M", 1, Q), RAD_INT("n", c.hc), RAD_STR("dtype", "f32"), RAD_F64("value", 0.0) },
                {}, {}, { d.pre_f });
    d.pre1 = op(b, "fill", { RAD_RANGE("M", 1, Q), RAD_INT("n", 1), RAD_STR("dtype", "f32"), RAD_F64("value", 1.0) },
                {}, {}, { d.pre_f });
    d.q_tab = op(b, "rope_table",
                 { RAD_RANGE("M", 1, Q), RAD_INT("rot", c.rope), RAD_F64("theta", c.theta), RAD_F64("scale", 1.0),
                   RAD_STR("dtype", "f32") },
                 {}, {}, { m.cs_a });
    /* apos = position + (block - 1 - row % block): each row at its block's last position. */
    d.poff_op = op(b, "iota", { RAD_RANGE("M", 1, Q), RAD_INT("start", d.block - 1), RAD_INT("step", -1),
                                RAD_INT("period", d.block) },
                   {}, {}, { d.poff });
    /* The short block's, declared where it is issued (in poff_op's place) for the buffers' liveness. */
    if (d.block_lo > 0 &&
        !(d.poff_lo = op(b, "iota", { RAD_RANGE("M", 1, Q), RAD_INT("start", d.block_lo - 1), RAD_INT("step", -1),
                                      RAD_INT("period", d.block_lo) },
                         {}, {}, { d.poff })))
        return RAD_E_INVAL;
    d.apos_op = op(b, "add", { RAD_RANGE("M", 1, Q), RAD_INT("n", 1), RAD_STR("dtype", "i32") }, {}, { d.poff },
                   { d.apos });
    if (!d.embed || !d.expand || !d.pre0 || !d.pre1 || !d.q_tab || !d.poff_op || !d.apos_op) return RAD_E_INVAL;
    for (auto& L : d.st) DS_TRY(dsp_stage_ops(b, m, L));

    /* ---- the head: the last stage's ffn gates collapse the streams, its norm, the trunk's lm_head */
    d.contract = op(b, "mhc_collapse", { RAD_RANGE("M", 1, Q), RAD_INT("n", D), RAD_INT("hc", c.hc), RAD_STR("dtype", "bf16") },
                    {}, { d.h0, d.pre_f }, { d.x });
    d.fnorm = rmsnorm(b, m, Q, d.norm, D, d.x, d.fin);
    d.gather = op(b, "gather_rows", { RAD_RANGE("M", 1, Q), RAD_INT("n", D), RAD_STR("dtype", "bf16") }, {},
                  { d.fin }, { d.rows });
    const auto ht = m.wty.find(m.head);
    if (ht != m.wty.end())
        d.head = op(b, "logits_gemm_ggml",
                    { RAD_RANGE("M", 1, Q), RAD_INT("n_vocab", g.n_vocab), RAD_INT("n_embd", D),
                      RAD_STR("type", ht->second), RAD_STR("dtype", "bf16") },
                    { m.head }, { d.rows }, { d.logits });
    else
        d.head = op(b, "logits_gemm",
                    { RAD_RANGE("M", 1, Q), RAD_INT("n_vocab", g.n_vocab), RAD_INT("n_embd", D), RAD_STR("dtype", "bf16") },
                    { m.head }, { d.rows }, { d.logits });
    /* The candidates: this rank's top-R of its vocabulary shard, merged to the global top-R across
     * ranks (ties to the lower id, as an unsharded top-R). */
    if (g.world > 1) {
        d.topk = op(b, "row_topk", { RAD_RANGE("M", 1, Q), RAD_INT("N", g.n_vocab), RAD_INT("R", R),
                                     RAD_INT("vocab_off", g.vocab_off), RAD_STR("dtype", "f32") },
                    {}, { d.logits }, { d.cand_, d.unary, d.pairs });
        d.pgather = op(b, "all_gather", { RAD_INT("world_size", g.world), RAD_INT("numel", Q * R * 2),
                                          RAD_STR("dtype", "f32"), RAD_INT("row", R * 2) },
                       {}, { d.pairs }, { d.gath });
        d.merge = op(b, "row_topk_merge", { RAD_RANGE("M", 1, Q), RAD_INT("R", R), RAD_INT("world_size", g.world),
                                            RAD_STR("dtype", "f32") },
                     {}, { d.gath }, { d.cand_, d.unary });
        if (!d.pgather || !d.merge) return RAD_E_INVAL;
    } else {
        d.topk = op(b, "row_topk", { RAD_RANGE("M", 1, Q), RAD_INT("N", g.n_vocab), RAD_INT("R", R), RAD_STR("dtype", "f32") },
                    {}, { d.logits }, { d.cand_, d.unary });
    }
    /* The Markov walk: logits[i] + W2 . W1[prev(i)], prev(0) the anchor, greedily (dflash_select with no
     * hidden projection is exactly that bigram over the candidates). */
    d.select = op(b, "dflash_select",
                  { RAD_RANGE("M", 1, g.max_seqs), RAD_INT("steps", d.block), RAD_INT("top_k", R),
                    RAD_INT("rank", d.rank), RAD_INT("n_vocab", c.n_vocab), RAD_INT("anchor_stride", d.block),
                    RAD_STR("dtype", "bf16") },
                  { d.mk1, d.mk2 }, { d.cand_, d.unary }, { d.tokens });
    if (!d.contract || !d.fnorm || !d.gather || !d.head || !d.topk || !d.select) return RAD_E_INVAL;
    /* The short block's walk: its rows' candidates, the anchor every block_lo rows, into the front of
     * each sequence's proposal row (the pitch stays the drafter's). */
    if (d.block_lo > 0 &&
        !(d.select_lo = op(b, "dflash_select",
                           { RAD_RANGE("M", 1, g.max_seqs), RAD_INT("steps", d.block_lo), RAD_INT("top_k", R),
                             RAD_INT("rank", d.rank), RAD_INT("n_vocab", c.n_vocab),
                             RAD_INT("anchor_stride", d.block_lo), RAD_STR("dtype", "bf16") },
                           { d.mk1, d.mk2 }, { d.cand_, d.unary }, { d.tokens })))
        return RAD_E_INVAL;

    m.dsp_f8_n = m.f8_n - f8_n0;
    m.dsp_f8_kept = m.f8_kept - f8_kept0;
    m.dsp_f8_codes = m.f8_codes - f8_codes0;
    m.f8_n = f8_n0;
    m.f8_kept = f8_kept0;
    m.f8_codes = f8_codes0;

    RadDrafterDecl drf{};
    drf.name = "dspark";
    drf.kind = RAD_DRAFT_BLOCK;
    drf.depth = ctx->max_spec;
    drf.block = d.block;
    drf.proposal = d.tokens;
    drf.proposal_pitch = d.block;
    /* The query block's first row reads back `window` keys before itself. */
    drf.window = c.window + 1;
    drf.mask_token = (int32_t)d.noise;
    DS_TRY(rad_declare_drafter(b, &drf));
    rad_note(b, "DSpark drafter: %d stages at absolute layers %d..%d (window-only V4 blocks, %lld experts top-%lld), "
                "taps before %zu trunk layer(s), one pass over [anchor, noise %lld x %lld] proposes %lld tokens; "
                "attention over %lld context keys + the block's own; Markov rank %lld walked over the head's top %lld; "
                "the confidence head is not applied",
             d.n_stage, d.base, d.base + d.n_stage - 1, (long long)d.n_expert, (long long)d.top_k, d.taps.size(),
             (long long)d.noise, (long long)(d.block - 1), (long long)d.block, (long long)c.window,
             (long long)d.rank, (long long)R);
    if (d.block_lo > 0) {
        DS_TRY(rad_drafter_short_block(b, d.block_lo));
        rad_note(b, "DSpark drafter: also a short block of %lld rows ([anchor, noise x %lld], %lld tokens; "
                    "RADIANCE_EXPERIMENTAL_DS41_DSPARK_SHORT_BLOCK), run from --draft-short-from sequences on",
                 (long long)d.block_lo, (long long)(d.block_lo - 1), (long long)d.block_lo);
    }
    return RAD_OK;
}

/* The startup line of RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ATTN, one a rank (the shared expert's
 * slice differs by rank): the list and budget, what the layers read ahead of them (the reads each
 * layer's predecessor issues, Engram layers apart), the reads by kind (cut: crossed the budget),
 * and each Engram layer's MiB read ahead of its GEMM + behind it. So an A/B arm's server log
 * states the plan it ran. */
static void pf_note(const Model& m, int rank) {
    const pf::Cfg& pc = m.pfc;
    if (!pc.on) return;
    constexpr double MiB = 1048576.0;
    char t[192];
    std::string order, kinds, eng;
    for (const int k : pc.order) order += (order.empty() ? "" : ",") + std::string(pf::name(k));
    int whole[pf::kKinds] = {}, cut[pf::kKinds] = {};
    int n = 0, rmin = 0, rmax = 0;
    int64_t bsum = 0, bmin = 0, bmax = 0;
    for (const Layer& L : m.layers) {
        for (const std::vector<PfRead>* v : { &L.pf, &L.pf_post })
            for (const PfRead& r : *v)
                if (r.kind >= 0 && r.kind < pf::kKinds) ++(r.cut ? cut : whole)[r.kind];
        if (L.e_gemm) {
            int64_t a = 0, p = 0;
            for (const PfRead& r : L.pf) a += r.bytes;
            for (const PfRead& r : L.pf_post) p += r.bytes;
            std::snprintf(t, sizeof t, "%s%d %.1f + %.1f", eng.empty() ? "" : ", ", L.l, a / MiB, p / MiB);
            eng += t;
            continue;
        }
        if (L.pf.empty()) continue;
        int64_t s = 0;
        for (const PfRead& r : L.pf) s += r.bytes;
        const int cnt = (int)L.pf.size();
        rmin = n ? std::min(rmin, cnt) : cnt;
        rmax = n ? std::max(rmax, cnt) : cnt;
        bmin = n ? std::min(bmin, s) : s;
        bmax = n ? std::max(bmax, s) : s;
        bsum += s;
        ++n;
    }
    for (int k = 0; k < pf::kKinds; ++k) {
        if (!whole[k] && !cut[k]) continue;
        if (cut[k])
            std::snprintf(t, sizeof t, "%s%s %d (%d cut)", kinds.empty() ? "" : ", ", pf::name(k),
                          whole[k] + cut[k], cut[k]);
        else
            std::snprintf(t, sizeof t, "%s%s %d", kinds.empty() ? "" : ", ", pf::name(k), whole[k]);
        kinds += t;
    }
    std::string how;
    if (pc.budget > 0) {
        std::snprintf(t, sizeof t, "a budget of %.1f MiB an issue point over ", pc.budget / MiB);
        how = t + order;
    } else {
        how = order + (pc.listed ? " whole" : " whole (the default)");
    }
    if (!pc.note.empty()) how += "; ignored:" + pc.note;
    if (!n && eng.empty()) {
        std::fprintf(stderr, "radiance: deepseek41: rank %d: on-die reads (RADIANCE_EXPERIMENTAL_DS41_"
                             "PREFETCH_ATTN): %s; none declared (no kernel library serves `prefetch`, "
                             "or the trunk is llama.cpp's blocks)\n", rank, how.c_str());
        return;
    }
    const std::string engs = pc.engram ? "(MiB ahead of the GEMM + behind it) " + eng
                                       : std::string("not read (..._PREFETCH_ENGRAM unset)");
    std::fprintf(stderr, "radiance: deepseek41: rank %d: on-die reads (RADIANCE_EXPERIMENTAL_DS41_"
                         "PREFETCH_ATTN): %s; %d layers read ahead, %d-%d reads, %.1f MiB a layer "
                         "(%.1f-%.1f); by kind %s; Engram layers %s\n",
                 rank, how.c_str(), n, rmin, rmax, n ? bsum / MiB / n : 0.0, bmin / MiB, bmax / MiB,
                 kinds.c_str(), engs.c_str());
}

/* ------------------------------------------------------------------ declare */
/* RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8=1 (Model::f8), on only where the checkpoint says its dense
 * linears are E4M3 with an E8M0 exponent a 32x32 block (the release's config.json, which the
 * container keeps as imported_quantization_config) and a library declares the op. Anything else
 * is a note, and every linear stays on gemm_nt. RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8_EXCLUDE:
 * comma-separated declared weight names, or `*suffix` patterns, kept on gemm_nt. */
static bool dense_f8_gate(RadBuilder* b, const RadModelMeta* meta, Model& m) {
    const char* on = std::getenv("RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8");
    if (!on || std::strcmp(on, "1")) return false;
    const char* qm = rad_meta_gets(meta, "imported_quantization_config.quant_method", nullptr);
    const char* sf = rad_meta_gets(meta, "imported_quantization_config.scale_fmt", nullptr);
    const std::vector<int64_t> bs =
        int_list(rad_meta_gets(meta, "imported_quantization_config.weight_block_size", nullptr));
    if (!qm || std::strcmp(qm, "fp8") || !sf || std::strcmp(sf, "ue8m0") || bs.size() != 2 ||
        bs[0] != 32 || bs[1] != 32) {
        fprintf(stderr, "radiance: deepseek41: RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8 is set, but the "
                        "checkpoint does not say its dense linears are E4M3 with an E8M0 scale a 32x32 "
                        "block (imported_quantization_config: quant_method %s, scale_fmt %s, "
                        "weight_block_size %s); they stay on gemm_nt\n",
                qm ? qm : "-", sf ? sf : "-",
                rad_meta_gets(meta, "imported_quantization_config.weight_block_size", "-"));
        return false;
    }
    if (!rad_op_declared(b, kDenseF8)) {
        fprintf(stderr, "radiance: deepseek41: RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8 is set, but no "
                        "kernel library loaded declares %s; the dense linears stay on gemm_nt\n",
                kDenseF8);
        return false;
    }
    m.f8_skip.clear();
    if (const char* x = std::getenv("RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8_EXCLUDE"))
        for (const char* p = x; *p;) {
            const char* e = std::strchr(p, ',');
            const size_t len = e ? (size_t)(e - p) : std::strlen(p);
            if (len) m.f8_skip.emplace_back(p, len);
            p += len + (e ? 1 : 0);
        }
    return true;
}

/* RADIANCE_EXPERIMENTAL_DS41_DSPARK_FP8=1 (Model::dsp_f8): the DSpark drafter's dense linears
 * read by gemm_nt_bf16f8 as the trunk's are (lw8, MoeFP8::Config::shared_bf16_op), behind the
 * trunk's gate: the same checkpoint statement and library, the same exclusions. A note and
 * nothing else without it. The drafter's proposals are the same bits: the op gives gemm_nt's
 * bytes (no tune cache, as for the trunk's), so only its weight reads halve. */
static bool dsp_f8_gate(const Model& m) {
    const char* on = std::getenv("RADIANCE_EXPERIMENTAL_DS41_DSPARK_FP8");
    if (!on || std::strcmp(on, "1")) return false;
    if (!m.dsp.on) return false;   /* no drafter in this serve */
    if (!m.f8_on) {
        fprintf(stderr, "radiance: deepseek41: RADIANCE_EXPERIMENTAL_DS41_DSPARK_FP8 is set, but the "
                        "trunk's dense FP8 is off (RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8=1, which it "
                        "extends to the drafter); the drafter's linears stay on gemm_nt\n");
        return false;
    }
    return true;
}

/* ------------------------------------------------------------------ the consumer skip */
/* The per-row cut's grain (RadConsumerSuffix::row_grain): the chunk quantum, the window cache's
 * 16-token blocks, on which every chunk starts. */
static constexpr int32_t kCutRowGrain = 16;
/* RADIANCE_EXPERIMENTAL_DS41_CONSUMER_SKIP=1: the engine runs a prompt's early passes on a prefix of
 * the layers (core/sched/consumer_skip.h holds the schedule and its proof; rad_builder.h,
 * rad_declare_consumer_suffix, the contract). This plugin's half is the declaration below, made
 * only where the graph meets the contract, and step()'s cut.
 *
 * Why this graph meets it, layer by layer (layer() and step() above):
 *   - Compressed and index keys are stored only at the kv sources (`L.src`: l_st, k_st), 2, 8, 14
 *     and 20 in V4.1-Flash; B, the boundary, is the last. A layer past B reads compressed keys of
 *     src_of[l], the latest source at or below it of its ratio (read_cfg), so B's: its attention
 *     reads kv_cache(m.comp[r], src) and its indexer kv_cache(m.index[r], src). The index sources
 *     past B (24, 28, 32, 36) score their own queries against B's index keys and store none.
 *   - Every layer's window key (kv_g -> kv_n / kv_f -> kv_st) is computed from that layer's own
 *     normed input `xn` at the row, and attn_sparse_sink at row p reads the window cache at
 *     p - window + 1 .. p (libref ref_attn_sparse_sink). The window is W = c.window.
 *   - The rest is per row, or per pass and rebuilt in it: the candidate layer's keep and an index
 *     layer's selection are written earlier in the same pass for the same rows (the candidate layer
 *     is at or below B, and an index layer runs in full whenever a later layer does); Engram rows
 *     depend on tokens alone, at layers 1 and 14, below B; the compressor's ring (ratio 2) belongs
 *     to sources below B, which always run.
 *   - Layer F's key ops in a cut pass are the full layer's own handles in the full layer's order
 *     with the query, indexer, attention, output and FFN ops left out: its collapse and norm (or
 *     the previous layer's fused post_f + collapse + norm, pcn_a), kv_g, the norm/rope/round trip
 *     (kv_f, or kv_n, kv_rope, kv_q), kv_st, and at a source the compressor chain and the index
 *     and compressed key stores. None of them reads what a left-out op writes (c_tab rewrites
 *     cs_b with the same angles rope_b_step wrote, a pure function of the position, and keeps its
 *     order to the reads around it).
 * Refused (a line on rank 0 says why) for a graph off that shape: a kv source or an Engram layer
 * past the boundary, a candidate layer past it, a layer past it reading another source's keys,
 * the DSpark drafter (its context reads every prompt row's taps at its target layers) or any
 * speculation, and an engine that does not carry the cut (the weak reference is null). */
static void cut_declare(RadBuilder* b, Model& m, const RadBuildCtx* ctx) {
    const char* v = std::getenv("RADIANCE_EXPERIMENTAL_DS41_CONSUMER_SKIP");
    if (!v || std::strcmp(v, "1")) return;
    const auto& c = m.c;
    int B = -1;
    for (int64_t s : c.kv_src) B = std::max(B, (int)s);
    std::string why;
    if (B < 0 || B >= (int)c.n_layer - 1) {
        why = "no kv source before the last layer, so no layer to cut";
    } else if (c.window < 1) {
        why = "no window";
    } else if (m.dsp.on || ctx->max_spec > 0) {
        why = "speculation is on (the DSpark drafter's context reads every prompt row's taps at its "
              "target layers)";
    } else if (c.cand > B) {
        why = "the candidate layer " + std::to_string(c.cand) + " is past the boundary";
    } else {
        for (int l = B + 1; l < (int)c.n_layer && why.empty(); ++l) {
            if (c.is_src(l)) why = "layer " + std::to_string(l) + " stores keys past the boundary";
            else if (c.ple_index(l) >= 0) why = "layer " + std::to_string(l) + " has an Engram block";
            else if (c.ratio[(size_t)l] > 0 && c.src_of[(size_t)l] != B)
                why = "layer " + std::to_string(l) + " reads another source's compressed keys";
        }
    }
    if (why.empty() && !rad_declare_consumer_suffix)
        why = "this engine does not carry the cut (it is older than rad_declare_consumer_suffix)";
    if (why.empty()) {
        RadConsumerSuffix d{};
        d.struct_size = (uint32_t)sizeof d;
        d.n_layer = (int32_t)c.n_layer;
        d.boundary = B;
        d.window = (int32_t)c.window;
        /* Every op of a layer can run on a suffix of the step's rows that starts on the 16-row grid
         * (step(), "the per-row cut"): the engine uses it under large-chunk prefill. */
        d.row_grain = kCutRowGrain;
        const int rc = rad_declare_consumer_suffix(b, &d);
        if (rc != RAD_OK) why = "the engine refused the declaration (" + std::to_string(rc) + ")";
    }
    if (!why.empty()) {
        if (ctx->rank == 0)
            fprintf(stderr, "radiance: deepseek41: consumer skip refused: %s\n", why.c_str());
        rad_note(b, "consumer skip refused: %s", why.c_str());
        return;
    }
    m.cut_on = true;
    m.cut_b = B;
    m.cut_rows = rad_consumer_row_grain && rad_consumer_row_grain(b) == kCutRowGrain;
    if (ctx->rank == 0)
        fprintf(stderr, "radiance: deepseek41: consumer suffix declared: layers %d..%lld see earlier "
                        "rows only through their own %lld-row windows and layer %d's compressed and "
                        "index keys; a cut pass runs layers [0, F) in full and layer F's key ops alone\n",
                B + 1, (long long)c.n_layer - 1, (long long)c.window, B);
    rad_note(b, "consumer suffix declared: boundary %d, window %lld", B, (long long)c.window);
}

/* RADIANCE_EXPERIMENTAL_DS41_KV_REPLICATED=1: the compressed and index key caches declared
 * replicated (rad_kv_group_replicated), so the prefix cache's host tier (--prefix-cache-mib) keeps
 * one copy of an entry instead of every rank's: 6,400 B a 2-token entry instead of 25,600 at TP4,
 * four times the conversations in the same bytes. Every attention rank computes them whole from the
 * replicated residual stream with replicated weights (one latent KV head), so the four ranks hold
 * the same bytes: the Kh arms' per-rank hashes of kv_index_r* and kv_compressed_r* agree in every
 * prompt of the 29 Oct prefill window, and RADIANCE_DEBUG_TIER_VERIFY checks it at each restore.
 * Only with every rank attending (no attention-zero rank, whose groups are zeroed) and more than
 * one; it changes nothing but the tier's slot layout. */
static int kv_replicate(RadBuilder* b, Model& m) {
    const char* v = std::getenv("RADIANCE_EXPERIMENTAL_DS41_KV_REPLICATED");
    if (!v || std::strcmp(v, "1") != 0) return RAD_OK;
    const char* why = nullptr;
    if (!rad_kv_group_replicated) why = "this engine has no rad_kv_group_replicated";
    else if (m.g.world <= 1) why = "one rank holds every byte already";
    else if (m.zero || m.aw != m.g.world) why = "a rank keeps no attention cache (attention-zero)";
    if (why) {
        if (m.g.rank == 0)
            fprintf(stderr, "radiance: deepseek41: RADIANCE_EXPERIMENTAL_DS41_KV_REPLICATED refused: "
                            "%s\n", why);
        return RAD_OK;
    }
    int n = 0;
    std::string names;
    for (const auto* groups : { &m.index, &m.comp })
        for (const auto& kv : *groups) {
            DS_TRY(rad_kv_group_replicated(b, kv.second));
            ++n;
        }
    for (const auto& kv : m.index) names += (names.empty() ? "" : ",") + std::string("kv_index_r") + std::to_string(kv.first);
    for (const auto& kv : m.comp) names += (names.empty() ? "" : ",") + std::string("kv_compressed_r") + std::to_string(kv.first);
    if (m.g.rank == 0)
        fprintf(stderr, "radiance: deepseek41: RADIANCE_EXPERIMENTAL_DS41_KV_REPLICATED: %d paged "
                        "group(s) declared replicated (%s): a host-tier slot holds them once, not "
                        "once a rank\n", n, names.c_str());
    return RAD_OK;
}

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
    m.release = is_release(meta);
    m.ista = is_ista(meta);
    m.local_engram = local_ista_engram(meta);
    m.n_stage = (int)gi(meta, "nextn_predict_layers", 0);
    m.pfc = pf_env();
    naming = &m;
    DS_TRY(geometry(m, ctx));
    m.f8_on = dense_f8_gate(b, meta, m);
    DS_TRY(dsp_config(b, m, meta, ctx));
    m.dsp_f8 = dsp_f8_gate(m);
    m.router_bf16 = router_bf16_on();
    if (m.router_bf16 && !rad_op_declared(b, "gemm_nt_f32bf16")) {
        fprintf(stderr, "radiance: deepseek41: RADIANCE_EXPERIMENTAL_DS41_ROUTER_BF16=1 needs "
                        "gemm_nt_f32bf16, which no loaded kernel library declares (a libr4d built "
                        "with it)\n");
        return RAD_E_UNSUPPORTED;
    }
    const auto& c = m.c;
    auto& g = m.g;
    int64_t rmin = 0;
    for (int l = 0; l < c.n_layer; ++l)
        if (c.ratio[(size_t)l] > 0 && (rmin == 0 || c.ratio[(size_t)l] < rmin)) rmin = c.ratio[(size_t)l];
    m.work = g.max_tok + g.max_seqs;
    m.cmax = g.max_ctx / std::max<int64_t>(rmin, 1) + 1;
    index_runs(m);
    DS_TRY(buffers(b, m));
    /* The drafter's source rows, one hc-mean row a target layer, written by the trunk step. */
    /* With ..._DSPARK_CTX_DEFER persistent, and a window's rows more (Dsp::ctx_defer). */
    if (m.dsp.on && !m.zero) {
        m.dsp.src_rows = g.max_tok + (m.dsp.ctx_defer ? c.window + 1 : 0);
        if (!(m.dsp.src = decl_b(b, nm.f("dspark_src"), RAD_BF16,
                                 { m.dsp.src_rows, (int64_t)m.dsp.taps.size() * c.n_embd },
                                 m.dsp.ctx_defer ? RAD_BUF_PERSIST : RAD_BUF_TRANSIENT)))
            return RAD_E_INVAL;
    }

    const int64_t T = g.max_tok, D = c.n_embd;
    /* The vocabulary is split over the ranks (shard_extent): the lookup gives zeros outside this
     * rank's rows and an all-reduce completes it; the head's logits stay sharded for the sampler. */
    const int64_t v0 = g.vocab_off, v1 = g.vocab_off + g.n_vocab;
    m.token = lw(b, m, nm.f("token_embd.weight"), g.n_vocab, D, grp_model(), RAD_SHARD_ROW, v0, v1,
                 false, RAD_ACCESS_VOCAB);
    m.head = lw(b, m, nm.f("output.weight"), g.n_vocab, D, grp_model(), RAD_SHARD_ROW, v0, v1, false,
                RAD_ACCESS_VOCAB);
    if (!m.token || !m.head) return RAD_E_INVAL;
    m.final_gain = wt(b, nm.f("output_norm.weight"), RAD_F32, { D }, grp_model());
    /* A GGUF's table in llama.cpp's blocks is gathered from them (embed_lookup_ggml). */
    const auto tt = m.wty.find(m.token);
    if (tt != m.wty.end())
        m.embed = op(b, "embed_lookup_ggml",
                     { RAD_RANGE("M", 1, T), RAD_INT("n_embd", D), RAD_INT("n_vocab", g.n_vocab),
                       RAD_STR("type", tt->second), RAD_STR("dtype", "bf16"),
                       RAD_INT("vocab_offset", g.vocab_off) },
                     { m.token }, {}, { m.xcol });
    else
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
        /* Bf16, or a single-level affine encoding (codes, a scale per group, an optional
         * codebook: a GGUF's IQ4_NL, Q8_0, Q4_0) read in place from its stored planes, since the
         * table (103 GiB in the 1017) is too large to relayout or copy; it stays on whatever tier
         * holds the container's bytes. */
        for (size_t ti = 0; ti < (m.local_engram ? c.ple.size() : size_t(1)); ++ti) {
            const int64_t table_rows = m.local_engram ? ista::layer_rows[ti] : c.ple_rows;
            const char* tn = m.local_engram
                ? nm.f("per_layer_token_embd.l%lld.weight", (long long)c.ple[ti])
                : nm.f("per_layer_token_embd.weight");
            m.ple_table = m.ple_codes = m.ple_scale = m.ple_tab = 0;
            m.ple_group = 0;
            if (m.local_engram) {
                DS_TRY(map_copy(b, tn, nm.ckpt("blk.%lld.engram_embd.weight", (long long)c.ple[ti])));
            } else if (m.release) {
                /* Every Engram layer's table, joined in layer order (the GGUF's one table). */
                for (size_t i = 0; i < c.ple.size(); i += 8) {
                    std::vector<const char*> parts;
                    for (size_t k = i; k < c.ple.size() && k < i + 8; ++k)
                        parts.push_back(nm.ckpt("layers.%lld.engram.embed.weight", (long long)c.ple[k]));
                    RadNameMap mp = map_base(tn, RAD_MAP_CONCAT);
                    for (const char* p : parts) mp.src[mp.n_src++] = p;
                    DS_TRY(rad_decl_name_map(b, &mp));
                }
            } else {
                DS_TRY(map_copy(b, tn, tn));
            }
            RadEncoding te{};
            const bool known = weight_enc(b, tn, &te);
            if (!known || (rad_enc_is(&te, "plain") && te.n_planes == 1)) {
                m.ple_table = decl_w(b, tn, RAD_BF16, { table_rows, c.ple_dim }, RAD_ACCESS_VOCAB,
                                     RAD_SHARD_NONE, grp_model());
                if (!m.ple_table) return RAD_E_INVAL;
            } else {
                const RadEncPlane* cp = rad_enc_plane(&te, "codes");
                const RadEncPlane* sp = rad_enc_plane(&te, "scale");
                const RadEncPlane* tp = rad_enc_plane(&te, "table");
                const int64_t G = sp ? sp->block[1] : 0;
                if (!rad_enc_is(&te, "affine") || te.transform[0] || !cp || !sp || cp->block[0] != 1 ||
                    cp->block[1] != 1 || sp->block[0] != 1 || G <= 0 || c.ple_dim % G ||
                    te.n_planes != 2 + (tp ? 1 : 0)) {
                    char en[256];
                    rad_enc_format(&te, en, sizeof en);
                    fprintf(stderr, "radiance: deepseek41: the Engram table is %s; it is read as bf16 or "
                                    "as codes with one scale per group of a row and an optional "
                                    "codebook\n", en);
                    return RAD_E_DTYPE;
                }
                m.ple_group = G;
                m.ple_codes = decl_view(b, tn, nullptr, "codes", cp->dtype, { table_rows, c.ple_dim },
                                        RAD_ACCESS_VOCAB, RAD_SHARD_NONE, grp_model());
                m.ple_scale = decl_view(b, m.local_engram ? nm.f("per_layer_token_embd.l%lld.scale", (long long)c.ple[ti])
                                                 : nm.f("per_layer_token_embd.scale"), tn, "scale", sp->dtype,
                                        { table_rows, c.ple_dim / G }, RAD_ACCESS_VOCAB, RAD_SHARD_NONE,
                                        grp_model());
                if (tp)
                    m.ple_tab = decl_view(b, m.local_engram ? nm.f("per_layer_token_embd.l%lld.table", (long long)c.ple[ti])
                                                   : nm.f("per_layer_token_embd.table"), tn, "table", tp->dtype,
                                          { tp->extent[0], tp->extent[1] }, RAD_ACCESS_PER_TOKEN,
                                          RAD_SHARD_NONE, grp_model());
                if (!m.ple_codes || !m.ple_scale || (tp && !m.ple_tab)) return RAD_E_INVAL;
            }
            if (m.local_engram) m.ista_tables[ti] = {m.ple_table, m.ple_codes, m.ple_scale, m.ple_tab, m.ple_group};
        }
        m.tok_map = op(b, "token_map", { RAD_RANGE("M", 1, T), RAD_INT("n_vocab", c.n_vocab) },
                       { m.tmap }, {}, { m.ctok });
    }
    m.layers.resize((size_t)c.n_layer);
    for (int l = 0; l < c.n_layer; ++l) DS_TRY(layer(b, m, m.layers[(size_t)l], l));
    DS_TRY(kv_replicate(b, m));
    if (m.router_bf16 && g.rank == 0)
        fprintf(stderr, "radiance: deepseek41: RADIANCE_EXPERIMENTAL_DS41_ROUTER_BF16: %d routers "
                        "and %d compressor kv|gate projections read as bf16 (gemm_nt_f32bf16), "
                        "%.1f MiB less a card; the load refuses any f32 one with a nonzero low "
                        "half\n",
                m.n_router_bf16, m.n_comp_bf16, (double)m.bf16_saved / (1 << 20));
    m.pcn_fin = post_collapse_norm(b, m, T, m.final_gain, m.act.x, m.h1, m.h0, m.pre_f, m.xcol,
                                   m.final_hidden);
    if (!m.pcn_fin)
        m.cn_fin = collapse_norm(b, m, T, m.final_gain, m.h0, m.pre_f, m.xcol, m.final_hidden);
    if (!m.pcn_fin && !m.cn_fin) {
        m.contract = op(b, "mhc_collapse",
                        { RAD_RANGE("M", 1, T), RAD_INT("n", D), RAD_INT("hc", c.hc), RAD_STR("dtype", "bf16") },
                        {}, { m.h0, m.pre_f }, { m.xcol });
        m.fnorm = rmsnorm(b, m, T, m.final_gain, D, m.xcol, m.final_hidden);
    }
    m.gather = op(b, "gather_rows",
                  { RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n", D), RAD_STR("dtype", "bf16") },
                  {}, { m.final_hidden }, { m.logit_input });
    const auto ht = m.wty.find(m.head);
    if (ht != m.wty.end())
        m.head_op = op(b, "logits_gemm_ggml",
                       { RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n_vocab", g.n_vocab),
                         RAD_INT("n_embd", D), RAD_STR("type", ht->second), RAD_STR("dtype", "bf16") },
                       { m.head }, { m.logit_input }, { m.logits });
    else
        m.head_op = op(b, "logits_gemm",
                       { RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n_vocab", g.n_vocab),
                         RAD_INT("n_embd", D), RAD_STR("dtype", "bf16") },
                       { m.head }, { m.logit_input }, { m.logits });
    /* The drafter after everything the trunk step issues (llama_fp8's rule: declaration order is
     * liveness, and its passes run behind the step). */
    if (m.dsp.on) DS_TRY(dsp_declare(b, m, ctx));
    cut_declare(b, m, ctx);
    pf_note(m, ctx->rank);
    rad_note(b, "DeepSeek-V4.1: %lld layers, hc %lld, window %lld, %zu kv sources, %zu index "
                "sources, %lld experts top-%lld, Engram at %zu layers",
             (long long)c.n_layer, (long long)c.hc, (long long)c.window, c.kv_src.size(),
             c.idx_src.size(), (long long)c.n_expert, (long long)c.top_k, c.ple.size());
    if (m.f8_on) {
        /* In the server log, not only the graph dump: the A/B harness requires this line of an
         * arm that sets the knob (ab_window.sh's F8 part). */
        char msg[400];
        std::snprintf(msg, sizeof msg,
                      "deepseek41: rank %d: %d dense weights read by gemm_nt_bf16f8 "
                      "(RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8): %.1f MiB of E4M3 codes where bf16 held "
                      "%.1f MiB; %d left on gemm_nt%s",
                      ctx->rank, m.f8_n, (double)m.f8_codes / 1048576.0,
                      2.0 * (double)m.f8_codes / 1048576.0, m.f8_kept,
                      m.f8_skip.empty() ? "" : " (some by RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8_EXCLUDE)");
        fprintf(stderr, "radiance: %s\n", msg);
        rad_note(b, "%s", msg);
    }
    if (m.dsp_f8) {
        /* ab_window.sh's Df part requires this line of an arm that sets the knob, with N above 0,
         * and no other arm may have it. */
        char msg[400];
        std::snprintf(msg, sizeof msg,
                      "deepseek41: rank %d: DSpark drafter: %d dense weights read by gemm_nt_bf16f8 "
                      "(RADIANCE_EXPERIMENTAL_DS41_DSPARK_FP8): %.1f MiB of E4M3 codes where bf16 "
                      "held %.1f MiB; %d left on gemm_nt",
                      ctx->rank, m.dsp_f8_n, (double)m.dsp_f8_codes / 1048576.0,
                      2.0 * (double)m.dsp_f8_codes / 1048576.0, m.dsp_f8_kept);
        fprintf(stderr, "radiance: %s\n", msg);
        rad_note(b, "%s", msg);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ step */
/* The drafter's context pass (draft_pass < 0): the trunk step's rows, in order. */
static void dsp_context(RadCtx* x, Model& m, const RadBatch* batch) {
    const auto& d = m.dsp;
    const auto& g = m.g;
    const int64_t T = batch->n_tok, hd = m.c.head_dim;
    if (T <= 0 || m.zero) return;   /* an attention-zero rank keeps no keys (its cache is zeroed) */
    const auto* kb = kv_batch(batch, d.kv);
    if (!kb || !kb->slot_mapping) {
        rad_step_fail(x, "deepseek41: the DSpark cache has no slots on a context pass");
        return;
    }
    const RadOperand pos = praw(batch->positions, RAD_I32, T);
    /* Dsp::ctx_defer: the last T rows the trunk appended. A pass longer than what was appended is a
     * defect of the core's count; it reads from row 0 as before (the drafter's proposals may suffer,
     * nothing else can), and says so once. */
    RadOperand srows = brows(d.src, T);
    if (d.ctx_defer) {
        auto& dm = m.dsp;
        if (T > dm.tap_rows) {
            if (!dm.defer_warned)
                fprintf(stderr, "radiance: deepseek41: rank %d: a DSpark context pass of %lld rows after %lld "
                                "appended (RADIANCE_EXPERIMENTAL_DS41_DSPARK_CTX_DEFER count out of step); read "
                                "from row 0 (logged once)\n", rad_rank(x), (long long)T, (long long)dm.tap_rows);
            dm.defer_warned = true;
        } else {
            srows = brow_slice(d.src, dm.tap_rows - T, T, (int64_t)d.taps.size() * m.c.n_embd);
        }
        dm.tap_rows = 0;
    }
    RAD_ISSUE_N(x, d.c_tab, T, pos, brows(m.cs_a, g.max_ctx));
    RAD_ISSUE_N(x, d.c_fc, T, srows, RAD_W(d.fc), brows(d.c, T));
    RAD_ISSUE_N(x, d.c_norm, T, brows(d.c, T), RAD_W(d.main_norm), brows(d.cn, T));
    for (const auto& L : d.st) {
        RAD_ISSUE_N(x, L.d_kv, T, brows(d.cn, T), RAD_W(L.wkv), brows(d.ckv, T));
        RAD_ISSUE_N(x, L.d_kvn, T, brows(d.ckv, T), RAD_W(L.kv_norm), brows(d.ckvn, T));
        RAD_ISSUE_N(x, L.d_kvrope, T, brows(d.ckvn, T), brows(m.cs_a, g.max_ctx), pos);
        if (L.d_kvq) RAD_ISSUE_N(x, L.d_kvq, T, brows(d.ckvn, T));
        RAD_ISSUE_N(x, L.d_kvst, T, bcol(d.ckvn, 0, hd / 2, T), bcol(d.ckvn, hd / 2, hd / 2, T),
                    praw(kb->slot_mapping, RAD_I32, T), kv_cache(d.kv, L.l));
    }
}

/* The drafter's query pass (draft_pass > 0): `block` rows a drafting sequence, the anchor first. */
static void dsp_query(RadCtx* x, Model& m, const RadBatch* batch) {
    const auto& d = m.dsp;
    const auto& c = m.c;
    const auto& g = m.g;
    const int64_t T = batch->n_tok, S = batch->n_seq, D = c.n_embd;
    const int64_t HD = g.n_head * c.head_dim, hd = c.head_dim;
    if (T <= 0 || S <= 0) return;
    /* The drafter's own block, or its short one (Dsp::block_lo), whose rows' offsets, attention
     * window and walk are ops of their own. */
    const bool lo = d.block_lo > 0 && T == S * d.block_lo;
    if ((!lo && T != S * d.block) || T > d.Q) {
        rad_step_fail(x, "deepseek41: a DSpark query pass staged at a block this drafter has no ops for");
        return;
    }
    const RadOperand pos = praw(batch->positions, RAD_I32, T);
    const RadOperand cu = praw(batch->cu_seqlens, RAD_I32, S + 1);
    const auto* kb = kv_batch(batch, d.kv);
    if (!m.zero && (!kb || !kb->slot_mapping || !kb->block_table || !kb->seqused)) {
        rad_step_fail(x, "deepseek41: the DSpark cache has no block table on a query pass");
        return;
    }

    RAD_ISSUE_N(x, d.embed, T, praw(batch->token_ids, RAD_I32, T), RAD_W(m.token), brows(d.x, T));
    if (d.embed_ar) RAD_ISSUE_N(x, d.embed_ar, T * D, brows(d.x, T), RAD_NONE);
    for (int64_t i = 0; i < c.hc; ++i) RAD_ISSUE_N(x, d.expand, T, brows(d.x, T), bcol(d.h0, i * D, D, T));
    RAD_ISSUE_N(x, d.pre0, T, brows(d.pre_f, T));
    RAD_ISSUE_N(x, d.pre1, T, bcol(d.pre_f, 0, 1, T));
    RAD_ISSUE_N(x, d.q_tab, T, pos, brows(m.cs_a, g.max_ctx));
    RAD_ISSUE_N(x, lo ? d.poff_lo : d.poff_op, T, brows(d.poff, T));
    RAD_ISSUE_N(x, d.apos_op, T, pos, brows(d.poff, T), brows(d.apos, T));

    for (const auto& L : d.st) {
        RAD_ISSUE_N(x, L.mix_a, T, brows(d.h0, T), RAD_W(L.hc_a_fn), RAD_W(L.hc_a_scale), RAD_W(L.hc_a_base),
                    brows(d.pre_a, T), brows(d.post, T), brows(d.comb, T));
        RAD_ISSUE_N(x, L.col_a, T, brows(d.h0, T), brows(d.pre_f, T), brows(d.x, T));
        if (m.zero) {
            RAD_ISSUE_N(x, L.attn_fill, T, brows(d.attn_out, T));
        } else {
            RAD_ISSUE_N(x, L.norm_a, T, brows(d.x, T), RAD_W(L.attn_norm), brows(d.xn, T));
            RAD_ISSUE_N(x, L.q_a, T, brows(d.xn, T), RAD_W(L.wq_a), brows(d.qr, T));
            RAD_ISSUE_N(x, L.q_an, T, brows(d.qr, T), RAD_W(L.q_norm), brows(d.qrn, T));
            RAD_ISSUE_N(x, L.q_b, T, brows(d.qrn, T), RAD_W(L.wq_b), brows(d.q, T));
            RAD_ISSUE_N(x, L.q_rope, T, brows(d.q, T), brows(m.cs_a, g.max_ctx), pos);
            RAD_ISSUE_N(x, L.kv_g, T, brows(d.xn, T), RAD_W(L.wkv), brows(d.kv_, T));
            RAD_ISSUE_N(x, L.kv_n, T, brows(d.kv_, T), RAD_W(L.kv_norm), brows(d.kvn, T));
            RAD_ISSUE_N(x, L.kv_rope, T, brows(d.kvn, T), brows(m.cs_a, g.max_ctx), pos);
            if (L.kv_q) RAD_ISSUE_N(x, L.kv_q, T, brows(d.kvn, T));
            /* The block's keys past the committed end; the next context pass overwrites them. */
            RAD_ISSUE_N(x, L.kv_st, T, bcol(d.kvn, 0, hd / 2, T), bcol(d.kvn, hd / 2, hd / 2, T),
                        praw(kb->slot_mapping, RAD_I32, T), kv_cache(d.kv, L.l));
            RAD_ISSUE_N(x, lo ? L.attn_lo : L.attn, T, brows(d.q, T), kv_cache(d.kv, L.l),
                        praw2(kb->block_table, RAD_I32, S, kb->block_table_pitch), praw(kb->seqused, RAD_I32, S),
                        RAD_NONE, RAD_NONE, RAD_NONE, brows(d.apos, T), cu, RAD_W(L.sinks), brows(d.o, T));
            RAD_ISSUE_N(x, L.o_rope, T, brows(d.o, T), brows(m.cs_a, g.max_ctx), pos);
            if (L.o_q) RAD_ISSUE_N(x, L.o_q, T, brows(d.o, T));
            const int64_t gd = HD / m.gr;
            for (int j = 0; j < m.gr; ++j)
                RAD_ISSUE_N(x, L.o_a[(size_t)j], T, bcol(d.o, j * gd, gd, T), RAD_W(L.wo_a[(size_t)j]),
                            bcol(d.oa, j * c.o_rank, c.o_rank, T));
            RAD_ISSUE_N(x, L.o_b, T, brows(d.oa, T), RAD_W(L.wo_b), brows(d.attn_out, T));
        }
        if (L.ar_attn) RAD_ISSUE_N(x, L.ar_attn, T * D, brows(d.attn_out, T), RAD_NONE);
        RAD_ISSUE_N(x, L.post_a, T, brows(d.attn_out, T), brows(d.h0, T), brows(d.post, T), brows(d.comb, T),
                    brows(d.h1, T));
        RAD_ISSUE_N(x, L.mix_f, T, brows(d.h1, T), RAD_W(L.hc_f_fn), RAD_W(L.hc_f_scale), RAD_W(L.hc_f_base),
                    brows(d.pre_f, T), brows(d.post, T), brows(d.comb, T));
        RAD_ISSUE_N(x, L.col_f, T, brows(d.h1, T), brows(d.pre_a, T), brows(d.x, T));
        RAD_ISSUE_N(x, L.norm_f, T, brows(d.x, T), RAD_W(L.ffn_norm), brows(d.act.x, T));
        L.moe.step(x, batch);
        RAD_ISSUE_N(x, L.post_f, T, brows(d.act.x, T), brows(d.h1, T), brows(d.post, T), brows(d.comb, T),
                    brows(d.h0, T));
    }

    RAD_ISSUE_N(x, d.contract, T, brows(d.h0, T), brows(d.pre_f, T), brows(d.x, T));
    RAD_ISSUE_N(x, d.fnorm, T, brows(d.x, T), RAD_W(d.norm), brows(d.fin, T));
    const int64_t N = batch->n_out;
    if (N <= 0) return;
    RAD_ISSUE_N(x, d.gather, N, brows(d.fin, T), praw(batch->out_ids, RAD_I32, N), brows(d.rows, N));
    RAD_ISSUE_N(x, d.head, N, brows(d.rows, N), RAD_W(m.head), brows(d.logits, N));
    if (d.merge) {
        RAD_ISSUE_N(x, d.topk, N, brows(d.logits, N), brows(d.cand_, N), brows(d.unary, N), brows(d.pairs, N));
        RAD_ISSUE(x, d.pgather, brows(d.pairs, N), brows(d.gath, N));
        RAD_ISSUE_N(x, d.merge, N, brows(d.gath, N), brows(d.cand_, N), brows(d.unary, N));
    } else {
        RAD_ISSUE_N(x, d.topk, N, brows(d.logits, N), brows(d.cand_, N), brows(d.unary, N), RAD_NONE);
    }
    RAD_ISSUE_N(x, lo ? d.select_lo : d.select, S, brows(d.cand_, N), brows(d.unary, N), RAD_NONE,
                praw(batch->token_ids, RAD_I32, T), RAD_W(d.mk1), RAD_W(d.mk2), brows(d.tokens, S));
}

/* The ffn connection's mix, issued by the MoE block behind its routed gate_up (MoeFP8::Beside). */
struct MixBeside {
    const Layer* L;
    const Model* m;
    int64_t T;
    int64_t r0 = 0;   /* the layer's first row (the per-row cut); the mix runs on [r0, T) */
    /* The reads pf_tail issues (the next layer's Layer::pf) and their counters; null: none. */
    const std::vector<PfRead>* pf = nullptr;
    Model::PfStats* pfs = nullptr;
};
/* Reads `rd` into the on-die cache, in order, on lane 1 behind a join from lane 0 (so they start
 * once everything issued so far is done, and lane 0 goes on), or on lane 0 itself where launches
 * chain, as prefetch_beside does (rad_arch.h). They only read, so no value changes. `engram`: the
 * reads behind an Engram GEMM (Layer::pf_post), counted apart. */
static void pf_issue(RadCtx* x, const std::vector<PfRead>& rd, Model::PfStats& st, bool engram) {
    if (rd.empty()) return;
    const bool l0 = prefetch_lane0();
    if (!l0 && (rad_lane_join(x, 0, 1) != RAD_OK || rad_lane(x, 1) != RAD_OK)) {
        rad_lane(x, 0);
        return;
    }
    uint64_t n = 0, bytes = 0;
    for (const PfRead& r : rd)
        if (RAD_ISSUE(x, r.op, RAD_W(r.w)) == RAD_OK) {
            ++n;
            bytes += (uint64_t)r.bytes;
        }
    if (!l0) rad_lane(x, 0);
    if (!n) return;
    ++st.points;
    st.reads += n;
    st.bytes += bytes;
    if (engram) {
        ++st.epoints;
        st.ereads += n;
        st.ebytes += bytes;
    }
}
/* The next layer's reads, while the card waits for this layer's CPU rows (MoeFP8::Beside::tail,
 * called once the block's card work before the gather is issued). */
static void pf_tail(RadCtx* x, void* a) {
    const MixBeside& p = *(const MixBeside*)a;
    if (p.pf && p.pfs) pf_issue(x, *p.pf, *p.pfs, false);
}

static void mix_f_beside(RadCtx* x, void* a) {
    const MixBeside& p = *(const MixBeside*)a;
    const Layer& L = *p.L;
    const Model& m = *p.m;
    const int64_t T = p.T, r0 = p.r0, n = T - r0, hc = m.c.hc;
    RAD_ISSUE_N(x, L.mix_f, n, brow_slice(m.h1, r0, n, hc * m.c.n_embd), RAD_W(L.hc_f_fn), RAD_W(L.hc_f_scale),
                RAD_W(L.hc_f_base), brow_slice(m.pre_f, r0, n, hc), brow_slice(m.post, r0, n, hc),
                brow_slice(m.comb, r0, n, hc * hc));
}

static void step(RadCtx* x, const RadBatch* batch) {
    auto& m = *models[rad_rank(x)];
    /* A drafter pass: the sign gives the kind (rad_runtime.h); zero is a trunk step. */
    if (batch->draft_pass != 0) {
        if (!m.dsp.on) {
            rad_step_fail(x, "deepseek41: a drafter pass and no DSpark drafter is declared (the checkpoint "
                             "carries none, or declare ran at max_spec 0)");
            return;
        }
        if (batch->draft_pass > 0) dsp_query(x, m, batch);
        else dsp_context(x, m, batch);
        return;
    }
    const auto& c = m.c;
    const auto& g = m.g;
    const int64_t T = batch->n_tok, S = batch->n_seq, D = c.n_embd;
    const int64_t HD = g.n_head * c.head_dim, hd = c.head_dim;
    const RadOperand pos = praw(batch->positions, RAD_I32, T);
    const RadOperand cu = praw(batch->cu_seqlens, RAD_I32, S + 1);
    /* Dsp::ctx_defer: this step's taps go after the rows still waiting for a context pass. Past the
     * rows declared (the core's count out of step) they start over at row 0, logged once: the
     * drafter's proposals may suffer, nothing else reads these rows. */
    int64_t tap_off = 0;
    if (m.dsp.on && m.dsp.ctx_defer && m.dsp.src) {
        if (m.dsp.tap_rows + T > m.dsp.src_rows) {
            if (!m.dsp.defer_warned)
                fprintf(stderr, "radiance: deepseek41: rank %d: %lld DSpark tap rows waiting and %lld more do "
                                "not fit %lld (RADIANCE_EXPERIMENTAL_DS41_DSPARK_CTX_DEFER count out of step); "
                                "started over (logged once)\n", rad_rank(x), (long long)m.dsp.tap_rows,
                        (long long)T, (long long)m.dsp.src_rows);
            m.dsp.defer_warned = true;
            m.dsp.tap_rows = 0;
        }
        tap_off = m.dsp.tap_rows;
        m.dsp.tap_rows += T;
    }
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

    /* The Engram's row ids are a function of the batch's tokens alone (token_map, each Engram
     * layer's token history), never of an activation, so every Engram layer's ngram_ids is issued
     * here, before the first block, each into its own host buffer (rad_block_ple.h's ids() does
     * the same). A host-site gather (the table on the disk or RAM tier) joins on the last device
     * op that touched the host arena: these few ops, or the previous Engram layer's copy of its
     * rows. It then reads its rows while the device runs the blocks the host has already issued,
     * where ngram_ids at its layer (and device-domain rows) made the device drain before every
     * gather and idle through it. */
    for (auto& L : m.layers) {
        if (!L.e_ids) continue;
        const int l = L.l;
        const auto* tk = kv_batch(batch, m.tok);
        const int32_t* tk_idx = tk ? tk->state_index : nullptr;
        int64_t Dn = 0, DT = 0;
        batch_split(batch, &Dn, &DT);
        const int64_t P = S - Dn;
        if (Dn > 0)
            RAD_ISSUE_N(x, L.e_ids, DT, brows(m.ctok, DT), kv_cache(m.tok, l),
                        praw(batch->cu_seqlens, RAD_I32, Dn + 1), RAD_W(L.e_mult),
                        RAD_W(L.e_vsz), RAD_W(L.e_off), praw(tk_idx, RAD_I32, Dn),
                        praw(batch->ctx_lens, RAD_I32, Dn),
                        praw(batch->num_accepted, RAD_I32, Dn), brows(L.ids, DT), RAD_NONE);
        if (P > 0)
            RAD_ISSUE_N(x, L.e_ids, T, brows(m.ctok, T), kv_cache(m.tok, l),
                        praw(batch->cu_seqlens + Dn, RAD_I32, P + 1), RAD_W(L.e_mult),
                        RAD_W(L.e_vsz), RAD_W(L.e_off),
                        praw(tk_idx ? tk_idx + Dn : nullptr, RAD_I32, P),
                        praw(batch->ctx_lens + Dn, RAD_I32, P), RAD_NONE, brows(L.ids, T),
                        RAD_NONE);
    }

    /* The consumer skip (cut_declare): a pass the engine cut runs layers [0, F) in full, layer F's
     * key ops alone and nothing past it; no final norm, no head. Read only once the declaration
     * was taken: an engine without the call has no such field. */
    /* The per-row cut (cut_declare, RadBatch::consumer_row0): each layer L on its own rows, in full on
     * [row0[L], T) and its key ops on [row0[L - 1], T) (from 0 at and below the boundary), the rows
     * of every op offset to them and every per-row buffer kept at the step's row index; a layer with
     * no key rows issues nothing, and neither does any after it. consumer_full is then the first such
     * layer's F, which row0 already says. */
    const int32_t* const row0 = m.cut_on && m.cut_rows ? batch->consumer_row0 : nullptr;
    if (row0) {
        bool ok = S == 1 && batch->n_seq_decode == 0 && batch->n_spec == 0 && batch->n_mm_rows == 0 &&
                  batch->n_out <= 1;
        for (int l = 0; l < (int)c.n_layer && ok; ++l)
            ok = row0[l] >= 0 && row0[l] <= T && (l >= m.cut_b || row0[l] == 0) &&
                 (l <= m.cut_b || row0[l] >= row0[l - 1]);
        /* The sampled row (the chunk's last, S = 1) must be one every layer runs. */
        if (ok && batch->n_out > 0) ok = row0[c.n_layer - 1] <= T - 1;
        if (!ok) {
            rad_step_fail(x, "deepseek41: a per-row cut outside the declared consumer suffix (rows past the "
                             "step, a layer's rows not nested in the one below, a sampled row cut, or a "
                             "step that mixes decode rows, media or speculation)");
            return;
        }
        ++m.cuts.row_passes;
        for (int l = 0; l < (int)c.n_layer; ++l) {
            const int64_t kf = l <= m.cut_b ? 0 : row0[l - 1], ff = row0[l];
            m.cuts.rows_full += (uint64_t)(T - std::min<int64_t>(ff, T));
            m.cuts.rows_key += (uint64_t)std::max<int64_t>(std::min<int64_t>(ff, T) - kf, 0);
            m.cuts.rows_skipped += (uint64_t)std::min<int64_t>(kf, T);
        }
    }
    const int32_t cut_full = m.cut_on && !row0 ? batch->consumer_full : 0;
    if (cut_full != 0 &&
        (cut_full < m.cut_b || cut_full >= (int32_t)c.n_layer || batch->n_out != 0 || S != 1 ||
         batch->n_seq_decode != 0 || batch->n_spec != 0 || batch->n_mm_rows != 0)) {
        rad_step_fail(x, "deepseek41: a cut pass outside the declared consumer suffix (consumer_full "
                         "past its bounds, or a pass that samples, mixes decode rows or speculates)");
        return;
    }
    if (cut_full) {
        ++m.cuts.passes;
        m.cuts.full += (uint64_t)cut_full;
        m.cuts.key += 1;
        m.cuts.skipped += (uint64_t)(c.n_layer - cut_full - 1);
    }

    /* The on-die reads (Layer::pf), at decode heights only: prompt steps keep their reads to
     * themselves (rad_arch.h, kPrefetchRows). */
    const bool pf_live = batch->phase == RAD_PHASE_DECODE && T <= kPrefetchRows;
    const uint64_t pf_reads0 = m.pfs.reads;
    for (auto& L : m.layers) {
        const int l = L.l;
        const int64_t r = L.r;
        /* A cut pass: past layer F nothing; layer F its collapse, norm and key ops (below). */
        if (cut_full && l > cut_full) break;
        /* The rows this layer runs (the per-row cut; the whole step otherwise): its key ops -- the
         * collapse and norm that feed them, the window key, at a source the compressor and the index
         * keys -- on [rk, T), everything else on [rf, T). A source is at or below the boundary, so its
         * key rows are all of the step's. */
        const int64_t rk = row0 && l > m.cut_b ? row0[l - 1] : 0;
        const int64_t rf = row0 ? row0[l] : 0;
        if (row0 && rk >= T) break;
        const bool key_only = (cut_full && l == cut_full) || (row0 && rf >= T);
        const int64_t nk = T - rk, nf = T - rf;
        if (L.src && rk != 0) {
            rad_step_fail(x, "deepseek41: a per-row cut that leaves a key source's rows out");
            return;
        }
        const int64_t HN = c.hc * D, hc = c.hc;
        /* Rows [rk, T) / [rf, T) of a [T, w] buffer; a column slice of them; the rows' positions. */
        const auto KR = [&](rad_buf h, int64_t w) { return brow_slice(h, rk, nk, w); };
        const auto FR = [&](rad_buf h, int64_t w) { return brow_slice(h, rf, nf, w); };
        const auto FC = [&](rad_buf h, int64_t w, int64_t c0, int64_t cw) { return bcol_at(h, rf, w, c0, cw, nf); };
        const RadOperand kpos = praw(batch->positions + rk, RAD_I32, nk);
        const RadOperand fpos = praw(batch->positions + rf, RAD_I32, nf);
        if (L.e_ids) {
            const int64_t H = c.ple_heads;
            const int pi = c.ple_index(l);
            if (m.local_engram && (pi < 0 || pi >= int(sizeof(m.ista_tables) / sizeof(m.ista_tables[0])))) {
                rad_step_fail(x, "deepseek41: local Engram layer has no declared table");
                return;
            }
            const auto* et = m.local_engram ? &m.ista_tables[pi] : nullptr;
            const int64_t egroup = et ? et->group : m.ple_group;
            const rad_weight eplain = et ? et->plain : m.ple_table;
            const rad_weight ecodes = et ? et->codes : m.ple_codes;
            const rad_weight escale = et ? et->scale : m.ple_scale;
            const rad_weight etab = et ? et->tab : m.ple_tab;
            /* T rows of operands; the gather's band is T * H, one table row per head per token,
             * read through the ids and rows as flat (rad_block_ple.h does the same). */
            if (egroup > 0)
                RAD_ISSUE_N(x, L.e_emb, T * H, brows(L.ids, T), RAD_W(ecodes), RAD_W(escale),
                            etab ? RAD_W(etab) : RAD_NONE, brows(m.emb, T));
            else
                RAD_ISSUE_N(x, L.e_emb, T * H, brows(L.ids, T), RAD_W(eplain), brows(m.emb, T));
            RAD_ISSUE_N(x, L.e_copy, T, brows(m.emb, T), brows(m.embd, T));
            /* This layer's own reads go right behind its Engram GEMM, which would have evicted
             * them (..._PREFETCH_ENGRAM=1): beside the all_gather of the split rows, the gate and
             * the attention connection, until the attention's GEMVs reach them. */
            if (L.e_gather) {
                RAD_ISSUE_N(x, L.e_gemm, T, brows(m.embd, T), RAD_W(L.e_kv), brows(m.ekv_part, T));
                if (pf_live) pf_issue(x, L.pf_post, m.pfs, true);
                RAD_ISSUE_N(x, L.e_gather, T * m.esplit, brows(m.ekv_part, T), brows(m.ekv, T));
            } else {
                RAD_ISSUE_N(x, L.e_gemm, T, brows(m.embd, T), RAD_W(L.e_kv), brows(m.ekv, T));
                if (pf_live) pf_issue(x, L.pf_post, m.pfs, true);
            }
            RAD_ISSUE_N(x, L.e_gate_op, T, brows(m.h0, T), bcol(m.ekv, 0, c.hc * D, T),
                        bcol(m.ekv, c.hc * D, D, T), RAD_W(L.e_gate), brows(m.h1, T));
            RAD_ISSUE_N(x, L.e_back, T, brows(m.h1, T), brows(m.h0, T));
        }
        if (L.tap && !key_only)
            RAD_ISSUE_N(x, L.tap, T, brows(m.h0, T),
                        m.dsp.ctx_defer ? bcol_at(m.dsp.src, tap_off, (int64_t)m.dsp.taps.size() * D,
                                                  L.tap_col * D, D, T)
                                        : bcol(m.dsp.src, L.tap_col * D, D, T));
        /* Issue order of the connections' mixes (Ctx::haz_mark: a recorded pass leaves a dispatch
         * unordered when it touches nothing in flight). Each connection's collapse reads the gates
         * the previous connection's mix wrote, never what its own mix writes, and nothing reads a
         * mix's output before the next mhc_post. So, same kernels on the same operands:
         * - the collapse goes first. Where it is its own op (no fused mhc_post before it: the
         *   first layer, after an Engram block, the attention-zero rank) the mix's second launch,
         *   ordered anyway, is then the barrier the norm needs;
         * - on an attention rank the attention mix goes between the output projection's groups
         *   and attn_output_b, where its launches overlap those GEMVs instead of running nearly
         *   alone at the layer's start; on the attention-zero rank it follows the collapse. */
        const bool late = !m.zero;
        const auto mix_a = [&] {
            RAD_ISSUE_N(x, L.mix_a, nf, FR(m.h0, HN), RAD_W(L.hc_a_fn), RAD_W(L.hc_a_scale),
                        RAD_W(L.hc_a_base), FR(m.pre_a, hc), FR(m.post, hc), FR(m.comb, hc * hc));
        };
        const auto col_a = [&] {
            if (L.cn_a)
                RAD_ISSUE_N(x, L.cn_a, nk, KR(m.h0, HN), KR(m.pre_f, hc), KR(m.xcol, D),
                            RAD_W(L.attn_norm), KR(m.xn, D));
            else if (L.col_a)   /* 0 under pcn_a: issued for the previous post_f */
                RAD_ISSUE_N(x, L.col_a, nk, KR(m.h0, HN), KR(m.pre_f, hc), KR(m.xcol, D));
        };
        col_a();
        if (!late && !key_only) mix_a();
        if (m.zero) {
            /* An attention-zero rank stores no keys (its caches are zeroed). */
            if (!key_only) RAD_ISSUE_N(x, L.attn_fill, nf, FR(m.attn_out, D));
        } else {
            if (L.norm_a)
                RAD_ISSUE_N(x, L.norm_a, nk, KR(m.xcol, D), RAD_W(L.attn_norm), KR(m.xn, D));
            const rad_buf cs = r > 0 ? m.cs_b : m.cs_a;
            /* The compressor (at a kv source) and the indexer (at an index source) read xn and qrn
             * as the query and key chains do, and their only order with them is the compressed
             * angles: c_tab rewrites cs_b after the query and the window key read it, and at a
             * source the index query's rope reads it after. Each of their ops goes between the
             * query/key pairs at the first point its inputs are issued, so a recorded pass overlaps
             * three or four chains instead of running the compressor and the indexer alone after
             * the window key. Same kernels on the same operands. */
            const bool ctail = L.src && r > 1;
            int64_t rows = T, tpitch = 1;
            RadOperand page = RAD_NONE, bpos = pos;
            const auto* cb = L.src ? kv_batch(batch, m.comp[r]) : nullptr;
            const auto* tb = ctail ? kv_batch(batch, m.tail[r]) : nullptr;
            if (L.src) {
                const auto* ib = kv_batch(batch, m.index[r]);
                if (!cb || !ib || !cb->block_table || !ib->block_table) {
                    rad_step_fail(x, "deepseek41: a compressed group has no block table");
                    return;
                }
                page = praw(cb->slot_mapping, RAD_I32, T);
                if (ctail) {
                    if (!tb || !tb->state_index) {
                        rad_step_fail(x, "deepseek41: the compressor tail has no state slots");
                        return;
                    }
                    tpitch = tb->state_index_pitch > 0 ? tb->state_index_pitch : 1;
                    rows = m.work;
                    page = brows(m.page, rows);
                    bpos = brows(m.bpos, rows);
                }
            }
            /* The query and key chains share only xn. Issued alternately, each op's predecessor
             * is in the other chain, so a recorded pass can leave it unordered behind it (the
             * tape's hazard check) and the two chains' dispatches overlap; one after the other,
             * every op but kv_g waits for the one before. Same kernels on the same operands. */
            const int64_t ql = c.q_lora;
            const auto q_a = [&] { RAD_ISSUE_N(x, L.q_a, nf, FR(m.xn, D), RAD_W(L.wq_a), FR(m.qr, ql)); };
            const auto q_an = [&] { RAD_ISSUE_N(x, L.q_an, nf, FR(m.qr, ql), RAD_W(L.q_norm), FR(m.qrn, ql)); };
            const auto q_b = [&] { RAD_ISSUE_N(x, L.q_b, nf, FR(m.qrn, ql), RAD_W(L.wq_b), FR(m.q, HD)); };
            const auto q_rope = [&] { RAD_ISSUE_N(x, L.q_rope, nf, FR(m.q, HD), brows(cs, g.max_ctx), fpos); };
            const auto kv_g = [&] { RAD_ISSUE_N(x, L.kv_g, nk, KR(m.xn, D), RAD_W(L.wkv), KR(m.kv, hd)); };
            /* The window key's norm, rope and round trip: one op (kv_f) where declared, else
             * kv_n here and kv_rope / kv_q at the next two points. */
            const auto kv_n = [&] {
                if (L.kv_f)
                    RAD_ISSUE_N(x, L.kv_f, nk, KR(m.kv, hd), RAD_W(L.kv_norm), KR(m.kvn, hd),
                                brows(cs, g.max_ctx), kpos);
                else
                    RAD_ISSUE_N(x, L.kv_n, nk, KR(m.kv, hd), RAD_W(L.kv_norm), KR(m.kvn, hd));
            };
            const auto kv_rope = [&] {
                if (!L.kv_f) RAD_ISSUE_N(x, L.kv_rope, nk, KR(m.kvn, hd), brows(cs, g.max_ctx), kpos);
            };
            const auto kv_q = [&] {
                if (L.kv_q) RAD_ISSUE_N(x, L.kv_q, nk, KR(m.kvn, hd));
            };
            const auto kv_st = [&] {
                RAD_ISSUE_N(x, L.kv_st, nk, bcol_at(m.kvn, rk, hd, 0, hd / 2, nk),
                            bcol_at(m.kvn, rk, hd, hd / 2, hd / 2, nk),
                            praw(wb->slot_mapping + rk, RAD_I32, nk), kv_cache(m.win, l));
            };
            const auto c_gemm = [&] {
                if (L.src)
                    RAD_ISSUE_N(x, L.c_gemm, T, brows(m.xn, T), RAD_W(L.comp_w),
                                brows(ctail ? m.kvs : m.pooled, T));
            };
            const auto c_work = [&] {
                if (ctail)
                    RAD_ISSUE_N(x, L.c_work, T, brows(m.kvs, T), pos, cu, kv_cache(m.tail[r], l),
                                praw2(tb->state_index, RAD_I32, S, tpitch),
                                praw2(cb->block_table, RAD_I32, S, cb->block_table_pitch), RAD_NONE,
                                brows(m.stage, rows * r), brows(m.page, rows), brows(m.bpos, rows),
                                brows(m.nc, S));
            };
            const auto c_pool = [&] {
                if (ctail)
                    RAD_ISSUE_N(x, L.c_pool, rows, brows(m.stage, rows * r), brows(m.pooled, rows),
                                brows(m.page, rows));
            };
            const auto c_norm = [&] {
                if (L.src)
                    RAD_ISSUE_N(x, L.c_norm, rows, brows(m.pooled, rows), RAD_W(L.comp_norm),
                                brows(m.lat, rows));
            };
            const auto c_tab = [&] {
                if (L.src) RAD_ISSUE_N(x, L.c_tab, rows, bpos, brows(m.cs_b, g.max_ctx));
            };
            const auto k_gemm = [&] {
                if (L.src) RAD_ISSUE_N(x, L.k_gemm, rows, brows(m.lat, rows), RAD_W(L.ik_w), brows(m.ik, rows));
            };
            /* The index key's norm, rope and round trip (k_f where declared); the latent's rope and
             * round trip (l_f where declared); the index query's (i_f where declared). Each part
             * function issues the fused op at the chain's first point and nothing at the others. */
            const auto k_norm = [&] {
                if (!L.src) return;
                if (L.k_f)
                    RAD_ISSUE_N(x, L.k_f, rows, brows(m.ik, rows), RAD_W(L.ik_norm), brows(m.ikn, rows),
                                brows(m.cs_b, g.max_ctx), bpos);
                else
                    RAD_ISSUE_N(x, L.k_norm, rows, brows(m.ik, rows), RAD_W(L.ik_norm), brows(m.ikn, rows));
            };
            const auto k_rope = [&] {
                if (L.src && !L.k_f) RAD_ISSUE_N(x, L.k_rope, rows, brows(m.ikn, rows), brows(m.cs_b, g.max_ctx), bpos);
            };
            const auto k_q = [&] {
                if (L.src && L.k_q) RAD_ISSUE_N(x, L.k_q, rows, brows(m.ikn, rows));
            };
            const auto k_st = [&] {
                if (L.src) RAD_ISSUE_N(x, L.k_st, rows, brows(m.ikn, rows), page, kv_cache(m.index[r], l));
            };
            const auto l_rope = [&] {
                if (!L.src) return;
                if (L.l_f)
                    RAD_ISSUE_N(x, L.l_f, rows, brows(m.lat, rows), brows(m.cs_b, g.max_ctx), bpos);
                else
                    RAD_ISSUE_N(x, L.l_rope, rows, brows(m.lat, rows), brows(m.cs_b, g.max_ctx), bpos);
            };
            const auto l_q = [&] {
                if (L.src && L.l_q) RAD_ISSUE_N(x, L.l_q, rows, brows(m.lat, rows));
            };
            const auto l_st = [&] {
                if (L.src) RAD_ISSUE_N(x, L.l_st, rows, brows(m.lat, rows), page, kv_cache(m.comp[r], l));
            };
            const auto c_tail = [&] {
                if (ctail && L.c_tail)
                    RAD_ISSUE_N(x, L.c_tail, T, brows(m.kvs, T), pos, cu, kv_cache(m.tail[r], l),
                                praw2(tb->state_index, RAD_I32, S, tpitch));
            };
            const int64_t iqw = c.ih * c.idim;
            const auto i_q = [&] {
                if (L.i_score) RAD_ISSUE_N(x, L.i_q, nf, FR(m.qrn, ql), RAD_W(L.iq_w), FR(m.iq, iqw));
            };
            const auto i_rope = [&] {
                if (!L.i_score) return;
                if (L.i_f)
                    RAD_ISSUE_N(x, L.i_f, nf, FR(m.iq, iqw), brows(m.cs_b, g.max_ctx), fpos);
                else
                    RAD_ISSUE_N(x, L.i_rope, nf, FR(m.iq, iqw), brows(m.cs_b, g.max_ctx), fpos);
            };
            const auto i_qq = [&] {
                if (L.i_score && L.i_qq) RAD_ISSUE_N(x, L.i_qq, nf, FR(m.iq, iqw));
            };
            const auto i_w = [&] {
                if (L.i_score) RAD_ISSUE_N(x, L.i_w, nf, FR(m.xn, D), RAD_W(L.iproj), FR(m.iw, c.ih));
            };
            const auto i_score = [&] {   /* the scores, the candidates and the selection */
                if (!L.i_score) return;
                const int src = c.src_of[(size_t)l];
                const auto* ib = kv_batch(batch, m.index[r]);
                const int64_t live = std::min<int64_t>(
                    g.max_ctx / r + 1, ((int64_t)batch->max_ctx_len + batch->max_q_len) / r + 1);
                const int64_t kw = keep_words(live, c.cand_block);
                const bool uses = c.cand >= 0 && c.cand < l;
                /* Run by run (index_runs()): score R rows into the plane, then select from them
                 * before the next run overwrites it. Rows are independent, so each row's score,
                 * candidates and selection are the whole step's. Under the per-row cut the runs
                 * cover the layer's rows [rf, T): the operands start there (with one sequence every
                 * row is cu's sequence 0 wherever it starts), the keep mask and the selection at the
                 * step's own rows. */
                const int64_t n = (nf + m.irows - 1) / m.irows, R = (nf + n - 1) / n;
                for (int64_t p = 0; p * R < nf; ++p) {
                    const int64_t r0 = p * R, rows = std::min(R, nf - r0), at = rf + r0;
                    const RadOperand prow = praw(batch->positions + at, RAD_I32, rows);
                    RAD_ISSUE_N(x, L.i_runs[(size_t)p], R, FR(m.iq, iqw), FR(m.iw, c.ih),
                                kv_cache(m.index[r], src),
                                praw2(ib->block_table, RAD_I32, S, ib->block_table_pitch), fpos, cu,
                                bcol(m.score, 0, live, R));
                    if (L.i_cand)
                        RAD_ISSUE_N(x, L.i_cand, rows, bcol(m.score, 0, live, rows), prow,
                                    bcol_at(m.keep, at, m.kwords, 0, kw, rows));
                    RAD_ISSUE_N(x, L.i_topk, rows, bcol(m.score, 0, live, rows),
                                uses ? bcol_at(m.keep, at, m.kwords, 0, kw, rows) : RAD_NONE, prow,
                                brow_slice(m.sel, at, rows, c.itopk));
                }
            };

            if (key_only) {
                /* The consumer skip's key ops: the order below with the query, the indexer and
                 * everything after the stores left out, so every store gets the full layer's bytes
                 * (cut_declare). */
                c_gemm();
                kv_g();
                c_work();
                kv_n();
                c_pool();
                kv_rope();
                c_tail(); c_norm();
                kv_q();
                k_gemm(); c_tab();
                kv_st();
                k_norm();
                l_rope(); k_rope();
                l_q(); k_q(); l_st(); k_st();
                break;
            }
            c_gemm();
            q_a(); kv_g();
            i_w(); c_work();
            q_an(); kv_n();
            i_q(); c_pool();
            q_b(); kv_rope();
            if (!L.src) i_rope();   /* at a source it reads the angles c_tab rewrites */
            c_tail(); c_norm();
            q_rope(); kv_q();
            if (!L.src) i_qq();
            k_gemm(); c_tab();
            kv_st();
            k_norm();
            if (L.src) i_rope();
            l_rope(); k_rope();
            if (L.src) i_qq();
            l_q(); k_q(); l_st(); k_st(); i_score();

            /* The attention over the layer's rows: with one sequence its window base comes from the
             * operand's last row, the step's own (attn_dec_emit), wherever the rows start. */
            if (r > 0) {
                const int src = c.src_of[(size_t)l];
                const auto* cb = kv_batch(batch, m.comp[r]);
                RAD_ISSUE_N(x, L.attn, nf, FR(m.q, HD), kv_cache(m.win, l), wbt, wused,
                            kv_cache(m.comp[r], src),
                            praw2(cb->block_table, RAD_I32, S, cb->block_table_pitch), FR(m.sel, c.itopk),
                            fpos, cu, RAD_W(L.sinks), FR(m.o, HD));
            } else {
                RAD_ISSUE_N(x, L.attn, nf, FR(m.q, HD), kv_cache(m.win, l), wbt, wused, RAD_NONE,
                            RAD_NONE, RAD_NONE, fpos, cu, RAD_W(L.sinks), FR(m.o, HD));
            }
            if (L.o_f) {
                RAD_ISSUE_N(x, L.o_f, nf, FR(m.o, HD), brows(cs, g.max_ctx), fpos);
            } else {
                RAD_ISSUE_N(x, L.o_rope, nf, FR(m.o, HD), brows(cs, g.max_ctx), fpos);
                if (L.o_q) RAD_ISSUE_N(x, L.o_q, nf, FR(m.o, HD));
            }
            const int64_t gd = HD / m.gr, ow = m.gr * c.o_rank;
            for (int j = 0; j < m.gr; ++j)
                RAD_ISSUE_N(x, L.o_a[(size_t)j], nf, FC(m.o, HD, j * gd, gd), RAD_W(L.wo_a[(size_t)j]),
                            FC(m.oa, ow, j * c.o_rank, c.o_rank));
            if (late) mix_a();
            RAD_ISSUE_N(x, L.o_b, nf, FR(m.oa, ow), RAD_W(L.wo_b), FR(m.attn_out, D));
        }
        if (key_only) break;   /* an attention-zero rank's key layer: its collapse alone */
        if (L.ar_attn) RAD_ISSUE_N(x, L.ar_attn, nf * D, FR(m.attn_out, D), RAD_NONE);
        if (L.pcn_f)   /* post_a, col_f and norm_f in one */
            RAD_ISSUE_N(x, L.pcn_f, nf, FR(m.attn_out, D), FR(m.h0, HN), FR(m.post, hc),
                        FR(m.comb, hc * hc), FR(m.h1, HN), FR(m.pre_a, hc), FR(m.xcol, D),
                        RAD_W(L.ffn_norm), FR(m.act.x, D));
        else
            RAD_ISSUE_N(x, L.post_a, nf, FR(m.attn_out, D), FR(m.h0, HN), FR(m.post, hc),
                        FR(m.comb, hc * hc), FR(m.h1, HN));

        /* The ffn connection's mix is likewise needed first by the next mhc_post, so the MoE block
         * issues it (MoeFP8::Beside) behind its routed and shared gate_up, where a recorded pass
         * overlaps it with those GEMMs instead of with the collapse and the norm (or, under the
         * fused post, with nothing). It reads h1 and writes what post_a (fused or not) has read. */
        const auto col_f = [&] {
            if (L.pcn_f) {
            } else if (L.cn_f) {
                RAD_ISSUE_N(x, L.cn_f, nf, FR(m.h1, HN), FR(m.pre_a, hc), FR(m.xcol, D),
                            RAD_W(L.ffn_norm), FR(m.act.x, D));
            } else {
                RAD_ISSUE_N(x, L.col_f, nf, FR(m.h1, HN), FR(m.pre_a, hc), FR(m.xcol, D));
                RAD_ISSUE_N(x, L.norm_f, nf, FR(m.xcol, D), RAD_W(L.ffn_norm), FR(m.act.x, D));
            }
        };
        col_f();
        MixBeside mb{ &L, &m, T };
        mb.r0 = rf;
        /* The next layer's reads (decode passes only). Ahead of an Engram layer that is at most
         * its engram_kv, since its 79 MB GEMM would evict anything else before the attention
         * (pf_layer()); by default nothing. */
        const Layer* nl = L.l + 1 < c.n_layer ? &m.layers[(size_t)L.l + 1] : nullptr;
        if (nl && pf_live && !nl->pf.empty()) {
            mb.pf = &nl->pf;
            mb.pfs = &m.pfs;
        }
        MoeFP8::Beside bs;
        bs.fn = mix_f_beside;
        bs.arg = &mb;
        bs.tail = pf_tail;
        if (row0) {
            /* The layer's rows alone, one pass (a chunk is at most the block's pass height). */
            bs.first_row = rf;
            L.moe.pass(x, T, rf, nf, true, &bs);
            L.moe.route_report(x);
        } else {
            L.moe.step(x, batch, bs);
        }
        /* post_f, or post_f with the next collapse + norm (the next layer's pcn_a, or pcn_fin). Under the
         * per-row cut the next layer's key rows are this layer's rows. */
        const Layer* nx = L.l + 1 < c.n_layer ? &m.layers[(size_t)L.l + 1] : nullptr;
        if (nx && nx->pcn_a)
            RAD_ISSUE_N(x, nx->pcn_a, nf, FR(m.act.x, D), FR(m.h1, HN), FR(m.post, hc),
                        FR(m.comb, hc * hc), FR(m.h0, HN), FR(m.pre_f, hc), FR(m.xcol, D),
                        RAD_W(nx->attn_norm), FR(m.xn, D));
        else if (!nx && m.pcn_fin)
            RAD_ISSUE_N(x, m.pcn_fin, nf, FR(m.act.x, D), FR(m.h1, HN), FR(m.post, hc),
                        FR(m.comb, hc * hc), FR(m.h0, HN), FR(m.pre_f, hc), FR(m.xcol, D),
                        RAD_W(m.final_gain), FR(m.final_hidden, D));
        else
            RAD_ISSUE_N(x, L.post_f, nf, FR(m.act.x, D), FR(m.h1, HN), FR(m.post, hc),
                        FR(m.comb, hc * hc), FR(m.h0, HN));
    }
    if (m.pfs.reads != pf_reads0) ++m.pfs.steps;
    if (cut_full) return;   /* a cut pass samples nothing: no final norm, no head */
    /* Under the per-row cut the final norm runs on the last layer's rows, and nothing when it has none
     * (the step samples nothing then: checked above). */
    const int64_t fr = row0 ? row0[c.n_layer - 1] : 0, fn = T - fr;
    if (row0 && fr >= T) return;
    const int64_t HN = c.hc * D;
    if (m.pcn_fin) {
    } else if (m.cn_fin) {
        RAD_ISSUE_N(x, m.cn_fin, fn, brow_slice(m.h0, fr, fn, HN), brow_slice(m.pre_f, fr, fn, c.hc),
                    brow_slice(m.xcol, fr, fn, D), RAD_W(m.final_gain), brow_slice(m.final_hidden, fr, fn, D));
    } else {
        RAD_ISSUE_N(x, m.contract, fn, brow_slice(m.h0, fr, fn, HN), brow_slice(m.pre_f, fr, fn, c.hc),
                    brow_slice(m.xcol, fr, fn, D));
        RAD_ISSUE_N(x, m.fnorm, fn, brow_slice(m.xcol, fr, fn, D), RAD_W(m.final_gain),
                    brow_slice(m.final_hidden, fr, fn, D));
    }
    if (batch->n_out > 0) {
        RAD_ISSUE_N(x, m.gather, batch->n_out, brows(m.final_hidden, T),
                    praw(batch->out_ids, RAD_I32, batch->n_out), brows(m.logit_input, batch->n_out));
        RAD_ISSUE_N(x, m.head_op, batch->n_out, brows(m.logit_input, batch->n_out), RAD_W(m.head),
                    brows(m.logits, batch->n_out));
    }
}

static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    /* A DSpark drafter proposes the block it was trained at, every row predicting (block = depth). */
    const int64_t ns = gi(meta, "nextn_predict_layers", 0), blk = gi(meta, "dspark.block_size", 0);
    if (ns > 0 && blk > 0) {
        out->draft_depth = (int)blk;
        out->draft_depth_fixed = 1;
    }
    return RAD_OK;
}

/* RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ATTN's shutdown report, a line a rank (Model::PfStats),
 * from rad_plugin_close: the loader calls it once at shutdown, after the engine's own report and
 * once the rank threads are gone. */
static void pf_report() {
    constexpr double MiB = 1048576.0;
    for (int r = 0; r < MAX_RANKS; ++r) {
        const Model* m = models[r].get();
        if (!m || !m->pfc.on) continue;
        const Model::PfStats& s = m->pfs;
        const uint64_t tails = s.points - s.epoints, treads = s.reads - s.ereads;
        std::fprintf(stderr, "radiance: deepseek41: rank %d: on-die reads issued live in %llu decode "
                             "steps: %llu reads at %llu layer tails, %.2f GiB (%.1f MiB a step, %.1f "
                             "a tail); behind the Engram GEMMs %llu reads at %llu layers, %.2f GiB\n",
                     r, (unsigned long long)s.steps, (unsigned long long)treads,
                     (unsigned long long)tails, (double)(s.bytes - s.ebytes) / (1024.0 * MiB),
                     s.steps ? (double)s.bytes / MiB / (double)s.steps : 0.0,
                     tails ? (double)(s.bytes - s.ebytes) / MiB / (double)tails : 0.0,
                     (unsigned long long)s.ereads, (unsigned long long)s.epoints,
                     (double)s.ebytes / (1024.0 * MiB));
    }
}

/* The consumer skip's shutdown line, a line a rank (Model::CutStats): what its cut passes issued,
 * which the engine's own "consumer skip:" line counts from the batches; the two agree. */
static void cut_report() {
    for (int r = 0; r < MAX_RANKS; ++r) {
        const Model* m = models[r].get();
        if (!m || !m->cut_on) continue;
        const Model::CutStats& s = m->cuts;
        std::fprintf(stderr, "radiance: deepseek41: rank %d: consumer skip: %llu cut passes issued "
                             "%llu layer-passes in full, %llu key ops alone and skipped %llu\n",
                     r, (unsigned long long)s.passes, (unsigned long long)s.full,
                     (unsigned long long)s.key, (unsigned long long)s.skipped);
        if (s.row_passes)
            std::fprintf(stderr, "radiance: deepseek41: rank %d: consumer skip, per row: %llu steps cut by "
                                 "row issued %llu layer-rows in full, %llu key ops alone and skipped %llu\n",
                         r, (unsigned long long)s.row_passes, (unsigned long long)s.rows_full,
                         (unsigned long long)s.rows_key, (unsigned long long)s.rows_skipped);
    }
}
}  // namespace deepseek41

#ifndef RAD_ARCH_NO_EXPORTS
RAD_ARCH_PROBE(deepseek41)
RAD_ARCH_PLUGIN(deepseek41, "deepseek41", "", "0.1.0",
                "DeepSeek-V4.1: hyper-connection streams, window + sparse compressed latent "
                "attention, sqrt(softplus) MoE, Engram")
/* Called once at shutdown (abi/rad_abi.h): the on-die reads' and the consumer skip's counters. */
extern "C" void rad_plugin_close(void) {
    deepseek41::pf_report();
    deepseek41::cut_report();
}
#endif
