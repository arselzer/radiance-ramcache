/* DeepSeek V4.1 configuration and cross-layer ownership, shared by the forthcoming architecture
 * and its CPU reference tests. This is not a serving plugin. Sources:
 * official 1017 inference/model.py (1bae0e1088993b9b82de08f59de5be5a34e42170),
 * DeepInfer hugging_face/config.py and v4_modules/cross_layer.py (9bd9cf243fc5),
 * and Project Maya ds4_model.cu splitb (f26d8bab43f9).
 *
 * Accept both GGUF and flattened HF metadata, including Radiance's normalized HF id. Every
 * present spelling is checked: a conflicting alias is an error, not a first-key-wins choice.
 * Only the backbone runs; draft schedule entries remain recorded and never share its caches. */
#pragma once
#include "rad_builder.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <initializer_list>
#include <iterator>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace rad::deepseek41 {

struct LayerSources {
    int compress_ratio = 0;
    int kv = -1;               // compressed KV; -1 for window-only layers
    int index_k = -1;          // index-key compressor, usually the KV source
    int index = -1;            // top-k selection producer, may be more frequent than index_k
    int candidate = -1;        // includes the producing layer itself
    bool writes_kv(int layer) const { return kv == layer; }
    bool writes_index_k(int layer) const { return index_k == layer; }
    bool writes_index(int layer) const { return index == layer; }
    bool writes_candidates(int layer) const { return candidate == layer; }
    bool uses_candidates(int layer) const { return candidate >= 0 && candidate < layer; }
};

struct Config {
    int64_t n_layers = 0, n_embd = 0, n_head = 0, n_head_kv = 0;
    int64_t head_dim = 0, rope_head_dim = 0, q_lora_rank = 0, o_lora_rank = 0, o_groups = 0;
    int64_t n_ff = 0, n_vocab = 0, context_length = 0;
    int64_t n_expert = 0, n_expert_used = 0, n_expert_shared = 0;
    int64_t window_size = 0, index_n_heads = 0, index_head_dim = 0, index_topk = 0;
    double norm_eps = 0, route_scale = 0, gate_temp = 1;
    bool norm_topk_prob = true;
    std::string score_func;
    std::vector<double> routed_clamp, shared_clamp;
    std::string rope_type;
    double rope_theta = 0, compress_rope_theta = 0, rope_factor = 0;
    int64_t original_seq_len = 0;
    double beta_fast = 0, beta_slow = 0;
    int hc_mult = 0, hc_sinkhorn_iters = 0;
    double hc_eps = 0;
    bool hc_shifted = true;     // official 1017 default; unshifted checkpoints are refused
    int candidate_source = -1;
    int64_t candidate_topk_blocks = 0, candidate_block_size = 0;
    std::vector<int64_t> compress_ratios, kv_source_layers, index_k_source_layers, index_source_layers;
    std::vector<LayerSources> layers;
    int64_t draft_layers = 0;
    std::vector<int64_t> dspark_target_layers;
    bool engram_present = false;
    std::vector<int64_t> engram_layers, engram_rows;
    int64_t engram_max_ngram_size = 0, engram_vocab_size = 0;
    int64_t engram_n_heads = 0, engram_head_dim = 0, engram_compressed_vocab_size = 0, engram_pad_id = 0;
    std::string engram_source;
};

namespace config_detail {
inline int fail(std::string* why, const std::string& message, int code = RAD_E_INVAL) {
    if (why) *why = message;
    return code;
}
inline bool near(double a, double b) {
    return std::isfinite(a) && std::isfinite(b) &&
           (a == b || std::fabs(a - b) <= 1e-6 * std::max(std::fabs(a), std::fabs(b)));
}
inline bool integer(std::string_view text, int64_t* value) {
    if (text.empty()) return false;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), *value);
    return result.ec == std::errc() && result.ptr == text.data() + text.size();
}
inline bool real(std::string_view text, double* value) {
    if (text.empty()) return false;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), *value);
    return result.ec == std::errc() && result.ptr == text.data() + text.size() && std::isfinite(*value);
}
inline bool boolean(std::string_view text, bool* value) {
    if (text == "true" || text == "1") { *value = true; return true; }
    if (text == "false" || text == "0") { *value = false; return true; }
    return false;
}
inline bool space(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; }
/* GGUF arrays are comma separated; flattened HF arrays are space separated. Accept either
 * convention but refuse empty comma fields, trailing delimiters and partially parsed numbers. */
template<class T, class Parse>
inline bool list(std::string_view text, std::vector<T>* out, Parse parse) {
    out->clear();
    size_t first = 0, end = text.size();
    while (first < end && space(text[first])) ++first;
    while (end > first && space(text[end - 1])) --end;
    if (first < end && text[first] == '[') {
        if (text[end - 1] != ']') return false;
        ++first; --end;
    }
    bool after_comma = false;
    while (first < end) {
        while (first < end && space(text[first])) ++first;
        if (first == end) return !after_comma;
        const size_t start = first;
        while (first < end && !space(text[first]) && text[first] != ',') ++first;
        if (start == first) return false;
        T value{};
        if (!parse(text.substr(start, first - start), &value)) return false;
        out->push_back(value);
        if (out->size() > 65536) return false;
        after_comma = false;
        while (first < end && space(text[first])) ++first;
        if (first < end && text[first] == ',') { ++first; after_comma = true; }
    }
    return !after_comma;
}

struct Reader {
    std::map<std::string, const char*> values;
    std::string error;
    explicit Reader(const RadModelMeta& meta) {
        if (meta.n_kv < 0 || meta.n_kv > 1000000 ||
            (meta.n_kv && (!meta.kv_key || !meta.kv_val))) {
            error = "invalid DeepSeek metadata table"; return;
        }
        for (int i = 0; i < meta.n_kv; ++i) {
            if (!meta.kv_key[i] || !meta.kv_val[i]) { error = "null DeepSeek metadata entry"; return; }
            const auto inserted = values.emplace(meta.kv_key[i], meta.kv_val[i]);
            if (!inserted.second && std::string_view(inserted.first->second) != meta.kv_val[i]) {
                error = "conflicting metadata entry: " + std::string(meta.kv_key[i]); return;
            }
        }
    }
    static std::vector<std::string> keys(const char* gguf, const char* hf = nullptr) {
        std::vector<std::string> out;
        if (gguf && *gguf) out.push_back("deepseek41." + std::string(gguf));
        if (hf && *hf) { out.push_back("text_config." + std::string(hf)); out.emplace_back(hf); }
        return out;
    }
    const char* get(const std::string& key) const {
        const auto it = values.find(key); return it == values.end() ? nullptr : it->second;
    }
    template<class T, class Parse, class Equal>
    bool read(const std::vector<std::string>& aliases, T* out, Parse parse, Equal equal,
              std::optional<T> typed = {}, bool required = true) {
        if (!error.empty()) return false;
        bool have = typed.has_value();
        T value = typed.value_or(T{});
        for (const auto& key : aliases) {
            const char* raw = get(key);
            if (!raw) continue;
            T parsed{};
            if (!parse(std::string_view(raw), &parsed)) { error = "invalid metadata value: " + key; return false; }
            if (have && !equal(value, parsed)) { error = "conflicting metadata aliases: " + key; return false; }
            value = std::move(parsed); have = true;
        }
        if (!have && required) { error = "missing metadata: " + aliases.front(); return false; }
        if (have) *out = std::move(value);
        return true;
    }
    bool i(const char* gguf, const char* hf, int64_t* out, int64_t typed = 0, bool required = true) {
        return read<int64_t>(keys(gguf,hf),out,integer,std::equal_to<int64_t>(),
                             typed ? std::optional<int64_t>(typed) : std::nullopt,required);
    }
    bool f(const char* gguf, const char* hf, double* out, double typed = 0, bool required = true) {
        return read<double>(keys(gguf,hf),out,real,near,
                            typed ? std::optional<double>(typed) : std::nullopt,required);
    }
    bool b(const char* gguf, const char* hf, bool* out, bool required = true) {
        return read<bool>(keys(gguf,hf),out,boolean,std::equal_to<bool>(),{},required);
    }
    bool s(const char* gguf, const char* hf, std::string* out, bool required = true) {
        return read<std::string>(keys(gguf,hf),out,
                  [](std::string_view s,std::string* out) { *out = std::string(s); return true; },
                  std::equal_to<std::string>(),{},required);
    }
    bool ints(const char* gguf, const char* hf, std::vector<int64_t>* out, bool required = true) {
        return read<std::vector<int64_t>>(keys(gguf,hf),out,
             [](std::string_view s,std::vector<int64_t>* v) { return list(s,v,integer); },
             std::equal_to<std::vector<int64_t>>(),{},required);
    }
    bool reals(const char* gguf, const char* hf, std::vector<double>* out, bool required = true) {
        return read<std::vector<double>>(keys(gguf,hf),out,
             [](std::string_view s,std::vector<double>* v) { return list(s,v,real); },
             [](const auto& a,const auto& b) {
                 return a.size()==b.size() && std::equal(a.begin(),a.end(),b.begin(),near);
             },{},required);
    }
};

inline bool product_fits(std::initializer_list<int64_t> factors) {
    int64_t product=1;
    for (int64_t factor : factors) {
        if (factor<=0 || product>INT64_MAX/factor) return false;
        product*=factor;
    }
    return true;
}
inline bool sorted_ids(const std::vector<int64_t>& ids, int64_t layers) {
    return std::all_of(ids.begin(),ids.end(),[&](int64_t v){ return v >= 0 && v < layers; }) &&
           std::adjacent_find(ids.begin(),ids.end(),std::greater_equal<int64_t>()) == ids.end();
}
inline int nearest(const std::vector<int64_t>& ids, int64_t layer) {
    const auto it=std::upper_bound(ids.begin(),ids.end(),layer);
    return it==ids.begin() ? -1 : int(*std::prev(it));
}
} // namespace config_detail

/* Rebuild derived ownership transactionally. Invalid metadata never leaves partial source maps.
 * Small synthetic dimensions are allowed, but the supported model semantics are strict. */
inline int validate_config(Config* cfg, std::string* why = nullptr) {
    using namespace config_detail;
    if (!cfg) return fail(why,"null DeepSeek config");
    Config c=*cfg;
    if (c.n_layers <= 0 || c.n_layers > 65536 || c.n_embd <= 0 || c.n_head <= 0 || c.head_dim <= 0 ||
        c.n_ff <= 0 || c.n_vocab <= 0 || c.context_length <= 0 || c.q_lora_rank <= 0 || c.o_lora_rank <= 0)
        return fail(why,"nonpositive or unsupported DeepSeek geometry");
    if (c.n_embd > INT64_MAX/4 || c.n_head > INT64_MAX/c.head_dim || c.o_groups <= 0 ||
        c.n_head%c.o_groups || c.o_lora_rank > INT64_MAX/c.o_groups)
        return fail(why,"invalid grouped output geometry");
    if (c.n_head_kv != 1 || c.rope_head_dim <= 0 || c.rope_head_dim > c.head_dim || c.rope_head_dim%2)
        return fail(why,"expected one KV head and an even, in-range rotary dimension",RAD_E_UNSUPPORTED);
    if (c.n_expert <= 0 || c.n_expert_used <= 0 || c.n_expert_used > c.n_expert || c.n_expert_shared != 1)
        return fail(why,"invalid routing counts or unsupported shared-expert count");
    if (c.score_func != "sqrtsoftplus" || !c.norm_topk_prob || !std::isfinite(c.route_scale) || c.route_scale<=0 ||
        !std::isfinite(c.gate_temp) || c.gate_temp<=0)
        return fail(why,"expected normalized sqrtsoftplus routing with positive scale/temperature",RAD_E_UNSUPPORTED);
    if (!std::isfinite(c.norm_eps) || c.norm_eps<=0 || c.window_size<=0 || c.index_n_heads<=0 ||
        c.index_head_dim<c.rope_head_dim || c.index_topk<=0)
        return fail(why,"invalid normalization/window/index geometry");
    if (!product_fits({c.n_vocab,c.n_embd}) || !product_fits({c.n_expert,c.n_ff,c.n_embd}) ||
        !product_fits({c.q_lora_rank,c.n_embd}) || !product_fits({c.n_head,c.head_dim,c.q_lora_rank}) ||
        !product_fits({c.o_groups,c.o_lora_rank,c.n_embd}) || !product_fits({96,c.n_embd}) ||
        !product_fits({c.index_n_heads,c.index_head_dim,c.q_lora_rank}) ||
        !product_fits({c.context_length,c.head_dim}))
        return fail(why,"DeepSeek matrix or cache dimensions overflow int64");
    if (c.hc_mult != 4 || !c.hc_shifted || c.hc_sinkhorn_iters<=0 || c.hc_sinkhorn_iters>256 ||
        !std::isfinite(c.hc_eps) || c.hc_eps<=0)
        return fail(why,"only shifted four-stream mHC is supported",RAD_E_UNSUPPORTED);
    if (c.rope_type!="yarn" || !std::isfinite(c.rope_theta) || c.rope_theta<=1 ||
        !std::isfinite(c.compress_rope_theta) || c.compress_rope_theta<=1 ||
        !std::isfinite(c.rope_factor) || c.rope_factor<=0 || c.original_seq_len<=0 ||
        c.original_seq_len>c.context_length || !std::isfinite(c.beta_fast) || !std::isfinite(c.beta_slow) ||
        c.beta_slow<=0 || c.beta_fast<=c.beta_slow)
        return fail(why,"invalid or unsupported DeepSeek YaRN settings",RAD_E_UNSUPPORTED);
    for (const auto* limits : {&c.routed_clamp,&c.shared_clamp}) {
        if ((int64_t)limits->size()!=c.n_layers ||
            !std::all_of(limits->begin(),limits->end(),[](double v){return std::isfinite(v)&&v>=0;}))
            return fail(why,"SwiGLU clamp schedules must cover every backbone layer");
    }
    if (c.draft_layers < 0 || c.draft_layers > 65536-c.n_layers ||
        (int64_t)c.compress_ratios.size()!=c.n_layers+c.draft_layers)
        return fail(why,"compression schedule length does not match backbone plus draft layers");
    for (int64_t layer=0;layer<(int64_t)c.compress_ratios.size();++layer) {
        const int64_t ratio=c.compress_ratios[layer];
        if (ratio<0 || ratio>2 || (layer>=c.n_layers && ratio!=0))
            return fail(why,"only backbone compression ratios 0/1/2 and unshared window-only drafts are supported",RAD_E_UNSUPPORTED);
    }
    if (!c.dspark_target_layers.empty() && (!sorted_ids(c.dspark_target_layers,c.n_layers) ||
        (int64_t)c.dspark_target_layers.size()!=c.draft_layers))
        return fail(why,"invalid DSpark target layers");
    for (const auto* sources : {&c.kv_source_layers,&c.index_k_source_layers,&c.index_source_layers}) {
        if (!sorted_ids(*sources,c.n_layers)) return fail(why,"source layers must be sorted, unique backbone ids");
        for (int64_t source : *sources)
            if (!c.compress_ratios[source]) return fail(why,"a window-only layer cannot produce compressed shared state");
    }
    c.layers.assign((size_t)c.n_layers,{});
    int last_window = -1;
    for (int layer=0;layer<c.n_layers;++layer) {
        auto& dst=c.layers[layer];dst.compress_ratio=(int)c.compress_ratios[layer];
        if (!dst.compress_ratio) { last_window=layer; continue; }
        dst.kv=nearest(c.kv_source_layers,layer); dst.index_k=nearest(c.index_k_source_layers,layer);
        dst.index=nearest(c.index_source_layers,layer);
        for (int source : {dst.kv,dst.index_k,dst.index}) {
            if (source<0) return fail(why,"compressed layer has no preceding source");
            if (c.compress_ratios[source]!=dst.compress_ratio)
                return fail(why,"shared source and consumer compression ratios disagree");
            if (source <= last_window) return fail(why,"shared state crosses a window-only layer");
        }
        if (nearest(c.kv_source_layers,dst.index_k)!=dst.kv ||
            nearest(c.index_k_source_layers,dst.index)!=dst.index_k)
            return fail(why,"index-topk/index-K/KV source segments do not refine each other");
    }
    if (c.candidate_source>=0) {
        if (c.candidate_source>=c.n_layers || c.candidate_topk_blocks<=0 ||
            c.candidate_block_size<=0 ||
            c.candidate_topk_blocks > INT64_MAX/c.candidate_block_size ||
            c.candidate_topk_blocks*c.candidate_block_size<c.index_topk ||
            c.layers[c.candidate_source].index!=c.candidate_source)
            return fail(why,"invalid candidate producer, block geometry or capacity");
        for (int layer=c.candidate_source;layer<c.n_layers;++layer) {
            if (c.layers[layer].compress_ratio!=c.layers[c.candidate_source].compress_ratio)
                return fail(why,"candidate consumers must share the source compression ratio");
            c.layers[layer].candidate=c.candidate_source;
        }
    } else if (c.candidate_source != -1 || c.candidate_topk_blocks || c.candidate_block_size) {
        return fail(why,"disabled candidate filtering must have no candidate buffers");
    }
    if (c.engram_present) {
        if (c.engram_layers.empty() || c.engram_layers.size()!=c.engram_rows.size() ||
            !sorted_ids(c.engram_layers,c.n_layers) || c.engram_max_ngram_size<2 ||
            c.engram_max_ngram_size>64 || c.engram_n_heads<=0 || c.engram_n_heads>65536 ||
            c.engram_head_dim<=0 || c.engram_vocab_size<=0 || c.engram_compressed_vocab_size<=0 ||
            c.engram_compressed_vocab_size>c.n_vocab || c.engram_pad_id<0 || c.engram_pad_id>=c.n_vocab)
            return fail(why,"invalid Engram layer/hash/embedding geometry");
        const int64_t heads=c.engram_n_heads*(c.engram_max_ngram_size-1);
        if (c.engram_head_dim>INT64_MAX/heads ||
            !std::all_of(c.engram_rows.begin(),c.engram_rows.end(),[&](int64_t n){return n>=heads;}))
            return fail(why,"invalid Engram table extent");
    } else if (!c.engram_layers.empty() || !c.engram_rows.empty()) {
        return fail(why,"Engram table metadata is present while Engram is disabled");
    }
    *cfg=std::move(c);
    if (why) why->clear();
    return RAD_OK;
}

/* Explicitly normalize GGUF deepseek41 and HF deepseek_v41 (core id deepseekv41). HF's optional
 * gate_temp=1 and hc_shifted=true are the official 1017 architecture defaults; no geometry,
 * source schedule, routing function or Engram capacity is inferred from a model name. */
inline int normalize_config(const RadModelMeta& meta, Config* out, std::string* why = nullptr) {
    using namespace config_detail;
    if (!out) return fail(why,"null DeepSeek config destination");
    if (!meta.arch_id || (std::string_view(meta.arch_id)!="deepseek41" &&
        std::string_view(meta.arch_id)!="deepseek_v41" && std::string_view(meta.arch_id)!="deepseekv41"))
        return fail(why,"unsupported DeepSeek architecture id",RAD_E_UNSUPPORTED);
    Reader r(meta);Config c;
    auto arch_alias=[](std::string_view id) {return id=="deepseek41" || id=="deepseek_v41" || id=="deepseekv41";};
    for (const char* key : {"general.architecture","model_type"})
        if (const char* value=r.get(key);value && !arch_alias(value))
            return fail(why,std::string("conflicting architecture metadata: ")+key,RAD_E_FORMAT);
    if (const char* value=r.get("text_config.model_type");value &&
        std::string_view(value)!="deepseek_v41_text" && !arch_alias(value))
        return fail(why,"unsupported text model type",RAD_E_UNSUPPORTED);
    for (const char* key : {"hidden_act","text_config.hidden_act"})
        if (const char* value=r.get(key);value && std::string_view(value)!="silu")
            return fail(why,"only SwiGLU/silu activation is supported",RAD_E_UNSUPPORTED);
    for (const char* key : {"attention_bias","text_config.attention_bias"})
        if (const char* value=r.get(key);value) {
            bool enabled=false;
            if (!boolean(value,&enabled) || enabled)return fail(why,"attention biases are unsupported",RAD_E_UNSUPPORTED);
        }
    for (const char* key : {"attention_dropout","text_config.attention_dropout"})
        if (const char* value=r.get(key);value) {
            double dropout=0;
            if (!real(value,&dropout) || dropout!=0)return fail(why,"attention dropout must be zero",RAD_E_UNSUPPORTED);
        }
    r.i("block_count","num_hidden_layers",&c.n_layers,meta.n_layers);
    r.i("embedding_length","hidden_size",&c.n_embd,meta.n_embd);
    r.i("attention.head_count","num_attention_heads",&c.n_head,meta.n_head);
    r.i("attention.head_count_kv","num_key_value_heads",&c.n_head_kv,meta.n_head_kv);
    r.i("attention.key_length","head_dim",&c.head_dim,meta.head_dim);
    int64_t value_dim=c.head_dim;
    r.i("attention.value_length","v_head_dim",&value_dim,0,false);
    if (value_dim!=c.head_dim)r.error="latent KV key/value dimensions must match";
    r.i("rope.dimension_count","qk_rope_head_dim",&c.rope_head_dim);
    r.i("attention.q_lora_rank","q_lora_rank",&c.q_lora_rank);
    r.i("attention.output_lora_rank","o_lora_rank",&c.o_lora_rank);
    r.i("attention.output_group_count","o_groups",&c.o_groups);
    r.i("expert_feed_forward_length","moe_intermediate_size",&c.n_ff,meta.n_ff);
    r.i("vocab_size","vocab_size",&c.n_vocab,meta.n_vocab);
    r.i("context_length","max_position_embeddings",&c.context_length,meta.n_ctx_train);
    r.i("expert_count","n_routed_experts",&c.n_expert,meta.n_expert);
    r.i("expert_used_count","num_experts_per_tok",&c.n_expert_used,meta.n_expert_used);
    r.i("expert_shared_count","n_shared_experts",&c.n_expert_shared,meta.n_expert_shared);
    r.i("attention.sliding_window","sliding_window",&c.window_size);
    r.i("attention.indexer.head_count","index_n_heads",&c.index_n_heads);
    r.i("attention.indexer.key_length","index_head_dim",&c.index_head_dim);
    r.i("attention.indexer.top_k","index_topk",&c.index_topk);
    r.f("attention.layer_norm_rms_epsilon","rms_norm_eps",&c.norm_eps,meta.rms_eps);
    r.f("expert_weights_scale","routed_scaling_factor",&c.route_scale);
    r.f("expert_gate_temp","gate_temp",&c.gate_temp,0,false);
    r.b("expert_weights_norm","norm_topk_prob",&c.norm_topk_prob);
    r.s("expert_scoring_func","scoring_func",&c.score_func);
    if (const char* numeric=r.get("deepseek41.expert_gating_func")) {
        int64_t value=0;if (!integer(numeric,&value)||value!=4)r.error="unsupported GGUF expert gating function";
    }
    std::string method;
    r.s(nullptr,"topk_method",&method,false);
    if (!method.empty() && method!="noaux_tc")r.error="unsupported routing top-k method";
    r.reals("swiglu_clamp_exp",nullptr,&c.routed_clamp,false);
    r.reals("swiglu_clamp_shexp",nullptr,&c.shared_clamp,false);
    double clamp=0;
    if (r.get("swiglu_limit") || r.get("text_config.swiglu_limit")) {
        r.f(nullptr,"swiglu_limit",&clamp);
        if (c.n_layers>0 && c.n_layers<=65536) {
            for (auto* limits : {&c.routed_clamp,&c.shared_clamp}) {
                if (limits->empty())limits->assign((size_t)c.n_layers,clamp);
                else if (!std::all_of(limits->begin(),limits->end(),[&](double v){return near(v,clamp);}))
                    r.error="conflicting scalar and per-layer SwiGLU clamps";
            }
        }
    }
    /* HF and the core's translated GGUF YaRN spellings. */
    auto rope_keys=Reader::keys("rope.scaling.type", "rope_scaling.rope_type");
    rope_keys.push_back("rope_scaling.type");rope_keys.push_back("text_config.rope_scaling.type");
    r.read<std::string>(rope_keys,&c.rope_type,[](std::string_view s,std::string* o){*o=s;return true;},std::equal_to<std::string>());
    r.f("rope.freq_base","rope_theta",&c.rope_theta,meta.rope_theta);
    r.f("attention.compress_rope_freq_base","compress_rope_theta",&c.compress_rope_theta);
    r.f("rope.scaling.factor","rope_scaling.factor",&c.rope_factor);
    r.i("rope.scaling.original_context_length","rope_scaling.original_max_position_embeddings",&c.original_seq_len);
    r.f("rope.scaling.yarn_beta_fast","rope_scaling.beta_fast",&c.beta_fast);
    r.f("rope.scaling.yarn_beta_slow","rope_scaling.beta_slow",&c.beta_slow);
    int64_t hc_mult=0,hc_iters=0;
    r.i("hyper_connection.count","hc_mult",&hc_mult);
    r.i("hyper_connection.sinkhorn_iterations","hc_sinkhorn_iters",&hc_iters);
    if (hc_mult>INT32_MAX || hc_mult<0 || hc_iters>INT32_MAX || hc_iters<0)r.error="mHC count out of range";
    else {c.hc_mult=(int)hc_mult;c.hc_sinkhorn_iters=(int)hc_iters;}
    r.f("hyper_connection.epsilon","hc_eps",&c.hc_eps);
    r.b("hyper_connection.shifted","hc_shifted",&c.hc_shifted,false);
    r.ints("attention.compress_ratios","compress_ratios",&c.compress_ratios);
    r.ints("kv_source_layer_ids","kv_source_layer_ids",&c.kv_source_layers);
    r.ints("index_source_layer_ids","index_source_layer_ids",&c.index_source_layers);
    c.index_k_source_layers=c.kv_source_layers;
    r.ints("index_k_source_layer_ids","index_k_source_layer_ids",&c.index_k_source_layers,false);
    int64_t candidate=-1;
    r.i("attention.candidate_source_layer_id","candidate_source_layer_id",&candidate,0,false);
    if (candidate< -1 || candidate>INT32_MAX)r.error="candidate source id out of range";
    else c.candidate_source=(int)candidate;
    r.i("attention.candidate_topk_blocks","candidate_topk_blocks",&c.candidate_topk_blocks,0,false);
    r.i("attention.candidate_block_size","candidate_block_size",&c.candidate_block_size,0,false);
    r.i("draft_layer_count","num_nextn_predict_layers",&c.draft_layers,0,false);
    r.ints("dspark_target_layer_ids","dspark_target_layer_ids",&c.dspark_target_layers,false);
    if (!r.get("deepseek41.draft_layer_count") && !r.get("num_nextn_predict_layers") && !r.get("text_config.num_nextn_predict_layers") &&
        c.n_layers>0 && c.compress_ratios.size()>size_t(c.n_layers))c.draft_layers=(int64_t)c.compress_ratios.size()-c.n_layers;
    r.ints("engram.layer_ids","engram_layer_ids",&c.engram_layers,false);
    r.ints("engram.num_embeddings","engram_num_embeddings",&c.engram_rows,false);
    c.engram_present=!c.engram_layers.empty();
    r.b("engram.present",nullptr,&c.engram_present,false);
    if (c.engram_present) {
        r.i("engram_max_ngram_size","engram_max_ngram_size",&c.engram_max_ngram_size);
        r.i("engram_vocab_size","engram_vocab_size",&c.engram_vocab_size);
        r.i("engram_n_heads","engram_n_heads",&c.engram_n_heads);
        r.i("engram_head_dim","engram_head_dim",&c.engram_head_dim);
        r.i("engram_compressed_vocab_size","engram_compressed_vocab_size",&c.engram_compressed_vocab_size);
        r.i("engram_pad_token_id","engram_pad_token_id",&c.engram_pad_id);
        r.s("engram.source",nullptr,&c.engram_source,false);
    }
    if (!r.error.empty())return fail(why,r.error,RAD_E_FORMAT);
    const int status=validate_config(&c,why);
    if (status!=RAD_OK)return status;
    *out=std::move(c);return RAD_OK;
}

struct Partition { int64_t offset=0, width=0; };
/* Group-aligned column ownership. All ranks use the same range for gate/up rows and down
 * columns. Rotating the extra groups across layers balances aggregate packed expert bytes. */
inline int ffn_partition(int64_t total_ffn, int world, int rank, int64_t layer, int64_t block,
                         Partition* out, std::string* why = nullptr) {
    using namespace config_detail;
    if (!out || total_ffn<=0 || block<=0 || total_ffn%block || world<=0 || world>65536 ||
        rank<0 || rank>=world || layer<0 || total_ffn/block<world)
        return fail(why,"invalid group-aligned FFN partition");
    const int64_t groups=total_ffn/block,q=groups/world,remainder=groups%world;
    const int rotation=(int)(layer%world);
    auto count=[&](int member) {return q+(((member-rotation+world)%world)<remainder ? 1 : 0);};
    int64_t offset=0;for(int member=0;member<rank;++member)offset+=count(member);
    *out={offset*block,count(rank)*block};if(why)why->clear();return RAD_OK;
}
inline int validate_tp_geometry(const Config& c, int world, std::string* why = nullptr) {
    using namespace config_detail;
    if (world<=0 || c.n_head%world || c.o_groups%world || c.index_n_heads%world || c.n_vocab%world)
        return fail(why,"TP ranks must divide attention heads, output groups, index heads and vocabulary");
    Partition p;
    return ffn_partition(c.n_ff,world,0,0,256,&p,why);
}

} // namespace rad::deepseek41
