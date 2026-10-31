/* DS41's independent, default-off Engram gather row slices: bounds, ordering and full pitches.
 * No model file or card: the recording architecture ABI only. */
#define RAD_ARCH_NO_EXPORTS 1
#include "arch_fakes.h"
#include "../arch/deepseek41/deepseek41.cpp"
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace {
struct Env {
    std::map<std::string, std::string> kept;
    std::vector<std::string> absent;
    explicit Env(const char* value, bool split = true, bool tbo = false, bool prefetch = false) {
        const std::pair<const char*, const char*> vars[] = {
            { deepseek41::kAgRowsEnv, value }, { deepseek41::kArRowsEnv, "0" },
            { "RADIANCE_EXPERIMENTAL_DS41_ENGRAM_SPLIT", split ? "1" : "0" },
            { "RADIANCE_EXPERIMENTAL_SCHED_TBO", tbo ? "1" : "0" },
            { "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ATTN", prefetch ? "1" : "0" },
            { "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ENGRAM", prefetch ? "1" : "0" },
            { "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_MIB", "56" },
        };
        for (const auto& v : vars) {
            if (const char* old = std::getenv(v.first)) kept[v.first] = old;
            else absent.push_back(v.first);
            if (v.second) setenv(v.first, v.second, 1); else unsetenv(v.first);
        }
    }
    ~Env() {
        for (const auto& v : kept) setenv(v.first.c_str(), v.second.c_str(), 1);
        for (const auto& v : absent) unsetenv(v.c_str());
    }
};
MetaKV dspark_meta() {
    RadModelMeta base{};
    base.arch_id = "deepseek41";
    base.name = "ds41-flash-dspark-geometry";
    base.n_layers = 40;
    base.n_embd = 5120;
    base.n_head = 64;
    base.n_head_kv = 1;
    base.head_dim = 512;
    base.n_vocab = 129280;
    base.n_expert = 384;
    base.n_expert_used = 6;
    base.n_expert_shared = 1;
    base.rms_eps = 1e-6f;
    base.rope_theta = 10000.f;
    base.rope_scale = 1.f;
    MetaKV k(base);
    std::string ratios;
    for (int l = 0; l < 40; ++l) ratios += (l ? "," : "") + std::to_string(l < 2 ? 0 : l < 20 ? 2 : 1);
    k.set("deepseek41.block_count", "40")
        .set("deepseek41.rope.dimension_count", "64")
        .set("deepseek41.attention.q_lora_rank", "1280")
        .set("deepseek41.attention.output_group_count", "8")
        .set("deepseek41.attention.output_lora_rank", "1024")
        .set("deepseek41.attention.sliding_window", "128")
        .set("deepseek41.expert_feed_forward_length", "2304")
        .set("deepseek41.expert_weights_scale", "1.5")
        .set("deepseek41.expert_weights_norm", "1")
        .set("deepseek41.swiglu_clamp_exp", "10")
        .set("deepseek41.attention.indexer.head_count", "32")
        .set("deepseek41.attention.indexer.key_length", "128")
        .set("deepseek41.attention.indexer.top_k", "512")
        .set("deepseek41.attention.indexer.candidate_source_layer", "20")
        .set("deepseek41.attention.indexer.candidate_top_k_blocks", "2048")
        .set("deepseek41.attention.indexer.candidate_block_size", "8")
        .set("deepseek41.hyper_connection.count", "4")
        .set("deepseek41.hyper_connection.sinkhorn_iterations", "20")
        .set("deepseek41.hyper_connection.epsilon", "1e-6")
        .set("deepseek41.attention.compress_rope_freq_base", "160000")
        .set("deepseek41.attention.compress_ratios", ratios.c_str())
        .set("deepseek41.attention.kv_source_layers", "2,8,14,20")
        .set("deepseek41.attention.indexer.source_layers", "2,8,14,20,24,28,32,36")
        .set("deepseek41.ple.layers", "1,14")
        .set("deepseek41.ple.ngram_size", "4")
        .set("deepseek41.ple.heads_per_ngram", "8")
        .set("deepseek41.embedding_length_per_layer_input", "256")
        .set("deepseek41.engram.pad_token_id", "2")
        .set("deepseek41.ple.head_vocab_sizes", "1000,1000")
        .set("deepseek41.nextn_predict_layers", "3")
        .set("deepseek41.dspark.block_size", "5")
        .set("deepseek41.dspark.expert_count", "128")
        .set("deepseek41.dspark.expert_used_count", "3")
        .set("deepseek41.dspark.markov_rank", "256")
        .set("deepseek41.dspark.markov_top_k", "16")
        .set("deepseek41.dspark.noise_token_id", "128799")
        .set("deepseek41.dspark.target_layers", "37,38,39")
        .set("imported_quantization_config.quant_method", "fp8")
        .set("imported_quantization_config.scale_fmt", "ue8m0")
        .set("imported_quantization_config.weight_block_size", "32,32");
    return k;
}

struct Decl {
    RadBuilder b;
    int rc = RAD_E_INVAL;
    deepseek41::Model* m = nullptr;
};
std::unique_ptr<Decl> declare(int rank, int64_t C = 2048) {
    auto d = std::make_unique<Decl>();
    d->b.kv_block_of = [](const std::string& n) -> int64_t {
        for (const char* p : { "kv_compressed_r", "kv_index_r" })
            if (n.rfind(p, 0) == 0) return std::atoll(n.c_str() + std::strlen(p));
        return 0;
    };
    auto meta = dspark_meta();
    const RadEncoding plain = rad_enc_plain(RAD_BF16);
    for (const char* name : { "attn_q_a.weight", "attn_kv.weight", "attn_q_b.weight", "attn_output_a.",
                              "attn_output_b.weight", "engram_kv.weight", "ffn_gate_up_shexp.weight",
                              "ffn_down_shexp.weight", "dspark.fc.weight" })
        d->b.encs.push_back({ name, plain });
    RadBuildCtx ctx = build_ctx(rank, 4);
    ctx.max_tok = C; ctx.max_seqs = 8; ctx.max_spec = 5;
    stderr_of([&] { d->rc = deepseek41::declare(&d->b, meta.meta(), &ctx); });
    d->m = deepseek41::models[rank].get();
    return d;
}
std::vector<RecIssue> eng_ags(const Decl& d, const RadCtx& c) {
    std::vector<RecIssue> out;
    for (const auto& i : c.issues) {
        const auto* op = d.b.op_of(i.op);
        if (op->op == "all_gather" && pint(op, "row") == d.m->esplit && d.m->esplit > 0) out.push_back(i);
    }
    return out;
}
uint64_t other_issue_hash(const RadBuilder& b, const RadCtx& c, const Kit* k) {
    auto copy = std::make_unique<RadCtx>(c);
    auto& v = copy->issues;
    v.erase(std::remove_if(v.begin(), v.end(), [&](const RecIssue& i) {
        const auto* op = b.op_of(i.op);
        return op->op == "all_gather" && pstr(op, "dtype") == "bf16";
    }), v.end());
    copy->joins.clear(); // insertion indices change with extra gathers; ordered reads checked below
    std::vector<std::unique_ptr<RadCtx>> passes;
    passes.push_back(std::move(copy));
    return issue_hash(b, passes, { k });
}
} // namespace

TEST(ds41_ag_rows_defaults_validate_counts_and_require_the_existing_aligned_tp4_split) {
    deepseek41::Model m;
    m.g.world = 4; m.g.max_tok = 2048; m.esplit = 6400;
    for (const char* v : { (const char*)nullptr, "", "0" }) {
        REQUIRE_EQ(deepseek41::ag_rows_config(m, v), (int)RAD_OK);
        CHECK_EQ(m.ag_rows, int64_t(0));
    }
    REQUIRE_EQ(deepseek41::ag_rows_config(m, "1024"), (int)RAD_OK);
    CHECK_EQ(m.ag_rows, int64_t(1024));
    REQUIRE_EQ(deepseek41::ag_rows_config(m, "9999"), (int)RAD_OK);
    CHECK_EQ(m.ag_rows, int64_t(2048));
    for (const char* v : { "-1", "+1", "1x", " 1", "9223372036854775808" })
        CHECK_EQ(deepseek41::ag_rows_config(m, v), (int)RAD_E_INVAL);
    for (int64_t width : { 0, 6401 }) {
        m.esplit = width;
        REQUIRE_EQ(deepseek41::ag_rows_config(m, "1024"), (int)RAD_OK);
        CHECK_EQ(m.ag_rows, int64_t(0));
    }
    m.esplit = 6408;  // a different whole-word width is supported, no 6400/5120 hardcode
    REQUIRE_EQ(deepseek41::ag_rows_config(m, "1024"), (int)RAD_OK);
    CHECK_EQ(m.ag_rows, int64_t(1024));
    m.g.world = 3;
    REQUIRE_EQ(deepseek41::ag_rows_config(m, "1024"), (int)RAD_OK);
    CHECK_EQ(m.ag_rows, int64_t(0));
}

TEST(ds41_ag_rows_bounds_both_engram_layers_only_at_2k_and_4k_on_every_rank) {
    Env env("1024");
    for (int64_t C : { 2048, 4096 }) for (int rank = 0; rank < 4; ++rank) {
        auto d = declare(rank, C);
        REQUIRE_EQ(d->rc, (int)RAD_OK);
        REQUIRE(d->m && d->m->dsp.on);
        REQUIRE_EQ(d->m->esplit, int64_t(6400));
        CHECK_EQ(d->m->ar_rows, int64_t(0));
        int ag = 0, other_ag = 0, whole_ar = 0;
        for (const auto& o : d->b.ops) {
            if (o.op == "all_gather" && pint(&o, "row") == d->m->esplit) {
                CHECK_EQ(param(&o, "numel")->ihi, int64_t(1024 * 6400));
                CHECK_EQ(param(&o, "numel")->ival, int64_t(6400));
                ++ag;
            } else if (o.op == "all_gather") {
                CHECK_EQ(pstr(&o, "dtype"), std::string("f32"));
                CHECK_EQ(pint(&o, "numel"), d->m->dsp.Q * d->m->dsp.cand * 2);
                ++other_ag;
            } else if (o.op == "all_reduce" && param(&o, "numel")->ihi == C * 5120) ++whole_ar;
        }
        CHECK_EQ(ag, 2);
        CHECK_EQ(other_ag, 1);
        CHECK(whole_ar >= 81);
    }
    auto d = declare(0, 16);
    REQUIRE_EQ(d->rc, (int)RAD_OK);
    CHECK_EQ(d->m->ag_rows, int64_t(16));
}

TEST(ds41_ag_rows_zero_keeps_the_original_graph_and_issues_and_split_off_stays_off) {
    uint64_t graph = 0, issues = 0;
    for (const char* value : { (const char*)nullptr, "0" }) {
        Env env(value);
        auto d = declare(0);
        REQUIRE_EQ(d->rc, (int)RAD_OK);
        auto input = kit(d->b, {}, { 1025 });
        auto run = run_passes(d->b, deepseek41::step, { input.get() }, 0, 4);
        REQUIRE(steps_ok(run));
        if (!value) { graph = graph_hash(d->b); issues = issue_hash(d->b, run, { input.get() }); }
        else { CHECK_EQ(graph_hash(d->b), graph); CHECK_EQ(issue_hash(d->b, run, { input.get() }), issues); }
    }
    Env env("1024", false);
    auto d = declare(0);
    REQUIRE_EQ(d->rc, (int)RAD_OK);
    CHECK_EQ(d->m->esplit, int64_t(0));
    CHECK_EQ(d->m->ag_rows, int64_t(0));
    for (const auto& L : d->m->layers) CHECK_EQ(L.e_gather, rad_op(0));
}

TEST(ds41_ag_rows_covers_fence_edges_cap_edges_and_4k_without_moving_other_issues) {
    for (const char* cap : { "64", "1024" }) {
        Env env(cap);
        auto d = declare(0, 4096);
        REQUIRE_EQ(d->rc, (int)RAD_OK);
        for (int n : { 1, 20, 21, 63, 64, 65, 1023, 1024, 1025, 1044, 1045, 2048, 4096 }) {
            auto input = kit(d->b, {}, { n });
            auto run = run_passes(d->b, deepseek41::step, { input.get() }, 0, 4);
            REQUIRE(steps_ok(run));
            const auto ag = eng_ags(*d, *run[0]);
            size_t at = 0;
            for (int layer : { 1, 14 }) {
                const auto& L = d->m->layers[(size_t)layer];
                for (int64_t r = 0; r < n;) {
                    REQUIRE(at < ag.size());
                    const auto& a = ag[at++];
                    const int64_t nr = std::min<int64_t>(std::atoll(cap), n - r);
                    CHECK_EQ(a.op, L.e_gather);
                    CHECK_EQ(a.n, nr * 6400);
                    REQUIRE_EQ(a.opd.size(), size_t(2));
                    CHECK_EQ(a.opd[0].handle, d->m->ekv_part);
                    CHECK_EQ(a.opd[1].handle, d->m->ekv);
                    CHECK_EQ(a.opd[0].offset, r * 6400);
                    CHECK_EQ(a.opd[1].offset, r * 25600);
                    for (const auto& o : a.opd) { CHECK_EQ(o.rows, nr); CHECK_EQ(o.cols, int64_t(0)); }
                    r += nr;
                }
                // The one full GEMM precedes every slice; the gate still follows the last slice.
                size_t gemm = 0, first = run[0]->issues.size(), last = 0, gate = 0, gemms = 0;
                for (size_t i = 0; i < run[0]->issues.size(); ++i) {
                    const auto& a = run[0]->issues[i];
                    if (a.op == L.e_gemm) { gemm = i; ++gemms; CHECK_EQ(a.n, int64_t(n)); }
                    if (a.op == L.e_gather) { first = std::min(first, i); last = i; }
                    if (a.op == L.e_gate_op) { gate = i; CHECK_EQ(a.n, int64_t(n)); }
                }
                CHECK_EQ(gemms, size_t(1)); CHECK(gemm < first); CHECK(last < gate);
            }
            CHECK_EQ(at, ag.size());
        }
    }
}

TEST(ds41_ag_rows_rebases_actual_tbo_views_and_keeps_distinct_full_pitches) {
    Env env("2", true, true);
    auto d = declare(0);
    REQUIRE_EQ(d->rc, (int)RAD_OK);
    REQUIRE(d->m->tbo);
    auto input = kit(d->b, std::vector<int>(7, 1), {});
    REQUIRE(::rad::tbo::step_splits(&input->b));
    auto run = run_passes(d->b, deepseek41::step, { input.get() }, 0, 4);
    REQUIRE(steps_ok(run));
    const auto ag = eng_ags(*d, *run[0]);
    int seen[7] = {};
    for (const auto& a : ag) {
        REQUIRE_EQ(a.opd.size(), size_t(2));
        const int64_t first = a.opd[0].offset / 6400;
        CHECK_EQ(a.opd[0].offset % 6400, int64_t(0));
        CHECK_EQ(a.opd[1].offset, first * 25600);
        CHECK_EQ(a.opd[0].rows, a.opd[1].rows);
        CHECK_EQ(a.opd[0].cols, int64_t(0)); CHECK_EQ(a.opd[1].cols, int64_t(0));
        REQUIRE(first >= 0 && first + a.opd[0].rows <= 7);
        for (int64_t r = first; r < first + a.opd[0].rows; ++r) ++seen[r];
    }
    for (int count : seen) CHECK_EQ(count, 2); // exactly once per token at each Engram layer
    RadCtx x; x.b = &d->b;
    deepseek41::issue_ag_rows(&x, d->m->layers[1].e_gather, d->m->ekv_part, d->m->ekv,
                            13, 1025, 6400, 4, 1024);
    REQUIRE_EQ(x.issues.size(), size_t(2));
    CHECK_EQ(x.issues[1].opd[0].offset, int64_t(1037 * 6400));
    CHECK_EQ(x.issues[1].opd[1].offset, int64_t(1037 * 25600));
    CHECK_EQ(x.issues[1].n, int64_t(6400));
}

TEST(ds41_ag_rows_keeps_all_other_issue_arguments_and_engram_prefetch_order) {
    uint64_t reference = 0;
    std::unique_ptr<Kit> input;
    {
        Env env(nullptr, true, false, true);
        auto d = declare(0);
        REQUIRE_EQ(d->rc, (int)RAD_OK);
        input = kit(d->b, std::vector<int>(7, 1), {});
        auto run = run_passes(d->b, deepseek41::step, { input.get() }, 0, 4);
        REQUIRE(steps_ok(run));
        reference = other_issue_hash(d->b, *run[0], input.get());
    }
    Env env("4", true, false, true);
    auto d = declare(0);
    REQUIRE_EQ(d->rc, (int)RAD_OK);
    auto run = run_passes(d->b, deepseek41::step, { input.get() }, 0, 4);
    REQUIRE(steps_ok(run));
    CHECK_EQ(other_issue_hash(d->b, *run[0], input.get()), reference);
    for (int layer : { 1, 14 }) {
        const auto& L = d->m->layers[(size_t)layer];
        REQUIRE(!L.pf_post.empty());
        size_t gemm = 0, first = run[0]->issues.size();
        for (size_t i = 0; i < run[0]->issues.size(); ++i) {
            if (run[0]->issues[i].op == L.e_gemm) gemm = i;
            if (run[0]->issues[i].op == L.e_gather) first = std::min(first, i);
        }
        for (const auto& read : L.pf_post) {
            bool between = false;
            for (size_t i = gemm + 1; i < first; ++i) between |= run[0]->issues[i].op == read.op;
            CHECK(between);
        }
    }
}

RAD_TEST_MAIN()
