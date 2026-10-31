/* DS41's default-off exact AR row slices: graph bounds, every AR user's coverage and views.
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
    std::string old, old_es;
    bool had = false, had_es = false;
    explicit Env(const char* v) {
        if (const char* p = std::getenv(deepseek41::kArRowsEnv)) { had = true; old = p; }
        if (const char* p = std::getenv("RADIANCE_EXPERIMENTAL_DS41_ENGRAM_SPLIT")) { had_es = true; old_es = p; }
        if (v) setenv(deepseek41::kArRowsEnv, v, 1); else unsetenv(deepseek41::kArRowsEnv);
        setenv("RADIANCE_EXPERIMENTAL_DS41_ENGRAM_SPLIT", "1", 1);
    }
    ~Env() {
        if (had) setenv(deepseek41::kArRowsEnv, old.c_str(), 1); else unsetenv(deepseek41::kArRowsEnv);
        if (had_es) setenv("RADIANCE_EXPERIMENTAL_DS41_ENGRAM_SPLIT", old_es.c_str(), 1);
        else unsetenv("RADIANCE_EXPERIMENTAL_DS41_ENGRAM_SPLIT");
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
std::vector<RecIssue> ars(const RadBuilder& b, const RadCtx& c) {
    std::vector<RecIssue> out;
    for (const RecIssue& i : c.issues) if (b.op_of(i.op)->op == "all_reduce") out.push_back(i);
    return out;
}
} // namespace

TEST(ds41_ar_rows_defaults_off_validates_counts_and_keeps_other_worlds_widths_whole) {
    deepseek41::Model m;
    m.g.world = 4; m.g.n_embd = 5120; m.g.max_tok = 2048;
    for (const char* v : { (const char*)nullptr, "", "0" }) {
        REQUIRE_EQ(deepseek41::ar_rows_config(m, v), (int)RAD_OK);
        CHECK_EQ(m.ar_rows, int64_t(0));
    }
    REQUIRE_EQ(deepseek41::ar_rows_config(m, "1024"), (int)RAD_OK);
    CHECK_EQ(m.ar_rows, int64_t(1024));
    REQUIRE_EQ(deepseek41::ar_rows_config(m, "9999"), (int)RAD_OK);
    CHECK_EQ(m.ar_rows, int64_t(2048));
    for (const char* v : { "-1", "+1", "1x", " 1", "9223372036854775808" })
        CHECK_EQ(deepseek41::ar_rows_config(m, v), (int)RAD_E_INVAL);
    m.g.world = 3;
    REQUIRE_EQ(deepseek41::ar_rows_config(m, "1024"), (int)RAD_OK);
    CHECK_EQ(m.ar_rows, int64_t(0));
    m.g.world = 4; m.g.n_embd = 5136;
    REQUIRE_EQ(deepseek41::ar_rows_config(m, "1024"), (int)RAD_OK);
    CHECK_EQ(m.ar_rows, int64_t(0));
    rad::arch::MoeFP8 other;
    CHECK_EQ(other.ar_rows, int64_t(0));
}

TEST(ds41_ar_rows_bounds_every_trunk_draft_attention_and_embed_user_without_slicing_gather) {
    Env env("1024");
    constexpr int64_t D = 5120, C = 2048;
    for (int rank = 0; rank < 4; ++rank) {
        auto d = declare(rank, C);
        REQUIRE_EQ(d->rc, (int)RAD_OK);
        REQUIRE(d->m && d->m->dsp.on);
        int ar = 0, ag = 0;
        for (const RecOp& o : d->b.ops) {
            if (o.op == "all_reduce") {
                const RecParam* n = param(&o, "numel");
                REQUIRE(n);
                CHECK(n->ihi <= 1024 * D);
                ++ar;
            } else if (o.op == "all_gather" && pint(&o, "row") == d->m->esplit && d->m->esplit > 0) {
                CHECK_EQ(param(&o, "numel")->ihi, C * d->m->esplit);
                ++ag;
            }
        }
        CHECK(ar >= 1 + 2 * (40 + 3) + 1);
        for (const auto& L : d->m->layers) CHECK_EQ(L.moe.ar_rows, int64_t(1024));
        for (const auto& L : d->m->dsp.st) CHECK_EQ(L.moe.ar_rows, int64_t(1024));
        /* Engram splitting is an independent switch: where present its whole bound remains. */
        CHECK_EQ(ag, d->m->esplit > 0 ? 2 : 0);
    }
}

TEST(ds41_ar_rows_zero_keeps_the_original_graph_and_issue_arguments) {
    uint64_t graph = 0, issues = 0;
    for (const char* value : { (const char*)nullptr, "0" }) {
        Env env(value);
        auto d = declare(0);
        REQUIRE_EQ(d->rc, (int)RAD_OK);
        auto input = kit(d->b, {}, { 17 });
        auto run = run_passes(d->b, deepseek41::step, { input.get() }, 0, 4);
        REQUIRE(steps_ok(run));
        if (!value) { graph = graph_hash(d->b); issues = issue_hash(d->b, run, { input.get() }); }
        else { CHECK_EQ(graph_hash(d->b), graph); CHECK_EQ(issue_hash(d->b, run, { input.get() }), issues); }
        REQUIRE(param(d->b.op_of(d->m->embed_ar), "numel"));
        CHECK_EQ(param(d->b.op_of(d->m->embed_ar), "numel")->ihi, int64_t(2048 * 5120));
    }
}

TEST(ds41_ar_rows_preserves_full_row_coverage_at_small_steps_boundaries_and_ragged_tails) {
    constexpr int64_t D = 5120, C = 2048;
    std::vector<std::vector<RecIssue>> baseline;
    const std::vector<int> sizes = { 1, 17, 102, 103, 1024, 1025, 2048 };
    {
        Env env(nullptr);
        auto d = declare(0, C);
        REQUIRE_EQ(d->rc, (int)RAD_OK);
        for (int n : sizes) {
            auto k = kit(d->b, {}, { n });
            auto run = run_passes(d->b, deepseek41::step, { k.get() }, 0, 4);
            REQUIRE(steps_ok(run));
            baseline.push_back(ars(d->b, *run[0]));
        }
    }
    for (const char* cap : { "64", "1024" }) {
        Env env(cap);
        auto d = declare(0, C);
        REQUIRE_EQ(d->rc, (int)RAD_OK);
        for (size_t k = 0; k < sizes.size(); ++k) {
            auto input = kit(d->b, {}, { sizes[k] });
            auto run = run_passes(d->b, deepseek41::step, { input.get() }, 0, 4);
            REQUIRE(steps_ok(run));
            const auto sliced = ars(d->b, *run[0]);
            size_t at = 0;
            for (const RecIssue& full : baseline[k]) {
                int64_t done = 0;
                while (done < full.n) {
                    REQUIRE(at < sliced.size());
                    const RecIssue& s = sliced[at++];
                    CHECK_EQ(s.op, full.op);
                    CHECK(s.n > 0 && s.n <= std::atoll(cap) * D && s.n % D == 0);
                    REQUIRE(!s.opd.empty() && !full.opd.empty());
                    CHECK_EQ(s.opd[0].handle, full.opd[0].handle);
                    CHECK_EQ(s.opd[0].offset, full.opd[0].offset + done);
                    CHECK_EQ(s.opd[0].rows, s.n / D);
                    done += s.n;
                }
                CHECK_EQ(done, full.n);
            }
            CHECK_EQ(at, sliced.size());
        }
    }
}

TEST(ds41_ar_rows_rebases_suffixes_and_the_common_moe_path_without_changing_pitches) {
    RadBuilder b;
    const rad_op ar = rad_decl_op(&b, "all_reduce", nullptr, 0, nullptr, 0);
    RadCtx x; x.b = &b;
    constexpr int64_t D = 5120;
    deepseek41::issue_ar_rows(&x, ar, 7, 13, 1025, D, 1024);
    REQUIRE_EQ(x.issues.size(), size_t(2));
    CHECK_EQ(x.issues[0].opd[0].offset, 13 * D);
    CHECK_EQ(x.issues[1].opd[0].offset, 1037 * D);
    CHECK_EQ(x.issues[1].n, D);
    CHECK_EQ(x.issues[1].opd[0].cols, int64_t(0));   /* keep the buffer's declared full row pitch */
    rad::arch::MoeFP8 moe;
    moe.g.n_embd = D; moe.op_ar = ar; moe.w.h.x = 7; moe.ar_rows = 1024;
    x.issues.clear();
    moe.issue_ar(&x, 13, 1025, false);
    REQUIRE_EQ(x.issues.size(), size_t(2));
    CHECK_EQ(x.issues[0].opd[0].offset, 13 * D);
    CHECK_EQ(x.issues[1].opd[0].offset, 1037 * D);
    CHECK_EQ(x.issues[1].n, D);
}

RAD_TEST_MAIN()
