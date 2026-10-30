/* ds41_dspark_defer_arch_test.cpp -- deepseek41's half of two DSpark controls (core/sched/draft_conf.h),
 * on the recording builder (arch_fakes.h) at DeepSeek-V4.1-Flash's geometry with its three stages:
 *   - RADIANCE_EXPERIMENTAL_DS41_DSPARK_CTX_DEFER: the drafter's source rows are persistent with the
 *     drafter's window (129 rows) more; each trunk step's taps go after the rows still waiting; a
 *     context pass of T rows reads the last T of them and starts the count over; past the room the
 *     taps start over at row 0 (said once), and a pass longer than what waits reads from row 0;
 *   - RADIANCE_EXPERIMENTAL_DS41_DSPARK_CONF: the last stage's collapsed rows are persistent;
 *   - unset, both declare and issue what they did before (transient buffers, taps at row 0, the
 *     context pass at row 0). */
#define RAD_ARCH_NO_EXPORTS 1

#include "arch_fakes.h"

#include "../arch/deepseek41/deepseek41.cpp"

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr int kWorld = 4;

const char* const kVars[] = { "RADIANCE_EXPERIMENTAL_DS41_DSPARK_CTX_DEFER",
                              "RADIANCE_EXPERIMENTAL_DS41_DSPARK_CONF",
                              "RADIANCE_EXPERIMENTAL_DS41_DSPARK_SHORT_BLOCK",
                              "RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8",
                              "RADIANCE_EXPERIMENTAL_DS41_DSPARK_FP8",
                              "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ATTN",
                              "RADIANCE_EXPERIMENTAL_DS41_ENGRAM_SPLIT",
                              "RADIANCE_EXPERIMENTAL_DS41_ROUTER_BF16" };
struct Env {
    Env(std::initializer_list<std::pair<const char*, const char*>> kv) {
        for (const char* n : kVars) unsetenv(n);
        for (const auto& p : kv) setenv(p.first, p.second, 1);
    }
    ~Env() {
        for (const char* n : kVars) unsetenv(n);
    }
};

/* As ds41_dspark_fp8_arch_test's. */
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

int64_t block_of(const std::string& n) {
    for (const char* p : { "kv_compressed_r", "kv_index_r" })
        if (n.rfind(p, 0) == 0) return std::atoll(n.c_str() + std::strlen(p));
    return 0;
}

struct Decl {
    RadBuilder b;
    int rc = RAD_E_INVAL;
    std::string err;
    deepseek41::Model* m = nullptr;
};

std::unique_ptr<Decl> declare(int rank) {
    auto d = std::make_unique<Decl>();
    d->b.kv_block_of = block_of;
    MetaKV meta = dspark_meta();
    RadBuildCtx ctx = spec_ctx(rank, kWorld, 5);
    ctx.max_tok = 16;
    ctx.max_seqs = 1;
    d->err = stderr_of([&] { d->rc = deepseek41::declare(&d->b, meta.meta(), &ctx); });
    d->m = deepseek41::models[rank].get();
    return d;
}

/* The operands of every issue of `op`, in order. */
std::vector<RadOperand> operands_of(const RadCtx& c, rad_op op, size_t i) {
    std::vector<RadOperand> v;
    for (const auto& is : c.issues)
        if (is.op == op && is.opd.size() > i) v.push_back(is.opd[i]);
    return v;
}

}  // namespace

TEST(ds41_dspark_controls_off_declare_and_issue_as_before) {
    Env env({});
    auto d = declare(0);
    REQUIRE_EQ(d->rc, (int)RAD_OK);
    const auto& ds = d->m->dsp;
    REQUIRE(ds.on);
    CHECK(!ds.ctx_defer);
    CHECK(!ds.conf_x);
    const RecBuf* src = find_b(d->b, "dspark_src");
    const RecBuf* x = find_b(d->b, "dspark_collapsed");
    REQUIRE(src);
    REQUIRE(x);
    CHECK_EQ(src->d.shape[0], (int64_t)16);
    CHECK_EQ(src->d.kind, (int)RAD_BUF_TRANSIENT);
    CHECK_EQ(x->d.kind, (int)RAD_BUF_TRANSIENT);
    /* Two decode steps and a context pass: every tap and the pass's fc at row 0. */
    auto t1 = kit(d->b, { 1 }, {});
    auto t2 = kit(d->b, { 1 }, {});
    auto cp = context_pass(d->b, *t2);
    auto r = run_passes(d->b, deepseek41::step, { t1.get(), t2.get(), cp.get() }, 0, kWorld);
    REQUIRE(steps_ok(r));
    for (int p = 0; p < 2; ++p)
        for (const auto& L : d->m->layers) {
            if (!L.tap) continue;
            const auto o = operands_of(*r[(size_t)p], L.tap, 1);
            REQUIRE_EQ(o.size(), (size_t)1);
            CHECK_EQ(o[0].offset, (int64_t)L.tap_col * 5120);
        }
    const auto fc = operands_of(*r[2], ds.c_fc, 0);
    REQUIRE_EQ(fc.size(), (size_t)1);
    CHECK_EQ(fc[0].offset, (int64_t)0);
    CHECK_EQ(fc[0].rows, (int64_t)1);
}

TEST(ds41_dspark_ctx_defer_appends_taps_and_reads_the_last_rows) {
    Env env({ { "RADIANCE_EXPERIMENTAL_DS41_DSPARK_CTX_DEFER", "1" } });
    for (int rank = 0; rank < kWorld; ++rank) {
        auto d = declare(rank);
        REQUIRE_EQ(d->rc, (int)RAD_OK);
        auto& ds = d->m->dsp;
        REQUIRE(ds.ctx_defer);
        if (d->m->zero) continue;   /* an attention-zero rank keeps no keys and has no source rows */
        const RecBuf* src = find_b(d->b, "dspark_src");
        REQUIRE(src);
        CHECK_EQ(src->d.shape[0], (int64_t)(16 + 129));
        CHECK_EQ(src->d.kind, (int)RAD_BUF_PERSIST);
        CHECK_EQ(ds.src_rows, (int64_t)(16 + 129));
        const int64_t pitch = 3 * 5120;
        /* Three decode steps of one row, a two-row verify, then a context pass of three rows: the
         * taps at rows 0, 1, 2, 3..4; the pass reads rows 2..4, the last three. */
        auto t1 = kit(d->b, { 1 }, {});
        auto t2 = kit(d->b, { 1 }, {});
        auto t3 = kit(d->b, { 1 }, {});
        auto t4 = kit(d->b, { 2 }, {});
        auto cp = kit(d->b, { 3 }, {});
        cp->b.draft_pass = -1;
        auto r = run_passes(d->b, deepseek41::step, { t1.get(), t2.get(), t3.get(), t4.get(), cp.get() },
                            rank, kWorld);
        REQUIRE(steps_ok(r));
        const int64_t at[4] = { 0, 1, 2, 3 };
        for (int p = 0; p < 4; ++p)
            for (const auto& L : d->m->layers) {
                if (!L.tap) continue;
                const auto o = operands_of(*r[(size_t)p], L.tap, 1);
                REQUIRE_EQ(o.size(), (size_t)1);
                CHECK_EQ(o[0].offset, at[p] * pitch + (int64_t)L.tap_col * 5120);
                CHECK_EQ(o[0].rows, p == 3 ? (int64_t)2 : (int64_t)1);
                CHECK_EQ(o[0].cols, (int64_t)5120);
            }
        CHECK_EQ(ds.tap_rows, (int64_t)0);
        const auto fc = operands_of(*r[4], ds.c_fc, 0);
        REQUIRE_EQ(fc.size(), (size_t)1);
        CHECK_EQ(fc[0].offset, 2 * pitch);
        CHECK_EQ(fc[0].rows, (int64_t)3);
        /* The next trunk step starts over at row 0. */
        auto t5 = kit(d->b, { 1 }, {});
        auto r5 = run_passes(d->b, deepseek41::step, { t5.get() }, rank, kWorld);
        REQUIRE(steps_ok(r5));
        for (const auto& L : d->m->layers)
            if (L.tap) CHECK_EQ(operands_of(*r5[0], L.tap, 1)[0].offset, (int64_t)L.tap_col * 5120);
        CHECK_EQ(ds.tap_rows, (int64_t)1);
        CHECK(d->err.find("count out of step") == std::string::npos);
    }
}

TEST(ds41_dspark_ctx_defer_out_of_step_is_said_once_and_never_overruns) {
    Env env({ { "RADIANCE_EXPERIMENTAL_DS41_DSPARK_CTX_DEFER", "1" } });
    auto d = declare(0);
    REQUIRE_EQ(d->rc, (int)RAD_OK);
    auto& ds = d->m->dsp;
    /* Nine full steps (144 rows) fit 145; the tenth starts over at row 0. */
    std::vector<std::unique_ptr<Kit>> ks;
    std::vector<const Kit*> ps;
    for (int i = 0; i < 10; ++i) {
        ks.push_back(kit(d->b, { 16 }, {}));
        ps.push_back(ks.back().get());
    }
    std::string err = stderr_of([&] {
        auto r = run_passes(d->b, deepseek41::step, ps, 0, kWorld);
        CHECK(steps_ok(r));
        for (const auto& L : d->m->layers)
            if (L.tap) CHECK_EQ(operands_of(*r[9], L.tap, 1)[0].offset, (int64_t)L.tap_col * 5120);
    });
    CHECK(err.find("count out of step") != std::string::npos);
    CHECK_EQ(ds.tap_rows, (int64_t)16);
    /* A pass of what waits reads it; one longer than what waits (nothing now) reads from row 0, and
     * is said no more (once a run). */
    auto cp = kit(d->b, { 16 }, {});
    cp->b.draft_pass = -1;
    auto c1 = kit(d->b, { 1 }, {});
    c1->b.draft_pass = -1;
    err = stderr_of([&] {
        auto r = run_passes(d->b, deepseek41::step, { cp.get(), c1.get() }, 0, kWorld);
        CHECK(steps_ok(r));
        CHECK_EQ(operands_of(*r[0], ds.c_fc, 0)[0].offset, (int64_t)0);
        CHECK_EQ(operands_of(*r[0], ds.c_fc, 0)[0].rows, (int64_t)16);
        CHECK_EQ(operands_of(*r[1], ds.c_fc, 0)[0].offset, (int64_t)0);
        CHECK_EQ(operands_of(*r[1], ds.c_fc, 0)[0].rows, (int64_t)1);
    });
    CHECK_EQ(ds.tap_rows, (int64_t)0);
    CHECK(err.find("count out of step") == std::string::npos);
}

TEST(ds41_dspark_conf_keeps_the_collapsed_rows) {
    Env env({ { "RADIANCE_EXPERIMENTAL_DS41_DSPARK_CONF", "0.55" } });
    auto d = declare(0);
    REQUIRE_EQ(d->rc, (int)RAD_OK);
    CHECK(d->m->dsp.conf_x);
    CHECK(!d->m->dsp.ctx_defer);
    const RecBuf* x = find_b(d->b, "dspark_collapsed");
    REQUIRE(x);
    CHECK_EQ(x->d.kind, (int)RAD_BUF_PERSIST);
    CHECK_EQ(x->d.dtype, (uint32_t)RAD_BF16);
    CHECK_EQ(x->d.shape[1], (int64_t)5120);
    const RecBuf* src = find_b(d->b, "dspark_src");
    REQUIRE(src);
    CHECK_EQ(src->d.kind, (int)RAD_BUF_TRANSIENT);
    /* "0" is off. */
    Env off({ { "RADIANCE_EXPERIMENTAL_DS41_DSPARK_CONF", "0" },
              { "RADIANCE_EXPERIMENTAL_DS41_DSPARK_CTX_DEFER", "0" } });
    auto e = declare(0);
    REQUIRE_EQ(e->rc, (int)RAD_OK);
    CHECK(!e->m->dsp.conf_x);
    CHECK(!e->m->dsp.ctx_defer);
}

RAD_TEST_MAIN()
