/* ds41_dspark_defer_arch_test.cpp -- deepseek41's half of two DSpark controls (core/sched/draft_conf.h),
 * on the recording builder (arch_fakes.h) at DeepSeek-V4.1-Flash's geometry with its three stages:
 *   - RADIANCE_EXPERIMENTAL_DS41_DSPARK_CTX_DEFER: a persistent tap history beside the source rows, a
 *     row for each slot the core carves for the drafter's own cache (kv_dspark: a window of blocks a
 *     sequence and a step's); every trunk step taps into rows [0, T) as before and scatters them into
 *     the history at its kv_dspark slot_mapping once its last tap is in; a context pass gathers its
 *     rows back from the history by its own slot_mapping before its fc; an attention-zero rank has
 *     neither;
 *   - RADIANCE_EXPERIMENTAL_DS41_DSPARK_CONF: the last stage's collapsed rows are persistent;
 *   - unset, both declare and issue what they did before (transient buffers, taps at row 0, the
 *     context pass at row 0). */
#define RAD_ARCH_NO_EXPORTS 1

#include "arch_fakes.h"

#include "../arch/deepseek41/deepseek41.cpp"

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr int kWorld = 4;

const char* const kVars[] = { "RADIANCE_EXPERIMENTAL_DS41_DSPARK_CTX_DEFER",
                              rad_dflash_observe::kEnv,
                              rad_dflash_observe::kEarlyEnv,
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

std::unique_ptr<Decl> declare(int rank, int max_seqs = 1) {
    auto d = std::make_unique<Decl>();
    d->b.kv_block_of = block_of;
    MetaKV meta = dspark_meta();
    RadBuildCtx ctx = spec_ctx(rank, kWorld, 5);
    ctx.max_tok = 16;
    ctx.max_seqs = max_seqs;
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

/* The index of the first and last issue of `op` in a pass, -1 when none. */
int first_issue(const RadCtx& c, rad_op op) {
    for (size_t i = 0; i < c.issues.size(); ++i)
        if (c.issues[i].op == op) return (int)i;
    return -1;
}
int last_issue(const RadCtx& c, rad_op op) {
    int at = -1;
    for (size_t i = 0; i < c.issues.size(); ++i)
        if (c.issues[i].op == op) at = (int)i;
    return at;
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
    CHECK(!ds.prob_capture); CHECK(!ds.observe); CHECK(!ds.observations);
    CHECK(find_b(d->b, rad_dflash_observe::kBuffer) == nullptr);
    const RecBuf* src = find_b(d->b, "dspark_src");
    const RecBuf* x = find_b(d->b, "dspark_collapsed");
    REQUIRE(src);
    REQUIRE(x);
    CHECK_EQ(src->d.shape[0], (int64_t)16);
    CHECK_EQ(src->d.kind, (int)RAD_BUF_TRANSIENT);
    CHECK_EQ(x->d.kind, (int)RAD_BUF_TRANSIENT);
    CHECK(find_b(d->b, "dspark_hist") == nullptr);
    CHECK(!ds.hist); CHECK(!ds.hist_put); CHECK(!ds.hist_get);
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

/* The history's rows are the slots the core carves for kv_dspark (KVManager::window_pool_cap): the
 * window 128 + 5 + 5 is ceil(138 / 16) + 1 = 10 blocks, so 11 a sequence, and one step's block: 12
 * blocks at one sequence (the 192 tokens a served bank's report shows), 45 at four, 89 at eight. */
TEST(ds41_dspark_ctx_defer_declares_a_tap_history_a_row_a_drafter_cache_slot) {
    Env env({ { "RADIANCE_EXPERIMENTAL_DS41_DSPARK_CTX_DEFER", "1" } });
    const int seqs[3] = { 1, 4, 8 };
    const int64_t blocks[3] = { 12, 45, 89 };
    for (int k = 0; k < 3; ++k)
        for (int rank = 0; rank < kWorld; ++rank) {
            auto d = declare(rank, seqs[k]);
            REQUIRE_EQ(d->rc, (int)RAD_OK);
            const auto& ds = d->m->dsp;
            REQUIRE(ds.ctx_defer);
            if (d->m->zero) {   /* an attention-zero rank keeps no keys: no source rows, no history */
                CHECK(!ds.hist); CHECK(!ds.hist_put); CHECK(!ds.hist_get);
                CHECK(find_b(d->b, "dspark_hist") == nullptr);
                continue;
            }
            const RecBuf* src = find_b(d->b, "dspark_src");
            const RecBuf* hist = find_b(d->b, "dspark_hist");
            REQUIRE(src);
            REQUIRE(hist);
            CHECK_EQ(src->d.shape[0], (int64_t)16);
            CHECK_EQ(src->d.kind, (int)RAD_BUF_TRANSIENT);
            CHECK_EQ(hist->d.kind, (int)RAD_BUF_PERSIST);
            CHECK_EQ(hist->d.dtype, (uint32_t)RAD_BF16);
            CHECK_EQ(hist->d.shape[0], blocks[k] * 16);
            CHECK_EQ(hist->d.shape[1], (int64_t)3 * 5120);
            CHECK_EQ(ds.hist_rows, blocks[k] * 16);
            CHECK(ds.hist_put); CHECK(ds.hist_get);
            CHECK_EQ(ds.kv_bs, (int64_t)16);
        }
}

/* Two decode steps of one row, a two-row verify, then a context pass of three rows (whichever
 * sequences and positions the core puts in it). Every tap still goes to rows [0, T); the trunk step
 * then scatters its T rows into the history at its own kv_dspark slot_mapping, after its last tap;
 * the context pass gathers its T rows from the history at its slot_mapping before its fc reads them
 * from row 0. The lint (aliasing, spans, declared ranges, unwritten reads) finds nothing over the four
 * for the source rows, the history or the two ops; the whole lint also flags DeepSeek-V4.1 trunk
 * patterns this fake does not model (gemm_nt over column views of attn_o, the routed MoE's operands),
 * which is why no other ds41 arch test runs it. */
TEST(ds41_dspark_ctx_defer_scatters_each_steps_taps_and_gathers_a_pass_by_slot) {
    Env env({ { "RADIANCE_EXPERIMENTAL_DS41_DSPARK_CTX_DEFER", "1" } });
    for (int rank = 0; rank < kWorld; ++rank) {
        auto d = declare(rank, 4);
        REQUIRE_EQ(d->rc, (int)RAD_OK);
        const auto& ds = d->m->dsp;
        if (d->m->zero) continue;
        const int64_t pitch = 3 * 5120;
        auto t1 = kit(d->b, { 1 }, {});
        auto t2 = kit(d->b, { 1 }, {});
        auto t3 = kit(d->b, { 2 }, {});
        auto cp = kit(d->b, { 3 }, {});
        cp->b.draft_pass = -1;
        auto r = run_passes(d->b, deepseek41::step, { t1.get(), t2.get(), t3.get(), cp.get() }, rank, kWorld);
        REQUIRE(steps_ok(r));
        std::vector<std::string> mine;
        for (const std::string& f : lint_step(d->b, view(r)))
            for (const char* k : { "dspark_src", "dspark_hist", "scatter_rows", "gather_rows" })
                if (f.find(k) != std::string::npos) { mine.push_back(f); break; }
        CHECK_EQ(report_lint(mine), (size_t)0);
        const Kit* trunk[3] = { t1.get(), t2.get(), t3.get() };
        for (int p = 0; p < 3; ++p) {
            const int64_t T = trunk[p]->b.n_tok;
            int last_tap = -1;
            for (const auto& L : d->m->layers) {
                if (!L.tap) continue;
                const auto o = operands_of(*r[(size_t)p], L.tap, 1);
                REQUIRE_EQ(o.size(), (size_t)1);
                CHECK_EQ(o[0].offset, (int64_t)L.tap_col * 5120);
                CHECK_EQ(o[0].rows, T);
                last_tap = std::max(last_tap, last_issue(*r[(size_t)p], L.tap));
            }
            const int put = first_issue(*r[(size_t)p], ds.hist_put);
            REQUIRE(put >= 0);
            CHECK(put > last_tap);
            CHECK_EQ(last_issue(*r[(size_t)p], ds.hist_put), put);   /* once a step */
            CHECK_EQ(first_issue(*r[(size_t)p], ds.hist_get), -1);
            const auto& opd = r[(size_t)p]->issues[(size_t)put].opd;
            REQUIRE_EQ(opd.size(), (size_t)3);
            CHECK_EQ(opd[0].handle, ds.src); CHECK_EQ(opd[0].offset, (int64_t)0); CHECK_EQ(opd[0].rows, T);
            CHECK_EQ((int)opd[1].kind, (int)RAD_OPK_RAW);
            CHECK(opd[1].raw == (void*)trunk[p]->b.kv[ds.kv - 1].slot_mapping);
            CHECK_EQ(opd[1].rows, T);
            CHECK_EQ(opd[2].handle, ds.hist); CHECK_EQ(opd[2].rows, ds.hist_rows);
        }
        /* The context pass: the gather, then the fc over the rows it wrote. */
        const int get = first_issue(*r[3], ds.hist_get);
        const int fc = first_issue(*r[3], ds.c_fc);
        REQUIRE(get >= 0);
        CHECK(get < fc);
        CHECK_EQ(first_issue(*r[3], ds.hist_put), -1);
        const auto& g = r[3]->issues[(size_t)get].opd;
        REQUIRE_EQ(g.size(), (size_t)3);
        CHECK_EQ(g[0].handle, ds.hist); CHECK_EQ(g[0].rows, ds.hist_rows);
        CHECK_EQ((int)g[1].kind, (int)RAD_OPK_RAW);
        CHECK(g[1].raw == (void*)cp->b.kv[ds.kv - 1].slot_mapping);
        CHECK_EQ(g[1].rows, (int64_t)3);
        CHECK_EQ(g[2].handle, ds.src); CHECK_EQ(g[2].offset, (int64_t)0); CHECK_EQ(g[2].rows, (int64_t)3);
        const auto f = operands_of(*r[3], ds.c_fc, 0);
        REQUIRE_EQ(f.size(), (size_t)1);
        CHECK_EQ(f[0].handle, ds.src); CHECK_EQ(f[0].offset, (int64_t)0); CHECK_EQ(f[0].rows, (int64_t)3);
        CHECK_EQ(pitch, (int64_t)3 * 5120);
    }
}

/* A trunk step whose batch carries no kv_dspark slots cannot keep its taps: the step fails, said by
 * name, rather than scattering nowhere. */
TEST(ds41_dspark_ctx_defer_refuses_a_trunk_step_without_drafter_slots) {
    Env env({ { "RADIANCE_EXPERIMENTAL_DS41_DSPARK_CTX_DEFER", "1" } });
    auto d = declare(0);
    REQUIRE_EQ(d->rc, (int)RAD_OK);
    REQUIRE(!d->m->zero);
    auto t1 = kit(d->b, { 1 }, {});
    t1->kv[d->m->dsp.kv - 1].slot_mapping = nullptr;
    auto r = run_passes(d->b, deepseek41::step, { t1.get() }, 0, kWorld);
    REQUIRE_EQ(r.size(), (size_t)1);
    CHECK(r[0]->step_fail.find("tap history") != std::string::npos);
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

TEST(ds41_dspark_proposal_capture_is_persistent_and_issued_after_unchanged_selection) {
    for (int short_block : { 0, 3 }) {
        Env env({ { rad_dflash_observe::kEnv, "1" },
                  { "RADIANCE_EXPERIMENTAL_DS41_DSPARK_SHORT_BLOCK", short_block ? "3" : "0" } });
        auto d = declare(0, 4);
        REQUIRE_EQ(d->rc, (int)RAD_OK);
        const auto& ds = d->m->dsp;
        const RecBuf* obs = find_b(d->b, rad_dflash_observe::kBuffer);
        REQUIRE(obs);
        CHECK_EQ(obs->d.kind, (int)RAD_BUF_PERSIST);
        CHECK_EQ(obs->d.dtype, (uint32_t)RAD_F32);
        CHECK_EQ(obs->d.shape[0], (int64_t)4);
        CHECK_EQ(obs->d.shape[1], (int64_t)rad_dflash_observe::kWidth);
        auto q = query_pass(d->b, 2, short_block ? 3 : 5, short_block ? 3 : 5);
        auto runs = run_passes(d->b, deepseek41::step, { q.get() }, 0, kWorld);
        REQUIRE(steps_ok(runs));
        const rad_op select = short_block ? ds.select_lo : ds.select;
        const rad_op observe = short_block ? ds.observe_lo : ds.observe;
        const int sel = first_issue(*runs[0], select), cap = first_issue(*runs[0], observe);
        REQUIRE(sel >= 0); REQUIRE(cap > sel);
        const auto& before = runs[0]->issues[(size_t)sel].opd;
        const auto& after = runs[0]->issues[(size_t)cap].opd;
        REQUIRE_EQ(before.size(), (size_t)7); REQUIRE_EQ(after.size(), (size_t)8);
        for (size_t i = 0; i < before.size(); ++i) {
            CHECK_EQ(before[i].handle, after[i].handle);
            CHECK_EQ(before[i].rows, after[i].rows);
            CHECK_EQ(before[i].offset, after[i].offset);
        }
        CHECK_EQ(after[7].handle, ds.observations);
        CHECK_EQ(after[7].rows, (int64_t)2);
    }
}

TEST(ds41_dspark_early_fixed_controls_observe_the_same_db2_query_even_at_zero) {
    for (const char* k : { "0", "1", "2" }) {
        Env env({ { rad_dflash_observe::kEnv, "0" }, { rad_dflash_observe::kEarlyEnv, k },
                  { "RADIANCE_EXPERIMENTAL_DS41_DSPARK_SHORT_BLOCK", "2" } });
        auto d = declare(0);
        REQUIRE_EQ(d->rc, (int)RAD_OK);
        const auto& ds = d->m->dsp;
        const RecBuf* obs = find_b(d->b, rad_dflash_observe::kBuffer);
        REQUIRE(obs); CHECK_EQ(obs->d.kind, (int)RAD_BUF_PERSIST);
        auto q = query_pass(d->b, 1, 2, 2);
        auto runs = run_passes(d->b, deepseek41::step, { q.get() }, 0, kWorld);
        REQUIRE(steps_ok(runs));
        const int sel = first_issue(*runs[0], ds.select_lo), cap = first_issue(*runs[0], ds.observe_lo);
        REQUIRE(sel >= 0); REQUIRE(cap > sel);
        const auto& before = runs[0]->issues[(size_t)sel].opd;
        const auto& after = runs[0]->issues[(size_t)cap].opd;
        REQUIRE_EQ(before.size(), (size_t)7); REQUIRE_EQ(after.size(), (size_t)8);
        for (size_t j = 0; j < before.size(); ++j) {
            CHECK_EQ(before[j].handle, after[j].handle);
            CHECK_EQ(before[j].rows, after[j].rows);
            CHECK_EQ(before[j].offset, after[j].offset);
        }
        CHECK_EQ(after[7].handle, ds.observations); CHECK_EQ(after[7].rows, (int64_t)1);
    }
    Env off({ { rad_dflash_observe::kEnv, "0" }, { rad_dflash_observe::kEarlyEnv, "off" },
              { "RADIANCE_EXPERIMENTAL_DS41_DSPARK_SHORT_BLOCK", "2" } });
    auto d = declare(0);
    REQUIRE_EQ(d->rc, (int)RAD_OK);
    CHECK(find_b(d->b, rad_dflash_observe::kBuffer) == nullptr);
    CHECK_EQ(d->m->dsp.observe, (rad_op)0); CHECK_EQ(d->m->dsp.observe_lo, (rad_op)0);
}

RAD_TEST_MAIN()
