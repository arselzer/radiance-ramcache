/* ds41_tbo_arch_test.cpp -- deepseek41's two-batch overlap (RADIANCE_EXPERIMENTAL_SCHED_TBO=1,
 * core/sched/tbo.h) on the recording builder (arch_fakes.h) with DeepSeek-V4.1-Flash's own geometry
 * at TP4, no card or model file. The exactness argument, checked op by op:
 *
 *   1. each micro-batch issues exactly what a step of its rows alone issues: the split step's layer
 *      issues are an interleaving of the issues of a lone step of micro-batch A's sequences and of a
 *      lone step of B's (same op handles, issue counts, weights, cache groups and batch arrays; a
 *      buffer operand at the same offset, or for the step-tall buffers exactly B's first row's
 *      offset further on, the same for every use of a buffer); its prologue (embedding, Engram ids,
 *      rope tables) and its head are the batched step's, op for op;
 *   2. the interleaving never lets one micro-batch read bytes the other wrote: walking the split
 *      step's issues with each op's declared reads and writes, every element a micro-batch's op reads
 *      was last written by that micro-batch, the prologue, or by nothing this step; and the buffer
 *      plan under the declaration keeps every transient's bytes its own, so the step lint is clean;
 *   3. the order is the overlap's: for every layer, B's attention between A's gate_up and A's
 *      post, B's gate_up after A's post, and A's next attention between B's gate_up and B's post.
 *
 * With 1 and 2 the split step computes, row by row, what the lone steps compute; a row's result does
 * not depend on its step's other rows (the replies at 1, 2, 4 and 8 sequences are identical), so it
 * is what the batched step computes. Also: the switch off (or not declared) splits nothing; one
 * sequence, a prefill chunk or a mixed step is never split. */
#define RAD_ARCH_NO_EXPORTS 1

#include "arch_fakes.h"

#include "../arch/deepseek41/deepseek41.cpp"
#include "../core/sched/tbo.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace {

constexpr int kWorld = 4;

const char* const kVars[] = { "RADIANCE_EXPERIMENTAL_SCHED_TBO",
                              "RADIANCE_EXPERIMENTAL_DS41_CONSUMER_SKIP",
                              "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ATTN",
                              "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_MIB",
                              "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ENGRAM",
                              "RADIANCE_EXPERIMENTAL_DS41_ENGRAM_SPLIT",
                              "RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8",
                              "RADIANCE_EXPERIMENTAL_DS41_ROUTER_BF16",
                              "RADIANCE_EXPERIMENTAL_DS41_KV_REPLICATED" };
struct Env {
    Env(std::initializer_list<std::pair<const char*, const char*>> kv) { set(kv); }
    void set(std::initializer_list<std::pair<const char*, const char*>> kv) {
        for (const char* n : kVars) unsetenv(n);
        for (const auto& p : kv) setenv(p.first, p.second, 1);
    }
    ~Env() {
        for (const char* n : kVars) unsetenv(n);
    }
};

/* The release's geometry as llama.cpp's converter states it (ds41_consumer_skip_arch_test's). */
MetaKV flash_meta() {
    RadModelMeta base{};
    base.arch_id = "deepseek41";
    base.name = "ds41-flash-geometry";
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
        .set("deepseek41.ple.head_vocab_sizes", "1000,1000");
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
/* deepseek41's declare for `rank` of four at 16 rows a step (eight sequences), under the current
 * environment. */
std::unique_ptr<Decl> declare(int rank, int world = kWorld) {
    auto d = std::make_unique<Decl>();
    d->b.kv_block_of = block_of;
    RadBuildCtx ctx = build_ctx(rank, world);
    ctx.max_tok = 16;
    ctx.max_ctx = 8192;
    MetaKV meta = flash_meta();
    d->err = stderr_of([&] { d->rc = deepseek41::declare(&d->b, meta.meta(), &ctx); });
    d->m = deepseek41::models[rank].get();
    return d;
}

/* A step of `rows` decode sequences, one row each. */
std::unique_ptr<Kit> decode(const RadBuilder& b, int rows) {
    return kit(b, std::vector<int>((size_t)rows, 1), {});
}

/* A lone step of the sequences [s0, s0 + n) of `full` (a decode step, one row a sequence): every
 * per-row and per-sequence array at those sequences, cu_seqlens its first n + 1 entries (0 .. n),
 * the step-wide bounds (max_ctx_len, max_q_len) the full step's, as the split step passes them. */
struct Lone {
    RadBatch b{};
    std::vector<RadKVGroupBatch> kv;
};
std::unique_ptr<Lone> lone(const Kit& full, int64_t s0, int64_t n) {
    auto L = std::make_unique<Lone>();
    L->b = full.b;
    RadBatch& b = L->b;
    b.n_tok = n;
    b.n_seq = n;
    b.n_seq_decode = n;
    b.n_tok_decode = n;
    b.token_ids = full.b.token_ids + s0;
    b.positions = full.b.positions + s0;
    b.cu_seqlens = full.b.cu_seqlens;
    b.n_out = n;
    b.out_ids = full.b.out_ids;   /* 0 .. n-1 */
    b.q_lens = full.b.q_lens + s0;
    b.ctx_lens = full.b.ctx_lens + s0;
    b.num_accepted = full.b.num_accepted + s0;
    for (int g = 0; g < full.b.n_kv_groups; ++g) {
        RadKVGroupBatch k = full.b.kv[g];
        k.slot_mapping = k.slot_mapping + s0;
        k.block_table = k.block_table + s0 * k.block_table_pitch;
        k.seqused = k.seqused + s0;
        k.state_index = k.state_index + s0 * (k.state_index_pitch > 0 ? k.state_index_pitch : 1);
        L->kv.push_back(k);
    }
    b.kv = L->kv.data();
    return L;
}

std::unique_ptr<RadCtx> run(RadBuilder& b, const RadBatch& batch, int rank) {
    auto c = std::make_unique<RadCtx>();
    c->b = &b;
    c->batch = &batch;
    c->rank = rank;
    c->world = kWorld;
    deepseek41::step(c.get(), &batch);
    return c;
}

std::string op_name(const RadBuilder& b, rad_op h) {
    const RecOp* o = b.op_of(h);
    return o ? o->op : std::string("<none>");
}

/* A buffer's row pitch in elements (its declared numel over its first dimension) and numel. */
int64_t pitch_of(const RadBuilder& b, rad_buf h) {
    const RadBufDecl& d = b.bufs[(size_t)h - 1].d;
    int64_t n = 1;
    for (uint32_t i = 1; i < d.rank; ++i) n *= d.shape[i];
    return d.rank >= 1 ? n : 1;
}
int64_t rows_of_buf(const RadBuilder& b, rad_buf h) {
    const RadBufDecl& d = b.bufs[(size_t)h - 1].d;
    return d.rank >= 1 ? d.shape[0] : 1;
}

/* An issue that reads only weights: an on-die read (Layer::pf), which moves no value. */
bool weight_only(const RecIssue& i) {
    if (i.opd.empty()) return false;
    for (const RadOperand& o : i.opd)
        if (o.kind != RAD_OPK_WEIGHT) return false;
    return true;
}

bool same_operand(const RadOperand& x, const RadOperand& y) {
    return x.kind == y.kind && x.dtype == y.dtype && x.handle == y.handle && x.raw == y.raw &&
           x.offset == y.offset && x.rows == y.rows && x.cols == y.cols;
}
bool same_issue(const RecIssue& x, const RecIssue& y) {
    if (x.op != y.op || x.n != y.n || x.lane != y.lane || x.opd.size() != y.opd.size()) return false;
    for (size_t i = 0; i < x.opd.size(); ++i)
        if (!same_operand(x.opd[i], y.opd[i])) return false;
    return true;
}
/* x (the split step's) is y (a lone step's) on rows `shift` further on: every operand the same,
 * but a buffer's offset may be `shift` rows of its pitch further. */
bool shifted_issue(const RadBuilder& b, const RecIssue& x, const RecIssue& y, int64_t shift) {
    if (x.op != y.op || x.n != y.n || x.lane != y.lane || x.opd.size() != y.opd.size()) return false;
    for (size_t i = 0; i < x.opd.size(); ++i) {
        const RadOperand &p = x.opd[i], &q = y.opd[i];
        if (same_operand(p, q)) continue;
        if (p.kind != RAD_OPK_BUF || q.kind != RAD_OPK_BUF || p.handle != q.handle || p.rows != q.rows ||
            p.cols != q.cols || p.dtype != q.dtype)
            return false;
        /* Handle 0: a plane this fake geometry leaves undeclared (the lint's "handle out of range",
         * the batched step's too); its pitch is unknown, so any whole number of rows. */
        if (p.handle == 0) {
            if ((p.offset - q.offset) % shift != 0) return false;
            continue;
        }
        if (p.offset - q.offset != shift * pitch_of(b, (rad_buf)p.handle)) return false;
    }
    return true;
}

std::vector<RecIssue> value_issues(const std::vector<RecIssue>& v, size_t lo, size_t hi) {
    std::vector<RecIssue> out;
    for (size_t i = lo; i < hi && i < v.size(); ++i)
        if (!weight_only(v[i])) out.push_back(v[i]);
    return out;
}

/* Where a step's layer loop starts (its first issue not of the prologue: the embedding, the
 * Engram ids, the rope tables) and ends (one past its last layer's post). */
struct Region { size_t lo = 0, hi = 0; };
Region loop_region(const deepseek41::Model& m, const std::vector<RecIssue>& v) {
    std::set<rad_op> pro = { m.embed, m.embed_ar, m.expand, m.pre0, m.pre1, m.rope_a, m.rope_b_step,
                             m.tok_map };
    for (const auto& L : m.layers) pro.insert(L.e_ids);
    pro.erase(0);
    Region r;
    while (r.lo < v.size() && pro.count(v[r.lo].op)) ++r.lo;
    const rad_op last = m.pcn_fin ? m.pcn_fin : m.layers.back().post_f;
    for (size_t i = 0; i < v.size(); ++i)
        if (v[i].op == last) r.hi = i + 1;
    return r;
}

/* Whether `x` interleaves `a` and `b` (x[i + j] is a[i] or b[j] shifted), and if so each x entry's
 * micro-batch (0: a, 1: b). */
bool interleaves(const RadBuilder& bld, const std::vector<RecIssue>& x, const std::vector<RecIssue>& a,
                 const std::vector<RecIssue>& b, int64_t shift, std::vector<int>* tag) {
    const size_t na = a.size(), nb = b.size();
    if (x.size() != na + nb) {
        fprintf(stderr, "    %zu split issues against %zu + %zu lone ones\n", x.size(), na, nb);
        return false;
    }
    /* ok[i][j]: x's first i + j issues interleave a's first i and b's first j. */
    std::vector<std::vector<uint8_t>> ok(na + 1, std::vector<uint8_t>(nb + 1, 0));
    ok[0][0] = 1;
    size_t deepest = 0;
    for (size_t i = 0; i <= na; ++i)
        for (size_t j = 0; j <= nb; ++j) {
            if (!ok[i][j]) continue;
            deepest = std::max(deepest, i + j);
            if (i + j == x.size()) continue;
            const RecIssue& e = x[i + j];
            if (i < na && same_issue(e, a[i])) ok[i + 1][j] = 1;
            if (j < nb && shifted_issue(bld, e, b[j], shift)) ok[i][j + 1] = 1;
        }
    if (!ok[na][nb]) {
        fprintf(stderr, "    no interleaving: stuck at split issue %zu (%s)\n", deepest,
                deepest < x.size() ? op_name(bld, x[deepest].op).c_str() : "(end)");
        auto show = [&](const char* who, const RecIssue& e) {
            fprintf(stderr, "      %s %s n=%lld:", who, op_name(bld, e.op).c_str(), (long long)e.n);
            for (const RadOperand& o : e.opd)
                fprintf(stderr, " [k%d h%u off%lld r%lld c%lld]", o.kind, o.handle, (long long)o.offset,
                        (long long)o.rows, (long long)o.cols);
            fprintf(stderr, "\n");
        };
        if (deepest < x.size()) show("split", x[deepest]);
        for (size_t i = 0; i <= na; ++i) {
            const size_t j = deepest - i;
            if (deepest < i || j > nb || !ok[i][j]) continue;
            if (i < na) show("A", a[i]);
            if (j < nb) show("B", b[j]);
        }
        return false;
    }
    tag->assign(x.size(), -1);
    size_t i = na, j = nb;
    while (i + j > 0) {
        const RecIssue& e = x[i + j - 1];
        if (i > 0 && ok[i - 1][j] && same_issue(e, a[i - 1])) { (*tag)[i + j - 1] = 0; --i; }
        else { (*tag)[i + j - 1] = 1; --j; }
    }
    return true;
}

/* Every buffer B's issues shift, by one amount: a buffer is the step's (at B's rows) or B's own
 * scratch (from row 0), never both. */
bool shifts_consistent(const RadBuilder& bld, const std::vector<RecIssue>& x, const std::vector<RecIssue>& b,
                       const std::vector<int>& tag) {
    std::map<uint32_t, int64_t> delta;
    bool ok = true;
    size_t j = 0;
    for (size_t k = 0; k < x.size(); ++k) {
        if (tag[k] != 1) continue;
        const RecIssue &e = x[k], &y = b[j++];
        for (size_t i = 0; i < e.opd.size(); ++i) {
            if (e.opd[i].kind != RAD_OPK_BUF || e.opd[i].handle == 0) continue;
            const int64_t d = e.opd[i].offset - y.opd[i].offset;
            auto it = delta.find(e.opd[i].handle);
            if (it == delta.end()) delta[e.opd[i].handle] = d;
            else if (it->second != d) {
                fprintf(stderr, "    buffer '%s' at two shifts (%lld, %lld) in micro-batch B (%s)\n",
                        bld.bufs[e.opd[i].handle - 1].name.c_str(), (long long)it->second, (long long)d,
                        op_name(bld, e.op).c_str());
                ok = false;
            }
        }
    }
    return ok;
}

/* Which micro-batch wrote each element of each buffer last: painted segments [lo, hi). */
struct Paint {
    struct Seg { int64_t lo, hi; int who; };
    std::map<uint32_t, std::vector<Seg>> bufs;
    void write(uint32_t h, int64_t lo, int64_t hi, int who) {
        std::vector<Seg>& v = bufs[h];
        std::vector<Seg> out;
        for (const Seg& s : v) {
            if (s.hi <= lo || s.lo >= hi) { out.push_back(s); continue; }
            if (s.lo < lo) out.push_back({ s.lo, lo, s.who });
            if (s.hi > hi) out.push_back({ hi, s.hi, s.who });
        }
        out.push_back({ lo, hi, who });
        v.swap(out);
    }
    /* The writers other than `who` and `also` of any element in [lo, hi). */
    int foreign(uint32_t h, int64_t lo, int64_t hi, int who, int also) const {
        auto it = bufs.find(h);
        if (it == bufs.end()) return -1;
        for (const Seg& s : it->second)
            if (s.hi > lo && s.lo < hi && s.who != who && s.who != also) return s.who;
        return -1;
    }
};

/* An operand's element span in its buffer: offset, rows (the declared extent at 0), columns (the
 * pitch at 0); a column slice spans whole rows but its last. */
void span_of(const RadBuilder& b, const RadOperand& o, int64_t* lo, int64_t* hi) {
    const rad_buf h = (rad_buf)o.handle;
    const int64_t pitch = pitch_of(b, h);
    const int64_t rows = o.rows > 0 ? o.rows : rows_of_buf(b, h);
    const int64_t cols = o.cols > 0 ? o.cols : pitch;
    *lo = o.offset;
    *hi = o.offset + (rows - 1) * pitch + cols;
}

/* Tags: 0 A, 1 B, 2 the prologue, 3 the head. Reports every read of an element the other
 * micro-batch wrote last. */
std::vector<std::string> cross_reads(const RadBuilder& b, const std::vector<RecIssue>& v,
                                     const std::vector<int>& tag, const std::set<rad_buf>& tables = {}) {
    std::vector<std::string> bad;
    Paint p;
    for (size_t k = 0; k < v.size(); ++k) {
        const RecIssue& is = v[k];
        const RecOp* o = b.op_of(is.op);
        if (!o) continue;
        const int who = tag[k];
        std::vector<std::pair<int64_t, int64_t>> wr_spans;
        std::vector<uint32_t> wr_bufs;
        for (const RadOperand& od : is.opd) {
            if (od.kind != RAD_OPK_BUF || od.handle == 0 || od.handle > b.bufs.size()) continue;
            const rad_buf h = (rad_buf)od.handle;
            const bool r = in(o->reads, h), w = in(o->writes, h);
            int64_t lo = 0, hi = 0;
            span_of(b, od, &lo, &hi);
            if ((r || !w) && (who == 0 || who == 1) && !tables.count(h)) {
                const int f = p.foreign(h, lo, hi, who, 2);
                if (f >= 0 && bad.size() < 16)
                    bad.push_back("issue " + std::to_string(k) + " (" + op_name(b, is.op) + ", micro-batch " +
                                  (who ? "B" : "A") + ") reads '" + b.bufs[h - 1].name +
                                  "' where " + (f == 0 ? "A" : f == 1 ? "B" : "the head") + " wrote last");
            }
            if (w || !r) { wr_bufs.push_back(h); wr_spans.push_back({ lo, hi }); }
        }
        for (size_t i = 0; i < wr_bufs.size(); ++i) p.write(wr_bufs[i], wr_spans[i].first, wr_spans[i].second, who);
    }
    return bad;
}

/* The step lint's findings without their issue numbers. */
std::set<std::string> lint_kinds(const RadBuilder& b, const RadCtx* c) {
    std::set<std::string> out;
    for (std::string f : lint_step(b, { c })) {
        const size_t i = f.find("issue ");
        if (i != std::string::npos) {
            size_t j = i + 6;
            while (j < f.size() && std::isdigit((unsigned char)f[j])) ++j;
            f.erase(i, j - i);
        }
        out.insert(f);
    }
    return out;
}
/* The split step's lint finds nothing the batched step's does not (this fake geometry leaves a few
 * planes undeclared and reads a column slice into a GEMM, in every step). */
bool no_new_lint(const RadBuilder& b, const RadCtx* split, const RadCtx* batched) {
    const auto x = lint_kinds(b, split), w = lint_kinds(b, batched);
    bool ok = true;
    for (const std::string& f : x)
        if (!w.count(f)) {
            if (ok) fprintf(stderr, "    lint only in the split step: %s\n", f.c_str());
            ok = false;
        }
    return ok;
}

struct Split {
    std::unique_ptr<Kit> full;
    std::unique_ptr<RadCtx> split, batched;
    std::vector<int> tag;   /* every issue of the split step: 0 A, 1 B, 2 prologue, 3 head */
    bool interleaved = false;
};

/* One rank's split step of `S` sequences against its batched step and the lone steps of A's and
 * B's sequences (checks 1 and 2 of the file header). */
void check_split(Decl& d, int rank, int S, Split& out) {
    deepseek41::Model& m = *d.m;
    out.full = decode(d.b, S);
    REQUIRE(::rad::tbo::step_splits(&out.full->b));
    const int64_t Sa = ::rad::tbo::split_a(S), Sb = S - Sa;
    CHECK_EQ(::rad::tbo::rows_of(S, 0), Sa);
    CHECK_EQ(::rad::tbo::first_row(S, 1), Sa);
    REQUIRE(m.tbo);
    const uint64_t steps0 = m.tbo_steps;
    out.split = run(d.b, out.full->b, rank);
    REQUIRE(out.split->step_fail.empty());
    CHECK_EQ(m.tbo_steps, steps0 + 1);
    m.tbo = false;   /* the same graph, the step unsplit */
    out.batched = run(d.b, out.full->b, rank);
    auto la = lone(*out.full, 0, Sa), lb = lone(*out.full, Sa, Sb);
    auto ca = run(d.b, la->b, rank), cb = run(d.b, lb->b, rank);
    m.tbo = true;
    REQUIRE(out.batched->step_fail.empty());
    REQUIRE(ca->step_fail.empty());
    REQUIRE(cb->step_fail.empty());
    CHECK_EQ(m.tbo_steps, steps0 + 1);

    const auto& X = out.split->issues;
    const auto& W = out.batched->issues;
    const Region rx = loop_region(m, X), rw = loop_region(m, W), ra = loop_region(m, ca->issues),
                 rb = loop_region(m, cb->issues);
    REQUIRE(rx.hi > rx.lo);
    REQUIRE(rw.hi > rw.lo);
    /* The prologue and the head: the batched step's, op for op. */
    REQUIRE_EQ(rx.lo, rw.lo);
    bool same = true;
    for (size_t i = 0; i < rx.lo; ++i) same = same && same_issue(X[i], W[i]);
    CHECK(same);
    REQUIRE_EQ(X.size() - rx.hi, W.size() - rw.hi);
    same = true;
    for (size_t i = 0; i < X.size() - rx.hi; ++i) same = same && same_issue(X[rx.hi + i], W[rw.hi + i]);
    CHECK(same);
    /* The on-die reads: the batched step's (each layer's once, behind B's MoE), the Engram layers'
     * behind each micro-batch's Engram GEMM. */
    std::set<rad_op> rx_w, rw_w;
    for (size_t i = rx.lo; i < rx.hi; ++i) if (weight_only(X[i])) rx_w.insert(X[i].op);
    for (size_t i = rw.lo; i < rw.hi; ++i) if (weight_only(W[i])) rw_w.insert(W[i].op);
    CHECK(rx_w == rw_w);
    /* 1: the layers, an interleaving of the lone steps' (B's at its rows). */
    std::vector<size_t> at;   /* the split step's value issues' indices */
    std::vector<RecIssue> xv;
    for (size_t i = rx.lo; i < rx.hi; ++i)
        if (!weight_only(X[i])) { xv.push_back(X[i]); at.push_back(i); }
    const auto av = value_issues(ca->issues, ra.lo, ra.hi), bv = value_issues(cb->issues, rb.lo, rb.hi);
    std::vector<int> t;
    out.interleaved = interleaves(d.b, xv, av, bv, Sa, &t);
    CHECK(out.interleaved);
    if (!out.interleaved) return;
    CHECK(shifts_consistent(d.b, xv, bv, t));
    /* 2: no micro-batch reads what the other wrote. */
    out.tag.assign(X.size(), 2);
    for (size_t i = rx.hi; i < X.size(); ++i) out.tag[i] = 3;
    for (size_t i = rx.lo; i < rx.hi; ++i) out.tag[i] = 0;   /* on-die reads: no operand to check */
    for (size_t k = 0; k < at.size(); ++k) out.tag[at[k]] = t[k];
    /* The rope tables are by position (row p is rope(p), whichever op writes it: the prologue's
     * table ops for the step's positions, a compressor's for its pooled ones), so a micro-batch may
     * read rows the other wrote. Checked: nothing else writes them. */
    const std::set<rad_buf> tables = { m.cs_a, m.cs_b };
    std::set<rad_op> table_ops = { m.rope_a, m.rope_b_step };
    for (const auto& L : m.layers) table_ops.insert(L.c_tab);
    bool only_tables = true;
    for (const RecIssue& is : X) {
        const RecOp* o = d.b.op_of(is.op);
        if (!o) continue;
        for (rad_buf h : tables)
            if (in(o->writes, h) && !table_ops.count(is.op)) {
                fprintf(stderr, "    %s writes the rope table '%s'\n", op_name(d.b, is.op).c_str(),
                        d.b.bufs[h - 1].name.c_str());
                only_tables = false;
            }
    }
    CHECK(only_tables);
    /* ...and every op writing the compressed table is the step's own table op but for its height. */
    auto same_table = [&](rad_op x, rad_op y) {
        const RecOp *a = d.b.op_of(x), *c = d.b.op_of(y);
        if (!a || !c || a->op != c->op || a->p.size() != c->p.size()) return false;
        for (size_t i = 0; i < a->p.size(); ++i) {
            const RecParam &p = a->p[i], &q = c->p[i];
            if (p.key == "M") continue;
            if (p.key != q.key || p.kind != q.kind || p.ival != q.ival || p.ihi != q.ihi || p.dval != q.dval ||
                p.sval != q.sval)
                return false;
        }
        return true;
    };
    for (const auto& L : m.layers)
        if (L.c_tab) CHECK(same_table(L.c_tab, m.rope_b_step));
    const auto bad = cross_reads(d.b, X, out.tag, tables);
    for (const auto& s : bad) fprintf(stderr, "    %s\n", s.c_str());
    CHECK(bad.empty());
    CHECK(no_new_lint(d.b, out.split.get(), out.batched.get()));
}

/* The k-th issue of op h in micro-batch `who`, or SIZE_MAX. */
size_t nth(const Split& s, rad_op h, int who, int k) {
    const auto& X = s.split->issues;
    for (size_t i = 0; i < X.size(); ++i)
        if (X[i].op == h && s.tag[i] == who && k-- == 0) return i;
    return SIZE_MAX;
}

}  // namespace

TEST(tbo_off_declares_nothing_and_splits_nothing) {
    Env env({});
    auto d = declare(0);
    REQUIRE_EQ(d->rc, (int)RAD_OK);
    CHECK_EQ(d->b.two_batch_calls, 0);
    CHECK(!d->b.two_batch);
    CHECK(!d->m->tbo);
    CHECK(d->err.find("two-batch") == std::string::npos);
    auto k = decode(d->b, 4);
    auto c = run(d->b, k->b, 0);
    REQUIRE(c->step_fail.empty());
    CHECK_EQ(d->m->tbo_steps, (uint64_t)0);
    /* The graph is the switch's: the same ops declared either way. */
    std::vector<std::string> off;
    for (const auto& o : d->b.ops) off.push_back(o.op);
    env.set({ { "RADIANCE_EXPERIMENTAL_SCHED_TBO", "1" } });
    auto e = declare(0);
    REQUIRE_EQ(e->rc, (int)RAD_OK);
    std::vector<std::string> on;
    for (const auto& o : e->b.ops) on.push_back(o.op);
    CHECK(on == off);
    CHECK_EQ(e->b.bufs.size(), d->b.bufs.size());
}

TEST(tbo_declares_on_every_rank_and_says_so_once) {
    Env env({ { "RADIANCE_EXPERIMENTAL_SCHED_TBO", "1" } });
    for (int rank = 0; rank < kWorld; ++rank) {
        auto d = declare(rank);
        REQUIRE_EQ(d->rc, (int)RAD_OK);
        CHECK_EQ(d->b.two_batch_calls, 1);
        CHECK(d->b.two_batch);
        CHECK(d->m->tbo);
        const bool line = d->err.find("two-batch overlap (RADIANCE_EXPERIMENTAL_SCHED_TBO=1)") != std::string::npos;
        CHECK_EQ(line, rank == 0);
    }
    /* "1" only. */
    env.set({ { "RADIANCE_EXPERIMENTAL_SCHED_TBO", "2" } });
    auto d = declare(0);
    CHECK(!d->m->tbo);
    CHECK_EQ(d->b.two_batch_calls, 0);
}

/* Which steps split (tbo::step_splits, which the file stager and the routing witness share): a
 * pure decode step of two or more sequences, one row each, nothing else; the rest run as before. */
TEST(tbo_splits_only_pure_decode_steps_of_two_or_more) {
    Env env({ { "RADIANCE_EXPERIMENTAL_SCHED_TBO", "1" } });
    auto d = declare(0);
    REQUIRE(d->m->tbo);
    auto one = decode(d->b, 1);
    auto chunk = kit(d->b, {}, { 16 }, 64);
    auto mixed = kit(d->b, { 1, 1 }, { 8 }, 64);
    auto verify = kit(d->b, { 3, 3 }, {});
    CHECK(!::rad::tbo::step_splits(&one->b));
    CHECK(!::rad::tbo::step_splits(&chunk->b));
    CHECK(!::rad::tbo::step_splits(&mixed->b));
    CHECK(!::rad::tbo::step_splits(&verify->b));
    for (int S : { 2, 3, 8, 16 }) CHECK(::rad::tbo::step_splits(&decode(d->b, S)->b));
    auto cut = decode(d->b, 4);
    cut->b.consumer_full = 25;
    CHECK(!::rad::tbo::step_splits(&cut->b));
    auto draft = decode(d->b, 4);
    draft->b.draft_pass = 1;
    CHECK(!::rad::tbo::step_splits(&draft->b));
    CHECK_EQ(::rad::tbo::split_a(2), (int64_t)1);
    CHECK_EQ(::rad::tbo::split_a(5), (int64_t)3);
    CHECK_EQ(::rad::tbo::rows_of(5, 1), (int64_t)2);
    /* A step that does not split issues exactly what it issues with the switch off. */
    for (const Kit* k : { one.get(), chunk.get(), mixed.get() }) {
        auto on = run(d->b, k->b, 0);
        d->m->tbo = false;
        auto off = run(d->b, k->b, 0);
        d->m->tbo = true;
        REQUIRE_EQ(on->issues.size(), off->issues.size());
        bool same = true;
        for (size_t i = 0; i < on->issues.size(); ++i) same = same && same_issue(on->issues[i], off->issues[i]);
        CHECK(same);
    }
    CHECK_EQ(d->m->tbo_steps, (uint64_t)0);
}

/* Checks 1 and 2 at 2, 4, 5 and 8 sequences on every rank, plain and with the best configuration's
 * architecture switches (the Engram rows split over the ranks, on-die reads with a budget and at
 * the Engram layers, FP8 dense linears, bf16 routers). */
TEST(tbo_each_micro_batch_is_a_lone_step_of_its_rows_and_reads_only_its_own) {
    for (int cfg = 0; cfg < 2; ++cfg) {
        Env env({ { "RADIANCE_EXPERIMENTAL_SCHED_TBO", "1" } });
        if (cfg == 1)
            env.set({ { "RADIANCE_EXPERIMENTAL_SCHED_TBO", "1" },
                      { "RADIANCE_EXPERIMENTAL_DS41_ENGRAM_SPLIT", "1" },
                      { "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ATTN", "1" },
                      { "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_MIB", "56" },
                      { "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ENGRAM", "1" },
                      { "RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8", "1" },
                      { "RADIANCE_EXPERIMENTAL_DS41_ROUTER_BF16", "1" } });
        for (int rank = 0; rank < kWorld; ++rank) {
            auto d = declare(rank);
            REQUIRE_EQ(d->rc, (int)RAD_OK);
            REQUIRE(d->m->tbo);
            for (int S : { 2, 4, 5, 8 }) {
                fprintf(stderr, "  config %d rank %d, %d sequences\n", cfg, rank, S);
                Split s;
                check_split(*d, rank, S, s);
                if (!s.interleaved) continue;
                /* 3: the overlap's order, layer by layer. */
                const auto& Ls = d->m->layers;
                bool order = true;
                for (size_t l = 0; l < Ls.size(); ++l) {
                    const rad_op gu = Ls[l].moe.op_gu, attn = Ls[l].attn;
                    const rad_op post = l + 1 < Ls.size() && Ls[l + 1].pcn_a ? Ls[l + 1].pcn_a
                                        : l + 1 == Ls.size() && d->m->pcn_fin ? d->m->pcn_fin
                                                                              : Ls[l].post_f;
                    /* With a fused post the next layer's pcn_a is issued once a micro-batch per layer;
                     * occurrence l (pcn_a of layer l + 1 is that layer's only one). */
                    const size_t gu_a = nth(s, gu, 0, 0), gu_b = nth(s, gu, 1, 0), at_b = nth(s, attn, 1, 0),
                                 po_a = nth(s, post, 0, 0), po_b = nth(s, post, 1, 0);
                    const size_t at_a1 = l + 1 < Ls.size() ? nth(s, Ls[l + 1].attn, 0, 0) : SIZE_MAX - 1;
                    const bool ok = gu_a < at_b && at_b < po_a && po_a < gu_b && gu_b < po_b &&
                                    (l + 1 == Ls.size() || (gu_b < at_a1 && at_a1 < po_b));
                    if (!ok && order)
                        fprintf(stderr, "    layer %zu out of order: gu A %zu, attn B %zu, post A %zu, gu B %zu, "
                                        "attn A' %zu, post B %zu\n", l, gu_a, at_b, po_a, gu_b, at_a1, po_b);
                    order = order && ok;
                }
                CHECK(order);
            }
        }
    }
}

/* The buffer plan: under the declaration no transient shares bytes (the declared op order no longer
 * bounds a transient's live range); undeclared, the split step's order would read a transient after
 * a write of one planned into its bytes, which the lint shows. */
TEST(tbo_buffer_plan_keeps_every_transient_its_own) {
    Env env({ { "RADIANCE_EXPERIMENTAL_SCHED_TBO", "1" } });
    auto d = declare(0);
    REQUIRE(d->m->tbo);
    auto k = decode(d->b, 4);
    auto c = run(d->b, k->b, 0);
    REQUIRE(c->step_fail.empty());
    d->m->tbo = false;
    auto w = run(d->b, k->b, 0);
    d->m->tbo = true;
    CHECK(no_new_lint(d->b, c.get(), w.get()));
    for (const Live& l : liveness(d->b)) CHECK(!l.planned);
    /* With the declared order's plan the split order reads transients after writes of others planned
     * into their bytes: the plan rule above is what keeps it exact. */
    d->b.two_batch = false;
    const auto x = lint_kinds(d->b, c.get()), y = lint_kinds(d->b, w.get());
    size_t alias = 0;
    for (const std::string& f : x) alias += !y.count(f) && f.find("which may share its bytes") != std::string::npos;
    fprintf(stderr, "  aliasing reads the split order would have under the declared order's plan: %zu\n", alias);
    d->b.two_batch = true;
}

RAD_TEST_MAIN()
