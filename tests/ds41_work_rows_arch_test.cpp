/* ds41_work_rows_arch_test.cpp -- deepseek41's compressor tail at the rows a step needs (work_rows()),
 * on the recording builder (arch_fakes.h) with DeepSeek-V4.1-Flash's own geometry at TP4, no card or
 * model file.
 *
 * A source of ratio 2 (layers 2, 8 and 14 of the release) issued qsa_work's launch and the ops after it
 * (softmax_pool, the latent's rmsnorm and rope table, the index keys' gemm_nt, their norm, rope and
 * round trip, the latent's rope and round trip, both entry stores) at m.work rows, max_tok + max_seqs,
 * whatever the step: a one-row decode step ran them on 2,049 rows a layer at 2,048-row chunks and on
 * 1,025 at 1,024. Beside the norms' pinned width (LCP or EXACT_TALL) a step issues them at
 * S + (T - S) / r rows. Checked:
 *   - the fit is the pins, read as libr4d reads them; without them a step issues m.work's rows as before;
 *   - the bound holds every block a step can complete (each sequence length and start position at
 *     ratios 1 to 128) and is reached;
 *   - a decode step at 2,048-row chunks runs no op on more rows than its sequences need (without the
 *     pins the ops that ran on max_tok's rows are listed);
 *   - the same step at m.work and at the step's bound issues the same ops on the same operands but for
 *     those rows, and, run op by op on host forms of the compressor's ops from the same memory (garbage
 *     in every row past a step's blocks), writes the same bytes: every cache entry, the tail state, the
 *     angle table, nc, and every work row up to the step's block count; at a bound one short a block
 *     goes unstored, which the comparison sees;
 *   - the rows follow the step's counts, which the recorded passes' key holds (Ctx::pass_key): steps
 *     with the same counts at other positions issue the same shapes;
 *   - a two-batch-overlap split issues each micro-batch's at its own sequences.
 * The kernels' half (a row's bits at any launch height under the pins) is libr4d's: test_ds41tall
 * holds the norms and gemm_nt's tiled row to it, and the core's LCP admission refuses a band that is
 * not one kernel or a known pair. */
#define RAD_ARCH_NO_EXPORTS 1

#include "arch_fakes.h"

#include "../arch/deepseek41/deepseek41.cpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace {

constexpr int kWorld = 4;

const char* const kVars[] = { "RADIANCE_EXPERIMENTAL_DS41_LCP",
                              "RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL",
                              "RADIANCE_EXPERIMENTAL_DS41_CONSUMER_SKIP",
                              "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ATTN",
                              "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_MIB",
                              "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ENGRAM",
                              "RADIANCE_EXPERIMENTAL_DS41_ENGRAM_SPLIT",
                              "RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8",
                              "RADIANCE_EXPERIMENTAL_DS41_ROUTER_BF16",
                              "RADIANCE_EXPERIMENTAL_DS41_KV_REPLICATED",
                              "RADIANCE_EXPERIMENTAL_SCHED_TBO" };
struct Env {
    Env(std::initializer_list<std::pair<const char*, const char*>> kv) {
        for (const char* n : kVars) unsetenv(n);
        for (const auto& p : kv) setenv(p.first, p.second, 1);
    }
    ~Env() {
        for (const char* n : kVars) unsetenv(n);
    }
};
/* The window's switches the plugin reads (the adopted arm's Cs, Pa / Pm56 / Pe, Es and F8). */
#define WINDOW_SWITCHES                                                                             \
    { "RADIANCE_EXPERIMENTAL_DS41_CONSUMER_SKIP", "1" }, { "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ATTN", "1" }, \
        { "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_MIB", "56" }, { "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ENGRAM", "1" }, \
        { "RADIANCE_EXPERIMENTAL_DS41_ENGRAM_SPLIT", "1" }, { "RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8", "1" }

/* The release's geometry as llama.cpp's converter states it (ds41_shape_probe_arch_test's). */
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
        .set("deepseek41.ple.head_vocab_sizes", "1000,1000")
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
/* deepseek41's declare for `rank` of four, under the current environment. */
std::unique_ptr<Decl> declare(int rank, int64_t max_tok, int64_t max_seqs, int64_t max_ctx = 4096) {
    auto d = std::make_unique<Decl>();
    d->b.kv_block_of = block_of;
    RadBuildCtx ctx = build_ctx(rank, kWorld);
    ctx.max_tok = max_tok;
    ctx.max_seqs = max_seqs;
    ctx.max_ctx = max_ctx;
    MetaKV meta = flash_meta();
    d->err = stderr_of([&] { d->rc = deepseek41::declare(&d->b, meta.meta(), &ctx); });
    d->m = deepseek41::models[rank].get();
    return d;
}

/* A step as the core builds one (arch_fakes.h's kit()), each sequence `rows` rows from position `at`,
 * the decode sequences first. Every group's table gives sequence s the blocks [s * pitch, s * pitch +
 * pitch), as many as its compressed blocks need; its state slot is s. */
struct Seq { int rows; int32_t at; };
constexpr int64_t kPitch = 1024;
std::unique_ptr<Kit> step_of(const RadBuilder& bld, const std::vector<Seq>& dq, const std::vector<Seq>& pq) {
    auto k = std::make_unique<Kit>();
    std::vector<Seq> all(dq);
    all.insert(all.end(), pq.begin(), pq.end());
    const int64_t nd = (int64_t)dq.size(), ns = (int64_t)all.size();
    int64_t T = 0, max_q = 0, max_ctx = 0, td = 0, mqd = 0, mqp = 0;
    k->cu.push_back(0);
    for (int64_t s = 0; s < ns; ++s) {
        const bool dec = s < nd;
        const Seq& q = all[(size_t)s];
        for (int i = 0; i < q.rows; ++i) {
            k->tok.push_back((int32_t)(3 + (T + i) % 97));
            k->pos.push_back(q.at + i);
            if (dec || i + 1 == q.rows) k->out.push_back((int32_t)(T + i));
        }
        T += q.rows;
        k->cu.push_back((int32_t)T);
        k->ctxl.push_back(q.at);
        k->qlen.push_back(q.rows);
        k->nacc.push_back(1);
        k->dout.push_back((int32_t)T - 1);
        max_q = std::max<int64_t>(max_q, q.rows);
        max_ctx = std::max<int64_t>(max_ctx, q.at + q.rows);
        if (dec) { td += q.rows; mqd = std::max<int64_t>(mqd, q.rows); }
        else     mqp = std::max<int64_t>(mqp, q.rows);
    }
    k->tok.resize((size_t)T + 64, 5);
    for (int64_t i = 0; i < T; ++i) k->slot.push_back((int32_t)i);
    for (int64_t i = 0; i < ns * kPitch; ++i) k->table.push_back((int32_t)i);
    for (int64_t s = 0; s < ns; ++s) {
        k->used.push_back(k->ctxl[(size_t)s] + k->qlen[(size_t)s]);
        k->sidx.push_back((int32_t)s);
    }
    for (size_t g = 0; g < bld.kv_groups.size(); ++g) {
        RadKVGroupBatch gb{};
        gb.group = (rad_kvgroup)(g + 1);
        gb.slot_mapping = k->slot.data();
        gb.block_table = k->table.data();
        gb.block_table_pitch = kPitch;
        gb.seqused = k->used.data();
        gb.state_index = k->sidx.data();
        gb.state_index_pitch = 1;
        gb.block_size = bld.kv_block;
        gb.max_blocks = kPitch;
        k->kv.push_back(gb);
    }
    RadBatch& b = k->b;
    b.phase = pq.empty() ? RAD_PHASE_DECODE : dq.empty() ? RAD_PHASE_PREFILL : RAD_PHASE_MIXED;
    b.n_tok = T;
    b.n_seq = ns;
    b.token_ids = k->tok.data();
    b.positions = k->pos.data();
    b.cu_seqlens = k->cu.data();
    b.n_out = (int64_t)k->out.size();
    b.out_ids = k->out.data();
    b.q_lens = k->qlen.data();
    b.ctx_lens = k->ctxl.data();
    b.max_q_len = (int32_t)max_q;
    b.max_ctx_len = (int32_t)max_ctx;
    b.n_kv_groups = (int)k->kv.size();
    b.kv = k->kv.data();
    b.n_spec = mqd > 1 ? (int)(mqd - 1) : 0;
    b.num_accepted = k->nacc.data();
    b.n_seq_decode = nd;
    b.n_tok_decode = td;
    b.max_q_len_decode = (int32_t)mqd;
    b.max_q_len_prefill = (int32_t)mqp;
    return k;
}

/* The blocks a step completes at ratio r, as qsa_work counts them: those whose last position is one of
 * a sequence's rows. */
int64_t blocks_of(const RadBatch& b, int64_t r) {
    int64_t n = 0;
    for (int64_t s = 0; s < b.n_seq; ++s) {
        const int32_t t0 = b.cu_seqlens[s], t1 = b.cu_seqlens[s + 1];
        if (t1 > t0) n += (b.positions[t1 - 1] + 1) / r - b.positions[t0] / r;
    }
    return n;
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

/* The ops a source of ratio above 1 issues on its work rows (work_rows()), and the layer of each. */
std::map<rad_op, int> work_ops(const deepseek41::Model& m) {
    std::map<rad_op, int> s;
    for (const auto& L : m.layers)
        if (L.src && L.r > 1)
            for (rad_op h : { L.c_work, L.c_pool, L.c_norm, L.c_tab, L.k_gemm, L.k_f, L.k_norm, L.k_rope, L.k_q,
                              L.l_f, L.l_rope, L.l_q, L.k_st, L.l_st })
                if (h) s[h] = L.l;
    return s;
}

/* The work rows a work op runs on: qsa_work's launch (its page), every other's count. */
int64_t work_rows_of(const RadBuilder& b, const RecIssue& is) {
    return op_name(b, is.op) == "qsa_work" ? is.opd[8].rows : is.n;
}

/* The rows an issue reaches: its largest operand of a step-tall buffer (a transient device buffer of at
 * least max_tok rows, not an expert's or a sequence's array), the declared extent where it names none. */
int64_t issue_rows(const RadBuilder& b, const RecIssue& is, int64_t max_tok) {
    int64_t rows = 0;
    for (const RadOperand& o : is.opd) {
        if (o.kind != RAD_OPK_BUF || o.handle == 0 || o.handle > b.bufs.size()) continue;
        const RadBufDecl& d = b.bufs[o.handle - 1].d;
        if (d.kind != RAD_BUF_TRANSIENT || d.domain == RAD_DOMAIN_HOST || d.rank == 0 || d.shape[0] < max_tok)
            continue;
        rows = std::max(rows, o.rows > 0 ? o.rows : d.shape[0]);
    }
    return rows;
}

bool same_operand(const RadOperand& x, const RadOperand& y) {
    return x.kind == y.kind && x.dtype == y.dtype && x.handle == y.handle && x.raw == y.raw &&
           x.offset == y.offset && x.rows == y.rows && x.cols == y.cols;
}
/* Whether `x` is `y` (the step at m.work) but for its work rows: the same op, lane and operands, a
 * buffer operand's rows and the issue's count at most y's. */
bool same_but_rows(const RecIssue& x, const RecIssue& y, bool work) {
    if (x.op != y.op || x.lane != y.lane || x.opd.size() != y.opd.size()) return false;
    if (!work) {
        if (x.n != y.n) return false;
        for (size_t i = 0; i < x.opd.size(); ++i)
            if (!same_operand(x.opd[i], y.opd[i])) return false;
        return true;
    }
    if (x.n > y.n) return false;
    for (size_t i = 0; i < x.opd.size(); ++i) {
        const RadOperand &p = x.opd[i], &q = y.opd[i];
        if (p.kind != q.kind || p.dtype != q.dtype || p.handle != q.handle || p.raw != q.raw ||
            p.offset != q.offset || p.cols != q.cols || p.rows > q.rows)
            return false;
    }
    return true;
}

/* The step lint's findings without their issue numbers. */
std::set<std::string> lint_kinds(const RadBuilder& b, const RadCtx* c) {
    std::set<std::string> out;
    for (std::string f : lint_step(b, { c })) {
        const size_t at = f.find("issue ");
        if (at != std::string::npos) {
            size_t e = at + 6;
            while (e < f.size() && std::isdigit((unsigned char)f[e])) ++e;
            f.erase(at, e - at);
        }
        out.insert(f);
    }
    return out;
}

/* ---- the compressor's ops on the host --------------------------------------------------------------
 * Host forms of the ops a source of ratio above 1 issues on its work rows, and of the tail store
 * between them: qsa_work's work list as r4d_qsa_work_bf16.hip walks it, every other op a row from that
 * row's operands alone, as libref computes them in kind (the round trips a block-scaled quantise and
 * dequantise). What they do to a row is beside the point; what is checked is that every real row reads
 * and writes the same bytes at either bound. Memory is made at first use, every buffer, weight and
 * cache filled from its handle (so two runs start alike, garbage in every row), the angle table with
 * each position's own angles (as every writer of a row writes it). */
uint64_t mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}
float fbits(uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
float bf(uint16_t h) { return fbits((uint32_t)h << 16); }
uint16_t to_bf(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
float unit(uint64_t s) { return (float)((double)(mix64(s) >> 40) / (double)(1 << 24) * 4.0 - 2.0); }

void fill(std::vector<uint8_t>& v, uint32_t dtype, uint64_t seed) {
    if (dtype == RAD_F32)
        for (size_t i = 0; i < v.size() / 4; ++i) ((float*)v.data())[i] = unit(seed * 7919 + i);
    else if (dtype == RAD_BF16)
        for (size_t i = 0; i < v.size() / 2; ++i) ((uint16_t*)v.data())[i] = to_bf(unit(seed * 7919 + i));
    else if (dtype == RAD_I32)
        for (size_t i = 0; i < v.size() / 4; ++i) ((int32_t*)v.data())[i] = (int32_t)(mix64(seed * 7919 + i) % 5000) - 100;
    else
        for (size_t i = 0; i < v.size(); ++i) v[i] = (uint8_t)mix64(seed * 7919 + i);
}

constexpr double kCompTheta = 160000.0;
/* Position p's angles (cos then sin, rot / 2 each). */
void angles(float* row, int64_t p, int64_t rot) {
    const int64_t half = rot / 2;
    for (int64_t i = 0; i < half; ++i) {
        const double a = (double)p * std::pow(kCompTheta, -2.0 * (double)i / (double)rot);
        row[i] = (float)std::cos(a);
        row[half + i] = (float)std::sin(a);
    }
}

struct Mem {
    const RadBuilder* b = nullptr;
    const deepseek41::Model* m = nullptr;
    int64_t blocks = 0, slots = 0;   /* a cache region's blocks; a tail's slots */
    std::map<uint32_t, std::vector<uint8_t>> buf, wt;
    std::map<std::pair<uint32_t, int64_t>, std::vector<uint8_t>> kv;
};

/* [rows, cols] elements of one dtype, rows `pitch` apart. */
struct View {
    uint8_t* p = nullptr;
    int64_t rows = 0, cols = 0, pitch = 0;
    template <class T> T& at(int64_t r, int64_t c) const { return ((T*)p)[r * pitch + c]; }
};

std::vector<uint8_t>& buffer(Mem& M, uint32_t h) {
    auto it = M.buf.find(h);
    if (it != M.buf.end()) return it->second;
    const RadBufDecl& d = M.b->bufs[h - 1].d;
    int64_t n = 1;
    for (uint32_t i = 0; i < d.rank; ++i) n *= d.shape[i];
    std::vector<uint8_t>& v = M.buf[h];
    v.assign((size_t)rad_dtype_bytes(d.dtype, n), 0);
    if ((rad_buf)h == M.m->cs_b) {
        const int64_t rot = d.shape[1];
        for (int64_t p = 0; p < d.shape[0]; ++p) angles((float*)v.data() + p * rot, p, rot);
    } else {
        fill(v, d.dtype, 1000 + h);
    }
    return v;
}

View view(Mem& M, const RadOperand& o) {
    View v;
    if (o.kind == RAD_OPK_BUF) {
        const RadBufDecl& d = M.b->bufs[o.handle - 1].d;
        int64_t width = 1;
        for (uint32_t i = 1; i < d.rank; ++i) width *= d.shape[i];
        v.p = buffer(M, o.handle).data() + rad_dtype_bytes(d.dtype, o.offset);
        v.pitch = width;
        v.rows = o.rows > 0 ? o.rows : d.shape[0];
        v.cols = o.cols > 0 ? o.cols : width;
    } else if (o.kind == RAD_OPK_RAW) {
        v.p = (uint8_t*)o.raw;
        v.rows = o.rows;
        v.cols = o.cols > 0 ? o.cols : 1;
        v.pitch = v.cols;
    } else if (o.kind == RAD_OPK_WEIGHT) {
        const RadWeightDecl& d = M.b->weights[o.handle - 1].d;
        int64_t width = 1, n = 1;
        for (uint32_t i = 0; i < d.rank; ++i) n *= d.shape[i];
        for (uint32_t i = 1; i < d.rank; ++i) width *= d.shape[i];
        auto it = M.wt.find(o.handle);
        if (it == M.wt.end()) {
            it = M.wt.emplace(o.handle, std::vector<uint8_t>((size_t)rad_dtype_bytes(d.dtype, n))).first;
            fill(it->second, d.dtype, 500000 + o.handle);
        }
        v.p = it->second.data();
        v.rows = d.rank ? d.shape[0] : 1;
        v.cols = width;
        v.pitch = width;
    }
    return v;
}

/* A cache operand's region for its layer: `rows` entries of `width` elements of `dtype`. */
View region(Mem& M, const RadOperand& o, int64_t rows, int64_t width, uint32_t dtype) {
    const auto key = std::make_pair(o.handle, o.offset);
    auto it = M.kv.find(key);
    if (it == M.kv.end()) {
        it = M.kv.emplace(key, std::vector<uint8_t>((size_t)rad_dtype_bytes(dtype, rows * width))).first;
        fill(it->second, dtype, 900000 + o.handle * 64 + (uint64_t)o.offset);
    }
    View v;
    v.p = it->second.data();
    v.rows = rows;
    v.cols = width;
    v.pitch = width;
    return v;
}

void rope_row(uint16_t* row, int64_t heads, int64_t hd, int64_t off, int64_t rot, const float* cs) {
    const int64_t half = rot / 2;
    for (int64_t h = 0; h < heads; ++h)
        for (int64_t i = 0; i < half; ++i) {
            uint16_t* a = row + h * hd + off + 2 * i;
            const float x = bf(a[0]), y = bf(a[1]), c = cs[i], s = cs[half + i];
            a[0] = to_bf(x * c - y * s);
            a[1] = to_bf(x * s + y * c);
        }
}
void roundtrip_row(uint16_t* row, int64_t n, int64_t block) {
    for (int64_t b0 = 0; b0 + block <= n; b0 += block) {
        float amax = 0.0f;
        for (int64_t i = b0; i < b0 + block; ++i) amax = std::max(amax, std::fabs(bf(row[i])));
        const float s = amax > 0.0f ? std::ldexp(1.0f, (int)std::ceil(std::log2(amax / 6.0f))) : 1.0f;
        for (int64_t i = b0; i < b0 + block; ++i)
            row[i] = to_bf(std::clamp(std::nearbyint(bf(row[i]) / s * 2.0f) / 2.0f, -6.0f, 6.0f) * s);
    }
}
void norm_row(const uint16_t* x, const float* w, uint16_t* y, int64_t n, double eps) {
    float ss = 0.0f;
    for (int64_t i = 0; i < n; ++i) ss += bf(x[i]) * bf(x[i]);
    const float s = 1.0f / std::sqrt(ss / (float)n + (float)eps);
    for (int64_t i = 0; i < n; ++i) y[i] = to_bf(bf(x[i]) * s * w[i]);
}

/* Runs one issue of the compressor's ops on M; false for an op it has no form of. */
bool exec(Mem& M, const RecIssue& is) {
    const RecOp& op = *M.b->op_of(is.op);
    const auto& d = is.opd;
    const auto P = [&](const char* k) { return (int64_t)pint(&op, k); };
    if (op.op == "qsa_work" || op.op == "qsa_tail_store") {
        const int64_t n = P("n"), r = P("ratio"), ring = P("ring");
        const View k = view(M, d[0]), pos = view(M, d[1]), cu = view(M, d[2]), sidx = view(M, d[4]);
        const View st = region(M, d[3], M.slots * ring, n, RAD_F32);
        const int64_t seqs = cu.rows - 1;
        const auto P0 = [&](int64_t t) { return pos.at<int32_t>(t, 0); };
        if (op.op == "qsa_tail_store") {   /* the last `ring` rows of each sequence */
            for (int64_t s = 0; s < seqs; ++s) {
                const int32_t t0 = cu.at<int32_t>(s, 0), t1 = cu.at<int32_t>(s + 1, 0), si = sidx.at<int32_t>(s, 0);
                if (si < 0) continue;
                for (int64_t t = std::max<int64_t>(t0, t1 - ring); t < t1; ++t)
                    for (int64_t c = 0; c < n; ++c) st.at<float>(si * ring + P0(t) % ring, c) = k.at<float>(t, c);
            }
            return true;
        }
        const View bt = view(M, d[5]), stage = view(M, d[7]), page = view(M, d[8]), bpos = view(M, d[9]),
                   nc = view(M, d[10]);
        const int64_t mw = page.rows;
        if (mw < seqs || stage.rows < mw * r) return false;
        for (int64_t w = 0; w < mw; ++w) {
            if (w < seqs) {
                const int32_t t0 = cu.at<int32_t>(w, 0), t1 = cu.at<int32_t>(w + 1, 0);
                nc.at<int32_t>(w, 0) = t1 > t0 ? (P0(t1 - 1) + 1) / (int32_t)r : 0;
            }
            int64_t s = -1, j = 0, run = 0;
            for (int64_t i = 0; i < seqs; ++i) {
                const int32_t t0 = cu.at<int32_t>(i, 0), t1 = cu.at<int32_t>(i + 1, 0);
                if (t1 <= t0) continue;
                const int64_t lo = P0(t0) / r, hi = (P0(t1 - 1) + 1) / r - 1, cnt = hi >= lo ? hi - lo + 1 : 0;
                if (w < run + cnt) { s = i; j = w - run + lo; break; }
                run += cnt;
            }
            if (s < 0) {
                page.at<int32_t>(w, 0) = -1;
                bpos.at<int32_t>(w, 0) = 0;
                continue;
            }
            const int32_t t0 = cu.at<int32_t>(s, 0), p0 = P0(t0), si = sidx.at<int32_t>(s, 0);
            page.at<int32_t>(w, 0) = si < 0 ? -1 : bt.at<int32_t>(s, j);
            bpos.at<int32_t>(w, 0) = (int32_t)(j * r);
            if (si < 0) continue;
            for (int64_t i = 0; i < r; ++i) {
                const int64_t q = j * r + i;
                for (int64_t c = 0; c < n; ++c)
                    stage.at<float>(w * r + i, c) =
                        q >= p0 ? k.at<float>(t0 + (q - p0), c) : st.at<float>(si * ring + q % ring, c);
            }
        }
        return true;
    }
    if (op.op == "softmax_pool") {
        const int64_t n = P("n"), r = P("ratio");
        const View stage = view(M, d[0]), out = view(M, d[1]), page = view(M, d[2]);
        for (int64_t w = 0; w < out.rows; ++w) {
            if (page.at<int32_t>(w, 0) < 0) continue;
            for (int64_t c = 0; c < n; ++c) {
                float mx = -INFINITY, den = 0.0f, acc = 0.0f;
                for (int64_t i = 0; i < r; ++i) mx = std::max(mx, stage.at<float>(w * r + i, n + c));
                for (int64_t i = 0; i < r; ++i) den += std::exp(stage.at<float>(w * r + i, n + c) - mx);
                for (int64_t i = 0; i < r; ++i)
                    acc += std::exp(stage.at<float>(w * r + i, n + c) - mx) / den * stage.at<float>(w * r + i, c);
                out.at<uint16_t>(w, c) = to_bf(acc);
            }
        }
        return true;
    }
    if (op.op == "rmsnorm") {
        const View x = view(M, d[0]), w = view(M, d[1]), y = view(M, d[2]);
        for (int64_t i = 0; i < y.rows; ++i)
            norm_row(&x.at<uint16_t>(i, 0), &w.at<float>(0, 0), &y.at<uint16_t>(i, 0), P("n"), pflt(&op, "eps"));
        return true;
    }
    if (op.op == "rope_table") {
        const View pos = view(M, d[0]), cs = view(M, d[1]);
        for (int64_t i = 0; i < pos.rows; ++i) {
            const int32_t p = pos.at<int32_t>(i, 0);
            if (p >= 0 && p < cs.rows) angles(&cs.at<float>(p, 0), p, P("rot"));
        }
        return true;
    }
    if (op.op == "gemm_nt") {
        const View a = view(M, d[0]), w = view(M, d[1]), y = view(M, d[2]);
        const int64_t N = P("N"), K = P("K");
        for (int64_t i = 0; i < y.rows; ++i)
            for (int64_t j = 0; j < N; ++j) {
                float s = 0.0f;
                for (int64_t k = 0; k < K; ++k) s += bf(a.at<uint16_t>(i, k)) * bf(w.at<uint16_t>(j, k));
                y.at<uint16_t>(i, j) = to_bf(s);
            }
        return true;
    }
    if (op.op == "rmsnorm_rope_roundtrip" || op.op == "rope_rows_roundtrip") {
        const bool norm = op.op == "rmsnorm_rope_roundtrip";
        const View x = view(M, d[0]), y = norm ? view(M, d[2]) : x, cs = view(M, d[norm ? 3 : 1]),
                   pos = view(M, d[norm ? 4 : 2]);
        const int64_t heads = P("n_head"), hd = P("head_dim");
        for (int64_t i = 0; i < y.rows; ++i) {
            uint16_t* row = &y.at<uint16_t>(i, 0);
            if (norm) norm_row(&x.at<uint16_t>(i, 0), &view(M, d[1]).at<float>(0, 0), row, P("n"), pflt(&op, "eps"));
            const int32_t p = pos.at<int32_t>(i, 0);
            if (p >= 0 && p < cs.rows) rope_row(row, heads, hd, P("off"), P("rot"), &cs.at<float>(p, 0));
            roundtrip_row(row, heads * hd, P("block"));
        }
        return true;
    }
    if (op.op == "entry_store") {
        const int64_t n = P("n");
        const View x = view(M, d[0]), page = view(M, d[1]), cache = region(M, d[2], M.blocks, n, RAD_BF16);
        for (int64_t w = 0; w < x.rows; ++w) {
            const int32_t pg = page.at<int32_t>(w, 0);
            if (pg < 0 || pg >= cache.rows) continue;
            for (int64_t c = 0; c < n; ++c) cache.at<uint16_t>(pg, c) = x.at<uint16_t>(w, c);
        }
        return true;
    }
    return false;
}

/* The compressor's issues of a step (every source of ratio above 1: its work ops and the tail store) run
 * on fresh memory; false when one has no host form. */
bool replay(Mem& M, const RadCtx& c) {
    std::set<rad_op> ops;
    for (const auto& [h, l] : work_ops(*M.m)) ops.insert(h);
    for (const auto& L : M.m->layers)
        if (L.src && L.r > 1 && L.c_tail) ops.insert(L.c_tail);
    for (const RecIssue& is : c.issues)
        if (ops.count(is.op) && !exec(M, is)) {
            std::fprintf(stderr, "    no host form of %s\n", op_name(*M.b, is.op).c_str());
            return false;
        }
    return true;
}

/* Where two runs' memory differs (empty: nowhere): every cache region, the angle table, nc and every
 * buffer in full, but the work rows (stage, page, bpos, pooled, latent, index keys) only up to the
 * step's `count` blocks, past which they are padding. */
std::string differ(const Mem& x, const Mem& y, int64_t count, int64_t r) {
    const deepseek41::Model& m = *x.m;
    std::map<uint32_t, int64_t> padded = { { m.stage, count * r }, { m.page, count }, { m.bpos, count },
                                            { m.pooled, count }, { m.lat, count }, { m.ik, count },
                                            { m.ikn, count } };
    if (x.kv.size() != y.kv.size() || x.buf.size() != y.buf.size()) return "different regions touched";
    for (const auto& [k, v] : x.kv) {
        const auto it = y.kv.find(k);
        if (it == y.kv.end() || it->second != v)
            return "cache group " + std::to_string(k.first) + " layer " + std::to_string(k.second);
    }
    for (const auto& [h, v] : x.buf) {
        const auto it = y.buf.find(h);
        if (it == y.buf.end()) return "buffer " + x.b->bufs[h - 1].name + " touched once";
        size_t bytes = v.size();
        const auto pd = padded.find(h);
        if (pd != padded.end()) {
            const RadBufDecl& d = x.b->bufs[h - 1].d;
            int64_t width = 1;
            for (uint32_t i = 1; i < d.rank; ++i) width *= d.shape[i];
            bytes = std::min(bytes, (size_t)rad_dtype_bytes(d.dtype, pd->second * width));
        }
        if (std::memcmp(v.data(), it->second.data(), bytes)) return "buffer " + x.b->bufs[h - 1].name;
    }
    return "";
}

}  // namespace

TEST(the_work_rows_fit_the_step_only_beside_the_pinned_norms) {
    struct Case { const char* lcp; const char* tall; bool fit; } cases[] = {
        { nullptr, nullptr, false }, { "1024", nullptr, true },  { "2048", nullptr, true },
        { "64", nullptr, true },     { "48", nullptr, false },   { "4096", nullptr, true },
        { "3072", nullptr, true },   { "4097", nullptr, false }, { "4112", nullptr, false },
        { "1000", nullptr, false },  { "1024x", nullptr, false }, { "", nullptr, false },
        { "0", nullptr, false },     { nullptr, "1", true },     { nullptr, "0", false },
        { nullptr, "2", false },     { "100", "1", true },
    };
    for (const Case& c : cases) {
        Env e{};
        if (c.lcp) setenv("RADIANCE_EXPERIMENTAL_DS41_LCP", c.lcp, 1);
        if (c.tall) setenv("RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL", c.tall, 1);
        auto d = declare(0, 1024, 1);
        REQUIRE_EQ(d->rc, RAD_OK);
        CHECK_EQ(d->m->work_fit, c.fit);
        if (d->m->work_fit != c.fit)
            std::fprintf(stderr, "  LCP %s, EXACT_TALL %s\n", c.lcp ? c.lcp : "unset", c.tall ? c.tall : "unset");
        /* Unfitted, every step takes m.work, the bound of any step; fitted, its own. */
        CHECK_EQ(d->m->work, (int64_t)1025);
        CHECK_EQ(deepseek41::work_rows(*d->m, 2, 1, 1), c.fit ? (int64_t)1 : (int64_t)1025);
        CHECK_EQ(deepseek41::work_rows(*d->m, 2, 1024, 1), c.fit ? (int64_t)512 : (int64_t)1025);
    }
}

TEST(the_bound_holds_every_block_a_step_completes) {
    Env e{ { "RADIANCE_EXPERIMENTAL_DS41_LCP", "1024" } };
    auto d = declare(0, 1024, 16);
    REQUIRE_EQ(d->rc, RAD_OK);
    REQUIRE(d->m->work_fit);
    const deepseek41::Model& m = *d->m;
    /* One sequence: n rows from any position complete at most (n - 1) / r + 1 blocks, and some start
     * reaches it. */
    for (int64_t r : { (int64_t)1, (int64_t)2, (int64_t)3, (int64_t)4, (int64_t)128 })
        for (int64_t n = 1; n <= 300; ++n) {
            int64_t most = 0;
            for (int64_t p0 = 0; p0 < 2 * r + 2; ++p0) most = std::max(most, (p0 + n) / r - p0 / r);
            CHECK_EQ(most, (n - 1) / r + 1);
            CHECK_EQ(deepseek41::work_rows(m, r, n, 1), std::min(m.work, (n - 1) / r + 1));
        }
    /* Several: the sum is at most S + (T - S) / r, reached by one sequence carrying the rows past one each. */
    const std::vector<std::vector<Seq>> steps = {
        { { 1, 41 }, { 1, 42 }, { 1, 43 }, { 1, 44 } },  { { 4, 41 }, { 4, 50 } },
        { { 1, 41 }, { 1, 42 }, { 1, 43 }, { 64, 100 } }, { { 1, 1 }, { 1, 3 }, { 1, 5 }, { 13, 7 } },
        { { 1000, 0 } },                                   { { 1000, 1 } },
        { { 7, 3 }, { 9, 5 }, { 11, 9 }, { 2, 1 } },
    };
    for (const auto& seqs : steps) {
        auto k = step_of(d->b, {}, seqs);
        const int64_t T = k->b.n_tok, S = k->b.n_seq;
        for (int64_t r : { (int64_t)2, (int64_t)4 }) {
            CHECK(blocks_of(k->b, r) <= deepseek41::work_rows(m, r, T, S));
            CHECK(deepseek41::work_rows(m, r, T, S) >= S);
        }
    }
    auto k = step_of(d->b, {}, { { 1, 1 }, { 1, 3 }, { 1, 5 }, { 13, 7 } });
    CHECK_EQ(blocks_of(k->b, 2), deepseek41::work_rows(m, 2, k->b.n_tok, k->b.n_seq));
}

/* The ops a one-row decode step ran on max_tok's rows, by name and layer; with the pins, none. */
TEST(a_decode_step_at_2048_row_chunks_runs_nothing_on_max_toks_rows) {
    for (int64_t T : { 2048, 4096 })
    for (int64_t S : { (int64_t)1, (int64_t)8 })
        for (bool fit : { false, true }) {
            Env e{ WINDOW_SWITCHES };
            if (fit) setenv("RADIANCE_EXPERIMENTAL_DS41_LCP", std::to_string(T).c_str(), 1);
            auto d = declare(0, T, S);
            REQUIRE_EQ(d->rc, RAD_OK);
            REQUIRE_EQ(d->m->work_fit, fit);
            auto k = step_of(d->b, std::vector<Seq>((size_t)S, Seq{ 1, 41 }), {});
            auto c = run(d->b, k->b, 0);
            REQUIRE(c->step_fail.empty());
            const auto wo = work_ops(*d->m);
            /* Issues reaching past the rows a decode step's experts take, a row a routed slot. */
            std::vector<std::string> tall;
            int64_t n_work = 0;
            for (const RecIssue& is : c->issues) {
                const auto w = wo.find(is.op);
                if (w != wo.end()) {
                    ++n_work;
                    CHECK_EQ(work_rows_of(d->b, is), fit ? S : T + S);
                }
                const int64_t rows = issue_rows(d->b, is, d->m->g.max_tok);
                if (rows <= S * d->m->c.top_k) continue;
                const std::string at = op_name(d->b, is.op) + (w != wo.end() ? "@" + std::to_string(w->second) : "") +
                                       " " + std::to_string(rows);
                CHECK(w != wo.end());
                if (w == wo.end()) std::fprintf(stderr, "  %s: not a work op\n", at.c_str());
                tall.push_back(at);
            }
            CHECK_EQ(n_work, (int64_t)27);   /* nine ops at each of the three sources of ratio 2 */
            if (fit) {
                CHECK(tall.empty());
                continue;
            }
            CHECK_EQ(tall.size(), (size_t)27);
            std::string list;
            for (const std::string& t : tall) list += (list.empty() ? "" : ", ") + t;
            std::fprintf(stderr, "  %lld sequence(s), m.work %lld: %zu issues past the step's rows: %s\n",
                         (long long)S, (long long)d->m->work, tall.size(), list.c_str());
        }
}

TEST(the_same_step_at_both_bounds_writes_the_same_bytes) {
    Env e{ WINDOW_SWITCHES, { "RADIANCE_EXPERIMENTAL_DS41_LCP", "256" } };
    struct Case { int64_t max_tok, max_seqs; std::vector<Seq> dq, pq; } cases[] = {
        { 256, 8, { { 1, 41 } }, {} },                                    /* completes a block */
        { 256, 8, { { 1, 42 } }, {} },                                    /* none */
        { 256, 8, { { 1, 40 }, { 1, 41 }, { 1, 42 }, { 1, 43 }, { 1, 44 }, { 1, 45 }, { 1, 47 }, { 1, 50 } }, {} },
        { 256, 8, { { 4, 41 }, { 4, 50 } }, {} },                         /* verify-shaped rows */
        { 256, 8, {}, { { 256, 0 } } },                                   /* a chunk, the bound reached */
        { 256, 8, {}, { { 256, 1041 } } },
        { 256, 8, {}, { { 37, 300 } } },
        { 256, 8, { { 1, 41 }, { 1, 42 }, { 1, 43 } }, { { 64, 100 } } }, /* mixed */
        { 2048, 1, { { 1, 1041 } }, {} },                                 /* the window's: 2,049 rows before */
    };
    for (const Case& cs : cases) {
        auto d = declare(0, cs.max_tok, cs.max_seqs);
        REQUIRE_EQ(d->rc, RAD_OK);
        deepseek41::Model& m = *d->m;
        REQUIRE(m.work_fit);
        for (const auto& L : m.layers)
            if (L.src && L.r > 1) REQUIRE(L.k_f && L.l_f && L.c_tail);
        auto k = step_of(d->b, cs.dq, cs.pq);
        const int64_t T = k->b.n_tok, S = k->b.n_seq, count = blocks_of(k->b, 2), rows = S + (T - S) / 2;
        m.work_fit = false;
        auto old_c = run(d->b, k->b, 0);
        m.work_fit = true;
        auto new_c = run(d->b, k->b, 0);
        REQUIRE(old_c->step_fail.empty());
        REQUIRE(new_c->step_fail.empty());
        /* The same issues, the work ops on fewer rows: the step's own bound, at least its blocks. */
        REQUIRE_EQ(new_c->issues.size(), old_c->issues.size());
        const auto wo = work_ops(m);
        CHECK(count <= rows);
        bool same = true;
        for (size_t i = 0; i < new_c->issues.size(); ++i) {
            const RecIssue &x = new_c->issues[i], &y = old_c->issues[i];
            const bool work = wo.count(x.op) != 0;
            if (!same_but_rows(x, y, work)) {
                if (same)
                    std::fprintf(stderr, "  %lld rows: issue %zu (%s) differs beyond its work rows\n", (long long)T, i,
                                 op_name(d->b, x.op).c_str());
                same = false;
            }
            if (work) {
                CHECK_EQ(work_rows_of(d->b, x), rows);
                CHECK_EQ(work_rows_of(d->b, y), m.work);
            }
        }
        CHECK(same);
        const auto lo = lint_kinds(d->b, old_c.get()), ln = lint_kinds(d->b, new_c.get());
        for (const std::string& f : ln) {
            CHECK(lo.count(f));
            if (!lo.count(f)) std::fprintf(stderr, "  lint only at the step's bound: %s\n", f.c_str());
        }
        /* The compressor run on the host from the same memory: the same bytes. */
        Mem x, y;
        for (Mem* M : { &x, &y }) {
            M->b = &d->b;
            M->m = &m;
            M->blocks = S * kPitch;
            M->slots = S;
        }
        REQUIRE(replay(x, *old_c));
        REQUIRE(replay(y, *new_c));
        const std::string why = differ(x, y, count, 2);
        CHECK(why.empty());
        std::fprintf(stderr, "  %lld rows over %lld sequence(s) at %lld-row chunks: %lld blocks, %lld work rows "
                             "(%lld before)%s%s\n", (long long)T, (long long)S, (long long)cs.max_tok,
                     (long long)count, (long long)rows, (long long)m.work, why.empty() ? "" : "; differs: ",
                     why.c_str());
        /* The control: one row short of the step's blocks leaves the last unstored. */
        if (count >= 2 && count - 1 >= S) {
            const int64_t keep = m.work;
            m.work = count - 1;
            auto short_c = run(d->b, k->b, 0);
            m.work = keep;
            REQUIRE(short_c->step_fail.empty());
            Mem z;
            z.b = &d->b;
            z.m = &m;
            z.blocks = S * kPitch;
            z.slots = S;
            REQUIRE(replay(z, *short_c));
            CHECK(differ(x, z, count - 1, 2).find("cache") == 0);
        }
    }
}

TEST(the_rows_a_step_issues_follow_the_pass_key) {
    Env e{ WINDOW_SWITCHES, { "RADIANCE_EXPERIMENTAL_DS41_LCP", "256" } };
    auto d = declare(0, 256, 8);
    REQUIRE_EQ(d->rc, RAD_OK);
    REQUIRE(d->m->work_fit);
    /* Pairs with the counts Ctx::pass_key reads alike (rows, sequences, the bounds) and other positions:
     * one completes blocks where the other completes none or fewer. */
    struct Pair { std::vector<Seq> da, pa, db, pb; } pairs[] = {
        { { { 1, 41 } }, {}, { { 1, 42 } }, {} },
        { { { 1, 41 }, { 1, 43 }, { 1, 45 } }, {}, { { 1, 40 }, { 1, 44 }, { 1, 46 } }, {} },
        { { { 3, 41 }, { 3, 50 } }, {}, { { 3, 40 }, { 3, 50 } }, {} },
        { {}, { { 37, 300 } }, {}, { { 37, 301 } } },
    };
    for (const Pair& p : pairs) {
        auto a = step_of(d->b, p.da, p.pa);
        auto b = step_of(d->b, p.db, p.pb);
        b->b.max_ctx_len = a->b.max_ctx_len;   /* the bound as the batch builder rounds it */
        CHECK(blocks_of(a->b, 2) != blocks_of(b->b, 2));
        auto ca = run(d->b, a->b, 0);
        auto cb = run(d->b, b->b, 0);
        REQUIRE(ca->step_fail.empty());
        REQUIRE(cb->step_fail.empty());
        REQUIRE_EQ(ca->issues.size(), cb->issues.size());
        bool same = true;
        for (size_t i = 0; i < ca->issues.size() && same; ++i) {
            const RecIssue &x = ca->issues[i], &y = cb->issues[i];
            same = x.op == y.op && x.n == y.n && x.lane == y.lane && x.opd.size() == y.opd.size();
            for (size_t j = 0; j < x.opd.size() && same; ++j) {
                const RadOperand &u = x.opd[j], &v = y.opd[j];
                same = u.kind == v.kind && u.dtype == v.dtype && u.handle == v.handle && u.offset == v.offset &&
                       u.rows == v.rows && u.cols == v.cols;
            }
            if (!same) std::fprintf(stderr, "  issue %zu (%s) has another shape\n", i, op_name(d->b, x.op).c_str());
        }
        CHECK(same);
    }
}

TEST(a_split_step_issues_each_micro_batchs_work_rows_at_its_sequences) {
    Env e{ WINDOW_SWITCHES, { "RADIANCE_EXPERIMENTAL_DS41_LCP", "256" }, { "RADIANCE_EXPERIMENTAL_SCHED_TBO", "1" } };
    auto d = declare(0, 256, 8);
    REQUIRE_EQ(d->rc, RAD_OK);
    REQUIRE(d->m->tbo);
    REQUIRE(d->m->work_fit);
    std::vector<Seq> dq;
    for (int s = 0; s < 7; ++s) dq.push_back({ 1, 40 + s });
    auto k = step_of(d->b, dq, {});
    REQUIRE(::rad::tbo::step_splits(&k->b));
    auto c = run(d->b, k->b, 0);
    REQUIRE(c->step_fail.empty());
    const auto wo = work_ops(*d->m);
    std::map<int, std::vector<int64_t>> per;   /* layer -> its qsa_work launches' rows */
    for (const RecIssue& is : c->issues) {
        const auto w = wo.find(is.op);
        if (w == wo.end()) continue;
        const int64_t rows = work_rows_of(d->b, is);
        CHECK(rows == 4 || rows == 3);
        if (op_name(d->b, is.op) == "qsa_work") per[w->second].push_back(rows);
    }
    REQUIRE_EQ(per.size(), (size_t)3);
    for (const auto& [l, v] : per) CHECK(v == std::vector<int64_t>({ 4, 3 }) || v == std::vector<int64_t>({ 3, 4 }));
}

RAD_TEST_MAIN()
